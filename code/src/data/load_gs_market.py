"""Load the GS electricity-market dataset prepared from ``data/GS(1).csv``."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/gs_market")
PROCESSED_PATH = Path("data/processed/gs_market.pkl")
RAW_DIR_2025 = Path("data/raw/gs_market_2025")
PROCESSED_PATH_2025 = Path("data/processed/gs_market_2025.pkl")
TRAIN_ONLY_NORMALIZATION = "train_only"
FULL_DATA_NORMALIZATION = "full_data"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _read_timeseries_matrix(path: Path) -> pd.DataFrame:
    df = _normalise_columns(pd.read_csv(path))
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Expected timestamp plus value columns in {path}")

    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="mixed")
    values = df.iloc[:, 1:].copy()
    values.columns = [str(col) for col in values.columns]
    values = values.apply(pd.to_numeric, errors="coerce")
    values.index = pd.DatetimeIndex(ts)
    values.index.name = "timestamp"
    values = values.loc[~values.index.isna()].sort_index()
    return values.groupby(values.index).mean()


def _load_split_indices(raw_dir: Path) -> dict[str, np.ndarray]:
    path = raw_dir / "split_indices.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _normalise_occupancy(
    occ_raw: np.ndarray,
    *,
    split_indices: dict[str, np.ndarray],
    normalization_source: str,
) -> tuple[np.ndarray, float, float]:
    if normalization_source == TRAIN_ONLY_NORMALIZATION and "train" in split_indices:
        source = occ_raw[split_indices["train"]]
    elif normalization_source == FULL_DATA_NORMALIZATION:
        source = occ_raw
    else:
        source = occ_raw

    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    eps = 1e-8
    occ_norm = (occ_raw - vmin) / (vmax - vmin + eps)
    return occ_norm.astype(np.float32), vmin, vmax


def load_occupancy(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "occupancy.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"GS market occupancy matrix not found: {path}. "
            "Run `python scripts/prepare_gs_market.py` first."
        )
    return _read_timeseries_matrix(path)


def load_price(raw_dir: Path = RAW_DIR, timestamps=None) -> pd.DataFrame:
    path = raw_dir / "price.csv"
    if not path.exists():
        return pd.DataFrame()
    price = _read_timeseries_matrix(path)
    if timestamps is not None:
        price = price.reindex(pd.DatetimeIndex(timestamps)).ffill().bfill()
    return price


def load_missing_mask(raw_dir: Path = RAW_DIR, timestamps=None) -> pd.DataFrame:
    path = raw_dir / "missing_mask.csv"
    if not path.exists():
        return pd.DataFrame()
    mask = _read_timeseries_matrix(path).astype(bool)
    if timestamps is not None:
        mask = mask.reindex(pd.DatetimeIndex(timestamps)).fillna(False).astype(bool)
    return mask


def load_node_meta(raw_dir: Path = RAW_DIR, node_ids: list[str] | None = None) -> pd.DataFrame:
    path = raw_dir / "nodes.csv"
    if not path.exists():
        return pd.DataFrame(index=node_ids or [])

    meta = _normalise_columns(pd.read_csv(path))
    if "node_id" in meta.columns:
        meta["node_id"] = meta["node_id"].astype(str)
        meta = meta.set_index("node_id")
    else:
        meta.index = meta.index.map(str)

    if node_ids is not None:
        meta = meta.reindex([str(node_id) for node_id in node_ids])
    return meta


def load_adjacency(
    raw_dir: Path = RAW_DIR,
    n_nodes: int | None = None,
    node_ids: list[str] | None = None,
) -> np.ndarray:
    path = raw_dir / "adjacency.csv"
    if not path.exists():
        return np.eye(n_nodes, dtype=np.float32) if n_nodes else np.empty((0, 0), dtype=np.float32)

    df = _normalise_columns(pd.read_csv(path))
    if "node_id" in df.columns:
        mat = df.set_index("node_id")
        mat.index = mat.index.map(str)
        mat.columns = [str(col) for col in mat.columns]
        if node_ids is not None:
            ordered = [str(node_id) for node_id in node_ids]
            mat = mat.reindex(index=ordered, columns=ordered)
    else:
        mat = df
    return mat.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def load_gs_market(
    raw_dir: Path = RAW_DIR,
    processed_path: Path | None = None,
    force_reprocess: bool = False,
    normalization_source: str = TRAIN_ONLY_NORMALIZATION,
) -> dict:
    processed_path = Path(processed_path) if processed_path is not None else PROCESSED_PATH
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("normalization_source") == normalization_source:
            return cached

    df = load_occupancy(raw_dir)
    node_ids = [str(col) for col in df.columns]
    occ_raw = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(dtype=np.float32)
    if np.isnan(occ_raw).any():
        occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    split_indices = _load_split_indices(raw_dir)
    occ_norm, vmin, vmax = _normalise_occupancy(
        occ_raw,
        split_indices=split_indices,
        normalization_source=normalization_source,
    )

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": pd.DatetimeIndex(df.index),
        "node_ids": node_ids,
        "node_meta": load_node_meta(raw_dir, node_ids=node_ids),
        "adj": load_adjacency(raw_dir, n_nodes=len(node_ids), node_ids=node_ids),
        "weather": pd.DataFrame(),
        "price": load_price(raw_dir, timestamps=df.index),
        "missing_mask": load_missing_mask(raw_dir, timestamps=df.index),
        "split_indices": split_indices,
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": normalization_source,
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[GS-Market] T={occ_raw.shape[0]}, N={occ_raw.shape[1]}, "
        f"raw range=[{vmin:.3f}, {vmax:.3f}], normalization={normalization_source}"
    )
    return result


def load_gs_market_2025(
    raw_dir: Path = RAW_DIR_2025,
    processed_path: Path | None = None,
    force_reprocess: bool = False,
    normalization_source: str = TRAIN_ONLY_NORMALIZATION,
) -> dict:
    return load_gs_market(
        raw_dir=raw_dir,
        processed_path=processed_path or PROCESSED_PATH_2025,
        force_reprocess=force_reprocess,
        normalization_source=normalization_source,
    )


if __name__ == "__main__":
    data = load_gs_market(force_reprocess=True)
    print(f"occupancy shape: {data['occupancy'].shape}")
    print(f"node_ids: {data['node_ids']}")
    print(f"split sizes: { {k: len(v) for k, v in data.get('split_indices', {}).items()} }")
