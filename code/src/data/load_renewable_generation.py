"""
Load processed renewable generation datasets prepared into the local Diff-DoRA layout.

Supported datasets:
  - renewable_solar
  - renewable_wind
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_SPLIT_RATIO = 0.6
FULL_DATA_NORMALIZATION = "full_data"
TRAIN_ONLY_NORMALIZATION = "train_only"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _read_timeseries_matrix(path: Path) -> pd.DataFrame:
    df = _normalise_columns(pd.read_csv(path))
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Expected a timestamp column plus at least one value column in {path}")

    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="mixed")
    values = df.iloc[:, 1:].copy()
    values.columns = [str(col) for col in values.columns]
    values = values.apply(pd.to_numeric, errors="coerce")
    values.index = pd.DatetimeIndex(ts)
    values.index.name = "timestamp"
    values = values.loc[~values.index.isna()].sort_index()
    values = values.groupby(values.index).mean()
    return values


def _normalise_occupancy(
    occ_raw: np.ndarray,
    *,
    normalization_source: str,
) -> tuple[np.ndarray, float, float]:
    if normalization_source == FULL_DATA_NORMALIZATION:
        source = occ_raw
    elif normalization_source == TRAIN_ONLY_NORMALIZATION:
        train_end = max(1, int(len(occ_raw) * TRAIN_SPLIT_RATIO))
        source = occ_raw[:train_end]
    else:
        raise ValueError(
            f"Unsupported normalization_source={normalization_source!r}; "
            f"expected {FULL_DATA_NORMALIZATION!r} or {TRAIN_ONLY_NORMALIZATION!r}."
        )

    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    eps = 1e-8
    occ_norm = (occ_raw - vmin) / (vmax - vmin + eps)
    return occ_norm.astype(np.float32), vmin, vmax


def _load_common(
    *,
    raw_dir: Path,
    processed_path: Path,
    force_reprocess: bool = False,
    normalization_source: str = FULL_DATA_NORMALIZATION,
) -> dict:
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("normalization_source", FULL_DATA_NORMALIZATION) == normalization_source:
            return cached

    occ_path = raw_dir / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(
            f"Renewable occupancy matrix not found: {occ_path}. "
            "Run `python scripts/prepare_renewable_generation.py --dataset ...` first."
        )

    df = _read_timeseries_matrix(occ_path)
    node_ids = [str(col) for col in df.columns]
    occ_raw = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(dtype=np.float32)
    if np.isnan(occ_raw).any():
        occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    occ_norm, vmin, vmax = _normalise_occupancy(
        occ_raw,
        normalization_source=normalization_source,
    )

    nodes_path = raw_dir / "nodes.csv"
    node_meta = pd.DataFrame(index=node_ids)
    if nodes_path.exists():
        meta = _normalise_columns(pd.read_csv(nodes_path))
        if "node_id" in meta.columns:
            meta["node_id"] = meta["node_id"].astype(str)
            meta = meta.set_index("node_id")
        else:
            meta.index = meta.index.map(str)
        node_meta = meta.reindex(node_ids)

    weather_path = raw_dir / "weather.csv"
    weather = pd.DataFrame()
    if weather_path.exists():
        weather = _read_timeseries_matrix(weather_path).reindex(df.index).ffill().bfill()

    price_path = raw_dir / "price.csv"
    price = pd.DataFrame()
    if price_path.exists():
        price = _read_timeseries_matrix(price_path).reindex(df.index).ffill().bfill()

    adj_path = raw_dir / "adjacency.csv"
    if adj_path.exists():
        adj_df = _normalise_columns(pd.read_csv(adj_path))
        if "node_id" in adj_df.columns:
            adj_df = adj_df.set_index("node_id")
            adj_df.index = adj_df.index.map(str)
            adj_df.columns = [str(col) for col in adj_df.columns]
            adj_df = adj_df.reindex(index=node_ids, columns=node_ids)
        adj = adj_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        adj = np.eye(len(node_ids), dtype=np.float32)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": pd.DatetimeIndex(df.index),
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": adj,
        "weather": weather,
        "price": price,
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": normalization_source,
    }
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_renewable_solar(
    raw_dir: Path = Path("data/raw/renewable_solar"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
    normalization_source: str = FULL_DATA_NORMALIZATION,
) -> dict:
    return _load_common(
        raw_dir=raw_dir,
        processed_path=Path(processed_path) if processed_path is not None else Path("data/processed/renewable_solar.pkl"),
        force_reprocess=force_reprocess,
        normalization_source=normalization_source,
    )


def load_renewable_wind(
    raw_dir: Path = Path("data/raw/renewable_wind"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
    normalization_source: str = FULL_DATA_NORMALIZATION,
) -> dict:
    return _load_common(
        raw_dir=raw_dir,
        processed_path=Path(processed_path) if processed_path is not None else Path("data/processed/renewable_wind.pkl"),
        force_reprocess=force_reprocess,
        normalization_source=normalization_source,
    )
