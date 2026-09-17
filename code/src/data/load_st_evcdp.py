"""
src/data/load_st_evcdp.py
-------------------------
Load and preprocess the ST-EVCDP EV charging occupancy dataset.

Expected raw layout after download:
  data/raw/st_evcdp/
    occupancy.csv       – shape (T, N), float in [0,1]
    nodes.csv           – columns: node_id, zone_type, lat, lon, capacity
    adjacency.csv       – columns: src, dst, weight   (or an .npy matrix)

Usage:
  from src.data.load_st_evcdp import load_st_evcdp
  data = load_st_evcdp()
  # returns dict with keys: occupancy, node_meta, adj
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/st_evcdp")
PROCESSED_PATH = Path("data/processed/st_evcdp.pkl")
TRAINNORM_H6_PROCESSED_PATH = Path("data/processed/st_evcdp_trainnorm_h6.pkl")
TRAIN_SPLIT_RATIO = 0.6
FULL_DATA_NORMALIZATION = "full_data"
TRAIN_ONLY_NORMALIZATION = "train_only"

EXPECTED_OCCUPANCY_SHAPE = (8640, 247)
EXPECTED_ADJ_SHAPE = (247, 247)
EXPECTED_WEATHER_SHAPE = (8640, 5)
EXPECTED_PRICE_SHAPE = (8640, 247)
REQUIRED_NODE_META_FIELDS = ("zone_type", "capacity", "area")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _load_time_index(raw_dir: Path = RAW_DIR) -> pd.DatetimeIndex | None:
    path = raw_dir / "time.csv"
    if not path.exists():
        return None

    df = _normalise_columns(pd.read_csv(path))
    lowered = {str(col).lower(): col for col in df.columns}
    required = ["year", "month", "day", "hour", "minute"]
    if any(col not in lowered for col in required):
        return None

    rename_map = {lowered[name]: name for name in required}
    if "second" in lowered:
        rename_map[lowered["second"]] = "second"
    df = df.rename(columns=rename_map)
    if "second" not in df.columns:
        df["second"] = 0

    ts = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]],
        errors="coerce",
    )
    if ts.isna().all():
        return None
    return pd.DatetimeIndex(ts)


def _read_aligned_timeseries(path: Path, *, time_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    df = _normalise_columns(pd.read_csv(path))
    value_df = df.iloc[:, 1:].copy()
    value_df.columns = [str(col) for col in value_df.columns]
    value_df = value_df.apply(pd.to_numeric, errors="coerce")

    if time_index is not None and len(time_index) == len(value_df):
        value_df.index = time_index
        value_df.index.name = "timestamp"
    else:
        index_values = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if not index_values.isna().all():
            value_df.index = pd.DatetimeIndex(index_values)
            value_df.index.name = str(df.columns[0])
        else:
            value_df.index = pd.Index(df.iloc[:, 0], name=str(df.columns[0]))

    if isinstance(value_df.index, pd.DatetimeIndex):
        value_df = value_df.sort_index()
    return value_df


def load_occupancy(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return occupancy DataFrame, index=DatetimeIndex, columns=node_id."""
    occ_path = raw_dir / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(
            f"Occupancy file not found: {occ_path}\n"
            "Please download ST-EVCDP and place files under data/raw/st_evcdp/."
        )
    return _read_aligned_timeseries(occ_path, time_index=_load_time_index(raw_dir))


def _ensure_cached_node_ids(cached: dict, raw_dir: Path) -> dict:
    if cached.get("node_ids") is not None:
        return cached
    try:
        cached["node_ids"] = [str(c) for c in load_occupancy(raw_dir).columns]
    except FileNotFoundError:
        occ = cached.get("occupancy")
        if occ is not None:
            cached["node_ids"] = [str(i) for i in range(occ.shape[1])]
    return cached


def load_price(raw_dir: Path = RAW_DIR, timestamps=None) -> pd.DataFrame:
    path = raw_dir / "price.csv"
    if not path.exists():
        return pd.DataFrame()

    df = _read_aligned_timeseries(path, time_index=_load_time_index(raw_dir))
    if timestamps is not None and isinstance(df.index, pd.DatetimeIndex):
        ts_index = pd.DatetimeIndex(timestamps)
        df = df.reindex(ts_index)
        df = df.ffill().bfill()
    return df


def load_weather(raw_dir: Path = RAW_DIR, timestamps=None) -> pd.DataFrame:
    xls_path = raw_dir / "SZweather20220619-20220718.xls"
    if not xls_path.exists():
        csv_path = raw_dir / "weather.csv"
        if not csv_path.exists():
            return pd.DataFrame()
        weather = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        weather.columns = [str(col) for col in weather.columns]
        weather = weather.sort_index()
    else:
        try:
            weather = pd.read_excel(xls_path)
        except Exception:
            return pd.DataFrame()

        weather = _normalise_columns(weather)
        time_col = weather.columns[0]
        weather = weather.rename(
            columns={
                time_col: "timestamp",
                "T": "temperature",
                "U": "humidity",
                "P": "pressure",
                "P0": "station_pressure",
                "Td": "dew_point",
            }
        )
        weather["timestamp"] = pd.to_datetime(
            weather["timestamp"],
            dayfirst=True,
            errors="coerce",
        )
        weather = weather.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        keep_cols = [
            col
            for col in ("temperature", "humidity", "pressure", "station_pressure", "dew_point")
            if col in weather.columns
        ]
        weather = weather[keep_cols].apply(pd.to_numeric, errors="coerce")

    if timestamps is not None and isinstance(weather.index, pd.DatetimeIndex):
        ts_index = pd.DatetimeIndex(timestamps)
        weather = weather.reindex(ts_index, method="ffill").bfill()
    return weather


def load_node_meta(raw_dir: Path = RAW_DIR, node_ids: list[str] | None = None) -> pd.DataFrame:
    """Return node metadata DataFrame, index=node_id."""
    path = raw_dir / "nodes.csv"
    if not path.exists():
        info_path = raw_dir / "information.csv"
        if not info_path.exists():
            return pd.DataFrame()  # allow graceful degradation

        df = _normalise_columns(pd.read_csv(info_path))
        rename_map = {
            "grid": "node_id",
            "count": "capacity",
            "lon": "lon",
            "la": "lat",
            "CBD": "cbd_flag",
        }
        df = df.rename(columns={src: dst for src, dst in rename_map.items() if src in df.columns})
        if "node_id" in df.columns:
            df["node_id"] = df["node_id"].astype(str)
            df = df.set_index("node_id")
        if "cbd_flag" in df.columns:
            cbd_flag = pd.to_numeric(df["cbd_flag"], errors="coerce").fillna(0)
            df["zone_type"] = np.where(cbd_flag > 0, "CBD", "Residential")
        if node_ids is not None:
            df = df.reindex([str(node_id) for node_id in node_ids])
        return df

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(str)
    if node_ids is not None:
        df = df.reindex([str(node_id) for node_id in node_ids])
    return df


def load_adjacency(
    raw_dir: Path = RAW_DIR,
    n_nodes: int | None = None,
    node_ids: list[str] | None = None,
) -> np.ndarray:
    """Return adjacency matrix as (N, N) ndarray."""
    npy_path = raw_dir / "adjacency.npy"
    if npy_path.exists():
        return np.load(npy_path)

    for csv_path in (raw_dir / "adjacency.csv", raw_dir / "adj.csv"):
        if not csv_path.exists():
            continue

        df = _normalise_columns(pd.read_csv(csv_path))
        lowered = {str(col).lower(): col for col in df.columns}
        if {"src", "dst"}.issubset(lowered):
            if n_nodes is None:
                n_nodes = max(df[lowered["src"]].max(), df[lowered["dst"]].max()) + 1
            adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for _, row in df.iterrows():
                adj[int(row[lowered["src"]]), int(row[lowered["dst"]])] = float(row.get(lowered.get("weight"), 1.0))
            return adj

        first_col = str(df.columns[0]).lower()
        if first_col in {"node_id", "grid", "id"} and df.shape[1] > 1:
            row_ids = df.iloc[:, 0].astype(str)
            mat_df = df.iloc[:, 1:].copy()
            mat_df.index = row_ids
            mat_df.columns = [str(col) for col in mat_df.columns]
            if node_ids is not None:
                ordered = [str(node_id) for node_id in node_ids]
                if set(ordered).issubset(mat_df.index) and set(ordered).issubset(mat_df.columns):
                    mat_df = mat_df.loc[ordered, ordered]
            mat = mat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if mat.shape[0] == mat.shape[1]:
                return mat

        mat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if mat.shape[0] == mat.shape[1]:
            return mat

    # fallback: identity
    if n_nodes:
        return np.eye(n_nodes, dtype=np.float32)
    return np.empty((0, 0), dtype=np.float32)


def normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalise to [0,1], return (normed, min, max)."""
    vmin = float(arr.min())
    vmax = float(arr.max())
    eps = 1e-8
    return (arr - vmin) / (vmax - vmin + eps), vmin, vmax


def _resolve_processed_path(
    processed_path: Path | None,
    normalization_source: str,
) -> Path:
    if processed_path is not None:
        return Path(processed_path)
    if normalization_source == TRAIN_ONLY_NORMALIZATION:
        return TRAINNORM_H6_PROCESSED_PATH
    return PROCESSED_PATH


def _normalise_occupancy(
    occ_raw: np.ndarray,
    *,
    normalization_source: str,
) -> tuple[np.ndarray, float, float]:
    if normalization_source == FULL_DATA_NORMALIZATION:
        return normalize(occ_raw)
    if normalization_source == TRAIN_ONLY_NORMALIZATION:
        train_end = int(len(occ_raw) * TRAIN_SPLIT_RATIO)
        occ_train = occ_raw[:train_end]
        vmin = float(occ_train.min())
        vmax = float(occ_train.max())
        eps = 1e-8
        occ_norm = (occ_raw - vmin) / (vmax - vmin + eps)
        return occ_norm.astype(np.float32), vmin, vmax
    raise ValueError(
        f"Unsupported normalization_source={normalization_source!r}; "
        f"expected one of {{{FULL_DATA_NORMALIZATION!r}, {TRAIN_ONLY_NORMALIZATION!r}}}."
    )


def validate_st_evcdp_processed(
    data: dict,
    *,
    strict: bool = True,
) -> dict:
    occupancy = data.get("occupancy")
    adj = data.get("adj")
    weather = data.get("weather")
    price = data.get("price")
    node_meta = data.get("node_meta")

    errors = []
    if occupancy is None or tuple(getattr(occupancy, "shape", ())) != EXPECTED_OCCUPANCY_SHAPE:
        errors.append(
            f"occupancy shape={getattr(occupancy, 'shape', None)!r} "
            f"(expected {EXPECTED_OCCUPANCY_SHAPE!r})"
        )
    if adj is None or tuple(getattr(adj, "shape", ())) != EXPECTED_ADJ_SHAPE:
        errors.append(f"adj shape={getattr(adj, 'shape', None)!r} (expected {EXPECTED_ADJ_SHAPE!r})")
    if weather is None or tuple(getattr(weather, "shape", ())) != EXPECTED_WEATHER_SHAPE:
        errors.append(
            f"weather shape={getattr(weather, 'shape', None)!r} "
            f"(expected {EXPECTED_WEATHER_SHAPE!r})"
        )
    if price is None or tuple(getattr(price, "shape", ())) != EXPECTED_PRICE_SHAPE:
        errors.append(
            f"price shape={getattr(price, 'shape', None)!r} "
            f"(expected {EXPECTED_PRICE_SHAPE!r})"
        )
    if node_meta is None or getattr(node_meta, "empty", True):
        errors.append("node_meta is missing or empty")
    else:
        missing_fields = [field for field in REQUIRED_NODE_META_FIELDS if field not in node_meta.columns]
        if missing_fields:
            errors.append(f"node_meta missing required fields: {missing_fields}")

    numeric_checks = [
        ("occupancy", occupancy),
        ("adj", adj),
    ]
    if weather is not None and not getattr(weather, "empty", True):
        numeric_checks.append(("weather", weather.to_numpy(dtype=np.float32)))
    if price is not None and not getattr(price, "empty", True):
        numeric_checks.append(("price", price.to_numpy(dtype=np.float32)))
    for name, arr in numeric_checks:
        if arr is None:
            continue
        if np.isnan(np.asarray(arr, dtype=np.float32)).any():
            errors.append(f"{name} contains NaN values")

    if errors and strict:
        raise ValueError("ST-EVCDP validation failed: " + "; ".join(errors))
    return {
        "occupancy_shape": getattr(occupancy, "shape", None),
        "adj_shape": getattr(adj, "shape", None),
        "weather_shape": getattr(weather, "shape", None),
        "price_shape": getattr(price, "shape", None),
        "node_meta_columns": [] if node_meta is None else list(node_meta.columns),
        "errors": errors,
    }


def load_st_evcdp(
    raw_dir: Path = RAW_DIR,
    processed_path: Path | None = None,
    force_reprocess: bool = False,
    normalization_source: str = FULL_DATA_NORMALIZATION,
) -> dict:
    """
    Load ST-EVCDP dataset.

    Returns
    -------
    {
        "occupancy": np.ndarray  shape (T, N),
        "timestamps": pd.DatetimeIndex,
        "node_meta":  pd.DataFrame,
        "adj":        np.ndarray shape (N, N),
        "norm_min":   float,
        "norm_max":   float,
    }
    """
    processed_path = _resolve_processed_path(processed_path, normalization_source)
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as f:
            cached = _ensure_cached_node_ids(pickle.load(f), raw_dir)
        cached_source = cached.get("normalization_source", FULL_DATA_NORMALIZATION)
        if cached_source == normalization_source:
            return cached

    df = load_occupancy(raw_dir)
    node_ids = [str(col) for col in df.columns]
    node_meta = load_node_meta(raw_dir, node_ids=node_ids)
    price = load_price(raw_dir, timestamps=df.index)
    weather = load_weather(raw_dir, timestamps=df.index)

    T, N = df.shape
    occ_raw = df.values.astype(np.float32)

    # Fill NaN by forward-fill then backward-fill
    occ_df = pd.DataFrame(occ_raw)
    occ_df = occ_df.ffill().bfill()
    occ_raw = occ_df.values.astype(np.float32)

    occ_norm, vmin, vmax = _normalise_occupancy(
        occ_raw,
        normalization_source=normalization_source,
    )

    adj = load_adjacency(raw_dir, n_nodes=N, node_ids=node_ids)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": df.index,
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": adj,
        "price": price,
        "weather": weather,
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": normalization_source,
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[ST-EVCDP] T={T}, N={N}, NaN%={np.isnan(occ_raw).mean():.4f}")
    return result


if __name__ == "__main__":
    data = load_st_evcdp()
    occ = data["occupancy"]
    print(f"occupancy shape: {occ.shape}, range [{occ.min():.3f}, {occ.max():.3f}]")
    print(f"node_meta columns: {list(data['node_meta'].columns)}")
    print(f"adj shape: {data['adj'].shape}")
