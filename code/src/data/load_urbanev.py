"""
src/data/load_urbanev.py
-------------------------
Load and preprocess the UrbanEV EV charging occupancy dataset.

Expected raw layout after download:
  data/raw/urbanev/
    occupancy.csv     – shape (T, N), float in [0,1]
    weather.csv       – columns: timestamp, temperature, humidity, ...
    price.csv         – columns: timestamp, price
    poi.csv           – columns: node_id, poi_category, poi_count
    adjacency.csv     – columns: src, dst, weight  (or adjacency.npy)

Usage:
  from src.data.load_urbanev import load_urbanev
  data = load_urbanev()
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/urbanev")
PROCESSED_PATH = Path("data/processed/urbanev.pkl")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _normalise_columns(pd.read_csv(path, **kwargs))


def _read_timeseries_matrix(path: Path) -> pd.DataFrame:
    df = _normalise_columns(pd.read_csv(path))
    value_df = df.iloc[:, 1:].copy()
    value_df.columns = [str(col) for col in value_df.columns]
    value_df = value_df.apply(pd.to_numeric, errors="coerce")

    index_values = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    if not index_values.isna().all():
        value_df.index = pd.DatetimeIndex(index_values)
        value_df.index.name = str(df.columns[0])
        value_df = value_df.sort_index()
    else:
        value_df.index = pd.Index(df.iloc[:, 0], name=str(df.columns[0]))
    return value_df


def _aggregate_node_meta(raw_dir: Path, node_ids: list[str]) -> pd.DataFrame:
    path = raw_dir / "inf.csv"
    if not path.exists():
        return pd.DataFrame()

    df = _normalise_columns(pd.read_csv(path))
    if "TAZID" in df.columns:
        df["TAZID"] = df["TAZID"].astype(str)
        agg = (
            df.groupby("TAZID", dropna=False)
            .agg(
                lon=("longitude", "mean"),
                lat=("latitude", "mean"),
                capacity=("charge_count", "sum"),
                area=("area", "mean"),
                perimeter=("perimeter", "mean"),
                station_count=("station_id", "count"),
            )
        )
        return agg.reindex([str(node_id) for node_id in node_ids])

    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
        df = df.set_index("station_id")
        rename_map = {
            "charge_count": "capacity",
            "longitude": "lon",
            "latitude": "lat",
        }
        df = df.rename(columns={src: dst for src, dst in rename_map.items() if src in df.columns})
        return df.reindex([str(node_id) for node_id in node_ids])

    return pd.DataFrame()


def _load_weather(raw_dir: Path, timestamps) -> pd.DataFrame:
    frames = []
    for name, prefix in (("weather_airport.csv", "airport"), ("weather_central.csv", "central")):
        path = raw_dir / name
        if not path.exists():
            continue
        df = _read_csv_safe(path)
        if df.empty:
            continue
        time_col = df.columns[0]
        df = df.rename(
            columns={
                time_col: "timestamp",
                "T": "temperature",
                "U": "humidity",
                "P": "pressure",
                "P0": "station_pressure",
                "Td": "dew_point",
                "nRAIN": "rain_code",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        keep_cols = [
            col
            for col in ("temperature", "humidity", "pressure", "station_pressure", "dew_point", "rain_code")
            if col in df.columns
        ]
        df = df[keep_cols].apply(pd.to_numeric, errors="coerce")
        df = df.add_prefix(f"{prefix}_")
        frames.append(df)

    if not frames:
        fallback = raw_dir / "weather.csv"
        if not fallback.exists():
            return pd.DataFrame()
        weather = pd.read_csv(fallback, index_col=0, parse_dates=True)
        weather.columns = [str(col) for col in weather.columns]
        weather = weather.sort_index()
        return weather.reindex(pd.DatetimeIndex(timestamps), method="ffill").bfill()

    weather = pd.concat(frames, axis=1).sort_index()
    for logical_name in ("temperature", "humidity", "pressure", "station_pressure", "dew_point"):
        cols = [col for col in weather.columns if col.endswith(f"_{logical_name}")]
        if cols:
            weather[logical_name] = weather[cols].mean(axis=1)
    rain_cols = [col for col in weather.columns if col.endswith("_rain_code")]
    if rain_cols:
        weather["rain_code"] = weather[rain_cols].max(axis=1)

    ts_index = pd.DatetimeIndex(timestamps)
    weather = weather.reindex(ts_index, method="ffill").bfill()
    return weather


def _load_price(raw_dir: Path, timestamps) -> pd.DataFrame:
    price_path = raw_dir / "price.csv"
    if price_path.exists():
        price = _read_timeseries_matrix(price_path)
    else:
        e_price = _read_timeseries_matrix(raw_dir / "e_price.csv") if (raw_dir / "e_price.csv").exists() else pd.DataFrame()
        s_price = _read_timeseries_matrix(raw_dir / "s_price.csv") if (raw_dir / "s_price.csv").exists() else pd.DataFrame()
        if e_price.empty and s_price.empty:
            return pd.DataFrame()
        if e_price.empty:
            price = s_price
        elif s_price.empty:
            price = e_price
        else:
            price = e_price.add(s_price, fill_value=0.0)

    if isinstance(price.index, pd.DatetimeIndex):
        ts_index = pd.DatetimeIndex(timestamps)
        price = price.reindex(ts_index)
        price = price.ffill().bfill()
    return price


def _load_adjacency(raw_dir: Path, n_nodes: int, node_ids: list[str]) -> np.ndarray:
    npy_path = raw_dir / "adjacency.npy"
    if npy_path.exists():
        return np.load(npy_path)

    for csv_path in (raw_dir / "adjacency.csv", raw_dir / "adj.csv"):
        if not csv_path.exists():
            continue

        df = _normalise_columns(pd.read_csv(csv_path))
        lowered = {str(col).lower(): col for col in df.columns}
        if {"src", "dst"}.issubset(lowered):
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
            ordered = [str(node_id) for node_id in node_ids]
            if set(ordered).issubset(mat_df.index) and set(ordered).issubset(mat_df.columns):
                mat_df = mat_df.loc[ordered, ordered]
            mat = mat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if mat.shape[0] == mat.shape[1]:
                return mat

        mat = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if mat.shape[0] == mat.shape[1]:
            return mat

    return np.eye(n_nodes, dtype=np.float32)


def _ensure_cached_node_ids(cached: dict, raw_dir: Path) -> dict:
    if cached.get("node_ids") is not None:
        return cached
    occ_path = raw_dir / "occupancy.csv"
    if occ_path.exists():
        cols = pd.read_csv(occ_path, nrows=0).columns.tolist()[1:]
        cached["node_ids"] = [str(c) for c in cols]
    else:
        occ = cached.get("occupancy")
        if occ is not None:
            cached["node_ids"] = [str(i) for i in range(occ.shape[1])]
    return cached


def normalize(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    vmin = float(arr.min())
    vmax = float(arr.max())
    eps = 1e-8
    return (arr - vmin) / (vmax - vmin + eps), vmin, vmax


def load_urbanev(
    raw_dir: Path = RAW_DIR,
    processed_path: Path = PROCESSED_PATH,
    force_reprocess: bool = False,
) -> dict:
    """
    Load UrbanEV dataset.

    Returns
    -------
    {
        "occupancy":  np.ndarray (T, N),
        "timestamps": pd.DatetimeIndex,
        "weather":    pd.DataFrame,
        "price":      pd.DataFrame,
        "poi":        pd.DataFrame,
        "adj":        np.ndarray (N, N),
        "norm_min":   float,
        "norm_max":   float,
    }
    """
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as f:
            return _ensure_cached_node_ids(pickle.load(f), raw_dir)

    occ_path = raw_dir / "occupancy.csv"
    if not occ_path.exists():
        raise FileNotFoundError(
            f"UrbanEV occupancy not found: {occ_path}\n"
            "Download UrbanEV and place files under data/raw/urbanev/."
        )

    occ_df = _read_timeseries_matrix(occ_path)
    T, N = occ_df.shape
    occ_raw = occ_df.values.astype(np.float32)
    occ_raw = pd.DataFrame(occ_raw).ffill().bfill().values.astype(np.float32)
    occ_norm, vmin, vmax = normalize(occ_raw)

    node_ids = [str(col) for col in occ_df.columns]
    node_meta = _aggregate_node_meta(raw_dir, node_ids)
    weather = _load_weather(raw_dir, occ_df.index)
    price = _load_price(raw_dir, occ_df.index)
    poi = _read_csv_safe(raw_dir / "poi.csv")

    # Adjacency
    adj = _load_adjacency(raw_dir, n_nodes=N, node_ids=node_ids)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": occ_df.index,
        "node_ids": node_ids,
        "node_meta": node_meta,
        "weather": weather,
        "price": price,
        "poi": poi,
        "adj": adj,
        "norm_min": vmin,
        "norm_max": vmax,
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[UrbanEV] T={T}, N={N}, NaN%={np.isnan(occ_raw).mean():.4f}")
    return result


if __name__ == "__main__":
    data = load_urbanev()
    print(f"occupancy: {data['occupancy'].shape}")
