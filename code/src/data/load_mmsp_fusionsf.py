"""Loader for the MMSP / FusionSF photovoltaic forecasting dataset.

The official FusionSF repository expects the dataset to be staged as:

```text
data/raw/aaai_2026/mmsp_fusionsf/data/solar_power/solar_power.csv
data/raw/aaai_2026/mmsp_fusionsf/data/nwp/nwp.csv
data/raw/aaai_2026/mmsp_fusionsf/data/satellite/satellite.npy
data/raw/aaai_2026/mmsp_fusionsf/data/satellite/satellite_coords.npy
data/raw/aaai_2026/mmsp_fusionsf/data/satellite/satellite_times.npy
```

The solar-power CSV used by FusionSF is a long table with columns similar to
`datetime, site, lat, lon, power`.  This loader also accepts a wide matrix with
one timestamp column and one PV-power column per plant/site, because mirrors and
manual exports often reshape the same data.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRAIN_SPLIT_RATIO = 0.7


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return _normalise_columns(pd.read_csv(path))
    except UnicodeDecodeError:
        return _normalise_columns(pd.read_csv(path, encoding="gbk"))


def _find_file(raw_dir: Path, candidates: Iterable[str], *, contains: Iterable[str] = ()) -> Path | None:
    raw_dir = Path(raw_dir)
    for rel in candidates:
        path = raw_dir / rel
        if path.exists():
            return path
    contains_lower = [token.lower() for token in contains]
    for suffix in ("*.csv", "*.CSV"):
        for path in raw_dir.glob(f"**/{suffix}"):
            lowered = str(path).lower()
            if all(token in lowered for token in contains_lower):
                return path
    return None


def _find_solar_power_file(raw_dir: Path) -> Path:
    path = _find_file(
        raw_dir,
        candidates=[
            "data/solar_power/solar_power.csv",
            "solar_power/solar_power.csv",
            "solar_power.csv",
        ],
        contains=["solar", "power"],
    )
    if path is None:
        raise FileNotFoundError(
            f"No FusionSF solar_power.csv found under {raw_dir}. "
            "Expected data/solar_power/solar_power.csv or solar_power/solar_power.csv."
        )
    return path


def _find_nwp_file(raw_dir: Path) -> Path | None:
    return _find_file(
        raw_dir,
        candidates=[
            "data/nwp/nwp.csv",
            "nwp/nwp.csv",
            "nwp.csv",
        ],
        contains=["nwp"],
    )


def _find_satellite_dir(raw_dir: Path) -> Path | None:
    candidates = [
        raw_dir / "data" / "satellite",
        raw_dir / "satellite",
    ]
    for path in candidates:
        if (path / "satellite.npy").exists():
            return path
    for path in raw_dir.glob("**/satellite.npy"):
        return path.parent
    return None


def _timestamp_column(df: pd.DataFrame, preferred: Iterable[str] = ()) -> str:
    lower_to_col = {str(col).lower(): col for col in df.columns}
    for name in [*preferred, "datetime", "timestamp", "time", "date_time", "time_stamp", "date", "fcst_date"]:
        if name.lower() in lower_to_col:
            col = str(lower_to_col[name.lower()])
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().any():
                return col
    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            return str(column)
    raise ValueError("Could not identify a timestamp column.")


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_col = {str(col).lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_to_col:
            return str(lower_to_col[name.lower()])
    return None


def _site_column(df: pd.DataFrame) -> str | None:
    return _first_existing_column(df, ["site", "site_id", "plant", "plant_id", "station", "station_id"])


def _target_column(df: pd.DataFrame, *, exclude: Iterable[str]) -> str:
    excluded = {str(col).lower() for col in exclude}
    preferred = [
        "power",
        "pv_power",
        "solar_power",
        "generation",
        "generated_power",
        "active_power",
        "target",
        "value",
    ]
    for column in preferred:
        actual = _first_existing_column(df, [column])
        if actual is not None and actual.lower() not in excluded:
            return actual

    numeric_candidates: list[str] = []
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in excluded or "lat" in lowered or "lon" in lowered:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().mean() > 0.8:
            numeric_candidates.append(str(column))
    if numeric_candidates:
        return numeric_candidates[0]
    raise ValueError("Could not identify PV power/target column.")


def _wide_value_columns(df: pd.DataFrame, *, timestamp_col: str) -> list[str]:
    out: list[str] = []
    for column in df.columns:
        if str(column) == timestamp_col:
            continue
        lowered = str(column).lower()
        if lowered in {"lat", "latitude", "lon", "longitude"}:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().mean() > 0.8:
            out.append(str(column))
    if not out:
        raise ValueError("Wide FusionSF solar-power table has no numeric site columns.")
    return out


def _node_id(site: object) -> str:
    text = str(site).strip()
    if not text:
        text = "unknown"
    return f"pv_site_{text}"


def _normalise_train_only(occ_raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    train_end = max(1, int(len(occ_raw) * TRAIN_SPLIT_RATIO))
    source = occ_raw[:train_end]
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    occ_norm = (occ_raw - vmin) / (vmax - vmin + 1e-8)
    return occ_norm.astype(np.float32), vmin, vmax


def _read_solar_power(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_csv(path)
    ts_col = _timestamp_column(df, preferred=["datetime"])
    site_col = _site_column(df)

    if site_col is None:
        value_cols = _wide_value_columns(df, timestamp_col=ts_col)
        working = df[[ts_col, *value_cols]].copy()
        working[ts_col] = pd.to_datetime(working[ts_col], errors="coerce", format="mixed")
        working = working.loc[working[ts_col].notna()].sort_values(ts_col)
        occupancy = working.set_index(ts_col)[value_cols].apply(pd.to_numeric, errors="coerce")
        occupancy.columns = [_node_id(col) for col in value_cols]
        node_meta = pd.DataFrame(
            {
                "node_id": list(occupancy.columns),
                "site_type": "pv",
                "type": "photovoltaic_power",
                "source_column": value_cols,
            }
        ).set_index("node_id")
        return occupancy, node_meta

    exclude = {ts_col, site_col}
    lat_col = _first_existing_column(df, ["lat", "latitude"])
    lon_col = _first_existing_column(df, ["lon", "lng", "longitude"])
    if lat_col:
        exclude.add(lat_col)
    if lon_col:
        exclude.add(lon_col)
    target_col = _target_column(df, exclude=exclude)

    working = df.copy()
    working[ts_col] = pd.to_datetime(working[ts_col], errors="coerce", format="mixed")
    working = working.loc[working[ts_col].notna()].sort_values([ts_col, site_col])
    working[site_col] = working[site_col].astype(str)
    working["node_id"] = working[site_col].map(_node_id)
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce").clip(lower=0.0)

    occupancy = working.pivot_table(index=ts_col, columns="node_id", values=target_col, aggfunc="mean")
    occupancy = occupancy.sort_index()

    meta_cols = [site_col]
    for column in (lat_col, lon_col):
        if column is not None:
            meta_cols.append(column)
    metadata = working.groupby("node_id")[meta_cols].first()
    metadata = metadata.rename(columns={site_col: "source_site"})
    if lat_col is not None and lat_col in metadata.columns:
        metadata["latitude"] = pd.to_numeric(metadata[lat_col], errors="coerce")
    if lon_col is not None and lon_col in metadata.columns:
        metadata["longitude"] = pd.to_numeric(metadata[lon_col], errors="coerce")
    metadata["node_id"] = metadata.index
    metadata["site_type"] = "pv"
    metadata["type"] = "photovoltaic_power"
    return occupancy, metadata


def _read_nwp(path: Path | None, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = _read_csv(path)
    ts_col = _timestamp_column(df, preferred=["fcst_date", "datetime"])
    working = df.copy()
    working[ts_col] = pd.to_datetime(working[ts_col], errors="coerce", format="mixed")
    working = working.loc[working[ts_col].notna()].sort_values(ts_col)
    excluded = {ts_col.lower(), "lat", "latitude", "lon", "lng", "longitude"}
    value_cols: list[str] = []
    for column in working.columns:
        lowered = str(column).lower()
        if lowered in excluded:
            continue
        values = pd.to_numeric(working[column], errors="coerce")
        if values.notna().mean() > 0.5:
            working[column] = values
            value_cols.append(str(column))
    if not value_cols:
        return pd.DataFrame()
    weather = working.groupby(ts_col)[value_cols].mean().sort_index()
    return weather.reindex(timestamps).interpolate(limit_direction="both").ffill().bfill()


def _satellite_metadata(raw_dir: Path) -> dict[str, str | bool]:
    satellite_dir = _find_satellite_dir(raw_dir)
    if satellite_dir is None:
        return {"available": False}
    files = {
        "satellite_npy": satellite_dir / "satellite.npy",
        "satellite_coords_npy": satellite_dir / "satellite_coords.npy",
        "satellite_times_npy": satellite_dir / "satellite_times.npy",
    }
    payload: dict[str, str | bool] = {"available": bool(files["satellite_npy"].exists())}
    for key, path in files.items():
        if path.exists():
            payload[key] = str(path)
    return payload


def _adjacency_from_metadata(node_meta: pd.DataFrame, node_ids: list[str]) -> np.ndarray:
    lat_col = _first_existing_column(node_meta, ["latitude", "lat"])
    lon_col = _first_existing_column(node_meta, ["longitude", "lon", "lng"])
    if lat_col is None or lon_col is None:
        return np.eye(len(node_ids), dtype=np.float32)
    coords = node_meta.reindex(node_ids)[[lat_col, lon_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all():
        return np.eye(len(node_ids), dtype=np.float32)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    scale = np.nanmedian(dist[dist > 0]) if np.any(dist > 0) else 1.0
    return np.exp(-dist / max(float(scale), 1e-6)).astype(np.float32)


def load_mmsp_fusionsf(
    raw_dir: Path = Path("data/raw/aaai_2026/mmsp_fusionsf"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    processed = Path(processed_path) if processed_path is not None else Path("data/processed/mmsp_fusionsf.pkl")
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as f:
            return pickle.load(f)

    raw_dir = Path(raw_dir)
    solar_power_file = _find_solar_power_file(raw_dir)
    occupancy_df, node_meta = _read_solar_power(solar_power_file)
    occupancy_df = occupancy_df.sort_index()
    occupancy_df = occupancy_df[~occupancy_df.index.duplicated(keep="last")]
    occupancy_df = occupancy_df.asfreq("h").interpolate(limit_direction="both").ffill().bfill()
    node_ids = [str(col) for col in occupancy_df.columns]
    node_meta = node_meta.reindex(node_ids)
    node_meta["node_id"] = node_ids
    node_meta["site_type"] = "pv"
    node_meta["type"] = "photovoltaic_power"

    weather = _read_nwp(_find_nwp_file(raw_dir), pd.DatetimeIndex(occupancy_df.index))
    occ_raw = occupancy_df.to_numpy(dtype=np.float32)
    if np.isnan(occ_raw).any():
        occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    occ_norm, vmin, vmax = _normalise_train_only(occ_raw)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": pd.DatetimeIndex(occupancy_df.index),
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": _adjacency_from_metadata(node_meta, node_ids),
        "weather": weather,
        "price": pd.DataFrame(),
        "satellite": _satellite_metadata(raw_dir),
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_only",
        "source_dataset": "MMSP/FusionSF",
        "source_file": str(solar_power_file),
    }
    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result
