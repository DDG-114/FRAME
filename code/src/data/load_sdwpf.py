"""Loader for SDWPF / KDD Cup 2022 wind-power data.

Supported raw layouts include the KDD Cup files:

```text
data/raw/aaai_2026/sdwpf/wtbdata_245days.csv
data/raw/aaai_2026/sdwpf/sdwpf_baidukddcup2022_full.CSV
data/raw/aaai_2026/sdwpf/sdwpf_baidukddcup2022_turb_location.CSV
```

and the Scientific Data full release files:

```text
data/raw/aaai_2026/sdwpf/sdwpf_2001_2112_full.csv
data/raw/aaai_2026/sdwpf/sdwpf_2001_2112_full.parquet
data/raw/aaai_2026/sdwpf/sdwpf_turb_location_elevation.csv
```
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAIN_SPLIT_RATIO = 0.7
TARGET_COLUMN = "Patv"
TURBINE_COLUMN = "TurbID"
SCADA_COLUMNS = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _find_data_file(raw_dir: Path) -> Path:
    preferred = [
        "sdwpf_2001_2112_full.parquet",
        "sdwpf_2001_2112_full.csv",
        "sdwpf_baidukddcup2022_full.CSV",
        "sdwpf_baidukddcup2022_full.csv",
        "wtbdata_245days.csv",
    ]
    for name in preferred:
        matches = list(raw_dir.glob(f"**/{name}"))
        if matches:
            return matches[0]
    candidates = []
    for suffix in ("*.parquet", "*.csv", "*.CSV"):
        for path in raw_dir.glob(f"**/{suffix}"):
            lowered = path.name.lower()
            if "location" in lowered or "turb" in lowered and "location" in lowered:
                continue
            if "sdwpf" in lowered or "wtbdata" in lowered:
                candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    raise FileNotFoundError(
        f"No SDWPF data file found under {raw_dir}. Expected wtbdata_245days.csv, "
        "sdwpf_baidukddcup2022_full.CSV, or sdwpf_2001_2112_full.parquet/csv."
    )


def _find_location_file(raw_dir: Path) -> Path | None:
    names = [
        "sdwpf_turb_location_elevation.csv",
        "sdwpf_baidukddcup2022_turb_location.CSV",
        "sdwpf_baidukddcup2022_turb_location.csv",
    ]
    for name in names:
        matches = list(raw_dir.glob(f"**/{name}"))
        if matches:
            return matches[0]
    for path in raw_dir.glob("**/*location*.csv"):
        return path
    for path in raw_dir.glob("**/*location*.CSV"):
        return path
    return None


def _read_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return _normalise_columns(pd.read_parquet(path))
    try:
        return _normalise_columns(pd.read_csv(path))
    except UnicodeDecodeError:
        return _normalise_columns(pd.read_csv(path, encoding="gbk"))


def _parse_tmstamp(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip()
    with_colon = raw.str.contains(":", regex=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if with_colon.any():
        parsed.loc[with_colon] = pd.to_datetime(raw.loc[with_colon], format="%H:%M", errors="coerce")
        still_missing = parsed.loc[with_colon].isna()
        if still_missing.any():
            idx = still_missing[still_missing].index
            parsed.loc[idx] = pd.to_datetime(raw.loc[idx], errors="coerce", format="mixed")
    if (~with_colon).any():
        numeric = pd.to_numeric(raw.loc[~with_colon], errors="coerce")
        hours = (numeric // 100).fillna(0).astype(int)
        minutes = (numeric % 100).fillna(0).astype(int)
        parsed.loc[~with_colon] = pd.Timestamp("2000-01-01") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
    return parsed.dt.hour.fillna(0).astype(int) * 60 + parsed.dt.minute.fillna(0).astype(int)


def _timestamp_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    lower_to_col = {str(col).lower(): col for col in df.columns}
    for key in ("timestamp", "datetime", "time", "date_time"):
        if key in lower_to_col:
            ts = pd.to_datetime(df[lower_to_col[key]], errors="coerce", format="mixed")
            if ts.notna().any():
                return pd.DatetimeIndex(ts)

    day_col = lower_to_col.get("day")
    tm_col = lower_to_col.get("tmstamp") or lower_to_col.get("timestamp")
    if day_col is None or tm_col is None:
        raise ValueError("Could not infer SDWPF timestamps. Expected timestamp or Day+Tmstamp columns.")
    day = pd.to_numeric(df[day_col], errors="coerce").fillna(1).astype(int)
    minutes = _parse_tmstamp(df[tm_col])
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(day - 1, unit="d") + pd.to_timedelta(minutes, unit="m")
    return pd.DatetimeIndex(ts)


def _read_location(raw_dir: Path, node_ids: list[str]) -> pd.DataFrame:
    location_path = _find_location_file(raw_dir)
    if location_path is None:
        meta = pd.DataFrame(index=node_ids)
        meta["node_id"] = node_ids
        meta["site_type"] = "wind"
        meta["type"] = "wind_power"
        return meta

    try:
        loc = _normalise_columns(pd.read_csv(location_path))
    except UnicodeDecodeError:
        loc = _normalise_columns(pd.read_csv(location_path, encoding="gbk"))
    lower_to_col = {str(col).lower(): col for col in loc.columns}
    id_col = lower_to_col.get("turbid") or lower_to_col.get("turb_id") or lower_to_col.get("turbine_id") or loc.columns[0]
    loc[id_col] = loc[id_col].astype(str)
    loc["node_id"] = "turbine_" + loc[id_col].astype(str)
    loc = loc.set_index("node_id")
    loc["site_type"] = "wind"
    loc["type"] = "wind_power"
    return loc.reindex(node_ids)


def _clean_target(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    return out.clip(lower=0.0)


def _normalise_train_only(occ_raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    train_end = max(1, int(len(occ_raw) * TRAIN_SPLIT_RATIO))
    source = occ_raw[:train_end]
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    occ_norm = (occ_raw - vmin) / (vmax - vmin + 1e-8)
    return occ_norm.astype(np.float32), vmin, vmax


def _build_from_frame(raw_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
    lower_to_col = {str(col).lower(): col for col in df.columns}
    turb_col = lower_to_col.get("turbid") or lower_to_col.get("turb_id") or lower_to_col.get("turbine_id")
    target_col = lower_to_col.get("patv")
    if turb_col is None or target_col is None:
        raise ValueError("SDWPF data must contain TurbID and Patv columns.")

    df = df.copy()
    df["timestamp"] = _timestamp_index(df)
    df = df.loc[df["timestamp"].notna()].sort_values(["timestamp", turb_col])
    df[turb_col] = df[turb_col].astype(str)
    df["node_id"] = "turbine_" + df[turb_col]
    df[target_col] = _clean_target(df[target_col])

    occupancy = df.pivot_table(index="timestamp", columns="node_id", values=target_col, aggfunc="mean")
    occupancy = occupancy.sort_index()
    occupancy = occupancy[~occupancy.index.duplicated(keep="last")]
    full_index = pd.date_range(occupancy.index.min(), occupancy.index.max(), freq="10min")
    occupancy = occupancy.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()
    node_ids = [str(col) for col in occupancy.columns]

    weather_cols = []
    for column in SCADA_COLUMNS:
        actual = lower_to_col.get(column.lower())
        if actual is not None and actual != target_col:
            weather_cols.append(actual)
    weather = pd.DataFrame(index=occupancy.index)
    if weather_cols:
        for column in weather_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        weather = df.groupby("timestamp")[weather_cols].mean().sort_index()
        weather = weather.reindex(occupancy.index).interpolate(limit_direction="both").ffill().bfill()

    node_meta = _read_location(raw_dir, node_ids)
    node_meta["node_id"] = node_ids
    node_meta["site_type"] = "wind"
    node_meta["type"] = "wind_power"
    node_meta = node_meta.reindex(node_ids)

    occ_raw = occupancy.to_numpy(dtype=np.float32)
    if np.isnan(occ_raw).any():
        occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    occ_norm, vmin, vmax = _normalise_train_only(occ_raw)

    adj = np.eye(len(node_ids), dtype=np.float32)
    x_cols = [col for col in node_meta.columns if str(col).lower() in {"x", "x_m", "longitude", "lon"}]
    y_cols = [col for col in node_meta.columns if str(col).lower() in {"y", "y_m", "latitude", "lat"}]
    if x_cols and y_cols:
        coords = node_meta[[x_cols[0], y_cols[0]]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if np.isfinite(coords).all():
            dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
            scale = np.nanmedian(dist[dist > 0]) if np.any(dist > 0) else 1.0
            adj = np.exp(-dist / max(float(scale), 1e-6)).astype(np.float32)

    return {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": pd.DatetimeIndex(occupancy.index),
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": adj,
        "weather": weather,
        "price": pd.DataFrame(),
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_only",
        "source_dataset": "SDWPF",
    }


def load_sdwpf(
    raw_dir: Path = Path("data/raw/aaai_2026/sdwpf"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    processed = Path(processed_path) if processed_path is not None else Path("data/processed/sdwpf.pkl")
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as f:
            return pickle.load(f)

    raw_dir = Path(raw_dir)
    data_file = _find_data_file(raw_dir)
    frame = _read_data(data_file)
    result = _build_from_frame(raw_dir, frame)
    result["source_file"] = str(data_file)
    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result
