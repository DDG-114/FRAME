"""Loader for the Great Britain GSP-group net-load forecasting data."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAIN_SPLIT_RATIO = 0.7
DEFAULT_GROUPS = ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P")
TARGET_COLUMN = "node"
TIME_COLUMN = "targetTime"
ISSUE_COLUMN = "issueTime"
CONTEXT_COLUMNS = [
    "x2T_weighted.mean_cell",
    "x2T_weighted.sd_cell",
    "x2T_weighted.mean_pcell",
    "x2T_weighted.mean_p_max_point",
    "WindSpd100_weighted.mean_cell",
    "WindSpd10_weighted.mean_cell",
    "SSRD_mean_2",
    "SSRD_weighted.mean_cell",
    "TP_weighted.mean_cell",
    "LCC_weighted.mean_cell",
    "MCC_weighted.mean_cell",
    "HCC_weighted.mean_cell",
    "SolarCap",
    "n2ex",
    "EMBEDDED_WIND_CAPACITY",
]
WEATHER_MODES = ("compact", "full", "none")


def _normalise_train_only(occ_raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    train_end = max(1, int(len(occ_raw) * TRAIN_SPLIT_RATIO))
    source = occ_raw[:train_end]
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    occ_norm = (occ_raw - vmin) / (vmax - vmin + 1e-8)
    return occ_norm.astype(np.float32), vmin, vmax


def _find_rda(raw_dir: Path) -> Path:
    for name in ("GSPGroup_NetD_v3_2.Rda", "GSPGroup_NetD_v3_2.rda"):
        path = raw_dir / name
        if path.exists():
            return path
    matches = sorted(raw_dir.glob("**/*NetD*.Rda")) + sorted(raw_dir.glob("**/*NetD*.rda"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No GSPGroup_NetD .Rda file found under {raw_dir}.")


def _timestamp(values: pd.Series) -> pd.DatetimeIndex:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.8:
        finite = numeric[np.isfinite(numeric)]
        median_abs = float(finite.abs().median()) if len(finite) else 0.0
        if median_abs > 1e17:
            unit = "ns"
        elif median_abs > 1e14:
            unit = "us"
        elif median_abs > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.DatetimeIndex(pd.to_datetime(numeric, unit=unit, utc=True).dt.tz_convert(None))
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", utc=True, format="mixed").dt.tz_convert(None))


def _read_node_data_from_rda(path: Path) -> dict[str, pd.DataFrame]:
    import rdata

    parsed = rdata.parser.parse_file(path)
    converted = rdata.conversion.convert(parsed)
    if not isinstance(converted, dict) or "NodeData" not in converted:
        raise ValueError(f"{path} did not contain a NodeData object.")
    node_data = converted["NodeData"]
    if not isinstance(node_data, dict):
        raise ValueError("NodeData is not a dictionary/list-like object after conversion.")
    return {str(key): value for key, value in node_data.items() if isinstance(value, pd.DataFrame)}


def _candidate_parquet_dir(raw_dir: Path) -> Path | None:
    for path in [raw_dir / "processed_parquet", raw_dir / "parquet", raw_dir / "node_parquet"]:
        if path.exists() and any(path.glob("*.parquet")):
            return path
    return None


def _read_node_data(raw_dir: Path, *, groups: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    parquet_dir = _candidate_parquet_dir(raw_dir)
    if parquet_dir is not None:
        out: dict[str, pd.DataFrame] = {}
        for group in groups:
            path = parquet_dir / f"{group}.parquet"
            if path.exists():
                out[group] = pd.read_parquet(path)
        if out:
            return out
    return _read_node_data_from_rda(_find_rda(raw_dir))


def convert_gb_gsp_rda_to_parquet(
    raw_dir: Path = Path("data/raw/aaai_2026/gb_gsp_netload"),
    output_dir: Path | None = None,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
) -> dict[str, Any]:
    """Convert the large R NodeData object to one parquet file per GSP group."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir) if output_dir is not None else raw_dir / "processed_parquet"
    node_data = _read_node_data_from_rda(_find_rda(raw_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    keep_columns = [ISSUE_COLUMN, TIME_COLUMN, TARGET_COLUMN, "settlement_period", "dow", "doy", "clock_hour", *CONTEXT_COLUMNS]
    for group in groups:
        if group not in node_data:
            continue
        frame = node_data[group]
        cols = [col for col in keep_columns if col in frame.columns]
        slim = frame.loc[:, cols].copy()
        slim["gsp_group"] = group
        path = output_dir / f"{group}.parquet"
        slim.to_parquet(path, index=False)
        written[group] = {"path": str(path), "rows": int(len(slim)), "columns": list(slim.columns)}
    return {"output_dir": str(output_dir), "written": written}


def _group_frame_to_series(group: str, frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if TIME_COLUMN not in frame.columns or TARGET_COLUMN not in frame.columns:
        raise ValueError(f"GSP group {group} must contain {TIME_COLUMN!r} and {TARGET_COLUMN!r}.")
    working = frame.copy()
    working["timestamp"] = _timestamp(working[TIME_COLUMN])
    if ISSUE_COLUMN in working.columns:
        working["issue_timestamp"] = _timestamp(working[ISSUE_COLUMN])
        lead_hours = (working["timestamp"] - working["issue_timestamp"]).dt.total_seconds() / 3600.0
        day_ahead = (working["issue_timestamp"].dt.hour == 0) & lead_hours.between(22.0, 47.5)
        if day_ahead.any():
            working = working.loc[day_ahead].copy()
    working = working.loc[working["timestamp"].notna()].sort_values("timestamp")
    working[TARGET_COLUMN] = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")
    working = working.loc[working[TARGET_COLUMN].notna()]
    target = working.groupby("timestamp")[TARGET_COLUMN].mean().sort_index()

    context_cols = [col for col in CONTEXT_COLUMNS if col in working.columns]
    weather = pd.DataFrame(index=target.index)
    if context_cols:
        for col in context_cols:
            working[col] = pd.to_numeric(working[col], errors="coerce")
        weather = working.groupby("timestamp")[context_cols].mean().sort_index()
    return target.rename(f"gsp_{group}"), weather


def _combine_weather(
    weather_by_group: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    *,
    weather_mode: str,
) -> pd.DataFrame:
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode={weather_mode!r}; expected one of {WEATHER_MODES}.")
    if weather_mode == "none" or not weather_by_group:
        return pd.DataFrame(index=index)

    if weather_mode == "full":
        frames = [frame.add_prefix(f"gsp_{group}_") for group, frame in sorted(weather_by_group.items()) if not frame.empty]
        if not frames:
            return pd.DataFrame(index=index)
        weather = pd.concat(frames, axis=1).sort_index()
        return weather.reindex(index).interpolate(limit_direction="both").ffill().bfill()

    compact = pd.DataFrame(index=index)
    for column in CONTEXT_COLUMNS:
        pieces = []
        for group, frame in sorted(weather_by_group.items()):
            if column in frame.columns:
                pieces.append(frame[column].rename(group))
        if pieces:
            compact[column] = pd.concat(pieces, axis=1).mean(axis=1)
    if compact.empty:
        return pd.DataFrame(index=index)
    return compact.reindex(index).interpolate(limit_direction="both").ffill().bfill()


def _group_capacity(frame: pd.DataFrame) -> float:
    if "SolarCap" not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame["SolarCap"], errors="coerce")
    if values.notna().any():
        return float(values.max())
    return 0.0


def _build_from_node_data(
    node_data: dict[str, pd.DataFrame],
    *,
    groups: tuple[str, ...],
    weather_mode: str,
) -> dict[str, Any]:
    series_frames: list[pd.Series] = []
    weather_by_group: dict[str, pd.DataFrame] = {}
    used_groups: list[str] = []
    capacities: list[float] = []
    for group in groups:
        frame = node_data.get(group)
        if frame is None:
            continue
        target, weather = _group_frame_to_series(group, frame)
        series_frames.append(target)
        if not weather.empty:
            weather_by_group[group] = weather
        used_groups.append(group)
        capacities.append(_group_capacity(frame))
    if not series_frames:
        raise RuntimeError("No GB GSP groups were loaded.")

    occupancy = pd.concat(series_frames, axis=1).sort_index()
    full_index = pd.date_range(occupancy.index.min(), occupancy.index.max(), freq="30min")
    occupancy = occupancy.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()
    weather = _combine_weather(weather_by_group, pd.DatetimeIndex(occupancy.index), weather_mode=weather_mode)

    node_ids = [str(col) for col in occupancy.columns]
    node_meta = pd.DataFrame(
        {
            "node_id": node_ids,
            "gsp_group": used_groups,
            "site_type": "net_load",
            "type": "regional_net_load",
            "capacity": capacities,
        }
    ).set_index("node_id")
    occ_raw = occupancy.to_numpy(dtype=np.float32)
    occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    occ_norm, vmin, vmax = _normalise_train_only(occ_raw)
    return {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": pd.DatetimeIndex(occupancy.index),
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": np.eye(len(node_ids), dtype=np.float32),
        "weather": weather,
        "price": pd.DataFrame(),
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_only",
        "source_dataset": "GB GSP net-load",
        "source_groups": used_groups,
        "weather_mode": weather_mode,
        "available_weather_modes": list(WEATHER_MODES),
    }


def load_gb_gsp_netload(
    raw_dir: Path = Path("data/raw/aaai_2026/gb_gsp_netload"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    weather_mode: str = "compact",
) -> dict:
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode={weather_mode!r}; expected one of {WEATHER_MODES}.")
    processed = (
        Path(processed_path)
        if processed_path is not None
        else Path(f"data/processed/gb_gsp_netload_{weather_mode}.pkl")
    )
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("weather_mode") == weather_mode:
            return cached

    raw_dir = Path(raw_dir)
    node_data = _read_node_data(raw_dir, groups=groups)
    result = _build_from_node_data(node_data, groups=groups, weather_mode=weather_mode)
    try:
        result["source_file"] = str(_find_rda(raw_dir))
    except FileNotFoundError:
        result["source_file"] = str(raw_dir)
    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result
