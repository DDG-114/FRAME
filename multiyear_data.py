"""AEMO 2022--2026 payloads with forecast-origin-safe chronological splits."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("load", "wind", "pv", "net_load")
REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
ISSUED = tuple(f"issued_{task}" for task in TASKS)
MONTHLY_QUICK_DAYS = (5, 12, 19, 26)
MONTHLY_QUICK_H2_HOURS = (0, 6, 12, 18)


def _root() -> Path:
    return Path(os.environ["AEMO_MULTYEAR_DATA_DIR"]).expanduser().resolve()


def _plan_path() -> Path:
    configured = os.environ.get("AEMO_SPLIT_PLAN")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "split_plan.json"


@lru_cache(maxsize=1)
def split_plan() -> dict:
    return json.loads(_plan_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def actuals(cadence_minutes: int) -> pd.DataFrame:
    frame = pd.read_parquet(_root() / "dataset" / f"actuals_{cadence_minutes}min.parquet")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["REGIONID"] = frame["REGIONID"].astype(str)
    frame = frame.loc[frame.REGIONID.isin(REGIONS)].sort_values(["timestamp", "REGIONID"])
    if frame.duplicated(["timestamp", "REGIONID"]).any():
        raise ValueError("Actual data contain duplicate timestamp-region rows")
    return frame


@lru_cache(maxsize=2)
def capacity(cadence_minutes: int) -> dict[str, pd.DataFrame]:
    rows = []
    for path in sorted((_root() / "data" / "native5min").glob("????-??/capacity.csv")):
        frame = pd.read_csv(path)
        frame["effective_month"] = pd.Timestamp(f"{path.parent.name}-01")
        rows.append(frame)
    monthly = pd.concat(rows, ignore_index=True).sort_values("effective_month")
    timestamps = pd.DatetimeIndex(sorted(actuals(cadence_minutes).timestamp.unique()))
    result = {}
    for task, column in (("wind", "Wind"), ("pv", "Solar")):
        pivot = monthly.pivot(index="effective_month", columns="REGIONID", values=column)
        expanded = pivot.reindex(timestamps.to_period("M").to_timestamp()).set_axis(timestamps)
        result[task] = expanded.reindex(columns=REGIONS).ffill().bfill().astype(np.float32)
    if (result["wind"] <= 0).any().any():
        raise ValueError("Wind capacity must be positive for every region and timestamp")
    return result


@lru_cache(maxsize=1)
def training_scales() -> dict[str, dict[str, tuple[float, float]]]:
    frame = actuals(30)
    end = pd.Timestamp(next(item["origin_end"] for item in split_plan()["splits"] if item["name"] == "train"))
    frame = frame.loc[frame.timestamp.lt(end)]
    scales: dict[str, dict[str, tuple[float, float]]] = {}
    wind_capacity = capacity(30)["wind"]
    for region in REGIONS:
        part = frame.loc[frame.REGIONID.eq(region)].set_index("timestamp")
        scales[region] = {"wind": (0.0, 1.0)}
        for task in ("load", "pv", "net_load"):
            values = pd.to_numeric(part[task], errors="coerce").dropna()
            low, high = float(values.min()), float(values.max())
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError(f"Invalid training scale for {region} {task}")
            if high == low:
                high = low + 1.0
            scales[region][task] = (low, high)
        wind = pd.to_numeric(part.wind, errors="coerce").to_numpy(np.float32)
        cap = wind_capacity[region].reindex(part.index).to_numpy(np.float32)
        if not np.isfinite(wind / cap).all():
            raise ValueError(f"Invalid wind capacity factor for {region}")
    return scales


def _normalize(values: pd.Series, task: str, region: str, times: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(np.float32)
    if task == "wind":
        cadence = 5 if times.dt.minute.mod(5).eq(0).all() and not times.dt.minute.mod(30).eq(0).all() else 30
        cap = capacity(cadence)["wind"][region]
        requested = pd.DatetimeIndex(times)
        lookup = cap.reindex(cap.index.union(requested.unique())).ffill().bfill()
        return raw / lookup.reindex(requested).to_numpy(np.float32)
    low, high = training_scales()[region][task]
    return (raw - low) / max(high - low, 1e-6)


def _issued_files(cadence_minutes: int) -> list[Path]:
    dataset = _root() / "dataset"
    if cadence_minutes == 5:
        return sorted((dataset / "official_2h_inputs").glob("aligned_inputs_2h_????-??.parquet"))
    return sorted(dataset.glob("issued_30min_????-??.parquet"))


@lru_cache(maxsize=2)
def issued(cadence_minutes: int) -> pd.DataFrame:
    files = _issued_files(cadence_minutes)
    if not files:
        raise FileNotFoundError(f"No issued forecast files for {cadence_minutes}-minute data")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame = frame.rename(columns={"region": "REGIONID"})
    frame["forecast_origin"] = pd.to_datetime(frame.forecast_origin)
    frame["valid_time"] = pd.to_datetime(frame.valid_time)
    frame["REGIONID"] = frame.REGIONID.astype(str)
    frame = frame.loc[frame.REGIONID.isin(REGIONS)].copy()
    for task in TASKS:
        if cadence_minutes == 5:
            release = pd.to_datetime(frame[f"{task}_available_at"], errors="coerce")
            available = frame[f"{task}_available"].fillna(False).astype(bool)
        else:
            release = pd.to_datetime(frame["available_at"], errors="coerce")
            available = pd.to_numeric(frame[task], errors="coerce").notna()
        if (release.loc[available] > frame.loc[available, "forecast_origin"]).any():
            raise ValueError(f"{task} contains a release after forecast origin")
        frame[f"issued_{task}"] = np.nan
        for region in REGIONS:
            mask = available & frame.REGIONID.eq(region)
            frame.loc[mask, f"issued_{task}"] = _normalize(
                frame.loc[mask, task], task, region, frame.loc[mask, "valid_time"]
            )
    provenance = (
        [name for task in TASKS for name in (f"{task}_available", f"{task}_available_at")]
        if cadence_minutes == 5
        else ["available_at"]
    )
    keep = ["forecast_origin", "valid_time", "REGIONID", *ISSUED, *provenance]
    frame = frame[keep].rename(columns={"REGIONID": "node_id"})
    frame = frame.sort_values(["forecast_origin", "node_id", "valid_time"])
    if frame.duplicated(["forecast_origin", "node_id", "valid_time"]).any():
        raise ValueError("Issued data contain duplicate origin-region-lead rows")
    return frame.reset_index(drop=True)


@lru_cache(maxsize=2)
def issued_lookup(cadence_minutes: int) -> dict[tuple[pd.Timestamp, str], pd.DataFrame]:
    frame = issued(cadence_minutes)
    return {
        (pd.Timestamp(origin), str(node)): group.set_index("valid_time").sort_index()
        for (origin, node), group in frame.groupby(["forecast_origin", "node_id"], sort=False)
    }


def _ranges_for_loader() -> dict[str, list[list[str]]]:
    entries = {item["name"]: item for item in split_plan()["splits"]}
    return {
        "train": [[entries["train"]["origin_start"], entries["train"]["origin_end"]]],
        "val": [
            [entries["validation"]["origin_start"], entries["validation"]["origin_end"]],
            [entries["calibration"]["origin_start"], entries["calibration"]["origin_end"]],
        ],
        "test": [
            [entries["test"]["origin_start"], entries["test"]["origin_end"]],
            [entries["confirmation"]["origin_start"], entries["confirmation"]["origin_end"]],
        ],
    }


def _split_indices(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    plan = split_plan()
    maximum = pd.Timedelta(hours=int(plan["maximum_horizon_hours"]))
    history = pd.Timedelta(days=7)
    ranges = _ranges_for_loader()
    result = {}
    for split, windows in ranges.items():
        start = min(pd.Timestamp(window[0]) for window in windows) - history
        end = max(pd.Timestamp(window[1]) for window in windows) + maximum
        result[split] = np.flatnonzero((timestamps >= start) & (timestamps < end))
    return result


@lru_cache(maxsize=1)
def origin_manifest() -> pd.DataFrame:
    path = _root() / "dataset" / "split_manifest" / "forecast_origins.csv"
    frame = pd.read_csv(path, parse_dates=["forecast_origin"])
    return frame


def _origin_filter(hours: int, sample_mode: str) -> pd.DatetimeIndex:
    frame = origin_manifest().loc[origin_manifest().horizon_hours.eq(hours)].copy()
    if sample_mode == "quick":
        development = frame.split.isin(["train", "validation", "calibration"]) & frame.quick_selected
        diagnostic = frame.split.eq("test") & frame.diagnostic_test_selected
        frame = frame.loc[development | diagnostic]
    elif sample_mode == "monthly_quick":
        origin = frame.forecast_origin.dt
        selected_time = origin.day.isin(MONTHLY_QUICK_DAYS) & origin.minute.eq(0)
        if hours == 2:
            selected_time &= origin.hour.isin(MONTHLY_QUICK_H2_HOURS)
        else:
            selected_time &= origin.hour.eq(0)
        frame = frame.loc[
            frame.split.isin(["train", "validation", "calibration", "test", "confirmation"])
            & selected_time
        ]
    elif sample_mode == "full":
        frame = frame.loc[
            frame.split.isin(["train", "validation", "calibration", "test", "confirmation"])
        ]
    else:
        raise ValueError("AEMO_SAMPLE_MODE must be quick, monthly_quick, or full")
    return pd.DatetimeIndex(frame.forecast_origin.unique()).sort_values()


def load_multiyear_task(task: str, hours: int) -> dict:
    if task not in TASKS or hours not in (2, 24, 168):
        raise ValueError(f"Unsupported task/horizon: {task}/{hours}")
    cadence_minutes = 5 if hours == 2 else 30
    raw = actuals(cadence_minutes)
    timestamps = pd.DatetimeIndex(sorted(raw.timestamp.unique()))
    expected = pd.MultiIndex.from_product([timestamps, REGIONS], names=["timestamp", "REGIONID"])
    grid = raw.set_index(["timestamp", "REGIONID"]).reindex(expected)
    values = grid[task].unstack("REGIONID").reindex(columns=REGIONS)
    normalized = np.empty(values.shape, dtype=np.float32)
    node_norm = {}
    for index, region in enumerate(REGIONS):
        if task == "wind":
            normalized[:, index] = values[region].to_numpy(np.float32) / capacity(cadence_minutes)["wind"][region].to_numpy(np.float32)
            low, high = 0.0, 1.0
        else:
            low, high = training_scales()[region][task]
            normalized[:, index] = (values[region].to_numpy(np.float32) - low) / (high - low)
        node_norm[region] = {"min": low, "max": high}

    sample_mode = os.environ.get("AEMO_SAMPLE_MODE", "quick")
    metadata = pd.DataFrame(index=REGIONS)
    metadata["region"] = REGIONS
    metadata["market"] = "AEMO NEM"
    metadata["site_type"] = f"aemo_{task}"
    metadata["latitude"] = [-32.2, -22.5, -30.0, -42.0, -36.5]
    metadata["longitude"] = [147.0, 144.5, 135.0, 146.5, 144.5]
    train_time = pd.Timestamp("2023-12-01")
    if task in {"wind", "pv"}:
        metadata["capacity"] = capacity(cadence_minutes)[task].loc[train_time].reindex(REGIONS).to_numpy()

    payload = {
        "occupancy": normalized,
        "timestamps": timestamps,
        "node_ids": list(REGIONS),
        "node_meta": metadata,
        "node_norm": node_norm,
        "norm_min": 0.0,
        "norm_max": 1.0,
        "split_indices": _split_indices(timestamps),
        "split_origin_ranges": _ranges_for_loader(),
        "forecast_origin_filter": _origin_filter(hours, sample_mode),
        "forecast_origin_from_history_end": True,
        "forecast_start_offset_steps": 1,
        "forecast_lead_steps": 1,
        "issued_exogenous": issued(cadence_minutes),
        "_issued_exogenous_lookup": issued_lookup(cadence_minutes),
        "issued_exogenous_columns": list(ISSUED),
        "covariate_labels": list(ISSUED),
        "issued_forecast_metadata": {
            "selection": "latest archive value released no later than forecast origin",
            "missing_leads": "retained as zero-valued tokens with a per-field, per-lead mask",
            "cadence_minutes": cadence_minutes,
        },
        "protocol": {
            "period": "2022-2026",
            "sample_mode": sample_mode,
            "horizon_hours": hours,
            "split_plan": str(_plan_path()),
            "missing_actual_points": int(grid[list(TASKS)].isna().sum().sum()),
            "window_eligibility": "common complete history and targets from split_manifest",
        },
    }
    if task == "wind":
        payload["target_offset_frame"] = pd.DataFrame(0.0, index=timestamps, columns=REGIONS)
        payload["target_scale_frame"] = capacity(cadence_minutes)["wind"]
    return payload
