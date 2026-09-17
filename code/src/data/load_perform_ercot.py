"""Load processed ARPA-E PERFORM ERCOT four-task probability data."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


TASK_COLUMNS = {"load": "load", "wind": "wind", "pv": "pv", "net_load": "net_load"}
ISSUED_COLUMNS = tuple(f"issued_{task}" for task in TASK_COLUMNS)


def _root() -> Path:
    configured = os.environ.get("PERFORM_ERCOT_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "outputs" / "perform_ercot_poc"


def _daily_splits(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    days = pd.DatetimeIndex(timestamps.normalize().unique()).sort_values()
    train_days = int(np.floor(len(days) * 0.6))
    val_days = int(np.floor(len(days) * 0.2))
    if train_days < 14 or val_days < 7 or len(days) - train_days - val_days < 7:
        raise ValueError("PERFORM ERCOT must retain adequate chronological train/val/test days")
    train_end = days[train_days - 1]
    val_end = days[train_days + val_days - 1]
    normalized = timestamps.normalize()
    return {
        "train": np.flatnonzero(normalized <= train_end),
        "val": np.flatnonzero((normalized > train_end) & (normalized <= val_end)),
        "test": np.flatnonzero(normalized > val_end),
    }


def load_perform_ercot_task(task: str) -> dict:
    if task not in TASK_COLUMNS:
        raise ValueError(f"Unknown PERFORM task {task!r}")
    root = _root()
    actuals = pd.read_csv(root / "perform_ercot_actuals.csv", low_memory=False)
    issued = pd.read_csv(root / "perform_ercot_issued_forecasts.csv", low_memory=False)
    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"])
    issued["forecast_origin"] = pd.to_datetime(issued["forecast_origin"])
    issued["valid_time"] = pd.to_datetime(issued["valid_time"])
    node_id = str(actuals["node_id"].iloc[0])
    actuals = actuals.loc[actuals["node_id"].astype(str).eq(node_id)].sort_values("timestamp")
    actuals = actuals.drop_duplicates("timestamp", keep="last")
    timestamps = pd.date_range(actuals["timestamp"].min(), actuals["timestamp"].max(), freq="1h")
    actuals = actuals.set_index("timestamp").reindex(timestamps)
    if actuals[list(TASK_COLUMNS.values())].isna().any().any():
        raise ValueError("PERFORM ERCOT actual time series is not a complete hourly grid")
    splits = _daily_splits(timestamps)
    train = actuals.iloc[splits["train"]]

    scales = {}
    for name, column in TASK_COLUMNS.items():
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        low, high = float(values.min()), float(values.max())
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = 0.0, 1.0
        scales[name] = (low, high)
    low, high = scales[task]
    occupancy = ((pd.to_numeric(actuals[TASK_COLUMNS[task]]) - low) / (high - low)).to_numpy(
        dtype=np.float32
    )[:, None]

    issued["node_id"] = issued["node_id"].astype(str)
    oracle = actuals.reset_index(names="valid_time")
    oracle["node_id"] = node_id
    for name, (field_low, field_high) in scales.items():
        field = f"issued_{name}"
        issued[field] = (
            pd.to_numeric(issued[f"{name}_forecast"], errors="coerce") - field_low
        ) / (field_high - field_low)
        oracle[field] = (
            pd.to_numeric(oracle[TASK_COLUMNS[name]], errors="coerce") - field_low
        ) / (field_high - field_low)
    issued = issued.sort_values(["forecast_origin", "valid_time"])
    complete_rows = issued[["forecast_origin", "valid_time", *ISSUED_COLUMNS]].dropna()
    origin_coverage = complete_rows.groupby("forecast_origin")["valid_time"].nunique()
    complete_origins = pd.DatetimeIndex(origin_coverage.index[origin_coverage.ge(24)]).sort_values()
    if len(complete_origins) < 100:
        raise ValueError("PERFORM ERCOT requires at least 100 complete 24-hour forecast origins")
    first_valid = (
        complete_rows.loc[complete_rows["forecast_origin"].eq(complete_origins[0]), "valid_time"]
        .sort_values()
        .iloc[0]
    )
    step_minutes = int(round((timestamps[1] - timestamps[0]).total_seconds() / 60.0))
    start_offset_minutes = int((first_valid - timestamps[0]).total_seconds() / 60.0) % (24 * 60)
    forecast_start_offset_steps = start_offset_minutes // step_minutes
    issued_store = issued[["forecast_origin", "valid_time", "node_id", *ISSUED_COLUMNS]].copy()
    oracle_store = oracle[["valid_time", "node_id", *ISSUED_COLUMNS]].copy()
    lead_steps = int(round(float(issued["lead_hours"].min())))
    if lead_steps < 1:
        raise ValueError("PERFORM day-ahead issue-to-valid lead must be positive")
    node_meta = pd.DataFrame(
        {"region": ["ERCOT"], "market": ["ARPA-E PERFORM"], "site_type": [f"perform_{task}"], "timezone": ["UTC"]},
        index=[node_id],
    )
    return {
        "occupancy": occupancy,
        "timestamps": timestamps,
        "node_ids": [node_id],
        "node_meta": node_meta,
        "node_norm": {node_id: {"min": low, "max": high}},
        "norm_min": 0.0,
        "norm_max": 1.0,
        "split_indices": splits,
        "forecast_lead_steps": lead_steps,
        "forecast_origin_from_history_end": True,
        "forecast_origin_filter": complete_origins,
        "forecast_target_start_minute_of_day": int(first_valid.hour * 60 + first_valid.minute),
        "forecast_start_offset_steps": int(forecast_start_offset_steps),
        "issued_exogenous": issued_store,
        "oracle_exogenous": oracle_store,
        "issued_exogenous_columns": list(ISSUED_COLUMNS),
        "covariate_labels": list(ISSUED_COLUMNS),
        "issued_forecast_metadata": {
            "source": "ARPA-E PERFORM ERCOT BA-level day-ahead deterministic or median forecasts",
            "probability_reference": str((root / "perform_ercot_official_quantiles.csv").resolve()),
            "selection": "exact daily issue_time and hourly forecast_time",
            "timezone": "UTC",
        },
        "protocol": {"system": "ERCOT", "task": task, "probability_levels": "1%-99%"},
    }


def load_perform_load() -> dict:
    return load_perform_ercot_task("load")


def load_perform_wind() -> dict:
    return load_perform_ercot_task("wind")


def load_perform_pv() -> dict:
    return load_perform_ercot_task("pv")


def load_perform_net_load() -> dict:
    return load_perform_ercot_task("net_load")
