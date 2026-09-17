"""Load the audited AEMO same-system Load/Wind/PV/Net-load tables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


TASK_COLUMNS = {
    "load": "load",
    "wind": "wind",
    "pv": "pv",
    "net_load": "net_load",
}
FORECAST_COLUMNS = {
    "load": "load_forecast",
    "wind": "wind_forecast",
    "pv": "pv_forecast",
    "net_load": "net_load_forecast",
}
ISSUED_COLUMNS = tuple(f"issued_{name}" for name in TASK_COLUMNS)
RELEVANT_NOTICE_TYPES = {
    "ADMINISTERED PRICE CAP",
    "CONSTRAINTS",
    "INTER-REGIONAL TRANSFER",
    "LOAD RESTORE",
    "LOAD SHED",
    "MARKET INTERVENTION",
    "MARKET SUSPENSION",
    "NON-CONFORMANCE",
    "POWER SYSTEM EVENTS",
    "PROTECTED EVENT",
    "RECALL GEN CAPACITY",
    "RECLASSIFY CONTINGENCY",
    "RESERVE NOTICE",
}


def _default_root() -> Path:
    configured = os.environ.get("AEMO_UNIFIED_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "outputs" / "aemo_unified_poc"


def _table_files(root: Path, name: str) -> list[Path]:
    if root.is_file():
        return [root] if root.name == name else []
    direct = root / name
    if direct.exists():
        return [direct]
    return sorted(root.glob(f"**/{name}"))


def _read_tables(root: Path, name: str) -> pd.DataFrame:
    files = _table_files(root, name)
    if not files:
        raise FileNotFoundError(f"No {name} found under {root}")
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=2)
def _read_market_notices(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    frame["regions"] = frame["regions"].fillna("NEM").astype(str)
    frame["type_id"] = frame["type_id"].fillna("").astype(str)
    frame["reason"] = frame["reason"].fillna("").astype(str)
    frame["external_reference"] = frame["external_reference"].fillna("").astype(str)
    return frame.loc[frame["type_id"].isin(RELEVANT_NOTICE_TYPES)].sort_values("available_at")


def _market_notice_contexts(
    path: str,
    *,
    origins: pd.DatetimeIndex,
    regions: list[str],
) -> dict[tuple[pd.Timestamp, str], str]:
    notices = _read_market_notices(path)
    contexts: dict[tuple[pd.Timestamp, str], str] = {}
    for region in regions:
        region_notices = notices.loc[
            notices["regions"].eq("NEM")
            | notices["regions"].str.split("|", regex=False).map(lambda values: region in values)
        ].reset_index(drop=True)
        notice_times = region_notices["available_at"].to_numpy(dtype="datetime64[ns]")
        for origin in origins:
            upper = int(np.searchsorted(notice_times, np.datetime64(origin), side="right"))
            lower_time = np.datetime64(origin - pd.Timedelta(hours=72))
            lower = int(np.searchsorted(notice_times, lower_time, side="right"))
            selected = region_notices.iloc[lower:upper].tail(3)
            entries = []
            for row in selected.itertuples(index=False):
                subject = row.external_reference or row.reason
                entries.append(f"[{row.type_id}] {subject}. {row.reason[:600]}")
            contexts[(pd.Timestamp(origin), region)] = " ".join(entries)
    return contexts


def _daily_split_indices(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    steps_per_day = 48
    complete_days = len(timestamps) // steps_per_day
    if complete_days < 5 or complete_days * steps_per_day != len(timestamps):
        raise ValueError("AEMO data must cover at least five complete days")
    train_days = max(1, int(np.floor(complete_days * 0.6)))
    val_days = max(1, int(np.floor(complete_days * 0.2)))
    if train_days + val_days >= complete_days:
        val_days = 1
        train_days = complete_days - 2
    train_end = train_days * steps_per_day
    val_end = (train_days + val_days) * steps_per_day
    return {
        "train": np.arange(0, train_end, dtype=np.int64),
        "val": np.arange(train_end, val_end, dtype=np.int64),
        "test": np.arange(val_end, len(timestamps), dtype=np.int64),
    }


def _complete_actual_grid(actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    actuals = actuals.copy()
    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"])
    actuals["REGIONID"] = actuals["REGIONID"].astype(str)
    actuals = actuals.sort_values("timestamp").drop_duplicates(["timestamp", "REGIONID"], keep="last")
    regions = sorted(actuals["REGIONID"].unique().tolist())
    timestamps = pd.date_range(actuals["timestamp"].min(), actuals["timestamp"].max(), freq="30min")
    expected = pd.MultiIndex.from_product([timestamps, regions], names=["timestamp", "REGIONID"])
    complete = actuals.set_index(["timestamp", "REGIONID"]).reindex(expected)
    missing = complete[list(TASK_COLUMNS.values())].isna().any(axis=1)
    if missing.any():
        first = complete.index[missing][0]
        raise ValueError(f"AEMO actual grid is incomplete; first missing row is {first}")
    return complete.reset_index(), timestamps, regions


def _node_scales(
    actuals: pd.DataFrame,
    *,
    timestamps: pd.DatetimeIndex,
    train_indices: np.ndarray,
    regions: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    train_end = timestamps[int(train_indices.max())]
    source = actuals.loc[actuals["timestamp"].le(train_end)]
    scales: dict[str, dict[str, tuple[float, float]]] = {}
    for region in regions:
        region_values = source.loc[source["REGIONID"].eq(region)]
        scales[region] = {}
        for task, column in TASK_COLUMNS.items():
            values = pd.to_numeric(region_values[column], errors="coerce").dropna()
            low = float(values.min())
            high = float(values.max())
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = 0.0, 1.0
            scales[region][task] = (low, high)
    return scales


def _wind_capacity_frame(
    files: list[Path],
    *,
    timestamps: pd.DatetimeIndex,
    regions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for path in files:
        frame = pd.read_csv(path)
        label = path.parent.name
        try:
            effective_month = pd.Timestamp(f"{label}-01")
        except ValueError:
            effective_month = timestamps.min().normalize().replace(day=1)
        frame["effective_month"] = effective_month
        records.append(frame)
    if not records:
        raise FileNotFoundError("AEMO wind capacity files are required for capacity-factor targets")
    monthly = pd.concat(records, ignore_index=True)
    monthly = monthly.sort_values("effective_month").drop_duplicates(
        ["effective_month", "REGIONID"], keep="last"
    )
    pivot = monthly.pivot(index="effective_month", columns="REGIONID", values="wind_capacity")
    months = pd.DatetimeIndex(timestamps.to_period("M").to_timestamp())
    expanded = pivot.reindex(pd.DatetimeIndex(months.unique()).sort_values()).ffill().bfill()
    capacity = expanded.reindex(months).set_axis(timestamps)
    capacity = capacity.reindex(columns=regions)
    if capacity.isna().any().any() or (capacity <= 0.0).any().any():
        raise ValueError("AEMO wind capacity must be positive for every timestamp and region")
    return capacity.astype(np.float32), monthly


def _standardize_context_value(value: pd.Series, low: float, high: float) -> pd.Series:
    scale = max(float(high) - float(low), 1e-6)
    return (pd.to_numeric(value, errors="coerce") - float(low)) / scale


def _complete_issued_origins(
    forecasts: pd.DataFrame,
    *,
    regions: list[str],
    horizon_steps: int = 336,
) -> pd.DatetimeIndex:
    """Retain origins with a complete seven-day forecast for every NEM region."""

    required = list(FORECAST_COLUMNS.values())
    frame = forecasts.copy()
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame["REGIONID"] = frame["REGIONID"].astype(str)
    lead = (frame["valid_time"] - frame["forecast_origin"]).dt.total_seconds() / (30.0 * 60.0)
    frame["lead_step"] = np.rint(lead).astype(np.int64)
    frame = frame.loc[
        frame["REGIONID"].isin(regions)
        & frame["lead_step"].between(1, horizon_steps)
        & frame[required].notna().all(axis=1)
    ]
    if frame.duplicated(["forecast_origin", "REGIONID", "lead_step"]).any():
        raise ValueError("AEMO issued forecasts contain duplicate origin-region-lead rows")
    coverage = frame.groupby(["forecast_origin", "REGIONID"])["lead_step"].agg(["count", "min", "max"])
    complete_pairs = coverage.loc[
        coverage["count"].eq(horizon_steps)
        & coverage["min"].eq(1)
        & coverage["max"].eq(horizon_steps)
    ]
    origin_counts = complete_pairs.reset_index().groupby("forecast_origin")["REGIONID"].nunique()
    origins = pd.DatetimeIndex(origin_counts.loc[origin_counts.eq(len(regions))].index).sort_values()
    if origins.empty:
        raise ValueError("AEMO has no origins with complete seven-day issued forecasts")
    return origins


def _versioned_exogenous(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    scales: dict[str, dict[str, tuple[float, float]]],
    wind_capacity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = forecasts.copy()
    forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
    forecasts["valid_time"] = pd.to_datetime(forecasts["valid_time"])
    forecasts["node_id"] = forecasts["REGIONID"].astype(str)
    actuals = actuals.copy()
    actuals["valid_time"] = pd.to_datetime(actuals["timestamp"])
    actuals["node_id"] = actuals["REGIONID"].astype(str)
    for region, task_scales in scales.items():
        issued_mask = forecasts["node_id"].eq(region)
        oracle_mask = actuals["node_id"].eq(region)
        for task, (low, high) in task_scales.items():
            output = f"issued_{task}"
            if task == "wind":
                issued_times = pd.DatetimeIndex(forecasts.loc[issued_mask, "valid_time"])
                oracle_times = pd.DatetimeIndex(actuals.loc[oracle_mask, "valid_time"])
                capacity_series = wind_capacity[region]
                issued_lookup = capacity_series.reindex(
                    capacity_series.index.union(pd.DatetimeIndex(issued_times.unique()))
                ).ffill().bfill()
                oracle_lookup = capacity_series.reindex(
                    capacity_series.index.union(pd.DatetimeIndex(oracle_times.unique()))
                ).ffill().bfill()
                issued_capacity = issued_lookup.loc[issued_times].to_numpy()
                oracle_capacity = oracle_lookup.loc[oracle_times].to_numpy()
                forecasts.loc[issued_mask, output] = (
                    pd.to_numeric(forecasts.loc[issued_mask, FORECAST_COLUMNS[task]], errors="coerce").to_numpy()
                    / issued_capacity
                )
                actuals.loc[oracle_mask, output] = (
                    pd.to_numeric(actuals.loc[oracle_mask, TASK_COLUMNS[task]], errors="coerce").to_numpy()
                    / oracle_capacity
                )
            else:
                forecasts.loc[issued_mask, output] = _standardize_context_value(
                    forecasts.loc[issued_mask, FORECAST_COLUMNS[task]], low, high
                )
                actuals.loc[oracle_mask, output] = _standardize_context_value(
                    actuals.loc[oracle_mask, TASK_COLUMNS[task]], low, high
                )
    issued_fields = ["forecast_origin", "valid_time", "node_id", *ISSUED_COLUMNS]
    for provenance in ("run_time", "version_time"):
        if provenance in forecasts.columns:
            issued_fields.append(provenance)
    issued = forecasts[issued_fields].sort_values(["forecast_origin", "valid_time", "node_id"])
    oracle = actuals[["valid_time", "node_id", *ISSUED_COLUMNS]].sort_values(["valid_time", "node_id"])
    return issued.reset_index(drop=True), oracle.reset_index(drop=True)


def load_aemo_unified_task(task: str) -> dict:
    if task not in TASK_COLUMNS:
        raise ValueError(f"Unknown AEMO task {task!r}")
    root = _default_root()
    actuals_raw = _read_tables(root, "aemo_unified_actuals.csv")
    forecasts = _read_tables(root, "aemo_unified_issued_forecasts.csv")
    actuals, timestamps, regions = _complete_actual_grid(actuals_raw)
    eligible_origins = _complete_issued_origins(forecasts, regions=regions)
    split_indices = _daily_split_indices(timestamps)
    capacity_files = _table_files(root, "aemo_unified_capacity.csv")
    wind_capacity, capacity_history = _wind_capacity_frame(
        capacity_files,
        timestamps=timestamps,
        regions=regions,
    )
    capacity = capacity_history.sort_values("effective_month").drop_duplicates("REGIONID", keep="last").set_index("REGIONID")
    scales = _node_scales(
        actuals,
        timestamps=timestamps,
        train_indices=split_indices["train"],
        regions=regions,
    )
    for region in regions:
        scales[region]["wind"] = (0.0, 1.0)

    raw_values = actuals.pivot(index="timestamp", columns="REGIONID", values=TASK_COLUMNS[task])
    raw_values = raw_values.reindex(index=timestamps, columns=regions)
    normalized = np.empty(raw_values.shape, dtype=np.float32)
    node_norm: dict[str, dict[str, float]] = {}
    for node_idx, region in enumerate(regions):
        low, high = scales[region][task]
        if task == "wind":
            normalized[:, node_idx] = (
                pd.to_numeric(raw_values[region], errors="coerce").to_numpy(np.float32)
                / wind_capacity[region].to_numpy(np.float32)
            )
        else:
            normalized[:, node_idx] = _standardize_context_value(raw_values[region], low, high).to_numpy(np.float32)
        node_norm[region] = {"min": low, "max": high}

    issued, oracle = _versioned_exogenous(
        forecasts,
        actuals,
        scales=scales,
        wind_capacity=wind_capacity,
    )
    node_meta = pd.DataFrame(index=regions)
    node_meta["region"] = regions
    node_meta["market"] = "AEMO NEM"
    node_meta["site_type"] = f"aemo_{task}"
    node_meta["timezone"] = "Australia/NEM regional time"
    centroids = {
        "NSW1": (-32.2, 147.0),
        "QLD1": (-22.5, 144.5),
        "VIC1": (-36.5, 144.5),
        "SA1": (-30.0, 135.0),
        "TAS1": (-42.0, 146.5),
    }
    node_meta["latitude"] = [centroids.get(region, (np.nan, np.nan))[0] for region in regions]
    node_meta["longitude"] = [centroids.get(region, (np.nan, np.nan))[1] for region in regions]
    if task == "wind" and "wind_capacity" in capacity:
        node_meta["capacity"] = capacity["wind_capacity"].reindex(regions)
    elif task == "pv" and "utility_pv_capacity" in capacity:
        node_meta["capacity"] = capacity["utility_pv_capacity"].reindex(regions)

    payload = {
        "occupancy": normalized,
        "timestamps": timestamps,
        "node_ids": regions,
        "node_meta": node_meta,
        "node_norm": node_norm,
        "norm_min": 0.0,
        "norm_max": 1.0,
        "split_indices": split_indices,
        "forecast_lead_steps": 1,
        "forecast_origin_from_history_end": True,
        "forecast_target_start_minute_of_day": 30,
        "forecast_origin_filter": eligible_origins,
        "issued_exogenous": issued,
        "oracle_exogenous": oracle,
        "issued_exogenous_columns": list(ISSUED_COLUMNS),
        "covariate_labels": list(ISSUED_COLUMNS),
        "issued_forecast_metadata": {
            "source": "AEMO STPASA and rooftop-PV forecast vintages",
            "selection": "latest forecast_origin at or before the common origin",
            "valid_time_resolution_minutes": 30,
            "data_root": str(root),
        },
        "protocol": {
            "system": "AEMO NEM",
            "task": task,
            "common_regions": regions,
            "forecast_origin": "daily 00:00; first target interval ends at 00:30",
            "eligible_complete_origins": int(len(eligible_origins)),
            "origin_eligibility": "all five regions have 336 issued half-hour leads",
        },
    }
    notice_path = os.environ.get("AEMO_MARKET_NOTICE_PATH")
    if notice_path:
        notice_path = str(Path(notice_path).expanduser().resolve())
        payload["issued_text_context"] = _market_notice_contexts(
            notice_path,
            origins=eligible_origins,
            regions=regions,
        )
        payload["issued_text_metadata"] = {
            "source": "AEMO MARKETNOTICEDATA",
            "path": notice_path,
            "causal_cutoff": "available_at <= forecast origin",
            "lookback_hours": 72,
            "max_notices": 3,
            "notice_types": sorted(RELEVANT_NOTICE_TYPES),
        }
    if task == "wind":
        payload["target_offset_frame"] = pd.DataFrame(0.0, index=timestamps, columns=regions)
        payload["target_scale_frame"] = wind_capacity.reindex(index=timestamps, columns=regions)
    return payload


def load_aemo_load() -> dict:
    return load_aemo_unified_task("load")


def load_aemo_wind() -> dict:
    return load_aemo_unified_task("wind")


def load_aemo_pv() -> dict:
    return load_aemo_unified_task("pv")


def load_aemo_net_load() -> dict:
    return load_aemo_unified_task("net_load")
