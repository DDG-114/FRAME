"""Dataset preparation for the EnergyCA MVP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.build_splits import build_splits
from src.data.loaders import load_dataset
from src.energy_context.context import build_energy_context, context_dim, estimate_frequency_minutes
from src.energy_context.issued_availability import (
    IssuedAvailability,
    provenance_columns,
    select_issued_block,
)
from src.energy_context.llm_context import CONTEXT_EMBEDDING_MODES, apply_llm_context_embeddings


@dataclass
class EnergyExample:
    dataset_name: str
    dataset_label: int
    node_id: str
    node_label: int
    task_type: str
    task_id: int
    numeric: np.ndarray
    history_len: int
    horizon: int
    history_exog_dim: int
    future_exog_dim: int
    context: np.ndarray
    target: np.ndarray
    target_raw: np.ndarray
    target_offset: np.ndarray
    target_scale: np.ndarray
    history_raw: np.ndarray
    norm_min: float
    norm_max: float
    context_text: str
    context_segments: list[str]
    history_timestamps: list[str]
    future_timestamps: list[str]
    forecast_start: str | None
    future_exog_mask: np.ndarray
    future_exog_available_at: np.ndarray
    future_exog_coverage: float
    latest_issued_available_at: str | None


class EnergyWindowDataset(Dataset):
    def __init__(self, examples: list[EnergyExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {
            "numeric": torch.tensor(ex.numeric, dtype=torch.float32),
            "history_len": torch.tensor(ex.history_len, dtype=torch.long),
            "horizon": torch.tensor(ex.horizon, dtype=torch.long),
            "history_exog_dim": torch.tensor(ex.history_exog_dim, dtype=torch.long),
            "future_exog_dim": torch.tensor(ex.future_exog_dim, dtype=torch.long),
            "context": torch.tensor(ex.context, dtype=torch.float32),
            "target": torch.tensor(ex.target, dtype=torch.float32),
            "target_raw": torch.tensor(ex.target_raw, dtype=torch.float32),
            "norm_min": torch.tensor(ex.norm_min, dtype=torch.float32),
            "norm_max": torch.tensor(ex.norm_max, dtype=torch.float32),
            "task_id": torch.tensor(ex.task_id, dtype=torch.long),
            "dataset_label": torch.tensor(ex.dataset_label, dtype=torch.long),
            "node_label": torch.tensor(ex.node_label, dtype=torch.long),
            "dataset_name": ex.dataset_name,
            "task_type": ex.task_type,
            "node_id": ex.node_id,
            "context_text": ex.context_text,
            "context_segments": ex.context_segments,
            "history_timestamps": ex.history_timestamps,
            "future_timestamps": ex.future_timestamps,
            "forecast_start": ex.forecast_start,
        }


def _time_features(timestamps) -> np.ndarray:
    try:
        hours = np.asarray(timestamps.hour, dtype=np.float32) + np.asarray(timestamps.minute, dtype=np.float32) / 60.0
        dows = np.asarray(timestamps.dayofweek, dtype=np.float32)
    except Exception:
        return np.zeros((len(timestamps), 4), dtype=np.float32)
    return np.stack(
        [
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.sin(2 * np.pi * dows / 7.0),
            np.cos(2 * np.pi * dows / 7.0),
        ],
        axis=-1,
    ).astype(np.float32)


def _select_evenly(values: list[int], max_count: int | None) -> list[int]:
    if max_count is None or max_count <= 0 or len(values) <= max_count:
        return values
    indices = np.linspace(0, len(values) - 1, num=int(max_count), dtype=int)
    return [values[int(i)] for i in np.unique(indices)]


def _node_ids_for(data: dict, n_nodes: int) -> list[str]:
    node_ids = data.get("node_ids")
    if node_ids:
        return [str(node_id) for node_id in node_ids]
    return [str(idx) for idx in range(n_nodes)]


def _forecast_lead_steps_for(data: dict) -> int:
    value = data.get("forecast_lead_steps")
    if value is not None:
        try:
            lead_steps = int(value or 1)
            if lead_steps >= 1:
                return lead_steps
        except Exception:
            pass
    protocol = data.get("protocol")
    if isinstance(protocol, dict):
        try:
            lead_steps = int(protocol.get("forecast_lead_steps", 1) or 1)
            if lead_steps >= 1:
                return lead_steps
        except Exception:
            pass
    return 1


def _raw_from_norm(values: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return values * (float(norm_max) - float(norm_min)) + float(norm_min)


def _timestamp_strings(values) -> list[str]:
    try:
        return [pd.Timestamp(value).isoformat() for value in values]
    except Exception:
        return [str(value) for value in values]


def _safe_nanmean(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.isnan(values).all():
        return default
    out = float(np.nanmean(values))
    return out if np.isfinite(out) else default


def _safe_nanstd(values: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.isnan(values).all():
        return default
    out = float(np.nanstd(values))
    return out if np.isfinite(out) else default


def _safe_nanpercentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.isnan(values).all():
        return default
    out = float(np.nanpercentile(values, q))
    return out if np.isfinite(out) else default


def _node_stat_value(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(features.get(key, default))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _level_band(value: float) -> str:
    if value <= 0.05:
        return "near-zero"
    if value <= 0.25:
        return "low"
    if value <= 0.60:
        return "medium"
    if value <= 0.85:
        return "high"
    return "very-high"


def _trend_band(slope: float) -> str:
    if slope >= 0.05:
        return "rising-fast"
    if slope >= 0.01:
        return "rising"
    if slope <= -0.05:
        return "falling-fast"
    if slope <= -0.01:
        return "falling"
    return "flat"


def _history_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float32)
    if y.size < 2:
        return 0.0
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=_safe_nanmean(y))
    x = np.linspace(-0.5, 0.5, num=len(y), dtype=np.float32)
    denom = float(np.sum(x * x))
    if denom <= 0.0:
        return 0.0
    out = float(np.sum(x * (y - float(np.mean(y)))) / denom)
    return out if np.isfinite(out) else 0.0


def _series_value(values: np.ndarray, index: int, *, default: float) -> float:
    if index < 0 or index >= len(values):
        return float(default)
    try:
        value = float(values[index])
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _window_values(values: np.ndarray, end: int, size: int) -> np.ndarray:
    if size <= 0 or len(values) == 0:
        return np.asarray([], dtype=np.float32)
    end = min(int(end), len(values) - 1)
    if end < 0:
        return np.asarray([], dtype=np.float32)
    start = max(0, end - int(size) + 1)
    if start > end:
        return np.asarray([], dtype=np.float32)
    return np.asarray(values[start : end + 1], dtype=np.float32)


def _window_numeric_features(
    *,
    history: np.ndarray,
    series: np.ndarray,
    history_end: int,
    forecast_start: int,
    day_steps: int,
) -> dict[str, float]:
    history = np.asarray(history, dtype=np.float32)
    series = np.asarray(series, dtype=np.float32)
    hist_mean = _safe_nanmean(history)
    hist_std = _safe_nanstd(history)
    hist_min = _safe_nanpercentile(history, 0)
    hist_max = _safe_nanpercentile(history, 100)
    hist_q10 = _safe_nanpercentile(history, 10)
    hist_q90 = _safe_nanpercentile(history, 90)
    hist_last = float(history[-1]) if history.size else hist_mean
    if not np.isfinite(hist_last):
        hist_last = hist_mean
    recent_7d = _window_values(series, history_end, 7 * max(1, int(day_steps)))
    recent_14d = _window_values(series, history_end, 14 * max(1, int(day_steps)))
    recent_7d_mean = _safe_nanmean(recent_7d, default=hist_mean)
    recent_7d_std = _safe_nanstd(recent_7d, default=hist_std)
    recent_14d_mean = _safe_nanmean(recent_14d, default=hist_mean)
    recent_14d_std = _safe_nanstd(recent_14d, default=hist_std)
    recent_7d_slope = _history_slope(recent_7d) if recent_7d.size >= 2 else 0.0
    recent_14d_slope = _history_slope(recent_14d) if recent_14d.size >= 2 else 0.0
    same_slot_prev_day = _series_value(series, forecast_start - day_steps, default=hist_last)
    same_slot_prev_week = _series_value(series, forecast_start - 7 * day_steps, default=same_slot_prev_day)
    same_slot_prev_2week = _series_value(series, forecast_start - 14 * day_steps, default=same_slot_prev_week)
    return {
        "history_mean": hist_mean,
        "history_std": hist_std,
        "history_min": hist_min,
        "history_max": hist_max,
        "history_q10": hist_q10,
        "history_q90": hist_q90,
        "history_last_minus_mean": hist_last - hist_mean,
        "history_slope": _history_slope(history),
        "history_zero_fraction": float(np.mean(history <= 0.02)) if history.size else 0.0,
        "recent_7d_mean": recent_7d_mean,
        "recent_7d_std": recent_7d_std,
        "recent_7d_slope": recent_7d_slope,
        "recent_14d_mean": recent_14d_mean,
        "recent_14d_std": recent_14d_std,
        "recent_14d_slope": recent_14d_slope,
        "same_slot_prev_day": same_slot_prev_day,
        "same_slot_prev_week": same_slot_prev_week,
        "same_slot_prev_2week": same_slot_prev_2week,
        "same_slot_prev_day_minus_prev_week": same_slot_prev_day - same_slot_prev_week,
        "same_slot_prev_week_minus_prev_2week": same_slot_prev_week - same_slot_prev_2week,
    }


def _window_context_suffix(
    *,
    history: np.ndarray,
    future_timestamps: list[str],
    future_exog: np.ndarray,
    future_exog_dim: int,
    availability_neutral: bool = False,
) -> str:
    """Text-only per-window summary for frozen LLM context encoders."""
    summary = _window_context_summary(
        history=history,
        future_timestamps=future_timestamps,
        future_exog=future_exog,
        future_exog_dim=future_exog_dim,
    )
    if availability_neutral:
        future_text = (
            f"future covariates mean/std={summary['future_exog_mean']:.3f}/"
            f"{summary['future_exog_std']:.3f}"
        )
    else:
        future_text = summary["future_exog_text"]
    return (
        " Window summary: "
        f"forecast_start={summary['start_desc']}; forecast_span={summary['start_text']} to {summary['end_text']}; "
        f"history_level={summary['history_level']}; trend={summary['trend']}; "
        f"history mean/std/min/max={summary['hist_mean']:.3f}/{summary['hist_std']:.3f}/{summary['hist_min']:.3f}/{summary['hist_max']:.3f}; "
            f"history q10/q90={summary['hist_q10']:.3f}/{summary['hist_q90']:.3f}; "
            f"last_minus_mean={summary['last_minus_mean']:.3f}; slope={summary['slope']:.3f}; "
            f"near_zero_fraction={summary['zero_fraction']:.3f}; {future_text}."
    )


def _window_context_summary(
    *,
    history: np.ndarray,
    future_timestamps: list[str],
    future_exog: np.ndarray,
    future_exog_dim: int,
) -> dict[str, Any]:
    """Structured per-window summary used by both flat text and segmented text."""
    history = np.asarray(history, dtype=np.float32)
    hist_mean = _safe_nanmean(history)
    hist_std = _safe_nanstd(history)
    hist_min = _safe_nanpercentile(history, 0)
    hist_max = _safe_nanpercentile(history, 100)
    hist_q10 = _safe_nanpercentile(history, 10)
    hist_q90 = _safe_nanpercentile(history, 90)
    hist_last = float(history[-1]) if history.size else hist_mean
    if not np.isfinite(hist_last):
        hist_last = hist_mean
    slope = _history_slope(history)
    zero_fraction = float(np.mean(history <= 0.02)) if history.size else 0.0

    start_text = future_timestamps[0] if future_timestamps else "unknown"
    end_text = future_timestamps[-1] if future_timestamps else "unknown"
    try:
        start = pd.Timestamp(start_text)
        start_desc = f"hour={start.hour}, dow={start.dayofweek}, weekend={int(start.dayofweek >= 5)}"
    except Exception:
        start_desc = "hour=unknown, dow=unknown, weekend=unknown"

    future_exog = np.asarray(future_exog, dtype=np.float32)
    if future_exog_dim > 0 and future_exog.size > 0:
        future_exog_mean = _safe_nanmean(future_exog)
        future_exog_std = _safe_nanstd(future_exog)
        future_exog_text = f"known future covariates mean/std={future_exog_mean:.3f}/{future_exog_std:.3f}"
    else:
        future_exog_mean = 0.0
        future_exog_std = 0.0
        future_exog_text = "no known future covariates"
    return {
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "hist_min": hist_min,
        "hist_max": hist_max,
        "hist_q10": hist_q10,
        "hist_q90": hist_q90,
        "hist_last": hist_last,
        "last_minus_mean": hist_last - hist_mean,
        "slope": slope,
        "zero_fraction": zero_fraction,
        "history_level": _level_band(hist_mean),
        "trend": _trend_band(slope),
        "start_text": start_text,
        "end_text": end_text,
        "start_desc": start_desc,
        "future_exog_mean": future_exog_mean,
        "future_exog_std": future_exog_std,
        "future_exog_text": future_exog_text,
    }


def _context_segments(
    *,
    task_type: str,
    dataset_name: str,
    node_id: str,
    frequency_minutes: float,
    has_weather: bool,
    has_price: bool,
    covariate_labels: list[str] | tuple[str, ...] | None = None,
    capacity: float,
    node_static_features: dict[str, float],
    window_summary: dict[str, Any],
) -> list[str]:
    if covariate_labels is not None:
        covariates = [str(label).strip() for label in covariate_labels if str(label).strip()]
    else:
        covariates = []
        if has_weather:
            covariates.append("weather")
        if has_price:
            covariates.append("price/system")
    covariate_text = ", ".join(covariates) if covariates else "none"
    return [
        f"Task: {task_type} forecasting.",
        (
            f"Dataset: {dataset_name}. Node: {node_id}. Capacity: {capacity:.3f}. "
            f"Train target mean/std: {_node_stat_value(node_static_features, 'target_train_mean'):.4f}/"
            f"{_node_stat_value(node_static_features, 'target_train_std'):.4f}."
        ),
        (
            f"Frequency: {frequency_minutes:.1f} minutes. "
            f"Forecast start: {window_summary['start_desc']}."
        ),
        (
            f"Known covariates: {covariate_text}. "
            f"Forecast span: {window_summary['start_text']} to {window_summary['end_text']}. "
            f"{window_summary['future_exog_text']}."
        ),
        (
            f"History summary: level={window_summary['history_level']}; trend={window_summary['trend']}; "
            f"mean/std/min/max={window_summary['hist_mean']:.3f}/{window_summary['hist_std']:.3f}/"
            f"{window_summary['hist_min']:.3f}/{window_summary['hist_max']:.3f}; "
            f"q10/q90={window_summary['hist_q10']:.3f}/{window_summary['hist_q90']:.3f}; "
            f"last_minus_mean={window_summary['last_minus_mean']:.3f}; "
            f"near_zero_fraction={window_summary['zero_fraction']:.3f}."
        ),
        (
            f"Future summary: forecast window {window_summary['start_text']} to {window_summary['end_text']}; "
            f"future covariates mean/std={window_summary['future_exog_mean']:.3f}/{window_summary['future_exog_std']:.3f}; "
            f"train prior q10/q90={_node_stat_value(node_static_features, 'target_train_q10'):.3f}/"
            f"{_node_stat_value(node_static_features, 'target_train_q90'):.3f}."
        ),
    ]


def _categorical_band(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def _aaai_exact_context_segments(
    *,
    task_type: str,
    dataset_name: str,
    node_id: str,
    node_meta: dict[str, Any],
    frequency_minutes: float,
    has_weather: bool,
    has_price: bool,
    covariate_labels: list[str] | tuple[str, ...] | None,
    capacity: float,
    window_summary: dict[str, Any],
    information_track: str,
    future_coverage: float,
    has_issued_fields: bool,
    latest_issued_origin: Any,
) -> list[str]:
    """Low-cardinality text schema specified by the AAAI manuscript.

    Continuous weather, target, and operating values remain in the numerical
    stream. Text contains field roles, identities, calendar categories,
    training-split-defined regimes, and explicit availability only.
    """

    def first_meta(keys: tuple[str, ...], default: str) -> str:
        for key in keys:
            value = node_meta.get(key)
            if value is None:
                continue
            label = str(value).strip()
            if label and label.lower() not in {"nan", "none", "null"}:
                return label
        return default

    try:
        start = pd.Timestamp(window_summary["start_text"])
        hour = int(start.hour)
        if hour < 6:
            day_period = "night"
        elif hour < 12:
            day_period = "morning"
        elif hour < 18:
            day_period = "afternoon"
        else:
            day_period = "evening"
        day_type = "weekend" if start.dayofweek >= 5 else "weekday"
        month = int(start.month)
        if month in {12, 1, 2}:
            season = "winter"
        elif month in {3, 4, 5}:
            season = "spring"
        elif month in {6, 7, 8}:
            season = "summer"
        else:
            season = "autumn"
    except Exception:
        day_period = day_type = season = "unknown"

    future_steps = 0
    try:
        start_time = pd.Timestamp(window_summary["start_text"])
        end_time = pd.Timestamp(window_summary["end_text"])
        future_steps = max(1, int(round((end_time - start_time).total_seconds() / 60.0 / frequency_minutes)) + 1)
    except Exception:
        pass
    horizon_hours = future_steps * max(float(frequency_minutes), 0.0) / 60.0
    if horizon_hours <= 6.0:
        horizon_class = "intra-day"
    elif horizon_hours <= 30.0:
        horizon_class = "day-ahead"
    else:
        horizon_class = "multi-day"

    if frequency_minutes <= 15:
        cadence = "high-frequency"
    elif frequency_minutes <= 30:
        cadence = "half-hourly"
    elif frequency_minutes <= 60:
        cadence = "hourly"
    else:
        cadence = "coarse"

    if covariate_labels is not None:
        covariates = sorted({str(label).strip() for label in covariate_labels if str(label).strip()})
    else:
        covariates = []
        if has_weather:
            covariates.append("weather")
        if has_price:
            covariates.append("price/system")
    covariate_text = ", ".join(covariates) if covariates else "none"
    future_weather = "available" if has_weather and future_coverage > 0.0 else "unavailable"
    if future_coverage <= 0.0:
        coverage_class = "none"
    elif future_coverage < 0.95:
        coverage_class = "partial"
    else:
        coverage_class = "complete"

    weather_level = _categorical_band(float(window_summary["future_exog_mean"]), low=-0.5, high=0.5)
    weather_variability = _categorical_band(float(window_summary["future_exog_std"]), low=0.5, high=1.25)
    near_zero = _categorical_band(float(window_summary["zero_fraction"]), low=0.1, high=0.5)
    region = first_meta(("area", "gsp_group", "zone_id", "source_site", "zone_type"), "unspecified")
    asset_type = first_meta(("site_type", "type", "zone_type"), task_type)
    capacity_state = "available" if capacity > 0.0 else "unavailable"
    vintage_state = "available" if latest_issued_origin is not None else "unavailable"
    issued_state = "available" if has_issued_fields else "unavailable"

    return [
        (
            f"[static] task={task_type}; asset_type={asset_type}; asset_id={node_id}; "
            f"region={region}; source={dataset_name}; capacity_field={capacity_state}."
        ),
        (
            f"[known-future] cadence={cadence}; horizon={horizon_class}; "
            f"day_period={day_period}; day_type={day_type}; season={season}."
        ),
        (
            f"[forecast] covariate_fields={covariate_text}; weather_level={weather_level}; "
            f"weather_variability={weather_variability}."
        ),
        (
            f"[observed] recent_level={window_summary['history_level']}; "
            f"recent_trend={window_summary['trend']}; near_zero_regime={near_zero}."
        ),
        (
            f"[availability] information_track={information_track}; history=available; "
            f"future_weather={future_weather}; issued_fields={issued_state}; "
            f"issued_coverage={coverage_class}; selected_vintage={vintage_state}."
        ),
    ]


def _node_capacity(meta: dict[str, Any]) -> float:
    for key in ("capacity", "station_capacity", "rated_capacity"):
        try:
            value = float(meta.get(key, 0.0))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return 0.0


def _node_meta_row(node_meta, node_id: str, node_idx: int) -> dict[str, Any]:
    if node_meta is None or getattr(node_meta, "empty", True):
        return {}
    for key in (node_id, str(node_id)):
        try:
            if key in node_meta.index:
                row = node_meta.loc[key]
                if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                    row = row.iloc[0]
                return row.to_dict()
        except Exception:
            pass
    try:
        if 0 <= int(node_idx) < len(node_meta):
            return node_meta.iloc[int(node_idx)].to_dict()
    except Exception:
        pass
    return {}


def _build_node_static_features(
    data: dict,
    split: dict,
    *,
    include_target_stats: bool,
) -> dict[str, dict[str, float]]:
    """Build node-level metadata features and optional train-split target summaries."""
    train_values = np.asarray(split.get("train"), dtype=np.float32)
    if train_values.ndim != 2 or train_values.size == 0:
        return {}
    timestamps = split.get("timestamps_train")
    node_ids = _node_ids_for(data, train_values.shape[1])
    node_meta = data.get("node_meta")
    capacities = []
    for idx, node_id in enumerate(node_ids):
        capacities.append(_node_capacity(_node_meta_row(node_meta, node_id=node_id, node_idx=idx)))
    capacity_order = np.argsort(np.argsort(np.asarray(capacities, dtype=np.float32)))
    capacity_denom = max(1, len(capacities) - 1)

    try:
        ts = pd.DatetimeIndex(timestamps)
        hours = np.asarray(ts.hour, dtype=np.int64)
        day_mask = (hours >= 6) & (hours < 18)
        night_mask = ~day_mask
        weekend_mask = np.asarray(ts.dayofweek >= 5)
    except Exception:
        hours = np.zeros(train_values.shape[0], dtype=np.int64)
        day_mask = np.zeros(train_values.shape[0], dtype=bool)
        night_mask = np.zeros(train_values.shape[0], dtype=bool)
        weekend_mask = np.zeros(train_values.shape[0], dtype=bool)

    features: dict[str, dict[str, float]] = {}
    for idx, node_id in enumerate(node_ids):
        node_features = {
            "node_index_scaled": float(idx / max(1, len(node_ids) - 1)),
            "capacity_rank": float(capacity_order[idx] / capacity_denom),
        }
        if not include_target_stats:
            features[str(node_id)] = node_features
            continue
        values = np.asarray(train_values[:, idx], dtype=np.float32)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            finite_values = np.zeros(1, dtype=np.float32)
        mean = _safe_nanmean(finite_values)
        std = float(np.nanstd(finite_values))
        if not np.isfinite(std):
            std = 0.0
        q10 = _safe_nanpercentile(finite_values, 10)
        q50 = _safe_nanpercentile(finite_values, 50)
        q90 = _safe_nanpercentile(finite_values, 90)
        hourly_means = []
        for hour in range(24):
            mask = hours == hour
            hourly_means.append(_safe_nanmean(values[mask], default=mean) if np.any(mask) else mean)
        peak_hour = int(np.nanargmax(hourly_means)) if hourly_means else 0
        node_features.update(
            {
                "target_train_mean": float(mean),
                "target_train_std": float(std),
                "target_train_min": _safe_nanpercentile(finite_values, 0),
                "target_train_max": _safe_nanpercentile(finite_values, 100),
                "target_train_q10": float(q10),
                "target_train_q50": float(q50),
                "target_train_q90": float(q90),
                "target_train_iqr": float(q90 - q10),
                "target_train_zero_fraction": float(np.mean(np.abs(finite_values) <= 1e-6)),
                "target_train_active_fraction": float(np.mean(finite_values > 1e-3)),
                "target_train_day_mean": _safe_nanmean(values[day_mask], default=mean) if np.any(day_mask) else mean,
                "target_train_night_mean": _safe_nanmean(values[night_mask], default=mean) if np.any(night_mask) else mean,
                "target_train_weekend_mean": _safe_nanmean(values[weekend_mask], default=mean)
                if np.any(weekend_mask)
                else mean,
                "target_train_peak_hour_sin": float(np.sin(2 * np.pi * peak_hour / 24.0)),
                "target_train_peak_hour_cos": float(np.cos(2 * np.pi * peak_hour / 24.0)),
            }
        )
        features[str(node_id)] = node_features
    return features


def _numeric_optional_frame(frame, timestamps) -> pd.DataFrame:
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame(index=pd.DatetimeIndex(timestamps))
    out = frame.copy()
    try:
        out.index = pd.DatetimeIndex(out.index)
    except Exception:
        out = pd.DataFrame(out)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, out.notna().any(axis=0)]
    if out.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(timestamps))
    return out.reindex(pd.DatetimeIndex(timestamps)).interpolate(limit_direction="both").ffill().bfill()


def _exogenous_frame(
    data: dict,
    timestamps,
    *,
    train_len: int,
    train_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames = []
    future_known_columns: list[str] = []
    # `price_history` is history-only context: it can enrich the observed window
    # but must not be exposed inside the forecast horizon.
    source_specs = (
        ("weather", True),
        ("price", True),
        ("price_history", False),
    )
    for key, future_known in source_specs:
        frame = _numeric_optional_frame(data.get(key), timestamps)
        if not frame.empty:
            frame = frame.add_prefix(f"{key}_")
            frames.append(frame)
            if future_known:
                future_known_columns.extend(str(col) for col in frame.columns)
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex(timestamps)), [], []

    exog = pd.concat(frames, axis=1)
    exog = exog.loc[:, ~exog.columns.duplicated()]
    if train_indices is not None and len(train_indices) > 0:
        valid_indices = np.asarray(train_indices, dtype=np.int64)
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < len(exog))]
        source = exog.iloc[valid_indices] if len(valid_indices) > 0 else exog.iloc[:1]
    else:
        train_end = max(1, min(int(train_len), len(exog)))
        source = exog.iloc[:train_end]
    mean = source.mean(axis=0)
    std = source.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    exog = ((exog - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exog = exog.astype(np.float32)
    exog_columns = [str(col) for col in exog.columns]
    future_known_set = set(future_known_columns)
    future_known_columns = [col for col in exog_columns if col in future_known_set]
    return exog, exog_columns, future_known_columns


def _prepared_exogenous(data: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    timestamps = data.get("timestamps")
    if timestamps is None:
        return pd.DataFrame(), [], []
    train_len = int(len(data.get("occupancy", [])) * 0.7) if data.get("occupancy") is not None else len(timestamps)
    train_indices = None
    try:
        split_indices = data.get("split_indices") or {}
        train_indices = np.asarray(split_indices.get("train", []), dtype=np.int64)
    except Exception:
        train_indices = None
    frame, columns, future_known_columns = _exogenous_frame(
        data,
        timestamps,
        train_len=train_len,
        train_indices=train_indices,
    )
    # Versioned operational forecasts are selected by forecast origin later,
    # inside `_build_split_examples`. Reserve their numeric fields here so all
    # datasets retain a fixed token layout without exposing future revisions.
    for column in data.get("issued_exogenous_columns", []):
        column = str(column)
        if column not in frame.columns:
            frame[column] = 0.0
        if column not in columns:
            columns.append(column)
        if column not in future_known_columns:
            future_known_columns.append(column)
    return frame.reindex(columns=columns), columns, future_known_columns


def _node_normalization(data: dict, node_id: str, node_idx: int) -> tuple[float, float]:
    """Return the train-fitted inverse transform for one spatial node."""

    node_norm = data.get("node_norm") or {}
    item = node_norm.get(str(node_id))
    if item is None:
        item = node_norm.get(node_idx)
    if isinstance(item, dict):
        norm_min = float(item.get("min", data.get("norm_min", 0.0)))
        norm_max = float(item.get("max", data.get("norm_max", 1.0)))
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        norm_min, norm_max = float(item[0]), float(item[1])
    else:
        norm_min = float(data.get("norm_min", 0.0))
        norm_max = float(data.get("norm_max", 1.0))
    if not np.isfinite(norm_min) or not np.isfinite(norm_max) or norm_max <= norm_min:
        return 0.0, 1.0
    return norm_min, norm_max


def _latest_issued_block(
    frame: pd.DataFrame,
    *,
    forecast_origin: pd.Timestamp,
    future_timestamps,
    node_id: str,
    columns: list[str],
    lookup: dict[tuple[pd.Timestamp, str], pd.DataFrame] | None = None,
) -> IssuedAvailability:
    """Select the latest forecast vintage that existed at one origin."""

    future_index = pd.DatetimeIndex(future_timestamps)
    exact = None if lookup is None else lookup.get((pd.Timestamp(forecast_origin), str(node_id)))
    if exact is not None:
        frame = exact.reset_index()
    return select_issued_block(
        frame,
        forecast_origin=forecast_origin,
        future_timestamps=future_index,
        node_id=node_id,
        columns=columns,
    )


def _oracle_block(
    frame: pd.DataFrame,
    *,
    future_timestamps,
    node_id: str,
    columns: list[str],
    lookup: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, float]:
    """Retrieve final observed covariates for the labelled oracle track."""

    empty = np.zeros((len(future_timestamps), len(columns)), dtype=np.float32)
    if frame is None or getattr(frame, "empty", True) or not columns:
        return empty, 0.0
    future_index = pd.DatetimeIndex(future_timestamps)
    exact = None if lookup is None else lookup.get(str(node_id))
    if exact is not None:
        numeric = exact.reindex(future_index)[columns]
        coverage = float(numeric.notna().to_numpy().mean()) if numeric.size else 0.0
        return numeric.fillna(0.0).to_numpy(dtype=np.float32), coverage
    required = {"valid_time", "node_id", *columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"oracle_exogenous is missing fields {sorted(missing)}")
    selected = frame.loc[frame["node_id"].astype(str).eq(str(node_id))].copy()
    selected["valid_time"] = pd.to_datetime(selected["valid_time"])
    selected = selected.sort_values("valid_time").drop_duplicates("valid_time", keep="last")
    values = selected.set_index("valid_time").reindex(future_index)[columns]
    numeric = values.apply(pd.to_numeric, errors="coerce")
    coverage = float(numeric.notna().to_numpy().mean()) if numeric.size else 0.0
    return numeric.fillna(0.0).to_numpy(dtype=np.float32), coverage


def _future_exogenous_block(
    *,
    data: dict,
    policy: str,
    base_values: np.ndarray,
    future_known_mask: np.ndarray,
    exog_columns: list[str],
    forecast_origin: pd.Timestamp,
    future_timestamps,
    node_id: str,
) -> IssuedAvailability:
    """Build one node's future block without crossing the forecast origin."""

    shape = np.asarray(base_values).shape
    unavailable = np.zeros(shape, dtype=bool)
    unpublished = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    if policy in {"zero", "common"}:
        return IssuedAvailability(
            np.zeros_like(base_values, dtype=np.float32), unavailable, unpublished, 0.0, None, None
        )

    block = np.asarray(base_values, dtype=np.float32).copy()
    available = np.broadcast_to(future_known_mask, shape).copy() if block.size else unavailable
    if block.size > 0 and future_known_mask.size > 0 and (~future_known_mask).any():
        block[:, ~future_known_mask] = 0.0
    declared = [str(column) for column in data.get("issued_exogenous_columns", [])]
    declared = [column for column in declared if column in exog_columns]
    if not declared:
        return IssuedAvailability(
            block, available, unpublished, 1.0 if block.size else 0.0, None, None
        )
    positions = [exog_columns.index(column) for column in declared]

    if policy in {"available", "issued"}:
        issued = _latest_issued_block(
            data.get("issued_exogenous"),
            forecast_origin=forecast_origin,
            future_timestamps=future_timestamps,
            node_id=node_id,
            columns=declared,
            lookup=data.get("_issued_exogenous_lookup"),
        )
        block[:, positions] = issued.values
        available[:, positions] = issued.available
        unpublished[:, positions] = issued.available_at
        return IssuedAvailability(
            block,
            available,
            unpublished,
            issued.coverage,
            issued.latest_forecast_origin,
            issued.latest_available_at,
        )
    if policy == "oracle":
        values, coverage = _oracle_block(
            data.get("oracle_exogenous"),
            future_timestamps=future_timestamps,
            node_id=node_id,
            columns=declared,
            lookup=data.get("_oracle_exogenous_lookup"),
        )
        block[:, positions] = values
        return IssuedAvailability(block, available, unpublished, coverage, None, None)
    raise ValueError(
        "future_exog_policy must be one of 'common', 'issued', 'oracle', "
        "'zero', or 'available'"
    )


def _prepare_versioned_exogenous_lookups(data: dict, policy: str) -> None:
    columns = [str(value) for value in data.get("issued_exogenous_columns", [])]
    if policy in {"available", "issued"} and columns and "_issued_exogenous_lookup" not in data:
        frame = data.get("issued_exogenous")
        if frame is not None and not getattr(frame, "empty", True):
            fields = list(dict.fromkeys([
                "forecast_origin", "valid_time", "node_id", *columns,
                *provenance_columns(frame, columns),
            ]))
            indexed = frame[fields].copy()
            indexed["forecast_origin"] = pd.to_datetime(indexed["forecast_origin"])
            indexed["valid_time"] = pd.to_datetime(indexed["valid_time"])
            indexed["node_id"] = indexed["node_id"].astype(str)
            indexed[columns] = indexed[columns].apply(pd.to_numeric, errors="coerce")
            data["_issued_exogenous_lookup"] = {
                (pd.Timestamp(origin), str(node_id)): group.set_index("valid_time").sort_index()
                for (origin, node_id), group in indexed.groupby(
                    ["forecast_origin", "node_id"],
                    sort=False,
                    observed=True,
                )
            }
    if policy == "oracle" and columns:
        frame = data.get("oracle_exogenous")
        if frame is not None and not getattr(frame, "empty", True):
            indexed = frame[["valid_time", "node_id", *columns]].copy()
            indexed["valid_time"] = pd.to_datetime(indexed["valid_time"])
            indexed["node_id"] = indexed["node_id"].astype(str)
            indexed[columns] = indexed[columns].apply(pd.to_numeric, errors="coerce")
            data["_oracle_exogenous_lookup"] = {
                str(node_id): group.sort_values("valid_time").drop_duplicates(
                    "valid_time",
                    keep="last",
                ).set_index("valid_time")[columns]
                for node_id, group in indexed.groupby("node_id", sort=False, observed=True)
            }


def _jiangsu_price_corr_core_columns(columns: list[str]) -> list[str]:
    exact = {
        "weather_boundary_系统负荷",
        "weather_boundary_江南风电",
        "weather_boundary_江北风电",
        "weather_boundary_江南光伏",
        "weather_boundary_江北光伏",
        "weather_market_system_load",
        "weather_market_pv_total",
        "weather_market_renewable_total",
        "weather_market_net_load_after_renewable",
        "weather_market_fire_competitive_space",
        "weather_market_fire_competitive_space_clipped",
        "weather_market_renewable_share",
        "weather_market_fire_space_share",
    }
    prefixes = (
        "weather_market_system_load_diff_",
        "weather_market_system_load_mean_",
        "weather_market_renewable_total_diff_",
        "weather_market_renewable_total_mean_",
        "weather_market_fire_competitive_space_diff_",
        "weather_market_fire_competitive_space_mean_",
        "weather_market_fire_competitive_space_clipped_diff_",
        "weather_market_fire_competitive_space_clipped_mean_",
    )
    return [
        col for col in columns
        if col in exact or any(col.startswith(prefix) for prefix in prefixes)
    ]


def _filter_exogenous_columns(
    *,
    dataset_name: str,
    exog_frame: pd.DataFrame,
    exog_columns: list[str],
    future_known_columns: list[str],
    exogenous_feature_profile: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if exogenous_feature_profile == "all" or not exog_columns:
        return exog_frame, exog_columns, future_known_columns
    if dataset_name != "jiangsu_price":
        return exog_frame, exog_columns, future_known_columns

    corr_core = set(_jiangsu_price_corr_core_columns(exog_columns))
    if exogenous_feature_profile == "jiangsu_price_corr_core":
        selected = [
            col for col in exog_columns
            if (not col.startswith("weather_")) or (col in corr_core)
        ]
    elif exogenous_feature_profile == "jiangsu_price_drop_corr_core":
        selected = [
            col for col in exog_columns
            if (not col.startswith("weather_")) or (col not in corr_core)
        ]
    else:
        raise ValueError(
            "exogenous_feature_profile must be one of: all, "
            "jiangsu_price_corr_core, jiangsu_price_drop_corr_core"
        )

    selected_set = set(selected)
    filtered_frame = exog_frame.reindex(columns=selected).copy()
    filtered_future_known = [col for col in future_known_columns if col in selected_set]
    return filtered_frame, selected, filtered_future_known


def _build_split_examples(
    *,
    dataset_name: str,
    split_name: str,
    data: dict,
    split_values: np.ndarray,
    split_timestamps,
    history_len: int,
    horizon: int,
    stride: int,
    max_nodes: int,
    max_windows: int,
    exog_frame: pd.DataFrame,
    exog_columns: list[str],
    future_known_columns: list[str] | None = None,
    node_static_features: dict[str, dict[str, float]] | None = None,
    context_identity_policy: str = "full",
    text_schema_variant: str = "current",
    future_exog_policy: str = "available",
    dataset_label: int = 0,
    node_label_map: dict[str, int] | None = None,
) -> list[EnergyExample]:
    forecast_lead_steps = _forecast_lead_steps_for(data)
    if forecast_lead_steps < 1:
        raise ValueError("forecast_lead_steps must be >= 1")
    # `forecast_lead_steps=1` means the history ends immediately before the
    # target start. Larger values insert a business cutoff gap before target.
    if split_values is None or len(split_values) < history_len + horizon + forecast_lead_steps - 1:
        return []

    split_values = np.asarray(split_values, dtype=np.float32)
    n_nodes = split_values.shape[1]
    node_ids = _node_ids_for(data, n_nodes)
    node_count = min(int(max_nodes), n_nodes) if max_nodes and max_nodes > 0 else n_nodes
    node_indices = list(range(node_count))
    forecast_start_offset_steps = int(data.get("forecast_start_offset_steps", 0) or 0)
    if forecast_start_offset_steps < 0 or forecast_start_offset_steps >= int(stride):
        raise ValueError("forecast_start_offset_steps must lie in [0, stride)")
    starts = list(
        range(
            int(history_len) + forecast_start_offset_steps,
            len(split_values) - int(horizon) + 1,
            int(stride),
        )
    )
    target_start_time = (data.get("split_target_start_times") or {}).get(split_name)
    if target_start_time is not None:
        try:
            target_start_time = pd.Timestamp(target_start_time)
            split_index = pd.DatetimeIndex(split_timestamps)
            starts = [start for start in starts if split_index[start] >= target_start_time]
        except Exception:
            pass
    target_start_minute = data.get("forecast_target_start_minute_of_day")
    if target_start_minute is not None:
        target_start_minute = int(target_start_minute)
        if not 0 <= target_start_minute < 24 * 60:
            raise ValueError("forecast_target_start_minute_of_day must lie in [0, 1440)")
        split_index = pd.DatetimeIndex(split_timestamps)
        starts = [
            start
            for start in starts
            if split_index[start].hour * 60 + split_index[start].minute == target_start_minute
        ]
    time_feat = _time_features(split_timestamps)
    frequency = estimate_frequency_minutes(split_timestamps)
    day_steps = max(1, int(round((24.0 * 60.0) / frequency))) if frequency > 0 else max(1, int(history_len))
    has_weather = data.get("weather") is not None and not getattr(data.get("weather"), "empty", True)
    has_price = (
        (data.get("price") is not None and not getattr(data.get("price"), "empty", True))
        or (data.get("price_history") is not None and not getattr(data.get("price_history"), "empty", True))
    )
    covariate_labels = data.get("covariate_labels")
    if not exog_frame.empty:
        exog_frame = exog_frame.reindex(pd.DatetimeIndex(split_timestamps)).reindex(columns=exog_columns).fillna(0.0)
        exog_values = exog_frame.to_numpy(dtype=np.float32)
    else:
        exog_values = np.zeros((len(split_values), len(exog_columns)), dtype=np.float32)
    exog_dim = int(len(exog_columns))
    future_known_set = set(str(col) for col in (future_known_columns or exog_columns))
    future_known_mask = np.asarray([str(col) in future_known_set for col in exog_columns], dtype=bool)
    node_static_features = node_static_features or {}
    node_label_map = node_label_map or {}
    dynamic_offset = data.get("target_offset_frame")
    dynamic_scale = data.get("target_scale_frame")
    if dynamic_offset is not None and not getattr(dynamic_offset, "empty", True):
        dynamic_offset = dynamic_offset.reindex(pd.DatetimeIndex(split_timestamps)).reindex(columns=node_ids)
    else:
        dynamic_offset = None
    if dynamic_scale is not None and not getattr(dynamic_scale, "empty", True):
        dynamic_scale = dynamic_scale.reindex(pd.DatetimeIndex(split_timestamps)).reindex(columns=node_ids)
    else:
        dynamic_scale = None

    examples: list[EnergyExample] = []
    try:
        split_timestamp_index = pd.DatetimeIndex(split_timestamps)
    except Exception:
        split_timestamp_index = split_timestamps
    # Filter out windows that cannot satisfy the strict business cutoff before
    # any downsampling. This keeps `max_windows=1` from collapsing to an empty
    # split and ensures the retained samples are the first valid boundary-aligned
    # windows, not the first raw stride positions.
    valid_starts: list[int] = []
    allowed_origin_values = data.get("forecast_origin_filter")
    allowed_origins = None
    if allowed_origin_values is not None:
        allowed_origins = {pd.Timestamp(value) for value in allowed_origin_values}
    origin_windows = (data.get("split_origin_ranges") or {}).get(split_name, [])
    origin_windows = [(pd.Timestamp(left), pd.Timestamp(right)) for left, right in origin_windows]
    for start in starts:
        history_end = start - forecast_lead_steps
        history_start = history_end - history_len + 1
        if history_start < 0:
            continue
        if allowed_origins is not None:
            origin_position = history_end if data.get("forecast_origin_from_history_end", False) else start
            candidate_origin = pd.Timestamp(split_timestamp_index[origin_position])
            if candidate_origin not in allowed_origins:
                continue
        else:
            origin_position = history_end if data.get("forecast_origin_from_history_end", False) else start
            candidate_origin = pd.Timestamp(split_timestamp_index[origin_position])
        if origin_windows and not any(left <= candidate_origin < right for left, right in origin_windows):
            continue
        valid_starts.append(start)
    starts = _select_evenly(valid_starts, max_windows)
    for start in starts:
        history_end = start - forecast_lead_steps
        history_start = history_end - history_len + 1
        hist_time = time_feat[history_start : history_end + 1].reshape(-1)
        future_time = time_feat[start : start + horizon].reshape(-1)
        hist_exog = exog_values[history_start : history_end + 1].reshape(-1)
        forecast_start_value = split_timestamp_index[start] if hasattr(split_timestamp_index, "__len__") else None
        target_start = pd.Timestamp(forecast_start_value) if forecast_start_value is not None else None
        if data.get("forecast_origin_from_history_end", False) and target_start is not None:
            forecast_start = pd.Timestamp(split_timestamp_index[history_end])
        else:
            forecast_start = target_start
        history_timestamps = _timestamp_strings(split_timestamp_index[history_start : history_end + 1])
        future_timestamps = _timestamp_strings(split_timestamp_index[start : start + horizon])
        for node_idx in node_indices:
            node_id = node_ids[node_idx] if node_idx < len(node_ids) else str(node_idx)
            norm_min, norm_max = _node_normalization(data, str(node_id), node_idx)
            if dynamic_offset is not None and dynamic_scale is not None:
                history_offset = dynamic_offset.iloc[history_start : history_end + 1, node_idx].to_numpy(np.float32)
                history_scale = dynamic_scale.iloc[history_start : history_end + 1, node_idx].to_numpy(np.float32)
                target_offset = dynamic_offset.iloc[start : start + horizon, node_idx].to_numpy(np.float32)
                target_scale = dynamic_scale.iloc[start : start + horizon, node_idx].to_numpy(np.float32)
                if not (
                    np.isfinite(history_offset).all()
                    and np.isfinite(history_scale).all()
                    and np.isfinite(target_offset).all()
                    and np.isfinite(target_scale).all()
                    and (history_scale > 0.0).all()
                    and (target_scale > 0.0).all()
                ):
                    raise ValueError(f"Invalid dynamic target normalization for {dataset_name}::{node_id}")
            else:
                history_offset = np.full(history_len, norm_min, dtype=np.float32)
                history_scale = np.full(history_len, norm_max - norm_min, dtype=np.float32)
                target_offset = np.full(horizon, norm_min, dtype=np.float32)
                target_scale = np.full(horizon, norm_max - norm_min, dtype=np.float32)
            future_result = _future_exogenous_block(
                data=data,
                policy=future_exog_policy,
                base_values=exog_values[start : start + horizon],
                future_known_mask=future_known_mask,
                exog_columns=exog_columns,
                forecast_origin=forecast_start,
                future_timestamps=split_timestamp_index[start : start + horizon],
                node_id=str(node_id),
            )
            future_block = future_result.values
            future_coverage = future_result.coverage
            latest_issued_origin = future_result.latest_forecast_origin
            future_exog = future_block.reshape(-1)
            hist = split_values[history_start : history_end + 1, node_idx]
            series = split_values[:, node_idx]
            target = split_values[start : start + horizon, node_idx]
            node_meta = _node_meta_row(data.get("node_meta"), node_id=node_id, node_idx=node_idx)
            capacity = _node_capacity(node_meta)
            window_summary = _window_context_summary(
                history=hist,
                future_timestamps=future_timestamps,
                future_exog=future_exog,
                future_exog_dim=exog_dim,
            )
            window_numeric_features = _window_numeric_features(
                history=hist,
                series=series,
                history_end=history_end,
                forecast_start=start,
                day_steps=day_steps,
            )
            dynamic_context_text = _window_context_suffix(
                history=hist,
                future_timestamps=future_timestamps,
                future_exog=future_exog,
                future_exog_dim=exog_dim,
                availability_neutral=text_schema_variant == "no_availability",
            )
            information_track = {
                "zero": "common",
                "available": "issued",
            }.get(future_exog_policy, future_exog_policy)
            information_suffix = f" Forecast-information track: {information_track}."
            if data.get("issued_exogenous_columns"):
                information_suffix += f" Future-field coverage: {future_coverage:.3f}."
                if latest_issued_origin is not None:
                    information_suffix += (
                        " Latest selected forecast vintage: "
                        f"{pd.Timestamp(latest_issued_origin).isoformat()}."
                    )
            context = build_energy_context(
                dataset_name=dataset_name,
                node_id=node_id,
                node_idx=node_idx,
                node_meta=data.get("node_meta"),
                has_weather=has_weather,
                has_price=has_price,
                frequency_minutes=frequency,
                forecast_start=forecast_start,
                covariate_labels=covariate_labels,
                node_static_features=node_static_features.get(str(node_id), {}),
                window_numeric_features=window_numeric_features,
                context_identity_policy=context_identity_policy,
            )
            numeric = np.concatenate([hist, hist_time, hist_exog, future_time, future_exog]).astype(np.float32)
            node_key = f"{dataset_name}::{node_id}"
            if text_schema_variant == "operational":
                notice_text = (data.get("issued_text_context") or {}).get(
                    (pd.Timestamp(forecast_start), str(node_id)),
                    "",
                )
                context_segments = [
                    (
                        f"Task: {context.task_type} forecasting. Region: {node_id}. "
                        f"Forecast origin: {pd.Timestamp(forecast_start).isoformat()}."
                    ),
                    notice_text or "No relevant AEMO operational notice in the previous 72 hours.",
                ]
                context_text = " ".join(context_segments)
            elif text_schema_variant == "aaai_exact":
                context_segments = _aaai_exact_context_segments(
                    task_type=context.task_type,
                    dataset_name=dataset_name,
                    node_id=node_id,
                    node_meta=node_meta,
                    frequency_minutes=frequency,
                    has_weather=has_weather,
                    has_price=has_price,
                    covariate_labels=covariate_labels,
                    capacity=capacity,
                    window_summary=window_summary,
                    information_track=information_track,
                    future_coverage=future_coverage,
                    has_issued_fields=bool(data.get("issued_exogenous_columns")),
                    latest_issued_origin=latest_issued_origin,
                )
                context_text = " ".join(context_segments)
            elif text_schema_variant == "no_availability":
                # This ablation retains the forecast-origin-valid values while
                # removing per-window publication and coverage descriptions.
                context_text = context.text + dynamic_context_text
                context_segments = _context_segments(
                    task_type=context.task_type,
                    dataset_name=dataset_name,
                    node_id=node_id,
                    frequency_minutes=frequency,
                    has_weather=has_weather,
                    has_price=has_price,
                    covariate_labels=covariate_labels,
                    capacity=capacity,
                    node_static_features=node_static_features.get(str(node_id), {}),
                    window_summary=window_summary,
                )
            else:
                context_text = context.text + dynamic_context_text + information_suffix
                context_segments = _context_segments(
                    task_type=context.task_type,
                    dataset_name=dataset_name,
                    node_id=node_id,
                    frequency_minutes=frequency,
                    has_weather=has_weather,
                    has_price=has_price,
                    covariate_labels=covariate_labels,
                    capacity=capacity,
                    node_static_features=node_static_features.get(str(node_id), {}),
                    window_summary=window_summary,
                )
            examples.append(
                EnergyExample(
                    dataset_name=dataset_name,
                    dataset_label=int(dataset_label),
                    node_id=node_id,
                    node_label=int(node_label_map.get(node_key, node_idx)),
                    task_type=context.task_type,
                    task_id=context.task_id,
                    numeric=numeric,
                    history_len=int(history_len),
                    horizon=int(horizon),
                    history_exog_dim=exog_dim,
                    future_exog_dim=exog_dim,
                    context=context.vector,
                    target=target.astype(np.float32),
                    target_raw=(target * target_scale + target_offset).astype(np.float32),
                    target_offset=target_offset.astype(np.float32),
                    target_scale=target_scale.astype(np.float32),
                    history_raw=(hist * history_scale + history_offset).astype(np.float32),
                    norm_min=norm_min,
                    norm_max=norm_max,
                    context_text=context_text,
                    context_segments=context_segments,
                    history_timestamps=history_timestamps,
                    future_timestamps=future_timestamps,
                    forecast_start=forecast_start.isoformat() if forecast_start is not None else None,
                    future_exog_mask=future_result.available.astype(np.float32),
                    future_exog_available_at=future_result.available_at,
                    future_exog_coverage=float(future_result.coverage),
                    latest_issued_available_at=(
                        future_result.latest_available_at.isoformat()
                        if future_result.latest_available_at is not None
                        else None
                    ),
                )
            )
    return examples


def load_energy_datasets(
    dataset_names: list[str],
    *,
    history_len: int,
    horizon: int,
    stride: int,
    max_nodes_per_dataset: int = 8,
    max_train_windows_per_dataset: int = 512,
    max_eval_windows_per_dataset: int = 128,
    context_target_stats: str = "train",
    context_identity_policy: str = "full",
    text_schema_variant: str = "current",
    future_exog_policy: str = "available",
    exogenous_feature_profile: str = "all",
    context_embedding_mode: str = "numeric",
    llm_context_cache: str | None = None,
    llm_context_missing_policy: str = "error",
    embargo_steps: int = 0,
) -> tuple[dict[str, EnergyWindowDataset], dict[str, Any]]:
    """Load existing project datasets into train/val/test EnergyWindowDataset objects."""
    if context_target_stats not in {"train", "none"}:
        raise ValueError("context_target_stats must be either 'train' or 'none'")
    if context_identity_policy not in {"full", "no_hash", "semantic_only"}:
        raise ValueError("context_identity_policy must be 'full', 'no_hash', or 'semantic_only'")
    if text_schema_variant not in {"current", "aaai_exact", "no_availability", "operational"}:
        raise ValueError(
            "text_schema_variant must be 'current', 'aaai_exact', 'no_availability', or 'operational'"
        )
    if future_exog_policy not in {"available", "zero", "common", "issued", "oracle"}:
        raise ValueError(
            "future_exog_policy must be one of 'common', 'issued', 'oracle', "
            "'zero', or 'available'"
        )
    if int(embargo_steps) != embargo_steps or embargo_steps < 0:
        raise ValueError("embargo_steps must be a non-negative whole number")
    if context_embedding_mode not in CONTEXT_EMBEDDING_MODES:
        raise ValueError(f"context_embedding_mode must be one of {CONTEXT_EMBEDDING_MODES}")
    if context_embedding_mode != "numeric" and not llm_context_cache:
        raise ValueError("llm_context_cache is required when context_embedding_mode is not 'numeric'")
    splits_out = {"train": [], "val": [], "test": []}
    manifest: dict[str, Any] = {
        "datasets": {},
        "history_len": history_len,
        "horizon": horizon,
        "stride": stride,
        "context_target_stats": context_target_stats,
        "context_identity_policy": context_identity_policy,
        "text_schema_variant": text_schema_variant,
        "future_exog_policy": future_exog_policy,
        "exogenous_feature_profile": exogenous_feature_profile,
        "context_embedding_mode": context_embedding_mode,
        "llm_context_cache": llm_context_cache,
        "llm_context_missing_policy": llm_context_missing_policy,
        "embargo_steps": int(embargo_steps),
    }
    loaded_items: list[tuple[str, dict, dict, pd.DataFrame, list[str], list[str]]] = []
    global_exog_columns: list[str] = []
    global_future_known_columns: list[str] = []
    dataset_label_map = {name: idx for idx, name in enumerate(dataset_names)}
    node_label_map: dict[str, int] = {}

    for dataset_name in dataset_names:
        raw = load_dataset(dataset_name)
        _prepare_versioned_exogenous_lookups(raw, future_exog_policy)
        split = build_splits(
            raw,
            dataset_name=dataset_name,
            embargo_steps=int(embargo_steps),
        )
        exog_frame, exog_columns, future_known_columns = _prepared_exogenous(raw)
        exog_frame, exog_columns, future_known_columns = _filter_exogenous_columns(
            dataset_name=dataset_name,
            exog_frame=exog_frame,
            exog_columns=exog_columns,
            future_known_columns=future_known_columns,
            exogenous_feature_profile=exogenous_feature_profile,
        )
        for column in exog_columns:
            if column not in global_exog_columns:
                global_exog_columns.append(column)
        for column in future_known_columns:
            if column not in global_future_known_columns:
                global_future_known_columns.append(column)
        node_ids = _node_ids_for(raw, int(np.asarray(raw["occupancy"]).shape[1]))
        for node_id in node_ids:
            node_key = f"{dataset_name}::{node_id}"
            if node_key not in node_label_map:
                node_label_map[node_key] = len(node_label_map)
        loaded_items.append((dataset_name, raw, split, exog_frame, exog_columns, future_known_columns))

    for dataset_name, raw, split, exog_frame, _local_exog_columns, _local_future_known_columns in loaded_items:
        node_static_features = _build_node_static_features(
            raw,
            split,
            include_target_stats=context_target_stats == "train",
        )
        dataset_manifest = {
            "nodes_total": int(np.asarray(raw["occupancy"]).shape[1]),
            "norm_min": float(raw.get("norm_min", 0.0)),
            "norm_max": float(raw.get("norm_max", 1.0)),
            "forecast_lead_steps": int(raw.get("forecast_lead_steps", 1) or 1),
            "forecast_start_offset_steps": int(raw.get("forecast_start_offset_steps", 0) or 0),
            "examples": {},
            "tasks": {},
            "sampled_nodes": [],
            "exogenous_columns": [],
            "exogenous_dim": 0,
            "future_known_exogenous_columns": [],
            "issued_exogenous_columns": [str(value) for value in raw.get("issued_exogenous_columns", [])],
            "forecast_origin_from_history_end": bool(raw.get("forecast_origin_from_history_end", False)),
            "issued_forecast_metadata": dict(raw.get("issued_forecast_metadata") or {}),
            "issued_text_metadata": dict(raw.get("issued_text_metadata") or {}),
            "context_static_features": {
                "source": "train_split_target_and_node_metadata"
                if context_target_stats == "train"
                else "node_metadata_only_no_target_stats",
                "node_count": len(node_static_features),
                "fields": list(next(iter(node_static_features.values())).keys()) if node_static_features else [],
            },
        }
        for split_name in ("train", "val", "test"):
            max_windows = max_train_windows_per_dataset if split_name == "train" else max_eval_windows_per_dataset
            examples = _build_split_examples(
                dataset_name=dataset_name,
                split_name=split_name,
                data=raw,
                split_values=split.get(split_name),
                split_timestamps=split.get(f"timestamps_{split_name}"),
                history_len=history_len,
                horizon=horizon,
                stride=stride,
                max_nodes=max_nodes_per_dataset,
                max_windows=max_windows,
                exog_frame=exog_frame,
                exog_columns=global_exog_columns,
                future_known_columns=global_future_known_columns,
                node_static_features=node_static_features,
                context_identity_policy=context_identity_policy,
                text_schema_variant=text_schema_variant,
                future_exog_policy=future_exog_policy,
                dataset_label=int(dataset_label_map.get(dataset_name, 0)),
                node_label_map=node_label_map,
            )
            splits_out[split_name].extend(examples)
            dataset_manifest["examples"][split_name] = len(examples)
            if split_name == "train":
                task_counts: dict[str, int] = {}
                sampled_nodes: list[str] = []
                for ex in examples:
                    task_counts[ex.task_type] = task_counts.get(ex.task_type, 0) + 1
                    if ex.node_id not in sampled_nodes:
                        sampled_nodes.append(ex.node_id)
                if examples:
                    dataset_manifest["exogenous_dim"] = int(examples[0].history_exog_dim)
                    dataset_manifest["exogenous_columns"] = list(global_exog_columns)
                    dataset_manifest["future_known_exogenous_columns"] = list(global_future_known_columns)
                dataset_manifest["tasks"] = task_counts
                dataset_manifest["sampled_nodes"] = sampled_nodes
        manifest["datasets"][dataset_name] = dataset_manifest

    llm_context_manifest = None
    if context_embedding_mode != "numeric":
        all_examples = [example for examples in splits_out.values() for example in examples]
        llm_context_manifest = apply_llm_context_embeddings(
            all_examples,
            cache_path=str(llm_context_cache),
            mode=context_embedding_mode,
            missing_policy=llm_context_missing_policy,
        )
        manifest["llm_context"] = llm_context_manifest

    datasets = {name: EnergyWindowDataset(examples) for name, examples in splits_out.items()}
    manifest["examples_total"] = {name: len(ds) for name, ds in datasets.items()}
    first_example = next((example for examples in splits_out.values() for example in examples), None)
    manifest["context_dim"] = int(first_example.context.shape[0]) if first_example is not None else 0
    manifest["base_numeric_context_dim"] = int(context_dim())
    manifest["forecast_lead_steps"] = _forecast_lead_steps_for(next(iter(loaded_items))[1]) if loaded_items else 1
    manifest["forecast_start_offset_steps"] = (
        int(next(iter(loaded_items))[1].get("forecast_start_offset_steps", 0) or 0)
        if loaded_items
        else 0
    )
    if llm_context_manifest is not None:
        manifest["llm_context_dim"] = int(llm_context_manifest.get("llm_context_dim", 0))
    exog_dim = int(len(global_exog_columns))
    manifest["exogenous_columns"] = list(global_exog_columns)
    manifest["future_known_exogenous_columns"] = list(global_future_known_columns)
    manifest["history_exog_dim"] = exog_dim
    manifest["future_exog_dim"] = exog_dim
    manifest["dataset_label_map"] = dict(dataset_label_map)
    manifest["node_label_map"] = dict(node_label_map)
    manifest["dataset_label_count"] = int(len(dataset_label_map))
    manifest["node_label_count"] = int(len(node_label_map))
    manifest["numeric_layout"] = {
        "history_target": int(history_len),
        "history_time": int(4 * history_len),
        "history_exogenous": int(exog_dim * history_len),
        "future_time": int(4 * horizon),
        "future_exogenous": int(exog_dim * horizon),
    }
    manifest["numeric_dim"] = int(history_len + 4 * history_len + exog_dim * history_len + 4 * horizon + exog_dim * horizon)
    return datasets, manifest
