"""Structured energy context helpers for the EnergyCA MVP."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

TASK_TYPES = (
    "load",
    "wind",
    "pv",
    "net_load",
    "market_signal",
    "price",
    "ev_demand",
    "unknown",
)
TASK_TO_ID = {name: idx for idx, name in enumerate(TASK_TYPES)}

BASE_CONTEXT_FIELD_NAMES = [
    *[f"task_onehot_{name}" for name in TASK_TYPES],
    "capacity_scaled",
    "has_weather",
    "has_price",
    "frequency_scaled",
    "forecast_hour_sin",
    "forecast_hour_cos",
    "forecast_dow_sin",
    "forecast_dow_cos",
    "is_weekend",
    "forecast_dayofyear_sin",
    "forecast_dayofyear_cos",
    "forecast_month_sin",
    "forecast_month_cos",
]

STATIC_CONTEXT_FIELD_NAMES = [
    "node_index_scaled",
    "capacity_rank",
    "latitude_scaled",
    "longitude_scaled",
    "coverage_log_scaled",
    "target_train_mean",
    "target_train_std",
    "target_train_min",
    "target_train_max",
    "target_train_q10",
    "target_train_q50",
    "target_train_q90",
    "target_train_iqr",
    "target_train_zero_fraction",
    "target_train_active_fraction",
    "target_train_day_mean",
    "target_train_night_mean",
    "target_train_weekend_mean",
    "target_train_peak_hour_sin",
    "target_train_peak_hour_cos",
]

WINDOW_CONTEXT_FIELD_NAMES = [
    "history_mean",
    "history_std",
    "history_min",
    "history_max",
    "history_q10",
    "history_q90",
    "history_last_minus_mean",
    "history_slope",
    "history_zero_fraction",
    "recent_7d_mean",
    "recent_7d_std",
    "recent_7d_slope",
    "recent_14d_mean",
    "recent_14d_std",
    "recent_14d_slope",
    "same_slot_prev_day",
    "same_slot_prev_week",
    "same_slot_prev_2week",
    "same_slot_prev_day_minus_prev_week",
    "same_slot_prev_week_minus_prev_2week",
]

HASH_CONTEXT_FIELD_NAMES = [
    "node_hash_sin",
    "node_hash_cos",
    "group_hash_sin",
    "group_hash_cos",
    "source_hash_sin",
    "source_hash_cos",
    "meta_hash_sin",
    "meta_hash_cos",
]

CONTEXT_FIELD_NAMES = (
    BASE_CONTEXT_FIELD_NAMES + STATIC_CONTEXT_FIELD_NAMES + WINDOW_CONTEXT_FIELD_NAMES + HASH_CONTEXT_FIELD_NAMES
)


@dataclass(frozen=True)
class EnergyContext:
    """Structured text plus numeric context used by MVP models."""

    text: str
    vector: np.ndarray
    task_type: str
    task_id: int


def infer_task_type(dataset_name: str, node_id: str = "", node_meta: dict[str, Any] | None = None) -> str:
    """Infer a coarse energy task label from dataset and node metadata."""
    node_meta = node_meta or {}
    haystack = " ".join(
        str(value).lower()
        for value in (
            dataset_name,
            node_id,
            node_meta.get("site_type", ""),
            node_meta.get("source_column", ""),
            node_meta.get("type", ""),
            node_meta.get("zone_type", ""),
        )
    )
    if any(token in haystack for token in ("solar", "pv", "photovoltaic")):
        return "pv"
    if "wind" in haystack:
        return "wind"
    if "net_load" in haystack or "net-load" in haystack or "net load" in haystack:
        return "net_load"
    if "price" in haystack:
        return "price"
    if any(token in haystack for token in ("market", "generation", "renewable", "tie_line", "storage")):
        return "market_signal"
    if any(token in haystack for token in ("ev", "charging", "occupancy", "st_evcdp", "urbanev")):
        return "ev_demand"
    if any(token in haystack for token in ("load", "demand", "wotai")):
        return "load"
    return "unknown"


def estimate_frequency_minutes(timestamps) -> float:
    """Estimate median timestamp interval in minutes."""
    try:
        ts = pd.DatetimeIndex(timestamps)
    except Exception:
        return 0.0
    if len(ts) < 2:
        return 0.0
    deltas = pd.Series(ts).diff().dropna().dt.total_seconds().to_numpy(dtype=float) / 60.0
    if deltas.size == 0:
        return 0.0
    return float(np.nanmedian(deltas))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _feature_value(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _safe_float(features.get(key, default), default=default)


def _first_present(meta: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return value
    return ""


def _stable_hash_unit(value: Any) -> float:
    text = str(value)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return float(integer / float((1 << 64) - 1))


def _hash_sincos(value: Any) -> tuple[float, float]:
    angle = 2.0 * math.pi * _stable_hash_unit(value)
    return float(math.sin(angle)), float(math.cos(angle))


def _node_meta_dict(node_meta, node_id: str, node_idx: int) -> dict[str, Any]:
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


def build_energy_context(
    *,
    dataset_name: str,
    node_id: str,
    node_idx: int,
    node_meta,
    has_weather: bool,
    has_price: bool,
    frequency_minutes: float,
    forecast_start,
    covariate_labels: list[str] | tuple[str, ...] | None = None,
    node_static_features: dict[str, Any] | None = None,
    window_numeric_features: dict[str, Any] | None = None,
    context_identity_policy: str = "full",
) -> EnergyContext:
    """Build structured context text and a compact numeric context vector."""
    if context_identity_policy not in {"full", "no_hash", "semantic_only"}:
        raise ValueError("context_identity_policy must be 'full', 'no_hash', or 'semantic_only'")
    meta = _node_meta_dict(node_meta, node_id=node_id, node_idx=node_idx)
    node_static_features = node_static_features or {}
    window_numeric_features = window_numeric_features or {}
    task_type = infer_task_type(dataset_name, node_id=node_id, node_meta=meta)
    task_id = TASK_TO_ID.get(task_type, TASK_TO_ID["unknown"])

    try:
        start = pd.Timestamp(forecast_start)
        hour = float(start.hour + start.minute / 60.0)
        dow = float(start.dayofweek)
        doy = float(start.dayofyear)
        month = float(start.month)
        is_weekend = 1.0 if start.dayofweek >= 5 else 0.0
    except Exception:
        hour = 0.0
        dow = 0.0
        doy = 0.0
        month = 0.0
        is_weekend = 0.0

    capacity = _safe_float(
        meta.get("capacity", meta.get("station_capacity", meta.get("rated_capacity", 0.0))),
        default=0.0,
    )
    capacity_scaled = float(np.log1p(max(capacity, 0.0)) / 10.0)
    freq_hours = float(max(frequency_minutes, 0.0) / 60.0)
    latitude = _safe_float(meta.get("latitude", meta.get("lat", 0.0)), default=0.0)
    longitude = _safe_float(meta.get("longitude", meta.get("lon", meta.get("lng", 0.0))), default=0.0)
    coverage_ratio = _safe_float(meta.get("coverage_ratio", 0.0), default=0.0)

    group_signature = _first_present(
        meta,
        ("gsp_group", "zone_id", "zone_type", "area", "source_site", "site_type", "type"),
    )
    source_signature = _first_present(
        meta,
        ("source_column", "raw_file", "source_variant", "source_site", "node_id"),
    )
    meta_signature = "|".join(
        f"{key}={meta.get(key)}"
        for key in sorted(meta)
        if key
        in {
            "area",
            "capacity",
            "coverage_ratio",
            "gsp_group",
            "lat",
            "latitude",
            "lon",
            "longitude",
            "raw_file",
            "source_column",
            "source_site",
            "source_variant",
            "type",
            "zone_id",
            "zone_type",
        }
    )

    one_hot = np.zeros(len(TASK_TYPES), dtype=np.float32)
    one_hot[task_id] = 1.0
    numeric = np.asarray(
        [
            capacity_scaled,
            1.0 if has_weather else 0.0,
            1.0 if has_price else 0.0,
            min(freq_hours, 24.0) / 24.0,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            is_weekend,
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ],
        dtype=np.float32,
    )
    static_numeric = np.asarray(
        [
            0.0
            if context_identity_policy == "semantic_only"
            else _feature_value(node_static_features, "node_index_scaled"),
            0.0
            if context_identity_policy == "semantic_only"
            else _feature_value(node_static_features, "capacity_rank"),
            float(np.clip(latitude / 90.0, -1.0, 1.0)),
            float(np.clip(longitude / 180.0, -1.0, 1.0)),
            float(np.log1p(max(coverage_ratio, 0.0)) / 5.0),
            _feature_value(node_static_features, "target_train_mean"),
            _feature_value(node_static_features, "target_train_std"),
            _feature_value(node_static_features, "target_train_min"),
            _feature_value(node_static_features, "target_train_max"),
            _feature_value(node_static_features, "target_train_q10"),
            _feature_value(node_static_features, "target_train_q50"),
            _feature_value(node_static_features, "target_train_q90"),
            _feature_value(node_static_features, "target_train_iqr"),
            _feature_value(node_static_features, "target_train_zero_fraction"),
            _feature_value(node_static_features, "target_train_active_fraction"),
            _feature_value(node_static_features, "target_train_day_mean"),
            _feature_value(node_static_features, "target_train_night_mean"),
            _feature_value(node_static_features, "target_train_weekend_mean"),
            _feature_value(node_static_features, "target_train_peak_hour_sin"),
            _feature_value(node_static_features, "target_train_peak_hour_cos"),
        ],
        dtype=np.float32,
    )
    window_numeric = np.asarray(
        [
            _feature_value(window_numeric_features, "history_mean"),
            _feature_value(window_numeric_features, "history_std"),
            _feature_value(window_numeric_features, "history_min"),
            _feature_value(window_numeric_features, "history_max"),
            _feature_value(window_numeric_features, "history_q10"),
            _feature_value(window_numeric_features, "history_q90"),
            _feature_value(window_numeric_features, "history_last_minus_mean"),
            _feature_value(window_numeric_features, "history_slope"),
            _feature_value(window_numeric_features, "history_zero_fraction"),
            _feature_value(window_numeric_features, "recent_7d_mean"),
            _feature_value(window_numeric_features, "recent_7d_std"),
            _feature_value(window_numeric_features, "recent_7d_slope"),
            _feature_value(window_numeric_features, "recent_14d_mean"),
            _feature_value(window_numeric_features, "recent_14d_std"),
            _feature_value(window_numeric_features, "recent_14d_slope"),
            _feature_value(window_numeric_features, "same_slot_prev_day"),
            _feature_value(window_numeric_features, "same_slot_prev_week"),
            _feature_value(window_numeric_features, "same_slot_prev_2week"),
            _feature_value(window_numeric_features, "same_slot_prev_day_minus_prev_week"),
            _feature_value(window_numeric_features, "same_slot_prev_week_minus_prev_2week"),
        ],
        dtype=np.float32,
    )
    if context_identity_policy == "full":
        hash_values = [
            *_hash_sincos(f"{dataset_name}:{node_id}"),
            *_hash_sincos(f"{dataset_name}:{group_signature}"),
            *_hash_sincos(f"{dataset_name}:{source_signature}"),
            *_hash_sincos(f"{dataset_name}:{meta_signature or node_id}"),
        ]
    else:
        hash_values = [0.0] * len(HASH_CONTEXT_FIELD_NAMES)
    hash_features = np.asarray(hash_values, dtype=np.float32)
    vector = np.concatenate([one_hot, numeric, static_numeric, window_numeric, hash_features]).astype(np.float32)

    if covariate_labels is not None:
        covariates = [str(label).strip() for label in covariate_labels if str(label).strip()]
    else:
        covariates = []
        if has_weather:
            covariates.append("weather")
        if has_price:
            covariates.append("price/system")
    covariate_text = ", ".join(covariates) if covariates else "none"
    text = (
        f"Task: {task_type} forecasting. "
        f"Dataset: {dataset_name}. Node: {node_id}. "
        f"Frequency: {frequency_minutes:.1f} minutes. "
        f"Known covariates: {covariate_text}. "
        f"Capacity: {capacity:.3f}. "
        f"Train target mean/std: "
        f"{_feature_value(node_static_features, 'target_train_mean'):.4f}/"
        f"{_feature_value(node_static_features, 'target_train_std'):.4f}."
    )
    return EnergyContext(text=text, vector=vector, task_type=task_type, task_id=task_id)


def context_field_names() -> list[str]:
    return list(CONTEXT_FIELD_NAMES)


def context_dim() -> int:
    return len(CONTEXT_FIELD_NAMES)
