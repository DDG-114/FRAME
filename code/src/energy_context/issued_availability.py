"""Causal, per-field availability contract for issued AEMO forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IssuedAvailability:
    values: np.ndarray
    available: np.ndarray
    available_at: np.ndarray
    coverage: float
    latest_forecast_origin: pd.Timestamp | None
    latest_available_at: pd.Timestamp | None

    def __iter__(self):
        """Keep the legacy three-value unpacking contract."""

        yield self.values
        yield self.coverage
        yield self.latest_forecast_origin


def _base_name(column: str) -> str:
    return column.removeprefix("issued_")


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def release_column(frame: pd.DataFrame, column: str) -> str | None:
    base = _base_name(column)
    return _first_present(
        frame,
        (
            f"{column}_available_at",
            f"{base}_available_at",
            "available_at",
            "version_time",
            "forecast_origin",
        ),
    )


def availability_column(frame: pd.DataFrame, column: str) -> str | None:
    base = _base_name(column)
    return _first_present(frame, (f"{column}_available", f"{base}_available"))


def provenance_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        for name in (release_column(frame, column), availability_column(frame, column)):
            if name is not None and name not in selected:
                selected.append(name)
    return selected


def select_issued_block(
    frame: pd.DataFrame,
    *,
    forecast_origin: pd.Timestamp,
    future_timestamps,
    node_id: str,
    columns: list[str],
) -> IssuedAvailability:
    """Select values whose field-specific publication time is at the origin or earlier."""

    origin = pd.Timestamp(forecast_origin)
    future_index = pd.DatetimeIndex(future_timestamps)
    if future_index.has_duplicates:
        raise ValueError("future_timestamps must be unique")
    if (future_index <= origin).any():
        raise ValueError("Every requested issued lead must be later than forecast_origin")

    shape = (len(future_index), len(columns))
    values = np.zeros(shape, dtype=np.float32)
    available = np.zeros(shape, dtype=bool)
    available_at = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    if frame is None or getattr(frame, "empty", True) or not columns:
        return IssuedAvailability(values, available, available_at, 0.0, None, None)

    source = frame.reset_index() if "valid_time" not in frame.columns and frame.index.name == "valid_time" else frame.copy()
    required = {"valid_time", "node_id", *columns}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"issued_exogenous is missing fields {sorted(missing)}")
    for column in columns:
        if release_column(source, column) is None:
            raise ValueError(f"issued_exogenous has no publication timestamp for {column}")

    source["valid_time"] = pd.to_datetime(source["valid_time"])
    source = source.loc[
        source["node_id"].astype(str).eq(str(node_id))
        & source["valid_time"].isin(future_index)
    ].copy()
    if "forecast_origin" in source.columns:
        source["forecast_origin"] = pd.to_datetime(source["forecast_origin"])
        source = source.loc[source["forecast_origin"].le(origin)]
    if source.empty:
        return IssuedAvailability(values, available, available_at, 0.0, None, None)

    chosen_origins: list[pd.Timestamp] = []
    chosen_releases: list[pd.Timestamp] = []
    for position, column in enumerate(columns):
        published = release_column(source, column)
        flag = availability_column(source, column)
        candidates = source.copy()
        candidates["_value"] = pd.to_numeric(candidates[column], errors="coerce")
        candidates["_available_at"] = pd.to_datetime(candidates[published], errors="coerce")
        eligible = (
            np.isfinite(candidates["_value"])
            & candidates["_available_at"].notna()
            & candidates["_available_at"].le(origin)
        )
        if flag is not None:
            eligible &= candidates[flag].fillna(False).astype(bool)
        candidates = candidates.loc[eligible]
        if candidates.empty:
            continue
        order = ["valid_time", "_available_at"]
        if "forecast_origin" in candidates.columns:
            order.append("forecast_origin")
        chosen = candidates.sort_values(order).drop_duplicates("valid_time", keep="last")
        chosen = chosen.set_index("valid_time").reindex(future_index)
        numeric = pd.to_numeric(chosen["_value"], errors="coerce")
        release = pd.to_datetime(chosen["_available_at"], errors="coerce")
        mask = numeric.notna() & np.isfinite(numeric) & release.notna() & release.le(origin)
        values[:, position] = numeric.where(mask, 0.0).to_numpy(dtype=np.float32)
        available[:, position] = mask.to_numpy(dtype=bool)
        release_values = release.to_numpy(dtype="datetime64[ns]")
        available_at[available[:, position], position] = release_values[available[:, position]]
        chosen_releases.extend(pd.Timestamp(value) for value in release.loc[mask])
        if "forecast_origin" in chosen.columns:
            chosen_origins.extend(pd.Timestamp(value) for value in chosen.loc[mask, "forecast_origin"].dropna())

    coverage = float(available.mean()) if available.size else 0.0
    latest_origin = max(chosen_origins) if chosen_origins else None
    latest_release = max(chosen_releases) if chosen_releases else None
    if latest_release is not None and latest_release > origin:
        raise AssertionError("Issued publication time crossed forecast_origin")
    return IssuedAvailability(
        values=values,
        available=available,
        available_at=available_at,
        coverage=coverage,
        latest_forecast_origin=latest_origin,
        latest_available_at=latest_release,
    )
