"""Loader for the GEFCom2014 load forecasting track.

Expected staging layouts include either:

```text
data/raw/aaai_2026/gefc2014_load/load/Task 1/L1-train.csv
data/raw/aaai_2026/gefc2014_load/load/Task 2/L2-train.csv
...
```

or the same structure nested below an extracted `GEFCom2014 Data` directory.
The loader is deliberately tolerant because public mirrors package the
competition archive slightly differently.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRAIN_SPLIT_RATIO = 0.7
TASK15_HOLDOUT_START = pd.Timestamp("2011-12-01 00:00:00")
TASK15_VAL_START = pd.Timestamp("2011-11-01 00:00:00")
TASK15_HISTORY_PREFIX_HOURS = 168


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    return df


def _task_number(path: Path) -> int:
    match = re.search(r"Task\s*([0-9]+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _find_train_files(raw_dir: Path) -> list[Path]:
    patterns = [
        "**/Task */L*-train.csv",
        "**/Task */L *-train.csv",
        "**/*train*.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.glob(pattern))
    unique = sorted({path.resolve() for path in files}, key=lambda p: (_task_number(p), str(p)))
    return [Path(path) for path in unique]


def _find_extended_excel(raw_dir: Path) -> Path | None:
    candidates = list(raw_dir.glob("**/GEFCom2014-E*.xlsx")) + list(raw_dir.glob("**/GEFCom2014-E*.xls"))
    return candidates[0] if candidates else None


def _find_task15_solution_load(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("**/solution15_L.csv"))
    return candidates[0] if candidates else None


def _find_task15_solution_temperature(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("**/solution15_L_temperature.csv"))
    return candidates[0] if candidates else None


def _combine_date_hour(df: pd.DataFrame) -> pd.Series | None:
    lower_to_col = {str(col).lower(): col for col in df.columns}
    date_col = lower_to_col.get("date")
    hour_col = lower_to_col.get("hour")
    if date_col is None or hour_col is None:
        return None
    date = pd.to_datetime(df[date_col], errors="coerce")
    hour = pd.to_numeric(df[hour_col], errors="coerce").fillna(1)
    return date + pd.to_timedelta(hour.astype(int).clip(lower=1) - 1, unit="h")


def _compact_gefc_timestamp_candidates(value: str) -> list[pd.Timestamp]:
    raw = str(value).strip()
    match = re.fullmatch(r"([0-9]{2,8})\s+([0-9]{1,2})(?::([0-9]{1,2}))?", raw)
    if not match:
        return []
    date_digits, hour_text, minute_text = match.groups()
    if len(date_digits) < 6:
        return []
    year = int(date_digits[-4:])
    md = date_digits[:-4]
    candidates: list[tuple[int, int]] = []
    if len(md) == 2:
        candidates.append((int(md[0]), int(md[1])))
    elif len(md) == 3:
        first_two_month = int(md[:2])
        if 10 <= first_two_month <= 12:
            candidates.append((first_two_month, int(md[2:])))
        candidates.append((int(md[0]), int(md[1:])))
    elif len(md) == 4:
        candidates.append((int(md[:2]), int(md[2:])))

    hour = int(hour_text)
    minute = int(minute_text or 0)
    out: list[pd.Timestamp] = []
    for month, day in candidates:
        try:
            base = pd.Timestamp(year=year, month=month, day=day)
            if hour == 0:
                timestamp = base - pd.Timedelta(hours=1) + pd.Timedelta(minutes=minute)
            elif hour == 24:
                timestamp = base + pd.Timedelta(hours=23, minutes=minute)
            else:
                timestamp = base + pd.Timedelta(hours=hour - 1, minutes=minute)
            if timestamp not in out:
                out.append(timestamp)
        except ValueError:
            continue
    return out


def _parse_compact_gefc_timestamp(values: pd.Series) -> pd.Series:
    """Parse official GEFCom2014 load timestamps such as `112001 1:00`.

    The date part is month/day/year without separators, with the year occupying
    the last four digits. Three-digit month/day prefixes are ambiguous: for
    example, `1212011` can mean either Jan 21 or Dec 1. The parser therefore
    resolves candidates across the whole series by selecting the path that is
    closest to hourly continuity.

    The load-track compact timestamp follows the same hour convention as the
    official solution file's `date`/`hour` columns: hour 1 is the first hour of
    the day. Some files encode the final hour as the next date with hour 0
    (for example `112012 0:00` means 2011-12-31 23:00).
    """
    candidate_rows = [_compact_gefc_timestamp_candidates(value) for value in values.astype(str)]
    if not any(candidate_rows):
        return pd.Series([pd.NaT] * len(values), index=values.index, dtype="datetime64[ns]")

    prev_candidates: list[pd.Timestamp] = []
    prev_costs: list[float] = []
    backpointers: list[list[int | None]] = []
    candidate_rows_with_nat: list[list[pd.Timestamp | pd.NaT]] = []

    for candidates in candidate_rows:
        if not candidates:
            candidate_rows_with_nat.append([pd.NaT])
            backpointers.append([None])
            prev_candidates = []
            prev_costs = []
            continue

        candidate_rows_with_nat.append(candidates)
        if not prev_candidates:
            prev_candidates = candidates
            prev_costs = [0.0] * len(candidates)
            backpointers.append([None] * len(candidates))
            continue

        costs: list[float] = []
        pointers: list[int] = []
        for candidate in candidates:
            transition_costs = []
            for prev_idx, previous in enumerate(prev_candidates):
                expected = previous + pd.Timedelta(hours=1)
                diff_hours = abs((candidate - expected).total_seconds()) / 3600.0
                transition_costs.append(prev_costs[prev_idx] + diff_hours)
            best_idx = int(np.argmin(transition_costs))
            costs.append(float(transition_costs[best_idx]))
            pointers.append(best_idx)
        prev_candidates = candidates
        prev_costs = costs
        backpointers.append(pointers)

    parsed: list[pd.Timestamp | pd.NaT] = [pd.NaT] * len(candidate_rows_with_nat)
    current_idx: int | None = int(np.argmin(prev_costs)) if prev_costs else None
    for row_idx in range(len(candidate_rows_with_nat) - 1, -1, -1):
        candidates = candidate_rows_with_nat[row_idx]
        if current_idx is None or pd.isna(candidates[current_idx]):
            parsed[row_idx] = pd.NaT
            current_idx = None
            continue
        parsed[row_idx] = candidates[current_idx]
        current_idx = backpointers[row_idx][current_idx]
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns]")


def _timestamp_series(df: pd.DataFrame) -> pd.Series:
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in {"timestamp", "time", "datetime", "date_time"}:
            compact = _parse_compact_gefc_timestamp(df[column])
            if compact.notna().mean() >= 0.8:
                return compact
            ts = pd.to_datetime(df[column], errors="coerce", format="mixed")
            if ts.notna().mean() >= 0.8:
                return ts
            if compact.notna().any():
                return compact
    combined = _combine_date_hour(df)
    if combined is not None and combined.notna().any():
        return combined
    raise ValueError("Could not identify timestamp columns. Expected TIMESTAMP/time or Date+Hour.")


def _zone_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in {"zoneid", "zone_id", "zone", "area", "region"}:
            return str(column)
    return None


def _load_column(df: pd.DataFrame) -> str:
    candidates = []
    for column in df.columns:
        lowered = str(column).lower()
        if "load" in lowered and "forecast" not in lowered:
            candidates.append(str(column))
    if candidates:
        return candidates[0]
    for column in df.columns:
        if str(column).lower() in {"y", "target", "demand"}:
            return str(column)
    raise ValueError("Could not identify load/target column.")


def _temperature_columns(df: pd.DataFrame, *, exclude: Iterable[str]) -> list[str]:
    excluded = {str(col).lower() for col in exclude}
    out: list[str] = []
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in excluded:
            continue
        if lowered in {"t", "temp", "temperature"} or re.fullmatch(r"t[0-9]+", lowered):
            out.append(str(column))
    if out:
        return out

    numeric_columns = []
    for column in df.columns:
        lowered = str(column).lower()
        if lowered in excluded or lowered in {"hour", "month", "day", "year"}:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().mean() > 0.8:
            numeric_columns.append(str(column))
    return numeric_columns


def _read_task_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _normalise_columns(pd.read_csv(path))
    ts = _timestamp_series(df)
    load_col = _load_column(df)
    zone_col = _zone_column(df)
    exclude = {load_col}
    if zone_col:
        exclude.add(zone_col)
    exclude.update([col for col in df.columns if str(col).lower() in {"timestamp", "time", "datetime", "date", "hour"}])
    temp_cols = _temperature_columns(df, exclude=exclude)

    working = df.copy()
    working["timestamp"] = pd.DatetimeIndex(ts)
    working = working.loc[working["timestamp"].notna()].sort_values("timestamp")
    working[load_col] = pd.to_numeric(working[load_col], errors="coerce")
    working = working.loc[working[load_col].notna()]
    if zone_col:
        working[zone_col] = working[zone_col].astype(str)
        occ = working.pivot_table(index="timestamp", columns=zone_col, values=load_col, aggfunc="mean")
        occ.columns = [f"zone_{col}" for col in occ.columns]
        node_meta = pd.DataFrame(
            {
                "node_id": list(occ.columns),
                "zone_id": [str(col).replace("zone_", "", 1) for col in occ.columns],
                "site_type": "load",
                "type": "demand",
            }
        ).set_index("node_id")
    else:
        occ = working.set_index("timestamp")[[load_col]].rename(columns={load_col: "load_zone_1"})
        node_meta = pd.DataFrame(
            {"node_id": ["load_zone_1"], "zone_id": ["1"], "site_type": ["load"], "type": ["demand"]}
        ).set_index("node_id")

    weather = pd.DataFrame(index=occ.index)
    if temp_cols:
        temp = working[["timestamp", *temp_cols]].copy()
        for column in temp_cols:
            temp[column] = pd.to_numeric(temp[column], errors="coerce")
        weather = temp.groupby("timestamp")[temp_cols].mean().sort_index()
    return occ, weather, node_meta


def _read_extended_excel(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _normalise_columns(pd.read_excel(path))
    ts = _timestamp_series(df)
    load_col = _load_column(df)
    temp_cols = _temperature_columns(
        df,
        exclude={
            load_col,
            *[col for col in df.columns if str(col).lower() in {"timestamp", "time", "datetime", "date", "hour"}],
        },
    )
    working = df.copy()
    working["timestamp"] = pd.DatetimeIndex(ts)
    working = working.loc[working["timestamp"].notna()].sort_values("timestamp")
    occ = working.set_index("timestamp")[[load_col]].rename(columns={load_col: "load_zone_1"})
    weather = pd.DataFrame(index=occ.index)
    if temp_cols:
        weather = working.set_index("timestamp")[temp_cols].apply(pd.to_numeric, errors="coerce")
    node_meta = pd.DataFrame(
        {"node_id": ["load_zone_1"], "zone_id": ["1"], "site_type": ["load"], "type": ["demand"]}
    ).set_index("node_id")
    return occ, weather, node_meta


def _load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_files = _find_train_files(raw_dir)
    if train_files:
        occ_frames: list[pd.DataFrame] = []
        weather_frames: list[pd.DataFrame] = []
        node_meta_frames: list[pd.DataFrame] = []
        for path in train_files:
            occ, weather, node_meta = _read_task_csv(path)
            occ_frames.append(occ)
            if not weather.empty:
                weather_frames.append(weather)
            node_meta_frames.append(node_meta)
        occupancy = pd.concat(occ_frames).sort_index()
        occupancy = occupancy[~occupancy.index.duplicated(keep="last")]
        occupancy = occupancy.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        weather = pd.DataFrame(index=occupancy.index)
        if weather_frames:
            weather = pd.concat(weather_frames).sort_index()
            weather = weather[~weather.index.duplicated(keep="last")]
            weather = weather.reindex(occupancy.index).ffill().bfill()
        node_meta = pd.concat(node_meta_frames)
        node_meta = node_meta[~node_meta.index.duplicated(keep="last")]
        node_meta = node_meta.reindex(occupancy.columns)
        return occupancy, weather, node_meta

    excel = _find_extended_excel(raw_dir)
    if excel:
        occupancy, weather, node_meta = _read_extended_excel(excel)
        occupancy = occupancy.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        weather = weather.reindex(occupancy.index).ffill().bfill() if not weather.empty else weather
        return occupancy, weather, node_meta

    raise FileNotFoundError(
        f"No GEFCom2014 load train files found under {raw_dir}. "
        "Expected load/Task n/Ln-train.csv or GEFCom2014-E.xlsx."
    )


def _load_task15_solution(raw_dir: Path, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    load_path = _find_task15_solution_load(raw_dir)
    temp_path = _find_task15_solution_temperature(raw_dir)
    if load_path is None and temp_path is None:
        raise FileNotFoundError(
            f"No Task 15 solution files found under {raw_dir}. "
            "Expected solution15_L.csv and/or solution15_L_temperature.csv."
        )

    if load_path is not None:
        occ, _, _ = _read_task_csv(load_path)
    else:
        occ, _, _ = _read_task_csv(temp_path)  # type: ignore[arg-type]
    if len(columns) == 1 and list(occ.columns) != columns:
        occ = occ.rename(columns={occ.columns[0]: columns[0]})
    occ = occ.reindex(columns=columns)

    weather = pd.DataFrame(index=occ.index)
    if temp_path is not None:
        _, weather, _ = _read_task_csv(temp_path)
        weather = weather.reindex(occ.index).interpolate(limit_direction="both").ffill().bfill()
    return occ, weather


def _normalise_train_only(occ_raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    train_end = max(1, int(len(occ_raw) * TRAIN_SPLIT_RATIO))
    source = occ_raw[:train_end]
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    occ_norm = (occ_raw - vmin) / (vmax - vmin + 1e-8)
    return occ_norm.astype(np.float32), vmin, vmax


def _normalise_from_indices(occ_raw: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, float, float]:
    if len(train_indices) == 0:
        return _normalise_train_only(occ_raw)
    source = occ_raw[np.asarray(train_indices, dtype=np.int64)]
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    occ_norm = (occ_raw - vmin) / (vmax - vmin + 1e-8)
    return occ_norm.astype(np.float32), vmin, vmax


def _task15_split_indices(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    test_prefix_start = TASK15_HOLDOUT_START - pd.Timedelta(hours=TASK15_HISTORY_PREFIX_HOURS)
    positions = np.arange(len(timestamps), dtype=np.int64)
    train = positions[timestamps < TASK15_VAL_START]
    val = positions[(timestamps >= TASK15_VAL_START) & (timestamps < TASK15_HOLDOUT_START)]
    test = positions[timestamps >= test_prefix_start]
    return {"train": train, "val": val, "test": test}


def load_gefc2014_load(
    raw_dir: Path = Path("data/raw/aaai_2026/gefc2014_load"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    processed = Path(processed_path) if processed_path is not None else Path("data/processed/gefc2014_load.pkl")
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as f:
            return pickle.load(f)

    occupancy_df, weather, node_meta = _load_raw(Path(raw_dir))
    occupancy_df = occupancy_df.sort_index()
    occupancy_df = occupancy_df[~occupancy_df.index.duplicated(keep="last")]
    occupancy_df = occupancy_df.asfreq("h").interpolate(limit_direction="both").ffill().bfill()
    weather = weather.reindex(occupancy_df.index).interpolate(limit_direction="both").ffill().bfill() if not weather.empty else weather

    node_ids = [str(col) for col in occupancy_df.columns]
    occ_raw = occupancy_df.to_numpy(dtype=np.float32)
    if np.isnan(occ_raw).any():
        occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    occ_norm, vmin, vmax = _normalise_train_only(occ_raw)
    adj = np.eye(len(node_ids), dtype=np.float32)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": pd.DatetimeIndex(occupancy_df.index),
        "node_ids": node_ids,
        "node_meta": node_meta.reindex(node_ids),
        "adj": adj,
        "weather": weather,
        "price": pd.DataFrame(),
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_only",
        "source_dataset": "GEFCom2014-Load",
    }
    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_gefc2014_load_task15(
    raw_dir: Path = Path("data/raw/aaai_2026/gefc2014_load"),
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    """Load GEFCom2014-Load with the official Task 15 solution as test target.

    The train/validation split remains chronological over released train files:
    train targets end before November 2011, validation targets are November
    2011, and test targets are the official December 2011 Task 15 solution. The
    test split includes the last seven November days only as history prefix so
    day-ahead windows can start exactly on 2011-12-01.
    """
    processed = (
        Path(processed_path)
        if processed_path is not None
        else Path("data/processed/gefc2014_load_task15.pkl")
    )
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as f:
            return pickle.load(f)

    occupancy_hist, weather_hist, node_meta = _load_raw(Path(raw_dir))
    occupancy_hist = occupancy_hist.sort_index()
    occupancy_hist = occupancy_hist[~occupancy_hist.index.duplicated(keep="last")]
    occupancy_hist = occupancy_hist.asfreq("h").interpolate(limit_direction="both").ffill().bfill()
    weather_hist = (
        weather_hist.reindex(occupancy_hist.index).interpolate(limit_direction="both").ffill().bfill()
        if not weather_hist.empty
        else weather_hist
    )

    solution_occ, solution_weather = _load_task15_solution(Path(raw_dir), [str(col) for col in occupancy_hist.columns])
    solution_occ = solution_occ.sort_index()
    solution_occ = solution_occ[~solution_occ.index.duplicated(keep="last")]
    solution_occ = solution_occ.asfreq("h").interpolate(limit_direction="both").ffill().bfill()
    solution_occ = solution_occ.loc[solution_occ.index >= TASK15_HOLDOUT_START]
    solution_weather = solution_weather.reindex(solution_occ.index).interpolate(limit_direction="both").ffill().bfill()

    occupancy_df = pd.concat([occupancy_hist, solution_occ]).sort_index()
    occupancy_df = occupancy_df[~occupancy_df.index.duplicated(keep="last")]
    occupancy_df = occupancy_df.asfreq("h").interpolate(limit_direction="both").ffill().bfill()

    weather_frames = []
    if not weather_hist.empty:
        weather_frames.append(weather_hist)
    if not solution_weather.empty:
        weather_frames.append(solution_weather)
    weather = pd.DataFrame(index=occupancy_df.index)
    if weather_frames:
        weather = pd.concat(weather_frames).sort_index()
        weather = weather[~weather.index.duplicated(keep="last")]
        weather = weather.reindex(occupancy_df.index).interpolate(limit_direction="both").ffill().bfill()

    timestamps = pd.DatetimeIndex(occupancy_df.index)
    split_indices = _task15_split_indices(timestamps)
    node_ids = [str(col) for col in occupancy_df.columns]
    occ_raw = occupancy_df.to_numpy(dtype=np.float32)
    if np.isnan(occ_raw).any():
        occ_raw = np.nan_to_num(occ_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    occ_norm, vmin, vmax = _normalise_from_indices(occ_raw, split_indices["train"])
    adj = np.eye(len(node_ids), dtype=np.float32)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occ_raw,
        "timestamps": timestamps,
        "node_ids": node_ids,
        "node_meta": node_meta.reindex(node_ids),
        "adj": adj,
        "weather": weather,
        "price": pd.DataFrame(),
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "task15_train_indices",
        "source_dataset": "GEFCom2014-Load Task 15",
        "split_indices": split_indices,
        "split_target_start_times": {"test": TASK15_HOLDOUT_START},
        "protocol": {
            "name": "gefc2014_task15_holdout",
            "train_end_exclusive": TASK15_VAL_START.isoformat(),
            "val_start": TASK15_VAL_START.isoformat(),
            "test_target_start": TASK15_HOLDOUT_START.isoformat(),
            "test_history_prefix_hours": TASK15_HISTORY_PREFIX_HOURS,
        },
    }
    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result
