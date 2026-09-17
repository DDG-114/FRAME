"""Loader for the Jiangsu horizontal-project short-term load dataset."""
from __future__ import annotations

import pickle
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


RAW_DIR = Path("江苏横向/_extracted/负荷数据")
# Keep a separate cache for the strict day-ahead variant so old relaxed
# pickles cannot silently leak into new runs.
PROCESSED_PATH = Path("data/processed/jiangsu_load_strict_v2.pkl")
LONG_PROCESSED_PATH = Path("data/processed/jiangsu_load_long.pkl")

ACTUAL_PATH = RAW_DIR / "近一年负荷实际数据.xlsx"
BOUNDARY_PATH = RAW_DIR / "近一年负荷边界数据.xlsx"
CLEARING_PATH = RAW_DIR / "近一年负荷出清数据.xlsx"

TARGET_COLUMN = "系统负荷"
CLEAN_START = pd.Timestamp("2025-09-03 00:15:00")
VAL_START = pd.Timestamp("2026-04-25 00:15:00")
TEST_START = pd.Timestamp("2026-05-25 00:15:00")
# D-1 09:00 -> D 00:15 on a 15-minute grid.
STRICT_FORECAST_LEAD_STEPS = 61
# Keep extra rows before validation/test starts so the first forecast window can
# still use the full strict history cutoff.
STRICT_WARMUP = pd.Timedelta(days=3)

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _tag(name: str) -> str:
    return f"{{{MAIN_NS}}}{name}"


def _column_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch) - ord("A") + 1
    return idx - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall(_tag("si")):
        text = "".join(node.text or "" for node in si.iter(_tag("t")))
        strings.append(text)
    return strings


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {
        rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall(f"{{{REL_NS}}}Relationship")
    }
    first_sheet = workbook.find(f"{_tag('sheets')}/{_tag('sheet')}")
    if first_sheet is None:
        raise ValueError("Workbook does not contain any sheet.")
    rid = first_sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
    target = rid_to_target[rid]
    return target if target.startswith("xl/") else f"xl/{target}"


def _read_rows(path: Path) -> list[list[object]]:
    with zipfile.ZipFile(path) as zf:
        shared = _read_shared_strings(zf)
        sheet_path = _first_sheet_path(zf)
        rows: list[list[object]] = []
        with zf.open(sheet_path) as fh:
            for _, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag != _tag("row"):
                    continue
                values: dict[int, object] = {}
                max_idx = -1
                for cell in elem.findall(_tag("c")):
                    ref = cell.attrib.get("r", "A1")
                    idx = _column_index(ref)
                    max_idx = max(max_idx, idx)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(_tag("v"))
                    if cell_type == "s":
                        value = shared[int(value_node.text)] if value_node is not None and value_node.text else ""
                    elif cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.iter(_tag("t")))
                    else:
                        raw = value_node.text if value_node is not None else ""
                        try:
                            value = float(raw) if raw != "" else ""
                        except ValueError:
                            value = raw
                    values[idx] = value
                if max_idx >= 0:
                    rows.append([values.get(i, "") for i in range(max_idx + 1)])
                elem.clear()
        return rows


def _safe_header(header_row: list[object], width: int) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for idx in range(width):
        name = str(header_row[idx]).strip() if idx < len(header_row) else ""
        if not name:
            name = f"COL_{idx + 1}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count > 1:
            name = f"{name}_{count}"
        names.append(name)
    return names


def _parse_timestamp(date_value: object, time_value: object) -> pd.Timestamp:
    date_text = str(date_value).strip()
    timestamp = pd.to_datetime(date_text, errors="coerce", dayfirst=True)
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse date cell: {date_value!r}")
    base = pd.Timestamp(timestamp).normalize()
    if time_value in ("", None):
        return base
    minutes = int(round(float(time_value) * 24.0 * 60.0))
    return base + pd.to_timedelta(minutes, unit="m")


def _frame_from_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Jiangsu load workbook: {path}")
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in workbook: {path}")
    width = max(len(row) for row in rows)
    header = _safe_header(rows[0], width)
    frame = pd.DataFrame(rows[1:], columns=header)
    frame["timestamp"] = [
        _parse_timestamp(row.get("日期", ""), row.get("时间", ""))
        for _, row in frame.iterrows()
    ]
    frame = frame.drop(columns=[col for col in ("日期", "时间") if col in frame.columns])
    for column in frame.columns:
        if column == "timestamp":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _normalise_from_indices(values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    source = values[train_idx] if len(train_idx) > 0 else values
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    denom = max(vmax - vmin, 1e-8)
    return ((values - vmin) / denom).astype(np.float32), vmin, vmax


def _prepare_combined_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actual = _frame_from_xlsx(ACTUAL_PATH)
    boundary = _frame_from_xlsx(BOUNDARY_PATH).add_prefix("boundary_")
    clearing = _frame_from_xlsx(CLEARING_PATH).add_prefix("clearing_")

    combined = actual.join(boundary, how="inner").join(clearing, how="inner")
    combined = combined.loc[combined.index >= CLEAN_START].copy()
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq="15min")
    combined = combined.reindex(full_index)
    if TARGET_COLUMN not in actual.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} in Jiangsu actual load file.")

    combined = combined.loc[combined[TARGET_COLUMN].notna()].copy()
    combined = combined.loc[combined[TARGET_COLUMN] > 0].copy()

    boundary_cols = [col for col in combined.columns if col.startswith("boundary_")]
    clearing_cols = [col for col in combined.columns if col.startswith("clearing_")]
    combined.loc[:, boundary_cols] = combined.loc[:, boundary_cols].interpolate(limit_direction="both").ffill().bfill()
    combined.loc[:, clearing_cols] = combined.loc[:, clearing_cols].interpolate(limit_direction="both").ffill().bfill()
    combined.loc[:, TARGET_COLUMN] = combined.loc[:, TARGET_COLUMN].interpolate(limit_direction="both").ffill().bfill()

    # The strict short-term load benchmark should only expose the boundary
    # forecast family as future-known input. Clearing is published later and is
    # therefore kept out of the model input path.
    weather = combined.loc[:, boundary_cols].copy()
    price = pd.DataFrame(index=combined.index)
    return combined, weather, price


def _prepare_long_components(
    clean_start: pd.Timestamp = CLEAN_START,
) -> dict:
    actual = _frame_from_xlsx(ACTUAL_PATH)
    boundary = _frame_from_xlsx(BOUNDARY_PATH)
    clearing = _frame_from_xlsx(CLEARING_PATH)

    actual = actual.loc[actual.index >= clean_start].copy()
    boundary = boundary.loc[boundary.index >= clean_start].copy()
    clearing = clearing.loc[clearing.index >= clean_start].copy()

    full_start = min(actual.index.min(), boundary.index.min(), clearing.index.min()).floor("D")
    full_end = max(actual.index.max(), boundary.index.max(), clearing.index.max()).ceil("D")
    full_index = pd.date_range(full_start, full_end, freq="15min")

    def _reindex_and_fill(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reindex(full_index).sort_index()
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[:, :] = frame.loc[:, :].interpolate(limit_direction="both").ffill().bfill()
        return frame

    actual = _reindex_and_fill(actual)
    boundary = _reindex_and_fill(boundary)
    clearing = _reindex_and_fill(clearing)

    combined = pd.concat(
        [
            actual.add_prefix("actual_"),
            boundary.add_prefix("boundary_"),
            clearing.add_prefix("clearing_"),
        ],
        axis=1,
    )

    return {
        "timestamps": pd.DatetimeIndex(full_index),
        "actual": actual,
        "boundary": boundary,
        "clearing": clearing,
        "combined": combined,
        "actual_columns": list(actual.columns),
        "boundary_columns": list(boundary.columns),
        "clearing_columns": list(clearing.columns),
        "clean_start": clean_start.isoformat(),
        "full_start": pd.Timestamp(full_start).isoformat(),
        "full_end": pd.Timestamp(full_end).isoformat(),
        "source_dataset": "Jiangsu horizontal load long-horizon",
    }


def load_jiangsu_load_long(
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    processed = Path(processed_path) if processed_path is not None else LONG_PROCESSED_PATH
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as fh:
            return pickle.load(fh)

    result = _prepare_long_components()
    result["protocol"] = {
        "name": "jiangsu_long_horizon_load",
        "clean_start": result["clean_start"],
        "full_start": result["full_start"],
        "full_end": result["full_end"],
        "target": TARGET_COLUMN,
        "forecast_horizons_days": [1, 2, 3, 4, 5, 6],
        "recommended_train_origin_start": "2025-09-18T00:00:00",
        "recommended_train_origin_end": "2026-05-18T00:00:00",
        "recommended_test_origin_start": "2026-05-25T00:00:00",
        "recommended_test_origin_end": "2026-06-17T00:00:00",
    }

    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_jiangsu_load(
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    processed = Path(processed_path) if processed_path is not None else PROCESSED_PATH
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as fh:
            return pickle.load(fh)

    combined, weather, price = _prepare_combined_frame()
    timestamps = pd.DatetimeIndex(combined.index)
    occ_raw = combined[[TARGET_COLUMN]].to_numpy(dtype=np.float32)

    train_idx = np.flatnonzero((timestamps >= CLEAN_START) & (timestamps < VAL_START))
    val_idx = np.flatnonzero((timestamps >= (VAL_START - STRICT_WARMUP)) & (timestamps < TEST_START))
    test_idx = np.flatnonzero(timestamps >= (TEST_START - STRICT_WARMUP))
    occ_norm, vmin, vmax = _normalise_from_indices(occ_raw, train_idx)

    capacity = float(np.nanmax(occ_raw)) if occ_raw.size else 0.0
    node_meta = pd.DataFrame(
        [
            {
                "node_id": "jiangsu_system_load",
                "site_type": "load",
                "source_column": TARGET_COLUMN,
                "source_site": "jiangsu_horizontal",
                "area": "jiangsu",
                "capacity": capacity,
                "raw_file": str(ACTUAL_PATH),
                "source_variant": "actual_target_boundary_only_strict_day_ahead",
            }
        ]
    ).set_index("node_id")

    result = {
        "occupancy": occ_norm.reshape(-1, 1),
        "occupancy_raw": occ_raw.reshape(-1, 1),
        "timestamps": timestamps,
        "node_ids": ["jiangsu_system_load"],
        "node_meta": node_meta,
        "adj": np.eye(1, dtype=np.float32),
        "weather": weather,
        "price": price,
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_jiangsu_proxy_before_validation",
        "forecast_lead_steps": STRICT_FORECAST_LEAD_STEPS,
        "covariate_labels": ["boundary forecast"],
        "split_indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "split_target_start_times": {
            "val": VAL_START,
            "test": TEST_START,
        },
        "protocol": {
            "name": "jiangsu_strict_day_ahead_load",
            "clean_start": CLEAN_START.isoformat(),
            "train_end": VAL_START.isoformat(),
            "val_start": VAL_START.isoformat(),
            "test_start": TEST_START.isoformat(),
            "target": TARGET_COLUMN,
            "future_covariates": "boundary_only",
            "forecast_lead_steps": STRICT_FORECAST_LEAD_STEPS,
            "warmup_days": int(STRICT_WARMUP / pd.Timedelta(days=1)),
            "information_cutoff": "D-1 09:00",
        },
        "source_dataset": "Jiangsu horizontal load strict day-ahead",
    }

    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


if __name__ == "__main__":
    data = load_jiangsu_load(force_reprocess=True)
    print(f"occupancy shape: {data['occupancy'].shape}")
    print(f"raw range: {data['norm_min']:.3f}..{data['norm_max']:.3f}")
    for split, idx in data["split_indices"].items():
        ts = data["timestamps"][idx]
        print(f"{split}: {len(idx)} rows, {ts[0]}..{ts[-1]}")
