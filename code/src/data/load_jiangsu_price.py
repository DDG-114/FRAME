"""Load the Jiangsu spot-price forecasting dataset.

The raw materials currently available under ``江苏横向/_extracted`` provide
quarter-hour price curves for three market units (#4/#5/#6) in two forms:

* day-ahead price files with ``节点临时`` and ``节点最终``
* realtime price files with ``节点临时``, ``节点最终``, ``统一临时`` and ``统一最终``

This loader keeps the primary forecasting target as the day-ahead
``节点最终`` curve for each unit, and exposes the other observed price series
as historical context in ``price_history`` / ``day_ahead`` / ``realtime`` tables.

The aim is to keep the data interface compatible with the rest of the project
while staying conservative about leakage:

* no future-known exogenous variables are staged yet
* only historical observed price curves are exposed as auxiliary context
* the split is chronological and uses the latest available June window as test
"""
from __future__ import annotations

import pickle
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from src.data.load_jiangsu_load import load_jiangsu_load


RAW_DIR_DAY_AHEAD = Path("江苏横向/_extracted/日前电价")
RAW_DIR_REALTIME = Path("江苏横向/_extracted/实时电价")
PROCESSED_PATH = Path("data/processed/jiangsu_price.pkl")

DAY_AHEAD_FILES = {
    4: RAW_DIR_DAY_AHEAD / "26年后#4机组日前电价.xlsx",
    5: RAW_DIR_DAY_AHEAD / "26年后#5机组日前电价.xlsx",
    6: RAW_DIR_DAY_AHEAD / "26年后#6机组日前电价.xlsx",
}
REALTIME_FILES = {
    4: RAW_DIR_REALTIME / "26年以来#4机组实时电价数据.xlsx",
    5: RAW_DIR_REALTIME / "26年以来#5机组实时电价数据.xlsx",
    6: RAW_DIR_REALTIME / "26年以来#6机组实时电价数据.xlsx",
}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

TRAIN_START = pd.Timestamp("2026-01-01 00:15:00")
VAL_START = pd.Timestamp("2026-05-01 00:15:00")
TEST_START = pd.Timestamp("2026-06-01 00:15:00")


def _series_or_zeros(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(np.float32)
    return pd.Series(0.0, index=frame.index, dtype=np.float32)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    out = pd.to_numeric(numerator, errors="coerce") / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _diff_feature(series: pd.Series, periods: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").diff(periods).fillna(0.0).astype(np.float32)


def _rolling_mean_feature(series: pd.Series, window: int) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .rolling(window=window, min_periods=1)
        .mean()
        .fillna(0.0)
        .astype(np.float32)
    )


def _build_boundary_market_features(boundary: pd.DataFrame) -> pd.DataFrame:
    if boundary.empty:
        return pd.DataFrame(index=boundary.index)

    system_load = _series_or_zeros(boundary, "boundary_系统负荷")
    import_east = _series_or_zeros(boundary, "boundary_华东受电")
    south_wind = _series_or_zeros(boundary, "boundary_江南风电")
    north_wind = _series_or_zeros(boundary, "boundary_江北风电")
    south_pv = _series_or_zeros(boundary, "boundary_江南光伏")
    north_pv = _series_or_zeros(boundary, "boundary_江北光伏")
    south_gas = _series_or_zeros(boundary, "boundary_燃机江南")
    north_gas = _series_or_zeros(boundary, "boundary_燃机江北")
    south_storage = _series_or_zeros(boundary, "boundary_储能江南")
    north_storage = _series_or_zeros(boundary, "boundary_储能江北")
    south_coal = _series_or_zeros(boundary, "boundary_煤电江南")
    north_coal = _series_or_zeros(boundary, "boundary_煤电江北")

    wind_total = (south_wind + north_wind).astype(np.float32)
    pv_total = (south_pv + north_pv).astype(np.float32)
    renewable_total = (wind_total + pv_total).astype(np.float32)
    gas_total = (south_gas + north_gas).astype(np.float32)
    storage_total = (south_storage + north_storage).astype(np.float32)
    coal_total = (south_coal + north_coal).astype(np.float32)
    thermal_total = (gas_total + coal_total).astype(np.float32)

    net_load_after_renewable = (system_load - renewable_total).astype(np.float32)
    net_load_after_import_renewable = (system_load - import_east - renewable_total).astype(np.float32)
    fire_competitive_space = (
        system_load - import_east - renewable_total - gas_total - storage_total
    ).astype(np.float32)
    fire_competitive_space_clipped = fire_competitive_space.clip(lower=0.0).astype(np.float32)

    market = pd.DataFrame(index=boundary.index)
    market["market_system_load"] = system_load
    market["market_import_east"] = import_east
    market["market_wind_total"] = wind_total
    market["market_pv_total"] = pv_total
    market["market_renewable_total"] = renewable_total
    market["market_gas_total"] = gas_total
    market["market_storage_total"] = storage_total
    market["market_coal_total"] = coal_total
    market["market_thermal_total"] = thermal_total
    market["market_net_load_after_renewable"] = net_load_after_renewable
    market["market_net_load_after_import_renewable"] = net_load_after_import_renewable
    market["market_fire_competitive_space"] = fire_competitive_space
    market["market_fire_competitive_space_clipped"] = fire_competitive_space_clipped

    market["market_renewable_share"] = _safe_ratio(renewable_total, system_load)
    market["market_wind_share"] = _safe_ratio(wind_total, system_load)
    market["market_pv_share"] = _safe_ratio(pv_total, system_load)
    market["market_import_share"] = _safe_ratio(import_east, system_load)
    market["market_gas_share"] = _safe_ratio(gas_total, system_load)
    market["market_storage_share"] = _safe_ratio(storage_total, system_load)
    market["market_coal_share"] = _safe_ratio(coal_total, system_load)
    market["market_fire_space_share"] = _safe_ratio(fire_competitive_space_clipped, system_load)
    market["market_net_load_after_renewable_share"] = _safe_ratio(net_load_after_renewable, system_load)
    market["market_net_load_after_import_renewable_share"] = _safe_ratio(
        net_load_after_import_renewable,
        system_load,
    )

    market["market_wind_region_diff"] = (north_wind - south_wind).astype(np.float32)
    market["market_pv_region_diff"] = (north_pv - south_pv).astype(np.float32)
    market["market_gas_region_diff"] = (north_gas - south_gas).astype(np.float32)
    market["market_storage_region_diff"] = (north_storage - south_storage).astype(np.float32)
    market["market_coal_region_diff"] = (north_coal - south_coal).astype(np.float32)
    market["market_renewable_region_diff"] = (
        (north_wind + north_pv) - (south_wind + south_pv)
    ).astype(np.float32)

    for name, series in (
        ("market_system_load", system_load),
        ("market_renewable_total", renewable_total),
        ("market_import_east", import_east),
        ("market_fire_competitive_space", fire_competitive_space),
        ("market_fire_competitive_space_clipped", fire_competitive_space_clipped),
        ("market_net_load_after_import_renewable", net_load_after_import_renewable),
    ):
        market[f"{name}_diff_1"] = _diff_feature(series, 1)
        market[f"{name}_diff_4"] = _diff_feature(series, 4)
        market[f"{name}_mean_4"] = _rolling_mean_feature(series, 4)

    return market.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


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


def _frame_from_xlsx(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Jiangsu price workbook: {path}")
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in workbook: {path}")
    width = max(len(row) for row in rows)
    header = _safe_header(rows[0], width)
    frame = pd.DataFrame(rows[1:], columns=header)
    if "日期" not in frame.columns or "时间" not in frame.columns:
        raise ValueError(f"Expected columns `日期` and `时间` in workbook: {path}")
    frame["timestamp"] = [
        _parse_timestamp(row.get("日期", ""), row.get("时间", ""))
        for _, row in frame.iterrows()
    ]
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    frame = frame.drop(columns=[col for col in ("日期", "时间") if col in frame.columns])
    for column in frame.columns:
        if column == "timestamp":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame, {"rows": int(len(frame)), "duplicate_timestamps": duplicate_count}


def _rename_columns(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [f"{prefix}_{str(col).strip()}" for col in out.columns]
    return out


def _load_unit_tables() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[str, dict[str, int]]]:
    day_ahead: dict[int, pd.DataFrame] = {}
    realtime: dict[int, pd.DataFrame] = {}
    audit: dict[str, dict[str, int]] = {}

    for unit_no, path in DAY_AHEAD_FILES.items():
        frame, info = _frame_from_xlsx(path)
        audit[f"day_ahead_{unit_no}"] = {"rows": info["rows"], "duplicate_timestamps": info["duplicate_timestamps"]}
        day_ahead[unit_no] = _rename_columns(frame, prefix=f"u{unit_no}_da")

    for unit_no, path in REALTIME_FILES.items():
        frame, info = _frame_from_xlsx(path)
        audit[f"realtime_{unit_no}"] = {"rows": info["rows"], "duplicate_timestamps": info["duplicate_timestamps"]}
        realtime[unit_no] = _rename_columns(frame, prefix=f"u{unit_no}_rt")

    return day_ahead, realtime, audit


def _combine_frames(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    ordered = [frames[key] for key in sorted(frames)]
    combined = pd.concat(ordered, axis=1, join="inner").sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def _build_block_split_indices(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    if len(index) % 96 != 0:
        raise ValueError(f"Expected complete 15-minute days, got {len(index)} rows.")

    day_starts = index[::96]
    train_days = np.flatnonzero(day_starts < VAL_START)
    val_days = np.flatnonzero((day_starts >= VAL_START) & (day_starts < TEST_START))
    test_days = np.flatnonzero(day_starts >= TEST_START)

    def _expand(days: np.ndarray) -> np.ndarray:
        if len(days) == 0:
            return np.asarray([], dtype=np.int64)
        return np.concatenate([np.arange(day * 96, day * 96 + 96, dtype=np.int64) for day in days])

    return {
        "train": _expand(train_days),
        "val": _expand(val_days),
        "test": _expand(test_days),
    }


def _normalise_from_indices(values: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    source = values[train_idx] if len(train_idx) > 0 else values
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    denom = max(vmax - vmin, 1e-8)
    return ((values - vmin) / denom).astype(np.float32), vmin, vmax


def _make_node_meta(node_ids: list[str], units: list[int], node_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for node_id, unit_no in zip(node_ids, units, strict=True):
        rows.append(
            {
                "node_id": node_id,
                "unit_no": unit_no,
                "site_type": "price",
                "source_column": node_columns[node_id],
                "source_site": "jiangsu_horizontal",
                "market_scope": "day_ahead",
                "area": "jiangsu",
                "unit": "CNY/MWh",
                "price_type": "day_ahead_final",
            }
        )
    return pd.DataFrame(rows).set_index("node_id")


def load_jiangsu_price(
    processed_path: Path | None = None,
    force_reprocess: bool = False,
) -> dict:
    processed = Path(processed_path) if processed_path is not None else PROCESSED_PATH
    if processed.exists() and not force_reprocess:
        with processed.open("rb") as fh:
            return pickle.load(fh)

    day_ahead_tables, realtime_tables, audit = _load_unit_tables()
    day_ahead_combined = _combine_frames(day_ahead_tables)
    realtime_combined = _combine_frames(realtime_tables)

    # The timestamps are identical across all files after de-duplication, so the
    # day-ahead and realtime tables can be joined directly for audit purposes.
    combined_index = day_ahead_combined.index.intersection(realtime_combined.index)
    day_ahead_combined = day_ahead_combined.reindex(combined_index)
    realtime_combined = realtime_combined.reindex(combined_index)
    day_ahead_combined = day_ahead_combined.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()
    realtime_combined = realtime_combined.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()

    node_ids = [f"jiangsu_price_u{unit_no}" for unit_no in sorted(day_ahead_tables)]
    units = sorted(day_ahead_tables)

    # Primary forecasting target: day-ahead final price for each unit.
    day_ahead_target_cols = {node_id: f"u{unit_no}_da_节点最终" for node_id, unit_no in zip(node_ids, units, strict=True)}
    day_ahead_temp_cols = {node_id: f"u{unit_no}_da_节点临时" for node_id, unit_no in zip(node_ids, units, strict=True)}
    realtime_node_final_cols = {node_id: f"u{unit_no}_rt_节点最终" for node_id, unit_no in zip(node_ids, units, strict=True)}
    realtime_unified_final_cols = {node_id: f"u{unit_no}_rt_统一最终" for node_id, unit_no in zip(node_ids, units, strict=True)}

    day_ahead_target = pd.DataFrame(
        {node_id: pd.to_numeric(day_ahead_combined[col], errors="coerce") for node_id, col in day_ahead_target_cols.items()},
        index=combined_index,
    )
    day_ahead_temp = pd.DataFrame(
        {node_id: pd.to_numeric(day_ahead_combined[col], errors="coerce") for node_id, col in day_ahead_temp_cols.items()},
        index=combined_index,
    )
    realtime_node_final = pd.DataFrame(
        {node_id: pd.to_numeric(realtime_combined[col], errors="coerce") for node_id, col in realtime_node_final_cols.items()},
        index=combined_index,
    )
    realtime_unified_final = pd.DataFrame(
        {node_id: pd.to_numeric(realtime_combined[col], errors="coerce") for node_id, col in realtime_unified_final_cols.items()},
        index=combined_index,
    )

    # Historical price context. These columns are all observed values and are
    # safe to expose as history-end context features.
    price_history = pd.DataFrame(index=combined_index)
    price_history = price_history.join(day_ahead_target)
    price_history = price_history.join(day_ahead_target.add_suffix("_da_final"))
    price_history = price_history.join(day_ahead_temp.add_suffix("_da_temp"))
    price_history = price_history.join(realtime_node_final.add_suffix("_rt_node_final"))
    price_history = price_history.join(realtime_unified_final.add_suffix("_rt_unified_final"))

    occupancy_raw = day_ahead_target.to_numpy(dtype=np.float32)
    split_indices = _build_block_split_indices(combined_index)
    train_idx = split_indices["train"]
    occ_norm, vmin, vmax = _normalise_from_indices(occupancy_raw, train_idx)

    adj = np.ones((len(node_ids), len(node_ids)), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)

    node_meta = _make_node_meta(node_ids, units, {node_id: day_ahead_target_cols[node_id] for node_id in node_ids})

    day_ahead_audit = {
        "rows": int(len(day_ahead_combined)),
        "columns": int(day_ahead_combined.shape[1]),
        "start": str(day_ahead_combined.index.min()),
        "end": str(day_ahead_combined.index.max()),
    }
    realtime_audit = {
        "rows": int(len(realtime_combined)),
        "columns": int(realtime_combined.shape[1]),
        "start": str(realtime_combined.index.min()),
        "end": str(realtime_combined.index.max()),
    }

    load_context = load_jiangsu_load()
    load_weather = load_context.get("weather")
    if load_weather is not None and not getattr(load_weather, "empty", True):
        boundary_weather = load_weather.reindex(combined_index).interpolate(limit_direction="both").ffill().bfill()
        market_weather = _build_boundary_market_features(boundary_weather)
        weather = pd.concat([boundary_weather, market_weather], axis=1)
        weather = weather.loc[:, ~weather.columns.duplicated()].astype(np.float32)
    else:
        weather = pd.DataFrame(index=combined_index)

    result = {
        "occupancy": occ_norm,
        "occupancy_raw": occupancy_raw,
        "timestamps": pd.DatetimeIndex(combined_index),
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": adj,
        "weather": weather,
        "price": pd.DataFrame(index=combined_index),
        "price_history": price_history,
        "day_ahead": day_ahead_combined,
        "day_ahead_temp": day_ahead_temp,
        "realtime": realtime_combined,
        "day_ahead_target": day_ahead_target,
        "realtime_target": realtime_node_final,
        "realtime_unified_target": realtime_unified_final,
        "split_indices": split_indices,
        "split_target_start_times": {
            "val": VAL_START,
            "test": TEST_START,
        },
        "forecast_lead_steps": 61,
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_2026_q1_q2_before_validation",
        "price_audit": {
            "day_ahead": day_ahead_audit,
            "realtime": realtime_audit,
            "duplicates": audit,
        },
        "protocol": {
            "name": "jiangsu_spot_price_2026",
            "target": "day_ahead_node_final",
            "forecast_lead_steps": 61,
            "forecast_horizons_days": [1, 2, 3, 4, 5, 6],
            "train_start": TRAIN_START.isoformat(),
            "val_start": VAL_START.isoformat(),
            "test_start": TEST_START.isoformat(),
            "split_policy": "train=2026-01..2026-04, val=2026-05, test=2026-06-01..2026-06-23",
            "market_rule_note": "江苏现货市场：日前申报窗口在 D-1 09:00-11:00，日前出清在 D-1 17:00 左右；当前 loader 保留历史观测价格，并把 D-1 09:00 后可见的负荷边界进一步扩展成竞价空间、占比、区域差和短周期变化率特征。",
            "derived_boundary_feature_groups": [
                "fire_competitive_space",
                "renewable_share",
                "import_share",
                "gas_share",
                "storage_share",
                "coal_share",
                "regional_differences",
                "diff_1_diff_4_mean_4",
            ],
            "source_files": {
                "day_ahead": {str(unit_no): str(path) for unit_no, path in DAY_AHEAD_FILES.items()},
                "realtime": {str(unit_no): str(path) for unit_no, path in REALTIME_FILES.items()},
            },
        },
        "source_dataset": "Jiangsu spot price day-ahead/realtime 2026",
    }

    processed.parent.mkdir(parents=True, exist_ok=True)
    with processed.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


if __name__ == "__main__":
    data = load_jiangsu_price(force_reprocess=True)
    print(f"occupancy shape: {data['occupancy'].shape}")
    print(f"raw range: {data['norm_min']:.3f}..{data['norm_max']:.3f}")
    for split, idx in data["split_indices"].items():
        ts = data["timestamps"][idx]
        print(f"{split}: {len(idx)} rows, {ts[0]}..{ts[-1]}")
