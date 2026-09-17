#!/usr/bin/env python3
"""Audit causal release times and complete aligned origins in processed AEMO data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_aemo_unified_poc import latest_at_origin


EXPECTED_REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
FORECAST_COLUMNS = (
    "load_forecast",
    "wind_forecast",
    "pv_forecast",
    "net_load_forecast",
)
READ_COLUMNS = (
    "forecast_origin",
    "valid_time",
    "lead_hours",
    "REGIONID",
    *FORECAST_COLUMNS,
    "run_time",
    "version_time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--archive_audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def all_true(mapping: dict[str, Any]) -> bool:
    return bool(mapping) and all(value is True for value in mapping.values())


def latest_vintage_behavior() -> bool:
    origin = pd.Timestamp("2024-01-02 00:00:00")
    valid_time = origin + pd.Timedelta(minutes=30)
    frame = pd.DataFrame(
        {
            "REGIONID": ["NSW1", "NSW1", "NSW1"],
            "valid_time": [valid_time, valid_time, valid_time],
            "run_time": [
                origin - pd.Timedelta(hours=12),
                origin - pd.Timedelta(hours=1),
                origin + pd.Timedelta(hours=1),
            ],
            "value": [1.0, 2.0, 3.0],
        }
    )
    selected = latest_at_origin(frame, origin, "run_time")
    return bool(len(selected) == 1 and float(selected.iloc[0]["value"]) == 2.0)


def audit_issued_frame(
    frame: pd.DataFrame,
    *,
    expected_regions: Iterable[str] = EXPECTED_REGIONS,
    horizon_steps: int = 336,
) -> dict[str, Any]:
    expected_regions = tuple(expected_regions)
    missing_columns = sorted(set(READ_COLUMNS) - set(frame.columns))
    if missing_columns:
        return {
            "checks": {"required_columns": False},
            "missing_columns": missing_columns,
        }

    data = frame.loc[:, READ_COLUMNS].copy()
    for column in ("forecast_origin", "valid_time", "run_time", "version_time"):
        data[column] = pd.to_datetime(data[column], errors="coerce", format="mixed")
    for column in ("lead_hours", *FORECAST_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["REGIONID"] = data["REGIONID"].astype(str)

    origin = data["forecast_origin"]
    valid = data["valid_time"]
    run_time = data["run_time"]
    version_time = data["version_time"]
    computed_lead_hours = (valid - origin).dt.total_seconds() / 3600.0
    lead_step_float = computed_lead_hours * 2.0
    rounded_lead_step = np.rint(lead_step_float.to_numpy(dtype=np.float64))
    finite_lead = np.isfinite(rounded_lead_step)
    data["lead_step"] = np.where(finite_lead, rounded_lead_step, -1).astype(np.int64)

    duplicate_rows = int(
        data.duplicated(["forecast_origin", "REGIONID", "lead_step"]).sum()
    )
    coverage = data.loc[
        data["lead_step"].between(1, horizon_steps)
    ].groupby(["forecast_origin", "REGIONID"])["lead_step"].agg(
        count="count", unique="nunique", minimum="min", maximum="max"
    )
    complete_pairs = coverage.loc[
        coverage["count"].eq(horizon_steps)
        & coverage["unique"].eq(horizon_steps)
        & coverage["minimum"].eq(1)
        & coverage["maximum"].eq(horizon_steps)
    ]
    complete_origin_counts = (
        complete_pairs.reset_index()
        .groupby("forecast_origin")["REGIONID"]
        .nunique()
    )
    complete_origins = int(complete_origin_counts.eq(len(expected_regions)).sum())
    observed_regions = sorted(data["REGIONID"].dropna().unique().tolist())

    checks = {
        "required_columns": True,
        "datetime_fields_parse": bool(
            origin.notna().all() and valid.notna().all() and run_time.notna().all()
        ),
        "forecast_origins_are_daily_midnight": bool(
            origin.notna().all()
            and (origin.dt.hour.eq(0) & origin.dt.minute.eq(0) & origin.dt.second.eq(0)).all()
        ),
        "stpasa_run_available_by_origin": bool(
            run_time.notna().all() and run_time.le(origin).all()
        ),
        "rooftop_version_available_or_absent_by_origin": bool(
            (version_time.isna() | version_time.le(origin)).all()
        ),
        "targets_strictly_after_origin": bool(valid.gt(origin).all()),
        "declared_lead_matches_valid_time": bool(
            np.isfinite(data["lead_hours"].to_numpy(dtype=np.float64)).all()
            and np.allclose(
                data["lead_hours"].to_numpy(dtype=np.float64),
                computed_lead_hours.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
            )
            and np.allclose(
                lead_step_float.to_numpy(dtype=np.float64),
                rounded_lead_step,
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "lead_range_is_one_to_336_half_hours": bool(
            data["lead_step"].min() == 1 and data["lead_step"].max() == horizon_steps
        ),
        "forecast_values_finite": bool(
            np.isfinite(data.loc[:, FORECAST_COLUMNS].to_numpy(dtype=np.float64)).all()
        ),
        "five_expected_regions": set(observed_regions) == set(expected_regions),
        "unique_origin_region_lead_rows": duplicate_rows == 0,
        "at_least_1000_complete_seven_day_origins": complete_origins >= 1000,
        "latest_pre_origin_vintage_selector": latest_vintage_behavior(),
    }
    return {
        "checks": checks,
        "rows": int(len(data)),
        "origins": int(origin.nunique()),
        "complete_seven_day_origins": complete_origins,
        "regions": observed_regions,
        "origin_start": str(origin.min()),
        "origin_end": str(origin.max()),
        "lead_step_min": int(data["lead_step"].min()),
        "lead_step_max": int(data["lead_step"].max()),
        "duplicate_rows": duplicate_rows,
        "missing_rooftop_version_rows": int(version_time.isna().sum()),
        "missing_stpasa_run_rows": int(run_time.isna().sum()),
        "stpasa_runs_after_origin": int(run_time.gt(origin).sum()),
        "rooftop_versions_after_origin": int(version_time.gt(origin).sum()),
        "test_targets_or_predictions_used_for_selection": False,
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.data_root.glob("*/aemo_unified_issued_forecasts.csv"))
    if not files:
        raise FileNotFoundError(
            f"No processed AEMO issued-forecast files found under {args.data_root}"
        )
    frames = [pd.read_csv(path, usecols=list(READ_COLUMNS)) for path in files]
    result = audit_issued_frame(pd.concat(frames, ignore_index=True))
    archive = json.loads(args.archive_audit.read_text(encoding="utf-8"))
    checks = dict(result.get("checks", {}))
    checks.update(
        {
            "thirty_six_month_files": len(files) == 36,
            "archive_audit_complete": archive.get("status") == "complete",
            "archive_and_protocol_complete_origin_counts_match": (
                archive.get("complete_seven_day_origins")
                == result.get("complete_seven_day_origins")
            ),
            "archive_and_protocol_lead_ranges_match": (
                archive.get("lead_step_min") == result.get("lead_step_min")
                and archive.get("lead_step_max") == result.get("lead_step_max")
            ),
        }
    )
    report = {
        "status": "complete" if all_true(checks) else "failed",
        "scope": (
            "processed issued-input release times, forecast-origin alignment, "
            "lead indexing, regional coverage, and latest-vintage selection"
        ),
        "data_root": str(args.data_root.resolve()),
        "archive_audit": str(args.archive_audit.resolve()),
        "monthly_files": len(files),
        **{key: value for key, value in result.items() if key != "checks"},
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise SystemExit("AEMO issued-input protocol audit failed")


if __name__ == "__main__":
    main()
