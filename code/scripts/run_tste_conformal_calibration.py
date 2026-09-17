#!/usr/bin/env python
"""Calibrate FRAME quantile forecasts with chronological CQR and ACI."""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.metrics import probabilistic_metrics
from src.energyca_paper.training import LEAD_TIME_BINS_HOURS


ORIGINAL_CADENCE_MINUTES = {"load": 60, "wind": 10, "pv": 60, "net_load": 30}
COVERAGES = (0.50, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol", choices=["original", "aemo", "perform"], default="original")
    parser.add_argument(
        "--artifact_method",
        help=(
            "Optional probability-baseline method. When set, read calibration_artifacts/"
            "METHOD and artifacts/METHOD from the supplied run directory."
        ),
    )
    parser.add_argument("--aci_eta", type=float, default=0.01)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def endpoint_indices(levels: np.ndarray, coverage: float) -> tuple[int, int]:
    alpha = 1.0 - coverage
    lower = int(np.argmin(np.abs(levels - alpha / 2.0)))
    upper = int(np.argmin(np.abs(levels - (1.0 - alpha / 2.0))))
    if not np.isclose(levels[lower], alpha / 2.0) or not np.isclose(levels[upper], 1.0 - alpha / 2.0):
        raise ValueError(f"Quantile grid must contain endpoints for {coverage:.0%} coverage")
    return lower, upper


def bin_masks(horizon: int, cadence_minutes: int) -> list[tuple[str, np.ndarray]]:
    lead_hours = (np.arange(horizon, dtype=np.float64) + 1.0) * cadence_minutes / 60.0
    return [(label, (lead_hours > lower) & (lead_hours <= upper)) for lower, upper, label in LEAD_TIME_BINS_HOURS if np.any((lead_hours > lower) & (lead_hours <= upper))]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    index = min(len(ordered) - 1, int(np.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1)
    return float(max(0.0, ordered[index]))


def cqr_quantiles(
    calibration: np.ndarray,
    calibration_target: np.ndarray,
    test: np.ndarray,
    *,
    levels: np.ndarray,
    cadence_minutes: int,
    nonnegative: bool,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    corrected = test.copy()
    qhats: dict[str, dict[str, float]] = {}
    for coverage in COVERAGES:
        alpha = 1.0 - coverage
        lower_index, upper_index = endpoint_indices(levels, coverage)
        by_bin: dict[str, float] = {}
        for label, mask in bin_masks(test.shape[1], cadence_minutes):
            score = np.maximum(
                calibration[:, mask, lower_index] - calibration_target[:, mask],
                calibration_target[:, mask] - calibration[:, mask, upper_index],
            )
            qhat = conformal_quantile(score, alpha)
            corrected[:, mask, lower_index] = test[:, mask, lower_index] - qhat
            corrected[:, mask, upper_index] = test[:, mask, upper_index] + qhat
            by_bin[label] = qhat
        qhats[str(int(coverage * 100))] = by_bin
    if nonnegative:
        corrected = np.maximum(corrected, 0.0)
    return np.maximum.accumulate(corrected, axis=-1), qhats


def aci_quantiles(
    raw: np.ndarray,
    target: np.ndarray,
    *,
    levels: np.ndarray,
    cadence_minutes: int,
    cqr_qhats: dict[str, dict[str, float]],
    eta: float,
    scale: float,
    nonnegative: bool,
    forecast_start: np.ndarray,
) -> np.ndarray:
    """Apply ACI with corrections frozen at each forecast origin.

    Outcomes update the correction only after their valid time is no later than
    the next forecast issue time. All regions sharing an issue time therefore
    receive the same pre-issue correction and cannot update one another.
    """
    corrected = raw.copy()
    starts = pd.to_datetime(
        np.asarray(forecast_start).astype(str), utc=True, errors="raise"
    )
    if len(starts) != raw.shape[0]:
        raise ValueError("forecast_start rows do not match quantile rows")
    start_ns = starts.asi8
    unique_origins = np.unique(start_ns)
    cadence_ns = int(cadence_minutes * 60 * 1_000_000_000)
    lead_ns = (np.arange(raw.shape[1], dtype=np.int64) + 1) * cadence_ns
    for coverage in COVERAGES:
        alpha = 1.0 - coverage
        lower_index, upper_index = endpoint_indices(levels, coverage)
        state = {
            label: float(value)
            for label, value in cqr_qhats[str(int(coverage * 100))].items()
        }
        lead_bins = [
            (label, np.flatnonzero(mask))
            for label, mask in bin_masks(raw.shape[1], cadence_minutes)
        ]
        pending: dict[int, dict[str, list[float]]] = {}
        pending_times: list[int] = []
        for origin_ns in unique_origins:
            while pending_times and pending_times[0] <= int(origin_ns):
                valid_ns = heapq.heappop(pending_times)
                by_bin = pending.pop(valid_ns)
                for label, misses in by_bin.items():
                    miss_rate = float(np.mean(misses))
                    state[label] = max(
                        0.0,
                        state[label] + eta * scale * (miss_rate - alpha),
                    )

            rows = np.flatnonzero(start_ns == origin_ns)
            for label, leads in lead_bins:
                correction = state[label]
                lower = raw[rows[:, None], leads[None, :], lower_index] - correction
                upper = raw[rows[:, None], leads[None, :], upper_index] + correction
                corrected[rows[:, None], leads[None, :], lower_index] = np.minimum(
                    corrected[rows[:, None], leads[None, :], lower_index], lower
                )
                corrected[rows[:, None], leads[None, :], upper_index] = np.maximum(
                    corrected[rows[:, None], leads[None, :], upper_index], upper
                )
                observed = target[rows[:, None], leads[None, :]]
                missed = (observed < lower) | (observed > upper)
                for column, lead in enumerate(leads):
                    valid_ns = int(origin_ns + lead_ns[lead])
                    if valid_ns not in pending:
                        pending[valid_ns] = {}
                        heapq.heappush(pending_times, valid_ns)
                    pending[valid_ns].setdefault(label, []).extend(
                        missed[:, column].astype(float).tolist()
                    )
    if nonnegative:
        corrected = np.maximum(corrected, 0.0)
    return np.maximum.accumulate(corrected, axis=-1)


def metric_rows(task: str, method: str, quantiles: np.ndarray, target: np.ndarray, levels: np.ndarray, cadence_minutes: int) -> list[dict[str, Any]]:
    rows = [{"task": task, "method": method, "lead_bin": "all", **probabilistic_metrics(quantiles, target, levels=levels)}]
    for label, mask in bin_masks(target.shape[1], cadence_minutes):
        rows.append({"task": task, "method": method, "lead_bin": label, **probabilistic_metrics(quantiles[:, mask], target[:, mask], levels=levels)})
    return rows


def main() -> None:
    args = parse_args()
    if args.artifact_method:
        calibration_root = (
            Path(args.calibration_dir) / "calibration_artifacts" / args.artifact_method
        )
        test_root = Path(args.test_dir) / "artifacts" / args.artifact_method
    else:
        calibration_root = Path(args.calibration_dir) / "artifacts"
        test_root = Path(args.test_dir) / "artifacts"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(path.name for path in calibration_root.iterdir() if path.is_dir() and (test_root / path.name / "full.npz").exists())
    if not tasks:
        raise FileNotFoundError("No shared calibration/test quantile artifacts were found")
    rows: list[dict[str, Any]] = []
    parameters: dict[str, Any] = {}
    for task in tasks:
        calibration_data = np.load(calibration_root / task / "full.npz")
        test_data = np.load(test_root / task / "full.npz")
        required = {"quantiles_raw", "quantile_levels", "target_raw"}
        if not required.issubset(calibration_data.files) or not required.issubset(test_data.files):
            raise ValueError(f"Task {task} does not contain quantile forecast artifacts")
        levels = np.asarray(test_data["quantile_levels"], dtype=np.float64)
        calibration_quantiles = np.asarray(calibration_data["quantiles_raw"], dtype=np.float64)
        calibration_target = np.asarray(calibration_data["target_raw"], dtype=np.float64)
        test_quantiles = np.asarray(test_data["quantiles_raw"], dtype=np.float64)
        test_target = np.asarray(test_data["target_raw"], dtype=np.float64)
        if "forecast_start" not in test_data.files:
            raise ValueError(f"Task {task} does not contain forecast origins for ACI")
        test_forecast_start = np.asarray(test_data["forecast_start"])
        if not np.array_equal(levels, np.asarray(calibration_data["quantile_levels"], dtype=np.float64)):
            raise ValueError(f"Task {task} has inconsistent quantile grids")
        base_task = task.split("_h", 1)[0]
        if args.protocol == "aemo":
            cadence = 30
        elif args.protocol == "perform":
            cadence = 60
        else:
            cadence = ORIGINAL_CADENCE_MINUTES[base_task]
        nonnegative = base_task != "net_load"
        cqr, qhats = cqr_quantiles(
            calibration_quantiles,
            calibration_target,
            test_quantiles,
            levels=levels,
            cadence_minutes=cadence,
            nonnegative=nonnegative,
        )
        scale = max(float(np.mean(np.abs(calibration_target))), 1e-6)
        aci = aci_quantiles(
            test_quantiles,
            test_target,
            levels=levels,
            cadence_minutes=cadence,
            cqr_qhats=qhats,
            eta=args.aci_eta,
            scale=scale,
            nonnegative=nonnegative,
            forecast_start=test_forecast_start,
        )
        rows.extend(metric_rows(task, "raw", test_quantiles, test_target, levels, cadence))
        rows.extend(metric_rows(task, "cqr", cqr, test_target, levels, cadence))
        rows.extend(metric_rows(task, "aci", aci, test_target, levels, cadence))
        task_dir = output_dir / "artifacts" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_dir / "cqr.npz", quantiles_raw=cqr.astype(np.float32), target_raw=test_target.astype(np.float32), quantile_levels=levels.astype(np.float32))
        np.savez_compressed(task_dir / "aci.npz", quantiles_raw=aci.astype(np.float32), target_raw=test_target.astype(np.float32), quantile_levels=levels.astype(np.float32))
        parameters[task] = {"task_scale": scale, "cqr_qhats": qhats}
    write_csv(output_dir / "probability_metrics.csv", rows)
    (output_dir / "calibration_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    calibration_summary_path = Path(args.calibration_dir) / "summary.json"
    calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    summary = {
        "calibration_dir": str(Path(args.calibration_dir)),
        "test_dir": str(Path(args.test_dir)),
        "split": calibration_summary.get("split", "calibration"),
        "calibration_fraction": float(calibration_summary["calibration_fraction"]),
        "aci_eta": float(args.aci_eta),
        "protocol": args.protocol,
        "artifact_method": args.artifact_method,
        "aci_update_policy": "forecast_origin_grouped_delayed_outcomes",
        "aci_correction_frozen_within_forecast_origin": True,
        "aci_same_origin_cross_region_updates": False,
        "aci_outcomes_required_available_by_issue_time": True,
        "tasks": tasks,
        "metric_rows": len(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
