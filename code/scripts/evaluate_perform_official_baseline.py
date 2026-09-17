#!/usr/bin/env python
"""Match PERFORM's official ERCOT forecasts to an exact FRAME test split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.metrics import mase, point_metrics, probabilistic_metrics
from src.energyca_paper.training import lead_time_metric_rows


LEVELS = np.arange(1, 100, dtype=np.float64) / 100.0
TASKS = ("load", "wind", "pv", "net_load")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_run_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_origin_rows(frame: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    selected = frame.loc[frame["forecast_origin"].eq(origin)].sort_values("valid_time")
    if len(selected) < horizon:
        raise ValueError(f"PERFORM origin {origin} has {len(selected)} rows, expected at least {horizon}")
    selected = selected.iloc[:horizon]
    if selected["valid_time"].duplicated().any():
        raise ValueError(f"PERFORM origin {origin} contains duplicate valid times")
    return selected


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    quantile_table = pd.read_csv(data_root / "perform_ercot_official_quantiles.csv", low_memory=False)
    issued_table = pd.read_csv(data_root / "perform_ercot_issued_forecasts.csv", low_memory=False)
    for frame in (quantile_table, issued_table):
        frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
        frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    quantile_columns = [f"q{index:02d}" for index in range(1, 100)]
    if not set(quantile_columns).issubset(quantile_table.columns):
        raise ValueError("PERFORM official quantile table must contain q01 through q99")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    matched_tasks: list[str] = []
    for task in TASKS:
        frame_artifact_path = Path(args.frame_run_dir) / "artifacts" / task / "full.npz"
        if not frame_artifact_path.exists():
            continue
        frame_artifact = np.load(frame_artifact_path)
        target = np.asarray(frame_artifact["target_raw"], dtype=np.float64)
        origins = pd.to_datetime(np.asarray(frame_artifact["forecast_start"]).astype(str))
        history = np.asarray(frame_artifact["history_raw"], dtype=np.float64)
        predictions = np.empty_like(target)
        quantiles = np.empty(target.shape + (99,), dtype=np.float64) if task != "net_load" else None
        for row_index, origin in enumerate(origins):
            if task == "net_load":
                selected = select_origin_rows(issued_table, origin, target.shape[1])
                predictions[row_index] = selected["net_load_forecast"].to_numpy(np.float64)
                reference_target = None
            else:
                selected = select_origin_rows(
                    quantile_table.loc[quantile_table["task"].eq(task)], origin, target.shape[1]
                )
                predictions[row_index] = selected["deterministic"].to_numpy(np.float64)
                quantiles[row_index] = selected[quantile_columns].to_numpy(np.float64)
                reference_target = selected["actual"].to_numpy(np.float64)
            if reference_target is not None and not np.allclose(reference_target, target[row_index], rtol=1e-5, atol=1e-4):
                raise ValueError(f"PERFORM target mismatch for {task} at {origin}")
        metrics = point_metrics(predictions, target)
        metrics["mase"] = mase(predictions, target, insample=history.reshape(-1), seasonality=24)
        metrics["examples"] = int(target.shape[0])
        if quantiles is not None:
            metrics.update(probabilistic_metrics(quantiles, target, levels=LEVELS))
        metrics_rows.append({"method": "perform_official", "task": task, **metrics})
        artifact = {
            "prediction_raw": predictions.astype(np.float32),
            "target_raw": target.astype(np.float32),
            "history_raw": history.astype(np.float32),
            "forecast_start": np.asarray(frame_artifact["forecast_start"]),
            "node_id": np.asarray(frame_artifact["node_id"]),
            "task": np.asarray(task),
            "method": np.asarray("perform_official"),
        }
        if quantiles is not None:
            artifact["quantiles_raw"] = quantiles.astype(np.float32)
            artifact["quantile_levels"] = LEVELS.astype(np.float32)
        task_dir = output_dir / "artifacts" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_dir / "full.npz", **artifact)
        lead_rows.extend(
            {"method": "perform_official", **row}
            for row in lead_time_metric_rows(
                artifact,
                task=task,
                intervention="full",
                cadence_minutes=60,
            )
        )
        matched_tasks.append(task)
    if not matched_tasks:
        raise FileNotFoundError(f"No FRAME task artifacts found under {args.frame_run_dir}")
    write_csv(output_dir / "test_metrics.csv", metrics_rows)
    write_csv(output_dir / "lead_time_metrics.csv", lead_rows)
    summary = {
        "frame_run_dir": str(Path(args.frame_run_dir).resolve()),
        "perform_data_root": str(data_root.resolve()),
        "method": "perform_official",
        "matched_tasks": matched_tasks,
        "marginal_probability_tasks": [task for task in matched_tasks if task != "net_load"],
        "net_load_probability_scope": "point forecast only; marginal component quantiles are not combined",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
