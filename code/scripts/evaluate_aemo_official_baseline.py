#!/usr/bin/env python
"""Match AEMO-issued forecasts to an exact FRAME test artifact."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.metrics import mase, point_metrics
from src.energyca_paper.training import lead_time_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_run_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_issued(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("**/aemo_unified_issued_forecasts.csv"))
    if not paths:
        raise FileNotFoundError(f"No AEMO issued-forecast tables found under {root}")
    frame = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame["REGIONID"] = frame["REGIONID"].astype(str)
    return frame.sort_values(["forecast_origin", "valid_time", "REGIONID"]).drop_duplicates(
        ["forecast_origin", "valid_time", "REGIONID"], keep="last"
    )


def match_prediction(
    issued: pd.DataFrame,
    *,
    task: str,
    origins: np.ndarray,
    nodes: np.ndarray,
    horizon: int,
    vintage_cache: dict[pd.Timestamp, pd.DataFrame],
) -> np.ndarray:
    value_column = f"{task}_forecast"
    if value_column not in issued:
        raise ValueError(f"AEMO table is missing {value_column}")
    predictions = np.empty((len(origins), horizon), dtype=np.float32)
    cadence = pd.Timedelta(minutes=30)
    for row, (origin_text, node) in enumerate(zip(origins.astype(str), nodes.astype(str))):
        origin = pd.Timestamp(str(origin_text))
        if origin not in vintage_cache:
            available = issued.loc[issued["forecast_origin"].le(origin)]
            if available.empty:
                raise ValueError(f"No AEMO forecast vintage is available by {origin}")
            latest = (
                available.sort_values("forecast_origin")
                .drop_duplicates(["valid_time", "REGIONID"], keep="last")
                .set_index(["valid_time", "REGIONID"])
            )
            vintage_cache[origin] = latest
        lookup = vintage_cache[origin][value_column]
        keys = [(origin + cadence * (lead + 1), str(node)) for lead in range(horizon)]
        values = lookup.reindex(pd.MultiIndex.from_tuples(keys, names=lookup.index.names))
        if values.isna().any():
            first = keys[int(np.flatnonzero(values.isna().to_numpy())[0])]
            raise ValueError(f"Missing AEMO official forecast for {task} at {first}")
        predictions[row] = values.to_numpy(np.float32)
    return predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    frame_root = Path(args.frame_run_dir) / "artifacts"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    issued = read_issued(Path(args.data_root))
    rows = []
    lead_rows = []
    sources = []
    vintage_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    for task_dir in sorted(path for path in frame_root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"(load|wind|pv|net_load)(?:_h\d+)?", task_dir.name)
        artifact_path = task_dir / "full.npz"
        if not match or not artifact_path.exists():
            continue
        task = match.group(1)
        artifact = np.load(artifact_path)
        target = np.asarray(artifact["target_raw"], dtype=np.float32)
        prediction = match_prediction(
            issued,
            task=task,
            origins=np.asarray(artifact["forecast_start"]),
            nodes=np.asarray(artifact["node_id"]),
            horizon=target.shape[1],
            vintage_cache=vintage_cache,
        )
        metrics = point_metrics(prediction, target)
        metrics["mase"] = mase(
            prediction,
            target,
            insample=np.asarray(artifact["history_raw"]).reshape(-1),
            seasonality=48,
        )
        metrics["examples"] = int(target.shape[0])
        rows.append({"method": "aemo_official", "task": task_dir.name, **metrics})
        baseline_artifact = {
            "prediction_raw": prediction,
            "target_raw": target,
            "history_raw": np.asarray(artifact["history_raw"], dtype=np.float32),
            "forecast_start": np.asarray(artifact["forecast_start"]),
            "node_id": np.asarray(artifact["node_id"]),
            "task": np.asarray(task),
            "method": np.asarray("aemo_official"),
        }
        task_output = output_dir / "artifacts" / task_dir.name
        task_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_output / "full.npz", **baseline_artifact)
        lead_rows.extend(
            {"method": "aemo_official", **row}
            for row in lead_time_metric_rows(
                baseline_artifact,
                task=task_dir.name,
                intervention="full",
                cadence_minutes=30,
            )
        )
        sources.append(str(artifact_path.resolve()))
    if not rows:
        raise FileNotFoundError(f"No FRAME test artifacts found under {frame_root}")
    write_csv(output_dir / "test_metrics.csv", rows)
    write_csv(output_dir / "lead_time_metrics.csv", lead_rows)
    summary = {
        "frame_run_dir": str(Path(args.frame_run_dir).resolve()),
        "aemo_data_root": str(Path(args.data_root).resolve()),
        "method": "aemo_official",
        "matched_tasks": len(rows),
        "frame_artifacts": sources,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
