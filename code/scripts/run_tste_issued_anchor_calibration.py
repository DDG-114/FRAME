#!/usr/bin/env python
"""Calibrate FRAME's residual contribution against an issued operational trajectory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.metrics import mase, point_metrics, probabilistic_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["aemo", "perform"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_calibration_dir", required=True)
    parser.add_argument("--output_test_dir", required=True)
    return parser.parse_args()


def weighted_median_coefficient(
    prediction: np.ndarray,
    issued: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return the [0, 1] coefficient that minimizes calibration MAE."""

    delta = (np.asarray(prediction) - np.asarray(issued)).reshape(-1)
    target_delta = (np.asarray(target) - np.asarray(issued)).reshape(-1)
    usable = np.abs(delta) > 1e-8
    if not np.any(usable):
        return 0.0
    ratios = target_delta[usable] / delta[usable]
    weights = np.abs(delta[usable])
    order = np.argsort(ratios)
    ratios = ratios[order]
    weights = weights[order]
    index = int(np.searchsorted(np.cumsum(weights), 0.5 * float(np.sum(weights))))
    return float(np.clip(ratios[min(index, len(ratios) - 1)], 0.0, 1.0))


def artifact_paths(root: Path) -> dict[str, Path]:
    paths = {path.parent.name: path for path in (root / "artifacts").glob("*/full.npz")}
    if not paths:
        raise FileNotFoundError(f"No full artifacts found below {root / 'artifacts'}")
    return paths


def base_task(artifact_name: str) -> str:
    return artifact_name.rsplit("_h", 1)[0] if "_h" in artifact_name else artifact_name


def load_issued_frame(protocol: str, data_root: Path, origins: set[str]) -> pd.DataFrame:
    if protocol == "perform":
        paths = [data_root / "perform_ercot_issued_forecasts.csv"]
        node_column = "node_id"
    else:
        paths = sorted(data_root.glob("*/aemo_unified_issued_forecasts.csv"))
        node_column = "REGIONID"
    if not paths or any(not path.exists() for path in paths):
        raise FileNotFoundError(f"Issued forecast files are missing below {data_root}")

    usecols = [
        "forecast_origin",
        "valid_time",
        node_column,
        "load_forecast",
        "wind_forecast",
        "pv_forecast",
        "net_load_forecast",
    ]
    frames: list[pd.DataFrame] = []
    origin_times = {pd.Timestamp(value) for value in origins}
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=["forecast_origin", "valid_time"],
        )
        frame = frame[frame["forecast_origin"].isin(origin_times)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No issued forecasts match the artifact forecast origins")
    issued = pd.concat(frames, ignore_index=True)
    issued = issued.rename(columns={node_column: "node_id"})
    issued["node_id"] = issued["node_id"].astype(str)
    return issued.drop_duplicates(["forecast_origin", "valid_time", "node_id"])


def issued_arrays(
    artifacts: dict[str, Path],
    issued: pd.DataFrame,
) -> dict[str, np.ndarray]:
    grouped = {
        (origin, str(node)): group.sort_values("valid_time")
        for (origin, node), group in issued.groupby(["forecast_origin", "node_id"], sort=False)
    }
    arrays: dict[str, np.ndarray] = {}
    for name, path in artifacts.items():
        archive = np.load(path, allow_pickle=False)
        horizon = int(archive["prediction_raw"].shape[1])
        column = f"{base_task(name)}_forecast"
        rows: list[np.ndarray] = []
        for origin, node in zip(archive["forecast_start"], archive["node_id"], strict=True):
            key = (pd.Timestamp(str(origin)), str(node))
            if key not in grouped:
                raise KeyError(f"Missing issued trajectory for {name}, origin={key[0]}, node={key[1]}")
            values = grouped[key][column].to_numpy(dtype=np.float64)
            if len(values) < horizon or not np.isfinite(values[:horizon]).all():
                raise ValueError(f"Incomplete issued trajectory for {name}, origin={key[0]}, node={key[1]}")
            rows.append(values[:horizon])
        arrays[name] = np.stack(rows).astype(np.float32)
    return arrays


def copy_and_adjust(
    source: Path,
    destination: Path,
    issued: np.ndarray,
    coefficient: float,
) -> dict[str, float]:
    archive = np.load(source, allow_pickle=False)
    payload = {key: archive[key] for key in archive.files}
    prediction = np.asarray(payload["prediction_raw"], dtype=np.float64)
    target = np.asarray(payload["target_raw"], dtype=np.float64)
    adjusted = issued + coefficient * (prediction - issued)
    task = base_task(source.parent.name)
    if task != "net_load":
        adjusted = np.maximum(adjusted, 0.0)
    shift = adjusted - prediction
    payload["prediction_raw"] = adjusted.astype(np.float32)
    payload["issued_prediction_raw"] = np.asarray(issued, dtype=np.float32)
    payload["issued_anchor_coefficient"] = np.asarray(coefficient, dtype=np.float32)
    if "quantiles_raw" in payload:
        quantiles = np.asarray(payload["quantiles_raw"], dtype=np.float64) + shift[..., None]
        if task != "net_load":
            quantiles = np.maximum(quantiles, 0.0)
        payload["quantiles_raw"] = np.maximum.accumulate(quantiles, axis=-1).astype(np.float32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    metrics = point_metrics(adjusted, target)
    seasonality = 24 if source.parent.name == task else 48
    metrics["mase"] = mase(
        adjusted,
        target,
        insample=np.asarray(payload["history_raw"]).reshape(-1),
        seasonality=seasonality,
    )
    if "quantiles_raw" in payload:
        metrics.update(
            probabilistic_metrics(
                payload["quantiles_raw"],
                target,
                levels=payload["quantile_levels"],
            )
        )
    return metrics


def main() -> None:
    args = parse_args()
    calibration_dir = Path(args.calibration_dir)
    test_dir = Path(args.test_dir)
    output_calibration_dir = Path(args.output_calibration_dir)
    output_test_dir = Path(args.output_test_dir)
    calibration_artifacts = artifact_paths(calibration_dir)
    test_artifacts = artifact_paths(test_dir)
    if calibration_artifacts.keys() != test_artifacts.keys():
        raise ValueError("Calibration and test artifact task sets differ")
    calibration_source_summary = json.loads(
        (calibration_dir / "summary.json").read_text(encoding="utf-8")
    )

    origin_strings: set[str] = set()
    for path in [*calibration_artifacts.values(), *test_artifacts.values()]:
        archive = np.load(path, allow_pickle=False)
        origin_strings.update(str(value).replace("T", " ") for value in archive["forecast_start"])
    issued_frame = load_issued_frame(args.protocol, Path(args.data_root), origin_strings)
    calibration_issued = issued_arrays(calibration_artifacts, issued_frame)
    test_issued = issued_arrays(test_artifacts, issued_frame)

    coefficients: dict[str, float] = {}
    calibration_metrics: dict[str, dict[str, float]] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    for name in sorted(calibration_artifacts):
        archive = np.load(calibration_artifacts[name], allow_pickle=False)
        coefficient = weighted_median_coefficient(
            archive["prediction_raw"],
            calibration_issued[name],
            archive["target_raw"],
        )
        coefficients[name] = coefficient
        calibration_metrics[name] = copy_and_adjust(
            calibration_artifacts[name],
            output_calibration_dir / "artifacts" / name / "full.npz",
            calibration_issued[name],
            coefficient,
        )
        test_metrics[name] = copy_and_adjust(
            test_artifacts[name],
            output_test_dir / "artifacts" / name / "full.npz",
            test_issued[name],
            coefficient,
        )

    summary = {
        "protocol": args.protocol,
        "fit_split": "calibration",
        "calibration_fraction": float(
            calibration_source_summary.get("calibration_fraction", 0.5)
        ),
        "objective": "global MAE",
        "coefficient_constraint": [0.0, 1.0],
        "coefficients": coefficients,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "calibration_source": str(calibration_dir),
        "test_source": str(test_dir),
    }
    for root, split, metrics in (
        (output_calibration_dir, "calibration", calibration_metrics),
        (output_test_dir, "test", test_metrics),
    ):
        root.mkdir(parents=True, exist_ok=True)
        split_summary = {**summary, "split": split, "metrics": metrics}
        (root / "summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
