#!/usr/bin/env python
"""Aggregate the unified AEMO TSTE suite and run paired block tests."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TASK_PATTERN = re.compile(r"(load|wind|pv|net_load)_h(1|2|3|7)$")
TASKS = ("load", "wind", "pv", "net_load")
POINT_BASELINES = ("persistence", "seasonal_naive", "weekly_seasonal_naive", "dlinear")
ISSUED_POINT_BASELINES = ("issued_bias", "issued_residual_linear")
PROBABILITY_BASELINES = (
    "seasonal_empirical",
    "issued_empirical",
    "dlinear_quantile",
    "lstm_quantile",
    "patchtst_quantile",
)
ISSUED_PROBABILITY_BASELINES = (
    "dlinear_issued_quantile",
    "lstm_issued_quantile",
    "patchtst_issued_quantile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/aemo_formal_suite")
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--block_origins", type=int, default=7)
    parser.add_argument(
        "--sensitivity_bootstrap_samples",
        type=int,
        default=5000,
        help="Moving-block draws per block-length sensitivity condition.",
    )
    parser.add_argument(
        "--sensitivity_block_origins",
        nargs="+",
        type=int,
        default=[1, 7, 14],
        help="Forecast-origin block lengths reported beside the primary seven-origin test.",
    )
    parser.add_argument(
        "--include_evidence_upgrade",
        action="store_true",
        help="Require and aggregate the added issued-information controls and controlled ablations.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_parts(label: str) -> tuple[str, int]:
    match = TASK_PATTERN.fullmatch(label)
    if not match:
        raise ValueError(f"Unexpected AEMO task label {label!r}")
    return match.group(1), int(match.group(2))


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if not match:
        raise ValueError(f"Cannot identify seed in {path}")
    return int(match.group(1))


def mean_std_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...], metric_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        summary = dict(zip(group_fields, key))
        summary["seeds"] = len({item.get("seed") for item in items})
        for field in metric_fields:
            values = np.asarray([float(item[field]) for item in items if field in item and item[field] not in {"", None}], dtype=np.float64)
            if values.size:
                summary[f"{field}_mean"] = float(values.mean())
                summary[f"{field}_std"] = float(values.std(ddof=0))
        output.append(summary)
    return output


def collect_point_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "issued_anchor" / "test").glob("seed*/summary.json")):
        seed = seed_from_path(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        for task_label, values in summary["metrics"].items():
            task, horizon = task_parts(task_label)
            rows.append({"family": "FRAME", "method": "FRAME-Issued-Anchor", "track": "issued", "seed": seed, "task": task, "horizon_days": horizon, **{key: float(values[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "frame_qwen").glob("*seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        for row in read_csv(path):
            if row["intervention"] != "full":
                continue
            task, horizon = task_parts(row["task"])
            rows.append({"family": "FRAME-ablation", "method": "FRAME-Raw-Residual", "track": "issued", "seed": seed, "task": task, "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "frame").glob("*seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        match = re.search(r"formal_numeric_(common|issued|oracle)_", path.parent.name)
        if not match:
            continue
        track = match.group(1)
        for row in read_csv(path):
            if row["intervention"] != "full":
                continue
            task, horizon = task_parts(row["task"])
            rows.append({"family": "FRAME-ablation", "method": f"FRAME-Numeric-{track.title()}", "track": track, "seed": seed, "task": task, "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "numeric_issued_anchor" / "test").glob("seed*/summary.json")):
        seed = seed_from_path(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        for task_label, values in summary["metrics"].items():
            task, horizon = task_parts(task_label)
            rows.append({"family": "FRAME-ablation", "method": "FRAME-Numeric-Issued-Anchor", "track": "issued", "seed": seed, "task": task, "horizon_days": horizon, **{key: float(values[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    ablation_names = {
        "numeric-schema": "FRAME-Numeric-Schema",
        "no-adapters": "FRAME-No-Adapters",
        "no-rank-gate": "FRAME-No-Rank-Gate",
        "no-strength-gate": "FRAME-No-Strength-Gate",
        "no-issued-target-anchor": "FRAME-No-Issued-Target-Anchor",
    }
    for path in sorted((root / "controlled_ablations").glob("*variant-*seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        variant_match = re.search(r"variant-([a-z-]+)_seed", path.parent.name)
        if not variant_match or variant_match.group(1) not in ablation_names:
            continue
        for row in read_csv(path):
            if row["intervention"] != "full":
                continue
            task, horizon = task_parts(row["task"])
            rows.append({"family": "controlled ablation", "method": ablation_names[variant_match.group(1)], "track": "issued", "seed": seed, "task": task, "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "aemo_official").glob("seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        for row in read_csv(path):
            task, horizon = task_parts(row["task"])
            rows.append({"family": "business baseline", "method": "AEMO-Official", "track": "issued", "seed": seed, "task": task, "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "point_baselines").glob("formal_aemo_*seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)d_", path.parent.name)
        if not horizon_match:
            continue
        horizon = int(horizon_match.group(1))
        for row in read_csv(path):
            rows.append({"family": "history-only baseline", "method": row["method"], "track": "common", "seed": seed, "task": row["task"], "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "issued_point_baselines").glob("formal_aemo_*_issued_seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)d_", path.parent.name)
        if not horizon_match:
            continue
        horizon = int(horizon_match.group(1))
        for row in read_csv(path):
            rows.append({"family": "issued residual baseline", "method": row["method"], "track": "issued", "seed": seed, "task": row["task"], "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    return rows


def collect_probability_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "issued_anchor" / "conformal").glob("seed*/probability_metrics.csv")):
        seed = seed_from_path(path)
        for row in read_csv(path):
            if row["lead_bin"] != "all":
                continue
            task, horizon = task_parts(row["task"])
            values = {key: float(row[key]) for key in ("pinball", "crps", "ncrps", "wis", "picp_90", "ace_90", "pinaw_90", "quantile_crossing_rate")}
            rows.append({"source": "FRAME-Issued-Anchor", "seed": seed, "method": row["method"], "task": task, "horizon_days": horizon, **values})
    for path in sorted((root / "probability_conformal").glob("seed*/probability_metrics.csv")):
        seed = seed_from_path(path)
        for row in read_csv(path):
            if row["lead_bin"] != "all":
                continue
            task, horizon = task_parts(row["task"])
            values = {key: float(row[key]) for key in ("pinball", "crps", "ncrps", "wis", "picp_90", "ace_90", "pinaw_90", "quantile_crossing_rate")}
            rows.append({"source": "FRAME-Raw-Residual", "seed": seed, "method": row["method"], "task": task, "horizon_days": horizon, **values})
    for path in sorted((root / "probability_baselines").glob("formal_aemo_*seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)d_", path.parent.name)
        if not horizon_match:
            continue
        horizon = int(horizon_match.group(1))
        for row in read_csv(path):
            values = {key: float(row[key]) for key in ("pinball", "crps", "ncrps", "wis", "picp_90", "ace_90", "pinaw_90", "quantile_crossing_rate")}
            rows.append({"source": "baseline", "seed": seed, "method": row["method"], "task": row["task"], "horizon_days": horizon, **values})
    for path in sorted((root / "issued_probability_baselines").glob("formal_aemo_*_issued_seed*/test_metrics.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)d_", path.parent.name)
        if not horizon_match:
            continue
        horizon = int(horizon_match.group(1))
        for row in read_csv(path):
            values = {key: float(row[key]) for key in ("pinball", "crps", "ncrps", "wis", "picp_90", "ace_90", "pinaw_90", "quantile_crossing_rate")}
            rows.append({"source": "issued residual baseline", "seed": seed, "method": row["method"], "task": row["task"], "horizon_days": horizon, **values})
    return rows


def collect_holdout_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "region_holdout_qwen").glob("*holdout-*seed*/test_metrics.csv")):
        region_match = re.search(r"holdout-(NSW1|QLD1|VIC1|SA1|TAS1)", path.parent.name)
        if not region_match:
            continue
        seed = seed_from_path(path)
        for row in read_csv(path):
            if row["intervention"] != "full":
                continue
            task, horizon = task_parts(row["task"])
            rows.append({"method": "FRAME-Issued", "region": region_match.group(1), "seed": seed, "task": task, "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    for path in sorted((root / "region_holdout_issued_baselines").glob("*holdout-*seed*/test_metrics.csv")):
        region_match = re.search(r"holdout-(NSW1|QLD1|VIC1|SA1|TAS1)", path.parent.name)
        horizon_match = re.search(r"_H(1|2|3|7)d_", path.parent.name)
        if not region_match or not horizon_match:
            continue
        seed = seed_from_path(path)
        horizon = int(horizon_match.group(1))
        for row in read_csv(path):
            rows.append({"method": row["method"], "region": region_match.group(1), "seed": seed, "task": row["task"], "horizon_days": horizon, **{key: float(row[key]) for key in ("mae", "rmse", "nmae", "smape", "mase")}})
    return rows


def collect_reserve_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "issued_anchor" / "reserve_value").glob("seed*_H*.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)\.csv$", path.name)
        if not horizon_match:
            continue
        for row in read_csv(path):
            values = {
                key: float(value)
                for key, value in row.items()
                if key not in {"method", "cost_ratio"}
            }
            source = "issued baseline" if row["method"].startswith("issued_") else "FRAME-Issued-Anchor"
            rows.append({
                "source": source,
                "seed": seed,
                "horizon_days": int(horizon_match.group(1)),
                "method": row["method"],
                "cost_ratio": float(row.get("cost_ratio") or 10.0),
                **values,
            })
    for path in sorted((root / "issued_anchor" / "reserve_frontier").glob("seed*_H*.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)\.csv$", path.name)
        if not horizon_match:
            continue
        for row in read_csv(path):
            values = {
                key: float(value)
                for key, value in row.items()
                if key not in {"method", "cost_ratio"}
            }
            source = "issued baseline" if row["method"].startswith("issued_") else "FRAME-Issued-Anchor"
            rows.append({
                "source": source,
                "seed": seed,
                "horizon_days": int(horizon_match.group(1)),
                "method": row["method"],
                "cost_ratio": float(row.get("cost_ratio") or 10.0),
                **values,
            })
    for path in sorted((root / "reserve_value").glob("seed*_H*.csv")):
        seed = seed_from_path(path)
        horizon_match = re.search(r"_H(1|2|3|7)\.csv$", path.name)
        if not horizon_match:
            continue
        for row in read_csv(path):
            values = {
                key: float(value)
                for key, value in row.items()
                if key not in {"method", "cost_ratio"}
            }
            rows.append({
                "source": "FRAME-Raw-Residual",
                "seed": seed,
                "horizon_days": int(horizon_match.group(1)),
                "method": row["method"],
                "cost_ratio": float(row.get("cost_ratio") or 10.0),
                **values,
            })
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    identity = ("source", "seed", "horizon_days", "method", "cost_ratio")
    for row in rows:
        key = tuple(row[field] for field in identity)
        if key in unique and unique[key] != row:
            raise ValueError(f"Conflicting reserve rows for {key}")
        unique[key] = row
    return list(unique.values())


def artifact_error_by_origin(path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path)
    prediction = np.asarray(payload["prediction_raw"], dtype=np.float64)
    target = np.asarray(payload["target_raw"], dtype=np.float64)
    origins = np.asarray(payload["forecast_start"]).astype(str)
    nodes = np.asarray(payload["node_id"]).astype(str)
    if prediction.shape != target.shape or prediction.shape[0] != origins.size or origins.size != nodes.size:
        raise ValueError(f"Inconsistent artifact {path}")
    scale = max(float(np.mean(np.abs(target))), 1e-8)
    sample_error = np.mean(np.abs(prediction - target), axis=1) / scale
    grouped: dict[str, list[float]] = defaultdict(list)
    for origin, _node, error in zip(origins, nodes, sample_error):
        grouped[origin].append(float(error))
    keys = sorted(grouped)
    return keys, np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)


def crps_by_sample(quantiles: np.ndarray, target: np.ndarray, levels: np.ndarray) -> np.ndarray:
    residual = target[..., None] - quantiles
    pinball = np.maximum(levels * residual, (levels - 1.0) * residual)
    score_by_level = np.mean(pinball, axis=1)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return 2.0 * integrate(score_by_level, levels, axis=1) / max(float(levels[-1] - levels[0]), 1e-8)


def probability_error_by_origin(path: Path, origins_path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path)
    origin_payload = np.load(origins_path)
    quantiles = np.asarray(payload["quantiles_raw"], dtype=np.float64)
    target = np.asarray(payload["target_raw"], dtype=np.float64)
    levels = np.asarray(payload["quantile_levels"], dtype=np.float64)
    origins = np.asarray(origin_payload["forecast_start"]).astype(str)
    nodes = np.asarray(origin_payload["node_id"]).astype(str)
    if quantiles.shape[0] != origins.size or origins.size != nodes.size:
        raise ValueError(f"Origin array does not match quantile artifact {path}")
    values = crps_by_sample(quantiles, target, levels)
    grouped: dict[str, list[float]] = defaultdict(list)
    for origin, _node, value in zip(origins, nodes, values):
        grouped[origin].append(float(value))
    keys = sorted(grouped)
    return keys, np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)


def circular_bootstrap(values: np.ndarray, samples: int, block_size: int, seed: int) -> np.ndarray:
    count = values.size
    block = min(max(1, block_size), count)
    blocks_per_draw = int(np.ceil(count / block))
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        starts = rng.integers(0, count, size=blocks_per_draw)
        indices = np.concatenate([(start + np.arange(block)) % count for start in starts])[:count]
        output[draw] = float(values[indices].mean())
    return output


def apply_holm(rows: list[dict[str, Any]], *, family_field: str | None = None) -> None:
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        families[str(row.get(family_field, "all"))].append(index)
    for family, indices in families.items():
        order = sorted(indices, key=lambda index: rows[index]["p_value_two_sided"])
        running = 0.0
        for rank, index in enumerate(order):
            running = max(
                running,
                min(1.0, rows[index]["p_value_two_sided"] * (len(order) - rank)),
            )
            rows[index]["p_value_holm"] = running
            rows[index]["comparison_family"] = family


def difference_summary(values: np.ndarray, samples: int, block_size: int, seed: int) -> dict[str, float]:
    bootstrap = circular_bootstrap(values, samples, block_size, seed)
    p_low = (np.count_nonzero(bootstrap <= 0.0) + 1.0) / (samples + 1.0)
    p_high = (np.count_nonzero(bootstrap >= 0.0) + 1.0) / (samples + 1.0)
    return {
        "gain": float(values.mean()),
        "gain_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "gain_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "p_value_two_sided": float(min(1.0, 2.0 * min(p_low, p_high))),
    }


def require_aligned_point_artifacts(
    left: Path,
    right: Path,
    left_origins: Path | None = None,
    right_origins: Path | None = None,
) -> None:
    with np.load(left, allow_pickle=True) as left_payload, np.load(right, allow_pickle=True) as right_payload:
        if not np.array_equal(left_payload["target_raw"], right_payload["target_raw"]):
            raise ValueError(f"target_raw mismatch: {left} vs {right}")
    with np.load(left_origins or left, allow_pickle=True) as left_payload, np.load(
        right_origins or right,
        allow_pickle=True,
    ) as right_payload:
        for key in ("forecast_start", "node_id"):
            if not np.array_equal(left_payload[key], right_payload[key]):
                raise ValueError(f"{key} mismatch: {left} vs {right}")


def information_comparisons(root: Path, samples: int, block_size: int) -> list[dict[str, Any]]:
    comparisons = (
        (
            "operational_information_value",
            "common",
            "issued",
            "historical and scheduled inputs",
            "operational forecasts available at the prediction time",
        ),
        (
            "realized_covariate_ceiling",
            "issued",
            "oracle",
            "operational forecasts available at the prediction time",
            "realized-covariate diagnostic",
        ),
    )
    rows: list[dict[str, Any]] = []
    for family, reference_track, evaluated_track, reference_label, evaluated_label in comparisons:
        for horizon in (1, 2, 3, 7):
            for task in TASKS:
                seed_differences: list[np.ndarray] = []
                reference_values: list[np.ndarray] = []
                evaluated_values: list[np.ndarray] = []
                origins_reference: list[str] | None = None
                for seed in (42, 2027, 3407):
                    run_prefix = "formal_numeric_{}_load-wind-pv-net_load_H1-2-3-7_L7d_seen-regions_seed{}"
                    reference_path = (
                        root / "frame" / run_prefix.format(reference_track, seed)
                        / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    )
                    evaluated_path = (
                        root / "frame" / run_prefix.format(evaluated_track, seed)
                        / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    )
                    require_aligned_point_artifacts(reference_path, evaluated_path)
                    reference_origins, reference_error = artifact_error_by_origin(reference_path)
                    evaluated_origins, evaluated_error = artifact_error_by_origin(evaluated_path)
                    if reference_origins != evaluated_origins:
                        raise ValueError(f"Origin mismatch: {reference_path} vs {evaluated_path}")
                    if origins_reference is None:
                        origins_reference = reference_origins
                    elif origins_reference != reference_origins:
                        raise ValueError(f"Seed-level origins differ for {family}, {task}, H{horizon}")
                    seed_differences.append(reference_error - evaluated_error)
                    reference_values.append(reference_error)
                    evaluated_values.append(evaluated_error)
                difference = np.mean(np.stack(seed_differences), axis=0)
                reference_error = np.mean(np.stack(reference_values), axis=0)
                evaluated_error = np.mean(np.stack(evaluated_values), axis=0)
                statistics = difference_summary(difference, samples, block_size, 10_500 + len(rows))
                rows.append(
                    {
                        "comparison_family": family,
                        "task": task,
                        "horizon_days": horizon,
                        "reference_input": reference_label,
                        "evaluated_input": evaluated_label,
                        "seeds": 3,
                        "origins": len(origins_reference or []),
                        "block_origins": min(block_size, len(origins_reference or [])),
                        "reference_nmae": float(reference_error.mean()),
                        "evaluated_nmae": float(evaluated_error.mean()),
                        "nmae_gain": float(statistics["gain"]),
                        "nmae_gain_percent": float(
                            100.0 * statistics["gain"] / max(float(reference_error.mean()), 1e-12)
                        ),
                        **statistics,
                    }
                )
    apply_holm(rows, family_field="comparison_family")
    return rows


def paired_comparisons(
    root: Path,
    samples: int,
    block_size: int,
    include_evidence_upgrade: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = (42, 2027, 3407)
    for horizon in (1, 2, 3, 7):
        for task in ("load", "wind", "pv", "net_load"):
            point_baselines = POINT_BASELINES
            if include_evidence_upgrade:
                point_baselines = (*point_baselines, *ISSUED_POINT_BASELINES)
            for baseline in (*point_baselines, "aemo_official"):
                seed_differences: list[np.ndarray] = []
                frame_values: list[np.ndarray] = []
                baseline_values: list[np.ndarray] = []
                origins_reference: list[str] | None = None
                for seed in seeds:
                    frame_path = root / "issued_anchor" / "test" / f"seed{seed}" / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    if baseline == "aemo_official":
                        baseline_path = root / "aemo_official" / f"seed{seed}" / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    elif baseline in ISSUED_POINT_BASELINES:
                        baseline_path = (
                            root
                            / "issued_point_baselines"
                            / f"formal_aemo_L7d_H{horizon}d_S1d_issued_seed{seed}"
                            / "artifacts"
                            / baseline
                            / task
                            / "full.npz"
                        )
                    else:
                        baseline_path = root / "point_baselines" / f"formal_aemo_L7d_H{horizon}d_S1d_seed{seed}" / "artifacts" / baseline / task / "full.npz"
                    require_aligned_point_artifacts(frame_path, baseline_path)
                    frame_origins, frame_error = artifact_error_by_origin(frame_path)
                    baseline_origins, baseline_error = artifact_error_by_origin(baseline_path)
                    if frame_origins != baseline_origins:
                        raise ValueError(f"Origin mismatch: {frame_path} vs {baseline_path}")
                    if origins_reference is None:
                        origins_reference = frame_origins
                    elif origins_reference != frame_origins:
                        raise ValueError(f"Seed-level origins differ for {task}, H{horizon}")
                    seed_differences.append(baseline_error - frame_error)
                    frame_values.append(frame_error)
                    baseline_values.append(baseline_error)
                difference = np.mean(np.stack(seed_differences), axis=0)
                frame_error = np.mean(np.stack(frame_values), axis=0)
                baseline_error = np.mean(np.stack(baseline_values), axis=0)
                bootstrap = circular_bootstrap(difference, samples, block_size, 2700 + horizon * 100 + len(rows))
                p_low = (np.count_nonzero(bootstrap <= 0.0) + 1.0) / (samples + 1.0)
                p_high = (np.count_nonzero(bootstrap >= 0.0) + 1.0) / (samples + 1.0)
                gain = float(difference.mean())
                rows.append({
                    "task": task,
                    "horizon_days": horizon,
                    "baseline": baseline,
                    "seeds": len(seeds),
                    "origins": len(origins_reference or []),
                    "block_origins": min(block_size, len(origins_reference or [])),
                    "frame_nmae": float(frame_error.mean()),
                    "baseline_nmae": float(baseline_error.mean()),
                    "nmae_gain": gain,
                    "nmae_gain_percent": float(100.0 * gain / max(float(baseline_error.mean()), 1e-12)),
                    "gain_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "gain_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "p_value_two_sided": float(min(1.0, 2.0 * min(p_low, p_high))),
                })
    apply_holm(rows)
    return rows


def controlled_ablation_comparisons(root: Path, samples: int, block_size: int) -> list[dict[str, Any]]:
    """Test each controlled design removal against the matched full FRAME run."""

    rows: list[dict[str, Any]] = []
    seeds = (42, 2027, 3407)
    variants = {
        "numeric_schema": "numeric-schema",
        "no_adapters": "no-adapters",
        "no_rank_gate": "no-rank-gate",
        "no_strength_gate": "no-strength-gate",
        "no_issued_target_anchor": "no-issued-target-anchor",
    }
    for label, variant in variants.items():
        for horizon in (1, 2, 3, 7):
            for task in ("load", "wind", "pv", "net_load"):
                differences: list[np.ndarray] = []
                full_values: list[np.ndarray] = []
                ablated_values: list[np.ndarray] = []
                origins_reference: list[str] | None = None
                for seed in seeds:
                    full_path = root / "frame_qwen" / (
                        "formal_qwen_issued_load-wind-pv-net_load_"
                        f"H1-2-3-7_L7d_seen-regions_seed{seed}"
                    ) / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    ablated_path = root / "controlled_ablations" / (
                        "formal_qwen_issued_load-wind-pv-net_load_"
                        f"H1-2-3-7_L7d_seen-regions_variant-{variant}_seed{seed}"
                    ) / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    require_aligned_point_artifacts(full_path, ablated_path)
                    full_origins, full_error = artifact_error_by_origin(full_path)
                    ablated_origins, ablated_error = artifact_error_by_origin(ablated_path)
                    if full_origins != ablated_origins:
                        raise ValueError(f"Ablation origin mismatch: {full_path} vs {ablated_path}")
                    if origins_reference is None:
                        origins_reference = full_origins
                    elif origins_reference != full_origins:
                        raise ValueError(f"Ablation seed origins differ for {task}, H{horizon}")
                    differences.append(ablated_error - full_error)
                    full_values.append(full_error)
                    ablated_values.append(ablated_error)
                difference = np.mean(np.stack(differences), axis=0)
                statistics = difference_summary(difference, samples, block_size, 9100 + len(rows))
                full_error = np.mean(np.stack(full_values), axis=0)
                ablated_error = np.mean(np.stack(ablated_values), axis=0)
                rows.append({
                    "ablation": label,
                    "task": task,
                    "horizon_days": horizon,
                    "seeds": len(seeds),
                    "origins": len(origins_reference or []),
                    "block_origins": min(block_size, len(origins_reference or [])),
                    "full_nmae": float(full_error.mean()),
                    "ablated_nmae": float(ablated_error.mean()),
                    "nmae_gain": float(statistics["gain"]),
                    "nmae_gain_percent": float(
                        100.0 * statistics["gain"] / max(float(ablated_error.mean()), 1e-12)
                    ),
                    **statistics,
                })

    for horizon in (1, 2, 3, 7):
        for task in ("load", "wind", "pv", "net_load"):
            differences = []
            full_values = []
            ablated_values = []
            origins_reference = None
            for seed in seeds:
                full_path = root / "issued_anchor" / "test" / f"seed{seed}" / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                ablated_path = root / "numeric_issued_anchor" / "test" / f"seed{seed}" / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                require_aligned_point_artifacts(full_path, ablated_path)
                full_origins, full_error = artifact_error_by_origin(full_path)
                ablated_origins, ablated_error = artifact_error_by_origin(ablated_path)
                if full_origins != ablated_origins:
                    raise ValueError(f"Numeric-anchor origin mismatch: {full_path} vs {ablated_path}")
                if origins_reference is None:
                    origins_reference = full_origins
                elif origins_reference != full_origins:
                    raise ValueError(f"Numeric-anchor seed origins differ for {task}, H{horizon}")
                differences.append(ablated_error - full_error)
                full_values.append(full_error)
                ablated_values.append(ablated_error)
            difference = np.mean(np.stack(differences), axis=0)
            statistics = difference_summary(difference, samples, block_size, 9700 + len(rows))
            full_error = np.mean(np.stack(full_values), axis=0)
            ablated_error = np.mean(np.stack(ablated_values), axis=0)
            rows.append({
                "ablation": "numeric_issued_anchor",
                "task": task,
                "horizon_days": horizon,
                "seeds": len(seeds),
                "origins": len(origins_reference or []),
                "block_origins": min(block_size, len(origins_reference or [])),
                "full_nmae": float(full_error.mean()),
                "ablated_nmae": float(ablated_error.mean()),
                "nmae_gain": float(statistics["gain"]),
                "nmae_gain_percent": float(
                    100.0 * statistics["gain"] / max(float(ablated_error.mean()), 1e-12)
                ),
                **statistics,
            })
    apply_holm(rows)
    return rows


def holdout_comparisons(root: Path, samples: int, block_size: int) -> list[dict[str, Any]]:
    """Compare held-region FRAME predictions with issued-information controls."""

    rows: list[dict[str, Any]] = []
    for region in ("NSW1", "QLD1", "VIC1", "SA1", "TAS1"):
        for horizon in (1, 2, 3, 7):
            for task in TASKS:
                for baseline in ISSUED_POINT_BASELINES:
                    differences: list[np.ndarray] = []
                    frame_values: list[np.ndarray] = []
                    baseline_values: list[np.ndarray] = []
                    origins_reference: list[str] | None = None
                    for seed in (42, 2027, 3407):
                        frame_path = root / "region_holdout_qwen" / (
                            "formal_qwen_issued_load-wind-pv-net_load_"
                            f"H1-2-3-7_L7d_holdout-{region}_seed{seed}"
                        ) / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                        baseline_path = root / "region_holdout_issued_baselines" / (
                            f"formal_aemo_L7d_H{horizon}d_S1d_issued_holdout-{region}_seed{seed}"
                        ) / "artifacts" / baseline / task / "full.npz"
                        require_aligned_point_artifacts(frame_path, baseline_path)
                        frame_origins, frame_error = artifact_error_by_origin(frame_path)
                        baseline_origins, baseline_error = artifact_error_by_origin(baseline_path)
                        if frame_origins != baseline_origins:
                            raise ValueError(
                                f"Holdout origin mismatch: {frame_path} vs {baseline_path}"
                            )
                        if origins_reference is None:
                            origins_reference = frame_origins
                        elif origins_reference != frame_origins:
                            raise ValueError(
                                f"Holdout origins differ across seeds for {region}, {task}, H{horizon}"
                            )
                        differences.append(baseline_error - frame_error)
                        frame_values.append(frame_error)
                        baseline_values.append(baseline_error)
                    difference = np.mean(np.stack(differences), axis=0)
                    statistics = difference_summary(
                        difference,
                        samples,
                        block_size,
                        10100 + len(rows),
                    )
                    frame_error = np.mean(np.stack(frame_values), axis=0)
                    baseline_error = np.mean(np.stack(baseline_values), axis=0)
                    rows.append(
                        {
                            "comparison_family": "leave_one_region_out",
                            "region": region,
                            "task": task,
                            "horizon_days": horizon,
                            "baseline": baseline,
                            "seeds": 3,
                            "origins": len(origins_reference or []),
                            "block_origins": min(block_size, len(origins_reference or [])),
                            "frame_nmae": float(frame_error.mean()),
                            "baseline_nmae": float(baseline_error.mean()),
                            "nmae_gain": float(statistics["gain"]),
                            "nmae_gain_percent": float(
                                100.0 * statistics["gain"] / max(float(baseline_error.mean()), 1e-12)
                            ),
                            **statistics,
                        }
                    )
    apply_holm(rows, family_field="comparison_family")
    return rows


def probability_comparisons(
    root: Path,
    samples: int,
    block_size: int,
    include_evidence_upgrade: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = (42, 2027, 3407)
    for horizon in (1, 2, 3, 7):
        for task in ("load", "wind", "pv", "net_load"):
            probability_baselines = PROBABILITY_BASELINES
            if include_evidence_upgrade:
                probability_baselines = (*probability_baselines, *ISSUED_PROBABILITY_BASELINES)
            comparisons = [("raw", reference) for reference in probability_baselines]
            comparisons.extend(
                [
                    ("cqr", "issued_empirical"),
                    ("aci", "issued_empirical"),
                ]
            )
            for method, reference in comparisons:
                differences: list[np.ndarray] = []
                frame_values: list[np.ndarray] = []
                reference_values: list[np.ndarray] = []
                origins_reference: list[str] | None = None
                for seed in seeds:
                    run_name = (
                        "formal_qwen_issued_load-wind-pv-net_load_"
                        f"H1-2-3-7_L7d_seen-regions_seed{seed}"
                    )
                    raw_path = root / "issued_anchor" / "test" / f"seed{seed}" / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    if method == "raw":
                        frame_path = raw_path
                    else:
                        frame_path = (
                            root
                            / "issued_anchor"
                            / "conformal"
                            / f"seed{seed}"
                            / "artifacts"
                            / f"{task}_h{horizon}"
                            / f"{method}.npz"
                        )
                    baseline_root = (
                        root / "issued_probability_baselines"
                        if reference in ISSUED_PROBABILITY_BASELINES
                        else root / "probability_baselines"
                    )
                    baseline_tag = "_issued" if reference in ISSUED_PROBABILITY_BASELINES else ""
                    reference_path = (
                        baseline_root
                        / f"formal_aemo_L7d_H{horizon}d{baseline_tag}_seed{seed}"
                        / "artifacts"
                        / reference
                        / task
                        / "full.npz"
                    )
                    require_aligned_point_artifacts(
                        frame_path,
                        reference_path,
                        raw_path,
                        reference_path,
                    )
                    origins, frame_crps = probability_error_by_origin(frame_path, raw_path)
                    reference_origins, reference_crps = probability_error_by_origin(
                        reference_path,
                        reference_path,
                    )
                    if origins != reference_origins:
                        raise ValueError(
                            f"AEMO probability origins differ for {task}, H{horizon}, {method}, {reference}"
                        )
                    if origins_reference is None:
                        origins_reference = origins
                    elif origins_reference != origins:
                        raise ValueError(f"AEMO probability origins differ across seeds for {task}, H{horizon}")
                    differences.append(reference_crps - frame_crps)
                    frame_values.append(frame_crps)
                    reference_values.append(reference_crps)
                difference = np.mean(np.stack(differences), axis=0)
                frame_crps = float(np.mean(frame_values))
                reference_crps = float(np.mean(reference_values))
                statistics = difference_summary(
                    difference,
                    samples,
                    block_size,
                    6100 + len(rows),
                )
                rows.append(
                    {
                        "task": task,
                        "horizon_days": horizon,
                        "method": method,
                        "reference": reference,
                        "comparison_family": (
                            "raw_vs_probability_baselines"
                            if method == "raw"
                            else f"{method}_vs_issued_empirical"
                        ),
                        "seeds": len(seeds),
                        "origins": len(origins_reference or []),
                        "block_origins": min(block_size, len(origins_reference or [])),
                        "frame_crps": frame_crps,
                        "reference_crps": reference_crps,
                        "gain_percent": 100.0 * statistics["gain"] / max(reference_crps, 1e-12),
                        **statistics,
                    }
                )
    apply_holm(rows, family_field="comparison_family")
    return rows


def probability_block_sensitivity(
    root: Path,
    *,
    samples: int,
    block_lengths: tuple[int, ...],
    include_evidence_upgrade: bool,
) -> list[dict[str, Any]]:
    """Repeat the declared probability tests across predeclared block lengths."""

    rows: list[dict[str, Any]] = []
    for block_length in block_lengths:
        if block_length < 1:
            raise ValueError("Sensitivity block lengths must be positive")
        for row in probability_comparisons(
            root,
            samples,
            block_length,
            include_evidence_upgrade,
        ):
            rows.append(
                {
                    "sensitivity_bootstrap_samples": samples,
                    "sensitivity_block_origins": block_length,
                    **row,
                }
            )
    return rows


def sharing_run_dir(
    root: Path,
    configuration: str,
    task: str,
    horizon: int,
    seed: int,
) -> Path:
    """Resolve the one checkpoint containing a task-horizon condition."""

    if configuration == "fully_shared":
        parent = root / "frame_qwen"
        tasks = TASKS
        horizons = (1, 2, 3, 7)
    elif configuration == "task_specific":
        parent = root / "sharing_ablations" / configuration
        tasks = (task,)
        horizons = (1, 2, 3, 7)
    elif configuration == "horizon_specific":
        parent = root / "sharing_ablations" / configuration
        tasks = TASKS
        horizons = (horizon,)
    elif configuration == "independent":
        parent = root / "sharing_ablations" / configuration
        tasks = (task,)
        horizons = (horizon,)
    else:
        raise ValueError(f"Unknown sharing configuration {configuration}")
    name = (
        "formal_qwen_issued_"
        f"{'-'.join(tasks)}_H{'-'.join(str(value) for value in horizons)}_"
        f"L7d_seen-regions_seed{seed}"
    )
    return parent / name


def sharing_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_count = {
        "fully_shared": 1,
        "task_specific": len(TASKS),
        "horizon_specific": 4,
        "independent": len(TASKS) * 4,
    }
    display_name = {
        "fully_shared": "FRAME-Fully-Shared",
        "task_specific": "FRAME-Task-Specific",
        "horizon_specific": "FRAME-Horizon-Specific",
        "independent": "FRAME-Independent",
    }
    for configuration in display_name:
        for seed in (42, 2027, 3407):
            loaded: dict[Path, tuple[dict[str, Any], list[dict[str, str]]]] = {}
            for task in TASKS:
                for horizon in (1, 2, 3, 7):
                    run_dir = sharing_run_dir(root, configuration, task, horizon, seed)
                    if run_dir not in loaded:
                        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                        loaded[run_dir] = (summary, read_csv(run_dir / "test_metrics.csv"))
                    summary, metrics = loaded[run_dir]
                    row = next(
                        item
                        for item in metrics
                        if item["intervention"] == "full" and item["task"] == f"{task}_h{horizon}"
                    )
                    parameters = summary["parameter_summary"]
                    rows.append(
                        {
                            "configuration": configuration,
                            "method": display_name[configuration],
                            "seed": seed,
                            "task": task,
                            "horizon_days": horizon,
                            "models_per_seed": model_count[configuration],
                            "checkpoint_trainable_parameters": int(parameters["trainable"]),
                            "deployment_trainable_parameters": int(parameters["trainable"]) * model_count[configuration],
                            **{
                                key: float(row[key])
                                for key in ("mae", "rmse", "nmae", "smape", "mase")
                            },
                        }
                    )
    return rows


def sharing_comparisons(root: Path, samples: int, block_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration in ("task_specific", "horizon_specific", "independent"):
        for task in TASKS:
            for horizon in (1, 2, 3, 7):
                differences: list[np.ndarray] = []
                full_values: list[np.ndarray] = []
                comparison_values: list[np.ndarray] = []
                origins_reference: list[str] | None = None
                for seed in (42, 2027, 3407):
                    full_path = sharing_run_dir(root, "fully_shared", task, horizon, seed) / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    comparison_path = sharing_run_dir(root, configuration, task, horizon, seed) / "artifacts" / f"{task}_h{horizon}" / "full.npz"
                    require_aligned_point_artifacts(full_path, comparison_path)
                    full_origins, full_error = artifact_error_by_origin(full_path)
                    comparison_origins, comparison_error = artifact_error_by_origin(comparison_path)
                    if full_origins != comparison_origins:
                        raise ValueError(f"Sharing origin mismatch: {full_path} vs {comparison_path}")
                    if origins_reference is None:
                        origins_reference = full_origins
                    elif origins_reference != full_origins:
                        raise ValueError(f"Sharing origins differ across seeds for {configuration}, {task}, H{horizon}")
                    differences.append(comparison_error - full_error)
                    full_values.append(full_error)
                    comparison_values.append(comparison_error)
                difference = np.mean(np.stack(differences), axis=0)
                statistics = difference_summary(difference, samples, block_size, 11100 + len(rows))
                full_error = np.mean(np.stack(full_values), axis=0)
                comparison_error = np.mean(np.stack(comparison_values), axis=0)
                rows.append(
                    {
                        "comparison_family": "sharing_controls",
                        "configuration": configuration,
                        "task": task,
                        "horizon_days": horizon,
                        "seeds": 3,
                        "origins": len(origins_reference or []),
                        "block_origins": min(block_size, len(origins_reference or [])),
                        "fully_shared_nmae": float(full_error.mean()),
                        "comparison_nmae": float(comparison_error.mean()),
                        "nmae_gain": float(statistics["gain"]),
                        "nmae_gain_percent": float(
                            100.0 * statistics["gain"] / max(float(comparison_error.mean()), 1e-12)
                        ),
                        **statistics,
                    }
                )
    apply_holm(rows, family_field="comparison_family")
    return rows


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 500:
        raise ValueError("Use at least 500 moving-block bootstrap draws")
    if args.sensitivity_bootstrap_samples < 500:
        raise ValueError("Use at least 500 draws for the block-length sensitivity table")
    root = Path(args.root)
    table_root = root / "tables"
    point = collect_point_rows(root)
    probability = collect_probability_rows(root)
    holdout = collect_holdout_rows(root)
    reserve = collect_reserve_rows(root)
    point_summary = mean_std_rows(point, ("family", "method", "track", "task", "horizon_days"), ("mae", "rmse", "nmae", "smape", "mase"))
    probability_summary = mean_std_rows(probability, ("source", "method", "task", "horizon_days"), ("pinball", "crps", "ncrps", "wis", "picp_90", "ace_90", "pinaw_90", "quantile_crossing_rate"))
    reserve_summary = mean_std_rows(reserve, ("source", "method", "horizon_days", "cost_ratio"), ("mean_up_reserve", "mean_down_reserve", "mean_total_reserve", "reserve_shortfall_rate", "mean_imbalance_energy", "mean_reserve_cost", "mean_imbalance_cost", "mean_total_cost"))
    write_csv(table_root / "aemo_point_seed_rows.csv", point)
    write_csv(table_root / "aemo_point_summary.csv", point_summary)
    write_csv(table_root / "aemo_probability_seed_rows.csv", probability)
    write_csv(table_root / "aemo_probability_summary.csv", probability_summary)
    write_csv(table_root / "aemo_region_holdout.csv", holdout)
    holdout_summary = mean_std_rows(
        holdout,
        ("method", "region", "task", "horizon_days"),
        ("mae", "rmse", "nmae", "smape", "mase"),
    )
    write_csv(table_root / "aemo_region_holdout_summary.csv", holdout_summary)
    write_csv(table_root / "aemo_reserve_seed_rows.csv", reserve)
    write_csv(table_root / "aemo_reserve_summary.csv", reserve_summary)
    comparisons = paired_comparisons(
        root,
        args.bootstrap_samples,
        args.block_origins,
        args.include_evidence_upgrade,
    )
    write_csv(table_root / "aemo_paired_block_tests.csv", comparisons)
    ablation_comparisons: list[dict[str, Any]] = []
    information_comparison_rows: list[dict[str, Any]] = []
    if args.include_evidence_upgrade:
        ablation_comparisons = controlled_ablation_comparisons(
            root,
            args.bootstrap_samples,
            args.block_origins,
        )
        write_csv(table_root / "aemo_ablation_block_tests.csv", ablation_comparisons)
        information_comparison_rows = information_comparisons(
            root,
            args.bootstrap_samples,
            args.block_origins,
        )
        write_csv(table_root / "aemo_information_block_tests.csv", information_comparison_rows)
        sharing = sharing_rows(root)
        sharing_summary = mean_std_rows(
            sharing,
            ("configuration", "method", "task", "horizon_days", "models_per_seed"),
            ("checkpoint_trainable_parameters", "deployment_trainable_parameters", "mae", "rmse", "nmae", "smape", "mase"),
        )
        write_csv(table_root / "aemo_sharing_seed_rows.csv", sharing)
        write_csv(table_root / "aemo_sharing_summary.csv", sharing_summary)
        sharing_comparison_rows = sharing_comparisons(root, args.bootstrap_samples, args.block_origins)
        write_csv(table_root / "aemo_sharing_block_tests.csv", sharing_comparison_rows)
        holdout_comparison_rows = holdout_comparisons(
            root,
            args.bootstrap_samples,
            args.block_origins,
        )
        write_csv(table_root / "aemo_region_holdout_block_tests.csv", holdout_comparison_rows)
    else:
        sharing = []
        sharing_comparison_rows = []
        holdout_comparison_rows = []
    probability_tests = probability_comparisons(
        root,
        args.bootstrap_samples,
        args.block_origins,
        args.include_evidence_upgrade,
    )
    write_csv(table_root / "aemo_probability_block_tests.csv", probability_tests)
    if args.include_evidence_upgrade:
        probability_sensitivity = probability_block_sensitivity(
            root,
            samples=args.sensitivity_bootstrap_samples,
            block_lengths=tuple(args.sensitivity_block_origins),
            include_evidence_upgrade=True,
        )
        write_csv(table_root / "aemo_probability_block_sensitivity.csv", probability_sensitivity)
    else:
        probability_sensitivity = []
    summary = {
        "point_seed_rows": len(point),
        "probability_seed_rows": len(probability),
        "holdout_rows": len(holdout),
        "holdout_summary_rows": len(holdout_summary),
        "holdout_comparisons": len(holdout_comparison_rows),
        "reserve_seed_rows": len(reserve),
        "paired_comparisons": len(comparisons),
        "ablation_comparisons": len(ablation_comparisons),
        "information_comparisons": len(information_comparison_rows),
        "sharing_seed_rows": len(sharing),
        "sharing_comparisons": len(sharing_comparison_rows),
        "probability_comparisons": len(probability_tests),
        "probability_block_sensitivity_rows": len(probability_sensitivity),
        "bootstrap_samples": args.bootstrap_samples,
        "block_origins": args.block_origins,
        "sensitivity_bootstrap_samples": args.sensitivity_bootstrap_samples,
        "sensitivity_block_origins": args.sensitivity_block_origins,
        "include_evidence_upgrade": args.include_evidence_upgrade,
    }
    (table_root / "aggregation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
