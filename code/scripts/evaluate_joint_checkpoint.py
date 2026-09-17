#!/usr/bin/env python
"""Re-evaluate one completed joint checkpoint without retraining it.

This utility exists to keep qualitative artifacts tied to the exact checkpoint
used for formal metrics. It rebuilds the recorded chronological test splits,
applies the requested controlled interventions, and writes the same artifact
contract consumed by ``plot_four_task_controls.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.config import ModelConfig, TaskSpec
from src.energyca_paper.data import collate_task_batch, load_task_bundle
from src.energyca_paper.model import ForwardControls
from src.energyca_paper.training import (
    TrainingConfig,
    evaluate_tasks,
    lead_time_metric_rows,
    load_trained_model,
    make_loaders,
)


INTERVENTIONS = (
    "full",
    "zero_context",
    "mean_context",
    "shuffle_context",
    "missing_fields_masked",
    "missing_fields_unmasked",
    "delayed_context",
    "wrong_task",
    "wrong_region",
    "wrong_release_time",
    "shuffle_fields",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export formal test metrics and artifacts from an existing checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_root", default="outputs/context_caches/formal")
    parser.add_argument("--embedding_mode", choices=["numeric", "qwen"], default="qwen")
    parser.add_argument("--protocol", choices=["original", "aemo", "perform"], default="original")
    parser.add_argument("--profile", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--history_days", type=int, default=7)
    parser.add_argument(
        "--text_schema_variant",
        choices=["current", "aaai_exact"],
        default="current",
    )
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--split", choices=["val", "calibration", "test"], default="test")
    parser.add_argument(
        "--future_exog_policy",
        choices=["common", "issued", "oracle", "available", "zero"],
        default="available",
    )
    parser.add_argument("--calibration_fraction", type=float, default=0.0)
    parser.add_argument("--split_embargo_days", type=int, default=0)
    parser.add_argument("--calibration_embargo_days", type=int, default=0)
    parser.add_argument("--group_by_node", action="store_true")
    parser.add_argument("--interventions", nargs="+", choices=INTERVENTIONS, default=["full", "wrong_task"])
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_bundles(
    checkpoint: dict[str, Any],
    *,
    cache_root: Path,
    future_exog_policy: str,
    calibration_fraction: float,
    embedding_mode: str,
    split_embargo_days: int,
    calibration_embargo_days: int,
    protocol: str,
    profile: str,
    history_days: int,
    text_schema_variant: str,
) -> dict[str, Any]:
    model_config = ModelConfig(**checkpoint["model_config"])
    bundles: dict[str, Any] = {}
    for task, raw_spec in checkpoint["task_specs"].items():
        spec = TaskSpec(**raw_spec)
        cache_path = None
        if embedding_mode == "qwen":
            if protocol == "original":
                cache_path = cache_root / task / "qwen3_context_cache.npz"
            else:
                match = re.fullmatch(r"(load|wind|pv|net_load)(?:_h(\d+))?", task)
                if not match:
                    raise ValueError(f"Cannot resolve {protocol} cache for checkpoint task {task!r}")
                base_task = match.group(1)
                horizon_days = int(match.group(2) or round(spec.horizon * spec.cadence_minutes / 1440))
                group = f"L{history_days}d_H{horizon_days}d_{future_exog_policy}"
                cache_path = cache_root / group / profile / base_task / "qwen3_context_cache.npz"
        if cache_path is not None and not cache_path.exists():
            raise FileNotFoundError(f"Missing recorded frozen-LLM cache: {cache_path}")
        bundles[task] = load_task_bundle(
            spec,
            model_config=model_config,
            llm_cache=cache_path,
            require_llm=embedding_mode == "qwen",
            future_exog_policy=future_exog_policy,
            calibration_fraction=calibration_fraction,
            split_embargo_steps=(24 * 60 // spec.cadence_minutes) * split_embargo_days,
            calibration_embargo_days=calibration_embargo_days,
            text_schema_variant=text_schema_variant,
        )
    return bundles


def make_calibration_loaders(bundles: dict[str, Any], config: TrainingConfig) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for task, bundle in bundles.items():
        if "calibration" not in bundle.datasets:
            raise ValueError("--split calibration requires --calibration_fraction greater than zero")
        loaders[task] = DataLoader(
            bundle.datasets["calibration"],
            batch_size=config.batch_size(task),
            shuffle=False,
            collate_fn=collate_task_batch,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def make_node_grouped_loaders(bundles: dict[str, Any], split: str) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for task, bundle in bundles.items():
        dataset = bundle.datasets[split]
        groups: dict[str, list[int]] = {}
        for index, example in enumerate(dataset.examples):
            groups.setdefault(str(example.node_id), []).append(index)
        loaders[task] = DataLoader(
            dataset,
            batch_sampler=list(groups.values()),
            collate_fn=collate_task_batch,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("This evaluation protocol requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundles = load_bundles(
        checkpoint,
        cache_root=Path(args.cache_root),
        future_exog_policy=args.future_exog_policy,
        calibration_fraction=args.calibration_fraction,
        embedding_mode=args.embedding_mode,
        split_embargo_days=args.split_embargo_days,
        calibration_embargo_days=args.calibration_embargo_days,
        protocol=args.protocol,
        profile=args.profile,
        history_days=args.history_days,
        text_schema_variant=args.text_schema_variant,
    )
    training_config = TrainingConfig(**checkpoint["training_config"])
    _, val_loaders, test_loaders = make_loaders(bundles, training_config)
    if args.split == "val":
        loaders = val_loaders
    elif args.split == "calibration":
        loaders = make_calibration_loaders(bundles, training_config)
    else:
        loaders = test_loaders
    if args.group_by_node:
        loaders = make_node_grouped_loaders(bundles, args.split)
    model = load_trained_model(checkpoint_path, device)
    controls = ForwardControls(**checkpoint["controls"])
    task_scales = {task: float(value) for task, value in checkpoint["task_scales"].items()}

    test_metrics: dict[str, Any] = {}
    lead_metric_rows: list[dict[str, Any]] = []
    for intervention in args.interventions:
        metrics, artifacts = evaluate_tasks(
            model,
            loaders,
            task_scales=task_scales,
            controls=controls,
            intervention=intervention,
            device=device,
            use_bf16=training_config.use_bf16,
        )
        test_metrics[intervention] = metrics
        for task, artifact in artifacts.items():
            artifact_dir = output_dir / "artifacts" / task
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact_dir / f"{intervention}.npz", **artifact)
            lead_metric_rows.extend(
                lead_time_metric_rows(
                    artifact,
                    task=task,
                    intervention=intervention,
                    cadence_minutes=bundles[task].spec.cadence_minutes,
                )
            )

    if lead_metric_rows:
        with (output_dir / "lead_time_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(lead_metric_rows[0]))
            writer.writeheader()
            writer.writerows(lead_metric_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "source_run": checkpoint["run_name"],
        "best_epoch": checkpoint["best_epoch"],
        "best_mean_val_nmae": checkpoint["best_mean_val_nmae"],
        "single_checkpoint": True,
        # Keep the exact frozen training/model configuration next to every
        # released prediction artifact.  The final evidence audit must not
        # have to infer settings from a directory name or a live code state.
        "model_config": checkpoint["model_config"],
        "training_config": checkpoint["training_config"],
        "parameter_summary": model.parameter_summary(),
        "controls": asdict(controls),
        "interventions": list(args.interventions),
        "split": args.split,
        "future_exog_policy": args.future_exog_policy,
        "calibration_fraction": args.calibration_fraction,
        "embedding_mode": args.embedding_mode,
        "protocol": args.protocol,
        "text_schema_variant": args.text_schema_variant,
        "split_embargo_days": args.split_embargo_days,
        "calibration_embargo_days": args.calibration_embargo_days,
        "group_by_node": args.group_by_node,
        "metrics": test_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
