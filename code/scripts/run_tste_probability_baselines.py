#!/usr/bin/env python
"""Train matched history-only quantile baselines for AEMO or PERFORM."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.config import ModelConfig, TaskSpec, aemo_task_specs, perform_task_specs
from src.energyca_paper.data import TaskDataBundle, collate_task_batch, load_task_bundle, move_batch_to_device
from src.energyca_paper.metrics import (
    DEFAULT_QUANTILE_LEVELS,
    mase,
    point_metrics,
    probabilistic_metrics,
)
from src.energyca_paper.training import lead_time_metric_rows


TASKS = ("load", "wind", "pv", "net_load")
METHODS = (
    "seasonal_empirical",
    "issued_empirical",
    "dlinear_quantile",
    "lstm_quantile",
    "patchtst_quantile",
    "dlinear_issued_quantile",
    "lstm_issued_quantile",
    "patchtst_issued_quantile",
)
DEFAULT_METHODS = (
    "seasonal_empirical",
    "issued_empirical",
    "dlinear_quantile",
    "lstm_quantile",
    "patchtst_quantile",
)
ISSUED_MODEL_METHODS = (
    "dlinear_issued_quantile",
    "lstm_issued_quantile",
    "patchtst_issued_quantile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["aemo", "perform"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--profile", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--history_days", type=int, default=7)
    parser.add_argument("--horizon_days", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--split_embargo_days", type=int, default=7)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--calibration_embargo_days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps_per_epoch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output_root", default="outputs/tste_probability_baselines")
    parser.add_argument("--run_tag", default="")
    parser.add_argument(
        "--evaluation_only_from_checkpoints",
        action="store_true",
        help=(
            "Rebuild test and chronological-calibration artifacts from the frozen "
            "best checkpoints in the resolved output directory, without retraining."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QuantileDLinear(nn.Module):
    def __init__(self, history: int, horizon: int, quantiles: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.linear = nn.Linear(history, horizon * quantiles)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.linear(history).view(history.shape[0], self.horizon, self.quantiles).sort(dim=-1).values


class QuantileLSTM(nn.Module):
    def __init__(self, horizon: int, quantiles: int, hidden: int = 64) -> None:
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon * quantiles)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(history.unsqueeze(-1))
        return self.head(hidden[-1]).view(history.shape[0], self.horizon, self.quantiles).sort(dim=-1).values


class QuantilePatchTST(nn.Module):
    def __init__(self, history: int, horizon: int, quantiles: int, *, patch: int, stride: int, d_model: int = 64) -> None:
        super().__init__()
        if history < patch:
            raise ValueError("PatchTST history must contain at least one patch")
        self.patch = patch
        self.stride = stride
        self.horizon = horizon
        self.quantiles = quantiles
        patches = 1 + (history - patch) // stride
        self.embed = nn.Linear(patch, d_model)
        self.position = nn.Parameter(torch.zeros(1, patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, horizon * quantiles)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        patches = history.unfold(1, self.patch, self.stride)
        encoded = self.encoder(self.embed(patches) + self.position[:, : patches.shape[1]])
        return self.head(encoded.mean(dim=1)).view(history.shape[0], self.horizon, self.quantiles).sort(dim=-1).values


def make_model(method: str, spec: TaskSpec, quantiles: int) -> nn.Module:
    base_method = method.replace("_issued", "")
    if base_method == "dlinear_quantile":
        return QuantileDLinear(spec.history_len, spec.horizon, quantiles)
    if base_method == "lstm_quantile":
        return QuantileLSTM(spec.horizon, quantiles)
    if base_method == "patchtst_quantile":
        day = 24 * 60 // spec.cadence_minutes
        return QuantilePatchTST(spec.history_len, spec.horizon, quantiles, patch=day, stride=max(1, day // 2))
    raise ValueError(f"Unknown trainable method {method}")


def constrain_normalized(
    quantiles: torch.Tensor,
    task: str,
    *,
    aemo_capacity_factor: bool,
) -> torch.Tensor:
    if task == "net_load":
        return quantiles
    quantiles = torch.clamp_min(quantiles, 0.0)
    if task == "wind" and aemo_capacity_factor:
        quantiles = torch.clamp_max(quantiles, 1.0)
    return quantiles


def to_raw(quantiles: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    return quantiles * batch["target_scale"].unsqueeze(-1) + batch["target_offset"].unsqueeze(-1)


def issued_target_from_batch(batch: dict[str, Any], spec: TaskSpec) -> torch.Tensor:
    return batch["tokens"][:, spec.history_len :, 5 + int(spec.task_id)]


def model_quantiles(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    spec: TaskSpec,
    method: str,
    aemo_capacity_factor: bool,
) -> torch.Tensor:
    history = batch["tokens"][:, : spec.history_len, 0]
    quantiles = model(history)
    if method in ISSUED_MODEL_METHODS:
        quantiles = quantiles + issued_target_from_batch(batch, spec).unsqueeze(-1)
    return constrain_normalized(
        quantiles,
        spec.key,
        aemo_capacity_factor=aemo_capacity_factor,
    )


def pinball_loss(quantiles: torch.Tensor, batch: dict[str, Any], levels: torch.Tensor, scale: float) -> torch.Tensor:
    residual = (batch["target_raw"].unsqueeze(-1) - to_raw(quantiles, batch)) / scale
    tau = levels.to(residual).view(1, 1, -1)
    return torch.maximum(tau * residual, (tau - 1.0) * residual).mean()


def loaders(bundle: TaskDataBundle, batch_size: int, seed: int) -> dict[str, DataLoader]:
    kwargs = {"batch_size": batch_size, "collate_fn": collate_task_batch, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    generator = torch.Generator().manual_seed(seed + bundle.spec.task_id)
    output = {
        "train": DataLoader(bundle.datasets["train"], shuffle=True, generator=generator, **kwargs),
        "val": DataLoader(bundle.datasets["val"], shuffle=False, **kwargs),
        "test": DataLoader(bundle.datasets["test"], shuffle=False, **kwargs),
    }
    if "calibration" in bundle.datasets:
        output["calibration"] = DataLoader(
            bundle.datasets["calibration"], shuffle=False, **kwargs
        )
    if "confirmation" in bundle.datasets:
        output["confirmation"] = DataLoader(
            bundle.datasets["confirmation"], shuffle=False, **kwargs
        )
    return output


def training_scale(bundle: TaskDataBundle) -> float:
    values = np.concatenate([np.asarray(example.target_raw).reshape(-1) for example in bundle.datasets["train"].examples])
    return max(float(np.mean(np.abs(values))), 1e-6)


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    task: str,
    method: str,
    levels: np.ndarray,
    device: torch.device,
    aemo_capacity_factor: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    quantiles, targets, histories, origins, nodes, issued_paths = [], [], [], [], [], []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                normalized = model_quantiles(
                    model,
                    batch,
                    spec=loader.dataset.spec,
                    method=method,
                    aemo_capacity_factor=aemo_capacity_factor,
                )
            quantiles.append(to_raw(normalized.float(), batch).cpu().numpy())
            targets.append(batch["target_raw"].cpu().numpy())
            histories.append(batch["history_raw"].cpu().numpy())
            origins.extend(str(value) for value in raw_batch["forecast_start"])
            nodes.extend(str(value) for value in raw_batch["node_id"])
            if method in ISSUED_MODEL_METHODS:
                issued_paths.append(
                    to_raw(
                        issued_target_from_batch(batch, loader.dataset.spec).unsqueeze(-1),
                        batch,
                    )
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
    quantile_array = np.concatenate(quantiles)
    artifact = {
        "quantiles_raw": quantile_array.astype(np.float32),
        "prediction_raw": quantile_array[..., int(np.argmin(np.abs(levels - 0.5)))].astype(np.float32),
        "target_raw": np.concatenate(targets).astype(np.float32),
        "history_raw": np.concatenate(histories).astype(np.float32),
        "quantile_levels": levels.astype(np.float32),
        "forecast_start": np.asarray(origins, dtype=str),
        "node_id": np.asarray(nodes, dtype=str),
    }
    if issued_paths:
        artifact["issued_prediction_raw"] = np.concatenate(issued_paths).astype(np.float32)
    return artifact


def seasonal_prediction(example: Any, spec: TaskSpec) -> np.ndarray:
    period = 7 * 24 * 60 // spec.cadence_minutes
    history = np.asarray(example.history_raw, dtype=np.float64)
    cycle = history[-period:]
    return np.tile(cycle, math.ceil(spec.horizon / cycle.size))[: spec.horizon]


def empirical_artifact(
    bundle: TaskDataBundle,
    *,
    split: str,
    levels: np.ndarray,
) -> dict[str, np.ndarray]:
    train_examples = bundle.datasets["train"].examples
    residuals = np.stack([np.asarray(example.target_raw, dtype=np.float64) - seasonal_prediction(example, bundle.spec) for example in train_examples])
    residual_quantiles = np.quantile(residuals, levels, axis=0).T
    quantiles, targets, histories, origins, nodes = [], [], [], [], []
    for example in bundle.datasets[split].examples:
        forecast = seasonal_prediction(example, bundle.spec)
        task_quantiles = forecast[:, None] + residual_quantiles
        if bundle.spec.key != "net_load":
            task_quantiles = np.maximum(task_quantiles, 0.0)
        quantiles.append(np.maximum.accumulate(task_quantiles, axis=-1))
        targets.append(np.asarray(example.target_raw))
        histories.append(np.asarray(example.history_raw))
        origins.append(str(example.forecast_start))
        nodes.append(str(example.node_id))
    quantile_array = np.stack(quantiles)
    return {
        "quantiles_raw": quantile_array.astype(np.float32),
        "prediction_raw": quantile_array[..., int(np.argmin(np.abs(levels - 0.5)))].astype(np.float32),
        "target_raw": np.stack(targets).astype(np.float32),
        "history_raw": np.stack(histories).astype(np.float32),
        "quantile_levels": levels.astype(np.float32),
        "forecast_start": np.asarray(origins, dtype=str),
        "node_id": np.asarray(nodes, dtype=str),
    }


def issued_point(dataset: Any, index: int) -> np.ndarray:
    item = dataset[index]
    horizon = dataset.spec.horizon
    task_column = 5 + dataset.spec.task_id
    issued_normalized = item["tokens"][-horizon:, task_column].numpy()
    return (
        issued_normalized * item["target_scale"].numpy()
        + item["target_offset"].numpy()
    )


def issued_empirical_artifact(
    bundle: TaskDataBundle,
    *,
    split: str,
    levels: np.ndarray,
) -> dict[str, np.ndarray]:
    train = bundle.datasets["train"]
    residuals = np.stack(
        [
            np.asarray(example.target_raw, dtype=np.float64) - issued_point(train, index)
            for index, example in enumerate(train.examples)
        ]
    )
    residual_quantiles = np.quantile(residuals, levels, axis=0).T
    quantiles, targets, histories, origins, nodes = [], [], [], [], []
    evaluation = bundle.datasets[split]
    for index, example in enumerate(evaluation.examples):
        task_quantiles = issued_point(evaluation, index)[:, None] + residual_quantiles
        if bundle.spec.key != "net_load":
            task_quantiles = np.maximum(task_quantiles, 0.0)
        quantiles.append(np.maximum.accumulate(task_quantiles, axis=-1))
        targets.append(np.asarray(example.target_raw))
        histories.append(np.asarray(example.history_raw))
        origins.append(str(example.forecast_start))
        nodes.append(str(example.node_id))
    quantile_array = np.stack(quantiles)
    return {
        "quantiles_raw": quantile_array.astype(np.float32),
        "prediction_raw": quantile_array[..., int(np.argmin(np.abs(levels - 0.5)))].astype(np.float32),
        "target_raw": np.stack(targets).astype(np.float32),
        "history_raw": np.stack(histories).astype(np.float32),
        "quantile_levels": levels.astype(np.float32),
        "forecast_start": np.asarray(origins, dtype=str),
        "node_id": np.asarray(nodes, dtype=str),
    }


def artifact_metrics(
    artifact: dict[str, np.ndarray],
    *,
    cadence_minutes: int,
) -> dict[str, float]:
    metrics = point_metrics(artifact["prediction_raw"], artifact["target_raw"])
    metrics["mase"] = mase(
        artifact["prediction_raw"],
        artifact["target_raw"],
        insample=artifact["history_raw"].reshape(-1),
        seasonality=24 * 60 // cadence_minutes,
    )
    metrics.update(probabilistic_metrics(artifact["quantiles_raw"], artifact["target_raw"], levels=artifact["quantile_levels"]))
    metrics["examples"] = int(artifact["target_raw"].shape[0])
    return metrics


def evaluate_pinball(
    model: nn.Module,
    loader: DataLoader,
    *,
    task: str,
    method: str,
    levels: torch.Tensor,
    scale: float,
    device: torch.device,
    aemo_capacity_factor: bool,
) -> float:
    model.eval()
    values = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model_quantiles(
                    model,
                    batch,
                    spec=loader.dataset.spec,
                    method=method,
                    aemo_capacity_factor=aemo_capacity_factor,
                )
                values.append(float(pinball_loss(prediction, batch, levels, scale).cpu()))
    return float(np.mean(values))


def train_model(method: str, bundle: TaskDataBundle, task_loaders: dict[str, DataLoader], *, args: argparse.Namespace, epochs: int, steps: int, patience: int, output_dir: Path, device: torch.device, levels: np.ndarray) -> tuple[nn.Module, list[dict[str, float]]]:
    model = make_model(method, bundle.spec, len(levels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scale = training_scale(bundle)
    level_tensor = torch.tensor(levels, dtype=torch.float32, device=device)
    iterator = iter(task_loaders["train"])
    best = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint = output_dir / "best_checkpoint.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(task_loaders["train"])
                raw_batch = next(iterator)
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model_quantiles(
                    model,
                    batch,
                    spec=bundle.spec,
                    method=method,
                    aemo_capacity_factor=args.protocol == "aemo",
                )
                loss = pinball_loss(prediction, batch, level_tensor, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate_pinball(
            model,
            task_loaders["val"],
            task=bundle.spec.key,
            method=method,
            levels=level_tensor,
            scale=scale,
            device=device,
            aemo_capacity_factor=args.protocol == "aemo",
        )
        row = {"epoch": float(epoch), "train_pinball": float(np.mean(losses)), "val_pinball": val}
        history.append(row)
        print(json.dumps({"method": method, "task": bundle.spec.key, **row}), flush=True)
        if val < best:
            best = val
            stale = 0
            torch.save({"state_dict": model.state_dict(), "method": method, "spec": bundle.spec.to_dict(), "best_val_pinball": best}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model, history


def load_frozen_model(
    method: str,
    bundle: TaskDataBundle,
    *,
    checkpoint: Path,
    device: torch.device,
    quantile_count: int,
) -> nn.Module:
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Frozen baseline checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("method") != method:
        raise ValueError(
            f"Checkpoint method mismatch for {checkpoint}: {payload.get('method')} != {method}"
        )
    if payload.get("spec") != bundle.spec.to_dict():
        raise ValueError(f"Checkpoint task specification mismatch for {checkpoint}")
    model = make_model(method, bundle.spec, quantile_count).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.protocol == "aemo":
        os.environ["AEMO_UNIFIED_DATA_DIR"] = str(Path(args.data_root).resolve())
        factory = aemo_task_specs
    else:
        os.environ["PERFORM_ERCOT_DATA_DIR"] = str(Path(args.data_root).resolve())
        factory = perform_task_specs
    if args.profile == "smoke":
        if args.protocol == "aemo":
            train_windows, eval_windows = 8, 4
        else:
            train_windows, eval_windows = 16, 8
        epochs, steps, patience = 2, 4, 2
    else:
        train_windows, eval_windows, epochs, steps, patience = None, None, 20, 128, 5
    epochs = args.epochs if args.epochs is not None else epochs
    steps = args.steps_per_epoch if args.steps_per_epoch is not None else steps
    patience = args.patience if args.patience is not None else patience
    specs = factory(history_days=args.history_days, horizon_days=args.horizon_days, train_windows_per_node=train_windows, eval_windows_per_node=eval_windows)
    model_config = ModelConfig(task_output_modes=("relu", "unit_interval" if args.protocol == "aemo" else "relu", "relu", "identity"))
    bundles = {task: load_task_bundle(specs[task], model_config=model_config, llm_cache=None, require_llm=False, future_exog_policy="issued", calibration_fraction=args.calibration_fraction, split_embargo_steps=(24 * 60 // specs[task].cadence_minutes) * args.split_embargo_days, calibration_embargo_days=args.calibration_embargo_days) for task in args.tasks}
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This experiment protocol requires CUDA training")
    device = torch.device(args.device)
    set_seed(args.seed)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    output_dir = Path(args.output_root) / f"{args.profile}_{args.protocol}_L{args.history_days}d_H{args.horizon_days}d{tag}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_manifests.json").write_text(
        json.dumps({task: bundle.manifest for task, bundle in bundles.items()}, indent=2),
        encoding="utf-8",
    )
    existing_summary: dict[str, Any] = {}
    summary_path = output_dir / "summary.json"
    if args.evaluation_only_from_checkpoints:
        if summary_path.is_file() and summary_path.stat().st_size > 0:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "protocol": args.protocol,
                "history_days": args.history_days,
                "horizon_days": args.horizon_days,
                "information_track": "issued",
                "split_embargo_days": args.split_embargo_days,
                "calibration_fraction": args.calibration_fraction,
                "calibration_embargo_days": args.calibration_embargo_days,
                "seed": args.seed,
            }
            mismatched = {
                key: {"expected": value, "observed": existing_summary.get(key)}
                for key, value in expected.items()
                if existing_summary.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    "Existing baseline summary does not match the requested protocol: "
                    + json.dumps(mismatched, sort_keys=True)
                )
    started = time.perf_counter()
    metric_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    model_summaries: dict[str, Any] = {}
    for task, bundle in bundles.items():
        task_loaders = loaders(bundle, args.batch_size, args.seed)
        if "calibration" not in task_loaders:
            raise ValueError("Probability baselines require a chronological calibration split")
        for method in args.methods:
            method_started = time.perf_counter()
            method_dir = output_dir / method / task
            method_dir.mkdir(parents=True, exist_ok=True)
            if method == "seasonal_empirical":
                artifact = empirical_artifact(bundle, split="test", levels=levels)
                calibration_artifact = empirical_artifact(
                    bundle, split="calibration", levels=levels
                )
                histories[f"{method}:{task}"] = []
                model_summaries[f"{method}:{task}"] = {
                    "total_parameters": 0,
                    "trainable_parameters": 0,
                }
            elif method == "issued_empirical":
                artifact = issued_empirical_artifact(bundle, split="test", levels=levels)
                calibration_artifact = issued_empirical_artifact(
                    bundle, split="calibration", levels=levels
                )
                histories[f"{method}:{task}"] = []
                model_summaries[f"{method}:{task}"] = {
                    "total_parameters": 0,
                    "trainable_parameters": 0,
                }
            else:
                if args.evaluation_only_from_checkpoints:
                    model = load_frozen_model(
                        method,
                        bundle,
                        checkpoint=method_dir / "best_checkpoint.pt",
                        device=device,
                        quantile_count=len(levels),
                    )
                    history = existing_summary.get("history", {}).get(
                        f"{method}:{task}", []
                    )
                else:
                    model, history = train_model(method, bundle, task_loaders, args=args, epochs=epochs, steps=steps, patience=patience, output_dir=method_dir, device=device, levels=levels)
                artifact = predict_model(
                    model,
                    task_loaders["test"],
                    task=task,
                    method=method,
                    levels=levels,
                    device=device,
                    aemo_capacity_factor=args.protocol == "aemo",
                )
                calibration_artifact = predict_model(
                    model,
                    task_loaders["calibration"],
                    task=task,
                    method=method,
                    levels=levels,
                    device=device,
                    aemo_capacity_factor=args.protocol == "aemo",
                )
                histories[f"{method}:{task}"] = history
                model_summaries[f"{method}:{task}"] = {
                    "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                    "trainable_parameters": int(
                        sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        )
                    ),
                }
            metrics = artifact_metrics(
                artifact, cadence_minutes=bundle.spec.cadence_minutes
            )
            metric_rows.append({"method": method, "task": task, **metrics})
            artifact_dir = output_dir / "artifacts" / method / task
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact_dir / "full.npz", **artifact)
            calibration_artifact_dir = output_dir / "calibration_artifacts" / method / task
            calibration_artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                calibration_artifact_dir / "full.npz", **calibration_artifact
            )
            model_summaries[f"{method}:{task}"]["train_and_evaluation_seconds"] = float(
                time.perf_counter() - method_started
            )
            lead_rows.extend({"method": method, **row} for row in lead_time_metric_rows(artifact, task=task, intervention="full", cadence_minutes=bundle.spec.cadence_minutes))
    write_csv(output_dir / "test_metrics.csv", metric_rows)
    write_csv(output_dir / "lead_time_metrics.csv", lead_rows)
    current_runtime = float(time.perf_counter() - started)
    summary = dict(existing_summary) if args.evaluation_only_from_checkpoints else {}
    summary.update({"protocol": args.protocol, "data_root": str(Path(args.data_root).resolve()), "history_days": args.history_days, "horizon_days": args.horizon_days, "information_track": "issued", "split_embargo_days": args.split_embargo_days, "calibration_fraction": args.calibration_fraction, "calibration_embargo_days": args.calibration_embargo_days, "seed": args.seed, "methods": args.methods, "tasks": args.tasks, "quantile_levels": levels.tolist(), "device": str(device), "history": histories, "issued_residual_methods": [method for method in args.methods if method in ISSUED_MODEL_METHODS], "calibration_artifacts": True, "data_manifests": str(output_dir / "data_manifests.json"), "model_summaries": model_summaries})
    if args.evaluation_only_from_checkpoints:
        summary["evaluation_rebuild_seconds"] = current_runtime
        summary["calibration_artifacts_rebuilt_from_frozen_checkpoints"] = True
        summary["checkpoint_reuse_verified"] = True
        summary["summary_rebuilt_from_checkpoints"] = not bool(existing_summary)
    else:
        summary["runtime_seconds"] = current_runtime
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "rows": len(metric_rows)}))


if __name__ == "__main__":
    main()
