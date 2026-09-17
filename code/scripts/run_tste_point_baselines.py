#!/usr/bin/env python
"""Run auditable history-only point-forecast baselines under the TSTE protocol."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energyca_paper.config import (
    ModelConfig,
    TaskSpec,
    aemo_task_specs,
    paper_task_specs,
    perform_task_specs,
)
from src.energyca_paper.data import (
    PaperEnergyDataset,
    TaskDataBundle,
    collate_task_batch,
    load_task_bundle,
    move_batch_to_device,
)
from src.energyca_paper.metrics import mase, point_metrics
from src.energyca_paper.training import LEAD_TIME_BINS_HOURS, lead_time_metric_rows


TASKS = ("load", "wind", "pv", "net_load")
REGIONS = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")
METHODS = (
    "issued_official",
    "persistence",
    "seasonal_naive",
    "weekly_seasonal_naive",
    "dlinear",
    "lstm_issued",
    "patchtst_issued",
    "timexer_issued",
    "tft_issued",
    "issued_bias",
    "issued_residual_linear",
)
DEFAULT_METHODS = (
    "persistence",
    "seasonal_naive",
    "weekly_seasonal_naive",
    "dlinear",
)


class DirectLinear(nn.Module):
    """Direct multi-step linear baseline using only the normalized target history."""

    def __init__(self, history_len: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(history_len, horizon)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.linear(history)


class IssuedResidualLSTM(nn.Module):
    """Predict a residual around an issued trajectory from target history."""

    def __init__(self, horizon: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(history.unsqueeze(-1))
        return self.head(hidden[-1])


class IssuedResidualPatchTST(nn.Module):
    """Patch-based residual model around an issued trajectory."""

    def __init__(self, history: int, horizon: int, *, patch: int, stride: int, d_model: int = 64) -> None:
        super().__init__()
        if history < patch:
            raise ValueError("PatchTST history must contain at least one patch")
        patches = 1 + (history - patch) // stride
        self.patch = patch
        self.stride = stride
        self.embed = nn.Linear(patch, d_model)
        self.position = nn.Parameter(torch.zeros(1, patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, horizon)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        patches = history.unfold(1, self.patch, self.stride)
        encoded = self.encoder(self.embed(patches) + self.position[:, : patches.shape[1]])
        return self.head(encoded.mean(dim=1))


class IssuedResidualTimeXer(nn.Module):
    """TimeXer-style residual model with endogenous patches and exogenous variate tokens."""

    def __init__(
        self,
        history: int,
        horizon: int,
        *,
        patch: int,
        stride: int,
        token_features: int = 9,
        context_dim: int = 69,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        patches = 1 + (history - patch) // stride
        self.history = history
        self.horizon = horizon
        self.patch = patch
        self.stride = stride
        self.token_features = token_features
        self.target_projection = nn.Linear(patch, d_model)
        self.target_position = nn.Parameter(torch.zeros(1, patches, d_model))
        self.exogenous_projection = nn.Linear(history + horizon, d_model)
        self.exogenous_type = nn.Parameter(torch.zeros(1, token_features - 1, d_model))
        self.static_projection = nn.Linear(context_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, 4, dropout=0.1, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, horizon)
        nn.init.normal_(self.target_position, std=0.02)
        nn.init.normal_(self.exogenous_type, std=0.02)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        target = tokens[:, : self.history, 0]
        patches = target.unfold(1, self.patch, self.stride)
        target_tokens = self.target_projection(patches) + self.target_position[:, : patches.shape[1]]
        exogenous = tokens[:, :, 1 : self.token_features].transpose(1, 2)
        exogenous_tokens = self.exogenous_projection(exogenous) + self.exogenous_type
        static_token = self.static_projection(context).unsqueeze(1)
        auxiliary_tokens = torch.cat([exogenous_tokens, static_token], dim=1)
        cross, _ = self.cross_attention(target_tokens, auxiliary_tokens, auxiliary_tokens, need_weights=False)
        encoded = self.encoder(target_tokens + cross)
        return self.head(encoded.mean(dim=1))


class GatedResidualNetwork(nn.Module):
    """Compact gated residual block used by the protocol-matched TFT baseline."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value + self.gate(value) * self.candidate(value))


class IssuedResidualTFT(nn.Module):
    """Compact TFT residual model with variable selection, recurrence, and temporal attention."""

    def __init__(
        self,
        history: int,
        *,
        token_features: int = 9,
        context_dim: int = 69,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.history = history
        self.token_features = token_features
        self.variable_weight = nn.Parameter(torch.empty(token_features, d_model))
        self.variable_bias = nn.Parameter(torch.zeros(token_features, d_model))
        self.variable_gate = nn.Linear(token_features, token_features)
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.static_projection = nn.Linear(context_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, 4, dropout=0.1, batch_first=True)
        self.enrichment = GatedResidualNetwork(d_model)
        self.output = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.variable_weight)

    def select_variables(self, values: torch.Tensor) -> torch.Tensor:
        gate = torch.softmax(self.variable_gate(values), dim=-1)
        embedded = values.unsqueeze(-1) * self.variable_weight + self.variable_bias
        return torch.sum(gate.unsqueeze(-1) * embedded, dim=2)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        selected = self.select_variables(tokens[:, :, : self.token_features])
        history, state = self.encoder(selected[:, : self.history])
        future, _ = self.decoder(selected[:, self.history :], state)
        memory = torch.cat([history, future], dim=1)
        attended, _ = self.attention(future, memory, memory, need_weights=False)
        enriched = self.enrichment(future + attended + self.static_projection(context).unsqueeze(1))
        return self.output(enriched).squeeze(-1)


class IssuedResidualLinear(nn.Module):
    """Numerically correct an issued target trajectory at the forecast origin."""

    def __init__(self, history_len: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(history_len + horizon, horizon)

    def forward(self, history: torch.Tensor, issued: torch.Tensor) -> torch.Tensor:
        return issued + self.linear(torch.cat([history, issued], dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--protocol", choices=["original", "aemo", "perform"], default="original")
    parser.add_argument("--data_root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--history_days", type=int, default=7)
    parser.add_argument("--horizon_days", type=int, default=1)
    parser.add_argument("--forecast_stride_days", type=int, default=1)
    parser.add_argument(
        "--future_exog_policy",
        choices=["common", "issued", "oracle", "available", "zero"],
        default="common",
    )
    parser.add_argument("--split_embargo_days", type=int, default=0)
    parser.add_argument("--holdout_region", choices=REGIONS)
    parser.add_argument(
        "--run_tag",
        default="",
        help="Optional label appended to the output directory, such as issued or holdout-NSW1.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps_per_epoch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_root", default="outputs/tste_point_baselines")
    parser.add_argument("--batch_size_load", type=int, default=32)
    parser.add_argument("--batch_size_wind", type=int, default=4)
    parser.add_argument("--batch_size_pv", type=int, default=32)
    parser.add_argument("--batch_size_net_load", type=int, default=32)
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--strict_determinism", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, *, strict: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_profile(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.profile == "smoke":
        epochs, steps, patience = 3, 4, 3
    else:
        epochs, steps, patience = 20, 128, 5
    return (
        epochs if args.epochs is None else args.epochs,
        steps if args.steps_per_epoch is None else args.steps_per_epoch,
        patience if args.patience is None else args.patience,
    )


def batch_size(args: argparse.Namespace, task: str) -> int:
    return int(getattr(args, f"batch_size_{task}"))


def make_loaders(bundle: TaskDataBundle, args: argparse.Namespace) -> dict[str, DataLoader]:
    kwargs = {
        "batch_size": batch_size(args, bundle.spec.key),
        "collate_fn": collate_task_batch,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(args.seed + bundle.spec.task_id)
    return {
        "train": DataLoader(bundle.datasets["train"], shuffle=True, generator=generator, **kwargs),
        "val": DataLoader(bundle.datasets["val"], shuffle=False, **kwargs),
        "test": DataLoader(bundle.datasets["test"], shuffle=False, **kwargs),
    }


def task_scale(bundle: TaskDataBundle) -> float:
    target = np.concatenate(
        [np.asarray(example.target_raw, dtype=np.float64).reshape(-1) for example in bundle.datasets["train"].examples]
    )
    return max(float(np.mean(np.abs(target))), 1e-6)


def normalized_to_raw(prediction: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    return prediction * batch["target_scale"] + batch["target_offset"]


def issued_target_from_batch(batch: dict[str, Any], spec: TaskSpec) -> torch.Tensor:
    """Return the target-specific issued trajectory from the shared token schema."""

    return batch["tokens"][:, spec.history_len :, 5 + int(spec.task_id)]


def normalized_target_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    return (batch["target_raw"] - batch["target_offset"]) / batch["target_scale"]


def baseline_prediction(
    method: str,
    batch: dict[str, Any],
    *,
    spec: TaskSpec,
    model: nn.Module | None,
    issued_bias: torch.Tensor | None,
) -> torch.Tensor:
    history = batch["tokens"][:, : spec.history_len, 0]
    if method == "issued_official":
        prediction = issued_target_from_batch(batch, spec)
    elif method == "persistence":
        prediction = history[:, -1:].expand(-1, spec.horizon)
    elif method == "seasonal_naive":
        daily_period = int(24 * 60 / spec.cadence_minutes)
        cycle = history[:, -daily_period:]
        repeats = (spec.horizon + daily_period - 1) // daily_period
        prediction = cycle.repeat(1, repeats)[:, : spec.horizon]
    elif method == "weekly_seasonal_naive":
        weekly_period = int(7 * 24 * 60 / spec.cadence_minutes)
        if spec.history_len < weekly_period:
            raise ValueError("weekly_seasonal_naive uses a seven-day input history")
        cycle = history[:, -weekly_period:]
        repeats = (spec.horizon + weekly_period - 1) // weekly_period
        prediction = cycle.repeat(1, repeats)[:, : spec.horizon]
    elif method == "dlinear":
        if model is None:
            raise ValueError("dlinear requires a trained model")
        prediction = model(history)
    elif method in {"lstm_issued", "patchtst_issued"}:
        if model is None:
            raise ValueError(f"{method} requires a trained model")
        prediction = issued_target_from_batch(batch, spec) + model(history)
    elif method in {"timexer_issued", "tft_issued"}:
        if model is None:
            raise ValueError(f"{method} requires a trained model")
        prediction = issued_target_from_batch(batch, spec) + model(
            batch["tokens"], batch["numeric_context"]
        )
    elif method == "issued_bias":
        if issued_bias is None:
            raise ValueError("issued_bias requires a fitted calibration vector")
        prediction = issued_target_from_batch(batch, spec) + issued_bias.unsqueeze(0)
    elif method == "issued_residual_linear":
        if model is None:
            raise ValueError("issued_residual_linear requires a trained model")
        prediction = model(history, issued_target_from_batch(batch, spec))
    else:
        raise ValueError(f"Unknown method {method}")
    if spec.key == "net_load" or (spec.cadence_minutes == 5 and spec.key == "load"):
        return prediction
    prediction = torch.clamp_min(prediction, 0.0)
    if spec.key in {"wind", "pv"} and spec.cadence_minutes != 5:
        prediction = torch.clamp_max(prediction, 1.0)
    return prediction


def evaluate(
    *,
    method: str,
    loader: DataLoader,
    spec: TaskSpec,
    model: nn.Module | None,
    issued_bias: torch.Tensor | None,
    scale: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if model is not None:
        model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    forecast_starts: list[str] = []
    node_ids: list[str] = []
    losses: list[float] = []
    issued_predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            prediction_norm = baseline_prediction(
                method,
                batch,
                spec=spec,
                model=model,
                issued_bias=issued_bias,
            )
            prediction_raw = normalized_to_raw(prediction_norm, batch)
            losses.append(
                float(F.huber_loss((prediction_raw - batch["target_raw"]) / scale, torch.zeros_like(prediction_raw)).cpu())
            )
            predictions.append(prediction_raw.cpu().numpy())
            targets.append(batch["target_raw"].cpu().numpy())
            histories.append(batch["history_raw"].cpu().numpy())
            forecast_starts.extend("" if value is None else str(value) for value in raw_batch["forecast_start"])
            node_ids.extend(str(value) for value in raw_batch["node_id"])
            if method.startswith("issued_"):
                issued_predictions.append(
                    normalized_to_raw(issued_target_from_batch(batch, spec), batch).cpu().numpy()
                )

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    history = np.concatenate(histories, axis=0)
    metrics = point_metrics(prediction, target)
    daily_period = int(24 * 60 / spec.cadence_minutes)
    metrics["mase"] = mase(prediction, target, insample=history.reshape(-1), seasonality=daily_period)
    metrics["loss"] = float(np.mean(losses))
    metrics["examples"] = int(prediction.shape[0])
    artifact = {
        "prediction_raw": prediction.astype(np.float32),
        "target_raw": target.astype(np.float32),
        "history_raw": history.astype(np.float32),
        "forecast_start": np.asarray(forecast_starts, dtype=str),
        "node_id": np.asarray(node_ids, dtype=str),
        "task": np.asarray(spec.key),
        "method": np.asarray(method),
    }
    if issued_predictions:
        artifact["issued_prediction_raw"] = np.concatenate(issued_predictions, axis=0).astype(np.float32)
    return metrics, artifact


def fit_issued_bias(
    *,
    loader: DataLoader,
    spec: TaskSpec,
    device: torch.device,
) -> torch.Tensor:
    """Fit one lead-wise operational bias vector using train origins only."""

    residuals: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            residuals.append(
                normalized_target_from_batch(batch) - issued_target_from_batch(batch, spec)
            )
    if not residuals:
        raise ValueError(f"No train examples are available for {spec.key}")
    return torch.cat(residuals, dim=0).mean(dim=0)


def train_direct_model(
    *,
    bundle: TaskDataBundle,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    epochs: int,
    steps: int,
    patience: int,
    output_dir: Path,
    device: torch.device,
    method: str,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    spec = bundle.spec
    set_seed(args.seed + spec.task_id, strict=args.strict_determinism)
    if method == "dlinear":
        model: nn.Module = DirectLinear(spec.history_len, spec.horizon).to(device)
    elif method == "lstm_issued":
        model = IssuedResidualLSTM(spec.horizon).to(device)
    elif method == "patchtst_issued":
        day = 24 * 60 // spec.cadence_minutes
        model = IssuedResidualPatchTST(
            spec.history_len,
            spec.horizon,
            patch=day,
            stride=max(1, day // 2),
        ).to(device)
    elif method == "timexer_issued":
        day = 24 * 60 // spec.cadence_minutes
        model = IssuedResidualTimeXer(
            spec.history_len,
            spec.horizon,
            patch=day,
            stride=max(1, day // 2),
        ).to(device)
    elif method == "tft_issued":
        model = IssuedResidualTFT(spec.history_len).to(device)
    elif method == "issued_residual_linear":
        model = IssuedResidualLinear(spec.history_len, spec.horizon).to(device)
    else:
        raise ValueError(f"Unsupported direct model {method}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scale = task_scale(bundle)
    iterator = iter(loaders["train"])
    best_score = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint = output_dir / f"best_{method}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for _ in range(steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loaders["train"])
                raw_batch = next(iterator)
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction_norm = baseline_prediction(
                method,
                batch,
                spec=spec,
                model=model,
                issued_bias=None,
            )
            prediction_raw = normalized_to_raw(prediction_norm, batch)
            loss = F.huber_loss(
                (prediction_raw - batch["target_raw"]) / scale,
                torch.zeros_like(prediction_raw),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_metrics, _ = evaluate(
            method=method,
            loader=loaders["val"],
            spec=spec,
            model=model,
            issued_bias=None,
            scale=scale,
            device=device,
        )
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_nmae": val_metrics["nmae"]}
        history.append(row)
        print(json.dumps({"task": spec.key, **row}), flush=True)
        if val_metrics["nmae"] < best_score:
            best_score = val_metrics["nmae"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"state_dict": model.state_dict(), "spec": spec.to_dict(), "best_epoch": epoch}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model, history, best_epoch


def apply_region_holdout(
    bundle: TaskDataBundle,
    *,
    region: str,
    model_config: ModelConfig,
    future_exog_available: bool,
) -> TaskDataBundle:
    datasets = {}
    for split, dataset in bundle.datasets.items():
        examples = (
            [example for example in dataset.examples if str(example.node_id) == region]
            if split == "test"
            else [example for example in dataset.examples if str(example.node_id) != region]
        )
        if not examples:
            raise ValueError(f"Region holdout {region} leaves {split} empty for {bundle.spec.key}")
        datasets[split] = PaperEnergyDataset(
            examples,
            spec=bundle.spec,
            model_config=model_config,
            require_llm=False,
            future_exog_available=future_exog_available,
        )
    manifest = dict(bundle.manifest)
    manifest["region_holdout"] = {
        "region": region,
        "train_and_validation": "all other NEM regions",
        "test": region,
        "examples": {split: len(dataset) for split, dataset in datasets.items()},
    }
    return TaskDataBundle(spec=bundle.spec, datasets=datasets, manifest=manifest)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if any(method.startswith("issued_") for method in args.methods) and args.future_exog_policy not in {
        "issued",
        "available",
    }:
        raise ValueError(
            "Issued residual baselines require --future_exog_policy issued or available"
        )
    if args.data_root:
        if args.protocol == "aemo":
            os.environ["AEMO_UNIFIED_DATA_DIR"] = str(Path(args.data_root).expanduser().resolve())
        elif args.protocol == "perform":
            os.environ["PERFORM_ERCOT_DATA_DIR"] = str(Path(args.data_root).expanduser().resolve())
    epochs, steps, patience = resolve_profile(args)
    spec_factory = {
        "original": paper_task_specs,
        "aemo": aemo_task_specs,
        "perform": perform_task_specs,
    }[args.protocol]
    specs = spec_factory(
        history_days=args.history_days,
        horizon_days=args.horizon_days,
        forecast_stride_days=args.forecast_stride_days,
    )
    model_config = ModelConfig()
    bundles = {
        task: load_task_bundle(
            specs[task],
            model_config=model_config,
            llm_cache=None,
            require_llm=False,
            future_exog_policy=args.future_exog_policy,
            split_embargo_steps=(24 * 60 // specs[task].cadence_minutes) * args.split_embargo_days,
        )
        for task in args.tasks
    }
    if args.holdout_region:
        if args.protocol != "aemo":
            raise ValueError("--holdout_region is only defined for the AEMO protocol")
        bundles = {
            task: apply_region_holdout(
                bundle,
                region=args.holdout_region,
                model_config=model_config,
                future_exog_available=args.future_exog_policy in {"available", "issued", "oracle"},
            )
            for task, bundle in bundles.items()
        }
    protocol = f"L{args.history_days}d_H{args.horizon_days}d_S{args.forecast_stride_days}d"
    tag_parts = [
        part
        for part in (
            args.run_tag,
            f"holdout-{args.holdout_region}" if args.holdout_region else "",
        )
        if part
    ]
    tag = "_" + "_".join(tag_parts) if tag_parts else ""
    output_dir = Path(args.output_root) / f"{args.profile}_{args.protocol}_{protocol}{tag}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("This baseline protocol requires CUDA training")
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    manifests = {task: bundle.manifest for task, bundle in bundles.items()}

    for task, bundle in bundles.items():
        loaders = make_loaders(bundle, args)
        scale = task_scale(bundle)
        task_summary: dict[str, Any] = {}
        trained_models: dict[str, nn.Module] = {}
        train_histories: dict[str, list[dict[str, float]]] = {}
        best_epochs: dict[str, int] = {}
        for method in (
            "dlinear",
            "lstm_issued",
            "patchtst_issued",
            "timexer_issued",
            "tft_issued",
            "issued_residual_linear",
        ):
            if method not in args.methods:
                continue
            method_dir = output_dir / method / task
            method_dir.mkdir(parents=True, exist_ok=True)
            trained_models[method], train_histories[method], best_epochs[method] = train_direct_model(
                bundle=bundle,
                loaders=loaders,
                args=args,
                epochs=epochs,
                steps=steps,
                patience=patience,
                output_dir=method_dir,
                device=device,
                method=method,
            )
        issued_bias = (
            fit_issued_bias(loader=loaders["train"], spec=bundle.spec, device=device)
            if "issued_bias" in args.methods
            else None
        )
        for method in args.methods:
            model = trained_models.get(method)
            metrics, artifact = evaluate(
                method=method,
                loader=loaders["test"],
                spec=bundle.spec,
                model=model,
                issued_bias=issued_bias,
                scale=scale,
                device=device,
            )
            task_summary[method] = {
                "test": metrics,
                "best_epoch": best_epochs.get(method),
                "history": train_histories.get(method, []),
                "information_scope": (
                    "target history and target-specific issued trajectory"
                    if method in {
                        "issued_official",
                        "issued_bias",
                        "issued_residual_linear",
                        "lstm_issued",
                        "patchtst_issued",
                        "timexer_issued",
                        "tft_issued",
                    }
                    else "target history only"
                ),
            }
            metrics_rows.append({"method": method, "task": task, **metrics})
            artifact_dir = output_dir / "artifacts" / method / task
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact_dir / "full.npz", **artifact)
            for row in lead_time_metric_rows(
                artifact,
                task=task,
                intervention="full",
                cadence_minutes=bundle.spec.cadence_minutes,
            ):
                lead_rows.append({"method": method, **row})
        summaries[task] = task_summary

    write_csv(output_dir / "test_metrics.csv", metrics_rows)
    write_csv(output_dir / "lead_time_metrics.csv", lead_rows)
    summary = {
        "run_name": output_dir.name,
        "seed": args.seed,
        "profile": args.profile,
        "protocol": args.protocol,
        "methods": args.methods,
        "tasks": args.tasks,
        "history_days": args.history_days,
        "horizon_days": args.horizon_days,
        "forecast_stride_days": args.forecast_stride_days,
        "future_exog_policy": args.future_exog_policy,
        "split_embargo_days": args.split_embargo_days,
        "calibration_fraction": 0.0,
        "training_config": {
            "epochs": epochs,
            "steps_per_epoch": steps,
            "patience": patience,
            "learning_rate": args.learning_rate,
            "batch_sizes": {task: batch_size(args, task) for task in args.tasks},
            "strict_determinism": args.strict_determinism,
        },
        "holdout_region": args.holdout_region,
        "device": str(device),
        "runtime": {
            "train_and_evaluation_seconds": float(time.perf_counter() - started_at),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
                if device.type == "cuda"
                else 0.0
            ),
        },
        "results": summaries,
        "manifests": manifests,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_name": output_dir.name, "metrics": len(metrics_rows), "lead_rows": len(lead_rows)}))


if __name__ == "__main__":
    main()
