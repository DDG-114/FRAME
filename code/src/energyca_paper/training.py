"""Task-balanced joint training and auditable evaluation utilities."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .metrics import mase, point_metrics, probabilistic_metrics

from .config import ModelConfig, TaskSpec
from .data import (
    PaperEnergyDataset,
    TaskDataBundle,
    collate_task_batch,
    intervene_batch,
    move_batch_to_device,
)
from .model import ForwardControls, SharedEnergyCALLM


LEAD_TIME_BINS_HOURS = (
    (0.0, 24.0, "0-24h"),
    (24.0, 48.0, "24-48h"),
    (48.0, 72.0, "48-72h"),
    (72.0, 168.0, "72-168h"),
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 20
    steps_per_epoch: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    lambda_residual: float = 1e-4
    lambda_quantile: float = 1.0
    loss_kind: str = "huber"
    patience: int = 5
    use_bf16: bool = True
    num_workers: int = 0
    batch_size_load: int = 16
    batch_size_wind: int = 2
    batch_size_pv: int = 16
    batch_size_net_load: int = 16
    eval_batch_multiplier: int = 8
    batch_by_horizon: bool = True
    strict_determinism: bool = False
    semantic_warmup_epochs: int = 2
    semantic_ramp_epochs: int = 0
    semantic_learning_rate_scale: float = 0.25
    initial_checkpoint: str | None = None
    semantic_only: bool = False
    operating_state_weight: float = 0.0
    direct_residual_loss_weight: float = 0.0

    def batch_size(self, task: str) -> int:
        base_task = task.split("_h", 1)[0]
        return int(getattr(self, f"batch_size_{base_task}"))

    def eval_batch_size(self, task: str) -> int:
        return self.batch_size(task) * int(self.eval_batch_multiplier)

    def __post_init__(self) -> None:
        if self.loss_kind not in {"huber", "l1"}:
            raise ValueError("loss_kind must be 'huber' or 'l1'")
        if self.eval_batch_multiplier < 1:
            raise ValueError("eval_batch_multiplier must be at least one")
        if self.semantic_warmup_epochs < 0:
            raise ValueError("semantic_warmup_epochs must be non-negative")
        if self.semantic_ramp_epochs < 0:
            raise ValueError("semantic_ramp_epochs must be non-negative")
        if not 0.0 < self.semantic_learning_rate_scale <= 1.0:
            raise ValueError("semantic_learning_rate_scale must lie in (0, 1]")
        if not 0.0 <= self.operating_state_weight <= 2.0:
            raise ValueError("operating_state_weight must lie in [0, 2]")
        if self.direct_residual_loss_weight < 0.0:
            raise ValueError("direct_residual_loss_weight must be non-negative")


def set_seed(seed: int, *, strict_determinism: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)


def _semantic_scale_for_epoch(
    epoch: int,
    *,
    warmup_epochs: int,
    ramp_epochs: int,
    target_scale: float,
) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(target_scale)
    progress = min((epoch - warmup_epochs + 1) / ramp_epochs, 1.0)
    return float(target_scale) * progress


def _cycle(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "tokens": batch["tokens"],
        "numeric_context": batch["numeric_context"],
        "schema_context": batch["schema_context"],
        "llm_context": batch["llm_context"],
        "availability": batch["availability"],
        "history_len": batch["history_len"],
        "horizon": batch["horizon"],
        "task_id": batch["task_id"],
        "cadence_id": batch["cadence_id"],
    }


def _merge_same_horizon_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge task batches whose temporal dimensions are identical.

    The shared model accepts a task identifier per sample, so load, wind, PV,
    and net-load examples for the same forecast horizon can use one forward
    pass.  Task losses remain separate after the pass, preserving equal task
    weighting in the joint objective.
    """

    if not batches:
        raise ValueError("At least one batch is required")
    horizon = batches[0]["horizon"][0]
    if any(torch.any(batch["horizon"] != horizon) for batch in batches[1:]):
        raise ValueError("Only batches with the same forecast horizon may be merged")
    merged: dict[str, Any] = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged[key] = [item for value in values for item in value]
        else:
            raise TypeError(f"Unsupported batch field for horizon merge: {key}")
    return merged


def _training_task_groups(
    iterators: dict[str, Iterator[dict[str, Any]]],
    *,
    batch_by_horizon: bool,
) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
    """Draw one batch per task and group compatible batches by horizon."""

    pending = [(task, next(iterator)) for task, iterator in iterators.items()]
    if not batch_by_horizon:
        return [((task,), [batch]) for task, batch in pending]
    grouped: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for task, batch in pending:
        horizon_values = batch["horizon"]
        if horizon_values.ndim != 1 or not torch.all(horizon_values == horizon_values[0]):
            raise ValueError(f"Task batch {task} has inconsistent forecast horizons")
        grouped.setdefault(int(horizon_values[0]), []).append((task, batch))
    return [
        (tuple(task for task, _ in items), [batch for _, batch in items])
        for _, items in sorted(grouped.items())
    ]


def compute_training_scales(bundles: dict[str, TaskDataBundle]) -> dict[str, float]:
    """Compute task scales from training targets only."""

    scales: dict[str, float] = {}
    for task, bundle in bundles.items():
        values = np.concatenate(
            [np.asarray(example.target_raw, dtype=np.float64).reshape(-1) for example in bundle.datasets["train"].examples]
        )
        scale = float(np.mean(np.abs(values)))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.std(values))
        scales[task] = max(scale, 1e-6)
    return scales


def scaled_huber_loss(
    prediction_normalized: torch.Tensor,
    batch: dict[str, Any],
    *,
    task_scale: float,
    loss_kind: str = "huber",
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction_raw = prediction_normalized * batch["target_scale"] + batch["target_offset"]
    residual = (prediction_raw - batch["target_raw"]) / float(task_scale)
    if loss_kind == "l1":
        point_loss = residual.abs()
    else:
        point_loss = F.huber_loss(
            residual,
            torch.zeros_like(residual),
            delta=1.0,
            reduction="none",
        )
    sample_loss = point_loss.flatten(1).mean(dim=1)
    if sample_weights is None:
        return sample_loss.mean()
    weights = sample_weights.to(device=sample_loss.device, dtype=sample_loss.dtype)
    return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def direct_residual_supervision_loss(
    predicted_residual: torch.Tensor,
    base_prediction: torch.Tensor,
    batch: dict[str, Any],
    *,
    loss_kind: str,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    target_normalized = (
        batch["target_raw"] - batch["target_offset"]
    ) / batch["target_scale"].clamp_min(1e-6)
    target_residual = target_normalized - base_prediction.detach()
    if loss_kind == "l1":
        point_loss = (predicted_residual - target_residual).abs()
    else:
        point_loss = F.huber_loss(
            predicted_residual,
            target_residual,
            delta=1.0,
            reduction="none",
        )
    sample_loss = point_loss.flatten(1).mean(dim=1)
    if sample_weights is None:
        return sample_loss.mean()
    weights = sample_weights.to(device=sample_loss.device, dtype=sample_loss.dtype)
    return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def operating_state_sample_weights(
    prediction_normalized: torch.Tensor,
    batch: dict[str, Any],
    *,
    task_scale: float,
    strength: float,
) -> torch.Tensor:
    """Emphasize hard forecasts and volatile forecast-origin histories."""

    prediction_raw = prediction_normalized * batch["target_scale"] + batch["target_offset"]
    error = (prediction_raw.detach() - batch["target_raw"]).abs().flatten(1).mean(dim=1)
    history = batch["history_raw"].detach().flatten(1)
    ramp = history.diff(dim=1).abs().mean(dim=1)
    volatility = history.std(dim=1, unbiased=False)
    signals = torch.stack((error, ramp, volatility), dim=1) / float(task_scale)
    order = signals.argsort(dim=0).argsort(dim=0).to(dtype=signals.dtype)
    ranks = order / max(signals.shape[0] - 1, 1)
    score = ranks.mean(dim=1)
    weights = (1.0 + float(strength) * (score - score.mean())).clamp_min(0.25)
    return weights / weights.mean().clamp_min(1e-8)


def scaled_quantile_pinball_loss(
    quantiles_normalized: torch.Tensor,
    batch: dict[str, Any],
    *,
    levels: torch.Tensor,
    task_scale: float,
) -> torch.Tensor:
    """Pinball loss in the physical target scale, normalized per task."""

    quantiles_raw = (
        quantiles_normalized * batch["target_scale"].unsqueeze(2)
        + batch["target_offset"].unsqueeze(2)
    )
    residual = (batch["target_raw"].unsqueeze(-1) - quantiles_raw) / float(task_scale)
    tau = levels.to(device=residual.device, dtype=residual.dtype).view(1, 1, -1)
    return torch.maximum(tau * residual, (tau - 1.0) * residual).mean()


def make_loaders(
    bundles: dict[str, TaskDataBundle],
    config: TrainingConfig,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader], dict[str, DataLoader]]:
    train: dict[str, DataLoader] = {}
    val: dict[str, DataLoader] = {}
    test: dict[str, DataLoader] = {}
    for task, bundle in bundles.items():
        generator = torch.Generator().manual_seed(config.seed + bundle.spec.task_id)
        common_kwargs = dict(
            collate_fn=collate_task_batch,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_kwargs = dict(
            batch_size=config.batch_size(task),
            **common_kwargs,
        )
        evaluation_kwargs = dict(
            batch_size=config.eval_batch_size(task),
            **common_kwargs,
        )
        train[task] = DataLoader(
            bundle.datasets["train"],
            shuffle=True,
            generator=generator,
            drop_last=False,
            **train_kwargs,
        )
        val[task] = DataLoader(bundle.datasets["val"], shuffle=False, **evaluation_kwargs)
        test[task] = DataLoader(bundle.datasets["test"], shuffle=False, **evaluation_kwargs)
    return train, val, test


def evaluate_loader(
    model: SharedEnergyCALLM,
    loader: DataLoader,
    *,
    task: str,
    task_scale: float,
    controls: ForwardControls,
    intervention: str,
    device: torch.device,
    use_bf16: bool,
    shuffle_seed: int = 2027,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    semantic_states: list[np.ndarray] = []
    semantic_gates: list[np.ndarray] = []
    residual_codes: list[np.ndarray] = []
    seasonal_gates: list[np.ndarray] = []
    issued_anchor_gates: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    quantile_levels: np.ndarray | None = None
    forecast_starts: list[str] = []
    node_ids: list[str] = []
    context_texts: list[str] = []
    losses: list[float] = []
    generator = torch.Generator(device=device.type).manual_seed(shuffle_seed)

    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            batch = intervene_batch(batch, intervention, generator=generator)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                output = model(**_model_inputs(batch), controls=controls)
                loss = scaled_huber_loss(output["prediction"], batch, task_scale=task_scale)
            prediction_raw = (
                output["prediction"].float() * batch["target_scale"]
                + batch["target_offset"]
            )
            predictions.append(prediction_raw.cpu().numpy())
            targets.append(batch["target_raw"].cpu().numpy())
            histories.append(batch["history_raw"].cpu().numpy())
            alphas.append(output["alpha"].float().cpu().numpy())
            gammas.append(output["gamma"].float().cpu().numpy())
            activations.append(output["activation"].float().cpu().numpy())
            semantic_states.append(output["semantic_state"].float().cpu().numpy())
            semantic_gates.append(output["semantic_gate"].float().cpu().numpy())
            residual_codes.append(output["residual_code"].float().cpu().numpy())
            seasonal_gates.append(output["seasonal_gate"].float().cpu().numpy())
            issued_anchor_gates.append(output["issued_anchor_gate"].float().cpu().numpy())
            if "quantiles" in output:
                raw_quantiles = (
                    output["quantiles"].float() * batch["target_scale"].unsqueeze(-1)
                    + batch["target_offset"].unsqueeze(-1)
                )
                quantiles.append(raw_quantiles.cpu().numpy())
                quantile_levels = output["quantile_levels"].float().cpu().numpy()
            forecast_starts.extend("" if value is None else str(value) for value in raw_batch["forecast_start"])
            node_ids.extend(str(value) for value in raw_batch["node_id"])
            context_texts.extend(str(value) for value in raw_batch["context_text"])
            losses.append(float(loss.cpu()))

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    history = np.concatenate(histories, axis=0)
    alpha_array = np.concatenate(alphas, axis=0)
    gamma_array = np.concatenate(gammas, axis=0)
    activation_array = np.concatenate(activations, axis=0)
    semantic_state_array = np.concatenate(semantic_states, axis=0)
    semantic_gate_array = np.concatenate(semantic_gates, axis=0)
    residual_code_array = np.concatenate(residual_codes, axis=0)
    metrics = point_metrics(prediction, target)
    cadence_minutes = int(loader.dataset.spec.cadence_minutes)
    daily_period = int(24 * 60 / cadence_minutes)
    metrics["mase"] = mase(
        prediction,
        target,
        insample=history.reshape(-1),
        seasonality=daily_period,
    )
    metrics["loss"] = float(np.mean(losses))
    metrics["examples"] = int(prediction.shape[0])
    metrics["mean_adapter_alpha"] = float(np.mean(alpha_array))
    metrics["mean_abs_rank_gate"] = float(np.mean(np.abs(gamma_array)))
    metrics["effective_rank_l1"] = float(np.mean(np.sum(np.abs(gamma_array), axis=-1)))
    metrics["active_rank_0p1"] = float(
        np.mean(np.sum(np.abs(gamma_array) >= 0.1, axis=-1))
    )
    metrics["active_rank_0p25"] = float(
        np.mean(np.sum(np.abs(gamma_array) >= 0.25, axis=-1))
    )
    metrics["mean_adapter_activation"] = float(np.mean(activation_array))
    metrics["mean_semantic_state_l2"] = float(
        np.mean(np.linalg.norm(semantic_state_array, axis=-1))
    )
    metrics["mean_semantic_gate"] = float(np.mean(semantic_gate_array))
    metrics["mean_residual_code_l2"] = float(
        np.mean(np.linalg.norm(residual_code_array, axis=-1))
    )
    operational_context = np.asarray(
        [
            "AEMO operational notice" in text
            or "AEMO ELECTRICITY MARKET NOTICE" in text
            for text in context_texts
        ],
        dtype=bool,
    )
    if operational_context.any():
        notice_present = np.asarray(
            ["No relevant AEMO operational notice" not in text for text in context_texts],
            dtype=bool,
        )
        metrics["notice_present_examples"] = int(notice_present.sum())
        metrics["notice_absent_examples"] = int((~notice_present).sum())
        if notice_present.any():
            metrics["notice_present_nmae"] = point_metrics(
                prediction[notice_present], target[notice_present]
            )["nmae"]
        if (~notice_present).any():
            metrics["notice_absent_nmae"] = point_metrics(
                prediction[~notice_present], target[~notice_present]
            )["nmae"]
    if quantiles:
        metrics.update(probabilistic_metrics(np.concatenate(quantiles, axis=0), target, levels=quantile_levels))
    per_window_mae = np.mean(np.abs(prediction - target), axis=1)
    per_window_nmae = per_window_mae / (np.mean(np.abs(target), axis=1) + 1e-6)
    artifact = {
        "prediction_raw": prediction.astype(np.float32),
        "target_raw": target.astype(np.float32),
        "history_raw": history.astype(np.float32),
        "alpha": alpha_array.astype(np.float32),
        "gamma": gamma_array.astype(np.float32),
        "activation": activation_array.astype(np.float32),
        "semantic_state": semantic_state_array.astype(np.float32),
        "semantic_gate": semantic_gate_array.astype(np.float32),
        "residual_code": residual_code_array.astype(np.float32),
        "seasonal_gate": np.concatenate(seasonal_gates, axis=0).astype(np.float32),
        "issued_anchor_gate": np.concatenate(issued_anchor_gates, axis=0).astype(np.float32),
        "window_nmae": per_window_nmae.astype(np.float32),
        "forecast_start": np.asarray(forecast_starts, dtype=str),
        "node_id": np.asarray(node_ids, dtype=str),
        "notice_present": (
            notice_present.astype(np.uint8)
            if operational_context.any()
            else np.zeros(prediction.shape[0], dtype=np.uint8)
        ),
        "task": np.asarray(task),
        "intervention": np.asarray(intervention),
    }
    if quantiles:
        artifact["quantiles_raw"] = np.concatenate(quantiles, axis=0).astype(np.float32)
        artifact["quantile_levels"] = np.asarray(quantile_levels, dtype=np.float32)
    return metrics, artifact


def lead_time_metric_rows(
    artifact: dict[str, np.ndarray],
    *,
    task: str,
    intervention: str,
    cadence_minutes: int,
) -> list[dict[str, Any]]:
    """Summarize point errors over physical lead-time bins for long horizons."""

    prediction = artifact["prediction_raw"]
    target = artifact["target_raw"]
    lead_hours = (np.arange(prediction.shape[1], dtype=np.float64) + 1.0) * cadence_minutes / 60.0
    rows: list[dict[str, Any]] = []
    for lower, upper, label in LEAD_TIME_BINS_HOURS:
        mask = (lead_hours > lower) & (lead_hours <= upper)
        if not np.any(mask):
            continue
        metrics = point_metrics(prediction[:, mask], target[:, mask])
        if "quantiles_raw" in artifact:
            metrics.update(
                probabilistic_metrics(
                    artifact["quantiles_raw"][:, mask],
                    target[:, mask],
                    levels=artifact["quantile_levels"],
                )
            )
        rows.append(
            {
                "intervention": intervention,
                "task": task,
                "lead_bin": label,
                "lead_start_hours": lower,
                "lead_end_hours": upper,
                "lead_steps": int(mask.sum()),
                "examples": int(prediction.shape[0]),
                **metrics,
            }
        )
    return rows


def evaluate_tasks(
    model: SharedEnergyCALLM,
    loaders: dict[str, DataLoader],
    *,
    task_scales: dict[str, float],
    controls: ForwardControls,
    intervention: str,
    device: torch.device,
    use_bf16: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    metrics: dict[str, dict[str, float]] = {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for task, loader in loaders.items():
        metrics[task], artifacts[task] = evaluate_loader(
            model,
            loader,
            task=task,
            task_scale=task_scales[task],
            controls=controls,
            intervention=intervention,
            device=device,
            use_bf16=use_bf16,
        )
    return metrics, artifacts


def mean_task_nmae(metrics: dict[str, dict[str, float]]) -> float:
    return float(np.mean([task_metrics["nmae"] for task_metrics in metrics.values()]))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_joint_model(
    *,
    bundles: dict[str, TaskDataBundle],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    controls: ForwardControls,
    output_dir: str | Path,
    run_name: str,
    validation_interventions: tuple[str, ...] = ("full",),
    evaluate_test: bool = True,
    test_interventions: tuple[str, ...] = (
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
    ),
) -> dict[str, Any]:
    """Train one shared checkpoint with equal task contribution per update."""

    set_seed(
        training_config.seed,
        strict_determinism=training_config.strict_determinism,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = SharedEnergyCALLM(model_config).to(device)
    if training_config.initial_checkpoint:
        initial_payload = torch.load(
            training_config.initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if model_config.use_direct_semantic_residual:
            incompatible = model.load_state_dict(initial_payload["model_state"], strict=False)
            allowed_missing = {
                name
                for name in model.state_dict()
                if name.startswith("direct_semantic_conditioner.")
                or name.startswith("direct_semantic_decoder.")
                or name.startswith("direct_semantic_field_gate.")
                or name.startswith("direct_numeric_projection.")
            }
            if not set(incompatible.missing_keys).issubset(allowed_missing) or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Initial checkpoint is incompatible with the direct semantic residual model: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            model.load_state_dict(initial_payload["model_state"], strict=True)
    semantic_parameters = list(model.context_encoder.semantic_residual_parameters())
    semantic_parameters.extend(model.direct_semantic_parameters())
    semantic_parameter_ids = {id(parameter) for parameter in semantic_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in semantic_parameter_ids
    ]
    if training_config.semantic_only:
        for parameter in base_parameters:
            parameter.requires_grad_(False)
        optimizer_groups = [
            {
                "params": semantic_parameters,
                "lr": training_config.learning_rate * training_config.semantic_learning_rate_scale,
            }
        ]
    else:
        optimizer_groups = [
            {"params": base_parameters, "lr": training_config.learning_rate},
            {
                "params": semantic_parameters,
                "lr": training_config.learning_rate * training_config.semantic_learning_rate_scale,
            },
        ]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=training_config.weight_decay)
    train_loaders, val_loaders, test_loaders = make_loaders(bundles, training_config)
    iterators = {task: _cycle(loader) for task, loader in train_loaders.items()}
    task_scales = compute_training_scales(bundles)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = -1
    stale_epochs = 0
    checkpoint_path = output_dir / "best_checkpoint.pt"
    initial_validation_nmae: float | None = None

    def save_checkpoint() -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model_config.to_dict(),
                "training_config": asdict(training_config),
                "controls": asdict(controls),
                "task_specs": {task: bundle.spec.to_dict() for task, bundle in bundles.items()},
                "task_scales": task_scales,
                "best_epoch": best_epoch,
                "best_mean_val_nmae": best_score,
                "run_name": run_name,
            },
            checkpoint_path,
        )

    if training_config.initial_checkpoint:
        initial_metrics, _ = evaluate_tasks(
            model,
            val_loaders,
            task_scales=task_scales,
            controls=controls,
            intervention="full",
            device=device,
            use_bf16=training_config.use_bf16,
        )
        initial_validation_nmae = mean_task_nmae(initial_metrics)
        best_score = initial_validation_nmae
        best_epoch = 0
        save_checkpoint()

    semantic_warmup_epochs = min(training_config.semantic_warmup_epochs, training_config.epochs)
    for epoch in range(training_config.epochs):
        epoch_controls = replace(
            controls,
            semantic_residual_scale=_semantic_scale_for_epoch(
                epoch,
                warmup_epochs=semantic_warmup_epochs,
                ramp_epochs=training_config.semantic_ramp_epochs,
                target_scale=controls.semantic_residual_scale,
            ),
        )
        if training_config.semantic_only:
            model.eval()
            model.context_encoder.train()
            if model_config.use_direct_semantic_residual:
                model.direct_semantic_conditioner.train()
                model.direct_semantic_decoder.train()
                if model_config.direct_residual_source == "numeric":
                    model.direct_numeric_projection.train()
                if model_config.use_field_conditioned_semantic:
                    model.direct_semantic_field_gate.train()
        else:
            model.train()
        epoch_losses: list[float] = []
        for _ in range(training_config.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            task_losses: list[float] = []
            for group_tasks, group_batches in _training_task_groups(
                iterators,
                batch_by_horizon=training_config.batch_by_horizon,
            ):
                device_batches = [move_batch_to_device(batch, device) for batch in group_batches]
                batch = _merge_same_horizon_batches(device_batches)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=training_config.use_bf16 and device.type == "cuda",
                ):
                    output = model(**_model_inputs(batch), controls=epoch_controls)
                    task_offset = 0
                    loss = torch.zeros((), dtype=output["prediction"].dtype, device=device)
                    for task, task_batch in zip(group_tasks, device_batches):
                        task_count = task_batch["target_raw"].shape[0]
                        task_slice = slice(task_offset, task_offset + task_count)
                        task_offset += task_count
                        sample_weights = None
                        if training_config.operating_state_weight > 0.0:
                            sample_weights = operating_state_sample_weights(
                                output["prediction"][task_slice],
                                task_batch,
                                task_scale=task_scales[task],
                                strength=training_config.operating_state_weight,
                            )
                        prediction_loss = scaled_huber_loss(
                            output["prediction"][task_slice],
                            task_batch,
                            task_scale=task_scales[task],
                            loss_kind=training_config.loss_kind,
                            sample_weights=sample_weights,
                        )
                        quantile_loss = torch.zeros_like(prediction_loss)
                        if "quantiles" in output:
                            quantile_loss = scaled_quantile_pinball_loss(
                                output["quantiles"][task_slice],
                                task_batch,
                                levels=output["quantile_levels"],
                                task_scale=task_scales[task],
                            )
                        residual_supervision = torch.zeros_like(prediction_loss)
                        if training_config.direct_residual_loss_weight > 0.0:
                            residual_supervision = direct_residual_supervision_loss(
                                output["direct_semantic_residual"][task_slice],
                                output["base_raw_prediction"][task_slice],
                                task_batch,
                                loss_kind=training_config.loss_kind,
                                sample_weights=sample_weights,
                            )
                        task_loss = (
                            prediction_loss
                            + training_config.lambda_quantile * quantile_loss
                            + training_config.direct_residual_loss_weight * residual_supervision
                        )
                        loss = loss + task_loss / len(iterators)
                        task_losses.append(float(task_loss.detach()))
                    loss = loss + (
                        training_config.lambda_residual
                        * output["residual_penalty"]
                        * len(group_tasks)
                        / len(iterators)
                    )
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            epoch_losses.append(float(np.mean(task_losses)))

        val_metrics, _ = evaluate_tasks(
            model,
            val_loaders,
            task_scales=task_scales,
            controls=epoch_controls,
            intervention="full",
            device=device,
            use_bf16=training_config.use_bf16,
        )
        score = mean_task_nmae(val_metrics)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "mean_val_nmae": score,
            "semantic_residual_scale": epoch_controls.semantic_residual_scale,
            "val": val_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(_json_ready(epoch_row)), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            stale_epochs = 0
            save_checkpoint()
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_metrics: dict[str, Any] = {}
    for intervention in validation_interventions:
        metrics, _ = evaluate_tasks(
            model,
            val_loaders,
            task_scales=task_scales,
            controls=controls,
            intervention=intervention,
            device=device,
            use_bf16=training_config.use_bf16,
        )
        validation_metrics[intervention] = metrics
    test_metrics: dict[str, Any] = {}
    lead_metric_rows: list[dict[str, Any]] = []
    if evaluate_test:
        for intervention in test_interventions:
            metrics, artifacts = evaluate_tasks(
                model,
                test_loaders,
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

    summary = {
        "run_name": run_name,
        "seed": training_config.seed,
        "single_checkpoint": True,
        "tasks": list(bundles),
        "best_epoch": best_epoch,
        "best_mean_val_nmae": best_score,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_evaluated": evaluate_test,
        "model_config": model_config.to_dict(),
        "training_config": asdict(training_config),
        "controls": asdict(controls),
        "validation_interventions": list(validation_interventions),
        "test_interventions": list(test_interventions),
        "task_scales": task_scales,
        "parameter_summary": model.parameter_summary(),
        "runtime": {
            "train_and_evaluation_seconds": float(time.perf_counter() - started_at),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
                if device.type == "cuda"
                else 0.0
            ),
        },
        "initialization": {
            "checkpoint": training_config.initial_checkpoint,
            "semantic_only": training_config.semantic_only,
            "initial_validation_nmae": initial_validation_nmae,
        },
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    metric_rows: list[dict[str, Any]] = []
    for intervention, task_metrics in test_metrics.items():
        for task, metrics in task_metrics.items():
            metric_rows.append({"intervention": intervention, "task": task, **metrics})
    if metric_rows:
        write_metric_csv(output_dir / "test_metrics.csv", metric_rows)
    if lead_metric_rows:
        write_metric_csv(output_dir / "lead_time_metrics.csv", lead_metric_rows)
    return summary


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> SharedEnergyCALLM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SharedEnergyCALLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model
