"""Paper-protocol data adapters for the four jointly trained tasks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ModelConfig, TaskSpec


AVAILABILITY_FIELDS = (
    "task_semantics",
    "asset_attributes",
    "geographic_attributes",
    "calendar",
    "weather",
    "price_or_system",
    "recent_operating_state",
    "future_exogenous",
)


def _hashed_schema_context(text: str, *, dimension: int) -> np.ndarray:
    """Encode the same low-cardinality text fields without a language model.

    This is the numeric-schema control for the pretrained-encoder comparison.
    It hashes role tags and complete ``field=value`` items from the deterministic
    schema, so it never receives the continuous context vector that the paper
    keeps exclusively in the numerical forecasting stream.
    """

    if dimension < 1:
        raise ValueError("schema context dimension must be positive")
    items = re.findall(r"\[[^\]]+\]|[A-Za-z_]+=[^;.\n]+", str(text))
    vector = np.zeros(dimension, dtype=np.float32)
    for item in items:
        normalized = " ".join(item.strip().lower().split())
        digest = hashlib.blake2b(
            normalized.encode("utf-8", errors="strict"), digest_size=16
        ).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    if items:
        vector /= math.sqrt(len(items))
    return vector


@dataclass
class TaskDataBundle:
    spec: TaskSpec
    datasets: dict[str, "PaperEnergyDataset"]
    manifest: dict[str, Any]


def _split_context(
    context: np.ndarray,
    *,
    numeric_context_dim: int,
    llm_context_dim: int,
    require_llm: bool,
) -> tuple[np.ndarray, np.ndarray]:
    context = np.asarray(context, dtype=np.float32).reshape(-1)
    if context.size < numeric_context_dim:
        raise ValueError(
            f"Context has {context.size} values, expected at least {numeric_context_dim}"
        )
    numeric = context[:numeric_context_dim].copy()
    llm = context[numeric_context_dim:].copy()
    if llm.size == 0:
        if require_llm:
            raise ValueError("A frozen-LLM context embedding is required for this run")
        llm = np.zeros(llm_context_dim, dtype=np.float32)
    if llm.size != llm_context_dim:
        raise ValueError(f"LLM context has {llm.size} values, expected {llm_context_dim}")
    return numeric, llm


def _availability_vector(
    numeric_context: np.ndarray,
    *,
    history_len: int,
    future_exog_available: bool,
    availability_mask_enabled: bool = True,
) -> np.ndarray:
    """Build the explicit field-group availability mask used by the model.

    The vendored context schema has stable field positions: task one-hot at
    0:8, capacity/weather/price at 8:11, and latitude/longitude at 23:25.
    Availability is kept separate from field values so that an observed zero is
    not confused with an unavailable field inside the controller.
    """

    if not availability_mask_enabled:
        # Keep the input shape but remove all window-specific availability
        # information.  Constant-one slots represent an unspecified interface,
        # rather than treating every field as numerically missing.
        return np.ones(len(AVAILABILITY_FIELDS), dtype=np.float32)
    ctx = np.asarray(numeric_context, dtype=np.float32)
    asset_available = float(np.any(np.abs(ctx[[8, 21, 22]]) > 1e-8))
    geography_available = float(np.any(np.abs(ctx[23:25]) > 1e-8))
    return np.asarray(
        [
            1.0,
            asset_available,
            geography_available,
            1.0,
            float(ctx[9] > 0.5),
            float(ctx[10] > 0.5),
            float(history_len > 0),
            float(future_exog_available),
        ],
        dtype=np.float32,
    )


def decode_numeric_tokens(
    numeric: np.ndarray,
    *,
    history_len: int,
    horizon: int,
    history_exog_dim: int,
    future_exog_dim: int,
    max_exog_dim: int,
    availability_mask_enabled: bool = True,
) -> np.ndarray:
    """Convert the legacy flattened vector into shared per-time-step tokens."""

    if history_exog_dim != future_exog_dim:
        raise ValueError("History and future exogenous dimensions must match")
    exog_dim = int(history_exog_dim)
    if exog_dim > max_exog_dim:
        raise ValueError(f"Exogenous dimension {exog_dim} exceeds max_exog_dim={max_exog_dim}")

    values = np.asarray(numeric, dtype=np.float32).reshape(-1)
    cursor = 0
    history_target = values[cursor : cursor + history_len]
    cursor += history_len
    history_time = values[cursor : cursor + 4 * history_len].reshape(history_len, 4)
    cursor += 4 * history_len
    history_exog = values[cursor : cursor + exog_dim * history_len].reshape(
        history_len, exog_dim
    )
    cursor += exog_dim * history_len
    future_time = values[cursor : cursor + 4 * horizon].reshape(horizon, 4)
    cursor += 4 * horizon
    future_exog = values[cursor : cursor + exog_dim * horizon].reshape(horizon, exog_dim)
    cursor += exog_dim * horizon
    if cursor != values.size:
        raise ValueError(f"Numeric layout consumed {cursor} values but input has {values.size}")

    sequence_len = history_len + horizon
    token_dim = 1 + 4 + 2 * max_exog_dim + 2
    tokens = np.zeros((sequence_len, token_dim), dtype=np.float32)
    tokens[:history_len, 0] = history_target
    tokens[:history_len, 1:5] = history_time
    tokens[history_len:, 1:5] = future_time
    if exog_dim:
        tokens[:history_len, 5 : 5 + exog_dim] = history_exog
        tokens[history_len:, 5 : 5 + exog_dim] = future_exog
        mask_start = 5 + max_exog_dim
        tokens[:, mask_start : mask_start + exog_dim] = 1.0
    tokens[:history_len, -2] = 1.0
    tokens[history_len:, -1] = 1.0
    if not availability_mask_enabled:
        mask_start = 5 + max_exog_dim
        tokens[:, mask_start : mask_start + max_exog_dim] = 1.0
        tokens[:, -2:] = 1.0
    return tokens


class PaperEnergyDataset(Dataset):
    """A task-homogeneous view with a common token and context contract."""

    def __init__(
        self,
        examples: Iterable[Any],
        *,
        spec: TaskSpec,
        model_config: ModelConfig,
        require_llm: bool,
        future_exog_available: bool,
        availability_mask_enabled: bool = True,
    ) -> None:
        self.examples = list(examples)
        self.spec = spec
        self.model_config = model_config
        self.require_llm = require_llm
        self.future_exog_available = future_exog_available
        self.availability_mask_enabled = availability_mask_enabled
        self.schema_contexts = [
            _hashed_schema_context(
                str(example.context_text),
                dimension=model_config.schema_context_dim,
            )
            for example in self.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ex = self.examples[index]
        numeric_context, llm_context = _split_context(
            ex.context,
            numeric_context_dim=self.model_config.numeric_context_dim,
            llm_context_dim=self.model_config.llm_context_dim,
            require_llm=self.require_llm,
        )
        tokens = decode_numeric_tokens(
            ex.numeric,
            history_len=int(ex.history_len),
            horizon=int(ex.horizon),
            history_exog_dim=int(ex.history_exog_dim),
            future_exog_dim=int(ex.future_exog_dim),
            max_exog_dim=self.model_config.max_exog_dim,
            availability_mask_enabled=self.availability_mask_enabled,
        )
        availability = _availability_vector(
            numeric_context,
            history_len=int(ex.history_len),
            future_exog_available=self.future_exog_available,
            availability_mask_enabled=self.availability_mask_enabled,
        )
        if hasattr(ex, 'future_exog_mask'):
            mask_start = 5 + self.model_config.max_exog_dim
            tokens[int(ex.history_len):, mask_start:mask_start + int(ex.future_exog_dim)] = ex.future_exog_mask
            availability[-1] = float(np.any(ex.future_exog_mask))
        norm_min = float(ex.norm_min)
        norm_max = float(ex.norm_max)
        return {
            "tokens": torch.from_numpy(tokens),
            "numeric_context": torch.from_numpy(numeric_context),
            "schema_context": torch.from_numpy(self.schema_contexts[index]),
            "llm_context": torch.from_numpy(llm_context),
            "availability": torch.from_numpy(availability),
            "target": torch.from_numpy(np.asarray(ex.target, dtype=np.float32)),
            "target_raw": torch.from_numpy(np.asarray(ex.target_raw, dtype=np.float32)),
            "target_offset": torch.from_numpy(np.asarray(ex.target_offset, dtype=np.float32)),
            "target_scale": torch.from_numpy(np.asarray(ex.target_scale, dtype=np.float32)),
            "history_raw": torch.from_numpy(np.asarray(ex.history_raw, dtype=np.float32)),
            "history_len": int(ex.history_len),
            "horizon": int(ex.horizon),
            "task_id": int(self.spec.task_id),
            "cadence_id": int(self.spec.cadence_id),
            "norm_min": norm_min,
            "norm_max": norm_max,
            "dataset_name": str(ex.dataset_name),
            "node_id": str(ex.node_id),
            "forecast_start": ex.forecast_start,
            "context_text": str(ex.context_text),
        }


def collate_task_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = (
        "tokens",
        "numeric_context",
        "schema_context",
        "llm_context",
        "availability",
        "target",
        "target_raw",
        "target_offset",
        "target_scale",
        "history_raw",
    )
    batch: dict[str, Any] = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    for key in ("history_len", "horizon", "task_id", "cadence_id"):
        batch[key] = torch.tensor([item[key] for item in items], dtype=torch.long)
    for key in ("norm_min", "norm_max"):
        batch[key] = torch.tensor([item[key] for item in items], dtype=torch.float32)
    for key in ("dataset_name", "node_id", "forecast_start", "context_text"):
        batch[key] = [item[key] for item in items]
    return batch


def load_task_bundle(
    spec: TaskSpec,
    *,
    model_config: ModelConfig,
    llm_cache: str | Path | None,
    require_llm: bool = True,
    future_exog_policy: str = "available",
    calibration_fraction: float = 0.0,
    split_embargo_steps: int = 0,
    calibration_embargo_days: int = 0,
    text_schema_variant: str = "current",
    availability_mask_enabled: bool = True,
) -> TaskDataBundle:
    """Load one task using the paper's native cadence and chronological split."""

    from src.energy_context.data import load_energy_datasets

    embedding_mode = "concat" if llm_cache is not None else "numeric"
    if require_llm and llm_cache is None:
        raise ValueError(f"Task {spec.key} requires an LLM cache")
    if future_exog_policy not in {"available", "zero", "common", "issued", "oracle"}:
        raise ValueError(
            "future_exog_policy must be one of 'common', 'issued', 'oracle', "
            "'zero', or 'available'"
        )
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")
    if int(calibration_embargo_days) != calibration_embargo_days or calibration_embargo_days < 0:
        raise ValueError("calibration_embargo_days must be a non-negative whole number")
    datasets, manifest = load_energy_datasets(
        [spec.dataset],
        history_len=spec.history_len,
        horizon=spec.horizon,
        stride=spec.stride,
        max_nodes_per_dataset=spec.max_nodes,
        max_train_windows_per_dataset=spec.train_windows_per_node,
        max_eval_windows_per_dataset=spec.eval_windows_per_node,
        context_target_stats="train",
        context_identity_policy="semantic_only",
        text_schema_variant=text_schema_variant,
        future_exog_policy=future_exog_policy,
        context_embedding_mode=embedding_mode,
        llm_context_cache=str(llm_cache) if llm_cache is not None else None,
        llm_context_missing_policy="error",
        embargo_steps=int(split_embargo_steps),
    )
    wrapped = {
        split: PaperEnergyDataset(
            dataset.examples,
            spec=spec,
            model_config=model_config,
            require_llm=require_llm,
            future_exog_available=future_exog_policy in {"available", "issued", "oracle"},
            availability_mask_enabled=availability_mask_enabled,
        )
        for split, dataset in datasets.items()
    }
    if calibration_fraction > 0.0:
        selection_dataset = wrapped["val"]
        if len(selection_dataset) < 2:
            raise ValueError("A chronological calibration split requires at least two validation examples")
        origins = sorted({str(example.forecast_start) for example in selection_dataset.examples})
        if len(origins) < 2:
            raise ValueError("Calibration requires at least two distinct forecast origins")
        calibration_count = int(round(len(origins) * calibration_fraction))
        calibration_count = min(max(calibration_count, 1), len(origins) - 1)
        calibration_origins = set(origins[-calibration_count:])
        selection_origins = set(origins[:-calibration_count])
        if calibration_embargo_days:
            calibration_start = datetime.fromisoformat(min(calibration_origins))
            selection_limit = calibration_start - timedelta(days=int(calibration_embargo_days))
            selection_origins = {
                origin
                for origin in selection_origins
                if datetime.fromisoformat(origin) <= selection_limit
            }
        selection_examples = [
            example for example in selection_dataset.examples
            if str(example.forecast_start) in selection_origins
        ]
        calibration_examples = [
            example for example in selection_dataset.examples
            if str(example.forecast_start) in calibration_origins
        ]
        if not selection_examples or not calibration_examples:
            raise ValueError("Calibration split and embargo must retain both selection and calibration origins")
        wrapped["val"] = PaperEnergyDataset(
            selection_examples,
            spec=spec,
            model_config=model_config,
            require_llm=require_llm,
            future_exog_available=future_exog_policy in {"available", "issued", "oracle"},
            availability_mask_enabled=availability_mask_enabled,
        )
        wrapped["calibration"] = PaperEnergyDataset(
            calibration_examples,
            spec=spec,
            model_config=model_config,
            require_llm=require_llm,
            future_exog_available=future_exog_policy in {"available", "issued", "oracle"},
            availability_mask_enabled=availability_mask_enabled,
        )
    manifest = dict(manifest)
    manifest["paper_task_spec"] = spec.to_dict()
    manifest["availability_fields"] = list(AVAILABILITY_FIELDS)
    manifest["availability_mask_enabled"] = bool(availability_mask_enabled)
    manifest["calibration_fraction"] = float(calibration_fraction)
    manifest["split_embargo_steps"] = int(split_embargo_steps)
    manifest["calibration_embargo_days"] = int(calibration_embargo_days)
    manifest["text_schema_variant"] = text_schema_variant
    manifest["model_selection_val_examples"] = len(wrapped["val"])
    manifest["calibration_examples"] = len(wrapped.get("calibration", []))
    return TaskDataBundle(spec=spec, datasets=wrapped, manifest=manifest)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def intervene_batch(
    batch: dict[str, Any],
    intervention: str,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Return a shallow-copied batch under a controlled context intervention."""

    out = dict(batch)
    if intervention == "full":
        return out
    if intervention == "zero_context":
        out["numeric_context"] = torch.zeros_like(batch["numeric_context"])
        out["schema_context"] = torch.zeros_like(batch["schema_context"])
        out["llm_context"] = torch.zeros_like(batch["llm_context"])
        out["availability"] = torch.zeros_like(batch["availability"])
        return out
    if intervention == "mean_context":
        for key in ("numeric_context", "schema_context", "llm_context", "availability"):
            mean = batch[key].mean(dim=0, keepdim=True)
            out[key] = mean.expand_as(batch[key]).clone()
        return out
    if intervention == "shuffle_context":
        size = batch["numeric_context"].shape[0]
        permutation = torch.randperm(size, generator=generator, device=batch["numeric_context"].device)
        for key in ("numeric_context", "schema_context", "llm_context", "availability"):
            out[key] = batch[key][permutation]
        return out
    if intervention == "llm_zero":
        out["llm_context"] = torch.zeros_like(batch["llm_context"])
        return out
    if intervention == "llm_shuffle":
        size = batch["llm_context"].shape[0]
        permutation = torch.randperm(size, generator=generator, device=batch["llm_context"].device)
        out["llm_context"] = batch["llm_context"][permutation]
        return out
    if intervention in {"missing_fields_masked", "missing_fields_unmasked"}:
        tokens = batch["tokens"].clone()
        max_exog_dim = (tokens.shape[-1] - 7) // 2
        mask_start = 5 + max_exog_dim
        for row, history_len in enumerate(batch["history_len"].tolist()):
            tokens[row, int(history_len) :, 5 : 5 + max_exog_dim] = 0.0
            if intervention == "missing_fields_masked":
                tokens[row, int(history_len) :, mask_start : mask_start + max_exog_dim] = 0.0
        out["tokens"] = tokens
        if intervention == "missing_fields_masked":
            availability = batch["availability"].clone()
            availability[:, -1] = 0.0
            out["availability"] = availability
        return out
    if intervention == "delayed_context":
        permutation = torch.arange(
            batch["numeric_context"].shape[0], device=batch["numeric_context"].device
        )
        groups: dict[str, list[int]] = {}
        for index, node_id in enumerate(batch["node_id"]):
            groups.setdefault(str(node_id), []).append(index)
        for indices in groups.values():
            ordered = sorted(indices, key=lambda index: str(batch["forecast_start"][index]))
            for position in range(1, len(ordered)):
                permutation[ordered[position]] = ordered[position - 1]
        for key in ("numeric_context", "schema_context", "llm_context", "availability"):
            out[key] = batch[key][permutation]
        return out
    if intervention in {"wrong_region", "wrong_release_time"}:
        group_values = (
            batch["forecast_start"] if intervention == "wrong_region" else batch["node_id"]
        )
        distinct_values = (
            batch["node_id"] if intervention == "wrong_region" else batch["forecast_start"]
        )
        permutation = torch.arange(
            batch["numeric_context"].shape[0],
            device=batch["numeric_context"].device,
        )
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(group_values):
            groups.setdefault(str(value), []).append(index)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            for offset, index in enumerate(indices):
                candidates = [
                    candidate
                    for candidate in indices
                    if str(distinct_values[candidate]) != str(distinct_values[index])
                ]
                if candidates:
                    permutation[index] = candidates[offset % len(candidates)]
        unchanged = permutation == torch.arange(
            permutation.shape[0],
            device=permutation.device,
        )
        if torch.any(unchanged):
            fallback = torch.roll(permutation, shifts=1)
            permutation = torch.where(unchanged, fallback, permutation)
        for key in ("numeric_context", "schema_context", "llm_context", "availability"):
            out[key] = batch[key][permutation]
        return out
    if intervention == "shuffle_fields":
        numeric = batch["numeric_context"].clone()
        if numeric.shape[1] <= 8:
            raise ValueError("shuffle_fields requires non-task numeric context fields")
        field_permutation = torch.randperm(
            numeric.shape[1] - 8,
            generator=generator,
            device=numeric.device,
        )
        numeric[:, 8:] = numeric[:, 8:][:, field_permutation]
        out["numeric_context"] = numeric
        availability = batch["availability"].clone()
        availability_permutation = torch.randperm(
            availability.shape[1] - 1,
            generator=generator,
            device=availability.device,
        )
        availability[:, 1:] = availability[:, 1:][:, availability_permutation]
        out["availability"] = availability
        schema = batch["schema_context"].clone()
        schema_permutation = torch.randperm(
            schema.shape[1], generator=generator, device=schema.device
        )
        out["schema_context"] = schema[:, schema_permutation]
        return out
    if intervention == "wrong_task":
        out["task_id"] = (batch["task_id"] + 1) % 4
        numeric = batch["numeric_context"].clone()
        numeric[:, :8] = 0.0
        numeric.scatter_(1, out["task_id"].unsqueeze(1), 1.0)
        out["numeric_context"] = numeric
        return out
    raise ValueError(f"Unknown intervention: {intervention}")
