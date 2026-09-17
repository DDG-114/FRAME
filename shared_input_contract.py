from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.energy_context.context import CONTEXT_FIELD_NAMES
from src.energyca_paper.data import AVAILABILITY_FIELDS, _hashed_schema_context


TASKS = ("load", "wind", "pv", "net_load")
CONTRACT_VERSION = "aemo-shared-facts-v1"
JOINT_HISTORY_COLUMNS = (9, 10, 11, 12)
JOINT_HISTORY_MASK_COLUMNS = (34, 35, 36, 37)
ISSUED_VALUE_COLUMNS = (5, 6, 7, 8)
ISSUED_MASK_COLUMNS = (30, 31, 32, 33)
COMMON_CONTEXT_FIELDS = (
    "has_weather",
    "has_price",
    "frequency_scaled",
    "forecast_hour_sin",
    "forecast_hour_cos",
    "forecast_dow_sin",
    "forecast_dow_cos",
    "is_weekend",
    "forecast_dayofyear_sin",
    "forecast_dayofyear_cos",
    "forecast_month_sin",
    "forecast_month_cos",
    "latitude_scaled",
    "longitude_scaled",
    "coverage_log_scaled",
)
COMMON_CONTEXT_INDICES = tuple(CONTEXT_FIELD_NAMES.index(name) for name in COMMON_CONTEXT_FIELDS)
BASELINE_DYNAMIC_FIELDS = (
    *(f"history_{task}" for task in TASKS),
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    *(f"issued_{task}" for task in TASKS),
    *(f"issued_{task}_available" for task in TASKS),
    "is_history",
    "is_future",
)


def _key(example: Any) -> tuple[str, str]:
    return str(example.forecast_start), str(example.node_id)


def _base_dataset(dataset):
    return dataset.base if isinstance(dataset, SharedInputDataset) else dataset


def _index(dataset) -> dict[tuple[str, str], int]:
    result = {_key(example): index for index, example in enumerate(dataset.examples)}
    if len(result) != len(dataset.examples):
        raise ValueError("Duplicate forecast-origin/node keys in shared-input dataset")
    return result


def _canonical_context(contexts: list[np.ndarray]) -> np.ndarray:
    values = [np.asarray(context, dtype=np.float32).reshape(-1) for context in contexts]
    dimension = len(CONTEXT_FIELD_NAMES)
    if any(len(value) < dimension for value in values):
        raise ValueError("Numeric context is shorter than the declared context schema")
    common = np.zeros(dimension, dtype=np.float32)
    reference = values[0]
    for index in COMMON_CONTEXT_INDICES:
        if not all(np.isclose(reference[index], value[index], atol=1e-6, rtol=1e-6) for value in values[1:]):
            raise ValueError(f"Shared context field differs across tasks: {CONTEXT_FIELD_NAMES[index]}")
        common[index] = reference[index]
    return common


def _exclude_capacity_text(value: str) -> str:
    return re.sub(r"capacity_field=(?:available|unavailable)", "capacity_field=excluded", str(value))


class SharedInputDataset(Dataset):
    def __init__(self, base, peers: tuple[Any, ...]):
        self.base = base
        self.peers = peers
        self._index_cache: dict[int, tuple[int, int, dict[tuple[str, str], int]]] = {}

    @property
    def examples(self):
        return self.base.examples

    @examples.setter
    def examples(self, value):
        self.base.examples = value

    @property
    def schema_contexts(self):
        return self.base.schema_contexts

    @schema_contexts.setter
    def schema_contexts(self, value):
        self.base.schema_contexts = value

    @property
    def require_llm(self):
        return self.base.require_llm

    @require_llm.setter
    def require_llm(self, value):
        self.base.require_llm = value

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __len__(self):
        return len(self.base)

    def _mapping(self, dataset):
        cached = self._index_cache.get(id(dataset))
        signature = (id(dataset.examples), len(dataset.examples))
        if cached is None or cached[:2] != signature:
            cached = (*signature, _index(dataset))
            self._index_cache[id(dataset)] = cached
        return cached[2]

    def __getitem__(self, index):
        item = dict(self.base[index])
        example = self.base.examples[index]
        key = _key(example)
        peer_examples = [peer.examples[self._mapping(peer)[key]] for peer in self.peers]
        history_len = int(item["history_len"])
        histories = torch.stack(
            [torch.as_tensor(np.asarray(peer.numeric, dtype=np.float32)[:history_len]) for peer in peer_examples],
            dim=-1,
        )
        tokens = item["tokens"].clone()
        if tokens.shape[-1] <= JOINT_HISTORY_MASK_COLUMNS[-1]:
            raise ValueError("Token layout has no reserved shared-history columns")
        tokens[:, JOINT_HISTORY_COLUMNS] = 0.0
        tokens[:, JOINT_HISTORY_MASK_COLUMNS] = 0.0
        tokens[:history_len, JOINT_HISTORY_COLUMNS] = histories
        tokens[:history_len, JOINT_HISTORY_MASK_COLUMNS] = 1.0
        tokens[:history_len, ISSUED_VALUE_COLUMNS] = 0.0
        tokens[:history_len, ISSUED_MASK_COLUMNS] = 0.0
        item["tokens"] = tokens
        item["numeric_context"] = item["numeric_context"].clone()
        item["availability"] = item["availability"].clone()
        return item

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_index_cache"] = {}
        return state


def contract_manifest() -> dict[str, Any]:
    excluded = [name for name in CONTEXT_FIELD_NAMES if name not in COMMON_CONTEXT_FIELDS]
    return {
        "version": CONTRACT_VERSION,
        "task_order": list(TASKS),
        "facts": [
            "four synchronized task histories through the forecast origin",
            "four calendar fields for history and forecast timestamps",
            "four AEMO forecasts selected at or before the forecast origin",
            "per-field per-lead issued-forecast availability",
            "common cadence, calendar, geography, and coverage metadata",
        ],
        "future_actual_targets": "excluded",
        "task_specific_numeric_context": "excluded",
        "common_numeric_context_fields": list(COMMON_CONTEXT_FIELDS),
        "excluded_numeric_context_fields": excluded,
        "frame_joint_history_token_columns": list(JOINT_HISTORY_COLUMNS),
        "frame_joint_history_mask_columns": list(JOINT_HISTORY_MASK_COLUMNS),
        "baseline_dynamic_fields": list(BASELINE_DYNAMIC_FIELDS),
        "baseline_static_fields": [*CONTEXT_FIELD_NAMES, *AVAILABILITY_FIELDS],
        "semantic_schema": (
            "task/region/calendar/availability labels and categorical summaries derived from the shared "
            "histories and issued fields; target training statistics and capacity state are excluded"
        ),
    }


def apply_shared_input_contract(bundles: dict[str, Any]) -> dict[str, Any]:
    dimension = len(CONTEXT_FIELD_NAMES)
    for hours in (2, 24, 168):
        selected = [bundles[f"{task}_h{hours}hours"] for task in TASKS]
        splits = tuple(selected[0].datasets)
        if any(tuple(bundle.datasets) != splits for bundle in selected[1:]):
            raise ValueError(f"Split names differ across H{hours} task bundles")
        for split in splits:
            datasets = [_base_dataset(bundle.datasets[split]) for bundle in selected]
            mappings = [_index(dataset) for dataset in datasets]
            keys = [_key(example) for example in datasets[0].examples]
            if any(set(mapping) != set(keys) for mapping in mappings[1:]):
                raise ValueError(f"Forecast-origin/node keys differ across H{hours} {split}")
            for key in keys:
                examples = [dataset.examples[mapping[key]] for dataset, mapping in zip(datasets, mappings)]
                history_lengths = {int(example.history_len) for example in examples}
                if len(history_lengths) != 1:
                    raise ValueError(f"History lengths differ across tasks for H{hours} {split} {key}")
                history_len = history_lengths.pop()
                reference = np.asarray(examples[0].numeric, dtype=np.float32)[history_len:]
                if not all(
                    np.allclose(reference, np.asarray(example.numeric, dtype=np.float32)[history_len:], atol=1e-6, rtol=1e-6)
                    for example in examples[1:]
                ):
                    raise ValueError(f"Calendar/issued values differ across tasks for H{hours} {split} {key}")
                reference_mask = np.asarray(examples[0].future_exog_mask, dtype=np.float32)
                if not all(
                    np.array_equal(reference_mask, np.asarray(example.future_exog_mask, dtype=np.float32))
                    for example in examples[1:]
                ):
                    raise ValueError(f"Issued masks differ across tasks for H{hours} {split} {key}")
                common = _canonical_context([example.context[:dimension] for example in examples])
                for example in examples:
                    tail = np.asarray(example.context, dtype=np.float32).reshape(-1)[dimension:]
                    example.context = np.concatenate((common, tail)).astype(np.float32)
                    example.context_text = _exclude_capacity_text(example.context_text)
                    example.context_segments = [_exclude_capacity_text(value) for value in example.context_segments]
            for dataset in datasets:
                dataset.schema_contexts = [
                    _hashed_schema_context(str(example.context_text), dimension=dataset.model_config.schema_context_dim)
                    for example in dataset.examples
                ]
            for bundle, dataset in zip(selected, datasets):
                bundle.datasets[split] = SharedInputDataset(dataset, tuple(datasets))
                bundle.manifest["shared_input_contract"] = contract_manifest()
    return bundles


def build_baseline_arrays(bundles: dict[str, Any], hours: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    selected = [bundles[f"{task}_h{hours}hours"] for task in TASKS]
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    splits = ("train", "val", "calibration", "test")
    if all("confirmation" in bundle.datasets for bundle in selected):
        splits = (*splits, "confirmation")
    for split in splits:
        datasets = [bundle.datasets[split] for bundle in selected]
        mappings = [_index(dataset) for dataset in datasets]
        keys = [_key(example) for example in datasets[0].examples]
        if any(set(mapping) != set(keys) for mapping in mappings[1:]):
            raise ValueError(f"Forecast-origin/node keys differ across H{hours} {split}")
        rows = {name: [] for name in ("x", "context", "y", "raw", "scale", "offset", "history")}
        for key in keys:
            items = [dataset[mapping[key]] for dataset, mapping in zip(datasets, mappings)]
            tokens = items[0]["tokens"]
            for item in items[1:]:
                if not torch.allclose(tokens[:, JOINT_HISTORY_COLUMNS], item["tokens"][:, JOINT_HISTORY_COLUMNS]):
                    raise ValueError(f"Shared histories differ across task views for {key}")
            x = torch.cat(
                (
                    tokens[:, JOINT_HISTORY_COLUMNS],
                    tokens[:, 1:9],
                    tokens[:, ISSUED_MASK_COLUMNS],
                    tokens[:, -2:],
                ),
                dim=-1,
            )
            values = {
                "x": x,
                "context": torch.cat((items[0]["numeric_context"], items[0]["availability"])),
                "y": torch.stack([item["target"] for item in items], dim=-1),
                "raw": torch.stack([item["target_raw"] for item in items], dim=-1),
                "scale": torch.stack([item["target_scale"] for item in items], dim=-1),
                "offset": torch.stack([item["target_offset"] for item in items], dim=-1),
                "history": torch.stack([item["history_raw"] for item in items], dim=-1),
            }
            if torch.count_nonzero(x[selected[0].spec.history_len :, :4]):
                raise ValueError(f"Future actual targets entered baseline inputs for {key}")
            for name, value in values.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite shared input: H{hours} {split} {key} {name}")
                rows[name].append(value.cpu().numpy())
        arrays = {name: np.stack(values).astype(np.float32) for name, values in rows.items()}
        arrays["forecast_start"] = np.asarray([key[0] for key in keys])
        arrays["node_id"] = np.asarray([key[1] for key in keys])
        split_arrays[split] = arrays
    metadata = {
        **contract_manifest(),
        "hours": hours,
        "history_length": selected[0].spec.history_len,
        "horizon": selected[0].spec.horizon,
        "splits": {split: len(values["y"]) for split, values in split_arrays.items()},
        "normalization": "native train-fitted target normalization; no additional feature transform",
        "physics_scale": float(np.abs(split_arrays["train"]["raw"][..., 0]).mean()),
    }
    return split_arrays, metadata


def write_baseline_arrays(bundles: dict[str, Any], destination: Path) -> None:
    for hours in (2, 24, 168):
        arrays, metadata = build_baseline_arrays(bundles, hours)
        folder = Path(destination) / f"h{hours}"
        folder.mkdir(parents=True, exist_ok=True)
        for split, values in arrays.items():
            np.savez_compressed(folder / f"{split}.npz", **values)
        (folder / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
