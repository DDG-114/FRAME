#!/usr/bin/env python
"""Build frozen Qwen3 context caches under the exact paper window protocol."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.energy_context.data import load_energy_datasets
from src.energy_context.llm_context import context_text_id, write_llm_context_cache
from src.energyca_paper.config import aemo_task_specs, paper_task_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["load", "wind", "pv", "net_load"])
    parser.add_argument("--protocol", choices=["original", "aemo", "perform"], default="original")
    parser.add_argument("--profile", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--train_windows_per_node", type=int)
    parser.add_argument("--eval_windows_per_node", type=int)
    parser.add_argument("--max_nodes", type=int)
    parser.add_argument("--history_days", type=int, default=7)
    parser.add_argument("--horizon_days", type=int, default=1)
    parser.add_argument("--forecast_stride_days", type=int, default=1)
    parser.add_argument(
        "--future_exog_policy",
        choices=["common", "issued", "oracle", "available", "zero"],
        default="available",
    )
    parser.add_argument("--split_embargo_days", type=int, default=0)
    parser.add_argument("--market_notice_path")
    parser.add_argument("--model_path", default="models/Qwen3-8B")
    parser.add_argument("--representation", choices=["qwen", "tfidf"], default="qwen")
    parser.add_argument("--tfidf_dimension", type=int, default=4096)
    parser.add_argument("--output_root", default="outputs/context_caches")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--pooling", choices=["mean", "last_token"], default="mean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--pool_normalization",
        choices=["none", "l2"],
        default="l2",
        help="Keep l2 for archived runs; use none for the paper-formula aaai_exact variant.",
    )
    parser.add_argument(
        "--text_schema_variant",
        choices=["current", "aaai_exact", "no_availability", "operational"],
        default="current",
    )
    return parser.parse_args()


def resolve_limits(args: argparse.Namespace) -> tuple[int | None, int | None, int | None]:
    if args.profile == "smoke":
        train_windows = 8
        eval_windows = 4
        max_nodes = 1
    else:
        train_windows = None
        eval_windows = None
        max_nodes = None
    if args.train_windows_per_node is not None:
        train_windows = int(args.train_windows_per_node)
    if args.eval_windows_per_node is not None:
        eval_windows = int(args.eval_windows_per_node)
    if args.max_nodes is not None:
        max_nodes = int(args.max_nodes)
    return train_windows, eval_windows, max_nodes


def collect_contexts(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    train_windows, eval_windows, max_nodes = resolve_limits(args)
    if args.protocol == "aemo":
        spec_factory = aemo_task_specs
    elif args.protocol == "perform":
        from src.energyca_paper.config import perform_task_specs

        spec_factory = perform_task_specs
    else:
        spec_factory = paper_task_specs
    spec_kwargs = {
        "train_windows_per_node": train_windows,
        "eval_windows_per_node": eval_windows,
        "history_days": args.history_days,
        "horizon_days": args.horizon_days,
        "forecast_stride_days": args.forecast_stride_days,
    }
    if args.protocol != "perform":
        spec_kwargs["max_nodes_override"] = max_nodes
    elif max_nodes not in {None, 1}:
        raise ValueError("PERFORM ERCOT contains one BA-level node")
    specs = spec_factory(**spec_kwargs)
    collected: dict[str, dict[str, Any]] = {}
    for task in args.tasks:
        spec = specs[task]
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
            text_schema_variant=args.text_schema_variant,
            future_exog_policy=args.future_exog_policy,
            context_embedding_mode="numeric",
            embargo_steps=(24 * 60 // spec.cadence_minutes) * args.split_embargo_days,
        )
        text_by_id: dict[str, str] = {}
        train_ids: set[str] = set()
        split_counts: dict[str, int] = {}
        for split_name, dataset in datasets.items():
            split_counts[split_name] = len(dataset.examples)
            for example in dataset.examples:
                text_id = context_text_id(example.context_text)
                text_by_id.setdefault(text_id, example.context_text)
                if split_name == "train":
                    train_ids.add(text_id)
        ids = sorted(text_by_id)
        ordered_train_ids = sorted(train_ids)
        collected[task] = {
            "spec": spec,
            "ids": ids,
            "texts": [text_by_id[text_id] for text_id in ids],
            "train_ids": ordered_train_ids,
            "train_texts": [text_by_id[text_id] for text_id in ordered_train_ids],
            "split_counts": split_counts,
            "data_manifest": manifest,
        }
        print(
            json.dumps(
                {
                    "task": task,
                    "unique_contexts": len(ids),
                    "split_counts": split_counts,
                    "spec": spec.to_dict(),
                }
            ),
            flush=True,
        )
    return collected


def prompt(text: str, *, text_schema_variant: str) -> str:
    if text_schema_variant == "aaai_exact":
        return text
    if text_schema_variant == "no_availability":
        return (
            "Encode this forecast-origin-valid energy context. Preserve task semantics, "
            "asset and geographic attributes, calendar and weather conditions, and recent "
            "operating state.\n"
            f"{text}"
        )
    if text_schema_variant == "operational":
        return (
            "Encode the operational conditions known at this forecast origin. Preserve the "
            "region, notice type, affected equipment or reserve condition, and timing.\n"
            f"{text}"
        )
    return (
        "Encode this forecast-origin-valid energy context. Preserve task semantics, "
        "asset and geographic attributes, calendar and weather conditions, recent "
        "operating state, and explicit field availability. Do not infer unavailable "
        "future targets.\n"
        f"{text}"
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def cache_matches_collected_contexts(
    path: Path,
    item: dict[str, Any],
    *,
    pool_normalization: str,
    text_schema_variant: str,
    model_path: str,
    prompt_policy: str,
    representation: str,
    tfidf_dimension: int,
    pooling: str = "mean",
    max_length: int = 192,
) -> bool:
    """Reuse a cache only when texts and encoder protocol match exactly."""

    if not path.exists():
        return False
    try:
        payload = np.load(path)
        cached_ids = np.asarray(payload["ids"]).astype(str)
        cached_texts = np.asarray(payload["texts"]).astype(str)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, json.JSONDecodeError, FileNotFoundError):
        return False
    matches_texts = (
        np.array_equal(cached_ids, np.asarray(item["ids"], dtype=str))
        and np.array_equal(cached_texts, np.asarray(item["texts"], dtype=str))
        and metadata.get("text_schema_variant") == text_schema_variant
        and metadata.get("prompt_policy") == prompt_policy
    )
    if not matches_texts:
        return False
    if representation == "qwen":
        return (
            metadata.get("representation", "qwen") == "qwen"
            and metadata.get("pool_normalization") == pool_normalization
            and metadata.get("model_path") == str(model_path)
            and metadata.get("pooling", "mean") == pooling
            and int(metadata.get("max_length", 192)) == int(max_length)
        )
    return (
        metadata.get("representation") == "tfidf"
        and int(metadata.get("embedding_dim", 0)) == int(tfidf_dimension)
        and metadata.get("fit_scope")
        == "unique_contexts_observed_in_chronological_training_windows_only"
        and metadata.get("train_id_sha256") == _ids_digest(item["train_ids"])
    )


@torch.inference_mode()
def encode_task(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    pool_normalization: str,
    text_schema_variant: str,
    pooling: str = "mean",
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            [
                prompt(text, text_schema_variant=text_schema_variant)
                for text in texts[start : start + batch_size]
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        result = model(**encoded, use_cache=False)
        hidden = result.last_hidden_state
        if pooling == "last_token":
            last_index = encoded["attention_mask"].sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device),
                last_index,
            ]
        else:
            mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled.float()
        if pool_normalization == "l2":
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        outputs.append(pooled.cpu().numpy())
        completed = min(start + batch_size, len(texts))
        if completed == len(texts) or (start // batch_size + 1) % 100 == 0:
            print(json.dumps({"encoded": completed, "total": len(texts)}), flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


_TFIDF_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _tfidf_terms(text: str) -> list[str]:
    tokens = _TFIDF_TOKEN.findall(str(text).lower())
    return tokens + [f"{left}__{right}" for left, right in zip(tokens, tokens[1:])]


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def encode_tfidf(
    texts: list[str],
    *,
    train_texts: list[str],
    dimension: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Fit TF-IDF only on chronological training texts and encode every split."""

    if dimension < 1:
        raise ValueError("tfidf_dimension must be positive")
    if not train_texts:
        raise ValueError("TF-IDF requires at least one chronological training context")
    document_frequency: Counter[str] = Counter()
    for text in train_texts:
        document_frequency.update(set(_tfidf_terms(text)))
    vocabulary = sorted(document_frequency, key=lambda term: (-document_frequency[term], term))[:dimension]
    index = {term: offset for offset, term in enumerate(vocabulary)}
    train_count = len(train_texts)
    idf = np.asarray(
        [math.log((1 + train_count) / (1 + document_frequency[term])) + 1.0 for term in vocabulary],
        dtype=np.float32,
    )
    embeddings = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        counts = Counter(_tfidf_terms(text))
        for term, count in counts.items():
            column = index.get(term)
            if column is not None:
                embeddings[row, column] = (1.0 + math.log(float(count))) * idf[column]
        norm = float(np.linalg.norm(embeddings[row]))
        if norm > 0.0:
            embeddings[row] /= norm
    return embeddings, vocabulary, idf


def main() -> None:
    args = parse_args()
    if args.text_schema_variant == "operational":
        if not args.market_notice_path:
            raise ValueError("--market_notice_path is required for the operational text schema")
        os.environ["AEMO_MARKET_NOTICE_PATH"] = str(Path(args.market_notice_path).expanduser().resolve())
    if args.representation == "tfidf" and args.tfidf_dimension != 4096:
        raise ValueError("The FRAME TF-IDF control uses the fixed 4096-dimensional LLM interface")
    collected = collect_contexts(args)
    output_root = Path(args.output_root) / args.profile
    prompt_policy = (
        "direct deterministic schema"
        if args.text_schema_variant == "aaai_exact"
        else (
            "forecast-origin AEMO operational notices v1"
            if args.text_schema_variant == "operational"
            else "forecast-origin-valid energy context v1"
        )
    )
    cache_suffix = {
        "no_availability": "_no_availability",
        "operational": "_operational",
    }.get(args.text_schema_variant, "")
    pending_tasks: list[str] = []
    for task in args.tasks:
        cache_name = (
            f"qwen3_context_cache{cache_suffix}.npz"
            if args.representation == "qwen"
            else f"tfidf4096_context_cache{cache_suffix}.npz"
        )
        cache_path = output_root / task / cache_name
        if cache_matches_collected_contexts(
            cache_path,
            collected[task],
            pool_normalization=args.pool_normalization,
            text_schema_variant=args.text_schema_variant,
            model_path=args.model_path,
            prompt_policy=prompt_policy,
            representation=args.representation,
            tfidf_dimension=args.tfidf_dimension,
            pooling=args.pooling,
            max_length=args.max_length,
        ):
            print(json.dumps({"task": task, "cache": str(cache_path), "status": "reused"}), flush=True)
        else:
            pending_tasks.append(task)
    if not pending_tasks:
        return
    tokenizer = None
    model = None
    if args.representation == "qwen":
        device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=resolve_dtype(args.dtype),
            device_map={"": int(device.index or 0)} if device.type == "cuda" else None,
            attn_implementation="sdpa",
        )
        if device.type != "cuda":
            model.to(device)
        model.eval()

    for task in pending_tasks:
        item = collected[task]
        if args.representation == "qwen":
            embeddings = encode_task(
                item["texts"],
                tokenizer=tokenizer,
                model=model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                pool_normalization=args.pool_normalization,
                text_schema_variant=args.text_schema_variant,
                pooling=args.pooling,
            )
            vocabulary: list[str] = []
            idf = np.asarray([], dtype=np.float32)
        else:
            embeddings, vocabulary, idf = encode_tfidf(
                item["texts"],
                train_texts=item["train_texts"],
                dimension=args.tfidf_dimension,
            )
        cache_name = (
            f"qwen3_context_cache{cache_suffix}.npz"
            if args.representation == "qwen"
            else f"tfidf4096_context_cache{cache_suffix}.npz"
        )
        output_path = output_root / task / cache_name
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "representation": args.representation,
            "encoder": (
                f"Qwen3-8B frozen {args.pooling.replace('_', '-')} final hidden state"
                if args.representation == "qwen"
                else "train-fit TF-IDF unigram and bigram context representation"
            ),
            "model_path": str(args.model_path) if args.representation == "qwen" else None,
            "embedding_dim": int(embeddings.shape[1]),
            "unique_context_texts": len(item["texts"]),
            "split_example_counts": item["split_counts"],
            "paper_task_spec": item["spec"].to_dict(),
            "data": item["data_manifest"],
            "prompt_policy": prompt_policy,
            "pool_normalization": args.pool_normalization,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "text_schema_variant": args.text_schema_variant,
            "normalized": args.pool_normalization == "l2",
            "args": vars(args),
        }
        if args.representation == "tfidf":
            metadata.update(
                {
                    "fit_scope": "unique_contexts_observed_in_chronological_training_windows_only",
                    "fit_unique_contexts": len(item["train_texts"]),
                    "all_unique_contexts": len(item["texts"]),
                    "term_rule": "lowercase alphanumeric unigrams plus adjacent bigrams; TF=1+log(count); IDF=log((1+n)/(1+df))+1; l2 normalized",
                    "train_id_sha256": _ids_digest(item["train_ids"]),
                }
            )
        write_llm_context_cache(
            path=output_path,
            ids=item["ids"],
            texts=item["texts"],
            embeddings=embeddings,
            metadata=metadata,
        )
        manifest_path = output_path.with_suffix(".json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.representation == "tfidf":
            np.savez_compressed(
                output_path.with_name("tfidf4096_transform.npz"),
                feature_names=np.asarray(vocabulary, dtype=str),
                idf=idf,
                train_ids=np.asarray(item["train_ids"], dtype=str),
            )
        print(
            json.dumps(
                {
                    "task": task,
                    "representation": args.representation,
                    "cache": str(output_path),
                    "embedding_dim": metadata["embedding_dim"],
                    "unique_context_texts": metadata["unique_context_texts"],
                    "split_example_counts": metadata["split_example_counts"],
                    "paper_task_spec": metadata["paper_task_spec"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
