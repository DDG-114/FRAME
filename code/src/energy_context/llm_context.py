"""LLM context embedding cache utilities for EnergyCA experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CONTEXT_EMBEDDING_MODES = ("numeric", "llm", "concat")


def context_text_id(text: str) -> str:
    """Stable id for a rendered energy-context text."""
    return hashlib.blake2b(str(text).encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def load_llm_context_cache(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load an .npz context embedding cache keyed by context_text_id."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing LLM context cache: {cache_path}")
    payload = np.load(cache_path, allow_pickle=False)
    ids = [str(value) for value in payload["ids"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(ids) != embeddings.shape[0]:
        raise ValueError(f"Invalid LLM context cache shape in {cache_path}")
    metadata: dict[str, Any] = {}
    if "metadata_json" in payload.files:
        metadata_raw = str(payload["metadata_json"].item())
        metadata = json.loads(metadata_raw) if metadata_raw else {}
    return dict(zip(ids, embeddings, strict=True)), metadata


def apply_llm_context_embeddings(
    examples: list[Any],
    *,
    cache_path: str | Path,
    mode: str,
    missing_policy: str = "error",
) -> dict[str, Any]:
    """Mutate EnergyExample.context with cached LLM embeddings.

    mode="llm" replaces the numeric context vector with the LLM vector.
    mode="concat" appends the LLM vector after the numeric context vector.
    mode="numeric" is accepted as a no-op so callers can pass through config.
    """
    if mode not in CONTEXT_EMBEDDING_MODES:
        raise ValueError(f"mode must be one of {CONTEXT_EMBEDDING_MODES}, got {mode!r}")
    if missing_policy not in {"error", "zero"}:
        raise ValueError("missing_policy must be 'error' or 'zero'")
    if mode == "numeric":
        return {
            "mode": mode,
            "cache_path": None,
            "llm_context_dim": 0,
            "missing_count": 0,
            "applied_examples": len(examples),
        }

    cache, metadata = load_llm_context_cache(cache_path)
    if not cache:
        raise ValueError(f"LLM context cache is empty: {cache_path}")
    first_embedding = next(iter(cache.values()))
    llm_dim = int(first_embedding.shape[0])
    missing = 0
    for ex in examples:
        text_id = context_text_id(str(ex.context_text))
        embedding = cache.get(text_id)
        if embedding is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(
                    "LLM context cache does not contain text id "
                    f"{text_id} for context: {str(ex.context_text)[:160]}"
                )
            embedding = np.zeros(llm_dim, dtype=np.float32)
        embedding = np.asarray(embedding, dtype=np.float32)
        if mode == "llm":
            ex.context = embedding.copy()
        else:
            ex.context = np.concatenate([np.asarray(ex.context, dtype=np.float32), embedding], axis=0).astype(np.float32)

    return {
        "mode": mode,
        "cache_path": str(Path(cache_path)),
        "llm_context_dim": llm_dim,
        "missing_count": int(missing),
        "applied_examples": int(len(examples)),
        "cache": metadata,
    }


def write_llm_context_cache(
    *,
    path: str | Path,
    ids: list[str],
    texts: list[str],
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write an .npz cache in the format consumed by load_llm_context_cache."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(ids) != embeddings.shape[0] or len(texts) != embeddings.shape[0]:
        raise ValueError("ids, texts, and embeddings must have the same first dimension")
    np.savez_compressed(
        out_path,
        ids=np.asarray(ids, dtype=str),
        texts=np.asarray(texts, dtype=str),
        embeddings=embeddings,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
