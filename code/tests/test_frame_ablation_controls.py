from __future__ import annotations

import numpy as np
import torch

from scripts.build_paper_context_caches import encode_tfidf
from src.energyca_paper.config import ModelConfig
from src.energyca_paper.data import _availability_vector, decode_numeric_tokens, intervene_batch
from src.energyca_paper.training import (
    _semantic_scale_for_epoch,
    direct_residual_supervision_loss,
    operating_state_sample_weights,
)
from src.energyca_paper.model import ForwardControls, SharedEnergyCALLM, semantic_cell_mask


def test_semantic_scale_warms_up_and_ramps_to_target() -> None:
    values = [
        _semantic_scale_for_epoch(
            epoch,
            warmup_epochs=2,
            ramp_epochs=4,
            target_scale=0.5,
        )
        for epoch in range(7)
    ]
    assert values == [0.0, 0.0, 0.125, 0.25, 0.375, 0.5, 0.5]


def test_task_horizon_semantic_mask_preserves_numeric_route() -> None:
    mask = semantic_cell_mask(
        torch.tensor([0, 1, 2, 3, 1, 3]),
        torch.tensor([48, 48, 48, 48, 336, 336]),
        torch.full((6,), 2, dtype=torch.long),
        policy="task_horizon",
        dtype=torch.float32,
    )
    assert mask.squeeze(1).tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def test_load_netload_semantic_mask_routes_operational_context() -> None:
    mask = semantic_cell_mask(
        torch.tensor([0, 1, 2, 3]),
        torch.full((4,), 336, dtype=torch.long),
        torch.full((4,), 2, dtype=torch.long),
        policy="load_netload",
        dtype=torch.float32,
    )
    assert mask.squeeze(1).tolist() == [1.0, 0.0, 0.0, 1.0]


def test_direct_semantic_residual_starts_from_exact_numeric_prediction() -> None:
    config = ModelConfig(
        d_model=16,
        d_context=16,
        d_residual=16,
        rank=2,
        dropout=0.0,
        use_physical_lead_time=True,
        use_direct_semantic_residual=True,
        use_field_conditioned_semantic=True,
        task_output_modes=("identity",) * 4,
    )
    model = SharedEnergyCALLM(config).eval()
    inputs = {
        "tokens": torch.randn(2, 50, config.token_dim),
        "numeric_context": torch.randn(2, config.numeric_context_dim),
        "schema_context": torch.randn(2, config.schema_context_dim),
        "llm_context": torch.randn(2, config.llm_context_dim),
        "availability": torch.ones(2, config.availability_dim),
        "history_len": torch.full((2,), 48, dtype=torch.long),
        "horizon": torch.full((2,), 2, dtype=torch.long),
        "task_id": torch.zeros(2, dtype=torch.long),
        "cadence_id": torch.full((2,), 2, dtype=torch.long),
    }
    full = model(**inputs, controls=ForwardControls(context_mode="full"))["prediction"]
    numeric = model(**inputs, controls=ForwardControls(context_mode="mlp_schema"))["prediction"]
    assert torch.equal(full, numeric)


def test_numeric_direct_residual_starts_from_exact_base_prediction() -> None:
    config = ModelConfig(
        d_model=16,
        d_context=16,
        d_residual=16,
        rank=2,
        dropout=0.0,
        use_physical_lead_time=True,
        use_direct_semantic_residual=True,
        direct_residual_source="numeric",
        task_output_modes=("identity",) * 4,
    )
    model = SharedEnergyCALLM(config).eval()
    inputs = {
        "tokens": torch.randn(2, 50, config.token_dim),
        "numeric_context": torch.randn(2, config.numeric_context_dim),
        "schema_context": torch.randn(2, config.schema_context_dim),
        "llm_context": torch.randn(2, config.llm_context_dim),
        "availability": torch.ones(2, config.availability_dim),
        "history_len": torch.full((2,), 48, dtype=torch.long),
        "horizon": torch.full((2,), 2, dtype=torch.long),
        "task_id": torch.zeros(2, dtype=torch.long),
        "cadence_id": torch.full((2,), 2, dtype=torch.long),
    }
    output = model(**inputs, controls=ForwardControls(context_mode="full"))
    assert torch.equal(output["prediction"], output["base_raw_prediction"])


def test_task_specific_direct_residual_preserves_output_shape() -> None:
    config = ModelConfig(
        d_model=16,
        d_context=16,
        d_residual=16,
        rank=2,
        dropout=0.0,
        use_physical_lead_time=True,
        use_direct_semantic_residual=True,
        direct_residual_head="task_specific",
        task_output_modes=("identity",) * 4,
    )
    model = SharedEnergyCALLM(config).eval()
    inputs = {
        "tokens": torch.randn(4, 50, config.token_dim),
        "numeric_context": torch.randn(4, config.numeric_context_dim),
        "schema_context": torch.randn(4, config.schema_context_dim),
        "llm_context": torch.randn(4, config.llm_context_dim),
        "availability": torch.ones(4, config.availability_dim),
        "history_len": torch.full((4,), 48, dtype=torch.long),
        "horizon": torch.full((4,), 2, dtype=torch.long),
        "task_id": torch.arange(4, dtype=torch.long),
        "cadence_id": torch.full((4,), 2, dtype=torch.long),
    }
    output = model(**inputs, controls=ForwardControls(context_mode="full"))
    assert output["prediction"].shape == (4, 2)
    assert torch.equal(output["prediction"], output["base_raw_prediction"])


def test_direct_residual_supervision_is_zero_for_exact_residual() -> None:
    batch = {
        "target_raw": torch.tensor([[3.0, 7.0]]),
        "target_offset": torch.tensor([[1.0, 1.0]]),
        "target_scale": torch.tensor([[2.0, 2.0]]),
    }
    base = torch.tensor([[0.5, 2.0]])
    exact = torch.tensor([[0.5, 1.0]])
    loss = direct_residual_supervision_loss(exact, base, batch, loss_kind="huber")
    assert torch.equal(loss, torch.zeros_like(loss))


def test_operating_state_weights_are_normalized_and_prioritize_hard_samples() -> None:
    batch = {
        "target_raw": torch.tensor([[0.0, 0.0], [2.0, 4.0], [8.0, 12.0]]),
        "target_scale": torch.ones(3, 2),
        "target_offset": torch.zeros(3, 2),
        "history_raw": torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 6.0, -2.0]]
        ),
    }
    weights = operating_state_sample_weights(
        torch.zeros(3, 2),
        batch,
        task_scale=1.0,
        strength=1.0,
    )
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[-1] > weights[0]


def test_llm_only_interventions_preserve_numeric_context() -> None:
    batch = {
        "numeric_context": torch.tensor([[1.0], [2.0]]),
        "llm_context": torch.tensor([[10.0], [20.0]]),
    }
    zeroed = intervene_batch(batch, "llm_zero")
    shuffled = intervene_batch(
        batch,
        "llm_shuffle",
        generator=torch.Generator().manual_seed(2),
    )
    assert torch.equal(zeroed["numeric_context"], batch["numeric_context"])
    assert torch.equal(shuffled["numeric_context"], batch["numeric_context"])
    assert torch.count_nonzero(zeroed["llm_context"]) == 0


def test_tfidf_is_fixed_width_and_fit_on_train_contexts_only() -> None:
    embeddings, vocabulary, _ = encode_tfidf(
        ["alpha stable", "heldout novel"],
        train_texts=["alpha stable"],
        dimension=4096,
    )
    assert embeddings.shape == (2, 4096)
    assert "alpha" in vocabulary
    assert "heldout" not in vocabulary
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)
    assert np.allclose(embeddings[1], 0.0)


def test_no_availability_uses_constant_interface_slots() -> None:
    tokens = decode_numeric_tokens(
        np.zeros(22, dtype=np.float32),
        history_len=2,
        horizon=2,
        history_exog_dim=1,
        future_exog_dim=1,
        max_exog_dim=3,
        availability_mask_enabled=False,
    )
    availability = _availability_vector(
        np.zeros(25, dtype=np.float32),
        history_len=2,
        future_exog_available=False,
        availability_mask_enabled=False,
    )
    assert np.all(tokens[:, 8:11] == 1.0)
    assert np.all(tokens[:, -2:] == 1.0)
    assert np.all(availability == 1.0)


def test_missing_future_fields_compare_truthful_and_stale_masks() -> None:
    tokens = torch.ones((2, 4, 11), dtype=torch.float32)
    batch = {
        "tokens": tokens,
        "history_len": torch.tensor([2, 2]),
        "availability": torch.ones((2, 8), dtype=torch.float32),
    }
    masked = intervene_batch(batch, "missing_fields_masked")
    unmasked = intervene_batch(batch, "missing_fields_unmasked")
    assert torch.all(masked["tokens"][:, 2:, 5:7] == 0.0)
    assert torch.all(masked["tokens"][:, 2:, 7:9] == 0.0)
    assert torch.all(masked["availability"][:, -1] == 0.0)
    assert torch.all(unmasked["tokens"][:, 2:, 5:7] == 0.0)
    assert torch.all(unmasked["tokens"][:, 2:, 7:9] == 1.0)
    assert torch.all(unmasked["availability"][:, -1] == 1.0)


def test_delayed_context_uses_previous_origin_within_region() -> None:
    batch = {
        "numeric_context": torch.tensor([[1.0], [2.0], [3.0]]),
        "schema_context": torch.tensor([[10.0], [20.0], [30.0]]),
        "llm_context": torch.tensor([[100.0], [200.0], [300.0]]),
        "availability": torch.tensor([[1.0], [2.0], [3.0]]),
        "node_id": ["NSW1", "NSW1", "NSW1"],
        "forecast_start": ["2024-01-01", "2024-01-02", "2024-01-03"],
    }
    delayed = intervene_batch(batch, "delayed_context")
    assert delayed["numeric_context"].squeeze(1).tolist() == [1.0, 1.0, 2.0]
    assert delayed["llm_context"].squeeze(1).tolist() == [100.0, 100.0, 200.0]
