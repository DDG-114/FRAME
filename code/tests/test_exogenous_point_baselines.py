from __future__ import annotations

import torch

from scripts.run_tste_point_baselines import IssuedResidualTFT, IssuedResidualTimeXer


def test_exogenous_point_baselines_produce_horizon_predictions() -> None:
    batch, history, horizon = 2, 48, 24
    tokens = torch.randn(batch, history + horizon, 57)
    context = torch.randn(batch, 69)
    models = (
        IssuedResidualTimeXer(history, horizon, patch=12, stride=6),
        IssuedResidualTFT(history),
    )
    for model in models:
        prediction = model(tokens, context)
        assert prediction.shape == (batch, horizon)
        assert torch.isfinite(prediction).all()
