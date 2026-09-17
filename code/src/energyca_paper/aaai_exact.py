"""Formula-faithful components for the AAAI EnergyCA-LLM method.

This module is intentionally isolated from the current TSTE implementation.
It follows Equations (2), (5), (6), and (7) in ``Overleaf_AAAI2027_Upload/
paper.tex`` so that recovery experiments can select the implementation
explicitly without changing archived formal results.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class AAAIExactSemanticProjection(nn.Module):
    """Masked-mean Qwen state followed by the paper's trainable projection and LN.

    Context caches contain the already masked-mean-pooled final hidden state, so
    the runtime module implements the remaining ``LN(W_e z_bar)`` operation.
    Numeric fields and availability are deliberately excluded from this path.
    """

    def __init__(self, llm_context_dim: int, d_context: int) -> None:
        super().__init__()
        self.projection = nn.Linear(llm_context_dim, d_context, bias=False)
        self.norm = nn.LayerNorm(d_context)

    def forward(self, frozen_llm_context: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(frozen_llm_context.detach()))


class AAAIExactSchemaMLP(nn.Module):
    """Non-language schema control used in the manuscript encoder ablation."""

    def __init__(self, schema_context_dim: int, availability_dim: int, d_context: int) -> None:
        super().__init__()
        input_dim = schema_context_dim + availability_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * d_context),
            nn.GELU(),
            nn.Linear(2 * d_context, d_context),
            nn.LayerNorm(d_context),
        )

    def forward(
        self,
        schema_context: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(torch.cat([schema_context, availability], dim=-1))


class AAAIExactAttentionPool(nn.Module):
    """Learned attention pooling over valid historical tokens only."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, mean=0.0, std=d_model**-0.5)

    def forward(self, hidden: torch.Tensor, history_len: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, tokens, channels]")
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        valid = positions < history_len.unsqueeze(1)
        scores = torch.einsum("btd,d->bt", hidden, self.query) / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bt,btd->bd", weights, hidden)


class AAAIExactResidualController(nn.Module):
    """Equation (5): availability-aware additive and multiplicative control."""

    def __init__(
        self,
        *,
        d_context: int,
        d_model: int,
        d_residual: int,
        availability_dim: int,
    ) -> None:
        super().__init__()
        self.availability_projection = nn.Linear(
            availability_dim, d_context, bias=False
        )
        self.additive_projection = nn.Linear(
            2 * d_context + d_model, d_residual, bias=False
        )
        self.semantic_interaction = nn.Linear(
            d_context, d_residual, bias=False
        )
        self.numeric_interaction = nn.Linear(d_model, d_residual, bias=False)
        self.output_projection = nn.Linear(
            2 * d_residual, d_residual, bias=False
        )
        self.norm = nn.LayerNorm(d_residual)

    def forward(
        self,
        semantic_state: torch.Tensor,
        base_state: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        availability_state = self.availability_projection(availability)
        additive = torch.nn.functional.silu(
            self.additive_projection(
                torch.cat([semantic_state, base_state, availability_state], dim=-1)
            )
        )
        multiplicative = self.semantic_interaction(semantic_state) * self.numeric_interaction(base_state)
        return self.norm(
            self.output_projection(torch.cat([additive, multiplicative], dim=-1))
        )


class AAAIExactRankGatedResidualAdapter(nn.Module):
    """Equations (6)--(7) with zero-initialized residual basis B."""

    def __init__(
        self,
        *,
        d_model: int,
        d_residual: int,
        rank: int,
        alpha_init_bias: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.adapter_A = nn.Linear(d_model, rank, bias=False)
        self.adapter_B = nn.Linear(rank, d_model, bias=False)
        self.rank_controller = nn.Linear(d_residual, rank, bias=False)
        self.strength_controller = nn.Linear(d_residual, 1)
        nn.init.kaiming_uniform_(self.adapter_A.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.adapter_B.weight)
        nn.init.zeros_(self.strength_controller.weight)
        nn.init.constant_(self.strength_controller.bias, alpha_init_bias)

    def forward(
        self,
        hidden: torch.Tensor,
        residual_code: torch.Tensor,
        *,
        enabled: bool,
        rank_gate: bool,
        strength_gate: bool,
        fixed_rank_value: float = 1.0,
        fixed_strength_value: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = hidden.shape[0]
        if rank_gate:
            gamma = torch.tanh(self.rank_controller(residual_code))
        else:
            gamma = torch.full(
                (batch_size, self.rank),
                float(fixed_rank_value),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        if strength_gate:
            alpha = torch.sigmoid(self.strength_controller(residual_code))
        else:
            alpha = torch.full(
                (batch_size, 1),
                float(fixed_strength_value),
                device=hidden.device,
                dtype=hidden.dtype,
            )

        delta = self.adapter_B(self.adapter_A(hidden) * gamma.unsqueeze(1))
        if enabled:
            adapted = hidden + alpha.unsqueeze(1) * delta
        else:
            adapted = hidden
            delta = torch.zeros_like(delta)
            alpha = torch.zeros_like(alpha)
            gamma = torch.zeros_like(gamma)
        residual_energy = delta.square().mean(dim=(1, 2))
        return adapted, alpha.squeeze(-1), gamma, residual_energy
