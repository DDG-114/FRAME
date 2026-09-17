"""One-checkpoint shared EnergyCA-LLM model with four rank-gated adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .aaai_exact import (
    AAAIExactAttentionPool,
    AAAIExactRankGatedResidualAdapter,
    AAAIExactResidualController,
    AAAIExactSchemaMLP,
    AAAIExactSemanticProjection,
)


ContextMode = Literal[
    "full",
    "no_context",
    "task_asset",
    "mlp_schema",
    "capacity_matched_schema",
]
SemanticCellPolicy = Literal["global", "task_horizon", "load_netload"]


def apply_task_output_modes(
    values: torch.Tensor,
    task_id: torch.Tensor,
    modes: tuple[str, ...],
) -> torch.Tensor:
    """Apply task-specific physical links while preserving batch shape."""

    if values.shape[0] != task_id.shape[0]:
        raise ValueError("task_id batch dimension must match model output")
    if torch.any(task_id < 0) or torch.any(task_id >= len(modes)):
        raise ValueError("task_id is outside the configured output-mode table")
    constrained = values
    broadcast_shape = (task_id.shape[0],) + (1,) * (values.ndim - 1)
    for index, mode in enumerate(modes):
        mask = (task_id == index).reshape(broadcast_shape)
        if mode == "relu":
            candidate = F.relu(values)
        elif mode == "softplus":
            candidate = F.softplus(values, beta=10.0)
        elif mode == "unit_interval":
            candidate = values.clamp(0.0, 1.0)
        elif mode == "identity":
            candidate = values
        else:
            raise ValueError(f"Unknown task output mode: {mode}")
        constrained = torch.where(mask, candidate, constrained)
    return constrained


@dataclass
class ForwardControls:
    context_mode: ContextMode = "full"
    use_task_interface: bool = True
    use_adapters: bool = True
    rank_gate: bool = True
    strength_gate: bool = True
    use_exogenous_context: bool = True
    fixed_rank_value: float = 1.0
    fixed_strength_value: float = 1.0
    semantic_residual_scale: float = 1.0
    semantic_cell_policy: SemanticCellPolicy = "global"


def semantic_cell_mask(
    task_id: torch.Tensor,
    horizon: torch.Tensor,
    cadence_id: torch.Tensor,
    *,
    policy: SemanticCellPolicy,
    dtype: torch.dtype,
) -> torch.Tensor:
    if policy == "global":
        return torch.ones((task_id.shape[0], 1), device=task_id.device, dtype=dtype)
    if policy == "load_netload":
        return (task_id.eq(0) | task_id.eq(3)).to(dtype=dtype).unsqueeze(-1)
    cadence_minutes = torch.tensor(
        PhysicalLeadTimeEncoding._CADENCE_MINUTES,
        device=horizon.device,
        dtype=dtype,
    )[cadence_id]
    horizon_days = horizon.to(dtype=dtype) * cadence_minutes / 1440.0
    enabled = task_id.eq(0) | task_id.eq(2) | horizon_days.ge(6.5)
    return enabled.to(dtype=dtype).unsqueeze(-1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        if length > self.encoding.shape[0]:
            raise ValueError(f"Sequence length {length} exceeds positional limit {self.encoding.shape[0]}")
        return self.encoding[:length].to(dtype=dtype)


class PhysicalLeadTimeEncoding(nn.Module):
    """Encode each forecast query by physical lead time rather than step index."""

    _CADENCE_MINUTES = (60.0, 10.0, 30.0, 5.0)

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        cadence_id: torch.Tensor,
        horizon_len: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cadence_lookup = torch.tensor(
            self._CADENCE_MINUTES,
            device=cadence_id.device,
            dtype=dtype,
        )
        if torch.any(cadence_id < 0) or torch.any(cadence_id >= len(self._CADENCE_MINUTES)):
            raise ValueError("cadence_id is outside the physical lead-time encoding table")
        step = torch.arange(1, horizon_len + 1, device=cadence_id.device, dtype=dtype)
        lead_hours = cadence_lookup[cadence_id].unsqueeze(1) * step.unsqueeze(0) / 60.0
        lead_days = lead_hours / 24.0
        features = torch.stack(
            (
                lead_days,
                torch.log1p(lead_hours),
                torch.sin(2.0 * torch.pi * lead_days),
                torch.cos(2.0 * torch.pi * lead_days),
            ),
            dim=-1,
        )
        return self.projection(features)


class SemanticContextEncoder(nn.Module):
    """Keep the numeric context as the base and add gated semantic corrections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.llm_context_dim = config.llm_context_dim
        self.numeric_norm = nn.LayerNorm(config.numeric_context_dim)
        self.llm_norm = nn.LayerNorm(config.llm_context_dim, elementwise_affine=False)
        self.numeric_projection = nn.Linear(config.numeric_context_dim, config.d_context)
        self.llm_projection = nn.Linear(config.llm_context_dim, config.d_context, bias=False)
        self.availability_projection = nn.Linear(config.availability_dim, config.d_context)
        self.numeric_base_norm = nn.LayerNorm(config.d_context)
        self.horizon_projection = nn.Linear(2, config.d_context)
        self.semantic_delta = nn.Linear(config.d_context, config.d_context)
        gate_input_dim = 3 * config.d_context + 2 * config.d_model
        self.semantic_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, config.d_context),
            nn.GELU(),
            nn.Linear(config.d_context, config.d_context),
        )
        nn.init.zeros_(self.semantic_delta.weight)
        nn.init.zeros_(self.semantic_delta.bias)
        nn.init.zeros_(self.semantic_gate[-1].weight)
        nn.init.constant_(self.semantic_gate[-1].bias, config.semantic_gate_init_bias)

    def semantic_residual_parameters(self):
        for module in (
            self.llm_projection,
            self.horizon_projection,
            self.semantic_delta,
            self.semantic_gate,
        ):
            yield from module.parameters()

    def forward(
        self,
        numeric_context: torch.Tensor,
        frozen_llm_context: torch.Tensor,
        availability: torch.Tensor,
        task_state: torch.Tensor,
        cadence_state: torch.Tensor,
        horizon: torch.Tensor,
        cadence_id: torch.Tensor,
        *,
        include_llm: bool,
        numeric_as_llm: bool = False,
        semantic_residual_scale: float = 1.0,
        semantic_cell_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_normalized = self.numeric_norm(numeric_context)
        numeric_state = self.numeric_projection(numeric_normalized)
        availability_state = self.availability_projection(availability)
        numeric_base = self.numeric_base_norm(numeric_state + availability_state)
        if not include_llm and not numeric_as_llm:
            return numeric_base, torch.zeros_like(numeric_base)
        if numeric_as_llm:
            repeats = (self.llm_context_dim + numeric_normalized.shape[1] - 1) // numeric_normalized.shape[1]
            expanded_schema = numeric_normalized.repeat(1, repeats)[:, : self.llm_context_dim]
            llm_state = self.llm_projection(self.llm_norm(expanded_schema))
        else:
            llm_state = self.llm_projection(self.llm_norm(frozen_llm_context.detach()))
        cadence_minutes = torch.tensor(
            PhysicalLeadTimeEncoding._CADENCE_MINUTES,
            device=horizon.device,
            dtype=numeric_base.dtype,
        )[cadence_id]
        horizon_days = horizon.to(dtype=numeric_base.dtype) * cadence_minutes / 1440.0
        horizon_features = torch.stack((horizon_days / 7.0, torch.log1p(horizon_days)), dim=-1)
        horizon_state = self.horizon_projection(horizon_features)
        gate = torch.sigmoid(
            self.semantic_gate(
                torch.cat(
                    [numeric_base, availability_state, horizon_state, task_state, cadence_state],
                    dim=-1,
                )
            )
        )
        gate = gate * float(semantic_residual_scale)
        if semantic_cell_scale is not None:
            gate = gate * semantic_cell_scale.to(device=gate.device, dtype=gate.dtype)
        semantic_update = torch.tanh(self.semantic_delta(llm_state))
        return numeric_base + gate * semantic_update, gate


class SharedTemporalBlock(nn.Module):
    """A shared local-temporal block with future-to-history retrieval."""

    def __init__(self, d_model: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.history_attention_norm = nn.LayerNorm(d_model)
        self.history_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4 if d_model % 4 == 0 else 1,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, *, history_length: int) -> torch.Tensor:
        temporal = self.temporal_norm(hidden).transpose(1, 2)
        temporal = self.pointwise(F.gelu(self.depthwise(temporal))).transpose(1, 2)
        hidden = hidden + self.dropout(temporal)

        # Every forecast query retrieves from the entire observed history. This
        # preserves one shared backbone while avoiding a 31-step receptive-field
        # ceiling for the 1008-step wind windows.
        attention_input = self.history_attention_norm(hidden)
        history = attention_input[:, :history_length]
        future = attention_input[:, history_length:]
        if future.shape[1] > 0:
            retrieved, _ = self.history_attention(
                query=future,
                key=history,
                value=history,
                need_weights=False,
            )
            hidden = torch.cat(
                [hidden[:, :history_length], hidden[:, history_length:] + self.dropout(retrieved)],
                dim=1,
            )
        hidden = hidden + self.dropout(self.channel_mlp(self.channel_norm(hidden)))
        return hidden


class RankGatedResidualAdapter(nn.Module):
    """Global low-rank bases controlled by per-window rank and strength gates."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rank = config.rank
        self.adapter_A = nn.Linear(config.d_model, config.rank, bias=False)
        self.adapter_B = nn.Linear(config.rank, config.d_model, bias=False)
        self.rank_controller = nn.Linear(config.d_residual, config.rank)
        self.strength_controller = nn.Linear(config.d_residual, 1)
        nn.init.kaiming_uniform_(self.adapter_A.weight, a=5**0.5)
        nn.init.zeros_(self.adapter_B.weight)
        nn.init.zeros_(self.strength_controller.weight)
        nn.init.constant_(self.strength_controller.bias, config.alpha_init_bias)

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

        low_rank = self.adapter_A(hidden) * gamma.unsqueeze(1)
        delta = self.adapter_B(low_rank)
        if enabled:
            adapted = hidden + alpha.unsqueeze(1) * delta
        else:
            adapted = hidden
            delta = torch.zeros_like(delta)
            alpha = torch.zeros_like(alpha)
            gamma = torch.zeros_like(gamma)
        residual_energy = delta.square().mean(dim=(1, 2))
        return adapted, alpha.squeeze(-1), gamma, residual_energy


class SharedEnergyCALLM(nn.Module):
    """A single shared model for all four paper task families."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.token_stem = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.d_model),
        )
        self.position = SinusoidalPositionEncoding(cfg.d_model)
        self.task_embedding = nn.Embedding(cfg.num_tasks, cfg.d_model)
        self.cadence_embedding = nn.Embedding(cfg.num_cadences, cfg.d_model)
        self.physical_lead_time = (
            PhysicalLeadTimeEncoding(cfg.d_model) if cfg.use_physical_lead_time else None
        )
        self.context_encoder = SemanticContextEncoder(cfg)
        self.method_variant = cfg.method_variant
        if self.method_variant == "aaai_exact":
            self.aaai_semantic_projection = AAAIExactSemanticProjection(
                cfg.llm_context_dim,
                cfg.d_context,
            )
            self.aaai_schema_norm = nn.LayerNorm(
                cfg.schema_context_dim,
                elementwise_affine=False,
            )
            self.aaai_schema_mlp = AAAIExactSchemaMLP(
                cfg.schema_context_dim,
                cfg.availability_dim,
                cfg.d_context,
            )
            self.aaai_attention_pool = AAAIExactAttentionPool(cfg.d_model)
            self.aaai_residual_controller = AAAIExactResidualController(
                d_context=cfg.d_context,
                d_model=cfg.d_model,
                d_residual=cfg.d_residual,
                availability_dim=cfg.availability_dim,
            )
        self.task_asset_encoder = nn.Sequential(
            nn.Linear(2 * cfg.d_model + 5, cfg.d_context),
            nn.GELU(),
            nn.Linear(cfg.d_context, cfg.d_context),
        )
        self.residual_code = nn.Sequential(
            nn.LayerNorm(cfg.d_context + cfg.d_model),
            nn.Linear(cfg.d_context + cfg.d_model, 2 * cfg.d_residual),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.d_residual, cfg.d_residual),
        )
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                SharedTemporalBlock(cfg.d_model, dilation=dilations[index], dropout=cfg.dropout)
                for index in range(cfg.num_layers)
            ]
        )
        if self.method_variant == "aaai_exact":
            self.adapters = nn.ModuleList(
                [
                    AAAIExactRankGatedResidualAdapter(
                        d_model=cfg.d_model,
                        d_residual=cfg.d_residual,
                        rank=cfg.rank,
                        alpha_init_bias=cfg.alpha_init_bias,
                    )
                    for _ in range(cfg.num_layers)
                ]
            )
        else:
            self.adapters = nn.ModuleList(
                [RankGatedResidualAdapter(cfg) for _ in range(cfg.num_layers)]
            )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        if cfg.use_quantile_head:
            levels = torch.tensor(cfg.quantile_levels, dtype=torch.float32)
            normal_scores = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2.0 * levels - 1.0)
            self.register_buffer("quantile_levels", levels, persistent=True)
            self.register_buffer("quantile_normal_scores", normal_scores, persistent=True)
            self.quantile_scale_head = nn.Linear(cfg.d_model, 1)
            nn.init.zeros_(self.quantile_scale_head.weight)
            nn.init.constant_(self.quantile_scale_head.bias, -3.0)
        self.decoder_variant = cfg.decoder_variant
        if self.decoder_variant == "contextual_mlp":
            decoder_input_dim = cfg.token_dim + 4 * cfg.d_model + cfg.d_context
            self.contextual_decoder = nn.Sequential(
                nn.LayerNorm(decoder_input_dim),
                nn.Linear(decoder_input_dim, 2 * cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(2 * cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )
            nn.init.zeros_(self.contextual_decoder[-1].weight)
            nn.init.zeros_(self.contextual_decoder[-1].bias)
        if cfg.use_direct_semantic_residual:
            if cfg.direct_residual_source == "numeric":
                self.direct_numeric_projection = nn.Sequential(
                    nn.LayerNorm(cfg.numeric_context_dim),
                    nn.Linear(cfg.numeric_context_dim, cfg.d_context, bias=False),
                )
            condition_dim = 4 * cfg.d_model
            self.direct_semantic_conditioner = nn.Sequential(
                nn.LayerNorm(condition_dim),
                nn.Linear(condition_dim, cfg.d_context),
                nn.GELU(),
            )
            direct_output_dim = cfg.num_tasks if cfg.direct_residual_head == "task_specific" else 1
            self.direct_semantic_decoder = nn.Sequential(
                nn.LayerNorm(cfg.d_context, elementwise_affine=False),
                nn.Linear(cfg.d_context, cfg.d_context, bias=False),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_context, direct_output_dim, bias=False),
            )
            nn.init.zeros_(self.direct_semantic_decoder[-1].weight)
            if cfg.use_field_conditioned_semantic:
                field_input_dim = cfg.numeric_context_dim + cfg.availability_dim
                self.direct_semantic_field_gate = nn.Sequential(
                    nn.LayerNorm(field_input_dim),
                    nn.Linear(field_input_dim, cfg.d_context),
                    nn.GELU(),
                    nn.Linear(cfg.d_context, cfg.d_context),
                )
                nn.init.zeros_(self.direct_semantic_field_gate[-1].weight)
                nn.init.zeros_(self.direct_semantic_field_gate[-1].bias)
        self.seasonal_anchor_mode = cfg.seasonal_anchor_mode
        if cfg.use_issued_target_anchor:
            issued_logit = torch.logit(torch.tensor(float(cfg.issued_anchor_init)))
            self.issued_anchor_logit = nn.Parameter(issued_logit.repeat(cfg.num_tasks))
        else:
            self.register_parameter("issued_anchor_logit", None)
        if self.seasonal_anchor_mode == "gated_daily_weekly":
            anchor_input_dim = cfg.d_context + 3 * cfg.d_model
            self.seasonal_anchor_gate = nn.Sequential(
                nn.LayerNorm(anchor_input_dim),
                nn.Linear(anchor_input_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )
            nn.init.zeros_(self.seasonal_anchor_gate[-1].weight)
            nn.init.zeros_(self.seasonal_anchor_gate[-1].bias)

    def _base_summary(self, hidden: torch.Tensor, history_len: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = positions < history_len.unsqueeze(1)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def direct_semantic_parameters(self):
        if not self.config.use_direct_semantic_residual:
            return
        yield from self.direct_semantic_conditioner.parameters()
        yield from self.direct_semantic_decoder.parameters()
        if self.config.direct_residual_source == "numeric":
            yield from self.direct_numeric_projection.parameters()
        if self.config.use_field_conditioned_semantic:
            yield from self.direct_semantic_field_gate.parameters()

    def _semantic_state(
        self,
        numeric_context: torch.Tensor,
        schema_context: torch.Tensor,
        llm_context: torch.Tensor,
        availability: torch.Tensor,
        task_state: torch.Tensor,
        cadence_state: torch.Tensor,
        horizon: torch.Tensor,
        task_id: torch.Tensor,
        cadence_id: torch.Tensor,
        mode: ContextMode,
        semantic_residual_scale: float,
        semantic_cell_policy: SemanticCellPolicy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_gate = torch.zeros(
            numeric_context.shape[0],
            self.config.d_context,
            device=numeric_context.device,
            dtype=numeric_context.dtype,
        )
        if mode == "no_context":
            return torch.zeros_like(zero_gate), zero_gate
        if mode == "task_asset":
            asset_fields = numeric_context[:, [8, 21, 22, 23, 24]]
            return self.task_asset_encoder(torch.cat([task_state, cadence_state, asset_fields], dim=-1)), zero_gate
        if mode == "mlp_schema":
            if self.method_variant == "aaai_exact":
                return self.aaai_schema_mlp(schema_context, availability), zero_gate
            return self.context_encoder(
                numeric_context,
                llm_context,
                availability,
                task_state,
                cadence_state,
                horizon,
                cadence_id,
                include_llm=False,
            )
        if mode == "capacity_matched_schema":
            if self.method_variant == "aaai_exact":
                schema_normalized = self.aaai_schema_norm(schema_context)
                repeats = (
                    self.config.llm_context_dim + schema_normalized.shape[1] - 1
                ) // schema_normalized.shape[1]
                expanded = schema_normalized.repeat(1, repeats)[
                    :, : self.config.llm_context_dim
                ]
                return self.aaai_semantic_projection(expanded), zero_gate
            return self.context_encoder(
                numeric_context,
                llm_context,
                availability,
                task_state,
                cadence_state,
                horizon,
                cadence_id,
                include_llm=False,
                numeric_as_llm=True,
                semantic_residual_scale=semantic_residual_scale,
            )
        if mode == "full" and self.method_variant == "aaai_exact":
            return self.aaai_semantic_projection(llm_context), zero_gate
        if mode == "full":
            if self.config.use_direct_semantic_residual:
                return self.context_encoder(
                    numeric_context,
                    llm_context,
                    availability,
                    task_state,
                    cadence_state,
                    horizon,
                    cadence_id,
                    include_llm=False,
                )
            cell_scale = semantic_cell_mask(
                task_id,
                horizon,
                cadence_id,
                policy=semantic_cell_policy,
                dtype=numeric_context.dtype,
            )
            return self.context_encoder(
                numeric_context,
                llm_context,
                availability,
                task_state,
                cadence_state,
                horizon,
                cadence_id,
                include_llm=True,
                semantic_residual_scale=semantic_residual_scale,
                semantic_cell_scale=cell_scale,
            )
        raise ValueError(f"Unknown context mode: {mode}")

    def forward(
        self,
        *,
        tokens: torch.Tensor,
        numeric_context: torch.Tensor,
        llm_context: torch.Tensor,
        availability: torch.Tensor,
        history_len: torch.Tensor,
        horizon: torch.Tensor,
        task_id: torch.Tensor,
        cadence_id: torch.Tensor,
        schema_context: torch.Tensor | None = None,
        controls: ForwardControls | None = None,
    ) -> dict[str, torch.Tensor]:
        controls = controls or ForwardControls()
        if schema_context is None:
            repeats = (
                self.config.schema_context_dim + numeric_context.shape[1] - 1
            ) // numeric_context.shape[1]
            schema_context = numeric_context.repeat(1, repeats)[
                :, : self.config.schema_context_dim
            ]
        if torch.any(horizon != horizon[0]):
            raise ValueError("Each batch must be task-homogeneous with one horizon")
        if torch.any(history_len != history_len[0]):
            raise ValueError("Each batch must be task-homogeneous with one history length")

        if not controls.use_exogenous_context:
            tokens = tokens.clone()
            exog_start = 5
            exog_end = exog_start + 2 * self.config.max_exog_dim
            tokens[..., exog_start:exog_end] = 0.0
        hidden = self.token_stem(tokens)
        hidden = hidden + self.position(hidden.shape[1], dtype=hidden.dtype).unsqueeze(0)
        task_state = self.task_embedding(task_id)
        cadence_state = self.cadence_embedding(cadence_id)
        if controls.use_task_interface:
            hidden = hidden + task_state.unsqueeze(1) + cadence_state.unsqueeze(1)
        lead_time_state = None
        if self.physical_lead_time is not None:
            horizon_len = int(horizon[0].item())
            hidden = hidden.clone()
            lead_time_state = self.physical_lead_time(
                cadence_id,
                horizon_len,
                dtype=hidden.dtype,
            )
            hidden[:, -horizon_len:] = hidden[:, -horizon_len:] + lead_time_state
        if self.method_variant == "aaai_exact":
            base_summary = self.aaai_attention_pool(hidden, history_len)
        else:
            base_summary = self._base_summary(hidden, history_len)
        semantic_state, semantic_gate = self._semantic_state(
            numeric_context,
            schema_context,
            llm_context,
            availability,
            task_state,
            cadence_state,
            horizon,
            task_id,
            cadence_id,
            controls.context_mode,
            controls.semantic_residual_scale,
            controls.semantic_cell_policy,
        )
        if self.method_variant == "aaai_exact":
            residual_code = self.aaai_residual_controller(
                semantic_state,
                base_summary,
                availability,
            )
        else:
            residual_code = self.residual_code(torch.cat([semantic_state, base_summary], dim=-1))

        alphas: list[torch.Tensor] = []
        gammas: list[torch.Tensor] = []
        residual_energies: list[torch.Tensor] = []
        for block, adapter in zip(self.blocks, self.adapters, strict=True):
            hidden = block(hidden, history_length=int(history_len[0].item()))
            hidden, alpha, gamma, residual_energy = adapter(
                hidden,
                residual_code,
                enabled=controls.use_adapters,
                rank_gate=controls.rank_gate,
                strength_gate=controls.strength_gate,
                fixed_rank_value=controls.fixed_rank_value,
                fixed_strength_value=controls.fixed_strength_value,
            )
            alphas.append(alpha)
            gammas.append(gamma)
            residual_energies.append(residual_energy)

        horizon_len = int(horizon[0].item())
        history_length = int(history_len[0].item())
        cadence_minutes = PhysicalLeadTimeEncoding._CADENCE_MINUTES[int(cadence_id[0].item())]
        daily_period = int(24 * 60 / cadence_minutes)
        if history_length < daily_period:
            raise ValueError("A full daily seasonal cycle must be available in history")
        future_hidden = hidden[:, -horizon_len:]
        last_daily_cycle = tokens[:, history_length - daily_period : history_length, 0]
        repeats = (horizon_len + daily_period - 1) // daily_period
        seasonal_baseline = last_daily_cycle.repeat(1, repeats)[:, :horizon_len]
        if self.seasonal_anchor_mode == "gated_daily_weekly":
            weekly_period = 7 * daily_period
            if history_length < weekly_period:
                raise ValueError("A seven-day seasonal cycle must be available for the gated daily-weekly anchor")
            last_weekly_cycle = tokens[:, history_length - weekly_period : history_length, 0]
            weekly_repeats = (horizon_len + weekly_period - 1) // weekly_period
            weekly_baseline = last_weekly_cycle.repeat(1, weekly_repeats)[:, :horizon_len]
            seasonal_gate_input = torch.cat(
                [semantic_state, base_summary, task_state, cadence_state],
                dim=-1,
            )
            seasonal_gate = torch.sigmoid(self.seasonal_anchor_gate(seasonal_gate_input))
            seasonal_baseline = (1.0 - seasonal_gate) * seasonal_baseline + seasonal_gate * weekly_baseline
        else:
            seasonal_gate = torch.zeros(
                tokens.shape[0],
                1,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        issued_anchor_gate = torch.zeros_like(seasonal_gate)
        if self.config.use_issued_target_anchor:
            future_tokens = tokens[:, -horizon_len:]
            anchor_positions = (5 + task_id).view(-1, 1, 1).expand(-1, horizon_len, 1)
            issued_anchor = torch.gather(future_tokens, dim=2, index=anchor_positions).squeeze(-1)
            issued_anchor_gate = (
                torch.sigmoid(self.issued_anchor_logit[task_id]).unsqueeze(1)
                * availability[:, -1:].to(dtype=tokens.dtype)
            )
            mask_positions = (5 + self.config.max_exog_dim + task_id).view(-1, 1, 1).expand(-1, horizon_len, 1)
            issued_anchor_gate = issued_anchor_gate * torch.gather(future_tokens, 2, mask_positions).squeeze(-1)
            seasonal_baseline = (
                (1.0 - issued_anchor_gate) * seasonal_baseline
                + issued_anchor_gate * issued_anchor
            )
        residual = self.decoder(self.decoder_norm(future_hidden)).squeeze(-1)
        if self.decoder_variant == "contextual_mlp":
            direct_inputs = torch.cat(
                [
                    tokens[:, -horizon_len:],
                    future_hidden,
                    base_summary.unsqueeze(1).expand(-1, horizon_len, -1),
                    task_state.unsqueeze(1).expand(-1, horizon_len, -1),
                    cadence_state.unsqueeze(1).expand(-1, horizon_len, -1),
                    semantic_state.unsqueeze(1).expand(-1, horizon_len, -1),
                ],
                dim=-1,
            )
            residual = residual + self.contextual_decoder(direct_inputs).squeeze(-1)
        direct_semantic_residual = torch.zeros_like(residual)
        if self.config.use_direct_semantic_residual and controls.context_mode == "full":
            if lead_time_state is None:
                raise RuntimeError("direct semantic residuals require physical lead-time encoding")
            if self.config.direct_residual_source == "numeric":
                source_state = self.direct_numeric_projection(numeric_context.detach())
            else:
                source_state = self.context_encoder.llm_projection(
                    self.context_encoder.llm_norm(llm_context.detach())
                )
            direct_condition = self.direct_semantic_conditioner(
                torch.cat(
                    [
                        future_hidden,
                        task_state.unsqueeze(1).expand(-1, horizon_len, -1),
                        cadence_state.unsqueeze(1).expand(-1, horizon_len, -1),
                        lead_time_state,
                    ],
                    dim=-1,
                )
            )
            semantic_vector = torch.tanh(source_state)
            if self.config.use_field_conditioned_semantic:
                field_gate = 2.0 * torch.sigmoid(
                    self.direct_semantic_field_gate(
                        torch.cat([numeric_context, availability], dim=-1)
                    )
                )
                semantic_vector = semantic_vector * field_gate
            direct_logits = self.direct_semantic_decoder(
                semantic_vector.unsqueeze(1) * direct_condition
            )
            if self.config.direct_residual_head == "task_specific":
                task_positions = task_id.view(-1, 1, 1).expand(-1, horizon_len, 1)
                direct_logits = torch.gather(direct_logits, dim=-1, index=task_positions)
            direct_semantic_residual = 0.1 * torch.tanh(direct_logits.squeeze(-1))
            cell_scale = semantic_cell_mask(
                task_id,
                horizon,
                cadence_id,
                policy=controls.semantic_cell_policy,
                dtype=residual.dtype,
            )
            direct_semantic_residual = (
                direct_semantic_residual
                * float(controls.semantic_residual_scale)
                * cell_scale
            )
        base_raw_prediction = seasonal_baseline + residual
        raw_prediction = base_raw_prediction + direct_semantic_residual
        prediction = apply_task_output_modes(
            raw_prediction,
            task_id,
            self.config.task_output_modes,
        )
        quantiles = None
        quantile_scale = None
        if self.config.use_quantile_head:
            quantile_scale = F.softplus(self.quantile_scale_head(self.decoder_norm(future_hidden)).squeeze(-1)) + 1e-4
            raw_quantiles = (
                raw_prediction.unsqueeze(-1)
                + quantile_scale.unsqueeze(-1) * self.quantile_normal_scores.to(dtype=prediction.dtype)
            )
            quantiles = apply_task_output_modes(
                raw_quantiles,
                task_id,
                self.config.task_output_modes,
            )
        alpha_tensor = torch.stack(alphas, dim=1)
        gamma_tensor = torch.stack(gammas, dim=1)
        activation = alpha_tensor.unsqueeze(-1) * gamma_tensor.abs()
        residual_penalty = torch.stack(residual_energies, dim=1).mean()
        output = {
            "prediction": prediction,
            "alpha": alpha_tensor,
            "gamma": gamma_tensor,
            "activation": activation,
            "residual_penalty": residual_penalty,
            "semantic_state": semantic_state,
            "semantic_gate": semantic_gate,
            "residual_code": residual_code,
            "seasonal_gate": seasonal_gate,
            "issued_anchor_gate": issued_anchor_gate,
            "base_raw_prediction": base_raw_prediction,
            "direct_semantic_residual": direct_semantic_residual,
        }
        if quantiles is not None and quantile_scale is not None:
            output["quantiles"] = quantiles
            output["quantile_scale"] = quantile_scale
            output["quantile_levels"] = self.quantile_levels
        return output

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        adapter = sum(parameter.numel() for module in self.adapters for parameter in module.parameters())
        return {"total": total, "trainable": trainable, "adapter": adapter}
