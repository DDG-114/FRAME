"""Configuration objects for the paper-faithful EnergyCA-LLM experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset: str
    task_id: int
    cadence_id: int
    cadence_minutes: int
    history_len: int
    horizon: int
    stride: int
    max_nodes: int
    train_windows_per_node: int
    eval_windows_per_node: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paper_task_specs(
    *,
    train_windows_per_node: int | None = None,
    eval_windows_per_node: int | None = None,
    max_nodes_override: int | None = None,
    history_days: int = 7,
    horizon_days: int = 1,
    forecast_stride_days: int = 1,
) -> dict[str, TaskSpec]:
    """Return native-cadence task protocols on a common physical-time scale.

    The defaults recover the AAAI protocol: seven days of history, one-day
    targets, and one daily forecast origin. The TSTE extension changes the
    physical prediction horizon while retaining each task's native cadence.
    """

    for name, value in (
        ("history_days", history_days),
        ("horizon_days", horizon_days),
        ("forecast_stride_days", forecast_stride_days),
    ):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive whole number of days")
    history_days = int(history_days)
    horizon_days = int(horizon_days)
    forecast_stride_days = int(forecast_stride_days)

    raw = {
        "load": dict(
            dataset="gefc2014_load_task15",
            task_id=0,
            cadence_id=0,
            cadence_minutes=60,
            history_len=24 * history_days,
            horizon=24 * horizon_days,
            stride=24 * forecast_stride_days,
            max_nodes=1,
            train_windows_per_node=1024,
            eval_windows_per_node=256,
        ),
        "wind": dict(
            dataset="sdwpf",
            task_id=1,
            cadence_id=1,
            cadence_minutes=10,
            history_len=144 * history_days,
            horizon=144 * horizon_days,
            stride=144 * forecast_stride_days,
            max_nodes=16,
            train_windows_per_node=512,
            eval_windows_per_node=128,
        ),
        "pv": dict(
            dataset="mmsp_fusionsf",
            task_id=2,
            cadence_id=0,
            cadence_minutes=60,
            history_len=24 * history_days,
            horizon=24 * horizon_days,
            stride=24 * forecast_stride_days,
            max_nodes=20,
            train_windows_per_node=1024,
            eval_windows_per_node=256,
        ),
        "net_load": dict(
            dataset="gb_gsp_netload",
            task_id=3,
            cadence_id=2,
            cadence_minutes=30,
            history_len=48 * history_days,
            horizon=48 * horizon_days,
            stride=48 * forecast_stride_days,
            max_nodes=1,
            train_windows_per_node=1024,
            eval_windows_per_node=256,
        ),
    }
    specs: dict[str, TaskSpec] = {}
    for key, values in raw.items():
        if train_windows_per_node is not None:
            values["train_windows_per_node"] = int(train_windows_per_node)
        if eval_windows_per_node is not None:
            values["eval_windows_per_node"] = int(eval_windows_per_node)
        if max_nodes_override is not None:
            values["max_nodes"] = min(values["max_nodes"], int(max_nodes_override))
        specs[key] = TaskSpec(key=key, **values)
    return specs


def aemo_task_specs(
    *,
    history_days: int = 7,
    horizon_days: int = 1,
    forecast_stride_days: int = 1,
    train_windows_per_node: int | None = None,
    eval_windows_per_node: int | None = None,
    max_nodes_override: int | None = None,
) -> dict[str, TaskSpec]:
    """Return the common-cadence, common-region AEMO four-task protocol."""

    for name, value in (
        ("history_days", history_days),
        ("horizon_days", horizon_days),
        ("forecast_stride_days", forecast_stride_days),
    ):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive whole number of days")
    history_len = 48 * int(history_days)
    horizon = 48 * int(horizon_days)
    stride = 48 * int(forecast_stride_days)
    specs: dict[str, TaskSpec] = {}
    for task_id, key in enumerate(("load", "wind", "pv", "net_load")):
        specs[key] = TaskSpec(
            key=key,
            dataset=f"aemo_{key}",
            task_id=task_id,
            cadence_id=2,
            cadence_minutes=30,
            history_len=history_len,
            horizon=horizon,
            stride=stride,
            max_nodes=min(5, int(max_nodes_override)) if max_nodes_override is not None else 5,
            train_windows_per_node=(
                int(train_windows_per_node) if train_windows_per_node is not None else 1024
            ),
            eval_windows_per_node=(
                int(eval_windows_per_node) if eval_windows_per_node is not None else 256
            ),
        )
    return specs


def perform_task_specs(
    *,
    history_days: int = 7,
    horizon_days: int = 2,
    forecast_stride_days: int = 1,
    train_windows_per_node: int | None = None,
    eval_windows_per_node: int | None = None,
) -> dict[str, TaskSpec]:
    """Return the synchronized ERCOT PERFORM day-ahead probability protocol."""

    if horizon_days not in {1, 2}:
        raise ValueError("PERFORM day-ahead evaluation supports 1- or 2-day horizons")
    for name, value in (("history_days", history_days), ("forecast_stride_days", forecast_stride_days)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive whole number of days")
    specs: dict[str, TaskSpec] = {}
    for task_id, key in enumerate(("load", "wind", "pv", "net_load")):
        specs[key] = TaskSpec(
            key=key,
            dataset=f"perform_{key}",
            task_id=task_id,
            cadence_id=0,
            cadence_minutes=60,
            history_len=24 * int(history_days),
            horizon=24 * int(horizon_days),
            stride=24 * int(forecast_stride_days),
            max_nodes=1,
            train_windows_per_node=int(train_windows_per_node) if train_windows_per_node is not None else 1024,
            eval_windows_per_node=int(eval_windows_per_node) if eval_windows_per_node is not None else 256,
        )
    return specs


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = 57
    numeric_context_dim: int = 69
    schema_context_dim: int = 256
    llm_context_dim: int = 4096
    availability_dim: int = 8
    max_exog_dim: int = 25
    d_model: int = 128
    d_context: int = 128
    d_residual: int = 128
    num_layers: int = 4
    rank: int = 8
    num_tasks: int = 4
    num_cadences: int = 3
    dropout: float = 0.1
    alpha_init_bias: float = -2.0
    semantic_gate_init_bias: float = -2.0
    use_direct_semantic_residual: bool = False
    direct_residual_source: str = "qwen"
    direct_residual_head: str = "shared"
    use_field_conditioned_semantic: bool = False
    method_variant: str = "current"
    decoder_variant: str = "seasonal_linear"
    seasonal_anchor_mode: str = "daily"
    use_physical_lead_time: bool = False
    use_quantile_head: bool = False
    use_issued_target_anchor: bool = False
    issued_anchor_init: float = 0.9
    task_output_modes: tuple[str, ...] = ("relu", "relu", "relu", "relu")
    quantile_levels: tuple[float, ...] = (
        0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
        0.95, 0.975, 0.99,
    )

    def __post_init__(self) -> None:
        expected = 1 + 4 + 2 * self.max_exog_dim + 2
        if self.token_dim != expected:
            raise ValueError(f"token_dim must be {expected} for max_exog_dim={self.max_exog_dim}")
        if self.num_layers != 4:
            raise ValueError("The paper contract requires exactly four adapted layers")
        if self.rank not in {2, 4, 8, 16}:
            raise ValueError("Adapter rank must be one of 2, 4, 8, or 16")
        if not -10.0 <= float(self.semantic_gate_init_bias) <= 0.0:
            raise ValueError("semantic_gate_init_bias must lie between -10 and 0")
        if self.use_direct_semantic_residual and not self.use_physical_lead_time:
            raise ValueError("direct semantic residuals require physical lead-time encoding")
        if self.direct_residual_source not in {"qwen", "numeric"}:
            raise ValueError("direct_residual_source must be 'qwen' or 'numeric'")
        if self.direct_residual_head not in {"shared", "task_specific"}:
            raise ValueError("direct_residual_head must be 'shared' or 'task_specific'")
        if self.use_field_conditioned_semantic and not self.use_direct_semantic_residual:
            raise ValueError("field-conditioned semantics require direct semantic residuals")
        if self.use_field_conditioned_semantic and self.direct_residual_source != "qwen":
            raise ValueError("field-conditioned semantics require the Qwen residual source")
        if self.method_variant not in {"current", "aaai_exact"}:
            raise ValueError("method_variant must be 'current' or 'aaai_exact'")
        if self.decoder_variant not in {"seasonal_linear", "contextual_mlp"}:
            raise ValueError("decoder_variant must be 'seasonal_linear' or 'contextual_mlp'")
        if self.seasonal_anchor_mode not in {"daily", "gated_daily_weekly"}:
            raise ValueError("seasonal_anchor_mode must be 'daily' or 'gated_daily_weekly'")
        if len(self.task_output_modes) != self.num_tasks:
            raise ValueError("task_output_modes must provide one mode per task")
        valid_output_modes = {"relu", "softplus", "unit_interval", "identity"}
        invalid_modes = set(self.task_output_modes) - valid_output_modes
        if invalid_modes:
            raise ValueError(f"Unknown task output modes: {sorted(invalid_modes)}")
        if any(not 0.0 < float(level) < 1.0 for level in self.quantile_levels):
            raise ValueError("quantile_levels must lie strictly between zero and one")
        if self.use_issued_target_anchor and self.max_exog_dim < self.num_tasks:
            raise ValueError("issued target anchors require one future exogenous field per task")
        if not 0.0 < float(self.issued_anchor_init) < 1.0:
            raise ValueError("issued_anchor_init must lie strictly between zero and one")
        if tuple(sorted(float(level) for level in self.quantile_levels)) != tuple(float(level) for level in self.quantile_levels):
            raise ValueError("quantile_levels must be strictly increasing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
