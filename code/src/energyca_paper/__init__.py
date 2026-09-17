"""Paper-faithful shared EnergyCA-LLM implementation."""

from .config import ModelConfig, TaskSpec, paper_task_specs
from .model import SharedEnergyCALLM

__all__ = ["ModelConfig", "SharedEnergyCALLM", "TaskSpec", "paper_task_specs"]

