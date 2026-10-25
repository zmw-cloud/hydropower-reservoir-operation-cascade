"""Monthly dual-site synthetic streamflow generator."""

from .configuration import GeneratorConfig, ScenarioConfig, load_configuration
from .io_utils import MonthlyHistory, load_monthly_history
from .joint_monthly import (
    GeneratedScenario,
    JointMonthlyModel,
    generate_scenario,
    prepare_joint_monthly_model,
)

__all__ = [
    "GeneratedScenario",
    "GeneratorConfig",
    "JointMonthlyModel",
    "MonthlyHistory",
    "ScenarioConfig",
    "generate_scenario",
    "load_configuration",
    "load_monthly_history",
    "prepare_joint_monthly_model",
]
