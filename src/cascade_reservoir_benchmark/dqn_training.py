"""Single-seed training orchestration for the P4 cascade Double DQN.

Only the configured 400-year training sequence is accepted by this module.
Development mode uses its first 320 years for normalization and learning and
its final 80 years as one continuous greedy validation trajectory.  Final mode
fits and trains from scratch on all 400 years and performs no model selection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .dqn_adapter import (
    INFLOW_COLUMNS,
    STATE_ORDER,
    CascadeDQNAdapter,
    JointActionGrid,
    StateNormalizer,
    normalized_inflow_sha256,
    sample_january_start_index,
    slice_episode_window,
)
from .dqn_agent import ACTION_MASK_MODE, DoubleDQNAgent
from .dqn_agent import ACTION_CONDITIONED_MODE, INDEXED_OUTPUT_MODE
from .dqn_evaluation import evaluate_greedy_trajectory
from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    _as_month_period_index,
)
from .sdp import reservoir_pair_sha256


CHECKPOINT_SCHEMA_VERSION = 3
TRAINING_MODES = ("development", "final")
DEVELOPMENT_YEARS = 320
VALIDATION_YEARS = 80
TOTAL_TRAINING_YEARS = DEVELOPMENT_YEARS + VALIDATION_YEARS
EPISODE_MONTHS = 240
DQN_REWARD_SCALE_GWH = 7332.48

TRAINING_HISTORY_COLUMNS = (
    "episode",
    "episode_start_index",
    "episode_start_date",
    "episode_end_date",
    "episode_transitions",
    "environment_steps",
    "learning_steps",
    "learning_updates",
    "raw_episode_cascade_generation_gwh",
    "raw_episode_penalized_objective_gwh",
    "episode_failed_station_months",
    "episode_failure_penalty_gwh",
    "normalized_episode_reward",
    "reward_scale_gwh",
    "mean_loss",
    "last_loss",
    "mean_predicted_q",
    "mean_target_q",
    "mean_gradient_norm",
    "epsilon",
    "mean_feasible_action_count",
    "min_feasible_action_count",
    "xiaowan_target_unreached_count",
    "xiaowan_target_unreached_rate",
    "nuozhadu_target_unreached_count",
    "nuozhadu_target_unreached_rate",
    "either_reservoir_target_unreached_count",
    "either_reservoir_target_unreached_rate",
    "validation_performed",
    "checkpoint_event",
)

VALIDATION_HISTORY_COLUMNS = (
    "episode",
    "environment_steps",
    "learning_steps",
    "validation_wall_time_seconds",
    "validation_months",
    "validation_total_cascade_generation_gwh",
    "validation_mean_annual_cascade_generation_gwh",
    "validation_total_penalized_objective_gwh",
    "validation_mean_annual_penalized_objective_gwh",
    "validation_failed_station_months",
    "xiaowan_firm_power_reliability",
    "nuozhadu_firm_power_reliability",
    "cascade_firm_power_reliability",
    "xiaowan_target_unreached_count",
    "xiaowan_target_unreached_rate",
    "nuozhadu_target_unreached_count",
    "nuozhadu_target_unreached_rate",
    "either_reservoir_target_unreached_count",
    "either_reservoir_target_unreached_rate",
    "score",
    "selected_best_validation_checkpoint",
)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _validate_frozen_reward_scale(metadata: Mapping[str, Any]) -> None:
    if not np.isclose(
        float(metadata["scale_gwh"]),
        DQN_REWARD_SCALE_GWH,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise CascadeSpecError(
            "Shared DQN reward scaling no longer equals 7332.48 GWh."
        )


@dataclass(frozen=True)
class DQNTrainingPaths:
    """Deterministic output locations for one mode and random seed."""

    output_root: Path
    run_root: Path
    experiment_id: str | None
    configuration_snapshot_json: Path
    metadata_json: Path
    training_history_csv: Path
    validation_history_csv: Path | None
    latest_checkpoint: Path
    best_validation_checkpoint: Path | None
    final_checkpoint: Path | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "experiment_id": self.experiment_id,
            "configuration_snapshot_json": str(self.configuration_snapshot_json),
            "metadata_json": str(self.metadata_json),
            "training_history_csv": str(self.training_history_csv),
            "validation_history_csv": (
                None
                if self.validation_history_csv is None
                else str(self.validation_history_csv)
            ),
            "latest_checkpoint": str(self.latest_checkpoint),
            "best_validation_checkpoint": (
                None
                if self.best_validation_checkpoint is None
                else str(self.best_validation_checkpoint)
            ),
            "final_checkpoint": (
                None if self.final_checkpoint is None else str(self.final_checkpoint)
            ),
        }


@dataclass(frozen=True)
class DQNTrainingResult:
    """Completed or resumed single-seed training outputs and in-memory records."""

    mode: str
    experiment_id: str | None
    seed: int
    completed_episodes: int
    environment_steps: int
    learning_steps: int
    learning_starts: int
    training_wall_time_seconds: float
    validation_wall_time_seconds: float
    best_validation_score: float | None
    best_validation_episode: int | None
    configuration_sha256: str
    normalization: dict[str, Any]
    action_grid: dict[str, Any]
    reward_scaling: dict[str, Any]
    training_inflow_sha256: str
    source_training_inflow_sha256: str
    paths: DQNTrainingPaths
    selected_checkpoint: Path
    training_records: tuple[dict[str, Any], ...]
    validation_records: tuple[dict[str, Any], ...]
    budget_extension_provenance: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "completed_episodes": self.completed_episodes,
            "environment_steps": self.environment_steps,
            "learning_steps": self.learning_steps,
            "learning_starts": self.learning_starts,
            "training_wall_time_seconds": self.training_wall_time_seconds,
            "validation_wall_time_seconds": self.validation_wall_time_seconds,
            "best_validation_score": self.best_validation_score,
            "best_validation_episode": self.best_validation_episode,
            "configuration_sha256": self.configuration_sha256,
            "normalization": copy.deepcopy(self.normalization),
            "action_grid": copy.deepcopy(self.action_grid),
            "reward_scaling": copy.deepcopy(self.reward_scaling),
            "training_inflow_sha256": self.training_inflow_sha256,
            "source_training_inflow_sha256": self.source_training_inflow_sha256,
            "paths": self.paths.as_dict(),
            "selected_checkpoint": str(self.selected_checkpoint),
            "training_records": [dict(record) for record in self.training_records],
            "validation_records": [dict(record) for record in self.validation_records],
            "budget_extension_provenance": copy.deepcopy(
                self.budget_extension_provenance
            ),
        }


@dataclass(frozen=True)
class _TrainingSettings:
    mode: str
    experiment_id: str | None
    seed: int
    episodes: int
    episode_months: int
    initial_levels_m: tuple[float, float]
    action_level_step_m: float
    development_years: int
    validation_years: int
    validation_interval_episodes: int
    validation_score: str
    firm_power_failure_penalty_gwh: float
    reward_reference_days: float
    reward_clip: tuple[float, float] | None
    normalization_clip: bool
    hidden_dimensions: tuple[int, ...]
    q_network_mode: str
    action_score_chunk_size: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_schedule: str
    epsilon_decay_steps: int
    target_update_interval: int
    replay_capacity: int
    batch_size: int
    learning_starts: int
    max_grad_norm: float | None
    device: str
    learn_every_steps: int
    updates_per_learning_event: int
    logging_interval_episodes: int
    checkpoint_interval_episodes: int


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CascadeSpecError(f"DQN configuration section {key!r} must be a mapping.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CascadeSpecError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise CascadeSpecError(f"{name} must be a positive integer.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CascadeSpecError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result < 0:
        raise CascadeSpecError(f"{name} must be a nonnegative integer.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CascadeSpecError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CascadeSpecError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise CascadeSpecError(f"{name} must be a finite number.")
    return result


def _plain_snapshot(value: Any) -> Any:
    """Convert a YAML-style configuration to stable JSON-compatible values."""
    if isinstance(value, Mapping):
        return {
            str(key): _plain_snapshot(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return [_plain_snapshot(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise CascadeSpecError(
        f"DQN configuration contains an unsupported value of type {type(value).__name__}."
    )


def configuration_sha256(config_snapshot: Mapping[str, Any]) -> str:
    """Hash the complete canonical DQN configuration snapshot."""
    canonical = _plain_snapshot(config_snapshot)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _budget_extension_semantics_snapshot(
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize the three fields allowed to differ for budget extension."""
    snapshot = copy.deepcopy(dict(config_snapshot))
    snapshot.pop("budget_extension", None)
    snapshot["experiment_id"] = "<budget-extension-experiment>"
    training = snapshot.get("training")
    outputs = snapshot.get("outputs")
    if not isinstance(training, Mapping) or not isinstance(outputs, Mapping):
        raise CascadeSpecError(
            "Budget extension requires training and outputs configuration sections."
        )
    snapshot["training"] = dict(training)
    snapshot["training"]["episodes"] = "<extended-training-budget>"
    snapshot["outputs"] = dict(outputs)
    snapshot["outputs"]["root"] = "<isolated-extension-output>"
    return _plain_snapshot(snapshot)


def _budget_extension_settings(
    config_snapshot: Mapping[str, Any],
    settings: _TrainingSettings,
) -> dict[str, Any]:
    raw = config_snapshot.get("budget_extension")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        raise CascadeSpecError(
            "An explicit budget_extension.enabled=true section is required."
        )
    source_experiment_id = raw.get("source_experiment_id")
    if source_experiment_id != "dqn_1m_l9_run02":
        raise CascadeSpecError(
            "Budget extension source_experiment_id must be dqn_1m_l9_run02."
        )
    source_budget = _positive_int(
        raw.get("source_training_budget_episodes"),
        "budget_extension.source_training_budget_episodes",
    )
    target_budget = _positive_int(
        raw.get("target_training_budget_episodes"),
        "budget_extension.target_training_budget_episodes",
    )
    if source_budget != 3000 or target_budget != 5000:
        raise CascadeSpecError(
            "Run02 budget extension must be exactly 3000 to 5000 episodes."
        )
    if settings.mode != "development" or settings.episodes != target_budget:
        raise CascadeSpecError(
            "Budget extension requires development mode and a 5000-episode target."
        )
    if settings.experiment_id != "dqn_1m_run02_budget_extension_5000":
        raise CascadeSpecError("Budget extension experiment_id is not isolated.")
    return {
        "source_experiment_id": str(source_experiment_id),
        "source_training_budget_episodes": source_budget,
        "target_training_budget_episodes": target_budget,
    }


def _optional_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CascadeSpecError(f"{name} must be a hexadecimal SHA-256 string.")
    result = value.lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise CascadeSpecError(f"{name} must be a hexadecimal SHA-256 string.")
    return result


def experiment_id_from_config(config: Mapping[str, Any]) -> str | None:
    """Return a filesystem-safe experiment identifier, or legacy-mode null."""
    raw = config.get("experiment_id")
    if raw is None:
        return None
    if not isinstance(raw, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", raw
    ):
        raise CascadeSpecError(
            "experiment_id must be null/omitted or match "
            "[A-Za-z0-9][A-Za-z0-9._-]{0,63}."
        )
    return raw


def _parse_reward_clip(reward: Mapping[str, Any]) -> tuple[float, float] | None:
    settings = reward.get("clip")
    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise CascadeSpecError("reward.clip must be a mapping or null.")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise CascadeSpecError("reward.clip.enabled must be boolean.")
    if not enabled:
        return None
    lower = _finite_float(settings.get("lower"), "reward.clip.lower")
    upper = _finite_float(settings.get("upper"), "reward.clip.upper")
    if upper <= lower:
        raise CascadeSpecError("reward.clip.upper must exceed reward.clip.lower.")
    return lower, upper


def _resolve_device(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadeSpecError("training.device must be a nonempty string.")
    requested = value.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise CascadeSpecError(f"Invalid DQN device {value!r}.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CascadeSpecError("A CUDA DQN device was requested but CUDA is unavailable.")
    return str(device)


def _settings_from_config(
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> _TrainingSettings:
    if not isinstance(config, Mapping):
        raise CascadeSpecError("DQN configuration must be a mapping.")
    if mode not in TRAINING_MODES:
        raise CascadeSpecError(f"mode must be one of {TRAINING_MODES}.")
    seed_value = _nonnegative_int(seed, "seed")

    problem = _mapping(config, "problem")
    information = _mapping(config, "information")
    reward = _mapping(config, "reward")
    normalization = _mapping(config, "normalization")
    validation = _mapping(config, "validation")
    agent = _mapping(config, "agent")
    network = _mapping(config, "network")
    training = _mapping(config, "training")

    if validation.get("enabled_in_development_mode") is not True:
        raise CascadeSpecError(
            "DQN V1 requires validation.enabled_in_development_mode=true."
        )
    if information.get("one_step_perfect_forecast") is not True:
        raise CascadeSpecError("DQN V1 requires one_step_perfect_forecast=true.")
    if list(information.get("state_order", [])) != list(STATE_ORDER):
        raise CascadeSpecError("Configured DQN state order does not match the V1 adapter.")
    if information.get("nuozhadu_total_inflow_is_action_dependent") is not True:
        raise CascadeSpecError(
            "DQN V1 requires NuoZhaDu total inflow to remain action-dependent."
        )
    if reward.get("objective") != "generation_gwh_minus_penalty_gwh_per_failed_station_month":
        raise CascadeSpecError("DQN reward objective is incompatible.")
    failure_penalty = _finite_float(
        reward.get("firm_power_failure_penalty_gwh"),
        "reward.firm_power_failure_penalty_gwh",
    )
    if failure_penalty < 0.0:
        raise CascadeSpecError("DQN firm-power failure penalty must be nonnegative.")
    if reward.get("failure_definition") != "station_specific_monthly_power_below_firm_power":
        raise CascadeSpecError("DQN firm-power failure definition is incompatible.")
    if reward.get("deficit_energy_role") != "diagnostic_only_not_in_objective":
        raise CascadeSpecError("DQN deficit energy must remain diagnostic only.")
    if _finite_float(reward.get("scale_gwh"), "reward.scale_gwh") != (
        DQN_REWARD_SCALE_GWH
    ):
        raise CascadeSpecError("DQN reward scale is frozen at 7332.48 GWh.")
    if normalization.get("method") != StateNormalizer.method:
        raise CascadeSpecError("DQN V1 requires componentwise_minmax normalization.")
    normalization_clip = normalization.get("clip")
    if normalization_clip is not True:
        raise CascadeSpecError("DQN V1 requires normalization.clip=true.")

    action_step = _finite_float(
        problem.get("action_level_step_m"), "problem.action_level_step_m"
    )
    if action_step <= 0.0:
        raise CascadeSpecError("problem.action_level_step_m must be positive.")
    problem_episode_months = _positive_int(
        problem.get("episode_months"), "problem.episode_months"
    )
    training_episode_months = _positive_int(
        training.get("episode_months"), "training.episode_months"
    )
    if problem_episode_months != EPISODE_MONTHS or training_episode_months != EPISODE_MONTHS:
        raise CascadeSpecError("DQN V1 training episodes must contain 240 transitions.")

    initial = np.asarray(problem.get("initial_levels_m"), dtype=float)
    if initial.shape != (2,) or np.any(~np.isfinite(initial)):
        raise CascadeSpecError("problem.initial_levels_m must contain two finite values.")
    if not np.array_equal(initial, np.array([1240.0, 812.0], dtype=float)):
        raise CascadeSpecError("DQN V1 initial levels must be [1240.0, 812.0] m.")

    development_years = _positive_int(
        validation.get("development_years"), "validation.development_years"
    )
    validation_years = _positive_int(
        validation.get("validation_years"), "validation.validation_years"
    )
    if (
        development_years != DEVELOPMENT_YEARS
        or validation_years != VALIDATION_YEARS
    ):
        raise CascadeSpecError(
            "DQN development mode requires the confirmed chronological 320/80-year split."
        )
    validation_score = validation.get("score")
    if validation_score != "mean_annual_penalized_objective_gwh":
        raise CascadeSpecError(
            "DQN validation score must be mean_annual_penalized_objective_gwh."
        )

    hidden_raw = network.get("hidden_dimensions")
    if isinstance(hidden_raw, (str, bytes)) or not isinstance(hidden_raw, Sequence):
        raise CascadeSpecError("network.hidden_dimensions must be a sequence.")
    hidden_dimensions = tuple(
        _positive_int(value, f"network.hidden_dimensions[{index}]")
        for index, value in enumerate(hidden_raw)
    )
    if not hidden_dimensions:
        raise CascadeSpecError("network.hidden_dimensions cannot be empty.")
    q_network_mode = network.get("action_representation", INDEXED_OUTPUT_MODE)
    if q_network_mode not in {INDEXED_OUTPUT_MODE, ACTION_CONDITIONED_MODE}:
        raise CascadeSpecError(
            "network.action_representation must be indexed_output or "
            "action_conditioned."
        )
    action_score_chunk_size = _positive_int(
        network.get("action_score_chunk_size", 256),
        "network.action_score_chunk_size",
    )

    epsilon_schedule = agent.get("epsilon_schedule")
    if epsilon_schedule != "linear":
        raise CascadeSpecError(
            "DQN V1 requires agent.epsilon_schedule='linear'."
        )

    gradient_raw = agent.get("gradient_clip_norm")
    max_grad_norm = (
        None
        if gradient_raw is None
        else _finite_float(gradient_raw, "agent.gradient_clip_norm")
    )
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise CascadeSpecError("agent.gradient_clip_norm must be positive when enabled.")

    return _TrainingSettings(
        mode=mode,
        experiment_id=experiment_id_from_config(config),
        seed=seed_value,
        episodes=_positive_int(training.get("episodes"), "training.episodes"),
        episode_months=EPISODE_MONTHS,
        initial_levels_m=(float(initial[0]), float(initial[1])),
        action_level_step_m=action_step,
        development_years=development_years,
        validation_years=validation_years,
        validation_interval_episodes=_positive_int(
            validation.get("interval_episodes"), "validation.interval_episodes"
        ),
        validation_score=str(validation_score),
        firm_power_failure_penalty_gwh=failure_penalty,
        reward_reference_days=_finite_float(
            reward.get("reward_reference_days"), "reward.reward_reference_days"
        ),
        reward_clip=_parse_reward_clip(reward),
        normalization_clip=normalization_clip,
        hidden_dimensions=hidden_dimensions,
        q_network_mode=str(q_network_mode),
        action_score_chunk_size=action_score_chunk_size,
        learning_rate=_finite_float(agent.get("learning_rate"), "agent.learning_rate"),
        gamma=_finite_float(agent.get("gamma"), "agent.gamma"),
        epsilon_start=_finite_float(
            agent.get("epsilon_start"), "agent.epsilon_start"
        ),
        epsilon_min=_finite_float(agent.get("epsilon_end"), "agent.epsilon_end"),
        epsilon_schedule=epsilon_schedule,
        epsilon_decay_steps=_positive_int(
            agent.get("epsilon_decay_steps"), "agent.epsilon_decay_steps"
        ),
        target_update_interval=_positive_int(
            agent.get("target_update_interval"), "agent.target_update_interval"
        ),
        replay_capacity=_positive_int(
            agent.get("replay_buffer_size"), "agent.replay_buffer_size"
        ),
        batch_size=_positive_int(agent.get("batch_size"), "agent.batch_size"),
        learning_starts=_nonnegative_int(
            agent.get("learning_starts"), "agent.learning_starts"
        ),
        max_grad_norm=max_grad_norm,
        device=_resolve_device(training.get("device")),
        learn_every_steps=_positive_int(
            training.get("learn_every_steps"), "training.learn_every_steps"
        ),
        updates_per_learning_event=_positive_int(
            training.get("updates_per_learning_event"),
            "training.updates_per_learning_event",
        ),
        logging_interval_episodes=_positive_int(
            training.get("logging_interval_episodes"),
            "training.logging_interval_episodes",
        ),
        checkpoint_interval_episodes=_positive_int(
            training.get("checkpoint_interval_episodes"),
            "training.checkpoint_interval_episodes",
        ),
    )


def inspect_dqn_configuration(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Validate one DQN configuration and describe its derived dimensions."""
    settings = _settings_from_config(config, mode=mode, seed=seed)
    action_grid = JointActionGrid(
        reservoir_1,
        reservoir_2,
        step_m=settings.action_level_step_m,
    )
    return {
        "experiment_id": settings.experiment_id,
        "mode": settings.mode,
        "seed": settings.seed,
        "state_order": list(STATE_ORDER),
        "state_dimension": len(STATE_ORDER),
        "action_level_step_m": action_grid.step_m,
        "xiaowan_level_count": action_grid.shape[0],
        "nuozhadu_level_count": action_grid.shape[1],
        "action_grid_shape": list(action_grid.shape),
        "joint_action_count": action_grid.action_count,
        "action_encoding": action_grid.state_dict()["encoding"],
        "action_mask_mode": ACTION_MASK_MODE,
        "hidden_dimensions": list(settings.hidden_dimensions),
        "q_network_dimensions": (
            [len(STATE_ORDER) + 2, *settings.hidden_dimensions, 1]
            if settings.q_network_mode == ACTION_CONDITIONED_MODE
            else [len(STATE_ORDER), *settings.hidden_dimensions, action_grid.action_count]
        ),
        "policy_observation_dimension": len(STATE_ORDER),
        "q_network_mode": settings.q_network_mode,
        "action_feature_dimension": (
            2 if settings.q_network_mode == ACTION_CONDITIONED_MODE else 0
        ),
        "action_score_chunk_size": settings.action_score_chunk_size,
        "episodes": settings.episodes,
        "episode_months": settings.episode_months,
        "environment_steps_budget": settings.episodes * settings.episode_months,
        "learning_rate": settings.learning_rate,
        "gamma": settings.gamma,
        "replay_buffer_size": settings.replay_capacity,
        "learning_starts": settings.learning_starts,
        "batch_size": settings.batch_size,
        "epsilon_start": settings.epsilon_start,
        "epsilon_end": settings.epsilon_min,
        "epsilon_schedule": settings.epsilon_schedule,
        "epsilon_decay_steps": settings.epsilon_decay_steps,
        "target_update_interval": settings.target_update_interval,
        "validation_interval_episodes": settings.validation_interval_episodes,
        "device": settings.device,
    }


def split_training_inflows(
    training_frame: pd.DataFrame,
    *,
    mode: str,
    development_years: int = DEVELOPMENT_YEARS,
    validation_years: int = VALIDATION_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return the permitted learning frame and internal validation frame.

    The input must be exactly one configured 400-year training sequence.  No
    independent evaluation sequence is accepted or opened here.
    """
    if mode not in TRAINING_MODES:
        raise CascadeSpecError(f"mode must be one of {TRAINING_MODES}.")
    development = _positive_int(development_years, "development_years")
    validation = _positive_int(validation_years, "validation_years")
    expected_months = 12 * (development + validation)
    if not isinstance(training_frame, pd.DataFrame):
        raise CascadeSpecError("training_frame must be a pandas DataFrame.")
    if len(training_frame) != expected_months:
        raise CascadeSpecError(
            f"DQN requires exactly {expected_months} configured training months; "
            f"received {len(training_frame)}."
        )
    missing = {"date", *INFLOW_COLUMNS} - set(training_frame.columns)
    if missing or training_frame.columns.duplicated().any():
        raise CascadeSpecError(
            "DQN training inflows require unique date/source columns; "
            f"missing: {sorted(missing)}."
        )
    periods = _as_month_period_index(training_frame["date"])
    expected_periods = pd.period_range(periods[0], periods[-1], freq="M")
    if not periods.equals(expected_periods):
        raise CascadeSpecError("DQN training inflows must be ordered continuous months.")
    if periods[0].month != 1 or periods[-1].month != 12:
        raise CascadeSpecError("DQN training inflows must contain complete calendar years.")
    normalized_inflow_sha256(training_frame)

    source = training_frame.reset_index(drop=True).copy()
    if mode == "final":
        return source, None
    split_index = 12 * development
    learning = source.iloc[:split_index].reset_index(drop=True).copy()
    validation_frame = source.iloc[split_index:].reset_index(drop=True).copy()
    return learning, validation_frame


def build_training_paths(
    output_root: str | Path,
    *,
    mode: str,
    seed: int,
    output_sections: Mapping[str, Any] | None = None,
    experiment_id: str | None = None,
) -> DQNTrainingPaths:
    """Construct one seed's non-overlapping checkpoint and history paths."""
    if mode not in TRAINING_MODES:
        raise CascadeSpecError(f"mode must be one of {TRAINING_MODES}.")
    seed_value = _nonnegative_int(seed, "seed")
    sections = dict(output_sections or {})
    checkpoints_name = str(sections.get("checkpoints", "checkpoints"))
    training_name = str(sections.get("logs", "training"))
    validation_name = str(sections.get("validation", "validation"))
    for name, value in (
        ("outputs.checkpoints", checkpoints_name),
        ("outputs.logs", training_name),
        ("outputs.validation", validation_name),
    ):
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise CascadeSpecError(f"{name} must be a safe relative directory name.")

    root = Path(output_root)
    validated_experiment_id = experiment_id_from_config(
        {"experiment_id": experiment_id}
    )
    if validated_experiment_id is None:
        run_root = root
        checkpoint_directory = root / checkpoints_name / mode / f"seed_{seed_value}"
        training_csv = root / training_name / mode / f"seed_{seed_value}_training.csv"
        configuration_snapshot_json = (
            root / training_name / mode / f"seed_{seed_value}_config_snapshot.json"
        )
        metadata_json = (
            root / training_name / mode / f"seed_{seed_value}_metadata.json"
        )
    else:
        run_root = (
            root
            / "experiments"
            / validated_experiment_id
            / f"seed_{seed_value}"
            / mode
        )
        checkpoint_directory = run_root / checkpoints_name
        training_csv = run_root / training_name / "history.csv"
        configuration_snapshot_json = run_root / "config_snapshot.json"
        metadata_json = run_root / "metadata.json"
    if mode == "development":
        validation_csv: Path | None = (
            root / validation_name / mode / f"seed_{seed_value}_validation.csv"
            if validated_experiment_id is None
            else run_root / validation_name / "history.csv"
        )
        best_checkpoint: Path | None = checkpoint_directory / "best_validation.pt"
        final_checkpoint: Path | None = None
    else:
        validation_csv = None
        best_checkpoint = None
        final_checkpoint = checkpoint_directory / "final.pt"
    return DQNTrainingPaths(
        output_root=root,
        run_root=run_root,
        experiment_id=validated_experiment_id,
        configuration_snapshot_json=configuration_snapshot_json,
        metadata_json=metadata_json,
        training_history_csv=training_csv,
        validation_history_csv=validation_csv,
        latest_checkpoint=checkpoint_directory / "latest.pt",
        best_validation_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
    )


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically replace a checkpoint using a temporary file beside it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a DQN training checkpoint container."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"DQN training checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise CascadeSpecError("DQN training checkpoint must contain a dictionary.")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CascadeSpecError("Unsupported DQN training checkpoint schema.")
    required = {
        "checkpoint_role",
        "experiment_id",
        "mode",
        "seed",
        "completed_episode",
        "learning_starts",
        "epsilon_start",
        "epsilon_end",
        "epsilon_schedule",
        "epsilon_decay_steps",
        "training_wall_time_seconds",
        "validation_wall_time_seconds",
        "agent_state",
        "normalizer",
        "action_grid",
        "reward_scaling",
        "configuration_snapshot",
        "configuration_sha256",
        "reservoir_physics_sha256",
        "training_inflow_sha256",
        "source_training_inflow_sha256",
        "episode_sampler_rng_state",
        "training_records",
        "validation_records",
    }
    missing = required - set(payload)
    if missing:
        raise CascadeSpecError(
            f"DQN training checkpoint is missing fields: {sorted(missing)}."
        )
    return payload


def _checkpoint_payload(
    *,
    role: str,
    settings: _TrainingSettings,
    agent: DoubleDQNAgent,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    reward_scaling: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    config_hash: str,
    reservoir_physics_hash: str,
    physical_configuration_hash: str | None,
    training_inflow_hash: str,
    source_training_inflow_hash: str,
    validation_inflow_hash: str | None,
    completed_episode: int,
    training_wall_time_seconds: float,
    validation_wall_time_seconds: float,
    sampler_rng: np.random.Generator,
    best_validation_score: float | None,
    best_validation_episode: int | None,
    training_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    budget_extension_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "experiment_id": settings.experiment_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": settings.mode,
        "seed": settings.seed,
        "completed_episode": int(completed_episode),
        "learning_starts": settings.learning_starts,
        "epsilon_start": settings.epsilon_start,
        "epsilon_end": settings.epsilon_min,
        "epsilon_schedule": settings.epsilon_schedule,
        "epsilon_decay_steps": settings.epsilon_decay_steps,
        "training_wall_time_seconds": float(training_wall_time_seconds),
        "validation_wall_time_seconds": float(validation_wall_time_seconds),
        "training_budget_episodes": settings.episodes,
        "environment_steps": int(agent.env_steps),
        "learning_steps": int(agent.learning_steps),
        "best_validation_score": best_validation_score,
        "best_validation_episode": best_validation_episode,
        "agent_state": agent.agent_state(),
        "normalizer": normalizer.state_dict(),
        "action_grid": action_grid.state_dict(),
        "reward_scaling": copy.deepcopy(dict(reward_scaling)),
        "configuration_snapshot": copy.deepcopy(dict(config_snapshot)),
        "configuration_sha256": config_hash,
        "reservoir_physics_sha256": reservoir_physics_hash,
        "physical_configuration_sha256": physical_configuration_hash,
        "training_inflow_sha256": training_inflow_hash,
        "source_training_inflow_sha256": source_training_inflow_hash,
        "validation_inflow_sha256": validation_inflow_hash,
        "episode_sampler_rng_state": copy.deepcopy(sampler_rng.bit_generator.state),
        "training_records": [dict(record) for record in training_records],
        "validation_records": [dict(record) for record in validation_records],
        "budget_extension_provenance": (
            None
            if budget_extension_provenance is None
            else copy.deepcopy(dict(budget_extension_provenance))
        ),
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    role: str,
    settings: _TrainingSettings,
    agent: DoubleDQNAgent,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    reward_scaling: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    config_hash: str,
    reservoir_physics_hash: str,
    physical_configuration_hash: str | None,
    training_inflow_hash: str,
    source_training_inflow_hash: str,
    validation_inflow_hash: str | None,
    completed_episode: int,
    training_wall_time_seconds: float,
    validation_wall_time_seconds: float,
    sampler_rng: np.random.Generator,
    best_validation_score: float | None,
    best_validation_episode: int | None,
    training_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    budget_extension_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Build and atomically persist a complete resumable checkpoint."""
    payload = _checkpoint_payload(
        role=role,
        settings=settings,
        agent=agent,
        normalizer=normalizer,
        action_grid=action_grid,
        reward_scaling=reward_scaling,
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        reservoir_physics_hash=reservoir_physics_hash,
        physical_configuration_hash=physical_configuration_hash,
        training_inflow_hash=training_inflow_hash,
        source_training_inflow_hash=source_training_inflow_hash,
        validation_inflow_hash=validation_inflow_hash,
        completed_episode=completed_episode,
        training_wall_time_seconds=training_wall_time_seconds,
        validation_wall_time_seconds=validation_wall_time_seconds,
        sampler_rng=sampler_rng,
        best_validation_score=best_validation_score,
        best_validation_episode=best_validation_episode,
        training_records=training_records,
        validation_records=validation_records,
        budget_extension_provenance=budget_extension_provenance,
    )
    atomic_torch_save(payload, path)


def _atomic_write_history(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        pd.DataFrame(records, columns=columns).to_csv(
            temporary_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _normalized_action_features(action_grid: JointActionGrid) -> np.ndarray:
    """Return row-major normalized target levels for every stable action ID."""
    xiaowan = action_grid.xiaowan_levels_m
    nuozhadu = action_grid.nuozhadu_levels_m
    xiaowan_normalized = (xiaowan - xiaowan[0]) / (xiaowan[-1] - xiaowan[0])
    nuozhadu_normalized = (nuozhadu - nuozhadu[0]) / (
        nuozhadu[-1] - nuozhadu[0]
    )
    upstream, downstream = np.meshgrid(
        xiaowan_normalized, nuozhadu_normalized, indexing="ij"
    )
    features = np.column_stack((upstream.ravel(), downstream.ravel())).astype(
        np.float32
    )
    if features.shape != (action_grid.action_count, 2):
        raise CascadeSpecError("Action feature table does not match the joint grid.")
    return features


def _new_agent(
    settings: _TrainingSettings, action_grid: JointActionGrid
) -> DoubleDQNAgent:
    action_features = (
        _normalized_action_features(action_grid)
        if settings.q_network_mode == ACTION_CONDITIONED_MODE
        else None
    )
    return DoubleDQNAgent(
        state_dimension=len(STATE_ORDER),
        action_count=action_grid.action_count,
        hidden_dimensions=settings.hidden_dimensions,
        learning_rate=settings.learning_rate,
        gamma=settings.gamma,
        epsilon_start=settings.epsilon_start,
        epsilon_min=settings.epsilon_min,
        epsilon_schedule=settings.epsilon_schedule,
        epsilon_decay_steps=settings.epsilon_decay_steps,
        learning_starts=settings.learning_starts,
        target_update_interval=settings.target_update_interval,
        replay_capacity=settings.replay_capacity,
        batch_size=settings.batch_size,
        max_grad_norm=settings.max_grad_norm,
        seed=settings.seed,
        device=settings.device,
        q_network_mode=settings.q_network_mode,
        action_features=action_features,
        action_score_chunk_size=settings.action_score_chunk_size,
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _validation_record(
    *,
    episode: int,
    agent: DoubleDQNAgent,
    summary: Mapping[str, Any],
    elapsed_seconds: float,
    selected: bool,
) -> dict[str, Any]:
    reliability = summary["firm_power_reliability"]
    counts = summary["target_unreached_count"]
    rates = summary["target_unreached_rate"]
    score = float(summary["mean_annual_penalized_objective_gwh"])
    return {
        "episode": episode,
        "environment_steps": int(agent.env_steps),
        "learning_steps": int(agent.learning_steps),
        "validation_wall_time_seconds": float(elapsed_seconds),
        "validation_months": int(summary["months"]),
        "validation_total_cascade_generation_gwh": float(
            summary["total_cascade_generation_gwh"]
        ),
        "validation_mean_annual_cascade_generation_gwh": float(
            summary["mean_annual_cascade_generation_gwh"]
        ),
        "validation_total_penalized_objective_gwh": float(
            summary["total_penalized_objective_gwh"]
        ),
        "validation_mean_annual_penalized_objective_gwh": score,
        "validation_failed_station_months": int(
            summary["failed_station_months"]
        ),
        "xiaowan_firm_power_reliability": float(reliability["xiaowan"]),
        "nuozhadu_firm_power_reliability": float(reliability["nuozhadu"]),
        "cascade_firm_power_reliability": float(reliability["cascade"]),
        "xiaowan_target_unreached_count": int(counts["xiaowan"]),
        "xiaowan_target_unreached_rate": float(rates["xiaowan"]),
        "nuozhadu_target_unreached_count": int(counts["nuozhadu"]),
        "nuozhadu_target_unreached_rate": float(rates["nuozhadu"]),
        "either_reservoir_target_unreached_count": int(
            counts["either_reservoir"]
        ),
        "either_reservoir_target_unreached_rate": float(
            rates["either_reservoir"]
        ),
        "score": score,
        "selected_best_validation_checkpoint": bool(selected),
    }


def _restore_latest(
    *,
    path: Path,
    settings: _TrainingSettings,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    agent: DoubleDQNAgent,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    reward_scaling: Mapping[str, Any],
    sampler_rng: np.random.Generator,
    config_hash: str,
    physical_configuration_hash: str | None,
    training_inflow_hash: str,
    source_training_inflow_hash: str,
    validation_inflow_hash: str | None,
) -> tuple[
    int,
    float | None,
    int | None,
    float,
    float,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    checkpoint = load_training_checkpoint(path, map_location=agent.device)
    expected = {
        "checkpoint_role": "latest",
        "experiment_id": settings.experiment_id,
        "mode": settings.mode,
        "seed": settings.seed,
        "training_budget_episodes": settings.episodes,
        "learning_starts": settings.learning_starts,
        "epsilon_start": settings.epsilon_start,
        "epsilon_end": settings.epsilon_min,
        "epsilon_schedule": settings.epsilon_schedule,
        "epsilon_decay_steps": settings.epsilon_decay_steps,
        "configuration_sha256": config_hash,
        "reservoir_physics_sha256": reservoir_pair_sha256(
            reservoir_1, reservoir_2
        ),
        "physical_configuration_sha256": physical_configuration_hash,
        "training_inflow_sha256": training_inflow_hash,
        "source_training_inflow_sha256": source_training_inflow_hash,
        "validation_inflow_sha256": validation_inflow_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CascadeSpecError(
                f"Latest DQN checkpoint {key} does not match this training run."
            )
    if configuration_sha256(checkpoint["configuration_snapshot"]) != checkpoint[
        "configuration_sha256"
    ]:
        raise CascadeSpecError(
            "Latest DQN checkpoint configuration snapshot does not match its hash."
        )

    saved_normalizer = StateNormalizer.from_state_dict(
        checkpoint["normalizer"], reservoir_1, reservoir_2
    )
    saved_grid = JointActionGrid.from_state_dict(
        checkpoint["action_grid"], reservoir_1, reservoir_2
    )
    if saved_normalizer.state_dict() != normalizer.state_dict():
        raise CascadeSpecError(
            "Latest DQN checkpoint normalization does not match the permitted training subset."
        )
    if saved_grid.state_dict() != action_grid.state_dict():
        raise CascadeSpecError("Latest DQN checkpoint action grid does not match this run.")
    if checkpoint["reward_scaling"] != dict(reward_scaling):
        raise CascadeSpecError("Latest DQN checkpoint reward scaling does not match this run.")

    completed_episode = _nonnegative_int(
        checkpoint["completed_episode"], "checkpoint completed_episode"
    )
    if completed_episode > settings.episodes:
        raise CascadeSpecError(
            "Latest DQN checkpoint exceeds the configured fixed training budget."
        )
    training_records = [dict(record) for record in checkpoint["training_records"]]
    validation_records = [dict(record) for record in checkpoint["validation_records"]]
    if len(training_records) != completed_episode:
        raise CascadeSpecError(
            "Latest DQN checkpoint training history is inconsistent with its episode counter."
        )
    expected_validation_records = (
        sum(
            int(
                episode % settings.validation_interval_episodes == 0
                or episode == settings.episodes
            )
            for episode in range(1, completed_episode + 1)
        )
        if settings.mode == "development"
        else 0
    )
    if len(validation_records) != expected_validation_records:
        raise CascadeSpecError(
            "Latest DQN checkpoint validation history is inconsistent with its episode counter."
        )

    training_wall_time_seconds = _finite_float(
        checkpoint["training_wall_time_seconds"],
        "checkpoint training_wall_time_seconds",
    )
    validation_wall_time_seconds = _finite_float(
        checkpoint["validation_wall_time_seconds"],
        "checkpoint validation_wall_time_seconds",
    )
    if training_wall_time_seconds < 0.0 or validation_wall_time_seconds < 0.0:
        raise CascadeSpecError("Checkpoint wall-clock durations must be nonnegative.")

    agent.load_training_state(checkpoint["agent_state"], restore_rng=True)
    if agent.env_steps != int(checkpoint["environment_steps"]):
        raise CascadeSpecError("Checkpoint environment-step counters disagree.")
    if agent.learning_steps != int(checkpoint["learning_steps"]):
        raise CascadeSpecError("Checkpoint learning-step counters disagree.")
    sampler_rng.bit_generator.state = copy.deepcopy(
        checkpoint["episode_sampler_rng_state"]
    )

    best_score_raw = checkpoint.get("best_validation_score")
    best_score = None if best_score_raw is None else float(best_score_raw)
    if best_score is not None and not np.isfinite(best_score):
        raise CascadeSpecError("Checkpoint best validation score must be finite.")
    best_episode_raw = checkpoint.get("best_validation_episode")
    best_episode = None if best_episode_raw is None else int(best_episode_raw)
    if (best_score is None) != (best_episode is None):
        raise CascadeSpecError("Checkpoint best validation score/episode are inconsistent.")
    provenance_raw = checkpoint.get("budget_extension_provenance")
    if provenance_raw is not None and not isinstance(provenance_raw, Mapping):
        raise CascadeSpecError("Checkpoint budget-extension provenance is invalid.")
    return (
        completed_episode,
        best_score,
        best_episode,
        training_wall_time_seconds,
        validation_wall_time_seconds,
        training_records,
        validation_records,
        (
            None
            if provenance_raw is None
            else copy.deepcopy(dict(provenance_raw))
        ),
    )


def _checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_budget_extension(
    *,
    path: Path,
    settings: _TrainingSettings,
    config_snapshot: Mapping[str, Any],
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    agent: DoubleDQNAgent,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    reward_scaling: Mapping[str, Any],
    sampler_rng: np.random.Generator,
    physical_configuration_hash: str | None,
    training_inflow_hash: str,
    source_training_inflow_hash: str,
    validation_inflow_hash: str | None,
) -> tuple[
    int,
    float | None,
    int | None,
    float,
    float,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    extension = _budget_extension_settings(config_snapshot, settings)
    checkpoint = load_training_checkpoint(path, map_location=agent.device)
    source_snapshot = checkpoint.get("configuration_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise CascadeSpecError("Source Run02 checkpoint has no configuration snapshot.")
    source_config_hash = str(checkpoint["configuration_sha256"])
    if configuration_sha256(source_snapshot) != source_config_hash:
        raise CascadeSpecError("Source Run02 checkpoint configuration hash is invalid.")
    if configuration_sha256(
        _budget_extension_semantics_snapshot(source_snapshot)
    ) != configuration_sha256(
        _budget_extension_semantics_snapshot(config_snapshot)
    ):
        raise CascadeSpecError(
            "Budget extension differs from Run02 in fields other than the "
            "experiment id, output root, and episode budget."
        )

    expected = {
        "checkpoint_role": "latest",
        "experiment_id": extension["source_experiment_id"],
        "mode": "development",
        "seed": settings.seed,
        "training_budget_episodes": extension[
            "source_training_budget_episodes"
        ],
        "learning_starts": settings.learning_starts,
        "epsilon_start": settings.epsilon_start,
        "epsilon_end": settings.epsilon_min,
        "epsilon_schedule": settings.epsilon_schedule,
        "epsilon_decay_steps": settings.epsilon_decay_steps,
        "reservoir_physics_sha256": reservoir_pair_sha256(
            reservoir_1, reservoir_2
        ),
        "physical_configuration_sha256": physical_configuration_hash,
        "training_inflow_sha256": training_inflow_hash,
        "source_training_inflow_sha256": source_training_inflow_hash,
        "validation_inflow_sha256": validation_inflow_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CascadeSpecError(
                f"Budget-extension source checkpoint {key} is incompatible."
            )

    completed_episode = _nonnegative_int(
        checkpoint["completed_episode"], "source checkpoint completed_episode"
    )
    source_budget = int(extension["source_training_budget_episodes"])
    if completed_episode != source_budget:
        raise CascadeSpecError(
            "Budget extension must start from the fully completed episode-3000 latest checkpoint."
        )
    if settings.episodes <= completed_episode:
        raise CascadeSpecError("The extended budget must exceed the source budget.")
    if int(checkpoint["environment_steps"]) != (
        completed_episode * settings.episode_months
    ):
        raise CascadeSpecError(
            "Source Run02 checkpoint is not at the episode-3000 environment-step boundary."
        )

    saved_normalizer = StateNormalizer.from_state_dict(
        checkpoint["normalizer"], reservoir_1, reservoir_2
    )
    saved_grid = JointActionGrid.from_state_dict(
        checkpoint["action_grid"], reservoir_1, reservoir_2
    )
    if saved_normalizer.state_dict() != normalizer.state_dict():
        raise CascadeSpecError("Source Run02 normalizer is incompatible.")
    if saved_grid.state_dict() != action_grid.state_dict():
        raise CascadeSpecError("Source Run02 action grid is incompatible.")
    if checkpoint["reward_scaling"] != dict(reward_scaling):
        raise CascadeSpecError("Source Run02 reward scaling is incompatible.")

    training_records = [dict(record) for record in checkpoint["training_records"]]
    validation_records = [dict(record) for record in checkpoint["validation_records"]]
    if len(training_records) != completed_episode:
        raise CascadeSpecError("Source Run02 training history is incomplete.")
    expected_validation_records = completed_episode // settings.validation_interval_episodes
    if len(validation_records) != expected_validation_records:
        raise CascadeSpecError("Source Run02 validation history is incomplete.")

    training_wall_time_seconds = _finite_float(
        checkpoint["training_wall_time_seconds"],
        "source checkpoint training_wall_time_seconds",
    )
    validation_wall_time_seconds = _finite_float(
        checkpoint["validation_wall_time_seconds"],
        "source checkpoint validation_wall_time_seconds",
    )
    if training_wall_time_seconds < 0.0 or validation_wall_time_seconds < 0.0:
        raise CascadeSpecError("Source checkpoint wall-clock durations are invalid.")

    agent.load_training_state(checkpoint["agent_state"], restore_rng=True)
    if agent.env_steps != int(checkpoint["environment_steps"]):
        raise CascadeSpecError("Source Run02 environment-step counters disagree.")
    if agent.learning_steps != int(checkpoint["learning_steps"]):
        raise CascadeSpecError("Source Run02 learning-step counters disagree.")
    sampler_rng.bit_generator.state = copy.deepcopy(
        checkpoint["episode_sampler_rng_state"]
    )

    best_score_raw = checkpoint.get("best_validation_score")
    best_episode_raw = checkpoint.get("best_validation_episode")
    if best_score_raw is None or best_episode_raw is None:
        raise CascadeSpecError("Source Run02 latest checkpoint has no selected best model.")
    best_score = float(best_score_raw)
    best_episode = int(best_episode_raw)
    if not np.isfinite(best_score) or not 1 <= best_episode <= completed_episode:
        raise CascadeSpecError("Source Run02 best-validation metadata is invalid.")

    provenance = {
        "kind": "dqn_training_budget_extension",
        "source_checkpoint_role": "latest",
        "source_checkpoint_path": str(path.resolve()),
        "source_checkpoint_sha256": _checkpoint_file_sha256(path),
        "source_experiment_id": extension["source_experiment_id"],
        "source_training_budget_episodes": source_budget,
        "source_completed_episode": completed_episode,
        "source_environment_steps": int(checkpoint["environment_steps"]),
        "source_learning_steps": int(checkpoint["learning_steps"]),
        "source_configuration_sha256": source_config_hash,
        "source_reservoir_physics_sha256": checkpoint[
            "reservoir_physics_sha256"
        ],
        "source_physical_configuration_sha256": checkpoint.get(
            "physical_configuration_sha256"
        ),
        "source_training_inflow_sha256": checkpoint["training_inflow_sha256"],
        "source_full_training_inflow_sha256": checkpoint[
            "source_training_inflow_sha256"
        ],
        "source_validation_inflow_sha256": checkpoint.get(
            "validation_inflow_sha256"
        ),
        "target_experiment_id": settings.experiment_id,
        "target_training_budget_episodes": settings.episodes,
        "continuation_first_episode": completed_episode + 1,
    }
    return (
        completed_episode,
        best_score,
        best_episode,
        training_wall_time_seconds,
        validation_wall_time_seconds,
        training_records,
        validation_records,
        provenance,
    )


def _rebase_budget_extension_best_checkpoint(
    *,
    source_latest_path: Path,
    destination: Path,
    settings: _TrainingSettings,
    config_snapshot: Mapping[str, Any],
    config_hash: str,
    provenance: Mapping[str, Any],
    expected_best_score: float,
    expected_best_episode: int,
) -> dict[str, Any]:
    source_best_path = source_latest_path.parent / "best_validation.pt"
    source_best = load_training_checkpoint(
        source_best_path, map_location="cpu"
    )
    expected = {
        "checkpoint_role": "best_validation",
        "experiment_id": provenance["source_experiment_id"],
        "mode": "development",
        "seed": settings.seed,
        "training_budget_episodes": provenance[
            "source_training_budget_episodes"
        ],
        "completed_episode": expected_best_episode,
        "best_validation_episode": expected_best_episode,
        "configuration_sha256": provenance["source_configuration_sha256"],
        "reservoir_physics_sha256": provenance[
            "source_reservoir_physics_sha256"
        ],
        "physical_configuration_sha256": provenance[
            "source_physical_configuration_sha256"
        ],
        "training_inflow_sha256": provenance["source_training_inflow_sha256"],
        "source_training_inflow_sha256": provenance[
            "source_full_training_inflow_sha256"
        ],
        "validation_inflow_sha256": provenance[
            "source_validation_inflow_sha256"
        ],
    }
    for key, value in expected.items():
        if source_best.get(key) != value:
            raise CascadeSpecError(
                f"Source Run02 best-validation checkpoint {key} is incompatible."
            )
    if not np.isclose(
        float(source_best.get("best_validation_score", float("nan"))),
        expected_best_score,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise CascadeSpecError(
            "Source Run02 best-validation checkpoint score is incompatible."
        )
    if configuration_sha256(source_best["configuration_snapshot"]) != str(
        source_best["configuration_sha256"]
    ):
        raise CascadeSpecError("Source Run02 best checkpoint configuration hash is invalid.")

    updated_provenance = copy.deepcopy(dict(provenance))
    updated_provenance.update(
        {
            "source_best_validation_checkpoint_path": str(
                source_best_path.resolve()
            ),
            "source_best_validation_checkpoint_sha256": (
                _checkpoint_file_sha256(source_best_path)
            ),
        }
    )
    rebased = copy.deepcopy(source_best)
    rebased.update(
        {
            "experiment_id": settings.experiment_id,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_budget_episodes": settings.episodes,
            "configuration_snapshot": copy.deepcopy(dict(config_snapshot)),
            "configuration_sha256": config_hash,
            "budget_extension_provenance": updated_provenance,
        }
    )
    atomic_torch_save(rebased, destination)
    return updated_provenance


def inspect_dqn_budget_extension(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    training_frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    source_checkpoint: str | Path,
    physical_configuration_sha256: str | None = None,
    monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Fully validate a Run02 episode-3000 extension source without training."""
    settings = _settings_from_config(config, mode=mode, seed=seed)
    learning_frame, validation_frame = split_training_inflows(
        training_frame,
        mode=settings.mode,
        development_years=settings.development_years,
        validation_years=settings.validation_years,
    )
    config_snapshot = _plain_snapshot(config)
    normalizer = StateNormalizer.fit(
        learning_frame,
        reservoir_1,
        reservoir_2,
        clip=settings.normalization_clip,
    )
    action_grid = JointActionGrid(
        reservoir_1, reservoir_2, step_m=settings.action_level_step_m
    )
    cpu_settings = replace(settings, device="cpu")
    agent = _new_agent(cpu_settings, action_grid)
    sampler_rng = np.random.default_rng(settings.seed)
    probe_flows = learning_frame.loc[:, list(INFLOW_COLUMNS)].to_numpy(dtype=float)
    probe_periods = _as_month_period_index(learning_frame["date"])
    probe_env = CascadeReservoirEnv(
        reservoir_1,
        reservoir_2,
        probe_flows[: settings.episode_months + 1],
        timestamps=probe_periods[: settings.episode_months + 1],
        initial_levels_m=settings.initial_levels_m,
        firm_power_failure_penalty_gwh=settings.firm_power_failure_penalty_gwh,
        monthly_max_target_levels_m=monthly_max_target_levels_m,
        require_energy_parameters=True,
    )
    probe_adapter = CascadeDQNAdapter(
        probe_env,
        normalizer,
        action_grid,
        reward_reference_days=settings.reward_reference_days,
        reward_clip=settings.reward_clip,
        max_transitions=settings.episode_months,
    )
    _validate_frozen_reward_scale(probe_adapter.reward_metadata)
    source_path = Path(source_checkpoint)
    (
        completed_episode,
        best_score,
        best_episode,
        _,
        _,
        training_records,
        validation_records,
        provenance,
    ) = _restore_budget_extension(
        path=source_path,
        settings=cpu_settings,
        config_snapshot=config_snapshot,
        reservoir_1=reservoir_1,
        reservoir_2=reservoir_2,
        agent=agent,
        normalizer=normalizer,
        action_grid=action_grid,
        reward_scaling=probe_adapter.reward_metadata,
        sampler_rng=sampler_rng,
        physical_configuration_hash=_optional_sha256(
            physical_configuration_sha256,
            "physical_configuration_sha256",
        ),
        training_inflow_hash=normalized_inflow_sha256(learning_frame),
        source_training_inflow_hash=normalized_inflow_sha256(training_frame),
        validation_inflow_hash=(
            None
            if validation_frame is None
            else normalized_inflow_sha256(validation_frame)
        ),
    )
    source_best_path = source_path.parent / "best_validation.pt"
    source_best = load_training_checkpoint(source_best_path, map_location="cpu")
    if (
        source_best.get("checkpoint_role") != "best_validation"
        or source_best.get("experiment_id") != provenance["source_experiment_id"]
        or int(source_best.get("seed", -1)) != settings.seed
        or int(source_best.get("completed_episode", -1)) != int(best_episode)
        or not np.isclose(
            float(source_best.get("best_validation_score", float("nan"))),
            float(best_score),
            rtol=0.0,
            atol=1.0e-12,
        )
        or source_best.get("configuration_sha256")
        != provenance["source_configuration_sha256"]
        or source_best.get("reservoir_physics_sha256")
        != provenance["source_reservoir_physics_sha256"]
        or source_best.get("training_inflow_sha256")
        != provenance["source_training_inflow_sha256"]
        or source_best.get("validation_inflow_sha256")
        != provenance["source_validation_inflow_sha256"]
    ):
        raise CascadeSpecError(
            "Run02 best-validation checkpoint does not match its episode-3000 latest checkpoint."
        )
    return {
        "compatible": True,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_role": "latest",
        "source_experiment_id": provenance["source_experiment_id"],
        "source_seed": settings.seed,
        "source_completed_episode": completed_episode,
        "source_environment_steps": int(agent.env_steps),
        "source_learning_steps": int(agent.learning_steps),
        "source_training_history_rows": len(training_records),
        "source_validation_history_rows": len(validation_records),
        "source_best_validation_episode": int(best_episode),
        "source_best_validation_score": float(best_score),
        "source_best_validation_checkpoint": str(source_best_path.resolve()),
        "target_experiment_id": settings.experiment_id,
        "target_training_budget_episodes": settings.episodes,
        "continuation_first_episode": completed_episode + 1,
        "semantic_configuration_match": True,
        "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
    }


def train_dqn_seed(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    training_frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    output_root: str | Path | None = None,
    resume_latest: bool = False,
    budget_extension_checkpoint: str | Path | None = None,
    physical_configuration_sha256: str | None = None,
    monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DQNTrainingResult:
    """Train one independent seed under the confirmed P4 DQN V1 protocol.

    ``training_frame`` must be the configured 400-year *training* split.  This
    API deliberately has no evaluation-frame argument, preventing evaluation
    inflows from entering normalization, replay, validation, or model choice.
    """
    if not isinstance(reservoir_1, ReservoirSpec) or not isinstance(
        reservoir_2, ReservoirSpec
    ):
        raise CascadeSpecError("train_dqn_seed requires two ReservoirSpec objects.")
    if not isinstance(resume_latest, bool):
        raise TypeError("resume_latest must be boolean.")
    if resume_latest and budget_extension_checkpoint is not None:
        raise CascadeSpecError(
            "Ordinary resume and budget-extension import are mutually exclusive."
        )
    extension_checkpoint_path = (
        None
        if budget_extension_checkpoint is None
        else Path(budget_extension_checkpoint)
    )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or null.")

    settings = _settings_from_config(config, mode=mode, seed=seed)
    training_timer_started = time.perf_counter()
    if not np.array_equal(
        np.asarray(settings.initial_levels_m, dtype=float),
        np.array(
            [reservoir_1.normal_level_m, reservoir_2.normal_level_m], dtype=float
        ),
    ):
        raise CascadeSpecError(
            "DQN V1 episode initial levels must equal both reservoirs' normal levels."
        )
    learning_frame, validation_frame = split_training_inflows(
        training_frame,
        mode=settings.mode,
        development_years=settings.development_years,
        validation_years=settings.validation_years,
    )

    config_snapshot = _plain_snapshot(config)
    config_hash = configuration_sha256(config_snapshot)
    physical_config_hash = _optional_sha256(
        physical_configuration_sha256, "physical_configuration_sha256"
    )
    physics_hash = reservoir_pair_sha256(reservoir_1, reservoir_2)
    source_training_hash = normalized_inflow_sha256(training_frame)
    training_hash = normalized_inflow_sha256(learning_frame)
    validation_hash = (
        None
        if validation_frame is None
        else normalized_inflow_sha256(validation_frame)
    )

    normalizer = StateNormalizer.fit(
        learning_frame,
        reservoir_1,
        reservoir_2,
        clip=settings.normalization_clip,
    )
    action_grid = JointActionGrid(
        reservoir_1, reservoir_2, step_m=settings.action_level_step_m
    )
    agent = _new_agent(settings, action_grid)
    sampler_rng = np.random.default_rng(settings.seed)

    outputs = _mapping(config, "outputs")
    configured_root = outputs.get("root") if output_root is None else output_root
    if configured_root is None:
        raise CascadeSpecError("outputs.root is required when output_root is not supplied.")
    paths = build_training_paths(
        configured_root,
        mode=settings.mode,
        seed=settings.seed,
        output_sections=outputs,
        experiment_id=settings.experiment_id,
    )

    probe_flows = learning_frame.loc[:, list(INFLOW_COLUMNS)].to_numpy(dtype=float)
    probe_periods = _as_month_period_index(learning_frame["date"])
    probe_env = CascadeReservoirEnv(
        reservoir_1,
        reservoir_2,
        probe_flows[: settings.episode_months + 1],
        timestamps=probe_periods[: settings.episode_months + 1],
        initial_levels_m=settings.initial_levels_m,
        firm_power_failure_penalty_gwh=settings.firm_power_failure_penalty_gwh,
        monthly_max_target_levels_m=monthly_max_target_levels_m,
        require_energy_parameters=True,
    )
    probe_adapter = CascadeDQNAdapter(
        probe_env,
        normalizer,
        action_grid,
        reward_reference_days=settings.reward_reference_days,
        reward_clip=settings.reward_clip,
        max_transitions=settings.episode_months,
    )
    reward_scaling = probe_adapter.reward_metadata
    _validate_frozen_reward_scale(reward_scaling)

    completed_episode = 0
    best_validation_score: float | None = None
    best_validation_episode: int | None = None
    training_wall_time_offset = 0.0
    validation_wall_time_seconds = 0.0
    training_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    budget_extension_provenance: dict[str, Any] | None = None
    restored_history = False
    if resume_latest:
        (
            completed_episode,
            best_validation_score,
            best_validation_episode,
            training_wall_time_offset,
            validation_wall_time_seconds,
            training_records,
            validation_records,
            budget_extension_provenance,
        ) = _restore_latest(
            path=paths.latest_checkpoint,
            settings=settings,
            reservoir_1=reservoir_1,
            reservoir_2=reservoir_2,
            agent=agent,
            normalizer=normalizer,
            action_grid=action_grid,
            reward_scaling=reward_scaling,
            sampler_rng=sampler_rng,
            config_hash=config_hash,
            physical_configuration_hash=physical_config_hash,
            training_inflow_hash=training_hash,
            source_training_inflow_hash=source_training_hash,
            validation_inflow_hash=validation_hash,
        )
        restored_history = True
    elif extension_checkpoint_path is not None:
        (
            completed_episode,
            best_validation_score,
            best_validation_episode,
            training_wall_time_offset,
            validation_wall_time_seconds,
            training_records,
            validation_records,
            budget_extension_provenance,
        ) = _restore_budget_extension(
            path=extension_checkpoint_path,
            settings=settings,
            config_snapshot=config_snapshot,
            reservoir_1=reservoir_1,
            reservoir_2=reservoir_2,
            agent=agent,
            normalizer=normalizer,
            action_grid=action_grid,
            reward_scaling=reward_scaling,
            sampler_rng=sampler_rng,
            physical_configuration_hash=physical_config_hash,
            training_inflow_hash=training_hash,
            source_training_inflow_hash=source_training_hash,
            validation_inflow_hash=validation_hash,
        )
        if paths.best_validation_checkpoint is None:
            raise CascadeSpecError(
                "Budget extension requires a development best-checkpoint path."
            )
        budget_extension_provenance = _rebase_budget_extension_best_checkpoint(
            source_latest_path=extension_checkpoint_path,
            destination=paths.best_validation_checkpoint,
            settings=settings,
            config_snapshot=config_snapshot,
            config_hash=config_hash,
            provenance=budget_extension_provenance,
            expected_best_score=float(best_validation_score),
            expected_best_episode=int(best_validation_episode),
        )
        restored_history = True

    if restored_history:
        _atomic_write_history(
            training_records, TRAINING_HISTORY_COLUMNS, paths.training_history_csv
        )
        if paths.validation_history_csv is not None:
            _atomic_write_history(
                validation_records,
                VALIDATION_HISTORY_COLUMNS,
                paths.validation_history_csv,
            )

    def current_training_wall_time_seconds() -> float:
        return training_wall_time_offset + (
            time.perf_counter() - training_timer_started
        )

    def save_role(path: Path, role: str, episode: int) -> None:
        save_training_checkpoint(
            path,
            role=role,
            settings=settings,
            agent=agent,
            normalizer=normalizer,
            action_grid=action_grid,
            reward_scaling=reward_scaling,
            config_snapshot=config_snapshot,
            config_hash=config_hash,
            reservoir_physics_hash=physics_hash,
            physical_configuration_hash=physical_config_hash,
            training_inflow_hash=training_hash,
            source_training_inflow_hash=source_training_hash,
            validation_inflow_hash=validation_hash,
            completed_episode=episode,
            training_wall_time_seconds=current_training_wall_time_seconds(),
            validation_wall_time_seconds=validation_wall_time_seconds,
            sampler_rng=sampler_rng,
            best_validation_score=best_validation_score,
            best_validation_episode=best_validation_episode,
            training_records=training_records,
            validation_records=validation_records,
            budget_extension_provenance=budget_extension_provenance,
        )

    for episode in range(completed_episode + 1, settings.episodes + 1):
        start_index = sample_january_start_index(
            learning_frame,
            sampler_rng,
            episode_months=settings.episode_months,
        )
        window = slice_episode_window(
            learning_frame,
            start_index,
            episode_months=settings.episode_months,
        )
        periods = _as_month_period_index(window["date"])
        flows = window.loc[:, list(INFLOW_COLUMNS)].to_numpy(dtype=float)
        env = CascadeReservoirEnv(
            reservoir_1,
            reservoir_2,
            flows,
            timestamps=periods,
            initial_levels_m=settings.initial_levels_m,
            firm_power_failure_penalty_gwh=settings.firm_power_failure_penalty_gwh,
            monthly_max_target_levels_m=monthly_max_target_levels_m,
            require_energy_parameters=True,
        )
        adapter = CascadeDQNAdapter(
            env,
            normalizer,
            action_grid,
            reward_reference_days=settings.reward_reference_days,
            reward_clip=settings.reward_clip,
            max_transitions=settings.episode_months,
        )

        state, _ = adapter.reset(seed=settings.seed)
        action_mask = adapter.get_feasible_action_mask()
        raw_episode_generation_gwh = 0.0
        raw_episode_penalized_objective_gwh = 0.0
        episode_failed_station_months = 0
        episode_failure_penalty_gwh = 0.0
        normalized_episode_reward = 0.0
        losses: list[float] = []
        predicted_q_values: list[float] = []
        target_q_values: list[float] = []
        gradient_norms: list[float] = []
        feasible_action_counts: list[int] = []
        learning_updates = 0
        last_loss: float | None = None

        for transition_index in range(settings.episode_months):
            feasible_action_counts.append(int(action_mask.sum()))
            action_index = agent.select_action(
                state, greedy=False, action_mask=action_mask
            )
            next_state, scaled_reward, terminated, truncated, info = adapter.step(
                action_index
            )
            is_final_transition = transition_index == settings.episode_months - 1
            if terminated:
                raise CascadeSpecError(
                    "A 241-month DQN training window reached a physical data terminal."
                )
            if next_state is None:
                raise CascadeSpecError(
                    "A DQN training transition did not return its real next observation."
                )
            if truncated != is_final_transition:
                raise CascadeSpecError(
                    "DQN training truncation did not occur exactly after 240 transitions."
                )
            next_action_mask = adapter.get_feasible_action_mask()

            # The final 240-month boundary is artificial.  All stored training
            # transitions therefore retain their real next-state bootstrap.
            agent.store_transition(
                state,
                action_index,
                scaled_reward,
                next_state,
                bootstrap_terminal=False,
                next_action_mask=next_action_mask,
            )
            raw_episode_generation_gwh += float(info["generation_energy_gwh"])
            raw_episode_penalized_objective_gwh += float(
                info["penalized_reward_gwh"]
            )
            episode_failed_station_months += int(info["failed_station_months"])
            episode_failure_penalty_gwh += float(info["failure_penalty_gwh"])
            normalized_episode_reward += float(scaled_reward)

            if (
                agent.env_steps >= settings.learning_starts
                and agent.can_learn()
                and agent.env_steps % settings.learn_every_steps == 0
            ):
                for _ in range(settings.updates_per_learning_event):
                    learning = agent.learn()
                    if learning is None:
                        break
                    learning_updates += 1
                    last_loss = float(learning["loss"])
                    losses.append(last_loss)
                    predicted_q_values.append(float(learning["mean_predicted_q"]))
                    target_q_values.append(float(learning["mean_target_q"]))
                    gradient_norms.append(float(learning["gradient_norm"]))
            state = next_state
            action_mask = next_action_mask

        diagnostics = adapter.episode_diagnostics
        counts = diagnostics["target_unreached_count"]
        rates = diagnostics["target_unreached_rate"]
        should_validate = bool(
            settings.mode == "development"
            and (
                episode % settings.validation_interval_episodes == 0
                or episode == settings.episodes
            )
        )
        selected_best = False
        validation_record: dict[str, Any] | None = None
        if should_validate:
            if validation_frame is None:
                raise CascadeSpecError(
                    "Development training is missing its internal validation frame."
                )
            validation_agent_counters = (
                int(agent.env_steps),
                int(agent.learning_steps),
                len(agent.replay),
                float(agent.epsilon),
            )
            validation_started = time.perf_counter()
            _, validation_summary = evaluate_greedy_trajectory(
                reservoir_1,
                reservoir_2,
                agent,
                normalizer,
                action_grid,
                validation_frame,
                settings.initial_levels_m,
                training_inflow_sha256=training_hash,
                seed=settings.seed,
                reward_reference_days=settings.reward_reference_days,
                reward_clip=settings.reward_clip,
                firm_power_failure_penalty_gwh=(
                    settings.firm_power_failure_penalty_gwh
                ),
                monthly_max_target_levels_m=monthly_max_target_levels_m,
                trajectory_kind="validation",
                collect_rows=False,
            )
            validation_elapsed_seconds = time.perf_counter() - validation_started
            validation_wall_time_seconds += validation_elapsed_seconds
            if validation_agent_counters != (
                int(agent.env_steps),
                int(agent.learning_steps),
                len(agent.replay),
                float(agent.epsilon),
            ):
                raise CascadeSpecError(
                    "Greedy validation modified DQN training counters, replay, or epsilon."
                )
            validation_score = float(
                validation_summary["mean_annual_penalized_objective_gwh"]
            )
            if not np.isfinite(validation_score):
                raise CascadeSpecError("DQN validation produced a non-finite score.")
            selected_best = bool(
                best_validation_score is None
                or validation_score > best_validation_score
            )
            if selected_best:
                best_validation_score = validation_score
                best_validation_episode = episode
            validation_record = _validation_record(
                episode=episode,
                agent=agent,
                summary=validation_summary,
                elapsed_seconds=validation_elapsed_seconds,
                selected=selected_best,
            )
            validation_records.append(validation_record)

        should_save_latest = bool(
            episode % settings.checkpoint_interval_episodes == 0
            or should_validate
            or episode == settings.episodes
        )
        checkpoint_events: list[str] = []
        if should_save_latest:
            checkpoint_events.append("latest")
        if selected_best:
            checkpoint_events.append("best_validation")
        if settings.mode == "final" and episode == settings.episodes:
            checkpoint_events.append("final")

        training_record = {
            "episode": episode,
            "episode_start_index": start_index,
            "episode_start_date": str(periods[0]),
            "episode_end_date": str(periods[settings.episode_months - 1]),
            "episode_transitions": settings.episode_months,
            "environment_steps": int(agent.env_steps),
            "learning_steps": int(agent.learning_steps),
            "learning_updates": learning_updates,
            "raw_episode_cascade_generation_gwh": raw_episode_generation_gwh,
            "raw_episode_penalized_objective_gwh": (
                raw_episode_penalized_objective_gwh
            ),
            "episode_failed_station_months": episode_failed_station_months,
            "episode_failure_penalty_gwh": episode_failure_penalty_gwh,
            "normalized_episode_reward": normalized_episode_reward,
            "reward_scale_gwh": float(reward_scaling["scale_gwh"]),
            "mean_loss": _mean_or_none(losses),
            "last_loss": last_loss,
            "mean_predicted_q": _mean_or_none(predicted_q_values),
            "mean_target_q": _mean_or_none(target_q_values),
            "mean_gradient_norm": _mean_or_none(gradient_norms),
            "epsilon": float(agent.epsilon),
            "mean_feasible_action_count": float(
                np.mean(feasible_action_counts)
            ),
            "min_feasible_action_count": int(min(feasible_action_counts)),
            "xiaowan_target_unreached_count": int(counts["xiaowan"]),
            "xiaowan_target_unreached_rate": float(rates["xiaowan"]),
            "nuozhadu_target_unreached_count": int(counts["nuozhadu"]),
            "nuozhadu_target_unreached_rate": float(rates["nuozhadu"]),
            "either_reservoir_target_unreached_count": int(
                counts["either_reservoir"]
            ),
            "either_reservoir_target_unreached_rate": float(
                rates["either_reservoir"]
            ),
            "validation_performed": should_validate,
            "checkpoint_event": ",".join(checkpoint_events),
        }
        training_records.append(training_record)

        if progress_callback is not None and (
            episode % settings.logging_interval_episodes == 0
            or should_validate
            or should_save_latest
        ):
            progress_callback(
                {
                    "event": "training_progress",
                    "mode": settings.mode,
                    "seed": settings.seed,
                    "training": dict(training_record),
                    "validation": (
                        None if validation_record is None else dict(validation_record)
                    ),
                    "paths": paths.as_dict(),
                }
            )

        if selected_best:
            if paths.best_validation_checkpoint is None:
                raise CascadeSpecError(
                    "Development mode is missing a best-validation checkpoint path."
                )
            save_role(paths.best_validation_checkpoint, "best_validation", episode)
        if should_save_latest:
            save_role(paths.latest_checkpoint, "latest", episode)

        should_write_history = bool(
            episode % settings.logging_interval_episodes == 0
            or should_validate
            or should_save_latest
            or episode == settings.episodes
        )
        if should_write_history:
            _atomic_write_history(
                training_records,
                TRAINING_HISTORY_COLUMNS,
                paths.training_history_csv,
            )
            if paths.validation_history_csv is not None:
                _atomic_write_history(
                    validation_records,
                    VALIDATION_HISTORY_COLUMNS,
                    paths.validation_history_csv,
                )
        completed_episode = episode

    if settings.mode == "final":
        if paths.final_checkpoint is None:
            raise CascadeSpecError("Final mode is missing its final checkpoint path.")
        # Also rewrite this after resuming an already-complete latest checkpoint:
        # a previous process may have stopped after latest but before final.
        save_role(paths.final_checkpoint, "final", completed_episode)
        selected_checkpoint = paths.final_checkpoint
    else:
        if paths.best_validation_checkpoint is None:
            raise CascadeSpecError(
                "Development mode is missing its best-validation checkpoint path."
            )
        if best_validation_score is None or best_validation_episode is None:
            raise CascadeSpecError(
                "Development training completed without a validation selection."
            )
        if not paths.best_validation_checkpoint.is_file():
            raise CascadeSpecError(
                "Development training has no persisted best-validation checkpoint."
            )
        selected_payload = load_training_checkpoint(
            paths.best_validation_checkpoint, map_location="cpu"
        )
        selected_expectations = {
            "checkpoint_role": "best_validation",
            "experiment_id": settings.experiment_id,
            "mode": settings.mode,
            "seed": settings.seed,
            "completed_episode": best_validation_episode,
            "best_validation_episode": best_validation_episode,
            "best_validation_score": best_validation_score,
            "configuration_sha256": config_hash,
            "reservoir_physics_sha256": physics_hash,
            "training_inflow_sha256": training_hash,
        }
        for key, value in selected_expectations.items():
            if selected_payload.get(key) != value:
                raise CascadeSpecError(
                    f"Best-validation DQN checkpoint {key} does not match the selected model."
                )
        selected_checkpoint = paths.best_validation_checkpoint

    return DQNTrainingResult(
        mode=settings.mode,
        experiment_id=settings.experiment_id,
        seed=settings.seed,
        completed_episodes=completed_episode,
        environment_steps=int(agent.env_steps),
        learning_steps=int(agent.learning_steps),
        learning_starts=settings.learning_starts,
        training_wall_time_seconds=current_training_wall_time_seconds(),
        validation_wall_time_seconds=validation_wall_time_seconds,
        best_validation_score=best_validation_score,
        best_validation_episode=best_validation_episode,
        configuration_sha256=config_hash,
        normalization=normalizer.state_dict(),
        action_grid=action_grid.state_dict(),
        reward_scaling=dict(reward_scaling),
        training_inflow_sha256=training_hash,
        source_training_inflow_sha256=source_training_hash,
        paths=paths,
        selected_checkpoint=selected_checkpoint,
        training_records=tuple(dict(record) for record in training_records),
        validation_records=tuple(dict(record) for record in validation_records),
        budget_extension_provenance=budget_extension_provenance,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEVELOPMENT_YEARS",
    "DQNTrainingPaths",
    "DQNTrainingResult",
    "EPISODE_MONTHS",
    "TOTAL_TRAINING_YEARS",
    "TRAINING_HISTORY_COLUMNS",
    "TRAINING_MODES",
    "VALIDATION_HISTORY_COLUMNS",
    "VALIDATION_YEARS",
    "atomic_torch_save",
    "build_training_paths",
    "configuration_sha256",
    "experiment_id_from_config",
    "inspect_dqn_budget_extension",
    "inspect_dqn_configuration",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "split_training_inflows",
    "train_dqn_seed",
]
