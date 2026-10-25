"""Reproducible development/final training for the cascade discrete PPO."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
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
)
from .dqn_training import (
    configuration_sha256,
    experiment_id_from_config,
    split_training_inflows,
)
from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    _as_month_period_index,
)
from .ppo_adapter import (
    CascadePPOAdapter,
    SynchronousPPOEnvironmentSet,
    make_episode_sampler_rngs,
)
from .ppo_agent import (
    ACTION_CONDITIONED_ACTOR_MODE,
    ACTION_MASK_MODE,
    DiscretePPOAgent,
    PPORolloutBatch,
)
from .ppo_evaluation import evaluate_deterministic_trajectory
from .sdp import reservoir_pair_sha256


PPO_CHECKPOINT_SCHEMA_VERSION = 3
PPO_TRAINING_MODES = ("development", "final")
DEVELOPMENT_YEARS = 320
VALIDATION_YEARS = 80
EPISODE_MONTHS = 240
NUM_ENVS = 8
N_STEPS = 240
ROLLOUT_SIZE = NUM_ENVS * N_STEPS
DEFAULT_TOTAL_ENVIRONMENT_STEPS = 720000
ENTROPY_NORMALIZATION = "divide_by_log_action_count"
PPO_REWARD_SCALE_GWH = 7332.48
PPO_ACTION_GRID_SHAPE = (75, 48)
PPO_ACTION_COUNT = 3600


ProgressCallback = Callable[[Mapping[str, Any]], None]


PPO_TRAINING_HISTORY_COLUMNS = (
    "update_number", "environment_steps", "completed_episode_windows",
    "rollout_size", "minibatch_updates", "episode_start_indices_json",
    "episode_start_dates_json", "episode_end_dates_json",
    "episode_bootstrap_dates_json", "raw_episode_generation_values_json",
    "raw_episode_penalized_objective_values_json",
    "episode_failed_station_months_values_json",
    "episode_failure_penalty_values_gwh_json",
    "normalized_episode_reward_values_json", "raw_generation_mean_gwh",
    "raw_generation_std_gwh", "raw_penalized_objective_mean_gwh",
    "raw_penalized_objective_std_gwh", "failed_station_months_mean",
    "failure_penalty_mean_gwh", "normalized_reward_mean",
    "normalized_reward_std", "policy_loss", "value_loss", "raw_entropy",
    "normalized_entropy", "rollout_raw_entropy", "rollout_normalized_entropy",
    "total_loss", "approx_kl", "clip_fraction", "explained_variance",
    "gradient_norm", "actor_gradient_norm", "critic_gradient_norm",
    "learning_rate", "advantage_mean_before_normalization",
    "advantage_std_before_normalization", "advantage_mean_after_normalization",
    "advantage_std_after_normalization", "return_mean", "return_std",
    "old_value_prediction_mean", "old_value_prediction_std",
    "final_value_prediction_mean", "final_value_prediction_std",
    "ratio_mean", "ratio_std", "ratio_max", "effective_action_count",
    "rollout_effective_action_count", "max_action_probability",
    "rollout_max_action_probability", "unique_sampled_actions",
    "mean_feasible_action_count", "min_feasible_action_count",
    "terminated_count", "truncated_count", "bootstrap_allowed_count",
    "xiaowan_target_unreached_count", "xiaowan_target_unreached_rate",
    "nuozhadu_target_unreached_count", "nuozhadu_target_unreached_rate",
    "either_reservoir_target_unreached_count",
    "either_reservoir_target_unreached_rate",
    "xiaowan_target_abs_error_mean_m", "xiaowan_target_abs_error_max_m",
    "xiaowan_target_abs_error_p95_m", "nuozhadu_target_abs_error_mean_m",
    "nuozhadu_target_abs_error_max_m", "nuozhadu_target_abs_error_p95_m",
    "rollout_wall_time_seconds", "optimization_wall_time_seconds",
    "cumulative_training_wall_time_seconds",
    "cumulative_validation_wall_time_seconds", "samples_per_second",
    "validation_performed", "validation_score",
    "best_validation_score", "best_validation_step", "checkpoint_event",
)

PPO_VALIDATION_HISTORY_COLUMNS = (
    "update_number", "environment_steps", "validation_wall_time_seconds",
    "validation_months", "validation_total_cascade_generation_gwh",
    "validation_mean_annual_cascade_generation_gwh",
    "validation_total_penalized_objective_gwh",
    "validation_mean_annual_penalized_objective_gwh",
    "validation_total_failed_station_months",
    "validation_total_failure_penalty_gwh",
    "xiaowan_firm_power_reliability", "nuozhadu_firm_power_reliability",
    "cascade_firm_power_reliability", "xiaowan_target_unreached_count",
    "xiaowan_target_unreached_rate", "nuozhadu_target_unreached_count",
    "nuozhadu_target_unreached_rate",
    "either_reservoir_target_unreached_count",
    "either_reservoir_target_unreached_rate", "mean_raw_policy_entropy",
    "mean_normalized_policy_entropy", "mean_max_action_probability",
    "unique_selected_actions", "score",
    "selected_best_validation_checkpoint",
)


@dataclass(frozen=True)
class PPOTrainingPaths:
    output_root: Path
    run_root: Path
    experiment_id: str
    configuration_snapshot_json: Path
    metadata_json: Path
    training_history_csv: Path
    validation_history_csv: Path | None
    latest_checkpoint: Path
    best_validation_checkpoint: Path | None
    final_checkpoint: Path | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            name: None if value is None else str(value)
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class PPOTrainingResult:
    mode: str
    experiment_id: str
    seed: int
    completed_updates: int
    environment_steps: int
    completed_episode_windows: int
    training_wall_time_seconds: float
    validation_wall_time_seconds: float
    best_validation_score: float | None
    best_validation_step: int | None
    configuration_sha256: str
    normalization: dict[str, Any]
    action_grid: dict[str, Any]
    reward_scaling: dict[str, Any]
    paths: PPOTrainingPaths
    selected_checkpoint: Path

    def as_dict(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.__dict__)
        payload["paths"] = self.paths.as_dict()
        payload["selected_checkpoint"] = str(self.selected_checkpoint)
        return payload


@dataclass(frozen=True)
class _PPOSettings:
    mode: str
    experiment_id: str
    seed: int
    action_level_step_m: float
    initial_levels_m: tuple[float, float]
    firm_power_failure_penalty_gwh: float
    reward_reference_days: float
    reward_clip: tuple[float, float] | None
    normalization_clip: bool
    actor_hidden_dimensions: tuple[int, ...]
    critic_hidden_dimensions: tuple[int, ...]
    action_score_chunk_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    normalized_entropy_coefficient: float
    value_loss_coefficient: float
    max_gradient_norm: float
    minibatch_size: int
    update_epochs: int
    advantage_normalization: bool
    adam_epsilon: float
    total_environment_steps: int
    num_envs: int
    n_steps: int
    episode_months: int
    validation_interval_environment_steps: int
    logging_interval_updates: int
    checkpoint_interval_updates: int
    device: str

    @property
    def rollout_size(self) -> int:
        return self.num_envs * self.n_steps

    @property
    def total_updates(self) -> int:
        return self.total_environment_steps // self.rollout_size


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CascadeSpecError(f"PPO configuration section {key!r} must be a mapping.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CascadeSpecError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise CascadeSpecError(f"{name} must be positive.")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CascadeSpecError(f"{name} must be numeric.") from exc
    if not np.isfinite(result):
        raise CascadeSpecError(f"{name} must be finite.")
    return result


def _resolve_device(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadeSpecError("training.device must be a nonempty string.")
    requested = value.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CascadeSpecError("CUDA was requested for PPO but is unavailable.")
    return str(device)


def _parse_reward_clip(reward: Mapping[str, Any]) -> tuple[float, float] | None:
    clip = reward.get("clip")
    if not isinstance(clip, Mapping):
        raise CascadeSpecError("reward.clip must be a mapping.")
    if clip.get("enabled") is False:
        return None
    if clip.get("enabled") is not True:
        raise CascadeSpecError("reward.clip.enabled must be boolean.")
    lower = _finite_float(clip.get("lower"), "reward.clip.lower")
    upper = _finite_float(clip.get("upper"), "reward.clip.upper")
    if upper <= lower:
        raise CascadeSpecError("reward.clip upper must exceed lower.")
    return lower, upper


def _hidden_dimensions(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CascadeSpecError(f"{name} must be a sequence.")
    result = tuple(
        _positive_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise CascadeSpecError(f"{name} cannot be empty.")
    return result


def _settings_from_config(
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> _PPOSettings:
    if mode not in PPO_TRAINING_MODES:
        raise CascadeSpecError(f"mode must be one of {PPO_TRAINING_MODES}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise CascadeSpecError("seed must be an integer.")
    seed_value = int(seed)
    if seed_value < 0:
        raise CascadeSpecError("seed must be nonnegative.")
    experiment_id = experiment_id_from_config(config)
    if experiment_id is None:
        raise CascadeSpecError("Formal PPO configurations require experiment_id.")

    problem = _mapping(config, "problem")
    information = _mapping(config, "information")
    reward = _mapping(config, "reward")
    normalization = _mapping(config, "normalization")
    validation = _mapping(config, "validation")
    ppo = _mapping(config, "ppo")
    network = _mapping(config, "network")
    training = _mapping(config, "training")

    if information.get("one_step_perfect_forecast") is not True:
        raise CascadeSpecError("PPO requires one_step_perfect_forecast=true.")
    if list(information.get("state_order", [])) != list(STATE_ORDER):
        raise CascadeSpecError("PPO state order must match the DQN state order.")
    if information.get("nuozhadu_total_inflow_is_action_dependent") is not True:
        raise CascadeSpecError("PPO requires action-dependent NuoZhaDu total inflow.")
    if reward.get("objective") != "generation_gwh_minus_penalty_gwh_per_failed_station_month":
        raise CascadeSpecError("PPO reward objective is incompatible.")
    failure_penalty = _finite_float(
        reward.get("firm_power_failure_penalty_gwh"),
        "reward.firm_power_failure_penalty_gwh",
    )
    if failure_penalty < 0.0:
        raise CascadeSpecError("PPO firm-power failure penalty must be nonnegative.")
    if reward.get("failure_definition") != "station_specific_monthly_power_below_firm_power":
        raise CascadeSpecError("PPO firm-power failure definition is incompatible.")
    if reward.get("deficit_energy_role") != "diagnostic_only_not_in_objective":
        raise CascadeSpecError("PPO deficit energy must remain diagnostic only.")
    if _finite_float(reward.get("scale_gwh"), "reward.scale_gwh") != (
        PPO_REWARD_SCALE_GWH
    ):
        raise CascadeSpecError("PPO reward scale is frozen at 7332.48 GWh.")
    if normalization.get("method") != StateNormalizer.method:
        raise CascadeSpecError("PPO requires componentwise_minmax normalization.")
    if normalization.get("clip") is not True:
        raise CascadeSpecError("PPO requires clipped state normalization.")
    if validation.get("enabled_in_development_mode") is not True:
        raise CascadeSpecError("PPO development validation must be enabled.")
    if int(validation.get("development_years")) != DEVELOPMENT_YEARS:
        raise CascadeSpecError("PPO development learning must use 320 years.")
    if int(validation.get("validation_years")) != VALIDATION_YEARS:
        raise CascadeSpecError("PPO validation must use the final 80 training years.")
    if validation.get("score") != "mean_annual_penalized_objective_gwh":
        raise CascadeSpecError("PPO validation score is incompatible.")

    action_step = _finite_float(
        problem.get("action_level_step_m"), "problem.action_level_step_m"
    )
    if action_step != 1.0:
        raise CascadeSpecError("Formal PPO requires a 1.0 m action grid.")
    initial = np.asarray(problem.get("initial_levels_m"), dtype=float)
    if not np.array_equal(initial, np.array([1240.0, 812.0], dtype=float)):
        raise CascadeSpecError("PPO initial levels must be [1240.0, 812.0].")
    episode_months = _positive_int(
        training.get("episode_months"), "training.episode_months"
    )
    num_envs = _positive_int(training.get("num_envs"), "training.num_envs")
    n_steps = _positive_int(training.get("n_steps"), "training.n_steps")
    if episode_months != EPISODE_MONTHS or n_steps != N_STEPS:
        raise CascadeSpecError("PPO requires 240-step complete artificial episodes.")
    if num_envs != NUM_ENVS:
        raise CascadeSpecError("PPO requires eight logical training environments.")
    total_steps = _positive_int(
        training.get("total_environment_steps"),
        "training.total_environment_steps",
    )
    rollout_size = num_envs * n_steps
    if total_steps % rollout_size != 0:
        raise CascadeSpecError("PPO total steps must be divisible by rollout size.")
    minibatch_size = _positive_int(
        ppo.get("minibatch_size"), "ppo.minibatch_size"
    )
    if rollout_size % minibatch_size != 0:
        raise CascadeSpecError("PPO rollout size must divide evenly into minibatches.")

    gamma = _finite_float(ppo.get("gamma"), "ppo.gamma")
    if gamma != 0.97:
        raise CascadeSpecError("PPO gamma is frozen at 0.97.")
    if ppo.get("entropy_normalization") != ENTROPY_NORMALIZATION:
        raise CascadeSpecError(
            "PPO entropy must be normalized by log(action_count)."
        )
    if ppo.get("advantage_normalization") is not True:
        raise CascadeSpecError("PPO advantage normalization must be enabled.")
    if ppo.get("value_clipping") is not False:
        raise CascadeSpecError("PPO value clipping must be disabled.")
    if ppo.get("target_kl_early_stop") is not False:
        raise CascadeSpecError("PPO target-KL early stopping must be disabled.")
    if ppo.get("lr_schedule") != "constant":
        raise CascadeSpecError("PPO learning-rate schedule must be constant.")
    if network.get("activation") != "tanh":
        raise CascadeSpecError("PPO activation must be tanh.")
    if network.get("separate_actor_critic") is not True:
        raise CascadeSpecError("PPO Actor and Critic must be separate networks.")
    if network.get("action_representation") != ACTION_CONDITIONED_ACTOR_MODE:
        raise CascadeSpecError(
            "PPO Actor must use action_representation='action_conditioned'."
        )
    if int(network.get("action_feature_dimension", -1)) != 2:
        raise CascadeSpecError("PPO action feature dimension must equal two.")
    action_score_chunk_size = _positive_int(
        network.get("action_score_chunk_size"),
        "network.action_score_chunk_size",
    )
    if network.get("layer_norm") is not False:
        raise CascadeSpecError("PPO LayerNorm must be disabled.")
    if network.get("initialization") != "orthogonal":
        raise CascadeSpecError("PPO initialization must be orthogonal.")
    if _finite_float(network.get("hidden_gain"), "network.hidden_gain") != float(
        np.sqrt(2.0)
    ):
        raise CascadeSpecError("PPO hidden orthogonal gain must equal sqrt(2).")
    if _finite_float(
        network.get("actor_output_gain"), "network.actor_output_gain"
    ) != 0.01:
        raise CascadeSpecError("PPO Actor output gain is frozen at 0.01.")
    if _finite_float(
        network.get("critic_output_gain"), "network.critic_output_gain"
    ) != 1.0:
        raise CascadeSpecError("PPO Critic output gain is frozen at 1.0.")
    if ppo.get("action_distribution") != "categorical_logits":
        raise CascadeSpecError("PPO must use Categorical logits.")
    if ppo.get("deterministic_policy") != "argmax_logits":
        raise CascadeSpecError("PPO deterministic policy must be argmax logits.")
    actor_hidden = _hidden_dimensions(
        network.get("actor_hidden_dimensions"),
        "network.actor_hidden_dimensions",
    )
    critic_hidden = _hidden_dimensions(
        network.get("critic_hidden_dimensions"),
        "network.critic_hidden_dimensions",
    )
    if actor_hidden != (512, 256) or critic_hidden != (256, 256):
        raise CascadeSpecError("PPO Actor/Critic hidden dimensions are frozen.")
    if minibatch_size != 320:
        raise CascadeSpecError("PPO minibatch_size is frozen at 320.")
    if _positive_int(ppo.get("update_epochs"), "ppo.update_epochs") != 10:
        raise CascadeSpecError("PPO update_epochs is frozen at 10.")
    if _finite_float(
        ppo.get("value_loss_coefficient"), "ppo.value_loss_coefficient"
    ) != 0.5:
        raise CascadeSpecError("PPO value-loss coefficient is frozen at 0.5.")
    if _finite_float(ppo.get("max_gradient_norm"), "ppo.max_gradient_norm") != 0.5:
        raise CascadeSpecError("PPO max gradient norm is frozen at 0.5.")
    if _finite_float(ppo.get("adam_epsilon"), "ppo.adam_epsilon") != 1.0e-5:
        raise CascadeSpecError("PPO Adam epsilon is frozen at 1e-5.")
    learning_rate = _finite_float(ppo.get("learning_rate"), "ppo.learning_rate")
    gae_lambda = _finite_float(ppo.get("gae_lambda"), "ppo.gae_lambda")
    clip_range = _finite_float(ppo.get("clip_range"), "ppo.clip_range")
    entropy_coefficient = _finite_float(
        ppo.get("normalized_entropy_coefficient"),
        "ppo.normalized_entropy_coefficient",
    )
    if learning_rate <= 0.0:
        raise CascadeSpecError("PPO learning rate must be positive.")
    if not 0.0 <= gae_lambda <= 1.0:
        raise CascadeSpecError("PPO GAE lambda must lie in [0, 1].")
    if not 0.0 < clip_range < 1.0:
        raise CascadeSpecError("PPO clip range must lie in (0, 1).")
    if entropy_coefficient < 0.0:
        raise CascadeSpecError("PPO entropy coefficient must be nonnegative.")

    return _PPOSettings(
        mode=mode,
        experiment_id=experiment_id,
        seed=seed_value,
        action_level_step_m=action_step,
        initial_levels_m=(float(initial[0]), float(initial[1])),
        firm_power_failure_penalty_gwh=failure_penalty,
        reward_reference_days=_finite_float(
            reward.get("reward_reference_days"), "reward.reward_reference_days"
        ),
        reward_clip=_parse_reward_clip(reward),
        normalization_clip=True,
        actor_hidden_dimensions=actor_hidden,
        critic_hidden_dimensions=critic_hidden,
        action_score_chunk_size=action_score_chunk_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        normalized_entropy_coefficient=entropy_coefficient,
        value_loss_coefficient=_finite_float(
            ppo.get("value_loss_coefficient"), "ppo.value_loss_coefficient"
        ),
        max_gradient_norm=_finite_float(
            ppo.get("max_gradient_norm"), "ppo.max_gradient_norm"
        ),
        minibatch_size=minibatch_size,
        update_epochs=_positive_int(ppo.get("update_epochs"), "ppo.update_epochs"),
        advantage_normalization=True,
        adam_epsilon=_finite_float(ppo.get("adam_epsilon"), "ppo.adam_epsilon"),
        total_environment_steps=total_steps,
        num_envs=num_envs,
        n_steps=n_steps,
        episode_months=episode_months,
        validation_interval_environment_steps=_positive_int(
            validation.get("interval_environment_steps"),
            "validation.interval_environment_steps",
        ),
        logging_interval_updates=_positive_int(
            training.get("logging_interval_updates"),
            "training.logging_interval_updates",
        ),
        checkpoint_interval_updates=_positive_int(
            training.get("checkpoint_interval_updates"),
            "training.checkpoint_interval_updates",
        ),
        device=_resolve_device(training.get("device")),
    )


def build_ppo_training_paths(
    output_root: str | Path,
    *,
    mode: str,
    seed: int,
    experiment_id: str,
) -> PPOTrainingPaths:
    if mode not in PPO_TRAINING_MODES:
        raise CascadeSpecError(f"mode must be one of {PPO_TRAINING_MODES}.")
    root = Path(output_root)
    run_root = root / "experiments" / experiment_id / f"seed_{int(seed)}" / mode
    checkpoint_dir = run_root / "checkpoints"
    return PPOTrainingPaths(
        output_root=root,
        run_root=run_root,
        experiment_id=experiment_id,
        configuration_snapshot_json=run_root / "config_snapshot.json",
        metadata_json=run_root / "metadata.json",
        training_history_csv=run_root / "training" / "history.csv",
        validation_history_csv=(
            run_root / "validation" / "history.csv"
            if mode == "development"
            else None
        ),
        latest_checkpoint=checkpoint_dir / "latest.pt",
        best_validation_checkpoint=(
            checkpoint_dir / "best_validation.pt"
            if mode == "development"
            else None
        ),
        final_checkpoint=(checkpoint_dir / "final.pt" if mode == "final" else None),
    )


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
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(dict(payload), temporary_path)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _normalized_action_features(action_grid: JointActionGrid) -> np.ndarray:
    """Return row-major normalized target levels for all PPO action IDs."""
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
        raise CascadeSpecError("PPO action features do not match the action grid.")
    return features


def _new_agent(
    settings: _PPOSettings, action_grid: JointActionGrid
) -> DiscretePPOAgent:
    return DiscretePPOAgent(
        state_dimension=len(STATE_ORDER),
        action_count=action_grid.action_count,
        action_features=_normalized_action_features(action_grid),
        action_score_chunk_size=settings.action_score_chunk_size,
        actor_hidden_dimensions=settings.actor_hidden_dimensions,
        critic_hidden_dimensions=settings.critic_hidden_dimensions,
        learning_rate=settings.learning_rate,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
        clip_range=settings.clip_range,
        normalized_entropy_coefficient=settings.normalized_entropy_coefficient,
        value_loss_coefficient=settings.value_loss_coefficient,
        max_gradient_norm=settings.max_gradient_norm,
        minibatch_size=settings.minibatch_size,
        update_epochs=settings.update_epochs,
        advantage_normalization=settings.advantage_normalization,
        adam_epsilon=settings.adam_epsilon,
        seed=settings.seed,
        device=settings.device,
    )


def _validate_formal_action_grid(action_grid: JointActionGrid) -> None:
    if tuple(action_grid.shape) != PPO_ACTION_GRID_SHAPE:
        raise CascadeSpecError(
            "Formal PPO requires a 75 by 48 joint target-level grid."
        )
    if action_grid.action_count != PPO_ACTION_COUNT:
        raise CascadeSpecError("Formal PPO requires exactly 3600 joint actions.")


def _mlp_parameter_count(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    output_dimension: int,
) -> int:
    dimensions = [input_dimension, *hidden_dimensions, output_dimension]
    return int(
        sum(
            dimensions[index] * dimensions[index + 1]
            + dimensions[index + 1]
            for index in range(len(dimensions) - 1)
        )
    )


def inspect_ppo_configuration(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    settings = _settings_from_config(config, mode=mode, seed=seed)
    action_grid = JointActionGrid(
        reservoir_1,
        reservoir_2,
        step_m=settings.action_level_step_m,
    )
    _validate_formal_action_grid(action_grid)
    state_trunk_parameters = _mlp_parameter_count(
        len(STATE_ORDER), settings.actor_hidden_dimensions, 0
    )
    final_hidden = settings.actor_hidden_dimensions[-1]
    embedding_dimension = min(128, final_hidden)
    actor_parameters = (
        state_trunk_parameters
        + final_hidden * embedding_dimension + embedding_dimension
        + 2 * embedding_dimension + embedding_dimension
        + embedding_dimension * embedding_dimension + embedding_dimension
        + 2 * embedding_dimension + embedding_dimension
        + embedding_dimension + 1
    )
    critic_parameters = _mlp_parameter_count(
        len(STATE_ORDER), settings.critic_hidden_dimensions, 1
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
        "action_mask_mode": ACTION_MASK_MODE,
        "actor_mode": ACTION_CONDITIONED_ACTOR_MODE,
        "policy_observation_dimension": len(STATE_ORDER),
        "action_feature_dimension": 2,
        "action_features": [
            "normalized_xiaowan_target_level",
            "normalized_nuozhadu_target_level",
        ],
        "action_score_chunk_size": settings.action_score_chunk_size,
        "actor_state_trunk_dimensions": [
            len(STATE_ORDER), *settings.actor_hidden_dimensions
        ],
        "actor_embedding_dimension": embedding_dimension,
        "critic_dimensions": [
            len(STATE_ORDER), *settings.critic_hidden_dimensions, 1
        ],
        "actor_parameter_count": actor_parameters,
        "critic_parameter_count": critic_parameters,
        "total_parameter_count": actor_parameters + critic_parameters,
        "num_envs": settings.num_envs,
        "n_steps": settings.n_steps,
        "rollout_batch_size": settings.rollout_size,
        "minibatch_size": settings.minibatch_size,
        "minibatches_per_epoch": settings.rollout_size // settings.minibatch_size,
        "update_epochs": settings.update_epochs,
        "optimizer_steps_per_update": (
            settings.update_epochs * settings.rollout_size // settings.minibatch_size
        ),
        "total_environment_steps": settings.total_environment_steps,
        "total_updates": settings.total_updates,
        "completed_episode_windows": settings.total_updates * settings.num_envs,
        "validation_interval_environment_steps": (
            settings.validation_interval_environment_steps
        ),
        "entropy_normalization": ENTROPY_NORMALIZATION,
        "maximum_raw_entropy": float(np.log(action_grid.action_count)),
        "reward_scale_gwh": PPO_REWARD_SCALE_GWH,
        "device": settings.device,
    }


def _reward_metadata(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    learning_frame: pd.DataFrame,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    settings: _PPOSettings,
    monthly_max_target_levels_m: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    periods = _as_month_period_index(learning_frame["date"])
    flows = learning_frame.loc[:, list(INFLOW_COLUMNS)].to_numpy(dtype=float)
    env = CascadeReservoirEnv(
        reservoir_1,
        reservoir_2,
        flows[: settings.episode_months + 1],
        timestamps=periods[: settings.episode_months + 1],
        initial_levels_m=settings.initial_levels_m,
        firm_power_failure_penalty_gwh=(
            settings.firm_power_failure_penalty_gwh
        ),
        monthly_max_target_levels_m=monthly_max_target_levels_m,
        require_energy_parameters=True,
    )
    adapter = CascadePPOAdapter(
        env,
        normalizer,
        action_grid,
        reward_reference_days=settings.reward_reference_days,
        reward_clip=settings.reward_clip,
        max_transitions=settings.episode_months,
    )
    metadata = adapter.reward_metadata
    if not np.isclose(
        float(metadata["scale_gwh"]),
        PPO_REWARD_SCALE_GWH,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise CascadeSpecError(
            "Shared DQN reward scaling no longer equals the frozen PPO scale."
        )
    return metadata


def _validation_record(
    *,
    update_number: int,
    environment_steps: int,
    summary: Mapping[str, Any],
    selected: bool,
) -> dict[str, Any]:
    reliability = summary["firm_power_reliability"]
    counts = summary["target_unreached_count"]
    rates = summary["target_unreached_rate"]
    score = float(summary["mean_annual_penalized_objective_gwh"])
    return {
        "update_number": update_number,
        "environment_steps": environment_steps,
        "validation_wall_time_seconds": float(
            summary["validation_wall_time_seconds"]
        ),
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
        "validation_total_failed_station_months": int(
            summary["failed_station_months"]
        ),
        "validation_total_failure_penalty_gwh": float(
            summary["total_failure_penalty_gwh"]
        ),
        "xiaowan_firm_power_reliability": float(reliability["xiaowan"]),
        "nuozhadu_firm_power_reliability": float(reliability["nuozhadu"]),
        "cascade_firm_power_reliability": float(reliability["cascade"]),
        "xiaowan_target_unreached_count": int(counts["xiaowan"]),
        "xiaowan_target_unreached_rate": float(rates["xiaowan"]),
        "nuozhadu_target_unreached_count": int(counts["nuozhadu"]),
        "nuozhadu_target_unreached_rate": float(rates["nuozhadu"]),
        "either_reservoir_target_unreached_count": int(counts["either_reservoir"]),
        "either_reservoir_target_unreached_rate": float(rates["either_reservoir"]),
        "mean_raw_policy_entropy": float(summary["mean_raw_policy_entropy"]),
        "mean_normalized_policy_entropy": float(
            summary["mean_normalized_policy_entropy"]
        ),
        "mean_max_action_probability": float(
            summary["mean_max_action_probability"]
        ),
        "unique_selected_actions": int(summary["unique_selected_actions"]),
        "score": score,
        "selected_best_validation_checkpoint": bool(selected),
    }


def _checkpoint_payload(
    *,
    role: str,
    settings: _PPOSettings,
    agent: DiscretePPOAgent,
    vector_env: SynchronousPPOEnvironmentSet,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    reward_scaling: Mapping[str, Any],
    configuration_snapshot: Mapping[str, Any],
    configuration_hash: str,
    reservoir_physics_hash: str,
    physical_configuration_hash: str | None,
    training_inflow_hash: str,
    source_training_inflow_hash: str,
    validation_inflow_hash: str | None,
    completed_update: int,
    environment_steps: int,
    next_validation_threshold: int,
    training_wall_time_seconds: float,
    validation_wall_time_seconds: float,
    best_validation_score: float | None,
    best_validation_step: int | None,
    training_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PPO_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": role,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": settings.mode,
        "experiment_id": settings.experiment_id,
        "seed": settings.seed,
        "completed_update": int(completed_update),
        "environment_steps": int(environment_steps),
        "completed_episode_windows": int(completed_update * settings.num_envs),
        "next_validation_threshold": int(next_validation_threshold),
        "total_environment_steps_budget": settings.total_environment_steps,
        "rollout_size": settings.rollout_size,
        "num_envs": settings.num_envs,
        "n_steps": settings.n_steps,
        "rollout_buffer_saved": False,
        "checkpoint_boundary": "complete_rollout_and_ppo_update",
        "agent_state": agent.training_state(),
        "episode_sampler_rng_states": vector_env.sampler_rng_states(),
        "normalizer": normalizer.state_dict(),
        "action_grid": action_grid.state_dict(),
        "reward_scaling": copy.deepcopy(dict(reward_scaling)),
        "configuration_snapshot": copy.deepcopy(dict(configuration_snapshot)),
        "configuration_sha256": configuration_hash,
        "reservoir_physics_sha256": reservoir_physics_hash,
        "physical_configuration_sha256": physical_configuration_hash,
        "training_inflow_sha256": training_inflow_hash,
        "source_training_inflow_sha256": source_training_inflow_hash,
        "validation_inflow_sha256": validation_inflow_hash,
        "training_wall_time_seconds": float(training_wall_time_seconds),
        "validation_wall_time_seconds": float(validation_wall_time_seconds),
        "best_validation_score": best_validation_score,
        "best_validation_step": best_validation_step,
        "training_records": [dict(record) for record in training_records],
        "validation_records": [dict(record) for record in validation_records],
    }


def load_ppo_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise CascadeSpecError("PPO checkpoint must contain a dictionary.")
    if payload.get("schema_version") != PPO_CHECKPOINT_SCHEMA_VERSION:
        raise CascadeSpecError("Unsupported PPO checkpoint schema.")
    required = {
        "checkpoint_role", "mode", "experiment_id", "seed",
        "completed_update", "environment_steps", "completed_episode_windows",
        "next_validation_threshold", "total_environment_steps_budget",
        "rollout_size", "num_envs", "n_steps", "checkpoint_boundary",
        "agent_state", "episode_sampler_rng_states", "normalizer",
        "action_grid", "reward_scaling", "configuration_snapshot",
        "configuration_sha256", "reservoir_physics_sha256",
        "physical_configuration_sha256", "training_inflow_sha256",
        "source_training_inflow_sha256", "validation_inflow_sha256",
        "training_wall_time_seconds", "validation_wall_time_seconds",
        "best_validation_score", "best_validation_step", "training_records",
        "validation_records", "rollout_buffer_saved",
    }
    missing = required - set(payload)
    if missing:
        raise CascadeSpecError(f"PPO checkpoint is missing: {sorted(missing)}")
    if payload["rollout_buffer_saved"] is not False:
        raise CascadeSpecError("PPO checkpoints must not contain partial rollouts.")
    if payload["checkpoint_boundary"] != "complete_rollout_and_ppo_update":
        raise CascadeSpecError("PPO checkpoint is not at an update boundary.")
    return payload


def _restore_latest(
    *,
    path: Path,
    settings: _PPOSettings,
    agent: DiscretePPOAgent,
    vector_env: SynchronousPPOEnvironmentSet,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    reward_scaling: Mapping[str, Any],
    configuration_hash: str,
    reservoir_physics_hash: str,
    physical_configuration_hash: str | None,
    training_inflow_hash: str,
    source_training_inflow_hash: str,
    validation_inflow_hash: str | None,
) -> tuple[
    int, int, int, float, float, float | None, int | None,
    list[dict[str, Any]], list[dict[str, Any]],
]:
    checkpoint = load_ppo_checkpoint(path, map_location=agent.device)
    expected = {
        "checkpoint_role": "latest",
        "mode": settings.mode,
        "experiment_id": settings.experiment_id,
        "seed": settings.seed,
        "total_environment_steps_budget": settings.total_environment_steps,
        "rollout_size": settings.rollout_size,
        "num_envs": settings.num_envs,
        "n_steps": settings.n_steps,
        "configuration_sha256": configuration_hash,
        "reservoir_physics_sha256": reservoir_physics_hash,
        "physical_configuration_sha256": physical_configuration_hash,
        "training_inflow_sha256": training_inflow_hash,
        "source_training_inflow_sha256": source_training_inflow_hash,
        "validation_inflow_sha256": validation_inflow_hash,
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise CascadeSpecError(
                f"Latest PPO checkpoint {name} does not match this run."
            )
    if configuration_sha256(checkpoint["configuration_snapshot"]) != configuration_hash:
        raise CascadeSpecError("PPO checkpoint configuration snapshot hash is invalid.")
    saved_normalizer = StateNormalizer.from_state_dict(
        checkpoint["normalizer"],
        vector_env.reservoir_1,
        vector_env.reservoir_2,
    )
    saved_grid = JointActionGrid.from_state_dict(
        checkpoint["action_grid"],
        vector_env.reservoir_1,
        vector_env.reservoir_2,
    )
    if saved_normalizer.state_dict() != normalizer.state_dict():
        raise CascadeSpecError("PPO checkpoint normalizer is incompatible.")
    if saved_grid.state_dict() != action_grid.state_dict():
        raise CascadeSpecError("PPO checkpoint action grid is incompatible.")
    if checkpoint["reward_scaling"] != dict(reward_scaling):
        raise CascadeSpecError("PPO checkpoint reward scaling is incompatible.")

    completed_update = int(checkpoint["completed_update"])
    environment_steps = int(checkpoint["environment_steps"])
    if environment_steps != completed_update * settings.rollout_size:
        raise CascadeSpecError("PPO checkpoint is not at a complete update boundary.")
    if completed_update > settings.total_updates:
        raise CascadeSpecError("PPO checkpoint exceeds the configured budget.")
    training_records = [dict(item) for item in checkpoint["training_records"]]
    validation_records = [dict(item) for item in checkpoint["validation_records"]]
    if len(training_records) != completed_update:
        raise CascadeSpecError("PPO training history disagrees with update count.")
    agent.load_training_state(checkpoint["agent_state"], restore_rng=True)
    vector_env.load_sampler_rng_states(checkpoint["episode_sampler_rng_states"])
    return (
        completed_update,
        environment_steps,
        int(checkpoint["next_validation_threshold"]),
        float(checkpoint["training_wall_time_seconds"]),
        float(checkpoint["validation_wall_time_seconds"]),
        None
        if checkpoint.get("best_validation_score") is None
        else float(checkpoint["best_validation_score"]),
        None
        if checkpoint.get("best_validation_step") is None
        else int(checkpoint["best_validation_step"]),
        training_records,
        validation_records,
    )


def _rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_numpy = left["numpy"]
    right_numpy = right["numpy"]
    if (
        left_numpy[0] != right_numpy[0]
        or not np.array_equal(left_numpy[1], right_numpy[1])
        or left_numpy[2:] != right_numpy[2:]
    ):
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = list(left.get("torch_cuda", []))
    right_cuda = list(right.get("torch_cuda", []))
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(a, b) for a, b in zip(left_cuda, right_cuda, strict=True)
    )


def train_ppo_seed(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    training_frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    output_root: str | Path | None = None,
    resume_latest: bool = False,
    physical_configuration_sha256: str | None = None,
    monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PPOTrainingResult:
    """Train one independently seeded PPO policy at complete update boundaries."""
    settings = _settings_from_config(config, mode=mode, seed=seed)
    validation_config = _mapping(config, "validation")
    learning_frame, validation_frame = split_training_inflows(
        training_frame,
        mode=mode,
        development_years=int(validation_config["development_years"]),
        validation_years=int(validation_config["validation_years"]),
    )
    configuration_snapshot = copy.deepcopy(dict(config))
    configuration_hash = configuration_sha256(configuration_snapshot)
    reservoir_physics_hash = reservoir_pair_sha256(reservoir_1, reservoir_2)
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
        reservoir_1,
        reservoir_2,
        step_m=settings.action_level_step_m,
    )
    _validate_formal_action_grid(action_grid)
    agent = _new_agent(settings, action_grid)
    sampler_rngs = make_episode_sampler_rngs(settings.seed, settings.num_envs)
    vector_env = SynchronousPPOEnvironmentSet(
        reservoir_1,
        reservoir_2,
        learning_frame,
        normalizer,
        action_grid,
        sampler_rngs,
        episode_months=settings.episode_months,
        initial_levels_m=settings.initial_levels_m,
        reward_reference_days=settings.reward_reference_days,
        reward_clip=settings.reward_clip,
        firm_power_failure_penalty_gwh=(
            settings.firm_power_failure_penalty_gwh
        ),
        monthly_max_target_levels_m=monthly_max_target_levels_m,
    )
    reward_scaling = _reward_metadata(
        reservoir_1,
        reservoir_2,
        learning_frame,
        normalizer,
        action_grid,
        settings,
        monthly_max_target_levels_m,
    )
    configured_output = _mapping(config, "outputs").get("root")
    selected_output = output_root if output_root is not None else configured_output
    if selected_output is None or not str(selected_output).strip():
        raise CascadeSpecError("outputs.root must be a nonempty path.")
    root = Path(selected_output)
    paths = build_ppo_training_paths(
        root,
        mode=settings.mode,
        seed=settings.seed,
        experiment_id=settings.experiment_id,
    )

    completed_update = 0
    environment_steps = 0
    next_validation_threshold = settings.validation_interval_environment_steps
    training_wall_time_offset = 0.0
    validation_wall_time_seconds = 0.0
    best_validation_score: float | None = None
    best_validation_step: int | None = None
    training_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    if resume_latest:
        (
            completed_update,
            environment_steps,
            next_validation_threshold,
            training_wall_time_offset,
            validation_wall_time_seconds,
            best_validation_score,
            best_validation_step,
            training_records,
            validation_records,
        ) = _restore_latest(
            path=paths.latest_checkpoint,
            settings=settings,
            agent=agent,
            vector_env=vector_env,
            normalizer=normalizer,
            action_grid=action_grid,
            reward_scaling=reward_scaling,
            configuration_hash=configuration_hash,
            reservoir_physics_hash=reservoir_physics_hash,
            physical_configuration_hash=physical_configuration_sha256,
            training_inflow_hash=training_hash,
            source_training_inflow_hash=source_training_hash,
            validation_inflow_hash=validation_hash,
        )
        _atomic_write_history(
            training_records,
            PPO_TRAINING_HISTORY_COLUMNS,
            paths.training_history_csv,
        )
        if paths.validation_history_csv is not None:
            _atomic_write_history(
                validation_records,
                PPO_VALIDATION_HISTORY_COLUMNS,
                paths.validation_history_csv,
            )

    training_timer_started = time.perf_counter()

    def current_training_wall_time() -> float:
        return training_wall_time_offset + (
            time.perf_counter() - training_timer_started
        )

    def save_checkpoint(path: Path, role: str) -> None:
        payload = _checkpoint_payload(
            role=role,
            settings=settings,
            agent=agent,
            vector_env=vector_env,
            normalizer=normalizer,
            action_grid=action_grid,
            reward_scaling=reward_scaling,
            configuration_snapshot=configuration_snapshot,
            configuration_hash=configuration_hash,
            reservoir_physics_hash=reservoir_physics_hash,
            physical_configuration_hash=physical_configuration_sha256,
            training_inflow_hash=training_hash,
            source_training_inflow_hash=source_training_hash,
            validation_inflow_hash=validation_hash,
            completed_update=completed_update,
            environment_steps=environment_steps,
            next_validation_threshold=next_validation_threshold,
            training_wall_time_seconds=current_training_wall_time(),
            validation_wall_time_seconds=validation_wall_time_seconds,
            best_validation_score=best_validation_score,
            best_validation_step=best_validation_step,
            training_records=training_records,
            validation_records=validation_records,
        )
        _atomic_torch_save(payload, path)

    for update_number in range(completed_update + 1, settings.total_updates + 1):
        rollout_started = time.perf_counter()
        states = vector_env.reset_rollout()
        rollout_states = np.zeros(
            (settings.n_steps, settings.num_envs, len(STATE_ORDER)),
            dtype=np.float32,
        )
        rollout_next_states = np.zeros_like(rollout_states)
        rollout_actions = np.zeros(
            (settings.n_steps, settings.num_envs), dtype=np.int64
        )
        rollout_log_probs = np.zeros(
            (settings.n_steps, settings.num_envs), dtype=np.float32
        )
        rollout_rewards = np.zeros_like(rollout_log_probs)
        rollout_values = np.zeros_like(rollout_log_probs)
        rollout_terminated = np.zeros_like(rollout_log_probs, dtype=np.bool_)
        rollout_truncated = np.zeros_like(rollout_log_probs, dtype=np.bool_)
        rollout_bootstrap_allowed = np.zeros_like(
            rollout_log_probs, dtype=np.bool_
        )
        rollout_entropies = np.zeros_like(rollout_log_probs)
        rollout_max_probabilities = np.zeros_like(rollout_log_probs)
        rollout_action_masks = np.zeros(
            (
                settings.n_steps,
                settings.num_envs,
                action_grid.action_count,
            ),
            dtype=np.bool_,
        )

        for step in range(settings.n_steps):
            action_masks = vector_env.get_feasible_action_masks()
            decision = agent.act(
                states,
                deterministic=False,
                action_masks=action_masks,
            )
            transition = vector_env.step(decision["actions"])
            rollout_states[step] = states
            rollout_next_states[step] = transition.next_states
            rollout_actions[step] = decision["actions"]
            rollout_log_probs[step] = decision["log_probs"]
            rollout_rewards[step] = transition.rewards
            rollout_values[step] = decision["values"]
            rollout_terminated[step] = transition.terminated
            rollout_truncated[step] = transition.truncated
            rollout_bootstrap_allowed[step] = transition.bootstrap_allowed
            rollout_entropies[step] = decision["raw_entropies"]
            rollout_max_probabilities[step] = decision[
                "max_action_probabilities"
            ]
            rollout_action_masks[step] = action_masks
            states = transition.next_states

        if np.any(rollout_terminated):
            raise CascadeSpecError("A formal PPO training rollout terminated physically.")
        if np.any(rollout_truncated[:-1]) or not np.all(rollout_truncated[-1]):
            raise CascadeSpecError(
                "PPO truncation must occur for all environments only at step 240."
            )
        if not np.all(rollout_bootstrap_allowed):
            raise CascadeSpecError("Every formal PPO training step must allow bootstrap.")
        rollout_next_values = agent.predict_values(
            rollout_next_states.reshape(-1, len(STATE_ORDER))
        ).reshape(settings.n_steps, settings.num_envs)
        rollout = PPORolloutBatch(
            states=rollout_states,
            actions=rollout_actions,
            old_log_probs=rollout_log_probs,
            rewards=rollout_rewards,
            values=rollout_values,
            next_values=rollout_next_values,
            terminated=rollout_terminated,
            truncated=rollout_truncated,
            bootstrap_allowed=rollout_bootstrap_allowed,
            action_masks=rollout_action_masks,
        )
        rollout_wall_time = time.perf_counter() - rollout_started
        optimization_started = time.perf_counter()
        optimization_metrics = agent.update(rollout)
        optimization_wall_time = time.perf_counter() - optimization_started
        diagnostics = vector_env.rollout_diagnostics()
        environment_steps = update_number * settings.rollout_size
        completed_update = update_number

        should_validate = bool(
            settings.mode == "development"
            and (
                environment_steps >= next_validation_threshold
                or environment_steps == settings.total_environment_steps
            )
        )
        validation_record: dict[str, Any] | None = None
        selected_best = False
        validation_score: float | None = None
        if should_validate:
            if validation_frame is None:
                raise CascadeSpecError("Development PPO has no validation frame.")
            rng_before = agent.rng_state()
            sampler_before = vector_env.sampler_rng_states()
            _, validation_summary = evaluate_deterministic_trajectory(
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
            if not _rng_states_equal(rng_before, agent.rng_state()):
                raise CascadeSpecError("PPO validation modified training RNG state.")
            if json.dumps(sampler_before, sort_keys=True) != json.dumps(
                vector_env.sampler_rng_states(), sort_keys=True
            ):
                raise CascadeSpecError("PPO validation modified episode sampler RNGs.")
            elapsed = float(validation_summary["validation_wall_time_seconds"])
            validation_wall_time_seconds += elapsed
            validation_score = float(
                validation_summary["mean_annual_penalized_objective_gwh"]
            )
            if not np.isfinite(validation_score):
                raise CascadeSpecError("PPO validation score must be finite.")
            selected_best = bool(
                best_validation_score is None
                or validation_score > best_validation_score
            )
            if selected_best:
                best_validation_score = validation_score
                best_validation_step = environment_steps
            validation_record = _validation_record(
                update_number=update_number,
                environment_steps=environment_steps,
                summary=validation_summary,
                selected=selected_best,
            )
            validation_records.append(validation_record)
            while next_validation_threshold <= environment_steps:
                next_validation_threshold += (
                    settings.validation_interval_environment_steps
                )

        counts = diagnostics["target_unreached_count"]
        rates = diagnostics["target_unreached_rate"]
        errors = diagnostics["target_absolute_error_m"]
        checkpoint_events: list[str] = []
        should_save_latest = bool(
            update_number % settings.checkpoint_interval_updates == 0
            or should_validate
            or update_number == settings.total_updates
        )
        if should_save_latest:
            checkpoint_events.append("latest")
        if selected_best:
            checkpoint_events.append("best_validation")
        if settings.mode == "final" and update_number == settings.total_updates:
            checkpoint_events.append("final")

        record: dict[str, Any] = {
            "update_number": update_number,
            "environment_steps": environment_steps,
            "completed_episode_windows": update_number * settings.num_envs,
            "rollout_size": settings.rollout_size,
            "minibatch_updates": int(optimization_metrics["minibatch_updates"]),
            "episode_start_indices_json": json.dumps(
                diagnostics["episode_start_indices"], separators=(",", ":")
            ),
            "episode_start_dates_json": json.dumps(
                diagnostics["episode_start_dates"], separators=(",", ":")
            ),
            "episode_end_dates_json": json.dumps(
                diagnostics["episode_end_dates"], separators=(",", ":")
            ),
            "episode_bootstrap_dates_json": json.dumps(
                diagnostics["episode_bootstrap_dates"], separators=(",", ":")
            ),
            "raw_episode_generation_values_json": json.dumps(
                diagnostics["raw_episode_cascade_generation_gwh"],
                separators=(",", ":"),
            ),
            "raw_episode_penalized_objective_values_json": json.dumps(
                diagnostics["raw_episode_penalized_objective_gwh"],
                separators=(",", ":"),
            ),
            "episode_failed_station_months_values_json": json.dumps(
                diagnostics["episode_failed_station_months"],
                separators=(",", ":"),
            ),
            "episode_failure_penalty_values_gwh_json": json.dumps(
                diagnostics["episode_failure_penalty_gwh"],
                separators=(",", ":"),
            ),
            "normalized_episode_reward_values_json": json.dumps(
                diagnostics["normalized_episode_rewards"], separators=(",", ":")
            ),
            "raw_generation_mean_gwh": diagnostics["raw_generation_mean_gwh"],
            "raw_generation_std_gwh": diagnostics["raw_generation_std_gwh"],
            "raw_penalized_objective_mean_gwh": diagnostics[
                "raw_penalized_objective_mean_gwh"
            ],
            "raw_penalized_objective_std_gwh": diagnostics[
                "raw_penalized_objective_std_gwh"
            ],
            "failed_station_months_mean": diagnostics[
                "failed_station_months_mean"
            ],
            "failure_penalty_mean_gwh": diagnostics[
                "failure_penalty_mean_gwh"
            ],
            "normalized_reward_mean": diagnostics["normalized_reward_mean"],
            "normalized_reward_std": diagnostics["normalized_reward_std"],
            "rollout_raw_entropy": float(rollout_entropies.mean()),
            "rollout_normalized_entropy": float(
                rollout_entropies.mean() / np.log(action_grid.action_count)
            ),
            "rollout_effective_action_count": float(
                np.exp(rollout_entropies).mean()
            ),
            "rollout_max_action_probability": float(
                rollout_max_probabilities.mean()
            ),
            "terminated_count": int(rollout_terminated.sum()),
            "truncated_count": int(rollout_truncated.sum()),
            "bootstrap_allowed_count": int(rollout_bootstrap_allowed.sum()),
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
            "xiaowan_target_abs_error_mean_m": errors["xiaowan"]["mean"],
            "xiaowan_target_abs_error_max_m": errors["xiaowan"]["max"],
            "xiaowan_target_abs_error_p95_m": errors["xiaowan"]["p95"],
            "nuozhadu_target_abs_error_mean_m": errors["nuozhadu"]["mean"],
            "nuozhadu_target_abs_error_max_m": errors["nuozhadu"]["max"],
            "nuozhadu_target_abs_error_p95_m": errors["nuozhadu"]["p95"],
            "rollout_wall_time_seconds": rollout_wall_time,
            "optimization_wall_time_seconds": optimization_wall_time,
            "cumulative_training_wall_time_seconds": current_training_wall_time(),
            "cumulative_validation_wall_time_seconds": validation_wall_time_seconds,
            "samples_per_second": settings.rollout_size
            / max(rollout_wall_time, 1.0e-12),
            "validation_performed": should_validate,
            "validation_score": validation_score,
            "best_validation_score": best_validation_score,
            "best_validation_step": best_validation_step,
            "checkpoint_event": ",".join(checkpoint_events),
        }
        record.update(optimization_metrics)
        training_records.append(record)

        if selected_best:
            if paths.best_validation_checkpoint is None:
                raise CascadeSpecError("Development PPO has no best checkpoint path.")
            save_checkpoint(paths.best_validation_checkpoint, "best_validation")
        if should_save_latest:
            save_checkpoint(paths.latest_checkpoint, "latest")

        should_write_history = bool(
            update_number % settings.logging_interval_updates == 0
            or should_validate
            or should_save_latest
            or update_number == settings.total_updates
        )
        if should_write_history:
            _atomic_write_history(
                training_records,
                PPO_TRAINING_HISTORY_COLUMNS,
                paths.training_history_csv,
            )
            if paths.validation_history_csv is not None:
                _atomic_write_history(
                    validation_records,
                    PPO_VALIDATION_HISTORY_COLUMNS,
                    paths.validation_history_csv,
                )
        if progress_callback is not None and should_write_history:
            progress_callback(
                {
                    "event": "ppo_training_progress",
                    "mode": settings.mode,
                    "experiment_id": settings.experiment_id,
                    "seed": settings.seed,
                    "training": dict(record),
                    "validation": (
                        None
                        if validation_record is None
                        else dict(validation_record)
                    ),
                    "paths": paths.as_dict(),
                }
            )

    if environment_steps != settings.total_environment_steps:
        raise CascadeSpecError("PPO training did not consume its exact step budget.")
    if completed_update != settings.total_updates:
        raise CascadeSpecError("PPO training did not finish all configured updates.")

    if settings.mode == "final":
        if paths.final_checkpoint is None:
            raise CascadeSpecError("Final PPO has no final checkpoint path.")
        save_checkpoint(paths.final_checkpoint, "final")
        selected_checkpoint = paths.final_checkpoint
    else:
        if (
            paths.best_validation_checkpoint is None
            or best_validation_score is None
            or best_validation_step is None
            or not paths.best_validation_checkpoint.is_file()
        ):
            raise CascadeSpecError("Development PPO has no selected best checkpoint.")
        selected_checkpoint = paths.best_validation_checkpoint

    return PPOTrainingResult(
        mode=settings.mode,
        experiment_id=settings.experiment_id,
        seed=settings.seed,
        completed_updates=completed_update,
        environment_steps=environment_steps,
        completed_episode_windows=completed_update * settings.num_envs,
        training_wall_time_seconds=current_training_wall_time(),
        validation_wall_time_seconds=validation_wall_time_seconds,
        best_validation_score=best_validation_score,
        best_validation_step=best_validation_step,
        configuration_sha256=configuration_hash,
        normalization=normalizer.state_dict(),
        action_grid=action_grid.state_dict(),
        reward_scaling=dict(reward_scaling),
        paths=paths,
        selected_checkpoint=selected_checkpoint,
    )


__all__ = [
    "DEFAULT_TOTAL_ENVIRONMENT_STEPS",
    "ENTROPY_NORMALIZATION",
    "EPISODE_MONTHS",
    "N_STEPS",
    "NUM_ENVS",
    "PPO_ACTION_COUNT",
    "PPO_ACTION_GRID_SHAPE",
    "PPOTrainingPaths",
    "PPOTrainingResult",
    "PPO_CHECKPOINT_SCHEMA_VERSION",
    "PPO_REWARD_SCALE_GWH",
    "PPO_TRAINING_HISTORY_COLUMNS",
    "PPO_TRAINING_MODES",
    "PPO_VALIDATION_HISTORY_COLUMNS",
    "ROLLOUT_SIZE",
    "build_ppo_training_paths",
    "inspect_ppo_configuration",
    "load_ppo_checkpoint",
    "train_ppo_seed",
]
