"""Reproducible training for continuous-target autoregressive PPO."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .continuous_ppo_adapter import (
    ACTION_GEOMETRY_DIMENSION,
    CONTINUOUS_ACTION_TRANSFORM,
    ContinuousCascadePPOAdapter,
    ContinuousSynchronousPPOEnvironmentSet,
    NUOZHADU_GEOMETRY_DIMENSION,
    XIAOWAN_GEOMETRY_DIMENSION,
    make_continuous_episode_sampler_rngs,
)
from .continuous_ppo_agent import (
    CONTINUOUS_ACTION_DIMENSION,
    CONTINUOUS_ACTION_MODE,
    ContinuousPPOAgent,
    ContinuousPPORolloutBatch,
)
from .continuous_ppo_evaluation import evaluate_continuous_ppo_trajectory
from .dqn_adapter import INFLOW_COLUMNS, STATE_ORDER, StateNormalizer, normalized_inflow_sha256
from .dqn_training import configuration_sha256, experiment_id_from_config, split_training_inflows
from .environment import CascadeReservoirEnv, CascadeSpecError, ReservoirSpec, _as_month_period_index
from .sdp import reservoir_pair_sha256


CONTINUOUS_PPO_CHECKPOINT_SCHEMA_VERSION = 1
CONTINUOUS_PPO_TRAINING_MODES = ("development", "final")
DEVELOPMENT_YEARS = 320
VALIDATION_YEARS = 80
EPISODE_MONTHS = 240
DEFAULT_NUM_ENVS = 8
DEFAULT_N_STEPS = 240
CONTINUOUS_PPO_REWARD_SCALE_GWH = 7332.48

ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ContinuousPPOTrainingPaths:
    output_root: Path
    run_root: Path
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
class ContinuousPPOTrainingResult:
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
    reward_scaling: dict[str, Any]
    paths: ContinuousPPOTrainingPaths
    selected_checkpoint: Path

    def as_dict(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.__dict__)
        payload["paths"] = self.paths.as_dict()
        payload["selected_checkpoint"] = str(self.selected_checkpoint)
        return payload


@dataclass(frozen=True)
class _Settings:
    mode: str
    experiment_id: str
    seed: int
    initial_levels_m: tuple[float, float]
    firm_power_failure_penalty_gwh: float
    reward_reference_days: float
    reward_clip: tuple[float, float] | None
    actor_hidden_dimensions: tuple[int, ...]
    critic_hidden_dimensions: tuple[int, ...]
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coefficient: float
    value_loss_coefficient: float
    max_gradient_norm: float
    minibatch_size: int
    update_epochs: int
    target_kl: float
    advantage_normalization: bool
    adam_epsilon: float
    lr_schedule: str
    final_learning_rate_fraction: float
    initial_log_std: float
    min_log_std: float
    max_log_std: float
    squash_epsilon: float
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
        raise CascadeSpecError(f"Configuration section {key!r} must be a mapping.")
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


def _hidden_dimensions(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CascadeSpecError(f"{name} must be a sequence.")
    result = tuple(_positive_int(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not result:
        raise CascadeSpecError(f"{name} cannot be empty.")
    return result


def _resolve_device(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadeSpecError("training.device must be a nonempty string.")
    requested = value.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CascadeSpecError("CUDA requested but unavailable.")
    return str(device)


def _parse_reward_clip(reward: Mapping[str, Any]) -> tuple[float, float] | None:
    clip = _mapping(reward, "clip")
    if clip.get("enabled") is False:
        return None
    if clip.get("enabled") is not True:
        raise CascadeSpecError("reward.clip.enabled must be boolean.")
    lower = _finite_float(clip.get("lower"), "reward.clip.lower")
    upper = _finite_float(clip.get("upper"), "reward.clip.upper")
    if upper <= lower:
        raise CascadeSpecError("reward clip upper must exceed lower.")
    return lower, upper


def _settings_from_config(
    config: Mapping[str, Any], *, mode: str, seed: int
) -> _Settings:
    if mode not in CONTINUOUS_PPO_TRAINING_MODES:
        raise CascadeSpecError(f"mode must be one of {CONTINUOUS_PPO_TRAINING_MODES}.")
    experiment_id = experiment_id_from_config(config)
    if experiment_id is None:
        raise CascadeSpecError("Continuous PPO requires experiment_id.")
    problem = _mapping(config, "problem")
    information = _mapping(config, "information")
    reward = _mapping(config, "reward")
    normalization = _mapping(config, "normalization")
    validation = _mapping(config, "validation")
    ppo = _mapping(config, "ppo")
    network = _mapping(config, "network")
    training = _mapping(config, "training")

    if list(information.get("state_order", [])) != list(STATE_ORDER):
        raise CascadeSpecError("Continuous PPO state order must match fixed five-state order.")
    if information.get("one_step_perfect_forecast") is not True:
        raise CascadeSpecError("Continuous PPO requires one-step perfect current inflows.")
    if information.get("geometry_features_are_current_state_derived") is not True:
        raise CascadeSpecError(
            "V16 requires current-state-derived feasible geometry features."
        )
    if reward.get("objective") != "generation_gwh_minus_penalty_gwh_per_failed_station_month":
        raise CascadeSpecError("Continuous PPO reward objective differs.")
    penalty = _finite_float(
        reward.get("firm_power_failure_penalty_gwh"),
        "reward.firm_power_failure_penalty_gwh",
    )
    if penalty < 0.0:
        raise CascadeSpecError("failure penalty must be nonnegative.")
    if _finite_float(reward.get("scale_gwh"), "reward.scale_gwh") != CONTINUOUS_PPO_REWARD_SCALE_GWH:
        raise CascadeSpecError("Continuous PPO reward scale must remain 7332.48 GWh.")
    if normalization.get("method") != StateNormalizer.method or normalization.get("clip") is not True:
        raise CascadeSpecError("Continuous PPO requires clipped componentwise normalization.")
    if int(validation.get("development_years")) != DEVELOPMENT_YEARS or int(
        validation.get("validation_years")
    ) != VALIDATION_YEARS:
        raise CascadeSpecError("Continuous PPO requires the 320/80 development split.")
    if validation.get("score") != "mean_annual_penalized_objective_gwh":
        raise CascadeSpecError("Continuous PPO validation score differs.")

    initial = np.asarray(problem.get("initial_levels_m"), dtype=float)
    if initial.shape != (2,) or np.any(~np.isfinite(initial)):
        raise CascadeSpecError("problem.initial_levels_m must contain two values.")
    if network.get("action_distribution") != CONTINUOUS_ACTION_MODE:
        raise CascadeSpecError("Continuous PPO action distribution differs.")
    if network.get("action_transform") != CONTINUOUS_ACTION_TRANSFORM:
        raise CascadeSpecError("Continuous PPO action transform differs.")
    if list(network.get("xiaowan_geometry_features", [])) != [
        "lower_level", "upper_level", "interval_width"
    ]:
        raise CascadeSpecError("V16 XiaoWan geometry feature contract differs.")
    if list(network.get("nuozhadu_geometry_features", [])) != [
        "xiaowan_target", "lower_level", "upper_level", "interval_width"
    ]:
        raise CascadeSpecError("V16 NuoZhaDu geometry feature contract differs.")
    actor_hidden = _hidden_dimensions(
        network.get("actor_hidden_dimensions"), "network.actor_hidden_dimensions"
    )
    critic_hidden = _hidden_dimensions(
        network.get("critic_hidden_dimensions"), "network.critic_hidden_dimensions"
    )
    episode_months = _positive_int(training.get("episode_months"), "training.episode_months")
    n_steps = _positive_int(training.get("n_steps"), "training.n_steps")
    num_envs = _positive_int(training.get("num_envs"), "training.num_envs")
    if episode_months != EPISODE_MONTHS or n_steps != EPISODE_MONTHS:
        raise CascadeSpecError("Continuous PPO requires 240-month windows and rollouts.")
    total_steps = _positive_int(
        training.get("total_environment_steps"), "training.total_environment_steps"
    )
    rollout_size = n_steps * num_envs
    if total_steps % rollout_size != 0:
        raise CascadeSpecError("total_environment_steps must divide by rollout size.")
    minibatch = _positive_int(ppo.get("minibatch_size"), "ppo.minibatch_size")
    if rollout_size % minibatch != 0:
        raise CascadeSpecError("rollout size must divide into minibatches.")
    lr_schedule = str(ppo.get("lr_schedule"))
    if lr_schedule != "linear_decay":
        raise CascadeSpecError(
            "Stabilized continuous PPO requires lr_schedule='linear_decay'."
        )
    final_lr_fraction = _finite_float(
        ppo.get("final_learning_rate_fraction"),
        "ppo.final_learning_rate_fraction",
    )
    if not 0.0 <= final_lr_fraction < 1.0:
        raise CascadeSpecError(
            "final_learning_rate_fraction must lie in [0, 1)."
        )
    return _Settings(
        mode=mode,
        experiment_id=experiment_id,
        seed=int(seed),
        initial_levels_m=(float(initial[0]), float(initial[1])),
        firm_power_failure_penalty_gwh=penalty,
        reward_reference_days=_finite_float(
            reward.get("reward_reference_days"), "reward.reward_reference_days"
        ),
        reward_clip=_parse_reward_clip(reward),
        actor_hidden_dimensions=actor_hidden,
        critic_hidden_dimensions=critic_hidden,
        learning_rate=_finite_float(ppo.get("learning_rate"), "ppo.learning_rate"),
        gamma=_finite_float(ppo.get("gamma"), "ppo.gamma"),
        gae_lambda=_finite_float(ppo.get("gae_lambda"), "ppo.gae_lambda"),
        clip_range=_finite_float(ppo.get("clip_range"), "ppo.clip_range"),
        entropy_coefficient=_finite_float(
            ppo.get("entropy_coefficient"), "ppo.entropy_coefficient"
        ),
        value_loss_coefficient=_finite_float(
            ppo.get("value_loss_coefficient"), "ppo.value_loss_coefficient"
        ),
        max_gradient_norm=_finite_float(
            ppo.get("max_gradient_norm"), "ppo.max_gradient_norm"
        ),
        minibatch_size=minibatch,
        update_epochs=_positive_int(ppo.get("update_epochs"), "ppo.update_epochs"),
        target_kl=_finite_float(ppo.get("target_kl"), "ppo.target_kl"),
        advantage_normalization=bool(ppo.get("advantage_normalization")),
        adam_epsilon=_finite_float(ppo.get("adam_epsilon"), "ppo.adam_epsilon"),
        lr_schedule=lr_schedule,
        final_learning_rate_fraction=final_lr_fraction,
        initial_log_std=_finite_float(ppo.get("initial_log_std"), "ppo.initial_log_std"),
        min_log_std=_finite_float(ppo.get("min_log_std"), "ppo.min_log_std"),
        max_log_std=_finite_float(ppo.get("max_log_std"), "ppo.max_log_std"),
        squash_epsilon=_finite_float(ppo.get("squash_epsilon"), "ppo.squash_epsilon"),
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


def build_continuous_ppo_training_paths(
    output_root: str | Path, *, mode: str, seed: int, experiment_id: str
) -> ContinuousPPOTrainingPaths:
    root = Path(output_root)
    run_root = root / "experiments" / experiment_id / f"seed_{int(seed)}" / mode
    checkpoint_dir = run_root / "checkpoints"
    return ContinuousPPOTrainingPaths(
        output_root=root,
        run_root=run_root,
        configuration_snapshot_json=run_root / "config_snapshot.json",
        metadata_json=run_root / "metadata.json",
        training_history_csv=run_root / "training" / "history.csv",
        validation_history_csv=(
            run_root / "validation" / "history.csv" if mode == "development" else None
        ),
        latest_checkpoint=checkpoint_dir / "latest.pt",
        best_validation_checkpoint=(
            checkpoint_dir / "best_validation.pt" if mode == "development" else None
        ),
        final_checkpoint=checkpoint_dir / "final.pt" if mode == "final" else None,
    )


def inspect_continuous_ppo_configuration(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    settings = _settings_from_config(config, mode=mode, seed=seed)
    return {
        "experiment_id": settings.experiment_id,
        "mode": settings.mode,
        "seed": settings.seed,
        "state_order": list(STATE_ORDER),
        "state_dimension": len(STATE_ORDER),
        "continuous_action_dimension": CONTINUOUS_ACTION_DIMENSION,
        "action_geometry_dimension": ACTION_GEOMETRY_DIMENSION,
        "xiaowan_geometry_dimension": XIAOWAN_GEOMETRY_DIMENSION,
        "nuozhadu_geometry_dimension": NUOZHADU_GEOMETRY_DIMENSION,
        "action_distribution": CONTINUOUS_ACTION_MODE,
        "action_transform": CONTINUOUS_ACTION_TRANSFORM,
        "actor_hidden_dimensions": list(settings.actor_hidden_dimensions),
        "critic_hidden_dimensions": list(settings.critic_hidden_dimensions),
        "initial_log_std": settings.initial_log_std,
        "min_log_std": settings.min_log_std,
        "max_log_std": settings.max_log_std,
        "target_kl": settings.target_kl,
        "lr_schedule": settings.lr_schedule,
        "final_learning_rate_fraction": settings.final_learning_rate_fraction,
        "num_envs": settings.num_envs,
        "n_steps": settings.n_steps,
        "rollout_batch_size": settings.rollout_size,
        "minibatch_size": settings.minibatch_size,
        "total_environment_steps": settings.total_environment_steps,
        "total_updates": settings.total_updates,
        "completed_episode_windows": settings.total_updates * settings.num_envs,
        "firm_power_failure_penalty_gwh": settings.firm_power_failure_penalty_gwh,
        "reward_scale_gwh": CONTINUOUS_PPO_REWARD_SCALE_GWH,
        "device": settings.device,
        "xiaowan_operating_bounds_m": [
            reservoir_1.dead_level_m,
            reservoir_1.normal_level_m,
        ],
        "nuozhadu_operating_bounds_m": [
            reservoir_2.dead_level_m,
            reservoir_2.normal_level_m,
        ],
    }


def _atomic_write_history(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
        pd.DataFrame(records).to_csv(
            temporary, index=False, float_format="%.17g", lineterminator="\n"
        )
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_continuous_ppo_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CONTINUOUS_PPO_CHECKPOINT_SCHEMA_VERSION:
        raise CascadeSpecError("Unsupported continuous PPO checkpoint.")
    return payload


def _new_agent(settings: _Settings) -> ContinuousPPOAgent:
    return ContinuousPPOAgent(
        state_dimension=len(STATE_ORDER),
        actor_hidden_dimensions=settings.actor_hidden_dimensions,
        critic_hidden_dimensions=settings.critic_hidden_dimensions,
        learning_rate=settings.learning_rate,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
        clip_range=settings.clip_range,
        entropy_coefficient=settings.entropy_coefficient,
        value_loss_coefficient=settings.value_loss_coefficient,
        max_gradient_norm=settings.max_gradient_norm,
        minibatch_size=settings.minibatch_size,
        update_epochs=settings.update_epochs,
        target_kl=settings.target_kl,
        advantage_normalization=settings.advantage_normalization,
        adam_epsilon=settings.adam_epsilon,
        initial_log_std=settings.initial_log_std,
        min_log_std=settings.min_log_std,
        max_log_std=settings.max_log_std,
        squash_epsilon=settings.squash_epsilon,
        seed=settings.seed,
        device=settings.device,
    )


def _reward_metadata(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    learning_frame: pd.DataFrame,
    normalizer: StateNormalizer,
    settings: _Settings,
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
        firm_power_failure_penalty_gwh=settings.firm_power_failure_penalty_gwh,
        monthly_max_target_levels_m=monthly_max_target_levels_m,
        require_energy_parameters=True,
    )
    adapter = ContinuousCascadePPOAdapter(
        env,
        normalizer,
        reward_reference_days=settings.reward_reference_days,
        reward_clip=settings.reward_clip,
        max_transitions=settings.episode_months,
    )
    return adapter.reward_metadata


def _rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["python"] == right["python"]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left.get("torch_cuda", [])) == len(right.get("torch_cuda", []))
        and all(
            torch.equal(a, b)
            for a, b in zip(left.get("torch_cuda", []), right.get("torch_cuda", []), strict=True)
        )
    )


def train_continuous_ppo_seed(
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
) -> ContinuousPPOTrainingResult:
    settings = _settings_from_config(config, mode=mode, seed=seed)
    validation_config = _mapping(config, "validation")
    learning_frame, validation_frame = split_training_inflows(
        training_frame,
        mode=mode,
        development_years=int(validation_config["development_years"]),
        validation_years=int(validation_config["validation_years"]),
    )
    snapshot = copy.deepcopy(dict(config))
    config_hash = configuration_sha256(snapshot)
    physics_hash = reservoir_pair_sha256(reservoir_1, reservoir_2)
    source_hash = normalized_inflow_sha256(training_frame)
    training_hash = normalized_inflow_sha256(learning_frame)
    validation_hash = (
        None if validation_frame is None else normalized_inflow_sha256(validation_frame)
    )
    normalizer = StateNormalizer.fit(
        learning_frame, reservoir_1, reservoir_2, clip=True
    )
    agent = _new_agent(settings)
    sampler_rngs = make_continuous_episode_sampler_rngs(
        settings.seed, settings.num_envs
    )
    vector_env = ContinuousSynchronousPPOEnvironmentSet(
        reservoir_1,
        reservoir_2,
        learning_frame,
        normalizer,
        sampler_rngs,
        episode_months=settings.episode_months,
        initial_levels_m=settings.initial_levels_m,
        reward_reference_days=settings.reward_reference_days,
        reward_clip=settings.reward_clip,
        firm_power_failure_penalty_gwh=settings.firm_power_failure_penalty_gwh,
        monthly_max_target_levels_m=monthly_max_target_levels_m,
    )
    reward_scaling = _reward_metadata(
        reservoir_1,
        reservoir_2,
        learning_frame,
        normalizer,
        settings,
        monthly_max_target_levels_m,
    )
    configured_root = _mapping(config, "outputs").get("root")
    selected_root = output_root if output_root is not None else configured_root
    if selected_root is None:
        raise CascadeSpecError("outputs.root is required.")
    paths = build_continuous_ppo_training_paths(
        selected_root,
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
        checkpoint = load_continuous_ppo_checkpoint(
            paths.latest_checkpoint, map_location=agent.device
        )
        expected = {
            "experiment_id": settings.experiment_id,
            "mode": settings.mode,
            "seed": settings.seed,
            "configuration_sha256": config_hash,
            "reservoir_physics_sha256": physics_hash,
            "physical_configuration_sha256": physical_configuration_sha256,
            "training_inflow_sha256": training_hash,
            "source_training_inflow_sha256": source_hash,
            "validation_inflow_sha256": validation_hash,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise CascadeSpecError(f"Continuous checkpoint {name} differs.")
        agent.load_training_state(checkpoint["agent_state"], restore_rng=True)
        vector_env.load_sampler_rng_states(checkpoint["episode_sampler_rng_states"])
        completed_update = int(checkpoint["completed_update"])
        environment_steps = int(checkpoint["environment_steps"])
        next_validation_threshold = int(checkpoint["next_validation_threshold"])
        training_wall_time_offset = float(checkpoint["training_wall_time_seconds"])
        validation_wall_time_seconds = float(checkpoint["validation_wall_time_seconds"])
        best_validation_score = checkpoint.get("best_validation_score")
        best_validation_step = checkpoint.get("best_validation_step")
        training_records = list(checkpoint["training_records"])
        validation_records = list(checkpoint["validation_records"])

    training_started = time.perf_counter()

    def current_wall_time() -> float:
        return training_wall_time_offset + time.perf_counter() - training_started

    def checkpoint_payload(role: str) -> dict[str, Any]:
        return {
            "schema_version": CONTINUOUS_PPO_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_role": role,
            "experiment_id": settings.experiment_id,
            "mode": settings.mode,
            "seed": settings.seed,
            "configuration_snapshot": snapshot,
            "configuration_sha256": config_hash,
            "reservoir_physics_sha256": physics_hash,
            "physical_configuration_sha256": physical_configuration_sha256,
            "training_inflow_sha256": training_hash,
            "source_training_inflow_sha256": source_hash,
            "validation_inflow_sha256": validation_hash,
            "normalizer": normalizer.state_dict(),
            "reward_scaling": reward_scaling,
            "agent_state": agent.training_state(),
            "episode_sampler_rng_states": vector_env.sampler_rng_states(),
            "completed_update": completed_update,
            "environment_steps": environment_steps,
            "next_validation_threshold": next_validation_threshold,
            "training_wall_time_seconds": current_wall_time(),
            "validation_wall_time_seconds": validation_wall_time_seconds,
            "best_validation_score": best_validation_score,
            "best_validation_step": best_validation_step,
            "training_records": copy.deepcopy(training_records),
            "validation_records": copy.deepcopy(validation_records),
        }

    for update_number in range(completed_update + 1, settings.total_updates + 1):
        if settings.total_updates <= 1:
            progress = 1.0
        else:
            progress = (update_number - 1) / (settings.total_updates - 1)
        learning_rate = settings.learning_rate * (
            1.0
            - progress * (1.0 - settings.final_learning_rate_fraction)
        )
        agent.set_learning_rate(learning_rate)
        rollout_started = time.perf_counter()
        states = vector_env.reset_rollout()
        shape = (settings.n_steps, settings.num_envs)
        rollout_states = np.zeros((*shape, len(STATE_ORDER)), dtype=np.float32)
        rollout_next_states = np.zeros_like(rollout_states)
        rollout_latent = np.zeros((*shape, 2), dtype=np.float32)
        rollout_log_probs = np.zeros(shape, dtype=np.float32)
        rollout_rewards = np.zeros(shape, dtype=np.float32)
        rollout_values = np.zeros(shape, dtype=np.float32)
        rollout_terminated = np.zeros(shape, dtype=np.bool_)
        rollout_truncated = np.zeros(shape, dtype=np.bool_)
        rollout_bootstrap = np.zeros(shape, dtype=np.bool_)
        rollout_active = np.zeros((*shape, 2), dtype=np.bool_)
        rollout_geometry = np.zeros(
            (*shape, ACTION_GEOMETRY_DIMENSION), dtype=np.float32
        )
        rollout_entropy = np.zeros(shape, dtype=np.float32)
        rollout_std = np.zeros((*shape, 2), dtype=np.float32)

        for step in range(settings.n_steps):
            xiaowan_active = vector_env.get_xiaowan_active_dimensions()
            xiaowan_geometry = vector_env.get_xiaowan_geometry_features()
            xiaowan_decision = agent.act_xiaowan(
                states,
                xiaowan_geometry,
                deterministic=False,
                xiaowan_active=xiaowan_active,
            )
            nuozhadu_geometry, nuozhadu_active = (
                vector_env.get_nuozhadu_geometry_features(
                    xiaowan_decision["fraction"]
                )
            )
            decision = agent.complete_action(
                states,
                xiaowan_decision,
                xiaowan_geometry,
                nuozhadu_geometry,
                deterministic=False,
                nuozhadu_active=nuozhadu_active,
            )
            transition = vector_env.step(decision["action_fractions"])
            if not np.array_equal(
                xiaowan_active,
                transition.active_action_dimensions[:, 0],
            ):
                raise CascadeSpecError(
                    "Continuous upstream action activity changed during mapping."
                )
            if not np.array_equal(
                nuozhadu_active,
                transition.active_action_dimensions[:, 1],
            ):
                raise CascadeSpecError(
                    "Continuous downstream action activity changed during mapping."
                )
            active_float = transition.active_action_dimensions.astype(np.float32)
            rollout_states[step] = states
            rollout_next_states[step] = transition.next_states
            rollout_latent[step] = decision["latent_actions"]
            rollout_log_probs[step] = (
                decision["component_log_probs"] * active_float
            ).sum(axis=1)
            rollout_rewards[step] = transition.rewards
            rollout_values[step] = decision["values"]
            rollout_terminated[step] = transition.terminated
            rollout_truncated[step] = transition.truncated
            rollout_bootstrap[step] = transition.bootstrap_allowed
            rollout_active[step] = transition.active_action_dimensions
            rollout_geometry[step] = decision["action_geometry"]
            rollout_entropy[step] = (
                -decision["component_log_probs"] * active_float
            ).sum(axis=1)
            rollout_std[step] = decision["action_std"]
            states = transition.next_states

        if np.any(rollout_terminated):
            raise CascadeSpecError("Continuous PPO rollout terminated physically.")
        if np.any(rollout_truncated[:-1]) or not np.all(rollout_truncated[-1]):
            raise CascadeSpecError("Continuous truncation must occur at step 240.")
        if not np.all(rollout_bootstrap):
            raise CascadeSpecError("Continuous nonterminal rollout must bootstrap.")
        rollout_next_values = agent.predict_values(
            rollout_next_states.reshape(-1, len(STATE_ORDER))
        ).reshape(shape)
        rollout = ContinuousPPORolloutBatch(
            states=rollout_states,
            latent_actions=rollout_latent,
            old_log_probs=rollout_log_probs,
            rewards=rollout_rewards,
            values=rollout_values,
            next_values=rollout_next_values,
            terminated=rollout_terminated,
            truncated=rollout_truncated,
            bootstrap_allowed=rollout_bootstrap,
            active_action_dimensions=rollout_active,
            action_geometry=rollout_geometry,
        )
        rollout_wall_time = time.perf_counter() - rollout_started
        optimization_started = time.perf_counter()
        optimization = agent.update(rollout)
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
                raise CascadeSpecError("Continuous development run lacks validation data.")
            rng_before = agent.rng_state()
            sampler_before = vector_env.sampler_rng_states()
            _, summary = evaluate_continuous_ppo_trajectory(
                reservoir_1,
                reservoir_2,
                agent,
                normalizer,
                validation_frame,
                settings.initial_levels_m,
                training_inflow_sha256=training_hash,
                seed=settings.seed,
                reward_reference_days=settings.reward_reference_days,
                reward_clip=settings.reward_clip,
                firm_power_failure_penalty_gwh=settings.firm_power_failure_penalty_gwh,
                monthly_max_target_levels_m=monthly_max_target_levels_m,
                trajectory_kind="validation",
                collect_rows=False,
            )
            if not _rng_states_equal(rng_before, agent.rng_state()):
                raise CascadeSpecError("Continuous validation modified agent RNG state.")
            if json.dumps(sampler_before, sort_keys=True) != json.dumps(
                vector_env.sampler_rng_states(), sort_keys=True
            ):
                raise CascadeSpecError("Continuous validation modified sampler RNGs.")
            elapsed = float(summary["validation_wall_time_seconds"])
            validation_wall_time_seconds += elapsed
            validation_score = float(summary["mean_annual_penalized_objective_gwh"])
            selected_best = bool(
                best_validation_score is None or validation_score > best_validation_score
            )
            if selected_best:
                best_validation_score = validation_score
                best_validation_step = environment_steps
            validation_record = {
                "update_number": update_number,
                "environment_steps": environment_steps,
                "validation_wall_time_seconds": elapsed,
                "validation_months": int(summary["months"]),
                "validation_mean_annual_cascade_generation_gwh": float(
                    summary["mean_annual_cascade_generation_gwh"]
                ),
                "validation_mean_annual_penalized_objective_gwh": validation_score,
                "validation_total_failed_station_months": int(summary["failed_station_months"]),
                "xiaowan_firm_power_reliability": float(
                    summary["firm_power_reliability"]["xiaowan"]
                ),
                "nuozhadu_firm_power_reliability": float(
                    summary["firm_power_reliability"]["nuozhadu"]
                ),
                "cascade_firm_power_reliability": float(
                    summary["firm_power_reliability"]["cascade"]
                ),
                "score": validation_score,
                "selected_best_validation_checkpoint": selected_best,
            }
            validation_records.append(validation_record)
            while next_validation_threshold <= environment_steps:
                next_validation_threshold += settings.validation_interval_environment_steps

        checkpoint_events: list[str] = []
        save_latest = bool(
            update_number % settings.checkpoint_interval_updates == 0
            or should_validate
            or update_number == settings.total_updates
        )
        if save_latest:
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
            "episode_start_indices_json": json.dumps(diagnostics["episode_start_indices"]),
            "episode_start_dates_json": json.dumps(diagnostics["episode_start_dates"]),
            "raw_episode_generation_values_json": json.dumps(
                diagnostics["raw_episode_cascade_generation_gwh"]
            ),
            "raw_episode_penalized_objective_values_json": json.dumps(
                diagnostics["raw_episode_penalized_objective_gwh"]
            ),
            "episode_failed_station_months_values_json": json.dumps(
                diagnostics["episode_failed_station_months"]
            ),
            "raw_generation_mean_gwh": diagnostics["raw_generation_mean_gwh"],
            "raw_generation_std_gwh": diagnostics["raw_generation_std_gwh"],
            "raw_penalized_objective_mean_gwh": diagnostics[
                "raw_penalized_objective_mean_gwh"
            ],
            "raw_penalized_objective_std_gwh": diagnostics[
                "raw_penalized_objective_std_gwh"
            ],
            "failed_station_months_mean": diagnostics["failed_station_months_mean"],
            "failure_penalty_mean_gwh": diagnostics["failure_penalty_mean_gwh"],
            "normalized_reward_mean": diagnostics["normalized_reward_mean"],
            "normalized_reward_std": diagnostics["normalized_reward_std"],
            "rollout_policy_entropy": float(rollout_entropy.mean()),
            "rollout_xiaowan_action_std": float(rollout_std[:, :, 0].mean()),
            "rollout_nuozhadu_action_std": float(rollout_std[:, :, 1].mean()),
            "xiaowan_interval_width_mean_m": diagnostics["xiaowan_interval_width_mean_m"],
            "nuozhadu_interval_width_mean_m": diagnostics["nuozhadu_interval_width_mean_m"],
            "xiaowan_fraction_mean": diagnostics["xiaowan_fraction_mean"],
            "nuozhadu_fraction_mean": diagnostics["nuozhadu_fraction_mean"],
            "xiaowan_fixed_dimension_rate": diagnostics["xiaowan_fixed_dimension_rate"],
            "nuozhadu_fixed_dimension_rate": diagnostics["nuozhadu_fixed_dimension_rate"],
            "rollout_wall_time_seconds": rollout_wall_time,
            "optimization_wall_time_seconds": optimization_wall_time,
            "cumulative_training_wall_time_seconds": current_wall_time(),
            "cumulative_validation_wall_time_seconds": validation_wall_time_seconds,
            "samples_per_second": settings.rollout_size / max(rollout_wall_time, 1.0e-12),
            "validation_performed": should_validate,
            "validation_score": validation_score,
            "best_validation_score": best_validation_score,
            "best_validation_step": best_validation_step,
            "checkpoint_event": ",".join(checkpoint_events),
        }
        record.update(optimization)
        training_records.append(record)

        if selected_best:
            if paths.best_validation_checkpoint is None:
                raise CascadeSpecError("Continuous development run lacks best path.")
            _atomic_torch_save(checkpoint_payload("best_validation"), paths.best_validation_checkpoint)
        if save_latest:
            _atomic_torch_save(checkpoint_payload("latest"), paths.latest_checkpoint)
        if (
            update_number % settings.logging_interval_updates == 0
            or should_validate
            or save_latest
            or update_number == settings.total_updates
        ):
            _atomic_write_history(training_records, paths.training_history_csv)
            if paths.validation_history_csv is not None:
                _atomic_write_history(validation_records, paths.validation_history_csv)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "continuous_ppo_training_progress",
                        "mode": settings.mode,
                        "experiment_id": settings.experiment_id,
                        "seed": settings.seed,
                        "training": dict(record),
                        "validation": validation_record,
                        "paths": paths.as_dict(),
                    }
                )

    if settings.mode == "final":
        if paths.final_checkpoint is None:
            raise CascadeSpecError("Continuous final run lacks final path.")
        _atomic_torch_save(checkpoint_payload("final"), paths.final_checkpoint)
        selected_checkpoint = paths.final_checkpoint
    else:
        if paths.best_validation_checkpoint is None or not paths.best_validation_checkpoint.is_file():
            raise CascadeSpecError("Continuous development run has no best checkpoint.")
        selected_checkpoint = paths.best_validation_checkpoint
    return ContinuousPPOTrainingResult(
        mode=settings.mode,
        experiment_id=settings.experiment_id,
        seed=settings.seed,
        completed_updates=completed_update,
        environment_steps=environment_steps,
        completed_episode_windows=completed_update * settings.num_envs,
        training_wall_time_seconds=current_wall_time(),
        validation_wall_time_seconds=validation_wall_time_seconds,
        best_validation_score=best_validation_score,
        best_validation_step=best_validation_step,
        configuration_sha256=config_hash,
        normalization=normalizer.state_dict(),
        reward_scaling=reward_scaling,
        paths=paths,
        selected_checkpoint=selected_checkpoint,
    )


__all__ = [
    "CONTINUOUS_PPO_CHECKPOINT_SCHEMA_VERSION",
    "CONTINUOUS_PPO_TRAINING_MODES",
    "ContinuousPPOTrainingPaths",
    "ContinuousPPOTrainingResult",
    "build_continuous_ppo_training_paths",
    "inspect_continuous_ppo_configuration",
    "load_continuous_ppo_checkpoint",
    "train_continuous_ppo_seed",
]
