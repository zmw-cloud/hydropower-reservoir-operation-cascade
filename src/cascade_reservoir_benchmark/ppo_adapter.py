"""Thin PPO adapters over the validated cascade environment and DQN utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dqn_adapter import (
    INFLOW_COLUMNS,
    STATE_ORDER,
    CascadeDQNAdapter,
    JointActionGrid,
    StateNormalizer,
    sample_january_start_index,
    slice_episode_window,
)
from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    _as_month_period_index,
)


class CascadePPOAdapter:
    """PPO-named view that delegates state/action/reward semantics unchanged."""

    def __init__(
        self,
        env: CascadeReservoirEnv,
        normalizer: StateNormalizer,
        action_grid: JointActionGrid,
        *,
        reward_reference_days: float,
        reward_clip: Sequence[float] | None,
        max_transitions: int | None,
    ) -> None:
        self._delegate = CascadeDQNAdapter(
            env,
            normalizer,
            action_grid,
            reward_reference_days=reward_reference_days,
            reward_clip=reward_clip,
            max_transitions=max_transitions,
        )

    @property
    def env(self) -> CascadeReservoirEnv:
        return self._delegate.env

    @property
    def normalizer(self) -> StateNormalizer:
        return self._delegate.normalizer

    @property
    def action_grid(self) -> JointActionGrid:
        return self._delegate.action_grid

    @property
    def reward_metadata(self) -> dict[str, Any]:
        return self._delegate.reward_metadata

    @property
    def episode_diagnostics(self) -> dict[str, Any]:
        return self._delegate.episode_diagnostics

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._delegate.reset(seed=seed, options=options)

    def step(
        self, action_index: int
    ) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        return self._delegate.step(action_index)

    def get_feasible_action_mask(self) -> np.ndarray:
        return self._delegate.get_feasible_action_mask()


@dataclass(frozen=True)
class VectorStepResult:
    next_states: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_allowed: np.ndarray
    infos: tuple[dict[str, Any], ...]


def make_episode_sampler_rngs(
    master_seed: int,
    num_envs: int,
) -> list[np.random.Generator]:
    """Derive independent deterministic January samplers from one seed."""
    if isinstance(master_seed, bool) or not isinstance(
        master_seed, (int, np.integer)
    ):
        raise CascadeSpecError("master_seed must be an integer.")
    if isinstance(num_envs, bool) or not isinstance(num_envs, (int, np.integer)):
        raise CascadeSpecError("num_envs must be an integer.")
    count = int(num_envs)
    if int(master_seed) < 0 or count <= 0:
        raise CascadeSpecError("master_seed must be nonnegative and num_envs positive.")
    children = np.random.SeedSequence(int(master_seed)).spawn(count)
    return [np.random.default_rng(child) for child in children]


class SynchronousPPOEnvironmentSet:
    """Eight logical environments, each holding one independent 241-month window."""

    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        learning_frame: pd.DataFrame,
        normalizer: StateNormalizer,
        action_grid: JointActionGrid,
        sampler_rngs: Sequence[np.random.Generator],
        *,
        episode_months: int,
        initial_levels_m: Sequence[float],
        reward_reference_days: float,
        reward_clip: Sequence[float] | None,
        firm_power_failure_penalty_gwh: float,
        monthly_max_target_levels_m: Sequence[Sequence[float]] | None,
    ) -> None:
        if not sampler_rngs:
            raise CascadeSpecError("At least one PPO training environment is required.")
        if any(not isinstance(rng, np.random.Generator) for rng in sampler_rngs):
            raise CascadeSpecError("Every PPO episode sampler must be a Generator.")
        if int(episode_months) <= 0:
            raise CascadeSpecError("episode_months must be positive.")
        initial = np.asarray(initial_levels_m, dtype=float)
        if initial.shape != (2,) or np.any(~np.isfinite(initial)):
            raise CascadeSpecError("initial_levels_m must contain two finite values.")
        self.reservoir_1 = reservoir_1
        self.reservoir_2 = reservoir_2
        self.learning_frame = learning_frame
        self.normalizer = normalizer
        self.action_grid = action_grid
        self.sampler_rngs = list(sampler_rngs)
        self.num_envs = len(self.sampler_rngs)
        self.episode_months = int(episode_months)
        self.initial_levels_m = (float(initial[0]), float(initial[1]))
        self.reward_reference_days = float(reward_reference_days)
        self.reward_clip = None if reward_clip is None else tuple(reward_clip)
        self.firm_power_failure_penalty_gwh = float(
            firm_power_failure_penalty_gwh
        )
        self.monthly_max_target_levels_m = monthly_max_target_levels_m
        self.adapters: list[CascadePPOAdapter] = []
        self.current_states: np.ndarray | None = None
        self.current_step = 0
        self.start_indices = np.zeros(self.num_envs, dtype=np.int64)
        self.start_dates: list[str] = []
        self.end_dates: list[str] = []
        self.bootstrap_dates: list[str] = []
        self.raw_episode_generation_gwh = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.raw_episode_penalized_objective_gwh = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.episode_failed_station_months = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self.episode_failure_penalty_gwh = np.zeros(
            self.num_envs, dtype=np.float64
        )
        self.normalized_episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.absolute_target_errors_xiaowan: list[float] = []
        self.absolute_target_errors_nuozhadu: list[float] = []

    def reset_rollout(self) -> np.ndarray:
        """Sample eight independent January windows and reset all physical states."""
        self.adapters = []
        states: list[np.ndarray] = []
        self.start_dates = []
        self.end_dates = []
        self.bootstrap_dates = []
        for env_index, rng in enumerate(self.sampler_rngs):
            start_index = sample_january_start_index(
                self.learning_frame,
                rng,
                episode_months=self.episode_months,
            )
            window = slice_episode_window(
                self.learning_frame,
                start_index,
                episode_months=self.episode_months,
            )
            periods = _as_month_period_index(window["date"])
            flows = window.loc[:, list(INFLOW_COLUMNS)].to_numpy(dtype=float)
            env = CascadeReservoirEnv(
                self.reservoir_1,
                self.reservoir_2,
                flows,
                timestamps=periods,
                initial_levels_m=self.initial_levels_m,
                firm_power_failure_penalty_gwh=(
                    self.firm_power_failure_penalty_gwh
                ),
                monthly_max_target_levels_m=(
                    self.monthly_max_target_levels_m
                ),
                require_energy_parameters=True,
            )
            adapter = CascadePPOAdapter(
                env,
                self.normalizer,
                self.action_grid,
                reward_reference_days=self.reward_reference_days,
                reward_clip=self.reward_clip,
                max_transitions=self.episode_months,
            )
            state, _ = adapter.reset()
            self.adapters.append(adapter)
            states.append(state)
            self.start_indices[env_index] = start_index
            self.start_dates.append(str(periods[0]))
            self.end_dates.append(str(periods[self.episode_months - 1]))
            self.bootstrap_dates.append(str(periods[self.episode_months]))
        self.current_states = np.stack(states).astype(np.float32, copy=False)
        self.current_step = 0
        self.raw_episode_generation_gwh[:] = 0.0
        self.raw_episode_penalized_objective_gwh[:] = 0.0
        self.episode_failed_station_months[:] = 0
        self.episode_failure_penalty_gwh[:] = 0.0
        self.normalized_episode_rewards[:] = 0.0
        self.absolute_target_errors_xiaowan = []
        self.absolute_target_errors_nuozhadu = []
        return self.current_states.copy()

    def step(self, actions: Sequence[int] | np.ndarray) -> VectorStepResult:
        if self.current_states is None or len(self.adapters) != self.num_envs:
            raise CascadeSpecError("Call reset_rollout before stepping PPO environments.")
        if self.current_step >= self.episode_months:
            raise CascadeSpecError("The current PPO rollout is already complete.")
        action_array = np.asarray(actions)
        if action_array.shape != (self.num_envs,) or not np.issubdtype(
            action_array.dtype, np.integer
        ):
            raise CascadeSpecError(
                f"actions must contain {self.num_envs} integer action indices."
            )
        next_states: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        bootstrap_allowed = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        expected_final = self.current_step == self.episode_months - 1

        for env_index, (adapter, action) in enumerate(
            zip(self.adapters, action_array, strict=True)
        ):
            next_state, reward, is_terminated, is_truncated, info = adapter.step(
                int(action)
            )
            if is_terminated:
                raise CascadeSpecError(
                    "PPO training reached a physical data terminal; "
                    "the 241-month window invariant is broken."
                )
            if next_state is None:
                raise CascadeSpecError(
                    "A nonterminal PPO transition did not return its real next state."
                )
            if bool(is_truncated) != expected_final:
                raise CascadeSpecError(
                    "PPO artificial truncation must occur exactly after 240 transitions."
                )
            allowed = bool(info.get("bootstrap_allowed"))
            if not allowed:
                raise CascadeSpecError(
                    "A nonterminal PPO training transition must allow bootstrap."
                )
            rewards[env_index] = float(reward)
            terminated[env_index] = bool(is_terminated)
            truncated[env_index] = bool(is_truncated)
            bootstrap_allowed[env_index] = allowed
            next_states.append(next_state)
            infos.append(dict(info))
            self.raw_episode_generation_gwh[env_index] += float(
                info["generation_energy_gwh"]
            )
            self.raw_episode_penalized_objective_gwh[env_index] += float(
                info["penalized_reward_gwh"]
            )
            self.episode_failed_station_months[env_index] += int(
                info["failed_station_months"]
            )
            self.episode_failure_penalty_gwh[env_index] += float(
                info["failure_penalty_gwh"]
            )
            self.normalized_episode_rewards[env_index] += float(reward)
            self.absolute_target_errors_xiaowan.append(
                abs(float(info["xiaowan_target_error_m"]))
            )
            self.absolute_target_errors_nuozhadu.append(
                abs(float(info["nuozhadu_target_error_m"]))
            )

        self.current_step += 1
        self.current_states = np.stack(next_states).astype(np.float32, copy=False)
        return VectorStepResult(
            next_states=self.current_states.copy(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            bootstrap_allowed=bootstrap_allowed,
            infos=tuple(infos),
        )

    def get_feasible_action_masks(self) -> np.ndarray:
        """Stack one physics-derived mask for every active logical environment."""
        if self.current_states is None or len(self.adapters) != self.num_envs:
            raise CascadeSpecError(
                "Call reset_rollout before requesting PPO action masks."
            )
        masks = np.stack(
            [adapter.get_feasible_action_mask() for adapter in self.adapters]
        ).astype(np.bool_, copy=False)
        expected = (self.num_envs, self.action_grid.action_count)
        if masks.shape != expected or np.any(~masks.any(axis=1)):
            raise CascadeSpecError(
                f"PPO feasible-action masks must have shape {expected} and one valid action per row."
            )
        return masks.copy()

    def rollout_diagnostics(self) -> dict[str, Any]:
        if self.current_step != self.episode_months:
            raise CascadeSpecError("PPO rollout diagnostics require a complete rollout.")
        counts_xw = 0
        counts_nzd = 0
        counts_either = 0
        for adapter in self.adapters:
            counts = adapter.episode_diagnostics["target_unreached_count"]
            counts_xw += int(counts["xiaowan"])
            counts_nzd += int(counts["nuozhadu"])
            counts_either += int(counts["either_reservoir"])
        transitions = self.num_envs * self.episode_months

        def error_summary(values: Sequence[float]) -> dict[str, float]:
            array = np.asarray(values, dtype=float)
            return {
                "mean": float(array.mean()),
                "max": float(array.max()),
                "p95": float(np.quantile(array, 0.95)),
            }

        return {
            "completed_episode_windows": self.num_envs,
            "rollout_transitions": transitions,
            "episode_start_indices": self.start_indices.tolist(),
            "episode_start_dates": list(self.start_dates),
            "episode_end_dates": list(self.end_dates),
            "episode_bootstrap_dates": list(self.bootstrap_dates),
            "raw_episode_cascade_generation_gwh": (
                self.raw_episode_generation_gwh.tolist()
            ),
            "raw_episode_penalized_objective_gwh": (
                self.raw_episode_penalized_objective_gwh.tolist()
            ),
            "episode_failed_station_months": (
                self.episode_failed_station_months.tolist()
            ),
            "episode_failure_penalty_gwh": (
                self.episode_failure_penalty_gwh.tolist()
            ),
            "normalized_episode_rewards": self.normalized_episode_rewards.tolist(),
            "raw_generation_mean_gwh": float(
                self.raw_episode_generation_gwh.mean()
            ),
            "raw_generation_std_gwh": float(
                self.raw_episode_generation_gwh.std(ddof=0)
            ),
            "raw_penalized_objective_mean_gwh": float(
                self.raw_episode_penalized_objective_gwh.mean()
            ),
            "raw_penalized_objective_std_gwh": float(
                self.raw_episode_penalized_objective_gwh.std(ddof=0)
            ),
            "failed_station_months_mean": float(
                self.episode_failed_station_months.mean()
            ),
            "failure_penalty_mean_gwh": float(
                self.episode_failure_penalty_gwh.mean()
            ),
            "normalized_reward_mean": float(
                self.normalized_episode_rewards.mean()
            ),
            "normalized_reward_std": float(
                self.normalized_episode_rewards.std(ddof=0)
            ),
            "target_unreached_count": {
                "xiaowan": counts_xw,
                "nuozhadu": counts_nzd,
                "either_reservoir": counts_either,
            },
            "target_unreached_rate": {
                "xiaowan": counts_xw / transitions,
                "nuozhadu": counts_nzd / transitions,
                "either_reservoir": counts_either / transitions,
            },
            "target_absolute_error_m": {
                "xiaowan": error_summary(self.absolute_target_errors_xiaowan),
                "nuozhadu": error_summary(self.absolute_target_errors_nuozhadu),
            },
        }

    def sampler_rng_states(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(rng.bit_generator.state) for rng in self.sampler_rngs]

    def load_sampler_rng_states(self, states: Sequence[Mapping[str, Any]]) -> None:
        if len(states) != self.num_envs:
            raise CascadeSpecError("PPO checkpoint sampler RNG count is incompatible.")
        for rng, state in zip(self.sampler_rngs, states, strict=True):
            rng.bit_generator.state = copy.deepcopy(dict(state))


__all__ = [
    "CascadePPOAdapter",
    "SynchronousPPOEnvironmentSet",
    "VectorStepResult",
    "make_episode_sampler_rngs",
]
