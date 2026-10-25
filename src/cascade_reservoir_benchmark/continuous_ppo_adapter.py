"""Continuous PPO adapters with exact state-dependent feasible intervals."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dqn_adapter import (
    INFLOW_COLUMNS,
    StateNormalizer,
    raw_state_from_env,
    sample_january_start_index,
    slice_episode_window,
)
from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    _as_month_period_index,
    _transition,
)


CONTINUOUS_ACTION_TRANSFORM = (
    "autoregressive_fraction_to_exact_physical_feasible_interval_v1"
)
XIAOWAN_GEOMETRY_DIMENSION = 3
NUOZHADU_GEOMETRY_DIMENSION = 4
ACTION_GEOMETRY_DIMENSION = (
    XIAOWAN_GEOMETRY_DIMENSION + NUOZHADU_GEOMETRY_DIMENSION
)
INTERVAL_TOLERANCE_M = 1.0e-10


def _reachable_level_bounds(
    spec: ReservoirSpec,
    *,
    current_storage_1e9m3: float,
    inflow_m3s: float,
    seconds: float,
    monthly_max_level_m: float,
) -> tuple[float, float]:
    if seconds <= 0.0 or inflow_m3s < 0.0:
        raise CascadeSpecError("Continuous action bounds require positive time and inflow.")
    available = float(current_storage_1e9m3) + float(inflow_m3s) * seconds / 1.0e9
    dead_storage = float(spec.storage_from_level(spec.dead_level_m))
    maximum_level = min(float(spec.normal_level_m), float(monthly_max_level_m))
    maximum_storage = min(float(spec.storage_from_level(maximum_level)), available)
    minimum_storage = dead_storage
    if spec.max_total_release_m3s is not None:
        minimum_storage = max(
            minimum_storage,
            available - float(spec.max_total_release_m3s) * seconds / 1.0e9,
        )
    minimum_storage = max(dead_storage, minimum_storage)
    maximum_storage = max(dead_storage, maximum_storage)
    lower = float(spec.level_from_storage(minimum_storage))
    upper = float(spec.level_from_storage(maximum_storage))
    lower = max(float(spec.dead_level_m), min(lower, maximum_level))
    upper = max(float(spec.dead_level_m), min(upper, maximum_level))
    if lower > upper + INTERVAL_TOLERANCE_M:
        raise CascadeSpecError(
            f"{spec.name}: continuous feasible target interval is empty."
        )
    if lower > upper:
        lower = upper
    return lower, upper


class ContinuousCascadePPOAdapter:
    """Map two normalized fractions to exact reachable cascade target levels."""

    def __init__(
        self,
        env: CascadeReservoirEnv,
        normalizer: StateNormalizer,
        *,
        reward_reference_days: float,
        reward_clip: Sequence[float] | None,
        max_transitions: int | None,
        target_reached_atol_m: float = 1.0e-7,
    ) -> None:
        if not isinstance(env, CascadeReservoirEnv):
            raise TypeError("env must be a CascadeReservoirEnv.")
        if not isinstance(normalizer, StateNormalizer):
            raise TypeError("normalizer must be a StateNormalizer.")
        if not np.isfinite(reward_reference_days) or reward_reference_days <= 0.0:
            raise ValueError("reward_reference_days must be positive.")
        if max_transitions is not None and int(max_transitions) <= 0:
            raise ValueError("max_transitions must be positive when provided.")
        self.env = env
        self.normalizer = normalizer
        self.reward_reference_days = float(reward_reference_days)
        self.reward_clip = None if reward_clip is None else tuple(float(v) for v in reward_clip)
        self.max_transitions = None if max_transitions is None else int(max_transitions)
        self.target_reached_atol_m = float(target_reached_atol_m)
        maximum_energy = (
            (float(env.reservoir_1.max_power_kw) + float(env.reservoir_2.max_power_kw))
            * self.reward_reference_days
            * 24.0
            / 1.0e6
        )
        if maximum_energy <= 0.0 or not np.isfinite(maximum_energy):
            raise CascadeSpecError("Continuous PPO requires finite power parameters.")
        self.reward_scale_gwh = float(maximum_energy)
        self._steps = 0
        self._ready = False
        self._target_unreached = np.zeros(2, dtype=np.int64)
        self._any_target_unreached = 0

    @property
    def reward_metadata(self) -> dict[str, Any]:
        return {
            "objective": "generation_gwh_minus_penalty_gwh_per_failed_station_month",
            "scale_gwh": self.reward_scale_gwh,
            "reference_days": self.reward_reference_days,
            "clip": self.reward_clip,
            "firm_power_failure_penalty_gwh": float(
                self.env.firm_power_failure_penalty_gwh
            ),
            "continuous_action_transform": CONTINUOUS_ACTION_TRANSFORM,
        }

    @property
    def episode_diagnostics(self) -> dict[str, Any]:
        denominator = self._steps if self._steps else 1
        return {
            "steps": self._steps,
            "target_unreached_count": {
                "xiaowan": int(self._target_unreached[0]),
                "nuozhadu": int(self._target_unreached[1]),
                "either_reservoir": int(self._any_target_unreached),
            },
            "target_unreached_rate": {
                "xiaowan": float(self._target_unreached[0] / denominator),
                "nuozhadu": float(self._target_unreached[1] / denominator),
                "either_reservoir": float(self._any_target_unreached / denominator),
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.env.reset(seed=seed, options=None if options is None else dict(options))
        raw = raw_state_from_env(self.env)
        self._steps = 0
        self._ready = True
        self._target_unreached[:] = 0
        self._any_target_unreached = 0
        return self.normalizer.transform(raw), {
            "state_raw": raw.copy(),
            "continuous_action_transform": CONTINUOUS_ACTION_TRANSFORM,
        }

    def _geometry(
        self, action_fractions: Sequence[float] | np.ndarray
    ) -> dict[str, Any]:
        if not self._ready:
            raise CascadeSpecError("Reset continuous PPO adapter before acting.")
        fractions = np.asarray(action_fractions, dtype=float)
        if fractions.shape != (2,) or np.any(~np.isfinite(fractions)):
            raise CascadeSpecError("Continuous action fractions must contain two finite values.")
        if np.any(fractions < 0.0) or np.any(fractions > 1.0):
            raise CascadeSpecError("Continuous action fractions must lie in [0, 1].")
        seconds = self.env._seconds_this_step()
        inflow_1, inflow_2_local = self.env.local_inflows_m3s[self.env.index]
        monthly = self.env.monthly_max_target_levels_m[self.env._month() - 1]
        xw_bounds = _reachable_level_bounds(
            self.env.reservoir_1,
            current_storage_1e9m3=float(self.env.storages_1e9m3[0]),
            inflow_m3s=float(inflow_1),
            seconds=seconds,
            monthly_max_level_m=float(monthly[0]),
        )
        xw_width = xw_bounds[1] - xw_bounds[0]
        xw_target = xw_bounds[0] + float(fractions[0]) * xw_width
        upstream = _transition(
            self.env.reservoir_1,
            float(self.env.storages_1e9m3[0]),
            float(inflow_1),
            xw_target,
            seconds,
        )
        routed = float(self.env.router.route(upstream.total_discharge_m3s))
        nzd_inflow = float(inflow_2_local) + routed
        nzd_bounds = _reachable_level_bounds(
            self.env.reservoir_2,
            current_storage_1e9m3=float(self.env.storages_1e9m3[1]),
            inflow_m3s=nzd_inflow,
            seconds=seconds,
            monthly_max_level_m=float(monthly[1]),
        )
        nzd_width = nzd_bounds[1] - nzd_bounds[0]
        nzd_target = nzd_bounds[0] + float(fractions[1]) * nzd_width
        return {
            "fractions": fractions.copy(),
            "targets_m": np.array((xw_target, nzd_target), dtype=float),
            "lower_bounds_m": np.array((xw_bounds[0], nzd_bounds[0]), dtype=float),
            "upper_bounds_m": np.array((xw_bounds[1], nzd_bounds[1]), dtype=float),
            "interval_widths_m": np.array((xw_width, nzd_width), dtype=float),
            "active_dimensions": np.array(
                (xw_width > INTERVAL_TOLERANCE_M, nzd_width > INTERVAL_TOLERANCE_M),
                dtype=np.bool_,
            ),
            "preview_routed_inflow_m3s": routed,
        }

    def xiaowan_action_active(self) -> bool:
        """Whether XiaoWan has a non-degenerate feasible interval this month."""
        geometry = self._geometry((0.5, 0.5))
        return bool(geometry["active_dimensions"][0])

    def xiaowan_geometry_features(self) -> np.ndarray:
        """Normalized current feasible geometry known before the action."""
        geometry = self._geometry((0.5, 0.5))
        spec = self.env.reservoir_1
        scale = float(spec.normal_level_m - spec.dead_level_m)
        if scale <= 0.0:
            raise CascadeSpecError("XiaoWan operating range must be positive.")
        lower = float(geometry["lower_bounds_m"][0])
        upper = float(geometry["upper_bounds_m"][0])
        return np.asarray(
            (
                (lower - float(spec.dead_level_m)) / scale,
                (upper - float(spec.dead_level_m)) / scale,
                (upper - lower) / scale,
            ),
            dtype=np.float32,
        )

    def nuozhadu_geometry_features(
        self, xiaowan_fraction: float
    ) -> np.ndarray:
        """Normalized downstream geometry after fixing the upstream action."""
        geometry = self._geometry((float(xiaowan_fraction), 0.5))
        xw = self.env.reservoir_1
        nzd = self.env.reservoir_2
        xw_scale = float(xw.normal_level_m - xw.dead_level_m)
        nzd_scale = float(nzd.normal_level_m - nzd.dead_level_m)
        if xw_scale <= 0.0 or nzd_scale <= 0.0:
            raise CascadeSpecError("Reservoir operating ranges must be positive.")
        xw_target = float(geometry["targets_m"][0])
        lower = float(geometry["lower_bounds_m"][1])
        upper = float(geometry["upper_bounds_m"][1])
        return np.asarray(
            (
                (xw_target - float(xw.dead_level_m)) / xw_scale,
                (lower - float(nzd.dead_level_m)) / nzd_scale,
                (upper - float(nzd.dead_level_m)) / nzd_scale,
                (upper - lower) / nzd_scale,
            ),
            dtype=np.float32,
        )

    def nuozhadu_action_active(self, xiaowan_fraction: float) -> bool:
        geometry = self._geometry((float(xiaowan_fraction), 0.5))
        return bool(geometry["active_dimensions"][1])

    def step(
        self, action_fractions: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        geometry = self._geometry(action_fractions)
        state_raw = raw_state_from_env(self.env)
        _, raw_reward, terminated, env_truncated, info = self.env.step(
            geometry["targets_m"]
        )
        upstream = info["reservoir_1"]
        downstream = info["reservoir_2"]
        actual = np.array(
            (float(upstream["end_level_m"]), float(downstream["end_level_m"])),
            dtype=float,
        )
        reached = np.isclose(
            actual,
            geometry["targets_m"],
            rtol=0.0,
            atol=self.target_reached_atol_m,
        )
        if not bool(np.all(reached)):
            raise CascadeSpecError(
                "Continuous feasible transform produced an unreachable target."
            )
        if bool(upstream["target_level_clipped"]) or bool(downstream["target_level_clipped"]):
            raise CascadeSpecError("Continuous PPO target was clipped by the environment.")
        self._steps += 1
        self._target_unreached += np.logical_not(reached).astype(np.int64)
        self._any_target_unreached += int(not bool(np.all(reached)))
        scaled_unclipped = float(raw_reward) / self.reward_scale_gwh
        scaled_reward = (
            float(np.clip(scaled_unclipped, self.reward_clip[0], self.reward_clip[1]))
            if self.reward_clip is not None
            else scaled_unclipped
        )
        artificial = bool(
            self.max_transitions is not None and self._steps == self.max_transitions
        )
        if artificial and terminated:
            raise CascadeSpecError(
                "Continuous training reached data terminal; provide a bootstrap month."
            )
        truncated = bool(env_truncated or artificial)
        if terminated:
            next_state = None
            self._ready = False
        else:
            next_state = self.normalizer.transform(raw_state_from_env(self.env))
        info.update(
            {
                "state_raw": state_raw,
                "next_state_normalized": None if next_state is None else next_state.copy(),
                "continuous_action_fractions": geometry["fractions"],
                "continuous_target_levels_m": geometry["targets_m"],
                "continuous_lower_bounds_m": geometry["lower_bounds_m"],
                "continuous_upper_bounds_m": geometry["upper_bounds_m"],
                "continuous_interval_widths_m": geometry["interval_widths_m"],
                "continuous_active_dimensions": geometry["active_dimensions"],
                "continuous_action_transform": CONTINUOUS_ACTION_TRANSFORM,
                "xiaowan_target_error_m": float(actual[0] - geometry["targets_m"][0]),
                "nuozhadu_target_error_m": float(actual[1] - geometry["targets_m"][1]),
                "generation_energy_gwh": float(info["generation_energy_gwh"]),
                "penalized_reward_gwh": float(info["penalized_reward_gwh"]),
                "failure_penalty_gwh": float(info["failure_penalty_gwh"]),
                "failed_station_months": int(info["failed_station_months"]),
                "normalized_reward_unclipped": scaled_unclipped,
                "normalized_reward": scaled_reward,
                "reward_scale_gwh": self.reward_scale_gwh,
                "artificial_window_truncation": artificial,
                "bootstrap_allowed": not bool(terminated),
            }
        )
        return next_state, scaled_reward, bool(terminated), truncated, info


@dataclass(frozen=True)
class ContinuousVectorStepResult:
    next_states: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_allowed: np.ndarray
    active_action_dimensions: np.ndarray
    infos: tuple[dict[str, Any], ...]


def make_continuous_episode_sampler_rngs(
    master_seed: int, num_envs: int
) -> list[np.random.Generator]:
    if int(master_seed) < 0 or int(num_envs) <= 0:
        raise CascadeSpecError("master_seed must be nonnegative and num_envs positive.")
    children = np.random.SeedSequence(int(master_seed)).spawn(int(num_envs))
    return [np.random.default_rng(child) for child in children]


class ContinuousSynchronousPPOEnvironmentSet:
    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        learning_frame: pd.DataFrame,
        normalizer: StateNormalizer,
        sampler_rngs: Sequence[np.random.Generator],
        *,
        episode_months: int,
        initial_levels_m: Sequence[float],
        reward_reference_days: float,
        reward_clip: Sequence[float] | None,
        firm_power_failure_penalty_gwh: float,
        monthly_max_target_levels_m: Sequence[Sequence[float]] | None,
    ) -> None:
        self.reservoir_1 = reservoir_1
        self.reservoir_2 = reservoir_2
        self.learning_frame = learning_frame
        self.normalizer = normalizer
        self.sampler_rngs = list(sampler_rngs)
        self.num_envs = len(self.sampler_rngs)
        self.episode_months = int(episode_months)
        self.initial_levels_m = tuple(float(v) for v in initial_levels_m)
        self.reward_reference_days = float(reward_reference_days)
        self.reward_clip = None if reward_clip is None else tuple(reward_clip)
        self.firm_power_failure_penalty_gwh = float(firm_power_failure_penalty_gwh)
        self.monthly_max_target_levels_m = monthly_max_target_levels_m
        self.adapters: list[ContinuousCascadePPOAdapter] = []
        self.current_states: np.ndarray | None = None
        self.current_step = 0
        self.start_indices = np.zeros(self.num_envs, dtype=np.int64)
        self.start_dates: list[str] = []
        self.end_dates: list[str] = []
        self.bootstrap_dates: list[str] = []
        self.raw_episode_generation_gwh = np.zeros(self.num_envs, dtype=np.float64)
        self.raw_episode_penalized_objective_gwh = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_failed_station_months = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_failure_penalty_gwh = np.zeros(self.num_envs, dtype=np.float64)
        self.normalized_episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.interval_widths: list[np.ndarray] = []
        self.action_fractions: list[np.ndarray] = []

    def reset_rollout(self) -> np.ndarray:
        self.adapters = []
        states: list[np.ndarray] = []
        self.start_dates = []
        self.end_dates = []
        self.bootstrap_dates = []
        for index, rng in enumerate(self.sampler_rngs):
            start = sample_january_start_index(
                self.learning_frame, rng, episode_months=self.episode_months
            )
            window = slice_episode_window(
                self.learning_frame, start, episode_months=self.episode_months
            )
            periods = _as_month_period_index(window["date"])
            flows = window.loc[:, list(INFLOW_COLUMNS)].to_numpy(dtype=float)
            env = CascadeReservoirEnv(
                self.reservoir_1,
                self.reservoir_2,
                flows,
                timestamps=periods,
                initial_levels_m=self.initial_levels_m,
                firm_power_failure_penalty_gwh=self.firm_power_failure_penalty_gwh,
                monthly_max_target_levels_m=self.monthly_max_target_levels_m,
                require_energy_parameters=True,
            )
            adapter = ContinuousCascadePPOAdapter(
                env,
                self.normalizer,
                reward_reference_days=self.reward_reference_days,
                reward_clip=self.reward_clip,
                max_transitions=self.episode_months,
            )
            state, _ = adapter.reset()
            self.adapters.append(adapter)
            states.append(state)
            self.start_indices[index] = start
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
        self.interval_widths = []
        self.action_fractions = []
        return self.current_states.copy()

    def step(self, action_fractions: np.ndarray) -> ContinuousVectorStepResult:
        fractions = np.asarray(action_fractions, dtype=float)
        if fractions.shape != (self.num_envs, 2):
            raise CascadeSpecError(
                f"continuous actions must have shape ({self.num_envs}, 2)."
            )
        next_states: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        bootstrap = np.zeros(self.num_envs, dtype=np.bool_)
        active = np.zeros((self.num_envs, 2), dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        expected_final = self.current_step == self.episode_months - 1
        for index, (adapter, action) in enumerate(
            zip(self.adapters, fractions, strict=True)
        ):
            next_state, reward, term, trunc, info = adapter.step(action)
            if term or next_state is None:
                raise CascadeSpecError("Continuous training window reached physical terminal.")
            if bool(trunc) != expected_final or not bool(info["bootstrap_allowed"]):
                raise CascadeSpecError("Continuous PPO truncation/bootstrap invariant failed.")
            next_states.append(next_state)
            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
            bootstrap[index] = bool(info["bootstrap_allowed"])
            active[index] = np.asarray(info["continuous_active_dimensions"], dtype=np.bool_)
            infos.append(dict(info))
            self.raw_episode_generation_gwh[index] += float(info["generation_energy_gwh"])
            self.raw_episode_penalized_objective_gwh[index] += float(info["penalized_reward_gwh"])
            self.episode_failed_station_months[index] += int(info["failed_station_months"])
            self.episode_failure_penalty_gwh[index] += float(info["failure_penalty_gwh"])
            self.normalized_episode_rewards[index] += float(reward)
            self.interval_widths.append(
                np.asarray(info["continuous_interval_widths_m"], dtype=float)
            )
            self.action_fractions.append(
                np.asarray(info["continuous_action_fractions"], dtype=float)
            )
        self.current_step += 1
        self.current_states = np.stack(next_states).astype(np.float32, copy=False)
        return ContinuousVectorStepResult(
            next_states=self.current_states.copy(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            bootstrap_allowed=bootstrap,
            active_action_dimensions=active,
            infos=tuple(infos),
        )

    def get_xiaowan_active_dimensions(self) -> np.ndarray:
        if self.current_states is None or len(self.adapters) != self.num_envs:
            raise CascadeSpecError("Reset rollout before requesting action geometry.")
        return np.asarray(
            [adapter.xiaowan_action_active() for adapter in self.adapters],
            dtype=np.bool_,
        )

    def get_xiaowan_geometry_features(self) -> np.ndarray:
        if self.current_states is None or len(self.adapters) != self.num_envs:
            raise CascadeSpecError("Reset rollout before requesting action geometry.")
        return np.stack(
            [adapter.xiaowan_geometry_features() for adapter in self.adapters]
        ).astype(np.float32, copy=False)

    def get_nuozhadu_geometry_features(
        self, xiaowan_fractions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        fractions = np.asarray(xiaowan_fractions, dtype=float)
        if fractions.shape != (self.num_envs,):
            raise CascadeSpecError(
                f"xiaowan_fractions must have shape ({self.num_envs},)."
            )
        features = np.stack(
            [
                adapter.nuozhadu_geometry_features(fraction)
                for adapter, fraction in zip(
                    self.adapters, fractions, strict=True
                )
            ]
        ).astype(np.float32, copy=False)
        active = np.asarray(
            [
                adapter.nuozhadu_action_active(fraction)
                for adapter, fraction in zip(
                    self.adapters, fractions, strict=True
                )
            ],
            dtype=np.bool_,
        )
        return features, active

    def rollout_diagnostics(self) -> dict[str, Any]:
        if self.current_step != self.episode_months:
            raise CascadeSpecError("Continuous rollout diagnostics require a complete rollout.")
        transitions = self.num_envs * self.episode_months
        widths = np.stack(self.interval_widths)
        fractions = np.stack(self.action_fractions)
        return {
            "completed_episode_windows": self.num_envs,
            "rollout_transitions": transitions,
            "episode_start_indices": self.start_indices.tolist(),
            "episode_start_dates": list(self.start_dates),
            "episode_end_dates": list(self.end_dates),
            "episode_bootstrap_dates": list(self.bootstrap_dates),
            "raw_episode_cascade_generation_gwh": self.raw_episode_generation_gwh.tolist(),
            "raw_episode_penalized_objective_gwh": self.raw_episode_penalized_objective_gwh.tolist(),
            "episode_failed_station_months": self.episode_failed_station_months.tolist(),
            "episode_failure_penalty_gwh": self.episode_failure_penalty_gwh.tolist(),
            "normalized_episode_rewards": self.normalized_episode_rewards.tolist(),
            "raw_generation_mean_gwh": float(self.raw_episode_generation_gwh.mean()),
            "raw_generation_std_gwh": float(self.raw_episode_generation_gwh.std(ddof=0)),
            "raw_penalized_objective_mean_gwh": float(self.raw_episode_penalized_objective_gwh.mean()),
            "raw_penalized_objective_std_gwh": float(self.raw_episode_penalized_objective_gwh.std(ddof=0)),
            "failed_station_months_mean": float(self.episode_failed_station_months.mean()),
            "failure_penalty_mean_gwh": float(self.episode_failure_penalty_gwh.mean()),
            "normalized_reward_mean": float(self.normalized_episode_rewards.mean()),
            "normalized_reward_std": float(self.normalized_episode_rewards.std(ddof=0)),
            "xiaowan_interval_width_mean_m": float(widths[:, 0].mean()),
            "nuozhadu_interval_width_mean_m": float(widths[:, 1].mean()),
            "xiaowan_fraction_mean": float(fractions[:, 0].mean()),
            "nuozhadu_fraction_mean": float(fractions[:, 1].mean()),
            "xiaowan_fixed_dimension_rate": float((widths[:, 0] <= INTERVAL_TOLERANCE_M).mean()),
            "nuozhadu_fixed_dimension_rate": float((widths[:, 1] <= INTERVAL_TOLERANCE_M).mean()),
        }

    def sampler_rng_states(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(rng.bit_generator.state) for rng in self.sampler_rngs]

    def load_sampler_rng_states(self, states: Sequence[Mapping[str, Any]]) -> None:
        if len(states) != self.num_envs:
            raise CascadeSpecError("Continuous PPO sampler RNG count differs.")
        for rng, state in zip(self.sampler_rngs, states, strict=True):
            rng.bit_generator.state = copy.deepcopy(dict(state))


__all__ = [
    "ACTION_GEOMETRY_DIMENSION",
    "CONTINUOUS_ACTION_TRANSFORM",
    "ContinuousCascadePPOAdapter",
    "ContinuousSynchronousPPOEnvironmentSet",
    "ContinuousVectorStepResult",
    "INTERVAL_TOLERANCE_M",
    "NUOZHADU_GEOMETRY_DIMENSION",
    "XIAOWAN_GEOMETRY_DIMENSION",
    "make_continuous_episode_sampler_rngs",
]
