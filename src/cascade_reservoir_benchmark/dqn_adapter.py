"""State, action, and environment adapter for the P4 cascade Double DQN.

This module deliberately contains no reservoir physics.  It translates the
shared :class:`CascadeReservoirEnv` interface into the five-component network
state and a configurable discrete joint target-level action grid.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .ddp import make_level_grid
from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    _as_month_period_index,
)


STATE_ORDER = (
    "month",
    "xiaowan_level_m",
    "nuozhadu_level_m",
    "xiaowan_natural_inflow_m3s",
    "nuozhadu_interval_inflow_m3s",
)
INFLOW_COLUMNS = ("xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s")
ACTION_ENCODING = "row_major_xiaowan_index_then_nuozhadu_index"
TARGET_REACHED_ATOL_M = 1.0e-8


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise CascadeSpecError(f"{label} must be a mapping.")
    actual = set(payload)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise CascadeSpecError(
            f"{label} keys are invalid; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise CascadeSpecError(f"{name} must be a positive integer.")
    return int(value)


def _validated_inflow_frame(frame: pd.DataFrame) -> tuple[pd.PeriodIndex, np.ndarray]:
    if not isinstance(frame, pd.DataFrame):
        raise CascadeSpecError("Inflow data must be a pandas DataFrame.")
    missing = {"date", *INFLOW_COLUMNS} - set(frame.columns)
    if missing or frame.columns.duplicated().any():
        raise CascadeSpecError(
            f"Inflow data require unique date/source columns; missing: {sorted(missing)}."
        )
    if len(frame) == 0:
        raise CascadeSpecError("Inflow data must contain at least one month.")
    periods = _as_month_period_index(frame["date"])
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    if not periods.equals(expected):
        raise CascadeSpecError("Inflow dates must be unique, ordered, and continuous months.")
    flows = (
        frame.loc[:, list(INFLOW_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if np.any(~np.isfinite(flows)) or np.any(flows < 0.0):
        raise CascadeSpecError("External inflows must be finite and nonnegative.")
    return periods, flows


class StateNormalizer:
    """Componentwise min-max normalization for the fixed DQN state order."""

    schema_version = 1
    method = "componentwise_minmax"

    def __init__(self, lower: Sequence[float], upper: Sequence[float], *, clip: bool = True) -> None:
        if not isinstance(clip, bool):
            raise CascadeSpecError("State-normalization clip must be a boolean.")
        lower_array = np.asarray(lower, dtype=float)
        upper_array = np.asarray(upper, dtype=float)
        if lower_array.shape != (len(STATE_ORDER),) or upper_array.shape != (len(STATE_ORDER),):
            raise CascadeSpecError("State-normalization bounds must each contain five values.")
        if np.any(~np.isfinite(lower_array)) or np.any(~np.isfinite(upper_array)):
            raise CascadeSpecError("State-normalization bounds must be finite.")
        if np.any(upper_array <= lower_array):
            raise CascadeSpecError("Every state-normalization upper bound must exceed its lower bound.")
        self.lower = lower_array.copy()
        self.upper = upper_array.copy()
        self.clip = clip

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        *,
        clip: bool = True,
    ) -> "StateNormalizer":
        """Fit only the two inflow bounds from the supplied training frame.

        Month and level bounds come from the physical reservoir specs.  The
        method never opens another dataset, so callers control whether the
        supplied frame is the 320-year development subset or full 400-year
        final-training sequence.
        """
        _, flows = _validated_inflow_frame(frame)
        lower = np.array(
            [
                1.0,
                reservoir_1.dead_level_m,
                reservoir_2.dead_level_m,
                float(flows[:, 0].min()),
                float(flows[:, 1].min()),
            ],
            dtype=float,
        )
        upper = np.array(
            [
                12.0,
                reservoir_1.normal_level_m,
                reservoir_2.normal_level_m,
                float(flows[:, 0].max()),
                float(flows[:, 1].max()),
            ],
            dtype=float,
        )
        return cls(lower, upper, clip=clip)

    def validate_for_specs(self, reservoir_1: ReservoirSpec, reservoir_2: ReservoirSpec) -> None:
        expected_lower = np.array(
            [1.0, reservoir_1.dead_level_m, reservoir_2.dead_level_m], dtype=float
        )
        expected_upper = np.array(
            [12.0, reservoir_1.normal_level_m, reservoir_2.normal_level_m], dtype=float
        )
        if not np.array_equal(self.lower[:3], expected_lower) or not np.array_equal(
            self.upper[:3], expected_upper
        ):
            raise CascadeSpecError(
                "State-normalization month/level bounds do not match the active reservoir specs."
            )

    def transform(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=float)
        if values.ndim == 0 or values.shape[-1] != len(STATE_ORDER):
            raise CascadeSpecError("A DQN state must have five components on its final axis.")
        if np.any(~np.isfinite(values)):
            raise CascadeSpecError("DQN state components must be finite.")
        normalized = (values - self.lower) / (self.upper - self.lower)
        if self.clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "state_order": list(STATE_ORDER),
            "clip": self.clip,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
    ) -> "StateNormalizer":
        expected_keys = {"schema_version", "method", "state_order", "clip", "lower", "upper"}
        _require_exact_keys(payload, expected_keys, "State-normalization metadata")
        if payload["schema_version"] != cls.schema_version:
            raise CascadeSpecError("Unsupported state-normalization schema version.")
        if payload["method"] != cls.method:
            raise CascadeSpecError("Unsupported state-normalization method.")
        if payload["state_order"] != list(STATE_ORDER):
            raise CascadeSpecError("Saved state-normalization order is incompatible with DQN V1.")
        normalizer = cls(payload["lower"], payload["upper"], clip=payload["clip"])
        normalizer.validate_for_specs(reservoir_1, reservoir_2)
        return normalizer


class JointActionGrid:
    """Stable row-major encoding of the two target-level grids."""

    schema_version = 1

    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        step_m: float = 2.0,
    ) -> None:
        if isinstance(step_m, bool) or not np.isfinite(step_m) or step_m <= 0.0:
            raise CascadeSpecError("DQN action-grid step must be finite and positive.")
        self.step_m = float(step_m)
        self.xiaowan_levels_m = make_level_grid(
            reservoir_1.dead_level_m, reservoir_1.normal_level_m, self.step_m
        )
        self.nuozhadu_levels_m = make_level_grid(
            reservoir_2.dead_level_m, reservoir_2.normal_level_m, self.step_m
        )
        self.shape = (len(self.xiaowan_levels_m), len(self.nuozhadu_levels_m))
        self.action_count = int(self.shape[0] * self.shape[1])

    @staticmethod
    def _index(value: Any, size: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise CascadeSpecError(f"{name} must be an integer.")
        index = int(value)
        if not 0 <= index < size:
            raise CascadeSpecError(f"{name} is outside [0, {size - 1}].")
        return index

    def encode(self, xiaowan_grid_index: int, nuozhadu_grid_index: int) -> int:
        upstream = self._index(xiaowan_grid_index, self.shape[0], "xiaowan_grid_index")
        downstream = self._index(
            nuozhadu_grid_index, self.shape[1], "nuozhadu_grid_index"
        )
        return upstream * self.shape[1] + downstream

    def decode(self, action_index: int) -> tuple[int, int]:
        encoded = self._index(action_index, self.action_count, "action_index")
        upstream, downstream = divmod(encoded, self.shape[1])
        return int(upstream), int(downstream)

    def action(self, action_index: int) -> tuple[float, float]:
        upstream, downstream = self.decode(action_index)
        return (
            float(self.xiaowan_levels_m[upstream]),
            float(self.nuozhadu_levels_m[downstream]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoding": ACTION_ENCODING,
            "step_m": self.step_m,
            "shape": list(self.shape),
            "action_count": self.action_count,
            "xiaowan_levels_m": self.xiaowan_levels_m.tolist(),
            "nuozhadu_levels_m": self.nuozhadu_levels_m.tolist(),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
    ) -> "JointActionGrid":
        expected_keys = {
            "schema_version",
            "encoding",
            "step_m",
            "shape",
            "action_count",
            "xiaowan_levels_m",
            "nuozhadu_levels_m",
        }
        _require_exact_keys(payload, expected_keys, "Joint-action metadata")
        if payload["schema_version"] != cls.schema_version:
            raise CascadeSpecError("Unsupported joint-action schema version.")
        if payload["encoding"] != ACTION_ENCODING:
            raise CascadeSpecError("Saved joint-action encoding is incompatible with DQN V1.")
        grid = cls(reservoir_1, reservoir_2, step_m=payload["step_m"])
        if payload["shape"] != list(grid.shape) or payload["action_count"] != grid.action_count:
            raise CascadeSpecError("Saved joint-action dimensions do not match the regenerated grids.")
        try:
            saved_upstream = np.asarray(payload["xiaowan_levels_m"], dtype=float)
            saved_downstream = np.asarray(payload["nuozhadu_levels_m"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise CascadeSpecError("Saved joint-action level grids must be numeric.") from exc
        if not np.array_equal(saved_upstream, grid.xiaowan_levels_m) or not np.array_equal(
            saved_downstream, grid.nuozhadu_levels_m
        ):
            raise CascadeSpecError("Saved joint-action levels do not match the active reservoir specs.")
        return grid


def raw_state_from_env_observation(
    observation: Sequence[float] | np.ndarray,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
) -> np.ndarray:
    """Convert ``[V1,V2,I1,I2_local,month]`` to the DQN water-level state."""
    values = np.asarray(observation, dtype=float)
    if values.shape != (5,) or np.any(~np.isfinite(values)):
        raise CascadeSpecError("Cascade environment observation must contain five finite values.")
    if values[2] < 0.0 or values[3] < 0.0:
        raise CascadeSpecError("Cascade environment observation inflows must be nonnegative.")
    month = int(round(float(values[4])))
    if not np.isclose(values[4], month, rtol=0.0, atol=0.0) or not 1 <= month <= 12:
        raise CascadeSpecError("Cascade environment observation month must be an integer from 1 to 12.")
    return np.array(
        [
            float(month),
            reservoir_1.level_from_storage(float(values[0])),
            reservoir_2.level_from_storage(float(values[1])),
            float(values[2]),
            float(values[3]),
        ],
        dtype=float,
    )


def raw_state_from_env(env: CascadeReservoirEnv) -> np.ndarray:
    """Read the exact current physical storage and external flows from ``env``."""
    if not isinstance(env, CascadeReservoirEnv):
        raise CascadeSpecError("raw_state_from_env requires a CascadeReservoirEnv.")
    if not 0 <= env.index < len(env.local_inflows_m3s):
        raise CascadeSpecError("Cascade environment index is outside its inflow sequence.")
    inflow_1, inflow_2_local = env.local_inflows_m3s[env.index]
    month = (
        int(env.timestamps[env.index].month)
        if env.timestamps is not None
        else env.index % 12 + 1
    )
    return np.array(
        [
            float(month),
            env.reservoir_1.level_from_storage(float(env.storages_1e9m3[0])),
            env.reservoir_2.level_from_storage(float(env.storages_1e9m3[1])),
            float(inflow_1),
            float(inflow_2_local),
        ],
        dtype=float,
    )


class CascadeDQNAdapter:
    """Thin DQN view over :class:`CascadeReservoirEnv`.

    For a 240-transition training episode, give the environment a 241-month
    inflow window and set ``max_transitions=240``.  The final returned state is
    then the real month-241 observation and the boundary is reported only as a
    truncation, so Double DQN may bootstrap through it.
    """

    def __init__(
        self,
        env: CascadeReservoirEnv,
        normalizer: StateNormalizer,
        action_grid: JointActionGrid,
        *,
        reward_reference_days: float = 30.4,
        reward_clip: Sequence[float] | None = None,
        max_transitions: int | None = None,
        target_reached_atol_m: float = TARGET_REACHED_ATOL_M,
    ) -> None:
        if not isinstance(env, CascadeReservoirEnv):
            raise CascadeSpecError("CascadeDQNAdapter requires a CascadeReservoirEnv.")
        if not isinstance(normalizer, StateNormalizer):
            raise CascadeSpecError("normalizer must be a StateNormalizer.")
        if not isinstance(action_grid, JointActionGrid):
            raise CascadeSpecError("action_grid must be a JointActionGrid.")
        normalizer.validate_for_specs(env.reservoir_1, env.reservoir_2)
        expected_grid = JointActionGrid(env.reservoir_1, env.reservoir_2, action_grid.step_m)
        if not np.array_equal(action_grid.xiaowan_levels_m, expected_grid.xiaowan_levels_m) or not np.array_equal(
            action_grid.nuozhadu_levels_m, expected_grid.nuozhadu_levels_m
        ):
            raise CascadeSpecError("DQN action grid does not match the environment reservoir specs.")
        if not (
            env.reservoir_1.energy_parameters_available
            and env.reservoir_2.energy_parameters_available
        ):
            raise CascadeSpecError("DQN V1 requires complete energy parameters for both reservoirs.")
        if isinstance(reward_reference_days, bool) or not np.isfinite(reward_reference_days) or reward_reference_days <= 0.0:
            raise CascadeSpecError("reward_reference_days must be finite and positive.")
        max_powers = (env.reservoir_1.max_power_kw, env.reservoir_2.max_power_kw)
        if any(value is None or not np.isfinite(value) or value <= 0.0 for value in max_powers):
            raise CascadeSpecError("Reward scaling requires positive max_power_kw for both reservoirs.")
        self.reward_reference_days = float(reward_reference_days)
        self.reward_scale_gwh = (
            (float(max_powers[0]) + float(max_powers[1]))
            * self.reward_reference_days
            * 24.0
            / 1.0e6
        )
        self.reward_clip = self._validate_reward_clip(reward_clip)
        if max_transitions is None:
            self.max_transitions = None
        else:
            self.max_transitions = _positive_integer(max_transitions, "max_transitions")
            if self.max_transitions >= len(env.local_inflows_m3s):
                raise CascadeSpecError(
                    "A limited training run needs at least max_transitions + 1 inflow months "
                    "for a real bootstrap observation."
                )
        if isinstance(target_reached_atol_m, bool) or not np.isfinite(target_reached_atol_m) or target_reached_atol_m < 0.0:
            raise CascadeSpecError("target_reached_atol_m must be finite and nonnegative.")
        self.target_reached_atol_m = float(target_reached_atol_m)
        self.env = env
        self.normalizer = normalizer
        self.action_grid = action_grid
        self._steps = 0
        self._ready = False
        self._current_raw_state: np.ndarray | None = None
        self._target_unreached = np.zeros(2, dtype=np.int64)
        self._any_target_unreached = 0
        self._action_mask_cache_key: tuple[float | int, ...] | None = None
        self._action_mask_cache: np.ndarray | None = None

    @staticmethod
    def _validate_reward_clip(reward_clip: Sequence[float] | None) -> tuple[float, float] | None:
        if reward_clip is None:
            return None
        values = np.asarray(reward_clip, dtype=float)
        if values.shape != (2,) or np.any(~np.isfinite(values)) or values[1] <= values[0]:
            raise CascadeSpecError("reward_clip must be null or two increasing finite values.")
        return float(values[0]), float(values[1])

    @property
    def current_raw_state(self) -> np.ndarray:
        if not self._ready or self._current_raw_state is None:
            raise CascadeSpecError("Reset the DQN adapter before requesting its current state.")
        return self._current_raw_state.copy()

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
                "xiaowan": float(self._target_unreached[0] / denominator) if self._steps else 0.0,
                "nuozhadu": float(self._target_unreached[1] / denominator) if self._steps else 0.0,
                "either_reservoir": float(self._any_target_unreached / denominator) if self._steps else 0.0,
            },
        }

    @property
    def reward_metadata(self) -> dict[str, Any]:
        return {
            "objective": "generation_gwh_minus_penalty_gwh_per_failed_station_month",
            "firm_power_failure_penalty_gwh": (
                self.env.firm_power_failure_penalty_gwh
            ),
            "failure_definition": "station_specific_monthly_power_below_firm_power",
            "deficit_energy_role": "diagnostic_only_not_in_objective",
            "reference_days": self.reward_reference_days,
            "scale_gwh": self.reward_scale_gwh,
            "clip": list(self.reward_clip) if self.reward_clip is not None else None,
        }

    def get_feasible_action_mask(self) -> np.ndarray:
        """Return the shared environment's mask in stable joint-action order."""
        if not self._ready or self._current_raw_state is None:
            raise CascadeSpecError(
                "Reset the DQN adapter before requesting its action mask."
            )
        inflow_1, inflow_2_local = self.env.local_inflows_m3s[self.env.index]
        cache_key = (
            int(self.env.index),
            float(self.env._month()),
            float(self.env.storages_1e9m3[0]),
            float(self.env.storages_1e9m3[1]),
            float(inflow_1),
            float(inflow_2_local),
            float(self.env._seconds_this_step()),
        )
        if (
            self._action_mask_cache_key == cache_key
            and self._action_mask_cache is not None
        ):
            return self._action_mask_cache.copy()
        mask = self.env.get_feasible_action_mask(
            self.action_grid.xiaowan_levels_m,
            self.action_grid.nuozhadu_levels_m,
            target_reached_atol_m=self.target_reached_atol_m,
        )
        if mask.shape != (self.action_grid.action_count,) or mask.dtype != np.bool_:
            raise CascadeSpecError(
                "Cascade environment returned an incompatible feasible-action mask."
            )
        self._action_mask_cache_key = cache_key
        self._action_mask_cache = mask.copy()
        return mask.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._ready = False
        self._current_raw_state = None
        self._action_mask_cache_key = None
        self._action_mask_cache = None
        _, env_info = self.env.reset(seed=seed, options=options)
        if self.max_transitions is not None and self.env.index + self.max_transitions >= len(
            self.env.local_inflows_m3s
        ):
            raise CascadeSpecError(
                "This reset position does not leave max_transitions + 1 real observations."
            )
        raw_state = raw_state_from_env(self.env)
        network_state = self.normalizer.transform(raw_state)
        self._steps = 0
        self._target_unreached[:] = 0
        self._any_target_unreached = 0
        self._current_raw_state = raw_state
        self._ready = True
        info = dict(env_info)
        info.update(
            {
                "raw_state": raw_state.copy(),
                "normalized_state": network_state.copy(),
                "state_order": list(STATE_ORDER),
            }
        )
        return network_state, info

    def step(
        self, action_index: int
    ) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        if not self._ready or self._current_raw_state is None:
            raise CascadeSpecError("Reset the DQN adapter before stepping it.")
        if self.max_transitions is not None and self._steps >= self.max_transitions:
            raise CascadeSpecError("The configured training transition limit has been reached.")
        state_raw = self._current_raw_state.copy()
        upstream_index, downstream_index = self.action_grid.decode(action_index)
        feasible_mask = self.get_feasible_action_mask()
        if not bool(feasible_mask[int(action_index)]):
            raise CascadeSpecError(
                f"DQN/PPO attempted physically infeasible action index {action_index}."
            )
        targets = self.action_grid.action(action_index)
        _, raw_reward, terminated, env_truncated, env_info = self.env.step(targets)
        self._action_mask_cache_key = None
        self._action_mask_cache = None
        if not np.isfinite(raw_reward):
            raise CascadeSpecError("Cascade environment returned a non-finite physical reward.")
        info = dict(env_info)
        try:
            upstream = info["reservoir_1"]
            downstream = info["reservoir_2"]
            actual_levels = (
                float(upstream["end_level_m"]),
                float(downstream["end_level_m"]),
            )
            reported_targets = (
                float(upstream["requested_target_level_m"]),
                float(downstream["requested_target_level_m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CascadeSpecError("Cascade environment transition diagnostics are incomplete.") from exc
        if not np.allclose(reported_targets, targets, rtol=0.0, atol=0.0):
            raise CascadeSpecError("Cascade environment reported different requested target levels.")
        reached = tuple(
            bool(np.isclose(actual, requested, rtol=0.0, atol=self.target_reached_atol_m))
            for actual, requested in zip(actual_levels, targets)
        )
        self._steps += 1
        self._target_unreached += np.logical_not(reached).astype(np.int64)
        self._any_target_unreached += int(not all(reached))

        scaled_unclipped = float(raw_reward) / self.reward_scale_gwh
        scaled_reward = (
            float(np.clip(scaled_unclipped, self.reward_clip[0], self.reward_clip[1]))
            if self.reward_clip is not None
            else scaled_unclipped
        )
        artificial_truncation = bool(
            self.max_transitions is not None and self._steps == self.max_transitions
        )
        if artificial_truncation and terminated:
            raise CascadeSpecError(
                "Training reached the environment data end; provide one extra inflow month "
                "for the bootstrap observation."
            )
        truncated = bool(env_truncated or artificial_truncation)
        if terminated:
            next_raw_state = None
            next_network_state = None
            self._current_raw_state = None
            self._ready = False
        else:
            next_raw_state = raw_state_from_env(self.env)
            next_network_state = self.normalizer.transform(next_raw_state)
            self._current_raw_state = next_raw_state

        info.update(
            {
                "action_index": int(action_index),
                "feasible_action_count": int(feasible_mask.sum()),
                "selected_action_feasible": True,
                "action_mask_mode": "environment_transition_reachability_and_monthly_limits_v2",
                "xiaowan_action_grid_index": upstream_index,
                "nuozhadu_action_grid_index": downstream_index,
                "requested_target_level_xiaowan_m": targets[0],
                "requested_target_level_nuozhadu_m": targets[1],
                "actual_end_level_xiaowan_m": actual_levels[0],
                "actual_end_level_nuozhadu_m": actual_levels[1],
                "xiaowan_target_reached": reached[0],
                "nuozhadu_target_reached": reached[1],
                "xiaowan_target_error_m": actual_levels[0] - targets[0],
                "nuozhadu_target_error_m": actual_levels[1] - targets[1],
                "raw_reward_gwh": float(raw_reward),
                "generation_energy_gwh": float(info["generation_energy_gwh"]),
                "penalized_reward_gwh": float(info["penalized_reward_gwh"]),
                "failure_penalty_gwh": float(info["failure_penalty_gwh"]),
                "failed_station_months": int(info["failed_station_months"]),
                "xiaowan_firm_power_failure": bool(
                    info["xiaowan_firm_power_failure"]
                ),
                "nuozhadu_firm_power_failure": bool(
                    info["nuozhadu_firm_power_failure"]
                ),
                "normalized_reward_unclipped": scaled_unclipped,
                "normalized_reward": scaled_reward,
                "reward_scale_gwh": self.reward_scale_gwh,
                "state_raw": state_raw,
                "next_state_raw": None if next_raw_state is None else next_raw_state.copy(),
                "next_state_normalized": (
                    None if next_network_state is None else next_network_state.copy()
                ),
                "artificial_window_truncation": artificial_truncation,
                "bootstrap_allowed": not bool(terminated),
                "episode_diagnostics": self.episode_diagnostics,
            }
        )
        return next_network_state, scaled_reward, bool(terminated), truncated, info


def valid_january_start_indices(
    frame: pd.DataFrame, *, episode_months: int = 240
) -> np.ndarray:
    """Enumerate January starts with transitions plus one bootstrap month."""
    months = _positive_integer(episode_months, "episode_months")
    periods, _ = _validated_inflow_frame(frame)
    candidates = np.flatnonzero(periods.month == 1).astype(np.int64)
    return candidates[candidates + months < len(frame)]


def sample_january_start_index(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    episode_months: int = 240,
) -> int:
    """Uniformly sample one valid January using the caller's seeded generator."""
    if not isinstance(rng, np.random.Generator):
        raise CascadeSpecError("rng must be a numpy.random.Generator.")
    candidates = valid_january_start_indices(frame, episode_months=episode_months)
    if len(candidates) == 0:
        raise CascadeSpecError(
            "The inflow frame has no January with a complete episode and bootstrap month."
        )
    return int(rng.choice(candidates))


def slice_episode_window(
    frame: pd.DataFrame,
    start_index: int,
    *,
    episode_months: int = 240,
) -> pd.DataFrame:
    """Return ``episode_months + 1`` rows for valid nonterminal bootstrap."""
    months = _positive_integer(episode_months, "episode_months")
    candidates = valid_january_start_indices(frame, episode_months=months)
    if isinstance(start_index, bool) or not isinstance(start_index, (int, np.integer)):
        raise CascadeSpecError("start_index must be an integer.")
    start = int(start_index)
    if not np.any(candidates == start):
        raise CascadeSpecError(
            "start_index must be a January with a complete episode and bootstrap month."
        )
    return frame.iloc[start : start + months + 1].reset_index(drop=True).copy()


def normalized_inflow_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical monthly external-inflow content, independent of CSV formatting."""
    periods, flows = _validated_inflow_frame(frame)
    digest = hashlib.sha256()
    for period, (upstream, local) in zip(periods, flows):
        digest.update(
            f"{period},{float(upstream).hex()},{float(local).hex()}\n".encode("utf-8")
        )
    return digest.hexdigest()


__all__ = [
    "ACTION_ENCODING",
    "CascadeDQNAdapter",
    "INFLOW_COLUMNS",
    "JointActionGrid",
    "STATE_ORDER",
    "StateNormalizer",
    "normalized_inflow_sha256",
    "raw_state_from_env",
    "raw_state_from_env_observation",
    "sample_january_start_index",
    "slice_episode_window",
    "valid_january_start_indices",
]

