"""Joint deterministic dynamic programming as the P4 perfect-foresight bound."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .environment import CascadeSpecError, ReservoirSpec, _as_month_period_index, _transition


_CHECKPOINT_VERSION = 3
_TARGET_STORAGE_ATOL_1E9M3 = 1.0e-10
_LEVEL_ATOL_M = 1.0e-8
_CONSTRAINT_FINGERPRINT = (
    b"monthly_max_hard_terminal_and_firm_power_failure_count_penalty_v1"
)


def make_level_grid(dead_level_m: float, normal_level_m: float, step_m: float) -> np.ndarray:
    """Build a grid that always includes both physical bounds."""
    if step_m <= 0.0:
        raise CascadeSpecError("DDP grid step must be positive.")
    levels = np.arange(dead_level_m, normal_level_m + 1.0e-9, step_m, dtype=float)
    if not np.isclose(levels[-1], normal_level_m):
        levels = np.append(levels, normal_level_m)
    return levels


class CascadeDDP:
    """Finite-horizon two-reservoir DDP with complete realized inflow foresight.

    This class is expressly an offline upper bound. Its action semantics and
    physical transition are shared with the P4 environment, but it is not a
    deployable controller and must not be compared as one.
    """

    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        local_inflows_m3s: Sequence[Sequence[float]],
        timestamps: Sequence[object],
        grid_step_m: float = 0.5,
        firm_power_failure_penalty_gwh: float = 0.0,
        monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
    ) -> None:
        inflows = np.asarray(local_inflows_m3s, dtype=float)
        if inflows.ndim != 2 or inflows.shape[1] != 2:
            raise CascadeSpecError("DDP local inflows must have shape [months, 2].")
        if len(timestamps) != len(inflows):
            raise CascadeSpecError("DDP timestamps must align with inflows.")
        if not (reservoir_1.energy_parameters_available and reservoir_2.energy_parameters_available):
            raise CascadeSpecError("DDP requires complete P4 V1 energy parameters.")
        self.reservoir_1 = reservoir_1
        self.reservoir_2 = reservoir_2
        self.inflows = inflows
        self.timestamps = _as_month_period_index(timestamps)
        self.levels_1 = make_level_grid(reservoir_1.dead_level_m, reservoir_1.normal_level_m, grid_step_m)
        self.levels_2 = make_level_grid(reservoir_2.dead_level_m, reservoir_2.normal_level_m, grid_step_m)
        self.storages_1 = np.array([reservoir_1.storage_from_level(level) for level in self.levels_1])
        self.storages_2 = np.array([reservoir_2.storage_from_level(level) for level in self.levels_2])
        self.grid_step_m = float(grid_step_m)
        self.firm_power_failure_penalty_gwh = float(
            firm_power_failure_penalty_gwh
        )
        if (
            not np.isfinite(self.firm_power_failure_penalty_gwh)
            or self.firm_power_failure_penalty_gwh < 0.0
        ):
            raise CascadeSpecError(
                "DDP firm_power_failure_penalty_gwh must be a finite "
                "nonnegative value."
            )

        if monthly_max_target_levels_m is None:
            monthly_max = np.column_stack(
                (
                    np.full(12, reservoir_1.normal_level_m, dtype=float),
                    np.full(12, reservoir_2.normal_level_m, dtype=float),
                )
            )
        else:
            monthly_max = np.asarray(monthly_max_target_levels_m, dtype=float)
        if monthly_max.shape != (12, 2) or np.any(~np.isfinite(monthly_max)):
            raise CascadeSpecError(
                "DDP monthly_max_target_levels_m must be a finite [12, 2] table."
            )
        if np.any(
            (monthly_max[:, 0] < reservoir_1.dead_level_m)
            | (monthly_max[:, 0] > reservoir_1.normal_level_m)
            | (monthly_max[:, 1] < reservoir_2.dead_level_m)
            | (monthly_max[:, 1] > reservoir_2.normal_level_m)
        ):
            raise CascadeSpecError(
                "DDP monthly maximum target levels must lie within physical level bounds."
            )
        self.monthly_max_target_levels_m = monthly_max
        self.monthly_action_counts_1 = np.searchsorted(
            self.levels_1,
            monthly_max[:, 0] + _LEVEL_ATOL_M,
            side="right",
        ).astype(int)
        self.monthly_action_counts_2 = np.searchsorted(
            self.levels_2,
            monthly_max[:, 1] + _LEVEL_ATOL_M,
            side="right",
        ).astype(int)
        if np.any(self.monthly_action_counts_1 <= 0) or np.any(
            self.monthly_action_counts_2 <= 0
        ):
            raise CascadeSpecError("DDP monthly operating limits remove every action.")

    def _validated_level_pair(
        self,
        levels_m: Sequence[float],
        *,
        label: str,
    ) -> tuple[float, float]:
        levels = np.asarray(levels_m, dtype=float)
        if levels.shape != (2,) or np.any(~np.isfinite(levels)):
            raise CascadeSpecError(f"{label} must contain two finite values.")
        if not (
            self.reservoir_1.dead_level_m
            <= levels[0]
            <= self.reservoir_1.normal_level_m
            and self.reservoir_2.dead_level_m
            <= levels[1]
            <= self.reservoir_2.normal_level_m
        ):
            raise CascadeSpecError(f"{label} must lie within both reservoir bounds.")
        return float(levels[0]), float(levels[1])

    def _prepare_terminal_constraint(
        self,
        terminal_levels_m: Sequence[float] | None,
    ) -> tuple[tuple[float, float], tuple[int, int]] | None:
        if terminal_levels_m is None:
            return None
        levels = self._validated_level_pair(
            terminal_levels_m,
            label="terminal_levels_m",
        )
        state_1 = int(np.argmin(np.abs(self.levels_1 - levels[0])))
        state_2 = int(np.argmin(np.abs(self.levels_2 - levels[1])))
        if not (
            np.isclose(self.levels_1[state_1], levels[0], rtol=0.0, atol=_LEVEL_ATOL_M)
            and np.isclose(
                self.levels_2[state_2], levels[1], rtol=0.0, atol=_LEVEL_ATOL_M
            )
        ):
            raise CascadeSpecError(
                "Strict DDP terminal levels must be present on both water-level grids."
            )
        return levels, (state_1, state_2)

    @staticmethod
    def _initialize_terminal_value(
        terminal_value: np.ndarray,
        terminal_constraint: tuple[tuple[float, float], tuple[int, int]] | None,
    ) -> None:
        if terminal_constraint is None:
            terminal_value[:, :] = 0.0
            return
        terminal_value[:, :] = -np.inf
        state_1, state_2 = terminal_constraint[1]
        terminal_value[state_1, state_2] = 0.0

    @staticmethod
    def _target_is_reached(end_storage_1e9m3: float, target_storage_1e9m3: float) -> bool:
        return bool(
            np.isclose(
                end_storage_1e9m3,
                target_storage_1e9m3,
                rtol=0.0,
                atol=_TARGET_STORAGE_ATOL_1E9M3,
            )
        )

    def _reward_breakdown(
        self,
        upstream: object,
        downstream: object,
    ) -> dict[str, float | int | bool]:
        """Return generation, station failures, diagnostics, and DDP reward."""
        generation_energy = float(upstream.energy_gwh) + float(downstream.energy_gwh)
        xiaowan_deficit = float(upstream.firm_deficit_gwh)
        nuozhadu_deficit = float(downstream.firm_deficit_gwh)
        xiaowan_failure = bool(xiaowan_deficit > 0.0)
        nuozhadu_failure = bool(nuozhadu_deficit > 0.0)
        failed_station_months = int(xiaowan_failure) + int(nuozhadu_failure)
        failure_penalty = self.firm_power_failure_penalty_gwh * (
            failed_station_months
        )
        return {
            "generation_energy_gwh": generation_energy,
            "xiaowan_deficit_energy_gwh": xiaowan_deficit,
            "nuozhadu_deficit_energy_gwh": nuozhadu_deficit,
            "xiaowan_firm_power_failure": xiaowan_failure,
            "nuozhadu_firm_power_failure": nuozhadu_failure,
            "failed_station_months": failed_station_months,
            "failure_penalty_gwh": failure_penalty,
            # Compatibility alias for output-processing code that expects a
            # generic penalty column. In V6 this is a failure-count penalty,
            # not a deficit-energy penalty.
            "penalty_energy_gwh": failure_penalty,
            "penalized_reward_gwh": generation_energy - failure_penalty,
        }

    def _monthly_action_limits(self, t: int) -> tuple[int, int, float, float]:
        """Return action counts and maxima for the current end-of-month target.

        State ``t`` is the current month-start state.  The action is the current
        month-end target (and next month-start level), so May uses the flood
        limit needed at the beginning of June.
        """
        month_index = int(self.timestamps[t].month) - 1
        return (
            int(self.monthly_action_counts_1[month_index]),
            int(self.monthly_action_counts_2[month_index]),
            float(self.monthly_max_target_levels_m[month_index, 0]),
            float(self.monthly_max_target_levels_m[month_index, 1]),
        )

    def _checkpoint_fingerprint(
        self,
        initial_levels_m: tuple[float, float],
        terminal_levels_m: tuple[float, float] | None = None,
    ) -> str:
        """Identify data, grids, physics, operating limits, and boundaries."""
        digest = hashlib.sha256()
        terminal_array = (
            np.asarray(terminal_levels_m, dtype=np.float64)
            if terminal_levels_m is not None
            else np.asarray([np.nan, np.nan], dtype=np.float64)
        )
        for array in (
            self.inflows,
            self.levels_1,
            self.levels_2,
            self.storages_1,
            self.storages_2,
            self.monthly_max_target_levels_m,
            np.asarray(initial_levels_m, dtype=np.float64),
            terminal_array,
            np.asarray(
                [self.grid_step_m, self.firm_power_failure_penalty_gwh],
                dtype=np.float64,
            ),
        ):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        digest.update(repr(self.reservoir_1).encode("utf-8"))
        digest.update(repr(self.reservoir_2).encode("utf-8"))
        digest.update("|".join(str(period) for period in self.timestamps).encode("utf-8"))
        digest.update(_CONSTRAINT_FINGERPRINT)
        digest.update(str(_CHECKPOINT_VERSION).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _write_checkpoint(
        checkpoint_path: Path,
        *,
        fingerprint: str,
        months: int,
        next_t: int,
        value_next: np.ndarray,
        policy_start: int,
        policies_1_suffix: np.ndarray,
        policies_2_suffix: np.ndarray,
    ) -> None:
        """Atomically replace the checkpoint so a partial write is never resumed."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                checkpoint_version=np.asarray(_CHECKPOINT_VERSION, dtype=np.int64),
                fingerprint=np.asarray(fingerprint),
                months=np.asarray(months, dtype=np.int64),
                completed_months=np.asarray(months - policy_start, dtype=np.int64),
                next_t=np.asarray(next_t, dtype=np.int64),
                value_next=np.asarray(value_next, dtype=float),
                policy_start=np.asarray(policy_start, dtype=np.int64),
                policies_1_suffix=np.asarray(policies_1_suffix, dtype=np.int16),
                policies_2_suffix=np.asarray(policies_2_suffix, dtype=np.int16),
            )
        os.replace(temporary_path, checkpoint_path)

    def _restore_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        fingerprint: str,
        months: int,
        value: np.ndarray,
        policies_1: np.ndarray,
        policies_2: np.ndarray,
    ) -> tuple[int, int]:
        """Restore the solved suffix and return the next month index to solve."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DDP checkpoint not found: {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            if int(saved["checkpoint_version"].item()) != _CHECKPOINT_VERSION:
                raise CascadeSpecError("Unsupported DDP checkpoint version.")
            if str(saved["fingerprint"].item()) != fingerprint:
                raise CascadeSpecError(
                    "DDP checkpoint does not match current inflow, grid, operating limits, or boundary levels."
                )
            if int(saved["months"].item()) != months:
                raise CascadeSpecError("DDP checkpoint horizon does not match the current run.")
            next_t = int(saved["next_t"].item())
            policy_start = int(saved["policy_start"].item())
            if next_t < -1 or next_t >= months or policy_start != next_t + 1:
                raise CascadeSpecError("DDP checkpoint contains an invalid recursion position.")
            expected_value_shape = (len(self.levels_1), len(self.levels_2))
            if saved["value_next"].shape != expected_value_shape:
                raise CascadeSpecError("DDP checkpoint value-grid shape does not match the current grid.")
            expected_policy_shape = (
                months - policy_start,
                len(self.levels_1),
                len(self.levels_2),
            )
            if (
                saved["policies_1_suffix"].shape != expected_policy_shape
                or saved["policies_2_suffix"].shape != expected_policy_shape
            ):
                raise CascadeSpecError("DDP checkpoint policy shape does not match the current grid.")
            value[policy_start, :, :] = saved["value_next"]
            policies_1[policy_start:, :, :] = saved["policies_1_suffix"]
            policies_2[policy_start:, :, :] = saved["policies_2_suffix"]
            completed_months = int(saved["completed_months"].item())
        return next_t, completed_months

    def _transition_value(self, t: int, state_1: int, state_2: int, action_1: int, action_2: int) -> tuple[float, object, object]:
        seconds = float(self.timestamps[t].days_in_month) * 86400.0
        inflow_1, inflow_2_local = self.inflows[t]
        upstream = _transition(self.reservoir_1, self.storages_1[state_1], float(inflow_1), self.levels_1[action_1], seconds)
        downstream = _transition(self.reservoir_2, self.storages_2[state_2], float(inflow_2_local + upstream.total_discharge_m3s), self.levels_2[action_2], seconds)
        immediate = self._reward_breakdown(upstream, downstream)[
            "penalized_reward_gwh"
        ]
        return immediate, upstream, downstream

    def solve(
        self,
        initial_levels_m: tuple[float, float],
        progress_every_months: int | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every_months: int | None = None,
        resume: bool = False,
        *,
        terminal_levels_m: tuple[float, float] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        initial_levels_m = self._validated_level_pair(
            initial_levels_m,
            label="initial_levels_m",
        )
        terminal_constraint = self._prepare_terminal_constraint(terminal_levels_m)
        normalized_terminal_levels = (
            terminal_constraint[0] if terminal_constraint is not None else None
        )
        months = len(self.inflows)
        value = np.full((months + 1, len(self.levels_1), len(self.levels_2)), -np.inf, dtype=float)
        self._initialize_terminal_value(value[-1], terminal_constraint)
        policies_1 = np.zeros((months, len(self.levels_1), len(self.levels_2)), dtype=np.int16)
        policies_2 = np.zeros_like(policies_1)
        checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint_file is not None and (
            checkpoint_every_months is None or checkpoint_every_months <= 0
        ):
            raise CascadeSpecError("DDP checkpoint interval must be positive.")
        if resume and checkpoint_file is None:
            raise CascadeSpecError("DDP resume requires a checkpoint path.")

        fingerprint = self._checkpoint_fingerprint(
            initial_levels_m,
            normalized_terminal_levels,
        )
        resumed_completed_months = 0
        if resume:
            start_t, resumed_completed_months = self._restore_checkpoint(
                checkpoint_file,
                fingerprint=fingerprint,
                months=months,
                value=value,
                policies_1=policies_1,
                policies_2=policies_2,
            )
            print(
                f"Resumed DDP checkpoint at {resumed_completed_months}/{months} solved months: "
                f"{checkpoint_file}",
                flush=True,
            )
        else:
            start_t = months - 1
            if checkpoint_file is not None:
                self._write_checkpoint(
                    checkpoint_file,
                    fingerprint=fingerprint,
                    months=months,
                    next_t=start_t,
                    value_next=value[-1],
                    policy_start=months,
                    policies_1_suffix=policies_1[months:],
                    policies_2_suffix=policies_2[months:],
                )
                print(f"Initialized DDP checkpoint: {checkpoint_file}", flush=True)

        for t in range(start_t, -1, -1):
            action_count_1, action_count_2, _, _ = self._monthly_action_limits(t)
            for state_1 in range(len(self.levels_1)):
                for state_2 in range(len(self.levels_2)):
                    best_value = -np.inf
                    best_action = (state_1, state_2)
                    for action_1 in range(action_count_1):
                        for action_2 in range(action_count_2):
                            immediate, upstream, downstream = self._transition_value(t, state_1, state_2, action_1, action_2)
                            if not self._target_is_reached(
                                upstream.end_storage_1e9m3,
                                self.storages_1[action_1],
                            ) or not self._target_is_reached(
                                downstream.end_storage_1e9m3,
                                self.storages_2[action_2],
                            ):
                                continue
                            candidate = immediate + value[t + 1, action_1, action_2]
                            if candidate > best_value:
                                best_value = candidate
                                best_action = (action_1, action_2)
                    value[t, state_1, state_2] = best_value
                    policies_1[t, state_1, state_2] = best_action[0]
                    policies_2[t, state_1, state_2] = best_action[1]
            completed_months = months - t
            if progress_every_months and (
                completed_months % progress_every_months == 0 or t == 0
            ):
                print(
                    f"DDP backward recursion solved {completed_months}/{months} months.",
                    flush=True,
                )
            if checkpoint_file is not None and (
                completed_months % int(checkpoint_every_months) == 0 or t == 0
            ):
                self._write_checkpoint(
                    checkpoint_file,
                    fingerprint=fingerprint,
                    months=months,
                    next_t=t - 1,
                    value_next=value[t],
                    policy_start=t,
                    policies_1_suffix=policies_1[t:],
                    policies_2_suffix=policies_2[t:],
                )
                print(
                    f"Saved DDP checkpoint after {completed_months}/{months} months: "
                    f"{checkpoint_file}",
                    flush=True,
                )

        state_1 = int(np.argmin(np.abs(self.levels_1 - initial_levels_m[0])))
        state_2 = int(np.argmin(np.abs(self.levels_2 - initial_levels_m[1])))
        if not np.isfinite(value[0, state_1, state_2]):
            raise CascadeSpecError(
                "DDP found no feasible trajectory from the configured initial levels "
                "through the monthly operating limits to the strict terminal levels."
            )
        rows: list[dict] = []
        previous_end_storage_1: float | None = None
        previous_end_storage_2: float | None = None
        for t in range(months):
            _, _, monthly_max_1, monthly_max_2 = self._monthly_action_limits(t)
            action_1 = int(policies_1[t, state_1, state_2])
            action_2 = int(policies_2[t, state_1, state_2])
            immediate, upstream, downstream = self._transition_value(t, state_1, state_2, action_1, action_2)
            reward = self._reward_breakdown(upstream, downstream)
            start_storage_1 = float(self.storages_1[state_1])
            start_storage_2 = float(self.storages_2[state_2])
            storage_jump_1_m3 = (
                0.0
                if previous_end_storage_1 is None
                else (start_storage_1 - previous_end_storage_1) * 1.0e9
            )
            storage_jump_2_m3 = (
                0.0
                if previous_end_storage_2 is None
                else (start_storage_2 - previous_end_storage_2) * 1.0e9
            )
            rows.append(
                {
                    "date": f"{self.timestamps[t].year:04d}-{self.timestamps[t].month:02d}-01",
                    "start_level_1_m": self.levels_1[state_1],
                    "start_level_2_m": self.levels_2[state_2],
                    "target_level_1_m": self.levels_1[action_1],
                    "target_level_2_m": self.levels_2[action_2],
                    "monthly_max_level_1_m": monthly_max_1,
                    "monthly_max_level_2_m": monthly_max_2,
                    "at_monthly_max_level_1": bool(
                        np.isclose(
                            upstream.end_level_m,
                            monthly_max_1,
                            rtol=0.0,
                            atol=_LEVEL_ATOL_M,
                        )
                    ),
                    "at_monthly_max_level_2": bool(
                        np.isclose(
                            downstream.end_level_m,
                            monthly_max_2,
                            rtol=0.0,
                            atol=_LEVEL_ATOL_M,
                        )
                    ),
                    "is_flood_limit_action_month": bool(
                        monthly_max_1 < self.reservoir_1.normal_level_m
                        or monthly_max_2 < self.reservoir_2.normal_level_m
                    ),
                    "storage_jump_1_m3": storage_jump_1_m3,
                    "storage_jump_2_m3": storage_jump_2_m3,
                    # Preserve reward_gwh as the actual cascade generation used by
                    # existing output consumers; the Bellman objective is exposed
                    # separately as penalized_reward_gwh.
                    "reward_gwh": reward["generation_energy_gwh"],
                    **reward,
                    "inflow_1_m3s": float(self.inflows[t, 0]),
                    "inflow_2_local_m3s": float(self.inflows[t, 1]),
                    "routed_inflow_m3s": upstream.total_discharge_m3s,
                    "reservoir_1": asdict(upstream),
                    "reservoir_2": asdict(downstream),
                }
            )
            previous_end_storage_1 = upstream.end_storage_1e9m3
            previous_end_storage_2 = downstream.end_storage_1e9m3
            state_1 = action_1
            state_2 = action_2

        schedule = pd.DataFrame(rows)
        final_levels = (
            float(schedule.iloc[-1]["reservoir_1"]["end_level_m"]),
            float(schedule.iloc[-1]["reservoir_2"]["end_level_m"]),
        )
        terminal_errors = (
            (
                final_levels[0] - normalized_terminal_levels[0],
                final_levels[1] - normalized_terminal_levels[1],
            )
            if normalized_terminal_levels is not None
            else (None, None)
        )
        if normalized_terminal_levels is not None and not (
            np.isclose(
                final_levels[0],
                normalized_terminal_levels[0],
                rtol=0.0,
                atol=_LEVEL_ATOL_M,
            )
            and np.isclose(
                final_levels[1],
                normalized_terminal_levels[1],
                rtol=0.0,
                atol=_LEVEL_ATOL_M,
            )
        ):
            raise CascadeSpecError(
                "DDP forward reconstruction violated the strict terminal constraint."
            )
        summary = {
            "method": "DDP",
            "information_set": "complete_realized_future_inflow_offline_upper_bound_only",
            "months": months,
            "grid_step_m": self.grid_step_m,
            "state_grid_shape": [len(self.levels_1), len(self.levels_2)],
            "joint_action_count": len(self.levels_1) * len(self.levels_2),
            "objective_gwh": float(schedule.penalized_reward_gwh.sum()),
            "total_generation_gwh": float(schedule.generation_energy_gwh.sum()),
            "total_xiaowan_deficit_energy_gwh": float(
                schedule.xiaowan_deficit_energy_gwh.sum()
            ),
            "total_nuozhadu_deficit_energy_gwh": float(
                schedule.nuozhadu_deficit_energy_gwh.sum()
            ),
            "total_penalty_energy_gwh": float(schedule.penalty_energy_gwh.sum()),
            "total_failed_station_months": int(schedule.failed_station_months.sum()),
            "xiaowan_failed_months": int(
                schedule.xiaowan_firm_power_failure.sum()
            ),
            "nuozhadu_failed_months": int(
                schedule.nuozhadu_firm_power_failure.sum()
            ),
            "total_failure_penalty_gwh": float(
                schedule.failure_penalty_gwh.sum()
            ),
            "firm_power_failure_penalty_gwh": (
                self.firm_power_failure_penalty_gwh
            ),
            "objective_definition": (
                "generation_energy_gwh_minus_gwh_equivalent_penalty_per_"
                "failed_station_month"
            ),
            "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
            "checkpoint_every_months": checkpoint_every_months,
            "resumed_from_checkpoint": bool(resume),
            "resumed_completed_months": resumed_completed_months,
            "initial_levels_m": list(initial_levels_m),
            "terminal_levels_m": (
                list(normalized_terminal_levels)
                if normalized_terminal_levels is not None
                else None
            ),
            "terminal_constraint": (
                "hard_final_end_state"
                if normalized_terminal_levels is not None
                else None
            ),
            "final_level_1_m": final_levels[0],
            "final_level_2_m": final_levels[1],
            "terminal_error_1_m": terminal_errors[0],
            "terminal_error_2_m": terminal_errors[1],
            "monthly_max_target_levels_m": {
                "xiaowan": self.monthly_max_target_levels_m[:, 0].tolist(),
                "nuozhadu": self.monthly_max_target_levels_m[:, 1].tolist(),
            },
            "monthly_max_semantics": "applies_to_end_of_month_target_level",
        }
        return schedule, summary


def summarize_ddp_trajectory(
    schedule: pd.DataFrame,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
) -> dict:
    """Build JSON-serializable physical diagnostics from a DDP trajectory."""
    if schedule.empty:
        raise CascadeSpecError("DDP trajectory diagnostics require at least one month.")

    months = np.asarray([int(str(value)[5:7]) for value in schedule["date"]], dtype=int)
    r1 = schedule["reservoir_1"]
    r2 = schedule["reservoir_2"]

    def detail(series: pd.Series, key: str) -> np.ndarray:
        return np.asarray([float(item[key]) for item in series], dtype=float)

    r1_end_level = detail(r1, "end_level_m")
    r2_end_level = detail(r2, "end_level_m")
    r1_start_storage = detail(r1, "start_storage_1e9m3")
    r2_start_storage = detail(r2, "start_storage_1e9m3")
    r1_end_storage = detail(r1, "end_storage_1e9m3")
    r2_end_storage = detail(r2, "end_storage_1e9m3")
    r1_inflow = detail(r1, "inflow_m3s")
    r2_inflow = detail(r2, "inflow_m3s")
    r1_turbine = detail(r1, "turbine_discharge_m3s")
    r2_turbine = detail(r2, "turbine_discharge_m3s")
    r1_spill = detail(r1, "spill_discharge_m3s")
    r2_spill = detail(r2, "spill_discharge_m3s")
    r1_energy = detail(r1, "energy_gwh")
    r2_energy = detail(r2, "energy_gwh")
    r1_water_balance = detail(r1, "water_balance_error_m3")
    r2_water_balance = detail(r2, "water_balance_error_m3")

    target_1 = schedule["target_level_1_m"].to_numpy(dtype=float)
    target_2 = schedule["target_level_2_m"].to_numpy(dtype=float)
    monthly_max_1 = schedule["monthly_max_level_1_m"].to_numpy(dtype=float)
    monthly_max_2 = schedule["monthly_max_level_2_m"].to_numpy(dtype=float)
    flood_action = schedule["is_flood_limit_action_month"].to_numpy(dtype=bool)

    monthly: dict[str, dict[str, float]] = {}
    for month in range(1, 13):
        mask = months == month
        if not np.any(mask):
            continue
        monthly[str(month)] = {
            "xiaowan_mean_end_level_m": float(np.mean(r1_end_level[mask])),
            "nuozhadu_mean_end_level_m": float(np.mean(r2_end_level[mask])),
            "xiaowan_mean_inflow_m3s": float(np.mean(r1_inflow[mask])),
            "nuozhadu_mean_local_inflow_m3s": float(
                np.mean(schedule.loc[mask, "inflow_2_local_m3s"].to_numpy(dtype=float))
            ),
            "nuozhadu_mean_effective_inflow_m3s": float(np.mean(r2_inflow[mask])),
            "xiaowan_mean_turbine_discharge_m3s": float(np.mean(r1_turbine[mask])),
            "nuozhadu_mean_turbine_discharge_m3s": float(np.mean(r2_turbine[mask])),
            "xiaowan_mean_spill_discharge_m3s": float(np.mean(r1_spill[mask])),
            "nuozhadu_mean_spill_discharge_m3s": float(np.mean(r2_spill[mask])),
        }

    equivalent_years = len(schedule) / 12.0
    storage_jump_1_m3 = (r1_start_storage[1:] - r1_end_storage[:-1]) * 1.0e9
    storage_jump_2_m3 = (r2_start_storage[1:] - r2_end_storage[:-1]) * 1.0e9
    turbine_tolerance = 1.0e-6
    level_tolerance = _LEVEL_ATOL_M

    return {
        "mean_annual_generation_gwh": {
            "xiaowan": float(np.sum(r1_energy) / equivalent_years),
            "nuozhadu": float(np.sum(r2_energy) / equivalent_years),
            "cascade": float(np.sum(r1_energy + r2_energy) / equivalent_years),
        },
        "monthly_means": monthly,
        "spill_month_fraction": {
            "xiaowan": float(np.mean(r1_spill > 1.0e-9)),
            "nuozhadu": float(np.mean(r2_spill > 1.0e-9)),
        },
        "maximum_turbine_discharge_month_fraction": {
            "xiaowan": float(
                np.mean(r1_turbine >= float(reservoir_1.max_turbine_discharge_m3s) - turbine_tolerance)
            ),
            "nuozhadu": float(
                np.mean(r2_turbine >= float(reservoir_2.max_turbine_discharge_m3s) - turbine_tolerance)
            ),
        },
        "monthly_max_level_month_fraction": {
            "xiaowan": float(np.mean(np.abs(r1_end_level - monthly_max_1) <= level_tolerance)),
            "nuozhadu": float(np.mean(np.abs(r2_end_level - monthly_max_2) <= level_tolerance)),
        },
        "flood_limit_exceedance_count": {
            "xiaowan": int(np.sum(flood_action & (r1_end_level > monthly_max_1 + level_tolerance))),
            "nuozhadu": int(np.sum(flood_action & (r2_end_level > monthly_max_2 + level_tolerance))),
        },
        "target_unreached_count": {
            "xiaowan": int(np.sum(np.abs(r1_end_level - target_1) > level_tolerance)),
            "nuozhadu": int(np.sum(np.abs(r2_end_level - target_2) > level_tolerance)),
        },
        "max_abs_water_balance_error_m3": float(
            max(np.max(np.abs(r1_water_balance)), np.max(np.abs(r2_water_balance)))
        ),
        "storage_jump": {
            "xiaowan_max_abs_m3": float(
                np.max(np.abs(storage_jump_1_m3)) if storage_jump_1_m3.size else 0.0
            ),
            "nuozhadu_max_abs_m3": float(
                np.max(np.abs(storage_jump_2_m3)) if storage_jump_2_m3.size else 0.0
            ),
            "cascade_max_abs_m3": float(
                np.max(np.abs(storage_jump_1_m3 + storage_jump_2_m3))
                if storage_jump_1_m3.size
                else 0.0
            ),
        },
    }
