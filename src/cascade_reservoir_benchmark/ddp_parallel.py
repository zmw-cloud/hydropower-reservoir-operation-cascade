"""Process-parallel deterministic dynamic programming for the reservoir cascade."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ddp import CascadeDDP
from .environment import CascadeSpecError, ReservoirSpec, _transition


def nearest_storage_state(storage_grid: np.ndarray, storage_1e9m3: float) -> int:
    """Return the nearest storage-grid index, preferring the lower index on ties."""
    if not np.isfinite(storage_1e9m3):
        return int(np.argmin(np.abs(storage_grid - storage_1e9m3)))

    right = int(np.searchsorted(storage_grid, storage_1e9m3, side="left"))
    if right <= 0:
        return 0
    if right >= len(storage_grid):
        return len(storage_grid) - 1

    left = right - 1
    left_distance = np.abs(storage_grid[left] - storage_1e9m3)
    right_distance = np.abs(storage_grid[right] - storage_1e9m3)
    return left if left_distance <= right_distance else right


def _make_state_blocks(
    num_states: int,
    num_workers: int,
    state_block_size: int | None = None,
) -> list[tuple[int, int]]:
    """Partition flattened joint states into contiguous, non-overlapping blocks."""
    if state_block_size is None:
        target_blocks = min(num_states, max(1, num_workers * 4))
        state_block_size = max(1, (num_states + target_blocks - 1) // target_blocks)
    return [
        (start, min(start + state_block_size, num_states))
        for start in range(0, num_states, state_block_size)
    ]


@dataclass(frozen=True)
class _WorkerContext:
    reservoir_2: ReservoirSpec
    storages_1: np.ndarray
    levels_2: np.ndarray
    storages_2: np.ndarray
    firm_power_failure_penalty_gwh: float
    value_next: np.ndarray
    upstream_end_storage: np.ndarray
    upstream_discharge: np.ndarray
    upstream_energy: np.ndarray
    upstream_firm_deficit: np.ndarray


_WORKER_CONTEXT: _WorkerContext | None = None


@dataclass(frozen=True)
class _TransitionBatch:
    """Array-valued transition outputs used by the Bellman hot path."""

    end_storage: np.ndarray
    total_discharge: np.ndarray
    energy: np.ndarray
    firm_deficit: np.ndarray


def _transition_batch(
    spec: ReservoirSpec,
    start_storages: np.ndarray,
    inflow_m3s: float,
    target_levels: np.ndarray,
    seconds: float,
) -> _TransitionBatch:
    """Evaluate every start-storage/target-level pair with NumPy broadcasting.

    This is numerically equivalent to ``environment._transition`` for the four
    quantities needed by the backward recursion, but it avoids constructing one
    ``ReservoirTransition`` object per state-action pair.
    """
    if seconds <= 0.0:
        raise CascadeSpecError("A monthly period must have positive duration.")
    if inflow_m3s < 0.0:
        raise CascadeSpecError(f"{spec.name}: inflow must be nonnegative.")

    start = np.asarray(start_storages, dtype=np.float64).reshape(-1, 1)
    target_level = np.clip(
        np.asarray(target_levels, dtype=np.float64),
        spec.dead_level_m,
        spec.normal_level_m,
    ).reshape(1, -1)
    target_storage = np.interp(
        target_level,
        spec.levels_m,
        spec.storages_1e9m3,
    )
    available_storage = start + float(inflow_m3s) * seconds / 1.0e9
    end_storage = np.minimum(target_storage, available_storage)
    end_storage = np.minimum(end_storage, spec.normal_storage_1e9m3)
    end_storage = np.maximum(end_storage, spec.dead_storage_1e9m3)

    total_discharge = (
        start + float(inflow_m3s) * seconds / 1.0e9 - end_storage
    ) * 1.0e9 / seconds
    if np.any(total_discharge < -1.0e-9):
        raise CascadeSpecError(f"{spec.name}: transition produced a negative release.")
    total_discharge = np.maximum(total_discharge, 0.0)

    if spec.max_total_release_m3s is not None:
        capped = total_discharge > float(spec.max_total_release_m3s)
        if np.any(capped):
            capped_discharge = float(spec.max_total_release_m3s)
            capped_storage = start + (
                float(inflow_m3s) - capped_discharge
            ) * seconds / 1.0e9
            capped_storage = np.clip(
                capped_storage,
                spec.dead_storage_1e9m3,
                spec.normal_storage_1e9m3,
            )
            end_storage = np.where(capped, capped_storage, end_storage)
            total_discharge = np.where(capped, capped_discharge, total_discharge)

    end_level = np.interp(
        end_storage,
        spec.storages_1e9m3,
        spec.levels_m,
    )
    start_level = np.interp(
        start,
        spec.storages_1e9m3,
        spec.levels_m,
    )
    tailwater = np.polyval(
        spec._tailwater_polynomial_coefficients,
        total_discharge,
    )
    if np.any(~np.isfinite(tailwater)):
        raise CascadeSpecError(
            f"{spec.name}: tailwater polynomial produced a non-finite value."
        )
    head = np.maximum(0.0, (start_level + end_level) / 2.0 - tailwater)
    turbine_limit = (
        float(spec.max_turbine_discharge_m3s)
        if spec.max_turbine_discharge_m3s is not None
        else float("inf")
    )
    turbine = np.minimum(total_discharge, turbine_limit)
    power_coefficient = float(spec.power_coefficient_kw_per_m3s_m)
    max_power = float(spec.max_power_kw)
    power_limited_turbine = np.zeros_like(head)
    np.divide(
        max_power,
        power_coefficient * head,
        out=power_limited_turbine,
        where=head > 0.0,
    )
    turbine = np.where(
        head <= 0.0,
        0.0,
        np.minimum(turbine, power_limited_turbine),
    )
    power = np.minimum(max_power, power_coefficient * turbine * head)
    hours = seconds / 3600.0
    energy = power * hours / 1.0e6
    firm_deficit = np.maximum(0.0, float(spec.firm_power_kw) - power) * hours / 1.0e6
    return _TransitionBatch(
        end_storage=end_storage,
        total_discharge=total_discharge,
        energy=energy,
        firm_deficit=firm_deficit,
    )


def _shared_array_view(
    buffer: Any,
    dtype: np.dtype[Any] | type,
    shape: tuple[int, ...],
    *,
    read_only: bool = False,
) -> np.ndarray:
    view = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    if read_only:
        view.setflags(write=False)
    return view


def _initialize_worker(
    reservoir_2: ReservoirSpec,
    storages_1: np.ndarray,
    levels_2: np.ndarray,
    storages_2: np.ndarray,
    firm_power_failure_penalty_gwh: float,
    value_next_buffer: Any,
    upstream_end_storage_buffer: Any,
    upstream_discharge_buffer: Any,
    upstream_energy_buffer: Any,
    upstream_firm_deficit_buffer: Any,
    value_shape: tuple[int, int],
    upstream_shape: tuple[int, int],
) -> None:
    """Attach a worker to immutable model data and parent-managed shared arrays."""
    global _WORKER_CONTEXT

    worker_storages_1 = np.asarray(storages_1, dtype=float)
    worker_levels_2 = np.asarray(levels_2, dtype=float)
    worker_storages_2 = np.asarray(storages_2, dtype=float)
    worker_storages_1.setflags(write=False)
    worker_levels_2.setflags(write=False)
    worker_storages_2.setflags(write=False)
    _WORKER_CONTEXT = _WorkerContext(
        reservoir_2=reservoir_2,
        storages_1=worker_storages_1,
        levels_2=worker_levels_2,
        storages_2=worker_storages_2,
        firm_power_failure_penalty_gwh=float(
            firm_power_failure_penalty_gwh
        ),
        value_next=_shared_array_view(
            value_next_buffer, np.float64, value_shape, read_only=True
        ),
        upstream_end_storage=_shared_array_view(
            upstream_end_storage_buffer, np.float64, upstream_shape, read_only=True
        ),
        upstream_discharge=_shared_array_view(
            upstream_discharge_buffer, np.float64, upstream_shape, read_only=True
        ),
        upstream_energy=_shared_array_view(
            upstream_energy_buffer, np.float64, upstream_shape, read_only=True
        ),
        upstream_firm_deficit=_shared_array_view(
            upstream_firm_deficit_buffer, np.float64, upstream_shape, read_only=True
        ),
    )


def _solve_state_block(
    task: tuple[int, int, float, np.float64, int, int],
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Solve all joint actions for a contiguous block of joint states."""
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Parallel DDP worker was not initialized.")

    (
        block_start,
        block_stop,
        seconds,
        inflow_2_local,
        action_count_1,
        action_count_2,
    ) = task
    num_levels_2 = len(context.levels_2)
    block_length_1 = block_stop - block_start
    block_values = np.full((block_length_1, num_levels_2), -np.inf, dtype=np.float64)
    block_policies_1 = np.empty((block_length_1, num_levels_2), dtype=np.int16)
    block_policies_2 = np.broadcast_to(
        np.arange(num_levels_2, dtype=np.int16),
        (block_length_1, num_levels_2),
    ).copy()

    state_2_rows = np.arange(num_levels_2)
    target_storages_2 = context.storages_2[:action_count_2]
    for offset, state_1 in enumerate(range(block_start, block_stop)):
        block_policies_1[offset, :] = state_1
        for action_1 in range(action_count_1):
            if not CascadeDDP._target_is_reached(
                context.upstream_end_storage[state_1, action_1],
                context.storages_1[action_1],
            ):
                continue
            upstream_discharge = context.upstream_discharge[state_1, action_1]
            upstream_energy = context.upstream_energy[state_1, action_1]
            upstream_firm_deficit = context.upstream_firm_deficit[state_1, action_1]
            inflow_2 = float(inflow_2_local + upstream_discharge)
            downstream = _transition_batch(
                context.reservoir_2,
                context.storages_2,
                inflow_2,
                context.levels_2[:action_count_2],
                seconds,
            )
            feasible = np.isclose(
                downstream.end_storage,
                target_storages_2.reshape(1, -1),
                rtol=0.0,
                atol=1.0e-10,
            )
            upstream_failure = float(upstream_firm_deficit > 0.0)
            downstream_failure = (downstream.firm_deficit > 0.0).astype(
                np.float64
            )
            immediate = (
                float(upstream_energy)
                + downstream.energy
                - context.firm_power_failure_penalty_gwh
                * (upstream_failure + downstream_failure)
            )
            candidates = immediate + context.value_next[
                action_1, :action_count_2
            ].reshape(1, -1)
            candidates = np.where(feasible, candidates, -np.inf)
            best_action_2 = np.argmax(candidates, axis=1)
            best_for_action_1 = candidates[state_2_rows, best_action_2]
            improved = best_for_action_1 > block_values[offset]
            block_values[offset, improved] = best_for_action_1[improved]
            block_policies_1[offset, improved] = action_1
            block_policies_2[offset, improved] = best_action_2[improved]

    return (
        block_start,
        block_stop,
        block_values,
        block_policies_1,
        block_policies_2,
    )


class ParallelCascadeDDP(CascadeDDP):
    """State-parallel counterpart of :class:`CascadeDDP` for one CPU node."""

    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        local_inflows_m3s: Sequence[Sequence[float]],
        timestamps: Sequence[object],
        grid_step_m: float = 0.5,
        firm_power_failure_penalty_gwh: float = 0.0,
        monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
        *,
        num_workers: int = 8,
        state_block_size: int | None = None,
        multiprocessing_start_method: str | None = None,
    ) -> None:
        super().__init__(
            reservoir_1=reservoir_1,
            reservoir_2=reservoir_2,
            local_inflows_m3s=local_inflows_m3s,
            timestamps=timestamps,
            grid_step_m=grid_step_m,
            firm_power_failure_penalty_gwh=firm_power_failure_penalty_gwh,
            monthly_max_target_levels_m=monthly_max_target_levels_m,
        )
        if isinstance(num_workers, bool) or int(num_workers) != num_workers or num_workers <= 0:
            raise CascadeSpecError("Parallel DDP num_workers must be a positive integer.")
        if state_block_size is not None and (
            isinstance(state_block_size, bool)
            or int(state_block_size) != state_block_size
            or state_block_size <= 0
        ):
            raise CascadeSpecError(
                "Parallel DDP state_block_size must be a positive integer."
            )
        if multiprocessing_start_method is not None:
            available_methods = mp.get_all_start_methods()
            if multiprocessing_start_method not in available_methods:
                raise CascadeSpecError(
                    "Parallel DDP multiprocessing_start_method must be one of "
                    f"{available_methods}."
                )
        if len(self.levels_1) - 1 > np.iinfo(np.int16).max:
            raise CascadeSpecError("Reservoir 1 action grid exceeds the int16 policy range.")
        if len(self.levels_2) - 1 > np.iinfo(np.int16).max:
            raise CascadeSpecError("Reservoir 2 action grid exceeds the int16 policy range.")

        self.num_workers = int(num_workers)
        self.state_block_size = (
            int(state_block_size) if state_block_size is not None else None
        )
        self.multiprocessing_start_method = multiprocessing_start_method

    def _build_upstream_cache(
        self,
        t: int,
        upstream_end_storage: np.ndarray,
        upstream_discharge: np.ndarray,
        upstream_energy: np.ndarray,
        upstream_firm_deficit: np.ndarray,
    ) -> float:
        """Fill the exact per-month reservoir-1 transition cache."""
        seconds = float(self.timestamps[t].days_in_month) * 86400.0
        inflow_1 = self.inflows[t, 0]
        upstream = _transition_batch(
            self.reservoir_1,
            self.storages_1,
            float(inflow_1),
            self.levels_1,
            seconds,
        )
        np.copyto(upstream_end_storage, upstream.end_storage)
        np.copyto(upstream_discharge, upstream.total_discharge)
        np.copyto(upstream_energy, upstream.energy)
        np.copyto(upstream_firm_deficit, upstream.firm_deficit)
        return seconds

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
        """Solve the finite-horizon DDP with state blocks evaluated in processes."""
        initial_levels_m = self._validated_level_pair(
            initial_levels_m,
            label="initial_levels_m",
        )
        terminal_constraint = self._prepare_terminal_constraint(terminal_levels_m)
        normalized_terminal_levels = (
            terminal_constraint[0] if terminal_constraint is not None else None
        )
        months = len(self.inflows)
        num_levels_1 = len(self.levels_1)
        num_levels_2 = len(self.levels_2)
        value_shape = (num_levels_1, num_levels_2)
        upstream_shape = (num_levels_1, num_levels_1)
        num_joint_states = num_levels_1 * num_levels_2

        value = np.full((months + 1, *value_shape), -np.inf, dtype=float)
        self._initialize_terminal_value(value[-1], terminal_constraint)
        policies_1 = np.zeros((months, *value_shape), dtype=np.int16)
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

        process_context = (
            mp.get_context(self.multiprocessing_start_method)
            if self.multiprocessing_start_method is not None
            else mp.get_context()
        )
        value_next_buffer = process_context.RawArray("d", num_joint_states)
        upstream_cache_size = num_levels_1 * num_levels_1
        upstream_end_storage_buffer = process_context.RawArray(
            "d", upstream_cache_size
        )
        upstream_discharge_buffer = process_context.RawArray(
            "d", upstream_cache_size
        )
        upstream_energy_buffer = process_context.RawArray("d", upstream_cache_size)
        upstream_firm_deficit_buffer = process_context.RawArray(
            "d", upstream_cache_size
        )

        value_next = _shared_array_view(
            value_next_buffer, np.float64, value_shape
        )
        upstream_end_storage = _shared_array_view(
            upstream_end_storage_buffer, np.float64, upstream_shape
        )
        upstream_discharge = _shared_array_view(
            upstream_discharge_buffer, np.float64, upstream_shape
        )
        upstream_energy = _shared_array_view(
            upstream_energy_buffer, np.float64, upstream_shape
        )
        upstream_firm_deficit = _shared_array_view(
            upstream_firm_deficit_buffer, np.float64, upstream_shape
        )
        upstream_block_size = (
            None
            if self.state_block_size is None
            else max(1, (self.state_block_size + num_levels_2 - 1) // num_levels_2)
        )
        state_blocks = _make_state_blocks(
            num_levels_1, self.num_workers, upstream_block_size
        )

        np.copyto(value_next, value[start_t + 1])
        with process_context.Pool(
            processes=self.num_workers,
            initializer=_initialize_worker,
            initargs=(
                self.reservoir_2,
                self.storages_1,
                self.levels_2,
                self.storages_2,
                self.firm_power_failure_penalty_gwh,
                value_next_buffer,
                upstream_end_storage_buffer,
                upstream_discharge_buffer,
                upstream_energy_buffer,
                upstream_firm_deficit_buffer,
                value_shape,
                upstream_shape,
            ),
        ) as pool:
            for t in range(start_t, -1, -1):
                action_count_1, action_count_2, _, _ = self._monthly_action_limits(t)
                seconds = self._build_upstream_cache(
                    t,
                    upstream_end_storage,
                    upstream_discharge,
                    upstream_energy,
                    upstream_firm_deficit,
                )
                inflow_2_local = self.inflows[t, 1]
                tasks = (
                    (
                        block_start,
                        block_stop,
                        seconds,
                        inflow_2_local,
                        action_count_1,
                        action_count_2,
                    )
                    for block_start, block_stop in state_blocks
                )
                for (
                    block_start,
                    block_stop,
                    block_values,
                    block_policies_1,
                    block_policies_2,
                ) in pool.imap_unordered(_solve_state_block, tasks, chunksize=1):
                    value[t, block_start:block_stop, :] = block_values
                    policies_1[t, block_start:block_stop, :] = block_policies_1
                    policies_2[t, block_start:block_stop, :] = block_policies_2
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
                np.copyto(value_next, value[t])

        state_1 = int(np.argmin(np.abs(self.levels_1 - initial_levels_m[0])))
        state_2 = int(np.argmin(np.abs(self.levels_2 - initial_levels_m[1])))
        if not np.isfinite(value[0, state_1, state_2]):
            raise CascadeSpecError(
                "Parallel DDP found no feasible trajectory from the configured initial "
                "levels through the monthly operating limits to the strict terminal levels."
            )
        rows: list[dict] = []
        previous_end_storage_1: float | None = None
        previous_end_storage_2: float | None = None
        for t in range(months):
            _, _, monthly_max_1, monthly_max_2 = self._monthly_action_limits(t)
            action_1 = int(policies_1[t, state_1, state_2])
            action_2 = int(policies_2[t, state_1, state_2])
            immediate, upstream, downstream = self._transition_value(
                t, state_1, state_2, action_1, action_2
            )
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
                        np.isclose(upstream.end_level_m, monthly_max_1, rtol=0.0, atol=1.0e-8)
                    ),
                    "at_monthly_max_level_2": bool(
                        np.isclose(downstream.end_level_m, monthly_max_2, rtol=0.0, atol=1.0e-8)
                    ),
                    "is_flood_limit_action_month": bool(
                        monthly_max_1 < self.reservoir_1.normal_level_m
                        or monthly_max_2 < self.reservoir_2.normal_level_m
                    ),
                    "storage_jump_1_m3": storage_jump_1_m3,
                    "storage_jump_2_m3": storage_jump_2_m3,
                    # Preserve reward_gwh as actual generation and expose the
                    # reliability-aware Bellman reward in a separate column.
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
            np.isclose(final_levels[0], normalized_terminal_levels[0], rtol=0.0, atol=1.0e-8)
            and np.isclose(final_levels[1], normalized_terminal_levels[1], rtol=0.0, atol=1.0e-8)
        ):
            raise CascadeSpecError(
                "Parallel DDP forward reconstruction violated the strict terminal constraint."
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
