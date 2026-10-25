"""Parallel DDP with continuous storage carry-over and interpolated continuation."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ddp_parallel import ParallelCascadeDDP, _transition_batch
from .environment import CascadeSpecError, ReservoirSpec, ReservoirTransition, _transition


SOLVER_VARIANT = "interpolated_continuation"
_CHECKPOINT_SOLVER_VARIANT = (
    "interpolated_continuation_vectorized_failure_count_v1"
)
_CHECKPOINT_VERSION = 5


def storage_interpolation_bracket(
    storage_grid: np.ndarray,
    storage_1e9m3: float,
) -> tuple[int, int, float]:
    """Locate bracketing storage states and the upper-state interpolation weight."""
    if not np.isfinite(storage_1e9m3):
        raise CascadeSpecError("DDP interpolation storage must be finite.")
    if storage_1e9m3 <= storage_grid[0]:
        return 0, 0, 0.0

    last = len(storage_grid) - 1
    if storage_1e9m3 >= storage_grid[last]:
        return last, last, 0.0

    high = int(np.searchsorted(storage_grid, storage_1e9m3, side="left"))
    if storage_1e9m3 == storage_grid[high]:
        return high, high, 0.0

    low = high - 1
    low_storage = float(storage_grid[low])
    high_storage = float(storage_grid[high])
    weight = (float(storage_1e9m3) - low_storage) / (high_storage - low_storage)
    return low, high, float(weight)


def interpolated_continuation_value(
    value_next: np.ndarray,
    index_1_low: int,
    index_1_high: int,
    weight_1: float,
    index_2_low: int,
    index_2_high: int,
    weight_2: float,
) -> float:
    """Interpolate continuation values while preserving infeasible states.

    Hard terminal and monthly operating constraints introduce ``-inf`` grid
    values.  Zero-weight corners are ignored to avoid ``0 * -inf = nan``;
    every positively weighted corner must be feasible.
    """
    one_minus_weight_1 = 1.0 - weight_1
    one_minus_weight_2 = 1.0 - weight_2
    weighted_corners = (
        (one_minus_weight_1 * one_minus_weight_2, value_next[index_1_low, index_2_low]),
        (weight_1 * one_minus_weight_2, value_next[index_1_high, index_2_low]),
        (one_minus_weight_1 * weight_2, value_next[index_1_low, index_2_high]),
        (weight_1 * weight_2, value_next[index_1_high, index_2_high]),
    )
    total = 0.0
    for weight, value in weighted_corners:
        if weight <= 0.0:
            continue
        if not np.isfinite(value):
            return -np.inf
        total += float(weight * value)
    return float(total)


def _storage_interpolation_brackets(
    storage_grid: np.ndarray,
    storages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of :func:`storage_interpolation_bracket`."""
    values = np.asarray(storages, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise CascadeSpecError("DDP interpolation storage must be finite.")
    last = len(storage_grid) - 1
    high = np.searchsorted(storage_grid, values, side="left")
    high = np.clip(high, 0, last).astype(np.int64, copy=False)
    low = np.maximum(high - 1, 0)
    on_grid_or_boundary = (
        (values <= storage_grid[0])
        | (values >= storage_grid[last])
        | (values == storage_grid[high])
    )
    low = np.where(on_grid_or_boundary, high, low).astype(np.int64, copy=False)
    weight = np.zeros_like(values, dtype=np.float64)
    interpolated = low != high
    weight[interpolated] = (
        values[interpolated] - storage_grid[low[interpolated]]
    ) / (
        storage_grid[high[interpolated]] - storage_grid[low[interpolated]]
    )
    return low, high, weight


def _interpolated_continuation_batch(
    value_next: np.ndarray,
    index_1_low: int,
    index_1_high: int,
    weight_1: float,
    index_2_low: np.ndarray,
    index_2_high: np.ndarray,
    weight_2: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a matrix of continuation values."""
    one_minus_weight_1 = 1.0 - weight_1
    one_minus_weight_2 = 1.0 - weight_2
    weighted_corners = (
        (
            one_minus_weight_1 * one_minus_weight_2,
            value_next[index_1_low, index_2_low],
        ),
        (
            weight_1 * one_minus_weight_2,
            value_next[index_1_high, index_2_low],
        ),
        (
            one_minus_weight_1 * weight_2,
            value_next[index_1_low, index_2_high],
        ),
        (
            weight_1 * weight_2,
            value_next[index_1_high, index_2_high],
        ),
    )
    total = np.zeros_like(weight_2, dtype=np.float64)
    valid = np.ones_like(weight_2, dtype=bool)
    for weight, value in weighted_corners:
        positive = weight > 0.0
        finite = np.isfinite(value)
        valid &= ~positive | finite
        usable = positive & finite
        total[usable] += weight[usable] * value[usable]
    return np.where(valid, total, -np.inf)


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


@dataclass(frozen=True)
class _InterpolatedWorkerContext:
    reservoir_2: ReservoirSpec
    storages_1: np.ndarray
    levels_2: np.ndarray
    storages_2: np.ndarray
    firm_power_failure_penalty_gwh: float
    value_next: np.ndarray
    upstream_end_storage: np.ndarray
    upstream_index_low: np.ndarray
    upstream_index_high: np.ndarray
    upstream_weight: np.ndarray
    upstream_discharge: np.ndarray
    upstream_energy: np.ndarray
    upstream_firm_deficit: np.ndarray


_WORKER_CONTEXT: _InterpolatedWorkerContext | None = None


def _initialize_worker(
    reservoir_2: ReservoirSpec,
    storages_1: np.ndarray,
    levels_2: np.ndarray,
    storages_2: np.ndarray,
    firm_power_failure_penalty_gwh: float,
    value_next_buffer: Any,
    upstream_end_storage_buffer: Any,
    upstream_index_low_buffer: Any,
    upstream_index_high_buffer: Any,
    upstream_weight_buffer: Any,
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
    _WORKER_CONTEXT = _InterpolatedWorkerContext(
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
        upstream_index_low=_shared_array_view(
            upstream_index_low_buffer, np.int64, upstream_shape, read_only=True
        ),
        upstream_index_high=_shared_array_view(
            upstream_index_high_buffer, np.int64, upstream_shape, read_only=True
        ),
        upstream_weight=_shared_array_view(
            upstream_weight_buffer, np.float64, upstream_shape, read_only=True
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
) -> tuple[int, int, np.ndarray]:
    """Solve one block using continuous end storage for continuation values."""
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Interpolated parallel DDP worker was not initialized.")

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

    state_2_rows = np.arange(num_levels_2)
    target_storages_2 = context.storages_2[:action_count_2]
    for offset, state_1 in enumerate(range(block_start, block_stop)):
        for action_1 in range(action_count_1):
            if not ParallelCascadeDDP._target_is_reached(
                context.upstream_end_storage[state_1, action_1],
                context.storages_1[action_1],
            ):
                continue
            index_1_low = int(context.upstream_index_low[state_1, action_1])
            index_1_high = int(context.upstream_index_high[state_1, action_1])
            weight_1 = float(context.upstream_weight[state_1, action_1])
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
            index_2_low, index_2_high, weight_2 = _storage_interpolation_brackets(
                context.storages_2,
                downstream.end_storage,
            )
            continuation = _interpolated_continuation_batch(
                context.value_next,
                index_1_low,
                index_1_high,
                weight_1,
                index_2_low,
                index_2_high,
                weight_2,
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
            candidates = np.where(feasible, immediate + continuation, -np.inf)
            best_action_2 = np.argmax(candidates, axis=1)
            best_for_action_1 = candidates[state_2_rows, best_action_2]
            improved = best_for_action_1 > block_values[offset]
            block_values[offset, improved] = best_for_action_1[improved]

    return block_start, block_stop, block_values


def _storage_jump_diagnostics(
    jumps_m3: np.ndarray,
    external_natural_inflow_volume_m3: float,
) -> dict[str, float | int | None]:
    """Summarize measured interperiod storage discontinuities without masking them."""
    positive = jumps_m3[jumps_m3 > 0.0]
    negative = jumps_m3[jumps_m3 < 0.0]
    positive_sum = float(positive.sum())
    negative_sum = float(negative.sum())
    net_sum = float(jumps_m3.sum())
    absolute_sum = float(np.abs(jumps_m3).sum())
    maximum_absolute = float(np.abs(jumps_m3).max()) if jumps_m3.size else 0.0

    def ratio(value: float) -> float | None:
        if external_natural_inflow_volume_m3 == 0.0:
            return None
        return float(value / external_natural_inflow_volume_m3)

    return {
        "interval_count": int(jumps_m3.size),
        "positive_jump_count": int((jumps_m3 > 0.0).sum()),
        "negative_jump_count": int((jumps_m3 < 0.0).sum()),
        "zero_jump_count": int((jumps_m3 == 0.0).sum()),
        "cumulative_positive_storage_jump_m3": positive_sum,
        "cumulative_negative_storage_jump_m3": negative_sum,
        "cumulative_absolute_storage_jump_m3": absolute_sum,
        "net_storage_jump_m3": net_sum,
        "max_abs_storage_jump_m3": maximum_absolute,
        "cumulative_positive_jump_to_external_inflow_ratio": ratio(positive_sum),
        "cumulative_negative_jump_to_external_inflow_ratio": ratio(negative_sum),
        "cumulative_absolute_jump_to_external_inflow_ratio": ratio(absolute_sum),
        "net_jump_to_external_inflow_ratio": ratio(net_sum),
    }


class InterpolatedParallelCascadeDDP(ParallelCascadeDDP):
    """Parallel DDP with interpolated value continuation and continuous rollout."""

    def _checkpoint_fingerprint(
        self,
        initial_levels_m: tuple[float, float],
        terminal_levels_m: tuple[float, float] | None = None,
    ) -> str:
        base_fingerprint = super()._checkpoint_fingerprint(
            initial_levels_m,
            terminal_levels_m,
        )
        digest = hashlib.sha256()
        digest.update(base_fingerprint.encode("ascii"))
        digest.update(_CHECKPOINT_SOLVER_VARIANT.encode("ascii"))
        digest.update(str(_CHECKPOINT_VERSION).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _write_interpolated_checkpoint(
        checkpoint_path: Path,
        *,
        fingerprint: str,
        months: int,
        next_t: int,
        value_start: int,
        value_suffix: np.ndarray,
    ) -> None:
        """Atomically save every solved value layer needed by continuous rollout."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                checkpoint_version=np.asarray(_CHECKPOINT_VERSION, dtype=np.int64),
                solver_variant=np.asarray(_CHECKPOINT_SOLVER_VARIANT),
                fingerprint=np.asarray(fingerprint),
                months=np.asarray(months, dtype=np.int64),
                completed_months=np.asarray(months - value_start, dtype=np.int64),
                next_t=np.asarray(next_t, dtype=np.int64),
                value_start=np.asarray(value_start, dtype=np.int64),
                value_suffix=np.asarray(value_suffix, dtype=float),
            )
        os.replace(temporary_path, checkpoint_path)

    def _restore_interpolated_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        fingerprint: str,
        months: int,
        value: np.ndarray,
    ) -> tuple[int, int]:
        """Restore the complete solved suffix required by continuous rollout."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Interpolated DDP checkpoint not found: {checkpoint_path}")

        with np.load(checkpoint_path, allow_pickle=False) as saved:
            if "solver_variant" not in saved.files:
                raise CascadeSpecError(
                    "Interpolated DDP cannot resume a nearest-projection checkpoint."
                )
            if int(saved["checkpoint_version"].item()) != _CHECKPOINT_VERSION:
                raise CascadeSpecError("Unsupported interpolated DDP checkpoint version.")
            if str(saved["solver_variant"].item()) != _CHECKPOINT_SOLVER_VARIANT:
                raise CascadeSpecError("Checkpoint belongs to a different DDP solver variant.")
            if str(saved["fingerprint"].item()) != fingerprint:
                raise CascadeSpecError(
                    "Interpolated DDP checkpoint does not match the current data, grid, or initial levels."
                )
            if int(saved["months"].item()) != months:
                raise CascadeSpecError(
                    "Interpolated DDP checkpoint horizon does not match the current run."
                )

            next_t = int(saved["next_t"].item())
            value_start = int(saved["value_start"].item())
            if (
                next_t < -1
                or next_t >= months
                or value_start != next_t + 1
            ):
                raise CascadeSpecError(
                    "Interpolated DDP checkpoint contains an invalid recursion position."
                )

            expected_value_shape = (
                months - value_start + 1,
                len(self.levels_1),
                len(self.levels_2),
            )
            if saved["value_suffix"].shape != expected_value_shape:
                raise CascadeSpecError(
                    "Interpolated DDP checkpoint value suffix has the wrong shape."
                )

            completed_months = int(saved["completed_months"].item())
            if completed_months != months - value_start:
                raise CascadeSpecError(
                    "Interpolated DDP checkpoint completed-month count is inconsistent."
                )
            value[value_start:, :, :] = saved["value_suffix"]
        return next_t, completed_months

    def _build_upstream_interpolation_cache(
        self,
        t: int,
        upstream_end_storage: np.ndarray,
        upstream_index_low: np.ndarray,
        upstream_index_high: np.ndarray,
        upstream_weight: np.ndarray,
        upstream_discharge: np.ndarray,
        upstream_energy: np.ndarray,
        upstream_firm_deficit: np.ndarray,
    ) -> float:
        """Fill the per-month upstream physics and interpolation cache."""
        seconds = float(self.timestamps[t].days_in_month) * 86400.0
        inflow_1 = self.inflows[t, 0]
        upstream = _transition_batch(
            self.reservoir_1,
            self.storages_1,
            float(inflow_1),
            self.levels_1,
            seconds,
        )
        index_low, index_high, weight = _storage_interpolation_brackets(
            self.storages_1,
            upstream.end_storage,
        )
        np.copyto(upstream_end_storage, upstream.end_storage)
        np.copyto(upstream_index_low, index_low)
        np.copyto(upstream_index_high, index_high)
        np.copyto(upstream_weight, weight)
        np.copyto(upstream_discharge, upstream.total_discharge)
        np.copyto(upstream_energy, upstream.energy)
        np.copyto(upstream_firm_deficit, upstream.firm_deficit)
        return seconds

    def _continuous_forward_reconstruction(
        self,
        value: np.ndarray,
        initial_levels_m: tuple[float, float],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Reconstruct a physically continuous trajectory from the value tables."""
        months = len(self.inflows)
        current_storage_1 = self.reservoir_1.storage_from_level(initial_levels_m[0])
        current_storage_2 = self.reservoir_2.storage_from_level(initial_levels_m[1])
        previous_end_storage_1: float | None = None
        previous_end_storage_2: float | None = None
        storage_jumps_1_m3: list[float] = []
        storage_jumps_2_m3: list[float] = []
        external_natural_inflow_volume_m3 = 0.0
        rows: list[dict] = []

        for t in range(months):
            action_count_1, action_count_2, monthly_max_1, monthly_max_2 = (
                self._monthly_action_limits(t)
            )
            seconds = float(self.timestamps[t].days_in_month) * 86400.0
            inflow_1, inflow_2_local = self.inflows[t]
            external_natural_inflow_volume_m3 += (
                float(inflow_1) + float(inflow_2_local)
            ) * seconds

            storage_jump_1_m3 = (
                0.0
                if previous_end_storage_1 is None
                else (current_storage_1 - previous_end_storage_1) * 1.0e9
            )
            storage_jump_2_m3 = (
                0.0
                if previous_end_storage_2 is None
                else (current_storage_2 - previous_end_storage_2) * 1.0e9
            )
            if previous_end_storage_1 is not None:
                storage_jumps_1_m3.append(storage_jump_1_m3)
                storage_jumps_2_m3.append(storage_jump_2_m3)

            start_level_1 = self.reservoir_1.level_from_storage(current_storage_1)
            start_level_2 = self.reservoir_2.level_from_storage(current_storage_2)
            best_value = -np.inf
            best_action: tuple[int, int] | None = None
            best_immediate: float | None = None
            best_upstream: ReservoirTransition | None = None
            best_downstream: ReservoirTransition | None = None

            upstream_batch = _transition_batch(
                self.reservoir_1,
                np.asarray([current_storage_1]),
                float(inflow_1),
                self.levels_1[:action_count_1],
                seconds,
            )
            index_1_low, index_1_high, weight_1 = _storage_interpolation_brackets(
                self.storages_1,
                upstream_batch.end_storage,
            )
            target_storages_2 = self.storages_2[:action_count_2]
            for action_1 in range(action_count_1):
                if not self._target_is_reached(
                    float(upstream_batch.end_storage[0, action_1]),
                    self.storages_1[action_1],
                ):
                    continue
                inflow_2 = float(
                    inflow_2_local + upstream_batch.total_discharge[0, action_1]
                )
                downstream_batch = _transition_batch(
                    self.reservoir_2,
                    np.asarray([current_storage_2]),
                    inflow_2,
                    self.levels_2[:action_count_2],
                    seconds,
                )
                feasible = np.isclose(
                    downstream_batch.end_storage[0],
                    target_storages_2,
                    rtol=0.0,
                    atol=1.0e-10,
                )
                index_2_low, index_2_high, weight_2 = _storage_interpolation_brackets(
                    self.storages_2,
                    downstream_batch.end_storage,
                )
                continuation = _interpolated_continuation_batch(
                    value[t + 1],
                    int(index_1_low[0, action_1]),
                    int(index_1_high[0, action_1]),
                    float(weight_1[0, action_1]),
                    index_2_low,
                    index_2_high,
                    weight_2,
                )[0]
                upstream_failure = float(
                    upstream_batch.firm_deficit[0, action_1] > 0.0
                )
                downstream_failure = (
                    downstream_batch.firm_deficit[0] > 0.0
                ).astype(np.float64)
                immediate = (
                    float(upstream_batch.energy[0, action_1])
                    + downstream_batch.energy[0]
                    - self.firm_power_failure_penalty_gwh
                    * (upstream_failure + downstream_failure)
                )
                candidates = np.where(feasible, immediate + continuation, -np.inf)
                action_2 = int(np.argmax(candidates))
                candidate = float(candidates[action_2])
                if candidate > best_value:
                    best_value = candidate
                    best_action = (action_1, action_2)

            if best_action is not None:
                best_upstream = _transition(
                    self.reservoir_1,
                    current_storage_1,
                    float(inflow_1),
                    self.levels_1[best_action[0]],
                    seconds,
                )
                best_downstream = _transition(
                    self.reservoir_2,
                    current_storage_2,
                    float(inflow_2_local + best_upstream.total_discharge_m3s),
                    self.levels_2[best_action[1]],
                    seconds,
                )
                reward = self._reward_breakdown(best_upstream, best_downstream)
                best_immediate = reward["penalized_reward_gwh"]

            if (
                best_action is None
                or best_immediate is None
                or best_upstream is None
                or best_downstream is None
            ):
                raise CascadeSpecError(
                    f"Interpolated DDP found no finite action at month index {t}."
                )

            rows.append(
                {
                    "date": f"{self.timestamps[t].year:04d}-{self.timestamps[t].month:02d}-01",
                    "start_level_1_m": start_level_1,
                    "start_level_2_m": start_level_2,
                    "target_level_1_m": self.levels_1[best_action[0]],
                    "target_level_2_m": self.levels_2[best_action[1]],
                    "monthly_max_level_1_m": monthly_max_1,
                    "monthly_max_level_2_m": monthly_max_2,
                    "at_monthly_max_level_1": bool(
                        np.isclose(
                            best_upstream.end_level_m,
                            monthly_max_1,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
                    ),
                    "at_monthly_max_level_2": bool(
                        np.isclose(
                            best_downstream.end_level_m,
                            monthly_max_2,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
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
                    "inflow_1_m3s": float(inflow_1),
                    "inflow_2_local_m3s": float(inflow_2_local),
                    "routed_inflow_m3s": best_upstream.total_discharge_m3s,
                    "reservoir_1": asdict(best_upstream),
                    "reservoir_2": asdict(best_downstream),
                }
            )
            previous_end_storage_1 = best_upstream.end_storage_1e9m3
            previous_end_storage_2 = best_downstream.end_storage_1e9m3
            current_storage_1 = best_upstream.end_storage_1e9m3
            current_storage_2 = best_downstream.end_storage_1e9m3

        jumps_1 = np.asarray(storage_jumps_1_m3, dtype=np.float64)
        jumps_2 = np.asarray(storage_jumps_2_m3, dtype=np.float64)
        continuity = {
            "external_natural_inflow_volume_m3": float(
                external_natural_inflow_volume_m3
            ),
            "reservoir_1": _storage_jump_diagnostics(
                jumps_1,
                external_natural_inflow_volume_m3,
            ),
            "reservoir_2": _storage_jump_diagnostics(
                jumps_2,
                external_natural_inflow_volume_m3,
            ),
            "cascade_total": _storage_jump_diagnostics(
                jumps_1 + jumps_2,
                external_natural_inflow_volume_m3,
            ),
        }
        return pd.DataFrame(rows), continuity

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
        """Solve with interpolated continuation and a continuous-storage rollout."""
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

        value = np.full((months + 1, *value_shape), -np.inf, dtype=np.float64)
        self._initialize_terminal_value(value[-1], terminal_constraint)
        checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint_file is not None and (
            checkpoint_every_months is None or checkpoint_every_months <= 0
        ):
            raise CascadeSpecError("Interpolated DDP checkpoint interval must be positive.")
        if resume and checkpoint_file is None:
            raise CascadeSpecError("Interpolated DDP resume requires a checkpoint path.")

        fingerprint = self._checkpoint_fingerprint(
            initial_levels_m,
            normalized_terminal_levels,
        )
        resumed_completed_months = 0
        if resume:
            start_t, resumed_completed_months = self._restore_interpolated_checkpoint(
                checkpoint_file,
                fingerprint=fingerprint,
                months=months,
                value=value,
            )
            print(
                f"Resumed interpolated DDP checkpoint at {resumed_completed_months}/{months} "
                f"solved months: {checkpoint_file}",
                flush=True,
            )
        else:
            start_t = months - 1
            if checkpoint_file is not None:
                self._write_interpolated_checkpoint(
                    checkpoint_file,
                    fingerprint=fingerprint,
                    months=months,
                    next_t=start_t,
                    value_start=months,
                    value_suffix=value[months:],
                )
                print(
                    f"Initialized interpolated DDP checkpoint: {checkpoint_file}",
                    flush=True,
                )

        process_context = (
            mp.get_context(self.multiprocessing_start_method)
            if self.multiprocessing_start_method is not None
            else mp.get_context()
        )
        value_next_buffer = process_context.RawArray("d", num_joint_states)
        upstream_cache_size = num_levels_1 * num_levels_1
        upstream_end_storage_buffer = process_context.RawArray("d", upstream_cache_size)
        upstream_index_low_buffer = process_context.RawArray("q", upstream_cache_size)
        upstream_index_high_buffer = process_context.RawArray("q", upstream_cache_size)
        upstream_weight_buffer = process_context.RawArray("d", upstream_cache_size)
        upstream_discharge_buffer = process_context.RawArray("d", upstream_cache_size)
        upstream_energy_buffer = process_context.RawArray("d", upstream_cache_size)
        upstream_firm_deficit_buffer = process_context.RawArray(
            "d", upstream_cache_size
        )

        value_next = _shared_array_view(value_next_buffer, np.float64, value_shape)
        upstream_end_storage = _shared_array_view(
            upstream_end_storage_buffer, np.float64, upstream_shape
        )
        upstream_index_low = _shared_array_view(
            upstream_index_low_buffer, np.int64, upstream_shape
        )
        upstream_index_high = _shared_array_view(
            upstream_index_high_buffer, np.int64, upstream_shape
        )
        upstream_weight = _shared_array_view(
            upstream_weight_buffer, np.float64, upstream_shape
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
            num_levels_1,
            self.num_workers,
            upstream_block_size,
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
                upstream_index_low_buffer,
                upstream_index_high_buffer,
                upstream_weight_buffer,
                upstream_discharge_buffer,
                upstream_energy_buffer,
                upstream_firm_deficit_buffer,
                value_shape,
                upstream_shape,
            ),
        ) as pool:
            for t in range(start_t, -1, -1):
                action_count_1, action_count_2, _, _ = self._monthly_action_limits(t)
                seconds = self._build_upstream_interpolation_cache(
                    t,
                    upstream_end_storage,
                    upstream_index_low,
                    upstream_index_high,
                    upstream_weight,
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
                ) in pool.imap_unordered(_solve_state_block, tasks, chunksize=1):
                    value[t, block_start:block_stop, :] = block_values
                completed_months = months - t
                if progress_every_months and (
                    completed_months % progress_every_months == 0 or t == 0
                ):
                    print(
                        f"Interpolated DDP backward recursion solved "
                        f"{completed_months}/{months} months.",
                        flush=True,
                    )
                if checkpoint_file is not None and (
                    completed_months % int(checkpoint_every_months) == 0 or t == 0
                ):
                    self._write_interpolated_checkpoint(
                        checkpoint_file,
                        fingerprint=fingerprint,
                        months=months,
                        next_t=t - 1,
                        value_start=t,
                        value_suffix=value[t:],
                    )
                    print(
                        f"Saved interpolated DDP checkpoint after "
                        f"{completed_months}/{months} months: {checkpoint_file}",
                        flush=True,
                    )
                np.copyto(value_next, value[t])

        initial_state_1 = int(np.argmin(np.abs(self.levels_1 - initial_levels_m[0])))
        initial_state_2 = int(np.argmin(np.abs(self.levels_2 - initial_levels_m[1])))
        if not np.isfinite(value[0, initial_state_1, initial_state_2]):
            raise CascadeSpecError(
                "Interpolated DDP found no feasible trajectory from the configured "
                "initial levels through the monthly operating limits to the strict terminal levels."
            )

        schedule, storage_continuity = self._continuous_forward_reconstruction(
            value,
            initial_levels_m,
        )
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
                "Interpolated DDP forward reconstruction violated the strict terminal constraint."
            )
        summary = {
            "method": "DDP",
            "solver_variant": SOLVER_VARIANT,
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
            "checkpoint_solver_variant": _CHECKPOINT_SOLVER_VARIANT,
            "resumed_from_checkpoint": bool(resume),
            "resumed_completed_months": resumed_completed_months,
            "storage_continuity": storage_continuity,
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
