"""One-step-perfect-forecast monthly SDP for the P4 cascade."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .ddp import make_level_grid
from .environment import (
    CascadeSpecError,
    ReservoirSpec,
    _transition_arrays,
)
from .sdp_probability import MonthlyInflowModel


POLICY_SCHEMA_VERSION = 4
N_INFLOW_CLASSES = 10
POLICY_INFORMATION_SET = "month_levels_current_source_classes_one_step_perfect_forecast"
DEFAULT_WATER_LEVEL_STEP_M = 1.0
SDP_SOLVER_VARIANT = "bilinear_continuation"
_LEVEL_ATOL_M = 1.0e-8


def format_water_level_step_label(water_level_step: float) -> str:
    """Return the stable output label used to isolate different SDP grids."""
    if isinstance(water_level_step, bool) or not np.isfinite(water_level_step) or water_level_step <= 0.0:
        raise CascadeSpecError("SDP water-level step must be positive and finite.")
    return f"{float(water_level_step):g}m"


def sdp_output_variant_path(water_level_step: float) -> Path:
    """Return the isolated step/solver directory below an SDP output root."""
    return Path(format_water_level_step_label(water_level_step)) / SDP_SOLVER_VARIANT


def _nearest_indices(grid: np.ndarray, values: np.ndarray | float) -> np.ndarray:
    """Return ``np.argmin(abs(grid-value))`` indices without a full distance cube."""
    values = np.asarray(values, dtype=float)
    right = np.searchsorted(grid, values, side="left")
    right = np.clip(right, 0, len(grid) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(grid[right] - values) < np.abs(grid[left] - values)
    return np.where(choose_right, right, left).astype(np.intp, copy=False)


def _storage_interpolation_brackets(
    storage_grid: np.ndarray,
    storage_1e9m3: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of the verified DDP storage brackets.

    Queries outside the physical grid are clamped to its end points. Exact
    grid queries use a single state with zero upper-state weight.
    """
    grid = np.asarray(storage_grid, dtype=float)
    values = np.asarray(storage_1e9m3, dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise CascadeSpecError("SDP interpolation storage grid must be finite and strictly increasing.")
    if np.any(~np.isfinite(values)):
        raise CascadeSpecError("SDP interpolation storage queries must be finite.")

    clipped = np.clip(values, grid[0], grid[-1])
    high = np.searchsorted(grid, clipped, side="left")
    high = np.clip(high, 0, len(grid) - 1).astype(np.intp, copy=False)
    low = np.maximum(high - 1, 0)
    exact = clipped == grid[high]
    low = np.where(exact, high, low).astype(np.intp, copy=False)
    denominator = grid[high] - grid[low]
    weight = np.divide(
        clipped - grid[low],
        denominator,
        out=np.zeros_like(clipped, dtype=float),
        where=denominator > 0.0,
    )
    return low, high, weight


def bilinear_interpolate_storage_values(
    storage_grid_1: np.ndarray,
    storage_grid_2: np.ndarray,
    value_grid: np.ndarray,
    storage_1_1e9m3: np.ndarray | float,
    storage_2_1e9m3: np.ndarray | float,
) -> np.ndarray:
    """Interpolate a two-reservoir value table at continuous physical storage.

    This follows the latest DDP solver: interpolation applies only to the
    continuation value. It does not interpolate policy actions or overwrite
    either reservoir's physical storage.
    """
    values = np.asarray(value_grid, dtype=float)
    grid_1 = np.asarray(storage_grid_1, dtype=float)
    grid_2 = np.asarray(storage_grid_2, dtype=float)
    if values.shape != (len(grid_1), len(grid_2)):
        raise CascadeSpecError("SDP interpolation value table does not match its storage grids.")

    low_1, high_1, weight_1 = _storage_interpolation_brackets(grid_1, storage_1_1e9m3)
    low_2, high_2, weight_2 = _storage_interpolation_brackets(grid_2, storage_2_1e9m3)
    low_1, high_1, weight_1, low_2, high_2, weight_2 = np.broadcast_arrays(
        low_1, high_1, weight_1, low_2, high_2, weight_2
    )

    lower_row = values[low_1, low_2].astype(float, copy=True)
    lower_row += weight_2 * (values[low_1, high_2] - lower_row)
    upper_row = values[high_1, low_2].astype(float, copy=True)
    upper_row += weight_2 * (values[high_1, high_2] - upper_row)
    return lower_row + weight_1 * (upper_row - lower_row)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def reservoir_pair_sha256(reservoir_1: ReservoirSpec, reservoir_2: ReservoirSpec) -> str:
    """Fingerprint every physical input that can affect an SDP transition."""
    records = []
    for spec in (reservoir_1, reservoir_2):
        records.append({
            "name": spec.name,
            "dead_level_m": spec.dead_level_m,
            "normal_level_m": spec.normal_level_m,
            "levels_m": np.asarray(spec.levels_m, dtype=float).tolist(),
            "storages_1e9m3": np.asarray(spec.storages_1e9m3, dtype=float).tolist(),
            "tailwater_discharge_m3s": np.asarray(spec.tailwater_discharge_m3s, dtype=float).tolist(),
            "tailwater_levels_m": np.asarray(spec.tailwater_levels_m, dtype=float).tolist(),
            "max_turbine_discharge_m3s": spec.max_turbine_discharge_m3s,
            "max_power_kw": spec.max_power_kw,
            "firm_power_kw": spec.firm_power_kw,
            "power_coefficient_kw_per_m3s_m": spec.power_coefficient_kw_per_m3s_m,
            "max_total_release_m3s": spec.max_total_release_m3s,
            "high_flow_tailwater_anchors": [list(item) for item in spec.high_flow_tailwater_anchors],
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SDPPolicy:
    """A five-state policy for one-step-perfect-forecast operation.

    The action tables are indexed by ``[month, Hxw, Hnzd, Kxw, Knzd_local]``.
    ``Kxw`` is the observed class of the current month's XiaoWan natural
    inflow, and ``Knzd_local`` is the observed class of the current month's
    NuoZhaDu interval inflow.  The routed XiaoWan release is generated by the
    shared cascade physics after the joint target-level action is selected.
    """

    levels_1: np.ndarray
    levels_2: np.ndarray
    action_indices_1: np.ndarray
    action_indices_2: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        levels_1 = np.asarray(self.levels_1, dtype=float)
        levels_2 = np.asarray(self.levels_2, dtype=float)
        action_1 = np.asarray(self.action_indices_1)
        action_2 = np.asarray(self.action_indices_2)
        expected = (12, len(levels_1), len(levels_2), N_INFLOW_CLASSES, N_INFLOW_CLASSES)
        if len(levels_1) == 0 or len(levels_2) == 0:
            raise CascadeSpecError("SDP policy level grids cannot be empty.")
        if np.any(~np.isfinite(levels_1)) or np.any(~np.isfinite(levels_2)):
            raise CascadeSpecError("SDP policy levels must be finite.")
        if np.any(np.diff(levels_1) <= 0.0) or np.any(np.diff(levels_2) <= 0.0):
            raise CascadeSpecError("SDP policy level grids must increase strictly.")
        if action_1.shape != expected or action_2.shape != expected:
            raise CascadeSpecError(f"SDP action tables must have shape {expected}.")
        if not np.issubdtype(action_1.dtype, np.integer) or not np.issubdtype(action_2.dtype, np.integer):
            raise CascadeSpecError("SDP action tables must contain integer grid indices.")
        if np.any(action_1 < 0) or np.any(action_1 >= len(levels_1)):
            raise CascadeSpecError("SDP XiaoWan action index is outside its level grid.")
        if np.any(action_2 < 0) or np.any(action_2 >= len(levels_2)):
            raise CascadeSpecError("SDP NuoZhaDu action index is outside its level grid.")
        if not isinstance(self.metadata, dict) or self.metadata.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise CascadeSpecError(
                f"Unsupported or missing SDP policy metadata schema; "
                f"one-step-perfect-forecast schema {POLICY_SCHEMA_VERSION} is required."
            )
        if self.metadata.get("information_set") != POLICY_INFORMATION_SET:
            raise CascadeSpecError("SDP policy metadata has an incompatible information set.")
        if self.metadata.get("class_numbering") != "zero_based_0_to_9":
            raise CascadeSpecError("SDP policy metadata has incompatible inflow-class numbering.")
        if self.metadata.get("n_inflow_classes") != N_INFLOW_CLASSES:
            raise CascadeSpecError("SDP policy metadata must specify ten inflow classes per source.")
        if self.metadata.get("state_grid_shape") != list(expected[1:]):
            raise CascadeSpecError("SDP policy metadata state grid does not match its action tables.")
        if self.metadata.get("action_table_shape") != list(expected):
            raise CascadeSpecError("SDP policy metadata action-table shape does not match its arrays.")
        step = self.metadata.get("water_level_step_m")
        if isinstance(step, bool):
            raise CascadeSpecError("SDP policy metadata has an invalid water-level step.")
        try:
            step = float(step)
        except (TypeError, ValueError) as exc:
            raise CascadeSpecError("SDP policy metadata has an invalid water-level step.") from exc
        if not np.isfinite(step) or step <= 0.0:
            raise CascadeSpecError("SDP policy metadata has an invalid water-level step.")
        for name, levels in (("XiaoWan", levels_1), ("NuoZhaDu", levels_2)):
            differences = np.diff(levels)
            if len(differences) > 1 and not np.allclose(differences[:-1], step, rtol=0.0, atol=1.0e-9):
                raise CascadeSpecError(f"SDP {name} policy grid does not match its metadata water-level step.")
            if len(differences) and differences[-1] > step + 1.0e-9:
                raise CascadeSpecError(f"SDP {name} policy grid does not match its metadata water-level step.")

    def require_water_level_step(self, expected_step_m: float) -> None:
        """Reject a policy produced on a different state/action level grid."""
        if isinstance(expected_step_m, bool) or not np.isfinite(expected_step_m) or expected_step_m <= 0.0:
            raise CascadeSpecError("Expected SDP water-level step must be positive and finite.")
        actual_step_m = float(self.metadata["water_level_step_m"])
        if not np.isclose(actual_step_m, float(expected_step_m), rtol=0.0, atol=1.0e-12):
            raise CascadeSpecError(
                f"SDP policy water-level step {actual_step_m:g} m does not match "
                f"the requested {float(expected_step_m):g} m grid."
            )

    @staticmethod
    def _validate_current_classes(current_classes: Sequence[int]) -> tuple[int, int]:
        if len(current_classes) != 2:
            raise CascadeSpecError("current_classes must contain XiaoWan-natural and NuoZhaDu-interval classes.")
        result = []
        for value in current_classes:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or not 0 <= value < N_INFLOW_CLASSES
            ):
                raise CascadeSpecError("Current inflow classes must be zero-based integers from 0 to 9.")
            result.append(int(value))
        return result[0], result[1]

    def action(
        self,
        month: int,
        levels_m: Sequence[float],
        current_classes: Sequence[int],
    ) -> tuple[float, float]:
        """Look up targets after observing both current-month source classes."""
        if isinstance(month, bool) or not isinstance(month, (int, np.integer)) or not 1 <= month <= 12:
            raise CascadeSpecError("month must be an integer from 1 to 12.")
        levels = np.asarray(levels_m, dtype=float)
        if levels.shape != (2,) or np.any(~np.isfinite(levels)):
            raise CascadeSpecError("levels_m must contain two finite current water levels.")
        class_1, class_2 = self._validate_current_classes(current_classes)
        state_1 = int(_nearest_indices(self.levels_1, levels[0]))
        state_2 = int(_nearest_indices(self.levels_2, levels[1]))
        key = (int(month) - 1, state_1, state_2, class_1, class_2)
        action_1 = int(self.action_indices_1[key])
        action_2 = int(self.action_indices_2[key])
        return float(self.levels_1[action_1]), float(self.levels_2[action_2])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        levels_1 = np.asarray(self.levels_1, dtype=float)
        levels_2 = np.asarray(self.levels_2, dtype=float)
        actions_1 = np.asarray(self.action_indices_1, dtype=np.int16)
        actions_2 = np.asarray(self.action_indices_2, dtype=np.int16)
        metadata = {
            **self.metadata,
            "levels_1_sha256": _array_sha256(levels_1),
            "levels_2_sha256": _array_sha256(levels_2),
            "action_indices_1_sha256": _array_sha256(actions_1),
            "action_indices_2_sha256": _array_sha256(actions_2),
        }
        np.savez_compressed(
            path,
            levels_1=levels_1,
            levels_2=levels_2,
            action_indices_1=actions_1,
            action_indices_2=actions_2,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SDPPolicy":
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {"levels_1", "levels_2", "action_indices_1", "action_indices_2", "metadata_json"}
                if set(archive.files) != required:
                    raise CascadeSpecError("SDP policy archive has an unexpected file schema.")
                levels_1 = archive["levels_1"]
                levels_2 = archive["levels_2"]
                actions_1 = archive["action_indices_1"]
                actions_2 = archive["action_indices_2"]
                metadata = json.loads(str(archive["metadata_json"].item()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise CascadeSpecError(f"Cannot load SDP policy archive: {path}.") from exc
        if not isinstance(metadata, dict) or metadata.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise CascadeSpecError(
                f"Unsupported SDP policy archive schema; "
                f"one-step-perfect-forecast schema {POLICY_SCHEMA_VERSION} is required."
            )
        arrays = {
            "levels_1": levels_1,
            "levels_2": levels_2,
            "action_indices_1": actions_1,
            "action_indices_2": actions_2,
        }
        for name, array in arrays.items():
            expected_hash = metadata.pop(f"{name}_sha256", None)
            if expected_hash != _array_sha256(array):
                raise CascadeSpecError(f"SDP policy checksum mismatch for {name}.")
        return cls(levels_1, levels_2, actions_1, actions_2, metadata)

    def export_csv(self, path: str | Path) -> Path:
        """Write every current-month inflow-class key and its target levels."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "month",
            "xiaowan_level_m",
            "nuozhadu_level_m",
            "current_xiaowan_natural_inflow_class",
            "current_nuozhadu_interval_inflow_class",
            "target_xiaowan_level_m",
            "target_nuozhadu_level_m",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            for month in range(12):
                for state_1, level_1 in enumerate(self.levels_1):
                    for state_2, level_2 in enumerate(self.levels_2):
                        for class_1 in range(N_INFLOW_CLASSES):
                            for class_2 in range(N_INFLOW_CLASSES):
                                key = (month, state_1, state_2, class_1, class_2)
                                target_1 = self.levels_1[self.action_indices_1[key]]
                                target_2 = self.levels_2[self.action_indices_2[key]]
                                writer.writerow([
                                    month + 1,
                                    float(level_1),
                                    float(level_2),
                                    class_1,
                                    class_2,
                                    float(target_1),
                                    float(target_2),
                                ])
        return path


class CascadeSDP:
    """Periodic average-reward SDP with one-step-perfect inflow forecasts."""

    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        inflow_model: MonthlyInflowModel,
        water_level_step: float = DEFAULT_WATER_LEVEL_STEP_M,
        firm_power_failure_penalty_gwh: float = 0.0,
        monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
        cache_dir: str | Path | None = None,
        water_balance_tolerance_m3: float = 1.0e-3,
        max_physics_batch_elements: int = 2_000_000,
    ) -> None:
        inflow_model._validate()
        if water_level_step <= 0.0 or not np.isfinite(water_level_step):
            raise CascadeSpecError("SDP water_level_step must be positive and finite.")
        if (
            firm_power_failure_penalty_gwh < 0.0
            or not np.isfinite(firm_power_failure_penalty_gwh)
        ):
            raise CascadeSpecError(
                "SDP firm-power failure penalty must be finite and nonnegative."
            )
        if water_balance_tolerance_m3 < 0.0 or not np.isfinite(water_balance_tolerance_m3):
            raise CascadeSpecError("SDP water-balance tolerance must be finite and nonnegative.")
        if (
            isinstance(max_physics_batch_elements, bool)
            or not isinstance(max_physics_batch_elements, (int, np.integer))
            or max_physics_batch_elements <= 0
        ):
            raise CascadeSpecError("max_physics_batch_elements must be a positive integer.")
        if not (reservoir_1.energy_parameters_available and reservoir_2.energy_parameters_available):
            raise CascadeSpecError("SDP requires complete P4 V1 energy parameters.")
        self.reservoir_1 = reservoir_1
        self.reservoir_2 = reservoir_2
        self.inflow_model = inflow_model
        self.levels_1 = make_level_grid(reservoir_1.dead_level_m, reservoir_1.normal_level_m, water_level_step)
        self.levels_2 = make_level_grid(reservoir_2.dead_level_m, reservoir_2.normal_level_m, water_level_step)
        self.storages_1 = np.array([reservoir_1.storage_from_level(level) for level in self.levels_1])
        self.storages_2 = np.array([reservoir_2.storage_from_level(level) for level in self.levels_2])
        self.water_level_step = float(water_level_step)
        self.firm_power_failure_penalty_gwh = float(
            firm_power_failure_penalty_gwh
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
                "SDP monthly_max_target_levels_m must be a finite [12, 2] table."
            )
        if np.any(
            (monthly_max[:, 0] < reservoir_1.dead_level_m)
            | (monthly_max[:, 0] > reservoir_1.normal_level_m)
            | (monthly_max[:, 1] < reservoir_2.dead_level_m)
            | (monthly_max[:, 1] > reservoir_2.normal_level_m)
        ):
            raise CascadeSpecError(
                "SDP monthly maximum target levels must lie within physical level bounds."
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
            raise CascadeSpecError("SDP monthly operating limits remove every action.")
        self.water_balance_tolerance_m3 = float(water_balance_tolerance_m3)
        self.max_physics_batch_elements = int(max_physics_batch_elements)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_abs_water_balance_error_m3 = 0.0
        self.bellman_evaluations = 0

    def _stage_value(self, transition) -> np.ndarray:
        failure = np.asarray(transition.firm_deficit_gwh, dtype=float) > 0.0
        return (
            transition.energy_gwh
            - self.firm_power_failure_penalty_gwh
            * failure.astype(float, copy=False)
        )

    def _record_balance(self, transition) -> None:
        maximum = float(np.max(np.abs(transition.water_balance_error_m3)))
        self.max_abs_water_balance_error_m3 = max(self.max_abs_water_balance_error_m3, maximum)

    def _expected_over_current_classes(self, month_index: int, values: np.ndarray) -> np.ndarray:
        """Contract observed-class values to the pre-observation monthly value."""
        n_1, n_2 = len(self.levels_1), len(self.levels_2)
        expected_shape = (n_1, n_2, N_INFLOW_CLASSES, N_INFLOW_CLASSES)
        if values.shape != expected_shape:
            raise CascadeSpecError(f"SDP observed-class value table must have shape {expected_shape}.")
        probabilities_1 = self.inflow_model.probabilities[0, month_index]
        probabilities_2 = self.inflow_model.probabilities[1, month_index]
        return np.einsum("ijab,a,b->ij", values, probabilities_1, probabilities_2, optimize=True)

    def _bellman_month(
        self,
        month_index: int,
        future_expected_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve one month after its two source-inflow classes are observed.

        ``future_expected_value`` is the next month's value before its inflow
        classes are observed.  Consequently, maximization is conditional on
        the known current class pair while uncertainty enters only through the
        already-contracted continuation value for the next month.
        """
        n_1, n_2 = len(self.levels_1), len(self.levels_2)
        if future_expected_value.shape != (n_1, n_2):
            raise CascadeSpecError("SDP continuation value has an incompatible state grid.")
        seconds = float(self.inflow_model.mean_days[month_index]) * 86400.0
        representatives_1 = self.inflow_model.representatives[0, month_index]
        representatives_2 = self.inflow_model.representatives[1, month_index]
        values = np.empty((n_1, n_2, N_INFLOW_CLASSES, N_INFLOW_CLASSES), dtype=float)
        action_1 = np.empty_like(values, dtype=np.int16)
        action_2 = np.empty_like(values, dtype=np.int16)

        action_count_1 = int(self.monthly_action_counts_1[month_index])
        action_count_2 = int(self.monthly_action_counts_2[month_index])
        start_1 = self.storages_1[:, None]
        target_1 = self.levels_1[None, :action_count_1]
        start_2 = self.storages_2[None, None, :, None]
        target_2 = self.levels_2[None, None, None, :action_count_2]
        elements_per_state_1 = action_count_1 * n_2 * action_count_2
        state_1_chunk = max(1, self.max_physics_batch_elements // elements_per_state_1)
        for class_1 in range(N_INFLOW_CLASSES):
            upstream = _transition_arrays(
                self.reservoir_1,
                start_1,
                float(representatives_1[class_1]),
                target_1,
                seconds,
            )
            self._record_balance(upstream)
            upstream_value = self._stage_value(upstream)
            for class_2 in range(N_INFLOW_CLASSES):
                for start in range(0, n_1, state_1_chunk):
                    stop = min(n_1, start + state_1_chunk)
                    effective_inflow_2 = (
                        upstream.total_discharge_m3s[start:stop, :, None, None]
                        + float(representatives_2[class_2])
                    )
                    downstream = _transition_arrays(
                        self.reservoir_2,
                        start_2,
                        effective_inflow_2,
                        target_2,
                        seconds,
                    )
                    self._record_balance(downstream)
                    continuation = bilinear_interpolate_storage_values(
                        self.storages_1,
                        self.storages_2,
                        future_expected_value,
                        upstream.end_storage_1e9m3[start:stop, :, None, None],
                        downstream.end_storage_1e9m3,
                    )
                    q_values = (
                        upstream_value[start:stop, :, None, None]
                        + self._stage_value(downstream)
                        + continuation
                    )
                    candidates = q_values.transpose(0, 2, 1, 3).reshape(
                        stop - start,
                        n_2,
                        action_count_1 * action_count_2,
                    )
                    best_joint = np.argmax(candidates, axis=2)
                    values[start:stop, :, class_1, class_2] = np.take_along_axis(
                        candidates,
                        best_joint[:, :, None],
                        axis=2,
                    )[:, :, 0]
                    action_1[start:stop, :, class_1, class_2] = (best_joint // action_count_2).astype(np.int16)
                    action_2[start:stop, :, class_1, class_2] = (best_joint % action_count_2).astype(np.int16)

        self.bellman_evaluations += int(
            n_1
            * n_2
            * action_count_1
            * action_count_2
            * N_INFLOW_CLASSES
            * N_INFLOW_CLASSES
        )
        return values, action_1, action_2

    def _write_checkpoint(
        self,
        iteration: int,
        january_expected_relative_value: np.ndarray,
        actions_1: np.ndarray,
        actions_2: np.ndarray,
        residual_gwh: float,
    ) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_dir / "sdp_last_completed_iteration.npz",
            schema_version=np.asarray(POLICY_SCHEMA_VERSION),
            information_set=np.asarray(POLICY_INFORMATION_SET),
            solver_variant=np.asarray(SDP_SOLVER_VARIANT),
            continuation_value_lookup=np.asarray(
                "bilinear_interpolation_at_continuous_end_storage"
            ),
            iteration=np.asarray(iteration),
            january_expected_relative_value=january_expected_relative_value,
            action_indices_1=actions_1,
            action_indices_2=actions_2,
            residual_gwh=np.asarray(residual_gwh),
            training_sha256=np.asarray(self.inflow_model.metadata["training_sha256"]),
            water_level_step=np.asarray(self.water_level_step),
        )

    def solve(
        self,
        *,
        max_iterations: int = 400,
        tolerance_gwh: float = 1.0e-5,
        stable_iterations: int = 2,
        progress: bool = True,
    ) -> SDPPolicy:
        """Run cyclic relative value iteration and return a stationary monthly rule."""
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
            raise CascadeSpecError("max_iterations must be a positive integer.")
        if not np.isfinite(tolerance_gwh) or tolerance_gwh < 0.0:
            raise CascadeSpecError("tolerance_gwh must be finite and nonnegative.")
        if isinstance(stable_iterations, bool) or not isinstance(stable_iterations, (int, np.integer)) or stable_iterations <= 0:
            raise CascadeSpecError("stable_iterations must be a positive integer.")

        n_1, n_2 = len(self.levels_1), len(self.levels_2)
        january_expected_relative = np.zeros((n_1, n_2), dtype=float)
        previous_actions_1 = previous_actions_2 = None
        stable_count = 0
        converged = False
        residual = float("inf")
        annual_gain = 0.0
        actions_1 = np.zeros(
            (12, n_1, n_2, N_INFLOW_CLASSES, N_INFLOW_CLASSES),
            dtype=np.int16,
        )
        actions_2 = np.zeros_like(actions_1)
        anchor = (n_1 - 1, n_2 - 1)

        for iteration in range(1, int(max_iterations) + 1):
            future_expected = january_expected_relative
            for month_index in range(11, -1, -1):
                current_values, actions_1[month_index], actions_2[month_index] = self._bellman_month(
                    month_index,
                    future_expected,
                )
                future_expected = self._expected_over_current_classes(month_index, current_values)
                if progress:
                    print(
                        f"SDP iteration {iteration}: solved month {month_index + 1:02d}/12 "
                        f"on {n_1}x{n_2} level states and "
                        f"{N_INFLOW_CLASSES}x{N_INFLOW_CLASSES} observed class pairs.",
                        flush=True,
                    )
            january_expected = future_expected
            annual_gain = float(january_expected[anchor])
            updated_expected_relative = january_expected - annual_gain
            residual = float(np.max(np.abs(updated_expected_relative - january_expected_relative)))
            same_policy = (
                previous_actions_1 is not None
                and np.array_equal(actions_1, previous_actions_1)
                and np.array_equal(actions_2, previous_actions_2)
            )
            stable_count = stable_count + 1 if same_policy else 0
            self._write_checkpoint(iteration, updated_expected_relative, actions_1, actions_2, residual)
            if progress:
                print(
                    f"SDP annual iteration {iteration}: relative residual={residual:.6g} GWh, "
                    f"stable policy passes={stable_count}.",
                    flush=True,
                )
            january_expected_relative = updated_expected_relative
            previous_actions_1 = actions_1.copy()
            previous_actions_2 = actions_2.copy()
            # The storage-control problem can contain multiple recurrent
            # classes, so an anchored relative value need not converge in the
            # sup norm even after every decision has stopped changing.  The
            # retained single-reservoir SDP also uses policy stability as its
            # convergence criterion; the value residual remains a diagnostic.
            if stable_count >= stable_iterations:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                f"SDP relative value iteration did not converge in {max_iterations} annual iterations "
                f"(residual={residual:.6g} GWh, stable policy passes={stable_count})."
            )
        metadata = {
            "schema_version": POLICY_SCHEMA_VERSION,
            "method": "SDP",
            "information_set": POLICY_INFORMATION_SET,
            "action": "next_month_start_target_levels_m",
            "probability_model": "independent_product_of_monthly_empirical_marginals",
            "forecast_horizon_months": 1,
            "current_inflow_observation": "class_known_before_action",
            "next_month_inflow_treatment": "expectation_before_next_month_class_observation",
            "inflow_class_sources": ["xiaowan_natural_inflow", "nuozhadu_interval_inflow"],
            "class_numbering": "zero_based_0_to_9",
            "n_inflow_classes": N_INFLOW_CLASSES,
            "water_level_step_m": self.water_level_step,
            "state_level_grid_step_m": self.water_level_step,
            "candidate_target_level_action_step_m": self.water_level_step,
            "state_grid_shape": [n_1, n_2, N_INFLOW_CLASSES, N_INFLOW_CLASSES],
            "action_table_shape": [12, n_1, n_2, N_INFLOW_CLASSES, N_INFLOW_CLASSES],
            "joint_action_count": n_1 * n_2,
            "training_inflow_sha256": self.inflow_model.metadata["training_sha256"],
            "reservoir_physics_sha256": reservoir_pair_sha256(self.reservoir_1, self.reservoir_2),
            "training_expected_years": self.inflow_model.metadata["expected_years"],
            "training_stage_duration": "mean_calendar_days_for_each_month_over_training_period",
            "firm_power_failure_penalty_gwh": (
                self.firm_power_failure_penalty_gwh
            ),
            "failure_definition": (
                "station_specific_monthly_power_strictly_below_firm_power"
            ),
            "objective_definition": (
                "generation_energy_gwh_minus_gwh_equivalent_penalty_per_"
                "failed_station_month"
            ),
            "deficit_energy_role": "diagnostic_only_not_in_sdp_objective",
            "monthly_max_target_levels_m": self.monthly_max_target_levels_m.tolist(),
            "monthly_max_semantics": "applies_to_end_of_month_target_level",
            "flood_limit_action_months": [5, 6, 7, 8, 9, 10],
            "solver": "cyclic_relative_value_iteration",
            "solver_variant": SDP_SOLVER_VARIANT,
            "continuation_value_lookup": "bilinear_interpolation_at_continuous_end_storage",
            "policy_state_lookup": "nearest_level_indices_for_discrete_policy_table_only",
            "physical_state_propagation": "continuous_storage_without_grid_projection",
            "converged": True,
            "iterations": iteration,
            "relative_value_residual_gwh": residual,
            "relative_value_residual_within_tolerance": residual <= tolerance_gwh,
            "stable_policy_passes": stable_count,
            "annual_gain_anchor_gwh": annual_gain,
            "bellman_state_action_scenario_evaluations": self.bellman_evaluations,
            "max_abs_training_water_balance_error_m3": self.max_abs_water_balance_error_m3,
            "water_balance_tolerance_m3": self.water_balance_tolerance_m3,
            "max_physics_batch_elements": self.max_physics_batch_elements,
            "training_water_balance_within_tolerance": self.max_abs_water_balance_error_m3 <= self.water_balance_tolerance_m3,
        }
        return SDPPolicy(self.levels_1, self.levels_2, actions_1.copy(), actions_2.copy(), metadata)
