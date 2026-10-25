"""Unified monthly two-reservoir cascade environment.

The deployment observation deliberately contains only quantities available at
the start of the current month.  DDP may use a full realized sequence outside
this class as an offline upper bound; it must not change this interface.
"""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import re


class CascadeSpecError(ValueError):
    """Raised when source-backed cascade inputs are incomplete or inconsistent."""


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_month_period_index(values: Sequence[object]) -> pd.PeriodIndex:
    """Parse monthly labels without the nanosecond Timestamp year-2262 limit."""
    labels = [str(value).strip()[:7] for value in values]
    try:
        periods = pd.PeriodIndex(labels, freq="M")
    except (TypeError, ValueError) as exc:
        raise CascadeSpecError("Monthly dates must be valid year-month labels.") from exc
    if periods.hasnans:
        raise CascadeSpecError("Monthly dates must be complete.")
    return periods


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small mapping/list YAML subset used by project configs.

    PyYAML remains the preferred parser. This fallback keeps the physics smoke
    test usable before the project dependency environment is installed.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, raw_value = raw_line.strip().partition(":")
        if not separator:
            raise CascadeSpecError(f"Unsupported configuration line: {raw_line!r}")
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue
        if value == "null":
            current[key] = None
        elif value == "true":
            current[key] = True
        elif value == "false":
            current[key] = False
        elif value.startswith("[") and value.endswith("]"):
            try:
                current[key] = literal_eval(value)
            except (SyntaxError, ValueError):
                current[key] = [item.strip() for item in value[1:-1].split(",")]
        else:
            try:
                current[key] = float(value) if "." in value else int(value)
            except ValueError:
                current[key] = value
    return root


@dataclass(frozen=True)
class ReservoirSpec:
    """One reservoir's source-backed curves and operating parameters."""

    name: str
    dead_level_m: float
    normal_level_m: float
    levels_m: np.ndarray
    storages_1e9m3: np.ndarray
    tailwater_discharge_m3s: np.ndarray
    tailwater_levels_m: np.ndarray
    max_turbine_discharge_m3s: float | None = None
    max_power_kw: float | None = None
    firm_power_kw: float | None = None
    power_coefficient_kw_per_m3s_m: float | None = None
    max_total_release_m3s: float | None = None
    high_flow_tailwater_anchors: tuple[tuple[float, float], ...] = ()
    _tailwater_discharges: np.ndarray = field(init=False, repr=False, compare=False)
    _tailwater_levels: np.ndarray = field(init=False, repr=False, compare=False)
    _tailwater_polynomial_coefficients: np.ndarray = field(init=False, repr=False, compare=False)
    _tailwater_polynomial_degree: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dead_level_m > self.normal_level_m:
            raise CascadeSpecError(f"{self.name}: dead level exceeds normal level.")
        if len(self.levels_m) < 2 or len(self.tailwater_discharge_m3s) < 2:
            raise CascadeSpecError(f"{self.name}: each curve needs at least two points.")
        if not np.all(np.diff(self.levels_m) > 0.0):
            raise CascadeSpecError(f"{self.name}: level-storage levels must increase.")
        if not np.all(np.diff(self.storages_1e9m3) > 0.0):
            raise CascadeSpecError(f"{self.name}: storage curve must increase.")
        if not np.all(np.diff(self.tailwater_discharge_m3s) > 0.0):
            raise CascadeSpecError(f"{self.name}: tailwater discharge curve must increase.")
        if not np.all(np.diff(self.tailwater_levels_m) >= 0.0):
            raise CascadeSpecError(f"{self.name}: tailwater level curve must not decrease.")
        if self.dead_level_m < self.levels_m[0] or self.normal_level_m > self.levels_m[-1]:
            raise CascadeSpecError(f"{self.name}: operating levels are outside its storage curve.")
        anchors = np.asarray(self.high_flow_tailwater_anchors, dtype=float)
        if anchors.size:
            if anchors.ndim != 2 or anchors.shape[1] != 2:
                raise CascadeSpecError(f"{self.name}: high-flow anchors must be [discharge, level] pairs.")
            if np.any(anchors[:, 0] <= self.tailwater_discharge_m3s[-1]) or np.any(np.diff(anchors[:, 0]) <= 0.0):
                raise CascadeSpecError(f"{self.name}: high-flow anchor discharges must increase beyond the measured curve.")
            if anchors[0, 1] < self.tailwater_levels_m[-1] or np.any(np.diff(anchors[:, 1]) < 0.0):
                raise CascadeSpecError(f"{self.name}: high-flow anchor levels must not decrease.")
        if anchors.size:
            tailwater_discharges = np.concatenate((self.tailwater_discharge_m3s, anchors[:, 0]))
            tailwater_levels = np.concatenate((self.tailwater_levels_m, anchors[:, 1]))
        else:
            tailwater_discharges = self.tailwater_discharge_m3s.copy()
            tailwater_levels = self.tailwater_levels_m.copy()
        object.__setattr__(self, "_tailwater_discharges", tailwater_discharges)
        object.__setattr__(self, "_tailwater_levels", tailwater_levels)
        degree = min(4, len(tailwater_discharges) - 1)
        coefficients = np.polyfit(tailwater_discharges, tailwater_levels, degree)
        object.__setattr__(self, "_tailwater_polynomial_coefficients", coefficients)
        object.__setattr__(self, "_tailwater_polynomial_degree", degree)

    @classmethod
    def from_curve_directory(cls, name: str, settings: Mapping[str, Any], curve_dir: Path) -> "ReservoirSpec":
        storage = pd.read_csv(curve_dir / "level_storage_curve.csv")
        tailwater = pd.read_csv(curve_dir / "tailwater_discharge_curve.csv")
        return cls(
            name=name,
            dead_level_m=float(settings["dead_level_m"]),
            normal_level_m=float(settings["normal_level_m"]),
            levels_m=storage["water_level_m"].to_numpy(dtype=float),
            storages_1e9m3=storage["storage_1e9_m3"].to_numpy(dtype=float),
            tailwater_discharge_m3s=tailwater["discharge_m3s"].to_numpy(dtype=float),
            tailwater_levels_m=tailwater["tailwater_level_m"].to_numpy(dtype=float),
            max_turbine_discharge_m3s=_optional_float(settings.get("max_turbine_discharge_m3s")),
            max_power_kw=_optional_float(settings.get("max_power_kw")),
            firm_power_kw=_optional_float(settings.get("firm_power_kw")),
            power_coefficient_kw_per_m3s_m=_optional_float(
                settings.get("power_coefficient_kw_per_m3s_m")
            ),
            max_total_release_m3s=_optional_float(settings.get("max_total_release_m3s")),
            high_flow_tailwater_anchors=tuple(
                (float(item[0]), float(item[1]))
                for item in settings.get("high_flow_tailwater_anchors", [])
            ),
        )

    @property
    def dead_storage_1e9m3(self) -> float:
        return self.storage_from_level(self.dead_level_m)

    @property
    def normal_storage_1e9m3(self) -> float:
        return self.storage_from_level(self.normal_level_m)

    @property
    def energy_parameters_available(self) -> bool:
        return all(
            value is not None and value > 0.0
            for value in (
                self.max_power_kw,
                self.firm_power_kw,
                self.power_coefficient_kw_per_m3s_m,
            )
        )

    def storage_from_level(self, level_m: float) -> float:
        return float(np.interp(level_m, self.levels_m, self.storages_1e9m3))

    def level_from_storage(self, storage_1e9m3: float) -> float:
        return float(np.interp(storage_1e9m3, self.storages_1e9m3, self.levels_m))

    def tailwater_from_discharge(self, discharge_m3s: float) -> tuple[float, bool]:
        discharges = self._tailwater_discharges
        lower, upper = discharges[[0, -1]]
        extrapolated = bool(discharge_m3s < lower or discharge_m3s > upper)
        poly = np.poly1d(self._tailwater_polynomial_coefficients)
        tailwater = float(poly(float(discharge_m3s)))
        if not np.isfinite(tailwater):
            raise CascadeSpecError(f"{self.name}: tailwater polynomial produced a non-finite value.")
        return tailwater, extrapolated


@dataclass(frozen=True)
class ReservoirTransition:
    """One mass-balanced reservoir transition over the current month."""

    requested_target_level_m: float
    end_level_m: float
    start_storage_1e9m3: float
    end_storage_1e9m3: float
    inflow_m3s: float
    total_discharge_m3s: float
    turbine_discharge_m3s: float | None
    spill_discharge_m3s: float | None
    unallocated_release_m3s: float | None
    power_kw: float | None
    energy_gwh: float | None
    firm_deficit_gwh: float | None
    tailwater_level_m: float | None
    head_m: float | None
    target_level_clipped: bool
    tailwater_discharge_extrapolated: bool
    water_balance_error_m3: float


@dataclass(frozen=True)
class ReservoirTransitionArrays:
    """Broadcast NumPy counterpart of :class:`ReservoirTransition`.

    SDP evaluates the same physical transition for many state/action/inflow
    combinations.  Keeping the vectorized implementation in this shared
    physics module avoids placing a second reservoir model in the algorithm.
    The scalar ``_transition`` used by DDP remains unchanged.
    """

    end_storage_1e9m3: np.ndarray
    end_level_m: np.ndarray
    total_discharge_m3s: np.ndarray
    power_kw: np.ndarray
    energy_gwh: np.ndarray
    firm_deficit_gwh: np.ndarray
    water_balance_error_m3: np.ndarray


def _transition_arrays(
    spec: ReservoirSpec,
    start_storage_1e9m3: np.ndarray | float,
    inflow_m3s: np.ndarray | float,
    requested_target_level_m: np.ndarray | float,
    seconds: float,
) -> ReservoirTransitionArrays:
    """Evaluate ``_transition`` semantics over broadcast-compatible arrays.

    This batch API is intentionally narrow: it returns only the fields needed
    by Bellman recursion.  Evaluation and CSV output still use
    :class:`CascadeReservoirEnv`, so all detailed reporting comes directly
    from the established scalar simulator.
    """
    if seconds <= 0.0:
        raise CascadeSpecError("A monthly period must have positive duration.")
    if not spec.energy_parameters_available:
        raise CascadeSpecError(f"{spec.name}: batch energy transition requires complete energy parameters.")

    start, inflow, requested = np.broadcast_arrays(
        np.asarray(start_storage_1e9m3, dtype=float),
        np.asarray(inflow_m3s, dtype=float),
        np.asarray(requested_target_level_m, dtype=float),
    )
    if np.any(~np.isfinite(start)) or np.any(~np.isfinite(inflow)) or np.any(~np.isfinite(requested)):
        raise CascadeSpecError(f"{spec.name}: batch transition inputs must be finite.")
    if np.any(inflow < 0.0):
        raise CascadeSpecError(f"{spec.name}: inflow must be nonnegative.")

    target_level = np.clip(requested, spec.dead_level_m, spec.normal_level_m)
    target_storage = np.interp(target_level, spec.levels_m, spec.storages_1e9m3)
    available_storage = start + inflow * seconds / 1.0e9
    end_storage = np.minimum(np.minimum(target_storage, available_storage), spec.normal_storage_1e9m3)
    end_storage = np.maximum(end_storage, spec.dead_storage_1e9m3)
    total_discharge = (start + inflow * seconds / 1.0e9 - end_storage) * 1.0e9 / seconds
    if np.any(total_discharge < -1.0e-9):
        raise CascadeSpecError(f"{spec.name}: transition produced a negative release.")
    total_discharge = np.maximum(0.0, total_discharge)
    if spec.max_total_release_m3s is not None:
        capped = total_discharge > spec.max_total_release_m3s
        total_discharge = np.where(capped, float(spec.max_total_release_m3s), total_discharge)
        capped_storage = start + (inflow - total_discharge) * seconds / 1.0e9
        capped_storage = np.clip(capped_storage, spec.dead_storage_1e9m3, spec.normal_storage_1e9m3)
        end_storage = np.where(capped, capped_storage, end_storage)

    end_level = np.interp(end_storage, spec.storages_1e9m3, spec.levels_m)
    water_balance_error = (end_storage - start) * 1.0e9 - (inflow - total_discharge) * seconds
    tailwater = np.polyval(spec._tailwater_polynomial_coefficients, total_discharge)
    if np.any(~np.isfinite(tailwater)):
        raise CascadeSpecError(f"{spec.name}: tailwater polynomial produced a non-finite value.")
    start_level = np.interp(start, spec.storages_1e9m3, spec.levels_m)
    head = np.maximum(0.0, (start_level + end_level) / 2.0 - tailwater)
    turbine_limit = (
        float(spec.max_turbine_discharge_m3s)
        if spec.max_turbine_discharge_m3s is not None
        else float("inf")
    )
    turbine = np.minimum(total_discharge, turbine_limit)
    capacity_turbine = np.divide(
        float(spec.max_power_kw),
        float(spec.power_coefficient_kw_per_m3s_m) * head,
        out=np.full_like(head, np.inf),
        where=head > 0.0,
    )
    turbine = np.where(head <= 0.0, 0.0, np.minimum(turbine, capacity_turbine))
    power = np.minimum(
        float(spec.max_power_kw),
        float(spec.power_coefficient_kw_per_m3s_m) * turbine * head,
    )
    hours = seconds / 3600.0
    energy = power * hours / 1.0e6
    firm_deficit = np.maximum(0.0, float(spec.firm_power_kw) - power) * hours / 1.0e6
    return ReservoirTransitionArrays(
        end_storage_1e9m3=end_storage,
        end_level_m=end_level,
        total_discharge_m3s=total_discharge,
        power_kw=power,
        energy_gwh=energy,
        firm_deficit_gwh=firm_deficit,
        water_balance_error_m3=water_balance_error,
    )


class ZeroDelayRouter:
    """P4-v1 router: the upstream total release reaches reservoir 2 in-month."""

    mode = "zero_delay_direct_transfer"

    def route(self, upstream_total_discharge_m3s: float) -> float:
        return float(upstream_total_discharge_m3s)


def _transition(
    spec: ReservoirSpec,
    start_storage_1e9m3: float,
    inflow_m3s: float,
    requested_target_level_m: float,
    seconds: float,
) -> ReservoirTransition:
    if seconds <= 0.0:
        raise CascadeSpecError("A monthly period must have positive duration.")
    if inflow_m3s < 0.0:
        raise CascadeSpecError(f"{spec.name}: inflow must be nonnegative.")
    target_level = float(np.clip(requested_target_level_m, spec.dead_level_m, spec.normal_level_m))
    target_storage = spec.storage_from_level(target_level)
    available_storage = start_storage_1e9m3 + inflow_m3s * seconds / 1.0e9
    end_storage = min(target_storage, available_storage, spec.normal_storage_1e9m3)
    end_storage = max(end_storage, spec.dead_storage_1e9m3)
    total_discharge = (start_storage_1e9m3 + inflow_m3s * seconds / 1.0e9 - end_storage) * 1.0e9 / seconds
    if total_discharge < -1.0e-9:
        raise CascadeSpecError(f"{spec.name}: transition produced a negative release.")
    total_discharge = max(0.0, total_discharge)
    if spec.max_total_release_m3s is not None and total_discharge > spec.max_total_release_m3s:
        total_discharge = float(spec.max_total_release_m3s)
        end_storage = start_storage_1e9m3 + (inflow_m3s - total_discharge) * seconds / 1.0e9
        end_storage = min(max(end_storage, spec.dead_storage_1e9m3), spec.normal_storage_1e9m3)
    end_level = spec.level_from_storage(end_storage)
    water_balance_error = (end_storage - start_storage_1e9m3) * 1.0e9 - (inflow_m3s - total_discharge) * seconds

    if not spec.energy_parameters_available:
        return ReservoirTransition(
            requested_target_level_m=float(requested_target_level_m), end_level_m=end_level,
            start_storage_1e9m3=float(start_storage_1e9m3), end_storage_1e9m3=end_storage,
            inflow_m3s=float(inflow_m3s), total_discharge_m3s=total_discharge,
            turbine_discharge_m3s=None, spill_discharge_m3s=None,
            unallocated_release_m3s=total_discharge, power_kw=None, energy_gwh=None,
            firm_deficit_gwh=None, tailwater_level_m=None, head_m=None,
            target_level_clipped=not np.isclose(target_level, requested_target_level_m),
            tailwater_discharge_extrapolated=False, water_balance_error_m3=water_balance_error,
        )

    tailwater, tailwater_extrapolated = spec.tailwater_from_discharge(total_discharge)
    head = max(0.0, (spec.level_from_storage(start_storage_1e9m3) + end_level) / 2.0 - tailwater)
    turbine_limit = (
        float(spec.max_turbine_discharge_m3s)
        if spec.max_turbine_discharge_m3s is not None
        else float("inf")
    )
    turbine = min(total_discharge, turbine_limit)
    if head <= 0.0:
        turbine = 0.0
    else:
        turbine = min(turbine, float(spec.max_power_kw) / (float(spec.power_coefficient_kw_per_m3s_m) * head))
    power = min(float(spec.max_power_kw), float(spec.power_coefficient_kw_per_m3s_m) * turbine * head)
    hours = seconds / 3600.0
    energy = power * hours / 1.0e6
    firm_deficit = max(0.0, float(spec.firm_power_kw) - power) * hours / 1.0e6
    return ReservoirTransition(
        requested_target_level_m=float(requested_target_level_m), end_level_m=end_level,
        start_storage_1e9m3=float(start_storage_1e9m3), end_storage_1e9m3=end_storage,
        inflow_m3s=float(inflow_m3s), total_discharge_m3s=total_discharge,
        turbine_discharge_m3s=turbine, spill_discharge_m3s=total_discharge - turbine,
        unallocated_release_m3s=None, power_kw=power, energy_gwh=energy,
        firm_deficit_gwh=firm_deficit, tailwater_level_m=tailwater, head_m=head,
        target_level_clipped=not np.isclose(target_level, requested_target_level_m),
        tailwater_discharge_extrapolated=tailwater_extrapolated, water_balance_error_m3=water_balance_error,
    )


class CascadeReservoirEnv:
    """Gymnasium-style environment for a serial two-reservoir monthly system.

    Its observation is ``[V1, V2, I1, I2_local, month]``.  ``I2_local`` is the
    XiaoWan-NuoZhaDu incremental natural inflow; the NuoZhaDu effective inflow
    is formed by adding routed XiaoWan total release in the current month.
    """

    def __init__(
        self,
        reservoir_1: ReservoirSpec,
        reservoir_2: ReservoirSpec,
        local_inflows_m3s: Sequence[Sequence[float]],
        timestamps: Sequence[object] | None = None,
        initial_levels_m: Sequence[float] | None = None,
        firm_power_failure_penalty_gwh: float = 0.0,
        monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
        require_energy_parameters: bool = True,
        router: ZeroDelayRouter | None = None,
    ) -> None:
        inflows = np.asarray(local_inflows_m3s, dtype=float)
        if inflows.ndim != 2 or inflows.shape[1] != 2 or len(inflows) == 0:
            raise CascadeSpecError("local_inflows_m3s must have shape [months, 2].")
        if np.any(~np.isfinite(inflows)) or np.any(inflows < 0.0):
            raise CascadeSpecError("Local inflows must be finite and nonnegative.")
        if require_energy_parameters and not (reservoir_1.energy_parameters_available and reservoir_2.energy_parameters_available):
            raise CascadeSpecError("Energy experiment blocked: source-backed machine parameters are incomplete.")
        self.reservoir_1 = reservoir_1
        self.reservoir_2 = reservoir_2
        self.local_inflows_m3s = inflows
        self.timestamps = _as_month_period_index(timestamps) if timestamps is not None else None
        if self.timestamps is not None and len(self.timestamps) != len(inflows):
            raise CascadeSpecError("timestamps length must equal the number of inflow months.")
        self.initial_levels_m = tuple(initial_levels_m or (reservoir_1.normal_level_m, reservoir_2.normal_level_m))
        self.firm_power_failure_penalty_gwh = float(
            firm_power_failure_penalty_gwh
        )
        if (
            not np.isfinite(self.firm_power_failure_penalty_gwh)
            or self.firm_power_failure_penalty_gwh < 0.0
        ):
            raise CascadeSpecError(
                "firm_power_failure_penalty_gwh must be finite and nonnegative."
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
                "monthly_max_target_levels_m must be a finite [12, 2] table."
            )
        if np.any(
            (monthly_max[:, 0] < reservoir_1.dead_level_m)
            | (monthly_max[:, 0] > reservoir_1.normal_level_m)
            | (monthly_max[:, 1] < reservoir_2.dead_level_m)
            | (monthly_max[:, 1] > reservoir_2.normal_level_m)
        ):
            raise CascadeSpecError(
                "Monthly maximum target levels must lie within reservoir bounds."
            )
        self.monthly_max_target_levels_m = monthly_max.copy()
        self.router = router or ZeroDelayRouter()
        self.index = 0
        self.storages_1e9m3 = np.zeros(2, dtype=float)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (5,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (2,)

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        del seed
        options = dict(options or {})
        self.index = int(options.get("start_index", 0))
        if not 0 <= self.index < len(self.local_inflows_m3s):
            raise CascadeSpecError("start_index is outside the inflow series.")
        levels = options.get("initial_levels_m", self.initial_levels_m)
        if len(levels) != 2:
            raise CascadeSpecError("initial_levels_m must contain two values.")
        self.storages_1e9m3 = np.array([
            self.reservoir_1.storage_from_level(float(levels[0])),
            self.reservoir_2.storage_from_level(float(levels[1])),
        ])
        return self._observation(), {"routing_mode": self.router.mode, "energy_parameters_available": self._energy_available}

    @property
    def _energy_available(self) -> bool:
        return self.reservoir_1.energy_parameters_available and self.reservoir_2.energy_parameters_available

    def _seconds_this_step(self) -> float:
        if self.timestamps is None:
            return 30.4 * 86400.0
        return float(self.timestamps[self.index].days_in_month) * 86400.0

    def _month(self) -> int:
        return int(self.timestamps[self.index].month) if self.timestamps is not None else self.index % 12 + 1

    def _observation(self) -> np.ndarray:
        inflow_1, inflow_2_local = self.local_inflows_m3s[self.index]
        return np.array([self.storages_1e9m3[0], self.storages_1e9m3[1], inflow_1, inflow_2_local, self._month()], dtype=np.float32)

    def get_feasible_action_mask(
        self,
        xiaowan_target_levels_m: Sequence[float] | np.ndarray,
        nuozhadu_target_levels_m: Sequence[float] | np.ndarray,
        *,
        target_reached_atol_m: float = 1.0e-8,
    ) -> np.ndarray:
        """Return row-major joint actions whose requested targets are reachable.

        The calculation is side-effect free and reuses the shared vectorized
        reservoir transition. XiaoWan's realized release is routed before the
        NuoZhaDu feasibility calculation, exactly as in :meth:`step`.
        """
        levels_1 = np.asarray(xiaowan_target_levels_m, dtype=float)
        levels_2 = np.asarray(nuozhadu_target_levels_m, dtype=float)
        for name, levels, spec in (
            ("XiaoWan", levels_1, self.reservoir_1),
            ("NuoZhaDu", levels_2, self.reservoir_2),
        ):
            if (
                levels.ndim != 1
                or len(levels) == 0
                or np.any(~np.isfinite(levels))
                or np.any(np.diff(levels) <= 0.0)
                or levels[0] < spec.dead_level_m
                or levels[-1] > spec.normal_level_m
            ):
                raise CascadeSpecError(
                    f"{name} feasible-action target grid must be finite, increasing, "
                    "nonempty, and inside the operating bounds."
                )
        if (
            isinstance(target_reached_atol_m, bool)
            or not np.isfinite(target_reached_atol_m)
            or target_reached_atol_m < 0.0
        ):
            raise CascadeSpecError(
                "target_reached_atol_m must be finite and nonnegative."
            )
        if not 0 <= self.index < len(self.local_inflows_m3s):
            raise CascadeSpecError(
                "Cannot calculate an action mask outside the inflow sequence."
            )
        if (
            self.storages_1e9m3.shape != (2,)
            or np.any(~np.isfinite(self.storages_1e9m3))
        ):
            raise CascadeSpecError(
                "Reset the cascade environment before requesting an action mask."
            )

        seconds = self._seconds_this_step()
        inflow_1, inflow_2_local = self.local_inflows_m3s[self.index]
        upstream = _transition_arrays(
            self.reservoir_1,
            float(self.storages_1e9m3[0]),
            float(inflow_1),
            levels_1,
            seconds,
        )
        routed = np.asarray(
            [self.router.route(float(value)) for value in upstream.total_discharge_m3s],
            dtype=float,
        )
        downstream = _transition_arrays(
            self.reservoir_2,
            float(self.storages_1e9m3[1]),
            float(inflow_2_local) + routed[:, None],
            levels_2[None, :],
            seconds,
        )
        upstream_reached = np.isclose(
            upstream.end_level_m,
            levels_1,
            rtol=0.0,
            atol=float(target_reached_atol_m),
        )
        downstream_reached = np.isclose(
            downstream.end_level_m,
            levels_2[None, :],
            rtol=0.0,
            atol=float(target_reached_atol_m),
        )
        month_index = self._month() - 1
        monthly_max = self.monthly_max_target_levels_m[month_index]
        within_monthly_limits_1 = levels_1 <= monthly_max[0] + target_reached_atol_m
        within_monthly_limits_2 = levels_2 <= monthly_max[1] + target_reached_atol_m
        joint = (
            upstream_reached[:, None]
            & downstream_reached
            & within_monthly_limits_1[:, None]
            & within_monthly_limits_2[None, :]
        )

        current_levels = (
            self.reservoir_1.level_from_storage(float(self.storages_1e9m3[0])),
            self.reservoir_2.level_from_storage(float(self.storages_1e9m3[1])),
        )
        current_1 = np.flatnonzero(
            np.isclose(levels_1, current_levels[0], rtol=0.0, atol=float(target_reached_atol_m))
        )
        current_2 = np.flatnonzero(
            np.isclose(levels_2, current_levels[1], rtol=0.0, atol=float(target_reached_atol_m))
        )
        current_within_limits = bool(
            current_levels[0] <= monthly_max[0] + target_reached_atol_m
            and current_levels[1] <= monthly_max[1] + target_reached_atol_m
        )
        if (
            current_within_limits
            and len(current_1)
            and len(current_2)
            and not bool(joint[int(current_1[0]), int(current_2[0])])
        ):
            raise CascadeSpecError(
                "The current-level hold action must remain physically feasible."
            )
        mask = joint.reshape(-1).astype(np.bool_, copy=False)
        if not np.any(mask):
            raise CascadeSpecError("The current physical state has no feasible joint action.")
        return mask.copy()

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict]:
        targets = np.asarray(action, dtype=float)
        if targets.shape != (2,) or np.any(~np.isfinite(targets)):
            raise CascadeSpecError("action must contain two finite target water levels in metres.")
        monthly_max = self.monthly_max_target_levels_m[self._month() - 1]
        normal_levels = np.array(
            [self.reservoir_1.normal_level_m, self.reservoir_2.normal_level_m],
            dtype=float,
        )
        active_monthly_limit = monthly_max < normal_levels - 1.0e-8
        if np.any(active_monthly_limit & (targets > monthly_max + 1.0e-8)):
            raise CascadeSpecError(
                "Requested target levels exceed the current month's hard operating limits."
            )
        seconds = self._seconds_this_step()
        inflow_1, inflow_2_local = self.local_inflows_m3s[self.index]
        upstream = _transition(self.reservoir_1, self.storages_1e9m3[0], inflow_1, targets[0], seconds)
        routed = self.router.route(upstream.total_discharge_m3s)
        downstream = _transition(self.reservoir_2, self.storages_1e9m3[1], inflow_2_local + routed, targets[1], seconds)
        self.storages_1e9m3 = np.array([upstream.end_storage_1e9m3, downstream.end_storage_1e9m3])
        reward = 0.0
        generation_energy_gwh = 0.0
        xiaowan_failure = False
        nuozhadu_failure = False
        failed_station_months = 0
        failure_penalty_gwh = 0.0
        if self._energy_available:
            generation_energy_gwh = float(
                upstream.energy_gwh + downstream.energy_gwh
            )
            xiaowan_failure = bool(float(upstream.power_kw) < float(self.reservoir_1.firm_power_kw))
            nuozhadu_failure = bool(float(downstream.power_kw) < float(self.reservoir_2.firm_power_kw))
            failed_station_months = int(xiaowan_failure) + int(nuozhadu_failure)
            failure_penalty_gwh = (
                self.firm_power_failure_penalty_gwh * failed_station_months
            )
            reward = generation_energy_gwh - failure_penalty_gwh
        terminated = self.index >= len(self.local_inflows_m3s) - 1
        info = {
            "month": self._month(), "inflow_1_m3s": float(inflow_1), "inflow_2_local_m3s": float(inflow_2_local),
            "routed_inflow_m3s": routed, "effective_inflow_2_m3s": float(inflow_2_local + routed),
            "reservoir_1": upstream.__dict__, "reservoir_2": downstream.__dict__,
            "energy_parameters_available": self._energy_available,
            "generation_energy_gwh": generation_energy_gwh,
            "xiaowan_firm_power_failure": xiaowan_failure,
            "nuozhadu_firm_power_failure": nuozhadu_failure,
            "failed_station_months": failed_station_months,
            "firm_power_failure_penalty_gwh": self.firm_power_failure_penalty_gwh,
            "failure_penalty_gwh": failure_penalty_gwh,
            "penalty_energy_gwh": failure_penalty_gwh,
            "penalized_reward_gwh": float(reward),
            "monthly_max_target_level_1_m": float(monthly_max[0]),
            "monthly_max_target_level_2_m": float(monthly_max[1]),
        }
        if not terminated:
            self.index += 1
            next_observation = self._observation()
        else:
            next_observation = np.array([self.storages_1e9m3[0], self.storages_1e9m3[1], 0.0, 0.0, self._month() % 12 + 1], dtype=np.float32)
        return next_observation, reward, terminated, False, info


def load_specs_from_yaml(config_path: str | Path, *, require_experiment_ready: bool = True) -> tuple[ReservoirSpec, ReservoirSpec, dict]:
    """Load source-backed reservoir specs and reject a non-ready real experiment."""
    config_path = Path(config_path).resolve()
    text = config_path.read_text(encoding="utf-8")
    if require_experiment_ready and re.search(r"^\s*local_inflow_experiment_ready:\s*false\s*$", text, re.MULTILINE):
        raise CascadeSpecError("Experiment blocked: I2_local is not yet source-validated.")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        config = _parse_simple_yaml(text)
    else:
        config = yaml.safe_load(text)
    if require_experiment_ready and not config["data"]["local_inflow_experiment_ready"]:
        raise CascadeSpecError("Experiment blocked: I2_local is not yet source-validated.")
    root = config_path.parents[1]
    specs = []
    for station in ("xiaowan", "nuozhadu"):
        settings = config["reservoirs"][station]
        specs.append(ReservoirSpec.from_curve_directory(station, settings, root / settings["curve_dir"]))
    return specs[0], specs[1], config
