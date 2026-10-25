"""Closed-loop evaluation for a one-step perfect-forecast SDP policy."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    ReservoirTransition,
    _as_month_period_index,
)
from .sdp_probability import MonthlyInflowModel, SOURCES
from .sdp import reservoir_pair_sha256


BASE_SCHEDULE_COLUMNS = [
    "date",
    "start_level_1_m",
    "start_level_2_m",
    "target_level_1_m",
    "target_level_2_m",
    "reward_gwh",
    "generation_energy_gwh",
    "penalized_reward_gwh",
    "xiaowan_firm_power_failure",
    "nuozhadu_firm_power_failure",
    "failed_station_months",
    "failure_penalty_gwh",
    "penalty_energy_gwh",
    "inflow_1_m3s",
    "inflow_2_local_m3s",
    "routed_inflow_m3s",
]
TRANSITION_FIELD_NAMES = [item.name for item in fields(ReservoirTransition)]
DDP_COMPATIBLE_COLUMNS = (
    BASE_SCHEDULE_COLUMNS
    + [f"r1_{name}" for name in TRANSITION_FIELD_NAMES]
    + [f"r2_{name}" for name in TRANSITION_FIELD_NAMES]
)


def _evaluation_hash(periods: pd.PeriodIndex, flows: np.ndarray) -> str:
    digest = hashlib.sha256()
    for period, (upstream, local) in zip(periods, flows):
        digest.update(f"{period},{float(upstream).hex()},{float(local).hex()}\n".encode("utf-8"))
    return digest.hexdigest()


def evaluate_policy(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    policy,
    inflow_model: MonthlyInflowModel,
    joint: pd.DataFrame,
    initial_levels_m: Sequence[float],
    *,
    firm_power_failure_penalty_gwh: float = 0.0,
    monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
    water_balance_tolerance_m3: float = 1.0e-3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate a policy in one continuous physical trajectory.

    At each month, the realized XiaoWan natural inflow and NuoZhaDu local
    incremental inflow are classified before the decision.  The policy sees
    those current-month classes as a one-step perfect forecast.  The same two
    continuous exogenous flows are then used by ``env.step``.  The environment
    remains alive across the full evaluation period, so every action determines
    the physical storage from which the following month starts.
    """
    inflow_model._validate()
    missing = {"date", *SOURCES} - set(joint.columns)
    if missing:
        raise CascadeSpecError(f"SDP evaluation input is missing columns: {sorted(missing)}.")
    if joint.columns.duplicated().any():
        raise CascadeSpecError("SDP evaluation columns must be unique.")
    if len(initial_levels_m) != 2:
        raise CascadeSpecError("initial_levels_m must contain two values.")
    initial_levels = np.asarray(initial_levels_m, dtype=float)
    if np.any(~np.isfinite(initial_levels)) or not (
        reservoir_1.dead_level_m <= initial_levels[0] <= reservoir_1.normal_level_m
        and reservoir_2.dead_level_m <= initial_levels[1] <= reservoir_2.normal_level_m
    ):
        raise CascadeSpecError("Initial levels must be finite and within both reservoirs' operating bounds.")
    if (
        not np.isfinite(firm_power_failure_penalty_gwh)
        or firm_power_failure_penalty_gwh < 0.0
    ):
        raise CascadeSpecError(
            "Evaluation firm-power failure penalty must be finite and nonnegative."
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
            "Evaluation monthly_max_target_levels_m must be a finite [12, 2] table."
        )
    if np.any(
        (monthly_max[:, 0] < reservoir_1.dead_level_m)
        | (monthly_max[:, 0] > reservoir_1.normal_level_m)
        | (monthly_max[:, 1] < reservoir_2.dead_level_m)
        | (monthly_max[:, 1] > reservoir_2.normal_level_m)
    ):
        raise CascadeSpecError(
            "Evaluation monthly maximum target levels must lie within physical bounds."
        )
    if not np.isfinite(water_balance_tolerance_m3) or water_balance_tolerance_m3 < 0.0:
        raise CascadeSpecError("Evaluation water-balance tolerance must be finite and nonnegative.")
    if hasattr(policy, "metadata"):
        training_hash = policy.metadata.get("training_inflow_sha256")
        if training_hash is not None and training_hash != inflow_model.metadata["training_sha256"]:
            raise CascadeSpecError("SDP policy and inflow probability artifacts came from different training data.")
        physics_hash = policy.metadata.get("reservoir_physics_sha256")
        if physics_hash is not None and physics_hash != reservoir_pair_sha256(reservoir_1, reservoir_2):
            raise CascadeSpecError("SDP policy was trained with different reservoir physics.")
        policy_penalty = policy.metadata.get("firm_power_failure_penalty_gwh")
        if policy_penalty is not None and not np.isclose(
            float(policy_penalty),
            firm_power_failure_penalty_gwh,
            rtol=0.0,
            atol=0.0,
        ):
            raise CascadeSpecError("SDP policy and evaluation firm-power penalties differ.")

    periods = _as_month_period_index(joint["date"])
    flows = joint.loc[:, list(SOURCES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if len(flows) == 0 or np.any(~np.isfinite(flows)) or np.any(flows < 0.0):
        raise CascadeSpecError("SDP evaluation flows must be nonempty, finite, and nonnegative.")
    if len(periods) != len(flows):
        raise CascadeSpecError("SDP evaluation dates and flows must align.")
    expected_periods = pd.period_range(periods[0], periods[-1], freq="M")
    if len(periods) != len(expected_periods) or not periods.equals(expected_periods):
        raise CascadeSpecError("SDP evaluation dates must be unique, ordered, and continuous months.")

    env = CascadeReservoirEnv(
        reservoir_1,
        reservoir_2,
        flows,
        timestamps=periods,
        initial_levels_m=(float(initial_levels[0]), float(initial_levels[1])),
        # The shared environment retains the legacy deficit-energy reward.
        # Evaluate physical generation there, then apply the V6 failed-station-
        # month objective explicitly below.
        firm_power_failure_penalty_gwh=0.0,
        require_energy_parameters=True,
    )
    env.reset()
    rows: list[dict] = []
    audit_rows: list[dict] = []
    out_of_range_counts = np.zeros((2, 2), dtype=int)  # source, below/above
    target_unreached = np.zeros(2, dtype=int)
    maximum_balance_error = 0.0
    extrapolated_tailwater_steps = np.zeros(2, dtype=int)
    previous_end_storages: tuple[float, float] | None = None
    previous_end_levels: tuple[float, float] | None = None
    storage_jumps_m3: list[tuple[float, float]] = []
    level_jumps_m: list[tuple[float, float]] = []
    firm_power_months = np.zeros(2, dtype=int)
    failed_station_months = np.zeros(2, dtype=int)
    joint_satisfied_months = 0

    for index, period in enumerate(periods):
        month = int(period.month)
        current_flows = (float(flows[index, 0]), float(flows[index, 1]))
        current_classes = inflow_model.classify(month, current_flows)
        start_storages = (float(env.storages_1e9m3[0]), float(env.storages_1e9m3[1]))
        start_levels = (
            reservoir_1.level_from_storage(start_storages[0]),
            reservoir_2.level_from_storage(start_storages[1]),
        )
        prior_end_storages = previous_end_storages
        prior_end_levels = previous_end_levels
        if prior_end_storages is None or prior_end_levels is None:
            current_storage_jumps_m3 = (None, None)
            current_level_jumps_m = (None, None)
        else:
            current_storage_jumps_m3 = (
                (start_storages[0] - prior_end_storages[0]) * 1.0e9,
                (start_storages[1] - prior_end_storages[1]) * 1.0e9,
            )
            current_level_jumps_m = (
                float(start_levels[0]) - prior_end_levels[0],
                float(start_levels[1]) - prior_end_levels[1],
            )
            storage_jumps_m3.append(current_storage_jumps_m3)
            level_jumps_m.append(current_level_jumps_m)
        targets = policy.action(month, start_levels, current_classes)
        if len(targets) != 2 or np.any(~np.isfinite(np.asarray(targets, dtype=float))):
            raise CascadeSpecError("SDP policy must return two finite target levels.")
        if np.any(np.asarray(targets, dtype=float) > monthly_max[month - 1] + 1.0e-8):
            raise CascadeSpecError(
                "SDP policy returned a target above the configured monthly maximum level."
            )

        _, generation_energy, terminated, truncated, info = env.step(targets)
        if truncated:
            raise CascadeSpecError("Unexpected truncation in deterministic SDP evaluation.")
        realized_flows = (float(info["inflow_1_m3s"]), float(info["inflow_2_local_m3s"]))
        if realized_flows != current_flows:
            raise CascadeSpecError("SDP forecast classes and physical transition flows are misaligned.")
        bounds = inflow_model.edges[:, month - 1]
        below = (realized_flows[0] < bounds[0, 0], realized_flows[1] < bounds[1, 0])
        above = (realized_flows[0] > bounds[0, -1], realized_flows[1] > bounds[1, -1])
        for source in range(2):
            out_of_range_counts[source, 0] += int(below[source])
            out_of_range_counts[source, 1] += int(above[source])

        upstream = info["reservoir_1"]
        downstream = info["reservoir_2"]
        xiaowan_failure = bool(float(upstream["firm_deficit_gwh"]) > 0.0)
        nuozhadu_failure = bool(float(downstream["firm_deficit_gwh"]) > 0.0)
        monthly_failed_stations = int(xiaowan_failure) + int(nuozhadu_failure)
        failure_penalty = (
            float(firm_power_failure_penalty_gwh) * monthly_failed_stations
        )
        penalized_reward = float(generation_energy) - failure_penalty
        failed_station_months[0] += int(xiaowan_failure)
        failed_station_months[1] += int(nuozhadu_failure)
        firm_power_months[0] += int(not xiaowan_failure)
        firm_power_months[1] += int(not nuozhadu_failure)
        joint_satisfied_months += int(
            not xiaowan_failure and not nuozhadu_failure
        )
        for source, details in enumerate((upstream, downstream)):
            target_unreached[source] += int(not np.isclose(
                float(details["end_level_m"]), float(details["requested_target_level_m"]), rtol=0.0, atol=1.0e-8
            ))
            extrapolated_tailwater_steps[source] += int(details["tailwater_discharge_extrapolated"])
            maximum_balance_error = max(maximum_balance_error, abs(float(details["water_balance_error_m3"])))
        previous_end_storages = (
            float(upstream["end_storage_1e9m3"]),
            float(downstream["end_storage_1e9m3"]),
        )
        previous_end_levels = (
            float(upstream["end_level_m"]),
            float(downstream["end_level_m"]),
        )

        row = {
            "date": f"{period.year:04d}-{period.month:02d}-01",
            "start_level_1_m": float(start_levels[0]),
            "start_level_2_m": float(start_levels[1]),
            "target_level_1_m": float(targets[0]),
            "target_level_2_m": float(targets[1]),
            # Keep reward_gwh as actual cascade generation for DDP-compatible
            # consumers. The Bellman/evaluation objective is explicit below.
            "reward_gwh": float(generation_energy),
            "generation_energy_gwh": float(generation_energy),
            "penalized_reward_gwh": penalized_reward,
            "xiaowan_firm_power_failure": xiaowan_failure,
            "nuozhadu_firm_power_failure": nuozhadu_failure,
            "failed_station_months": monthly_failed_stations,
            "failure_penalty_gwh": failure_penalty,
            "penalty_energy_gwh": failure_penalty,
            "inflow_1_m3s": realized_flows[0],
            "inflow_2_local_m3s": realized_flows[1],
            "routed_inflow_m3s": float(info["routed_inflow_m3s"]),
        }
        row.update({f"r1_{key}": value for key, value in upstream.items()})
        row.update({f"r2_{key}": value for key, value in downstream.items()})
        rows.append(row)
        audit_rows.append({
            "date": row["date"],
            "month": month,
            "decision_start_level_1_m": float(start_levels[0]),
            "decision_start_level_2_m": float(start_levels[1]),
            "decision_start_storage_1_1e9m3": start_storages[0],
            "decision_start_storage_2_1e9m3": start_storages[1],
            "previous_end_storage_1_1e9m3": None if prior_end_storages is None else prior_end_storages[0],
            "previous_end_storage_2_1e9m3": None if prior_end_storages is None else prior_end_storages[1],
            "xiaowan_inter_month_storage_jump_m3": current_storage_jumps_m3[0],
            "nuozhadu_inter_month_storage_jump_m3": current_storage_jumps_m3[1],
            "xiaowan_inter_month_level_jump_m": current_level_jumps_m[0],
            "nuozhadu_inter_month_level_jump_m": current_level_jumps_m[1],
            "current_xiaowan_inflow_class_at_decision": int(current_classes[0]),
            "current_nuozhadu_local_inflow_class_at_decision": int(current_classes[1]),
            "realized_xiaowan_natural_inflow_m3s": realized_flows[0],
            "realized_nuozhadu_local_inflow_m3s": realized_flows[1],
            "target_level_1_m": float(targets[0]),
            "target_level_2_m": float(targets[1]),
            "xiaowan_firm_power_failure": xiaowan_failure,
            "nuozhadu_firm_power_failure": nuozhadu_failure,
            "failed_station_months": monthly_failed_stations,
            "xiaowan_inflow_below_training_range": bool(below[0]),
            "xiaowan_inflow_above_training_range": bool(above[0]),
            "nuozhadu_local_inflow_below_training_range": bool(below[1]),
            "nuozhadu_local_inflow_above_training_range": bool(above[1]),
        })
        if terminated != (index == len(periods) - 1):
            raise CascadeSpecError("Cascade environment terminated at an unexpected evaluation month.")

    schedule = pd.DataFrame(rows, columns=DDP_COMPATIBLE_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    storage_jumps = np.asarray(storage_jumps_m3, dtype=float).reshape(-1, 2)
    level_jumps = np.asarray(level_jumps_m, dtype=float).reshape(-1, 2)
    maximum_storage_jumps = (
        np.max(np.abs(storage_jumps), axis=0) if len(storage_jumps) else np.zeros(2, dtype=float)
    )
    maximum_level_jumps = (
        np.max(np.abs(level_jumps), axis=0) if len(level_jumps) else np.zeros(2, dtype=float)
    )
    maximum_cascade_storage_jump = (
        float(np.max(np.abs(np.sum(storage_jumps, axis=1)))) if len(storage_jumps) else 0.0
    )
    metadata = getattr(policy, "metadata", {})
    summary = {
        "method": "SDP",
        "information_set": "month_levels_current_source_classes_one_step_perfect_forecast",
        "closed_loop_state_storage": "continuous_physical_storage_grid_used_only_for_policy_lookup",
        "months": len(schedule),
        "grid_step_m": metadata.get("water_level_step_m"),
        "state_grid_shape": metadata.get("state_grid_shape"),
        "joint_action_count": metadata.get("joint_action_count"),
        "solver_variant": metadata.get("solver_variant"),
        "continuation_value_lookup": metadata.get("continuation_value_lookup"),
        "policy_state_lookup": metadata.get("policy_state_lookup"),
        "objective_gwh": float(schedule["penalized_reward_gwh"].sum()),
        "total_generation_gwh": float(schedule["generation_energy_gwh"].sum()),
        "total_xiaowan_deficit_energy_gwh": float(
            schedule["r1_firm_deficit_gwh"].sum()
        ),
        "total_nuozhadu_deficit_energy_gwh": float(
            schedule["r2_firm_deficit_gwh"].sum()
        ),
        "firm_power_failure_penalty_gwh": float(
            firm_power_failure_penalty_gwh
        ),
        "objective_definition": (
            "generation_energy_gwh_minus_gwh_equivalent_penalty_per_"
            "failed_station_month"
        ),
        "failure_definition": (
            "station_specific_monthly_power_strictly_below_firm_power"
        ),
        "deficit_energy_role": "diagnostic_only_not_in_sdp_objective",
        "firm_power_reliability": {
            "xiaowan": float(firm_power_months[0] / len(schedule)),
            "nuozhadu": float(firm_power_months[1] / len(schedule)),
        },
        "firm_power_satisfied_months": {
            "xiaowan": int(firm_power_months[0]),
            "nuozhadu": int(firm_power_months[1]),
        },
        "firm_power_failed_months": {
            "xiaowan": int(failed_station_months[0]),
            "nuozhadu": int(failed_station_months[1]),
        },
        "total_failed_station_months": int(failed_station_months.sum()),
        "total_failure_penalty_gwh": float(schedule["failure_penalty_gwh"].sum()),
        "total_penalty_energy_gwh": float(schedule["penalty_energy_gwh"].sum()),
        "joint_firm_power_satisfaction": float(
            joint_satisfied_months / len(schedule)
        ),
        "joint_firm_power_satisfied_months": int(joint_satisfied_months),
        "monthly_max_target_levels_m": monthly_max.tolist(),
        "monthly_max_semantics": "applies_to_end_of_month_target_level",
        "evaluation_inflow_sha256": _evaluation_hash(periods, flows),
        "training_inflow_sha256": inflow_model.metadata["training_sha256"],
        "forecast_timing": "current_month_source_classes_known_before_policy_action",
        "forecast_flow_application": "same_realized_continuous_source_flows_used_in_physical_transition",
        "nuozhadu_forecast_inflow_definition": "local_incremental_inflow",
        "nuozhadu_total_inflow_formation": "local_incremental_inflow_plus_routed_xiaowan_total_discharge_in_environment",
        "out_of_training_range_counts": {
            "xiaowan_below": int(out_of_range_counts[0, 0]),
            "xiaowan_above": int(out_of_range_counts[0, 1]),
            "nuozhadu_local_below": int(out_of_range_counts[1, 0]),
            "nuozhadu_local_above": int(out_of_range_counts[1, 1]),
        },
        "target_level_unreached_steps": {
            "xiaowan": int(target_unreached[0]),
            "nuozhadu": int(target_unreached[1]),
        },
        "tailwater_extrapolated_steps": {
            "xiaowan": int(extrapolated_tailwater_steps[0]),
            "nuozhadu": int(extrapolated_tailwater_steps[1]),
        },
        "max_abs_water_balance_error_m3": maximum_balance_error,
        "water_balance_tolerance_m3": float(water_balance_tolerance_m3),
        "water_balance_within_tolerance": maximum_balance_error <= water_balance_tolerance_m3,
        "inter_month_storage_continuity": {
            "comparisons": int(len(storage_jumps)),
            "xiaowan_max_abs_jump_m3": float(maximum_storage_jumps[0]),
            "nuozhadu_max_abs_jump_m3": float(maximum_storage_jumps[1]),
            "cascade_max_abs_net_jump_m3": maximum_cascade_storage_jump,
            "xiaowan_max_abs_level_jump_m": float(maximum_level_jumps[0]),
            "nuozhadu_max_abs_level_jump_m": float(maximum_level_jumps[1]),
            "tolerance_m3": float(water_balance_tolerance_m3),
            "within_tolerance": bool(
                np.all(maximum_storage_jumps <= water_balance_tolerance_m3)
            ),
        },
        "inter_month_storage_continuity_within_tolerance": bool(
            np.all(maximum_storage_jumps <= water_balance_tolerance_m3)
        ),
    }
    return schedule, audit, summary
