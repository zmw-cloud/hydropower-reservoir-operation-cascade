"""Greedy closed-loop trajectories for the P4 cascade Double DQN.

The helper in this module evaluates one seed at a time.  It resets one shared
physical environment once, observes the current month's two external inflows
before every decision, and carries the simulator's realized storage forward
for the complete validation or evaluation sequence.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .dqn_adapter import (
    INFLOW_COLUMNS,
    STATE_ORDER,
    CascadeDQNAdapter,
    JointActionGrid,
    StateNormalizer,
    normalized_inflow_sha256,
)
from .dqn_agent import DoubleDQNAgent
from .environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
    _as_month_period_index,
)
from .sdp import reservoir_pair_sha256
from .sdp_evaluation import DDP_COMPATIBLE_COLUMNS


DQN_INFORMATION_SET = (
    "month_levels_current_external_inflows_one_step_perfect_forecast"
)
NORMALIZED_STATE_COLUMNS = tuple(
    f"normalized_state_{component}" for component in STATE_ORDER
)
DQN_DIAGNOSTIC_COLUMNS = (
    "year_index",
    "calendar_year",
    "calendar_month",
    "action_index",
    "xiaowan_action_grid_index",
    "nuozhadu_action_grid_index",
    "feasible_action_count",
    "selected_action_feasible",
    *NORMALIZED_STATE_COLUMNS,
    "xiaowan_target_reached",
    "nuozhadu_target_reached",
    "either_reservoir_target_unreached",
    "xiaowan_target_error_m",
    "nuozhadu_target_error_m",
    "xiaowan_firm_power_satisfied",
    "nuozhadu_firm_power_satisfied",
    "cascade_firm_power_satisfied",
)
DQN_EVALUATION_COLUMNS = [
    *DDP_COMPATIBLE_COLUMNS,
    *DQN_DIAGNOSTIC_COLUMNS,
]


def _validated_trajectory_inputs(
    joint: pd.DataFrame,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    initial_levels_m: Sequence[float],
    water_balance_tolerance_m3: float,
) -> tuple[pd.PeriodIndex, np.ndarray, tuple[float, float]]:
    if not isinstance(joint, pd.DataFrame):
        raise CascadeSpecError("DQN trajectory inflows must be a pandas DataFrame.")
    if len(joint) == 0:
        raise CascadeSpecError("DQN trajectory inflows must contain at least one month.")
    missing = {"date", *INFLOW_COLUMNS} - set(joint.columns)
    if missing or joint.columns.duplicated().any():
        raise CascadeSpecError(
            "DQN trajectory inflows require unique date/source columns; "
            f"missing: {sorted(missing)}."
        )
    periods = _as_month_period_index(joint["date"])
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    if not periods.equals(expected):
        raise CascadeSpecError(
            "DQN trajectory dates must be unique, ordered, and continuous months."
        )
    flows = (
        joint.loc[:, list(INFLOW_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if len(flows) == 0 or np.any(~np.isfinite(flows)) or np.any(flows < 0.0):
        raise CascadeSpecError(
            "DQN trajectory inflows must be nonempty, finite, and nonnegative."
        )

    initial_levels = np.asarray(initial_levels_m, dtype=float)
    if initial_levels.shape != (2,) or np.any(~np.isfinite(initial_levels)):
        raise CascadeSpecError("initial_levels_m must contain two finite values.")
    if not (
        reservoir_1.dead_level_m
        <= initial_levels[0]
        <= reservoir_1.normal_level_m
        and reservoir_2.dead_level_m
        <= initial_levels[1]
        <= reservoir_2.normal_level_m
    ):
        raise CascadeSpecError(
            "Initial levels must lie within both reservoirs' operating bounds."
        )
    if (
        isinstance(water_balance_tolerance_m3, bool)
        or not np.isfinite(water_balance_tolerance_m3)
        or water_balance_tolerance_m3 < 0.0
    ):
        raise CascadeSpecError(
            "water_balance_tolerance_m3 must be finite and nonnegative."
        )
    return periods, flows, (float(initial_levels[0]), float(initial_levels[1]))


def _validated_seed(agent: DoubleDQNAgent, seed: int | None) -> int:
    if not isinstance(agent, DoubleDQNAgent):
        raise TypeError("agent must be a DoubleDQNAgent.")
    if seed is None:
        return int(agent.seed)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    result = int(seed)
    if result != agent.seed:
        raise CascadeSpecError(
            "The trajectory seed must match the independently trained agent seed."
        )
    return result


def _validated_training_hash(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("training_inflow_sha256 must be a string.")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise CascadeSpecError(
            "training_inflow_sha256 must be a 64-character hexadecimal SHA-256."
        )
    return normalized


def _physical_row(
    period: pd.Period,
    state: np.ndarray,
    info: dict[str, Any],
    *,
    trajectory_index: int,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
) -> dict[str, Any]:
    upstream = info["reservoir_1"]
    downstream = info["reservoir_2"]
    raw_state = np.asarray(info["state_raw"], dtype=float)
    if raw_state.shape != (len(STATE_ORDER),):
        raise CascadeSpecError("DQN adapter returned an invalid raw decision state.")

    row: dict[str, Any] = {
        "date": f"{period.year:04d}-{period.month:02d}-01",
        "year_index": int(trajectory_index // 12 + 1),
        "calendar_year": int(period.year),
        "calendar_month": int(period.month),
        "start_level_1_m": float(raw_state[1]),
        "start_level_2_m": float(raw_state[2]),
        "target_level_1_m": float(upstream["requested_target_level_m"]),
        "target_level_2_m": float(downstream["requested_target_level_m"]),
        "reward_gwh": float(info["generation_energy_gwh"]),
        "penalized_reward_gwh": float(info["penalized_reward_gwh"]),
        "failure_penalty_gwh": float(info["failure_penalty_gwh"]),
        "failed_station_months": int(info["failed_station_months"]),
        "xiaowan_firm_power_failure": bool(info["xiaowan_firm_power_failure"]),
        "nuozhadu_firm_power_failure": bool(info["nuozhadu_firm_power_failure"]),
        "inflow_1_m3s": float(info["inflow_1_m3s"]),
        "inflow_2_local_m3s": float(info["inflow_2_local_m3s"]),
        "routed_inflow_m3s": float(info["routed_inflow_m3s"]),
        "action_index": int(info["action_index"]),
        "xiaowan_action_grid_index": int(info["xiaowan_action_grid_index"]),
        "nuozhadu_action_grid_index": int(info["nuozhadu_action_grid_index"]),
        "feasible_action_count": int(info["feasible_action_count"]),
        "selected_action_feasible": bool(info["selected_action_feasible"]),
        "xiaowan_target_reached": bool(info["xiaowan_target_reached"]),
        "nuozhadu_target_reached": bool(info["nuozhadu_target_reached"]),
        "either_reservoir_target_unreached": bool(
            not info["xiaowan_target_reached"]
            or not info["nuozhadu_target_reached"]
        ),
        "xiaowan_target_error_m": float(info["xiaowan_target_error_m"]),
        "nuozhadu_target_error_m": float(info["nuozhadu_target_error_m"]),
        "xiaowan_firm_power_satisfied": bool(
            float(upstream["power_kw"]) >= float(reservoir_1.firm_power_kw)
        ),
        "nuozhadu_firm_power_satisfied": bool(
            float(downstream["power_kw"]) >= float(reservoir_2.firm_power_kw)
        ),
        "cascade_firm_power_satisfied": bool(
            float(upstream["power_kw"]) + float(downstream["power_kw"])
            >= float(reservoir_1.firm_power_kw + reservoir_2.firm_power_kw)
        ),
    }
    row.update({f"r1_{key}": value for key, value in upstream.items()})
    row.update({f"r2_{key}": value for key, value in downstream.items()})
    row.update(
        {
            column: float(value)
            for column, value in zip(NORMALIZED_STATE_COLUMNS, state, strict=True)
        }
    )
    return row


def evaluate_greedy_trajectory(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    agent: DoubleDQNAgent,
    normalizer: StateNormalizer,
    action_grid: JointActionGrid,
    joint: pd.DataFrame,
    initial_levels_m: Sequence[float],
    *,
    training_inflow_sha256: str,
    seed: int | None = None,
    reward_reference_days: float = 30.4,
    reward_clip: Sequence[float] | None = None,
    firm_power_failure_penalty_gwh: float = 0.0,
    monthly_max_target_levels_m: Sequence[Sequence[float]] | None = None,
    water_balance_tolerance_m3: float = 1.0e-3,
    trajectory_kind: str = "evaluation",
    collect_rows: bool = True,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Run one seed's greedy policy over one continuous physical trajectory.

    ``joint`` supplies the current month's XiaoWan natural inflow and
    NuoZhaDu local incremental inflow to the five-component state before the
    action.  The routed XiaoWan release remains action-dependent and is only
    formed inside :class:`CascadeReservoirEnv`.  Setting ``collect_rows`` to
    false keeps validation memory use small while preserving the same summary.
    """
    trajectory_timer_started = time.perf_counter()
    if not isinstance(collect_rows, bool):
        raise TypeError("collect_rows must be a boolean.")
    if trajectory_kind not in {"validation", "evaluation"}:
        raise CascadeSpecError(
            "trajectory_kind must be 'validation' or 'evaluation'."
        )
    trajectory_seed = _validated_seed(agent, seed)
    training_hash = _validated_training_hash(training_inflow_sha256)
    periods, flows, initial_levels = _validated_trajectory_inputs(
        joint,
        reservoir_1,
        reservoir_2,
        initial_levels_m,
        water_balance_tolerance_m3,
    )
    if agent.state_dimension != len(STATE_ORDER):
        raise CascadeSpecError("Double DQN must use the five-component P4 V1 state.")
    if agent.action_count != action_grid.action_count:
        raise CascadeSpecError(
            "Double DQN output size does not match the joint target-level action grid."
        )

    env = CascadeReservoirEnv(
        reservoir_1,
        reservoir_2,
        flows,
        timestamps=periods,
        initial_levels_m=initial_levels,
        firm_power_failure_penalty_gwh=firm_power_failure_penalty_gwh,
        monthly_max_target_levels_m=monthly_max_target_levels_m,
        require_energy_parameters=True,
    )
    adapter = CascadeDQNAdapter(
        env,
        normalizer,
        action_grid,
        reward_reference_days=reward_reference_days,
        reward_clip=reward_clip,
    )

    rows: list[dict[str, Any]] | None = [] if collect_rows else None
    total_generation_gwh = 0.0
    total_penalized_objective_gwh = 0.0
    total_failure_penalty_gwh = 0.0
    failed_station_months = 0
    firm_months = np.zeros(3, dtype=np.int64)
    maximum_balance_error = 0.0
    feasible_action_counts: list[int] = []
    state, _ = adapter.reset(seed=trajectory_seed)

    online_was_training = agent.online_network.training
    target_was_training = agent.target_network.training
    agent.online_network.eval()
    agent.target_network.eval()
    try:
        with torch.no_grad():
            for index, period in enumerate(periods):
                action_mask = adapter.get_feasible_action_mask()
                action_index = agent.select_action(
                    state, greedy=True, action_mask=action_mask
                )
                next_state, _, terminated, truncated, info = adapter.step(action_index)
                if truncated:
                    raise CascadeSpecError(
                        "Unexpected truncation in the deterministic DQN trajectory."
                    )
                if terminated != (index == len(periods) - 1):
                    raise CascadeSpecError(
                        "Cascade environment terminated at an unexpected DQN trajectory month."
                    )

                upstream = info["reservoir_1"]
                downstream = info["reservoir_2"]
                upstream_power = float(upstream["power_kw"])
                downstream_power = float(downstream["power_kw"])
                total_generation_gwh += float(info["generation_energy_gwh"])
                total_penalized_objective_gwh += float(
                    info["penalized_reward_gwh"]
                )
                total_failure_penalty_gwh += float(info["failure_penalty_gwh"])
                failed_station_months += int(info["failed_station_months"])
                firm_months[0] += int(upstream_power >= float(reservoir_1.firm_power_kw))
                firm_months[1] += int(downstream_power >= float(reservoir_2.firm_power_kw))
                firm_months[2] += int(
                    upstream_power + downstream_power
                    >= float(reservoir_1.firm_power_kw)
                    + float(reservoir_2.firm_power_kw)
                )
                maximum_balance_error = max(
                    maximum_balance_error,
                    abs(float(upstream["water_balance_error_m3"])),
                    abs(float(downstream["water_balance_error_m3"])),
                )
                feasible_action_counts.append(int(info["feasible_action_count"]))
                if rows is not None:
                    rows.append(
                        _physical_row(
                            period,
                            state,
                            info,
                            trajectory_index=index,
                            reservoir_1=reservoir_1,
                            reservoir_2=reservoir_2,
                        )
                    )

                if terminated:
                    if next_state is not None:
                        raise CascadeSpecError(
                            "A terminal DQN transition unexpectedly returned a next state."
                        )
                else:
                    if next_state is None:
                        raise CascadeSpecError(
                            "A nonterminal DQN transition did not return its real next state."
                        )
                    state = next_state
    finally:
        agent.online_network.train(online_was_training)
        agent.target_network.train(target_was_training)

    months = len(periods)
    diagnostics = adapter.episode_diagnostics
    reliability = {
        "xiaowan": float(firm_months[0] / months),
        "nuozhadu": float(firm_months[1] / months),
        "cascade": float(firm_months[2] / months),
    }
    trajectory_wall_time_seconds = time.perf_counter() - trajectory_timer_started
    summary: dict[str, Any] = {
        "method": "Double DQN",
        "information_set": DQN_INFORMATION_SET,
        "forecast_timing": "current_month_external_inflows_known_before_policy_action",
        "nuozhadu_forecast_inflow_definition": "local_incremental_inflow",
        "nuozhadu_total_inflow_formation": (
            "local_incremental_inflow_plus_routed_xiaowan_total_discharge_in_environment"
        ),
        "closed_loop_state_storage": "continuous_physical_storage",
        "seed": trajectory_seed,
        "months": months,
        "years": float(months / 12.0),
        "action_grid_step_m": action_grid.step_m,
        "joint_action_count": action_grid.action_count,
        "action_mask_mode": "environment_transition_reachability_and_monthly_limits_v2",
        "mean_feasible_action_count": float(np.mean(feasible_action_counts)),
        "min_feasible_action_count": int(min(feasible_action_counts)),
        "total_cascade_generation_gwh": float(total_generation_gwh),
        "mean_annual_cascade_generation_gwh": float(
            total_generation_gwh * 12.0 / months
        ),
        "total_penalized_objective_gwh": float(total_penalized_objective_gwh),
        "mean_annual_penalized_objective_gwh": float(
            total_penalized_objective_gwh * 12.0 / months
        ),
        "firm_power_failure_penalty_gwh": float(
            firm_power_failure_penalty_gwh
        ),
        "total_failure_penalty_gwh": float(total_failure_penalty_gwh),
        "failed_station_months": int(failed_station_months),
        "objective_definition": (
            "generation_energy_gwh_minus_gwh_equivalent_penalty_per_"
            "failed_station_month"
        ),
        "firm_power_reliability": reliability,
        "target_unreached_count": diagnostics["target_unreached_count"],
        "target_unreached_rate": diagnostics["target_unreached_rate"],
        "max_abs_water_balance_error_m3": float(maximum_balance_error),
        "water_balance_tolerance_m3": float(water_balance_tolerance_m3),
        "water_balance_within_tolerance": bool(
            maximum_balance_error <= water_balance_tolerance_m3
        ),
        "normalization": normalizer.state_dict(),
        "reward_scaling": adapter.reward_metadata,
        "training_inflow_sha256": training_hash,
        "trajectory_kind": trajectory_kind,
        "trajectory_inflow_sha256": normalized_inflow_sha256(joint),
        "trajectory_wall_time_seconds": float(trajectory_wall_time_seconds),
        "reservoir_physics_sha256": reservoir_pair_sha256(
            reservoir_1, reservoir_2
        ),
    }
    summary[f"{trajectory_kind}_inflow_sha256"] = summary[
        "trajectory_inflow_sha256"
    ]
    summary[f"{trajectory_kind}_wall_time_seconds"] = float(
        trajectory_wall_time_seconds
    )
    schedule = (
        pd.DataFrame(rows, columns=DQN_EVALUATION_COLUMNS)
        if rows is not None
        else None
    )
    return schedule, summary


def summarize_monthly_physical_trajectory(
    schedule: pd.DataFrame,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build common 400-row yearly and aggregate metrics from monthly rows."""
    required = {
        "date",
        "reward_gwh",
        "penalized_reward_gwh",
        "failure_penalty_gwh",
        "failed_station_months",
        "r1_requested_target_level_m",
        "r1_end_level_m",
        "r1_total_discharge_m3s",
        "r1_spill_discharge_m3s",
        "r1_power_kw",
        "r1_energy_gwh",
        "r1_target_level_clipped",
        "r1_water_balance_error_m3",
        "r2_requested_target_level_m",
        "r2_end_level_m",
        "r2_total_discharge_m3s",
        "r2_spill_discharge_m3s",
        "r2_power_kw",
        "r2_energy_gwh",
        "r2_target_level_clipped",
        "r2_water_balance_error_m3",
    }
    missing = required - set(schedule.columns)
    if missing:
        raise CascadeSpecError(
            f"Monthly physical trajectory is missing columns: {sorted(missing)}"
        )
    if len(schedule) == 0 or len(schedule) % 12 != 0:
        raise CascadeSpecError(
            "Monthly physical trajectory must contain complete calendar years."
        )
    frame = schedule.copy().reset_index(drop=True)
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise CascadeSpecError(
            "Monthly physical trajectory contains duplicate columns: "
            f"{duplicated}."
        )
    periods = _as_month_period_index(frame["date"])
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    if (
        not periods.equals(expected)
        or periods[0].month != 1
        or periods[-1].month != 12
    ):
        raise CascadeSpecError(
            "Monthly physical trajectory must be continuous January-to-December years."
        )
    numeric_columns = sorted(required - {"date", "r1_target_level_clipped", "r2_target_level_clipped"})
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any().any() or np.any(
        ~np.isfinite(frame[numeric_columns].to_numpy(dtype=float))
    ):
        raise CascadeSpecError("Monthly physical metrics must be finite.")
    for column in ("r1_target_level_clipped", "r2_target_level_clipped"):
        if not pd.api.types.is_bool_dtype(frame[column]):
            normalized_boolean = frame[column].astype(str).str.strip().str.lower()
            if not normalized_boolean.isin({"true", "false"}).all():
                raise CascadeSpecError(f"{column} must contain boolean values.")
            frame[column] = normalized_boolean.eq("true")

    days = np.asarray(periods.days_in_month, dtype=float)
    seconds = days * 86400.0
    frame["year_index"] = np.arange(len(frame), dtype=np.int64) // 12 + 1
    frame["calendar_year"] = np.asarray(periods.year, dtype=np.int64)
    frame["r1_spill_volume_1e9m3"] = (
        frame["r1_spill_discharge_m3s"].to_numpy(dtype=float) * seconds / 1.0e9
    )
    frame["r2_spill_volume_1e9m3"] = (
        frame["r2_spill_discharge_m3s"].to_numpy(dtype=float) * seconds / 1.0e9
    )
    frame["r1_release_volume_1e9m3"] = (
        frame["r1_total_discharge_m3s"].to_numpy(dtype=float) * seconds / 1.0e9
    )
    frame["r2_release_volume_1e9m3"] = (
        frame["r2_total_discharge_m3s"].to_numpy(dtype=float) * seconds / 1.0e9
    )
    frame["xiaowan_target_error_m"] = (
        frame["r1_end_level_m"] - frame["r1_requested_target_level_m"]
    )
    frame["nuozhadu_target_error_m"] = (
        frame["r2_end_level_m"] - frame["r2_requested_target_level_m"]
    )
    frame["xiaowan_target_unreached"] = ~np.isclose(
        frame["xiaowan_target_error_m"], 0.0, rtol=0.0, atol=1.0e-8
    )
    frame["nuozhadu_target_unreached"] = ~np.isclose(
        frame["nuozhadu_target_error_m"], 0.0, rtol=0.0, atol=1.0e-8
    )
    frame["either_reservoir_target_unreached"] = (
        frame["xiaowan_target_unreached"]
        | frame["nuozhadu_target_unreached"]
    )
    frame["xiaowan_firm_power_satisfied"] = (
        frame["r1_power_kw"] >= float(reservoir_1.firm_power_kw)
    )
    frame["nuozhadu_firm_power_satisfied"] = (
        frame["r2_power_kw"] >= float(reservoir_2.firm_power_kw)
    )
    frame["cascade_firm_power_satisfied"] = (
        frame["r1_power_kw"] + frame["r2_power_kw"]
        >= float(reservoir_1.firm_power_kw + reservoir_2.firm_power_kw)
    )

    yearly_rows: list[dict[str, Any]] = []
    for year_index, group in frame.groupby("year_index", sort=True):
        yearly_rows.append(
            {
                "year_index": int(year_index),
                "calendar_year": int(group["calendar_year"].iloc[0]),
                "start_date": str(group["date"].iloc[0]),
                "end_date": str(group["date"].iloc[-1]),
                "months": int(len(group)),
                "xiaowan_annual_generation_gwh": float(group["r1_energy_gwh"].sum()),
                "nuozhadu_annual_generation_gwh": float(group["r2_energy_gwh"].sum()),
                "cascade_annual_generation_gwh": float(group["reward_gwh"].sum()),
                "annual_penalized_objective_gwh": float(
                    group["penalized_reward_gwh"].sum()
                ),
                "annual_failure_penalty_gwh": float(
                    group["failure_penalty_gwh"].sum()
                ),
                "annual_failed_station_months": int(
                    group["failed_station_months"].sum()
                ),
                "xiaowan_spill_volume_1e9m3": float(group["r1_spill_volume_1e9m3"].sum()),
                "nuozhadu_spill_volume_1e9m3": float(group["r2_spill_volume_1e9m3"].sum()),
                "cascade_spill_volume_1e9m3": float(
                    group["r1_spill_volume_1e9m3"].sum()
                    + group["r2_spill_volume_1e9m3"].sum()
                ),
                "xiaowan_release_volume_1e9m3": float(group["r1_release_volume_1e9m3"].sum()),
                "nuozhadu_release_volume_1e9m3": float(group["r2_release_volume_1e9m3"].sum()),
                "xiaowan_firm_power_reliability": float(group["xiaowan_firm_power_satisfied"].mean()),
                "nuozhadu_firm_power_reliability": float(group["nuozhadu_firm_power_satisfied"].mean()),
                "cascade_firm_power_reliability": float(group["cascade_firm_power_satisfied"].mean()),
                "xiaowan_target_unreached_count": int(group["xiaowan_target_unreached"].sum()),
                "nuozhadu_target_unreached_count": int(group["nuozhadu_target_unreached"].sum()),
                "either_reservoir_target_unreached_count": int(group["either_reservoir_target_unreached"].sum()),
                "xiaowan_mean_abs_target_error_m": float(group["xiaowan_target_error_m"].abs().mean()),
                "nuozhadu_mean_abs_target_error_m": float(group["nuozhadu_target_error_m"].abs().mean()),
                "xiaowan_target_level_clipped_count": int(group["r1_target_level_clipped"].sum()),
                "nuozhadu_target_level_clipped_count": int(group["r2_target_level_clipped"].sum()),
                "max_abs_water_balance_error_m3": float(
                    max(
                        group["r1_water_balance_error_m3"].abs().max(),
                        group["r2_water_balance_error_m3"].abs().max(),
                    )
                ),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    months = len(frame)
    metrics = {
        "months": months,
        "years": int(months // 12),
        "xiaowan_total_generation_gwh": float(frame["r1_energy_gwh"].sum()),
        "nuozhadu_total_generation_gwh": float(frame["r2_energy_gwh"].sum()),
        "cascade_total_generation_gwh": float(frame["reward_gwh"].sum()),
        "xiaowan_mean_annual_generation_gwh": float(frame["r1_energy_gwh"].sum() * 12.0 / months),
        "nuozhadu_mean_annual_generation_gwh": float(frame["r2_energy_gwh"].sum() * 12.0 / months),
        "cascade_mean_annual_generation_gwh": float(frame["reward_gwh"].sum() * 12.0 / months),
        "total_penalized_objective_gwh": float(
            frame["penalized_reward_gwh"].sum()
        ),
        "mean_annual_penalized_objective_gwh": float(
            frame["penalized_reward_gwh"].sum() * 12.0 / months
        ),
        "total_failure_penalty_gwh": float(frame["failure_penalty_gwh"].sum()),
        "failed_station_months": int(frame["failed_station_months"].sum()),
        "xiaowan_firm_power_reliability": float(frame["xiaowan_firm_power_satisfied"].mean()),
        "nuozhadu_firm_power_reliability": float(frame["nuozhadu_firm_power_satisfied"].mean()),
        "cascade_firm_power_reliability": float(frame["cascade_firm_power_satisfied"].mean()),
        "xiaowan_spill_volume_1e9m3": float(frame["r1_spill_volume_1e9m3"].sum()),
        "nuozhadu_spill_volume_1e9m3": float(frame["r2_spill_volume_1e9m3"].sum()),
        "cascade_spill_volume_1e9m3": float(
            frame["r1_spill_volume_1e9m3"].sum()
            + frame["r2_spill_volume_1e9m3"].sum()
        ),
        "xiaowan_target_unreached_count": int(frame["xiaowan_target_unreached"].sum()),
        "nuozhadu_target_unreached_count": int(frame["nuozhadu_target_unreached"].sum()),
        "either_reservoir_target_unreached_count": int(frame["either_reservoir_target_unreached"].sum()),
        "xiaowan_mean_abs_target_error_m": float(frame["xiaowan_target_error_m"].abs().mean()),
        "nuozhadu_mean_abs_target_error_m": float(frame["nuozhadu_target_error_m"].abs().mean()),
        "xiaowan_target_level_clipped_count": int(frame["r1_target_level_clipped"].sum()),
        "nuozhadu_target_level_clipped_count": int(frame["r2_target_level_clipped"].sum()),
        "max_abs_water_balance_error_m3": float(
            max(
                frame["r1_water_balance_error_m3"].abs().max(),
                frame["r2_water_balance_error_m3"].abs().max(),
            )
        ),
    }
    return yearly, metrics


__all__ = [
    "DQN_DIAGNOSTIC_COLUMNS",
    "DQN_EVALUATION_COLUMNS",
    "DQN_INFORMATION_SET",
    "NORMALIZED_STATE_COLUMNS",
    "evaluate_greedy_trajectory",
    "summarize_monthly_physical_trajectory",
]

