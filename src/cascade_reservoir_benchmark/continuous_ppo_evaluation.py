"""Deterministic continuous-target evaluation for autoregressive PPO."""

from __future__ import annotations

import time
from dataclasses import fields
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .continuous_ppo_adapter import (
    CONTINUOUS_ACTION_TRANSFORM,
    ContinuousCascadePPOAdapter,
)
from .continuous_ppo_agent import ContinuousPPOAgent
from .dqn_adapter import STATE_ORDER, StateNormalizer, normalized_inflow_sha256
from .dqn_evaluation import _validated_training_hash, _validated_trajectory_inputs
from .environment import CascadeReservoirEnv, CascadeSpecError, ReservoirSpec, ReservoirTransition
from .sdp import reservoir_pair_sha256


CONTINUOUS_PPO_INFORMATION_SET = (
    "month_levels_current_external_inflows_one_step_perfect_forecast"
)
TRANSITION_FIELDS = tuple(item.name for item in fields(ReservoirTransition))


def evaluate_continuous_ppo_trajectory(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    agent: ContinuousPPOAgent,
    normalizer: StateNormalizer,
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
    started = time.perf_counter()
    if trajectory_kind not in {"validation", "evaluation"}:
        raise CascadeSpecError("trajectory_kind must be validation or evaluation.")
    trajectory_seed = agent.seed if seed is None else int(seed)
    if trajectory_seed != agent.seed:
        raise CascadeSpecError("Continuous PPO evaluation seed differs from agent seed.")
    training_hash = _validated_training_hash(training_inflow_sha256)
    periods, flows, initial_levels = _validated_trajectory_inputs(
        joint,
        reservoir_1,
        reservoir_2,
        initial_levels_m,
        water_balance_tolerance_m3,
    )
    if agent.state_dimension != len(STATE_ORDER):
        raise CascadeSpecError("Continuous PPO must use the fixed five-component state.")

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
    adapter = ContinuousCascadePPOAdapter(
        env,
        normalizer,
        reward_reference_days=reward_reference_days,
        reward_clip=reward_clip,
        max_transitions=None,
    )
    state, _ = adapter.reset()
    rows: list[dict[str, Any]] | None = [] if collect_rows else None
    total_generation = 0.0
    total_penalized = 0.0
    total_failure_penalty = 0.0
    failed_station_months = 0
    firm_months = np.zeros(3, dtype=np.int64)
    maximum_balance_error = 0.0
    fractions: list[np.ndarray] = []
    interval_widths: list[np.ndarray] = []
    actor_was_training = agent.actor.training
    critic_was_training = agent.critic.training
    agent.actor.eval()
    agent.critic.eval()
    try:
        with torch.no_grad():
            for index, period in enumerate(periods):
                xiaowan_geometry = adapter.xiaowan_geometry_features()[None, :]
                xiaowan_decision = agent.act_xiaowan(
                    state,
                    xiaowan_geometry,
                    deterministic=True,
                    xiaowan_active=np.asarray(
                        [adapter.xiaowan_action_active()], dtype=np.bool_
                    ),
                )
                nuozhadu_geometry = adapter.nuozhadu_geometry_features(
                    float(xiaowan_decision["fraction"][0])
                )[None, :]
                decision = agent.complete_action(
                    state,
                    xiaowan_decision,
                    xiaowan_geometry,
                    nuozhadu_geometry,
                    deterministic=True,
                    nuozhadu_active=np.asarray(
                        [
                            adapter.nuozhadu_action_active(
                                float(xiaowan_decision["fraction"][0])
                            )
                        ],
                        dtype=np.bool_,
                    ),
                )
                selected_fraction = decision["action_fractions"][0]
                next_state, _, terminated, truncated, info = adapter.step(
                    selected_fraction
                )
                if truncated or terminated != (index == len(periods) - 1):
                    raise CascadeSpecError(
                        "Continuous PPO trajectory ended at an unexpected month."
                    )
                upstream = info["reservoir_1"]
                downstream = info["reservoir_2"]
                upstream_power = float(upstream["power_kw"])
                downstream_power = float(downstream["power_kw"])
                total_generation += float(info["generation_energy_gwh"])
                total_penalized += float(info["penalized_reward_gwh"])
                total_failure_penalty += float(info["failure_penalty_gwh"])
                failed_station_months += int(info["failed_station_months"])
                firm_months[0] += int(upstream_power >= float(reservoir_1.firm_power_kw))
                firm_months[1] += int(downstream_power >= float(reservoir_2.firm_power_kw))
                firm_months[2] += int(
                    upstream_power + downstream_power
                    >= float(reservoir_1.firm_power_kw + reservoir_2.firm_power_kw)
                )
                maximum_balance_error = max(
                    maximum_balance_error,
                    abs(float(upstream["water_balance_error_m3"])),
                    abs(float(downstream["water_balance_error_m3"])),
                )
                fractions.append(
                    np.asarray(info["continuous_action_fractions"], dtype=float)
                )
                interval_widths.append(
                    np.asarray(info["continuous_interval_widths_m"], dtype=float)
                )
                if rows is not None:
                    raw_state = np.asarray(info["state_raw"], dtype=float)
                    row: dict[str, Any] = {
                        "date": f"{period.year:04d}-{period.month:02d}-01",
                        "year_index": int(index // 12 + 1),
                        "calendar_year": int(period.year),
                        "calendar_month": int(period.month),
                        "start_level_1_m": float(raw_state[1]),
                        "start_level_2_m": float(raw_state[2]),
                        "target_level_1_m": float(upstream["requested_target_level_m"]),
                        "target_level_2_m": float(downstream["requested_target_level_m"]),
                        "reward_gwh": float(info["generation_energy_gwh"]),
                        "penalized_reward_gwh": float(info["penalized_reward_gwh"]),
                        "failed_station_months": int(info["failed_station_months"]),
                        "failure_penalty_gwh": float(info["failure_penalty_gwh"]),
                        "inflow_1_m3s": float(info["inflow_1_m3s"]),
                        "inflow_2_local_m3s": float(info["inflow_2_local_m3s"]),
                        "routed_inflow_m3s": float(info["routed_inflow_m3s"]),
                        "xiaowan_firm_power_satisfied": bool(
                            upstream_power >= float(reservoir_1.firm_power_kw)
                        ),
                        "nuozhadu_firm_power_satisfied": bool(
                            downstream_power >= float(reservoir_2.firm_power_kw)
                        ),
                        "cascade_firm_power_satisfied": bool(
                            upstream_power + downstream_power
                            >= float(reservoir_1.firm_power_kw + reservoir_2.firm_power_kw)
                        ),
                        "continuous_xiaowan_fraction": float(selected_fraction[0]),
                        "continuous_nuozhadu_fraction": float(selected_fraction[1]),
                        "continuous_xiaowan_lower_bound_m": float(
                            info["continuous_lower_bounds_m"][0]
                        ),
                        "continuous_xiaowan_upper_bound_m": float(
                            info["continuous_upper_bounds_m"][0]
                        ),
                        "continuous_nuozhadu_lower_bound_m": float(
                            info["continuous_lower_bounds_m"][1]
                        ),
                        "continuous_nuozhadu_upper_bound_m": float(
                            info["continuous_upper_bounds_m"][1]
                        ),
                        "continuous_xiaowan_interval_width_m": float(
                            info["continuous_interval_widths_m"][0]
                        ),
                        "continuous_nuozhadu_interval_width_m": float(
                            info["continuous_interval_widths_m"][1]
                        ),
                        "continuous_policy_log_prob": float(decision["log_probs"][0]),
                        "continuous_xiaowan_policy_std": float(decision["action_std"][0, 0]),
                        "continuous_nuozhadu_policy_std": float(decision["action_std"][0, 1]),
                    }
                    row.update({f"r1_{key}": upstream[key] for key in TRANSITION_FIELDS})
                    row.update({f"r2_{key}": downstream[key] for key in TRANSITION_FIELDS})
                    rows.append(row)
                if terminated:
                    if next_state is not None:
                        raise CascadeSpecError("Terminal continuous transition returned state.")
                else:
                    if next_state is None:
                        raise CascadeSpecError("Nonterminal continuous transition lacks state.")
                    state = next_state
    finally:
        agent.actor.train(actor_was_training)
        agent.critic.train(critic_was_training)

    months = len(periods)
    fraction_array = np.stack(fractions)
    width_array = np.stack(interval_widths)
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "method": "Continuous PPO",
        "policy_distribution": agent.configuration_state()["action_mode"],
        "continuous_action_transform": CONTINUOUS_ACTION_TRANSFORM,
        "information_set": CONTINUOUS_PPO_INFORMATION_SET,
        "closed_loop_state_storage": "continuous_physical_storage",
        "seed": trajectory_seed,
        "months": months,
        "years": float(months / 12.0),
        "total_cascade_generation_gwh": float(total_generation),
        "mean_annual_cascade_generation_gwh": float(total_generation * 12.0 / months),
        "total_penalized_objective_gwh": float(total_penalized),
        "mean_annual_penalized_objective_gwh": float(total_penalized * 12.0 / months),
        "firm_power_failure_penalty_gwh": float(firm_power_failure_penalty_gwh),
        "total_failure_penalty_gwh": float(total_failure_penalty),
        "failed_station_months": int(failed_station_months),
        "firm_power_reliability": {
            "xiaowan": float(firm_months[0] / months),
            "nuozhadu": float(firm_months[1] / months),
            "cascade": float(firm_months[2] / months),
        },
        "mean_action_fraction": {
            "xiaowan": float(fraction_array[:, 0].mean()),
            "nuozhadu": float(fraction_array[:, 1].mean()),
        },
        "mean_feasible_interval_width_m": {
            "xiaowan": float(width_array[:, 0].mean()),
            "nuozhadu": float(width_array[:, 1].mean()),
        },
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
        "trajectory_wall_time_seconds": float(elapsed),
        "reservoir_physics_sha256": reservoir_pair_sha256(reservoir_1, reservoir_2),
    }
    summary[f"{trajectory_kind}_inflow_sha256"] = summary["trajectory_inflow_sha256"]
    summary[f"{trajectory_kind}_wall_time_seconds"] = float(elapsed)
    return (pd.DataFrame(rows) if rows is not None else None), summary


__all__ = [
    "CONTINUOUS_PPO_INFORMATION_SET",
    "evaluate_continuous_ppo_trajectory",
]
