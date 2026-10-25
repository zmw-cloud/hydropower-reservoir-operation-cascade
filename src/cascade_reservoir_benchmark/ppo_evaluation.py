"""Deterministic continuous validation/evaluation for the cascade PPO policy."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .dqn_adapter import (
    STATE_ORDER,
    JointActionGrid,
    StateNormalizer,
    normalized_inflow_sha256,
)
from .dqn_evaluation import (
    DQN_EVALUATION_COLUMNS,
    _physical_row,
    _validated_training_hash,
    _validated_trajectory_inputs,
)
from .environment import CascadeReservoirEnv, CascadeSpecError, ReservoirSpec
from .ppo_adapter import CascadePPOAdapter
from .ppo_agent import DiscretePPOAgent
from .sdp import reservoir_pair_sha256


PPO_INFORMATION_SET = (
    "month_levels_current_external_inflows_one_step_perfect_forecast"
)
PPO_POLICY_COLUMNS = (
    "ppo_policy_raw_entropy",
    "ppo_policy_normalized_entropy",
    "ppo_policy_max_action_probability",
)
PPO_EVALUATION_COLUMNS = [*DQN_EVALUATION_COLUMNS, *PPO_POLICY_COLUMNS]


def _validated_seed(agent: DiscretePPOAgent, seed: int | None) -> int:
    if not isinstance(agent, DiscretePPOAgent):
        raise TypeError("agent must be a DiscretePPOAgent.")
    if seed is None:
        return agent.seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    result = int(seed)
    if result != agent.seed:
        raise CascadeSpecError(
            "The trajectory seed must match the independently trained PPO seed."
        )
    return result


def evaluate_deterministic_trajectory(
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    agent: DiscretePPOAgent,
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
    """Run argmax logits over one continuous physical trajectory."""
    timer_started = time.perf_counter()
    if trajectory_kind not in {"validation", "evaluation"}:
        raise CascadeSpecError(
            "trajectory_kind must be 'validation' or 'evaluation'."
        )
    if not isinstance(collect_rows, bool):
        raise TypeError("collect_rows must be boolean.")
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
        raise CascadeSpecError("PPO must use the fixed five-component state.")
    if agent.action_count != action_grid.action_count:
        raise CascadeSpecError("PPO Actor output does not match the action grid.")

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
    adapter = CascadePPOAdapter(
        env,
        normalizer,
        action_grid,
        reward_reference_days=reward_reference_days,
        reward_clip=reward_clip,
        max_transitions=None,
    )
    rows: list[dict[str, Any]] | None = [] if collect_rows else None
    total_generation_gwh = 0.0
    total_penalized_objective_gwh = 0.0
    total_failure_penalty_gwh = 0.0
    failed_station_months = 0
    firm_months = np.zeros(3, dtype=np.int64)
    maximum_balance_error = 0.0
    selected_actions: list[int] = []
    raw_entropies: list[float] = []
    max_probabilities: list[float] = []
    feasible_action_counts: list[int] = []
    state, _ = adapter.reset()

    actor_was_training = agent.actor.training
    critic_was_training = agent.critic.training
    agent.actor.eval()
    agent.critic.eval()
    try:
        with torch.no_grad():
            for index, period in enumerate(periods):
                action_mask = adapter.get_feasible_action_mask()
                decision = agent.act(
                    state,
                    deterministic=True,
                    action_masks=action_mask,
                )
                action_index = int(decision["actions"][0])
                raw_entropy = float(decision["raw_entropies"][0])
                max_probability = float(decision["max_action_probabilities"][0])
                next_state, _, terminated, truncated, info = adapter.step(
                    action_index
                )
                if truncated:
                    raise CascadeSpecError(
                        "Unexpected truncation in a continuous PPO trajectory."
                    )
                if terminated != (index == len(periods) - 1):
                    raise CascadeSpecError(
                        "PPO trajectory terminated at an unexpected month."
                    )

                upstream = info["reservoir_1"]
                downstream = info["reservoir_2"]
                upstream_power = float(upstream["power_kw"])
                downstream_power = float(downstream["power_kw"])
                total_generation_gwh += float(info["generation_energy_gwh"])
                total_penalized_objective_gwh += float(
                    info["penalized_reward_gwh"]
                )
                total_failure_penalty_gwh += float(
                    info["failure_penalty_gwh"]
                )
                failed_station_months += int(info["failed_station_months"])
                firm_months[0] += int(
                    upstream_power >= float(reservoir_1.firm_power_kw)
                )
                firm_months[1] += int(
                    downstream_power >= float(reservoir_2.firm_power_kw)
                )
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
                selected_actions.append(action_index)
                raw_entropies.append(raw_entropy)
                max_probabilities.append(max_probability)
                feasible_action_counts.append(int(info["feasible_action_count"]))
                if rows is not None:
                    row = _physical_row(
                        period,
                        state,
                        info,
                        trajectory_index=index,
                        reservoir_1=reservoir_1,
                        reservoir_2=reservoir_2,
                    )
                    row.update(
                        {
                            "ppo_policy_raw_entropy": raw_entropy,
                            "ppo_policy_normalized_entropy": raw_entropy
                            / math.log(action_grid.action_count),
                            "ppo_policy_max_action_probability": max_probability,
                        }
                    )
                    rows.append(row)

                if terminated:
                    if next_state is not None:
                        raise CascadeSpecError(
                            "Terminal PPO transition returned a next state."
                        )
                else:
                    if next_state is None:
                        raise CascadeSpecError(
                            "Nonterminal PPO transition has no next state."
                        )
                    state = next_state
    finally:
        agent.actor.train(actor_was_training)
        agent.critic.train(critic_was_training)

    months = len(periods)
    diagnostics = adapter.episode_diagnostics
    elapsed = time.perf_counter() - timer_started
    summary: dict[str, Any] = {
        "method": "PPO",
        "information_set": PPO_INFORMATION_SET,
        "policy_execution": "argmax_actor_logits",
        "forecast_timing": "current_month_external_inflows_known_before_action",
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
        "total_cascade_generation_gwh": float(total_generation_gwh),
        "mean_annual_cascade_generation_gwh": float(
            total_generation_gwh * 12.0 / months
        ),
        "total_penalized_objective_gwh": float(
            total_penalized_objective_gwh
        ),
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
        "firm_power_reliability": {
            "xiaowan": float(firm_months[0] / months),
            "nuozhadu": float(firm_months[1] / months),
            "cascade": float(firm_months[2] / months),
        },
        "target_unreached_count": diagnostics["target_unreached_count"],
        "target_unreached_rate": diagnostics["target_unreached_rate"],
        "unique_selected_actions": int(np.unique(selected_actions).size),
        "action_mask_mode": "environment_transition_reachability_and_monthly_limits_v2",
        "mean_feasible_action_count": float(np.mean(feasible_action_counts)),
        "min_feasible_action_count": int(min(feasible_action_counts)),
        "mean_raw_policy_entropy": float(np.mean(raw_entropies)),
        "mean_normalized_policy_entropy": float(
            np.mean(raw_entropies) / math.log(action_grid.action_count)
        ),
        "mean_max_action_probability": float(np.mean(max_probabilities)),
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
        "reservoir_physics_sha256": reservoir_pair_sha256(
            reservoir_1, reservoir_2
        ),
    }
    summary[f"{trajectory_kind}_inflow_sha256"] = summary[
        "trajectory_inflow_sha256"
    ]
    summary[f"{trajectory_kind}_wall_time_seconds"] = float(elapsed)
    schedule = (
        pd.DataFrame(rows, columns=PPO_EVALUATION_COLUMNS)
        if rows is not None
        else None
    )
    return schedule, summary


__all__ = [
    "PPO_EVALUATION_COLUMNS",
    "PPO_INFORMATION_SET",
    "evaluate_deterministic_trajectory",
]

