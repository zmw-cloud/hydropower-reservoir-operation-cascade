"""Focused tests for the shared DQN/PPO feasible-action mask contract."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from cascade_reservoir_benchmark.dqn_adapter import (
    CascadeDQNAdapter,
    JointActionGrid,
    StateNormalizer,
)
from cascade_reservoir_benchmark.dqn_agent import DoubleDQNAgent
from cascade_reservoir_benchmark.environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
)
from cascade_reservoir_benchmark.ppo_agent import (
    DiscretePPOAgent,
    PPORolloutBatch,
)


def make_spec(name: str) -> ReservoirSpec:
    return ReservoirSpec(
        name=name,
        dead_level_m=0.0,
        normal_level_m=10.0,
        levels_m=np.array([0.0, 5.0, 10.0]),
        storages_1e9m3=np.array([0.0, 5.0, 10.0]),
        tailwater_discharge_m3s=np.array([0.0, 100.0, 5000.0]),
        tailwater_levels_m=np.array([0.0, 1.0, 2.0]),
        max_turbine_discharge_m3s=1000.0,
        max_power_kw=10000000.0,
        firm_power_kw=1.0,
        power_coefficient_kw_per_m3s_m=1.0,
    )


def make_dqn_agent(action_count: int = 3) -> DoubleDQNAgent:
    return DoubleDQNAgent(
        state_dimension=5,
        action_count=action_count,
        hidden_dimensions=[8],
        learning_rate=1.0e-3,
        gamma=0.5,
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_schedule="linear",
        epsilon_decay_steps=1,
        learning_starts=0,
        target_update_interval=100,
        replay_capacity=4,
        batch_size=1,
        max_grad_norm=1.0,
        seed=1,
        device="cpu",
    )


def make_ppo_agent(action_count: int = 3) -> DiscretePPOAgent:
    action_features = np.column_stack(
        (
            np.linspace(0.0, 1.0, action_count, dtype=np.float32),
            np.linspace(1.0, 0.0, action_count, dtype=np.float32),
        )
    )
    return DiscretePPOAgent(
        state_dimension=5,
        action_count=action_count,
        action_features=action_features,
        action_score_chunk_size=2,
        actor_hidden_dimensions=[8],
        critic_hidden_dimensions=[8],
        learning_rate=1.0e-3,
        gamma=0.97,
        gae_lambda=0.95,
        clip_range=0.2,
        normalized_entropy_coefficient=0.01,
        value_loss_coefficient=0.5,
        max_gradient_norm=0.5,
        minibatch_size=2,
        update_epochs=1,
        advantage_normalization=True,
        adam_epsilon=1.0e-5,
        seed=1,
        device="cpu",
    )


class SharedEnvironmentMaskTests(unittest.TestCase):
    def test_mask_reuses_physics_keeps_row_major_order_and_has_no_side_effect(self):
        env = CascadeReservoirEnv(
            make_spec("xw"),
            make_spec("nzd"),
            local_inflows_m3s=[[0.0, 0.0]],
            timestamps=pd.to_datetime(["2000-01-01"]),
            initial_levels_m=(0.0, 0.0),
        )
        env.reset()
        storage_before = env.storages_1e9m3.copy()
        index_before = env.index
        levels = np.array([0.0, 5.0, 10.0])
        mask = env.get_feasible_action_mask(levels, levels)
        self.assertEqual(mask.shape, (9,))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[0])
        self.assertEqual(int(mask.sum()), 1)
        np.testing.assert_array_equal(env.storages_1e9m3, storage_before)
        self.assertEqual(env.index, index_before)

    def test_adapter_rejects_an_action_marked_infeasible(self):
        spec_1, spec_2 = make_spec("xw"), make_spec("nzd")
        env = CascadeReservoirEnv(
            spec_1,
            spec_2,
            local_inflows_m3s=[[0.0, 0.0], [0.0, 0.0]],
            timestamps=pd.to_datetime(["2000-01-01", "2000-02-01"]),
            initial_levels_m=(0.0, 0.0),
        )
        normalizer = StateNormalizer(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, 10.0, 10.0, 1.0, 1.0],
        )
        adapter = CascadeDQNAdapter(
            env,
            normalizer,
            JointActionGrid(spec_1, spec_2, step_m=5.0),
            max_transitions=1,
        )
        adapter.reset()
        with self.assertRaisesRegex(CascadeSpecError, "physically infeasible"):
            adapter.step(8)

    def test_flood_limit_is_a_hard_mask_and_step_constraint(self):
        monthly_limits = np.full((12, 2), 10.0, dtype=float)
        monthly_limits[4] = [5.0, 5.0]
        env = CascadeReservoirEnv(
            make_spec("xw"),
            make_spec("nzd"),
            local_inflows_m3s=[[0.0, 0.0]],
            timestamps=pd.to_datetime(["2000-05-01"]),
            initial_levels_m=(10.0, 10.0),
            monthly_max_target_levels_m=monthly_limits,
        )
        env.reset()
        levels = np.array([0.0, 5.0, 10.0])
        mask = env.get_feasible_action_mask(levels, levels).reshape(3, 3)
        self.assertFalse(mask[2, :].any())
        self.assertFalse(mask[:, 2].any())
        self.assertTrue(mask.any())
        with self.assertRaisesRegex(CascadeSpecError, "hard operating limits"):
            env.step([10.0, 10.0])

    def test_reward_penalizes_each_failed_station_month_once(self):
        high_firm = replace(make_spec("high-firm"), firm_power_kw=1.0e12)
        env = CascadeReservoirEnv(
            high_firm,
            high_firm,
            local_inflows_m3s=[[0.0, 0.0]],
            timestamps=pd.to_datetime(["2000-01-01"]),
            initial_levels_m=(0.0, 0.0),
            firm_power_failure_penalty_gwh=7.0,
        )
        env.reset()
        _, reward, _, _, info = env.step([0.0, 0.0])
        self.assertEqual(info["failed_station_months"], 2)
        self.assertEqual(info["failure_penalty_gwh"], 14.0)
        self.assertAlmostEqual(
            reward,
            info["generation_energy_gwh"] - 14.0,
        )


class DQNMaskTests(unittest.TestCase):
    def test_greedy_action_masks_the_largest_invalid_q_value(self):
        agent = make_dqn_agent()
        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()
            agent.online_network.layers[-1].bias.copy_(
                torch.tensor([1.0, 100.0, 5.0])
            )
        action = agent.select_action(
            np.zeros(5, dtype=np.float32),
            greedy=True,
            action_mask=np.array([True, False, True]),
        )
        self.assertEqual(action, 2)

    def test_double_dqn_target_uses_the_stored_next_state_mask(self):
        agent = make_dqn_agent()
        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()
            for parameter in agent.target_network.parameters():
                parameter.zero_()
            agent.online_network.layers[-1].bias.copy_(
                torch.tensor([1.0, 100.0, 5.0])
            )
            agent.target_network.layers[-1].bias.copy_(
                torch.tensor([2.0, 200.0, 3.0])
            )
        state = np.zeros(5, dtype=np.float32)
        agent.store_transition(
            state,
            0,
            0.0,
            state,
            bootstrap_terminal=False,
            next_action_mask=np.array([True, False, False]),
        )
        metrics = agent.learn()
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(float(metrics["mean_target_q"]), 1.0, places=6)


class PPOMaskTests(unittest.TestCase):
    def test_rollout_action_and_update_share_the_same_mask(self):
        agent = make_ppo_agent()
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.zero_()
            agent.actor.action_bias[0].weight[0, 0] = 1.0
            agent.actor.action_bias[2].weight[0, 0] = 4.0
        states = np.zeros((2, 1, 5), dtype=np.float32)
        masks = np.broadcast_to(
            np.array([True, False, True]), (2, 1, 3)
        ).copy()
        decision = agent.act(
            states.reshape(2, 5),
            deterministic=True,
            action_masks=masks.reshape(2, 3),
        )
        self.assertTrue(np.all(decision["actions"] == 2))
        rollout = PPORolloutBatch(
            states=states,
            actions=decision["actions"].reshape(2, 1),
            old_log_probs=decision["log_probs"].reshape(2, 1),
            rewards=np.ones((2, 1), dtype=np.float32),
            values=decision["values"].reshape(2, 1),
            next_values=np.zeros((2, 1), dtype=np.float32),
            terminated=np.zeros((2, 1), dtype=np.bool_),
            truncated=np.array([[False], [True]], dtype=np.bool_),
            bootstrap_allowed=np.ones((2, 1), dtype=np.bool_),
            action_masks=masks,
        )
        metrics = agent.update(rollout)
        self.assertEqual(metrics["min_feasible_action_count"], 2)
        self.assertEqual(metrics["mean_feasible_action_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
