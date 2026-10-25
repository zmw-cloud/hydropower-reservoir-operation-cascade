"""Focused tests for continuous PPO probability and feasible-action contracts."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cascade_reservoir_benchmark.continuous_ppo_adapter import (
    ContinuousCascadePPOAdapter,
)
from cascade_reservoir_benchmark.continuous_ppo_agent import (
    ContinuousPPOAgent,
    ContinuousPPORolloutBatch,
)
from cascade_reservoir_benchmark.dqn_adapter import StateNormalizer
from cascade_reservoir_benchmark.environment import CascadeReservoirEnv, ReservoirSpec


def make_spec(name: str, *, max_release: float | None) -> ReservoirSpec:
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
        max_total_release_m3s=max_release,
    )


def make_agent() -> ContinuousPPOAgent:
    return ContinuousPPOAgent(
        state_dimension=5,
        actor_hidden_dimensions=[8],
        critic_hidden_dimensions=[8],
        learning_rate=1.0e-3,
        gamma=0.97,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coefficient=0.001,
        value_loss_coefficient=0.5,
        max_gradient_norm=0.5,
        minibatch_size=2,
        update_epochs=1,
        target_kl=0.02,
        advantage_normalization=True,
        adam_epsilon=1.0e-5,
        initial_log_std=-0.5,
        min_log_std=-5.0,
        max_log_std=1.0,
        squash_epsilon=1.0e-6,
        seed=1,
        device="cpu",
    )


def make_decision(
    agent: ContinuousPPOAgent,
    states: np.ndarray,
    *,
    deterministic: bool = False,
):
    xiaowan_geometry = np.tile(
        np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        (len(states), 1),
    )
    upstream = agent.act_xiaowan(
        states,
        xiaowan_geometry,
        deterministic=deterministic,
    )
    nuozhadu_geometry = np.column_stack(
        (
            upstream["fraction"],
            np.zeros(len(states), dtype=np.float32),
            np.ones(len(states), dtype=np.float32),
            np.ones(len(states), dtype=np.float32),
        )
    )
    return agent.complete_action(
        states,
        upstream,
        xiaowan_geometry,
        nuozhadu_geometry,
        deterministic=deterministic,
    )


class ContinuousActorTests(unittest.TestCase):
    def test_autoregressive_action_and_log_probability_are_finite(self):
        agent = make_agent()
        states = np.zeros((3, 5), dtype=np.float32)
        decision = make_decision(agent, states, deterministic=False)
        self.assertEqual(decision["latent_actions"].shape, (3, 2))
        self.assertEqual(decision["action_fractions"].shape, (3, 2))
        self.assertTrue(np.isfinite(decision["component_log_probs"]).all())
        self.assertTrue((decision["action_fractions"] > 0.0).all())
        self.assertTrue((decision["action_fractions"] < 1.0).all())

    def test_rollout_update_uses_active_dimension_mask(self):
        agent = make_agent()
        states = np.zeros((2, 1, 5), dtype=np.float32)
        decision = make_decision(
            agent, states.reshape(2, 5), deterministic=False
        )
        active = np.array([[[True, False]], [[True, True]]], dtype=np.bool_)
        old_log_probs = (
            decision["component_log_probs"] * active.reshape(2, 2)
        ).sum(axis=1).reshape(2, 1)
        rollout = ContinuousPPORolloutBatch(
            states=states,
            latent_actions=decision["latent_actions"].reshape(2, 1, 2),
            old_log_probs=old_log_probs,
            rewards=np.ones((2, 1), dtype=np.float32),
            values=decision["values"].reshape(2, 1),
            next_values=np.zeros((2, 1), dtype=np.float32),
            terminated=np.zeros((2, 1), dtype=np.bool_),
            truncated=np.array([[False], [True]], dtype=np.bool_),
            bootstrap_allowed=np.ones((2, 1), dtype=np.bool_),
            active_action_dimensions=active,
            action_geometry=decision["action_geometry"].reshape(2, 1, 7),
        )
        metrics = agent.update(rollout)
        self.assertEqual(metrics["rollout_size"], 2)
        self.assertTrue(np.isfinite(float(metrics["total_loss"])))
        self.assertGreaterEqual(int(metrics["epochs_completed"]), 1)
        self.assertLessEqual(int(metrics["epochs_completed"]), 1)
        self.assertIn("early_stop_triggered", metrics)

    def test_learning_rate_can_be_scheduled(self):
        agent = make_agent()
        agent.set_learning_rate(2.5e-4)
        self.assertAlmostEqual(
            float(agent.optimizer.param_groups[0]["lr"]),
            2.5e-4,
        )


class ContinuousFeasibleTransformTests(unittest.TestCase):
    def test_fraction_maps_to_reachable_unclipped_targets(self):
        upstream = make_spec("xw", max_release=5000.0)
        downstream = make_spec("nzd", max_release=None)
        periods = pd.period_range("2000-01", periods=2, freq="M")
        env = CascadeReservoirEnv(
            upstream,
            downstream,
            local_inflows_m3s=[[200.0, 50.0], [200.0, 50.0]],
            timestamps=periods,
            initial_levels_m=(8.0, 8.0),
            firm_power_failure_penalty_gwh=1.0,
            monthly_max_target_levels_m=[(9.0, 9.0)] * 12,
            require_energy_parameters=True,
        )
        normalizer = StateNormalizer(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, 10.0, 10.0, 1000.0, 1000.0],
            clip=True,
        )
        adapter = ContinuousCascadePPOAdapter(
            env,
            normalizer,
            reward_reference_days=30.4,
            reward_clip=None,
            max_transitions=None,
        )
        adapter.reset()
        xw_geometry = adapter.xiaowan_geometry_features()
        self.assertEqual(xw_geometry.shape, (3,))
        nzd_geometry = adapter.nuozhadu_geometry_features(0.25)
        self.assertEqual(nzd_geometry.shape, (4,))
        _, _, terminated, truncated, info = adapter.step([0.25, 0.75])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.all(info["continuous_target_levels_m"] >= info["continuous_lower_bounds_m"]))
        self.assertTrue(np.all(info["continuous_target_levels_m"] <= info["continuous_upper_bounds_m"]))
        self.assertFalse(info["reservoir_1"]["target_level_clipped"])
        self.assertFalse(info["reservoir_2"]["target_level_clipped"])


if __name__ == "__main__":
    unittest.main()
