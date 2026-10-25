"""Focused checks for the unified cascade transition interface."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cascade_reservoir_benchmark.environment import (
    CascadeReservoirEnv,
    CascadeSpecError,
    ReservoirSpec,
)
from cascade_reservoir_benchmark.ddp import CascadeDDP
from cascade_reservoir_benchmark.data import load_configured_joint_inflows, load_observed_joint_inflows
from cascade_reservoir_benchmark.environment import load_specs_from_yaml


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


class CascadeEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CascadeReservoirEnv(
            make_spec("upstream"),
            make_spec("downstream"),
            local_inflows_m3s=[[100.0, 20.0], [200.0, 30.0]],
            timestamps=pd.to_datetime(["2000-01-01", "2000-02-01"]),
        )

    def test_observation_uses_current_month_only(self) -> None:
        observation, _ = self.env.reset()
        self.assertEqual(observation.shape, (5,))
        self.assertEqual(observation[2], 100.0)
        self.assertEqual(observation[3], 20.0)
        next_observation, _, _, _, _ = self.env.step([5.0, 5.0])
        self.assertEqual(next_observation[2], 200.0)
        self.assertEqual(next_observation[3], 30.0)

    def test_zero_delay_routing_and_water_balance(self) -> None:
        self.env.reset()
        _, _, _, _, info = self.env.step([10.0, 10.0])
        self.assertAlmostEqual(info["reservoir_1"]["total_discharge_m3s"], 100.0, places=7)
        self.assertAlmostEqual(info["routed_inflow_m3s"], 100.0, places=7)
        self.assertAlmostEqual(info["effective_inflow_2_m3s"], 120.0, places=7)
        self.assertLess(abs(info["reservoir_1"]["water_balance_error_m3"]), 1.0e-3)
        self.assertLess(abs(info["reservoir_2"]["water_balance_error_m3"]), 1.0e-3)

    def test_infeasible_target_is_clipped(self) -> None:
        self.env.reset()
        _, _, _, _, info = self.env.step([100.0, -10.0])
        self.assertTrue(info["reservoir_1"]["target_level_clipped"])
        self.assertTrue(info["reservoir_2"]["target_level_clipped"])
        self.assertLessEqual(info["reservoir_1"]["end_level_m"], 10.0)
        self.assertGreaterEqual(info["reservoir_2"]["end_level_m"], 0.0)

    def test_tailwater_uses_polynomial_extrapolation_with_high_flow_anchors(self) -> None:
        spec = replace(
            make_spec("upstream"),
            tailwater_discharge_m3s=np.array([0.0, 100.0, 1000.0]),
            tailwater_levels_m=np.array([0.0, 1.0, 3.0]),
            high_flow_tailwater_anchors=((5000.0, 8.0), (10000.0, 10.0)),
        )
        inside, inside_extrapolated = spec.tailwater_from_discharge(1000.0)
        middle, middle_extrapolated = spec.tailwater_from_discharge(3000.0)
        low, low_extrapolated = spec.tailwater_from_discharge(-100.0)
        high, high_extrapolated = spec.tailwater_from_discharge(12000.0)
        self.assertFalse(inside_extrapolated)
        self.assertFalse(middle_extrapolated)
        self.assertTrue(low_extrapolated)
        self.assertTrue(high_extrapolated)
        self.assertTrue(np.isfinite(inside))
        self.assertTrue(np.isfinite(middle))
        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))

    def test_missing_fixed_turbine_limit_uses_power_limited_flow(self) -> None:
        downstream = replace(make_spec("downstream"), max_turbine_discharge_m3s=None)
        env = CascadeReservoirEnv(
            make_spec("upstream"),
            downstream,
            local_inflows_m3s=[[10000.0, 20.0]],
            require_energy_parameters=True,
        )
        env.reset()
        _, _, _, _, info = env.step([0.0, 0.0])
        r2 = info["reservoir_2"]
        self.assertIsNotNone(r2["turbine_discharge_m3s"])
        self.assertLessEqual(r2["power_kw"], downstream.max_power_kw)
        self.assertGreaterEqual(r2["spill_discharge_m3s"], 0.0)

    def test_joint_ddp_is_labeled_as_perfect_foresight_upper_bound(self) -> None:
        solver = CascadeDDP(
            make_spec("upstream"),
            make_spec("downstream"),
            local_inflows_m3s=[[100.0, 20.0], [200.0, 30.0]],
            timestamps=pd.to_datetime(["2000-01-01", "2000-02-01"]),
            grid_step_m=5.0,
        )
        schedule, summary = solver.solve((10.0, 10.0))
        self.assertEqual(len(schedule), 2)
        self.assertEqual(summary["information_set"], "complete_realized_future_inflow_offline_upper_bound_only")
        self.assertEqual(summary["joint_action_count"], 9)

    def test_ddp_checkpoint_can_resume_completed_backward_recursion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "ddp.checkpoint.npz"
            timestamps = pd.to_datetime(["2000-01-01", "2000-02-01"])
            inflows = [[100.0, 20.0], [200.0, 30.0]]
            first_solver = CascadeDDP(
                make_spec("upstream"),
                make_spec("downstream"),
                local_inflows_m3s=inflows,
                timestamps=timestamps,
                grid_step_m=5.0,
            )
            first_schedule, _ = first_solver.solve(
                (10.0, 10.0),
                checkpoint_path=checkpoint,
                checkpoint_every_months=1,
            )
            self.assertTrue(checkpoint.exists())

            resumed_solver = CascadeDDP(
                make_spec("upstream"),
                make_spec("downstream"),
                local_inflows_m3s=inflows,
                timestamps=timestamps,
                grid_step_m=5.0,
            )
            resumed_schedule, resumed_summary = resumed_solver.solve(
                (10.0, 10.0),
                checkpoint_path=checkpoint,
                checkpoint_every_months=1,
                resume=True,
            )
            self.assertTrue(resumed_summary["resumed_from_checkpoint"])
            self.assertEqual(resumed_summary["resumed_completed_months"], 2)
            self.assertEqual(
                first_schedule[["target_level_1_m", "target_level_2_m"]].to_dict("records"),
                resumed_schedule[["target_level_1_m", "target_level_2_m"]].to_dict("records"),
            )

    def test_project_config_and_observed_inflows_load_without_future_leakage(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reservoir_1, reservoir_2, config = load_specs_from_yaml(root / "configs" / "xiaowan_nuozhadu_v1.yaml")
        joint = load_observed_joint_inflows(root / "data" / "xiaowan_nuozhadu")
        self.assertEqual(len(joint), 372)
        self.assertEqual(list(joint.columns), ["date", "xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s"])
        self.assertTrue(config["data"]["local_inflow_experiment_ready"])
        self.assertEqual(reservoir_1.max_turbine_discharge_m3s, 2340.0)
        self.assertAlmostEqual(reservoir_2.max_turbine_discharge_m3s, 3537.0)
        self.assertIsNone(reservoir_2.max_total_release_m3s)
        constraints = config["ddp_constraints"]
        self.assertEqual(constraints["action_grid_step_m"], 1.0)
        self.assertEqual(constraints["initial_levels_m"], [1240.0, 812.0])
        self.assertEqual(constraints["terminal_levels_m"], [1240.0, 812.0])
        self.assertEqual(constraints["flood_season_months"], [6, 7, 8, 9, 10])
        self.assertEqual(
            constraints["flood_limit_action_months"],
            [5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(
            constraints["monthly_max_target_levels_m"]["xiaowan"],
            [1240.0, 1240.0, 1240.0, 1240.0, 1236.0, 1236.0,
             1236.0, 1236.0, 1236.0, 1236.0, 1240.0, 1240.0],
        )
        self.assertEqual(
            constraints["monthly_max_target_levels_m"]["nuozhadu"],
            [812.0, 812.0, 812.0, 812.0, 804.0, 804.0,
             804.0, 804.0, 804.0, 804.0, 812.0, 812.0],
        )

    def test_configured_synthetic_train_and_evaluation_are_independent_400_year_inputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        _, _, config = load_specs_from_yaml(root / "configs" / "xiaowan_nuozhadu_v1.yaml")
        train = load_configured_joint_inflows(config, "train")
        evaluation = load_configured_joint_inflows(config, "evaluation")
        self.assertEqual(len(train), 4800)
        self.assertEqual(len(evaluation), 4800)
        self.assertEqual(train.date.iloc[0], "2009-01-01")
        self.assertEqual(train.date.iloc[-1], "2408-12-01")
        self.assertEqual(evaluation.date.iloc[-1], "2408-12-01")
        self.assertFalse(train.equals(evaluation))

    def test_missing_machine_parameters_block_energy_experiment(self) -> None:
        incomplete = replace(make_spec("downstream"), max_power_kw=None)
        with self.assertRaisesRegex(CascadeSpecError, "machine parameters"):
            CascadeReservoirEnv(
                make_spec("upstream"),
                incomplete,
                local_inflows_m3s=[[100.0, 20.0]],
                require_energy_parameters=True,
            )


if __name__ == "__main__":
    unittest.main()
