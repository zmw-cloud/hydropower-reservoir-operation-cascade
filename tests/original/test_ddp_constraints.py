"""Static regression targets for DDP operating and terminal constraints."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from cascade_reservoir_benchmark.ddp import CascadeDDP
from cascade_reservoir_benchmark.ddp_parallel import ParallelCascadeDDP
from cascade_reservoir_benchmark.ddp_parallel_interpolated import (
    InterpolatedParallelCascadeDDP,
    interpolated_continuation_value,
)
from cascade_reservoir_benchmark.environment import (
    CascadeSpecError,
    ReservoirSpec,
    _transition,
)


def make_spec(name: str) -> ReservoirSpec:
    return ReservoirSpec(
        name=name,
        dead_level_m=0.0,
        normal_level_m=10.0,
        levels_m=np.array([0.0, 5.0, 10.0]),
        storages_1e9m3=np.array([0.0, 5.0, 10.0]),
        tailwater_discharge_m3s=np.array([0.0, 100.0, 10000.0]),
        tailwater_levels_m=np.array([0.0, 1.0, 2.0]),
        max_turbine_discharge_m3s=1000.0,
        max_power_kw=10000000.0,
        firm_power_kw=1.0,
        power_coefficient_kw_per_m3s_m=1.0,
        max_total_release_m3s=None,
    )


def monthly_limits() -> list[list[float]]:
    return [
        [10.0, 10.0],
        [10.0, 10.0],
        [10.0, 10.0],
        [10.0, 10.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [10.0, 10.0],
        [10.0, 10.0],
    ]


class DDPConstraintTests(unittest.TestCase):
    def test_monthly_max_applies_to_end_of_month_action(self) -> None:
        solver = CascadeDDP(
            make_spec("upstream"),
            make_spec("downstream"),
            local_inflows_m3s=np.ones((12, 2)),
            timestamps=pd.period_range("2000-01", periods=12, freq="M"),
            grid_step_m=5.0,
            monthly_max_target_levels_m=monthly_limits(),
        )
        may = solver._monthly_action_limits(4)
        october = solver._monthly_action_limits(9)
        november = solver._monthly_action_limits(10)
        self.assertEqual(may[2:], (5.0, 5.0))
        self.assertEqual(october[2:], (5.0, 5.0))
        self.assertEqual(november[2:], (10.0, 10.0))

    def test_all_ddp_variants_enforce_hard_terminal_end_state(self) -> None:
        timestamps = pd.to_datetime(["2000-11-01", "2000-12-01"])
        inflows = [[100.0, 20.0], [200.0, 30.0]]
        for solver_class in (
            CascadeDDP,
            ParallelCascadeDDP,
            InterpolatedParallelCascadeDDP,
        ):
            kwargs = {"num_workers": 1} if solver_class is not CascadeDDP else {}
            solver = solver_class(
                make_spec("upstream"),
                make_spec("downstream"),
                inflows,
                timestamps,
                grid_step_m=5.0,
                monthly_max_target_levels_m=monthly_limits(),
                **kwargs,
            )
            schedule, summary = solver.solve(
                (10.0, 10.0),
                terminal_levels_m=(10.0, 10.0),
            )
            self.assertEqual(summary["terminal_error_1_m"], 0.0)
            self.assertEqual(summary["terminal_error_2_m"], 0.0)
            self.assertEqual(schedule.iloc[-1]["reservoir_1"]["end_level_m"], 10.0)
            self.assertEqual(schedule.iloc[-1]["reservoir_2"]["end_level_m"], 10.0)

    def test_unreachable_monthly_limit_is_infeasible_not_clipped(self) -> None:
        blocked = replace(make_spec("blocked"), max_total_release_m3s=0.0)
        solver = CascadeDDP(
            blocked,
            blocked,
            local_inflows_m3s=[[0.0, 0.0]],
            timestamps=pd.to_datetime(["2000-05-01"]),
            grid_step_m=5.0,
            monthly_max_target_levels_m=monthly_limits(),
        )
        with self.assertRaisesRegex(CascadeSpecError, "no feasible trajectory"):
            solver.solve((10.0, 10.0))

    def test_interpolation_preserves_infeasible_terminal_corners(self) -> None:
        values = np.array([[-np.inf, -np.inf], [-np.inf, 0.0]])
        exact_terminal = interpolated_continuation_value(
            values, 1, 1, 0.0, 1, 1, 0.0
        )
        mixed = interpolated_continuation_value(values, 0, 1, 0.5, 0, 1, 0.5)
        self.assertEqual(exact_terminal, 0.0)
        self.assertEqual(mixed, -np.inf)
        self.assertFalse(np.isnan(mixed))

    def test_turbine_limit_is_separate_from_total_release_and_spill(self) -> None:
        spec = replace(
            make_spec("nuozhadu"),
            max_turbine_discharge_m3s=3537.0,
            max_power_kw=5850000.0,
            power_coefficient_kw_per_m3s_m=8.5,
            max_total_release_m3s=None,
        )
        transition = _transition(
            spec,
            start_storage_1e9m3=spec.normal_storage_1e9m3,
            inflow_m3s=5000.0,
            requested_target_level_m=spec.normal_level_m,
            seconds=31.0 * 86400.0,
        )
        self.assertAlmostEqual(transition.total_discharge_m3s, 5000.0)
        self.assertLessEqual(transition.turbine_discharge_m3s, 3537.0)
        self.assertAlmostEqual(
            transition.spill_discharge_m3s,
            transition.total_discharge_m3s - transition.turbine_discharge_m3s,
        )
        self.assertLessEqual(transition.power_kw, 5850000.0)
        self.assertAlmostEqual(
            transition.energy_gwh,
            transition.power_kw * 31.0 * 24.0 / 1.0e6,
        )


if __name__ == "__main__":
    unittest.main()
