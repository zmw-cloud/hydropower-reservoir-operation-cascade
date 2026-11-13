"""Regression tests for the vectorized DDP Bellman kernel."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cascade_reservoir_benchmark.ddp import CascadeDDP
from cascade_reservoir_benchmark.ddp_parallel import (
    ParallelCascadeDDP,
    _transition_batch,
)
from cascade_reservoir_benchmark.ddp_parallel_interpolated import (
    InterpolatedParallelCascadeDDP,
)
from cascade_reservoir_benchmark.environment import ReservoirSpec, _transition


def make_spec(name: str, *, max_total_release_m3s: float | None = None) -> ReservoirSpec:
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
        firm_power_kw=100.0,
        power_coefficient_kw_per_m3s_m=1.0,
        max_total_release_m3s=max_total_release_m3s,
    )


class VectorizedDDPTests(unittest.TestCase):
    def test_transition_batch_matches_scalar_physics(self) -> None:
        for max_release in (None, 750.0):
            spec = make_spec("reservoir", max_total_release_m3s=max_release)
            starts = np.array([0.0, 5.0, 10.0])
            targets = np.array([0.0, 5.0, 10.0])
            seconds = 31.0 * 86400.0
            batch = _transition_batch(spec, starts, 900.0, targets, seconds)
            for state, start in enumerate(starts):
                for action, target in enumerate(targets):
                    scalar = _transition(spec, start, 900.0, target, seconds)
                    self.assertAlmostEqual(
                        batch.end_storage[state, action],
                        scalar.end_storage_1e9m3,
                        places=12,
                    )
                    self.assertAlmostEqual(
                        batch.total_discharge[state, action],
                        scalar.total_discharge_m3s,
                        places=10,
                    )
                    self.assertAlmostEqual(
                        batch.energy[state, action], scalar.energy_gwh, places=10
                    )
                    self.assertAlmostEqual(
                        batch.firm_deficit[state, action],
                        scalar.firm_deficit_gwh,
                        places=10,
                    )

    def test_vectorized_parallel_solvers_match_scalar_reference(self) -> None:
        inflows = [[100.0, 20.0], [200.0, 30.0], [150.0, 25.0]]
        timestamps = pd.period_range("2000-01", periods=3, freq="M")
        kwargs = dict(
            reservoir_1=make_spec("upstream"),
            reservoir_2=make_spec("downstream"),
            local_inflows_m3s=inflows,
            timestamps=timestamps,
            grid_step_m=5.0,
        )
        reference, _ = CascadeDDP(**kwargs).solve((10.0, 10.0))
        expected = reference[
            ["target_level_1_m", "target_level_2_m", "reward_gwh"]
        ].to_numpy()
        for solver_class in (ParallelCascadeDDP, InterpolatedParallelCascadeDDP):
            actual, _ = solver_class(**kwargs, num_workers=1).solve((10.0, 10.0))
            np.testing.assert_allclose(
                actual[["target_level_1_m", "target_level_2_m", "reward_gwh"]],
                expected,
                rtol=0.0,
                atol=1.0e-10,
            )

    def test_interpolated_vectorized_checkpoint_round_trip(self) -> None:
        kwargs = dict(
            reservoir_1=make_spec("upstream"),
            reservoir_2=make_spec("downstream"),
            local_inflows_m3s=[[100.0, 20.0], [200.0, 30.0]],
            timestamps=pd.period_range("2000-01", periods=2, freq="M"),
            grid_step_m=5.0,
            num_workers=1,
        )
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "interpolated.checkpoint.npz"
            first, _ = InterpolatedParallelCascadeDDP(**kwargs).solve(
                (10.0, 10.0),
                checkpoint_path=checkpoint,
                checkpoint_every_months=1,
            )
            resumed, summary = InterpolatedParallelCascadeDDP(**kwargs).solve(
                (10.0, 10.0),
                checkpoint_path=checkpoint,
                checkpoint_every_months=1,
                resume=True,
            )
            self.assertTrue(summary["resumed_from_checkpoint"])
            self.assertEqual(summary["resumed_completed_months"], 2)
            np.testing.assert_allclose(
                resumed[["target_level_1_m", "target_level_2_m", "reward_gwh"]],
                first[["target_level_1_m", "target_level_2_m", "reward_gwh"]],
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
