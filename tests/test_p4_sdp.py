"""Focused algorithm and physical-equivalence tests for P4 cascade SDP."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cascade_reservoir_benchmark.environment import (
    CascadeSpecError,
    ReservoirSpec,
    _parse_simple_yaml,
    _transition,
    _transition_arrays,
    load_specs_from_yaml,
)
from cascade_reservoir_benchmark.sdp import (
    DEFAULT_WATER_LEVEL_STEP_M,
    POLICY_INFORMATION_SET,
    POLICY_SCHEMA_VERSION,
    CascadeSDP,
    SDPPolicy,
    bilinear_interpolate_storage_values,
    format_water_level_step_label,
    sdp_output_variant_path,
)
from cascade_reservoir_benchmark.sdp_evaluation import DDP_COMPATIBLE_COLUMNS, evaluate_policy
from cascade_reservoir_benchmark.sdp_probability import MonthlyInflowModel


def make_spec(name: str, *, normal_level: float = 2.0) -> ReservoirSpec:
    return ReservoirSpec(
        name=name,
        dead_level_m=0.0,
        normal_level_m=normal_level,
        levels_m=np.array([0.0, normal_level / 2.0, normal_level]),
        storages_1e9m3=np.array([0.0, normal_level / 2.0, normal_level]),
        tailwater_discharge_m3s=np.array([0.0, 1000.0, 10000.0]),
        tailwater_levels_m=np.array([0.0, 0.1, 0.2]),
        max_turbine_discharge_m3s=5000.0,
        max_power_kw=10000000.0,
        firm_power_kw=100.0,
        power_coefficient_kw_per_m3s_m=8.5,
    )


def probability_frame(years: int = 2, *, upstream: float = 100.0, local: float = 20.0) -> pd.DataFrame:
    periods = pd.period_range("2001-01", periods=12 * years, freq="M")
    return pd.DataFrame({
        "date": [f"{period}-01" for period in periods],
        "xiaowan_inflow_m3s": upstream,
        "nuozhadu_local_inflow_m3s": local,
    })


def policy_metadata(model: MonthlyInflowModel, n_1: int = 3, n_2: int = 3) -> dict:
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "information_set": POLICY_INFORMATION_SET,
        "training_inflow_sha256": model.metadata["training_sha256"],
        "class_numbering": "zero_based_0_to_9",
        "n_inflow_classes": 10,
        "water_level_step_m": 1.0,
        "state_grid_shape": [n_1, n_2, 10, 10],
        "action_table_shape": [12, n_1, n_2, 10, 10],
        "joint_action_count": n_1 * n_2,
    }


class SharedBatchPhysicsTests(unittest.TestCase):
    def assert_batch_matches_scalar(self, spec, starts, inflows, targets, seconds):
        batch = _transition_arrays(spec, starts, inflows, targets, seconds)
        shape = np.broadcast_shapes(np.shape(starts), np.shape(inflows), np.shape(targets))
        broadcast_starts = np.broadcast_to(starts, shape)
        broadcast_inflows = np.broadcast_to(inflows, shape)
        broadcast_targets = np.broadcast_to(targets, shape)
        for index in np.ndindex(shape):
            scalar = _transition(
                spec,
                float(broadcast_starts[index]),
                float(broadcast_inflows[index]),
                float(broadcast_targets[index]),
                seconds,
            )
            for name in (
                "end_storage_1e9m3", "end_level_m", "total_discharge_m3s", "power_kw",
                "energy_gwh", "firm_deficit_gwh", "water_balance_error_m3",
            ):
                self.assertAlmostEqual(float(getattr(batch, name)[index]), float(getattr(scalar, name)), places=8)

    def test_batch_transition_matches_scalar_physics_field_by_field(self):
        spec = replace(make_spec("test"), max_total_release_m3s=300.0)
        starts = np.array([[0.0], [0.7], [2.0]])
        targets = np.array([[0.0, 0.8, 2.0, 99.0]])
        inflows = np.array([[0.0], [100.0], [1000.0]])
        seconds = 31.0 * 86400.0
        self.assert_batch_matches_scalar(spec, starts, inflows, targets, seconds)

    def test_batch_transition_matches_both_configured_reservoirs(self):
        root = Path(__file__).resolve().parents[1]
        specs = load_specs_from_yaml(root / "configs" / "xiaowan_nuozhadu_v1.yaml")[:2]
        for spec in specs:
            starts = np.array([
                [spec.dead_storage_1e9m3],
                [spec.storage_from_level((spec.dead_level_m + spec.normal_level_m) / 2.0)],
                [spec.normal_storage_1e9m3],
            ])
            targets = np.array([[spec.dead_level_m, (spec.dead_level_m + spec.normal_level_m) / 2.0, spec.normal_level_m]])
            inflows = np.array([[1.0], [3000.0], [50000.0]])
            with self.subTest(spec=spec.name):
                self.assert_batch_matches_scalar(spec, starts, inflows, targets, 30.0 * 86400.0)


class CascadeSDPTests(unittest.TestCase):
    def test_formal_default_is_one_metre_and_two_metre_grid_remains_available(self):
        root = Path(__file__).resolve().parents[1]
        settings = _parse_simple_yaml(
            (root / "configs" / "xiaowan_nuozhadu_sdp.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(DEFAULT_WATER_LEVEL_STEP_M, 1.0)
        self.assertEqual(settings["sdp"]["water_level_step"], 1.0)
        self.assertEqual(settings["sdp"]["supported_water_level_steps"], [1.0, 2.0])

        model = MonthlyInflowModel.fit(probability_frame(), expected_years=2)
        spec_1, spec_2 = make_spec("xw"), make_spec("nzd")
        formal = CascadeSDP(spec_1, spec_2, model)
        legacy = CascadeSDP(spec_1, spec_2, model, water_level_step=2.0)
        np.testing.assert_array_equal(formal.levels_1, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(formal.levels_2, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(legacy.levels_1, np.array([0.0, 2.0]))
        np.testing.assert_array_equal(legacy.levels_2, np.array([0.0, 2.0]))
        configured_1, configured_2 = load_specs_from_yaml(
            root / "configs" / "xiaowan_nuozhadu_v1.yaml"
        )[:2]
        configured = CascadeSDP(configured_1, configured_2, model)
        self.assertEqual((len(configured.levels_1), len(configured.levels_2)), (75, 48))
        np.testing.assert_allclose(np.diff(configured.levels_1), 1.0, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(np.diff(configured.levels_2), 1.0, rtol=0.0, atol=1.0e-12)
        self.assertEqual(len(configured.levels_1) * len(configured.levels_2), 3600)
        self.assertEqual(format_water_level_step_label(1.0), "1m")
        self.assertEqual(format_water_level_step_label(2.0), "2m")
        self.assertNotEqual(format_water_level_step_label(1.0), format_water_level_step_label(2.0))
        new_one_metre_root = Path("train/sdp") / sdp_output_variant_path(1.0)
        new_two_metre_root = Path("train/sdp") / sdp_output_variant_path(2.0)
        old_two_metre_root = Path("train/sdp/2m")
        self.assertNotEqual(new_one_metre_root, new_two_metre_root)
        self.assertNotEqual(new_two_metre_root, old_two_metre_root)

    def test_bilinear_continuation_matches_grid_midpoint_and_clamped_boundaries(self):
        grid_1 = np.array([0.0, 1.0])
        grid_2 = np.array([0.0, 2.0])
        values = np.array([[0.0, 4.0], [2.0, 6.0]])
        self.assertEqual(
            float(bilinear_interpolate_storage_values(grid_1, grid_2, values, 1.0, 2.0)),
            6.0,
        )
        self.assertEqual(
            float(bilinear_interpolate_storage_values(grid_1, grid_2, values, 0.5, 1.0)),
            3.0,
        )
        np.testing.assert_array_equal(
            bilinear_interpolate_storage_values(
                grid_1,
                grid_2,
                values,
                np.array([-1.0, 2.0]),
                np.array([3.0, -1.0]),
            ),
            np.array([4.0, 2.0]),
        )

    def test_bellman_maximizes_for_each_known_current_class_and_expects_only_next_month(self):
        frame = probability_frame()
        frame.loc[12:, "xiaowan_inflow_m3s"] = 200.0
        frame.loc[12:, "nuozhadu_local_inflow_m3s"] = 60.0
        model = MonthlyInflowModel.fit(frame, expected_years=2)
        spec_1, spec_2 = make_spec("xw"), make_spec("nzd")
        solver = CascadeSDP(
            spec_1,
            spec_2,
            model,
            water_level_step=1.0,
            firm_power_failure_penalty_gwh=0.7,
            max_physics_batch_elements=27,
        )
        next_month_values = np.arange(3 * 3 * 10 * 10, dtype=float).reshape(3, 3, 10, 10) / 10.0
        future = solver._expected_over_current_classes(1, next_month_values)
        scalar_future = np.zeros((3, 3), dtype=float)
        for next_state_1 in range(3):
            for next_state_2 in range(3):
                for next_class_1 in range(10):
                    for next_class_2 in range(10):
                        scalar_future[next_state_1, next_state_2] += (
                            model.probabilities[0, 1, next_class_1]
                            * model.probabilities[1, 1, next_class_2]
                            * next_month_values[next_state_1, next_state_2, next_class_1, next_class_2]
                        )
        np.testing.assert_allclose(future, scalar_future, rtol=0.0, atol=1.0e-13)

        values, actions_1, actions_2 = solver._bellman_month(0, future)
        seconds = model.mean_days[0] * 86400.0
        expected_values = np.full((3, 3, 10, 10), -np.inf)
        expected_actions_1 = np.zeros((3, 3, 10, 10), dtype=int)
        expected_actions_2 = np.zeros((3, 3, 10, 10), dtype=int)
        for state_1 in range(3):
            for state_2 in range(3):
                for class_1 in range(10):
                    for class_2 in range(10):
                        for action_1 in range(3):
                            upstream = _transition(
                                spec_1,
                                solver.storages_1[state_1],
                                model.representatives[0, 0, class_1],
                                solver.levels_1[action_1],
                                seconds,
                            )
                            for action_2 in range(3):
                                downstream = _transition(
                                    spec_2,
                                    solver.storages_2[state_2],
                                    model.representatives[1, 0, class_2] + upstream.total_discharge_m3s,
                                    solver.levels_2[action_2],
                                    seconds,
                                )
                                stage = (
                                    upstream.energy_gwh
                                    + downstream.energy_gwh
                                    - 0.7
                                    * (
                                        int(upstream.firm_deficit_gwh > 0.0)
                                        + int(downstream.firm_deficit_gwh > 0.0)
                                    )
                                )
                                continuation = float(bilinear_interpolate_storage_values(
                                    solver.storages_1,
                                    solver.storages_2,
                                    scalar_future,
                                    upstream.end_storage_1e9m3,
                                    downstream.end_storage_1e9m3,
                                ))
                                candidate = stage + continuation
                                key = (state_1, state_2, class_1, class_2)
                                if candidate > expected_values[key]:
                                    expected_values[key] = candidate
                                    expected_actions_1[key] = action_1
                                    expected_actions_2[key] = action_2
        np.testing.assert_allclose(values, expected_values, rtol=1.0e-13, atol=1.0e-13)
        np.testing.assert_array_equal(actions_1, expected_actions_1)
        np.testing.assert_array_equal(actions_2, expected_actions_2)

    def test_policy_action_table_uses_current_classes_and_csv_exports_every_key(self):
        model = MonthlyInflowModel.fit(probability_frame(), expected_years=2)
        levels = np.array([0.0, 1.0, 2.0])
        actions_1 = np.zeros((12, 3, 3, 10, 10), dtype=np.int16)
        actions_2 = np.zeros_like(actions_1)
        actions_1[0, 1, 1, 9, 9] = 2
        actions_2[0, 1, 1, 9, 9] = 1
        policy = SDPPolicy(
            levels,
            levels,
            actions_1,
            actions_2,
            policy_metadata(model),
        )
        self.assertEqual(policy.action_indices_1.shape, (12, 3, 3, 10, 10))
        self.assertEqual(policy.action(1, (1.1, 0.9), (0, 0)), (0.0, 0.0))
        self.assertEqual(policy.action(1, (1.1, 0.9), (9, 9)), (2.0, 1.0))
        with tempfile.TemporaryDirectory() as folder:
            archive = policy.save(Path(folder) / "policy.npz")
            loaded = SDPPolicy.load(archive)
            np.testing.assert_array_equal(loaded.action_indices_1, policy.action_indices_1)
            exported = loaded.export_csv(Path(folder) / "policy.csv")
            with exported.open(encoding="utf-8") as stream:
                self.assertEqual(sum(1 for _ in stream), 1 + 12 * 3 * 3 * 100)
            table = pd.read_csv(exported)
            self.assertEqual(
                list(table.columns),
                [
                    "month",
                    "xiaowan_level_m",
                    "nuozhadu_level_m",
                    "current_xiaowan_natural_inflow_class",
                    "current_nuozhadu_interval_inflow_class",
                    "target_xiaowan_level_m",
                    "target_nuozhadu_level_m",
                ],
            )
            self.assertEqual(len(table), 12 * 3 * 3 * 100)
        policy.require_water_level_step(1.0)
        with self.assertRaisesRegex(CascadeSpecError, "does not match"):
            policy.require_water_level_step(2.0)

    def test_schema_1_policy_is_rejected(self):
        model = MonthlyInflowModel.fit(probability_frame(), expected_years=2)
        legacy_metadata = policy_metadata(model)
        legacy_metadata["schema_version"] = 1
        actions = np.zeros((12, 3, 3, 10, 10), dtype=np.int16)
        with self.assertRaisesRegex(CascadeSpecError, "schema"):
            SDPPolicy(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
                actions,
                actions.copy(),
                legacy_metadata,
            )


class EvaluationTests(unittest.TestCase):
    def test_closed_loop_observes_current_classes_then_uses_real_flows_and_continuous_storage(self):
        training = probability_frame()
        training.loc[12:, "xiaowan_inflow_m3s"] = 200.0
        training.loc[12:, "nuozhadu_local_inflow_m3s"] = 60.0
        model = MonthlyInflowModel.fit(training, expected_years=2)
        spec_1 = make_spec("xw", normal_level=10.0)
        spec_2 = make_spec("nzd", normal_level=10.0)
        actions_1 = np.full((12, 3, 3, 10, 10), 1, dtype=np.int16)
        actions_2 = np.full_like(actions_1, 1)
        base = SDPPolicy(
            np.array([0.0, 5.0, 10.0]),
            np.array([0.0, 5.0, 10.0]),
            actions_1,
            actions_2,
            {**policy_metadata(model), "water_level_step_m": 5.0},
        )

        class SpyPolicy:
            metadata = base.metadata

            def __init__(self):
                self.calls = []

            def action(self, month, levels_m, current_classes):
                self.calls.append((month, tuple(levels_m), tuple(current_classes)))
                return base.action(month, levels_m, current_classes)

        spy = SpyPolicy()
        evaluation = pd.DataFrame({
            "date": ["2030-01-01", "2030-02-01", "2030-03-01"],
            "xiaowan_inflow_m3s": [112.5, 187.5, 250.0],
            "nuozhadu_local_inflow_m3s": [22.0, 55.0, 5.0],
        })
        schedule, audit, summary = evaluate_policy(
            spec_1, spec_2, spy, model, evaluation, (0.0, 0.0)
        )
        self.assertEqual(list(schedule.columns), DDP_COMPATIBLE_COLUMNS)
        for index, row in evaluation.iterrows():
            month = index + 1
            real_flows = (row["xiaowan_inflow_m3s"], row["nuozhadu_local_inflow_m3s"])
            self.assertEqual(spy.calls[index][2], model.classify(month, real_flows))
            self.assertEqual(schedule.loc[index, "inflow_1_m3s"], real_flows[0])
            self.assertEqual(schedule.loc[index, "inflow_2_local_m3s"], real_flows[1])
        self.assertNotEqual(spy.calls[1][1][0], 0.0)
        self.assertAlmostEqual(spy.calls[1][1][0], schedule.loc[0, "r1_end_level_m"])
        self.assertAlmostEqual(spy.calls[1][1][1], schedule.loc[0, "r2_end_level_m"])
        self.assertAlmostEqual(spy.calls[2][1][0], schedule.loc[1, "r1_end_level_m"])
        self.assertAlmostEqual(spy.calls[2][1][1], schedule.loc[1, "r2_end_level_m"])
        self.assertAlmostEqual(
            audit.loc[1, "decision_start_storage_1_1e9m3"],
            audit.loc[1, "previous_end_storage_1_1e9m3"],
        )
        self.assertAlmostEqual(
            audit.loc[1, "decision_start_storage_2_1e9m3"],
            audit.loc[1, "previous_end_storage_2_1e9m3"],
        )
        np.testing.assert_allclose(
            audit.loc[1:, [
                "xiaowan_inter_month_storage_jump_m3",
                "nuozhadu_inter_month_storage_jump_m3",
            ]].to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertAlmostEqual(schedule.loc[0, "routed_inflow_m3s"], schedule.loc[0, "r1_total_discharge_m3s"])
        self.assertEqual(
            audit.loc[1, "current_xiaowan_inflow_class_at_decision"],
            model.classify(2, (187.5, 55.0))[0],
        )
        self.assertEqual(summary["forecast_timing"], "current_month_source_classes_known_before_policy_action")
        self.assertEqual(summary["forecast_flow_application"], "same_realized_continuous_source_flows_used_in_physical_transition")
        self.assertEqual(summary["nuozhadu_forecast_inflow_definition"], "local_incremental_inflow")
        self.assertTrue(summary["water_balance_within_tolerance"])
        self.assertEqual(summary["inter_month_storage_continuity"]["comparisons"], 2)
        self.assertTrue(summary["inter_month_storage_continuity_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
