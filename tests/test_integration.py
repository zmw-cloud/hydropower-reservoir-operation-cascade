"""Small invariant and packaging checks; no formal training/evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

class IntegrationTests(unittest.TestCase):
    def test_recorded_results_and_units(self):
        import audit_repository
        result = audit_repository.audit()
        self.assertEqual(result["status"], "PASS")
        iso = next(x for x in result["representatives"] if x["case"] == "Case 1" and x["method"] == "ISO-RF")
        self.assertAlmostEqual(iso["station_rates_pct"][0], 81.97916666666667)

    def test_formal_scope_and_bound_artifacts(self):
        import run_workflow
        p = run_workflow.profile()
        self.assertEqual(p["formal_scenarios"], ["train", "evaluation"])
        for b in p["bindings"]:
            self.assertTrue(run_workflow.native(ROOT / b["source"]).is_file(), b)
            self.assertFalse(Path(b["target"]).is_absolute())
            self.assertNotIn("..", Path(b["target"]).parts)
        with self.assertRaises(ValueError):
            run_workflow.commands(p, "dqn", "drought")

    def test_paper_configurations_match_the_final_experiments(self):
        import yaml
        if (ROOT / "src/cascade_reservoir_benchmark").is_dir():
            dqn = yaml.safe_load((ROOT / "configs/paper/dqn5000.yaml").read_text(encoding="utf-8"))
            ppo = yaml.safe_load((ROOT / "configs/paper/ppo6000.yaml").read_text(encoding="utf-8"))
            self.assertEqual(dqn["network"]["action_representation"], "action_conditioned")
            self.assertEqual(dqn["training"]["episodes"], 4800)
            development = yaml.safe_load((ROOT / "configs/paper/dqn5000_development.yaml").read_text(encoding="utf-8"))
            self.assertEqual(development["training"]["episodes"], 3000)
            self.assertEqual(ppo["reward"]["firm_power_failure_penalty_gwh"], 6000)
            self.assertTrue(ppo["information"]["geometry_features_are_current_state_derived"])
        else:
            dqn = yaml.safe_load((ROOT / "data/experiment_settings.yaml").read_text(encoding="utf-8"))["dqn"]
            self.assertEqual(dqn["agent"]["memory_size"], 50000)
            self.assertEqual(dqn["agent"]["epsilon_decay_steps"], 72000)
            self.assertEqual(dqn["train"]["final_budget"], 2300)
            self.assertEqual(dqn["experiment"]["seed_list"], [1, 42, 100])

    def test_physical_invariants_on_one_month(self):
        if (ROOT / "src/reservoir_benchmark").is_dir():
            from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
            model = SharedReservoirModel(ROOT / "data")
            result = model.calculate_transition(1240., 1236., 1000., 31., month=5)
            self.assertLess(abs(result["water_balance_error_m3"]), .001)
            self.assertLessEqual(result["end_level"], 1236. + 1e-8)
            self.assertAlmostEqual(result["total_discharge"], result["turbine_discharge"] + result["spill_discharge"])
        else:
            from cascade_reservoir_benchmark.environment import load_specs_from_yaml, CascadeReservoirEnv
            one, two, cfg = load_specs_from_yaml(ROOT / "configs/xiaowan_nuozhadu_v1.yaml")
            limits = list(zip(cfg["ddp_constraints"]["monthly_max_target_levels_m"]["xiaowan"],
                              cfg["ddp_constraints"]["monthly_max_target_levels_m"]["nuozhadu"]))
            env = CascadeReservoirEnv(one, two, [[1000., 200.]], timestamps=["2009-05-01"],
                                      initial_levels_m=[1240., 812.], monthly_max_target_levels_m=limits)
            env.reset()
            _, _, _, _, info = env.step([1236., 804.])
            self.assertAlmostEqual(info["effective_inflow_2_m3s"], 200. + info["reservoir_1"]["total_discharge_m3s"])
            for k in ("reservoir_1", "reservoir_2"):
                self.assertLess(abs(info[k]["water_balance_error_m3"]), .001)

    def test_extreme_release_storage_conflict_is_reported(self):
        # An impossible flood inflow exceeds the release cap while storage is full.
        # Preserve, expose and document this historical edge behavior rather than
        # silently changing the physical implementation that produced the paper.
        if (ROOT / "src/reservoir_benchmark").is_dir():
            from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
            model = SharedReservoirModel(ROOT / "data")
            result = model.calculate_transition(1240., 1240., 30000., 31., month=1)
            self.assertTrue(result["max_total_release_triggered"])
            self.assertGreater(result["unresolved_upper_storage_excess_m3"], 0)
            self.assertGreater(abs(result["water_balance_error_m3"]), .001)
        else:
            from cascade_reservoir_benchmark.environment import load_specs_from_yaml, _transition
            one, _, _ = load_specs_from_yaml(ROOT / "configs/xiaowan_nuozhadu_v1.yaml")
            result = _transition(one, one.normal_storage_1e9m3, 30000., 1240., 31 * 86400.)
            self.assertEqual(result.total_discharge_m3s, 20683.)
            self.assertGreater(abs(result.water_balance_error_m3), .001)

if __name__ == "__main__":
    unittest.main()
