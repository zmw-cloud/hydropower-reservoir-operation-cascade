"""Check immutable release inputs and recorded trajectories; never run a model."""
from __future__ import annotations
import argparse
import calendar
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]

def native(p):
    p = Path(p).absolute()
    return Path("\\\\?\\" + str(p)) if os.name == "nt" and not str(p).startswith("\\\\?\\") else p

def read_json(p):
    return json.loads(native(p).read_text(encoding="utf-8-sig"))

def official_mean(d):
    for k in ("mean_annual_generation_gwh", "mean_annual_cascade_generation_gwh"):
        if k in d:
            return float(d[k])
    for k in ("total_generation_gwh", "cascade_generation_gwh"):
        if k in d:
            return float(d[k]) / 400
    raise ValueError("Unknown official generation schema")

def official_rates(d, single):
    if single:
        return [float(d["guarantee_rate"]) * 100]
    if "firm_power_reliability" in d:
        return [float(d["firm_power_reliability"][s]) * 100 for s in ("xiaowan", "nuozhadu")]
    if "xiaowan_firm_power_reliability" in d:
        return [float(d[s + "_firm_power_reliability"]) * 100 for s in ("xiaowan", "nuozhadu")]
    return [(1 - float(d[s + "_failed_months"]) / 4800) * 100 for s in ("xiaowan", "nuozhadu")]

def audit(check_hashes=True, plots=False):
    manifest = read_json(ROOT / "provenance/release_manifest.json")
    if check_hashes:
        for name, expected in manifest.items():
            b = native(ROOT / name).read_bytes()
            if hashlib.sha256(b).hexdigest() != expected["sha256"]:
                raise ValueError("Release file changed: " + name)
    config = read_json(ROOT / "configs/paper/reference_results.json")
    per_seed, aggregate, baselines = [], [], {}
    for rec in config["records"]:
        single = rec["case"] == "Case 1"
        rows = list(csv.DictReader(io.StringIO(native(ROOT / rec["trajectory"]).read_text(encoding="utf-8-sig"))))
        d = read_json(ROOT / rec["summary"])
        assert len(rows) == 4800, rec
        dates = [(int(x["Calendar_Year"]), int(x["Calendar_Month"])) if single else
                 (int(x["date"][:4]), int(x["date"][5:7])) for x in rows]
        assert dates[0] == (2009, 1) and dates[-1] == (2408, 12), rec
        assert all(y * 12 + m == 2009 * 12 + 1 + i for i, (y, m) in enumerate(dates)), rec
        flows = [tuple(float(x[k]) for k in (("Inflow_m3s",) if single else ("inflow_1_m3s", "inflow_2_local_m3s"))) for x in rows]
        if rec["case"] in baselines:
            assert flows == baselines[rec["case"]], "Different evaluation inflows"
        else:
            baselines[rec["case"]] = flows
        firm = config["firm_power_kw"][rec["case"]]
        powers = [tuple(float(x[k]) for k in (("Power_kW",) if single else ("r1_power_kw", "r2_power_kw"))) for x in rows]
        energy = [float(x["Electricity_GWh"]) if single else float(x["r1_energy_gwh"]) + float(x["r2_energy_gwh"]) for x in rows]
        err = max(abs(e - sum(p) * calendar.monthrange(y, m)[1] * 24 / 1e6)
                  for e, p, (y, m) in zip(energy, powers, dates))
        assert err < 1e-8, (rec, err)
        mean = math.fsum(energy) / 400
        assert abs(mean - official_mean(d)) < 1e-5, (rec, mean, official_mean(d))
        rates = [100 * sum(p[j] >= threshold for p in powers) / 4800 for j, threshold in enumerate(firm)]
        assert all(abs(x-y) < 1e-7 for x, y in zip(rates, official_rates(d, single))), (rec, rates)
        wb = max(abs(float(x[k])) for x in rows for k in
                 (("Water_Balance_Error_m3",) if single else ("r1_water_balance_error_m3", "r2_water_balance_error_m3")))
        assert wb < .001, (rec, wb)
        per_seed.append({**{k: rec[k] for k in ("case", "method", "penalty", "seed")},
                         "annual_generation_gwh": mean, "station_rates_pct": rates,
                         "both_station_rate_pct": 100 * sum(all(p[j] >= f for j, f in enumerate(firm)) for p in powers) / 4800,
                         "sum_power_rate_pct": 100 * sum(sum(p) >= sum(firm) for p in powers) / 4800,
                         "max_water_balance_error_m3": wb})
    for key in dict.fromkeys((x["case"], x["method"]) for x in per_seed):
        group = [x for x in per_seed if (x["case"], x["method"]) == key]
        means = [x["annual_generation_gwh"] for x in group]
        rates = [statistics.mean(x["station_rates_pct"][j] for x in group) for j in range(len(group[0]["station_rates_pct"]))]
        aggregate.append({"case": key[0], "method": key[1], "mean_annual_generation_gwh": statistics.mean(means),
                          "across_seed_sample_sd_gwh": statistics.stdev(means) if len(means) > 1 else None,
                          "station_rates_pct": rates, "minimum_of_mean_station_rates_pct": min(rates)})
    if plots:
        if not (ROOT / "configs/paper_results.json").is_file():
            raise ValueError("Single repository intentionally has no plotting scripts")
        sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
        paper = importlib.import_module("paper_results_data")
        paper.collect(paper.load_config())
        tradeoff = importlib.import_module("plot_generation_reliability_tradeoff")
        assert len(tradeoff.load_tradeoff_points()) > 0
        from cascade_reservoir_benchmark.comparison import load_comparison_bundle, build_analysis_tables
        bundle = load_comparison_bundle(ROOT, ROOT / "configs/comparison_ddp_sdp_iso_dqn_ppo.yaml")
        tables = build_analysis_tables(bundle)
        assert not tables.monthly_level_statistics.empty
        action = importlib.import_module("plot_dqn_ppo_action_behavior")
        _, runs = action.load_action_runs(ROOT / "configs/comparison_ddp_sdp_iso_dqn_ppo.yaml")
        assert set(runs) == {"DQN", "PPO"}
        # Hydrology is bundled independently; check its data/statistics without drawing.
        from importlib import util as import_util
        path = ROOT / "hydrology/figures/plot_figure2_hydrology_v2.py"
        spec = import_util.spec_from_file_location("bundled_hydrology_audit", path)
        hydro = import_util.module_from_spec(spec)
        spec.loader.exec_module(hydro)
        summary = hydro.calculate_summary({"HIST": hydro.read_historical(),
                    "TRAIN": hydro.read_synthetic(hydro.TRAIN_FILE, "TRAIN"),
                    "EVAL": hydro.read_synthetic(hydro.EVAL_FILE, "EVAL")})
        assert len(summary) == 108
        registry = read_json(ROOT / "configs/paper/figure_registry.json")
        assert len(registry["figures"]) == 8
    return {"status": "PASS", "release_files_checked": len(manifest) if check_hashes else 0,
            "evaluation_trajectories": len(per_seed), "per_seed": per_seed, "representatives": aggregate,
            "plot_inputs_checked": plots, "models_executed": False, "figures_generated": False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots", action="store_true", help="Check requested figure data only; do not render")
    p.add_argument("--report", type=Path, help="Write a NEW JSON under outputs/, using exclusive creation")
    args = p.parse_args()
    result = audit(plots=args.plots)
    if args.report:
        target = args.report.absolute()
        if not target.is_relative_to((ROOT / "outputs").absolute()):
            raise ValueError("Audit reports must be under this repository's outputs/")
        native(target.parent).mkdir(parents=True, exist_ok=True)
        with native(target).open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
