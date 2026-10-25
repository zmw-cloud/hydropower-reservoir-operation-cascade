"""Prepare isolated runs of the unchanged research algorithms.

Preparation copies inputs and selected frozen artifacts. Execution is a separate,
explicit command. It never evaluates or trains while inspecting/preparing.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

def native(path):
    p = Path(path).absolute()
    return Path("\\\\?\\" + str(p)) if os.name == "nt" and not str(p).startswith("\\\\?\\") else p

def profile():
    return json.loads((ROOT / "configs/paper/workflows.json").read_text(encoding="utf-8"))

def commands(p, method, stage):
    settings = p["methods"][method]
    lam = str(settings["penalty"])
    if p["system"] == "single":
        if method == "ddp" and stage == "solve":
            return [["scripts/run_single_comparison.py", "ddp-evaluation"]]
        if method == "sdp" and stage in ("train", "evaluate"):
            return [["scripts/run_single_comparison.py", "sdp-training" if stage == "train" else "sdp-evaluation"]]
        if method == "dqn" and stage in ("development", "final", "evaluate"):
            c = ["scripts/run_single_dqn.py", "evaluation" if stage == "evaluate" else stage, "--lambda", lam]
            return [c + ["--final-budget", str(settings["final_budget"])] if stage == "final" else c]
        if method == "iso" and stage in ("development", "lambda-scan", "select-lambda", "final-train", "evaluate"):
            return [["scripts/run_single_iso_rf.py", "evaluation" if stage == "evaluate" else stage]]
    else:
        config = settings.get("development_config", settings["config"]) if stage == "development" else settings["config"]
        if method == "ddp" and stage == "solve":
            return [["scripts/run_p4_v1_ddp_interpolated.py", "--split", "evaluation",
                     "--failure-penalty-gwh", lam]]
        if method == "sdp" and stage in ("train", "evaluate"):
            entry = "scripts/run_p4_v1_sdp.py"
            suffix = ["--failure-penalty-gwh", lam, "--skip-policy-csv"]
            return [[entry, "--stage", s] + suffix for s in (("preprocess", "train") if stage == "train" else ("evaluate",))]
        if method == "iso" and stage in ("development", "final-train", "evaluate"):
            return [["scripts/run_p4_v1_iso.py", stage, "--config", config]]
        if method in ("dqn", "ppo") and stage in ("development", "final", "evaluate"):
            entry = "scripts/run_p4_v1_dqn.py" if method == "dqn" else "scripts/run_p4_v1_continuous_ppo.py"
            return [[entry, "--config", config, "--stage", "evaluate" if stage == "evaluate" else "train",
                     "--mode", "development" if stage == "development" else "final"]]
    raise ValueError(f"Unsupported explicit stage: {method}/{stage}")

def prepare(method, stage, output=None):
    p = profile()
    cmds = commands(p, method, stage)
    run = output or ROOT / "outputs/runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{method}_{stage}_{uuid4().hex[:8]}"
    run = Path(run).absolute()
    if not run.is_relative_to((ROOT / "outputs").absolute()):
        raise ValueError("Runs must be new directories under this repository's outputs/")
    native(run).mkdir(parents=True, exist_ok=False)
    work = run / "workspace"
    native(work).mkdir()
    for name in ("src", "scripts", "data", "configs"):
        shutil.copytree(native(ROOT / name), native(work / name),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Archived SDP configuration differs only by the recorded original trailing newline.
    if p["system"] == "cascade" and method == "sdp":
        for filename in ("xiaowan_nuozhadu_v1.yaml", "xiaowan_nuozhadu_sdp.yaml"):
            src = ROOT / "provenance/original_configs/sdp" / filename
            shutil.copyfile(native(src), native(work / "configs" / filename))
    copied = []
    for item in p["bindings"]:
        if not item["source"].startswith("artifacts/" + ("iso_rf" if p["system"] == "single" and method == "iso" else method) + "/"):
            continue
        if stage not in item["stages"]:
            continue
        target = work / item["target"]
        if not target.absolute().is_relative_to(work.absolute()):
            raise ValueError("Artifact binding escapes the new workspace")
        native(target.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native(ROOT / item["source"]), native(target))
        copied.append(item)
    plan = {"system": p["system"], "algorithm": method, "stage": stage,
            "workspace": str(work), "commands": cmds, "artifact_bindings": copied,
            "status": "prepared_only", "training_or_simulation_executed": False}
    native(run / "run_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return run

def execute(run):
    run = Path(run).absolute()
    if not run.is_relative_to((ROOT / "outputs").absolute()):
        raise ValueError("Execution is restricted to prepared runs in this repository's outputs/")
    plan = json.loads(native(run / "run_plan.json").read_text(encoding="utf-8"))
    work = Path(plan["workspace"])
    if work != run / "workspace":
        raise ValueError("Prepared workspace path changed; prepare a fresh run after relocation")
    if plan["algorithm"] == "iso" and plan["stage"] == "evaluate":
        recorded = "1.7.2" if plan["system"] == "single" else "1.9.1"
        installed = importlib.metadata.version("scikit-learn")
        if installed != recorded:
            raise RuntimeError(f"Frozen RF recorded scikit-learn {recorded}; current interpreter has {installed}. "
                               "Use the matching RF environment; no model has been executed or resaved.")
    versions = {"python": sys.version}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "gymnasium", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    with native(run / "execution_started.json").open("x", encoding="utf-8") as f:
        json.dump({"algorithm": plan["algorithm"], "stage": plan["stage"], "runtime_versions": versions}, f, indent=2)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    codes = []
    with native(run / "execution.log").open("x", encoding="utf-8") as log:
        for cmd in plan["commands"]:
            # Extended Windows script paths propagate a long-path-safe __file__ root.
            command = [sys.executable, "-B", str(native(work / cmd[0])), *cmd[1:]]
            result = subprocess.run(command, cwd=native(work), env=env, stdout=log, stderr=subprocess.STDOUT)
            codes.append(result.returncode)
            if result.returncode:
                break
    native(run / "execution_result.json").write_text(json.dumps({"return_codes": codes}, indent=2), encoding="utf-8")
    return 0 if all(c == 0 for c in codes) else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("inspect", help="Print frozen profiles without loading models or running algorithms")
    prep = subs.add_parser("prepare", help="Create a fresh workspace; no algorithm is executed")
    prep.add_argument("--algorithm", choices=tuple(profile()["methods"]), required=True)
    prep.add_argument("--stage", required=True)
    prep.add_argument("--output-dir", type=Path)
    run = subs.add_parser("execute", help="Explicitly execute one previously prepared run")
    run.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    if args.action == "inspect":
        p = profile()
        print(json.dumps({k: v for k, v in p.items() if k != "bindings"}, ensure_ascii=False, indent=2))
    elif args.action == "prepare":
        print("PREPARED_ONLY=" + str(prepare(args.algorithm, args.stage, args.output_dir)))
    else:
        raise SystemExit(execute(args.run_directory))

if __name__ == "__main__":
    main()
