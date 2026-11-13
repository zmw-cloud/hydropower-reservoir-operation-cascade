"""Run research stages from source; generated dependencies remain under outputs/."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


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
        if method == "iso" and stage == "teacher":
            return [["scripts/run_p4_v1_ddp_interpolated.py", "--split", "train",
                     "--failure-penalty-gwh", lam, "--output-dir",
                     f"outputs/DDP/teachers/failure_penalty_{lam}"]]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("inspect", help="Show paper parameters and supported stages")
    run = subs.add_parser("run", help="Execute a stage, reusing dependencies generated under outputs/")
    run.add_argument("--algorithm", choices=tuple(profile()["methods"]), required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--show-commands", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    p = profile()
    if args.action == "inspect":
        print(json.dumps(p, ensure_ascii=False, indent=2))
        return
    if args.stage not in p["methods"][args.algorithm]["stages"]:
        parser.error("Unsupported stage; run inspect to see the supported stages")
    for command in commands(p, args.algorithm, args.stage):
        if args.show_commands:
            print(subprocess.list2cmdline([sys.executable, "-B", *command]))
        else:
            subprocess.run([sys.executable, "-B", str(ROOT / command[0]), *command[1:]],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
