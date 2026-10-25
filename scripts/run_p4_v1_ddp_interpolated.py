"""Run the interpolated-continuation parallel DDP perfect-foresight bound."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade_reservoir_benchmark.data import load_configured_joint_inflows
from cascade_reservoir_benchmark.ddp import summarize_ddp_trajectory
from cascade_reservoir_benchmark.ddp_parallel_interpolated import (
    InterpolatedParallelCascadeDDP,
)
from cascade_reservoir_benchmark.environment import load_specs_from_yaml


CONFIG = ROOT / "configs" / "xiaowan_nuozhadu_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "evaluation"], default="evaluation")
    parser.add_argument("--limit-months", type=int, default=None)
    parser.add_argument("--grid-step-m", type=float, default=None)
    parser.add_argument("--progress-every-months", type=int, default=1)
    parser.add_argument("--checkpoint-every-months", type=int, default=100)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("DDP_NUM_WORKERS", "8")),
    )
    parser.add_argument("--state-block-size", type=int, default=None)
    parser.add_argument("--multiprocessing-start-method", default=None)
    parser.add_argument(
        "--failure-penalty-gwh",
        type=float,
        default=None,
        help="Override reward.firm_power_failure_penalty_gwh without editing the physics config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Isolated output directory relative to the project root or an absolute path.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_months is not None and args.limit_months <= 0:
        raise ValueError("--limit-months must be positive.")
    if args.checkpoint_every_months <= 0:
        raise ValueError("--checkpoint-every-months must be positive.")

    reservoir_1, reservoir_2, config = load_specs_from_yaml(CONFIG)
    constraints = config["ddp_constraints"]
    reward_config = config.get("reward", {})
    if "firm_power_failure_penalty_gwh" not in reward_config:
        raise KeyError(
            "Missing reward.firm_power_failure_penalty_gwh in configuration."
        )
    reward_config = dict(reward_config)
    firm_power_failure_penalty_gwh = float(
        args.failure_penalty_gwh
        if args.failure_penalty_gwh is not None
        else reward_config["firm_power_failure_penalty_gwh"]
    )
    if (
        not math.isfinite(firm_power_failure_penalty_gwh)
        or firm_power_failure_penalty_gwh < 0.0
    ):
        raise ValueError(
            "reward.firm_power_failure_penalty_gwh must be a finite "
            "nonnegative value."
        )
    reward_config["firm_power_failure_penalty_gwh"] = (
        firm_power_failure_penalty_gwh
    )
    initial_levels = tuple(float(value) for value in constraints["initial_levels_m"])
    terminal_levels = tuple(float(value) for value in constraints["terminal_levels_m"])
    monthly_max_levels = list(
        zip(
            constraints["monthly_max_target_levels_m"]["xiaowan"],
            constraints["monthly_max_target_levels_m"]["nuozhadu"],
        )
    )
    joint = load_configured_joint_inflows(config, args.split)
    if args.limit_months is not None:
        joint = joint.head(args.limit_months)

    grid_step_m = float(
        args.grid_step_m
        if args.grid_step_m is not None
        else constraints["action_grid_step_m"]
    )
    if args.output_dir is None:
        output_dir = ROOT / "outputs" / "xiaowan_nuozhadu" / args.split / "ddp"
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_label = f"{args.split}_400y"
    grid_label = f"{grid_step_m:g}m"
    limit_label = f"_first{len(joint)}months" if args.limit_months is not None else ""
    output_stem = (
        f"ddp_p4_v1_{split_label}_{grid_label}{limit_label}_"
        "interpolated_failure_count_v1"
    )
    checkpoint_path = output_dir / "checkpoints" / f"{output_stem}.checkpoint.npz"

    solver = InterpolatedParallelCascadeDDP(
        reservoir_1,
        reservoir_2,
        joint[["xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s"]].to_numpy(),
        joint["date"],
        grid_step_m=grid_step_m,
        firm_power_failure_penalty_gwh=firm_power_failure_penalty_gwh,
        num_workers=args.num_workers,
        state_block_size=args.state_block_size,
        multiprocessing_start_method=args.multiprocessing_start_method,
        monthly_max_target_levels_m=monthly_max_levels,
    )
    schedule, summary = solver.solve(
        initial_levels,
        progress_every_months=args.progress_every_months,
        checkpoint_path=checkpoint_path,
        checkpoint_every_months=args.checkpoint_every_months,
        resume=args.resume,
        terminal_levels_m=terminal_levels,
    )

    trajectory_diagnostics = summarize_ddp_trajectory(
        schedule,
        reservoir_1,
        reservoir_2,
    )

    for number in (1, 2):
        details = schedule.pop(f"reservoir_{number}")
        for key in details.iloc[0]:
            schedule[f"r{number}_{key}"] = details.map(
                lambda item, key=key: item[key]
            )

    csv_path = output_dir / f"{output_stem}.csv"
    summary_path = csv_path.with_suffix(".json")
    schedule.to_csv(csv_path, index=False)
    summary.update(
        {
            "configuration": CONFIG.name,
            "inflow_split": split_label,
            "inflow_source": config["data"][f"{args.split}_inflow_file"],
            "ddp_constraints": constraints,
            "reward_configuration": reward_config,
            "physical_parameters": {
                "xiaowan": {
                    "normal_level_m": reservoir_1.normal_level_m,
                    "flood_limit_level_m": config["reservoirs"]["xiaowan"]["flood_limit_level_m"],
                    "max_turbine_discharge_m3s": reservoir_1.max_turbine_discharge_m3s,
                    "max_power_kw": reservoir_1.max_power_kw,
                    "firm_power_kw": reservoir_1.firm_power_kw,
                    "max_total_release_m3s": reservoir_1.max_total_release_m3s,
                },
                "nuozhadu": {
                    "normal_level_m": reservoir_2.normal_level_m,
                    "flood_limit_level_m": config["reservoirs"]["nuozhadu"]["flood_limit_level_m"],
                    "max_turbine_discharge_m3s": reservoir_2.max_turbine_discharge_m3s,
                    "max_power_kw": reservoir_2.max_power_kw,
                    "firm_power_kw": reservoir_2.firm_power_kw,
                    "max_total_release_m3s": reservoir_2.max_total_release_m3s,
                },
            },
            "trajectory_diagnostics": trajectory_diagnostics,
            "assumption_register": "docs/audit/uncertain_parameter_register_p4_v1.md",
            "warning": "DDP uses future inflow and is not a deployable policy.",
            "configured_grid_step_scenarios_m": [0.5, 1.0, 2.0, 5.0],
            "max_abs_water_balance_error_m3": trajectory_diagnostics[
                "max_abs_water_balance_error_m3"
            ],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
