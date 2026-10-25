"""Train and evaluate the P4 dual-reservoir one-step-forecast SDP policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade_reservoir_benchmark.data import load_configured_joint_inflows
from cascade_reservoir_benchmark.environment import _parse_simple_yaml, load_specs_from_yaml
from cascade_reservoir_benchmark.sdp import (
    CascadeSDP,
    SDP_SOLVER_VARIANT,
    SDPPolicy,
    format_water_level_step_label,
    sdp_output_variant_path,
)
from cascade_reservoir_benchmark.sdp_evaluation import evaluate_policy
from cascade_reservoir_benchmark.sdp_probability import MonthlyInflowModel


SDP_CONFIG = ROOT / "configs" / "xiaowan_nuozhadu_sdp.yaml"


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)
    return yaml.safe_load(text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def format_failure_penalty_label(value: float) -> str:
    """Return a stable path label for one failed-station-month penalty."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("SDP failure penalty must be finite and nonnegative.")
    text = f"{value:.15g}"
    return text.replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preprocess", "train", "evaluate", "all"], default="all")
    parser.add_argument(
        "--water-level-step",
        type=float,
        default=None,
        help="State and target-action level step in metres (formal default: 1; supported: 1 or 2).",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--tolerance-gwh", type=float, default=None)
    parser.add_argument("--stable-policy-iterations", type=int, default=None)
    parser.add_argument(
        "--failure-penalty-gwh",
        type=float,
        default=None,
        help=(
            "Override sdp.firm_power_failure_penalty_gwh for one isolated "
            "training/evaluation run."
        ),
    )
    parser.add_argument(
        "--skip-policy-csv",
        action="store_true",
        help="Save the compact policy archive but skip the multi-million-row policy CSV.",
    )
    parser.add_argument(
        "--reuse-probabilities",
        action="store_true",
        help="Load validated existing probability artifacts instead of refitting and rewriting them.",
    )
    parser.add_argument("--limit-evaluation-months", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _load_yaml(SDP_CONFIG)
    sdp_settings = settings["sdp"]
    solver_settings = sdp_settings["solver"]
    required_settings = {
        "n_inflow_classes": MonthlyInflowModel.n_inflow_classes,
        "probability_method": "monthly_equal_width_empirical_frequency",
        "source_dependence": "independent_marginal_product",
        "representative_inflow": "interval_midpoint",
        "information_set": "month_levels_current_source_classes_one_step_perfect_forecast",
        "forecast_horizon_months": 1,
        "forecast_xiaowan_inflow": "natural_inflow",
        "forecast_nuozhadu_inflow": "local_incremental_inflow",
        "nuozhadu_total_inflow": "local_incremental_inflow_plus_action_dependent_routed_xiaowan_release",
        "failure_definition": "station_specific_monthly_power_below_firm_power",
        "objective": "generation_gwh_minus_penalty_gwh_per_failed_station_month",
        "deficit_energy_role": "diagnostic_only_not_in_sdp_objective",
    }
    for key, expected in required_settings.items():
        if sdp_settings.get(key) != expected:
            raise ValueError(f"Unsupported SDP setting {key}={sdp_settings.get(key)!r}; expected {expected!r}.")
    physical_config = Path(settings["physical_config"])
    if not physical_config.is_absolute():
        physical_config = (ROOT / physical_config).resolve()
    reservoir_1, reservoir_2, physical = load_specs_from_yaml(physical_config)
    constraints = physical["ddp_constraints"]
    monthly_max_levels = list(
        zip(
            constraints["monthly_max_target_levels_m"]["xiaowan"],
            constraints["monthly_max_target_levels_m"]["nuozhadu"],
        )
    )

    water_level_step = float(
        args.water_level_step if args.water_level_step is not None else sdp_settings["water_level_step"]
    )
    failure_penalty_gwh = float(
        args.failure_penalty_gwh
        if args.failure_penalty_gwh is not None
        else sdp_settings["firm_power_failure_penalty_gwh"]
    )
    penalty_label = format_failure_penalty_label(failure_penalty_gwh)
    supported_steps = tuple(float(value) for value in sdp_settings["supported_water_level_steps"])
    if (
        not math.isfinite(water_level_step)
        or water_level_step <= 0.0
        or not any(math.isclose(water_level_step, value, rel_tol=0.0, abs_tol=1.0e-12) for value in supported_steps)
    ):
        choices = ", ".join(f"{value:g}" for value in supported_steps)
        raise ValueError(f"--water-level-step must be one of the configured values: {choices}.")
    grid_label = format_water_level_step_label(water_level_step)
    output_root = Path(settings["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    output_variant_path = sdp_output_variant_path(water_level_step)
    train_root = output_root / "train" / "sdp" / output_variant_path
    probability_dir = train_root / "probabilities"
    penalty_directory = Path(f"failure_penalty_{penalty_label}")
    policy_dir = train_root / "policy" / penalty_directory
    cache_dir = train_root / "cache" / penalty_directory
    artifact_stem = (
        f"sdp_p4_v4_one_step_forecast_{SDP_SOLVER_VARIANT}_train_400y_"
        f"{grid_label}_failure_count_penalty_{penalty_label}"
    )
    policy_path = policy_dir / f"{artifact_stem}.npz"
    policy_table_path = policy_dir / f"{artifact_stem}_five_state_policy.csv"

    if args.reuse_probabilities and args.stage == "preprocess":
        raise ValueError("--reuse-probabilities cannot be combined with --stage preprocess.")
    needs_training_data = args.stage in {"preprocess", "train", "all"}
    fitted_model = None
    if needs_training_data:
        if args.reuse_probabilities:
            fitted_model = MonthlyInflowModel.load(probability_dir)
            print(f"Loaded existing monthly source probabilities from {probability_dir}")
        else:
            train = load_configured_joint_inflows(physical, "train")
            fitted_model = MonthlyInflowModel.fit(
                train,
                expected_years=int(sdp_settings["expected_training_years"]),
            )

    if args.stage in {"preprocess", "train", "all"} and not args.reuse_probabilities:
        paths = fitted_model.save(probability_dir)
        print(f"Wrote monthly source probabilities to {probability_dir}")
        if args.stage == "preprocess":
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, ensure_ascii=False))

    if args.stage in {"train", "all"}:
        solver = CascadeSDP(
            reservoir_1,
            reservoir_2,
            fitted_model,
            water_level_step=water_level_step,
            firm_power_failure_penalty_gwh=failure_penalty_gwh,
            monthly_max_target_levels_m=monthly_max_levels,
            cache_dir=cache_dir,
            water_balance_tolerance_m3=float(sdp_settings["water_balance_tolerance_m3"]),
            max_physics_batch_elements=int(sdp_settings["max_physics_batch_elements"]),
        )
        policy = solver.solve(
            max_iterations=int(
                args.max_iterations if args.max_iterations is not None else solver_settings["max_iterations"]
            ),
            tolerance_gwh=float(
                args.tolerance_gwh if args.tolerance_gwh is not None else solver_settings["tolerance_gwh"]
            ),
            stable_iterations=int(
                args.stable_policy_iterations
                if args.stable_policy_iterations is not None
                else solver_settings["stable_policy_iterations"]
            ),
            progress=not args.quiet,
        )
        policy.metadata.update({
            "sdp_configuration": SDP_CONFIG.name,
            "sdp_configuration_sha256": _sha256(SDP_CONFIG),
            "physical_configuration": physical_config.name,
            "physical_configuration_sha256": _sha256(physical_config),
            "training_inflow_source": physical["data"]["train_inflow_file"],
        })
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy.save(policy_path)
        exported_policy_table = None
        if not args.skip_policy_csv:
            exported_policy_table = policy.export_csv(policy_table_path)
        training_summary = {
            **policy.metadata,
            "configuration": SDP_CONFIG.name,
            "configuration_sha256": _sha256(SDP_CONFIG),
            "physical_configuration": physical_config.name,
            "physical_configuration_sha256": _sha256(physical_config),
            "training_inflow_source": physical["data"]["train_inflow_file"],
            "probability_directory": str(probability_dir),
            "policy_archive": str(policy_path),
            "five_state_policy_table": (
                None if exported_policy_table is None else str(exported_policy_table)
            ),
            "firm_power_failure_penalty_gwh": failure_penalty_gwh,
        }
        training_summary_path = policy_path.with_suffix(".json")
        training_summary_path.write_text(
            json.dumps(training_summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote SDP policy to {policy_path}")

    if args.stage in {"evaluate", "all"}:
        model = MonthlyInflowModel.load(probability_dir)
        policy = SDPPolicy.load(policy_path)
        policy.require_water_level_step(water_level_step)
        if policy.metadata.get("solver_variant") != SDP_SOLVER_VARIANT:
            raise RuntimeError("The saved SDP policy does not use bilinear continuation values.")
        if policy.metadata.get("sdp_configuration_sha256") != _sha256(SDP_CONFIG):
            raise RuntimeError("The saved SDP policy does not match the current SDP configuration.")
        if policy.metadata.get("physical_configuration_sha256") != _sha256(physical_config):
            raise RuntimeError("The saved SDP policy does not match the current physical configuration.")
        evaluation = load_configured_joint_inflows(physical, "evaluation")
        if args.limit_evaluation_months is not None:
            if args.limit_evaluation_months <= 0:
                raise ValueError("--limit-evaluation-months must be positive.")
            evaluation = evaluation.head(args.limit_evaluation_months)
        schedule, state_audit, summary = evaluate_policy(
            reservoir_1,
            reservoir_2,
            policy,
            model,
            evaluation,
            (
                float(physical["reservoirs"]["xiaowan"]["initial_level_m"]),
                float(physical["reservoirs"]["nuozhadu"]["initial_level_m"]),
            ),
            firm_power_failure_penalty_gwh=failure_penalty_gwh,
            monthly_max_target_levels_m=monthly_max_levels,
            water_balance_tolerance_m3=float(sdp_settings["water_balance_tolerance_m3"]),
        )
        if summary["evaluation_inflow_sha256"] == summary["training_inflow_sha256"]:
            raise RuntimeError("Training and evaluation inflow content must be independent.")
        evaluation_dir = (
            output_root
            / "evaluation"
            / "sdp"
            / output_variant_path
            / penalty_directory
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        limit_label = f"_first{len(evaluation)}months" if args.limit_evaluation_months is not None else ""
        evaluation_stem = (
            f"sdp_p4_v4_one_step_forecast_{SDP_SOLVER_VARIANT}_evaluation_400y_"
            f"{grid_label}_failure_count_penalty_{penalty_label}{limit_label}"
        )
        csv_path = evaluation_dir / f"{evaluation_stem}.csv"
        audit_path = csv_path.with_name(csv_path.stem + "_state_audit.csv")
        summary_path = csv_path.with_suffix(".json")
        schedule.to_csv(csv_path, index=False)
        state_audit.to_csv(audit_path, index=False)
        summary.update({
            "configuration": SDP_CONFIG.name,
            "configuration_sha256": _sha256(SDP_CONFIG),
            "physical_configuration": physical_config.name,
            "physical_configuration_sha256": _sha256(physical_config),
            "inflow_split": "evaluation_400y",
            "inflow_source": physical["data"]["evaluation_inflow_file"],
            "policy_archive": str(policy_path),
            "probability_directory": str(probability_dir),
            "state_audit_csv": str(audit_path),
            "assumption_register": "docs/audit/uncertain_parameter_register_p4_v1.md",
        })
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {csv_path}")
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
