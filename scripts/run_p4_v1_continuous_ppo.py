"""Inspect, train, validate, and evaluate continuous-target cascade PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
import yaml

from cascade_reservoir_benchmark.continuous_ppo_agent import ContinuousPPOAgent
from cascade_reservoir_benchmark.continuous_ppo_evaluation import (
    evaluate_continuous_ppo_trajectory,
)
from cascade_reservoir_benchmark.continuous_ppo_training import (
    CONTINUOUS_PPO_TRAINING_MODES,
    build_continuous_ppo_training_paths,
    inspect_continuous_ppo_configuration,
    load_continuous_ppo_checkpoint,
    train_continuous_ppo_seed,
)
from cascade_reservoir_benchmark.data import load_configured_joint_inflows
from cascade_reservoir_benchmark.dqn_adapter import StateNormalizer, normalized_inflow_sha256
from cascade_reservoir_benchmark.dqn_training import (
    configuration_sha256,
    experiment_id_from_config,
    split_training_inflows,
)
from cascade_reservoir_benchmark.environment import CascadeSpecError, load_specs_from_yaml
from cascade_reservoir_benchmark.sdp import reservoir_pair_sha256


DEFAULT_CONFIG = ROOT / "configs" / "paper" / "ppo6000.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CascadeSpecError(f"Configuration must be a mapping: {path}")
    return payload


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CascadeSpecError(f"Configuration section {key!r} is missing.")
    return value


def _resolve_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CascadeSpecError(f"{label} must be a nonempty path.")
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolve_config_path(value: str | None) -> Path:
    return DEFAULT_CONFIG if value is None else _resolve_path(value, label="config")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _seeds(config: Mapping[str, Any], override: list[int] | None) -> tuple[int, ...]:
    values = list(_mapping(config, "training").get("seeds", [])) if override is None else override
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result) or any(value < 0 for value in result):
        raise CascadeSpecError("Continuous PPO seeds must be unique nonnegative integers.")
    return result


def _device(config: Mapping[str, Any]) -> str:
    requested = str(_mapping(config, "training").get("device", "auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise CascadeSpecError("CUDA requested but unavailable.")
    return requested


def _agent_from_checkpoint(
    checkpoint: Mapping[str, Any], *, device: str
) -> ContinuousPPOAgent:
    state = checkpoint.get("agent_state")
    if not isinstance(state, Mapping):
        raise CascadeSpecError("Continuous checkpoint lacks agent state.")
    settings = state.get("configuration")
    if not isinstance(settings, Mapping):
        raise CascadeSpecError("Continuous checkpoint lacks agent configuration.")
    agent = ContinuousPPOAgent(
        state_dimension=int(settings["state_dimension"]),
        actor_hidden_dimensions=settings["actor_hidden_dimensions"],
        critic_hidden_dimensions=settings["critic_hidden_dimensions"],
        learning_rate=float(settings["learning_rate"]),
        gamma=float(settings["gamma"]),
        gae_lambda=float(settings["gae_lambda"]),
        clip_range=float(settings["clip_range"]),
        entropy_coefficient=float(settings["entropy_coefficient"]),
        value_loss_coefficient=float(settings["value_loss_coefficient"]),
        max_gradient_norm=float(settings["max_gradient_norm"]),
        minibatch_size=int(settings["minibatch_size"]),
        update_epochs=int(settings["update_epochs"]),
        target_kl=float(settings["target_kl"]),
        advantage_normalization=bool(settings["advantage_normalization"]),
        adam_epsilon=float(settings["adam_epsilon"]),
        initial_log_std=float(settings["initial_log_std"]),
        min_log_std=float(settings["min_log_std"]),
        max_log_std=float(settings["max_log_std"]),
        squash_epsilon=float(settings["squash_epsilon"]),
        seed=int(settings["seed"]),
        device=device,
    )
    agent.load_inference_state(state)
    return agent


def _load_selected_policy(
    checkpoint_path: Path,
    *,
    config: Mapping[str, Any],
    mode: str,
    seed: int,
    reservoir_1,
    reservoir_2,
    training_frame: pd.DataFrame,
    physical_hash: str,
) -> tuple[ContinuousPPOAgent, StateNormalizer, dict[str, Any]]:
    checkpoint = load_continuous_ppo_checkpoint(
        checkpoint_path, map_location=_device(config)
    )
    validation = _mapping(config, "validation")
    learning, validation_frame = split_training_inflows(
        training_frame,
        mode=mode,
        development_years=int(validation["development_years"]),
        validation_years=int(validation["validation_years"]),
    )
    expected = {
        "checkpoint_role": "best_validation" if mode == "development" else "final",
        "experiment_id": experiment_id_from_config(config),
        "mode": mode,
        "seed": seed,
        "configuration_sha256": configuration_sha256(config),
        "reservoir_physics_sha256": reservoir_pair_sha256(reservoir_1, reservoir_2),
        "physical_configuration_sha256": physical_hash,
        "source_training_inflow_sha256": normalized_inflow_sha256(training_frame),
        "training_inflow_sha256": normalized_inflow_sha256(learning),
        "validation_inflow_sha256": (
            None if validation_frame is None else normalized_inflow_sha256(validation_frame)
        ),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise CascadeSpecError(f"Selected continuous checkpoint {name} differs.")
    normalizer = StateNormalizer.from_state_dict(
        checkpoint["normalizer"], reservoir_1, reservoir_2
    )
    expected_normalizer = StateNormalizer.fit(
        learning, reservoir_1, reservoir_2, clip=True
    )
    if normalizer.state_dict() != expected_normalizer.state_dict():
        raise CascadeSpecError("Continuous checkpoint normalizer differs.")
    return _agent_from_checkpoint(checkpoint, device=_device(config)), normalizer, checkpoint


def _progress(payload: Mapping[str, Any]) -> None:
    training = payload.get("training", {})
    validation = payload.get("validation")
    message = (
        f"Continuous PPO {payload.get('mode')} seed={payload.get('seed')} "
        f"update={training.get('update_number')} "
        f"steps={training.get('environment_steps')} "
        f"reward_gwh={training.get('raw_generation_mean_gwh')} "
        f"policy_loss={training.get('policy_loss')} "
        f"value_loss={training.get('value_loss')} "
        f"entropy={training.get('policy_entropy')} "
        f"approx_kl={training.get('approx_kl')} "
        f"lr={training.get('learning_rate')} "
        f"epochs={training.get('epochs_completed')} "
        f"kl_stop={training.get('early_stop_triggered')} "
        f"std=({training.get('xiaowan_action_std')},"
        f"{training.get('nuozhadu_action_std')})"
    )
    if isinstance(validation, Mapping):
        message += f" validation={validation.get('score')}"
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--stage", choices=["inspect", "train", "validate", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--mode", choices=list(CONTINUOUS_PPO_TRAINING_MODES), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    config = _load_yaml(config_path)
    experiment_id = experiment_id_from_config(config)
    if experiment_id is None:
        raise CascadeSpecError("Continuous PPO requires experiment_id.")
    physical_path = _resolve_path(
        _mapping(config, "problem").get("physical_config"), label="physical_config"
    )
    reservoir_1, reservoir_2, physical = load_specs_from_yaml(physical_path)
    constraints = _mapping(physical, "ddp_constraints")
    monthly_limits = list(
        zip(
            constraints["monthly_max_target_levels_m"]["xiaowan"],
            constraints["monthly_max_target_levels_m"]["nuozhadu"],
            strict=True,
        )
    )
    physical_hash = _sha256(physical_path)
    output_root = _resolve_path(_mapping(config, "outputs").get("root"), label="outputs.root")
    seeds = _seeds(config, args.seeds)

    if args.stage == "inspect":
        inspection = inspect_continuous_ppo_configuration(
            reservoir_1, reservoir_2, config, mode=args.mode, seed=seeds[0]
        )
        inspection.update(
            {
                "configuration": str(config_path),
                "configuration_file_sha256": _sha256(config_path),
                "physical_configuration": str(physical_path),
                "physical_configuration_file_sha256": physical_hash,
                "output_root": str(output_root),
                "monthly_max_target_levels_m": monthly_limits,
            }
        )
        print(json.dumps(inspection, indent=2, sort_keys=True, default=_json_default))
        return

    training_frame = load_configured_joint_inflows(physical, "train")
    if args.stage == "validate" and args.mode != "development":
        raise CascadeSpecError("Standalone validation requires development mode.")
    if args.stage == "evaluate" and args.mode != "final":
        raise CascadeSpecError("Independent evaluation requires final mode.")

    if args.stage in {"train", "all"}:
        for seed in seeds:
            paths = build_continuous_ppo_training_paths(
                output_root, mode=args.mode, seed=seed, experiment_id=experiment_id
            )
            _write_json(
                paths.configuration_snapshot_json,
                {
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "mode": args.mode,
                    "source_config": str(config_path),
                    "configuration_sha256": configuration_sha256(config),
                    "configuration": config,
                },
            )
            result = train_continuous_ppo_seed(
                reservoir_1,
                reservoir_2,
                training_frame,
                config,
                mode=args.mode,
                seed=seed,
                output_root=output_root,
                resume_latest=args.resume,
                physical_configuration_sha256=physical_hash,
                monthly_max_target_levels_m=monthly_limits,
                progress_callback=_progress,
            )
            metadata = result.as_dict()
            metadata.update(
                {
                    "continuous_ppo_configuration": str(config_path),
                    "continuous_ppo_configuration_file_sha256": _sha256(config_path),
                    "physical_configuration": str(physical_path),
                    "physical_configuration_file_sha256": physical_hash,
                    "training_inflow_source": physical["data"]["train_inflow_file"],
                    "hyperparameters": _mapping(config, "ppo"),
                    "network": _mapping(config, "network"),
                    "training": _mapping(config, "training"),
                }
            )
            _write_json(result.paths.metadata_json, metadata)
            print(
                f"Completed continuous PPO {args.mode} seed={seed}: "
                f"{result.selected_checkpoint}",
                flush=True,
            )

    run_trajectory = bool(
        args.stage in {"validate", "evaluate"}
        or (args.stage == "all" and args.mode == "final")
    )
    if not run_trajectory:
        return
    if args.stage == "validate":
        _, trajectory = split_training_inflows(
            training_frame,
            mode="development",
            development_years=int(_mapping(config, "validation")["development_years"]),
            validation_years=int(_mapping(config, "validation")["validation_years"]),
        )
        if trajectory is None:
            raise CascadeSpecError("Continuous validation trajectory is missing.")
        trajectory_kind = "validation"
    else:
        trajectory = load_configured_joint_inflows(physical, "evaluation")
        if normalized_inflow_sha256(trajectory) == normalized_inflow_sha256(training_frame):
            raise CascadeSpecError("Continuous training and evaluation inflows are identical.")
        trajectory_kind = "evaluation"

    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        paths = build_continuous_ppo_training_paths(
            output_root, mode=args.mode, seed=seed, experiment_id=experiment_id
        )
        checkpoint_path = (
            paths.best_validation_checkpoint if args.mode == "development" else paths.final_checkpoint
        )
        if checkpoint_path is None:
            raise CascadeSpecError("Continuous selected checkpoint path is missing.")
        agent, normalizer, checkpoint = _load_selected_policy(
            checkpoint_path,
            config=config,
            mode=args.mode,
            seed=seed,
            reservoir_1=reservoir_1,
            reservoir_2=reservoir_2,
            training_frame=training_frame,
            physical_hash=physical_hash,
        )
        scaling = checkpoint["reward_scaling"]
        schedule, summary = evaluate_continuous_ppo_trajectory(
            reservoir_1,
            reservoir_2,
            agent,
            normalizer,
            trajectory,
            _mapping(config, "problem")["initial_levels_m"],
            training_inflow_sha256=checkpoint["training_inflow_sha256"],
            seed=seed,
            reward_reference_days=float(scaling["reference_days"]),
            reward_clip=scaling["clip"],
            firm_power_failure_penalty_gwh=float(
                scaling["firm_power_failure_penalty_gwh"]
            ),
            monthly_max_target_levels_m=monthly_limits,
            trajectory_kind=trajectory_kind,
            collect_rows=True,
        )
        if schedule is None:
            raise CascadeSpecError("Continuous trajectory rows are missing.")
        summary.update(
            {
                "mode": args.mode,
                "experiment_id": experiment_id,
                "checkpoint": str(checkpoint_path),
                "checkpoint_role": checkpoint["checkpoint_role"],
                "checkpoint_environment_steps": checkpoint["environment_steps"],
                "configuration": str(config_path),
                "physical_configuration": str(physical_path),
                "training_inflow_source": physical["data"]["train_inflow_file"],
                "trajectory_source": (
                    physical["data"]["evaluation_inflow_file"]
                    if trajectory_kind == "evaluation"
                    else "chronological_final_80_years_of_training_split"
                ),
                "normalization_source": "frozen_training_checkpoint",
                "normalization_updated_during_trajectory": False,
            }
        )
        directory = paths.run_root / trajectory_kind
        csv_path = directory / f"{trajectory_kind}_monthly.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(csv_path, index=False)
        _write_json(csv_path.with_suffix(".json"), summary)
        summaries.append(summary)
        print(f"Wrote continuous PPO {trajectory_kind} seed={seed}: {csv_path}")

    aggregate_dir = output_root / "experiments" / experiment_id / trajectory_kind
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "seed": item["seed"],
                "mean_annual_generation_gwh": item["mean_annual_cascade_generation_gwh"],
                "mean_annual_penalized_objective_gwh": item[
                    "mean_annual_penalized_objective_gwh"
                ],
                "xiaowan_reliability": item["firm_power_reliability"]["xiaowan"],
                "nuozhadu_reliability": item["firm_power_reliability"]["nuozhadu"],
                "failed_station_months": item["failed_station_months"],
            }
            for item in summaries
        ]
    ).to_csv(aggregate_dir / f"{trajectory_kind}_seed_summary.csv", index=False)
    print(f"Wrote continuous PPO {trajectory_kind} cross-seed summary: {aggregate_dir}")


if __name__ == "__main__":
    main()
