"""Inspect, train, validate, and evaluate the P4 centralized cascade Double DQN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from cascade_reservoir_benchmark.data import (
    load_configured_joint_inflows,
    load_synthetic_joint_inflows,
)
from cascade_reservoir_benchmark.dqn_adapter import (
    JointActionGrid,
    StateNormalizer,
    normalized_inflow_sha256,
)
from cascade_reservoir_benchmark.dqn_agent import ACTION_MASK_MODE, DoubleDQNAgent
from cascade_reservoir_benchmark.dqn_evaluation import (
    evaluate_greedy_trajectory,
    summarize_monthly_physical_trajectory,
)
from cascade_reservoir_benchmark.dqn_training import (
    TRAINING_MODES,
    build_training_paths,
    configuration_sha256,
    experiment_id_from_config,
    inspect_dqn_budget_extension,
    inspect_dqn_configuration,
    load_training_checkpoint,
    split_training_inflows,
    train_dqn_seed,
)
from cascade_reservoir_benchmark.environment import (
    CascadeSpecError,
    _parse_simple_yaml,
    load_specs_from_yaml,
)
from cascade_reservoir_benchmark.sdp import reservoir_pair_sha256


DEFAULT_DQN_CONFIG = ROOT / "configs" / "paper" / "dqn5000.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        parsed = _parse_simple_yaml(text)
    else:
        parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise CascadeSpecError(f"DQN configuration must be a mapping: {path}")
    return parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CascadeSpecError(f"DQN configuration section {key!r} must be a mapping.")
    return value


def _evaluation_scenario(
    config: Mapping[str, Any], scenario: str
) -> tuple[Path, str]:
    scenarios = _mapping(config, "evaluation_scenarios")
    selected = scenarios.get(scenario)
    if not isinstance(selected, Mapping):
        raise CascadeSpecError(
            f"evaluation_scenarios.{scenario} must be a mapping."
        )
    role = selected.get("role")
    if not isinstance(role, str) or not role.strip():
        raise CascadeSpecError(
            f"evaluation_scenarios.{scenario}.role must be nonempty."
        )
    path = _resolve_path(
        selected.get("inflow_file"),
        label=f"evaluation_scenarios.{scenario}.inflow_file",
    )
    return path, role.strip()


def _resolve_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CascadeSpecError(f"{label} must be a nonempty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _resolve_config_path(value: str | None) -> Path:
    if value is None:
        return DEFAULT_DQN_CONFIG
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (ROOT / candidate).resolve()
    )


def _configured_seeds(config: Mapping[str, Any], override: Sequence[int] | None) -> list[int]:
    raw = override if override is not None else _mapping(config, "training").get("seeds")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise CascadeSpecError("training.seeds must be a sequence of integers.")
    seeds: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise CascadeSpecError("Every DQN seed must be an integer.")
        seed = int(value)
        if seed < 0:
            raise CascadeSpecError("DQN seeds must be nonnegative.")
        if seed in seeds:
            raise CascadeSpecError(f"Duplicate DQN seed: {seed}.")
        seeds.append(seed)
    if not seeds:
        raise CascadeSpecError("At least one DQN seed is required.")
    return seeds


def _device_from_config(config: Mapping[str, Any]) -> str:
    value = _mapping(config, "training").get("device")
    if not isinstance(value, str) or not value.strip():
        raise CascadeSpecError("training.device must be a nonempty string.")
    requested = value.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CascadeSpecError("CUDA was requested but is unavailable.")
    return str(device)


def _agent_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    device: str,
) -> DoubleDQNAgent:
    agent_state = checkpoint.get("agent_state")
    if not isinstance(agent_state, Mapping):
        raise CascadeSpecError("DQN checkpoint is missing agent_state.")
    settings = agent_state.get("configuration")
    if not isinstance(settings, Mapping):
        raise CascadeSpecError("DQN checkpoint is missing agent configuration.")
    agent = DoubleDQNAgent(
        state_dimension=int(settings["state_dimension"]),
        action_count=int(settings["action_count"]),
        hidden_dimensions=settings["hidden_dimensions"],
        learning_rate=float(settings["learning_rate"]),
        gamma=float(settings["gamma"]),
        epsilon_start=float(settings["epsilon_start"]),
        epsilon_min=float(settings["epsilon_min"]),
        epsilon_schedule=str(settings["epsilon_schedule"]),
        epsilon_decay_steps=int(settings["epsilon_decay_steps"]),
        learning_starts=int(settings["learning_starts"]),
        target_update_interval=int(settings["target_update_interval"]),
        replay_capacity=int(settings["replay_capacity"]),
        batch_size=int(settings["batch_size"]),
        max_grad_norm=settings["max_grad_norm"],
        seed=int(settings["seed"]),
        device=device,
        q_network_mode=str(settings.get("q_network_mode", "indexed_output")),
        action_features=settings.get("action_features"),
        action_score_chunk_size=int(settings.get("action_score_chunk_size", 256)),
    )
    agent.load_inference_state(agent_state)
    return agent


def _selected_checkpoint_path(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> Path:
    paths = build_training_paths(
        output_root,
        mode=mode,
        seed=seed,
        output_sections=_mapping(config, "outputs"),
        experiment_id=experiment_id_from_config(config),
    )
    selected = (
        paths.best_validation_checkpoint
        if mode == "development"
        else paths.final_checkpoint
    )
    if selected is None:
        raise CascadeSpecError(f"No selected checkpoint role is defined for mode {mode}.")
    return selected


def _load_selected_policy(
    checkpoint_path: Path,
    *,
    mode: str,
    seed: int,
    config: Mapping[str, Any],
    reservoir_1,
    reservoir_2,
    training_frame: pd.DataFrame,
    physical_configuration_sha256: str,
) -> tuple[DoubleDQNAgent, StateNormalizer, JointActionGrid, dict[str, Any]]:
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        map_location=_device_from_config(config),
    )
    expected_role = "best_validation" if mode == "development" else "final"
    validation = _mapping(config, "validation")
    learning_frame, validation_frame = split_training_inflows(
        training_frame,
        mode=mode,
        development_years=int(validation["development_years"]),
        validation_years=int(validation["validation_years"]),
    )
    expected = {
        "checkpoint_role": expected_role,
        "experiment_id": experiment_id_from_config(config),
        "mode": mode,
        "seed": seed,
        "configuration_sha256": configuration_sha256(config),
        "reservoir_physics_sha256": reservoir_pair_sha256(
            reservoir_1, reservoir_2
        ),
        "physical_configuration_sha256": physical_configuration_sha256,
        "source_training_inflow_sha256": normalized_inflow_sha256(training_frame),
        "training_inflow_sha256": normalized_inflow_sha256(learning_frame),
        "validation_inflow_sha256": (
            None
            if validation_frame is None
            else normalized_inflow_sha256(validation_frame)
        ),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise CascadeSpecError(
                f"Selected DQN checkpoint {key} does not match this workflow."
            )
    if configuration_sha256(checkpoint["configuration_snapshot"]) != checkpoint[
        "configuration_sha256"
    ]:
        raise CascadeSpecError("DQN checkpoint configuration snapshot hash is invalid.")

    normalizer = StateNormalizer.from_state_dict(
        checkpoint["normalizer"], reservoir_1, reservoir_2
    )
    action_grid = JointActionGrid.from_state_dict(
        checkpoint["action_grid"], reservoir_1, reservoir_2
    )
    expected_normalizer = StateNormalizer.fit(
        learning_frame,
        reservoir_1,
        reservoir_2,
        clip=bool(_mapping(config, "normalization")["clip"]),
    )
    if normalizer.state_dict() != expected_normalizer.state_dict():
        raise CascadeSpecError(
            "DQN checkpoint normalization does not match its permitted training subset."
        )
    expected_grid = JointActionGrid(
        reservoir_1,
        reservoir_2,
        step_m=float(_mapping(config, "problem")["action_level_step_m"]),
    )
    if action_grid.state_dict() != expected_grid.state_dict():
        raise CascadeSpecError(
            "DQN checkpoint joint action grid does not match the active configuration."
        )
    agent = _agent_from_checkpoint(
        checkpoint,
        device=_device_from_config(config),
    )
    return agent, normalizer, action_grid, checkpoint


def _evaluation_settings(
    checkpoint: Mapping[str, Any],
    *,
    reservoir_1,
    reservoir_2,
) -> tuple[float, Sequence[float] | None, float]:
    scaling = checkpoint.get("reward_scaling")
    if not isinstance(scaling, Mapping):
        raise CascadeSpecError("DQN checkpoint is missing reward scaling metadata.")
    if scaling.get("objective") != "generation_gwh_minus_penalty_gwh_per_failed_station_month":
        raise CascadeSpecError("DQN checkpoint has an incompatible reward objective.")
    failure_penalty = float(
        scaling.get("firm_power_failure_penalty_gwh", float("nan"))
    )
    if not np.isfinite(failure_penalty) or failure_penalty < 0.0:
        raise CascadeSpecError("DQN checkpoint has an invalid failure penalty.")
    reference_days = float(scaling["reference_days"])
    expected_scale = (
        (float(reservoir_1.max_power_kw) + float(reservoir_2.max_power_kw))
        * reference_days
        * 24.0
        / 1.0e6
    )
    if not np.isclose(
        float(scaling.get("scale_gwh", float("nan"))),
        expected_scale,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise CascadeSpecError(
            "DQN checkpoint reward scale does not match the active reservoir specs."
        )
    clip = scaling.get("clip")
    return reference_days, clip, failure_penalty


def _write_trajectory_outputs(
    schedule: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(csv_path, index=False)
    _write_json(csv_path.with_suffix(".json"), summary)


def _progress(payload: Mapping[str, Any]) -> None:
    training = payload.get("training", {})
    validation = payload.get("validation")
    message = (
        f"DQN {payload.get('mode')} seed={payload.get('seed')} "
        f"episode={training.get('episode')} "
        f"steps={training.get('environment_steps')} "
        f"learning_steps={training.get('learning_steps')} "
        f"epsilon={training.get('epsilon')} "
        f"raw_gwh={training.get('raw_episode_cascade_generation_gwh')} "
        f"normalized_reward={training.get('normalized_episode_reward')} "
        f"checkpoint={training.get('checkpoint_event')}"
    )
    if isinstance(validation, Mapping):
        message += (
            " validation_mean_annual_gwh="
            f"{validation.get('validation_mean_annual_cascade_generation_gwh')}"
        )
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "DQN YAML path relative to the project root. Default: "
            "configs/xiaowan_nuozhadu_dqn.yaml"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=[
            "inspect",
            "inspect-checkpoint",
            "train",
            "validate",
            "evaluate",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--mode", choices=list(TRAINING_MODES), required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--evaluation-scenario",
        choices=["baseline"],
        default="baseline",
        help="Final hold-out evaluation scenario; ignored outside evaluation.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume each seed from its strictly matching latest checkpoint.",
    )
    resume_group.add_argument(
        "--budget-extension-source",
        default=None,
        help=(
            "Import one completed Run02 episode-3000 latest.pt into an "
            "isolated 5000-episode development run. With --stage inspect, "
            "perform compatibility preflight only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    config = _load_yaml(config_path)
    experiment_id = experiment_id_from_config(config)
    physical_config = _resolve_path(
        _mapping(config, "problem").get("physical_config"),
        label="problem.physical_config",
    )
    reservoir_1, reservoir_2, physical = load_specs_from_yaml(physical_config)
    constraints = physical["ddp_constraints"]
    monthly_max_target_levels = list(
        zip(
            constraints["monthly_max_target_levels_m"]["xiaowan"],
            constraints["monthly_max_target_levels_m"]["nuozhadu"],
        )
    )
    physical_config_hash = _sha256(physical_config)
    output_root = _resolve_path(
        _mapping(config, "outputs").get("root"), label="outputs.root"
    )
    seeds = _configured_seeds(config, args.seeds)
    budget_extension_source = (
        None
        if args.budget_extension_source is None
        else _resolve_path(
            args.budget_extension_source,
            label="budget-extension-source",
        )
    )
    if budget_extension_source is not None:
        if args.stage not in {"inspect", "train"}:
            raise CascadeSpecError(
                "Budget extension is available only for inspect or train stage."
            )
        if args.mode != "development" or len(seeds) != 1:
            raise CascadeSpecError(
                "Budget extension requires development mode and exactly one seed."
            )
    if args.stage == "inspect":
        source_training_frame = load_configured_joint_inflows(physical, "train")
        inspection_learning_frame, inspection_validation_frame = split_training_inflows(
            source_training_frame,
            mode=args.mode,
            development_years=int(
                _mapping(config, "validation")["development_years"]
            ),
            validation_years=int(
                _mapping(config, "validation")["validation_years"]
            ),
        )
        inspection = inspect_dqn_configuration(
            reservoir_1,
            reservoir_2,
            config,
            mode=args.mode,
            seed=seeds[0],
        )
        inspection.update(
            {
                "configuration": str(config_path),
                "configuration_file_sha256": _sha256(config_path),
                "physical_configuration": str(physical_config),
                "physical_configuration_file_sha256": physical_config_hash,
                "output_root": str(output_root),
                "monthly_max_target_levels_m": monthly_max_target_levels,
                "action_mask_mode": ACTION_MASK_MODE,
                "source_training_months": len(source_training_frame),
                "learning_months": len(inspection_learning_frame),
                "internal_validation_months": (
                    0
                    if inspection_validation_frame is None
                    else len(inspection_validation_frame)
                ),
                "uses_full_400_year_training_sequence": bool(
                    args.mode == "final"
                    and len(source_training_frame) == 4800
                    and len(inspection_learning_frame) == 4800
                    and inspection_validation_frame is None
                ),
            }
        )
        if budget_extension_source is not None:
            inspection["budget_extension_preflight"] = (
                inspect_dqn_budget_extension(
                    reservoir_1,
                    reservoir_2,
                    source_training_frame,
                    config,
                    mode=args.mode,
                    seed=seeds[0],
                    source_checkpoint=budget_extension_source,
                    physical_configuration_sha256=physical_config_hash,
                    monthly_max_target_levels_m=monthly_max_target_levels,
                )
            )
        if "evaluation_scenarios" in config:
            scenario_records: dict[str, Any] = {}
            scenario_hashes: set[str] = set()
            configured_scenarios = _mapping(config, "evaluation_scenarios")
            if "baseline" not in configured_scenarios:
                raise CascadeSpecError(
                    "Configured DQN evaluation scenarios must include baseline."
                )
            unsupported_scenarios = set(configured_scenarios) - {
                "baseline", "drought", "wet"
            }
            if unsupported_scenarios:
                raise CascadeSpecError(
                    "Unsupported DQN evaluation scenarios: "
                    f"{sorted(unsupported_scenarios)}."
                )
            for scenario in configured_scenarios:
                scenario_path, scenario_role = _evaluation_scenario(
                    config, scenario
                )
                scenario_frame = load_synthetic_joint_inflows(
                    scenario_path,
                    expected_months=int(
                        physical["data"]["expected_months_per_split"]
                    ),
                )
                scenario_hash = normalized_inflow_sha256(scenario_frame)
                if scenario_hash == normalized_inflow_sha256(
                    source_training_frame
                ):
                    raise CascadeSpecError(
                        f"Evaluation scenario {scenario} duplicates training inflow."
                    )
                scenario_hashes.add(scenario_hash)
                scenario_records[scenario] = {
                    "role": scenario_role,
                    "inflow_file": str(scenario_path),
                    "months": len(scenario_frame),
                    "years": len(scenario_frame) // 12,
                    "inflow_sha256": scenario_hash,
                    "used_for_training_or_model_selection": False,
                    "used_for_penalty_calibration": bool(
                        scenario_role
                        == "penalty_calibration_baseline_evaluation"
                    ),
                }
            if len(scenario_hashes) != len(scenario_records):
                raise CascadeSpecError(
                    "Configured DQN evaluation inputs must be distinct."
                )
            inspection["evaluation_scenarios"] = scenario_records
        print(
            json.dumps(
                inspection,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
        )
        return
    training_frame = load_configured_joint_inflows(physical, "train")

    if args.stage == "inspect-checkpoint":
        if args.mode != "final":
            raise CascadeSpecError(
                "Checkpoint inspection for the selected model requires final mode."
            )
        for seed in seeds:
            checkpoint_path = _selected_checkpoint_path(
                output_root, config, mode="final", seed=seed
            )
            _, normalizer, action_grid, checkpoint = _load_selected_policy(
                checkpoint_path,
                config=config,
                mode="final",
                seed=seed,
                reservoir_1=reservoir_1,
                reservoir_2=reservoir_2,
                training_frame=training_frame,
                physical_configuration_sha256=physical_config_hash,
            )
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_role": checkpoint["checkpoint_role"],
                        "mode": checkpoint["mode"],
                        "experiment_id": checkpoint["experiment_id"],
                        "seed": checkpoint["seed"],
                        "completed_episode": checkpoint["completed_episode"],
                        "environment_steps": checkpoint["environment_steps"],
                        "learning_steps": checkpoint["learning_steps"],
                        "training_budget_episodes": checkpoint[
                            "training_budget_episodes"
                        ],
                        "configuration_sha256": checkpoint[
                            "configuration_sha256"
                        ],
                        "training_inflow_sha256": checkpoint[
                            "training_inflow_sha256"
                        ],
                        "source_training_inflow_sha256": checkpoint[
                            "source_training_inflow_sha256"
                        ],
                        "normalization": normalizer.state_dict(),
                        "action_grid": action_grid.state_dict(),
                        "selected_model": "episode_3000_final_checkpoint",
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
            )
        return

    if args.stage == "validate" and args.mode != "development":
        raise CascadeSpecError("Standalone validation is available only in development mode.")
    if args.stage == "evaluate" and args.mode != "final":
        raise CascadeSpecError(
            "Independent evaluation is available only in final mode."
        )

    if args.stage in {"train", "all"}:
        for seed in seeds:
            configured_paths = build_training_paths(
                output_root,
                mode=args.mode,
                seed=seed,
                output_sections=_mapping(config, "outputs"),
                experiment_id=experiment_id,
            )
            _write_json(
                configured_paths.configuration_snapshot_json,
                {
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "mode": args.mode,
                    "source_config": str(config_path),
                    "source_config_file_sha256": _sha256(config_path),
                    "configuration_sha256": configuration_sha256(config),
                    "configuration": config,
                },
            )
            result = train_dqn_seed(
                reservoir_1,
                reservoir_2,
                training_frame,
                config,
                mode=args.mode,
                seed=seed,
                output_root=output_root,
                resume_latest=args.resume,
                budget_extension_checkpoint=budget_extension_source,
                physical_configuration_sha256=physical_config_hash,
                monthly_max_target_levels_m=monthly_max_target_levels,
                progress_callback=_progress,
            )
            training_summary = result.as_dict()
            training_summary.pop("training_records", None)
            training_summary.pop("validation_records", None)
            training_summary.update(
                {
                    "action_mask_mode": ACTION_MASK_MODE,
                    "experiment_id": experiment_id,
                    "dqn_configuration": str(config_path),
                    "dqn_configuration_file_sha256": _sha256(config_path),
                    "physical_configuration": str(physical_config),
                    "physical_configuration_file_sha256": physical_config_hash,
                    "training_inflow_source": physical["data"]["train_inflow_file"],
                    "hyperparameters": {
                        "episodes": _mapping(config, "training")["episodes"],
                        "replay_buffer_size": _mapping(config, "agent")[
                            "replay_buffer_size"
                        ],
                        "learning_starts": _mapping(config, "agent")[
                            "learning_starts"
                        ],
                        "batch_size": _mapping(config, "agent")["batch_size"],
                        "epsilon_start": _mapping(config, "agent")[
                            "epsilon_start"
                        ],
                        "epsilon_end": _mapping(config, "agent")["epsilon_end"],
                        "epsilon_schedule": _mapping(config, "agent")[
                            "epsilon_schedule"
                        ],
                        "epsilon_decay_steps": _mapping(config, "agent")[
                            "epsilon_decay_steps"
                        ],
                        "learning_rate": _mapping(config, "agent")[
                            "learning_rate"
                        ],
                        "gamma": _mapping(config, "agent")["gamma"],
                        "target_update_interval": _mapping(config, "agent")[
                            "target_update_interval"
                        ],
                        "hidden_dimensions": _mapping(config, "network")[
                            "hidden_dimensions"
                        ],
                        "action_representation": _mapping(config, "network").get(
                            "action_representation", "indexed_output"
                        ),
                        "action_score_chunk_size": _mapping(config, "network").get(
                            "action_score_chunk_size", 256
                        ),
                    },
                }
            )
            _write_json(result.paths.metadata_json, training_summary)
            print(f"Completed DQN {args.mode} training seed={seed}: {result.selected_checkpoint}")

    run_trajectory = bool(
        args.stage == "validate"
        or args.stage == "evaluate"
        or (args.stage == "all" and args.mode == "final")
    )
    if run_trajectory:
        if args.stage == "validate":
            _, trajectory = split_training_inflows(
                training_frame,
                mode="development",
                development_years=int(_mapping(config, "validation")["development_years"]),
                validation_years=int(_mapping(config, "validation")["validation_years"]),
            )
            if trajectory is None:
                raise CascadeSpecError("Development validation frame is missing.")
            trajectory_label = "validation"
        else:
            scenario_path, scenario_role = _evaluation_scenario(
                config, args.evaluation_scenario
            )
            trajectory = load_synthetic_joint_inflows(
                scenario_path,
                expected_months=int(physical["data"]["expected_months_per_split"]),
            )
            if normalized_inflow_sha256(trajectory) == normalized_inflow_sha256(
                training_frame
            ):
                raise CascadeSpecError(
                    "DQN training and independent evaluation inflow sequences are identical."
                )
            trajectory_label = "evaluation"

        for seed in seeds:
            checkpoint_path = _selected_checkpoint_path(
                output_root, config, mode=args.mode, seed=seed
            )
            agent, normalizer, action_grid, checkpoint = _load_selected_policy(
                checkpoint_path,
                mode=args.mode,
                seed=seed,
                config=config,
                reservoir_1=reservoir_1,
                reservoir_2=reservoir_2,
                training_frame=training_frame,
                physical_configuration_sha256=physical_config_hash,
            )
            reference_days, reward_clip, failure_penalty = _evaluation_settings(
                checkpoint,
                reservoir_1=reservoir_1,
                reservoir_2=reservoir_2,
            )
            schedule, summary = evaluate_greedy_trajectory(
                reservoir_1,
                reservoir_2,
                agent,
                normalizer,
                action_grid,
                trajectory,
                _mapping(config, "problem")["initial_levels_m"],
                training_inflow_sha256=checkpoint["training_inflow_sha256"],
                seed=seed,
                reward_reference_days=reference_days,
                reward_clip=reward_clip,
                firm_power_failure_penalty_gwh=failure_penalty,
                monthly_max_target_levels_m=monthly_max_target_levels,
                trajectory_kind=trajectory_label,
                collect_rows=True,
            )
            if schedule is None:
                raise CascadeSpecError("DQN trajectory did not return its monthly records.")
            summary.update(
                {
                    "mode": args.mode,
                    "experiment_id": experiment_id,
                    "trajectory_kind": trajectory_label,
                    "trajectory_inflow_sha256": normalized_inflow_sha256(
                        trajectory
                    ),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_role": checkpoint["checkpoint_role"],
                    "checkpoint_episode": checkpoint["completed_episode"],
                    "dqn_configuration": str(config_path),
                    "dqn_configuration_file_sha256": _sha256(config_path),
                    "physical_configuration": str(physical_config),
                    "physical_configuration_file_sha256": physical_config_hash,
                    "training_inflow_source": physical["data"]["train_inflow_file"],
                    "trajectory_source": (
                        str(scenario_path)
                        if trajectory_label == "evaluation"
                        else "chronological_final_80_years_of_training_split"
                    ),
                }
            )
            yearly_summary: pd.DataFrame | None = None
            scenario_metrics: dict[str, Any] | None = None
            if trajectory_label == "evaluation":
                yearly_summary, scenario_metrics = summarize_monthly_physical_trajectory(
                    schedule,
                    reservoir_1,
                    reservoir_2,
                )
                if len(schedule) != 4800 or len(yearly_summary) != 400:
                    raise CascadeSpecError(
                        "Final DQN evaluation must produce 4800 monthly and 400 yearly rows."
                    )
                summary.update(scenario_metrics)
                summary.update(
                    {
                        "evaluation_scenario": args.evaluation_scenario,
                        "evaluation_scenario_role": scenario_role,
                        "evaluation_inflow_file": str(scenario_path),
                        "evaluation_inflow_sha256": normalized_inflow_sha256(
                            trajectory
                        ),
                        "normalization_source": "frozen_final_training_checkpoint",
                        "normalization_updated_during_evaluation": False,
                        "evaluation_data_used_for_model_selection": False,
                        "monthly_trajectory_rows": 4800,
                        "yearly_summary_rows": 400,
                    }
                )
            output_section = str(
                _mapping(config, "outputs").get(
                    trajectory_label, trajectory_label
                )
            )
            if experiment_id is None:
                csv_path = (
                    output_root
                    / output_section
                    / args.mode
                    / f"seed_{seed}_{trajectory_label}_monthly.csv"
                )
            else:
                run_paths = build_training_paths(
                    output_root,
                    mode=args.mode,
                    seed=seed,
                    output_sections=_mapping(config, "outputs"),
                    experiment_id=experiment_id,
                )
            if trajectory_label == "evaluation":
                scenario_directory = (
                    run_paths.run_root
                    / output_section
                    / args.evaluation_scenario
                )
                scenario_directory.mkdir(parents=True, exist_ok=True)
                monthly_path = scenario_directory / "monthly_trajectory.csv"
                yearly_path = scenario_directory / "yearly_summary.csv"
                summary_path = scenario_directory / "summary.json"
                schedule.to_csv(monthly_path, index=False)
                if yearly_summary is None:
                    raise CascadeSpecError("DQN yearly evaluation summary is missing.")
                yearly_summary.to_csv(yearly_path, index=False)
                _write_json(summary_path, summary)
                print(
                    f"Completed DQN {args.evaluation_scenario} evaluation "
                    f"seed={seed}: months=4800 years=400"
                )
                print(f"Wrote monthly trajectory: {monthly_path}")
                print(f"Wrote yearly summary: {yearly_path}")
                print(f"Wrote evaluation summary: {summary_path}")
            else:
                csv_path = (
                    run_paths.run_root
                    / output_section
                    / f"{trajectory_label}_monthly.csv"
                )
                _write_trajectory_outputs(schedule, summary, csv_path=csv_path)
                print(f"Wrote DQN {trajectory_label} seed={seed}: {csv_path}")


if __name__ == "__main__":
    main()
