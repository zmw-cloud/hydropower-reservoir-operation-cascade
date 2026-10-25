"""Development, final fitting, and continuous evaluation for cascade ISO."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from .data import load_synthetic_joint_inflows
from .environment import CascadeReservoirEnv, ReservoirSpec, _as_month_period_index, load_specs_from_yaml
from .iso_models import FEATURE_NAMES, TARGET_NAMES, MonthlyCascadeISOPolicy


CANONICAL_TEACHER_COLUMNS = ("date",) + FEATURE_NAMES + TARGET_NAMES


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_iso_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("ISO configuration must be a YAML mapping.")
    return config, config_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher_dataset(
    teacher_path: str | Path,
    column_mapping: Mapping[str, str],
    *,
    expected_years: int,
    required_solver_variant: str,
    required_objective: str,
    required_failure_penalty_gwh: float,
    required_terminal_constraint: str,
    required_initial_levels_m: tuple[float, float],
    required_terminal_levels_m: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the documented continuous-storage DDP schedule fields."""
    teacher_path = Path(teacher_path)
    summary_path = teacher_path.with_suffix(".json")
    if not summary_path.exists():
        raise FileNotFoundError(
            f"The DDP teacher summary is required to verify its solver variant: {summary_path}"
        )
    teacher_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_variant = teacher_summary.get("solver_variant")
    if actual_variant != required_solver_variant:
        raise ValueError(
            "ISO teacher must come from the corrected continuous-storage DDP; "
            f"required solver_variant={required_solver_variant!r}, got {actual_variant!r}."
        )
    reward = teacher_summary.get("reward_configuration", {})
    if reward.get("objective") != required_objective:
        raise ValueError(
            "DDP teacher objective is incompatible with ISO: "
            f"required {required_objective!r}, got {reward.get('objective')!r}."
        )
    actual_penalty = float(reward.get("firm_power_failure_penalty_gwh", np.nan))
    if not np.isclose(actual_penalty, required_failure_penalty_gwh, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "DDP teacher failure penalty is incompatible with ISO: "
            f"required {required_failure_penalty_gwh}, got {actual_penalty}."
        )
    if teacher_summary.get("terminal_constraint") != required_terminal_constraint:
        raise ValueError(
            "DDP teacher terminal constraint is incompatible with ISO: "
            f"required {required_terminal_constraint!r}, got "
            f"{teacher_summary.get('terminal_constraint')!r}."
        )
    actual_initial = np.asarray(teacher_summary.get("initial_levels_m"), dtype=float)
    actual_terminal = np.asarray(teacher_summary.get("terminal_levels_m"), dtype=float)
    if not np.allclose(actual_initial, required_initial_levels_m, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"DDP teacher initial levels are {actual_initial.tolist()}, not "
            f"{list(required_initial_levels_m)}."
        )
    if not np.allclose(actual_terminal, required_terminal_levels_m, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"DDP teacher terminal levels are {actual_terminal.tolist()}, not "
            f"{list(required_terminal_levels_m)}."
        )
    raw = pd.read_csv(teacher_path)
    missing_mapping = set(CANONICAL_TEACHER_COLUMNS) - set(column_mapping)
    if missing_mapping:
        raise ValueError(f"teacher_columns is missing canonical fields: {sorted(missing_mapping)}")
    source_columns = [column_mapping[name] for name in CANONICAL_TEACHER_COLUMNS]
    missing_source = set(source_columns) - set(raw.columns)
    if missing_source:
        raise ValueError(f"DDP teacher schedule is missing mapped fields: {sorted(missing_source)}")
    if len(set(source_columns)) != len(source_columns):
        raise ValueError("teacher_columns must map each canonical field to a distinct source field.")

    frame = raw.loc[:, source_columns].rename(
        columns={source: canonical for canonical, source in column_mapping.items()}
    )
    periods = _as_month_period_index(frame["date"])
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    if periods.duplicated().any() or not periods.equals(expected):
        raise ValueError("DDP teacher dates must be unique, increasing, and monthly continuous.")
    for name in FEATURE_NAMES + TARGET_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not np.all(
        np.isfinite(frame.loc[:, list(FEATURE_NAMES + TARGET_NAMES)].to_numpy(dtype=float))
    ):
        raise ValueError("DDP teacher features and target actions must all be finite numbers.")

    frame["year"] = periods.year
    frame["month"] = periods.month
    years = list(dict.fromkeys(int(year) for year in periods.year))
    if len(years) != expected_years:
        raise ValueError(f"Expected {expected_years} complete teacher years, got {len(years)}.")
    for year in years:
        months = frame.loc[frame["year"] == year, "month"].tolist()
        if months != list(range(1, 13)):
            raise ValueError(f"Teacher year {year} is not one complete January-December year.")
    frame["date"] = [f"{period.year:04d}-{period.month:02d}-01" for period in periods]
    return frame, teacher_summary


def split_development_years(
    teacher: pd.DataFrame, fit_years: int, validation_years: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    years = list(dict.fromkeys(int(year) for year in teacher["year"]))
    if len(years) != fit_years + validation_years:
        raise ValueError(
            f"Teacher contains {len(years)} years; development requires "
            f"{fit_years} fit + {validation_years} validation years."
        )
    fit_labels = years[:fit_years]
    validation_labels = years[fit_years:]
    fit = teacher.loc[teacher["year"].isin(fit_labels)].copy()
    validation = teacher.loc[teacher["year"].isin(validation_labels)].copy()
    metadata = {
        "split_method": "ordered_complete_calendar_years",
        "fit_year_count": len(fit_labels),
        "validation_year_count": len(validation_labels),
        "fit_first_year": fit_labels[0],
        "fit_last_year": fit_labels[-1],
        "validation_first_year": validation_labels[0],
        "validation_last_year": validation_labels[-1],
        "fit_months": len(fit),
        "validation_months": len(validation),
    }
    return fit, validation, metadata


def teacher_forced_metrics(
    policy: MonthlyCascadeISOPolicy, validation: pd.DataFrame
) -> dict[str, float]:
    predictions = np.empty((len(validation), 2), dtype=float)
    for month in range(1, 13):
        mask = validation["month"].to_numpy() == month
        x = validation.loc[mask, list(FEATURE_NAMES)].to_numpy(dtype=float)
        predictions[mask] = policy.month_models[month].predict(x)
    truth = validation.loc[:, list(TARGET_NAMES)].to_numpy(dtype=float)
    errors = predictions - truth
    return {
        "validation_xiaowan_target_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "validation_nuozhadu_target_rmse": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "validation_xiaowan_target_mae": float(np.mean(np.abs(errors[:, 0]))),
        "validation_nuozhadu_target_mae": float(np.mean(np.abs(errors[:, 1]))),
    }


def _transition_columns(row: dict[str, Any], prefix: str, transition: Mapping[str, Any]) -> None:
    for key, value in transition.items():
        row[f"{prefix}_{key}"] = value


def simulate_continuous_policy(
    policy: MonthlyCascadeISOPolicy,
    inflows: pd.DataFrame,
    reservoir_1: ReservoirSpec,
    reservoir_2: ReservoirSpec,
    *,
    initial_levels_m: tuple[float, float],
    terminal_levels_m: tuple[float, float],
    firm_power_failure_penalty_gwh: float,
    monthly_max_target_levels_m: np.ndarray,
    target_unreached_tolerance_m: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one unbroken trajectory exclusively through CascadeReservoirEnv."""
    periods = _as_month_period_index(inflows["date"])
    local_inflows = inflows[["xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s"]].to_numpy(
        dtype=float
    )
    environment = CascadeReservoirEnv(
        reservoir_1,
        reservoir_2,
        local_inflows,
        timestamps=periods,
        initial_levels_m=initial_levels_m,
        firm_power_failure_penalty_gwh=firm_power_failure_penalty_gwh,
        monthly_max_target_levels_m=monthly_max_target_levels_m,
    )
    environment.reset(options={"start_index": 0, "initial_levels_m": initial_levels_m})

    rows: list[dict[str, Any]] = []
    previous_end_storage: np.ndarray | None = None
    out_by_feature = np.zeros(len(FEATURE_NAMES), dtype=int)
    any_out_count = 0
    raw_clip_by_reservoir = np.zeros(2, dtype=int)
    unreached_by_reservoir = np.zeros(2, dtype=int)

    for index, period in enumerate(periods):
        start_storage = environment.storages_1e9m3.copy()
        start_levels = np.array(
            [
                reservoir_1.level_from_storage(start_storage[0]),
                reservoir_2.level_from_storage(start_storage[1]),
            ]
        )
        inflow_1, inflow_2_local = local_inflows[index]
        features = np.array([start_levels[0], start_levels[1], inflow_1, inflow_2_local])
        raw_targets, outside = policy.predict(int(period.month), features)
        monthly_max = monthly_max_target_levels_m[int(period.month) - 1]
        requested_targets = np.array(
            [
                np.clip(raw_targets[0], reservoir_1.dead_level_m, monthly_max[0]),
                np.clip(raw_targets[1], reservoir_2.dead_level_m, monthly_max[1]),
            ],
            dtype=float,
        )
        clipped = ~np.isclose(raw_targets, requested_targets, rtol=0.0, atol=1.0e-12)
        out_by_feature += outside.astype(int)
        any_out_count += int(np.any(outside))
        raw_clip_by_reservoir += clipped.astype(int)

        _, penalized_reward, terminated, truncated, info = environment.step(requested_targets)
        if truncated:
            raise RuntimeError("CascadeReservoirEnv unexpectedly truncated the ISO trajectory.")
        end_storage = environment.storages_1e9m3.copy()
        transition_1 = info["reservoir_1"]
        transition_2 = info["reservoir_2"]
        actual_end_levels = np.array(
            [transition_1["end_level_m"], transition_2["end_level_m"]], dtype=float
        )
        unreached = ~np.isclose(
            requested_targets,
            actual_end_levels,
            rtol=0.0,
            atol=target_unreached_tolerance_m,
        )
        unreached_by_reservoir += unreached.astype(int)
        storage_jump = (
            np.array([np.nan, np.nan])
            if previous_end_storage is None
            else (start_storage - previous_end_storage) * 1.0e9
        )
        row: dict[str, Any] = {
            "date": f"{period.year:04d}-{period.month:02d}-01",
            "start_level_1_m": start_levels[0],
            "start_level_2_m": start_levels[1],
            "start_storage_1_1e9m3": start_storage[0],
            "start_storage_2_1e9m3": start_storage[1],
            "inflow_1_m3s": inflow_1,
            "inflow_2_local_m3s": inflow_2_local,
            "monthly_max_target_level_1_m": monthly_max[0],
            "monthly_max_target_level_2_m": monthly_max[1],
            "raw_target_level_1_m": raw_targets[0],
            "raw_target_level_2_m": raw_targets[1],
            "requested_target_level_1_m": requested_targets[0],
            "requested_target_level_2_m": requested_targets[1],
            "actual_end_level_1_m": actual_end_levels[0],
            "actual_end_level_2_m": actual_end_levels[1],
            "end_storage_1_1e9m3": end_storage[0],
            "end_storage_2_1e9m3": end_storage[1],
            "raw_target_clipped_1": bool(clipped[0]),
            "raw_target_clipped_2": bool(clipped[1]),
            "target_unreached_1": bool(unreached[0]),
            "target_unreached_2": bool(unreached[1]),
            "out_of_training_domain": bool(np.any(outside)),
            "storage_jump_1_m3": storage_jump[0],
            "storage_jump_2_m3": storage_jump[1],
            "generation_energy_gwh": info["generation_energy_gwh"],
            "xiaowan_firm_power_failure": info["xiaowan_firm_power_failure"],
            "nuozhadu_firm_power_failure": info["nuozhadu_firm_power_failure"],
            "failed_station_months": info["failed_station_months"],
            "failure_penalty_gwh": info["failure_penalty_gwh"],
            "penalized_reward_gwh": penalized_reward,
        }
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            row[f"out_of_domain_{feature_name}"] = bool(outside[feature_index])
        _transition_columns(row, "r1", transition_1)
        _transition_columns(row, "r2", transition_2)
        rows.append(row)
        previous_end_storage = end_storage.copy()
        if terminated != (index == len(periods) - 1):
            raise RuntimeError("CascadeReservoirEnv termination did not match the inflow horizon.")

    schedule = pd.DataFrame(rows)
    month_count = len(schedule)
    target_clip_total = int(raw_clip_by_reservoir.sum())
    target_unreached_total = int(unreached_by_reservoir.sum())
    finite_jumps = schedule[["storage_jump_1_m3", "storage_jump_2_m3"]].iloc[1:]
    final_levels = np.array(
        [schedule.iloc[-1]["actual_end_level_1_m"], schedule.iloc[-1]["actual_end_level_2_m"]],
        dtype=float,
    )
    terminal_errors = final_levels - np.asarray(terminal_levels_m, dtype=float)
    summary: dict[str, Any] = {
        "months": month_count,
        "cascade_generation_gwh": float(
            schedule["r1_energy_gwh"].sum() + schedule["r2_energy_gwh"].sum()
        ),
        "xiaowan_generation_gwh": float(schedule["r1_energy_gwh"].sum()),
        "nuozhadu_generation_gwh": float(schedule["r2_energy_gwh"].sum()),
        "penalized_objective_gwh": float(schedule["penalized_reward_gwh"].sum()),
        "failure_penalty_gwh": float(firm_power_failure_penalty_gwh),
        "total_failure_penalty_gwh": float(schedule["failure_penalty_gwh"].sum()),
        "failed_station_months": int(schedule["failed_station_months"].sum()),
        "xiaowan_firm_power_reliability": float(
            1.0 - schedule["xiaowan_firm_power_failure"].mean()
        ),
        "nuozhadu_firm_power_reliability": float(
            1.0 - schedule["nuozhadu_firm_power_failure"].mean()
        ),
        "min_station_firm_power_reliability": float(
            min(
                1.0 - schedule["xiaowan_firm_power_failure"].mean(),
                1.0 - schedule["nuozhadu_firm_power_failure"].mean(),
            )
        ),
        "joint_firm_power_reliability": float(
            (~(
                schedule["xiaowan_firm_power_failure"]
                | schedule["nuozhadu_firm_power_failure"]
            )).mean()
        ),
        "target_unreached_count": target_unreached_total,
        "target_unreached_rate": target_unreached_total / (2 * month_count),
        "xiaowan_target_unreached_count": int(unreached_by_reservoir[0]),
        "nuozhadu_target_unreached_count": int(unreached_by_reservoir[1]),
        "target_clip_count": target_clip_total,
        "target_clip_rate": target_clip_total / (2 * month_count),
        "xiaowan_target_clip_count": int(raw_clip_by_reservoir[0]),
        "nuozhadu_target_clip_count": int(raw_clip_by_reservoir[1]),
        "out_of_domain_count": int(any_out_count),
        "out_of_domain_rate": any_out_count / month_count,
        "prediction_anomaly_count": 0,
        "xiaowan_level_min_m": float(
            min(schedule["start_level_1_m"].min(), schedule["actual_end_level_1_m"].min())
        ),
        "xiaowan_level_max_m": float(
            max(schedule["start_level_1_m"].max(), schedule["actual_end_level_1_m"].max())
        ),
        "nuozhadu_level_min_m": float(
            min(schedule["start_level_2_m"].min(), schedule["actual_end_level_2_m"].min())
        ),
        "nuozhadu_level_max_m": float(
            max(schedule["start_level_2_m"].max(), schedule["actual_end_level_2_m"].max())
        ),
        "max_abs_water_balance_error_1_m3": float(schedule["r1_water_balance_error_m3"].abs().max()),
        "max_abs_water_balance_error_2_m3": float(schedule["r2_water_balance_error_m3"].abs().max()),
        "max_abs_cascade_water_balance_error_m3": float(
            (schedule["r1_water_balance_error_m3"] + schedule["r2_water_balance_error_m3"])
            .abs()
            .max()
        ),
        "sum_abs_water_balance_error_1_m3": float(
            schedule["r1_water_balance_error_m3"].abs().sum()
        ),
        "sum_abs_water_balance_error_2_m3": float(
            schedule["r2_water_balance_error_m3"].abs().sum()
        ),
        "max_abs_storage_jump_1_m3": float(finite_jumps["storage_jump_1_m3"].abs().max()),
        "max_abs_storage_jump_2_m3": float(finite_jumps["storage_jump_2_m3"].abs().max()),
        "initial_levels_m": [float(initial_levels_m[0]), float(initial_levels_m[1])],
        "terminal_reference_levels_m": [
            float(terminal_levels_m[0]),
            float(terminal_levels_m[1]),
        ],
        "final_actual_levels_m": final_levels.tolist(),
        "terminal_level_errors_m": terminal_errors.tolist(),
        "terminal_constraint_enforced_on_policy": False,
        "monthly_max_target_levels_m": monthly_max_target_levels_m.tolist(),
        "flood_limit_exceedance_count": int(
            (
                (schedule["actual_end_level_1_m"] > schedule["monthly_max_target_level_1_m"] + 1.0e-8)
                | (schedule["actual_end_level_2_m"] > schedule["monthly_max_target_level_2_m"] + 1.0e-8)
            ).sum()
        ),
        "continuous_storage_carryover": True,
        "physical_simulator": "cascade_reservoir_benchmark.environment.CascadeReservoirEnv",
    }
    for index, name in enumerate(FEATURE_NAMES):
        summary[f"out_of_domain_{name}_count"] = int(out_by_feature[index])
        summary[f"out_of_domain_{name}_rate"] = float(out_by_feature[index] / month_count)
    return schedule, summary


def _continuous_segment_metrics(
    schedule: pd.DataFrame,
    *,
    segment_years: int,
) -> dict[str, float | int]:
    """Summarize contiguous validation segments without resetting reservoir state."""
    segment_months = int(segment_years) * 12
    if segment_months <= 0 or len(schedule) % segment_months != 0:
        raise ValueError(
            "Validation months must divide exactly into the configured segment length."
        )
    metrics: dict[str, float | int] = {}
    segment_minimum_reliabilities: list[float] = []
    for index, start in enumerate(range(0, len(schedule), segment_months), start=1):
        segment = schedule.iloc[start : start + segment_months]
        xiaowan_reliability = float(
            1.0 - segment["xiaowan_firm_power_failure"].mean()
        )
        nuozhadu_reliability = float(
            1.0 - segment["nuozhadu_firm_power_failure"].mean()
        )
        joint_reliability = float(
            (~(
                segment["xiaowan_firm_power_failure"]
                | segment["nuozhadu_firm_power_failure"]
            )).mean()
        )
        minimum_reliability = min(xiaowan_reliability, nuozhadu_reliability)
        segment_minimum_reliabilities.append(minimum_reliability)
        prefix = f"segment_{index:02d}_{segment_years}y"
        metrics[f"{prefix}_xiaowan_reliability"] = xiaowan_reliability
        metrics[f"{prefix}_nuozhadu_reliability"] = nuozhadu_reliability
        metrics[f"{prefix}_joint_reliability"] = joint_reliability
        metrics[f"{prefix}_min_station_reliability"] = minimum_reliability
        metrics[f"{prefix}_mean_annual_generation_gwh"] = float(
            segment["generation_energy_gwh"].sum() / segment_years
        )
        metrics[f"{prefix}_failed_station_months"] = int(
            segment["failed_station_months"].sum()
        )
    metrics["worst_segment_min_station_reliability"] = float(
        min(segment_minimum_reliabilities)
    )
    return metrics


def _candidate_definitions(config: Mapping[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    development = config["development"]
    enabled = set(development["candidate_model_families"])
    if "polynomial" in enabled:
        for index, alpha in enumerate(development["polynomial_alpha_candidates"], start=1):
            candidates.append(
                (f"polynomial_{index:02d}", "polynomial", {"degree": 3, "alpha": float(alpha)})
            )
    if "rbf" in enabled:
        for index, settings in enumerate(development["rbf_candidates"], start=1):
            candidates.append((f"rbf_{index:02d}", "rbf", dict(settings)))
    if "random_forest" in enabled:
        generator = development.get("random_forest_candidate_generator", {})
        if generator.get("enabled", False):
            candidates.extend(_generated_random_forest_candidates(generator))
        else:
            for index, settings in enumerate(
                development["random_forest_candidates"], start=1
            ):
                candidates.append(
                    (f"random_forest_{index:02d}", "random_forest", dict(settings))
                )
    if not candidates:
        raise ValueError("ISO configuration does not define any enabled candidates.")
    unknown = enabled - {"polynomial", "rbf", "random_forest"}
    if unknown:
        raise ValueError(f"Unsupported ISO candidate families: {sorted(unknown)}")
    return candidates


def _generated_random_forest_candidates(
    settings: Mapping[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Build a deterministic, level-balanced subset of the RF product space."""
    count = int(settings["candidate_count"])
    n_estimators = int(settings["n_estimators"])
    seed = int(settings["sampling_seed"])
    depths = list(settings["max_depth_values"])
    leaves = [int(value) for value in settings["min_samples_leaf_values"]]
    features = list(settings["max_features_values"])
    if count <= 0 or n_estimators <= 0:
        raise ValueError("Generated RF candidate_count and n_estimators must be positive.")
    if not depths or not leaves or not features:
        raise ValueError("Generated RF factor levels must be nonempty.")

    combinations = [
        {
            "n_estimators": n_estimators,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": feature,
        }
        for depth, leaf, feature in product(depths, leaves, features)
    ]
    if count > len(combinations):
        raise ValueError(
            f"Requested {count} RF candidates from only {len(combinations)} combinations."
        )

    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(combinations)).tolist()
    remaining = [combinations[index] for index in tie_order]
    selected: list[dict[str, Any]] = []
    depth_counts = {str(value): 0 for value in depths}
    leaf_counts = {value: 0 for value in leaves}
    feature_counts = {str(value): 0 for value in features}

    while len(selected) < count:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                depth_counts[str(remaining[index]["max_depth"])]
                + leaf_counts[int(remaining[index]["min_samples_leaf"])]
                + feature_counts[str(remaining[index]["max_features"])],
                index,
            ),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        depth_counts[str(chosen["max_depth"])] += 1
        leaf_counts[int(chosen["min_samples_leaf"])] += 1
        feature_counts[str(chosen["max_features"])] += 1

    return [
        (f"random_forest_coarse_{index:02d}", "random_forest", parameters)
        for index, parameters in enumerate(selected, start=1)
    ]


def _load_physics(config: Mapping[str, Any], root: Path):
    physics_path = _resolve(root, config["physics_config_path"])
    reservoir_1, reservoir_2, physics_config = load_specs_from_yaml(physics_path)
    return physics_path, reservoir_1, reservoir_2, physics_config


def _teacher_requirements(config: Mapping[str, Any]) -> dict[str, Any]:
    teacher = config["teacher"]
    return {
        "required_solver_variant": str(teacher["required_solver_variant"]),
        "required_objective": str(teacher["required_objective"]),
        "required_failure_penalty_gwh": float(teacher["required_failure_penalty_gwh"]),
        "required_terminal_constraint": str(teacher["required_terminal_constraint"]),
        "required_initial_levels_m": tuple(
            float(value) for value in teacher["required_initial_levels_m"]
        ),
        "required_terminal_levels_m": tuple(
            float(value) for value in teacher["required_terminal_levels_m"]
        ),
    }


def _experiment_contract(
    config: Mapping[str, Any],
    physics_config: Mapping[str, Any],
    teacher_summary: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float]:
    constraints = physics_config["ddp_constraints"]
    monthly = constraints["monthly_max_target_levels_m"]
    monthly_max = np.column_stack(
        (
            np.asarray(monthly["xiaowan"], dtype=float),
            np.asarray(monthly["nuozhadu"], dtype=float),
        )
    )
    if monthly_max.shape != (12, 2):
        raise ValueError("Physics configuration must define twelve monthly target limits.")
    teacher_monthly = teacher_summary.get("monthly_max_target_levels_m", {})
    teacher_monthly_array = np.column_stack(
        (
            np.asarray(teacher_monthly.get("xiaowan"), dtype=float),
            np.asarray(teacher_monthly.get("nuozhadu"), dtype=float),
        )
    )
    if teacher_monthly_array.shape != (12, 2) or not np.allclose(
        teacher_monthly_array, monthly_max, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("DDP teacher monthly limits differ from the active physics config.")

    teacher_physics = teacher_summary.get("physical_parameters", {})
    for station in ("xiaowan", "nuozhadu"):
        configured = physics_config["reservoirs"][station]
        recorded = teacher_physics.get(station, {})
        for key in (
            "normal_level_m",
            "flood_limit_level_m",
            "max_turbine_discharge_m3s",
            "max_power_kw",
            "firm_power_kw",
            "max_total_release_m3s",
        ):
            configured_value = configured.get(key)
            recorded_value = recorded.get(key)
            if configured_value is None or recorded_value is None:
                if configured_value is not None or recorded_value is not None:
                    raise ValueError(
                        f"DDP teacher {station}.{key} differs from the physics config."
                    )
            elif not np.isclose(
                float(configured_value),
                float(recorded_value),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"DDP teacher {station}.{key}={recorded_value} differs from "
                    f"the physics config value {configured_value}."
                )

    initial_levels = tuple(float(value) for value in constraints["initial_levels_m"])
    terminal_levels = tuple(float(value) for value in constraints["terminal_levels_m"])
    requirements = _teacher_requirements(config)
    if initial_levels != requirements["required_initial_levels_m"]:
        raise ValueError("ISO and physics initial levels are inconsistent.")
    if terminal_levels != requirements["required_terminal_levels_m"]:
        raise ValueError("ISO and physics terminal levels are inconsistent.")
    penalty = float(
        teacher_summary["reward_configuration"]["firm_power_failure_penalty_gwh"]
    )
    if config["objective"]["selection_metric"] not in {
        "validation_penalized_objective_gwh",
        "reliability_target_then_generation",
    }:
        raise ValueError("Unsupported ISO development selection metric.")
    return monthly_max, initial_levels, terminal_levels, penalty


def run_development(config_path: str | Path) -> tuple[Path, Path]:
    config, _ = load_iso_config(config_path)
    root = _project_root()
    teacher_path = _resolve(root, config["teacher"]["ddp_schedule_path"])
    expected_years = int(config["development"]["fit_years"]) + int(
        config["development"]["validation_years"]
    )
    teacher, teacher_summary = load_teacher_dataset(
        teacher_path,
        config["teacher"]["columns"],
        expected_years=expected_years,
        **_teacher_requirements(config),
    )
    fit, validation, split_metadata = split_development_years(
        teacher,
        int(config["development"]["fit_years"]),
        int(config["development"]["validation_years"]),
    )
    physics_path, reservoir_1, reservoir_2, physics_config = _load_physics(config, root)
    monthly_max, _, terminal_levels, failure_penalty = _experiment_contract(
        config, physics_config, teacher_summary
    )
    validation_inflows = validation.rename(
        columns={
            "inflow_1_m3s": "xiaowan_inflow_m3s",
            "inflow_2_local_m3s": "nuozhadu_local_inflow_m3s",
        }
    )[["date", "xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s"]]
    initial_levels = (
        float(validation.iloc[0]["start_level_1_m"]),
        float(validation.iloc[0]["start_level_2_m"]),
    )
    output_dir = _resolve(root, config["output_directory"]) / "development"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["random_seed"])
    tolerance = float(config["target_unreached_tolerance_m"])

    candidates = _candidate_definitions(config)
    results_path = output_dir / "iso_development_results.csv"
    contract_path = output_dir / "iso_development_contract.json"
    contract = {
        "teacher_sha256": _sha256(teacher_path),
        "teacher_summary_sha256": _sha256(teacher_path.with_suffix(".json")),
        "physics_config_sha256": _sha256(physics_path),
        "random_seed": seed,
        "candidate_definitions": [
            {
                "candidate_id": candidate_id,
                "model_type": model_type,
                "hyperparameters": hyperparameters,
            }
            for candidate_id, model_type, hyperparameters in candidates
        ],
    }
    resume_candidates = bool(
        config["development"].get("resume_completed_candidates", False)
    )
    if resume_candidates and results_path.exists():
        if not contract_path.exists():
            raise RuntimeError(
                "Cannot resume ISO development without iso_development_contract.json."
            )
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise RuntimeError(
                "Existing ISO development results use a different teacher, physics, "
                "seed, or candidate definition."
            )
        rows = pd.read_csv(results_path).to_dict(orient="records")
    else:
        rows: list[dict[str, Any]] = []
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    completed = {
        (str(row["candidate_id"]), str(row["hyperparameters"]))
        for row in rows
        if row.get("status") == "success"
    }
    for candidate_number, (candidate_id, model_type, hyperparameters) in enumerate(
        candidates, start=1
    ):
        hyperparameters_json = json.dumps(hyperparameters, sort_keys=True)
        if (candidate_id, hyperparameters_json) in completed:
            print(
                f"[{candidate_number}/{len(candidates)}] ISO candidate {candidate_id} "
                "already completed; skipped.",
                flush=True,
            )
            continue
        rows = [row for row in rows if str(row.get("candidate_id")) != candidate_id]
        print(
            f"[{candidate_number}/{len(candidates)}] ISO candidate {candidate_id} started.",
            flush=True,
        )
        started = time.perf_counter()
        base_row = {
            "candidate_id": candidate_id,
            "model_type": model_type,
            "hyperparameters": hyperparameters_json,
        }
        try:
            policy = MonthlyCascadeISOPolicy.fit(
                fit,
                model_type=model_type,
                hyperparameters=hyperparameters,
                random_seed=seed,
            )
            supervised = teacher_forced_metrics(policy, validation)
            closed_loop_schedule, closed_loop = simulate_continuous_policy(
                policy,
                validation_inflows,
                reservoir_1,
                reservoir_2,
                initial_levels_m=initial_levels,
                terminal_levels_m=terminal_levels,
                firm_power_failure_penalty_gwh=failure_penalty,
                monthly_max_target_levels_m=monthly_max,
                target_unreached_tolerance_m=tolerance,
            )
            segment_metrics = _continuous_segment_metrics(
                closed_loop_schedule,
                segment_years=int(
                    config["development"].get("validation_segment_years", 20)
                ),
            )
            row = {
                **base_row,
                "status": "success",
                "error_type": "",
                "error_message": "",
                "validation_penalized_objective_gwh": closed_loop[
                    "penalized_objective_gwh"
                ],
                "validation_cascade_generation_gwh": closed_loop[
                    "cascade_generation_gwh"
                ],
                "validation_failed_station_months": closed_loop[
                    "failed_station_months"
                ],
                "validation_total_failure_penalty_gwh": closed_loop[
                    "total_failure_penalty_gwh"
                ],
                "xiaowan_firm_power_reliability": closed_loop[
                    "xiaowan_firm_power_reliability"
                ],
                "nuozhadu_firm_power_reliability": closed_loop[
                    "nuozhadu_firm_power_reliability"
                ],
                "min_station_firm_power_reliability": closed_loop[
                    "min_station_firm_power_reliability"
                ],
                "joint_firm_power_reliability": closed_loop[
                    "joint_firm_power_reliability"
                ],
                "validation_mean_annual_generation_gwh": closed_loop[
                    "cascade_generation_gwh"
                ]
                / int(config["development"]["validation_years"]),
                **segment_metrics,
                **supervised,
                "target_unreached_count": closed_loop["target_unreached_count"],
                "target_unreached_rate": closed_loop["target_unreached_rate"],
                "target_clip_count": closed_loop["target_clip_count"],
                "target_clip_rate": closed_loop["target_clip_rate"],
                "out_of_domain_count": closed_loop["out_of_domain_count"],
                "out_of_domain_rate": closed_loop["out_of_domain_rate"],
                "flood_limit_exceedance_count": closed_loop[
                    "flood_limit_exceedance_count"
                ],
                "max_abs_water_balance_error_1_m3": closed_loop[
                    "max_abs_water_balance_error_1_m3"
                ],
                "max_abs_water_balance_error_2_m3": closed_loop[
                    "max_abs_water_balance_error_2_m3"
                ],
                "max_abs_cascade_water_balance_error_m3": closed_loop[
                    "max_abs_cascade_water_balance_error_m3"
                ],
                "sum_abs_water_balance_error_1_m3": closed_loop[
                    "sum_abs_water_balance_error_1_m3"
                ],
                "sum_abs_water_balance_error_2_m3": closed_loop[
                    "sum_abs_water_balance_error_2_m3"
                ],
                "max_abs_storage_jump_1_m3": closed_loop[
                    "max_abs_storage_jump_1_m3"
                ],
                "max_abs_storage_jump_2_m3": closed_loop[
                    "max_abs_storage_jump_2_m3"
                ],
                "xiaowan_level_min_m": closed_loop["xiaowan_level_min_m"],
                "xiaowan_level_max_m": closed_loop["xiaowan_level_max_m"],
                "nuozhadu_level_min_m": closed_loop["nuozhadu_level_min_m"],
                "nuozhadu_level_max_m": closed_loop["nuozhadu_level_max_m"],
                "terminal_level_error_1_m": closed_loop["terminal_level_errors_m"][0],
                "terminal_level_error_2_m": closed_loop["terminal_level_errors_m"][1],
                "prediction_anomaly_count": closed_loop["prediction_anomaly_count"],
            }
            for name in FEATURE_NAMES:
                row[f"out_of_domain_{name}_count"] = closed_loop[
                    f"out_of_domain_{name}_count"
                ]
                row[f"out_of_domain_{name}_rate"] = closed_loop[
                    f"out_of_domain_{name}_rate"
                ]
            print(
                f"[{candidate_number}/{len(candidates)}] ISO candidate {candidate_id} "
                f"completed: validation_penalized_objective_gwh="
                f"{closed_loop['penalized_objective_gwh']:.6f}.",
                flush=True,
            )
        except Exception as exc:
            row = {
                **base_row,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc).replace("\n", " "),
            }
            print(
                f"[{candidate_number}/{len(candidates)}] ISO candidate {candidate_id} "
                f"failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        row["wall_clock_seconds"] = time.perf_counter() - started
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)

    results = pd.DataFrame(rows)
    successful = results.loc[results["status"] == "success"].copy()
    if successful.empty:
        raise RuntimeError(f"All ISO candidates failed; inspect {results_path}.")
    missing_families = set(config["development"]["candidate_model_families"]) - set(
        successful["model_type"]
    )
    if missing_families:
        raise RuntimeError(
            f"No successful ISO candidate for families {sorted(missing_families)}; "
            f"inspect {results_path}."
        )
    selection_metric = config["objective"]["selection_metric"]
    reliability_target = float(config["objective"].get("reliability_target", 0.9))
    if selection_metric == "validation_penalized_objective_gwh":
        successful["meets_reliability_target"] = (
            (successful["xiaowan_firm_power_reliability"] >= reliability_target)
            & (successful["nuozhadu_firm_power_reliability"] >= reliability_target)
        )
        successful = successful.sort_values(
            ["validation_penalized_objective_gwh", "candidate_id"],
            ascending=[False, True],
            kind="stable",
        )
        target_feasible = bool(successful["meets_reliability_target"].any())
    elif selection_metric == "reliability_target_then_generation":
        successful["meets_reliability_target"] = (
            (successful["xiaowan_firm_power_reliability"] >= reliability_target)
            & (successful["nuozhadu_firm_power_reliability"] >= reliability_target)
        )
        successful["maximum_station_reliability_shortfall"] = np.maximum(
            np.maximum(
                0.0,
                reliability_target
                - successful["xiaowan_firm_power_reliability"].to_numpy(dtype=float),
            ),
            np.maximum(
                0.0,
                reliability_target
                - successful["nuozhadu_firm_power_reliability"].to_numpy(dtype=float),
            ),
        )
        eligible = successful.loc[successful["meets_reliability_target"]].copy()
        target_feasible = not eligible.empty
        if target_feasible:
            eligible = eligible.sort_values(
                [
                    "validation_mean_annual_generation_gwh",
                    "worst_segment_min_station_reliability",
                    "candidate_id",
                ],
                ascending=[False, False, True],
                kind="stable",
            )
            ineligible = successful.loc[
                ~successful["meets_reliability_target"]
            ].sort_values(
                [
                    "maximum_station_reliability_shortfall",
                    "validation_mean_annual_generation_gwh",
                    "candidate_id",
                ],
                ascending=[True, False, True],
                kind="stable",
            )
            successful = pd.concat([eligible, ineligible], ignore_index=True)
        else:
            successful = successful.sort_values(
                [
                    "maximum_station_reliability_shortfall",
                    "validation_mean_annual_generation_gwh",
                    "worst_segment_min_station_reliability",
                    "candidate_id",
                ],
                ascending=[True, False, False, True],
                kind="stable",
            )
    else:
        raise ValueError(f"Unsupported ISO selection metric: {selection_metric}")
    failed = results.loc[results["status"] != "success"].copy()
    results = pd.concat([successful, failed], ignore_index=True, sort=False)
    results.to_csv(results_path, index=False)
    best = successful.iloc[0]
    best_config = {
        "candidate_id": best["candidate_id"],
        "model_type": best["model_type"],
        "hyperparameters": json.loads(best["hyperparameters"]),
        "selection_metric": selection_metric,
        "selection_direction": (
            "maximize"
            if selection_metric == "validation_penalized_objective_gwh"
            else "reliability_constraint_then_maximize_generation"
        ),
        "selection_value": float(
            best["validation_penalized_objective_gwh"]
            if selection_metric == "validation_penalized_objective_gwh"
            else (
                best["validation_mean_annual_generation_gwh"]
                if target_feasible
                else best["min_station_firm_power_reliability"]
            )
        ),
        "selection_value_semantics": (
            "validation_penalized_objective_gwh"
            if selection_metric == "validation_penalized_objective_gwh"
            else (
                "validation_mean_annual_generation_gwh_among_reliability_feasible_candidates"
                if target_feasible
                else "maximum_validation_min_station_firm_power_reliability"
            )
        ),
        "reliability_target": reliability_target,
        "reliability_target_feasible": target_feasible,
        "xiaowan_firm_power_reliability": float(
            best["xiaowan_firm_power_reliability"]
        ),
        "nuozhadu_firm_power_reliability": float(
            best["nuozhadu_firm_power_reliability"]
        ),
        "min_station_firm_power_reliability": float(
            best["min_station_firm_power_reliability"]
        ),
        "joint_firm_power_reliability": float(
            best["joint_firm_power_reliability"]
        ),
        "worst_segment_min_station_reliability": float(
            best["worst_segment_min_station_reliability"]
        ),
        "validation_cascade_generation_gwh": float(
            best["validation_cascade_generation_gwh"]
        ),
        "validation_failed_station_months": int(
            best["validation_failed_station_months"]
        ),
        "firm_power_failure_penalty_gwh": failure_penalty,
        "teacher_ddp_schedule": str(teacher_path),
        "teacher_sha256": _sha256(teacher_path),
        "teacher_summary_sha256": _sha256(teacher_path.with_suffix(".json")),
        "physics_config": str(physics_path),
        "physics_config_sha256": _sha256(physics_path),
        "physical_simulator_source_sha256": _sha256(Path(__file__).with_name("environment.py")),
        "validation_initial_state_source": "first_validation_ddp_start_levels",
        "validation_initial_levels_m": list(initial_levels),
        "terminal_reference_levels_m": list(terminal_levels),
        "monthly_max_target_levels_m": monthly_max.tolist(),
        "split": split_metadata,
        "evaluation_data_used_for_selection": False,
        "random_seed": seed,
    }
    best_path = output_dir / "best_iso_config.json"
    best_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    return results_path, best_path


def run_final_training(config_path: str | Path) -> tuple[Path, Path]:
    config, _ = load_iso_config(config_path)
    root = _project_root()
    teacher_path = _resolve(root, config["teacher"]["ddp_schedule_path"])
    expected_years = int(config["development"]["fit_years"]) + int(
        config["development"]["validation_years"]
    )
    teacher, teacher_summary = load_teacher_dataset(
        teacher_path,
        config["teacher"]["columns"],
        expected_years=expected_years,
        **_teacher_requirements(config),
    )
    best_path = _resolve(root, config["final_selected_config_path"])
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if _sha256(teacher_path) != best["teacher_sha256"]:
        raise ValueError("DDP teacher changed after ISO development; repeat development first.")
    physics_path, _, _, physics_config = _load_physics(config, root)
    monthly_max, initial_levels, terminal_levels, failure_penalty = _experiment_contract(
        config, physics_config, teacher_summary
    )
    if _sha256(physics_path) != best["physics_config_sha256"]:
        raise ValueError("Cascade physics configuration changed after ISO development.")
    if best["selection_metric"] != config["objective"]["selection_metric"]:
        raise ValueError("ISO selection metric changed after development.")
    if not np.isclose(
        failure_penalty,
        float(best["firm_power_failure_penalty_gwh"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("ISO failure penalty changed after development.")
    policy = MonthlyCascadeISOPolicy.fit(
        teacher,
        model_type=best["model_type"],
        hyperparameters=best["hyperparameters"],
        random_seed=int(best["random_seed"]),
    )
    output_dir = _resolve(root, config["output_directory"]) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "iso_final_policy.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(policy, stream, protocol=pickle.HIGHEST_PROTOCOL)
    periods = _as_month_period_index(teacher["date"])
    metadata = {
        "model_family": policy.model_type,
        "hyperparameters": policy.hyperparameters,
        "monthly_model_count": len(policy.month_models),
        "feature_definitions": list(FEATURE_NAMES),
        "target_definitions": list(TARGET_NAMES),
        "feature_semantics": {
            "inflow_1_m3s": "Xiaowan natural inflow known for the current month",
            "inflow_2_local_m3s": "Nuozhadu interval natural inflow; excludes Xiaowan release",
        },
        "scaler_parameters_by_month": policy.scaler_metadata(),
        "monthly_fit_domains": {
            str(month): {name: list(bounds) for name, bounds in domain.items()}
            for month, domain in policy.month_domains.items()
        },
        "training_months": len(teacher),
        "training_years": expected_years,
        "training_period": [str(periods[0]), str(periods[-1])],
        "training_dataset_identifier": config["teacher"].get("dataset_identifier"),
        "teacher_ddp_schedule": str(teacher_path),
        "teacher_sha256": _sha256(teacher_path),
        "teacher_summary_sha256": _sha256(teacher_path.with_suffix(".json")),
        "teacher_action_fields": list(TARGET_NAMES),
        "teacher_solver_requirement": "continuous-storage interpolated-continuation DDP",
        "teacher_objective": teacher_summary["reward_configuration"]["objective"],
        "firm_power_failure_penalty_gwh": failure_penalty,
        "failure_definition": teacher_summary["reward_configuration"][
            "failure_definition"
        ],
        "initial_levels_m": list(initial_levels),
        "terminal_reference_levels_m": list(terminal_levels),
        "teacher_terminal_constraint": teacher_summary["terminal_constraint"],
        "monthly_max_target_levels_m": monthly_max.tolist(),
        "random_seed": int(best["random_seed"]),
        "continuous_storage_carryover": True,
        "physics_config": str(physics_path),
        "physics_config_sha256": _sha256(physics_path),
        "physical_simulator_source_sha256": _sha256(Path(__file__).with_name("environment.py")),
        "best_development_config": str(best_path),
        "selection_metric": best["selection_metric"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
    }
    metadata_path = output_dir / "iso_final_policy_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def run_final_evaluation(config_path: str | Path) -> list[tuple[Path, Path]]:
    config, _ = load_iso_config(config_path)
    root = _project_root()
    model_path = _resolve(root, config["final_model_path"])
    metadata_path = model_path.with_name("iso_final_policy_metadata.json")
    final_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as stream:
        policy = pickle.load(stream)
    if not isinstance(policy, MonthlyCascadeISOPolicy):
        raise TypeError("Final model artifact is not a MonthlyCascadeISOPolicy.")
    physics_path, reservoir_1, reservoir_2, physics_config = _load_physics(config, root)
    if _sha256(physics_path) != final_metadata["physics_config_sha256"]:
        raise ValueError("Cascade physics configuration differs from final ISO training metadata.")
    constraints = physics_config["ddp_constraints"]
    monthly_config = constraints["monthly_max_target_levels_m"]
    monthly_max = np.column_stack(
        (
            np.asarray(monthly_config["xiaowan"], dtype=float),
            np.asarray(monthly_config["nuozhadu"], dtype=float),
        )
    )
    if not np.allclose(
        monthly_max,
        np.asarray(final_metadata["monthly_max_target_levels_m"], dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Monthly target limits differ from final ISO training metadata.")
    initial_levels = tuple(float(value) for value in constraints["initial_levels_m"])
    terminal_levels = tuple(float(value) for value in constraints["terminal_levels_m"])
    failure_penalty = float(final_metadata["firm_power_failure_penalty_gwh"])
    if not np.isclose(
        failure_penalty,
        float(config["teacher"]["required_failure_penalty_gwh"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Final ISO failure penalty differs from the configured teacher contract.")
    expected_months = int(config["evaluation"]["expected_months_per_scenario"])
    output_root = _resolve(root, config["output_directory"]) / "evaluation"
    outputs: list[tuple[Path, Path]] = []
    for scenario, scenario_settings in config["evaluation"]["scenarios"].items():
        inflow_path = _resolve(root, scenario_settings["inflow_path"])
        inflows = load_synthetic_joint_inflows(inflow_path, expected_months=expected_months)
        schedule, summary = simulate_continuous_policy(
            policy,
            inflows,
            reservoir_1,
            reservoir_2,
            initial_levels_m=initial_levels,
            terminal_levels_m=terminal_levels,
            firm_power_failure_penalty_gwh=failure_penalty,
            monthly_max_target_levels_m=monthly_max,
            target_unreached_tolerance_m=float(config["target_unreached_tolerance_m"]),
        )
        scenario_dir = output_root / str(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        schedule_path = scenario_dir / "iso_schedule.csv"
        summary_path = scenario_dir / "iso_summary.json"
        schedule.to_csv(schedule_path, index=False)
        summary.update(
            {
                "scenario": scenario,
                "dataset_identifier": scenario_settings.get("dataset_identifier"),
                "inflow_path": str(inflow_path),
                "inflow_sha256": _sha256(inflow_path),
                "final_model_path": str(model_path),
                "final_model_sha256": _sha256(model_path),
                "physics_config": str(physics_path),
                "physics_config_sha256": _sha256(physics_path),
                "physical_simulator_source_sha256": _sha256(
                    Path(__file__).with_name("environment.py")
                ),
                "model_family": policy.model_type,
                "hyperparameters": policy.hyperparameters,
                "feature_order": list(FEATURE_NAMES),
                "target_order": list(TARGET_NAMES),
                "selection_metric": final_metadata["selection_metric"],
                "refit_or_selection_on_scenario": False,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs.append((schedule_path, summary_path))
    return outputs
