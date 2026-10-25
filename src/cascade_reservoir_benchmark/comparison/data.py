from __future__ import annotations

"""Load selected evaluation trajectories into one common internal schema."""

import json
import calendar
import os
from io import BytesIO
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class ReservoirSpec:
    code: str
    display_name: str
    firm_power_mw: float
    dead_level_m: float
    normal_level_m: float
    flood_limit_level_m: float
    monthly_max_target_levels_m: tuple[float, ...]


@dataclass
class ComparisonBundle:
    project_root: Path
    comparison_name: str
    output_directory: Path
    reference_rates_pct: tuple[float, ...]
    figure_dpi: int
    reservoir_specs: dict[str, ReservoirSpec]
    frames: dict[str, pd.DataFrame]
    selections: dict[str, dict[str, Any]]

    @property
    def algorithms(self) -> list[str]:
        return list(self.frames)


REQUIRED_RAW_COLUMNS = {
    "date",
    "start_level_1_m",
    "start_level_2_m",
    "inflow_1_m3s",
    "inflow_2_local_m3s",
    "xiaowan_firm_power_failure",
    "nuozhadu_firm_power_failure",
    "generation_energy_gwh",
    "r1_requested_target_level_m",
    "r1_end_level_m",
    "r1_inflow_m3s",
    "r1_total_discharge_m3s",
    "r1_turbine_discharge_m3s",
    "r1_spill_discharge_m3s",
    "r1_power_kw",
    "r1_energy_gwh",
    "r1_firm_deficit_gwh",
    "r2_requested_target_level_m",
    "r2_end_level_m",
    "r2_inflow_m3s",
    "r2_total_discharge_m3s",
    "r2_turbine_discharge_m3s",
    "r2_spill_discharge_m3s",
    "r2_power_kw",
    "r2_energy_gwh",
    "r2_firm_deficit_gwh",
}


def native_path(path: Path) -> Path:
    """Support extracted Windows result paths longer than MAX_PATH."""
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path.absolute()))
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    path = native_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return loaded


def _read_json(path: Path) -> dict[str, Any]:
    path = native_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        bad_count = int(values.isna().sum())
        raise ValueError(f"Column {column!r} contains {bad_count} non-numeric values")
    return values.astype(float)


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"Column {column!r} contains missing Boolean values")
        return values.astype(bool)

    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = normalized.map(mapping)
    if converted.isna().any():
        invalid = sorted(normalized.loc[converted.isna()].unique())
        raise ValueError(f"Column {column!r} contains invalid Boolean values: {invalid}")
    return converted.astype(bool)


def _build_reservoir_specs(physical: dict[str, Any]) -> dict[str, ReservoirSpec]:
    reservoirs = physical["reservoirs"]
    monthly_limits = physical["ddp_constraints"]["monthly_max_target_levels_m"]
    definitions = {
        "XW": ("xiaowan", "Xiaowan"),
        "NZD": ("nuozhadu", "Nuozhadu"),
    }
    specs: dict[str, ReservoirSpec] = {}
    for code, (config_key, display_name) in definitions.items():
        raw = reservoirs[config_key]
        monthly = tuple(float(value) for value in monthly_limits[config_key])
        if len(monthly) != 12:
            raise ValueError(f"Expected 12 monthly level limits for {display_name}")
        specs[code] = ReservoirSpec(
            code=code,
            display_name=display_name,
            firm_power_mw=float(raw["firm_power_kw"]) / 1000.0,
            dead_level_m=float(raw["dead_level_m"]),
            normal_level_m=float(raw["normal_level_m"]),
            flood_limit_level_m=float(raw["flood_limit_level_m"]),
            monthly_max_target_levels_m=monthly,
        )
    return specs


def _normalize_trajectory(
    raw: pd.DataFrame,
    algorithm: str,
    selection: dict[str, Any],
    specs: dict[str, ReservoirSpec],
) -> pd.DataFrame:
    # PPO exports satisfaction flags and unpenalized reward, not failure flags.
    # Convert only this documented schema; all existing consistency checks apply.
    if algorithm == "PPO":
        raw = raw.copy()
        for station in ("xiaowan", "nuozhadu"):
            raw[f"{station}_firm_power_failure"] = ~_boolean(
                raw, f"{station}_firm_power_satisfied"
            )
        raw["generation_energy_gwh"] = _numeric(raw, "reward_gwh")
    missing = sorted(REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing:
        raise KeyError(f"{algorithm} trajectory is missing columns: {missing}")

    date_text = raw["date"].astype(str).str.strip()
    date_parts = date_text.str.extract(
        r"^(?P<year>\d{4,})-(?P<month>\d{2})-(?P<day>\d{2})(?:[ T].*)?$"
    )
    if date_parts.isna().any().any():
        raise ValueError(f"{algorithm} trajectory contains invalid dates")
    years = date_parts["year"].astype(int)
    months = date_parts["month"].astype(int)
    days = date_parts["day"].astype(int)
    if not months.between(1, 12).all() or not (days == 1).all():
        raise ValueError(f"{algorithm} dates must be first-of-month values")
    serial_month = years.to_numpy(dtype=np.int64) * 12 + months.to_numpy(dtype=np.int64)
    if len(serial_month) > 1 and not np.all(np.diff(serial_month) == 1):
        raise ValueError(f"{algorithm} dates must be unique, increasing, and monthly continuous")
    days_in_month = pd.Series(
        [calendar.monthrange(int(year), int(month))[1] for year, month in zip(years, months)],
        index=raw.index,
        dtype=float,
    )

    routed_release = (
        _numeric(raw, "routed_inflow_m3s")
        if "routed_inflow_m3s" in raw.columns
        else _numeric(raw, "r1_total_discharge_m3s")
    )

    xw_energy = _numeric(raw, "r1_energy_gwh")
    nzd_energy = _numeric(raw, "r2_energy_gwh")
    cascade_energy = pd.to_numeric(raw["generation_energy_gwh"], errors="coerce")
    if cascade_energy.isna().all():
        cascade_energy = xw_energy + nzd_energy
    elif cascade_energy.isna().any():
        bad_count = int(cascade_energy.isna().sum())
        raise ValueError(
            f"Column 'generation_energy_gwh' contains {bad_count} missing values "
            "but is not wholly empty"
        )
    else:
        cascade_energy = cascade_energy.astype(float)

    frame = pd.DataFrame(
        {
            "algorithm": algorithm,
            "seed": int(selection.get("seed", 0)),
            "candidate_id": str(selection["candidate_id"]),
            "failure_penalty_gwh": float(selection["failure_penalty_gwh"]),
            "date": date_text,
            "year": years,
            "month": months,
            "month_hours": days_in_month * 24.0,
            "month_seconds": days_in_month * 86400.0,
            "generation_energy_gwh": cascade_energy,
            "xw_start_level_m": _numeric(raw, "start_level_1_m"),
            "xw_target_level_m": _numeric(raw, "r1_requested_target_level_m"),
            "xw_end_level_m": _numeric(raw, "r1_end_level_m"),
            "xw_natural_inflow_m3s": _numeric(raw, "r1_inflow_m3s"),
            "xw_total_release_m3s": _numeric(raw, "r1_total_discharge_m3s"),
            "xw_turbine_release_m3s": _numeric(raw, "r1_turbine_discharge_m3s"),
            "xw_spill_m3s": _numeric(raw, "r1_spill_discharge_m3s"),
            "xw_power_mw": _numeric(raw, "r1_power_kw") / 1000.0,
            "xw_energy_gwh": xw_energy,
            "xw_deficit_energy_gwh": _numeric(raw, "r1_firm_deficit_gwh"),
            "xw_failure": _boolean(raw, "xiaowan_firm_power_failure"),
            "nzd_start_level_m": _numeric(raw, "start_level_2_m"),
            "nzd_target_level_m": _numeric(raw, "r2_requested_target_level_m"),
            "nzd_end_level_m": _numeric(raw, "r2_end_level_m"),
            "nzd_local_inflow_m3s": _numeric(raw, "inflow_2_local_m3s"),
            "nzd_effective_inflow_m3s": _numeric(raw, "r2_inflow_m3s"),
            "nzd_total_release_m3s": _numeric(raw, "r2_total_discharge_m3s"),
            "nzd_turbine_release_m3s": _numeric(raw, "r2_turbine_discharge_m3s"),
            "nzd_spill_m3s": _numeric(raw, "r2_spill_discharge_m3s"),
            "nzd_power_mw": _numeric(raw, "r2_power_kw") / 1000.0,
            "nzd_energy_gwh": nzd_energy,
            "nzd_deficit_energy_gwh": _numeric(raw, "r2_firm_deficit_gwh"),
            "nzd_failure": _boolean(raw, "nuozhadu_firm_power_failure"),
            "routed_xw_release_m3s": routed_release,
        }
    )

    if len(frame) == 0 or len(frame) % 12 != 0:
        raise ValueError(f"{algorithm} trajectory must contain complete calendar years")
    if sorted(frame["month"].unique()) != list(range(1, 13)):
        raise ValueError(f"{algorithm} trajectory does not cover all calendar months")

    for prefix, code in (("xw", "XW"), ("nzd", "NZD")):
        spec = specs[code]
        calculated_failure = frame[f"{prefix}_power_mw"] < spec.firm_power_mw
        if not calculated_failure.equals(frame[f"{prefix}_failure"]):
            disagreements = int((calculated_failure != frame[f"{prefix}_failure"]).sum())
            raise ValueError(f"{algorithm} {code} failure flag disagrees in {disagreements} months")

        expected_energy = frame[f"{prefix}_power_mw"] * frame["month_hours"] / 1000.0
        if not np.allclose(expected_energy, frame[f"{prefix}_energy_gwh"], atol=1e-8, rtol=1e-10):
            raise ValueError(f"{algorithm} {code} monthly energy does not match power and month duration")

        deficit_mw = (spec.firm_power_mw - frame[f"{prefix}_power_mw"]).clip(lower=0.0)
        expected_deficit_energy = deficit_mw * frame["month_hours"] / 1000.0
        if not np.allclose(
            expected_deficit_energy,
            frame[f"{prefix}_deficit_energy_gwh"],
            atol=1e-8,
            rtol=1e-10,
        ):
            raise ValueError(f"{algorithm} {code} deficit energy is inconsistent")

    station_sum = frame["xw_energy_gwh"] + frame["nzd_energy_gwh"]
    if not np.allclose(station_sum, frame["generation_energy_gwh"], atol=1e-8, rtol=1e-10):
        raise ValueError(f"{algorithm} cascade energy differs from the two station sum")

    return frame


def _validate_summary(algorithm: str, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    trajectory_total = float(frame["generation_energy_gwh"].sum())
    summary_keys = {
        "ISO": "cascade_generation_gwh",
        "DQN": "total_cascade_generation_gwh",
        "PPO": "total_cascade_generation_gwh",
    }
    summary_key = summary_keys.get(algorithm, "total_generation_gwh")
    if summary_key in summary and not np.isclose(
        trajectory_total,
        float(summary[summary_key]),
        atol=1e-5,
        rtol=1e-12,
    ):
        raise ValueError(f"{algorithm} trajectory generation does not match {summary_key}")

    if "months" in summary and int(summary["months"]) != len(frame):
        raise ValueError(f"{algorithm} trajectory length does not match summary months")


def _validate_common_hydrology(frames: dict[str, pd.DataFrame]) -> None:
    algorithms = list(frames)
    reference = next(iter(frames[algorithms[0]].groupby("seed")))[1].reset_index(drop=True)
    for algorithm, frame in frames.items():
        for seed, candidate in frame.groupby("seed", sort=False):
            candidate = candidate.reset_index(drop=True)
            if not reference["date"].equals(candidate["date"]):
                raise ValueError(f"Date sequence differs for {algorithm} seed {seed}")
            for column in ("xw_natural_inflow_m3s", "nzd_local_inflow_m3s"):
                if not np.allclose(reference[column], candidate[column], atol=1e-10, rtol=0.0):
                    raise ValueError(f"{column} differs for {algorithm} seed {seed}")


def load_comparison_bundle(
    project_root: Path,
    comparison_config: Path | None = None,
) -> ComparisonBundle:
    project_root = project_root.resolve()
    config_path = comparison_config or project_root / "configs" / "comparison_ddp_sdp_iso.yaml"
    config = _read_yaml(config_path)
    comparison = config["comparison"]

    physical_path = project_root / str(comparison["physical_config"])
    physical = _read_yaml(physical_path)
    specs = _build_reservoir_specs(physical)

    selections = config["selected_results"]
    frames: dict[str, pd.DataFrame] = {}
    for algorithm, selection in selections.items():
        results = selection.get("results", [selection])
        if not results:
            raise ValueError(f"No selected results for {algorithm}")
        seed_ids = [int(result.get("seed", 0)) for result in results]
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError(f"Duplicate seeds for {algorithm}: {seed_ids}")
        normalized_frames = []
        for result in results:
            if "archive" in selection:
                archive_path = project_root / str(selection["archive"])
                with zipfile.ZipFile(native_path(archive_path)) as archive:
                    raw = pd.read_csv(BytesIO(archive.read(str(result["trajectory"]))))
                    summary = json.loads(archive.read(str(result["summary"])).decode("utf-8-sig"))
                trajectory_path = str(archive_path) + "::" + str(result["trajectory"])
                summary_path = str(archive_path) + "::" + str(result["summary"])
            else:
                trajectory_path = project_root / str(result["trajectory"])
                summary_path = project_root / str(result["summary"])
                raw = pd.read_csv(native_path(trajectory_path))
                summary = _read_json(summary_path)
            normalized = _normalize_trajectory(raw, algorithm, {**selection, **result}, specs)
            _validate_summary(algorithm, normalized, summary)
            if algorithm == "DQN" and selection.get("require_formal_evaluation", False):
                expected = {
                    "seed": int(result["seed"]), "months": 4800, "years": 400,
                    "firm_power_failure_penalty_gwh": float(selection["failure_penalty_gwh"]),
                    "checkpoint_role": "final", "mode": "final", "evaluation_scenario": "baseline",
                    "evaluation_data_used_for_model_selection": False,
                }
                for key, value in expected.items():
                    if summary.get(key) != value:
                        raise ValueError(f"DQN formal evaluation mismatch: {key}, {summary_path}")
                for station, prefix in (("xiaowan", "xw"), ("nuozhadu", "nzd")):
                    rate = 1.0 - float(normalized[f"{prefix}_failure"].mean())
                    if not np.isclose(rate, summary["firm_power_reliability"][station], atol=1e-12, rtol=0):
                        raise ValueError(f"DQN {station} reliability disagrees with formal summary")
                annual = normalized["generation_energy_gwh"].sum() / 400
                if not np.isclose(annual, summary["mean_annual_cascade_generation_gwh"], atol=1e-7, rtol=0):
                    raise ValueError("DQN formal annual generation disagrees with trajectory")
            if algorithm == "PPO":
                if int(summary["seed"]) != int(result["seed"]):
                    raise ValueError(f"PPO summary seed differs in {summary_path}")
                if float(summary["firm_power_failure_penalty_gwh"]) != float(selection["failure_penalty_gwh"]):
                    raise ValueError(f"PPO summary penalty differs in {summary_path}")
                for station, prefix in (("xiaowan", "xw"), ("nuozhadu", "nzd")):
                    rate = 1.0 - float(normalized[f"{prefix}_failure"].mean())
                    if not np.isclose(rate, summary["firm_power_reliability"][station], atol=1e-12):
                        raise ValueError(f"PPO {station} reliability disagrees with summary")
            normalized_frames.append(normalized)
            result["trajectory_path"] = trajectory_path
            result["summary_path"] = summary_path
        frames[algorithm] = pd.concat(normalized_frames, ignore_index=True)

    _validate_common_hydrology(frames)

    return ComparisonBundle(
        project_root=project_root,
        comparison_name=str(comparison["name"]),
        output_directory=project_root / str(comparison["output_directory"]),
        reference_rates_pct=tuple(float(value) for value in comparison["reference_guarantee_rates_pct"]),
        figure_dpi=int(comparison.get("figure_dpi", 600)),
        reservoir_specs=specs,
        frames=frames,
        selections=selections,
    )
