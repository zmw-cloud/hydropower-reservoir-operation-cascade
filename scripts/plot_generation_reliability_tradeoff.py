from __future__ import annotations

"""Plot observed generation--firm-power reliability trade-off points.

This script only reads existing result summaries.  It does not run or modify
DDP, SDP, ISO, DQN, or PPO. DDP and SDP use 400-year evaluation results.
ISO uses the existing 80-year validation results for a fixed RF41 model trained
from eleven DDP teacher-penalty trajectories.
PPO uses 80-year validation means over seeds 1, 42 and 100.
DQN uses eight penalties on 80-year validation data, averaging seeds 1, 42, 100.
"""

import json
import argparse
import os
import sys
from io import BytesIO
import zipfile
from pathlib import Path
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cascade_reservoir_benchmark.comparison.presentation import (
    FONT_RC, display_text, finalize_figure,
)
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
COMPARISON_ROOT = OUTPUTS_ROOT / "comparison_ddp_sdp_iso_dqn_ppo_formal"
FIGURES_DIR = COMPARISON_ROOT / "figures"
TABLES_DIR = COMPARISON_ROOT / "tables"

DDP_ROOT = PROJECT_ROOT / "reference_results/tradeoffs/DDP"
SDP_ROOT = PROJECT_ROOT / "reference_results/tradeoffs/SDP"
ISO_TEACHER_SWEEP_SUMMARY = PROJECT_ROOT / "reference_results/tradeoffs/iso_validation11.csv"
DQN_VALIDATION_ARCHIVE = PROJECT_ROOT / "reference_results/tradeoffs/dqn_validation8.zip"
PPO_SCAN_ROOT = PROJECT_ROOT / "reference_results/tradeoffs/ppo"
PPO_VALIDATION_SUMMARY = PPO_SCAN_ROOT / "continuous_ppo_v16_penalty_scan_validation/analysis/v16_penalty_scan_summary.csv"

REFERENCE_RATES_PCT = (80.0, 90.0)

LABEL_FT = 13
TICK_FT = 11
LEGEND_FT = 10
ANNOTATION_FT = 7.5

ALGORITHM_STYLES = {
    "DDP": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "SDP": {"color": "#2ca02c", "marker": "s", "linestyle": "-"},
    "ISO": {"color": "#d62728", "marker": "D", "linestyle": "-"},
    "DQN": {"color": "#9467bd", "marker": "^", "linestyle": "-"},
    "PPO": {"color": "#ff7f0e", "marker": "v", "linestyle": "-"},
}


def _native_path(path: Path) -> Path:
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path.absolute()))
    return path


def _read_json(path: Path) -> dict:
    path = _native_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _penalty_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _years_from_evaluation_json(path: Path) -> float:
    months = float(_read_json(path)["months"])
    if months <= 0 or months % 12 != 0:
        raise ValueError(f"Expected complete calendar years in {path}, got {months} months")
    return months / 12.0


def load_ddp_points() -> pd.DataFrame:
    summary_path = DDP_ROOT / "failure_penalty_sweep_summary.csv"
    summary = pd.read_csv(summary_path)
    records: list[dict] = []

    for row in summary.itertuples(index=False):
        penalty = float(row.failure_penalty_gwh)
        token = _penalty_token(penalty)
        result_dir = DDP_ROOT / f"penalty_{token}"
        json_files = sorted(result_dir.glob("ddp_*evaluation*.json"))
        if len(json_files) != 1:
            raise RuntimeError(f"Expected one DDP evaluation JSON in {result_dir}, found {len(json_files)}")
        years = _years_from_evaluation_json(json_files[0])
        total_generation = float(row.generation_gwh)
        records.append(
            {
                "algorithm": "DDP",
                "candidate_id": f"penalty_{token}",
                "failure_penalty_gwh": penalty,
                "xw_guarantee_rate_pct": float(row.xiaowan_reliability_pct),
                "nzd_guarantee_rate_pct": float(row.nuozhadu_reliability_pct),
                "min_guarantee_rate_pct": min(
                    float(row.xiaowan_reliability_pct),
                    float(row.nuozhadu_reliability_pct),
                ),
                "total_generation_gwh": total_generation,
                "period_years": years,
                "mean_annual_generation_gwh": total_generation / years,
                "data_split": "evaluation",
                "is_selected": False,
                "source_is_pareto": False,
                "source_summary": str(summary_path.relative_to(PROJECT_ROOT).as_posix()),
                "source_evaluation_json": str(json_files[0].relative_to(PROJECT_ROOT).as_posix()),
            }
        )
    return pd.DataFrame.from_records(records)


def load_sdp_points() -> pd.DataFrame:
    summary_path = SDP_ROOT / "failure_penalty_sweep" / "sdp_all_failure_penalty_summary.csv"
    summary = pd.read_csv(summary_path)
    # The 2000--16000 GWh extension produces a sharp downstream-reliability
    # reversal and is excluded from the manuscript trend figure.
    summary = summary.loc[summary["failure_penalty_gwh"] <= 1000.0].copy()
    records: list[dict] = []

    for row in summary.itertuples(index=False):
        penalty = float(row.failure_penalty_gwh)
        token = _penalty_token(penalty)
        result_dir = SDP_ROOT / f"failure_penalty_{token}"
        json_files = sorted(result_dir.glob("sdp_*evaluation*.json"))
        if len(json_files) != 1:
            raise RuntimeError(f"Expected one SDP evaluation JSON in {result_dir}, found {len(json_files)}")
        years = _years_from_evaluation_json(json_files[0])
        total_generation = float(row.total_generation_gwh)
        records.append(
            {
                "algorithm": "SDP",
                "candidate_id": f"penalty_{token}",
                "failure_penalty_gwh": penalty,
                "xw_guarantee_rate_pct": float(row.xiaowan_reliability_pct),
                "nzd_guarantee_rate_pct": float(row.nuozhadu_reliability_pct),
                "min_guarantee_rate_pct": min(
                    float(row.xiaowan_reliability_pct),
                    float(row.nuozhadu_reliability_pct),
                ),
                "total_generation_gwh": total_generation,
                "period_years": years,
                "mean_annual_generation_gwh": total_generation / years,
                "data_split": "evaluation",
                "is_selected": False,
                "source_is_pareto": False,
                "source_summary": str(summary_path.relative_to(PROJECT_ROOT).as_posix()),
                "source_evaluation_json": str(json_files[0].relative_to(PROJECT_ROOT).as_posix()),
            }
        )
    return pd.DataFrame.from_records(records)


def load_iso_points() -> pd.DataFrame:
    summary_path = ISO_TEACHER_SWEEP_SUMMARY
    candidates = pd.read_csv(summary_path).sort_values("teacher_failure_penalty_gwh")
    validation_years = 80.0
    if candidates.empty:
        raise ValueError(f"No ISO teacher-penalty sweep points in {summary_path}")
    if len(candidates) != 11:
        raise ValueError(f"Expected 11 ISO teacher-penalty points, found {len(candidates)}")

    records: list[dict] = []
    for row in candidates.itertuples(index=False):
        teacher_penalty = float(row.teacher_failure_penalty_gwh)
        xw_rate = 100.0 * float(row.xiaowan_reliability)
        nzd_rate = 100.0 * float(row.nuozhadu_reliability)
        mean_annual_generation = float(row.mean_annual_generation_gwh)
        records.append(
            {
                "algorithm": "ISO",
                "candidate_id": f"rf41_teacher_penalty_{_penalty_token(teacher_penalty)}",
                "failure_penalty_gwh": teacher_penalty,
                "xw_guarantee_rate_pct": xw_rate,
                "nzd_guarantee_rate_pct": nzd_rate,
                "min_guarantee_rate_pct": min(xw_rate, nzd_rate),
                "total_generation_gwh": mean_annual_generation * validation_years,
                "period_years": validation_years,
                "mean_annual_generation_gwh": mean_annual_generation,
                "data_split": "validation",
                "is_selected": teacher_penalty == 500.0,
                "source_is_pareto": False,
                "source_summary": str(summary_path.relative_to(PROJECT_ROOT).as_posix()),
                "source_evaluation_json": "",
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_dqn_validation(seeds: pd.DataFrame, summary: pd.DataFrame,
                             source: str) -> pd.DataFrame:
    """Aggregate validated seed metrics, not averaged powers or penalized rewards."""
    seeds = seeds.copy()
    summary = summary.copy()
    numeric = ["lambda", "seed", "xw_pct", "nzd_pct", "r_min_pct", "generation_twh",
               "best_episode", "validation_months", "validation_years"]
    for column in numeric:
        seeds[column] = pd.to_numeric(seeds[column], errors="raise")
    if not np.isfinite(seeds[numeric].to_numpy()).all():
        raise ValueError("DQN validation metrics contain non-finite values")
    expected = {0, 2000, 3000, 4000, 5000, 6000, 8000, 10000}
    if (len(seeds) != 24 or set(seeds["lambda"]) != expected
            or seeds.duplicated(["lambda", "seed"]).any()):
        raise ValueError("DQN validation requires eight penalties and 24 unique seed results")
    if not ((seeds["dataset"] == "validation").all()
            and (seeds["checkpoint_role"] == "best_validation").all()
            and (seeds["validation_months"] == 960).all()
            and (seeds["validation_years"] == 80).all()):
        raise ValueError("DQN scan must use 80-year validation and best-validation checkpoints")
    for column in ("xw_pct", "nzd_pct"):
        if not seeds[column].between(0, 100).all():
            raise ValueError(f"Invalid DQN rate: {column}")
        monthly_successes = seeds[column].to_numpy() * 960 / 100
        if not np.allclose(monthly_successes, np.round(monthly_successes), atol=1e-7, rtol=0):
            raise ValueError("DQN validation reliability does not correspond to 960 monthly samples")
    if (seeds["generation_twh"] < 0).any():
        raise ValueError("Negative DQN annual generation")
    if not np.allclose(seeds["r_min_pct"], seeds[["xw_pct", "nzd_pct"]].min(axis=1), atol=1e-8):
        raise ValueError("DQN seed minimum reliability inconsistent")
    summary["lambda"] = pd.to_numeric(summary["lambda"], errors="raise")
    if len(summary) != 8 or set(summary["lambda"]) != expected or summary["lambda"].duplicated().any():
        raise ValueError("DQN validation aggregate penalties are missing or duplicated")
    records = []
    for penalty, group in seeds.groupby("lambda", sort=True):
        if len(group) != 3 or set(group["seed"]) != {1, 42, 100}:
            raise ValueError(f"DQN lambda={penalty:g} requires seeds 1, 42, 100")
        row = summary.loc[summary["lambda"] == penalty].iloc[0]
        if float(row["seed_count"]) != 3:
            raise ValueError("DQN aggregate seed_count must be 3")
        for column, mean_column, std_column in (
            ("xw_pct", "xw_mean", "xw_std"), ("nzd_pct", "nzd_mean", "nzd_std"),
            ("r_min_pct", "r_min_mean", "r_min_std"),
            ("generation_twh", "generation_mean", "generation_std"),
        ):
            if not (np.isclose(group[column].mean(), float(row[mean_column]), atol=1e-8, rtol=0)
                    and np.isclose(group[column].std(ddof=1), float(row[std_column]), atol=1e-8, rtol=0)):
                raise ValueError(f"DQN lambda={penalty:g} aggregate mismatch: {column}")
        xw_rate, nzd_rate = float(group["xw_pct"].mean()), float(group["nzd_pct"].mean())
        generation = float(group["generation_twh"].mean()) * 1000.0
        records.append({
            "algorithm": "DQN", "candidate_id": f"lambda_{_penalty_token(penalty)}_mean_3seeds",
            "failure_penalty_gwh": float(penalty), "xw_guarantee_rate_pct": xw_rate,
            "nzd_guarantee_rate_pct": nzd_rate, "min_guarantee_rate_pct": min(xw_rate, nzd_rate),
            "total_generation_gwh": generation * 80, "period_years": 80.0,
            "mean_annual_generation_gwh": generation, "data_split": "validation_mean_3seeds",
            "is_selected": penalty == 5000, "source_is_pareto": False,
            "source_summary": source, "source_evaluation_json": "",
            "source_validation_json": "", "checkpoint_role": "best_validation",
        })
    return pd.DataFrame.from_records(records)


def load_dqn_points() -> pd.DataFrame:
    archive_path = DQN_VALIDATION_ARCHIVE
    with zipfile.ZipFile(_native_path(archive_path)) as archive:
        seeds = pd.read_csv(BytesIO(archive.read("dqn_validation_seed_results.csv")))
        summary = pd.read_csv(BytesIO(archive.read("dqn_validation_lambda_summary.csv")))
        for row in seeds.itertuples(index=False):
            folder = f"validation_runs/lambda{_penalty_token(float(row[0]))}/seed_{int(row.seed)}"
            metadata = json.loads(archive.read(folder + "/metadata.json"))
            history = pd.read_csv(BytesIO(archive.read(folder + "/validation/history.csv")))
            match = history.loc[history["episode"] == row.best_episode]
            if len(match) != 1 or metadata["best_validation_episode"] != row.best_episode:
                raise ValueError(f"DQN selected validation episode mismatch: {folder}")
            selected = match.iloc[0]
            if metadata["seed"] != row.seed or metadata["mode"] != "development":
                raise ValueError(f"DQN validation metadata mismatch: {folder}")
            comparisons = (
                (row.xw_pct, float(selected["xiaowan_firm_power_reliability"]) * 100),
                (row.nzd_pct, float(selected["nuozhadu_firm_power_reliability"]) * 100),
                (row.generation_twh * 1000, float(selected["validation_mean_annual_cascade_generation_gwh"])),
                (row.generation_twh * 1000 * 80, float(selected["validation_total_cascade_generation_gwh"])),
                (metadata["best_validation_score"], float(selected["validation_mean_annual_penalized_objective_gwh"])),
            )
            if selected["validation_months"] != 960 or not all(
                np.isclose(a, b, atol=1e-7, rtol=0) for a, b in comparisons
            ):
                raise ValueError(f"DQN validation history metrics disagree: {folder}")
            expected_objective = (row.generation_twh * 1000
                                  - float(row[0]) * float(selected["validation_failed_station_months"]) / 80)
            if not np.isclose(expected_objective, metadata["best_validation_score"], atol=1e-7, rtol=0):
                raise ValueError(f"DQN physical generation/penalized objective mismatch: {folder}")
    return aggregate_dqn_validation(seeds, summary,
        archive_path.relative_to(PROJECT_ROOT).as_posix() + "::dqn_validation_lambda_summary.csv")


def load_ppo_points() -> pd.DataFrame:
    """Read validation only, cross-check each scan mean against three seed summaries."""
    candidates = pd.read_csv(_native_path(PPO_VALIDATION_SUMMARY))
    expected_penalties = {0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000}
    if len(candidates) != 12 or set(candidates["lambda"]) != expected_penalties:
        raise ValueError("PPO validation scan must have twelve unique penalty groups")
    records = []
    for _, row in candidates.iterrows():
        penalty = float(row["lambda"])
        experiment_id = str(row["experiment_id"])
        root_name = (
            "continuous_ppo_v16_geometry_conditioned_lambda8000"
            if penalty == 8000 else "continuous_ppo_v16_penalty_scan_validation"
        )
        experiment_root = PPO_SCAN_ROOT / root_name / "experiments" / experiment_id
        seed_summary_path = experiment_root / "validation" / "validation_seed_summary.csv"
        seed_summary = pd.read_csv(_native_path(seed_summary_path))
        if len(seed_summary) != 3 or set(seed_summary["seed"]) != {1, 42, 100}:
            raise ValueError(f"PPO validation seeds must be 1, 42, 100: {seed_summary_path}")
        for source_col, seed_col, factor in (
            ("generation_twh", "mean_annual_generation_gwh", 0.001),
            ("xiaowan_reliability_pct", "xiaowan_reliability", 100.0),
            ("nuozhadu_reliability_pct", "nuozhadu_reliability", 100.0),
        ):
            if not np.isclose(float(row[source_col]), float(seed_summary[seed_col].mean()) * factor,
                              atol=1e-8, rtol=0.0):
                raise ValueError(f"PPO scan mean disagrees with seed summary: penalty={penalty}, {source_col}")
        json_paths = []
        for seed_row in seed_summary.itertuples(index=False):
            json_path = experiment_root / f"seed_{int(seed_row.seed)}" / "development" / "validation" / "validation_monthly.json"
            result = _read_json(json_path)
            if result["months"] != 960 or result["years"] != 80.0:
                raise ValueError(f"Expected PPO 80-year validation: {json_path}")
            if int(result["seed"]) != int(seed_row.seed) or float(result["firm_power_failure_penalty_gwh"]) != penalty:
                raise ValueError(f"Wrong PPO validation seed or penalty: {json_path}")
            if not np.isclose(result["mean_annual_cascade_generation_gwh"], seed_row.mean_annual_generation_gwh,
                              atol=1e-7, rtol=0.0):
                raise ValueError(f"PPO validation generation mismatch: {json_path}")
            for station, column in (("xiaowan", "xiaowan_reliability"), ("nuozhadu", "nuozhadu_reliability")):
                if not np.isclose(result["firm_power_reliability"][station], getattr(seed_row, column), atol=1e-12):
                    raise ValueError(f"PPO validation reliability mismatch: {json_path}")
            json_paths.append(json_path.relative_to(PROJECT_ROOT).as_posix())
        xw_rate = float(row["xiaowan_reliability_pct"])
        nzd_rate = float(row["nuozhadu_reliability_pct"])
        generation = float(row["generation_twh"]) * 1000.0
        records.append({
            "algorithm": "PPO",
            "candidate_id": f"lambda_{_penalty_token(penalty)}_mean_3seeds",
            "failure_penalty_gwh": penalty,
            "xw_guarantee_rate_pct": xw_rate,
            "nzd_guarantee_rate_pct": nzd_rate,
            "min_guarantee_rate_pct": min(xw_rate, nzd_rate),
            "total_generation_gwh": generation * 80.0,
            "period_years": 80.0,
            "mean_annual_generation_gwh": generation,
            "data_split": "validation_mean_3seeds",
            "is_selected": penalty == 6000.0,
            "source_is_pareto": False,
            "source_summary": PPO_VALIDATION_SUMMARY.relative_to(PROJECT_ROOT).as_posix(),
            "source_evaluation_json": "",  # Never mislabel a validation file as evaluation.
            "source_validation_json": ";".join(json_paths),
        })
    return pd.DataFrame.from_records(records)


POINT_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "DDP": load_ddp_points,
    "SDP": load_sdp_points,
    "ISO": load_iso_points,
    "DQN": load_dqn_points,
    "PPO": load_ppo_points,
}


def load_tradeoff_points() -> pd.DataFrame:
    points = pd.concat([loader() for loader in POINT_LOADERS.values()], ignore_index=True)
    numeric_columns = [
        "failure_penalty_gwh",
        "xw_guarantee_rate_pct",
        "nzd_guarantee_rate_pct",
        "min_guarantee_rate_pct",
        "total_generation_gwh",
        "period_years",
        "mean_annual_generation_gwh",
    ]
    if not np.isfinite(points[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Trade-off data contain missing numeric values")
    for column in ("xw_guarantee_rate_pct", "nzd_guarantee_rate_pct"):
        if not points[column].between(0.0, 100.0).all():
            raise ValueError(f"Invalid guarantee rate in {column}")
    return points.sort_values(["algorithm", "failure_penalty_gwh"]).reset_index(drop=True)


def build_nearest_reference_table(points: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    reservoirs = {
        "XW": "xw_guarantee_rate_pct",
        "NZD": "nzd_guarantee_rate_pct",
    }
    for reservoir, rate_column in reservoirs.items():
        for reference in REFERENCE_RATES_PCT:
            for algorithm, group in points.groupby("algorithm", sort=False):
                distances = (group[rate_column] - reference).abs()
                row = group.loc[distances.idxmin()]
                records.append(
                    {
                        "reservoir": reservoir,
                        "reference_rate_pct": reference,
                        "algorithm": algorithm,
                        "candidate_id": row["candidate_id"],
                        "failure_penalty_gwh": row["failure_penalty_gwh"],
                        "actual_guarantee_rate_pct": row[rate_column],
                        "absolute_difference_pct_points": abs(row[rate_column] - reference),
                        "mean_annual_generation_gwh": row["mean_annual_generation_gwh"],
                    }
                )
    return pd.DataFrame.from_records(records)


def build_balanced_reference_table(points: pd.DataFrame) -> pd.DataFrame:
    """Find one observed candidate per algorithm that is closest for both reservoirs.

    The primary distance is the larger of the two station-specific absolute
    deviations from the reference rate.  This keeps XW and NZD visible and does
    not replace them with an average guarantee rate.  Mean deviation and then
    higher generation are only tie breakers.
    """
    records: list[dict] = []
    for reference in REFERENCE_RATES_PCT:
        for algorithm, group in points.groupby("algorithm", sort=False):
            candidates = group.copy()
            candidates["xw_abs_difference_pct_points"] = (
                candidates["xw_guarantee_rate_pct"] - reference
            ).abs()
            candidates["nzd_abs_difference_pct_points"] = (
                candidates["nzd_guarantee_rate_pct"] - reference
            ).abs()
            candidates["max_station_abs_difference_pct_points"] = candidates[
                ["xw_abs_difference_pct_points", "nzd_abs_difference_pct_points"]
            ].max(axis=1)
            candidates["mean_station_abs_difference_pct_points"] = candidates[
                ["xw_abs_difference_pct_points", "nzd_abs_difference_pct_points"]
            ].mean(axis=1)
            row = candidates.sort_values(
                [
                    "max_station_abs_difference_pct_points",
                    "mean_station_abs_difference_pct_points",
                    "mean_annual_generation_gwh",
                ],
                ascending=[True, True, False],
            ).iloc[0]
            records.append(
                {
                    "reference_rate_pct": reference,
                    "algorithm": algorithm,
                    "candidate_id": row["candidate_id"],
                    "failure_penalty_gwh": row["failure_penalty_gwh"],
                    "xw_guarantee_rate_pct": row["xw_guarantee_rate_pct"],
                    "nzd_guarantee_rate_pct": row["nzd_guarantee_rate_pct"],
                    "min_guarantee_rate_pct": row["min_guarantee_rate_pct"],
                    "xw_abs_difference_pct_points": row["xw_abs_difference_pct_points"],
                    "nzd_abs_difference_pct_points": row["nzd_abs_difference_pct_points"],
                    "max_station_abs_difference_pct_points": row[
                        "max_station_abs_difference_pct_points"
                    ],
                    "mean_station_abs_difference_pct_points": row[
                        "mean_station_abs_difference_pct_points"
                    ],
                    "mean_annual_generation_gwh": row["mean_annual_generation_gwh"],
                }
            )
    return pd.DataFrame.from_records(records)


def _annotation_offset(algorithm: str, index: int) -> tuple[int, int]:
    if algorithm == "DDP":
        return (4, 7 if index % 2 == 0 else -11)
    if algorithm == "SDP":
        return (4, -11 if index % 2 == 0 else 7)
    return (7, 8)


def compute_pareto_front(points: pd.DataFrame, rate_column: str) -> pd.DataFrame:
    """Return points not dominated in guarantee rate and annual generation.

    Both objectives are maximized.  The result is sorted from lower to higher
    guarantee rate so joining the points produces the observed Pareto frontier
    without interpolation or smoothing.
    """
    valid = points.dropna(subset=[rate_column, "mean_annual_generation_gwh"]).copy()
    keep: list[bool] = []
    for index, row in valid.iterrows():
        at_least_as_good = (
            (valid[rate_column] >= row[rate_column])
            & (valid["mean_annual_generation_gwh"] >= row["mean_annual_generation_gwh"])
        )
        strictly_better = (
            (valid[rate_column] > row[rate_column])
            | (valid["mean_annual_generation_gwh"] > row["mean_annual_generation_gwh"])
        )
        dominated = bool((at_least_as_good & strictly_better).drop(index=index).any())
        keep.append(not dominated)
    return valid.loc[keep].sort_values(
        [rate_column, "mean_annual_generation_gwh"],
        ascending=[True, False],
    )


def _short_iso_candidate_id(candidate_id: str) -> str:
    suffix = candidate_id.rsplit("_", maxsplit=1)[-1]
    return f"RF{suffix}"


def draw_tradeoff_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    rate_column: str,
    reservoir_name: str,
    panel_label: str | None = None,
) -> None:
    for algorithm in POINT_LOADERS:
        group = points.loc[points["algorithm"] == algorithm].sort_values("failure_penalty_gwh")
        if group.empty:
            continue
        style = ALGORITHM_STYLES[algorithm]
        x = group[rate_column].to_numpy(dtype=float)
        y = group["mean_annual_generation_gwh"].to_numpy(dtype=float) / 1000.0
        if len(group) > 1:
            legend_label = {
                "DQN": "DQN validation (3-seed mean)",
                "PPO": "PPO validation (3-seed mean)",
            }.get(algorithm, display_text(algorithm))
            ax.plot(
                x,
                y,
                label=legend_label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.0,
                linewidth=1.5,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )

    ax.axvline(
        80.0,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.2,
        label="80% candidate reference",
        zorder=1,
    )
    ax.axvline(
        90.0,
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.2,
        label="90% target reference",
        zorder=1,
    )

    all_rates = points[rate_column]
    ax.set_xlim(max(0.0, float(all_rates.min()) - 4.0), 100.5)
    ax.set_xlabel(f"{display_text(reservoir_name)} firm-power guarantee rate (%)", fontsize=LABEL_FT)
    ax.set_ylabel("Mean annual generation (TWh/yr)", fontsize=LABEL_FT)
    ax.tick_params(axis="both", labelsize=TICK_FT)
    ax.grid(True, color="0.82", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)
    if panel_label:
        ax.text(
            0.015,
            0.985,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )


def save_single_panel(points: pd.DataFrame, rate_column: str, reservoir_name: str, stem: str) -> None:
    with plt.rc_context(FONT_RC):
        fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=600)
        draw_tradeoff_panel(ax, points, rate_column, reservoir_name)
        ax.legend(loc="best", frameon=False, fontsize=LEGEND_FT, ncol=2)
        finalize_figure(fig)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(FIGURES_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def save_combined_panel(points: pd.DataFrame) -> None:
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=600, sharey=True)
        draw_tradeoff_panel(axes[0], points, "xw_guarantee_rate_pct", "Reservoir X", "(a)")
        draw_tradeoff_panel(axes[1], points, "nzd_guarantee_rate_pct", "Reservoir N", "(b)")
        axes[1].set_ylabel("")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=LEGEND_FT,
        )
        finalize_figure(fig)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
        stem = "fig02_generation_reliability_tradeoff_combined"
        fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(FIGURES_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without saving or plotting.")
    args = parser.parse_args()
    points = load_tradeoff_points()
    counts = points.groupby("algorithm").size().to_dict()
    print(f"Loaded observed points: {counts}")
    if args.check_only:
        print("Trade-off data checks passed. No files or figures were written.")
        print(points.groupby(["algorithm", "data_split"]).size().to_string())
        print(points.loc[points["algorithm"].isin(["DQN", "PPO"]), [
            "algorithm", "failure_penalty_gwh", "xw_guarantee_rate_pct", "nzd_guarantee_rate_pct",
            "mean_annual_generation_gwh", "data_split",
        ]].to_string(index=False))
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for obsolete in TABLES_DIR.glob("iso_validation_pareto_front_*.csv"):
        obsolete.unlink()

    nearest = build_nearest_reference_table(points)
    balanced = build_balanced_reference_table(points)
    points.to_csv(TABLES_DIR / "tradeoff_points.csv", index=False, encoding="utf-8-sig")
    nearest.to_csv(TABLES_DIR / "reference_nearest_points.csv", index=False, encoding="utf-8-sig")
    balanced.to_csv(TABLES_DIR / "reference_balanced_candidates.csv", index=False, encoding="utf-8-sig")

    save_single_panel(points, "xw_guarantee_rate_pct", "Reservoir X", "fig02_tradeoff_xw")
    save_single_panel(points, "nzd_guarantee_rate_pct", "Reservoir N", "fig02_tradeoff_nzd")
    save_single_panel(
        points,
        "min_guarantee_rate_pct",
        "Weakest-station min(XW, NZD)",
        "fig02_tradeoff_rmin_auxiliary",
    )
    save_combined_panel(points)

    counts = points.groupby("algorithm").size().to_dict()
    print(f"Loaded observed points: {counts}")
    print(
        "Data split: DDP/SDP use 400-year evaluation results; "
        "DQN uses 80-year best-checkpoint validation means across three seeds and eight penalties; "
        "ISO uses fixed-RF41 80-year validation results across eleven DDP teacher penalties; "
        "PPO uses 80-year validation means across three seeds and twelve penalties."
    )
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables: {TABLES_DIR}")
    print("\nNearest observed points to the 80% and 90% references:")
    print(nearest.to_string(index=False))
    print("\nBalanced candidates when both station-specific rates are considered:")
    print(balanced.to_string(index=False))


if __name__ == "__main__":
    main()
