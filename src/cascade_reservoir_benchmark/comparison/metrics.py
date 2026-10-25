from __future__ import annotations

"""Metrics for the selected DDP--SDP--ISO comparison."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ComparisonBundle


@dataclass
class AnalysisTables:
    selected_performance: pd.DataFrame
    deficit_statistics: pd.DataFrame
    deficit_duration: pd.DataFrame
    monthly_power_statistics: pd.DataFrame
    monthly_level_statistics: pd.DataFrame
    spill_statistics: pd.DataFrame
    monthly_spill_statistics: pd.DataFrame
    annual_generation_by_year: pd.DataFrame
    annual_generation_statistics: pd.DataFrame
    summary_algorithms: pd.DataFrame


RESERVOIR_COLUMNS = {
    "XW": {
        "power": "xw_power_mw",
        "energy": "xw_energy_gwh",
        "level": "xw_end_level_m",
        "spill": "xw_spill_m3s",
        "turbine": "xw_turbine_release_m3s",
        "release": "xw_total_release_m3s",
        "deficit_energy": "xw_deficit_energy_gwh",
        "failure": "xw_failure",
    },
    "NZD": {
        "power": "nzd_power_mw",
        "energy": "nzd_energy_gwh",
        "level": "nzd_end_level_m",
        "spill": "nzd_spill_m3s",
        "turbine": "nzd_turbine_release_m3s",
        "release": "nzd_total_release_m3s",
        "deficit_energy": "nzd_deficit_energy_gwh",
        "failure": "nzd_failure",
    },
}


def build_selected_performance(bundle: ComparisonBundle) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm, frame in bundle.frames.items():
        seed_count = frame["seed"].nunique()
        n_months = len(frame) // seed_count
        n_years = frame["year"].nunique()
        xw_rate = 1.0 - float(frame["xw_failure"].mean())
        nzd_rate = 1.0 - float(frame["nzd_failure"].mean())
        total_generation = float(frame["generation_energy_gwh"].sum()) / seed_count
        rows.append(
            {
                "algorithm": algorithm,
                "candidate_id": bundle.selections[algorithm]["candidate_id"],
                "failure_penalty_gwh": float(bundle.selections[algorithm]["failure_penalty_gwh"]),
                "months": n_months,
                "seed_count": seed_count,
                "pooled_month_samples": len(frame),
                "years": n_years,
                "total_generation_gwh": total_generation,
                "mean_annual_generation_gwh": total_generation / n_years,
                "xw_guarantee_rate": xw_rate,
                "xw_guarantee_rate_pct": 100.0 * xw_rate,
                "nzd_guarantee_rate": nzd_rate,
                "nzd_guarantee_rate_pct": 100.0 * nzd_rate,
                "min_guarantee_rate": min(xw_rate, nzd_rate),
                "min_guarantee_rate_pct": 100.0 * min(xw_rate, nzd_rate),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_deficit_tables(bundle: ComparisonBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics: list[dict] = []
    duration_rows: list[dict] = []

    for algorithm, frame in bundle.frames.items():
        for reservoir, columns in RESERVOIR_COLUMNS.items():
            spec = bundle.reservoir_specs[reservoir]
            deficit = (spec.firm_power_mw - frame[columns["power"]]).clip(lower=0.0)
            positive = deficit > 0.0
            positive_values = deficit.loc[positive]
            seed_count = frame["seed"].nunique()
            deficit_months = float(positive.sum()) / seed_count
            n_months = len(frame) // seed_count
            guarantee_rate = 1.0 - deficit_months / n_months

            statistics.append(
                {
                    "algorithm": algorithm,
                    "candidate_id": bundle.selections[algorithm]["candidate_id"],
                    "reservoir": reservoir,
                    "firm_power_mw": spec.firm_power_mw,
                    "months": n_months,
                    "deficit_months": deficit_months,
                    "deficit_month_percentage": 100.0 * deficit_months / n_months,
                    "guarantee_rate": guarantee_rate,
                    "guarantee_rate_pct": 100.0 * guarantee_rate,
                    "mean_deficit_mw_deficit_months_only": (
                        float(positive_values.mean()) if deficit_months else 0.0
                    ),
                    "median_deficit_mw_deficit_months_only": (
                        float(positive_values.median()) if deficit_months else 0.0
                    ),
                    "max_deficit_mw": float(positive_values.max()) if deficit_months else 0.0,
                    "cumulative_deficit_energy_gwh": float(frame[columns["deficit_energy"]].sum()) / seed_count,
                }
            )

            sorted_values = np.sort(positive_values.to_numpy(dtype=float))[::-1]
            for rank, value in enumerate(sorted_values, start=1):
                duration_rows.append(
                    {
                        "algorithm": algorithm,
                        "reservoir": reservoir,
                        "rank": rank,
                        "percentage_of_deficit_months": 100.0 * rank / len(sorted_values),
                        "deficit_mw": float(value),
                    }
                )

    return pd.DataFrame.from_records(statistics), pd.DataFrame.from_records(duration_rows)


def build_monthly_level_statistics(bundle: ComparisonBundle) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm, frame in bundle.frames.items():
        for reservoir, columns in RESERVOIR_COLUMNS.items():
            for month, values in frame.groupby("month")[columns["level"]]:
                rows.append(
                    {
                        "algorithm": algorithm,
                        "reservoir": reservoir,
                        "month": int(month),
                        "sample_count": int(values.count()),
                        "mean_level_m": float(values.mean()),
                        "p10_level_m": float(values.quantile(0.10)),
                        "p25_level_m": float(values.quantile(0.25)),
                        "median_level_m": float(values.median()),
                        "p75_level_m": float(values.quantile(0.75)),
                        "p90_level_m": float(values.quantile(0.90)),
                    }
                )
    return pd.DataFrame.from_records(rows).sort_values(["reservoir", "algorithm", "month"])


def build_monthly_power_statistics(bundle: ComparisonBundle) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm, frame in bundle.frames.items():
        for reservoir, columns in RESERVOIR_COLUMNS.items():
            spec = bundle.reservoir_specs[reservoir]
            for month, values in frame.groupby("month")[columns["power"]]:
                rows.append(
                    {
                        "algorithm": algorithm,
                        "reservoir": reservoir,
                        "month": int(month),
                        "sample_count": int(values.count()),
                        "firm_power_mw": spec.firm_power_mw,
                        "mean_power_mw": float(values.mean()),
                        "min_power_mw": float(values.min()),
                        "p10_power_mw": float(values.quantile(0.10)),
                        "p25_power_mw": float(values.quantile(0.25)),
                        "median_power_mw": float(values.median()),
                        "p75_power_mw": float(values.quantile(0.75)),
                        "p90_power_mw": float(values.quantile(0.90)),
                        "max_power_mw": float(values.max()),
                        "monthly_guarantee_rate_pct": 100.0 * float((values >= spec.firm_power_mw).mean()),
                    }
                )
    return pd.DataFrame.from_records(rows).sort_values(["reservoir", "algorithm", "month"])


def build_spill_statistics(bundle: ComparisonBundle) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm, frame in bundle.frames.items():
        seed_count = frame["seed"].nunique()
        n_months = len(frame) // seed_count
        station_values: dict[str, dict[str, float]] = {}
        for reservoir, columns in RESERVOIR_COLUMNS.items():
            seconds = frame["month_seconds"].to_numpy(dtype=float)
            spill_volume = float(np.sum(frame[columns["spill"]].to_numpy(dtype=float) * seconds)) / seed_count
            turbine_volume = float(np.sum(frame[columns["turbine"]].to_numpy(dtype=float) * seconds)) / seed_count
            release_volume = float(np.sum(frame[columns["release"]].to_numpy(dtype=float) * seconds)) / seed_count
            spill_months = float((frame[columns["spill"]] > 0.0).sum()) / seed_count
            station_values[reservoir] = {
                "spill_volume_m3": spill_volume,
                "turbine_volume_m3": turbine_volume,
                "release_volume_m3": release_volume,
            }
            rows.append(
                {
                    "algorithm": algorithm,
                    "candidate_id": bundle.selections[algorithm]["candidate_id"],
                    "reservoir": reservoir,
                    "months": n_months,
                    "spill_months": spill_months,
                    "spill_month_percentage": 100.0 * spill_months / n_months,
                    "cumulative_spill_volume_m3": spill_volume,
                    "cumulative_spill_volume_1e8m3": spill_volume / 1e8,
                    "cumulative_turbine_volume_m3": turbine_volume,
                    "cumulative_turbine_volume_1e8m3": turbine_volume / 1e8,
                    "cumulative_total_release_volume_m3": release_volume,
                    "spill_to_total_release_ratio": (
                        spill_volume / release_volume if release_volume > 0.0 else np.nan
                    ),
                    "spill_ratio_definition": "spill_volume / total_release_volume",
                }
            )

        cascade_spill = sum(value["spill_volume_m3"] for value in station_values.values())
        cascade_turbine = sum(value["turbine_volume_m3"] for value in station_values.values())
        cascade_release = sum(value["release_volume_m3"] for value in station_values.values())
        cascade_spill_months = float(((frame["xw_spill_m3s"] > 0.0) | (frame["nzd_spill_m3s"] > 0.0)).sum()) / seed_count
        rows.append(
            {
                "algorithm": algorithm,
                "candidate_id": bundle.selections[algorithm]["candidate_id"],
                "reservoir": "Cascade",
                "months": n_months,
                "spill_months": cascade_spill_months,
                "spill_month_percentage": 100.0 * cascade_spill_months / n_months,
                "cumulative_spill_volume_m3": cascade_spill,
                "cumulative_spill_volume_1e8m3": cascade_spill / 1e8,
                "cumulative_turbine_volume_m3": cascade_turbine,
                "cumulative_turbine_volume_1e8m3": cascade_turbine / 1e8,
                "cumulative_total_release_volume_m3": cascade_release,
                "spill_to_total_release_ratio": (
                    cascade_spill / cascade_release if cascade_release > 0.0 else np.nan
                ),
                "spill_ratio_definition": (
                    "sum(station spill volumes) / sum(station total-release volumes)"
                ),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_monthly_spill_statistics(bundle: ComparisonBundle) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm, frame in bundle.frames.items():
        spill_series = {
            "XW": frame["xw_spill_m3s"],
            "NZD": frame["nzd_spill_m3s"],
            "Cascade": frame["xw_spill_m3s"] + frame["nzd_spill_m3s"],
        }
        spill_volume = {
            "XW": frame["xw_spill_m3s"] * frame["month_seconds"] / 1e8,
            "NZD": frame["nzd_spill_m3s"] * frame["month_seconds"] / 1e8,
            "Cascade": (
                frame["xw_spill_m3s"] + frame["nzd_spill_m3s"]
            )
            * frame["month_seconds"]
            / 1e8,
        }

        for reservoir in ("XW", "NZD", "Cascade"):
            working = pd.DataFrame(
                {
                    "month": frame["month"],
                    "spill_discharge_m3s": spill_series[reservoir],
                    "spill_volume_1e8m3": spill_volume[reservoir],
                }
            )
            for month, group in working.groupby("month", sort=True):
                discharge = group["spill_discharge_m3s"]
                volume = group["spill_volume_1e8m3"]
                rows.append(
                    {
                        "algorithm": algorithm,
                        "reservoir": reservoir,
                        "month": int(month),
                        "sample_count": int(discharge.count()),
                        "spill_years": float((discharge > 0.0).sum()) / frame["seed"].nunique(),
                        "spill_occurrence_rate_pct": 100.0 * float((discharge > 0.0).mean()),
                        "mean_spill_discharge_m3s": float(discharge.mean()),
                        "median_spill_discharge_m3s": float(discharge.median()),
                        "p75_spill_discharge_m3s": float(discharge.quantile(0.75)),
                        "p90_spill_discharge_m3s": float(discharge.quantile(0.90)),
                        "p95_spill_discharge_m3s": float(discharge.quantile(0.95)),
                        "max_spill_discharge_m3s": float(discharge.max()),
                        "mean_monthly_spill_volume_1e8m3": float(volume.mean()),
                        "total_400y_spill_volume_1e8m3": float(volume.sum()) / frame["seed"].nunique(),
                    }
                )

    return pd.DataFrame.from_records(rows).sort_values(["reservoir", "algorithm", "month"])


def build_annual_generation_tables(bundle: ComparisonBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows: list[dict] = []
    for algorithm, frame in bundle.frames.items():
        grouped = frame.groupby(["seed", "year"], sort=True).agg(
            xw_generation_gwh=("xw_energy_gwh", "sum"),
            nzd_generation_gwh=("nzd_energy_gwh", "sum"),
            cascade_generation_gwh=("generation_energy_gwh", "sum"),
        )
        for (seed, year), row in grouped.iterrows():
            annual_rows.append(
                {
                    "algorithm": algorithm,
                    "year": int(year),
                    "seed": int(seed),
                    "xw_generation_gwh": float(row["xw_generation_gwh"]),
                    "nzd_generation_gwh": float(row["nzd_generation_gwh"]),
                    "cascade_generation_gwh": float(row["cascade_generation_gwh"]),
                }
            )
    annual = pd.DataFrame.from_records(annual_rows)

    statistic_rows: list[dict] = []
    generation_columns = {
        "XW": "xw_generation_gwh",
        "NZD": "nzd_generation_gwh",
        "Cascade": "cascade_generation_gwh",
    }
    for algorithm, group in annual.groupby("algorithm", sort=False):
        for reservoir, column in generation_columns.items():
            values = group[column]
            statistic_rows.append(
                {
                    "algorithm": algorithm,
                    "reservoir": reservoir,
                    "annual_samples": int(values.count()),
                    "mean_annual_generation_gwh": float(values.mean()),
                    "median_annual_generation_gwh": float(values.median()),
                    "std_annual_generation_gwh": float(values.std(ddof=1)),
                    "min_annual_generation_gwh": float(values.min()),
                    "p10_annual_generation_gwh": float(values.quantile(0.10)),
                    "p25_annual_generation_gwh": float(values.quantile(0.25)),
                    "p75_annual_generation_gwh": float(values.quantile(0.75)),
                    "p90_annual_generation_gwh": float(values.quantile(0.90)),
                    "max_annual_generation_gwh": float(values.max()),
                }
            )
    statistics = pd.DataFrame.from_records(statistic_rows)
    return annual, statistics


def build_summary_table(
    bundle: ComparisonBundle,
    performance: pd.DataFrame,
    deficit: pd.DataFrame,
    spill: pd.DataFrame,
    annual_statistics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm in bundle.algorithms:
        perf = performance.loc[performance["algorithm"] == algorithm].iloc[0]
        xw_def = deficit.loc[
            (deficit["algorithm"] == algorithm) & (deficit["reservoir"] == "XW")
        ].iloc[0]
        nzd_def = deficit.loc[
            (deficit["algorithm"] == algorithm) & (deficit["reservoir"] == "NZD")
        ].iloc[0]
        xw_spill = spill.loc[
            (spill["algorithm"] == algorithm) & (spill["reservoir"] == "XW")
        ].iloc[0]
        nzd_spill = spill.loc[
            (spill["algorithm"] == algorithm) & (spill["reservoir"] == "NZD")
        ].iloc[0]
        cascade_spill = spill.loc[
            (spill["algorithm"] == algorithm) & (spill["reservoir"] == "Cascade")
        ].iloc[0]
        annual = annual_statistics.loc[
            (annual_statistics["algorithm"] == algorithm)
            & (annual_statistics["reservoir"] == "Cascade")
        ].iloc[0]
        frame = bundle.frames[algorithm]

        rows.append(
            {
                "algorithm": algorithm,
                "candidate_id": perf["candidate_id"],
                "failure_penalty_gwh": perf["failure_penalty_gwh"],
                "months": int(perf["months"]),
                "years": int(perf["years"]),
                "seed_count": int(perf["seed_count"]),
                "total_generation_GWh": perf["total_generation_gwh"],
                "mean_annual_generation_GWh": annual["mean_annual_generation_gwh"],
                "std_annual_generation_GWh": annual["std_annual_generation_gwh"],
                "XW_guarantee_rate": perf["xw_guarantee_rate"],
                "NZD_guarantee_rate": perf["nzd_guarantee_rate"],
                "min_guarantee_rate": perf["min_guarantee_rate"],
                "XW_guarantee_rate_pct": perf["xw_guarantee_rate_pct"],
                "NZD_guarantee_rate_pct": perf["nzd_guarantee_rate_pct"],
                "min_guarantee_rate_pct": perf["min_guarantee_rate_pct"],
                "XW_deficit_months": xw_def["deficit_months"],
                "NZD_deficit_months": nzd_def["deficit_months"],
                "XW_mean_deficit_MW": xw_def["mean_deficit_mw_deficit_months_only"],
                "NZD_mean_deficit_MW": nzd_def["mean_deficit_mw_deficit_months_only"],
                "XW_max_deficit_MW": xw_def["max_deficit_mw"],
                "NZD_max_deficit_MW": nzd_def["max_deficit_mw"],
                "XW_deficit_energy_GWh": xw_def["cumulative_deficit_energy_gwh"],
                "NZD_deficit_energy_GWh": nzd_def["cumulative_deficit_energy_gwh"],
                "XW_total_spill_m3": xw_spill["cumulative_spill_volume_m3"],
                "NZD_total_spill_m3": nzd_spill["cumulative_spill_volume_m3"],
                "cascade_total_spill_m3": cascade_spill["cumulative_spill_volume_m3"],
                "XW_spill_months": xw_spill["spill_months"],
                "NZD_spill_months": nzd_spill["spill_months"],
                "XW_spill_to_total_release_ratio": xw_spill["spill_to_total_release_ratio"],
                "NZD_spill_to_total_release_ratio": nzd_spill["spill_to_total_release_ratio"],
                "final_XW_level_m": float(frame.groupby("seed").tail(1)["xw_end_level_m"].mean()),
                "final_NZD_level_m": float(frame.groupby("seed").tail(1)["nzd_end_level_m"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_analysis_tables(bundle: ComparisonBundle) -> AnalysisTables:
    performance = build_selected_performance(bundle)
    deficit_statistics, deficit_duration = build_deficit_tables(bundle)
    monthly_power = build_monthly_power_statistics(bundle)
    monthly_levels = build_monthly_level_statistics(bundle)
    spill = build_spill_statistics(bundle)
    monthly_spill = build_monthly_spill_statistics(bundle)
    annual, annual_statistics = build_annual_generation_tables(bundle)
    summary = build_summary_table(
        bundle,
        performance,
        deficit_statistics,
        spill,
        annual_statistics,
    )
    return AnalysisTables(
        selected_performance=performance,
        deficit_statistics=deficit_statistics,
        deficit_duration=deficit_duration,
        monthly_power_statistics=monthly_power,
        monthly_level_statistics=monthly_levels,
        spill_statistics=spill,
        monthly_spill_statistics=monthly_spill,
        annual_generation_by_year=annual,
        annual_generation_statistics=annual_statistics,
        summary_algorithms=summary,
    )
