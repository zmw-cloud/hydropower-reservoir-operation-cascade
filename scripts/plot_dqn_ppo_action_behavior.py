"""One 2x2 figure of executed target-level action adjustments: DQN versus PPO.

Reads only the selected DQN5000 and PPO6000 formal evaluation runs in the
existing comparison YAML. No directory discovery, training, or simulation.

Action definition: r1/r2_requested_target_level_m, the target passed to the
physical simulator AFTER DQN action masking or PPO feasible-interval mapping.
These are not network outputs and are not the realized end-of-month levels.
Saved action targets must agree with those simulator inputs, without subsequent
target clipping; otherwise stop rather than silently substitute another metric.

Within each seed, abs(diff(target)) is computed along the entire chronological
evaluation trajectory. Only its first month is excluded; December-to-January
transitions remain. No differences are taken between seeds or reservoirs.

ECDF: evaluate each seed's right-continuous ECDF on a shared exact support, then
average the three ECDFs with equal seed weight. Shading is their min-max range.
Seasonal: mean adjustment within each seed/calendar month, followed by the
equal-weight three-seed mean and min-max range. Group by the CURRENT month.

CSV aggregate rows are the arithmetic mean and sample SD (ddof=1) of the three
seed-level statistics, including seed medians/quantiles/maxima. They are NOT
quantiles of pooled samples. P90/P95 use linear sample-quantile interpolation.
Seed bands are descriptive variability, not confidence intervals. Adjustment
magnitude alone does not establish generation causality or action-grid adequacy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import os
import sys
from pathlib import Path
import zipfile

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cascade_reservoir_benchmark.comparison.presentation import (
    ACTION_FIGSIZE, FIGURE_DPI, FONT_FAMILY, FONT_RC,
    LABEL_FT, TICK_FT, LEGEND_FT, PANEL_FT, RESERVOIR_LABELS, finalize_figure,
)
FIGURE_STEM = "fig_rl_action_adjustment_comparison"
BAND_ALPHA = 0.12
LINE_WIDTH = 1.5
ALGORITHMS = ("DQN", "PPO")
SEEDS = (1, 42, 100)
RESERVOIRS = {"XW": "Xiaowan", "NZD": "Nuozhadu"}
STYLES = {
    "DQN": {"color": "#7564A8", "marker": "^"},
    "PPO": {"color": "#D98932", "marker": "v"},
}
FONT_RC = {**FONT_RC, "path.simplify": False}
METRICS = (
    "mean_abs_adjustment_m", "median_abs_adjustment_m",
    "p90_abs_adjustment_m", "p95_abs_adjustment_m", "maximum_abs_adjustment_m",
)
USED_COLUMNS = {
    "date", "target_level_1_m", "target_level_2_m",
    "r1_requested_target_level_m", "r2_requested_target_level_m",
    "r1_target_level_clipped", "r2_target_level_clipped",
    "selected_action_feasible",
    "continuous_xiaowan_lower_bound_m", "continuous_xiaowan_upper_bound_m",
    "continuous_nuozhadu_lower_bound_m", "continuous_nuozhadu_upper_bound_m",
}


@dataclass
class ActionRun:
    dates: np.ndarray
    current_months: np.ndarray
    adjustments: dict[str, np.ndarray]
    source: str


def native_path(path: Path) -> Path:
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path.absolute()))
    return path


def configure_times_new_roman(font_directory: Path) -> None:
    """Register supplied TTF/OTF fonts and prevent a silent serif fallback."""
    if font_directory.is_dir():
        for path in sorted(font_directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
    try:
        for weight in ("normal", "bold"):
            properties = font_manager.FontProperties(family=FONT_FAMILY, weight=weight)
            font_manager.findfont(properties, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is not available. Upload its regular and bold TTF "
            "files to assets/fonts/ under the plotting project, or provide "
            "--font-dir /path/to/fonts. The requested font will not be silently replaced."
        ) from exc


def nice_upper_limit(maximum: float) -> float:
    """Use a tidy upper limit with headroom, retaining all data and seed bands."""
    if maximum <= 0:
        return 1.0
    ticks = MaxNLocator(nbins=5, min_n_ticks=3).tick_values(0, maximum * 1.08)
    return float(ticks[-1])


def boolean_values(series: pd.Series, name: str) -> np.ndarray:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError(f"Missing/invalid Boolean values in {name}")
    return result.to_numpy(dtype=bool)


def numeric_values(raw: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(raw[name], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing/non-finite action values in {name}")
    return values


def load_action_runs(config_path: Path) -> tuple[dict, dict]:
    with native_path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    runs: dict[str, dict[int, ActionRun]] = {name: {} for name in ALGORITHMS}
    reference_dates = None
    for algorithm in ALGORITHMS:
        selection = config["selected_results"][algorithm]
        expected_penalty = {"DQN": 5000.0, "PPO": 6000.0}[algorithm]
        if float(selection["failure_penalty_gwh"]) != expected_penalty:
            raise ValueError(f"This figure requires the selected {algorithm} lambda={expected_penalty:g}")
        results = selection["results"]
        if len(results) != 3 or {int(row["seed"]) for row in results} != set(SEEDS):
            raise ValueError(f"{algorithm} requires exactly seeds 1, 42, 100")
        for result in results:
            seed = int(result["seed"])
            if "archive" in selection:
                archive_path = PROJECT_ROOT / selection["archive"]
                with zipfile.ZipFile(native_path(archive_path)) as archive:
                    raw = pd.read_csv(BytesIO(archive.read(result["trajectory"])),
                                      usecols=lambda name: name in USED_COLUMNS)
                    summary = json.loads(archive.read(result["summary"]).decode("utf-8-sig"))
                source = str(archive_path) + "::" + result["trajectory"]
            else:
                trajectory = PROJECT_ROOT / result["trajectory"]
                raw = pd.read_csv(native_path(trajectory), usecols=lambda name: name in USED_COLUMNS)
                with native_path(PROJECT_ROOT / result["summary"]).open(encoding="utf-8-sig") as handle:
                    summary = json.load(handle)
                source = str(trajectory)
            if (int(summary["seed"]) != seed or int(summary["months"]) != 4800
                    or float(summary["years"]) != 400.0 or len(raw) != 4800
                    or float(summary["firm_power_failure_penalty_gwh"]) != expected_penalty):
                raise ValueError(f"Wrong formal evaluation selection: {algorithm}, seed={seed}")
            if algorithm == "DQN" and (summary.get("checkpoint_role") != "final"
                                       or summary.get("evaluation_scenario") != "baseline"):
                raise ValueError("DQN action figure must use the formal final baseline evaluation")

            dates = raw["date"].astype(str).str.strip()
            parts = dates.str.extract(r"^(\d{4,})-(\d{2})-(\d{2})(?:[ T].*)?$")
            if parts.isna().any().any():
                raise ValueError(f"Invalid monthly dates: {source}")
            years, months, days = (parts[i].to_numpy(dtype=int) for i in range(3))
            serial = years * 12 + months
            if not (np.all((months >= 1) & (months <= 12)) and np.all(days == 1)
                    and np.all(np.diff(serial) == 1)):
                raise ValueError("Formal action dates must be unique, chronological and monthly continuous")
            dates_array = dates.to_numpy()
            if reference_dates is None:
                reference_dates = dates_array
            elif not np.array_equal(reference_dates, dates_array):
                raise ValueError("DQN/PPO seeds must use the same complete evaluation date sequence")

            if algorithm == "DQN" and "selected_action_feasible" in raw:
                if not boolean_values(raw["selected_action_feasible"], "selected_action_feasible").all():
                    raise ValueError("DQN includes a saved action not marked feasible after action masking")
            adjustments = {}
            for code, index, station in (("XW", 1, "xiaowan"), ("NZD", 2, "nuozhadu")):
                target = numeric_values(raw, f"r{index}_requested_target_level_m")
                selected_action = numeric_values(raw, f"target_level_{index}_m")
                if not np.allclose(target, selected_action, atol=1e-7, rtol=0):
                    raise ValueError("Saved selected action differs from simulator target; do not use raw outputs")
                clipped = boolean_values(raw[f"r{index}_target_level_clipped"], f"r{index}_target_level_clipped")
                if clipped.any():
                    raise ValueError(
                        f"{algorithm} {station} has targets clipped after action mapping. "
                        "Confirm an effective executed-target field before plotting; "
                        "do not substitute realized end-of-month levels."
                    )
                if algorithm == "PPO":
                    lower = numeric_values(raw, f"continuous_{station}_lower_bound_m")
                    upper = numeric_values(raw, f"continuous_{station}_upper_bound_m")
                    if np.any(target < lower - 1e-7) or np.any(target > upper + 1e-7):
                        raise ValueError("PPO mapped target is outside its saved feasible interval")
                adjustments[code] = np.abs(np.diff(target))
            runs[algorithm][seed] = ActionRun(dates_array[1:], months[1:], adjustments, source)
    return config, runs


def summarize(runs: dict) -> pd.DataFrame:
    records = []
    for algorithm in ALGORITHMS:
        for code, reservoir in RESERVOIRS.items():
            seed_statistics = []
            for seed in SEEDS:
                run = runs[algorithm][seed]
                values = run.adjustments[code]
                statistics = {
                    "mean_abs_adjustment_m": float(values.mean()),
                    "median_abs_adjustment_m": float(np.median(values)),
                    "p90_abs_adjustment_m": float(np.quantile(values, 0.90)),
                    "p95_abs_adjustment_m": float(np.quantile(values, 0.95)),
                    "maximum_abs_adjustment_m": float(values.max()),
                }
                seed_statistics.append([statistics[name] for name in METRICS])
                records.append({
                    "algorithm": algorithm, "reservoir": reservoir, "row_type": "seed",
                    "seed": str(seed), "seed_count": 1, "n_adjustments": len(values),
                    "first_adjustment_month": run.dates[0], "last_adjustment_month": run.dates[-1],
                    **statistics, "trajectory_source": run.source,
                })
            statistics_matrix = np.asarray(seed_statistics, dtype=float)
            for row_type, values in (
                ("3-seed mean", statistics_matrix.mean(axis=0)),
                ("3-seed SD (ddof=1)", statistics_matrix.std(axis=0, ddof=1)),
            ):
                records.append({
                    "algorithm": algorithm, "reservoir": reservoir, "row_type": row_type,
                    "seed": "", "seed_count": 3, "n_adjustments": "",
                    "first_adjustment_month": runs[algorithm][SEEDS[0]].dates[0],
                    "last_adjustment_month": runs[algorithm][SEEDS[0]].dates[-1],
                    **dict(zip(METRICS, values)), "trajectory_source": "mean/SD of three seed-level statistics",
                })
    return pd.DataFrame.from_records(records)


def ecdf_matrix(runs: dict, algorithm: str, code: str, grid: np.ndarray) -> np.ndarray:
    curves = []
    for seed in SEEDS:
        values = np.sort(runs[algorithm][seed].adjustments[code])
        curves.append(np.searchsorted(values, grid, side="right") / len(values))
    return np.asarray(curves)


def seasonal_matrix(runs: dict, algorithm: str, code: str) -> np.ndarray:
    values = []
    for seed in SEEDS:
        run = runs[algorithm][seed]
        values.append([run.adjustments[code][run.current_months == month].mean()
                       for month in range(1, 13)])
    return np.asarray(values)


def draw(runs: dict, output_directory: Path) -> None:
    all_values = [runs[algorithm][seed].adjustments[code]
                  for algorithm in ALGORITHMS for seed in SEEDS for code in RESERVOIRS]
    largest = max(float(values.max()) for values in all_values)
    x_max = nice_upper_limit(largest)
    monthly = {(algorithm, code): seasonal_matrix(runs, algorithm, code)
               for algorithm in ALGORITHMS for code in RESERVOIRS}
    seasonal_max = max(float(values.max()) for values in monthly.values())
    seasonal_y_max = nice_upper_limit(seasonal_max)
    month_names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    months = np.arange(1, 13)
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(2, 2, figsize=ACTION_FIGSIZE)
        for column, (code, reservoir) in enumerate(RESERVOIRS.items()):
            reservoir = RESERVOIR_LABELS[code]
            ax = axes[0, column]
            # Exact union support preserves zero adjustments and discrete jumps.
            grid = np.unique(np.concatenate([
                np.array([0.0, x_max]),
                *[runs[algorithm][seed].adjustments[code]
                  for algorithm in ALGORITHMS for seed in SEEDS],
            ]))
            for algorithm in ALGORITHMS:
                style = STYLES[algorithm]
                curves = ecdf_matrix(runs, algorithm, code, grid)
                ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0),
                                color=style["color"], alpha=BAND_ALPHA, step="post",
                                linewidth=0, zorder=1)
                ax.step(grid, curves.mean(axis=0), where="post", color=style["color"],
                        linewidth=LINE_WIDTH, label=algorithm, zorder=3)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
            ax.set_xlabel("Absolute target-level adjustment (m)", labelpad=6,
                          fontfamily=FONT_FAMILY)
            ax.set_ylabel("Cumulative probability", labelpad=7, fontfamily=FONT_FAMILY)
            ax.set_title(f"({'ab'[column]}) {reservoir}", loc="left", fontsize=PANEL_FT,
                         fontweight="bold", fontfamily=FONT_FAMILY, pad=10)

            ax = axes[1, column]
            for algorithm in ALGORITHMS:
                style = STYLES[algorithm]
                values = monthly[algorithm, code]
                ax.fill_between(months, values.min(axis=0), values.max(axis=0),
                                color=style["color"], alpha=BAND_ALPHA, linewidth=0, zorder=1)
                ax.plot(months, values.mean(axis=0), color=style["color"], linestyle="-",
                        marker=style["marker"], markersize=4, linewidth=LINE_WIDTH,
                        markeredgecolor="white", markeredgewidth=0.4,
                        label=algorithm, zorder=3)
            ax.set_xticks(months)
            ax.set_xticklabels(month_names, fontfamily=FONT_FAMILY)
            ax.set_xlim(0.7, 12.3)
            ax.set_ylim(0, seasonal_y_max)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.set_xlabel("Calendar month", labelpad=6, fontfamily=FONT_FAMILY)
            ax.set_ylabel("Mean absolute target-level\nadjustment (m)", labelpad=7,
                          fontfamily=FONT_FAMILY)
            ax.set_title(f"({'cd'[column]}) {reservoir} seasonal adjustment",
                         loc="left", fontsize=PANEL_FT, fontweight="bold",
                         fontfamily=FONT_FAMILY, pad=10)

        for ax in axes.flat:
            ax.set_facecolor("white")
            ax.grid(True, color="0.84", linewidth=0.6, alpha=0.65)
            ax.set_axisbelow(True)
            ax.minorticks_off()
            ax.tick_params(axis="both", which="major", labelsize=TICK_FT,
                           direction="out", length=3.5, width=0.8, pad=4)
            for spine in ax.spines.values():
                spine.set_color("0.35")
                spine.set_linewidth(0.8)
            ax.xaxis.label.set_fontsize(LABEL_FT)
            ax.yaxis.label.set_fontsize(LABEL_FT)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(FONT_FAMILY)
            ax.xaxis.get_offset_text().set_fontfamily(FONT_FAMILY)
            ax.yaxis.get_offset_text().set_fontfamily(FONT_FAMILY)
        handles = [Line2D([0], [0], color=STYLES[name]["color"], linewidth=LINE_WIDTH,
                          label=f"{name} (3-seed mean)") for name in ALGORITHMS]
        handles.append(Patch(facecolor="#777777", alpha=BAND_ALPHA, edgecolor="none",
                             label="Shading: seed range (min–max)"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncol=3, frameon=False, prop={"family": FONT_FAMILY, "size": LEGEND_FT},
                   handlelength=2.4, handletextpad=0.65, columnspacing=1.6, borderaxespad=0)
        for text in fig.findobj(match=Text):
            text.set_fontfamily(FONT_FAMILY)
        finalize_figure(fig)
        fig.tight_layout(rect=(0.01, 0.01, 0.995, 0.93), h_pad=2.2, w_pad=2.2)
        for extension in ("pdf", "png", "svg"):
            fig.savefig(output_directory / f"{FIGURE_STEM}.{extension}",
                        dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/comparison_ddp_sdp_iso_dqn_ppo.yaml")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for this figure and its summary CSV; defaults to existing formal figures directory")
    parser.add_argument("--font-dir", type=Path, default=PROJECT_ROOT / "assets/fonts",
                        help="Optional directory containing Times New Roman TTF/OTF files")
    args = parser.parse_args(argv)
    font_directory = args.font_dir if args.font_dir.is_absolute() else PROJECT_ROOT / args.font_dir
    configure_times_new_roman(font_directory)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config, runs = load_action_runs(config_path)
    if args.output_dir is None:
        output = PROJECT_ROOT / config["comparison"]["output_directory"] / "figures"
    else:
        output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summarize(runs).to_csv(output / "dqn_ppo_action_behavior_summary.csv", index=False, encoding="utf-8-sig")
    draw(runs, output)
    print("DQN_PPO_ACTION_BEHAVIOR_PLOT_OK")
    print("Only the selected formal DQN/PPO evaluation trajectories were read; no policy simulation was run.")
    print("Per-seed adjustments: 4799; first month excluded, calendar-year transitions retained.")
    print("Lines: equal-weight 3-seed means; shading: seed min-max range, not confidence intervals.")
    print(f"OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main(None if "__file__" in globals() else [])
