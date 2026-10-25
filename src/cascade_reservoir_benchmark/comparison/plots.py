from __future__ import annotations

"""Publication-style plots for selected cascade-operation results."""

from pathlib import Path
import json

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import ComparisonBundle
from .metrics import AnalysisTables, RESERVOIR_COLUMNS
from .presentation import (
    FONT_RC, LABEL_FT, TICK_FT, LEGEND_FT, RESERVOIR_LABELS,
    display_text, finalize_figure,
)


ALGORITHM_STYLES = {
    "DDP": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "SDP": {"color": "#2ca02c", "marker": "s", "linestyle": "-"},
    "ISO": {"color": "#d62728", "marker": "D", "linestyle": "-"},
    "DQN": {"color": "#9467bd", "marker": "^", "linestyle": "-"},
    "PPO": {"color": "#ff7f0e", "marker": "v", "linestyle": "-"},
}


def _prepare_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="0.84", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=TICK_FT)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def _save(fig: plt.Figure, figures_dir: Path, stem: str, dpi: int) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    fig.savefig(figures_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figures_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_selected_generation_reliability(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
) -> None:
    performance = tables.selected_performance
    panels = [
        ("xw_guarantee_rate_pct", "Reservoir X guarantee rate (%)", "(a)"),
        ("nzd_guarantee_rate_pct", "Reservoir N guarantee rate (%)", "(b)"),
    ]
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), dpi=bundle.figure_dpi, sharex=True)
        for ax, (rate_column, ylabel, panel) in zip(axes, panels):
            for _, row in performance.iterrows():
                algorithm = str(row["algorithm"])
                style = ALGORITHM_STYLES[algorithm]
                x = float(row["mean_annual_generation_gwh"]) / 1000.0
                y = float(row[rate_column])
                ax.scatter(
                    x,
                    y,
                    s=62,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.6,
                    label=algorithm,
                    zorder=4,
                )
                ax.annotate(
                    algorithm,
                    (x, y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=LEGEND_FT,
                    color=style["color"],
                )

            for reference in bundle.reference_rates_pct:
                if np.isclose(reference, 80.0):
                    label = "80% comparison reference"
                    color, linestyle = "#7f7f7f", ":"
                else:
                    label = f"{reference:g}% target reference"
                    color, linestyle = "#4d4d4d", "--"
                ax.axhline(reference, color=color, linestyle=linestyle, linewidth=1.2, label=label)

            ax.set_xlabel("Mean annual cascade generation (TWh/yr)", fontsize=LABEL_FT)
            ax.set_ylabel(ylabel, fontsize=LABEL_FT)
            ax.set_ylim(max(0.0, float(performance[rate_column].min()) - 6.0), 100.5)
            _prepare_axes(ax)
            _panel_label(ax, panel)

        handles, labels = axes[0].get_legend_handles_labels()
        reference_handles = handles[-len(bundle.reference_rates_pct) :]
        reference_labels = labels[-len(bundle.reference_rates_pct) :]
        fig.legend(
            reference_handles,
            reference_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(reference_handles),
            frameon=False,
            fontsize=LEGEND_FT,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        _save(fig, figures_dir, "fig01_selected_generation_reliability", bundle.figure_dpi)


def plot_power_duration(bundle: ComparisonBundle, figures_dir: Path) -> None:
    panels = [("XW", "(a)"), ("NZD", "(b)")]
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), dpi=bundle.figure_dpi, sharex=True)
        for ax, (reservoir, panel) in zip(axes, panels):
            columns = RESERVOIR_COLUMNS[reservoir]
            spec = bundle.reservoir_specs[reservoir]
            for algorithm, frame in bundle.frames.items():
                style = ALGORITHM_STYLES[algorithm]
                values = np.sort(frame[columns["power"]].to_numpy(dtype=float))[::-1]
                exceedance = 100.0 * np.arange(1, len(values) + 1) / len(values)
                ax.plot(
                    exceedance,
                    values,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    label=algorithm,
                )
            ax.axhline(
                spec.firm_power_mw,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=f"Firm power ({spec.firm_power_mw:g} MW)",
            )
            ax.set_xlabel("Exceedance percentage of months (%)", fontsize=LABEL_FT)
            ax.set_ylabel(f"{RESERVOIR_LABELS[reservoir]} power (MW)", fontsize=LABEL_FT)
            ax.set_xlim(0.0, 100.0)
            _prepare_axes(ax)
            _panel_label(ax, panel)
            ax.legend(frameon=False, fontsize=LEGEND_FT, loc="best")
        fig.tight_layout()
        _save(fig, figures_dir, "fig03_power_duration", bundle.figure_dpi)


def _draw_grouped_monthly_boxplots(
    ax: plt.Axes,
    bundle: ComparisonBundle,
    reservoir: str,
    value_column: str,
    value_scale: float = 1.0,
    show_xlabel: bool = True,
) -> None:
    months = np.arange(1, 13, dtype=float)
    step = 0.18
    offsets = (np.arange(len(bundle.algorithms)) - (len(bundle.algorithms) - 1) / 2.0) * step
    width = 0.15

    for offset, algorithm in zip(offsets, bundle.algorithms):
        frame = bundle.frames[algorithm]
        datasets = [
            frame.loc[frame["month"] == month, value_column].to_numpy(dtype=float) * value_scale
            for month in range(1, 13)
        ]
        positions = months + offset
        color = ALGORITHM_STYLES[algorithm]["color"]
        result = ax.boxplot(
            datasets,
            positions=positions,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 0.85},
            whiskerprops={"color": "black", "linewidth": 0.65},
            capprops={"color": "black", "linewidth": 0.65},
            flierprops={
                "marker": "o",
                "markersize": 1.7,
                "alpha": 0.34,
                "markerfacecolor": color,
                "markeredgecolor": color,
            },
        )
        for box in result["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.58)
            box.set_edgecolor("black")
            box.set_linewidth(0.7)

        means = np.array([np.mean(values) for values in datasets], dtype=float)
        ax.scatter(
            positions,
            means,
            marker="o",
            s=10,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=5,
        )

    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(months)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if show_xlabel:
        ax.set_xticklabels(month_labels)
        ax.set_xlabel("Calendar month", fontsize=LABEL_FT)
    else:
        ax.set_xticklabels([])
    _prepare_axes(ax)


def plot_monthly_power_boxplots(bundle: ComparisonBundle, figures_dir: Path) -> None:
    panels = [("XW", "xw_power_mw", "(a)"), ("NZD", "nzd_power_mw", "(b)")]
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.4), dpi=bundle.figure_dpi, sharex=True)
        for index, (ax, (reservoir, column, panel)) in enumerate(zip(axes, panels)):
            spec = bundle.reservoir_specs[reservoir]
            _draw_grouped_monthly_boxplots(
                ax,
                bundle,
                reservoir,
                column,
                value_scale=1.0 / 1000.0,
                show_xlabel=index == len(panels) - 1,
            )
            ax.axhline(
                spec.firm_power_mw / 1000.0,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.1,
                label=f"Firm power ({spec.firm_power_mw / 1000.0:g} GW)",
            )
            ax.set_ylabel(f"{RESERVOIR_LABELS[reservoir]} power (GW)", fontsize=LABEL_FT)
            _panel_label(ax, panel)

        legend_handles = [
            Patch(
                facecolor=ALGORITHM_STYLES[algorithm]["color"],
                edgecolor="black",
                alpha=0.58,
                label=algorithm,
            )
            for algorithm in bundle.algorithms
        ]
        legend_handles.extend(
            [
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="Mean"),
                Line2D([], [], color="#2ca02c", linestyle="--", label="Firm-power requirement"),
            ]
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=LEGEND_FT,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        _save(fig, figures_dir, "fig03_monthly_power_boxplots", bundle.figure_dpi)


def plot_deficit_duration(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
) -> None:
    duration = tables.deficit_duration
    panels = [("XW", "Reservoir X", "(a)"), ("NZD", "Reservoir N", "(b)")]
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), dpi=bundle.figure_dpi, sharex=True)
        for ax, (reservoir, display_name, panel) in zip(axes, panels):
            for algorithm in bundle.algorithms:
                subset = duration.loc[
                    (duration["algorithm"] == algorithm) & (duration["reservoir"] == reservoir)
                ]
                if subset.empty:
                    continue
                style = ALGORITHM_STYLES[algorithm]
                ax.plot(
                    subset["percentage_of_deficit_months"],
                    subset["deficit_mw"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    label=algorithm,
                )
            ax.set_xlabel("Percentage of deficit months (%)", fontsize=LABEL_FT)
            ax.set_ylabel(f"{display_name} power deficit (MW)", fontsize=LABEL_FT)
            ax.set_xlim(0.0, 100.0)
            _prepare_axes(ax)
            _panel_label(ax, panel)
            ax.legend(frameon=False, fontsize=LEGEND_FT)
        fig.tight_layout()
        _save(fig, figures_dir, "fig04_deficit_duration", bundle.figure_dpi)


def _plot_monthly_levels(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
    central_column: str,
    central_label: str,
    stem: str,
) -> None:
    statistics = tables.monthly_level_statistics
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    panels = [("XW", "(a)"), ("NZD", "(b)")]

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=bundle.figure_dpi, sharex=True)
        for ax, (reservoir, panel) in zip(axes, panels):
            spec = bundle.reservoir_specs[reservoir]
            for algorithm in bundle.algorithms:
                subset = statistics.loc[
                    (statistics["algorithm"] == algorithm)
                    & (statistics["reservoir"] == reservoir)
                ].sort_values("month")
                style = ALGORITHM_STYLES[algorithm]
                x = subset["month"].to_numpy(dtype=float)
                central = subset[central_column].to_numpy(dtype=float)
                p10 = subset["p10_level_m"].to_numpy(dtype=float)
                p90 = subset["p90_level_m"].to_numpy(dtype=float)
                ax.fill_between(x, p10, p90, color=style["color"], alpha=0.10, linewidth=0.0)
                ax.plot(
                    x,
                    central,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.8,
                    linestyle=style["linestyle"],
                    linewidth=1.45,
                    label=f"{display_text(algorithm)} {central_label} (P10–P90)",
                )

            ax.axhline(
                spec.dead_level_m,
                color="#6b6b6b",
                linestyle=":",
                linewidth=1.1,
                label="Dead water level",
            )
            ax.axhline(
                spec.normal_level_m,
                color="#111111",
                linestyle="--",
                linewidth=1.1,
                label="Normal water level",
            )
            ax.plot(
                months,
                spec.monthly_max_target_levels_m,
                color="#e377c2",
                linestyle="-.",
                linewidth=1.25,
                drawstyle="steps-mid",
                label="Monthly maximum target level",
            )
            ax.set_xticks(months)
            ax.set_xticklabels(month_labels, rotation=0)
            ax.set_xlabel("Calendar month", fontsize=LABEL_FT)
            ax.set_ylabel(f"{RESERVOIR_LABELS[reservoir]} end-of-month level (m)", fontsize=LABEL_FT)
            ax.set_xlim(1.0, 12.0)
            _prepare_axes(ax)
            _panel_label(ax, panel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4 if len(handles) > 6 else 3,
            frameon=False,
            fontsize=8.8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
        _save(fig, figures_dir, stem, bundle.figure_dpi)


def plot_monthly_levels(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
) -> None:
    _plot_monthly_levels(
        bundle,
        tables,
        figures_dir,
        central_column="median_level_m",
        central_label="median",
        stem="fig05_monthly_level_statistics",
    )
    _plot_monthly_levels(
        bundle,
        tables,
        figures_dir,
        central_column="mean_level_m",
        central_label="mean",
        stem="fig05_monthly_level_mean_p10_p90",
    )


def plot_spill_comparison(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
) -> None:
    spill = tables.spill_statistics
    algorithms = bundle.algorithms
    reservoirs = ["XW", "NZD", "Cascade"]
    colors = {"XW": "#4c78a8", "NZD": "#f58518", "Cascade": "#7f7f7f"}
    labels = {"XW": "Reservoir X", "NZD": "Reservoir N", "Cascade": "Cascade total"}
    x = np.arange(len(algorithms), dtype=float)
    width = 0.24

    with plt.rc_context(FONT_RC):
        fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=bundle.figure_dpi)
        for index, reservoir in enumerate(reservoirs):
            values = []
            for algorithm in algorithms:
                row = spill.loc[
                    (spill["algorithm"] == algorithm) & (spill["reservoir"] == reservoir),
                    "cumulative_spill_volume_1e8m3",
                ]
                values.append(float(row.iloc[0]))
            bars = ax.bar(
                x + (index - 1) * width,
                values,
                width=width,
                color=colors[reservoir],
                label=labels[reservoir],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_xlabel("Algorithm", fontsize=LABEL_FT)
        ax.set_ylabel(r"Cumulative spill volume ($10^8$ m$^3$)", fontsize=LABEL_FT)
        maximum_value = float(spill["cumulative_spill_volume_1e8m3"].max())
        ax.set_ylim(0.0, maximum_value * 1.18 if maximum_value > 0.0 else 1.0)
        _prepare_axes(ax)
        ax.legend(frameon=False, fontsize=LEGEND_FT, ncol=3, loc="upper center")
        fig.tight_layout()
        _save(fig, figures_dir, "fig06_spill_comparison", bundle.figure_dpi)


def plot_monthly_mean_spill_volume(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
) -> None:
    statistics = tables.monthly_spill_statistics
    panels = [
        ("XW", "Reservoir X", "(a)"),
        ("NZD", "Reservoir N", "(b)"),
        ("Cascade", "Cascade total", "(c)"),
    ]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.4), dpi=bundle.figure_dpi, sharex=True)
        for ax, (reservoir, display_name, panel) in zip(axes, panels):
            for algorithm in bundle.algorithms:
                subset = statistics.loc[
                    (statistics["algorithm"] == algorithm)
                    & (statistics["reservoir"] == reservoir)
                ].sort_values("month")
                style = ALGORITHM_STYLES[algorithm]
                ax.plot(
                    subset["month"],
                    subset["mean_monthly_spill_volume_1e8m3"],
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4.0,
                    linestyle=style["linestyle"],
                    linewidth=1.45,
                    label=algorithm,
                )
            ax.set_ylabel(
                f"{display_name} mean spill\n" + r"volume ($10^8$ m$^3$)",
                fontsize=LABEL_FT,
            )
            _prepare_axes(ax)
            _panel_label(ax, panel)
        axes[-1].set_xticks(np.arange(1, 13))
        axes[-1].set_xticklabels(month_labels)
        axes[-1].set_xlabel("Calendar month", fontsize=LABEL_FT)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=len(bundle.algorithms),
            frameon=False,
            fontsize=LEGEND_FT,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        _save(fig, figures_dir, "fig06_monthly_mean_spill_volume", bundle.figure_dpi)


def plot_monthly_spill_boxplots(bundle: ComparisonBundle, figures_dir: Path) -> None:
    panels = [
        ("XW", "Reservoir X", "(a)"),
        ("NZD", "Reservoir N", "(b)"),
        ("Cascade", "Cascade total", "(c)"),
    ]
    months = np.arange(1, 13, dtype=float)
    step = 0.18
    offsets = (np.arange(len(bundle.algorithms)) - (len(bundle.algorithms) - 1) / 2.0) * step
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.8), dpi=bundle.figure_dpi, sharex=True)
        for panel_index, (ax, (reservoir, display_name, panel)) in enumerate(zip(axes, panels)):
            for offset, algorithm in zip(offsets, bundle.algorithms):
                frame = bundle.frames[algorithm]
                if reservoir == "XW":
                    values = frame["xw_spill_m3s"]
                elif reservoir == "NZD":
                    values = frame["nzd_spill_m3s"]
                else:
                    values = frame["xw_spill_m3s"] + frame["nzd_spill_m3s"]

                datasets = [
                    values.loc[frame["month"] == month].to_numpy(dtype=float)
                    for month in range(1, 13)
                ]
                positions = months + offset
                color = ALGORITHM_STYLES[algorithm]["color"]
                result = ax.boxplot(
                    datasets,
                    positions=positions,
                    widths=0.15,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=True,
                    medianprops={"color": "black", "linewidth": 0.85},
                    whiskerprops={"color": "black", "linewidth": 0.65},
                    capprops={"color": "black", "linewidth": 0.65},
                    flierprops={
                        "marker": "o",
                        "markersize": 1.7,
                        "alpha": 0.36,
                        "markerfacecolor": color,
                        "markeredgecolor": color,
                    },
                )
                for box in result["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(0.58)
                    box.set_edgecolor("black")
                    box.set_linewidth(0.7)

                means = np.array([np.mean(month_values) for month_values in datasets], dtype=float)
                ax.scatter(
                    positions,
                    means,
                    marker="o",
                    s=9,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=5,
                )

            ax.set_ylabel(f"{display_name} spill discharge (m³/s)", fontsize=LABEL_FT)
            ax.set_xlim(0.5, 12.5)
            _prepare_axes(ax)
            _panel_label(ax, panel)
            if panel_index < len(panels) - 1:
                ax.set_xticklabels([])

        axes[-1].set_xticks(months)
        axes[-1].set_xticklabels(month_labels)
        axes[-1].set_xlabel("Calendar month", fontsize=LABEL_FT)
        legend_handles = [
            Patch(
                facecolor=ALGORITHM_STYLES[algorithm]["color"],
                edgecolor="black",
                alpha=0.58,
                label=algorithm,
            )
            for algorithm in bundle.algorithms
        ]
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="Mean")
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=LEGEND_FT,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        _save(fig, figures_dir, "fig06_monthly_spill_boxplots", bundle.figure_dpi)


def _colored_boxplot(ax: plt.Axes, datasets: list[np.ndarray], algorithms: list[str]) -> None:
    boxplot = ax.boxplot(
        datasets,
        labels=algorithms,
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2.2, "alpha": 0.35},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )
    for box, algorithm in zip(boxplot["boxes"], algorithms):
        box.set_facecolor(ALGORITHM_STYLES[algorithm]["color"])
        box.set_alpha(0.62)
        box.set_edgecolor("black")
        box.set_linewidth(0.8)


def plot_annual_generation(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
    figures_dir: Path,
) -> None:
    annual = tables.annual_generation_by_year
    algorithms = bundle.algorithms

    with plt.rc_context(FONT_RC):
        fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=bundle.figure_dpi)
        datasets = [
            annual.loc[annual["algorithm"] == algorithm, "cascade_generation_gwh"].to_numpy(dtype=float)
            / 1000.0
            for algorithm in algorithms
        ]
        _colored_boxplot(ax, datasets, algorithms)
        ax.set_xlabel("Algorithm", fontsize=LABEL_FT)
        ax.set_ylabel("Annual cascade generation (TWh/yr)", fontsize=LABEL_FT)
        _prepare_axes(ax)
        fig.tight_layout()
        _save(fig, figures_dir, "fig07_annual_generation_boxplot", bundle.figure_dpi)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), dpi=bundle.figure_dpi)
        panels = [
            ("xw_generation_gwh", "Reservoir X annual generation (TWh/yr)", "(a)"),
            ("nzd_generation_gwh", "Reservoir N annual generation (TWh/yr)", "(b)"),
        ]
        for ax, (column, ylabel, panel) in zip(axes, panels):
            datasets = [
                annual.loc[annual["algorithm"] == algorithm, column].to_numpy(dtype=float) / 1000.0
                for algorithm in algorithms
            ]
            _colored_boxplot(ax, datasets, algorithms)
            ax.set_xlabel("Algorithm", fontsize=LABEL_FT)
            ax.set_ylabel(ylabel, fontsize=LABEL_FT)
            _prepare_axes(ax)
            _panel_label(ax, panel)
        fig.tight_layout()
        _save(fig, figures_dir, "fig07_annual_generation_boxplot_by_reservoir", bundle.figure_dpi)


def save_analysis_tables(tables: AnalysisTables, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "selected_performance_points.csv": tables.selected_performance,
        "deficit_statistics.csv": tables.deficit_statistics,
        "deficit_duration_points.csv": tables.deficit_duration,
        "monthly_power_statistics.csv": tables.monthly_power_statistics,
        "monthly_level_statistics.csv": tables.monthly_level_statistics,
        "spill_statistics.csv": tables.spill_statistics,
        "monthly_spill_statistics.csv": tables.monthly_spill_statistics,
        "annual_generation_by_year.csv": tables.annual_generation_by_year,
        "annual_generation_statistics.csv": tables.annual_generation_statistics,
        "summary_algorithms.csv": tables.summary_algorithms,
    }
    for filename, table in outputs.items():
        table.to_csv(tables_dir / filename, index=False, encoding="utf-8-sig")


def generate_all_selected_result_outputs(
    bundle: ComparisonBundle,
    tables: AnalysisTables,
) -> None:
    figures_dir = bundle.output_directory / "figures"
    tables_dir = bundle.output_directory / "tables"
    save_analysis_tables(tables, tables_dir)
    manifest = {
        "comparison_name": bundle.comparison_name,
        "selected_results": bundle.selections,
        "aggregation": {
            "reliability": "Mean of per-seed reliability on identical evaluation dates; never threshold mean power.",
            "cumulative_quantities": "Mean of per-seed full-period totals, not sum across seeds.",
            "distributions": "Pool individual seed-year/month samples; seeds are not independent hydrological years.",
            "p10_p90": "Empirical sample quantiles, not a confidence interval.",
            "seed_0": "Unspecified single-run seed label, not a claimed training seed.",
        },
    }
    with (tables_dir / "comparison_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    plot_selected_generation_reliability(bundle, tables, figures_dir)
    plot_monthly_power_boxplots(bundle, figures_dir)
    plot_power_duration(bundle, figures_dir)
    plot_deficit_duration(bundle, tables, figures_dir)
    plot_monthly_levels(bundle, tables, figures_dir)
    plot_spill_comparison(bundle, tables, figures_dir)
    plot_monthly_mean_spill_volume(bundle, tables, figures_dir)
    plot_monthly_spill_boxplots(bundle, figures_dir)
    plot_annual_generation(bundle, tables, figures_dir)
