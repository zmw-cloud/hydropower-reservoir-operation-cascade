"""Two new paper figures from frozen, independently audited evaluation results.

This entry is intentionally not executed during authoring. Run it in VS Code,
PyCharm, or a terminal when figures are wanted. No simulation/model is imported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np

from paper_results_data import (
    DEFAULT_CONFIG, PROJECT_ROOT, collect, export_audit, load_config,
    new_output, write_failure,
)

FONT_FAMILY = "Times New Roman"


def configure_font(config: dict, supplied: Path | None) -> dict:
    directory = supplied or Path(config["font_directory"])
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in (".ttf", ".otf"):
                font_manager.fontManager.addfont(str(path))
    family = FONT_FAMILY
    for weight in ("normal", "bold"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family, weight=weight),
                                  fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError(f"Font {family!r} is unavailable. Provide its regular/bold TTF files "
                               f"in {directory}, or use --font-dir. Data audits have been exported.") from exc
    return {
        "font.family": family, "font.serif": [family], "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    }


def style_axes(ax, family: str, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E3E3E3", linewidth=0.55, alpha=0.85)
    ax.tick_params(direction="out", length=3.5, width=0.75, labelsize=10)
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.75)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(family)


def save_figure(fig, directory: Path, stem: str, family: str,
                formats: tuple[str, ...] = ("pdf", "tiff")) -> None:
    for text in fig.findobj(match=Text):
        text.set_fontfamily(family)
    for extension in formats:
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(directory / f"{stem}.{extension}", format=extension, dpi=600,
                    bbox_inches="tight", pad_inches=0.04, facecolor="white", **options)
    plt.close(fig)


def plot_violins(result: dict, config: dict, directory: Path) -> None:
    annual = result["annual_seed_mean"]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 10.6), sharex=False, sharey=False)
    for panel, ax, case in zip(("(a)", "(b)"), axes, config["cases"]):
        methods = [method["method"] for method in case["methods"]]
        arrays = []
        for position, method in enumerate(methods, start=1):
            values = annual.loc[(annual["case"] == case["id"]) & (annual["method"] == method),
                                "annual_generation_twh"].to_numpy(dtype=float)
            if len(values) != 400 or not np.isfinite(values).all():
                raise ValueError(f"Expected400 finite annual plotting values: {case['id']} {method}")
            arrays.append(values)
            color = config["colors"][method]
            if np.ptp(values) > 0:
                violin = ax.violinplot([values], positions=[position], widths=0.78, points=200,
                                      showmeans=False, showmedians=False, showextrema=False,
                                      bw_method="scott")
                for body in violin["bodies"]:
                    outline = tuple(component * 0.62 for component in to_rgb(color))
                    body.set_alpha(None)
                    body.set_facecolor((*to_rgb(color), 0.90))
                    body.set_edgecolor((*outline, 1.0))
                    body.set_linewidth(1.35)
            else:
                ax.plot([position - 0.3, position + 0.3], [values[0], values[0]], color=color)
            ax.boxplot([values], positions=[position], widths=0.16, patch_artist=True,
                       manage_ticks=False, showfliers=True, whis=1.5,
                       boxprops={"facecolor": "white", "edgecolor": "#242424", "linewidth": 1.15},
                       medianprops={"color": "#111111", "linewidth": 1.6},
                       whiskerprops={"color": "#303030", "linewidth": 1.1},
                       capprops={"color": "#303030", "linewidth": 1.1},
                       flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": "#3A3A3A",
                                   "markeredgecolor": "#303030", "markeredgewidth": 0.5, "alpha": 0.75})
            ax.scatter(position, float(values.mean()), marker="D", s=32,
                       facecolor="white", edgecolor="#111111", linewidth=1.05, zorder=5)
        all_values = np.concatenate(arrays)
        span = float(np.ptp(all_values))
        padding = span * 0.06 if span > 0 else max(abs(float(all_values[0])) * 0.05, 0.1)
        ax.set_ylim(float(all_values.min()) - padding, float(all_values.max()) + padding)
        ax.set_xlim(0.4, len(methods) + 0.6)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods, fontfamily=FONT_FAMILY, fontsize=11.5)
        ax.set_xlabel("Scheduling method", labelpad=8, fontsize=12, fontfamily=FONT_FAMILY)
        ax.set_ylabel("Mean annual generation (TWh/yr)", labelpad=9,
                      fontsize=12, fontfamily=FONT_FAMILY)
        ax.set_title(f"{panel} {case['id']}: {case['label']}", loc="left",
                     fontweight="bold", fontsize=13, fontfamily=FONT_FAMILY, pad=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        style_axes(ax, config["font_family"])
        ax.tick_params(axis="x", labelsize=11.5)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, alpha=0.85)
    handles = [
        Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor="white",
               markeredgecolor="#111111", label="Mean"),
        Patch(facecolor="white", edgecolor="#242424", linewidth=1.15, label="Median and IQR"),
        Line2D([], [], linestyle="none", marker="o", markersize=2.5, markerfacecolor="#3A3A3A",
               markeredgecolor="#303030", label="Years outside 1.5×IQR"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=3,
               frameon=False, prop={"family": FONT_FAMILY, "size": 10},
               handlelength=1.6, columnspacing=2.0, handletextpad=0.65)
    fig.tight_layout(rect=(0.015, 0.015, 0.995, 0.952), h_pad=2.3)
    save_figure(fig, directory, "fig_annual_generation_violin", FONT_FAMILY,
                formats=("png", "pdf", "tiff"))


class ArrowLegendHandler(HandlerPatch):
    def create_artists(self, legend, original, xdescent, ydescent, width, height,
                       fontsize, transform):
        arrow = FancyArrowPatch((xdescent, ydescent + height / 2),
                                (xdescent + width, ydescent + height / 2),
                                arrowstyle="-|>", mutation_scale=fontsize,
                                linewidth=1.1, color="#555555")
        arrow.set_transform(transform)
        return [arrow]


def plot_transfer(result: dict, config: dict, directory: Path) -> None:
    points = result["points"]
    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    methods = list(config["colors"])
    target = config["target_reliability_pct"]
    for method in methods:
        group = points.loc[points["method"] == method]
        if group.empty:
            continue
        color = config["colors"][method]
        first = group.loc[group["case"] == "Case 1"]
        second = group.loc[group["case"] == "Case 2"]
        if len(first) == 1 and len(second) == 1:
            start, end = first.iloc[0], second.iloc[0]
            ax.annotate("", xy=(end["deviation_from_target_pp"], end["relative_generation_difference_pct"]),
                        xytext=(start["deviation_from_target_pp"], start["relative_generation_difference_pct"]),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.35,
                                    "alpha": 0.75, "shrinkA": 6, "shrinkB": 6}, zorder=2)
        for _, row in group.iterrows():
            marker = "o" if row["case"] == "Case 1" else "s"
            face = color if row["transfer_reliability_pct"] >= target else "white"
            ax.scatter(float(row["deviation_from_target_pp"]), float(row["relative_generation_difference_pct"]),
                       marker=marker, s=70, facecolor=face, edgecolor=color, linewidth=1.5, zorder=4)
        # One endpoint label per method keeps the arrows readable.
        if len(second) == 1:
            row = second.iloc[0]
            offsets = {"DDP": (8, 8), "SDP": (8, 8), "ISO-RF": (8, 8),
                       "DQN": (-7, -16), "PPO": (8, -12)}
            ax.annotate(method, (row["deviation_from_target_pp"], row["relative_generation_difference_pct"]),
                        xytext=offsets[method], textcoords="offset points", fontsize=10,
                        color=color, ha="right" if method == "DQN" else "left")
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax.axhline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    x = points["deviation_from_target_pp"].to_numpy(dtype=float)
    y = points["relative_generation_difference_pct"].to_numpy(dtype=float)
    x_pad, y_pad = max(float(np.ptp(x)) * 0.13, 1.0), max(float(np.ptp(y)) * 0.13, 0.6)
    ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
    ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
    ax.set_xlabel("Deviation from 80% reliability target (percentage points)", labelpad=8)
    ax.set_ylabel("Relative annual generation difference\nfrom DDP (%)", labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axes(ax, config["font_family"], "both")
    colors = [Line2D([], [], color=config["colors"][method], linewidth=2, label=method)
              for method in methods]
    fig.legend(handles=colors, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5,
               frameon=False, prop={"family": config["font_family"], "size": 10})
    arrow = FancyArrowPatch((0, 0), (1, 0), color="#555555", label="Case 1 to Case 2")
    semantics = [
        Line2D([], [], linestyle="none", marker="o", color="#555555", label="Case 1"),
        Line2D([], [], linestyle="none", marker="s", color="#555555", label="Case 2"),
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor="#555555", label="Reliability < 80%"),
        Line2D([], [], linestyle="none", marker="o", color="#555555", label="Reliability ≥ 80%"),
        arrow,
    ]
    fig.legend(handles=semantics, handler_map={FancyArrowPatch: ArrowLegendHandler()},
               loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5,
               frameon=False, prop={"family": config["font_family"], "size": 9.3},
               columnspacing=1.4, handlelength=2.0)
    fig.text(0.5, 0.015, "Case 2 reliability = min(mean-seed XW reliability, mean-seed NZD reliability). "
             "PPO has Case 2 only.", ha="center", fontsize=9, fontfamily=config["font_family"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    save_figure(fig, directory, "fig_single_to_cascade_transfer", config["font_family"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--single-root", type=Path)
    parser.add_argument("--cascade-results-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--figure", choices=("violin", "transfer", "both"), default="violin",
                        help="Default: render only the updated annual-generation violin figure")
    args = parser.parse_args()
    roots = {}
    if args.single_root:
        roots["single"] = str(args.single_root.resolve())
    if args.cascade_results_root:
        roots["cascade"] = str(args.cascade_results_root.resolve())
    config = load_config(args.config, roots)
    config["font_family"] = FONT_FAMILY
    output = new_output(config, args.output_dir)
    try:
        result = collect(config)  # All source/selection/unit/seed/mean checks precede figures.
        export_audit(result, output)
    except Exception as error:
        write_failure(output, error)
        raise
    rc = configure_font(config, args.font_dir)
    with plt.rc_context(rc):
        if args.figure in ("violin", "both"):
            plot_violins(result, config, output)
        if args.figure in ("transfer", "both"):
            plot_transfer(result, config, output)
    print(f"PAPER_RESULTS_FIGURES_OK: {args.figure}; vector PDF and600dpi raster output")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
