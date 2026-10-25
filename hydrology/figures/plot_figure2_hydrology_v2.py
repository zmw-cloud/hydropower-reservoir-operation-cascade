"""Figure 2: original data/statistics with an updated publication layout.

Dependencies: numpy>=1.22, pandas, matplotlib>=3.6, openpyxl.
Bundled inputs require no path edits. Outputs: ./figure2_outputs beside this script.
HIST interval inflow = Nuozhadu natural inflow - Xiaowan natural inflow.
Statistics: sample SD (ddof=1), linearly interpolated percentiles,
and Pearson correlation of paired flows within each calendar month.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Export figures without a graphical display.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.colors import to_rgb, to_rgba


# Paths are relative to this script, independent of the terminal directory.
BASE_DIR = Path(__file__).resolve().parent
HIST_FILE = BASE_DIR / "input/historical_monthly_inflow.xlsx"
TRAIN_FILE = BASE_DIR / "input/inflow_joint_monthly_train.csv"
EVAL_FILE = BASE_DIR / "input/inflow_joint_monthly_evaluation.csv"
OUTPUT_DIR = BASE_DIR / "figure2_outputs"
SUMMARY_FILE = OUTPUT_DIR / "Figure2_hydrology_validation_summary.csv"
STEM = "Figure2_hydrology_validation_v2"
FLOW_COLS = ["xiaowan_natural_inflow_m3s", "nuozhadu_interval_inflow_m3s"]
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
DATASETS = ("HIST", "TRAIN", "EVAL")
COLORS = {"HIST": "#7F8FA6", "TRAIN": "#E69F00", "EVAL": "#3A86C8"}
CORR_YLIM = (0.85, 1.00)
BOX_ALPHA = 0.38
BOX_WIDTH = 0.25
BOX_OFFSETS = (-0.30, 0.0, 0.30)
FIGSIZE = (7.6, 4.45)
STYLES = {
    "HIST": dict(ls="-", marker="o", box_ls="-"),
    "TRAIN": dict(ls="--", marker="s", box_ls=(0, (2.5, 1.2))),
    "EVAL": dict(ls=":", marker="^", box_ls=(0, (1, 1))),
}


def validate_monthly_data(frame, label, first_year, years):
    """Require complete paired observations; never silently drop missing data."""
    frame = frame.copy()
    for col in ["Year", "Month", *FLOW_COLS]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[["Year", "Month", *FLOW_COLS]].to_numpy()).all():
        raise ValueError(f"{label}: missing or nonfinite values.")
    for col in ["Year", "Month"]:
        if not (frame[col] == np.floor(frame[col])).all():
            raise ValueError(f"{label}: {col} must contain integers.")
        frame[col] = frame[col].astype(int)
    keys = pd.MultiIndex.from_frame(frame[["Year", "Month"]])
    expected = pd.MultiIndex.from_product(
        [range(first_year, first_year + years), range(1, 13)]
    )
    if (keys.has_duplicates or len(keys) != years * 12
            or len(expected.difference(keys)) or len(keys.difference(expected))):
        raise ValueError(f"{label}: expected {years} complete years from {first_year}.")
    if (frame[FLOW_COLS] < 0).any().any():
        raise ValueError(f"{label}: negative inflow; check variable definitions.")
    return frame.sort_values(["Year", "Month"]).reset_index(drop=True)


def read_historical():
    frames = []
    for sheet, field in [("小湾", FLOW_COLS[0]), ("糯扎渡", "nuozhadu_natural")]:
        data = pd.read_excel(
            HIST_FILE, sheet_name=sheet, usecols=["年月", "天然径流"], engine="openpyxl"
        ).rename(columns={"天然径流": field})
        dates = pd.to_datetime(data.pop("年月"), errors="raise")
        data["Year"], data["Month"] = dates.dt.year, dates.dt.month
        frames.append(data)
    paired = frames[0].merge(
        frames[1], on=["Year", "Month"], how="outer",
        validate="one_to_one", indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("HIST: station calendars do not match.")
    for col in [FLOW_COLS[0], "nuozhadu_natural"]:
        paired[col] = pd.to_numeric(paired[col], errors="raise")
    paired[FLOW_COLS[1]] = paired["nuozhadu_natural"] - paired[FLOW_COLS[0]]
    return validate_monthly_data(paired, "HIST", 1978, 31)


def read_synthetic(path, label):
    data = pd.read_csv(
        path, encoding="utf-8-sig", usecols=["Date", "Year", "Month", *FLOW_COLS]
    )
    # Avoid nanosecond datetime overflow for dates extending to 2408.
    date_parts = data["Date"].astype(str).str.extract(r"^(\d{4})-(\d{2})-01$")
    if date_parts.isna().any().any():
        raise ValueError(f"{label}: expected Date in YYYY-MM-01 format.")
    date_parts = date_parts.astype(int)
    if not (date_parts[0].eq(pd.to_numeric(data["Year"]))
            & date_parts[1].eq(pd.to_numeric(data["Month"]))).all():
        raise ValueError(f"{label}: Date and Year/Month disagree.")
    return validate_monthly_data(data, label, 2009, 400)


def calculate_summary(datasets):
    rows = []
    for panel, variable in zip(["a", "b"], FLOW_COLS):
        for month in range(1, 13):
            for label, data in datasets.items():
                values = data.loc[data.Month == month, variable].to_numpy(float)
                p05, p25, median, p75, p95 = np.percentile(
                    values, [5, 25, 50, 75, 95], method="linear"
                )
                rows.append(dict(
                    panel=panel, variable=variable, month=month,
                    month_name=MONTHS[month - 1], dataset=label, n=len(values),
                    units="m3/s", mean=values.mean(), std=values.std(ddof=1),
                    median=median, p25=p25, p75=p75, p05=p05, p95=p95,
                    pearson_r=np.nan,
                ))
    for month in range(1, 13):
        for label, data in datasets.items():
            paired = data.loc[data.Month == month, FLOW_COLS].to_numpy(float)
            if np.any(paired.std(axis=0, ddof=1) == 0):
                raise ValueError(f"{label}, month {month}: Pearson r is undefined.")
            rows.append(dict(
                panel="c", variable="paired_inflows", month=month,
                month_name=MONTHS[month - 1], dataset=label, n=len(paired),
                units="dimensionless", pearson_r=np.corrcoef(paired.T)[0, 1],
            ))
    return pd.DataFrame(rows)


def draw_figure(summary):
    with plt.rc_context({
        "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.titlesize": 9.3, "axes.labelsize": 8.8,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.8,
        "legend.fontsize": 8, "axes.linewidth": 0.6,
        "axes.edgecolor": "0.30", "text.color": "0.15",
        "axes.labelcolor": "0.15", "xtick.color": "0.25",
        "ytick.color": "0.25", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "white",
        "axes.facecolor": "white",
    }):
        fig = plt.figure(figsize=FIGSIZE)
        grid = fig.add_gridspec(
            2, 2, left=0.086, right=0.991, bottom=0.09, top=0.875,
            wspace=0.23, hspace=0.29, height_ratios=[1.9, 1],
        )
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, :])]
        titles = ["(a) Reservoir X natural inflow", "(b) Reservoir N interval inflow",
                  "(c) Monthly contemporaneous inflow correlation"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, loc="left", pad=6)
            ax.set_xlim(0.45, 12.55)
            ax.set_xticks(range(1, 13), MONTHS)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out", length=2.5, width=0.6, pad=3)
            ax.grid(axis="y", color="0.92", linewidth=0.45)
            ax.set_axisbelow(True)

        for panel, ax in zip(["a", "b"], axes[:2]):
            panel_rows = summary.loc[summary.panel == panel]
            for offset, label in zip(BOX_OFFSETS, DATASETS):
                table = panel_rows.loc[panel_rows.dataset == label].sort_values("month")
                style = STYLES[label]
                face = to_rgba(COLORS[label], alpha=BOX_ALPHA)
                edge = tuple(component * 0.68 for component in to_rgb(COLORS[label]))
                # Exact percentiles from the original statistical calculation.
                stats = [dict(q1=r.p25, med=r.median, q3=r.p75,
                              whislo=r.p05, whishi=r.p95)
                         for r in table.itertuples(index=False)]
                ax.bxp(
                    stats, positions=np.arange(1, 13) + offset, widths=BOX_WIDTH,
                    showfliers=False, patch_artist=True, manage_ticks=False,
                    boxprops=dict(facecolor=face, edgecolor=edge,
                                  linestyle=style["box_ls"], linewidth=0.65),
                    medianprops=dict(color=edge, linewidth=0.90),
                    whiskerprops=dict(color=edge, linewidth=0.65,
                                      linestyle=style["box_ls"]),
                    capprops=dict(color=edge, linewidth=0.65),
                )
            ax.set_ylabel(r"Monthly inflow (m$^3$/s)", labelpad=5)
            ax.set_ylim(0, panel_rows.p95.max() * 1.10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        correlation_rows = summary.loc[summary.panel == "c"]
        for label in DATASETS:
            table = correlation_rows.loc[correlation_rows.dataset == label].sort_values("month")
            style = STYLES[label]
            axes[2].plot(
                table.month, table.pearson_r, color=COLORS[label],
                linestyle=style["ls"], marker=style["marker"], linewidth=1.05,
                markersize=3.6, markerfacecolor="white", markeredgewidth=0.8,
                clip_on=False,
            )
        axes[2].set_ylabel("Pearson r", labelpad=5)
        axes[2].set_ylim(*CORR_YLIM)
        axes[2].set_yticks(np.linspace(*CORR_YLIM, 4))
        axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axes[2].axhline(1.0, color="0.87", linewidth=0.5, zorder=0, clip_on=False)

        handles = []
        for label in DATASETS:
            style = STYLES[label]
            face = to_rgba(COLORS[label], alpha=BOX_ALPHA)
            edge = tuple(component * 0.68 for component in to_rgb(COLORS[label]))
            handles.append((
                Patch(facecolor=face, edgecolor=edge,
                      linestyle=style["box_ls"], linewidth=0.65),
                Line2D([], [], color=COLORS[label], linestyle=style["ls"],
                       marker=style["marker"], markerfacecolor="white",
                       linewidth=1.05, markersize=3.6),
            ))
        fig.legend(
            handles, DATASETS, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper center", bbox_to_anchor=(0.5, 0.989), ncol=3,
            frameon=False, handlelength=3.1, handletextpad=0.65,
            columnspacing=1.8, borderaxespad=0,
        )
        return fig


def main():
    datasets = {
        "HIST": read_historical(),
        "TRAIN": read_synthetic(TRAIN_FILE, "TRAIN"),
        "EVAL": read_synthetic(EVAL_FILE, "EVAL"),
    }
    summary = calculate_summary(datasets)
    for label, data in datasets.items():
        per_month = int(data.groupby("Month").size().iloc[0])
        print(f"{label}: {len(data)} monthly records ({per_month} per calendar month)")
    print(f"Summary: {len(summary)} rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig",
                   float_format="%.12g", na_rep="")
    fig = draw_figure(summary)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ["pdf", "png", "svg"]:
            path = OUTPUT_DIR / f"{STEM}.{extension}"
            fig.savefig(path, dpi=600, facecolor="white")
            print(f"Saved: {path.resolve()}")
    plt.close(fig)
    print(f"Saved: {SUMMARY_FILE.resolve()}")
    print("PNG resolution: 600 dpi")
    print("Completed: Figure 2")
    return summary


if __name__ == "__main__":
    summary = main()
