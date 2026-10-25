"""Render only the six existing paper figures with anonymous display labels.

Uses the existing formal6.0.0 data loaders/statistics/plot functions unchanged.
The historical/synthetic inflow figure uses its original self-contained bundle.
Only visible text and presentation are changed; no result CSV is rewritten.
Run locally in PyCharm to produce the final figures.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from uuid import uuid4

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASCADE_ROOT = PROJECT_ROOT
HYDRO_SCRIPT = PROJECT_ROOT / "hydrology/figures/plot_figure2_hydrology_v2.py"
FIGURE_IDS = ("fig1", "fig3", "fig5", "fig6", "fig7", "fig8")


def load_script(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Existing plotting script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=FIGURE_IDS, default=list(FIGURE_IDS))
    parser.add_argument("--cascade-root", type=Path, default=CASCADE_ROOT)
    parser.add_argument("--hydrology-script", type=Path, default=HYDRO_SCRIPT)
    parser.add_argument("--font-dir", type=Path, default=PROJECT_ROOT / "assets/fonts")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cascade = args.cascade_root.resolve()
    # Prefer the actual formal plotting package, not16.0.0's older plot loaders.
    sys.path.insert(0, str(cascade / "src"))
    sys.path.insert(0, str(cascade / "scripts"))
    from cascade_reservoir_benchmark.comparison.presentation import (
        FONT_RC, FIGURE_DPI, finalize_figure,
    )
    action = load_script(cascade / "scripts/plot_dqn_ppo_action_behavior.py",
                         "paper_anonymous_action")
    action.configure_times_new_roman(args.font_dir)
    output = args.output_dir or PROJECT_ROOT / "outputs/paper_figures_anonymized" / (
        f"final_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    requested = set(args.figures)
    config_path = cascade / "configs/comparison_ddp_sdp_iso_dqn_ppo.yaml"
    print(f"FORMAL_CONFIG={config_path}")
    print(f"OUTPUT={output}")

    if "fig1" in requested:
        hydro = load_script(args.hydrology_script.resolve(), "paper_anonymous_hydrology")
        datasets = {
            "HIST": hydro.read_historical(),
            "TRAIN": hydro.read_synthetic(hydro.TRAIN_FILE, "TRAIN"),
            "EVAL": hydro.read_synthetic(hydro.EVAL_FILE, "EVAL"),
        }
        summary = hydro.calculate_summary(datasets)
        with plt.rc_context(FONT_RC):
            fig = hydro.draw_figure(summary)
            finalize_figure(fig)
            for extension in ("pdf", "png", "svg"):
                fig.savefig(output / f"{hydro.STEM}.{extension}", dpi=FIGURE_DPI,
                            facecolor="white")
            plt.close(fig)
        print(f"Fig.1 -> {hydro.STEM}.png/.pdf/.svg")

    if "fig3" in requested:
        tradeoff = load_script(cascade / "scripts/plot_generation_reliability_tradeoff.py",
                              "paper_anonymous_tradeoff")
        points = tradeoff.load_tradeoff_points()
        tradeoff.FIGURES_DIR = output
        tradeoff.save_combined_panel(points)
        print("Fig.3 -> fig02_generation_reliability_tradeoff_combined.png/.pdf")

    if requested.intersection({"fig5", "fig6", "fig7"}):
        from cascade_reservoir_benchmark.comparison import (
            load_comparison_bundle, build_analysis_tables,
        )
        from cascade_reservoir_benchmark.comparison import plots
        bundle = load_comparison_bundle(cascade, config_path)
        if "fig5" in requested:
            plots.plot_power_duration(bundle, output)
            print("Fig.5 -> fig03_power_duration.png/.pdf")
        if requested.intersection({"fig6", "fig7"}):
            tables = build_analysis_tables(bundle)
            if "fig6" in requested:
                plots._plot_monthly_levels(bundle, tables, output,
                    central_column="mean_level_m", central_label="mean",
                    stem="fig05_monthly_level_mean_p10_p90")
                print("Fig.6 -> fig05_monthly_level_mean_p10_p90.png/.pdf")
            if "fig7" in requested:
                plots.plot_monthly_mean_spill_volume(bundle, tables, output)
                print("Fig.7 -> fig06_monthly_mean_spill_volume.png/.pdf")

    if "fig8" in requested:
        _, runs = action.load_action_runs(config_path)
        action.draw(runs, output)
        print(f"Fig.8 -> {action.FIGURE_STEM}.png/.pdf/.svg")

    print("ANONYMIZED_PAPER_FIGURES_OK")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
