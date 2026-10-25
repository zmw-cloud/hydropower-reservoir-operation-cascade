"""End-to-end orchestration for all configured monthly scenarios."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from .configuration import GeneratorConfig, expand_change
from .io_utils import load_monthly_history
from .joint_monthly import generate_scenario, prepare_joint_monthly_model
from .output_writer import sha256_file, write_json, write_scenario_outputs
from .statistics import build_qc_bundle


def apply_overrides(
    config: GeneratorConfig,
    *,
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    synthetic_years: int | None = None,
    calendar_start_year: int | None = None,
) -> GeneratorConfig:
    updates = {}
    if input_file is not None:
        updates["input_file"] = Path(input_file).expanduser().resolve()
    if output_dir is not None:
        updates["output_dir"] = Path(output_dir).expanduser().resolve()
    if synthetic_years is not None:
        if synthetic_years < 2:
            raise ValueError("synthetic_years 必须至少为 2")
        updates["synthetic_years"] = int(synthetic_years)
    if calendar_start_year is not None:
        updates["calendar_start_year"] = int(calendar_start_year)
    updated = replace(config, **updates)
    if updated.calendar_start_year + updated.synthetic_years - 1 > 9999:
        raise ValueError("结束年份超过 9999")
    return updated


def _selected_scenarios(
    config: GeneratorConfig,
    scenario_names: Iterable[str] | None,
):
    if scenario_names is None:
        return config.scenarios
    requested = list(dict.fromkeys(scenario_names))
    available = {scenario.name: scenario for scenario in config.scenarios}
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"未知情景：{unknown}；可用={sorted(available)}")
    return tuple(available[name] for name in requested)


def load_inputs_and_model(config: GeneratorConfig):
    history = load_monthly_history(
        input_file=config.input_file,
        sheet_name=config.input_sheet,
        date_column=config.date_column,
        station_columns=config.station_columns,
    )
    model = prepare_joint_monthly_model(history, config.regularization)
    return history, model


def run_generation(
    config: GeneratorConfig,
    *,
    scenario_names: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """Generate, validate, and write all selected scenarios."""

    scenarios = _selected_scenarios(config, scenario_names)
    output_root = config.output_dir
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"输出目录非空，拒绝覆盖：{output_root}；请换目录或显式使用 --overwrite"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    history, model = load_inputs_and_model(config)
    manifest_scenarios = []
    summary_frames = []
    failed_scenarios = []

    for scenario in scenarios:
        mean_change = expand_change(
            scenario.mean_change_pct,
            config.station_names,
            f"scenarios.{scenario.name}.mean_change_pct",
        )
        sd_change = expand_change(
            scenario.sd_change_pct,
            config.station_names,
            f"scenarios.{scenario.name}.sd_change_pct",
        )
        generated = generate_scenario(
            model=model,
            synthetic_years=config.synthetic_years,
            seed=scenario.seed,
            mean_change_pct=mean_change,
            sd_change_pct=sd_change,
            minimum_calibration_samples=config.minimum_calibration_samples,
            calibration_multiplier=config.calibration_multiplier,
        )
        qc = build_qc_bundle(
            scenario_name=scenario.name,
            seed=scenario.seed,
            historical_cube=history.cube,
            generated=generated,
            model=model,
            mean_change_pct=mean_change,
            sd_change_pct=sd_change,
            tolerances=config.correlation_tolerances,
        )
        written = write_scenario_outputs(
            output_root=output_root,
            scenario_name=scenario.name,
            seed=scenario.seed,
            start_year=config.calendar_start_year,
            history=history,
            model=model,
            generated=generated,
            qc=qc,
            mean_change_pct=mean_change,
            sd_change_pct=sd_change,
        )
        summary_frames.append(qc.summary)
        if not qc.report["qc_passed"]:
            failed_scenarios.append(scenario.name)
        manifest_scenarios.append(
            {
                "name": scenario.name,
                "seed": scenario.seed,
                "qc_passed": qc.report["qc_passed"],
                "historical_overall_pearson": qc.report[
                    "historical_spatial_correlation"
                ]["overall_pearson"],
                "synthetic_overall_pearson": qc.report[
                    "synthetic_spatial_correlation"
                ]["overall_pearson"],
                "historical_deseasonalized_pearson": qc.report[
                    "historical_spatial_correlation"
                ]["deseasonalized_pearson"],
                "synthetic_deseasonalized_pearson": qc.report[
                    "synthetic_spatial_correlation"
                ]["deseasonalized_pearson"],
                "files": [str(path.relative_to(output_root)) for path in written],
            }
        )

    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = output_root / "reports" / "scenario_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    manifest = {
        "status": "passed" if not failed_scenarios else "failed_qc",
        "generator": "Target Synthetic Streamflow Generator - Monthly Dual Reservoir",
        "synthetic_years_per_scenario": config.synthetic_years,
        "calendar_labels": (
            f"{config.calendar_start_year}-01 to "
            f"{config.calendar_start_year + config.synthetic_years - 1}-12"
        ),
        "input_file": str(history.source_file),
        "input_sha256": sha256_file(history.source_file),
        "historical_months": int(len(history.frame)),
        "historical_years": history.n_years,
        "units": "m3/s monthly mean flow",
        "station_columns": dict(config.station_columns),
        "rotated_block_start_month": int(model.block_start_month_index + 1),
        "scenarios": manifest_scenarios,
        "failed_qc_scenarios": failed_scenarios,
    }
    write_json(output_root / "generation_manifest.json", manifest)

    if failed_scenarios and config.fail_on_qc:
        raise RuntimeError(
            f"以下情景未通过 QC：{failed_scenarios}；报告已写入 {output_root / 'reports'}"
        )
    return manifest
