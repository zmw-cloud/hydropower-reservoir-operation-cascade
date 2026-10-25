"""Command-line entry point for the monthly dual-reservoir generator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from GeneratorCodes.configuration import load_configuration
from GeneratorCodes.pipeline import apply_overrides, load_inputs_and_model, run_generation
from GeneratorCodes.statistics import spatial_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "小湾天然入流—糯扎渡区间入流月尺度联合随机生成器；"
            "输入/输出单位均为月平均 m3/s。"
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.json"),
        help="JSON 配置文件路径",
    )
    parser.add_argument("--input-file", help="临时覆盖输入 Excel 路径")
    parser.add_argument("--output-dir", help="临时覆盖输出目录")
    parser.add_argument("--years", type=int, help="临时覆盖每个情景的合成年数")
    parser.add_argument("--start-year", type=int, help="临时覆盖日期标签起始年")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="只运行指定情景；可重复使用，例如 --scenario train --scenario dry",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖目标目录内的同名输出文件",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验输入和相关模型，不生成文件",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_configuration(args.config)
        config = apply_overrides(
            config,
            input_file=args.input_file,
            output_dir=args.output_dir,
            synthetic_years=args.years,
            calendar_start_year=args.start_year,
        )

        if args.validate_only:
            history, model = load_inputs_and_model(config)
            metrics = spatial_metrics(history.cube)
            print("输入校验通过")
            print(f"历史期：{history.frame['Date'].min():%Y-%m} 至 {history.frame['Date'].max():%Y-%m}")
            print(f"完整历史年：{history.n_years}")
            print(f"总体空间相关：{metrics['overall_pearson']:.8f}")
            print(f"去季节空间相关：{metrics['deseasonalized_pearson']:.8f}")
            print(f"相关矩阵旋转块起始月：{model.block_start_month_index + 1}")
            return 0

        manifest = run_generation(
            config,
            scenario_names=args.scenarios,
            overwrite=args.overwrite,
        )
        print(f"完成：{config.output_dir}")
        for scenario in manifest["scenarios"]:
            print(
                f"  {scenario['name']}: QC={scenario['qc_passed']}, "
                f"总体相关={scenario['synthetic_overall_pearson']:.8f}, "
                f"去季节相关={scenario['synthetic_deseasonalized_pearson']:.8f}"
            )
        return 0
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
