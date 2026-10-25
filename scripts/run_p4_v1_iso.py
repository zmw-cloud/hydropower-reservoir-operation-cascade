"""Run one explicitly selected stage of the two-reservoir ISO workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade_reservoir_benchmark.iso_workflow import (
    run_development,
    run_final_evaluation,
    run_final_training,
)


DEFAULT_CONFIG = ROOT / "configs" / "paper" / "iso500.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-reservoir ISO stages. Run development, inspect/freeze its best config, "
            "then run final-train once and evaluate the frozen artifact."
        )
    )
    parser.add_argument(
        "stage",
        choices=("development", "final-train", "evaluate"),
        help="The single ISO stage to execute.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "development":
        results_path, best_path = run_development(args.config)
        print(f"Wrote {results_path}")
        print(f"Wrote {best_path}")
    elif args.stage == "final-train":
        model_path, metadata_path = run_final_training(args.config)
        print(f"Wrote {model_path}")
        print(f"Wrote {metadata_path}")
    else:
        for schedule_path, summary_path in run_final_evaluation(args.config):
            print(f"Wrote {schedule_path}")
            print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
