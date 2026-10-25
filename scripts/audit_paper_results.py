"""Data-only audit and CSV export for the two new paper figures; never plots."""
import argparse
from pathlib import Path

from paper_results_data import DEFAULT_CONFIG, collect, export_audit, load_config, new_output, write_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--single-root", type=Path)
    parser.add_argument("--cascade-results-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    roots = {}
    if args.single_root:
        roots["single"] = str(args.single_root.resolve())
    if args.cascade_results_root:
        roots["cascade"] = str(args.cascade_results_root.resolve())
    config = load_config(args.config, roots)
    output = new_output(config, args.output_dir, "audit")
    try:
        result = collect(config)
        export_audit(result, output)
    except Exception as error:
        write_failure(output, error)
        print(f"DATA_AUDIT_FAILED; see {output / 'data_audit_failure.json'}")
        raise
    print(result["points"][["case", "method", "penalty", "seed_count", "xw_reliability_pct",
          "nzd_reliability_pct", "mean_annual_generation_twh", "relative_generation_difference_pct"]].to_string(index=False))
    print(f"DATA_AUDIT_OK; no plotting code executed; OUTPUT={output}")


if __name__ == "__main__":
    main()
