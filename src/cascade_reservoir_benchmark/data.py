"""Validated loaders for observed and synthetic P4 V1 joint inflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .environment import CascadeSpecError, _as_month_period_index


SYNTHETIC_COLUMN_ALIASES = {
    "Date": "date",
    "Xiaowan_NaturalInflow_m3s": "xiaowan_inflow_m3s",
    "Nuozhadu_IntervalNaturalInflow_m3s": "nuozhadu_local_inflow_m3s",
    "xiaowan_natural_inflow_m3s": "xiaowan_inflow_m3s",
    "nuozhadu_interval_inflow_m3s": "nuozhadu_local_inflow_m3s",
}

JOINT_COLUMNS = ["date", "xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s"]


def _validate_joint_frame(frame: pd.DataFrame, expected_months: int | None = None) -> pd.DataFrame:
    frame = frame.loc[:, JOINT_COLUMNS].copy()
    periods = _as_month_period_index(frame["date"])
    for column in JOINT_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[JOINT_COLUMNS[1:]].isna().any().any() or (frame[JOINT_COLUMNS[1:]] <= 0.0).any().any():
        raise CascadeSpecError("Joint inflows must have complete, strictly positive dates and flows.")
    if periods.duplicated().any() or not periods.is_monotonic_increasing:
        raise CascadeSpecError("Joint inflow dates must be unique and increasing.")
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    if len(frame) != len(expected) or not periods.equals(expected):
        raise CascadeSpecError("Joint inflow record must contain continuous calendar months.")
    if expected_months is not None and len(frame) != expected_months:
        raise CascadeSpecError(f"Expected {expected_months} monthly records, got {len(frame)}.")
    frame["date"] = [f"{period.year:04d}-{period.month:02d}-01" for period in periods]
    return frame


def load_synthetic_joint_inflows(file_path: str | Path, expected_months: int = 4800) -> pd.DataFrame:
    """Load one independent joint 400-year synthetic sequence."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Synthetic inflow file not found: {file_path}")
    raw = pd.read_csv(file_path)
    rename = {
        column: SYNTHETIC_COLUMN_ALIASES[column]
        for column in raw.columns
        if column in SYNTHETIC_COLUMN_ALIASES
    }
    normalized = raw.rename(columns=rename)
    if normalized.columns.duplicated().any():
        raise CascadeSpecError("Synthetic inflow file contains duplicate canonical columns.")
    missing = set(JOINT_COLUMNS) - set(normalized.columns)
    if missing:
        raise CascadeSpecError(f"Synthetic inflow file missing columns: {sorted(missing)}")
    return _validate_joint_frame(normalized, expected_months)


def load_configured_joint_inflows(config: dict, split: str) -> pd.DataFrame:
    """Return the configured 400-year training or evaluation sequence.

    Relative paths are resolved from the cascade project root, so the same
    configuration works after copying the project to a cluster filesystem.
    """
    if split not in {"train", "evaluation"}:
        raise CascadeSpecError("split must be 'train' or 'evaluation'.")
    data = config["data"]
    file_path = Path(data[f"{split}_inflow_file"])
    if not file_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        file_path = (project_root / file_path).resolve()
    return load_synthetic_joint_inflows(
        file_path,
        expected_months=int(data["expected_months_per_split"]),
    )


def load_observed_joint_inflows(data_dir: str | Path) -> pd.DataFrame:
    """Load the aligned 1978-2008 XiaoWan-NuoZhaDu monthly inflow record.

    The source workbook stores natural runoff at each dam site.  The local
    XiaoWan-NuoZhaDu incremental inflow is therefore computed as downstream
    natural runoff minus upstream natural runoff.  Routed XiaoWan release is
    added only by the cascade simulator.
    """
    data_dir = Path(data_dir)
    source = data_dir / "raw" / "1-各电站天然月径流-改.xlsx"
    if not source.exists():
        candidates = sorted((data_dir / "raw").glob("1-*.xlsx"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Observed inflow source not found: {source}; candidates={candidates}"
            )
        source = candidates[0]
    book = pd.ExcelFile(source)
    if len(book.sheet_names) != 2:
        raise CascadeSpecError("Observed inflow workbook must have exactly two station sheets.")

    def read_sheet(sheet_name: str, output_name: str) -> pd.DataFrame:
        raw = pd.read_excel(source, sheet_name=sheet_name)
        if raw.shape[1] < 2:
            raise CascadeSpecError(f"{sheet_name}: expected date and inflow columns.")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
                output_name: pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            }
        )
        if frame.isna().any().any() or (frame[output_name] < 0.0).any():
            raise CascadeSpecError(f"{sheet_name}: dates and inflows must be complete and nonnegative.")
        if frame["date"].duplicated().any():
            raise CascadeSpecError(f"{sheet_name}: duplicate monthly dates are not allowed.")
        return frame

    upstream = read_sheet(book.sheet_names[0], "xiaowan_inflow_m3s")
    downstream = read_sheet(book.sheet_names[1], "nuozhadu_natural_inflow_m3s")
    joint = upstream.merge(downstream, on="date", how="inner", validate="one_to_one")
    joint["nuozhadu_local_inflow_m3s"] = (
        joint["nuozhadu_natural_inflow_m3s"] - joint["xiaowan_inflow_m3s"]
    )
    return _validate_joint_frame(joint)
