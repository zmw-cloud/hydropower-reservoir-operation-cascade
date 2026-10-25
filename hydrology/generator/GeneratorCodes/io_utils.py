"""Strict monthly input loading for Xiaowan and Nuozhadu interval inflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonthlyHistory:
    """Validated continuous monthly history."""

    source_file: Path
    sheet_name: str
    frame: pd.DataFrame
    cube: np.ndarray
    station_names: tuple[str, ...]
    source_columns: tuple[str, ...]
    calendar_years: np.ndarray

    @property
    def n_years(self) -> int:
        return int(self.cube.shape[0])


def load_monthly_history(
    input_file: str | Path,
    sheet_name: str,
    date_column: str,
    station_columns: tuple[tuple[str, str], ...],
) -> MonthlyHistory:
    """Read and validate one-row-per-month inflow observations."""

    path = Path(input_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入 Excel：{path}")

    workbook = pd.ExcelFile(path)
    if sheet_name not in workbook.sheet_names:
        raise ValueError(
            f"工作表 {sheet_name!r} 不存在；可用工作表={workbook.sheet_names}"
        )

    raw = pd.read_excel(path, sheet_name=sheet_name)
    source_columns = tuple(column for _, column in station_columns)
    required = (date_column,) + source_columns
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(f"输入 Excel 缺少列：{missing}")

    optional_calendar_columns = [
        column for column in ("year", "month") if column in raw.columns
    ]
    selected = raw.loc[:, list(required) + optional_calendar_columns].copy()
    selected[date_column] = pd.to_datetime(selected[date_column], errors="coerce")
    if selected[date_column].isna().any():
        bad_rows = selected.index[selected[date_column].isna()].tolist()[:10]
        raise ValueError(f"日期无法解析，示例行号：{bad_rows}")

    for column in source_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        if selected[column].isna().any():
            bad_rows = selected.index[selected[column].isna()].tolist()[:10]
            raise ValueError(f"列 {column!r} 存在缺失或非数值，示例行号：{bad_rows}")
        values = selected[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"列 {column!r} 存在非有限数值")
        if np.any(values <= 0.0):
            bad_rows = selected.index[selected[column] <= 0.0].tolist()[:10]
            raise ValueError(f"列 {column!r} 必须全部为正，示例行号：{bad_rows}")

    selected = selected.sort_values(date_column).reset_index(drop=True)
    periods = selected[date_column].dt.to_period("M")
    if periods.duplicated().any():
        duplicates = periods[periods.duplicated(keep=False)].astype(str).unique().tolist()
        raise ValueError(f"存在重复月份：{duplicates[:10]}")

    expected = pd.period_range(periods.iloc[0], periods.iloc[-1], freq="M")
    if len(expected) != len(periods) or not np.array_equal(
        periods.astype(str).to_numpy(), expected.astype(str).to_numpy()
    ):
        missing_periods = expected.difference(pd.PeriodIndex(periods)).astype(str).tolist()
        raise ValueError(f"月序列不连续；缺失月份示例：{missing_periods[:12]}")

    if periods.iloc[0].month != 1 or periods.iloc[-1].month != 12:
        raise ValueError("历史序列必须从 1 月开始并在 12 月结束")
    if len(selected) % 12 != 0:
        raise ValueError("历史月数必须是 12 的整数倍")

    station_names = tuple(name for name, _ in station_columns)
    frame = pd.DataFrame({"Date": periods.dt.to_timestamp()})
    for name, column in station_columns:
        frame[name] = selected[column].to_numpy(dtype=float)
    frame["Year"] = frame["Date"].dt.year
    frame["Month"] = frame["Date"].dt.month

    if "year" in selected.columns:
        source_year = pd.to_numeric(selected["year"], errors="coerce")
        if source_year.isna().any() or not np.array_equal(
            source_year.to_numpy(dtype=int), frame["Year"].to_numpy(dtype=int)
        ):
            raise ValueError("输入 year 列与 date 列不一致")
    if "month" in selected.columns:
        source_month = pd.to_numeric(selected["month"], errors="coerce")
        if source_month.isna().any() or not np.array_equal(
            source_month.to_numpy(dtype=int), frame["Month"].to_numpy(dtype=int)
        ):
            raise ValueError("输入 month 列与 date 列不一致")

    n_years = len(frame) // 12
    if n_years < 10:
        raise ValueError("至少需要 10 个完整历史年以估计联合相关结构")
    values = frame.loc[:, list(station_names)].to_numpy(dtype=float)
    cube = values.reshape(n_years, 12, len(station_names))
    calendar_years = frame["Year"].drop_duplicates().to_numpy(dtype=int)

    return MonthlyHistory(
        source_file=path,
        sheet_name=sheet_name,
        frame=frame,
        cube=cube,
        station_names=station_names,
        source_columns=source_columns,
        calendar_years=calendar_years,
    )
