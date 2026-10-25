"""Configuration loading and validation for the monthly generator."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScenarioConfig:
    """One independently generated hydrologic scenario."""

    name: str
    seed: int
    mean_change_pct: Any
    sd_change_pct: Any


@dataclass(frozen=True)
class GeneratorConfig:
    """Validated project configuration."""

    config_path: Path
    input_file: Path
    input_sheet: str
    date_column: str
    station_columns: tuple[tuple[str, str], ...]
    synthetic_years: int
    calendar_start_year: int
    output_dir: Path
    regularization: float
    minimum_calibration_samples: int
    calibration_multiplier: int
    scenarios: tuple[ScenarioConfig, ...]
    correlation_tolerances: dict[str, float]
    fail_on_qc: bool

    @property
    def station_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.station_columns)

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(column for _, column in self.station_columns)


DEFAULT_TOLERANCES = {
    "overall_spatial_abs_error": 0.03,
    "deseasonalized_spatial_abs_error": 0.05,
    "monthly_spatial_max_abs_error": 0.12,
    "log_correlation_rmse": 0.08,
    "monthly_mean_relative_error": 1e-6,
    "monthly_sd_relative_error": 1e-6,
}


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"配置项 {key!r} 必须是非空对象")
    return value


def load_configuration(config_path: str | Path) -> GeneratorConfig:
    """Load a UTF-8 JSON configuration and validate consequential fields."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")

    base_dir = path.parent
    input_file = _resolve_path(str(raw["input_file"]), base_dir)
    output_dir = _resolve_path(str(raw.get("output_dir", "outputs")), base_dir)

    station_mapping = _require_mapping(raw, "station_columns")
    station_columns = tuple((str(k), str(v)) for k, v in station_mapping.items())
    if len(station_columns) != 2:
        raise ValueError("本双库版本要求 station_columns 恰好包含两列")
    if len({name for name, _ in station_columns}) != 2:
        raise ValueError("内部站点名称必须唯一")
    if len({column for _, column in station_columns}) != 2:
        raise ValueError("Excel 源列名必须唯一")

    synthetic_years = int(raw.get("synthetic_years", 400))
    if synthetic_years < 2:
        raise ValueError("synthetic_years 必须至少为 2")

    calendar_start_year = int(raw.get("calendar_start_year", 2009))
    if calendar_start_year < 1900 or calendar_start_year + synthetic_years - 1 > 9999:
        raise ValueError("日期标签必须位于 Excel/Python 可表示的 1900—9999 年范围内")

    regularization = float(raw.get("regularization", 1e-4))
    if not 0.0 <= regularization < 1.0:
        raise ValueError("regularization 必须位于 [0, 1) 区间")

    minimum_calibration_samples = int(raw.get("minimum_calibration_samples", 20_000))
    calibration_multiplier = int(raw.get("calibration_multiplier", 50))
    if minimum_calibration_samples < 1000 or calibration_multiplier < 1:
        raise ValueError("相关结构校准样本数设置过小")

    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios 必须是非空数组")

    scenarios: list[ScenarioConfig] = []
    seen_names: set[str] = set()
    seen_seeds: set[int] = set()
    for item in raw_scenarios:
        if not isinstance(item, dict):
            raise ValueError("每个 scenario 必须是对象")
        name = str(item["name"]).strip()
        if not name or any(char in name for char in '<>:"/\\|?*'):
            raise ValueError(f"非法情景名称：{name!r}")
        seed = int(item["seed"])
        if seed < 0:
            raise ValueError(f"情景 {name!r} 的随机种子不能为负数")
        if name in seen_names:
            raise ValueError(f"情景名称重复：{name}")
        if seed in seen_seeds:
            raise ValueError(f"随机种子重复：{seed}；四组情景应相互独立")
        seen_names.add(name)
        seen_seeds.add(seed)
        scenarios.append(
            ScenarioConfig(
                name=name,
                seed=seed,
                mean_change_pct=item.get("mean_change_pct", 0.0),
                sd_change_pct=item.get("sd_change_pct", 0.0),
            )
        )

    tolerances = DEFAULT_TOLERANCES.copy()
    for key, value in raw.get("correlation_tolerances", {}).items():
        if key not in tolerances:
            raise ValueError(f"未知 QC 容差项：{key}")
        tolerances[key] = float(value)
    if any(value < 0 for value in tolerances.values()):
        raise ValueError("QC 容差不能为负数")

    return GeneratorConfig(
        config_path=path,
        input_file=input_file,
        input_sheet=str(raw.get("input_sheet", "interval_inflow")),
        date_column=str(raw.get("date_column", "date")),
        station_columns=station_columns,
        synthetic_years=synthetic_years,
        calendar_start_year=calendar_start_year,
        output_dir=output_dir,
        regularization=regularization,
        minimum_calibration_samples=minimum_calibration_samples,
        calibration_multiplier=calibration_multiplier,
        scenarios=tuple(scenarios),
        correlation_tolerances=tolerances,
        fail_on_qc=bool(raw.get("fail_on_qc", True)),
    )


def expand_change(
    value: Any,
    station_names: tuple[str, ...],
    label: str,
) -> np.ndarray:
    """Expand a scalar/list/station mapping into a [12, station] array."""

    n_stations = len(station_names)

    def expand_one(item: Any, context: str) -> np.ndarray:
        if np.isscalar(item) and not isinstance(item, (str, bytes)):
            return np.full(12, float(item), dtype=float)
        array = np.asarray(item, dtype=float)
        if array.shape != (12,):
            raise ValueError(f"{context} 必须是数字或 12 个逐月数值")
        return array

    if isinstance(value, dict):
        unknown = set(value) - set(station_names)
        missing = set(station_names) - set(value)
        if unknown or missing:
            raise ValueError(
                f"{label} 的站点键不匹配；缺少={sorted(missing)}，未知={sorted(unknown)}"
            )
        result = np.column_stack(
            [expand_one(value[name], f"{label}.{name}") for name in station_names]
        )
    else:
        monthly = expand_one(value, label)
        result = np.repeat(monthly[:, None], n_stations, axis=1)

    if not np.isfinite(result).all():
        raise ValueError(f"{label} 包含非有限数值")
    if np.any(result <= -100.0):
        raise ValueError(f"{label} 必须全部大于 -100%，以保证流量和标准差为正")
    return result
