"""Joint monthly stochastic generation in log space.

The original project generates monthly flow from daily observations one site at a
time and optionally disaggregates back to daily values.  This module works
directly with monthly mean discharge and extends the resampling/correlation
reconstruction to the full 12-month x 2-site vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import MonthlyHistory


@dataclass(frozen=True)
class JointMonthlyModel:
    """Historical parameters required by every generated scenario."""

    standardized_log_cube: np.ndarray
    historical_mean: np.ndarray
    historical_sd: np.ndarray
    observed_log_correlation: np.ndarray
    target_log_correlation: np.ndarray
    regularization: float
    station_names: tuple[str, ...]
    historical_years: np.ndarray
    block_start_month_index: int
    component_month_order: np.ndarray


@dataclass(frozen=True)
class GeneratedScenario:
    """One jointly generated [year, month, site] realization."""

    values: np.ndarray
    sampled_historical_year_indices: np.ndarray
    target_mean: np.ndarray
    target_sd: np.ndarray
    base_log_correlation: np.ndarray
    latent_log_correlation: np.ndarray


def stabilize_correlation(
    correlation: np.ndarray,
    regularization: float,
    eigenvalue_floor: float = 1e-8,
) -> np.ndarray:
    """Return a symmetric positive-definite correlation matrix.

    A small shrinkage toward the identity is necessary here because the two
    inflow series are extremely correlated and only 31 historical years are
    available for estimating a 24-dimensional matrix.
    """

    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("相关矩阵必须是方阵")
    if not np.isfinite(matrix).all():
        raise ValueError("相关矩阵包含非有限值")

    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    matrix = (1.0 - regularization) * matrix + regularization * np.eye(
        matrix.shape[0]
    )

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(matrix))
    matrix = matrix / np.outer(scale, scale)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _symmetric_matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * np.power(eigenvalues, exponent)) @ eigenvectors.T


def prepare_joint_monthly_model(
    history: MonthlyHistory,
    regularization: float,
) -> JointMonthlyModel:
    """Estimate monthly marginal moments and the joint 24-D log correlation."""

    cube = np.asarray(history.cube, dtype=float)
    if cube.ndim != 3 or cube.shape[1] != 12 or cube.shape[2] != 2:
        raise ValueError("历史数据形状必须为 [年, 12, 2]")
    if np.any(cube <= 0.0) or not np.isfinite(cube).all():
        raise ValueError("对数生成模型要求所有历史流量为有限正数")

    log_cube = np.log(cube)
    log_flat = log_cube.reshape(cube.shape[0], -1)
    log_mean = log_flat.mean(axis=0)
    log_sd = log_flat.std(axis=0, ddof=1)
    if np.any(log_sd <= 1e-12):
        raise ValueError("至少一个站点月份的对数流量方差为零")

    standardized_flat = (log_flat - log_mean) / log_sd
    standardized_log_cube = standardized_flat.reshape(cube.shape)

    block_start_month_index = _choose_weakest_transition_seam(
        standardized_log_cube
    )
    component_month_order = np.concatenate(
        [
            np.arange(block_start_month_index, 12),
            np.arange(0, block_start_month_index),
        ]
    )
    historical_blocks = _build_rotated_blocks(
        standardized_log_cube,
        block_start_month_index,
    )
    observed_correlation = np.corrcoef(
        historical_blocks.reshape(historical_blocks.shape[0], -1),
        rowvar=False,
    )
    target_correlation = stabilize_correlation(
        observed_correlation,
        regularization=regularization,
    )

    historical_mean = cube.mean(axis=0)
    historical_sd = cube.std(axis=0, ddof=1)
    if np.any(historical_sd <= 0.0):
        raise ValueError("至少一个站点月份的历史标准差不为正")

    return JointMonthlyModel(
        standardized_log_cube=standardized_log_cube,
        historical_mean=historical_mean,
        historical_sd=historical_sd,
        observed_log_correlation=observed_correlation,
        target_log_correlation=target_correlation,
        regularization=regularization,
        station_names=history.station_names,
        historical_years=history.calendar_years.copy(),
        block_start_month_index=block_start_month_index,
        component_month_order=component_month_order,
    )


def _transition_pairs(
    standardized_cube: np.ndarray,
    month_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    next_month = (month_index + 1) % 12
    if next_month == 0:
        return standardized_cube[:-1, month_index, :], standardized_cube[1:, 0, :]
    return standardized_cube[:, month_index, :], standardized_cube[:, next_month, :]


def _choose_weakest_transition_seam(standardized_cube: np.ndarray) -> int:
    """Place independent-block joins at the weakest observed month transition.

    The returned value is the first month inside a rotated 12-month block.  For
    the supplied Lancang data the weakest join is May->June, so blocks run from
    June through the following May and the strong Dec->Jan relation remains
    inside each reconstructed 24-dimensional vector.
    """

    scores: list[float] = []
    for month_index in range(12):
        previous, following = _transition_pairs(standardized_cube, month_index)
        station_correlations = [
            abs(float(np.corrcoef(previous[:, site], following[:, site])[0, 1]))
            for site in range(standardized_cube.shape[2])
        ]
        scores.append(float(np.mean(station_correlations)))
    weakest_transition = int(np.argmin(scores))
    return (weakest_transition + 1) % 12


def _build_rotated_blocks(
    cube: np.ndarray,
    block_start_month_index: int,
) -> np.ndarray:
    """Build circular overlapping blocks such as June(t) through May(t+1).

    Circular wrapping retains all historical years for every calendar month.
    Only the final block uses the first historical year as its successor; this
    contributes one transparent wrap pair while avoiding endpoint-induced loss
    of important spatial observations.
    """

    if block_start_month_index == 0:
        return cube.copy()
    following_year = np.roll(cube, shift=-1, axis=0)
    return np.concatenate(
        [
            cube[:, block_start_month_index:, :],
            following_year[:, :block_start_month_index, :],
        ],
        axis=1,
    )


def _sample_paired_months(
    standardized_cube: np.ndarray,
    historical_year_indices: np.ndarray,
    month_order: np.ndarray,
) -> np.ndarray:
    """Sample both sites from the same historical year for every month."""

    if historical_year_indices.ndim != 2 or historical_year_indices.shape[1] != 12:
        raise ValueError("历史年抽样矩阵形状必须为 [合成年数, 12]")
    month_indices = np.asarray(month_order, dtype=int)[None, :]
    sampled = standardized_cube[historical_year_indices, month_indices, :]
    return sampled.reshape(historical_year_indices.shape[0], -1)


def _coefficient_of_variation(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / values.mean())


def _positive_series_with_exact_moments(
    latent_values: np.ndarray,
    target_mean: float,
    target_sd: float,
) -> np.ndarray:
    """Map a latent score monotonically to positive values with exact moments."""

    if target_mean <= 0.0 or target_sd <= 0.0:
        raise ValueError("目标均值和标准差必须为正")
    latent = np.asarray(latent_values, dtype=float)
    latent_sd = latent.std(ddof=1)
    if latent.size < 2 or not np.isfinite(latent).all() or latent_sd <= 1e-12:
        raise ValueError("潜变量无法用于矩匹配")
    latent = (latent - latent.mean()) / latent_sd
    target_cv = target_sd / target_mean

    def cv_error(log_scale: float) -> float:
        exponent = log_scale * latent
        weights = np.exp(exponent - exponent.max())
        return _coefficient_of_variation(weights) - target_cv

    upper = 1.0
    while cv_error(upper) < 0.0 and upper < 128.0:
        upper *= 2.0
    if cv_error(upper) < 0.0:
        raise RuntimeError("当前有限样本无法达到指定的变异系数")

    lower = 0.0
    for _ in range(120):
        midpoint = (lower + upper) / 2.0
        if cv_error(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    log_scale = (lower + upper) / 2.0
    exponent = log_scale * latent
    weights = np.exp(exponent - exponent.max())
    values = weights * (target_mean / weights.mean())

    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise RuntimeError("矩匹配后出现非有限或非正流量")
    return values


def generate_scenario(
    model: JointMonthlyModel,
    synthetic_years: int,
    seed: int,
    mean_change_pct: np.ndarray,
    sd_change_pct: np.ndarray,
    minimum_calibration_samples: int,
    calibration_multiplier: int,
) -> GeneratedScenario:
    """Generate one independent scenario while preserving the joint structure."""

    if synthetic_years < 2:
        raise ValueError("synthetic_years 必须至少为 2")
    expected_shape = (12, len(model.station_names))
    if mean_change_pct.shape != expected_shape or sd_change_pct.shape != expected_shape:
        raise ValueError(f"目标变化矩阵必须为 {expected_shape}")
    if np.any(mean_change_pct <= -100.0) or np.any(sd_change_pct <= -100.0):
        raise ValueError("均值/标准差变化必须大于 -100%")

    historical_year_count = model.standardized_log_cube.shape[0]
    rng = np.random.default_rng(seed)
    generated_block_count = synthetic_years + 1
    sampled_indices = rng.integers(
        0,
        historical_year_count,
        size=(generated_block_count, 12),
    )
    resampled_raw = _sample_paired_months(
        model.standardized_log_cube,
        sampled_indices,
        model.component_month_order,
    )

    calibration_count = max(
        minimum_calibration_samples,
        synthetic_years * calibration_multiplier,
    )
    calibration_rng = np.random.default_rng(seed + 1_000_003)
    calibration_indices = calibration_rng.integers(
        0,
        historical_year_count,
        size=(calibration_count, 12),
    )
    calibration_raw = _sample_paired_months(
        model.standardized_log_cube,
        calibration_indices,
        model.component_month_order,
    )
    calibration_mean = calibration_raw.mean(axis=0)
    calibration_sd = calibration_raw.std(axis=0, ddof=1)
    if np.any(calibration_sd <= 1e-12):
        raise RuntimeError("校准抽样出现零方差")

    calibration_standardized = (
        calibration_raw - calibration_mean
    ) / calibration_sd
    observed_base_correlation = np.corrcoef(
        calibration_standardized,
        rowvar=False,
    )
    base_correlation = stabilize_correlation(
        observed_base_correlation,
        regularization=model.regularization,
    )

    resampled = (resampled_raw - calibration_mean) / calibration_sd
    base_inverse_sqrt = _symmetric_matrix_power(base_correlation, -0.5)
    target_sqrt = _symmetric_matrix_power(model.target_log_correlation, 0.5)
    latent_flat = resampled @ base_inverse_sqrt @ target_sqrt

    latent_blocks = latent_flat.reshape(generated_block_count, 12, len(model.station_names))
    stitched_latent = latent_blocks.reshape(-1, len(model.station_names))
    offset_to_january = (12 - model.block_start_month_index) % 12
    latent_calendar = stitched_latent[
        offset_to_january : offset_to_january + synthetic_years * 12
    ].reshape(synthetic_years, 12, len(model.station_names))

    target_mean = model.historical_mean * (1.0 + mean_change_pct / 100.0)
    target_sd = model.historical_sd * (1.0 + sd_change_pct / 100.0)
    flat_target_mean = target_mean.reshape(-1)
    flat_target_sd = target_sd.reshape(-1)

    latent_calendar_flat = latent_calendar.reshape(synthetic_years, -1)
    generated_flat = np.empty_like(latent_calendar_flat)
    for component in range(latent_calendar_flat.shape[1]):
        generated_flat[:, component] = _positive_series_with_exact_moments(
            latent_calendar_flat[:, component],
            target_mean=float(flat_target_mean[component]),
            target_sd=float(flat_target_sd[component]),
        )

    values = generated_flat.reshape(synthetic_years, 12, len(model.station_names))
    generated_log_cube = np.log(values)
    generated_log_mean = generated_log_cube.mean(axis=0)
    generated_log_sd = generated_log_cube.std(axis=0, ddof=1)
    standardized_generated_log_cube = (
        generated_log_cube - generated_log_mean
    ) / generated_log_sd
    generated_rotated_blocks = _build_rotated_blocks(
        standardized_generated_log_cube,
        model.block_start_month_index,
    )
    latent_correlation = np.corrcoef(
        generated_rotated_blocks.reshape(generated_rotated_blocks.shape[0], -1),
        rowvar=False,
    )

    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise RuntimeError("生成结果出现非有限或非正流量")

    return GeneratedScenario(
        values=values,
        sampled_historical_year_indices=sampled_indices,
        target_mean=target_mean,
        target_sd=target_sd,
        base_log_correlation=base_correlation,
        latent_log_correlation=latent_correlation,
    )
