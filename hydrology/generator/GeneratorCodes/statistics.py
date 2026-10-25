"""Quality-control metrics for historical and synthetic monthly inflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .joint_monthly import GeneratedScenario, JointMonthlyModel


MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


@dataclass(frozen=True)
class QCBundle:
    report: dict
    summary: pd.DataFrame
    monthly_statistics: pd.DataFrame
    spatial_correlations: pd.DataFrame
    cross_lag_correlations: pd.DataFrame
    autocorrelations: pd.DataFrame
    adjacent_month_correlations: pd.DataFrame
    historical_log_correlation: pd.DataFrame
    model_log_correlation: pd.DataFrame
    synthetic_log_correlation: pd.DataFrame


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size != right.size or left.size < 2:
        raise ValueError("相关系数要求长度相等且至少有两个样本")
    return float(np.corrcoef(left, right)[0, 1])


def _monthly_standardized_anomalies(
    cube: np.ndarray,
    log_transform: bool = False,
) -> np.ndarray:
    values = np.log(cube) if log_transform else np.asarray(cube, dtype=float)
    means = values.mean(axis=0)
    sds = values.std(axis=0, ddof=1)
    if np.any(sds <= 0.0):
        raise ValueError("去季节标准化遇到零标准差")
    return (values - means) / sds


def spatial_metrics(cube: np.ndarray) -> dict:
    """Spatial Pearson metrics for a [year, month, 2] cube.

    ``deseasonalized_pearson`` uses calendar-month z-score anomalies: each
    station-month series is centered by its own monthly mean and divided by its
    own sample standard deviation (ddof=1) before all months are concatenated.
    """

    values = np.asarray(cube, dtype=float)
    anomalies = _monthly_standardized_anomalies(values)
    log_anomalies = _monthly_standardized_anomalies(values, log_transform=True)
    per_month = {
        MONTH_NAMES[month]: _pearson(values[:, month, 0], values[:, month, 1])
        for month in range(12)
    }
    return {
        "overall_pearson": _pearson(values[:, :, 0].ravel(), values[:, :, 1].ravel()),
        "deseasonalized_pearson": _pearson(
            anomalies[:, :, 0].ravel(), anomalies[:, :, 1].ravel()
        ),
        "log_deseasonalized_pearson": _pearson(
            log_anomalies[:, :, 0].ravel(), log_anomalies[:, :, 1].ravel()
        ),
        "per_month_pearson": per_month,
    }


def _lagged_pair(
    left: np.ndarray,
    right: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return left[:-lag], right[lag:]
    if lag < 0:
        return left[-lag:], right[:lag]
    return left, right


def _cross_lag_frame(
    historical_cube: np.ndarray,
    synthetic_cube: np.ndarray,
    maximum_lag: int = 12,
) -> pd.DataFrame:
    history_z = _monthly_standardized_anomalies(historical_cube).reshape(-1, 2)
    synthetic_z = _monthly_standardized_anomalies(synthetic_cube).reshape(-1, 2)
    rows = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        hist_left, hist_right = _lagged_pair(history_z[:, 0], history_z[:, 1], lag)
        syn_left, syn_right = _lagged_pair(
            synthetic_z[:, 0], synthetic_z[:, 1], lag
        )
        rows.append(
            {
                "LagMonths": lag,
                "Definition": "corr(Xiaowan[t], NuozhaduInterval[t+lag])",
                "Historical": _pearson(hist_left, hist_right),
                "Synthetic": _pearson(syn_left, syn_right),
            }
        )
    return pd.DataFrame(rows)


def _autocorrelation_frame(
    historical_cube: np.ndarray,
    synthetic_cube: np.ndarray,
    station_names: tuple[str, ...],
    maximum_lag: int = 24,
) -> pd.DataFrame:
    history_z = _monthly_standardized_anomalies(historical_cube).reshape(-1, 2)
    synthetic_z = _monthly_standardized_anomalies(synthetic_cube).reshape(-1, 2)
    rows = []
    for station_index, station_name in enumerate(station_names):
        for lag in range(1, maximum_lag + 1):
            rows.append(
                {
                    "Station": station_name,
                    "LagMonths": lag,
                    "Historical": _pearson(
                        history_z[:-lag, station_index],
                        history_z[lag:, station_index],
                    ),
                    "Synthetic": _pearson(
                        synthetic_z[:-lag, station_index],
                        synthetic_z[lag:, station_index],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _transition_values(
    cube: np.ndarray,
    from_month: int,
    station_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    anomalies = _monthly_standardized_anomalies(cube, log_transform=True)
    to_month = (from_month + 1) % 12
    if to_month == 0:
        return anomalies[:-1, from_month, station_index], anomalies[1:, 0, station_index]
    return anomalies[:, from_month, station_index], anomalies[:, to_month, station_index]


def _adjacent_month_frame(
    historical_cube: np.ndarray,
    synthetic_cube: np.ndarray,
    station_names: tuple[str, ...],
    block_start_month_index: int,
) -> pd.DataFrame:
    rows = []
    seam_from_month = (block_start_month_index - 1) % 12
    for station_index, station_name in enumerate(station_names):
        for from_month in range(12):
            hist_left, hist_right = _transition_values(
                historical_cube, from_month, station_index
            )
            syn_left, syn_right = _transition_values(
                synthetic_cube, from_month, station_index
            )
            rows.append(
                {
                    "Station": station_name,
                    "FromMonth": MONTH_NAMES[from_month],
                    "ToMonth": MONTH_NAMES[(from_month + 1) % 12],
                    "IsBlockSeam": from_month == seam_from_month,
                    "HistoricalLogZPearson": _pearson(hist_left, hist_right),
                    "SyntheticLogZPearson": _pearson(syn_left, syn_right),
                }
            )
    return pd.DataFrame(rows)


def _matrix_frame(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _distribution_summary(cube: np.ndarray, station_names: tuple[str, ...]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for station_index, station_name in enumerate(station_names):
        values = cube[:, :, station_index].ravel()
        annual_mean = cube[:, :, station_index].mean(axis=1)
        summary[station_name] = {
            "minimum": float(values.min()),
            "p01": float(np.quantile(values, 0.01)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(values.max()),
            "overall_mean": float(values.mean()),
            "annual_mean_sd": float(annual_mean.std(ddof=1)),
        }
    ratio = cube[:, :, 1] / cube[:, :, 0]
    summary["nuozhadu_interval_to_xiaowan_ratio"] = {
        "minimum": float(ratio.min()),
        "p05": float(np.quantile(ratio, 0.05)),
        "median": float(np.median(ratio)),
        "mean": float(ratio.mean()),
        "p95": float(np.quantile(ratio, 0.95)),
        "maximum": float(ratio.max()),
    }
    return summary


def build_qc_bundle(
    scenario_name: str,
    seed: int,
    historical_cube: np.ndarray,
    generated: GeneratedScenario,
    model: JointMonthlyModel,
    mean_change_pct: np.ndarray,
    sd_change_pct: np.ndarray,
    tolerances: dict[str, float],
) -> QCBundle:
    """Calculate auditable statistics and pass/fail checks for one scenario."""

    synthetic_cube = generated.values
    historical_spatial = spatial_metrics(historical_cube)
    synthetic_spatial = spatial_metrics(synthetic_cube)

    monthly_rows = []
    synthetic_mean = synthetic_cube.mean(axis=0)
    synthetic_sd = synthetic_cube.std(axis=0, ddof=1)
    for station_index, station_name in enumerate(model.station_names):
        for month in range(12):
            historical_mean = float(model.historical_mean[month, station_index])
            historical_sd = float(model.historical_sd[month, station_index])
            target_mean = float(generated.target_mean[month, station_index])
            target_sd = float(generated.target_sd[month, station_index])
            actual_mean = float(synthetic_mean[month, station_index])
            actual_sd = float(synthetic_sd[month, station_index])
            monthly_rows.append(
                {
                    "Station": station_name,
                    "Month": month + 1,
                    "MonthName": MONTH_NAMES[month],
                    "HistoricalMean_m3s": historical_mean,
                    "HistoricalSD_m3s": historical_sd,
                    "RequestedMeanChange_pct": float(mean_change_pct[month, station_index]),
                    "RequestedSDChange_pct": float(sd_change_pct[month, station_index]),
                    "TargetMean_m3s": target_mean,
                    "TargetSD_m3s": target_sd,
                    "SyntheticMean_m3s": actual_mean,
                    "SyntheticSD_m3s": actual_sd,
                    "MeanRelativeError": abs(actual_mean - target_mean) / target_mean,
                    "SDRelativeError": abs(actual_sd - target_sd) / target_sd,
                }
            )
    monthly_statistics = pd.DataFrame(monthly_rows)

    spatial_rows = [
        {
            "Metric": "Overall Pearson (raw monthly flows)",
            "Historical": historical_spatial["overall_pearson"],
            "Synthetic": synthetic_spatial["overall_pearson"],
        },
        {
            "Metric": "Deseasonalized Pearson (calendar-month z anomalies)",
            "Historical": historical_spatial["deseasonalized_pearson"],
            "Synthetic": synthetic_spatial["deseasonalized_pearson"],
        },
        {
            "Metric": "Log-deseasonalized Pearson (calendar-month log-z anomalies)",
            "Historical": historical_spatial["log_deseasonalized_pearson"],
            "Synthetic": synthetic_spatial["log_deseasonalized_pearson"],
        },
    ]
    for month_name in MONTH_NAMES:
        spatial_rows.append(
            {
                "Metric": f"{month_name} same-month Pearson",
                "Historical": historical_spatial["per_month_pearson"][month_name],
                "Synthetic": synthetic_spatial["per_month_pearson"][month_name],
            }
        )
    spatial_correlations = pd.DataFrame(spatial_rows)
    spatial_correlations["AbsoluteError"] = (
        spatial_correlations["Synthetic"] - spatial_correlations["Historical"]
    ).abs()

    cross_lag = _cross_lag_frame(historical_cube, synthetic_cube)
    autocorrelations = _autocorrelation_frame(
        historical_cube,
        synthetic_cube,
        model.station_names,
    )
    adjacent_month = _adjacent_month_frame(
        historical_cube,
        synthetic_cube,
        model.station_names,
        model.block_start_month_index,
    )

    matrix_difference = generated.latent_log_correlation - model.target_log_correlation
    upper_triangle = np.triu_indices(matrix_difference.shape[0], k=1)
    upper_errors = np.abs(matrix_difference[upper_triangle])
    log_correlation_mae = float(np.mean(upper_errors))
    log_correlation_rmse = float(
        np.sqrt(np.mean(np.square(matrix_difference[upper_triangle])))
    )
    log_correlation_p95_abs_error = float(np.quantile(upper_errors, 0.95))
    log_correlation_max_abs_error = float(np.max(upper_errors))
    mean_max_error = float(monthly_statistics["MeanRelativeError"].max())
    sd_max_error = float(monthly_statistics["SDRelativeError"].max())
    overall_error = abs(
        synthetic_spatial["overall_pearson"] - historical_spatial["overall_pearson"]
    )
    deseasonalized_error = abs(
        synthetic_spatial["deseasonalized_pearson"]
        - historical_spatial["deseasonalized_pearson"]
    )
    per_month_errors = [
        abs(
            synthetic_spatial["per_month_pearson"][month]
            - historical_spatial["per_month_pearson"][month]
        )
        for month in MONTH_NAMES
    ]
    monthly_spatial_max_error = float(max(per_month_errors))

    checks = {
        "expected_shape": list(synthetic_cube.shape),
        "all_finite": bool(np.isfinite(synthetic_cube).all()),
        "all_positive": bool(np.all(synthetic_cube > 0.0)),
        "monthly_mean_max_relative_error": mean_max_error,
        "monthly_sd_max_relative_error": sd_max_error,
        "overall_spatial_abs_error": overall_error,
        "deseasonalized_spatial_abs_error": deseasonalized_error,
        "monthly_spatial_max_abs_error": monthly_spatial_max_error,
        "rotated_24d_log_correlation_mae": log_correlation_mae,
        "rotated_24d_log_correlation_rmse": log_correlation_rmse,
        "rotated_24d_log_correlation_p95_abs_error": log_correlation_p95_abs_error,
        "rotated_24d_log_correlation_max_abs_error": log_correlation_max_abs_error,
    }
    passed = bool(
        checks["all_finite"]
        and checks["all_positive"]
        and mean_max_error <= tolerances["monthly_mean_relative_error"]
        and sd_max_error <= tolerances["monthly_sd_relative_error"]
        and overall_error <= tolerances["overall_spatial_abs_error"]
        and deseasonalized_error <= tolerances["deseasonalized_spatial_abs_error"]
        and monthly_spatial_max_error <= tolerances["monthly_spatial_max_abs_error"]
        and log_correlation_rmse <= tolerances["log_correlation_rmse"]
    )

    summary = pd.DataFrame(
        [
            {
                "Scenario": scenario_name,
                "Seed": seed,
                "QC_Passed": passed,
                "HistoricalOverallPearson": historical_spatial["overall_pearson"],
                "SyntheticOverallPearson": synthetic_spatial["overall_pearson"],
                "HistoricalDeseasonalizedPearson": historical_spatial[
                    "deseasonalized_pearson"
                ],
                "SyntheticDeseasonalizedPearson": synthetic_spatial[
                    "deseasonalized_pearson"
                ],
                "MonthlyMeanMaxRelativeError": mean_max_error,
                "MonthlySDMaxRelativeError": sd_max_error,
                "Rotated24DLogCorrelationRMSE": log_correlation_rmse,
                "BlockStartMonth": MONTH_NAMES[model.block_start_month_index],
            }
        ]
    )

    labels = [
        f"{MONTH_NAMES[int(month)]}:{station}"
        for month in model.component_month_order
        for station in model.station_names
    ]
    observed_eigenvalues = np.linalg.eigvalsh(model.observed_log_correlation)
    target_eigenvalues = np.linalg.eigvalsh(model.target_log_correlation)
    generated_eigenvalues = np.linalg.eigvalsh(generated.latent_log_correlation)
    report = {
        "scenario": scenario_name,
        "random_seed": seed,
        "qc_passed": passed,
        "units": "m3/s monthly mean flow",
        "deseasonalized_definition": (
            "For each station and calendar month, subtract that month's sample mean "
            "and divide by that month's sample standard deviation (ddof=1), then "
            "calculate Pearson correlation after concatenating all months."
        ),
        "block_definition": {
            "start_month": MONTH_NAMES[model.block_start_month_index],
            "month_order": [MONTH_NAMES[int(m)] for m in model.component_month_order],
            "seam_reason": (
                "The 24-D reconstruction uses overlapping rotated years so Dec-to-Jan "
                "is internal; the independent-block seam is placed at the weakest "
                "historical adjacent-month transition."
            ),
        },
        "checks": checks,
        "tolerances": tolerances,
        "historical_spatial_correlation": historical_spatial,
        "synthetic_spatial_correlation": synthetic_spatial,
        "historical_distribution": _distribution_summary(
            historical_cube, model.station_names
        ),
        "synthetic_distribution": _distribution_summary(
            synthetic_cube, model.station_names
        ),
        "correlation_matrix_conditioning": {
            "observed_min_eigenvalue": float(observed_eigenvalues.min()),
            "observed_condition_number": float(np.linalg.cond(model.observed_log_correlation)),
            "regularization": model.regularization,
            "model_min_eigenvalue": float(target_eigenvalues.min()),
            "model_condition_number": float(np.linalg.cond(model.target_log_correlation)),
            "synthetic_min_eigenvalue": float(generated_eigenvalues.min()),
            "synthetic_condition_number": float(np.linalg.cond(generated.latent_log_correlation)),
        },
    }

    return QCBundle(
        report=report,
        summary=summary,
        monthly_statistics=monthly_statistics,
        spatial_correlations=spatial_correlations,
        cross_lag_correlations=cross_lag,
        autocorrelations=autocorrelations,
        adjacent_month_correlations=adjacent_month,
        historical_log_correlation=_matrix_frame(
            model.observed_log_correlation, labels
        ),
        model_log_correlation=_matrix_frame(model.target_log_correlation, labels),
        synthetic_log_correlation=_matrix_frame(
            generated.latent_log_correlation, labels
        ),
    )
