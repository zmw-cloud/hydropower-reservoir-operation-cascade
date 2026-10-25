"""Monthly four-input, two-output ISO policy models for the cascade system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURE_NAMES = (
    "start_level_1_m",
    "start_level_2_m",
    "inflow_1_m3s",
    "inflow_2_local_m3s",
)
TARGET_NAMES = ("target_level_1_m", "target_level_2_m")


class ScaledRBFRegressor:
    """Multi-output RBF interpolator with fitted X and Y standardization."""

    def __init__(
        self,
        *,
        kernel: str,
        smoothing: float,
        epsilon: float | None = None,
        neighbors: int | None = None,
        degree: int | None = None,
    ) -> None:
        self.kernel = str(kernel)
        self.smoothing = float(smoothing)
        self.epsilon = None if epsilon is None else float(epsilon)
        self.neighbors = None if neighbors is None else int(neighbors)
        self.degree = None if degree is None else int(degree)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.interpolator: RBFInterpolator | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScaledRBFRegressor":
        x_scaled = self.x_scaler.fit_transform(np.asarray(x, dtype=float))
        y_scaled = self.y_scaler.fit_transform(np.asarray(y, dtype=float))
        kwargs: dict[str, Any] = {
            "kernel": self.kernel,
            "smoothing": self.smoothing,
        }
        if self.epsilon is not None:
            kwargs["epsilon"] = self.epsilon
        if self.neighbors is not None:
            kwargs["neighbors"] = self.neighbors
        if self.degree is not None:
            kwargs["degree"] = self.degree
        self.interpolator = RBFInterpolator(x_scaled, y_scaled, **kwargs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.interpolator is None:
            raise RuntimeError("RBF model has not been fitted.")
        x_scaled = self.x_scaler.transform(np.asarray(x, dtype=float))
        y_scaled = self.interpolator(x_scaled)
        return self.y_scaler.inverse_transform(y_scaled)


def _build_model(model_type: str, hyperparameters: Mapping[str, Any], random_seed: int):
    params = dict(hyperparameters)
    if model_type == "polynomial":
        degree = int(params.pop("degree", 3))
        if degree != 3:
            raise ValueError("The ISO polynomial degree is fixed at 3.")
        alpha = float(params.pop("alpha"))
        if params:
            raise ValueError(f"Unknown polynomial hyperparameters: {sorted(params)}")
        return Pipeline(
            [
                ("input_scaler", StandardScaler()),
                ("polynomial", PolynomialFeatures(degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    if model_type == "rbf":
        return ScaledRBFRegressor(**params)
    if model_type == "random_forest":
        configured_seed = int(params.pop("random_state", random_seed))
        return RandomForestRegressor(random_state=configured_seed, n_jobs=-1, **params)
    raise ValueError(f"Unsupported ISO model_type: {model_type!r}")


@dataclass
class MonthlyCascadeISOPolicy:
    """Twelve calendar-month models sharing one family and hyperparameter set."""

    model_type: str
    hyperparameters: dict[str, Any]
    random_seed: int
    month_models: dict[int, Any]
    month_domains: dict[int, dict[str, tuple[float, float]]]

    @classmethod
    def fit(
        cls,
        frame,
        *,
        model_type: str,
        hyperparameters: Mapping[str, Any],
        random_seed: int,
    ) -> "MonthlyCascadeISOPolicy":
        month_models: dict[int, Any] = {}
        month_domains: dict[int, dict[str, tuple[float, float]]] = {}
        for month in range(1, 13):
            monthly = frame.loc[frame["month"] == month]
            if monthly.empty:
                raise ValueError(f"Teacher dataset has no samples for calendar month {month}.")
            x = monthly.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
            y = monthly.loc[:, list(TARGET_NAMES)].to_numpy(dtype=float)
            model = _build_model(model_type, hyperparameters, random_seed)
            model.fit(x, y)
            month_models[month] = model
            month_domains[month] = {
                name: (float(np.min(x[:, index])), float(np.max(x[:, index])))
                for index, name in enumerate(FEATURE_NAMES)
            }
        return cls(
            model_type=str(model_type),
            hyperparameters=dict(hyperparameters),
            random_seed=int(random_seed),
            month_models=month_models,
            month_domains=month_domains,
        )

    def predict(self, month: int, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if month not in self.month_models:
            raise ValueError(f"No ISO model is available for month {month}.")
        row = np.asarray(features, dtype=float).reshape(1, len(FEATURE_NAMES))
        prediction = np.asarray(self.month_models[month].predict(row), dtype=float).reshape(-1)
        if prediction.shape != (len(TARGET_NAMES),) or not np.all(np.isfinite(prediction)):
            raise FloatingPointError(
                f"{self.model_type} produced an invalid month-{month} target: {prediction!r}"
            )
        domain = self.month_domains[month]
        outside = np.array(
            [
                row[0, index] < domain[name][0] or row[0, index] > domain[name][1]
                for index, name in enumerate(FEATURE_NAMES)
            ],
            dtype=bool,
        )
        return prediction, outside

    def scaler_metadata(self) -> dict[str, Any]:
        """Return JSON-safe fitted scaler parameters for reproducibility metadata."""
        result: dict[str, Any] = {}
        for month, model in self.month_models.items():
            entry: dict[str, Any] = {}
            if isinstance(model, Pipeline):
                scaler = model.named_steps["input_scaler"]
                entry["input_mean"] = scaler.mean_.tolist()
                entry["input_scale"] = scaler.scale_.tolist()
            elif isinstance(model, ScaledRBFRegressor):
                entry["input_mean"] = model.x_scaler.mean_.tolist()
                entry["input_scale"] = model.x_scaler.scale_.tolist()
                entry["target_mean"] = model.y_scaler.mean_.tolist()
                entry["target_scale"] = model.y_scaler.scale_.tolist()
            result[str(month)] = entry
        return result
