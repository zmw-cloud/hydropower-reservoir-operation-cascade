"""Training-only monthly inflow marginals for one-step-perfect-forecast SDP.

The two sources are XiaoWan natural inflow and NuoZhaDu interval inflow.
Routed XiaoWan releases never enter this statistical model. Class numbers
are zero based throughout, including the exported probability CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd

from .environment import CascadeSpecError, _as_month_period_index


SOURCES = ("xiaowan_inflow_m3s", "nuozhadu_local_inflow_m3s")
PROBABILITY_FILES = (
    "xiaowan_inflow_probabilities.csv",
    "nuozhadu_local_inflow_probabilities.csv",
)
CSV_COLUMNS = (
    "month", "inflow_class", "lower_bound_m3s", "upper_bound_m3s",
    "representative_inflow_m3s", "sample_count", "training_sample_count",
    "probability",
)
CONVENTIONS = {
    "class_numbering": "zero_based_0_to_9",
    "classification": "10_equal_width_intervals_from_training_month_min_max",
    "interval_convention": "left_closed_right_open_last_interval_right_closed",
    "constant_month_convention": "all_samples_in_class_0_all_representatives_equal_constant",
    "out_of_range_convention": "clip_to_first_or_last_class_constant_month_to_class_0",
    "representative_inflow": "interval_midpoint",
    "probability_definition": "class_sample_count_divided_by_training_samples_per_month",
    "source_dependence": "independent_product_of_two_monthly_marginals",
    "temporal_dependence": "monthly_marginals_do_not_condition_on_previous_classes",
    "calendar_convention": "training_mean_calendar_days_per_month_evaluation_uses_actual_calendar_days",
    "training_hash_normalization": "UTF-8 LF-terminated rows YYYY-MM,float.hex(source_1),float.hex(source_2)",
}


def _calendar_days(periods: pd.PeriodIndex) -> np.ndarray:
    return np.array([
        np.mean([period.days_in_month for period in periods if period.month == month])
        for month in range(1, 13)
    ], dtype=float)


@dataclass
class MonthlyInflowModel:
    """Two independent seasonal marginals fitted only on a training split.

    Arrays use source order ``SOURCES`` and zero-based month/class axes:
    ``edges[2, 12, 11]`` and ``probabilities/representatives/counts[2, 12, 10]``.
    Public classification methods receive calendar months 1 through 12.
    """

    edges: np.ndarray
    probabilities: np.ndarray
    representatives: np.ndarray
    counts: np.ndarray
    mean_days: np.ndarray
    metadata: dict

    n_inflow_classes: ClassVar[int] = 10
    sources: ClassVar[tuple[str, str]] = SOURCES

    @classmethod
    def fit(cls, frame: pd.DataFrame, expected_years: int = 400) -> "MonthlyInflowModel":
        """Fit ten equal-width classes per source/month, preserving empty classes."""
        if isinstance(expected_years, bool) or not isinstance(expected_years, (int, np.integer)) or expected_years <= 0:
            raise CascadeSpecError("expected_years must be a positive integer.")
        missing = {"date", *SOURCES} - set(frame.columns)
        if missing or frame.columns.duplicated().any():
            raise CascadeSpecError(f"Training inflows require unique date/source columns; missing: {sorted(missing)}.")
        if len(frame) != 12 * expected_years:
            raise CascadeSpecError(f"Expected {12 * expected_years} training months, got {len(frame)}.")
        periods = _as_month_period_index(frame["date"])
        if periods[0].month != 1 or periods[-1].month != 12:
            raise CascadeSpecError("SDP training must contain complete January-December calendar years.")
        if not periods.equals(pd.period_range(periods[0], periods[-1], freq="M")):
            raise CascadeSpecError("SDP training dates must be unique, ordered, and continuous monthly records.")
        flows = frame.loc[:, list(SOURCES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(flows)) or np.any(flows < 0.0):
            raise CascadeSpecError("Training source inflows must be finite and nonnegative.")
        edges = np.empty((2, 12, 11), dtype=float)
        representatives = np.empty((2, 12, 10), dtype=float)
        counts = np.zeros((2, 12, 10), dtype=np.int64)
        for month in range(1, 13):
            samples = flows[periods.month == month]
            if len(samples) != expected_years:
                raise CascadeSpecError(f"Month {month} requires exactly {expected_years} training samples.")
            for source in range(2):
                values = samples[:, source]
                boundaries = np.linspace(values.min(), values.max(), 11)
                edges[source, month - 1] = boundaries
                representatives[source, month - 1] = boundaries[:-1] + np.diff(boundaries) / 2.0
                if boundaries[0] == boundaries[-1]:
                    indices = np.zeros(len(values), dtype=int)
                else:
                    indices = np.clip(np.searchsorted(boundaries, values, side="right") - 1, 0, 9)
                counts[source, month - 1] = np.bincount(indices, minlength=10)
        digest = hashlib.sha256()
        for period, (upstream, local) in zip(periods, flows):
            digest.update(f"{period},{float(upstream).hex()},{float(local).hex()}\n".encode("utf-8"))
        metadata = {
            "schema_version": 1,
            "sources": list(SOURCES),
            "n_inflow_classes": 10,
            "expected_years": int(expected_years),
            "months": len(frame),
            "start_month": str(periods[0]),
            "end_month": str(periods[-1]),
            "training_sha256": digest.hexdigest(),
            **CONVENTIONS,
        }
        model = cls(edges, counts / float(expected_years), representatives, counts, _calendar_days(periods), metadata)
        model._validate()
        return model

    def _validate(self) -> None:
        """Reject malformed models before use or export, including edited files."""
        expected_shapes = {
            "edges": (2, 12, 11), "probabilities": (2, 12, 10),
            "representatives": (2, 12, 10), "counts": (2, 12, 10),
            "mean_days": (12,),
        }
        for name, shape in expected_shapes.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape or not np.issubdtype(array.dtype, np.number) or np.any(~np.isfinite(array)):
                raise CascadeSpecError(f"SDP probability {name} must be a finite numeric array of shape {shape}.")
        if not isinstance(self.metadata, dict):
            raise CascadeSpecError("SDP probability metadata must be an object.")
        for key, value in {"schema_version": 1, "sources": list(SOURCES), "n_inflow_classes": 10, **CONVENTIONS}.items():
            if self.metadata.get(key) != value:
                raise CascadeSpecError(f"Unsupported SDP probability metadata {key}.")
        years = self.metadata.get("expected_years")
        if isinstance(years, bool) or not isinstance(years, int) or years <= 0 or self.metadata.get("months") != 12 * years:
            raise CascadeSpecError("SDP probability metadata must specify a positive whole number of complete training years.")
        digest = self.metadata.get("training_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise CascadeSpecError("SDP probability metadata requires a normalized training SHA256.")
        try:
            periods = pd.period_range(self.metadata["start_month"], self.metadata["end_month"], freq="M")
        except (KeyError, ValueError, TypeError) as exc:
            raise CascadeSpecError("SDP probability metadata has invalid calendar bounds.") from exc
        if len(periods) != 12 * years or periods[0].month != 1 or periods[-1].month != 12:
            raise CascadeSpecError("SDP probability metadata calendar does not match its training years.")
        if not np.allclose(self.mean_days, _calendar_days(periods), rtol=0.0, atol=1.0e-12):
            raise CascadeSpecError("SDP monthly training day lengths disagree with the saved calendar.")
        if np.any(self.edges < 0.0) or np.any(np.diff(self.edges, axis=2) < 0.0):
            raise CascadeSpecError("SDP inflow boundaries must be ordered and nonnegative.")
        if np.any(self.counts < 0) or np.any(self.counts != np.floor(self.counts)) or np.any(self.counts.sum(axis=2) != years):
            raise CascadeSpecError("SDP class counts must be nonnegative integers summing to training years per month.")
        if not np.allclose(self.probabilities, self.counts / float(years), rtol=0.0, atol=1.0e-14):
            raise CascadeSpecError("SDP probabilities must equal empirical class counts / training years.")
        if np.any(self.probabilities < 0.0) or not np.allclose(self.probabilities.sum(axis=2), 1.0, rtol=0.0, atol=1.0e-14):
            raise CascadeSpecError("SDP marginal probabilities must be nonnegative and sum to one.")
        midpoints = self.edges[:, :, :-1] + np.diff(self.edges, axis=2) / 2.0
        if not np.allclose(self.representatives, midpoints, rtol=1.0e-14, atol=1.0e-14):
            raise CascadeSpecError("SDP representative inflows must be interval midpoints.")
        for source in range(2):
            for month in range(12):
                boundaries = self.edges[source, month]
                if not np.allclose(boundaries, np.linspace(boundaries[0], boundaries[-1], 11), rtol=1.0e-14, atol=1.0e-14):
                    raise CascadeSpecError("SDP inflow intervals must be equal-width.")
                if boundaries[0] == boundaries[-1] and (self.counts[source, month, 0] != years or np.any(self.counts[source, month, 1:] != 0)):
                    raise CascadeSpecError("A constant training month must put every sample in class 0.")

    @staticmethod
    def _month_index(month: int) -> int:
        if isinstance(month, bool) or not isinstance(month, (int, np.integer)) or not 1 <= month <= 12:
            raise CascadeSpecError("month must be an integer from 1 to 12.")
        return int(month) - 1

    def classify(self, month: int, flows: Sequence[float]) -> tuple[int, int]:
        """Classify observed natural/local inflows using training bounds only."""
        month_index = self._month_index(month)
        values = np.asarray(flows, dtype=float)
        if values.shape != (2,) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise CascadeSpecError("Observed source flows must contain two finite nonnegative values.")
        classes = []
        for source in range(2):
            boundaries = self.edges[source, month_index]
            index = 0 if boundaries[0] == boundaries[-1] else int(np.clip(np.searchsorted(boundaries, values[source], side="right") - 1, 0, 9))
            classes.append(index)
        return classes[0], classes[1]

    def modal_classes(self, month: int) -> tuple[int, int]:
        """Return deterministic first-on-tie training modal classes for a month."""
        index = self._month_index(month)
        return int(np.argmax(self.counts[0, index])), int(np.argmax(self.counts[1, index]))

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Export two 120-row CSVs and their conventions/provenance JSON."""
        self._validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        file_metadata = {}
        for source, (name, filename) in enumerate(zip(SOURCES, PROBABILITY_FILES)):
            rows = []
            for month in range(12):
                for inflow_class in range(10):
                    rows.append((month + 1, inflow_class, self.edges[source, month, inflow_class], self.edges[source, month, inflow_class + 1], self.representatives[source, month, inflow_class], int(self.counts[source, month, inflow_class]), self.metadata["expected_years"], self.probabilities[source, month, inflow_class]))
            path = directory / filename
            pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
            paths[name] = path
            file_metadata[name] = {"filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        metadata = {**self.metadata, "mean_days": self.mean_days.tolist(), "probability_files": file_metadata}
        paths["metadata"] = directory / "probability_metadata.json"
        paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return paths

    @classmethod
    def load(cls, directory: str | Path) -> "MonthlyInflowModel":
        """Load a saved model, checking hashes, complete grids, and statistics."""
        directory = Path(directory)
        try:
            metadata = json.loads((directory / "probability_metadata.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise CascadeSpecError("Invalid SDP probability metadata JSON.") from exc
        if not isinstance(metadata, dict):
            raise CascadeSpecError("SDP probability metadata must be an object.")
        file_metadata = metadata.get("probability_files")
        if not isinstance(file_metadata, dict) or set(file_metadata) != set(SOURCES):
            raise CascadeSpecError("SDP probability metadata must identify exactly the two natural/local sources.")
        edges = np.empty((2, 12, 11), dtype=float)
        probabilities = np.empty((2, 12, 10), dtype=float)
        representatives = np.empty((2, 12, 10), dtype=float)
        counts = np.empty((2, 12, 10), dtype=float)
        for source, (name, filename) in enumerate(zip(SOURCES, PROBABILITY_FILES)):
            details = file_metadata[name]
            if not isinstance(details, dict) or details.get("filename") != filename:
                raise CascadeSpecError(f"Unexpected probability filename for {name}.")
            path = directory / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != details.get("sha256"):
                raise CascadeSpecError(f"SDP probability file checksum mismatch: {filename}.")
            try:
                frame = pd.read_csv(path, float_precision="round_trip")
                if tuple(frame.columns) != CSV_COLUMNS or len(frame) != 120:
                    raise CascadeSpecError(f"Probability file {filename} must contain the standard 120-row schema.")
                numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
                raise CascadeSpecError(f"Cannot parse SDP probability file: {filename}.") from exc
            if np.any(~np.isfinite(numeric)):
                raise CascadeSpecError("SDP probability files must contain finite numeric values.")
            if not np.array_equal(numeric[:, 0], np.repeat(np.arange(1, 13), 10)) or not np.array_equal(numeric[:, 1], np.tile(np.arange(10), 12)):
                raise CascadeSpecError("SDP probability rows must be the ordered 12 months by 10 unique classes.")
            if np.any(numeric[:, 6] != metadata.get("expected_years")):
                raise CascadeSpecError("SDP probability sample denominators disagree with training metadata.")
            records = numeric.reshape(12, 10, 8)
            if not np.array_equal(records[:, :-1, 3], records[:, 1:, 2]):
                raise CascadeSpecError("SDP probability intervals must share consecutive boundaries exactly.")
            edges[source, :, :-1] = records[:, :, 2]
            edges[source, :, -1] = records[:, -1, 3]
            representatives[source] = records[:, :, 4]
            counts[source] = records[:, :, 5]
            probabilities[source] = records[:, :, 7]
        try:
            mean_days = np.asarray(metadata.get("mean_days"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise CascadeSpecError("Invalid SDP monthly calendar durations.") from exc
        model = cls(edges, probabilities, representatives, counts, mean_days, metadata)
        model._validate()
        model.counts = model.counts.astype(np.int64)
        return model
