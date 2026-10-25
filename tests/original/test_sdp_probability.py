"""Statistical and artifact validation for training-only SDP inflow marginals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cascade_reservoir_benchmark.environment import CascadeSpecError
from cascade_reservoir_benchmark.sdp_probability import MonthlyInflowModel, PROBABILITY_FILES, SOURCES


def training_frame(years: int = 400) -> pd.DataFrame:
    periods = pd.period_range("2009-01", periods=12 * years, freq="M")
    # Deliberately unequal frequencies, source distributions, and seasonal scales.
    samples = np.zeros(years)
    samples[-max(1, years // 4):] = 100.0
    return pd.DataFrame({
        "date": [f"{period}-01" for period in periods],
        SOURCES[0]: np.repeat(samples, 12) + np.tile(np.arange(1, 13), years),
        SOURCES[1]: np.repeat(samples[::-1] * 2.0, 12) + np.tile(np.arange(1, 13) * 3.0, years),
    })


class MonthlyInflowModelTests(unittest.TestCase):
    def test_equal_width_empirical_marginals_and_calendar(self):
        model = MonthlyInflowModel.fit(training_frame())
        self.assertEqual(model.edges.shape, (2, 12, 11))
        np.testing.assert_array_equal(np.diff(model.edges[0], axis=1), 10.0)
        np.testing.assert_array_equal(np.diff(model.edges[1], axis=1), 20.0)
        np.testing.assert_array_equal(model.counts[:, :, 0], 300)
        np.testing.assert_array_equal(model.counts[:, :, 9], 100)
        np.testing.assert_array_equal(model.counts[:, :, 1:9], 0)
        np.testing.assert_array_equal(model.probabilities[:, :, 0], 0.75)
        np.testing.assert_array_equal(model.probabilities[:, :, 9], 0.25)
        np.testing.assert_allclose(model.probabilities.sum(axis=2), 1.0)
        np.testing.assert_array_equal(model.representatives, (model.edges[:, :, :-1] + model.edges[:, :, 1:]) / 2.0)
        self.assertEqual(model.mean_days[1], 28.2425)
        self.assertEqual(model.modal_classes(1), (0, 0))

    def test_classification_edges_clipping_and_source_order(self):
        model = MonthlyInflowModel.fit(training_frame())
        self.assertEqual(model.classify(1, (1.0, 3.0)), (0, 0))
        self.assertEqual(model.classify(1, (11.0, 23.0)), (1, 1))
        self.assertEqual(model.classify(1, (101.0, 203.0)), (9, 9))
        self.assertEqual(model.classify(1, (0.0, 1000.0)), (0, 9))
        self.assertEqual(model.classify(12, (12.0, 236.0)), (0, 9))
        for month, flows in [(0, (1, 2)), (13, (1, 2)), (1.0, (1, 2)), (1, (np.nan, 2)), (1, (-1, 2)), (1, (1,))]:
            with self.subTest(month=month, flows=flows), self.assertRaises(CascadeSpecError):
                model.classify(month, flows)

    def test_constant_month_is_deterministic_and_keeps_ten_classes(self):
        frame = training_frame(2)
        frame.loc[:, SOURCES[0]] = 0.0
        frame.loc[:, SOURCES[1]] = 7.5
        model = MonthlyInflowModel.fit(frame, expected_years=2)
        np.testing.assert_array_equal(model.counts[:, :, 0], 2)
        np.testing.assert_array_equal(model.counts[:, :, 1:], 0)
        np.testing.assert_array_equal(model.representatives[0], 0.0)
        np.testing.assert_array_equal(model.representatives[1], 7.5)
        self.assertEqual(model.classify(6, (999, 0)), (0, 0))

    def test_default_requires_exact_400_complete_years(self):
        with self.assertRaises(CascadeSpecError):
            MonthlyInflowModel.fit(training_frame(2))
        frame = training_frame(2)
        for invalid in [frame.iloc[:-1], frame.iloc[::-1], frame.assign(date=["2009-01-01"] * 24), frame.assign(xiaowan_inflow_m3s=np.inf), frame.assign(nuozhadu_local_inflow_m3s=-1)]:
            with self.subTest(), self.assertRaises(CascadeSpecError):
                MonthlyInflowModel.fit(invalid, expected_years=2)
        for years in [True, 0, -1, 2.0]:
            with self.subTest(years=years), self.assertRaises(CascadeSpecError):
                MonthlyInflowModel.fit(frame, expected_years=years)

    def test_hash_ignores_extra_columns_and_tracks_source_values(self):
        frame = training_frame(2)
        original = MonthlyInflowModel.fit(frame, 2)
        reordered = MonthlyInflowModel.fit(frame.assign(unused="ignored").loc[:, [SOURCES[1], "unused", "date", SOURCES[0]]], 2)
        self.assertEqual(original.metadata["training_sha256"], reordered.metadata["training_sha256"])
        frame.loc[0, SOURCES[0]] += 0.001
        self.assertNotEqual(original.metadata["training_sha256"], MonthlyInflowModel.fit(frame, 2).metadata["training_sha256"])

    def test_saved_model_round_trips_without_training_data(self):
        model = MonthlyInflowModel.fit(training_frame())
        with tempfile.TemporaryDirectory() as folder:
            paths = model.save(folder)
            for source in SOURCES:
                self.assertEqual(len(pd.read_csv(paths[source])), 120)
            loaded = MonthlyInflowModel.load(folder)
            for name in ["edges", "counts", "probabilities", "representatives", "mean_days"]:
                np.testing.assert_array_equal(getattr(model, name), getattr(loaded, name))
            self.assertEqual(loaded.metadata["training_sha256"], model.metadata["training_sha256"])

    def test_modified_probability_file_fails_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            MonthlyInflowModel.fit(training_frame(2), 2).save(folder)
            path = Path(folder) / PROBABILITY_FILES[0]
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(CascadeSpecError, "checksum"):
                MonthlyInflowModel.load(folder)

    def test_statistics_still_validated_when_file_checksum_is_updated(self):
        with tempfile.TemporaryDirectory() as folder:
            MonthlyInflowModel.fit(training_frame(2), 2).save(folder)
            path = Path(folder) / PROBABILITY_FILES[0]
            frame = pd.read_csv(path)
            frame.loc[0, "probability"] = 0.123
            frame.to_csv(path, index=False)
            metadata_path = Path(folder) / "probability_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["probability_files"][SOURCES[0]]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(CascadeSpecError, "empirical class counts"):
                MonthlyInflowModel.load(folder)

    def test_calendar_metadata_is_checked_against_training_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            MonthlyInflowModel.fit(training_frame(2), 2).save(folder)
            path = Path(folder) / "probability_metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["mean_days"][1] = 30.4
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(CascadeSpecError, "day lengths"):
                MonthlyInflowModel.load(folder)


if __name__ == "__main__":
    unittest.main()
