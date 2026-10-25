"""Read and audit frozen paper results. No plotting or reservoir-model imports."""
from __future__ import annotations

import calendar
from datetime import datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/paper_results.json"


def native(path: Path) -> Path:
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path.absolute()))
    return path


def load_config(path: Path = DEFAULT_CONFIG, roots: dict | None = None) -> dict:
    config = json.loads(native(path.resolve()).read_text(encoding="utf-8-sig"))
    config["roots"].update(roots or {})
    config["roots"] = {k: str((PROJECT_ROOT / v).resolve()) if not Path(v).is_absolute() else v for k, v in config["roots"].items()}
    if config["cascade_reliability_metric"] != "minimum_of_mean_station_reliabilities":
        raise ValueError("The user-approved Case2 metric is min(mean-seed XW rate, mean-seed NZD rate).")
    if {case["id"] for case in config["cases"]} != {"Case 1", "Case 2"}:
        raise ValueError("Both configured cases are required")
    return config


def new_output(config: dict, supplied: Path | None = None, prefix: str = "run") -> Path:
    if supplied is not None:
        directory = supplied.resolve()
    else:
        base = Path(config["output_directory"])
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        directory = base / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    native(directory).mkdir(parents=True, exist_ok=False)
    return directory


def get_nested(value: dict, dotted: str):
    for name in dotted.split("."):
        if not isinstance(value, dict) or name not in value:
            raise KeyError(f"Missing metadata key: {dotted}")
        value = value[name]
    return value


def expect_metadata(value: dict, expected: dict, location: str) -> None:
    for key, wanted in expected.items():
        actual = get_nested(value, key)
        if actual != wanted:
            raise ValueError(f"Wrong metadata at {location}: {key}={actual!r}; expected {wanted!r}")


class Reader:
    def __init__(self, config: dict):
        self.roots = {key: Path(value) for key, value in config["roots"].items()}
        self.sources: dict[str, dict] = {}
        self.metadata_cache: dict[str, dict] = {}

    def location(self, spec: dict, seed=None) -> str:
        root = self.roots[spec["root"]]
        if "archive" in spec:
            return str(root / spec["archive"]) + "::" + spec["member"].format(seed=seed)
        return str(root / spec["path"].format(seed=seed))

    def bytes(self, spec: dict, seed=None) -> bytes:
        location = self.location(spec, seed)
        root = self.roots[spec["root"]]
        try:
            if "archive" in spec:
                path = root / spec["archive"]
                with zipfile.ZipFile(native(path)) as archive:
                    data = archive.read(spec["member"].format(seed=seed))
            else:
                data = native(root / spec["path"].format(seed=seed)).read_bytes()
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise FileNotFoundError(f"Required frozen result is unavailable: {location}") from exc
        self.sources[location] = {"source": location, "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def json(self, spec: dict, seed=None) -> dict:
        location = self.location(spec, seed)
        if location not in self.metadata_cache:
            data = self.bytes(spec, seed).decode("utf-8-sig")
            if location.lower().endswith((".yaml", ".yml")):
                value = yaml.safe_load(data)
            else:
                value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError(f"Expected metadata mapping: {location}")
            self.metadata_cache[location] = value
        return self.metadata_cache[location]

    def csv(self, spec: dict, seed=None) -> pd.DataFrame:
        try:
            return pd.read_csv(BytesIO(self.bytes(spec, seed)), encoding="utf-8-sig")
        except pd.errors.ParserError as exc:
            raise ValueError(f"Invalid CSV: {self.location(spec, seed)}") from exc


def numeric(frame: pd.DataFrame, name: str, location: str) -> np.ndarray:
    if name not in frame:
        raise KeyError(f"Missing column {name!r} in {location}")
    values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing/non-finite column {name!r}: {location}")
    return values


def boolean(frame: pd.DataFrame, name: str, location: str) -> np.ndarray:
    if name not in frame:
        raise KeyError(f"Missing column {name!r}: {location}")
    values = frame[name].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    if values.isna().any():
        raise ValueError(f"Invalid failure flag {name!r}: {location}")
    return values.to_numpy(dtype=bool)


def close(actual: float, expected: float, tolerance: float, description: str) -> None:
    if not (np.isfinite(actual) and np.isfinite(expected)
            and np.isclose(actual, expected, atol=tolerance, rtol=0)):
        raise ValueError(f"{description}: computed={actual:.12g}, reference={expected:.12g}, atol={tolerance:g}")


def monthly_dates(raw: pd.DataFrame, schema: str, config: dict, location: str):
    if len(raw) != config["expected_months"]:
        raise ValueError(f"Expected4800 monthly records, found {len(raw)}: {location}")
    if schema == "single":
        year_values = numeric(raw, "Calendar_Year", location)
        month_values = numeric(raw, "Calendar_Month", location)
    else:
        if "date" not in raw:
            raise KeyError(f"Missing date column: {location}")
        parts = raw["date"].astype(str).str.strip().str.extract(
            r"^(\d{4,})-(\d{2})-(\d{2})(?:[ T].*)?$")
        if parts.isna().any().any() or not (parts[2].astype(int) == 1).all():
            raise ValueError(f"Invalid first-of-month date: {location}")
        year_values = parts[0].to_numpy(dtype=float)
        month_values = parts[1].to_numpy(dtype=float)
    if not (np.array_equal(year_values, np.round(year_values))
            and np.array_equal(month_values, np.round(month_values))):
        raise ValueError(f"Fractional calendar year/month: {location}")
    years, months = year_values.astype(int), month_values.astype(int)
    if not (np.all(years >= 1) and np.all((months >= 1) & (months <= 12))):
        raise ValueError(f"Calendar year/month outside valid range: {location}")
    serial = years.astype(np.int64) * 12 + months
    if len(np.unique(serial)) != len(serial) or not np.all(np.diff(serial) == 1):
        raise ValueError(f"Missing, duplicated, or unsorted monthly dates: {location}")
    unique_years = np.unique(years)
    if (len(unique_years) != config["expected_years"] or not np.all(np.diff(unique_years) == 1)
            or not np.array_equal(months, np.tile(np.arange(1, 13), config["expected_years"]))):
        raise ValueError(f"Expected400 complete consecutive calendar years, January to December: {location}")
    days = np.asarray([calendar.monthrange(int(y), int(m))[1] for y, m in zip(years, months)], dtype=float)
    if schema == "single":
        recorded = numeric(raw, "Duration_Days", location)
        if not np.allclose(recorded, days, rtol=0, atol=1e-8):
            raise ValueError(f"Recorded duration disagrees with actual calendar-month length: {location}")
    return years, months, days, serial


def read_reference(reader: Reader, reference: dict, method: dict) -> tuple[float, str]:
    frame = reader.csv(reference["source"])
    filters = dict(reference.get("filter", {}))
    if "method_column" in reference:
        value = reference.get("method_values", {}).get(method["method"], method["method"])
        filters[reference["method_column"]] = value
    for column, wanted in filters.items():
        if column not in frame:
            raise KeyError(f"Missing reference-table field: {column}")
        if isinstance(wanted, (int, float)):
            frame = frame.loc[pd.to_numeric(frame[column], errors="raise") == wanted]
        else:
            frame = frame.loc[frame[column] == wanted]
    if len(frame) != 1:
        raise ValueError(f"Expected one frozen reference row for {method['method']}, found {len(frame)}")
    if "penalty_column" in reference:
        close(float(frame.iloc[0][reference["penalty_column"]]), float(method["penalty"]), 1e-9,
              f"Reference penalty for {method['method']}")
    value = float(frame.iloc[0][reference["mean_column"]])
    return value, reader.location(reference["source"])


def summary_metrics(summary: dict, schema: str, method_name: str, months: int):
    if schema == "single":
        energies = [(float(summary["total_generation_gwh"]) / 400, "summary total/400"),
                    (float(summary["mean_annual_generation_gwh"]), "summary mean annual")]
        rates = [float(summary["guarantee_rate"]) * 100]
    else:
        total_key = "cascade_generation_gwh" if method_name == "ISO-RF" else (
            "total_cascade_generation_gwh" if method_name in ("DQN", "PPO") else "total_generation_gwh")
        energies = [(float(summary[total_key]) / 400, "summary total/400")]
        if method_name in ("DQN", "PPO"):
            energies.append((float(summary["mean_annual_cascade_generation_gwh"]), "summary mean annual"))
        if method_name == "DDP":
            rates = [(1 - float(summary[key]) / months) * 100
                     for key in ("xiaowan_failed_months", "nuozhadu_failed_months")]
        elif method_name == "ISO-RF":
            rates = [float(summary[key]) * 100 for key in
                     ("xiaowan_firm_power_reliability", "nuozhadu_firm_power_reliability")]
        else:
            rates = [float(summary["firm_power_reliability"][station]) * 100
                     for station in ("xiaowan", "nuozhadu")]
    return energies, rates


def collect(config: dict) -> dict:
    """Finish all checks before returning anything to the plotting entry."""
    reader = Reader(config)
    annual_seed_rows, annual_mean_rows, point_rows, check_rows = [], [], [], []
    tolerance = config["generation_tolerance_gwh"]
    rate_tolerance = config["rate_tolerance_pct_points"]
    for case in config["cases"]:
        reference_dates = reference_inflows = None
        for method in case["methods"]:
            name, seeds = method["method"], method["seeds"]
            if name in ("DQN", "PPO"):
                if len(seeds) != 3 or set(seeds) != {1, 42, 100}:
                    raise ValueError(f"Incomplete/duplicated formal seed set: {case['id']} {name}")
            elif seeds != [None]:
                raise ValueError(f"Expected one frozen policy: {case['id']} {name}")
            for evidence in method.get("evidence", []):
                value = reader.json(evidence["source"])
                expect_metadata(value, evidence["expected"], reader.location(evidence["source"]))
            method_annual, method_rates, official_means, used_sources = [], [], [], []
            for seed in seeds:
                label = "single-policy" if seed is None else str(seed)
                location = reader.location(method["trajectory"], seed)
                print(f"READ {case['id']} {name} lambda={method['penalty']} seed={label}: {location}")
                raw = reader.csv(method["trajectory"], seed)
                summary = reader.json(method["summary"], seed)
                expect_metadata(summary, method["expected_summary"], reader.location(method["summary"], seed))
                if int(summary["months"]) != config["expected_months"]:
                    raise ValueError(f"Summary months mismatch: {location}")
                if "years" in summary and float(summary["years"]) != config["expected_years"]:
                    raise ValueError(f"Summary years mismatch: {location}")
                if seed is not None and int(summary["seed"]) != seed:
                    raise ValueError(f"Summary seed mismatch: {location}")
                if "metadata" in method:
                    metadata = reader.json(method["metadata"], seed)
                    expect_metadata(metadata, method["expected_metadata"], reader.location(method["metadata"], seed))
                    if int(metadata["seed"]) != seed:
                        raise ValueError(f"Configuration snapshot seed mismatch: {location}")
                years, months, days, serial = monthly_dates(raw, case["schema"], config, location)
                if case["schema"] == "single":
                    energies = [numeric(raw, "Electricity_GWh", location)]
                    powers = [numeric(raw, "Power_kW", location)]
                    failures = [boolean(raw, "Firm_Power_Violation", location)]
                    inflows = np.column_stack([numeric(raw, "Inflow_m3s", location)])
                    energy_columns = ["Electricity_GWh"]
                else:
                    energy_columns = ["r1_energy_gwh", "r2_energy_gwh"]
                    energies = [numeric(raw, key, location) for key in energy_columns]
                    powers = [numeric(raw, key, location) for key in ("r1_power_kw", "r2_power_kw")]
                    if "xiaowan_firm_power_failure" in raw:
                        failures = [boolean(raw, key, location) for key in
                                    ("xiaowan_firm_power_failure", "nuozhadu_firm_power_failure")]
                    else:
                        failures = [~boolean(raw, key, location) for key in
                                    ("xiaowan_firm_power_satisfied", "nuozhadu_firm_power_satisfied")]
                    inflows = np.column_stack([numeric(raw, key, location) for key in
                                              ("inflow_1_m3s", "inflow_2_local_m3s")])
                if reference_dates is None:
                    reference_dates, reference_inflows = serial, inflows
                elif not (np.array_equal(serial, reference_dates)
                          and np.allclose(inflows, reference_inflows, atol=1e-8, rtol=0)):
                    raise ValueError(f"Different evaluation dates or natural inflows within {case['id']}: {location}")
                rates = []
                max_unit_error = 0.0
                for index, (energy, power, failure) in enumerate(zip(energies, powers, failures)):
                    if np.any(energy < 0) or np.any(power < 0):
                        raise ValueError(f"Negative physical energy/power: {location}")
                    expected_energy = power * days * 24 / 1e6  # kW h -> GWh
                    max_unit_error = max(max_unit_error, float(np.max(np.abs(energy - expected_energy))))
                    if not np.allclose(energy, expected_energy, atol=1e-7, rtol=1e-10):
                        raise ValueError(f"Monthly GWh/kW/calendar-duration consistency failed: {location}")
                    if not np.array_equal(failure, power < case["firm_power_kw"][index]):
                        raise ValueError(f"Failure flag disagrees with strict firm-power threshold: {location}")
                    rates.append(float((~failure).mean() * 100))
                monthly_energy = np.sum(np.vstack(energies), axis=0)
                if case["schema"] == "cascade" and "generation_energy_gwh" in raw:
                    recorded = pd.to_numeric(raw["generation_energy_gwh"], errors="raise")
                    if recorded.notna().any() and not np.allclose(
                            recorded.to_numpy(), monthly_energy, atol=1e-7, rtol=1e-10):
                        raise ValueError(f"Recorded cascade energy differs from station sum: {location}")
                annual = pd.DataFrame({"year": years, "energy_gwh": monthly_energy}).groupby("year")["energy_gwh"].sum()
                if len(annual) != 400:
                    raise ValueError(f"Expected400 annual values: {location}")
                observed_mean = float(annual.mean())
                official, official_rates = summary_metrics(summary, case["schema"], name, len(raw))
                for value, description in official:
                    close(observed_mean, value, tolerance, f"{case['id']} {name} seed={label}, {description}")
                for actual, expected in zip(rates, official_rates):
                    close(actual, expected, rate_tolerance, f"{case['id']} {name} seed={label}, station reliability")
                method_annual.append(annual)
                method_rates.append(rates)
                official_means.append(official[0][0])
                used_sources.append(location)
                for year, energy in annual.items():
                    annual_seed_rows.append({"case": case["id"], "method": name, "penalty": method["penalty"],
                        "seed": label, "calendar_year": int(year), "annual_generation_gwh": float(energy),
                        "annual_generation_twh": float(energy) / 1000})
                check_rows.append({"case": case["id"], "method": name, "penalty": method["penalty"],
                    "seed": label, "months": len(raw), "complete_years": len(annual),
                    "first_year": int(annual.index[0]), "last_year": int(annual.index[-1]),
                    "energy_columns": " + ".join(energy_columns), "input_energy_unit": "GWh/month",
                    "computed_mean_gwh": observed_mean, "official_mean_gwh": official[0][0],
                    "mean_difference_gwh": observed_mean - official[0][0],
                    "max_monthly_energy_unit_check_error_gwh": max_unit_error,
                    "trajectory_source": location, "summary_source": reader.location(method["summary"], seed),
                    "status": "PASS"})
            # Align by the actual calendar-year index. Never pool 1200 seed-years.
            matrix = pd.concat(method_annual, axis=1)
            if matrix.isna().any().any() or matrix.shape != (400, len(seeds)):
                raise ValueError(f"Seed annual alignment failed: {case['id']} {name}")
            mean_by_year = matrix.mean(axis=1)
            method_mean = float(mean_by_year.mean())
            close(method_mean, float(np.mean(official_means)), tolerance, f"Mean across seed annuals: {case['id']} {name}")
            reference = method.get("reference", case.get("reference"))
            reference_mean, reference_location = read_reference(reader, reference, method)
            close(method_mean, reference_mean, tolerance, f"Existing formal reference table: {case['id']} {name}")
            for year, energy in mean_by_year.items():
                annual_mean_rows.append({"case": case["id"], "method": name, "penalty": method["penalty"],
                    "seed_count": len(seeds), "calendar_year": int(year), "annual_generation_gwh": float(energy),
                    "annual_generation_twh": float(energy) / 1000})
            mean_rates = np.asarray(method_rates).mean(axis=0)
            scalar_rate = float(mean_rates[0] if case["schema"] == "single" else min(mean_rates))
            handling = ("Mean of annual energies aligned by calendar year across seeds1/42/100; "
                        "station reliability is mean of per-seed monthly reliability" if len(seeds) == 3
                        else "One frozen policy; no seed pooling")
            point_rows.append({"case": case["id"], "method": name, "penalty": method["penalty"],
                "seed_count": len(seeds), "seeds": ",".join("single-policy" if s is None else str(s) for s in seeds),
                "seed_handling": handling, "selection_data_stage": method["selection_stage"],
                "selection_rule": method["selection_rule"], "evaluation_stage": "independent_evaluation",
                "xw_reliability_pct": float(mean_rates[0]),
                "nzd_reliability_pct": float(mean_rates[1]) if len(mean_rates) == 2 else np.nan,
                "transfer_reliability_metric": "XW" if len(mean_rates) == 1 else "min(mean-seed XW, mean-seed NZD)",
                "transfer_reliability_pct": scalar_rate,
                "deviation_from_target_pp": scalar_rate - config["target_reliability_pct"],
                "mean_annual_generation_gwh": method_mean, "mean_annual_generation_twh": method_mean / 1000,
                "reference_table_mean_gwh": reference_mean, "reference_table_difference_gwh": method_mean - reference_mean,
                "annual_plot_sample_count": 400, "reference_table_source": reference_location,
                "trajectory_sources": " | ".join(used_sources), "selection_verified": True})
            print(f"PASS {case['id']} {name}: annual n=400, seeds={len(seeds)}, "
                  f"mean={method_mean:.8f} GWh/year, reference error={method_mean-reference_mean:.3g} GWh")
    points = pd.DataFrame(point_rows)
    for case_id, group in points.groupby("case", sort=False):
        ddp = group.loc[group["method"] == "DDP"]
        if len(ddp) != 1:
            raise ValueError(f"Exactly one same-case DDP reference is required: {case_id}")
        reference = float(ddp.iloc[0]["mean_annual_generation_gwh"])
        if reference <= 0:
            raise ValueError("DDP generation reference must be positive")
        points.loc[group.index, "same_case_ddp_generation_gwh"] = reference
        points.loc[group.index, "relative_generation_difference_pct"] = (
            (group["mean_annual_generation_gwh"] - reference) / reference * 100)
    report = {
        "status": "PASS", "configuration": config,
        "source_files": list(reader.sources.values()),
        "annual_seed_rows": len(annual_seed_rows), "annual_main_plot_rows": len(annual_mean_rows),
        "representative_point_count": len(points),
        "aggregation": "For each method, sum recorded monthly GWh within calendar year; align seeds by year; "
                       "average seeds within each year; divide GWh by1000 for TWh. Each violin has400 values.",
        "reliability": "Strict monthly station power>=firm power, per-seed first, then seed mean. "
                       "Case2 scalar=min(two station mean rates), explicitly approved by user; not joint reliability.",
        "selection": "All penalties/checkpoints are frozen existing choices. No penalty sweep or model is run.",
        "new_repository_vs_result_version": "Integrated repository contains local frozen result copies; original locations are recorded in provenance. "
                                             "PPO is pinned to confirmed final V16 geometry-conditioned6000 snapshots.",
        "figure_interpretation": "Arrows compare two problem instances at fixed representative points; "
                                 "they do not imply transfer-learning experiments or a causal treatment effect.",
        "plots_executed_during_audit": False,
    }
    return {"annual_by_seed": pd.DataFrame(annual_seed_rows),
            "annual_seed_mean": pd.DataFrame(annual_mean_rows), "points": points,
            "checks": pd.DataFrame(check_rows), "report": report}


def export_audit(result: dict, directory: Path) -> None:
    for key, filename in (("annual_by_seed", "annual_generation_by_seed.csv"),
                          ("annual_seed_mean", "annual_generation_400y_seed_mean.csv"),
                          ("points", "representative_points_audit.csv"),
                          ("checks", "per_seed_data_checks.csv")):
        result[key].to_csv(native(directory / filename), index=False, encoding="utf-8-sig")
    native(directory / "data_audit_report.json").write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")


def write_failure(directory: Path, error: Exception) -> None:
    native(directory / "data_audit_failure.json").write_text(json.dumps({
        "status": "FAIL", "error_type": type(error).__name__, "message": str(error),
        "figures_generated": False, "action": "Resolve source/schema/unit/selection discrepancy; do not silently alter results."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
