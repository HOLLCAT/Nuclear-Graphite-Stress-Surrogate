"""Shared utilities for the CT3 nuclear-graphite surrogate workflow.

The independent experimental unit is a complete FEM case.  Element rows are
spatial observations inside that case and must never be split across model
roles.  This module centralises file parsing, the frozen case manifest,
high-stress weighting and engineering evaluation metrics so every notebook
uses identical definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RANDOM_SEED = 42
EXPECTED_CASE_COUNT = 199
EXPECTED_ELEMENTS_PER_CASE = 400_360
EXPECTED_MISSING_CASE_NUMBERS = {25, 30}
EXPECTED_CASE_NUMBERS = set(range(0, 201)) - EXPECTED_MISSING_CASE_NUMBERS

TARGET_COL = "sigma_max_principal"
PHYSICAL_FEATURES = ["fluence_rate", "temperature", "weight_loss_rate"]
RAW_POSITION_FEATURES = ["rho", "theta", "z"]
PERIODIC_POSITION_FEATURES = ["rho", "theta_sin", "theta_cos", "z"]

FEATURE_SETS = {
    "physical_only": PHYSICAL_FEATURES,
    "position_only_periodic": PERIODIC_POSITION_FEATURES,
    "full_raw_coordinates": PHYSICAL_FEATURES + RAW_POSITION_FEATURES,
    "full_periodic_coordinates": PHYSICAL_FEATURES + PERIODIC_POSITION_FEATURES,
}

# These ranges are provisional engineering checks.  Values are flagged, not
# clipped or silently removed.  Stress sign handling remains an open decision.
PROVISIONAL_PHYSICAL_BOUNDS = {
    "fluence_rate": (0.0, 7.0),
    "temperature": (25.0, 600.0),
    "weight_loss_rate": (0.0, 2.0),
}
PROVISIONAL_STRESS_BOUNDS = (-70.0, 35.0)

TAIL_WEIGHT_LEVELS = (
    (0.90, 2.0),
    (0.95, 4.0),
    (0.99, 8.0),
)


COLUMN_ALIASES = {
    "element_id": ["ElementID", "Element ID", "Element_Number", "ElementNumber"],
    "x": ["X", "X Coordinate", "X_Coordinate"],
    "y": ["Y", "Y Coordinate", "Y_Coordinate"],
    "z": ["Z", "Z Coordinate", "Z_Coordinate"],
    "rho": ["Rho", "Radius", "Radial Coordinate", "Radial_Coordinate"],
    "theta": ["Theta", "Theta Rad", "Theta_Rad", "Angular Coordinate", "Angular_Coordinate"],
    "fluence_rate": ["FluenceRate", "Fluence Rate", "Fast Neutron Fluence Rate"],
    "temperature": ["Temperature", "Temp"],
    "weight_loss_rate": ["WeightLossRate", "Weight Loss Rate", "Weight_Loss_Rate"],
    "sigma_max_principal": ["MaxPrincipalStress", "Max Principal Stress", "Maximum Principal Stress"],
}


SELECTION_METRICS = (
    ("macro_rmse", 0.15, "min"),
    ("worst_case_rmse", 0.10, "min"),
    ("mean_top5_actual_rmse", 0.10, "min"),
    ("mean_p95_relative_error", 0.15, "min"),
    ("mean_p99_relative_error", 0.20, "min"),
    ("mean_p99_underprediction_fraction", 0.10, "min"),
    ("mean_top5pct_hotspot_overlap", 0.08, "max"),
    ("mean_top1pct_hotspot_overlap", 0.08, "max"),
    ("mean_top1_recall_in_predicted_top5", 0.04, "max"),
)


@dataclass(frozen=True)
class CasePaths:
    package_root: Path
    case_dir: Path
    output_root: Path
    manifest_path: Path


def resolve_package_root(cwd: Path | None = None) -> Path:
    current = (cwd or Path.cwd()).resolve()
    if current.name == "notebooks":
        return current.parent
    if current.name in {"src", "cluster", "shared"}:
        return current.parent
    return current


def resolve_case_directory(package_root: Path, override: str | Path | None = None) -> Path:
    env_override = os.environ.get("CT3_CASE_DIR")
    if override is not None:
        candidates = [Path(override)]
    elif env_override:
        candidates = [Path(env_override)]
    else:
        candidates = [
            package_root / "FE_Results_Cases_All",
            package_root / " Case Test Data" / "FE_Results_Cases_All",
            package_root / "Case Test Data" / "FE_Results_Cases_All",
            package_root.parent / "FE_Results_Cases_All",
            package_root.parent / "data" / "FE_Results_Cases_All",
            package_root.parent / " Case Test Data" / "FE_Results_Cases_All",
            package_root.parent / "Case Test Data" / "FE_Results_Cases_All",
        ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find FE_Results_Cases_All. Set CT3_CASE_DIR or edit CASE_DIR_OVERRIDE.\n"
        f"Checked:\n{rendered}"
    )


def build_paths(cwd: Path | None = None, case_dir_override: str | Path | None = None) -> CasePaths:
    package_root = resolve_package_root(cwd)
    case_dir = resolve_case_directory(package_root, case_dir_override)
    output_root = package_root / "outputs"
    manifest_path = package_root / "shared" / "frozen_case_split_manifest_199cases.csv"
    return CasePaths(package_root, case_dir, output_root, manifest_path)


def extract_case_number(path: Path) -> int:
    match = re.search(r"Case_(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract a case number from {path.name}")
    return int(match.group(1))


def case_id_from_number(case_number: int) -> str:
    return f"case_{case_number:02d}"


def count_data_rows(path: Path) -> int:
    newline_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            newline_count += chunk.count(b"\n")
    return max(newline_count - 1, 0)


def discover_case_files(
    case_dir: Path,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[list[Path], pd.DataFrame]:
    files = sorted(case_dir.glob("FE_Results_Case_*.txt"), key=extract_case_number)
    numbers = [extract_case_number(path) for path in files]
    if len(numbers) != len(set(numbers)):
        duplicate_numbers = sorted({number for number in numbers if numbers.count(number) > 1})
        raise ValueError(f"Duplicate case numbers found: {duplicate_numbers}")
    if len(files) != expected_case_count:
        raise ValueError(f"Expected {expected_case_count} case files, found {len(files)} in {case_dir}")
    if set(numbers) != EXPECTED_CASE_NUMBERS:
        missing = sorted(EXPECTED_CASE_NUMBERS - set(numbers))
        unexpected = sorted(set(numbers) - EXPECTED_CASE_NUMBERS)
        raise ValueError(f"Unexpected case-number coverage. Missing={missing}; unexpected={unexpected}")

    records = []
    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().strip()
        records.append({
            "case_id": case_id_from_number(extract_case_number(path)),
            "case_number": extract_case_number(path),
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "size_mb": path.stat().st_size / 1_000_000,
            "n_elements_from_line_count": count_data_rows(path),
            "header": header,
        })
    inventory = pd.DataFrame(records).sort_values("case_number").reset_index(drop=True)
    wrong_size = inventory[
        inventory["n_elements_from_line_count"] != EXPECTED_ELEMENTS_PER_CASE
    ]
    if not wrong_size.empty:
        details = wrong_size[["case_id", "n_elements_from_line_count"]].to_dict("records")
        raise ValueError(
            f"Every formal CT3 case must contain {EXPECTED_ELEMENTS_PER_CASE} elements. "
            f"Mismatches: {details[:10]}"
        )
    return files, inventory


def normalise_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_source_column(columns: Sequence[str], standard_name: str) -> str | None:
    normalised_lookup: dict[str, str] = {}
    for column in columns:
        key = normalise_column_name(column)
        if key in normalised_lookup and normalised_lookup[key] != column:
            raise ValueError(
                f"Ambiguous columns after normalisation: {normalised_lookup[key]!r} and {column!r}"
            )
        normalised_lookup[key] = column
    for alias in COLUMN_ALIASES[standard_name]:
        source = normalised_lookup.get(normalise_column_name(alias))
        if source is not None:
            return source
    return None


def circular_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def read_complete_case(
    path: Path,
    *,
    cylinder_origin_x: float = 0.0,
    cylinder_origin_y: float = 0.0,
    provided_theta_unit: str = "radians",
) -> tuple[pd.DataFrame, dict]:
    """Read every element and create a standard complete-case frame.

    X/Y/Z are treated as the exported post-deformation coordinates.  If Rho
    and Theta are absent, they are derived from those exported coordinates.
    The surrogate is therefore valid only when the same input fields are
    available at prediction time.
    """
    raw = pd.read_csv(path, skipinitialspace=True)
    raw.columns = [str(column).strip() for column in raw.columns]
    source_columns = {
        name: find_source_column(list(raw.columns), name)
        for name in COLUMN_ALIASES
    }
    always_required = ["element_id", "z", *PHYSICAL_FEATURES, TARGET_COL]
    missing_required = [name for name in always_required if source_columns[name] is None]
    if missing_required:
        raise ValueError(f"{path.name} is missing required fields: {missing_required}")

    has_xy = source_columns["x"] is not None and source_columns["y"] is not None
    has_rho_theta = source_columns["rho"] is not None and source_columns["theta"] is not None
    if not has_xy and not has_rho_theta:
        raise ValueError(f"{path.name} must provide either X/Y or Rho/Theta coordinates")

    standard = pd.DataFrame(index=raw.index)
    for name in always_required:
        standard[name] = pd.to_numeric(raw[source_columns[name]], errors="coerce")

    rho_from_xy = None
    theta_from_xy = None
    if has_xy:
        standard["x"] = pd.to_numeric(raw[source_columns["x"]], errors="coerce")
        standard["y"] = pd.to_numeric(raw[source_columns["y"]], errors="coerce")
        x_relative = standard["x"] - cylinder_origin_x
        y_relative = standard["y"] - cylinder_origin_y
        rho_from_xy = np.sqrt(x_relative ** 2 + y_relative ** 2)
        theta_from_xy = np.arctan2(y_relative, x_relative)
    else:
        standard["x"] = np.nan
        standard["y"] = np.nan

    if has_rho_theta:
        standard["rho"] = pd.to_numeric(raw[source_columns["rho"]], errors="coerce")
        standard["theta"] = pd.to_numeric(raw[source_columns["theta"]], errors="coerce")
        if provided_theta_unit == "degrees":
            standard["theta"] = np.deg2rad(standard["theta"])
        elif provided_theta_unit != "radians":
            raise ValueError("provided_theta_unit must be 'radians' or 'degrees'")
        coordinate_source = "provided_rho_theta"
    else:
        standard["rho"] = rho_from_xy
        standard["theta"] = theta_from_xy
        coordinate_source = "derived_from_exported_xy"

    standard["theta_sin"] = np.sin(standard["theta"])
    standard["theta_cos"] = np.cos(standard["theta"])

    numeric_columns = [
        "element_id", "x", "y", "z", "rho", "theta", "theta_sin", "theta_cos",
        *PHYSICAL_FEATURES, TARGET_COL,
    ]
    required_numeric_columns = [
        "element_id", "z", "rho", "theta", "theta_sin", "theta_cos",
        *PHYSICAL_FEATURES, TARGET_COL,
    ]
    missing_counts = standard[required_numeric_columns].isna().sum()
    if int(missing_counts.sum()) > 0:
        raise ValueError(
            f"{path.name} contains missing/non-numeric values: "
            f"{missing_counts[missing_counts > 0].to_dict()}"
        )
    infinite_counts = {
        column: int(np.isinf(standard[column].to_numpy(dtype=np.float64)).sum())
        for column in required_numeric_columns
    }
    if sum(infinite_counts.values()) > 0:
        raise ValueError(
            f"{path.name} contains infinite values: "
            f"{ {key: value for key, value in infinite_counts.items() if value > 0} }"
        )

    element_values = standard["element_id"].to_numpy(dtype=np.float64)
    if not np.allclose(element_values, np.round(element_values), rtol=0.0, atol=0.0):
        raise ValueError(f"{path.name} contains non-integer ElementID values")
    standard["element_id"] = np.round(element_values).astype(np.int64)
    if standard["element_id"].nunique(dropna=False) != len(standard):
        raise ValueError(f"{path.name} contains duplicate ElementID values")

    rho_crosscheck_max = np.nan
    theta_crosscheck_max = np.nan
    if has_xy and has_rho_theta:
        rho_crosscheck_max = float(np.max(np.abs(standard["rho"] - rho_from_xy)))
        theta_crosscheck_max = float(
            np.max(circular_angle_difference(standard["theta"].to_numpy(), theta_from_xy.to_numpy()))
        )

    case_number = extract_case_number(path)
    audit = {
        "case_id": case_id_from_number(case_number),
        "case_number": case_number,
        "source_file": path.name,
        "n_elements_read": len(standard),
        "element_id_unique": int(standard["element_id"].nunique()),
        "element_id_min": int(standard["element_id"].min()),
        "element_id_max": int(standard["element_id"].max()),
        "element_id_monotonic": bool(standard["element_id"].is_monotonic_increasing),
        "coordinate_source": coordinate_source,
        "rho_xy_crosscheck_max_abs": rho_crosscheck_max,
        "theta_xy_crosscheck_max_abs_rad": theta_crosscheck_max,
        "model_missing_count": int(standard[required_numeric_columns].isna().sum().sum()),
        "model_infinite_count": int(sum(infinite_counts.values())),
    }

    for column in numeric_columns:
        if column != "element_id" and column in standard.columns:
            standard[column] = standard[column].astype(np.float32)
    del raw
    return standard, audit


def physical_boundary_audit(frame: pd.DataFrame, case_id: str) -> list[dict]:
    records = []
    for feature, (lower, upper) in PROVISIONAL_PHYSICAL_BOUNDS.items():
        values = frame[feature].to_numpy(dtype=np.float64)
        below = values < lower
        above = values > upper
        records.append({
            "case_id": case_id,
            "variable": feature,
            "provisional_lower": lower,
            "provisional_upper": upper,
            "observed_min": float(values.min()),
            "observed_max": float(values.max()),
            "n_below": int(below.sum()),
            "n_above": int(above.sum()),
            "fraction_outside": float((below | above).mean()),
            "action": "flag_only_no_clipping",
        })
    stress = frame[TARGET_COL].to_numpy(dtype=np.float64)
    lower, upper = PROVISIONAL_STRESS_BOUNDS
    records.append({
        "case_id": case_id,
        "variable": TARGET_COL,
        "provisional_lower": lower,
        "provisional_upper": upper,
        "observed_min": float(stress.min()),
        "observed_max": float(stress.max()),
        "n_below": int((stress < lower).sum()),
        "n_above": int((stress > upper).sum()),
        "fraction_outside": float(((stress < lower) | (stress > upper)).mean()),
        "action": "flag_only_pending_stress_sign_and_mesh_review",
    })
    return records


def load_frozen_manifest(manifest_path: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen split manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"iteration", "case_id", "case_number", "similarity_group", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(manifest.columns))}")

    expected_case_ids = set(inventory["case_id"])
    if set(manifest["case_id"].unique()) != expected_case_ids:
        raise ValueError("Manifest case IDs do not match the discovered 199-case pool")
    if set(manifest["iteration"].unique()) != {1, 2, 3, 4}:
        raise ValueError("Manifest must contain exactly four development iterations")

    expected_counts = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    for iteration, rows in manifest.groupby("iteration"):
        if len(rows) != EXPECTED_CASE_COUNT or rows["case_id"].nunique() != EXPECTED_CASE_COUNT:
            raise ValueError(f"Iteration {iteration} does not contain every case exactly once")
        counts = rows["split"].value_counts().to_dict()
        if counts != expected_counts:
            raise ValueError(f"Iteration {iteration} has unexpected split counts: {counts}")
        leakage = rows.groupby("similarity_group")["split"].nunique()
        if (leakage > 1).any():
            raise ValueError(f"Iteration {iteration} splits at least one similarity group")

    final_sets = [
        frozenset(rows.loc[rows["split"] == "final_test", "case_id"])
        for _, rows in manifest.groupby("iteration")
    ]
    if len(set(final_sets)) != 1:
        raise ValueError("The 50-case final-test set changes between iterations")
    return manifest.sort_values(["iteration", "case_number"]).reset_index(drop=True)


def cases_for_role(manifest: pd.DataFrame, iteration: int, role: str) -> list[str]:
    rows = manifest[(manifest["iteration"] == iteration) & (manifest["split"] == role)]
    return rows.sort_values("case_number")["case_id"].tolist()


def development_case_ids(manifest: pd.DataFrame) -> list[str]:
    rows = manifest[(manifest["iteration"] == 1) & (manifest["split"] != "final_test")]
    return rows.sort_values("case_number")["case_id"].tolist()


def final_test_case_ids(manifest: pd.DataFrame) -> list[str]:
    return cases_for_role(manifest, 1, "final_test")


def make_case_tail_weights(
    target: np.ndarray,
    levels: Sequence[tuple[float, float]] = TAIL_WEIGHT_LEVELS,
) -> tuple[np.ndarray, dict]:
    """Create tiered upper-tail weights and normalise them to mean one.

    Every element remains in training.  Weighting changes emphasis rather than
    sampling.  Per-case normalisation prevents a high-tail case from receiving
    more total weight solely because its distribution is wider.
    """
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    weights = np.ones(len(values), dtype=np.float64)
    thresholds = {}
    for quantile, level_weight in levels:
        threshold = float(np.quantile(values, quantile))
        thresholds[f"q{int(round(100 * quantile)):02d}"] = threshold
        weights[values >= threshold] = float(level_weight)
    raw_mean = float(weights.mean())
    weights /= raw_mean
    audit = {
        **thresholds,
        "raw_weight_mean": raw_mean,
        "normalised_weight_mean": float(weights.mean()),
        "normalised_weight_min": float(weights.min()),
        "normalised_weight_max": float(weights.max()),
    }
    return weights.astype(np.float32), audit


def equal_fraction_hotspot_overlap(actual: np.ndarray, predicted: np.ndarray, fraction: float) -> float:
    n = len(actual)
    k = max(1, int(math.ceil(n * fraction)))
    actual_indices = np.argpartition(actual, n - k)[n - k:]
    predicted_indices = np.argpartition(predicted, n - k)[n - k:]
    overlap = np.intersect1d(actual_indices, predicted_indices, assume_unique=False).size
    return float(overlap / k)


def top_fraction_recall(
    actual: np.ndarray,
    predicted: np.ndarray,
    actual_fraction: float,
    predicted_fraction: float,
) -> float:
    n = len(actual)
    actual_k = max(1, int(math.ceil(n * actual_fraction)))
    predicted_k = max(1, int(math.ceil(n * predicted_fraction)))
    actual_indices = np.argpartition(actual, n - actual_k)[n - actual_k:]
    predicted_indices = np.argpartition(predicted, n - predicted_k)[n - predicted_k:]
    captured = np.intersect1d(actual_indices, predicted_indices, assume_unique=False).size
    return float(captured / actual_k)


def evaluate_prediction_arrays(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError(f"Prediction shape mismatch: {predicted.shape} vs {actual.shape}")
    if not np.isfinite(predicted).all():
        raise ValueError("Model produced non-finite predictions")

    error = predicted - actual
    absolute_error = np.abs(error)
    squared_error = error ** 2
    actual_mean = float(actual.mean())
    actual_p95 = float(np.quantile(actual, 0.95))
    actual_p99 = float(np.quantile(actual, 0.99))
    predicted_p95 = float(np.quantile(predicted, 0.95))
    predicted_p99 = float(np.quantile(predicted, 0.99))
    top5_threshold = actual_p95
    top5_mask = actual >= top5_threshold
    actual_sst = float(np.sum((actual - actual_mean) ** 2))
    sse = float(squared_error.sum())

    p95_denominator = max(abs(actual_p95), 1e-12)
    p99_denominator = max(abs(actual_p99), 1e-12)
    return {
        "n_elements": len(actual),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "r2": float(1.0 - sse / actual_sst) if actual_sst > 0 else np.nan,
        "bias_predicted_minus_actual": float(error.mean()),
        "top5_actual_rmse": float(np.sqrt(squared_error[top5_mask].mean())),
        "top5_actual_bias": float(error[top5_mask].mean()),
        "actual_mean": actual_mean,
        "predicted_mean": float(predicted.mean()),
        "actual_p95": actual_p95,
        "predicted_p95": predicted_p95,
        "p95_relative_error": float(abs(predicted_p95 - actual_p95) / p95_denominator),
        "p95_underprediction_fraction": float(max(actual_p95 - predicted_p95, 0.0) / p95_denominator),
        "actual_p99": actual_p99,
        "predicted_p99": predicted_p99,
        "p99_relative_error": float(abs(predicted_p99 - actual_p99) / p99_denominator),
        "p99_underprediction_fraction": float(max(actual_p99 - predicted_p99, 0.0) / p99_denominator),
        "actual_max": float(actual.max()),
        "predicted_max": float(predicted.max()),
        "max_relative_error": float(
            abs(float(predicted.max()) - float(actual.max())) / max(abs(float(actual.max())), 1e-12)
        ),
        "prediction_abs_max_ratio": float(
            np.max(np.abs(predicted)) / max(np.max(np.abs(actual)), 1e-12)
        ),
        "top5pct_hotspot_overlap": equal_fraction_hotspot_overlap(actual, predicted, 0.05),
        "top1pct_hotspot_overlap": equal_fraction_hotspot_overlap(actual, predicted, 0.01),
        "top1_recall_in_predicted_top5": top_fraction_recall(actual, predicted, 0.01, 0.05),
        "negative_prediction_fraction": float((predicted < 0).mean()),
        "sum_absolute_error": float(absolute_error.sum()),
        "sum_squared_error": sse,
        "actual_sum": float(actual.sum()),
        "actual_squared_sum": float(np.sum(actual ** 2)),
    }


def aggregate_case_metrics(case_metrics: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    records = []
    for keys, group in case_metrics.groupby(list(group_columns), observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        n_total = int(group["n_elements"].sum())
        sum_absolute_error = float(group["sum_absolute_error"].sum())
        sum_squared_error = float(group["sum_squared_error"].sum())
        actual_sum = float(group["actual_sum"].sum())
        actual_squared_sum = float(group["actual_squared_sum"].sum())
        total_sst = actual_squared_sum - (actual_sum ** 2) / n_total
        record.update({
            "n_cases": int(group["case_id"].nunique()),
            "n_elements_evaluated": n_total,
            "micro_mae": sum_absolute_error / n_total,
            "micro_rmse": math.sqrt(sum_squared_error / n_total),
            "micro_r2": 1.0 - sum_squared_error / total_sst if total_sst > 0 else np.nan,
            "macro_mae": float(group["mae"].mean()),
            "macro_rmse": float(group["rmse"].mean()),
            "macro_r2": float(group["r2"].mean()),
            "worst_case_rmse": float(group["rmse"].max()),
            "mean_top5_actual_rmse": float(group["top5_actual_rmse"].mean()),
            "mean_top5_actual_bias": float(group["top5_actual_bias"].mean()),
            "mean_p95_relative_error": float(group["p95_relative_error"].mean()),
            "mean_p95_underprediction_fraction": float(group["p95_underprediction_fraction"].mean()),
            "mean_p99_relative_error": float(group["p99_relative_error"].mean()),
            "mean_p99_underprediction_fraction": float(group["p99_underprediction_fraction"].mean()),
            "mean_top5pct_hotspot_overlap": float(group["top5pct_hotspot_overlap"].mean()),
            "mean_top1pct_hotspot_overlap": float(group["top1pct_hotspot_overlap"].mean()),
            "mean_top1_recall_in_predicted_top5": float(group["top1_recall_in_predicted_top5"].mean()),
            "max_prediction_abs_max_ratio": float(group["prediction_abs_max_ratio"].max()),
        })
        records.append(record)
    return pd.DataFrame(records)


def add_engineering_selection_score(
    table: pd.DataFrame,
    *,
    metric_prefix: str = "",
    score_column: str = "engineering_selection_score",
) -> pd.DataFrame:
    """Add a validation-only weighted percentile-rank score; lower is better."""
    out = table.copy()
    rank_columns = []
    for metric, weight, direction in SELECTION_METRICS:
        source = f"{metric_prefix}{metric}"
        if source not in out.columns:
            raise ValueError(f"Selection metric is unavailable: {source}")
        rank_column = f"{source}_selection_rank"
        out[rank_column] = out[source].rank(
            method="average",
            pct=True,
            ascending=(direction == "min"),
        )
        out[rank_column] *= weight
        rank_columns.append(rank_column)
    out[score_column] = out[rank_columns].sum(axis=1)
    return out


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def feature_set_from_selection(selection_path: Path) -> tuple[str, list[str]]:
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Feature-ablation selection is missing: {selection_path}. "
            "Run notebook 01 before baseline or symbolic regression."
        )
    selection = pd.read_csv(selection_path)
    if len(selection) != 1:
        raise ValueError("Feature-ablation selection must contain exactly one locked row")
    name = str(selection.iloc[0]["feature_set"])
    features = json.loads(selection.iloc[0]["features_json"])
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown selected feature set: {name}")
    if features != FEATURE_SETS[name]:
        raise ValueError("Selected feature list does not match the shared feature-set definition")
    return name, list(features)


def case_path_lookup(inventory: pd.DataFrame) -> dict[str, Path]:
    return {
        row.case_id: Path(row.file_path)
        for row in inventory.itertuples(index=False)
    }


def load_case_payload(
    case_id: str,
    path_by_case: dict[str, Path],
    features: Sequence[str],
) -> tuple[dict, dict]:
    frame, audit = read_complete_case(path_by_case[case_id])
    payload = {
        "X": np.ascontiguousarray(frame[list(features)].to_numpy(dtype=np.float32)),
        "y": np.ascontiguousarray(frame[TARGET_COL].to_numpy(dtype=np.float32)),
        "element_id": np.ascontiguousarray(frame["element_id"].to_numpy(dtype=np.int64)),
    }
    return payload, audit


def compute_full_scaling(
    case_ids: Sequence[str],
    path_by_case: dict[str, Path],
    features: Sequence[str],
) -> dict:
    x_sum = np.zeros(len(features), dtype=np.float64)
    x_squared_sum = np.zeros(len(features), dtype=np.float64)
    y_sum = 0.0
    y_squared_sum = 0.0
    n_rows = 0
    row_counts = {}
    for case_id in case_ids:
        payload, _ = load_case_payload(case_id, path_by_case, features)
        X = payload["X"]
        y = payload["y"]
        x_sum += X.sum(axis=0, dtype=np.float64)
        x_squared_sum += np.square(X, dtype=np.float64).sum(axis=0)
        y_sum += float(y.sum(dtype=np.float64))
        y_squared_sum += float(np.square(y, dtype=np.float64).sum())
        n_rows += len(y)
        row_counts[case_id] = len(y)
    x_mean = x_sum / n_rows
    x_variance = np.maximum(x_squared_sum / n_rows - x_mean ** 2, 0.0)
    x_std = np.sqrt(x_variance)
    y_mean = y_sum / n_rows
    y_variance = max(y_squared_sum / n_rows - y_mean ** 2, 0.0)
    y_std = math.sqrt(y_variance)
    if np.any(x_std <= 0) or y_std <= 0:
        raise ValueError("At least one full-training variable has zero variance")
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "n_rows": n_rows,
        "row_counts": row_counts,
    }


def assemble_full_training_arrays(
    case_ids: Sequence[str],
    path_by_case: dict[str, Path],
    features: Sequence[str],
    *,
    scale: bool,
    use_tail_weights: bool = True,
) -> dict:
    """Read and retain every element from the listed complete cases."""
    scaling = compute_full_scaling(case_ids, path_by_case, features)
    n_rows = int(scaling["n_rows"])
    X_all = np.empty((n_rows, len(features)), dtype=np.float32)
    y_all = np.empty(n_rows, dtype=np.float32)
    weights_all = np.empty(n_rows, dtype=np.float32)
    cursor = 0
    audit_records = []
    n_cases = len(case_ids)
    for case_id in case_ids:
        payload, coordinate_audit = load_case_payload(case_id, path_by_case, features)
        X = payload["X"]
        y = payload["y"]
        n_case_rows = len(y)
        if n_case_rows != scaling["row_counts"][case_id]:
            raise AssertionError(f"{case_id} changed row count between full-data passes")
        next_cursor = cursor + n_case_rows
        if scale:
            X_all[cursor:next_cursor] = (
                (X - scaling["x_mean"].astype(np.float32))
                / scaling["x_std"].astype(np.float32)
            )
            y_all[cursor:next_cursor] = (
                (y - np.float32(scaling["y_mean"])) / np.float32(scaling["y_std"])
            )
        else:
            X_all[cursor:next_cursor] = X
            y_all[cursor:next_cursor] = y

        if use_tail_weights:
            case_weights, weight_audit = make_case_tail_weights(y)
        else:
            case_weights = np.ones(n_case_rows, dtype=np.float32)
            weight_audit = {
                "normalised_weight_mean": 1.0,
                "normalised_weight_min": 1.0,
                "normalised_weight_max": 1.0,
            }
        case_equaliser = n_rows / (n_cases * n_case_rows)
        weights_all[cursor:next_cursor] = case_weights * np.float32(case_equaliser)
        audit_records.append({
            "case_id": case_id,
            "n_elements": n_case_rows,
            "all_elements_used": True,
            "case_equaliser": case_equaliser,
            "coordinate_source": coordinate_audit["coordinate_source"],
            **weight_audit,
        })
        cursor = next_cursor
    if cursor != n_rows:
        raise AssertionError(f"Assembled {cursor} rows; expected {n_rows}")
    weights_all /= np.float32(weights_all.mean(dtype=np.float64))
    return {
        "X": np.ascontiguousarray(X_all),
        "y": np.ascontiguousarray(y_all),
        "weights": np.ascontiguousarray(weights_all),
        "scaling": scaling,
        "audit": pd.DataFrame(audit_records),
    }


def scale_feature_matrix(X: np.ndarray, scaling: dict) -> np.ndarray:
    return np.ascontiguousarray(
        (X - scaling["x_mean"].astype(np.float32))
        / scaling["x_std"].astype(np.float32),
        dtype=np.float32,
    )


def unscale_target(y_scaled: np.ndarray, scaling: dict) -> np.ndarray:
    values = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
    return values * scaling["y_std"] + scaling["y_mean"]


def scaling_records(label: str | int, features: Sequence[str], scaling: dict) -> list[dict]:
    records = []
    for index, feature in enumerate(features):
        records.append({
            "label": label,
            "variable": feature,
            "role": "input",
            "mean": float(scaling["x_mean"][index]),
            "std": float(scaling["x_std"][index]),
        })
    records.append({
        "label": label,
        "variable": TARGET_COL,
        "role": "target",
        "mean": float(scaling["y_mean"]),
        "std": float(scaling["y_std"]),
    })
    return records


def assert_no_final_cases(case_ids: Iterable[str], manifest: pd.DataFrame) -> None:
    forbidden = set(final_test_case_ids(manifest))
    overlap = sorted(set(case_ids) & forbidden)
    if overlap:
        raise AssertionError(f"Locked final-test cases entered development processing: {overlap[:10]}")
