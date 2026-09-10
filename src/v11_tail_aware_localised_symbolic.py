"""V11 tail-aware localised residual symbolic-regression pilot.

V11 is an auditable increment over the corresponding completed V10 rotation. It
does not restart the full hierarchy.  Instead it addresses the two remaining
V10 weaknesses separately:

1. a case-level ridge formula calibrates the mean V10 residual; and
2. one recoverable PySR search learns a zero-case-mean local residual with
   stronger P95/P99 exposure and smooth hotspot radial-basis features.

The deployed expression is

    stress_v11 = stress_v10 + delta_mean(context)
                 + scale_v8 * (tail_raw - case_mean(tail_raw))

All terms use predictor fields only.  The case mean of ``tail_raw`` is a
spatial average of the formula output, not a stress-derived quantity.  Final
test cases remain sealed throughout this development pilot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from ct3_common import (  # noqa: E402
    RANDOM_SEED,
    TARGET_COL,
    aggregate_case_metrics,
    assert_no_final_cases,
    evaluate_prediction_arrays,
    read_complete_case,
    save_json,
)
from hierarchical_feasibility import (  # noqa: E402
    FeasibilityConfig,
    _case_summary_row,
    load_inputs,
)
from hierarchical_symbolic_regression import (  # noqa: E402
    CASE_FORMULA_FEATURES,
    SHAPE_FORMULA_FEATURES,
    _compiled_formula,
    _load_scaling,
    _save_scaling,
    _scaling_payload,
    _shortlist_shape_frontier,
    _tail_adaptation,
)
from signed_staged_residual_symbolic import (  # noqa: E402
    ALL_V10_FEATURES,
    BOUNDARY_DIRECTION_FEATURES,
    PHYSICAL_CONTEXT_FEATURES,
    STAGE_A_FEATURES,
    STAGE_B_FEATURES,
    STRESS_TIER_LABELS,
    STRESS_TIER_QUOTAS,
    SignedStagedPilotConfig,
    _atomic_json,
    _baseline_functions,
    _feature_indices,
    _predict_baseline_components,
    preflight_signed_staged_pilot,
)


V11_TIER_TARGET_MASS = [0.55, 0.15, 0.20, 0.10]
ACTUAL_TIER_PROBABILITY = [0.90, 0.05, 0.04, 0.01]
POSITIVE_TAIL_WEIGHT_MULTIPLIER = 1.75
N_HOTSPOT_RBF = 6
MEAN_CALIBRATION_FEATURES = list(PHYSICAL_CONTEXT_FEATURES)

V11_DIRECT_FEATURES = [
    "fluence_rate",
    "temperature",
    "weight_loss_rate",
    "fluence_rate_within_case_z",
    "temperature_within_case_z",
    "weight_loss_rate_within_case_z",
    "fluence_rate_mean",
    "fluence_rate_p95",
    "temperature_mean",
    "temperature_p95",
    "weight_loss_rate_mean",
    "weight_loss_rate_std",
    "rho_fraction_proxy",
    "z_fraction_proxy",
    "theta_fraction_proxy",
    "nearest_radial_boundary_fraction_proxy",
    "nearest_axial_boundary_fraction_proxy",
    "nearest_angular_boundary_fraction_proxy",
    "signed_radial_position_proxy",
    "signed_axial_position_proxy",
    "signed_angular_position_proxy",
    "radial_axial_signed_interaction_proxy",
    "radial_angular_signed_interaction_proxy",
    "axial_angular_signed_interaction_proxy",
    "boundary_corner_proximity_proxy",
]
V11_HARMONIC_FEATURES = [
    "theta_sin_2_proxy",
    "theta_cos_2_proxy",
    "theta_sin_4_proxy",
    "theta_cos_4_proxy",
]
V11_BASELINE_FEATURES = [
    "v10_fixed_shape_proxy",
    "v10_geometry_residual_proxy",
    "v10_physical_residual_proxy",
    "v10_normalised_shape_proxy",
]
V11_RBF_FEATURES = [f"hotspot_rbf_{index:02d}" for index in range(1, N_HOTSPOT_RBF + 1)]
V11_RBF_FEATURES += ["hotspot_rbf_max_proxy"]
V11_FEATURES = (
    V11_DIRECT_FEATURES
    + V11_HARMONIC_FEATURES
    + V11_BASELINE_FEATURES
    + V11_RBF_FEATURES
)


@dataclass(frozen=True)
class V11PilotConfig:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    rows_per_training_case: int = 5_000
    total_niterations: int = 500
    populations: int = 8
    segment_niterations: int = 100
    population_size: int = 40
    ncycles_per_iteration: int = 100
    batch_size: int = 50_000
    maxsize: int = 32
    maxdepth: int = 10
    julia_threads: int = 8
    no_activity_timeout_seconds: int = 45 * 60
    segment_wall_timeout_seconds: int = 3 * 60 * 60
    watchdog_poll_seconds: int = 60
    max_attempts_per_segment: int = 3
    max_candidates_for_full_validation: int = 20
    force_rebuild_training_cache: bool = False

    def validate(self) -> None:
        if self.iteration not in {1, 2, 3, 4}:
            raise ValueError("V11 iteration must be one of 1, 2, 3 or 4")
        if self.rows_per_training_case != sum(STRESS_TIER_QUOTAS):
            raise ValueError("V11 reuses exactly 5,000 V10 discovery rows per case")
        if self.total_niterations % self.segment_niterations != 0:
            raise ValueError("Total iterations must be divisible by segment iterations")
        if self.populations != 8:
            raise ValueError("V11 formal rotations are locked to eight populations")
        if not np.isclose(sum(V11_TIER_TARGET_MASS), 1.0):
            raise ValueError("V11 tier loss mass must sum to one")
        if self.segment_wall_timeout_seconds <= self.no_activity_timeout_seconds:
            raise ValueError("Segment wall timeout must exceed inactivity timeout")
        if len(set(V11_FEATURES)) != len(V11_FEATURES):
            raise ValueError("V11 feature registry contains duplicates")

    @property
    def n_segments(self) -> int:
        return self.total_niterations // self.segment_niterations

    @property
    def population_iterations_per_segment(self) -> int:
        return self.segment_niterations * self.populations

    @property
    def population_iterations_total(self) -> int:
        return self.total_niterations * self.populations


def output_directory(package_root: Path, config: V11PilotConfig) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "11_tail_aware_localised_symbolic"
        / config.output_subdir
    )


def v10_output_directory(package_root: Path, iteration: int = 1) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "10_signed_staged_residual_symbolic_pilot"
        / f"iteration_{iteration}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _v10_required_paths(package_root: Path, iteration: int = 1) -> dict[str, Path]:
    root = v10_output_directory(package_root, iteration)
    return {
        "completion": root / "pilot_complete.json",
        "composite": root / "selected_composite_formula.csv",
        "geometry_row": root / "stages" / "stage_geometry" / "selected_formula.csv",
        "physical_row": root / "stages" / "stage_physical" / "selected_formula.csv",
        "geometry_scaling": root / "training_cache" / "stage_a_scaling.csv",
        "physical_scaling": root / "training_cache" / "stage_b_scaling.csv",
        "all_X": root / "training_cache" / "all_v10_features.npy",
        "base_target": root / "training_cache" / "deployable_normalised_residual_target.npy",
        "sample_manifest": root / "training_cache" / "training_sample_manifest.csv.gz",
        "cache_metadata": root / "training_cache" / "cache_metadata.json",
        "split_metrics": root / "selected_models_split_metrics.csv",
    }


def _load_v10_state(package_root: Path, baseline: dict, iteration: int = 1) -> dict:
    paths = _v10_required_paths(package_root, iteration)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"V11 requires the completed local V10 Iteration-{iteration} artifacts:\n"
            + "\n".join(missing)
        )
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("The V10 pilot completion marker is not complete")
    geometry_row = pd.read_csv(paths["geometry_row"]).iloc[0]
    physical_row = pd.read_csv(paths["physical_row"]).iloc[0]
    geometry_scaling = _load_scaling(paths["geometry_scaling"])
    physical_scaling = _load_scaling(paths["physical_scaling"])
    composite = pd.read_csv(paths["composite"]).iloc[0]
    return {
        "paths": paths,
        "completion": completion,
        "geometry_row": geometry_row,
        "physical_row": physical_row,
        "geometry_scaling": geometry_scaling,
        "physical_scaling": physical_scaling,
        "geometry_function": _compiled_formula(geometry_row, geometry_scaling),
        "physical_function": _compiled_formula(physical_row, physical_scaling),
        "baseline_functions": _baseline_functions(baseline),
        "composite": composite,
    }


def preflight_v11_pilot(package_root: Path, config: V11PilotConfig) -> dict:
    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the reviewed V10 input and frozen-baseline checks, including the
    # 119/15/15/50 split and the guarantee that final elements are not read.
    v10_preflight = preflight_signed_staged_pilot(
        package_root,
        SignedStagedPilotConfig(
            iteration=config.iteration,
            output_subdir=f"iteration_{config.iteration}",
        ),
    )
    inputs = load_inputs(
        package_root,
        FeasibilityConfig(
            iteration=config.iteration,
            rows_per_case=config.rows_per_training_case,
            output_subdir=f"v11_tail_aware_input_check_iteration_{config.iteration}",
        ),
    )
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    v10_state = _load_v10_state(
        package_root,
        v10_preflight["baseline"],
        config.iteration,
    )
    worker = package_root / "scripts" / "run_v11_tail_segment.py"

    hash_rows = []
    for name, path in v10_state["paths"].items():
        # Hash every compact formula/metadata artifact and the exact sampled
        # arrays used by V11. This prevents silent baseline drift on resume.
        hash_rows.append({
            "artifact": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    hash_audit = pd.DataFrame(hash_rows)
    hash_audit.to_csv(output_dir / "v10_input_hash_audit.csv", index=False)

    metadata = json.loads(
        v10_state["paths"]["cache_metadata"].read_text(encoding="utf-8")
    )
    checks = pd.DataFrame([
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "v10_cache_rows", "value": metadata.get("n_rows"), "expected": 595_000},
        {"check": "v10_cache_train_ids", "value": metadata.get("train_ids"), "expected": inputs["train_ids"]},
        {"check": "v10_cache_feature_order", "value": metadata.get("all_features"), "expected": list(ALL_V10_FEATURES)},
        {"check": "v11_features", "value": len(V11_FEATURES), "expected": 40},
        {"check": "worker_script_exists", "value": worker.exists(), "expected": True},
        {"check": "recoverable_segments", "value": config.n_segments, "expected": 5},
        {"check": "population_iterations", "value": config.population_iterations_total, "expected": 4_000},
    ])
    checks["pass"] = [
        value == expected
        for value, expected in zip(checks["value"], checks["expected"])
    ]
    checks.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not checks["pass"].all():
        failed = checks.loc[~checks["pass"], "check"].tolist()
        raise RuntimeError(f"V11 preflight failed: {failed}")
    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "baseline": v10_preflight["baseline"],
        "v10": v10_state,
        "worker_path": worker,
        "hash_audit": hash_audit,
        "checks": checks,
    }


def _matrix_column(matrix: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(matrix[:, ALL_V10_FEATURES.index(name)], dtype=np.float64)


def _v10_sample_components(matrix: np.ndarray, state: dict) -> dict[str, np.ndarray]:
    fixed = state["baseline_functions"]["shape"](
        matrix[:, : len(SHAPE_FORMULA_FEATURES)]
    )
    geometry = state["geometry_function"](
        matrix[:, _feature_indices(STAGE_A_FEATURES)]
    )
    physical = state["physical_function"](
        matrix[:, _feature_indices(STAGE_B_FEATURES)]
    )
    return {
        "fixed": fixed,
        "geometry": geometry,
        "physical": physical,
        "normalised_shape": fixed + geometry + physical,
    }


def _fit_hotspot_rbf(matrix: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    top_mask = manifest["stress_tier"].eq("top_p99").to_numpy()
    fractions = np.column_stack([
        _matrix_column(matrix, "rho_fraction_proxy"),
        _matrix_column(matrix, "z_fraction_proxy"),
        _matrix_column(matrix, "theta_fraction_proxy"),
    ])
    top = fractions[top_mask]
    if len(top) < N_HOTSPOT_RBF * 100:
        raise ValueError("Insufficient top-P99 rows for stable hotspot centres")
    model = MiniBatchKMeans(
        n_clusters=N_HOTSPOT_RBF,
        random_state=RANDOM_SEED,
        batch_size=2048,
        n_init=10,
        max_iter=300,
    ).fit(top)
    labels = model.labels_
    rows = []
    for index, center in enumerate(model.cluster_centers_):
        distances = np.sqrt(np.sum((top[labels == index] - center) ** 2, axis=1))
        bandwidth = float(np.clip(1.5 * np.quantile(distances, 0.75), 0.04, 0.35))
        rows.append({
            "rbf_index": index + 1,
            "rho_fraction_center": float(center[0]),
            "z_fraction_center": float(center[1]),
            "theta_fraction_center": float(center[2]),
            "bandwidth": bandwidth,
            "top_p99_training_rows": int((labels == index).sum()),
        })
    return pd.DataFrame(rows).sort_values("rbf_index").reset_index(drop=True)


def _rbf_features(matrix: np.ndarray, centres: pd.DataFrame) -> np.ndarray:
    fractions = np.column_stack([
        _matrix_column(matrix, "rho_fraction_proxy"),
        _matrix_column(matrix, "z_fraction_proxy"),
        _matrix_column(matrix, "theta_fraction_proxy"),
    ])
    values = []
    for row in centres.itertuples(index=False):
        center = np.asarray([
            row.rho_fraction_center,
            row.z_fraction_center,
            row.theta_fraction_center,
        ])
        squared_distance = np.sum((fractions - center) ** 2, axis=1)
        values.append(np.exp(-squared_distance / (2.0 * row.bandwidth ** 2)))
    rbf = np.column_stack(values)
    return np.column_stack([rbf, np.max(rbf, axis=1)])


def build_v11_feature_matrix(
    matrix: np.ndarray,
    state: dict,
    centres: pd.DataFrame,
    components: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if components is None:
        components = _v10_sample_components(matrix, state)
    direct = np.column_stack([_matrix_column(matrix, name) for name in V11_DIRECT_FEATURES])
    theta_sin = _matrix_column(matrix, "theta_sin")
    theta_cos = _matrix_column(matrix, "theta_cos")
    sin_2 = 2.0 * theta_sin * theta_cos
    cos_2 = theta_cos ** 2 - theta_sin ** 2
    sin_4 = 2.0 * sin_2 * cos_2
    cos_4 = cos_2 ** 2 - sin_2 ** 2
    harmonics = np.column_stack([sin_2, cos_2, sin_4, cos_4])
    v10_features = np.column_stack([
        components["fixed"],
        components["geometry"],
        components["physical"],
        components["normalised_shape"],
    ])
    rbf = _rbf_features(matrix, centres)
    result = np.concatenate([direct, harmonics, v10_features, rbf], axis=1)
    if result.shape[1] != len(V11_FEATURES):
        raise AssertionError(
            f"V11 feature width {result.shape[1]} != {len(V11_FEATURES)}"
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def _tier_probability_mean(values: np.ndarray, tiers: np.ndarray) -> float:
    total = 0.0
    for index, probability in enumerate(ACTUAL_TIER_PROBABILITY):
        mask = tiers == STRESS_TIER_LABELS[index]
        if not mask.any():
            raise ValueError(f"Missing discovery tier {STRESS_TIER_LABELS[index]}")
        total += probability * float(np.mean(values[mask]))
    return total


def _case_scale(summary: pd.Series, state: dict) -> float:
    context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float64)[None, :]
    log_scale = float(state["baseline_functions"]["scale"](context)[0])
    if not np.isfinite(log_scale) or abs(log_scale) > 20.0:
        raise FloatingPointError("V8 scale formula is non-finite or explosive")
    return float(np.exp(log_scale))


def _fit_mean_calibration(case_table: pd.DataFrame) -> dict:
    # Geometry summaries are deliberately excluded here. They are strongly
    # collinear for this single brick geometry and produced large cancelling
    # coefficients in the first cache smoke test. The mean correction is kept
    # as a compact operating-condition formula, while geometry remains in the
    # zero-mean local residual branch.
    features = list(MEAN_CALIBRATION_FEATURES)
    X = case_table[features].to_numpy(dtype=np.float64)
    y = case_table["estimated_v10_mean_residual_mpa"].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X)
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 17)).fit(scaler.transform(X), y)
    coefficients = ridge.coef_ / scaler.scale_
    intercept = float(ridge.intercept_ - np.dot(ridge.coef_, scaler.mean_ / scaler.scale_))
    predicted = intercept + X @ coefficients
    audit = case_table[["case_id", "estimated_v10_mean_residual_mpa"]].copy()
    audit["predicted_mean_correction_mpa"] = predicted
    audit["error_mpa"] = predicted - y
    terms = [f"({coefficient:.16g})*{feature}" for feature, coefficient in zip(features, coefficients)]
    formula = f"({intercept:.16g})" + " + " + " + ".join(terms)
    return {
        "features": features,
        "coefficients": coefficients,
        "intercept": intercept,
        "alpha": float(ridge.alpha_),
        "formula": formula,
        "training_rmse": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "training_mae": float(np.mean(np.abs(predicted - y))),
        "audit": audit,
    }


def _mean_correction(summary: pd.Series, model: dict) -> float:
    values = summary[model["features"]].to_numpy(dtype=np.float64)
    return float(model["intercept"] + np.dot(values, model["coefficients"]))


def prepare_v11_training_cache(preflight: dict, config: V11PilotConfig) -> dict:
    output_dir = preflight["output_dir"]
    cache_dir = output_dir / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "X": cache_dir / "v11_tail_features.npy",
        "y": cache_dir / "v11_centered_tail_target.npy",
        "weights": cache_dir / "v11_tail_weights.npy",
        "scaling": cache_dir / "v11_tail_scaling.csv",
        "centres": cache_dir / "hotspot_rbf_centres.csv",
        "mean_model": cache_dir / "mean_calibration_model.json",
        "mean_audit": cache_dir / "mean_calibration_training_audit.csv",
        "weight_audit": cache_dir / "tail_weight_audit.csv",
        "metadata": cache_dir / "cache_metadata.json",
    }
    source_hashes = {
        row.artifact: row.sha256 for row in preflight["hash_audit"].itertuples(index=False)
    }
    expected = {
        "n_rows": 595_000,
        "features": list(V11_FEATURES),
        "train_ids": preflight["inputs"]["train_ids"],
        "tier_target_mass": V11_TIER_TARGET_MASS,
        "positive_tail_weight_multiplier": POSITIVE_TAIL_WEIGHT_MULTIPLIER,
        "mean_calibration_features": MEAN_CALIBRATION_FEATURES,
        "v10_source_hashes": source_hashes,
    }
    if all(path.exists() for path in paths.values()) and not config.force_rebuild_training_cache:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            mean_payload = json.loads(paths["mean_model"].read_text(encoding="utf-8"))
            mean_payload["coefficients"] = np.asarray(mean_payload["coefficients"], dtype=np.float64)
            mean_payload["formula"] = mean_payload["formula_original_variables"]
            return {
                "paths": paths,
                "metadata": metadata,
                "mean_model": mean_payload,
                "centres": pd.read_csv(paths["centres"]),
                "reused": True,
            }

    v10_paths = preflight["v10"]["paths"]
    all_X = np.load(v10_paths["all_X"], mmap_mode="r")
    base_target = np.load(v10_paths["base_target"], mmap_mode="r")
    manifest = pd.read_csv(v10_paths["sample_manifest"])
    if len(all_X) != len(base_target) or len(all_X) != len(manifest):
        raise ValueError("V10 training arrays and manifest are not row-aligned")
    if manifest["case_id"].drop_duplicates().tolist() != preflight["inputs"]["train_ids"]:
        raise ValueError(
            f"V10 sample manifest case order differs from frozen Iteration {config.iteration}"
        )

    components = _v10_sample_components(all_X, preflight["v10"])
    residual_norm = (
        np.asarray(base_target, dtype=np.float64)
        - components["geometry"]
        - components["physical"]
    )
    centres = _fit_hotspot_rbf(all_X, manifest)
    centres.to_csv(paths["centres"], index=False)
    V11_X = build_v11_feature_matrix(all_X, preflight["v10"], centres, components)

    case_rows = []
    centered_target = np.empty(len(residual_norm), dtype=np.float32)
    tiers = manifest["stress_tier"].astype(str).to_numpy()
    case_values = manifest["case_id"].astype(str).to_numpy()
    for case_id in preflight["inputs"]["train_ids"]:
        mask = case_values == case_id
        mean_norm = _tier_probability_mean(residual_norm[mask], tiers[mask])
        centered_target[mask] = (residual_norm[mask] - mean_norm).astype(np.float32)
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        row = {feature: float(summary[feature]) for feature in CASE_FORMULA_FEATURES}
        row.update({
            "case_id": case_id,
            "estimated_v10_mean_residual_normalised": mean_norm,
            "v8_predicted_scale": _case_scale(summary, preflight["v10"]),
        })
        row["estimated_v10_mean_residual_mpa"] = (
            row["estimated_v10_mean_residual_normalised"] * row["v8_predicted_scale"]
        )
        case_rows.append(row)
    case_table = pd.DataFrame(case_rows)
    mean_model = _fit_mean_calibration(case_table)
    mean_model["audit"].to_csv(paths["mean_audit"], index=False)
    serialisable_mean = {
        "method": "StandardScaler_plus_RidgeCV_converted_to_original_units",
        "features": mean_model["features"],
        "coefficients": mean_model["coefficients"].tolist(),
        "intercept": mean_model["intercept"],
        "alpha": mean_model["alpha"],
        "formula_original_variables": mean_model["formula"],
        "training_rmse": mean_model["training_rmse"],
        "training_mae": mean_model["training_mae"],
    }
    _atomic_json(paths["mean_model"], serialisable_mean)

    weights = np.empty(len(centered_target), dtype=np.float64)
    audit_rows = []
    for case_id in preflight["inputs"]["train_ids"]:
        case_mask = case_values == case_id
        for tier_index, tier_name in enumerate(STRESS_TIER_LABELS):
            mask = case_mask & (tiers == tier_name)
            weights[mask] = V11_TIER_TARGET_MASS[tier_index] / int(mask.sum())
    positive_tail = (centered_target > 0.0) & np.isin(tiers, ["p95_to_p99", "top_p99"])
    weights[positive_tail] *= POSITIVE_TAIL_WEIGHT_MULTIPLIER
    weights /= weights.mean()
    for tier_index, tier_name in enumerate(STRESS_TIER_LABELS):
        mask = tiers == tier_name
        audit_rows.append({
            "stress_tier": tier_name,
            "rows": int(mask.sum()),
            "configured_loss_mass_before_positive_multiplier": V11_TIER_TARGET_MASS[tier_index],
            "actual_normalised_weight_mass": float(weights[mask].sum() / weights.sum()),
            "positive_tail_rows": int((mask & positive_tail).sum()),
        })
    pd.DataFrame(audit_rows).to_csv(paths["weight_audit"], index=False)

    scaling = _scaling_payload(V11_X, centered_target, V11_FEATURES, weights)
    np.save(paths["X"], V11_X)
    np.save(paths["y"], centered_target)
    np.save(paths["weights"], weights.astype(np.float32))
    _save_scaling(scaling, "v11_centered_normalised_tail_residual", paths["scaling"])
    metadata = {
        **expected,
        "random_seed": RANDOM_SEED,
        "actual_tier_probability_for_mean_estimation": ACTUAL_TIER_PROBABILITY,
        "n_hotspot_rbf": N_HOTSPOT_RBF,
        "target": "V10 normalised residual minus its estimated complete-case mean",
        "tail_formula_is_centered_again_on_each_complete_case": True,
        "mean_calibration_formula": mean_model["formula"],
        "mean_calibration_alpha": mean_model["alpha"],
        "mean_calibration_train_rmse_mpa": mean_model["training_rmse"],
    }
    _atomic_json(paths["metadata"], metadata)
    del all_X, base_target, V11_X, residual_norm, centered_target, weights
    gc.collect()
    mean_model.pop("audit", None)
    return {
        "paths": paths,
        "metadata": metadata,
        "mean_model": mean_model,
        "centres": centres,
        "reused": False,
    }


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover
            process.terminate()
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
            process.wait(timeout=10)


def _activity_signature(paths: Sequence[Path]) -> tuple:
    result = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            result.append((str(path), None, None))
    return tuple(result)


def _append_csv(path: Path, row: dict) -> None:
    table = pd.DataFrame([row])
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True, sort=False)
    table.to_csv(path, index=False)


def _search_directory(output_dir: Path) -> Path:
    return output_dir / "stages" / "stage_tail_localised"


def _search_signature(preflight: dict, cache: dict, config: V11PilotConfig) -> dict:
    return {
        "method": "v11_mean_calibrated_centered_tail_localised_residual",
        "features": list(V11_FEATURES),
        "n_rows": cache["metadata"]["n_rows"],
        "train_ids": cache["metadata"]["train_ids"],
        "v10_source_hashes": cache["metadata"]["v10_source_hashes"],
        "tier_quotas": STRESS_TIER_QUOTAS,
        "tier_target_mass": V11_TIER_TARGET_MASS,
        "positive_tail_weight_multiplier": POSITIVE_TAIL_WEIGHT_MULTIPLIER,
        "total_niterations": config.total_niterations,
        "populations": config.populations,
        "segment_niterations": config.segment_niterations,
        "population_size": config.population_size,
        "ncycles_per_iteration": config.ncycles_per_iteration,
        "batch_size": config.batch_size,
        "maxsize": config.maxsize,
        "maxdepth": config.maxdepth,
        "operators": ["+", "-", "*", "/", "abs", "tanh", "gauss"],
        "elementwise_loss": "HuberLoss(1.0)",
        "precision": 64,
        "random_seed": RANDOM_SEED,
    }


def _completed_segments(stage_dir: Path, config: V11PilotConfig) -> list[int]:
    return [
        segment
        for segment in range(1, config.n_segments + 1)
        if (stage_dir / "segments" / f"segment_{segment:02d}" / "complete.json").exists()
    ]


def _write_progress(stage_dir: Path, config: V11PilotConfig) -> dict:
    completed = _completed_segments(stage_dir, config)
    contiguous = 0
    for segment in range(1, config.n_segments + 1):
        if segment in completed:
            contiguous = segment
        else:
            break
    payload = {
        "completed_segments": completed,
        "contiguous_completed_segments": contiguous,
        "total_segments": config.n_segments,
        "planned_population_iterations_completed": contiguous * config.population_iterations_per_segment,
        "target_population_iterations": config.population_iterations_total,
        "planned_progress_fraction": contiguous / config.n_segments,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(stage_dir / "search_progress.json", payload)
    return payload


def run_search_segment(
    preflight: dict,
    cache: dict,
    config: V11PilotConfig,
    segment: int,
) -> Path:
    stage_dir = _search_directory(preflight["output_dir"])
    segment_dir = stage_dir / "segments" / f"segment_{segment:02d}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    complete_path = segment_dir / "complete.json"
    canonical_frontier = segment_dir / "frontier.csv"
    if complete_path.exists() and canonical_frontier.exists():
        print(f"V11 tail segment {segment}/{config.n_segments}: reused", flush=True)
        return canonical_frontier

    run_id = f"v11_i{config.iteration}_tail_aware_localised_residual"
    run_directory = stage_dir / "search_state" / "pysr_runs" / run_id
    checkpoint = run_directory / "checkpoint.pkl"
    hall = run_directory / "hall_of_fame.csv"
    snapshots = stage_dir / "search_state" / "stable_snapshots"
    attempt_audit = stage_dir / "segment_attempt_audit.csv"
    for attempt in range(1, config.max_attempts_per_segment + 1):
        attempt_dir = segment_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "worker_stdout.log"
        stderr_path = attempt_dir / "worker_stderr.log"
        status_path = attempt_dir / "worker_status.json"
        resume = checkpoint.exists()
        command = [
            sys.executable,
            str(preflight["worker_path"]),
            "--features", str(cache["paths"]["X"]),
            "--target", str(cache["paths"]["y"]),
            "--weights", str(cache["paths"]["weights"]),
            "--scaling", str(cache["paths"]["scaling"]),
            "--feature-names", json.dumps(V11_FEATURES),
            "--state-root", str(stage_dir / "search_state"),
            "--segment-output", str(attempt_dir),
            "--run-id", run_id,
            "--segment", str(segment),
            "--attempt", str(attempt),
            "--niterations", str(config.segment_niterations),
            "--populations", str(config.populations),
            "--population-size", str(config.population_size),
            "--ncycles", str(config.ncycles_per_iteration),
            "--batch-size", str(config.batch_size),
            "--maxsize", str(config.maxsize),
            "--maxdepth", str(config.maxdepth),
            "--seed", str(RANDOM_SEED + 11_000),
        ]
        if resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["JULIA_NUM_THREADS"] = str(config.julia_threads)
        environment["PYTHONPATH"] = os.pathsep.join([
            str(preflight["package_root"] / "src"),
            environment.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep)
        started = time.time()
        reason = "process_exit"
        stalled = False
        wall_timeout = False
        print(
            f"V11 tail segment {segment}/{config.n_segments}, attempt {attempt}: "
            f"{'resume' if resume else 'fresh search'}",
            flush=True,
        )
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=preflight["package_root"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            activity_paths = [stdout_path, stderr_path, status_path, hall, checkpoint]
            last_signature = _activity_signature(activity_paths)
            last_activity = time.time()
            last_report = 0.0
            while process.poll() is None:
                now = time.time()
                elapsed = now - started
                signature = _activity_signature(activity_paths)
                if signature != last_signature:
                    last_signature = signature
                    last_activity = now
                inactive = now - last_activity
                if inactive >= config.no_activity_timeout_seconds:
                    reason = "no_file_activity_timeout"
                    stalled = True
                    _terminate_process_tree(process)
                    break
                if elapsed >= config.segment_wall_timeout_seconds:
                    reason = "segment_wall_timeout"
                    wall_timeout = True
                    _terminate_process_tree(process)
                    break
                if last_report == 0.0 or elapsed - last_report >= 300:
                    print(
                        f"V11 segment {segment}, attempt {attempt}: "
                        f"elapsed={elapsed / 60:.1f} min, "
                        f"inactive={inactive / 60:.1f} min, checkpoint={checkpoint.exists()}",
                        flush=True,
                    )
                    last_report = elapsed
                _atomic_json(segment_dir / "watchdog_status.json", {
                    "segment": segment,
                    "attempt": attempt,
                    "pid": process.pid,
                    "elapsed_seconds": elapsed,
                    "inactive_seconds": inactive,
                    "checkpoint_exists": checkpoint.exists(),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
                time.sleep(config.watchdog_poll_seconds)
            return_code = process.poll()

        worker_frontier = attempt_dir / "frontier.csv"
        audit = {
            "segment": segment,
            "attempt": attempt,
            "resume_from_checkpoint": resume,
            "return_code": return_code,
            "termination_reason": reason,
            "stalled": stalled,
            "wall_timed_out": wall_timeout,
            "elapsed_seconds": time.time() - started,
            "checkpoint_exists_after": checkpoint.exists(),
            "worker_frontier_exists": worker_frontier.exists(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _append_csv(attempt_audit, audit)
        if return_code == 0 and worker_frontier.exists() and checkpoint.exists():
            frontier = pd.read_csv(worker_frontier)
            frontier["completed_segment"] = segment
            frontier["completed_attempt"] = attempt
            frontier.to_csv(canonical_frontier, index=False)
            snapshot = snapshots / f"segment_{segment:02d}"
            snapshot.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, snapshot / "checkpoint.pkl")
            if hall.exists():
                shutil.copy2(hall, snapshot / "hall_of_fame.csv")
            _atomic_json(complete_path, {
                "status": "complete",
                "segment": segment,
                "attempt": attempt,
                "planned_population_iterations": config.population_iterations_per_segment,
                "elapsed_seconds": time.time() - started,
            })
            _write_progress(stage_dir, config)
            print(
                f"V11 tail segment {segment}: complete; planned cumulative "
                f"{segment * config.population_iterations_per_segment}/"
                f"{config.population_iterations_total}",
                flush=True,
            )
            return canonical_frontier

        previous = snapshots / f"segment_{segment - 1:02d}"
        if segment > 1 and (previous / "checkpoint.pkl").exists():
            run_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous / "checkpoint.pkl", checkpoint)
            if (previous / "hall_of_fame.csv").exists():
                shutil.copy2(previous / "hall_of_fame.csv", hall)
            recovery = f"rolled_back_to_stable_segment_{segment - 1:02d}"
        else:
            if run_directory.exists():
                destination = attempt_dir / "failed_run_directory"
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(run_directory), str(destination))
            recovery = "next_attempt_starts_fresh"
        table = pd.read_csv(attempt_audit)
        table.loc[
            (table["segment"] == segment) & (table["attempt"] == attempt),
            "recovery_action",
        ] = recovery
        table.to_csv(attempt_audit, index=False)
    raise RuntimeError(f"V11 segment {segment} failed after all retry attempts")


def run_symbolic_search(preflight: dict, cache: dict, config: V11PilotConfig) -> pd.DataFrame:
    stage_dir = _search_directory(preflight["output_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)
    signature = _search_signature(preflight, cache, config)
    signature_path = stage_dir / "search_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError("Existing V11 state has a different search signature")
    else:
        _atomic_json(signature_path, signature)
    latest = None
    for segment in range(1, config.n_segments + 1):
        latest = run_search_segment(preflight, cache, config, segment)
    if latest is None:
        raise RuntimeError("No V11 search segment completed")
    frontier = pd.read_csv(latest)
    frontier = (
        frontier.sort_values(["loss", "complexity"])
        .drop_duplicates("formula_scaled_sympy", keep="first")
        .reset_index(drop=True)
    )
    frontier["source_candidate_index"] = frontier["candidate_index"]
    frontier["candidate_index"] = np.arange(len(frontier), dtype=int)
    frontier["frontier_source"] = "v11_tail_localised_iteration1"
    frontier.to_csv(stage_dir / "frontier.csv", index=False)
    return frontier


def _full_case_components(
    frame: pd.DataFrame,
    summary: pd.Series,
    state: dict,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, np.ndarray]]:
    matrix, _, scale, fixed, baseline_stress = _predict_baseline_components(
        frame, summary, state["baseline_functions"]
    )
    geometry = state["geometry_function"](matrix[:, _feature_indices(STAGE_A_FEATURES)])
    physical = state["physical_function"](matrix[:, _feature_indices(STAGE_B_FEATURES)])
    components = {
        "fixed": fixed,
        "geometry": geometry,
        "physical": physical,
        "normalised_shape": fixed + geometry + physical,
    }
    v10_prediction = baseline_stress + scale * (geometry + physical)
    return matrix, scale, v10_prediction, components


def evaluate_candidates(
    candidates: pd.DataFrame,
    scaling: dict,
    preflight: dict,
    cache: dict,
    case_ids: Sequence[str],
    split: str,
    config: V11PilotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    functions = {
        int(row.candidate_index): _compiled_formula(row, scaling)
        for row in candidates.itertuples(index=False)
    }
    candidate_rows = []
    v10_rows = []
    mean_rows = []
    invalid: dict[int, str] = {}
    for position, case_id in enumerate(case_ids, start=1):
        print(
            f"[{position}/{len(case_ids)}] {split} complete-case V11: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        matrix, scale, v10_prediction, components = _full_case_components(
            frame, summary, preflight["v10"]
        )
        delta_mean = _mean_correction(summary, cache["mean_model"])
        mean_prediction = v10_prediction + delta_mean
        v10_rows.append({
            "iteration": config.iteration,
            "split": split,
            "model": "v10_signed_staged_symbolic",
            "case_id": case_id,
            **evaluate_prediction_arrays(actual, v10_prediction),
        })
        mean_rows.append({
            "iteration": config.iteration,
            "split": split,
            "model": "v11_mean_calibrated_v10",
            "case_id": case_id,
            **evaluate_prediction_arrays(actual, mean_prediction),
        })
        v11_X = build_v11_feature_matrix(
            matrix, preflight["v10"], cache["centres"], components
        )
        for candidate_index, function in functions.items():
            if candidate_index in invalid:
                continue
            try:
                raw = function(v11_X)
                centered = raw - float(np.mean(raw))
                predicted = mean_prediction + scale * centered
                candidate_rows.append({
                    "iteration": config.iteration,
                    "split": split,
                    "model": "v11_tail_aware_localised_candidate",
                    "candidate_index": candidate_index,
                    "case_id": case_id,
                    **evaluate_prediction_arrays(actual, predicted),
                })
            except Exception as exc:
                invalid[candidate_index] = repr(exc)
        del frame, actual, matrix, components, v11_X, v10_prediction, mean_prediction
        gc.collect()

    candidate_cases = pd.DataFrame(candidate_rows)
    references = {"v10": pd.DataFrame(v10_rows), "mean_only": pd.DataFrame(mean_rows)}
    records = []
    for candidate in candidates.itertuples(index=False):
        index = int(candidate.candidate_index)
        group = candidate_cases[candidate_cases["candidate_index"] == index]
        if index in invalid or group["case_id"].nunique() != len(case_ids):
            records.append({
                **candidate._asdict(),
                "candidate_valid": False,
                "invalid_reason": invalid.get(index, "incomplete_evaluation"),
            })
            continue
        aggregate = aggregate_case_metrics(group, ["candidate_index"]).iloc[0].to_dict()
        aggregate.pop("candidate_index", None)
        records.append({
            **candidate._asdict(),
            "candidate_valid": True,
            "invalid_reason": "",
            **{f"{split}_{key}": value for key, value in aggregate.items()},
            **{f"{split}_{key}": value for key, value in _tail_adaptation(group).items()},
        })
    return pd.DataFrame(records), candidate_cases, references


def _reference_aggregate(table: pd.DataFrame) -> pd.Series:
    return aggregate_case_metrics(table, ["iteration", "split", "model"]).iloc[0]


def score_and_select(
    metrics: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.Series:
    valid = metrics[metrics["candidate_valid"].fillna(False)].copy()
    if valid.empty:
        raise RuntimeError("No finite V11 candidate on every validation case")
    error_metrics = [
        ("validation_macro_rmse", 0.20),
        ("validation_mean_top5_actual_rmse", 0.15),
        ("validation_mean_p95_relative_error", 0.10),
        ("validation_mean_p99_relative_error", 0.20),
        ("validation_mean_p99_underprediction_fraction", 0.15),
    ]
    benefit_metrics = [
        ("validation_mean_top1pct_hotspot_overlap", 0.15),
        ("validation_mean_top1_recall_in_predicted_top5", 0.05),
    ]
    valid["engineering_selection_score"] = 0.0
    for column, weight in error_metrics:
        valid["engineering_selection_score"] += weight * valid[column].rank(
            pct=True, ascending=True, method="average"
        )
    for column, weight in benefit_metrics:
        valid["engineering_selection_score"] += weight * valid[column].rank(
            pct=True, ascending=False, method="average"
        )
    v10 = _reference_aggregate(references["v10"])
    v10_rmse = float(v10["macro_rmse"])
    valid["macro_rmse_improvement_vs_v10_fraction"] = (
        v10_rmse - valid["validation_macro_rmse"]
    ) / max(v10_rmse, 1e-12)
    valid["gate_numerical_guardrail"] = valid["validation_max_prediction_abs_max_ratio"] <= 5.0
    valid["gate_positive_macro_r2"] = valid["validation_macro_r2"] > 0.0
    valid["gate_improves_v10_rmse"] = valid["macro_rmse_improvement_vs_v10_fraction"] > 0.0
    valid["gate_improves_v10_rmse_by_1pct"] = valid["macro_rmse_improvement_vs_v10_fraction"] >= 0.01
    valid["gate_p95_relative_error"] = valid["validation_mean_p95_relative_error"] <= 0.15
    valid["gate_p99_relative_error"] = valid["validation_mean_p99_relative_error"] <= 0.15
    valid["gate_p99_underprediction"] = valid["validation_mean_p99_underprediction_fraction"] <= 0.10
    valid["gate_top1_hotspot_overlap"] = valid["validation_mean_top1pct_hotspot_overlap"] >= 0.60
    valid["gate_top1_recall"] = valid["validation_mean_top1_recall_in_predicted_top5"] >= 0.80
    gate_columns = [
        "gate_numerical_guardrail",
        "gate_positive_macro_r2",
        "gate_improves_v10_rmse_by_1pct",
        "gate_p95_relative_error",
        "gate_p99_relative_error",
        "gate_p99_underprediction",
        "gate_top1_hotspot_overlap",
        "gate_top1_recall",
    ]
    valid["all_v11_promotion_gates_pass"] = valid[gate_columns].all(axis=1)
    if valid["all_v11_promotion_gates_pass"].any():
        pool = valid[valid["all_v11_promotion_gates_pass"]].copy()
        status = "all_v11_promotion_gates"
    else:
        improved = valid[
            valid["gate_numerical_guardrail"]
            & valid["gate_positive_macro_r2"]
            & valid["gate_improves_v10_rmse"]
        ].copy()
        pool = improved if not improved.empty else valid[valid["gate_numerical_guardrail"]].copy()
        if pool.empty:
            pool = valid
        status = (
            "finite_positive_r2_and_improves_v10"
            if not improved.empty
            else "diagnostic_finite_candidates_only"
        )
    selected = pool.sort_values([
        "engineering_selection_score",
        "validation_mean_p99_underprediction_fraction",
        "validation_mean_top1pct_hotspot_overlap",
        "validation_macro_rmse",
        "complexity",
        "candidate_index",
    ], ascending=[True, True, False, True, True, True]).iloc[0].copy()
    selected["selection_pool_status"] = status
    selected["selection_method"] = (
        "validation-only tail-aware engineering rank; complexity is evaluated "
        "only after RMSE, P99 and hotspot behaviour"
    )
    selected["pilot_promotion_status"] = (
        "passes_all_v11_pilot_gates"
        if bool(selected["all_v11_promotion_gates_pass"])
        else "diagnostic_only_does_not_pass_all_v11_pilot_gates"
    )
    for column in valid.columns:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics.loc[valid.index, valid.columns] = valid
    metrics["selected_candidate"] = metrics["candidate_index"].eq(selected["candidate_index"])
    metrics.to_csv(output_dir / "candidate_validation_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "selected_formula.csv", index=False)
    return selected


def _save_formula(
    preflight: dict,
    cache: dict,
    selected: pd.Series,
    config: V11PilotConfig,
) -> dict:
    output_dir = preflight["output_dir"]
    v10_formula = str(preflight["v10"]["composite"]["composite_stress_formula"])
    log_scale = str(preflight["v10"]["composite"]["case_log_scale_formula"])
    mean_formula = str(cache["mean_model"]["formula"])
    tail_formula = str(selected["formula_original_variables"])
    fixed_shape_formula = str(preflight["v10"]["composite"]["fixed_global_shape_formula"])
    geometry_formula = str(preflight["v10"]["composite"]["geometry_residual_formula"])
    physical_formula = str(preflight["v10"]["composite"]["physical_residual_formula"])
    centered_tail = f"({tail_formula}) - case_mean({tail_formula})"
    composite = (
        f"({v10_formula}) + ({mean_formula}) + exp({log_scale}) * ({centered_tail})"
    )
    record = {
        "iteration": config.iteration,
        "method": "v10_plus_case_mean_calibration_plus_centered_tail_localised_residual",
        "v10_pilot_status": preflight["v10"]["composite"]["pilot_promotion_status"],
        "case_mean_calibration_formula": mean_formula,
        "tail_raw_formula": tail_formula,
        "tail_centering_rule": "tail_raw_minus_complete_case_predictor_only_mean",
        "composite_stress_formula": composite,
        "tail_complexity": int(selected["complexity"]),
        "pilot_promotion_status": selected["pilot_promotion_status"],
    }
    pd.DataFrame([record]).to_csv(output_dir / "selected_composite_formula.csv", index=False)
    lines = [
        "CT3 V11 tail-aware localised symbolic stress formula",
        "====================================================",
        "",
        "1. Frozen V10 stress predictor",
        f"sigma_v10 = {v10_formula}",
        "",
        "2. Case-level mean calibration (predictor summaries only)",
        f"delta_mu = {mean_formula}",
        "",
        "3. Tail/local hotspot residual",
        f"tail_raw = {tail_formula}",
        "tail_centered = tail_raw - case_mean(tail_raw)",
        "",
        "4. Combined V11 expression",
        f"sigma_v11 = {composite}",
        "",
        "The case mean above is the spatial mean of the predictor-only tail",
        "formula evaluated over the requested case. It is not derived from",
        "stress and therefore remains available at deployment.",
        "",
        "Smooth hotspot RBF definitions",
        "------------------------------",
        "rho_fraction_proxy = (rho-rho_min)/(rho_max-rho_min)",
        "z_fraction_proxy = (z-z_min)/(z_max-z_min)",
        "theta_fraction_proxy = (theta-theta_min)/(theta_max-theta_min)",
        "nearest_*_boundary_fraction_proxy = min(fraction, 1-fraction)",
        "signed_*_position_proxy = 2*fraction-1",
        "pairwise interaction proxies are products of signed positions",
        "boundary_corner_proximity_proxy = nearest_rho*nearest_z*nearest_theta",
        "",
    ]
    for row in cache["centres"].itertuples(index=False):
        lines.append(
            f"hotspot_rbf_{row.rbf_index:02d} = exp(-((rho_fraction-"
            f"{row.rho_fraction_center:.8g})^2 + (z_fraction-"
            f"{row.z_fraction_center:.8g})^2 + (theta_fraction-"
            f"{row.theta_fraction_center:.8g})^2) / "
            f"(2*{row.bandwidth:.8g}^2))"
        )
    lines.extend([
        "hotspot_rbf_max_proxy = max(hotspot_rbf_01, ..., hotspot_rbf_06)",
        "theta_sin_2_proxy = 2*sin(theta)*cos(theta)",
        "theta_cos_2_proxy = cos(theta)^2-sin(theta)^2",
        "theta_sin_4_proxy = 2*theta_sin_2_proxy*theta_cos_2_proxy",
        "theta_cos_4_proxy = theta_cos_2_proxy^2-theta_sin_2_proxy^2",
        "",
        f"v10_fixed_shape_proxy = {fixed_shape_formula}",
        f"v10_geometry_residual_proxy = {geometry_formula}",
        f"v10_physical_residual_proxy = {physical_formula}",
        "v10_normalised_shape_proxy = v10_fixed_shape_proxy + "
        "v10_geometry_residual_proxy + v10_physical_residual_proxy",
        "",
        "All RBF centres were fitted from training-case top-P99 locations only.",
        "The 50 final-test cases were not read.",
    ])
    (output_dir / "selected_composite_formula.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return record


def _save_plots(split_metrics: pd.DataFrame, case_metrics: pd.DataFrame, output_dir: Path) -> None:
    validation = split_metrics[split_metrics["split"] == "validation"].copy()
    metrics = [
        ("macro_rmse", "Macro RMSE", False),
        ("mean_p99_underprediction_fraction", "P99 underprediction", False),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (column, title, higher_better) in zip(axes, metrics):
        values = validation.sort_values(column, ascending=higher_better)
        axis.barh(values["model"], values[column], color="#246b73")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "v11_validation_model_comparison.png", dpi=180)
    plt.close(fig)

    selected = case_metrics[
        (case_metrics["split"] == "validation")
        & (case_metrics["model"] == "v11_tail_aware_localised_symbolic")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, tail in zip(axes, ["p95", "p99"]):
        actual = selected[f"actual_{tail}"]
        predicted = selected[f"predicted_{tail}"]
        axis.scatter(actual, predicted, color="#cf6b32", s=42)
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.plot([lower, upper], [lower, upper], "--", color="#555555")
        axis.set_xlabel(f"Actual {tail.upper()} stress")
        axis.set_ylabel(f"Predicted {tail.upper()} stress")
        axis.set_title(f"V11 {tail.upper()} adaptation")
    fig.tight_layout()
    fig.savefig(output_dir / "v11_selected_tail_adaptation.png", dpi=180)
    plt.close(fig)


def _start_sleep_inhibitor(output_dir: Path) -> subprocess.Popen | None:
    if platform.system() != "Darwin" or shutil.which("caffeinate") is None:
        return None
    process = subprocess.Popen(
        ["caffeinate", "-dimsu", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _atomic_json(output_dir / "sleep_inhibitor.json", {
        "enabled": True,
        "parent_pid": os.getpid(),
        "caffeinate_pid": process.pid,
    })
    return process


def _stop_sleep_inhibitor(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_v11_pilot(package_root: Path, config: V11PilotConfig) -> dict:
    started = time.time()
    preflight = preflight_v11_pilot(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "pilot_complete.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text(encoding="utf-8"))

    save_json(output_dir / "run_configuration.json", asdict(config))
    pd.DataFrame([
        {
            "feature": feature,
            "direct_v10_input": feature in V11_DIRECT_FEATURES,
            "periodic_harmonic": feature in V11_HARMONIC_FEATURES,
            "frozen_v10_component": feature in V11_BASELINE_FEATURES,
            "training_hotspot_rbf": feature in V11_RBF_FEATURES,
        }
        for feature in V11_FEATURES
    ]).to_csv(output_dir / "feature_registry.csv", index=False)
    _atomic_json(output_dir / "design_rationale.json", {
        "parent_model": (
            f"completed V10 Iteration-{config.iteration} signed staged residual model"
        ),
        "mean_calibration": "RidgeCV on 119 case-level predictor summaries",
        "tail_target": "V10 residual centered within each training case",
        "tail_formula_deployment": "raw tail formula centered over predictor field",
        "sampling_tier_quotas": STRESS_TIER_QUOTAS,
        "sampling_target_loss_mass": V11_TIER_TARGET_MASS,
        "positive_tail_weight_multiplier": POSITIVE_TAIL_WEIGHT_MULTIPLIER,
        "hotspot_features": "six training-only smooth RBF centres plus periodic harmonics",
        "loss": "HuberLoss(1.0) with static tail/positive-residual weights",
        "full_complete_case_validation": True,
        "final_test_cases_read": 0,
    })
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "prepare_v11_cache",
        "final_test_cases_read": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    inhibitor = _start_sleep_inhibitor(output_dir)
    try:
        cache = prepare_v11_training_cache(preflight, config)
        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "tail_localised_symbolic_search",
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        frontier = run_symbolic_search(preflight, cache, config)
        shortlist = _shortlist_shape_frontier(
            frontier, config.max_candidates_for_full_validation
        ).copy()
        stage_dir = _search_directory(output_dir)
        shortlist.to_csv(stage_dir / "full_validation_shortlist.csv", index=False)
        scaling = _load_scaling(cache["paths"]["scaling"])
        validation_metrics, validation_cases, validation_refs = evaluate_candidates(
            shortlist,
            scaling,
            preflight,
            cache,
            preflight["inputs"]["validation_ids"],
            "validation",
            config,
        )
        selected = score_and_select(validation_metrics, validation_refs, stage_dir)
        selected_table = pd.DataFrame([selected])
        _, internal_cases, internal_refs = evaluate_candidates(
            selected_table,
            scaling,
            preflight,
            cache,
            preflight["inputs"]["internal_ids"],
            "internal_test",
            config,
        )
        selected_index = int(selected["candidate_index"])
        selected_validation = validation_cases[
            validation_cases["candidate_index"] == selected_index
        ].copy()
        selected_validation["model"] = "v11_tail_aware_localised_symbolic"
        internal_cases["model"] = "v11_tail_aware_localised_symbolic"
        selected_cases = pd.concat([selected_validation, internal_cases], ignore_index=True)
        references = pd.concat([
            validation_refs["v10"],
            validation_refs["mean_only"],
            internal_refs["v10"],
            internal_refs["mean_only"],
        ], ignore_index=True)
        all_cases = pd.concat([references, selected_cases], ignore_index=True)
        split_metrics = aggregate_case_metrics(all_cases, ["iteration", "split", "model"])
        adaptation_rows = []
        for (split, model), group in all_cases.groupby(["split", "model"], observed=True):
            adaptation_rows.append({"split": split, "model": model, **_tail_adaptation(group)})
        adaptation = pd.DataFrame(adaptation_rows)
        all_cases.to_csv(
            output_dir / "selected_models_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        split_metrics.to_csv(output_dir / "selected_models_split_metrics.csv", index=False)
        adaptation.to_csv(output_dir / "selected_models_tail_adaptation.csv", index=False)
        formula_record = _save_formula(preflight, cache, selected, config)
        _save_plots(split_metrics, all_cases, output_dir)
        result = {
            "status": "complete",
            "iteration": config.iteration,
            "prototype_only": True,
            "parent_model": f"v10_iteration_{config.iteration}",
            "train_cases": len(preflight["inputs"]["train_ids"]),
            "validation_cases": len(preflight["inputs"]["validation_ids"]),
            "internal_test_cases": len(preflight["inputs"]["internal_ids"]),
            "final_test_cases_read": 0,
            "training_rows": 595_000,
            "planned_population_iterations": config.population_iterations_total,
            "completed_segments": config.n_segments,
            "selected_candidate_index": int(selected["candidate_index"]),
            "selection_pool_status": selected["selection_pool_status"],
            "pilot_promotion_status": formula_record["pilot_promotion_status"],
            "elapsed_seconds": time.time() - started,
            "output_directory": str(output_dir),
        }
        _atomic_json(completion_path, result)
        _atomic_json(output_dir / "run_status.json", {
            **result,
            "stage": "complete",
        })
        return result
    except Exception as exc:
        _atomic_json(output_dir / "run_status.json", {
            "status": "failed",
            "error": repr(exc),
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        raise
    finally:
        _stop_sleep_inhibitor(inhibitor)
