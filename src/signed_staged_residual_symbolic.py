"""V10 signed, staged residual symbolic-regression pilot for CT3 FEM stress.

The frozen V8 Candidate-5 hierarchy supplies case mean, positive case scale,
and a global shape. V10 searches two independent, recoverable corrections:

    stress = mean + exp(log_scale) * (
        fixed_shape + geometry_residual + physical_residual
    )

The final-test cases remain sealed. Candidate selection uses complete validation
cases and the selected combined formula is reported once on internal-test cases.
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
    LOCAL_FEATURES,
    STANDARDISED_LOCAL_FEATURES,
    _case_summary_row,
    _joint_stratum_codes,
    _stratified_indices,
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


BOUNDARY_DIRECTION_FEATURES = [
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
ALL_V10_FEATURES = SHAPE_FORMULA_FEATURES + BOUNDARY_DIRECTION_FEATURES

GEOMETRY_CONTEXT_FEATURES = [
    "rho_mean",
    "rho_std",
    "rho_max",
    "z_mean",
    "z_std",
    "z_max",
    "theta_sin_std",
]
PHYSICAL_CONTEXT_FEATURES = [
    "fluence_rate_mean",
    "fluence_rate_p95",
    "temperature_mean",
    "temperature_p95",
    "weight_loss_rate_mean",
    "weight_loss_rate_std",
]
GEOMETRY_LOCAL_FEATURES = ["rho", "theta_sin", "theta_cos", "z"]
PHYSICAL_LOCAL_FEATURES = ["fluence_rate", "temperature", "weight_loss_rate"]

STAGE_A_FEATURES = (
    GEOMETRY_LOCAL_FEATURES
    + [f"{feature}_within_case_z" for feature in GEOMETRY_LOCAL_FEATURES]
    + GEOMETRY_CONTEXT_FEATURES
    + BOUNDARY_DIRECTION_FEATURES
)
STAGE_B_FEATURES = (
    PHYSICAL_LOCAL_FEATURES
    + [f"{feature}_within_case_z" for feature in PHYSICAL_LOCAL_FEATURES]
    + PHYSICAL_CONTEXT_FEATURES
    + GEOMETRY_LOCAL_FEATURES
    + BOUNDARY_DIRECTION_FEATURES
)

STRESS_TIER_LABELS = ["below_p90", "p90_to_p95", "p95_to_p99", "top_p99"]
STRESS_TIER_QUOTAS = [3_000, 750, 1_000, 250]
STRESS_TIER_TARGET_MASS = [0.60, 0.15, 0.20, 0.05]


@dataclass(frozen=True)
class SignedStagedPilotConfig:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    rows_per_training_case: int = 5_000
    total_niterations_per_stage: int = 500
    populations: int = 8
    segment_niterations: int = 100
    population_size: int = 40
    ncycles_per_iteration: int = 100
    batch_size: int = 50_000
    stage_a_maxsize: int = 32
    stage_a_maxdepth: int = 10
    stage_b_maxsize: int = 26
    stage_b_maxdepth: int = 9
    julia_threads: int = 8
    no_activity_timeout_seconds: int = 45 * 60
    segment_wall_timeout_seconds: int = 3 * 60 * 60
    watchdog_poll_seconds: int = 60
    max_attempts_per_segment: int = 3
    max_candidates_for_full_validation: int = 16
    force_rebuild_training_cache: bool = False

    def validate(self) -> None:
        if self.iteration not in {1, 2, 3, 4}:
            raise ValueError("V10 iteration must be one of 1, 2, 3 or 4")
        if self.rows_per_training_case != sum(STRESS_TIER_QUOTAS):
            raise ValueError("V10 requires exactly 5,000 discovery rows per case")
        if self.total_niterations_per_stage % self.segment_niterations != 0:
            raise ValueError("Stage iterations must be divisible by segment iterations")
        if self.populations != 8:
            raise ValueError("The pilot comparison is locked to eight populations")
        if self.segment_wall_timeout_seconds <= self.no_activity_timeout_seconds:
            raise ValueError("Segment wall timeout must exceed inactivity timeout")
        if not np.isclose(sum(STRESS_TIER_TARGET_MASS), 1.0):
            raise ValueError("Stress-tier target loss mass must sum to one")
        if len(set(ALL_V10_FEATURES)) != len(ALL_V10_FEATURES):
            raise ValueError("V10 feature registry contains duplicate names")

    @property
    def n_segments_per_stage(self) -> int:
        return self.total_niterations_per_stage // self.segment_niterations

    @property
    def population_iterations_per_segment(self) -> int:
        return self.segment_niterations * self.populations

    @property
    def population_iterations_per_stage(self) -> int:
        return self.total_niterations_per_stage * self.populations

    @property
    def total_population_iterations(self) -> int:
        return 2 * self.population_iterations_per_stage


def output_directory(package_root: Path, config: SignedStagedPilotConfig) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "10_signed_staged_residual_symbolic_pilot"
        / config.output_subdir
    )


def frozen_baseline_directory(package_root: Path) -> Path:
    return Path(package_root) / "shared" / "v10_authoritative_v8_candidate5"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_baseline(package_root: Path) -> dict:
    source = frozen_baseline_directory(package_root)
    manifest_path = source / "baseline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen V10 baseline manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("baseline_id") != "v8_iteration1_local_candidate5":
        raise ValueError("Unexpected V10 baseline identity")
    hash_rows = []
    for relative, expected_hash in manifest["files_sha256"].items():
        path = source / relative
        if not path.exists():
            raise FileNotFoundError(f"Frozen V8 artifact is missing: {path}")
        actual_hash = _sha256(path)
        hash_rows.append({
            "file": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "pass": actual_hash == expected_hash,
        })
    hash_audit = pd.DataFrame(hash_rows)
    if not hash_audit["pass"].all():
        raise RuntimeError("At least one frozen V8 artifact failed SHA-256 verification")
    paths = {
        "mean_row": source / "selected_case_mean_formula.csv",
        "mean_scaling": source / "case_mean_scaling.csv",
        "scale_row": source / "selected_case_log_scale_formula.csv",
        "scale_scaling": source / "case_log_scale_scaling.csv",
        "shape_row": source / "selected_shape_formula.csv",
        "shape_scaling": source / "training_cache" / "shape_scaling.csv",
        "split_metrics": source / "selected_formula_split_metrics.csv",
    }
    return {
        "manifest": manifest,
        "hash_audit": hash_audit,
        "mean_row": pd.read_csv(paths["mean_row"]).iloc[0],
        "mean_scaling": _load_scaling(paths["mean_scaling"]),
        "scale_row": pd.read_csv(paths["scale_row"]).iloc[0],
        "scale_scaling": _load_scaling(paths["scale_scaling"]),
        "shape_row": pd.read_csv(paths["shape_row"]).iloc[0],
        "shape_scaling": _load_scaling(paths["shape_scaling"]),
        "split_metrics": pd.read_csv(paths["split_metrics"]),
        "paths": paths,
    }


def preflight_signed_staged_pilot(
    package_root: Path,
    config: SignedStagedPilotConfig,
) -> dict:
    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(
        package_root,
        FeasibilityConfig(
            iteration=config.iteration,
            rows_per_case=config.rows_per_training_case,
            output_subdir=(
                f"v10_signed_staged_input_check_iteration_{config.iteration}"
            ),
        ),
    )
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    baseline = _load_frozen_baseline(package_root)
    baseline["hash_audit"].to_csv(
        output_dir / "frozen_baseline_hash_audit.csv", index=False
    )
    worker_path = package_root / "scripts" / "run_signed_residual_stage_segment.py"
    checks = [
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "all_v10_features", "value": len(ALL_V10_FEATURES), "expected": 40},
        {"check": "stage_a_features", "value": len(STAGE_A_FEATURES), "expected": 28},
        {"check": "stage_b_features", "value": len(STAGE_B_FEATURES), "expected": 29},
        {"check": "worker_script_exists", "value": worker_path.exists(), "expected": True},
        {
            "check": "frozen_baseline_hashes",
            "value": bool(baseline["hash_audit"]["pass"].all()),
            "expected": True,
        },
        {
            "check": "population_iterations_per_stage",
            "value": config.population_iterations_per_stage,
            "expected": 4_000,
        },
        {
            "check": "recoverable_segments_per_stage",
            "value": config.n_segments_per_stage,
            "expected": 5,
        },
    ]
    table = pd.DataFrame(checks)
    table["pass"] = table["value"] == table["expected"]
    table.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not table["pass"].all():
        raise RuntimeError("V10 signed-staged preflight failed")
    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "baseline": baseline,
        "worker_path": worker_path,
        "checks": table,
    }


def _safe_span(lower: float, upper: float) -> float:
    return max(float(upper) - float(lower), 1e-8)


def build_v10_feature_matrix(frame: pd.DataFrame, summary: pd.Series) -> np.ndarray:
    """Build the reviewed V8 inputs plus signed case-relative boundary proxies."""

    local = frame[LOCAL_FEATURES].to_numpy(dtype=np.float32)
    means = np.asarray(
        [summary[f"{feature}_mean"] for feature in LOCAL_FEATURES],
        dtype=np.float32,
    )
    stds = np.maximum(
        np.asarray(
            [summary[f"{feature}_std"] for feature in LOCAL_FEATURES],
            dtype=np.float32,
        ),
        np.float32(1e-8),
    )
    within_case = (local - means) / stds
    context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float32)
    context_matrix = np.broadcast_to(context, (len(frame), len(context)))

    fractions = []
    for feature in ["rho", "z", "theta"]:
        values = frame[feature].to_numpy(dtype=np.float32)
        fraction = np.clip(
            (values - np.float32(summary[f"{feature}_min"]))
            / np.float32(
                _safe_span(summary[f"{feature}_min"], summary[f"{feature}_max"])
            ),
            0.0,
            1.0,
        )
        fractions.append(fraction)
    rho_fraction, z_fraction, theta_fraction = fractions
    nearest_rho = np.minimum(rho_fraction, 1.0 - rho_fraction)
    nearest_z = np.minimum(z_fraction, 1.0 - z_fraction)
    nearest_theta = np.minimum(theta_fraction, 1.0 - theta_fraction)
    signed_rho = 2.0 * rho_fraction - 1.0
    signed_z = 2.0 * z_fraction - 1.0
    signed_theta = 2.0 * theta_fraction - 1.0
    boundary = np.column_stack([
        rho_fraction,
        z_fraction,
        theta_fraction,
        nearest_rho,
        nearest_z,
        nearest_theta,
        signed_rho,
        signed_z,
        signed_theta,
        signed_rho * signed_z,
        signed_rho * signed_theta,
        signed_z * signed_theta,
        nearest_rho * nearest_z * nearest_theta,
    ]).astype(np.float32, copy=False)
    return np.ascontiguousarray(
        np.concatenate([local, within_case, context_matrix, boundary], axis=1),
        dtype=np.float32,
    )


def _stress_tiers(stress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q90, q95, q99 = np.quantile(stress, [0.90, 0.95, 0.99])
    tier = np.zeros(len(stress), dtype=np.int8)
    tier[stress >= q90] = 1
    tier[stress >= q95] = 2
    tier[stress >= q99] = 3
    return tier, np.asarray([q90, q95, q99], dtype=np.float64)


def _discovery_selection(
    case_id: str,
    stress: np.ndarray,
    local_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tier, quantiles = _stress_tiers(stress)
    strata = _joint_stratum_codes(local_matrix)
    case_number = int(case_id.split("_")[-1])
    selected_parts = []
    for tier_index, quota in enumerate(STRESS_TIER_QUOTAS):
        pool = np.flatnonzero(tier == tier_index)
        chosen = _stratified_indices(
            pool,
            strata,
            quota,
            RANDOM_SEED + 30_000 * case_number + 307 * tier_index,
        )
        if len(chosen) != quota:
            raise AssertionError(
                f"{case_id}: tier {tier_index} has {len(chosen)} rows, expected {quota}"
            )
        selected_parts.append(chosen)
    positions = np.sort(np.concatenate(selected_parts))
    selected_tier = tier[positions]
    weights = np.empty(len(positions), dtype=np.float64)
    for tier_index, target_mass in enumerate(STRESS_TIER_TARGET_MASS):
        mask = selected_tier == tier_index
        weights[mask] = target_mass / int(mask.sum())
    weights /= weights.mean()
    return positions, weights, selected_tier, quantiles


def _baseline_functions(baseline: dict) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "mean": _compiled_formula(baseline["mean_row"], baseline["mean_scaling"]),
        "scale": _compiled_formula(baseline["scale_row"], baseline["scale_scaling"]),
        "shape": _compiled_formula(baseline["shape_row"], baseline["shape_scaling"]),
    }


def _feature_indices(features: Sequence[str]) -> list[int]:
    lookup = {name: index for index, name in enumerate(ALL_V10_FEATURES)}
    return [lookup[name] for name in features]


def _predict_baseline_components(
    frame: pd.DataFrame,
    summary: pd.Series,
    baseline_functions: dict,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    matrix = build_v10_feature_matrix(frame, summary)
    context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float64)[None, :]
    predicted_mean = float(baseline_functions["mean"](context)[0])
    predicted_log_scale = float(baseline_functions["scale"](context)[0])
    if not np.isfinite(predicted_log_scale) or abs(predicted_log_scale) > 20.0:
        raise FloatingPointError("Frozen V8 log-scale formula is non-finite or explosive")
    predicted_scale = float(np.exp(predicted_log_scale))
    fixed_shape = baseline_functions["shape"](
        matrix[:, : len(SHAPE_FORMULA_FEATURES)]
    )
    baseline_stress = predicted_mean + predicted_scale * fixed_shape
    return matrix, predicted_mean, predicted_scale, fixed_shape, baseline_stress


def prepare_stage_a_training_cache(
    preflight: dict,
    config: SignedStagedPilotConfig,
) -> dict:
    cache_dir = preflight["output_dir"] / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_X": cache_dir / "all_v10_features.npy",
        "stage_a_X": cache_dir / "stage_a_features.npy",
        "base_target": cache_dir / "deployable_normalised_residual_target.npy",
        "weights": cache_dir / "tier_mass_weights.npy",
        "stage_a_scaling": cache_dir / "stage_a_scaling.csv",
        "metadata": cache_dir / "cache_metadata.json",
        "audit": cache_dir / "training_sample_audit.csv",
        "manifest": cache_dir / "training_sample_manifest.csv.gz",
    }
    expected_rows = len(preflight["inputs"]["train_ids"]) * config.rows_per_training_case
    if all(path.exists() for path in paths.values()) and not config.force_rebuild_training_cache:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("n_rows") == expected_rows
            and metadata.get("all_features") == list(ALL_V10_FEATURES)
            and metadata.get("stage_a_features") == list(STAGE_A_FEATURES)
            and metadata.get("train_ids") == preflight["inputs"]["train_ids"]
            and metadata.get("tier_quotas") == STRESS_TIER_QUOTAS
            and metadata.get("tier_target_loss_mass") == STRESS_TIER_TARGET_MASS
            and metadata.get("baseline_id")
            == preflight["baseline"]["manifest"]["baseline_id"]
        ):
            return {"paths": paths, "metadata": metadata, "reused": True}

    all_X = np.empty((expected_rows, len(ALL_V10_FEATURES)), dtype=np.float32)
    base_target = np.empty(expected_rows, dtype=np.float32)
    weights = np.empty(expected_rows, dtype=np.float32)
    baseline_functions = _baseline_functions(preflight["baseline"])
    audit_rows = []
    manifest_parts = []
    cursor = 0
    started = time.perf_counter()

    for position, case_id in enumerate(preflight["inputs"]["train_ids"], start=1):
        print(
            f"[{position}/{len(preflight['inputs']['train_ids'])}] "
            f"V10 signed-residual sample: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        stress = frame[TARGET_COL].to_numpy(dtype=np.float64)
        local_matrix = frame[LOCAL_FEATURES].to_numpy(dtype=np.float32)
        selected, selected_weights, selected_tier, quantiles = _discovery_selection(
            case_id, stress, local_matrix
        )
        sampled = frame.iloc[selected]
        matrix, _, predicted_scale, _, baseline_stress = _predict_baseline_components(
            sampled, summary, baseline_functions
        )
        target = (
            sampled[TARGET_COL].to_numpy(dtype=np.float64) - baseline_stress
        ) / predicted_scale
        next_cursor = cursor + len(sampled)
        all_X[cursor:next_cursor] = matrix
        base_target[cursor:next_cursor] = target.astype(np.float32)
        weights[cursor:next_cursor] = selected_weights.astype(np.float32)
        for tier_index, tier_name in enumerate(STRESS_TIER_LABELS):
            tier_mask = selected_tier == tier_index
            audit_rows.append({
                "case_id": case_id,
                "stress_tier": tier_name,
                "selected_rows": int(tier_mask.sum()),
                "target_loss_mass": STRESS_TIER_TARGET_MASS[tier_index],
                "actual_selected_loss_mass": float(
                    selected_weights[tier_mask].sum() / selected_weights.sum()
                ),
                "q90": float(quantiles[0]),
                "q95": float(quantiles[1]),
                "q99": float(quantiles[2]),
                "deployable_residual_rmse": float(np.sqrt(np.mean(target ** 2))),
                "negative_residual_fraction": float((target < 0).mean()),
            })
        manifest_parts.append(pd.DataFrame({
            "iteration": config.iteration,
            "case_id": case_id,
            "element_id": sampled["element_id"].to_numpy(dtype=np.int64),
            "stress_tier": [STRESS_TIER_LABELS[index] for index in selected_tier],
            "discovery_weight": selected_weights.astype(np.float32),
        }))
        cursor = next_cursor
        del frame, sampled, matrix, stress, local_matrix, target, baseline_stress
        gc.collect()

    if cursor != expected_rows:
        raise AssertionError(f"Assembled {cursor} rows, expected {expected_rows}")
    weights /= np.float32(weights.mean(dtype=np.float64))
    stage_a_indices = _feature_indices(STAGE_A_FEATURES)
    stage_a_X = np.ascontiguousarray(all_X[:, stage_a_indices], dtype=np.float32)
    stage_a_scaling = _scaling_payload(
        stage_a_X, base_target, STAGE_A_FEATURES, weights
    )
    np.save(paths["all_X"], all_X)
    np.save(paths["stage_a_X"], stage_a_X)
    np.save(paths["base_target"], base_target)
    np.save(paths["weights"], weights)
    _save_scaling(stage_a_scaling, "deployable_normalised_residual", paths["stage_a_scaling"])
    pd.DataFrame(audit_rows).to_csv(paths["audit"], index=False)
    pd.concat(manifest_parts, ignore_index=True).to_csv(
        paths["manifest"], index=False, compression="gzip"
    )
    metadata = {
        "n_rows": int(len(all_X)),
        "all_features": list(ALL_V10_FEATURES),
        "stage_a_features": list(STAGE_A_FEATURES),
        "stage_b_features": list(STAGE_B_FEATURES),
        "train_ids": preflight["inputs"]["train_ids"],
        "rows_per_case": config.rows_per_training_case,
        "random_seed": RANDOM_SEED,
        "tier_labels": STRESS_TIER_LABELS,
        "tier_quotas": STRESS_TIER_QUOTAS,
        "tier_target_loss_mass": STRESS_TIER_TARGET_MASS,
        "target": "(actual_stress - frozen_v8_predicted_stress) / frozen_v8_predicted_scale",
        "baseline_id": preflight["baseline"]["manifest"]["baseline_id"],
        "complete_case_validation": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(paths["metadata"], metadata)
    del all_X, stage_a_X, base_target, weights
    gc.collect()
    return {"paths": paths, "metadata": metadata, "reused": False}


def prepare_stage_b_training_cache(
    preflight: dict,
    cache: dict,
    selected_stage_a: pd.Series,
) -> dict:
    cache_dir = preflight["output_dir"] / "training_cache"
    paths = dict(cache["paths"])
    paths.update({
        "stage_b_X": cache_dir / "stage_b_features.npy",
        "stage_b_target": cache_dir / "remaining_after_geometry_residual.npy",
        "stage_b_scaling": cache_dir / "stage_b_scaling.csv",
        "stage_b_metadata": cache_dir / "stage_b_cache_metadata.json",
    })
    stage_a_signature = hashlib.sha256(
        str(selected_stage_a["formula_scaled_sympy"]).encode("utf-8")
    ).hexdigest()
    if all(paths[name].exists() for name in [
        "stage_b_X", "stage_b_target", "stage_b_scaling", "stage_b_metadata"
    ]):
        metadata = json.loads(paths["stage_b_metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("stage_a_formula_sha256") == stage_a_signature
            and metadata.get("stage_b_features") == list(STAGE_B_FEATURES)
        ):
            return {"paths": paths, "metadata": metadata, "reused": True}

    all_X = np.load(paths["all_X"], mmap_mode="r")
    base_target = np.load(paths["base_target"], mmap_mode="r")
    weights = np.load(paths["weights"], mmap_mode="r")
    stage_a_scaling = _load_scaling(paths["stage_a_scaling"])
    stage_a_function = _compiled_formula(selected_stage_a, stage_a_scaling)
    stage_a_X = np.asarray(all_X[:, _feature_indices(STAGE_A_FEATURES)])
    stage_a_prediction = stage_a_function(stage_a_X)
    stage_b_target = np.asarray(base_target, dtype=np.float64) - stage_a_prediction
    stage_b_X = np.ascontiguousarray(
        all_X[:, _feature_indices(STAGE_B_FEATURES)], dtype=np.float32
    )
    scaling = _scaling_payload(
        stage_b_X, stage_b_target, STAGE_B_FEATURES, weights
    )
    np.save(paths["stage_b_X"], stage_b_X)
    np.save(paths["stage_b_target"], stage_b_target.astype(np.float32))
    _save_scaling(scaling, "remaining_after_geometry_residual", paths["stage_b_scaling"])
    metadata = {
        "n_rows": int(len(stage_b_X)),
        "stage_b_features": list(STAGE_B_FEATURES),
        "stage_a_candidate_index": int(selected_stage_a["candidate_index"]),
        "stage_a_formula_sha256": stage_a_signature,
        "target": "deployable_normalised_residual_minus_selected_geometry_residual",
        "negative_target_fraction": float((stage_b_target < 0).mean()),
        "target_rmse": float(np.sqrt(np.mean(stage_b_target ** 2))),
    }
    _atomic_json(paths["stage_b_metadata"], metadata)
    del all_X, base_target, stage_a_X, stage_a_prediction, stage_b_X, stage_b_target
    gc.collect()
    return {"paths": paths, "metadata": metadata, "reused": False}


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


def _file_activity_signature(paths: Sequence[Path]) -> tuple:
    signature = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


def _append_attempt_audit(path: Path, record: dict) -> None:
    row = pd.DataFrame([record])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True, sort=False)
    row.to_csv(path, index=False)


def _stage_directory(output_dir: Path, stage: str) -> Path:
    return output_dir / "stages" / f"stage_{stage}"


def _completed_segments(stage_dir: Path, n_segments: int) -> list[int]:
    return [
        segment
        for segment in range(1, n_segments + 1)
        if (stage_dir / "segments" / f"segment_{segment:02d}" / "complete.json").exists()
    ]


def _write_stage_progress(
    stage_dir: Path,
    stage: str,
    config: SignedStagedPilotConfig,
) -> dict:
    completed = _completed_segments(stage_dir, config.n_segments_per_stage)
    contiguous = 0
    for segment in range(1, config.n_segments_per_stage + 1):
        if segment in completed:
            contiguous = segment
        else:
            break
    payload = {
        "stage": stage,
        "completed_segments": completed,
        "contiguous_completed_segments": contiguous,
        "total_segments": config.n_segments_per_stage,
        "planned_population_iterations_completed": (
            contiguous * config.population_iterations_per_segment
        ),
        "target_population_iterations": config.population_iterations_per_stage,
        "planned_progress_fraction": contiguous / config.n_segments_per_stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(stage_dir / "search_progress.json", payload)
    return payload


def _stage_spec(stage: str, cache: dict, config: SignedStagedPilotConfig) -> dict:
    if stage == "geometry":
        return {
            "features": list(STAGE_A_FEATURES),
            "X": cache["paths"]["stage_a_X"],
            "y": cache["paths"]["base_target"],
            "scaling": cache["paths"]["stage_a_scaling"],
            "maxsize": config.stage_a_maxsize,
            "maxdepth": config.stage_a_maxdepth,
        }
    if stage == "physical":
        return {
            "features": list(STAGE_B_FEATURES),
            "X": cache["paths"]["stage_b_X"],
            "y": cache["paths"]["stage_b_target"],
            "scaling": cache["paths"]["stage_b_scaling"],
            "maxsize": config.stage_b_maxsize,
            "maxdepth": config.stage_b_maxdepth,
        }
    raise ValueError(f"Unknown V10 stage: {stage}")


def _search_signature(
    stage: str,
    spec: dict,
    cache: dict,
    config: SignedStagedPilotConfig,
) -> dict:
    return {
        "method": "v10_signed_staged_deployable_normalised_residual",
        "stage": stage,
        "features": spec["features"],
        "n_rows": cache["metadata"]["n_rows"],
        "train_ids": cache["metadata"]["train_ids"],
        "tier_quotas": STRESS_TIER_QUOTAS,
        "tier_target_loss_mass": STRESS_TIER_TARGET_MASS,
        "total_niterations": config.total_niterations_per_stage,
        "populations": config.populations,
        "segment_niterations": config.segment_niterations,
        "population_size": config.population_size,
        "ncycles_per_iteration": config.ncycles_per_iteration,
        "batch_size": config.batch_size,
        "maxsize": spec["maxsize"],
        "maxdepth": spec["maxdepth"],
        "operators": ["+", "-", "*", "/", "abs", "tanh", "gauss"],
        "elementwise_loss": "HuberLoss(1.0)",
        "precision": 64,
        "random_seed": RANDOM_SEED,
    }


def run_stage_segment(
    preflight: dict,
    cache: dict,
    config: SignedStagedPilotConfig,
    stage: str,
    segment: int,
) -> Path:
    spec = _stage_spec(stage, cache, config)
    stage_dir = _stage_directory(preflight["output_dir"], stage)
    segment_dir = stage_dir / "segments" / f"segment_{segment:02d}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    complete_path = segment_dir / "complete.json"
    canonical_frontier = segment_dir / "frontier.csv"
    if complete_path.exists() and canonical_frontier.exists():
        print(
            f"V10 {stage} segment {segment}/{config.n_segments_per_stage}: reused",
            flush=True,
        )
        return canonical_frontier

    run_id = f"v10_i{config.iteration}_{stage}_signed_residual_pilot"
    run_directory = stage_dir / "search_state" / "pysr_runs" / run_id
    checkpoint_path = run_directory / "checkpoint.pkl"
    hall_path = run_directory / "hall_of_fame.csv"
    snapshot_root = stage_dir / "search_state" / "stable_snapshots"
    attempt_audit_path = stage_dir / "segment_attempt_audit.csv"

    for attempt in range(1, config.max_attempts_per_segment + 1):
        attempt_dir = segment_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "worker_stdout.log"
        stderr_path = attempt_dir / "worker_stderr.log"
        worker_status_path = attempt_dir / "worker_status.json"
        resume = checkpoint_path.exists()
        command = [
            sys.executable,
            str(preflight["worker_path"]),
            "--stage", stage,
            "--features", str(spec["X"]),
            "--target", str(spec["y"]),
            "--weights", str(cache["paths"]["weights"]),
            "--scaling", str(spec["scaling"]),
            "--feature-names", json.dumps(spec["features"]),
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
            "--maxsize", str(spec["maxsize"]),
            "--maxdepth", str(spec["maxdepth"]),
            "--seed", str(RANDOM_SEED + (0 if stage == "geometry" else 1_000)),
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
        wall_timed_out = False
        print(
            f"V10 {stage} segment {segment}/{config.n_segments_per_stage}, "
            f"attempt {attempt}: {'resume' if resume else 'fresh search'}",
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
            activity_paths = [
                stdout_path,
                stderr_path,
                worker_status_path,
                hall_path,
                checkpoint_path,
            ]
            last_signature = _file_activity_signature(activity_paths)
            last_activity = time.time()
            last_report = 0.0
            while process.poll() is None:
                now = time.time()
                elapsed = now - started
                signature = _file_activity_signature(activity_paths)
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
                    wall_timed_out = True
                    _terminate_process_tree(process)
                    break
                if elapsed - last_report >= 300 or last_report == 0:
                    print(
                        f"V10 {stage} segment {segment}, attempt {attempt}: "
                        f"elapsed={elapsed / 60:.1f} min, "
                        f"inactive={inactive / 60:.1f} min, "
                        f"checkpoint={checkpoint_path.exists()}",
                        flush=True,
                    )
                    last_report = elapsed
                _atomic_json(segment_dir / "watchdog_status.json", {
                    "stage": stage,
                    "segment": segment,
                    "attempt": attempt,
                    "pid": process.pid,
                    "elapsed_seconds": elapsed,
                    "inactive_seconds": inactive,
                    "checkpoint_exists": checkpoint_path.exists(),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
                time.sleep(config.watchdog_poll_seconds)
            return_code = process.poll()

        worker_frontier = attempt_dir / "frontier.csv"
        audit = {
            "stage": stage,
            "segment": segment,
            "attempt": attempt,
            "resume_from_checkpoint": resume,
            "return_code": return_code,
            "termination_reason": reason,
            "stalled": stalled,
            "wall_timed_out": wall_timed_out,
            "elapsed_seconds": time.time() - started,
            "checkpoint_exists_after": checkpoint_path.exists(),
            "worker_frontier_exists": worker_frontier.exists(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _append_attempt_audit(attempt_audit_path, audit)
        if return_code == 0 and worker_frontier.exists() and checkpoint_path.exists():
            frontier = pd.read_csv(worker_frontier)
            frontier["completed_segment"] = segment
            frontier["completed_attempt"] = attempt
            frontier.to_csv(canonical_frontier, index=False)
            snapshot = snapshot_root / f"segment_{segment:02d}"
            snapshot.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, snapshot / "checkpoint.pkl")
            if hall_path.exists():
                shutil.copy2(hall_path, snapshot / "hall_of_fame.csv")
            _atomic_json(complete_path, {
                "status": "complete",
                "stage": stage,
                "segment": segment,
                "attempt": attempt,
                "planned_population_iterations": config.population_iterations_per_segment,
                "elapsed_seconds": time.time() - started,
            })
            _write_stage_progress(stage_dir, stage, config)
            print(
                f"V10 {stage} segment {segment}: complete; planned cumulative "
                f"{segment * config.population_iterations_per_segment}/"
                f"{config.population_iterations_per_stage}",
                flush=True,
            )
            return canonical_frontier

        if run_directory.exists():
            interrupted = attempt_dir / "interrupted_state"
            interrupted.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                shutil.copy2(checkpoint_path, interrupted / "checkpoint.pkl")
            if hall_path.exists():
                shutil.copy2(hall_path, interrupted / "hall_of_fame.csv")
        previous = snapshot_root / f"segment_{segment - 1:02d}"
        if segment > 1 and (previous / "checkpoint.pkl").exists():
            run_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous / "checkpoint.pkl", checkpoint_path)
            if (previous / "hall_of_fame.csv").exists():
                shutil.copy2(previous / "hall_of_fame.csv", hall_path)
            recovery = f"rolled_back_to_stable_segment_{segment - 1:02d}"
        else:
            if run_directory.exists():
                destination = attempt_dir / "failed_run_directory"
                if destination.exists():
                    raise RuntimeError(f"Recovery destination exists: {destination}")
                shutil.move(str(run_directory), str(destination))
            recovery = "no_stable_checkpoint; next_attempt_starts_fresh"
        audit_table = pd.read_csv(attempt_audit_path)
        audit_table.loc[
            (audit_table["stage"] == stage)
            & (audit_table["segment"] == segment)
            & (audit_table["attempt"] == attempt),
            "recovery_action",
        ] = recovery
        audit_table.to_csv(attempt_audit_path, index=False)
        print(
            f"V10 {stage} segment {segment}, attempt {attempt} did not finish "
            f"({reason}); recovery={recovery}",
            flush=True,
        )

    raise RuntimeError(
        f"V10 {stage} segment {segment} failed "
        f"{config.max_attempts_per_segment} times"
    )


def run_stage_search(
    preflight: dict,
    cache: dict,
    config: SignedStagedPilotConfig,
    stage: str,
) -> pd.DataFrame:
    spec = _stage_spec(stage, cache, config)
    stage_dir = _stage_directory(preflight["output_dir"], stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    signature_path = stage_dir / "search_signature.json"
    signature = _search_signature(stage, spec, cache, config)
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError(
                f"Existing V10 {stage} state has a different signature. "
                "Use a new output_subdir."
            )
    else:
        _atomic_json(signature_path, signature)
    latest = None
    for segment in range(1, config.n_segments_per_stage + 1):
        latest = run_stage_segment(preflight, cache, config, stage, segment)
    if latest is None:
        raise RuntimeError(f"No V10 {stage} search segment completed")
    frontier = pd.read_csv(latest)
    frontier = (
        frontier.sort_values(["loss", "complexity"])
        .drop_duplicates("formula_scaled_sympy", keep="first")
        .reset_index(drop=True)
    )
    frontier["source_candidate_index"] = frontier["candidate_index"]
    frontier["candidate_index"] = np.arange(len(frontier), dtype=int)
    frontier["frontier_source"] = f"v10_{stage}_pilot_only"
    frontier.to_csv(stage_dir / "frontier.csv", index=False)
    return frontier


def _evaluate_stage_candidates(
    candidates: pd.DataFrame,
    candidate_scaling: dict,
    preflight: dict,
    case_ids: Sequence[str],
    split: str,
    config: SignedStagedPilotConfig,
    stage: str,
    fixed_stage_a: tuple[Callable[[np.ndarray], np.ndarray], pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    stage_features = STAGE_A_FEATURES if stage == "geometry" else STAGE_B_FEATURES
    stage_indices = _feature_indices(stage_features)
    functions = {
        int(row.candidate_index): _compiled_formula(row, candidate_scaling)
        for row in candidates.itertuples(index=False)
    }
    baseline_functions = _baseline_functions(preflight["baseline"])
    stage_a_function = fixed_stage_a[0] if fixed_stage_a is not None else None
    stage_a_indices = _feature_indices(STAGE_A_FEATURES)
    candidate_rows = []
    frozen_rows = []
    stage_a_rows = []
    invalid: dict[int, str] = {}

    for position, case_id in enumerate(case_ids, start=1):
        print(
            f"[{position}/{len(case_ids)}] {split} complete-case V10 {stage}: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        matrix, _, predicted_scale, _, baseline_stress = _predict_baseline_components(
            frame, summary, baseline_functions
        )
        frozen_rows.append({
            "iteration": config.iteration,
            "split": split,
            "model": "frozen_v8_candidate5",
            "case_id": case_id,
            **evaluate_prediction_arrays(actual, baseline_stress),
        })
        fixed_a_values = (
            stage_a_function(matrix[:, stage_a_indices])
            if stage_a_function is not None
            else np.zeros(len(matrix), dtype=np.float64)
        )
        if stage_a_function is not None:
            stage_a_prediction = baseline_stress + predicted_scale * fixed_a_values
            stage_a_rows.append({
                "iteration": config.iteration,
                "split": split,
                "model": "v10_geometry_residual_only",
                "case_id": case_id,
                **evaluate_prediction_arrays(actual, stage_a_prediction),
            })
        for candidate_index, function in functions.items():
            if candidate_index in invalid:
                continue
            try:
                correction = function(matrix[:, stage_indices])
                total_correction = fixed_a_values + correction
                predicted = baseline_stress + predicted_scale * total_correction
                candidate_rows.append({
                    "iteration": config.iteration,
                    "split": split,
                    "model": (
                        "v10_geometry_residual_candidate"
                        if stage == "geometry"
                        else "v10_signed_staged_residual_candidate"
                    ),
                    "stage": stage,
                    "candidate_index": candidate_index,
                    "case_id": case_id,
                    **evaluate_prediction_arrays(actual, predicted),
                })
            except Exception as exc:
                invalid[candidate_index] = repr(exc)
        del frame, actual, matrix, baseline_stress, fixed_a_values
        gc.collect()

    candidate_cases = pd.DataFrame(candidate_rows)
    references = {
        "frozen_v8": pd.DataFrame(frozen_rows),
        "stage_a": pd.DataFrame(stage_a_rows),
    }
    records = []
    for candidate in candidates.itertuples(index=False):
        candidate_index = int(candidate.candidate_index)
        group = candidate_cases[candidate_cases["candidate_index"] == candidate_index]
        if candidate_index in invalid or group["case_id"].nunique() != len(case_ids):
            records.append({
                **candidate._asdict(),
                "candidate_valid": False,
                "invalid_reason": invalid.get(candidate_index, "incomplete_evaluation"),
            })
            continue
        aggregate = aggregate_case_metrics(group, ["candidate_index"]).iloc[0].to_dict()
        aggregate.pop("candidate_index", None)
        adaptation = _tail_adaptation(group)
        records.append({
            **candidate._asdict(),
            "candidate_valid": True,
            "invalid_reason": "",
            **{f"{split}_{key}": value for key, value in aggregate.items()},
            **{f"{split}_{key}": value for key, value in adaptation.items()},
        })
    return pd.DataFrame(records), candidate_cases, references


def _reference_aggregate(case_metrics: pd.DataFrame) -> pd.Series:
    if case_metrics.empty:
        raise ValueError("Reference case metrics are empty")
    return aggregate_case_metrics(
        case_metrics, ["iteration", "split", "model"]
    ).iloc[0]


def _score_and_select_stage_candidate(
    candidate_metrics: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    output_dir: Path,
    stage: str,
) -> pd.Series:
    valid = candidate_metrics[candidate_metrics["candidate_valid"].fillna(False)].copy()
    if valid.empty:
        raise RuntimeError(f"No finite V10 {stage} candidate on all validation cases")
    error_metrics = [
        ("validation_macro_rmse", 0.25),
        ("validation_mean_top5_actual_rmse", 0.10),
        ("validation_mean_p95_relative_error", 0.15),
        ("validation_mean_p99_relative_error", 0.15),
        ("validation_mean_p99_underprediction_fraction", 0.10),
    ]
    benefit_metrics = [
        ("validation_mean_top1pct_hotspot_overlap", 0.15),
        ("validation_mean_top1_recall_in_predicted_top5", 0.10),
    ]
    valid["engineering_selection_score"] = 0.0
    for column, weight in error_metrics:
        valid["engineering_selection_score"] += weight * valid[column].rank(
            pct=True, method="average", ascending=True
        )
    for column, weight in benefit_metrics:
        valid["engineering_selection_score"] += weight * valid[column].rank(
            pct=True, method="average", ascending=False
        )

    frozen = _reference_aggregate(references["frozen_v8"])
    frozen_rmse = float(frozen["macro_rmse"])
    valid["macro_rmse_improvement_vs_v8_fraction"] = (
        frozen_rmse - valid["validation_macro_rmse"]
    ) / max(frozen_rmse, 1e-12)
    valid["gate_numerical_guardrail"] = (
        valid["validation_max_prediction_abs_max_ratio"] <= 5.0
    )
    valid["gate_positive_macro_r2"] = valid["validation_macro_r2"] > 0.0
    valid["gate_improves_v8_rmse"] = (
        valid["macro_rmse_improvement_vs_v8_fraction"] > 0.0
    )
    valid["gate_improves_v8_rmse_by_3pct"] = (
        valid["macro_rmse_improvement_vs_v8_fraction"] >= 0.03
    )
    valid["gate_p95_relative_error"] = (
        valid["validation_mean_p95_relative_error"] <= 0.15
    )
    valid["gate_p99_relative_error"] = (
        valid["validation_mean_p99_relative_error"] <= 0.15
    )
    valid["gate_p95_underprediction"] = (
        valid["validation_mean_p95_underprediction_fraction"] <= 0.10
    )
    valid["gate_p99_underprediction"] = (
        valid["validation_mean_p99_underprediction_fraction"] <= 0.10
    )
    valid["gate_top1_hotspot_overlap"] = (
        valid["validation_mean_top1pct_hotspot_overlap"] >= 0.60
    )
    valid["gate_top1_recall"] = (
        valid["validation_mean_top1_recall_in_predicted_top5"] >= 0.75
    )
    if not references["stage_a"].empty:
        stage_a = _reference_aggregate(references["stage_a"])
        valid["macro_rmse_improvement_vs_stage_a_fraction"] = (
            float(stage_a["macro_rmse"]) - valid["validation_macro_rmse"]
        ) / max(float(stage_a["macro_rmse"]), 1e-12)
        valid["gate_not_worse_than_stage_a_rmse"] = (
            valid["macro_rmse_improvement_vs_stage_a_fraction"] >= 0.0
        )
    else:
        valid["macro_rmse_improvement_vs_stage_a_fraction"] = np.nan
        valid["gate_not_worse_than_stage_a_rmse"] = True

    final_gate_columns = [
        "gate_numerical_guardrail",
        "gate_positive_macro_r2",
        "gate_improves_v8_rmse_by_3pct",
        "gate_not_worse_than_stage_a_rmse",
        "gate_p95_relative_error",
        "gate_p99_relative_error",
        "gate_p95_underprediction",
        "gate_p99_underprediction",
        "gate_top1_hotspot_overlap",
        "gate_top1_recall",
    ]
    valid["all_final_promotion_gates_pass"] = valid[final_gate_columns].all(axis=1)

    if stage == "physical" and valid["all_final_promotion_gates_pass"].any():
        pool = valid[valid["all_final_promotion_gates_pass"]].copy()
        pool_status = "all_final_promotion_gates"
    else:
        improved = valid[
            valid["gate_numerical_guardrail"]
            & valid["gate_positive_macro_r2"]
            & valid["gate_improves_v8_rmse"]
        ].copy()
        if not improved.empty:
            pool = improved
            pool_status = "finite_positive_r2_and_improves_frozen_v8"
        else:
            numerical = valid[valid["gate_numerical_guardrail"]].copy()
            pool = numerical if not numerical.empty else valid
            pool_status = "diagnostic_finite_candidates_only"

    selected = pool.sort_values([
        "engineering_selection_score",
        "validation_macro_rmse",
        "validation_mean_p99_relative_error",
        "complexity",
        "candidate_index",
    ]).iloc[0].copy()
    selected["selection_pool_status"] = pool_status
    selected["selection_method"] = (
        "validation-only engineering rank over RMSE, P95/P99, underprediction "
        "and hotspot metrics; complexity is a late tiebreaker"
    )
    if stage == "geometry":
        selected["pilot_promotion_status"] = (
            "geometry_stage_improves_frozen_v8"
            if bool(selected["gate_improves_v8_rmse"])
            else "geometry_stage_diagnostic_only"
        )
    else:
        selected["pilot_promotion_status"] = (
            "passes_all_v10_pilot_gates"
            if bool(selected["all_final_promotion_gates_pass"])
            else "diagnostic_only_does_not_pass_all_v10_pilot_gates"
        )

    for column in valid.columns:
        if column not in candidate_metrics.columns:
            candidate_metrics[column] = np.nan
    candidate_metrics.loc[valid.index, valid.columns] = valid
    candidate_metrics["selected_candidate"] = candidate_metrics["candidate_index"].eq(
        selected["candidate_index"]
    )
    candidate_metrics.to_csv(output_dir / "candidate_validation_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "selected_formula.csv", index=False)
    return selected


def _save_formula_artifacts(
    preflight: dict,
    selected_stage_a: pd.Series,
    selected_stage_b: pd.Series,
    config: SignedStagedPilotConfig,
) -> dict:
    baseline = preflight["baseline"]
    mean_formula = str(baseline["mean_row"]["formula_original_variables"])
    log_scale_formula = str(baseline["scale_row"]["formula_original_variables"])
    fixed_shape_formula = str(baseline["shape_row"]["formula_original_variables"])
    geometry_formula = str(selected_stage_a["formula_original_variables"])
    physical_formula = str(selected_stage_b["formula_original_variables"])
    corrected_shape = (
        f"({fixed_shape_formula}) + ({geometry_formula}) + ({physical_formula})"
    )
    composite = f"({mean_formula}) + exp({log_scale_formula}) * ({corrected_shape})"
    record = {
        "iteration": config.iteration,
        "method": "frozen_v8_plus_signed_geometry_then_physical_residuals",
        "baseline_id": baseline["manifest"]["baseline_id"],
        "case_mean_formula": mean_formula,
        "case_log_scale_formula": log_scale_formula,
        "fixed_global_shape_formula": fixed_shape_formula,
        "geometry_residual_formula": geometry_formula,
        "physical_residual_formula": physical_formula,
        "corrected_shape_formula": corrected_shape,
        "composite_stress_formula": composite,
        "geometry_complexity": int(selected_stage_a["complexity"]),
        "physical_complexity": int(selected_stage_b["complexity"]),
        "pilot_promotion_status": selected_stage_b["pilot_promotion_status"],
    }
    output_dir = preflight["output_dir"]
    pd.DataFrame([record]).to_csv(
        output_dir / "selected_composite_formula.csv", index=False
    )
    text = (
        "CT3 V10 signed staged symbolic stress formula\n"
        "=============================================\n\n"
        f"Frozen baseline: {record['baseline_id']}\n\n"
        f"1. Case mean\nmu = {mean_formula}\n\n"
        f"2. Positive case scale\nscale = exp({log_scale_formula})\n\n"
        f"3. Frozen V8 global shape\nfixed_shape = {fixed_shape_formula}\n\n"
        f"4. Signed geometry/boundary residual\ngeometry_residual = {geometry_formula}\n\n"
        f"5. Signed physical/local residual\nphysical_residual = {physical_formula}\n\n"
        f"6. Combined deployable expression\nsigma = {composite}\n\n"
        "Proxy definitions\n"
        "-----------------\n"
        "rho_fraction_proxy = (rho - rho_min) / (rho_max - rho_min)\n"
        "z_fraction_proxy = (z - z_min) / (z_max - z_min)\n"
        "theta_fraction_proxy = (theta - theta_min) / (theta_max - theta_min)\n"
        "nearest_*_boundary_fraction_proxy = min(fraction, 1 - fraction)\n"
        "signed_*_position_proxy = 2 * fraction - 1\n"
        "pairwise signed interaction proxies are products of signed positions\n"
        "boundary_corner_proximity_proxy = nearest_rho * nearest_z * nearest_theta\n\n"
        "All case summaries are calculated only from predictor fields. The 50 "
        "final-test cases were not read. Proxies are case-relative and are not "
        "confirmed distances to named FEM surfaces.\n"
    )
    (output_dir / "selected_composite_formula.txt").write_text(text, encoding="utf-8")
    return record


def _save_plots(split_metrics: pd.DataFrame, case_metrics: pd.DataFrame, output_dir: Path) -> None:
    validation = split_metrics[split_metrics["split"] == "validation"].copy()
    metrics = [
        ("macro_rmse", "Macro RMSE", False),
        ("mean_p99_relative_error", "Mean P99 relative error", False),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (column, title, higher_better) in zip(axes, metrics):
        values = validation.sort_values(column, ascending=higher_better)
        axis.barh(values["model"], values[column], color="#267a78")
        axis.set_title(title)
        axis.set_xlabel(column)
    fig.tight_layout()
    fig.savefig(output_dir / "v10_validation_model_comparison.png", dpi=180)
    plt.close(fig)

    selected_validation = case_metrics[
        (case_metrics["split"] == "validation")
        & (case_metrics["model"] == "v10_signed_staged_symbolic")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, tail in zip(axes, ["p95", "p99"]):
        actual = selected_validation[f"actual_{tail}"]
        predicted = selected_validation[f"predicted_{tail}"]
        axis.scatter(actual, predicted, color="#d97732", s=42)
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.plot([lower, upper], [lower, upper], "--", color="#555555")
        axis.set_xlabel(f"Actual {tail.upper()} stress")
        axis.set_ylabel(f"Predicted {tail.upper()} stress")
        axis.set_title(f"V10 {tail.upper()} adaptation")
    fig.tight_layout()
    fig.savefig(output_dir / "v10_selected_tail_adaptation.png", dpi=180)
    plt.close(fig)


def run_signed_staged_residual_pilot(
    package_root: Path,
    config: SignedStagedPilotConfig,
) -> dict:
    started = time.time()
    preflight = preflight_signed_staged_pilot(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "pilot_complete.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text(encoding="utf-8"))

    save_json(output_dir / "run_configuration.json", asdict(config))
    feature_rows = []
    for feature in ALL_V10_FEATURES:
        feature_rows.append({
            "feature": feature,
            "stage_a_geometry": feature in STAGE_A_FEATURES,
            "stage_b_physical": feature in STAGE_B_FEATURES,
            "derived_proxy": feature in BOUNDARY_DIRECTION_FEATURES,
        })
    pd.DataFrame(feature_rows).to_csv(output_dir / "feature_registry.csv", index=False)
    _atomic_json(output_dir / "design_rationale.json", {
        "frozen_baseline": "local V8 Candidate 5, verified by SHA-256",
        "target_alignment": "residual is defined against deployable predicted mean and scale",
        "stages": ["geometry_and_directional_boundary", "physical_and_local"],
        "sampling_tier_quotas": STRESS_TIER_QUOTAS,
        "sampling_target_loss_mass": STRESS_TIER_TARGET_MASS,
        "loss": "HuberLoss(1.0)",
        "batching": True,
        "full_complete_case_validation": True,
        "excluded_operator": "square, because the V9 residual required both signs",
        "final_test_cases_read": 0,
    })
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "prepare_stage_a_cache",
        "final_test_cases_read": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    sleep_inhibitor = _start_sleep_inhibitor(output_dir)
    try:
        cache = prepare_stage_a_training_cache(preflight, config)
        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "geometry_symbolic_search",
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        frontier_a = run_stage_search(preflight, cache, config, "geometry")
        shortlist_a = _shortlist_shape_frontier(
            frontier_a, config.max_candidates_for_full_validation
        ).copy()
        geometry_dir = _stage_directory(output_dir, "geometry")
        shortlist_a.to_csv(geometry_dir / "full_validation_shortlist.csv", index=False)
        scaling_a = _load_scaling(cache["paths"]["stage_a_scaling"])
        metrics_a, cases_a, refs_a = _evaluate_stage_candidates(
            shortlist_a,
            scaling_a,
            preflight,
            preflight["inputs"]["validation_ids"],
            "validation",
            config,
            "geometry",
        )
        selected_a = _score_and_select_stage_candidate(
            metrics_a, refs_a, geometry_dir, "geometry"
        )
        selected_a_function = _compiled_formula(selected_a, scaling_a)

        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "prepare_physical_residual_and_search",
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        cache_b = prepare_stage_b_training_cache(preflight, cache, selected_a)
        cache_b["metadata"] = cache["metadata"]
        frontier_b = run_stage_search(preflight, cache_b, config, "physical")
        shortlist_b = _shortlist_shape_frontier(
            frontier_b, config.max_candidates_for_full_validation
        ).copy()
        physical_dir = _stage_directory(output_dir, "physical")
        shortlist_b.to_csv(physical_dir / "full_validation_shortlist.csv", index=False)
        scaling_b = _load_scaling(cache_b["paths"]["stage_b_scaling"])
        metrics_b, cases_b, refs_b = _evaluate_stage_candidates(
            shortlist_b,
            scaling_b,
            preflight,
            preflight["inputs"]["validation_ids"],
            "validation",
            config,
            "physical",
            fixed_stage_a=(selected_a_function, selected_a),
        )
        selected_b = _score_and_select_stage_candidate(
            metrics_b, refs_b, physical_dir, "physical"
        )
        selected_b_index = int(selected_b["candidate_index"])
        selected_validation = cases_b[
            cases_b["candidate_index"] == selected_b_index
        ].copy()
        selected_validation["model"] = "v10_signed_staged_symbolic"

        selected_b_table = pd.DataFrame([selected_b])
        _, selected_internal, refs_internal = _evaluate_stage_candidates(
            selected_b_table,
            scaling_b,
            preflight,
            preflight["inputs"]["internal_ids"],
            "internal_test",
            config,
            "physical",
            fixed_stage_a=(selected_a_function, selected_a),
        )
        selected_internal["model"] = "v10_signed_staged_symbolic"
        selected_cases = pd.concat(
            [selected_validation, selected_internal], ignore_index=True
        )
        frozen_cases = pd.concat(
            [refs_b["frozen_v8"], refs_internal["frozen_v8"]], ignore_index=True
        )
        stage_a_cases = pd.concat(
            [refs_b["stage_a"], refs_internal["stage_a"]], ignore_index=True
        )
        all_case_metrics = pd.concat(
            [frozen_cases, stage_a_cases, selected_cases], ignore_index=True, sort=False
        )
        all_case_metrics.to_csv(
            output_dir / "selected_models_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        split_metrics = aggregate_case_metrics(
            all_case_metrics, ["iteration", "split", "model"]
        )
        split_metrics.to_csv(output_dir / "selected_models_split_metrics.csv", index=False)
        tail_rows = [
            {"split": split, "model": model, **_tail_adaptation(group)}
            for (split, model), group in all_case_metrics.groupby(["split", "model"])
        ]
        pd.DataFrame(tail_rows).to_csv(
            output_dir / "selected_models_tail_adaptation.csv", index=False
        )
        formula_record = _save_formula_artifacts(
            preflight, selected_a, selected_b, config
        )
        _save_plots(split_metrics, all_case_metrics, output_dir)
        progress_a = _write_stage_progress(geometry_dir, "geometry", config)
        progress_b = _write_stage_progress(physical_dir, "physical", config)
        payload = {
            "status": "complete",
            "iteration": config.iteration,
            "prototype_only": True,
            "method": "frozen_v8_plus_signed_geometry_then_physical_residuals",
            "baseline_id": preflight["baseline"]["manifest"]["baseline_id"],
            "train_cases": len(preflight["inputs"]["train_ids"]),
            "validation_cases": len(preflight["inputs"]["validation_ids"]),
            "internal_test_cases": len(preflight["inputs"]["internal_ids"]),
            "final_test_cases_read": 0,
            "training_rows": cache["metadata"]["n_rows"],
            "planned_population_iterations": config.total_population_iterations,
            "geometry_completed_segments": progress_a["contiguous_completed_segments"],
            "physical_completed_segments": progress_b["contiguous_completed_segments"],
            "selected_geometry_candidate": int(selected_a["candidate_index"]),
            "selected_physical_candidate": selected_b_index,
            "pilot_promotion_status": formula_record["pilot_promotion_status"],
            "elapsed_seconds": time.time() - started,
            "output_directory": str(output_dir),
        }
        _atomic_json(completion_path, payload)
        _atomic_json(output_dir / "run_status.json", payload)
        return payload
    except Exception as exc:
        _atomic_json(output_dir / "run_status.json", {
            "status": "failed_or_interrupted",
            "stage": "exception",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
            "recovery_instruction": (
                "Rerun the same notebook. Caches and completed PySR segments "
                "from both stages are reused."
            ),
            "final_test_cases_read": 0,
        })
        raise
    finally:
        _stop_sleep_inhibitor(sleep_inhibitor)
