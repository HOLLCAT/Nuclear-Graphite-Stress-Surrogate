"""V13 predictor-only structural residual symbolic-regression pilot.

V12 showed that a case-adaptive scalar gain almost reached the non-deployable
per-case oracle gain while leaving the P99 underprediction target unmet.  The
remaining error is therefore treated as a spatial/physical structure problem,
not another gain-calibration problem.

V13 freezes the validated V11 global-gain-0.90 predictor and searches for one
additional zero-case-mean symbolic residual:

    sigma_v13 = sigma_v11 + scale_v10 * (r_raw - case_mean(r_raw))

The residual formula may use only predictor-derived quantities.  Actual stress
is used to construct training targets, deterministic stress-tier discovery
samples and training weights, but never appears in a deployed feature.  Every
shortlisted candidate is selected on complete validation cases and evaluated
on complete internal/development cases.  The 50 final-test cases stay sealed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Sequence

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
)
from hierarchical_feasibility import _case_summary_row  # noqa: E402
from hierarchical_symbolic_regression import (  # noqa: E402
    _compiled_formula,
    _load_scaling,
    _save_scaling,
    _scaling_payload,
    _shortlist_shape_frontier,
    _tail_adaptation,
)
from signed_staged_residual_symbolic import (  # noqa: E402
    _atomic_json,
)
from v11_gain_stability_audit import (  # noqa: E402
    V11GainAuditConfig,
    _v11_required_paths,
    preflight_v11_gain_audit,
)
from v11_tail_aware_localised_symbolic import (  # noqa: E402
    V11_FEATURES,
    _activity_signature,
    _append_csv,
    _full_case_components,
    _matrix_column,
    _mean_correction,
    _start_sleep_inhibitor,
    _stop_sleep_inhibitor,
    _terminate_process_tree,
    build_v11_feature_matrix,
)


V13_OUTPUT_NAME = "15_v13_structural_tail_residual"
V12_OUTPUT_NAME = "13_v12_case_adaptive_tail_gain"
V12_OUTPUT_SUBDIR = "iteration_1"
V11_GLOBAL_GAIN = 0.90

STRESS_TIER_NAMES = ("below_p90", "p90_to_p95", "p95_to_p99", "top_p99")
V13_TIER_QUOTAS = (2_000, 800, 1_200, 1_000)
V13_TIER_LOSS_MASS = (0.35, 0.15, 0.25, 0.25)
V13_POSITIVE_TAIL_MULTIPLIER = 2.0

V13_LOCAL_PHYSICAL = [
    "fluence_rate_within_case_z",
    "temperature_within_case_z",
    "weight_loss_rate_within_case_z",
]
V13_CONTEXT = [
    "fluence_rate_mean",
    "fluence_rate_p95",
    "temperature_mean",
    "temperature_p95",
    "weight_loss_rate_mean",
    "weight_loss_rate_std",
]
V13_GEOMETRY = [
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
V13_HARMONICS = [
    "theta_sin_2_proxy",
    "theta_cos_2_proxy",
    "theta_sin_4_proxy",
    "theta_cos_4_proxy",
]
V13_PARENT_PROXIES = [
    "v10_normalised_shape_proxy",
    "v11_tail_centered_proxy",
    "v11_prediction_within_case_z_proxy",
    "v11_predicted_upper_gate_proxy",
]
V13_BOUNDARY_INTERACTIONS = [
    f"{physical}_by_{boundary}"
    for physical in ("fluence_z", "temperature_z", "weight_loss_z")
    for boundary in (
        "nearest_radial_boundary",
        "nearest_axial_boundary",
        "nearest_angular_boundary",
    )
]
V13_GATE_INTERACTIONS = [
    "fluence_z_by_v11_upper_gate",
    "temperature_z_by_v11_upper_gate",
    "weight_loss_z_by_v11_upper_gate",
]
V13_FEATURES = (
    V13_LOCAL_PHYSICAL
    + V13_CONTEXT
    + V13_GEOMETRY
    + V13_HARMONICS
    + V13_PARENT_PROXIES
    + V13_BOUNDARY_INTERACTIONS
    + V13_GATE_INTERACTIONS
)


@dataclass(frozen=True)
class V13Config:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    rows_per_training_case: int = 5_000
    total_niterations: int = 1_000
    populations: int = 8
    segment_niterations: int = 100
    population_size: int = 40
    ncycles_per_iteration: int = 100
    batch_size: int = 50_000
    maxsize: int = 36
    maxdepth: int = 11
    julia_threads: int = 8
    no_activity_timeout_seconds: int = 45 * 60
    segment_wall_timeout_seconds: int = 3 * 60 * 60
    watchdog_poll_seconds: int = 60
    max_attempts_per_segment: int = 3
    max_candidates_for_full_validation: int = 24
    force_rebuild_training_cache: bool = False
    minimum_validation_rmse_improvement_fraction: float = 0.005
    maximum_p95_degradation_absolute: float = 0.010
    maximum_all149_p95_degradation_absolute: float = 0.005
    maximum_p99_relative_error: float = 0.15
    maximum_p99_underprediction: float = 0.10
    minimum_top1_hotspot_overlap: float = 0.60
    minimum_top1_recall_in_predicted_top5: float = 0.80

    def validate(self) -> None:
        if self.iteration != 1:
            raise ValueError("V13 currently implements frozen Iteration 1 only")
        if self.rows_per_training_case != sum(V13_TIER_QUOTAS):
            raise ValueError("V13 tier quotas must equal rows_per_training_case")
        if self.total_niterations % self.segment_niterations != 0:
            raise ValueError("Total iterations must be divisible by segment iterations")
        if self.populations != 8:
            raise ValueError("V13 is locked to eight PySR populations")
        if not math.isclose(sum(V13_TIER_LOSS_MASS), 1.0, abs_tol=1e-12):
            raise ValueError("V13 tier loss mass must sum to one")
        if self.segment_wall_timeout_seconds <= self.no_activity_timeout_seconds:
            raise ValueError("Segment wall timeout must exceed inactivity timeout")
        if len(V13_FEATURES) != len(set(V13_FEATURES)):
            raise ValueError("V13 feature registry contains duplicates")
        if self.maximum_p99_underprediction <= 0:
            raise ValueError("P99 underprediction gate must be positive")

    @property
    def n_segments(self) -> int:
        return self.total_niterations // self.segment_niterations

    @property
    def population_iterations_per_segment(self) -> int:
        return self.segment_niterations * self.populations

    @property
    def population_iterations_total(self) -> int:
        return self.total_niterations * self.populations


def output_directory(package_root: Path, config: V13Config) -> Path:
    return Path(package_root) / "outputs" / V13_OUTPUT_NAME / config.output_subdir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v12_required_paths(package_root: Path) -> dict[str, Path]:
    root = Path(package_root) / "outputs" / V12_OUTPUT_NAME / V12_OUTPUT_SUBDIR
    return {
        "v12_completion": root / "v12_complete.json",
        "v12_run_status": root / "run_status.json",
        "v12_decision": root / "v12_promotion_decision.json",
        "v12_scope_metrics": root / "v12_comparison_scope_metrics.csv",
        "v12_selected_formula": root / "selected_deployment_formula.csv",
    }


def preflight_v13(package_root: Path, config: V13Config) -> dict:
    """Freeze V11/V12 provenance and verify that final cases remain sealed."""

    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    v11 = preflight_v11_gain_audit(package_root, V11GainAuditConfig())
    inputs = v11["inputs"]
    development_ids = inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"]
    if len(development_ids) != 149 or len(set(development_ids)) != 149:
        raise ValueError("V13 requires exactly 149 unique development cases")
    assert_no_final_cases(development_ids, inputs["manifest"])

    v12_paths = _v12_required_paths(package_root)
    v11_paths = _v11_required_paths(package_root)
    required = {
        **v12_paths,
        "v11_gain_formula": package_root
        / "outputs"
        / "12_v11_gain_stability_audit"
        / "iteration_1"
        / "selected_gain_formula.csv",
        "manifest": package_root / "shared" / "frozen_case_split_manifest_199cases.csv",
        "similarity_map": package_root / "shared" / "frozen_similarity_group_map_199cases.csv",
        "development_summary": package_root
        / "outputs"
        / "00_qc_sensitivity_ablation"
        / "development_case_summary.csv",
        "v13_source": package_root / "src" / "v13_structural_tail_residual.py",
        "v13_worker": package_root / "scripts" / "run_v13_structural_residual_segment.py",
        **{f"frozen_{name}": path for name, path in v11_paths.items()},
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("V13 required inputs are missing:\n" + "\n".join(missing))

    v12_completion = json.loads(v12_paths["v12_completion"].read_text(encoding="utf-8"))
    v12_decision = json.loads(v12_paths["v12_decision"].read_text(encoding="utf-8"))
    if v12_completion.get("status") != "complete":
        raise RuntimeError("The V12 run is not complete")
    if int(v12_completion.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError("V12 did not preserve the final-test seal")
    if v12_decision.get("promotion_status") != "retain_v11_global_gain_0_90":
        raise RuntimeError("V13 expects the reviewed V12 decision to retain V11 gain 0.90")

    gain_formula = pd.read_csv(required["v11_gain_formula"])
    if len(gain_formula) != 1:
        raise ValueError("Expected one frozen V11 gain formula")
    gain_formula = gain_formula.iloc[0]
    if not math.isclose(float(gain_formula["selected_tail_gain"]), V11_GLOBAL_GAIN, abs_tol=1e-12):
        raise ValueError("Frozen V11 tail gain is not 0.90")

    manifest = inputs["manifest"]
    iteration_manifest = manifest[manifest["iteration"].eq(config.iteration)].copy()
    development_manifest = iteration_manifest[~iteration_manifest["split"].eq("final_test")].copy()
    if set(development_manifest["case_id"]) != set(development_ids):
        raise ValueError("V13 development IDs differ from the frozen manifest")
    if set(iteration_manifest.loc[iteration_manifest["split"].eq("final_test"), "case_id"]) != set(inputs["final_ids"]):
        raise ValueError("V13 final-test IDs differ from the frozen manifest")

    summary = pd.read_csv(required["development_summary"])
    if len(summary) != 149 or set(summary["case_id"]) != set(development_ids):
        raise ValueError("Development predictor summary is incomplete")

    hash_rows = []
    for artifact, path in required.items():
        hash_rows.append({
            "artifact": artifact,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    hash_audit = pd.DataFrame(hash_rows)
    hash_audit.to_csv(output_dir / "frozen_input_hash_audit.csv", index=False)

    checks = pd.DataFrame([
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "development_cases", "value": len(development_ids), "expected": 149},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "v12_complete", "value": v12_completion.get("status"), "expected": "complete"},
        {"check": "v12_final_cases_read", "value": v12_completion.get("final_test_cases_read"), "expected": 0},
        {"check": "v12_retains_v11", "value": v12_decision.get("promotion_status"), "expected": "retain_v11_global_gain_0_90"},
        {"check": "frozen_v11_gain", "value": float(gain_formula["selected_tail_gain"]), "expected": V11_GLOBAL_GAIN},
        {"check": "v13_features", "value": len(V13_FEATURES), "expected": 42},
        {"check": "recoverable_segments", "value": config.n_segments, "expected": 10},
        {"check": "population_iterations", "value": config.population_iterations_total, "expected": 8_000},
    ])
    checks["pass"] = checks["value"] == checks["expected"]
    checks.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not checks["pass"].all():
        failed = checks.loc[~checks["pass"], "check"].tolist()
        raise RuntimeError(f"V13 preflight failed: {failed}")

    signature = {
        "config": asdict(config),
        "features": list(V13_FEATURES),
        "tier_names": list(STRESS_TIER_NAMES),
        "tier_quotas": list(V13_TIER_QUOTAS),
        "tier_loss_mass": list(V13_TIER_LOSS_MASS),
        "positive_tail_multiplier": V13_POSITIVE_TAIL_MULTIPLIER,
        "frozen_input_sha256": dict(zip(hash_audit["artifact"], hash_audit["sha256"])),
        "train_ids": inputs["train_ids"],
        "validation_ids": inputs["validation_ids"],
        "internal_ids": inputs["internal_ids"],
        "final_test_ids_inventoried_not_read": inputs["final_ids"],
    }
    signature["v13_signature_sha256"] = _canonical_hash(signature)
    signature_path = output_dir / "v13_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError("Existing V13 output has a different frozen signature")
    else:
        _atomic_json(signature_path, signature)

    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "development_ids": development_ids,
        "development_manifest": development_manifest,
        "summary": summary,
        "v11": v11,
        "v11_gain_formula": gain_formula,
        "v12_completion": v12_completion,
        "v12_decision": v12_decision,
        "required": required,
        "hash_audit": hash_audit,
        "checks": checks,
        "signature": signature,
        "worker_path": required["v13_worker"],
    }


def _v11_feature_column(matrix: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(matrix[:, V11_FEATURES.index(name)], dtype=np.float64)


def _repeat_summary(summary: pd.Series, name: str, n_rows: int) -> np.ndarray:
    value = float(summary[name])
    return np.full(n_rows, value, dtype=np.float64)


def build_v13_feature_matrix(
    matrix: np.ndarray,
    v11_matrix: np.ndarray,
    v11_prediction: np.ndarray,
    v11_tail_centered: np.ndarray,
    summary: pd.Series,
) -> np.ndarray:
    """Construct the explicit predictor-only V13 residual feature matrix."""

    n_rows = len(matrix)
    local = [_matrix_column(matrix, name) for name in V13_LOCAL_PHYSICAL]
    context = [_repeat_summary(summary, name, n_rows) for name in V13_CONTEXT]
    geometry = [_matrix_column(matrix, name) for name in V13_GEOMETRY]
    harmonics = [_v11_feature_column(v11_matrix, name) for name in V13_HARMONICS]

    prediction_std = float(np.std(v11_prediction, ddof=0))
    if not np.isfinite(prediction_std) or prediction_std <= 1e-12:
        prediction_z = np.zeros(n_rows, dtype=np.float64)
    else:
        prediction_z = (v11_prediction - float(np.mean(v11_prediction))) / prediction_std
    upper_gate = 0.5 * (np.tanh((prediction_z - 1.2815515655446004) / 0.50) + 1.0)
    parent = [
        _v11_feature_column(v11_matrix, "v10_normalised_shape_proxy"),
        np.asarray(v11_tail_centered, dtype=np.float64),
        prediction_z,
        upper_gate,
    ]

    boundaries = [
        _matrix_column(matrix, "nearest_radial_boundary_fraction_proxy"),
        _matrix_column(matrix, "nearest_axial_boundary_fraction_proxy"),
        _matrix_column(matrix, "nearest_angular_boundary_fraction_proxy"),
    ]
    boundary_interactions = [physical * boundary for physical in local for boundary in boundaries]
    gate_interactions = [physical * upper_gate for physical in local]
    values = local + context + geometry + harmonics + parent + boundary_interactions + gate_interactions
    result = np.column_stack(values)
    if result.shape != (n_rows, len(V13_FEATURES)):
        raise AssertionError(
            f"V13 feature shape {result.shape} != {(n_rows, len(V13_FEATURES))}"
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("V13 predictor feature matrix contains non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def _complete_v11_components(preflight: dict, case_id: str) -> dict:
    frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
    summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    matrix, scale, v10_prediction, components = _full_case_components(
        frame, summary, preflight["v11"]["v10"]
    )
    delta_mean = _mean_correction(summary, preflight["v11"]["mean_model"])
    v11_matrix = build_v11_feature_matrix(
        matrix,
        preflight["v11"]["v10"],
        preflight["v11"]["centres"],
        components,
    )
    tail_raw = preflight["v11"]["tail_function"](v11_matrix)
    tail_centered = tail_raw - float(np.mean(tail_raw))
    v11_prediction = (
        v10_prediction
        + delta_mean
        + V11_GLOBAL_GAIN * scale * tail_centered
    )
    v13_matrix = build_v13_feature_matrix(
        matrix,
        v11_matrix,
        v11_prediction,
        tail_centered,
        summary,
    )
    return {
        "frame": frame,
        "summary": summary,
        "actual": actual,
        "scale": float(scale),
        "v11_prediction": np.asarray(v11_prediction, dtype=np.float64),
        "v13_matrix": v13_matrix,
    }


def _exact_stress_tiers(actual: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    order = np.argsort(actual, kind="mergesort")
    n_rows = len(actual)
    boundaries = [0, int(0.90 * n_rows), int(0.95 * n_rows), int(0.99 * n_rows), n_rows]
    tier_index = np.empty(n_rows, dtype=np.int8)
    pools = []
    for index in range(4):
        pool = order[boundaries[index] : boundaries[index + 1]]
        tier_index[pool] = index
        pools.append(pool)
    return tier_index, pools


def _case_cache_path(output_dir: Path, case_id: str) -> Path:
    path = output_dir / "training_cache" / "case_samples" / f"{case_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _case_cache_valid(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["signature"].item()) == signature
                and data["X"].shape == (sum(V13_TIER_QUOTAS), len(V13_FEATURES))
                and data["y"].shape == (sum(V13_TIER_QUOTAS),)
                and data["tier_index"].shape == (sum(V13_TIER_QUOTAS),)
            )
    except Exception:
        return False


def _build_case_training_sample(preflight: dict, case_id: str) -> Path:
    output_dir = preflight["output_dir"]
    signature = preflight["signature"]["v13_signature_sha256"]
    path = _case_cache_path(output_dir, case_id)
    if _case_cache_valid(path, signature):
        return path

    values = _complete_v11_components(preflight, case_id)
    actual = values["actual"]
    scale = values["scale"]
    residual_normalised = (actual - values["v11_prediction"]) / scale
    residual_centered = residual_normalised - float(np.mean(residual_normalised))
    _, pools = _exact_stress_tiers(actual)
    case_number = int(case_id.split("_")[-1])
    rng = np.random.default_rng(RANDOM_SEED + 13_000 + case_number)
    selected_parts = []
    selected_tiers = []
    for tier_index, (pool, quota) in enumerate(zip(pools, V13_TIER_QUOTAS)):
        if len(pool) < quota:
            raise ValueError(f"{case_id} tier {STRESS_TIER_NAMES[tier_index]} has fewer than {quota} rows")
        chosen = rng.choice(pool, size=quota, replace=False)
        selected_parts.append(chosen)
        selected_tiers.append(np.full(quota, tier_index, dtype=np.int8))
    selected = np.concatenate(selected_parts)
    tiers = np.concatenate(selected_tiers)
    element_ids = values["frame"]["element_id"].to_numpy(dtype=np.int64)[selected]
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        signature=np.asarray(signature),
        X=values["v13_matrix"][selected].astype(np.float32),
        y=residual_centered[selected].astype(np.float32),
        tier_index=tiers,
        element_id=element_ids,
        scale=np.asarray(scale, dtype=np.float64),
        complete_case_residual_mean_normalised=np.asarray(
            float(np.mean(residual_normalised)), dtype=np.float64
        ),
    )
    os.replace(temporary, path)
    del values, actual, residual_normalised, residual_centered, selected
    gc.collect()
    return path


def prepare_v13_training_cache(preflight: dict, config: V13Config) -> dict:
    output_dir = preflight["output_dir"]
    cache_dir = output_dir / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "X": cache_dir / "v13_structural_features.npy",
        "y": cache_dir / "v13_centered_residual_target.npy",
        "weights": cache_dir / "v13_training_weights.npy",
        "scaling": cache_dir / "v13_scaling.csv",
        "manifest": cache_dir / "v13_training_sample_manifest.csv.gz",
        "weight_audit": cache_dir / "v13_weight_audit.csv",
        "metadata": cache_dir / "cache_metadata.json",
    }
    expected = {
        "signature": preflight["signature"]["v13_signature_sha256"],
        "n_rows": len(preflight["inputs"]["train_ids"]) * config.rows_per_training_case,
        "features": list(V13_FEATURES),
        "train_ids": preflight["inputs"]["train_ids"],
        "tier_quotas": list(V13_TIER_QUOTAS),
        "tier_loss_mass": list(V13_TIER_LOSS_MASS),
        "positive_tail_multiplier": V13_POSITIVE_TAIL_MULTIPLIER,
    }
    if all(path.exists() for path in paths.values()) and not config.force_rebuild_training_cache:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            return {"paths": paths, "metadata": metadata, "reused": True}

    case_paths = []
    train_ids = preflight["inputs"]["train_ids"]
    for position, case_id in enumerate(train_ids, start=1):
        print(f"[{position}/{len(train_ids)}] V13 deterministic training sample: {case_id}", flush=True)
        case_paths.append(_build_case_training_sample(preflight, case_id))

    X_parts = []
    y_parts = []
    tier_parts = []
    manifest_parts = []
    for case_id, path in zip(train_ids, case_paths):
        with np.load(path, allow_pickle=False) as data:
            X_parts.append(np.asarray(data["X"], dtype=np.float32))
            y_parts.append(np.asarray(data["y"], dtype=np.float32))
            tiers = np.asarray(data["tier_index"], dtype=np.int8)
            tier_parts.append(tiers)
            manifest_parts.append(pd.DataFrame({
                "iteration": config.iteration,
                "case_id": case_id,
                "element_id": np.asarray(data["element_id"], dtype=np.int64),
                "stress_tier": [STRESS_TIER_NAMES[index] for index in tiers],
            }))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    tier_index = np.concatenate(tier_parts, axis=0)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    case_values = manifest["case_id"].to_numpy(dtype=str)

    weights = np.zeros(len(y), dtype=np.float64)
    audit_rows = []
    for case_id in train_ids:
        case_mask = case_values == case_id
        for index, mass in enumerate(V13_TIER_LOSS_MASS):
            mask = case_mask & (tier_index == index)
            weights[mask] = mass / int(mask.sum())
    positive_tail = (y > 0.0) & np.isin(
        tier_index,
        [STRESS_TIER_NAMES.index("p95_to_p99"), STRESS_TIER_NAMES.index("top_p99")],
    )
    weights[positive_tail] *= V13_POSITIVE_TAIL_MULTIPLIER
    weights /= float(np.mean(weights))
    manifest["positive_local_underprediction"] = y > 0.0
    manifest["training_weight"] = weights
    for index, tier_name in enumerate(STRESS_TIER_NAMES):
        mask = tier_index == index
        audit_rows.append({
            "stress_tier": tier_name,
            "rows": int(mask.sum()),
            "rows_per_case": V13_TIER_QUOTAS[index],
            "configured_loss_mass_before_positive_multiplier": V13_TIER_LOSS_MASS[index],
            "actual_normalised_weight_mass": float(weights[mask].sum() / weights.sum()),
            "positive_local_underprediction_rows": int((mask & positive_tail).sum()),
        })

    scaling = _scaling_payload(X, y, V13_FEATURES, weights)
    np.save(paths["X"], X)
    np.save(paths["y"], y)
    np.save(paths["weights"], weights.astype(np.float32))
    _save_scaling(scaling, "v13_centered_v11_structural_residual", paths["scaling"])
    manifest.to_csv(paths["manifest"], index=False, compression="gzip")
    pd.DataFrame(audit_rows).to_csv(paths["weight_audit"], index=False)
    metadata = {
        **expected,
        "random_seed": RANDOM_SEED,
        "target": "complete-case-centred (actual - frozen V11 gain-0.90 prediction) / V10 scale",
        "selection_uses_complete_cases": True,
        "formula_centered_again_on_every_complete_case": True,
        "final_test_cases_read": 0,
    }
    _atomic_json(paths["metadata"], metadata)
    del X_parts, y_parts, tier_parts, X, y, tier_index, weights, manifest
    gc.collect()
    return {"paths": paths, "metadata": metadata, "reused": False}


def _search_directory(output_dir: Path) -> Path:
    return output_dir / "stages" / "stage_structural_residual"


def _search_signature(preflight: dict, cache: dict, config: V13Config) -> dict:
    return {
        "method": "v13_zero_case_mean_structural_tail_residual",
        "v13_signature": preflight["signature"]["v13_signature_sha256"],
        "features": list(V13_FEATURES),
        "n_rows": cache["metadata"]["n_rows"],
        "train_ids": cache["metadata"]["train_ids"],
        "tier_quotas": list(V13_TIER_QUOTAS),
        "tier_loss_mass": list(V13_TIER_LOSS_MASS),
        "positive_tail_multiplier": V13_POSITIVE_TAIL_MULTIPLIER,
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


def _completed_segments(stage_dir: Path, config: V13Config) -> list[int]:
    return [
        segment
        for segment in range(1, config.n_segments + 1)
        if (stage_dir / "segments" / f"segment_{segment:02d}" / "complete.json").exists()
    ]


def _write_progress(stage_dir: Path, config: V13Config) -> dict:
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
    config: V13Config,
    segment: int,
) -> Path:
    stage_dir = _search_directory(preflight["output_dir"])
    segment_dir = stage_dir / "segments" / f"segment_{segment:02d}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    complete_path = segment_dir / "complete.json"
    canonical_frontier = segment_dir / "frontier.csv"
    if complete_path.exists() and canonical_frontier.exists():
        print(f"V13 segment {segment}/{config.n_segments}: reused", flush=True)
        return canonical_frontier

    run_id = "v13_i1_structural_tail_residual"
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
            "--feature-names", json.dumps(V13_FEATURES),
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
            "--seed", str(RANDOM_SEED + 13_000),
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
            f"V13 segment {segment}/{config.n_segments}, attempt {attempt}: "
            f"{'checkpoint resume' if resume else 'fresh search'}",
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
                        f"V13 segment {segment}, attempt {attempt}: "
                        f"elapsed={elapsed / 60:.1f} min, inactive={inactive / 60:.1f} min, "
                        f"checkpoint={checkpoint.exists()}",
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
                f"V13 segment {segment}: complete; planned cumulative "
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
    raise RuntimeError(f"V13 segment {segment} failed after all retry attempts")


def run_symbolic_search(preflight: dict, cache: dict, config: V13Config) -> pd.DataFrame:
    stage_dir = _search_directory(preflight["output_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)
    signature = _search_signature(preflight, cache, config)
    signature_path = stage_dir / "search_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError("Existing V13 search state has a different signature")
    else:
        _atomic_json(signature_path, signature)
    latest = None
    for segment in range(1, config.n_segments + 1):
        latest = run_search_segment(preflight, cache, config, segment)
    if latest is None:
        raise RuntimeError("No V13 search segment completed")
    frontier = pd.read_csv(latest)
    frontier = (
        frontier.sort_values(["loss", "complexity"])
        .drop_duplicates("formula_scaled_sympy", keep="first")
        .reset_index(drop=True)
    )
    frontier["source_candidate_index"] = frontier["candidate_index"]
    frontier["candidate_index"] = np.arange(len(frontier), dtype=int)
    frontier["frontier_source"] = "v13_structural_tail_residual_iteration1"
    frontier.to_csv(stage_dir / "frontier.csv", index=False)
    return frontier


def evaluate_candidates(
    candidates: pd.DataFrame,
    scaling: dict,
    preflight: dict,
    case_ids: Sequence[str],
    split: str,
    config: V13Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    functions = {
        int(row.candidate_index): _compiled_formula(row, scaling)
        for row in candidates.itertuples(index=False)
    }
    candidate_rows = []
    baseline_rows = []
    invalid: dict[int, str] = {}
    for position, case_id in enumerate(case_ids, start=1):
        print(f"[{position}/{len(case_ids)}] {split} complete-case V13: {case_id}", flush=True)
        values = _complete_v11_components(preflight, case_id)
        actual = values["actual"]
        baseline = values["v11_prediction"]
        baseline_rows.append({
            "iteration": config.iteration,
            "split": split,
            "model": "v11_global_gain_0_90",
            "case_id": case_id,
            **evaluate_prediction_arrays(actual, baseline),
        })
        for candidate_index, function in functions.items():
            if candidate_index in invalid:
                continue
            try:
                raw = function(values["v13_matrix"])
                centered = raw - float(np.mean(raw))
                predicted = baseline + values["scale"] * centered
                candidate_rows.append({
                    "iteration": config.iteration,
                    "split": split,
                    "model": "v13_structural_residual_candidate",
                    "candidate_index": candidate_index,
                    "case_id": case_id,
                    **evaluate_prediction_arrays(actual, predicted),
                })
            except Exception as exc:
                invalid[candidate_index] = repr(exc)
        del values, actual, baseline
        gc.collect()

    candidate_cases = pd.DataFrame(candidate_rows)
    baseline_cases = pd.DataFrame(baseline_rows)
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
    return pd.DataFrame(records), candidate_cases, baseline_cases


def _aggregate_one(table: pd.DataFrame) -> pd.Series:
    return aggregate_case_metrics(table, ["iteration", "split", "model"]).iloc[0]


def score_and_select(
    metrics: pd.DataFrame,
    baseline_cases: pd.DataFrame,
    output_dir: Path,
    config: V13Config,
) -> pd.Series:
    valid = metrics[metrics["candidate_valid"].fillna(False)].copy()
    if valid.empty:
        raise RuntimeError("No finite V13 candidate on every validation case")
    baseline = _aggregate_one(baseline_cases)
    baseline_rmse = float(baseline["macro_rmse"])
    baseline_p95 = float(baseline["mean_p95_relative_error"])
    baseline_worst = float(baseline["worst_case_rmse"])
    baseline_max_ratio = float(baseline["max_prediction_abs_max_ratio"])

    rank_terms = [
        ("validation_macro_rmse", 0.20, True),
        ("validation_mean_top5_actual_rmse", 0.15, True),
        ("validation_mean_p95_relative_error", 0.10, True),
        ("validation_mean_p99_relative_error", 0.20, True),
        ("validation_mean_p99_underprediction_fraction", 0.15, True),
        ("validation_mean_top1pct_hotspot_overlap", 0.15, False),
        ("validation_mean_top1_recall_in_predicted_top5", 0.05, False),
    ]
    valid["engineering_selection_score"] = 0.0
    for column, weight, ascending in rank_terms:
        valid["engineering_selection_score"] += weight * valid[column].rank(
            pct=True, ascending=ascending, method="average"
        )
    valid["validation_macro_rmse_improvement_fraction"] = (
        baseline_rmse - valid["validation_macro_rmse"]
    ) / max(baseline_rmse, 1e-12)
    valid["gate_validation_rmse"] = (
        valid["validation_macro_rmse_improvement_fraction"]
        >= config.minimum_validation_rmse_improvement_fraction
    )
    valid["gate_validation_worst_case"] = (
        valid["validation_worst_case_rmse"] <= 1.10 * baseline_worst
    )
    valid["gate_validation_p95_preserved"] = (
        valid["validation_mean_p95_relative_error"]
        <= baseline_p95 + config.maximum_p95_degradation_absolute
    )
    valid["gate_validation_p99_relative"] = (
        valid["validation_mean_p99_relative_error"] <= config.maximum_p99_relative_error
    )
    valid["gate_validation_p99_underprediction"] = (
        valid["validation_mean_p99_underprediction_fraction"]
        <= config.maximum_p99_underprediction
    )
    valid["gate_validation_top1_overlap"] = (
        valid["validation_mean_top1pct_hotspot_overlap"]
        >= config.minimum_top1_hotspot_overlap
    )
    valid["gate_validation_top1_recall"] = (
        valid["validation_mean_top1_recall_in_predicted_top5"]
        >= config.minimum_top1_recall_in_predicted_top5
    )
    valid["gate_validation_numerical"] = (
        valid["validation_max_prediction_abs_max_ratio"]
        <= max(1.60, 1.05 * baseline_max_ratio)
    )
    gate_columns = [column for column in valid.columns if column.startswith("gate_validation_")]
    valid["all_v13_validation_gates_pass"] = valid[gate_columns].all(axis=1)
    if valid["all_v13_validation_gates_pass"].any():
        pool = valid[valid["all_v13_validation_gates_pass"]].copy()
        status = "all_v13_validation_gates"
    else:
        improved = valid[
            valid["gate_validation_numerical"]
            & (valid["validation_macro_rmse_improvement_fraction"] > 0.0)
        ].copy()
        pool = improved if not improved.empty else valid[valid["gate_validation_numerical"]].copy()
        if pool.empty:
            pool = valid
        status = "diagnostic_improving_candidates" if not improved.empty else "diagnostic_finite_candidates"
    selected = pool.sort_values(
        [
            "engineering_selection_score",
            "validation_mean_p99_underprediction_fraction",
            "validation_mean_top1pct_hotspot_overlap",
            "validation_macro_rmse",
            "complexity",
            "candidate_index",
        ],
        ascending=[True, True, False, True, True, True],
    ).iloc[0].copy()
    selected["selection_pool_status"] = status
    selected["selection_method"] = (
        "validation-only complete-case multi-objective selection; no internal or final case selects the formula"
    )
    for column in valid.columns:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics.loc[valid.index, valid.columns] = valid
    metrics["selected_candidate"] = metrics["candidate_index"].eq(selected["candidate_index"])
    metrics.to_csv(output_dir / "candidate_validation_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "selected_formula.csv", index=False)
    return selected


def _scope_metrics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    split_metrics = aggregate_case_metrics(case_metrics, ["iteration", "split", "model"])
    all_rows = case_metrics.copy()
    all_rows["split"] = "all_149_development"
    all_metrics = aggregate_case_metrics(all_rows, ["iteration", "split", "model"])
    return pd.concat([split_metrics, all_metrics], ignore_index=True, sort=False)


def _metric_row(scope_metrics: pd.DataFrame, split: str, model: str) -> pd.Series:
    rows = scope_metrics[
        scope_metrics["split"].eq(split) & scope_metrics["model"].eq(model)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one metric row for {split}/{model}, found {len(rows)}")
    return rows.iloc[0]


def promotion_decision(
    selected: pd.Series,
    scope_metrics: pd.DataFrame,
    config: V13Config,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    validation_pass = bool(selected["all_v13_validation_gates_pass"])
    rows.append({
        "gate": "validation_candidate_passes_all_predeclared_gates",
        "pass": validation_pass,
        "observed": validation_pass,
        "criterion": True,
    })
    for split in ["internal_test", "all_149_development"]:
        baseline = _metric_row(scope_metrics, split, "v11_global_gain_0_90")
        candidate = _metric_row(scope_metrics, split, "v13_structural_tail_residual")
        improvement = (
            float(baseline["macro_rmse"]) - float(candidate["macro_rmse"])
        ) / max(float(baseline["macro_rmse"]), 1e-12)
        minimum = (
            -0.005 if split == "internal_test"
            else config.minimum_validation_rmse_improvement_fraction
        )
        p95_allowance = (
            config.maximum_p95_degradation_absolute
            if split == "internal_test"
            else config.maximum_all149_p95_degradation_absolute
        )
        checks = [
            (f"{split}_macro_rmse", improvement >= minimum, improvement, minimum),
            (
                f"{split}_p95_preserved",
                float(candidate["mean_p95_relative_error"])
                <= float(baseline["mean_p95_relative_error"]) + p95_allowance,
                float(candidate["mean_p95_relative_error"]),
                float(baseline["mean_p95_relative_error"]) + p95_allowance,
            ),
            (
                f"{split}_p99_relative_error",
                float(candidate["mean_p99_relative_error"]) <= config.maximum_p99_relative_error,
                float(candidate["mean_p99_relative_error"]),
                config.maximum_p99_relative_error,
            ),
            (
                f"{split}_p99_underprediction",
                float(candidate["mean_p99_underprediction_fraction"])
                <= config.maximum_p99_underprediction,
                float(candidate["mean_p99_underprediction_fraction"]),
                config.maximum_p99_underprediction,
            ),
            (
                f"{split}_top1_hotspot_overlap",
                float(candidate["mean_top1pct_hotspot_overlap"])
                >= config.minimum_top1_hotspot_overlap,
                float(candidate["mean_top1pct_hotspot_overlap"]),
                config.minimum_top1_hotspot_overlap,
            ),
            (
                f"{split}_top1_recall",
                float(candidate["mean_top1_recall_in_predicted_top5"])
                >= config.minimum_top1_recall_in_predicted_top5,
                float(candidate["mean_top1_recall_in_predicted_top5"]),
                config.minimum_top1_recall_in_predicted_top5,
            ),
            (
                f"{split}_numerical_guardrail",
                float(candidate["max_prediction_abs_max_ratio"])
                <= max(1.60, 1.05 * float(baseline["max_prediction_abs_max_ratio"])),
                float(candidate["max_prediction_abs_max_ratio"]),
                max(1.60, 1.05 * float(baseline["max_prediction_abs_max_ratio"])),
            ),
        ]
        rows.extend(
            {"gate": name, "pass": bool(passed), "observed": observed, "criterion": criterion}
            for name, passed, observed, criterion in checks
        )
    gates = pd.DataFrame(rows)
    all_pass = bool(gates["pass"].all())
    decision = {
        "promotion_status": (
            "promote_v13_to_four_iteration_stability_testing"
            if all_pass
            else "retain_v11_global_gain_0_90"
        ),
        "all_gates_pass": all_pass,
        "failed_gates": gates.loc[~gates["pass"], "gate"].tolist(),
        "selected_candidate_index": int(selected["candidate_index"]),
        "selection_pool_status": selected["selection_pool_status"],
        "final_test_cases_read": 0,
    }
    return gates, decision


def _save_formula(
    preflight: dict,
    selected: pd.Series,
    decision: dict,
) -> None:
    output_dir = preflight["output_dir"]
    v11_formula = str(preflight["v11_gain_formula"]["combined_stress_formula"])
    log_scale = str(preflight["v11_gain_formula"]["v10_log_scale_formula"])
    residual_raw = str(selected["formula_original_variables"])
    residual_centered = f"({residual_raw}) - case_mean({residual_raw})"
    v13_formula = f"({v11_formula}) + exp({log_scale}) * ({residual_centered})"
    promoted = decision["promotion_status"].startswith("promote_v13")
    candidate_record = {
        "iteration": 1,
        "method": "frozen_v11_plus_zero_case_mean_structural_residual",
        "residual_raw_formula": residual_raw,
        "residual_centering_rule": "residual_raw_minus_complete_case_predictor_only_mean",
        "combined_stress_formula": v13_formula,
        "residual_complexity": int(selected["complexity"]),
        "promotion_status": decision["promotion_status"],
    }
    pd.DataFrame([candidate_record]).to_csv(
        output_dir / "v13_candidate_formula.csv", index=False
    )
    deployment_record = {
        "iteration": 1,
        "selected_model": "v13_structural_tail_residual" if promoted else "v11_global_gain_0_90",
        "combined_stress_formula": v13_formula if promoted else v11_formula,
        "promotion_status": decision["promotion_status"],
        "selected_for_deployment": True,
    }
    pd.DataFrame([deployment_record]).to_csv(
        output_dir / "selected_deployment_formula.csv", index=False
    )
    definitions = [
        "CT3 V13 structural residual symbolic stress candidate",
        "======================================================",
        "",
        "Frozen parent:",
        f"sigma_v11 = {v11_formula}",
        "",
        "V13 residual:",
        f"r_raw = {residual_raw}",
        "r_centered = r_raw - case_mean(r_raw)",
        "",
        "Candidate combined expression:",
        f"sigma_v13 = {v13_formula}",
        "",
        f"Promotion decision: {decision['promotion_status']}",
        "",
        "Predictor-only V13 feature definitions",
        "--------------------------------------",
        "*_within_case_z = (local predictor - case predictor mean) / case predictor std",
        "*_fraction_proxy = case-relative coordinate fraction in [0,1]",
        "nearest_*_boundary = min(fraction, 1-fraction)",
        "signed_*_position = 2*fraction-1",
        "theta harmonics use analytic sin/cos double- and quadruple-angle identities",
        "v10_normalised_shape_proxy = frozen V10 predictor shape before stress scaling",
        "v11_tail_centered_proxy = frozen V11 tail_raw - case_mean(V11 tail_raw)",
        "v11_prediction_within_case_z_proxy = (sigma_v11-case_mean(sigma_v11))/case_std(sigma_v11)",
        "v11_predicted_upper_gate_proxy = 0.5*(tanh((v11_prediction_z-1.2815515655)/0.5)+1)",
        "*_by_nearest_* features are explicit products of physical within-case z scores and boundary distances",
        "*_by_v11_upper_gate features are explicit products with the predictor-only smooth upper-tail gate",
        "",
        "Actual stress is not required by any deployed feature or case_mean operation.",
        "The 50 final-test case element files were not read.",
    ]
    (output_dir / "v13_candidate_formula.txt").write_text(
        "\n".join(definitions) + "\n", encoding="utf-8"
    )
    selected_text = (
        "V13 was promoted.\n\n" + "\n".join(definitions)
        if promoted
        else (
            "V13 did not pass every predeclared development gate.\n"
            "The frozen V11 global-gain-0.90 formula remains selected.\n\n"
            + f"sigma_selected = {v11_formula}\n"
        )
    )
    (output_dir / "selected_deployment_formula.txt").write_text(
        selected_text, encoding="utf-8"
    )


def _case_failure_registry(case_metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_id", "split"]
    baseline = case_metrics[case_metrics["model"].eq("v11_global_gain_0_90")].copy()
    candidate = case_metrics[case_metrics["model"].eq("v13_structural_tail_residual")].copy()
    keep = [
        "rmse",
        "p95_relative_error",
        "p99_relative_error",
        "p99_underprediction_fraction",
        "top1pct_hotspot_overlap",
        "top1_recall_in_predicted_top5",
        "top5_actual_bias",
    ]
    merged = baseline[keys + keep].merge(
        candidate[keys + keep], on=keys, suffixes=("_v11", "_v13"), validate="one_to_one"
    )
    for metric in keep:
        merged[f"delta_{metric}_v13_minus_v11"] = (
            merged[f"{metric}_v13"] - merged[f"{metric}_v11"]
        )
    merged["flag_p99_underprediction_above_10pct"] = (
        merged["p99_underprediction_fraction_v13"] > 0.10
    )
    merged["flag_p99_relative_error_above_15pct"] = (
        merged["p99_relative_error_v13"] > 0.15
    )
    merged["flag_top1_overlap_below_60pct"] = (
        merged["top1pct_hotspot_overlap_v13"] < 0.60
    )
    merged["flag_p95_relative_error_above_15pct"] = (
        merged["p95_relative_error_v13"] > 0.15
    )
    predictor_columns = [
        column
        for column in summary.columns
        if column == "case_id"
        or column.startswith("fluence_rate_")
        or column.startswith("temperature_")
        or column.startswith("weight_loss_rate_")
    ]
    merged = merged.merge(summary[predictor_columns], on="case_id", how="left", validate="one_to_one")
    return merged.sort_values(
        ["p99_underprediction_fraction_v13", "rmse_v13"], ascending=[False, False]
    ).reset_index(drop=True)


def _save_plots(scope_metrics: pd.DataFrame, registry: pd.DataFrame, output_dir: Path) -> None:
    all149 = scope_metrics[scope_metrics["split"].eq("all_149_development")].copy()
    metrics = [
        ("macro_rmse", "Macro RMSE", False),
        ("mean_p99_underprediction_fraction", "P99 underprediction", False),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (column, title, higher_better) in zip(axes, metrics):
        values = all149.sort_values(column, ascending=higher_better)
        axis.barh(values["model"], values[column], color=["#397a83", "#c56f3d"])
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "v13_all149_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        registry["rmse_v11"], registry["rmse_v13"],
        c=registry["p99_underprediction_fraction_v13"], cmap="viridis", s=32,
    )
    lower = min(registry["rmse_v11"].min(), registry["rmse_v13"].min())
    upper = max(registry["rmse_v11"].max(), registry["rmse_v13"].max())
    axes[0].plot([lower, upper], [lower, upper], "--", color="#555555")
    axes[0].set_xlabel("V11 case RMSE")
    axes[0].set_ylabel("V13 case RMSE")
    axes[0].set_title("Complete-case RMSE change")
    axes[1].scatter(
        registry["p99_underprediction_fraction_v11"],
        registry["p99_underprediction_fraction_v13"],
        c=registry["top1pct_hotspot_overlap_v13"], cmap="plasma", s=32,
    )
    axes[1].plot([0, 0.35], [0, 0.35], "--", color="#555555")
    axes[1].axhline(0.10, color="#b33a3a", linestyle=":")
    axes[1].set_xlabel("V11 P99 underprediction")
    axes[1].set_ylabel("V13 P99 underprediction")
    axes[1].set_title("Tail underprediction change")
    fig.tight_layout()
    fig.savefig(output_dir / "v13_case_metric_deltas.png", dpi=180)
    plt.close(fig)


def run_v13_pilot(package_root: Path, config: V13Config) -> dict:
    started = time.time()
    preflight = preflight_v13(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "v13_complete.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text(encoding="utf-8"))

    _atomic_json(output_dir / "run_configuration.json", asdict(config))
    pd.DataFrame([
        {
            "feature": feature,
            "predictor_only": True,
            "feature_family": (
                "local_physical" if feature in V13_LOCAL_PHYSICAL
                else "case_context" if feature in V13_CONTEXT
                else "geometry" if feature in V13_GEOMETRY
                else "periodic_harmonic" if feature in V13_HARMONICS
                else "frozen_parent_proxy" if feature in V13_PARENT_PROXIES
                else "boundary_interaction" if feature in V13_BOUNDARY_INTERACTIONS
                else "predicted_tail_interaction"
            ),
        }
        for feature in V13_FEATURES
    ]).to_csv(output_dir / "feature_registry.csv", index=False)
    _atomic_json(output_dir / "design_rationale.json", {
        "parent_model": "frozen V11 global gain 0.90",
        "v12_conclusion": "adaptive gain nearly matched oracle but failed all-case RMSE and P99-underprediction gates",
        "target": "complete-case-centred signed V11 residual divided by frozen V10 scale",
        "sampling": dict(zip(STRESS_TIER_NAMES, V13_TIER_QUOTAS)),
        "tier_loss_mass": dict(zip(STRESS_TIER_NAMES, V13_TIER_LOSS_MASS)),
        "positive_tail_multiplier": V13_POSITIVE_TAIL_MULTIPLIER,
        "selection": "validation-only complete-case engineering gates",
        "internal_role": "one-time development confirmation, never candidate selection",
        "all149_role": "descriptive development audit and promotion check",
        "final_test_cases_read": 0,
    })
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "prepare_recoverable_training_cache",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_test_cases_read": 0,
    })
    inhibitor = _start_sleep_inhibitor(output_dir)
    try:
        cache = prepare_v13_training_cache(preflight, config)
        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "recoverable_structural_residual_search",
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
        validation_metrics, validation_cases, validation_baseline = evaluate_candidates(
            shortlist,
            scaling,
            preflight,
            preflight["inputs"]["validation_ids"],
            "validation",
            config,
        )
        selected = score_and_select(
            validation_metrics, validation_baseline, stage_dir, config
        )
        selected_table = pd.DataFrame([selected])
        _, internal_cases, internal_baseline = evaluate_candidates(
            selected_table,
            scaling,
            preflight,
            preflight["inputs"]["internal_ids"],
            "internal_test",
            config,
        )
        _, train_cases, train_baseline = evaluate_candidates(
            selected_table,
            scaling,
            preflight,
            preflight["inputs"]["train_ids"],
            "train",
            config,
        )
        selected_index = int(selected["candidate_index"])
        validation_selected = validation_cases[
            validation_cases["candidate_index"].eq(selected_index)
        ].copy()
        candidate_cases = pd.concat(
            [train_cases, validation_selected, internal_cases], ignore_index=True
        )
        candidate_cases["model"] = "v13_structural_tail_residual"
        baseline_cases = pd.concat(
            [train_baseline, validation_baseline, internal_baseline], ignore_index=True
        )
        all_cases = pd.concat([baseline_cases, candidate_cases], ignore_index=True, sort=False)
        if all_cases["case_id"].nunique() != 149:
            raise AssertionError("V13 complete-case report does not contain all 149 development cases")
        scope_metrics = _scope_metrics(all_cases)
        gates, decision = promotion_decision(selected, scope_metrics, config)
        all_cases.to_csv(
            output_dir / "v13_case_metrics.csv.gz", index=False, compression="gzip"
        )
        scope_metrics.to_csv(output_dir / "v13_scope_metrics.csv", index=False)
        gates.to_csv(output_dir / "v13_promotion_gates.csv", index=False)
        _atomic_json(output_dir / "v13_promotion_decision.json", decision)
        registry = _case_failure_registry(all_cases, preflight["summary"])
        registry.to_csv(output_dir / "v13_case_failure_registry.csv", index=False)
        _save_formula(preflight, selected, decision)
        _save_plots(scope_metrics, registry, output_dir)
        result = {
            "status": "complete",
            "iteration": 1,
            "prototype_only": True,
            "parent_model": "v11_global_gain_0_90",
            "train_cases": 119,
            "validation_cases": 15,
            "internal_test_cases": 15,
            "development_cases_evaluated": 149,
            "final_test_cases_read": 0,
            "training_rows": 119 * config.rows_per_training_case,
            "planned_population_iterations": config.population_iterations_total,
            "completed_segments": config.n_segments,
            "selected_candidate_index": selected_index,
            "selection_pool_status": selected["selection_pool_status"],
            "promotion_status": decision["promotion_status"],
            "failed_promotion_gates": decision["failed_gates"],
            "elapsed_seconds": time.time() - started,
            "output_directory": str(output_dir),
        }
        _atomic_json(completion_path, result)
        _atomic_json(output_dir / "run_status.json", {**result, "stage": "complete"})
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
