"""V9 residual symbolic regression pilot for the CT3 graphite FEM surrogate.

The completed V8 model is retained as a fixed baseline:

    stress = case_mean + exp(case_log_scale) * global_shape

V9 searches only for a correction to the normalised global shape:

    stress = case_mean + exp(case_log_scale) * (global_shape + residual_shape)

The final-test cases remain sealed. Candidate formula selection uses complete
validation cases only. The selected pilot is then reported once on the
iteration-1 internal-test cases without further tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
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
import sympy as sp
from sklearn.ensemble import HistGradientBoostingRegressor


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


BOUNDARY_PROXY_FEATURES = [
    "rho_fraction_proxy",
    "z_fraction_proxy",
    "nearest_radial_boundary_fraction_proxy",
    "nearest_axial_boundary_fraction_proxy",
]
RESIDUAL_FORMULA_FEATURES = SHAPE_FORMULA_FEATURES + BOUNDARY_PROXY_FEATURES

STRESS_TIER_LABELS = ["below_p90", "p90_to_p95", "p95_to_p99", "top_p99"]
STRESS_TIER_QUOTAS = [2_500, 500, 1_000, 1_000]
STRESS_TIER_TARGET_MASS = [0.50, 0.10, 0.20, 0.20]


@dataclass(frozen=True)
class ResidualShapePilotConfig:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    rows_per_training_case: int = 5_000

    # A 4,000 population-iteration pilot. A later formal search can increase
    # this to 16,000 only if the residual proves learnable.
    total_niterations: int = 500
    populations: int = 8
    segment_niterations: int = 100
    population_size: int = 40
    ncycles_per_iteration: int = 100
    batch_size: int = 50_000
    maxsize: int = 28
    maxdepth: int = 10
    julia_threads: int = 8

    no_activity_timeout_seconds: int = 45 * 60
    segment_wall_timeout_seconds: int = 3 * 60 * 60
    watchdog_poll_seconds: int = 60
    max_attempts_per_segment: int = 3
    max_candidates_for_full_validation: int = 16
    hgb_max_iter: int = 100
    force_rebuild_training_cache: bool = False

    def validate(self) -> None:
        if self.iteration != 1:
            raise ValueError("The first V9 pilot is locked to frozen iteration 1")
        if self.rows_per_training_case != sum(STRESS_TIER_QUOTAS):
            raise ValueError("V9 requires exactly 5,000 discovery rows per training case")
        if self.total_niterations % self.segment_niterations != 0:
            raise ValueError("total_niterations must be divisible by segment_niterations")
        if self.populations != 8:
            raise ValueError("The comparison is locked to eight PySR populations")
        if self.segment_wall_timeout_seconds <= self.no_activity_timeout_seconds:
            raise ValueError("Segment wall timeout must exceed inactivity timeout")
        if len(STRESS_TIER_QUOTAS) != len(STRESS_TIER_TARGET_MASS):
            raise ValueError("Stress-tier quota and target-mass definitions differ")
        if not np.isclose(sum(STRESS_TIER_TARGET_MASS), 1.0):
            raise ValueError("Stress-tier target loss mass must sum to one")

    @property
    def n_segments(self) -> int:
        return self.total_niterations // self.segment_niterations

    @property
    def target_population_iterations(self) -> int:
        return self.total_niterations * self.populations

    @property
    def population_iterations_per_segment(self) -> int:
        return self.segment_niterations * self.populations


def output_directory(package_root: Path, config: ResidualShapePilotConfig) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "09_residual_shape_symbolic_pilot"
        / config.output_subdir
    )


def v8_directory(package_root: Path) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "08_recoverable_16000_single_formula"
        / "iteration_1"
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load_v8_baseline(package_root: Path) -> dict:
    source = v8_directory(package_root)
    paths = {
        "completion": source / "round_complete.json",
        "mean_row": source / "selected_case_mean_formula.csv",
        "mean_scaling": source / "case_mean_scaling.csv",
        "scale_row": source / "selected_case_log_scale_formula.csv",
        "scale_scaling": source / "case_log_scale_scaling.csv",
        "shape_row": source / "selected_shape_formula.csv",
        "shape_scaling": source / "training_cache" / "shape_scaling.csv",
        "split_metrics": source / "selected_formula_split_metrics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "V9 requires the completed V8 Iteration-1 baseline artifacts: "
            f"{missing}"
        )
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("V8 Iteration 1 is not marked complete")
    return {
        "mean_row": pd.read_csv(paths["mean_row"]).iloc[0],
        "mean_scaling": _load_scaling(paths["mean_scaling"]),
        "scale_row": pd.read_csv(paths["scale_row"]).iloc[0],
        "scale_scaling": _load_scaling(paths["scale_scaling"]),
        "shape_row": pd.read_csv(paths["shape_row"]).iloc[0],
        "shape_scaling": _load_scaling(paths["shape_scaling"]),
        "split_metrics": pd.read_csv(paths["split_metrics"]),
        "paths": paths,
    }


def preflight_residual_pilot(
    package_root: Path,
    config: ResidualShapePilotConfig,
) -> dict:
    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    feasibility_config = FeasibilityConfig(
        iteration=config.iteration,
        rows_per_case=config.rows_per_training_case,
        output_subdir="v9_residual_pilot_input_check",
    )
    inputs = load_inputs(package_root, feasibility_config)
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    baseline = _load_v8_baseline(package_root)
    worker_path = package_root / "scripts" / "run_residual_shape_segment.py"
    checks = [
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "residual_features", "value": len(RESIDUAL_FORMULA_FEATURES), "expected": 31},
        {"check": "worker_script_exists", "value": worker_path.exists(), "expected": True},
        {
            "check": "pilot_population_iterations",
            "value": config.target_population_iterations,
            "expected": 4_000,
        },
        {"check": "recoverable_segments", "value": config.n_segments, "expected": 5},
    ]
    table = pd.DataFrame(checks)
    table["pass"] = table["value"] == table["expected"]
    table.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not table["pass"].all():
        raise RuntimeError("V9 residual-pilot preflight failed")
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


def build_residual_feature_matrix(
    frame: pd.DataFrame,
    summary: pd.Series,
) -> np.ndarray:
    """Build 27 reviewed predictors plus four dimensionless boundary proxies."""

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

    rho = frame["rho"].to_numpy(dtype=np.float32)
    z = frame["z"].to_numpy(dtype=np.float32)
    rho_fraction = np.clip(
        (rho - np.float32(summary["rho_min"]))
        / np.float32(_safe_span(summary["rho_min"], summary["rho_max"])),
        0.0,
        1.0,
    )
    z_fraction = np.clip(
        (z - np.float32(summary["z_min"]))
        / np.float32(_safe_span(summary["z_min"], summary["z_max"])),
        0.0,
        1.0,
    )
    boundary = np.column_stack([
        rho_fraction,
        z_fraction,
        np.minimum(rho_fraction, 1.0 - rho_fraction),
        np.minimum(z_fraction, 1.0 - z_fraction),
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


def _v9_discovery_selection(
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
            RANDOM_SEED + 20_000 * case_number + 211 * tier_index,
        )
        if len(chosen) != quota:
            raise AssertionError(
                f"{case_id}: stress tier {tier_index} has {len(chosen)} rows, "
                f"expected {quota}"
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


def prepare_residual_training_cache(
    preflight: dict,
    config: ResidualShapePilotConfig,
) -> dict:
    cache_dir = preflight["output_dir"] / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "X": cache_dir / "residual_features.npy",
        "y": cache_dir / "residual_target.npy",
        "weights": cache_dir / "tier_mass_weights.npy",
        "scaling": cache_dir / "residual_scaling.csv",
        "metadata": cache_dir / "cache_metadata.json",
        "audit": cache_dir / "training_sample_audit.csv",
        "manifest": cache_dir / "training_sample_manifest.csv.gz",
    }
    expected_rows = len(preflight["inputs"]["train_ids"]) * config.rows_per_training_case
    if (
        all(path.exists() for path in paths.values())
        and not config.force_rebuild_training_cache
    ):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("n_rows") == expected_rows
            and metadata.get("features") == list(RESIDUAL_FORMULA_FEATURES)
            and metadata.get("train_ids") == preflight["inputs"]["train_ids"]
            and metadata.get("tier_quotas") == STRESS_TIER_QUOTAS
            and metadata.get("tier_target_loss_mass") == STRESS_TIER_TARGET_MASS
        ):
            return {"paths": paths, "metadata": metadata, "reused": True}

    X = np.empty((expected_rows, len(RESIDUAL_FORMULA_FEATURES)), dtype=np.float32)
    y = np.empty(expected_rows, dtype=np.float32)
    weights = np.empty(expected_rows, dtype=np.float32)
    baseline_functions = _baseline_functions(preflight["baseline"])
    shape_feature_count = len(SHAPE_FORMULA_FEATURES)
    audit_rows = []
    manifest_parts = []
    cursor = 0
    started = time.perf_counter()

    for position, case_id in enumerate(preflight["inputs"]["train_ids"], start=1):
        print(
            f"[{position}/{len(preflight['inputs']['train_ids'])}] "
            f"V9 residual sample: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        stress = frame[TARGET_COL].to_numpy(dtype=np.float64)
        local_matrix = frame[LOCAL_FEATURES].to_numpy(dtype=np.float32)
        selected, selected_weights, selected_tier, quantiles = _v9_discovery_selection(
            case_id, stress, local_matrix
        )
        sampled = frame.iloc[selected]
        matrix = build_residual_feature_matrix(sampled, summary)
        baseline_shape = baseline_functions["shape"](matrix[:, :shape_feature_count])
        actual_shape = (
            sampled[TARGET_COL].to_numpy(dtype=np.float64) - float(summary["stress_mean"])
        ) / float(summary["stress_p95"] - summary["stress_mean"])
        next_cursor = cursor + len(sampled)
        X[cursor:next_cursor] = matrix
        y[cursor:next_cursor] = (actual_shape - baseline_shape).astype(np.float32)
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
                "residual_rmse_before_correction": float(
                    np.sqrt(np.mean((actual_shape - baseline_shape) ** 2))
                ),
            })
        manifest_parts.append(pd.DataFrame({
            "iteration": config.iteration,
            "case_id": case_id,
            "element_id": sampled["element_id"].to_numpy(dtype=np.int64),
            "stress_tier": [STRESS_TIER_LABELS[index] for index in selected_tier],
            "discovery_weight": selected_weights.astype(np.float32),
        }))
        cursor = next_cursor
        del frame, sampled, matrix, stress, local_matrix, baseline_shape, actual_shape
        gc.collect()

    if cursor != expected_rows:
        raise AssertionError(f"Assembled {cursor} rows, expected {expected_rows}")
    weights /= np.float32(weights.mean(dtype=np.float64))
    scaling = _scaling_payload(X, y, RESIDUAL_FORMULA_FEATURES, weights)
    np.save(paths["X"], X)
    np.save(paths["y"], y)
    np.save(paths["weights"], weights)
    _save_scaling(scaling, "normalised_shape_residual", paths["scaling"])
    pd.DataFrame(audit_rows).to_csv(paths["audit"], index=False)
    pd.concat(manifest_parts, ignore_index=True).to_csv(
        paths["manifest"], index=False, compression="gzip"
    )
    metadata = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": list(RESIDUAL_FORMULA_FEATURES),
        "train_ids": preflight["inputs"]["train_ids"],
        "rows_per_case": config.rows_per_training_case,
        "random_seed": RANDOM_SEED,
        "tier_labels": STRESS_TIER_LABELS,
        "tier_quotas": STRESS_TIER_QUOTAS,
        "tier_target_loss_mass": STRESS_TIER_TARGET_MASS,
        "target": "actual_normalised_shape_minus_fixed_v8_global_shape",
        "complete_case_validation": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(paths["metadata"], metadata)
    del X, y, weights
    gc.collect()
    return {"paths": paths, "metadata": metadata, "reused": False}


def _predict_case_components(
    frame: pd.DataFrame,
    summary: pd.Series,
    baseline_functions: dict,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    matrix = build_residual_feature_matrix(frame, summary)
    context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float64)[None, :]
    predicted_mean = float(baseline_functions["mean"](context)[0])
    predicted_log_scale = float(baseline_functions["scale"](context)[0])
    if not np.isfinite(predicted_log_scale) or abs(predicted_log_scale) > 20.0:
        raise FloatingPointError("V8 case log-scale formula is non-finite or explosive")
    predicted_scale = float(np.exp(predicted_log_scale))
    baseline_shape = baseline_functions["shape"](
        matrix[:, : len(SHAPE_FORMULA_FEATURES)]
    )
    baseline_stress = predicted_mean + predicted_scale * baseline_shape
    return matrix, predicted_mean, predicted_scale, baseline_shape, baseline_stress


def run_residual_learnability_diagnostic(
    preflight: dict,
    cache: dict,
    config: ResidualShapePilotConfig,
) -> pd.DataFrame:
    """Estimate whether residual structure exists before spending PySR budget."""

    output_path = preflight["output_dir"] / "residual_learnability_split_metrics.csv"
    case_path = preflight["output_dir"] / "residual_learnability_case_metrics.csv.gz"
    if output_path.exists() and case_path.exists():
        return pd.read_csv(output_path)

    X = np.load(cache["paths"]["X"], mmap_mode="r")
    y = np.load(cache["paths"]["y"], mmap_mode="r")
    weights = np.load(cache["paths"]["weights"], mmap_mode="r")
    feature_index = {name: index for index, name in enumerate(RESIDUAL_FORMULA_FEATURES)}
    groups = {
        "residual_HGB_geometry_boundary": [
            feature_index[name]
            for name in [
                "rho", "theta_sin", "theta_cos", "z",
                "rho_within_case_z", "theta_sin_within_case_z",
                "theta_cos_within_case_z", "z_within_case_z",
                *BOUNDARY_PROXY_FEATURES,
            ]
        ],
        "residual_HGB_physical_fields": [
            feature_index[name]
            for name in [
                "fluence_rate", "temperature", "weight_loss_rate",
                "fluence_rate_within_case_z", "temperature_within_case_z",
                "weight_loss_rate_within_case_z",
            ]
        ],
        "residual_HGB_all_features": list(range(len(RESIDUAL_FORMULA_FEATURES))),
    }
    models = {}
    fit_rows = []
    for model_name, indices in groups.items():
        print(f"Residual learnability diagnostic: fitting {model_name}", flush=True)
        started = time.perf_counter()
        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=config.hgb_max_iter,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
        model.fit(np.asarray(X[:, indices]), np.asarray(y), sample_weight=np.asarray(weights))
        models[model_name] = (model, indices)
        fit_rows.append({
            "model": model_name,
            "n_features": len(indices),
            "fit_seconds": time.perf_counter() - started,
            "features_json": json.dumps(
                [RESIDUAL_FORMULA_FEATURES[index] for index in indices]
            ),
        })
    pd.DataFrame(fit_rows).to_csv(
        preflight["output_dir"] / "residual_learnability_model_audit.csv", index=False
    )

    baseline_functions = _baseline_functions(preflight["baseline"])
    rows = []
    for position, case_id in enumerate(preflight["inputs"]["validation_ids"], start=1):
        print(
            f"[{position}/{len(preflight['inputs']['validation_ids'])}] "
            f"Full validation residual diagnostic: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        matrix, predicted_mean, predicted_scale, _, baseline_stress = _predict_case_components(
            frame, summary, baseline_functions
        )
        rows.append({
            "iteration": config.iteration,
            "split": "validation",
            "model": "fixed_v8_global_trend",
            "case_id": case_id,
            **evaluate_prediction_arrays(actual, baseline_stress),
        })
        for model_name, (model, indices) in models.items():
            residual = model.predict(matrix[:, indices])
            predicted = baseline_stress + predicted_scale * residual
            rows.append({
                "iteration": config.iteration,
                "split": "validation",
                "model": model_name,
                "case_id": case_id,
                **evaluate_prediction_arrays(actual, predicted),
            })
        del frame, actual, matrix, baseline_stress
        gc.collect()

    case_metrics = pd.DataFrame(rows)
    case_metrics.to_csv(case_path, index=False, compression="gzip")
    split_metrics = aggregate_case_metrics(
        case_metrics, ["iteration", "split", "model"]
    )
    split_metrics.to_csv(output_path, index=False)
    baseline_rmse = float(
        split_metrics.loc[
            split_metrics["model"] == "fixed_v8_global_trend", "macro_rmse"
        ].iloc[0]
    )
    split_metrics["macro_rmse_improvement_vs_v8_fraction"] = (
        baseline_rmse - split_metrics["macro_rmse"]
    ) / max(baseline_rmse, 1e-12)
    split_metrics.to_csv(output_path, index=False)
    return split_metrics


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


def _completed_segments(output_dir: Path, n_segments: int) -> list[int]:
    return [
        segment
        for segment in range(1, n_segments + 1)
        if (output_dir / "segments" / f"segment_{segment:02d}" / "complete.json").exists()
    ]


def _write_progress(output_dir: Path, config: ResidualShapePilotConfig) -> dict:
    completed = _completed_segments(output_dir, config.n_segments)
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
        "planned_population_iterations_completed": (
            contiguous * config.population_iterations_per_segment
        ),
        "target_population_iterations": config.target_population_iterations,
        "planned_progress_fraction": contiguous / config.n_segments,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(output_dir / "search_progress.json", payload)
    return payload


def _search_signature(config: ResidualShapePilotConfig, cache: dict) -> dict:
    return {
        "method": "fixed_v8_global_shape_plus_one_residual_formula",
        "features": list(RESIDUAL_FORMULA_FEATURES),
        "n_rows": cache["metadata"]["n_rows"],
        "train_ids": cache["metadata"]["train_ids"],
        "tier_quotas": STRESS_TIER_QUOTAS,
        "tier_target_loss_mass": STRESS_TIER_TARGET_MASS,
        "total_niterations": config.total_niterations,
        "populations": config.populations,
        "segment_niterations": config.segment_niterations,
        "population_size": config.population_size,
        "ncycles_per_iteration": config.ncycles_per_iteration,
        "batch_size": config.batch_size,
        "maxsize": config.maxsize,
        "maxdepth": config.maxdepth,
        "operators": ["+", "-", "*", "/", "square", "abs", "gauss"],
        "random_seed": RANDOM_SEED,
    }


def run_residual_segment(
    preflight: dict,
    config: ResidualShapePilotConfig,
    cache: dict,
    segment: int,
) -> Path:
    output_dir = preflight["output_dir"]
    segment_dir = output_dir / "segments" / f"segment_{segment:02d}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    complete_path = segment_dir / "complete.json"
    canonical_frontier = segment_dir / "frontier.csv"
    if complete_path.exists() and canonical_frontier.exists():
        print(f"Segment {segment}/{config.n_segments}: completed result reused", flush=True)
        return canonical_frontier

    run_id = "v9_i1_residual_shape_pilot"
    run_directory = output_dir / "search_state" / "pysr_runs" / run_id
    checkpoint_path = run_directory / "checkpoint.pkl"
    hall_path = run_directory / "hall_of_fame.csv"
    snapshot_root = output_dir / "search_state" / "stable_snapshots"
    attempt_audit_path = output_dir / "segment_attempt_audit.csv"

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
            "--features", str(cache["paths"]["X"]),
            "--target", str(cache["paths"]["y"]),
            "--weights", str(cache["paths"]["weights"]),
            "--scaling", str(cache["paths"]["scaling"]),
            "--feature-names", json.dumps(list(RESIDUAL_FORMULA_FEATURES)),
            "--state-root", str(output_dir / "search_state"),
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
            "--seed", str(RANDOM_SEED),
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
            f"Segment {segment}/{config.n_segments}, attempt {attempt}: "
            f"starting ({'checkpoint resume' if resume else 'fresh search'})",
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
                stdout_path, stderr_path, worker_status_path, hall_path, checkpoint_path
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
                        f"Segment {segment}/{config.n_segments}, attempt {attempt}: "
                        f"elapsed={elapsed / 60:.1f} min, "
                        f"inactive={inactive / 60:.1f} min, "
                        f"checkpoint={checkpoint_path.exists()}",
                        flush=True,
                    )
                    last_report = elapsed
                _atomic_json(segment_dir / "watchdog_status.json", {
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
                "segment": segment,
                "attempt": attempt,
                "planned_population_iterations": config.population_iterations_per_segment,
                "elapsed_seconds": time.time() - started,
            })
            _write_progress(output_dir, config)
            print(
                f"Segment {segment}/{config.n_segments}: complete; planned cumulative "
                f"progress {segment * config.population_iterations_per_segment}/"
                f"{config.target_population_iterations}",
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
                    raise RuntimeError(f"Recovery destination already exists: {destination}")
                shutil.move(str(run_directory), str(destination))
            recovery = "no_stable_checkpoint; next_attempt_starts_fresh"
        audit_table = pd.read_csv(attempt_audit_path)
        audit_table.loc[
            (audit_table["segment"] == segment)
            & (audit_table["attempt"] == attempt),
            "recovery_action",
        ] = recovery
        audit_table.to_csv(attempt_audit_path, index=False)
        print(
            f"Segment {segment}, attempt {attempt} did not finish ({reason}); "
            f"recovery={recovery}",
            flush=True,
        )

    raise RuntimeError(
        f"Segment {segment} failed {config.max_attempts_per_segment} times. "
        "Rerun after reviewing its latest worker log; completed segments are retained."
    )


def run_residual_search(
    preflight: dict,
    cache: dict,
    config: ResidualShapePilotConfig,
) -> pd.DataFrame:
    signature_path = preflight["output_dir"] / "search_signature.json"
    signature = _search_signature(config, cache)
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError(
                "Existing V9 search state has a different signature. Use a new "
                "output_subdir instead of mixing experiments."
            )
    else:
        _atomic_json(signature_path, signature)
    latest = None
    for segment in range(1, config.n_segments + 1):
        latest = run_residual_segment(preflight, config, cache, segment)
    if latest is None:
        raise RuntimeError("No residual-search segment completed")
    frontier = pd.read_csv(latest)
    frontier = (
        frontier.sort_values(["loss", "complexity"])
        .drop_duplicates("formula_scaled_sympy", keep="first")
        .reset_index(drop=True)
    )
    frontier["source_candidate_index"] = frontier["candidate_index"]
    frontier["candidate_index"] = np.arange(len(frontier), dtype=int)
    frontier["frontier_source"] = "v9_residual_pilot_only"
    frontier.to_csv(preflight["output_dir"] / "residual_frontier.csv", index=False)
    return frontier


def _evaluate_residual_candidates(
    candidates: pd.DataFrame,
    residual_scaling: dict,
    preflight: dict,
    case_ids: Sequence[str],
    split: str,
    config: ResidualShapePilotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    functions = {
        int(row.candidate_index): _compiled_formula(row, residual_scaling)
        for row in candidates.itertuples(index=False)
    }
    baseline_functions = _baseline_functions(preflight["baseline"])
    metric_rows = []
    baseline_rows = []
    invalid: dict[int, str] = {}
    for position, case_id in enumerate(case_ids, start=1):
        print(
            f"[{position}/{len(case_ids)}] {split} full-case V9 evaluation: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        matrix, predicted_mean, predicted_scale, _, baseline_stress = _predict_case_components(
            frame, summary, baseline_functions
        )
        baseline_rows.append({
            "iteration": config.iteration,
            "split": split,
            "model": "fixed_v8_global_trend",
            "case_id": case_id,
            **evaluate_prediction_arrays(actual, baseline_stress),
        })
        for candidate_index, function in functions.items():
            if candidate_index in invalid:
                continue
            try:
                residual = function(matrix)
                predicted = baseline_stress + predicted_scale * residual
                metric_rows.append({
                    "iteration": config.iteration,
                    "split": split,
                    "model": "v9_trend_plus_symbolic_residual",
                    "candidate_index": candidate_index,
                    "case_id": case_id,
                    **evaluate_prediction_arrays(actual, predicted),
                })
            except Exception as exc:
                invalid[candidate_index] = repr(exc)
        del frame, actual, matrix, baseline_stress
        gc.collect()

    case_metrics = pd.DataFrame(metric_rows)
    baseline_cases = pd.DataFrame(baseline_rows)
    records = []
    for candidate in candidates.itertuples(index=False):
        candidate_index = int(candidate.candidate_index)
        group = case_metrics[case_metrics["candidate_index"] == candidate_index]
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
    return pd.DataFrame(records), case_metrics, baseline_cases


def _score_and_select_candidate(
    candidate_metrics: pd.DataFrame,
    baseline_cases: pd.DataFrame,
    output_dir: Path,
) -> pd.Series:
    valid = candidate_metrics[candidate_metrics["candidate_valid"].fillna(False)].copy()
    if valid.empty:
        raise RuntimeError("No residual candidate was finite on every validation case")
    error_metrics = [
        ("validation_macro_rmse", 0.20),
        ("validation_mean_top5_actual_rmse", 0.10),
        ("validation_mean_p95_relative_error", 0.10),
        ("validation_mean_p99_relative_error", 0.20),
        ("validation_mean_p99_underprediction_fraction", 0.15),
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

    baseline = aggregate_case_metrics(
        baseline_cases, ["iteration", "split", "model"]
    ).iloc[0]
    valid["macro_rmse_improvement_vs_v8_fraction"] = (
        float(baseline["macro_rmse"]) - valid["validation_macro_rmse"]
    ) / max(float(baseline["macro_rmse"]), 1e-12)
    valid["gate_numerical_guardrail"] = (
        valid["validation_max_prediction_abs_max_ratio"] <= 5.0
    )
    valid["gate_positive_macro_r2"] = valid["validation_macro_r2"] > 0.0
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
    valid["all_pilot_promotion_gates_pass"] = valid[[
        "gate_numerical_guardrail",
        "gate_positive_macro_r2",
        "gate_improves_v8_rmse_by_3pct",
        "gate_p95_relative_error",
        "gate_p99_relative_error",
        "gate_p95_underprediction",
        "gate_p99_underprediction",
        "gate_top1_hotspot_overlap",
        "gate_top1_recall",
    ]].all(axis=1)
    numerical = valid[
        valid["gate_numerical_guardrail"] & valid["gate_positive_macro_r2"]
    ].copy()
    pool = numerical if not numerical.empty else valid
    selected = pool.sort_values([
        "engineering_selection_score",
        "validation_macro_rmse",
        "validation_mean_p99_relative_error",
        "complexity",
        "candidate_index",
    ]).iloc[0].copy()
    selected["selection_method"] = (
        "validation-only multi-metric rank; complexity is the fourth tiebreaker, "
        "not a near-best-RMSE hard preference"
    )
    selected["pilot_promotion_status"] = (
        "passes_all_pilot_gates"
        if bool(selected["all_pilot_promotion_gates_pass"])
        else "prototype_selected_but_does_not_pass_all_pilot_gates"
    )
    for column in valid.columns:
        if column not in candidate_metrics.columns:
            candidate_metrics[column] = np.nan
    candidate_metrics.loc[valid.index, valid.columns] = valid
    candidate_metrics["selected_candidate"] = candidate_metrics["candidate_index"].eq(
        selected["candidate_index"]
    )
    candidate_metrics.to_csv(
        output_dir / "residual_candidate_validation_metrics.csv", index=False
    )
    pd.DataFrame([selected]).to_csv(
        output_dir / "selected_residual_formula.csv", index=False
    )
    return selected


def _save_formula_artifacts(
    preflight: dict,
    selected: pd.Series,
) -> dict:
    baseline = preflight["baseline"]
    mean_formula = str(baseline["mean_row"]["formula_original_variables"])
    log_scale_formula = str(baseline["scale_row"]["formula_original_variables"])
    trend_formula = str(baseline["shape_row"]["formula_original_variables"])
    residual_formula = str(selected["formula_original_variables"])
    shape_formula = f"({trend_formula}) + ({residual_formula})"
    composite = f"({mean_formula}) + exp({log_scale_formula}) * ({shape_formula})"
    record = {
        "iteration": 1,
        "method": "fixed_v8_global_shape_plus_v9_symbolic_residual",
        "case_mean_formula": mean_formula,
        "case_log_scale_formula": log_scale_formula,
        "fixed_global_shape_formula": trend_formula,
        "residual_shape_formula": residual_formula,
        "corrected_shape_formula": shape_formula,
        "composite_stress_formula": composite,
        "residual_complexity": int(selected["complexity"]),
        "pilot_promotion_status": selected["pilot_promotion_status"],
    }
    pd.DataFrame([record]).to_csv(
        preflight["output_dir"] / "selected_corrected_composite_formula.csv", index=False
    )
    text = (
        "CT3 V9 residual-corrected symbolic stress formula\n"
        "=================================================\n\n"
        f"Case mean:\nmu = {mean_formula}\n\n"
        f"Positive case scale:\nscale = exp({log_scale_formula})\n\n"
        f"Fixed V8 global shape:\nglobal_shape = {trend_formula}\n\n"
        f"V9 residual shape:\nresidual_shape = {residual_formula}\n\n"
        f"Corrected shape:\nshape = {shape_formula}\n\n"
        f"Deployable stress expression:\nsigma = {composite}\n\n"
        "Boundary proxy definitions:\n"
        "rho_fraction_proxy = (rho - rho_min) / (rho_max - rho_min)\n"
        "z_fraction_proxy = (z - z_min) / (z_max - z_min)\n"
        "nearest_radial_boundary_fraction_proxy = min(rho_fraction_proxy, "
        "1 - rho_fraction_proxy)\n"
        "nearest_axial_boundary_fraction_proxy = min(z_fraction_proxy, "
        "1 - z_fraction_proxy)\n\n"
        "These are case-relative geometric proxies, not confirmed physical "
        "distances to named FEM surfaces. Final-test cases were not read.\n"
    )
    (preflight["output_dir"] / "selected_corrected_composite_formula.txt").write_text(
        text, encoding="utf-8"
    )
    return record


def _save_pilot_plots(
    split_metrics: pd.DataFrame,
    selected_cases: pd.DataFrame,
    learnability: pd.DataFrame,
    output_dir: Path,
) -> None:
    comparison = pd.concat([learnability, split_metrics], ignore_index=True, sort=False)
    validation = (
        comparison[comparison["split"] == "validation"]
        .drop_duplicates(["split", "model"], keep="last")
        .copy()
    )
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
    fig.savefig(output_dir / "v9_validation_model_comparison.png", dpi=180)
    plt.close(fig)

    selected_validation = selected_cases[selected_cases["split"] == "validation"]
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
        axis.set_title(f"Selected V9 {tail.upper()} adaptation")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_v9_tail_adaptation.png", dpi=180)
    plt.close(fig)


def run_residual_shape_pilot(
    package_root: Path,
    config: ResidualShapePilotConfig,
) -> dict:
    started = time.time()
    preflight = preflight_residual_pilot(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "pilot_complete.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text(encoding="utf-8"))

    save_json(output_dir / "run_configuration.json", asdict(config))
    pd.DataFrame([
        {"feature": feature, "role": "residual_formula_predictor"}
        for feature in RESIDUAL_FORMULA_FEATURES
    ]).to_csv(output_dir / "feature_registry.csv", index=False)
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "prepare_training_cache",
        "final_test_cases_read": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    sleep_inhibitor = _start_sleep_inhibitor(output_dir)
    try:
        cache = prepare_residual_training_cache(preflight, config)
        learnability = run_residual_learnability_diagnostic(preflight, cache, config)
        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "recoverable_residual_symbolic_search",
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        frontier = run_residual_search(preflight, cache, config)
        shortlist = _shortlist_shape_frontier(
            frontier, config.max_candidates_for_full_validation
        ).copy()
        shortlist.to_csv(output_dir / "residual_full_validation_shortlist.csv", index=False)
        residual_scaling = _load_scaling(cache["paths"]["scaling"])
        validation_metrics, validation_cases, validation_baseline = (
            _evaluate_residual_candidates(
                shortlist,
                residual_scaling,
                preflight,
                preflight["inputs"]["validation_ids"],
                "validation",
                config,
            )
        )
        selected = _score_and_select_candidate(
            validation_metrics, validation_baseline, output_dir
        )
        selected_index = int(selected["candidate_index"])
        selected_validation = validation_cases[
            validation_cases["candidate_index"] == selected_index
        ].copy()
        selected_table = pd.DataFrame([selected])
        _, selected_internal, internal_baseline = _evaluate_residual_candidates(
            selected_table,
            residual_scaling,
            preflight,
            preflight["inputs"]["internal_ids"],
            "internal_test",
            config,
        )
        selected_cases = pd.concat(
            [selected_validation, selected_internal], ignore_index=True
        )
        baseline_cases = pd.concat(
            [validation_baseline, internal_baseline], ignore_index=True
        )
        selected_cases.to_csv(
            output_dir / "selected_formula_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        baseline_cases.to_csv(
            output_dir / "fixed_v8_baseline_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        split_metrics = pd.concat([
            aggregate_case_metrics(
                baseline_cases, ["iteration", "split", "model"]
            ),
            aggregate_case_metrics(
                selected_cases, ["iteration", "split", "model"]
            ),
        ], ignore_index=True, sort=False)
        split_metrics.to_csv(output_dir / "selected_formula_split_metrics.csv", index=False)
        tail_rows = [
            {"split": split, "model": model, **_tail_adaptation(group)}
            for (split, model), group in pd.concat(
                [baseline_cases, selected_cases], ignore_index=True
            ).groupby(["split", "model"])
        ]
        pd.DataFrame(tail_rows).to_csv(
            output_dir / "selected_formula_tail_adaptation.csv", index=False
        )
        formula_record = _save_formula_artifacts(preflight, selected)
        _save_pilot_plots(split_metrics, selected_cases, learnability, output_dir)
        progress = _write_progress(output_dir, config)
        payload = {
            "status": "complete",
            "iteration": config.iteration,
            "prototype_only": True,
            "method": "fixed_v8_global_shape_plus_one_symbolic_residual_formula",
            "train_cases": len(preflight["inputs"]["train_ids"]),
            "validation_cases": len(preflight["inputs"]["validation_ids"]),
            "internal_test_cases": len(preflight["inputs"]["internal_ids"]),
            "final_test_cases_read": 0,
            "training_rows": cache["metadata"]["n_rows"],
            "planned_population_iterations": config.target_population_iterations,
            "completed_segments": progress["contiguous_completed_segments"],
            "selected_residual_candidate": selected_index,
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
                "Rerun the same notebook. Training cache and completed PySR "
                "segments are reused."
            ),
            "final_test_cases_read": 0,
        })
        raise
    finally:
        _stop_sleep_inhibitor(sleep_inhibitor)
