"""Robust, restartable Iteration-1 hierarchical symbolic regression.

This module replaces one long element-shape PySR process with six bounded
worker processes.  Each worker writes its own Hall of Fame, and the parent
process enforces an external timeout so a stalled Julia process cannot consume
the whole experiment budget.

The final expression remains a single symbolic formula:

    stress = case_mean + exp(case_log_scale) * (shape_1 + shape_2 + shape_3)

The three shape terms are fitted sequentially to residuals.  Formula selection
uses complete validation cases; internal-test cases are evaluated only after
all three components have been fixed.  Locked final-test cases are never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import sympy as sp


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
    MODEL_FEATURES,
    STANDARDISED_LOCAL_FEATURES,
    _case_summary_row,
    assemble_training_sample,
    build_model_matrix,
    load_inputs,
)
from hierarchical_symbolic_regression import (  # noqa: E402
    CASE_FORMULA_FEATURES,
    SHAPE_FORMULA_FEATURES,
    _add_composite_selection_score,
    _compiled_formula,
    _evaluate_shape_candidates,
    _formula_exports,
    _load_scaling,
    _reference_metrics,
    _save_scaling,
    _scaling_payload,
    _shortlist_shape_frontier,
    _tail_adaptation,
)


BASE_LOCAL_AND_STANDARDISED = LOCAL_FEATURES + STANDARDISED_LOCAL_FEATURES

DERIVED_FEATURE_DEFINITIONS = {
    "rho_within_case_z_square": "rho_within_case_z ** 2",
    "z_within_case_z_square": "z_within_case_z ** 2",
    "rho_within_case_z_theta_sin": "rho_within_case_z * theta_sin",
    "rho_within_case_z_theta_cos": "rho_within_case_z * theta_cos",
    "z_within_case_z_theta_sin": "z_within_case_z * theta_sin",
    "z_within_case_z_theta_cos": "z_within_case_z * theta_cos",
    "fluence_rate_within_case_z_rho_within_case_z": (
        "fluence_rate_within_case_z * rho_within_case_z"
    ),
    "temperature_within_case_z_rho_within_case_z": (
        "temperature_within_case_z * rho_within_case_z"
    ),
    "weight_loss_rate_within_case_z_rho_within_case_z": (
        "weight_loss_rate_within_case_z * rho_within_case_z"
    ),
}

ROBUST_FEATURES = BASE_LOCAL_AND_STANDARDISED + list(DERIVED_FEATURE_DEFINITIONS)

GEOMETRY_COMPONENT_FEATURES = [
    "rho",
    "theta_sin",
    "theta_cos",
    "z",
    "rho_within_case_z",
    "theta_sin_within_case_z",
    "theta_cos_within_case_z",
    "z_within_case_z",
    "rho_within_case_z_square",
    "z_within_case_z_square",
    "rho_within_case_z_theta_sin",
    "rho_within_case_z_theta_cos",
    "z_within_case_z_theta_sin",
    "z_within_case_z_theta_cos",
]

PHYSICAL_RESIDUAL_FEATURES = [
    "fluence_rate",
    "temperature",
    "weight_loss_rate",
    "fluence_rate_within_case_z",
    "temperature_within_case_z",
    "weight_loss_rate_within_case_z",
    "rho_within_case_z",
    "z_within_case_z",
    "theta_sin",
    "theta_cos",
    "fluence_rate_within_case_z_rho_within_case_z",
    "temperature_within_case_z_rho_within_case_z",
    "weight_loss_rate_within_case_z_rho_within_case_z",
]

INTERACTION_RESIDUAL_FEATURES = BASE_LOCAL_AND_STANDARDISED + [
    "fluence_rate_within_case_z_rho_within_case_z",
    "temperature_within_case_z_rho_within_case_z",
    "weight_loss_rate_within_case_z_rho_within_case_z",
]

COMPONENT_FEATURES = {
    1: GEOMETRY_COMPONENT_FEATURES,
    2: PHYSICAL_RESIDUAL_FEATURES,
    3: INTERACTION_RESIDUAL_FEATURES,
}


@dataclass(frozen=True)
class RobustChunkedConfig:
    """Fixed settings for the new bounded Iteration-1 prototype."""

    iteration: int = 1
    output_subdir: str = "iteration_1"
    rows_per_training_case: int = 5_000
    components: int = 3
    chunks_per_component: int = 2
    niterations_per_chunk: int = 250
    populations: int = 4
    population_size: int = 30
    ncycles_per_iteration: int = 50
    batch_size: int = 20_000
    maxsize: int = 16
    maxdepth: int = 7
    julia_threads: int = 8
    worker_internal_timeout_seconds: int = 75 * 60
    worker_external_timeout_seconds: int = 90 * 60
    worker_poll_seconds: int = 30
    max_candidates_per_component: int = 12
    run_old_partial_audit: bool = True
    force_rebuild_training_cache: bool = False
    force_rerun_chunks: bool = False

    def validate(self) -> None:
        if self.iteration != 1:
            raise ValueError("This notebook is intentionally locked to Iteration 1")
        if self.rows_per_training_case != 5_000:
            raise ValueError("The discovery design is locked to 5,000 rows per case")
        if self.components != 3 or self.chunks_per_component != 2:
            raise ValueError("The robust design requires three components and two chunks each")
        if self.batch_size < 1 or self.batch_size > 100_000:
            raise ValueError("batch_size is outside the reviewed range")
        if self.worker_external_timeout_seconds <= self.worker_internal_timeout_seconds:
            raise ValueError("External timeout must exceed the PySR internal timeout")
        if self.julia_threads < 1:
            raise ValueError("julia_threads must be positive")


def package_output_dir(package_root: Path, config: RobustChunkedConfig) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "07_robust_chunked_hierarchical_symbolic"
        / config.output_subdir
    )


def old_iteration_dir(package_root: Path) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "06_hierarchical_symbolic_regression"
        / "iteration_1"
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _formula_expression(text: str) -> sp.Expr:
    locals_map = {
        "square": lambda value: value**2,
        "cube": lambda value: value**3,
        "abs": sp.Abs,
        "Abs": sp.Abs,
    }
    return sp.sympify(str(text), locals=locals_map)


def _unscale_expression(expression: sp.Expr, features: Sequence[str], scaling: dict) -> sp.Expr:
    replacements = {
        sp.Symbol(f"{feature}_scaled"): (
            sp.Symbol(feature) - sp.Float(scaling["x_mean"][index])
        ) / sp.Float(scaling["x_std"][index])
        for index, feature in enumerate(features)
    }
    return (
        sp.Float(scaling["y_mean"])
        + sp.Float(scaling["y_std"]) * expression.xreplace(replacements)
    )


def frontier_from_hall_of_fame(
    hall_path: Path,
    features: Sequence[str],
    scaling: dict,
    stage: str,
    run_id: str,
) -> pd.DataFrame:
    """Convert a raw PySR Hall of Fame into the project's canonical frontier."""

    raw = pd.read_csv(hall_path)
    # PySR writes raw Hall-of-Fame headers as Complexity/Loss/Equation, while
    # model.equations_ uses lower-case names. Recovery accepts both forms.
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"complexity", "loss", "equation"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{hall_path} is missing PySR columns: {missing}")
    records = []
    for candidate_index, row in raw.reset_index(drop=True).iterrows():
        expression = _formula_expression(row["equation"])
        original = _unscale_expression(expression, features, scaling)
        constant_free = re.sub(r"Float\([^\)]*\)", "CONST", sp.srepr(sp.factor(expression)))
        operators = sorted({
            node.func.__name__
            for node in sp.preorder_traversal(expression)
            if getattr(node, "args", ())
        })
        support = sorted(str(symbol) for symbol in expression.free_symbols)
        records.append({
            "stage": stage,
            "run_id": run_id,
            "candidate_index": int(candidate_index),
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "score": float(row.get("score", np.nan)),
            "equation": str(row["equation"]),
            "formula_scaled_sympy": str(expression),
            "formula_original_variables": str(original),
            "feature_support_json": json.dumps(support),
            "structure_signature": constant_free,
            "family_signature": json.dumps({
                "features": support,
                "operators": operators,
            }, sort_keys=True),
        })
    if not records:
        raise RuntimeError(f"No candidates were recovered from {hall_path}")
    return pd.DataFrame(records).sort_values(["complexity", "loss"]).reset_index(drop=True)


def _derived_feature_arrays(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rho_z = base["rho_within_case_z"]
    z_z = base["z_within_case_z"]
    return {
        "rho_within_case_z_square": rho_z**2,
        "z_within_case_z_square": z_z**2,
        "rho_within_case_z_theta_sin": rho_z * base["theta_sin"],
        "rho_within_case_z_theta_cos": rho_z * base["theta_cos"],
        "z_within_case_z_theta_sin": z_z * base["theta_sin"],
        "z_within_case_z_theta_cos": z_z * base["theta_cos"],
        "fluence_rate_within_case_z_rho_within_case_z": (
            base["fluence_rate_within_case_z"] * rho_z
        ),
        "temperature_within_case_z_rho_within_case_z": (
            base["temperature_within_case_z"] * rho_z
        ),
        "weight_loss_rate_within_case_z_rho_within_case_z": (
            base["weight_loss_rate_within_case_z"] * rho_z
        ),
    }


def build_robust_feature_matrix(model_matrix: np.ndarray) -> np.ndarray:
    indices = {feature: MODEL_FEATURES.index(feature) for feature in BASE_LOCAL_AND_STANDARDISED}
    base = {
        feature: np.asarray(model_matrix[:, index], dtype=np.float32)
        for feature, index in indices.items()
    }
    values = {**base, **_derived_feature_arrays(base)}
    return np.ascontiguousarray(
        np.column_stack([values[feature] for feature in ROBUST_FEATURES]),
        dtype=np.float32,
    )


def select_robust_features(matrix: np.ndarray, features: Sequence[str]) -> np.ndarray:
    indices = [ROBUST_FEATURES.index(feature) for feature in features]
    return np.ascontiguousarray(matrix[:, indices], dtype=np.float32)


def _feature_registry() -> pd.DataFrame:
    rows = []
    for feature in BASE_LOCAL_AND_STANDARDISED:
        rows.append({
            "feature": feature,
            "kind": "base_or_within_case_standardised",
            "definition": feature,
            "component_1": feature in COMPONENT_FEATURES[1],
            "component_2": feature in COMPONENT_FEATURES[2],
            "component_3": feature in COMPONENT_FEATURES[3],
        })
    for feature, definition in DERIVED_FEATURE_DEFINITIONS.items():
        rows.append({
            "feature": feature,
            "kind": "fixed_predictor_only_interaction",
            "definition": definition,
            "component_1": feature in COMPONENT_FEATURES[1],
            "component_2": feature in COMPONENT_FEATURES[2],
            "component_3": feature in COMPONENT_FEATURES[3],
        })
    return pd.DataFrame(rows)


def load_reused_case_formulas(package_root: Path) -> dict:
    source = old_iteration_dir(package_root)
    paths = {
        "mean_row": source / "selected_case_mean_formula.csv",
        "mean_scaling": source / "case_mean_scaling.csv",
        "scale_row": source / "selected_case_log_scale_formula.csv",
        "scale_scaling": source / "case_log_scale_scaling.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing completed case-level formula files: {missing}")
    return {
        "mean_row": pd.read_csv(paths["mean_row"]).iloc[0],
        "mean_scaling": _load_scaling(paths["mean_scaling"]),
        "scale_row": pd.read_csv(paths["scale_row"]).iloc[0],
        "scale_scaling": _load_scaling(paths["scale_scaling"]),
        "source_paths": paths,
    }


def preflight_robust_iteration(package_root: Path, config: RobustChunkedConfig) -> dict:
    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = package_output_dir(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    feasibility_config = FeasibilityConfig(
        iteration=config.iteration,
        rows_per_case=config.rows_per_training_case,
        output_subdir="iteration_1_pilot",
    )
    inputs = load_inputs(package_root, feasibility_config)
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    reused = load_reused_case_formulas(package_root)
    worker_path = package_root / "scripts" / "run_robust_shape_chunk.py"
    checks = [
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "worker_script_exists", "value": worker_path.exists(), "expected": True},
        {"check": "mean_formula_reusable", "value": True, "expected": True},
        {"check": "log_scale_formula_reusable", "value": True, "expected": True},
    ]
    table = pd.DataFrame(checks)
    table["pass"] = table["value"] == table["expected"]
    table.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not table["pass"].all():
        raise RuntimeError("Robust Iteration-1 preflight failed")
    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "feasibility_config": feasibility_config,
        "reused": reused,
        "checks": table,
        "worker_path": worker_path,
    }


def prepare_training_cache(preflight: dict, config: RobustChunkedConfig) -> dict:
    output_dir = preflight["output_dir"]
    cache_dir = output_dir / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "X": cache_dir / "robust_feature_pool.npy",
        "y_shape": cache_dir / "shape_target.npy",
        "weights": cache_dir / "discovery_weights.npy",
        "metadata": cache_dir / "cache_metadata.json",
    }
    can_reuse = all(path.exists() for path in paths.values()) and not config.force_rebuild_training_cache
    if can_reuse:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        expected_rows = len(preflight["inputs"]["train_ids"]) * config.rows_per_training_case
        if (
            metadata.get("n_rows") == expected_rows
            and metadata.get("features") == ROBUST_FEATURES
            and metadata.get("train_ids") == preflight["inputs"]["train_ids"]
        ):
            return {
                "X": np.load(paths["X"], mmap_mode="r"),
                "y_shape": np.load(paths["y_shape"], mmap_mode="r"),
                "weights": np.load(paths["weights"], mmap_mode="r"),
                "paths": paths,
                "metadata": metadata,
                "reused": True,
            }

    sample = assemble_training_sample(preflight["inputs"], preflight["feasibility_config"])
    robust = build_robust_feature_matrix(sample["X"])
    np.save(paths["X"], robust)
    np.save(paths["y_shape"], np.asarray(sample["y_shape"], dtype=np.float32))
    np.save(paths["weights"], np.asarray(sample["weights"], dtype=np.float32))
    sample["audit"].to_csv(cache_dir / "training_sample_audit.csv", index=False)
    sample["manifest"].to_csv(cache_dir / "training_sample_manifest.csv.gz", index=False)
    metadata = {
        "n_rows": int(len(robust)),
        "n_features": int(robust.shape[1]),
        "features": ROBUST_FEATURES,
        "train_ids": preflight["inputs"]["train_ids"],
        "rows_per_case": config.rows_per_training_case,
        "sampling": "fixed V4 joint-field and stress-tail stratified discovery design",
        "full_validation_policy": "all elements in every validation case",
    }
    _atomic_json(paths["metadata"], metadata)
    del sample, robust
    gc.collect()
    return {
        "X": np.load(paths["X"], mmap_mode="r"),
        "y_shape": np.load(paths["y_shape"], mmap_mode="r"),
        "weights": np.load(paths["weights"], mmap_mode="r"),
        "paths": paths,
        "metadata": metadata,
        "reused": False,
    }


def audit_old_partial_shape(preflight: dict, config: RobustChunkedConfig) -> pd.DataFrame | None:
    if not config.run_old_partial_audit:
        return None
    output_path = preflight["output_dir"] / "old_partial_shape_validation_audit.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    old_dir = old_iteration_dir(preflight["package_root"])
    hall_candidates = sorted((old_dir / "pysr_runs").glob("hierarchical_i1_shape_*/hall_of_fame.csv"))
    if not hall_candidates:
        return None
    old_scaling = _load_scaling(old_dir / "shape_scaling.csv")
    frontier = frontier_from_hall_of_fame(
        hall_candidates[-1],
        SHAPE_FORMULA_FEATURES,
        old_scaling,
        "old_partial_shape_diagnostic",
        hall_candidates[-1].parent.name,
    )
    metrics, case_metrics = _evaluate_shape_candidates(
        candidates=frontier,
        shape_scaling=old_scaling,
        mean_row=preflight["reused"]["mean_row"],
        mean_scaling=preflight["reused"]["mean_scaling"],
        scale_row=preflight["reused"]["scale_row"],
        scale_scaling=preflight["reused"]["scale_scaling"],
        inputs=preflight["inputs"],
        case_ids=preflight["inputs"]["validation_ids"],
        split="validation",
        iteration=config.iteration,
    )
    valid = metrics[metrics["candidate_valid"].fillna(False)].copy()
    reference = _reference_metrics(preflight["inputs"], config.iteration, "validation")
    scored = _add_composite_selection_score(valid, reference)
    scored["formal_reuse_allowed"] = False
    scored["diagnostic_conclusion"] = "partial stalled search; all candidates require replacement"
    scored.to_csv(output_path, index=False)
    case_metrics.to_csv(
        preflight["output_dir"] / "old_partial_shape_validation_case_metrics.csv",
        index=False,
    )
    return scored


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
        "platform": platform.system(),
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
        else:  # pragma: no cover - Windows fallback
            process.terminate()
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                process.kill()
            process.wait(timeout=10)


def _chunk_seed(component: int, chunk: int) -> int:
    return int(RANDOM_SEED + 1_000 + component * 100 + chunk)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_worker_chunk(
    preflight: dict,
    config: RobustChunkedConfig,
    cache: dict,
    component: int,
    chunk: int,
    target_path: Path,
    scaling_path: Path,
) -> pd.DataFrame:
    component_dir = preflight["output_dir"] / f"component_{component}"
    chunk_dir = component_dir / f"chunk_{chunk}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = chunk_dir / "frontier.csv"
    run_id = f"robust_i1_component_{component}_chunk_{chunk}"
    if config.force_rerun_chunks:
        run_id = f"{run_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    hall_path = chunk_dir / "pysr_runs" / run_id / "hall_of_fame.csv"
    features = COMPONENT_FEATURES[component]
    scaling = _load_scaling(scaling_path)
    signature_path = chunk_dir / "input_signature.json"
    expected_signature = {
        "target_sha256": _sha256_file(target_path),
        "scaling_sha256": _sha256_file(scaling_path),
        "features": list(features),
        "seed": _chunk_seed(component, chunk),
        "niterations": config.niterations_per_chunk,
        "populations": config.populations,
        "population_size": config.population_size,
        "ncycles_per_iteration": config.ncycles_per_iteration,
        "batch_size": config.batch_size,
        "maxsize": config.maxsize,
        "maxdepth": config.maxdepth,
    }
    if signature_path.exists() and not config.force_rerun_chunks:
        existing_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing_signature != expected_signature:
            raise RuntimeError(
                f"Component {component}, chunk {chunk} cache was produced from a "
                "different residual target or search configuration. Use a new "
                "output_subdir for the changed experiment."
            )
    elif not signature_path.exists() and (canonical_path.exists() or hall_path.exists()):
        raise RuntimeError(
            f"Component {component}, chunk {chunk} has cached search files but no "
            "input signature, so safe reuse cannot be verified."
        )
    _atomic_json(signature_path, expected_signature)

    resume_run_directory: Path | None = None
    bounded_completion_states = {
        "normal_completion_or_internal_timeout",
        "external_timeout_partial_hall_of_fame",
    }
    if not config.force_rerun_chunks:
        if canonical_path.exists():
            existing_frontier = pd.read_csv(canonical_path)
            completion_states = set(
                existing_frontier.get(
                    "worker_completion", pd.Series(dtype=str)
                ).dropna().astype(str)
            )
            if completion_states & bounded_completion_states:
                print(
                    f"Component {component}, chunk {chunk}: completed frontier reused",
                    flush=True,
                )
                _atomic_json(chunk_dir / "parent_worker_result.json", {
                    "component": component,
                    "chunk": chunk,
                    "run_id": run_id,
                    "status": "completed_frontier_reused",
                    "completion_states": sorted(completion_states),
                })
                return existing_frontier
            run_directory = hall_path.parent
            if (run_directory / "checkpoint.pkl").exists():
                resume_run_directory = run_directory
                print(
                    f"Component {component}, chunk {chunk}: incomplete frontier "
                    "found; resuming from checkpoint.pkl",
                    flush=True,
                )
            else:
                print(
                    f"Component {component}, chunk {chunk}: incomplete frontier "
                    "has no checkpoint; retained as a partial bounded result",
                    flush=True,
                )
                return existing_frontier
        if hall_path.exists() and hall_path.stat().st_size > 40:
            run_directory = hall_path.parent
            if resume_run_directory is None and (run_directory / "checkpoint.pkl").exists():
                resume_run_directory = run_directory
                print(
                    f"Component {component}, chunk {chunk}: interrupted Hall of Fame "
                    "found; resuming from checkpoint.pkl",
                    flush=True,
                )
            elif resume_run_directory is None:
                print(
                    f"Component {component}, chunk {chunk}: partial Hall of Fame "
                    "reused because no checkpoint is available",
                    flush=True,
                )
                frontier = frontier_from_hall_of_fame(
                    hall_path, features, scaling, f"component_{component}", run_id
                )
                frontier["component"] = component
                frontier["chunk"] = chunk
                frontier["seed"] = _chunk_seed(component, chunk)
                frontier["worker_completion"] = "recovered_partial_hall_of_fame"
                frontier.to_csv(canonical_path, index=False)
                return frontier

    command = [
        sys.executable,
        str(preflight["worker_path"]),
        "--feature-pool", str(cache["paths"]["X"]),
        "--feature-registry", json.dumps(ROBUST_FEATURES),
        "--target", str(target_path),
        "--weights", str(cache["paths"]["weights"]),
        "--scaling", str(scaling_path),
        "--features", json.dumps(features),
        "--component", str(component),
        "--chunk", str(chunk),
        "--seed", str(_chunk_seed(component, chunk)),
        "--output-dir", str(chunk_dir),
        "--run-id", run_id,
        "--niterations", str(config.niterations_per_chunk),
        "--populations", str(config.populations),
        "--population-size", str(config.population_size),
        "--ncycles", str(config.ncycles_per_iteration),
        "--batch-size", str(config.batch_size),
        "--maxsize", str(config.maxsize),
        "--maxdepth", str(config.maxdepth),
        "--internal-timeout", str(config.worker_internal_timeout_seconds),
    ]
    if resume_run_directory is not None:
        command.extend(["--resume-run-directory", str(resume_run_directory)])
    environment = os.environ.copy()
    environment["JULIA_NUM_THREADS"] = str(config.julia_threads)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(preflight["package_root"] / "src"),
        environment.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    stdout_path = chunk_dir / "worker_stdout.log"
    stderr_path = chunk_dir / "worker_stderr.log"
    started = time.time()
    log_mode = "a" if resume_run_directory is not None else "w"
    with stdout_path.open(log_mode, encoding="utf-8") as stdout, stderr_path.open(
        log_mode, encoding="utf-8"
    ) as stderr:
        if resume_run_directory is not None:
            marker = (
                f"\n\n===== RESUME {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"FROM {resume_run_directory} =====\n"
            )
            stdout.write(marker)
            stdout.flush()
            stderr.write(marker)
            stderr.flush()
        process = subprocess.Popen(
            command,
            cwd=preflight["package_root"],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=(os.name == "posix"),
        )
        last_message = 0.0
        timed_out = False
        while process.poll() is None:
            elapsed = time.time() - started
            if elapsed >= config.worker_external_timeout_seconds:
                timed_out = True
                _terminate_process_tree(process)
                break
            if elapsed - last_message >= 300 or last_message == 0:
                print(
                    f"Component {component}, chunk {chunk}: worker running "
                    f"for {elapsed / 60:.1f} min; Hall of Fame exists={hall_path.exists()}",
                    flush=True,
                )
                last_message = elapsed
            _atomic_json(chunk_dir / "parent_watchdog_status.json", {
                "component": component,
                "chunk": chunk,
                "pid": process.pid,
                "elapsed_seconds": elapsed,
                "external_timeout_seconds": config.worker_external_timeout_seconds,
                "hall_of_fame_exists": hall_path.exists(),
                "hall_of_fame_size_bytes": hall_path.stat().st_size if hall_path.exists() else 0,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            time.sleep(config.worker_poll_seconds)
        return_code = process.poll()

    status = {
        "component": component,
        "chunk": chunk,
        "run_id": run_id,
        "return_code": return_code,
        "timed_out_by_parent": timed_out,
        "elapsed_seconds": time.time() - started,
        "canonical_frontier_exists": canonical_path.exists(),
        "hall_of_fame_exists": hall_path.exists(),
        "resumed_from_checkpoint": resume_run_directory is not None,
    }
    _atomic_json(chunk_dir / "parent_worker_result.json", status)
    if return_code == 0 and canonical_path.exists():
        return pd.read_csv(canonical_path)
    if hall_path.exists() and hall_path.stat().st_size > 40:
        frontier = frontier_from_hall_of_fame(
            hall_path, features, scaling, f"component_{component}", run_id
        )
        frontier["component"] = component
        frontier["chunk"] = chunk
        frontier["seed"] = _chunk_seed(component, chunk)
        frontier["worker_completion"] = (
            "external_timeout_partial_hall_of_fame" if timed_out
            else "abnormal_exit_partial_hall_of_fame"
        )
        frontier.to_csv(canonical_path, index=False)
        return frontier
    raise RuntimeError(
        f"Component {component}, chunk {chunk} produced no recoverable candidates. "
        f"Inspect {stderr_path}."
    )


def _zero_residual_candidate(target: np.ndarray, weights: np.ndarray, scaling: dict) -> dict:
    scaled_constant = -float(scaling["y_mean"]) / float(scaling["y_std"])
    y_scaled = (np.asarray(target, dtype=np.float64) - scaling["y_mean"]) / scaling["y_std"]
    normalised_weights = np.asarray(weights, dtype=np.float64)
    normalised_weights /= normalised_weights.sum()
    loss = float(np.sum(normalised_weights * (y_scaled - scaled_constant) ** 2))
    return {
        "stage": "zero_residual_guardrail",
        "run_id": "deterministic_zero_candidate",
        "candidate_index": -1,
        "complexity": 1,
        "loss": loss,
        "score": 0.0,
        "equation": str(scaled_constant),
        "formula_scaled_sympy": str(sp.Float(scaled_constant)),
        "formula_original_variables": "0.0",
        "feature_support_json": "[]",
        "structure_signature": "ZERO_RESIDUAL",
        "family_signature": json.dumps({"features": [], "operators": []}),
        "component": np.nan,
        "chunk": 0,
        "seed": RANDOM_SEED,
        "worker_completion": "deterministic_guardrail",
    }


def merge_component_frontiers(
    frontiers: Sequence[pd.DataFrame],
    target: np.ndarray,
    weights: np.ndarray,
    scaling: dict,
    component: int,
    limit: int,
) -> pd.DataFrame:
    merged = pd.concat(frontiers, ignore_index=True, sort=False)
    merged = merged.sort_values(["loss", "complexity"]).drop_duplicates(
        "formula_scaled_sympy", keep="first"
    )
    merged = _shortlist_shape_frontier(merged, limit)
    zero = _zero_residual_candidate(target, weights, scaling)
    zero["component"] = component
    merged = pd.concat([merged, pd.DataFrame([zero])], ignore_index=True, sort=False)
    merged = merged.sort_values(["complexity", "loss"]).reset_index(drop=True)
    merged["source_candidate_index"] = merged["candidate_index"]
    merged["candidate_index"] = np.arange(len(merged), dtype=int)
    return merged


def _component_function(record: dict) -> Callable[[np.ndarray], np.ndarray]:
    return _compiled_formula(pd.Series(record["row"]), record["scaling"])


def _component_matrix(robust_matrix: np.ndarray, features: Sequence[str]) -> np.ndarray:
    return select_robust_features(robust_matrix, features)


def evaluate_component_candidates(
    *,
    candidates: pd.DataFrame,
    candidate_scaling: dict,
    candidate_features: Sequence[str],
    accumulated: Sequence[dict],
    preflight: dict,
    case_ids: Sequence[str],
    split: str,
    component: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_functions = {
        int(row.candidate_index): _compiled_formula(row, candidate_scaling)
        for row in candidates.itertuples(index=False)
    }
    accumulated_functions = [
        (record, _component_function(record)) for record in accumulated
    ]
    reused = preflight["reused"]
    mean_function = _compiled_formula(reused["mean_row"], reused["mean_scaling"])
    scale_function = _compiled_formula(reused["scale_row"], reused["scale_scaling"])
    metric_rows = []
    invalid: dict[int, str] = {}

    for position, case_id in enumerate(case_ids, start=1):
        print(
            f"[{position}/{len(case_ids)}] {split} complete-case evaluation: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float64)[None, :]
        predicted_mean = float(mean_function(context)[0])
        predicted_log_scale = float(scale_function(context)[0])
        if not np.isfinite(predicted_log_scale) or abs(predicted_log_scale) > 20.0:
            raise FloatingPointError(f"{case_id}: invalid case log-scale prediction")
        predicted_scale = float(np.exp(predicted_log_scale))
        model_matrix = build_model_matrix(frame, summary)
        robust_matrix = build_robust_feature_matrix(model_matrix)
        base_shape = np.zeros(len(frame), dtype=np.float64)
        for record, function in accumulated_functions:
            component_values = function(_component_matrix(robust_matrix, record["features"]))
            base_shape += component_values
        candidate_matrix = _component_matrix(robust_matrix, candidate_features)

        for candidate_index, function in candidate_functions.items():
            if candidate_index in invalid:
                continue
            try:
                correction = function(candidate_matrix)
                predicted = predicted_mean + predicted_scale * (base_shape + correction)
                if not np.isfinite(predicted).all():
                    raise FloatingPointError("non-finite composite prediction")
                metric_rows.append({
                    "iteration": 1,
                    "component": component,
                    "split": split,
                    "model": "robust_chunked_hierarchical_symbolic",
                    "candidate_index": candidate_index,
                    "case_id": case_id,
                    "predicted_case_mean": predicted_mean,
                    "predicted_case_scale": predicted_scale,
                    **evaluate_prediction_arrays(actual, predicted),
                })
            except Exception as exc:
                invalid[candidate_index] = repr(exc)
        del frame, actual, model_matrix, robust_matrix, candidate_matrix, base_shape
        gc.collect()

    case_metrics = pd.DataFrame(metric_rows)
    records = []
    accumulated_complexity = int(sum(record["row"]["complexity"] for record in accumulated))
    for candidate in candidates.itertuples(index=False):
        index = int(candidate.candidate_index)
        group = case_metrics[case_metrics["candidate_index"] == index]
        base = candidate._asdict()
        base["cumulative_shape_complexity"] = accumulated_complexity + int(candidate.complexity)
        if index in invalid or group["case_id"].nunique() != len(case_ids):
            records.append({
                **base,
                "candidate_valid": False,
                "invalid_reason": invalid.get(index, "incomplete_case_evaluation"),
            })
            continue
        aggregate = aggregate_case_metrics(group, ["candidate_index"]).iloc[0].to_dict()
        aggregate.pop("candidate_index", None)
        adaptation = _tail_adaptation(group)
        records.append({
            **base,
            "candidate_valid": True,
            "invalid_reason": "",
            **{f"{split}_{key}": value for key, value in aggregate.items()},
            **{f"{split}_{key}": value for key, value in adaptation.items()},
        })
    return pd.DataFrame(records), case_metrics


def select_component_candidate(
    candidate_metrics: pd.DataFrame,
    reference: pd.Series,
    component_dir: Path,
) -> pd.Series:
    valid = candidate_metrics[candidate_metrics["candidate_valid"].fillna(False)].copy()
    if valid.empty:
        raise RuntimeError("No finite complete-validation candidate remains")
    scored = _add_composite_selection_score(valid, reference)
    numerical = scored[scored["gate_numerical_guardrail"]].copy()
    if numerical.empty:
        raise RuntimeError("Every candidate failed the numerical guardrail")
    accepted = numerical[numerical["all_provisional_acceptance_gates_pass"]]
    pool = accepted if not accepted.empty else numerical
    pool = pool[
        np.isfinite(pool["reference_relative_engineering_score"])
        & np.isfinite(pool["validation_macro_rmse"])
    ].copy()
    if pool.empty:
        raise RuntimeError("Every numerically valid candidate has a non-finite selection score")

    # The engineering score already combines RMSE, p95/p99 errors,
    # underprediction and hotspot metrics.  Comparing against the independent
    # global minimum of every objective can create an empty intersection when
    # different candidates optimise different objectives.  First retain
    # candidates within 5% of the best composite score, then apply the RMSE
    # tolerance relative to the best RMSE inside that engineering-competitive
    # set.  This is a lexicographic engineering decision, not an accidental
    # intersection of two unrelated global minima.
    best_score = float(pool["reference_relative_engineering_score"].min())
    score_tolerance = 0.05 * max(abs(best_score), 1e-12)
    engineering_competitive = pool[
        pool["reference_relative_engineering_score"] <= best_score + score_tolerance
    ].copy()
    engineering_rmse_anchor = float(
        engineering_competitive["validation_macro_rmse"].min()
    )
    competitive = engineering_competitive[
        engineering_competitive["validation_macro_rmse"]
        <= 1.10 * engineering_rmse_anchor
    ].copy()
    if competitive.empty:
        # Defensive fallback; the score-minimising row should always satisfy
        # the two filters above, but an explicit fallback prevents another
        # opaque iloc failure if future metrics contain unusual values.
        competitive = pool.sort_values([
            "reference_relative_engineering_score",
            "validation_macro_rmse",
            "cumulative_shape_complexity",
            "candidate_index",
        ]).head(1)
    selected = competitive.sort_values([
        "cumulative_shape_complexity",
        "reference_relative_engineering_score",
        "validation_macro_rmse",
        "candidate_index",
    ]).iloc[0].copy()
    scored["selected_candidate"] = scored["candidate_index"].eq(selected["candidate_index"])
    scored["selection_pool_status"] = (
        "all_gates_pass" if not accepted.empty
        else "provisional_no_candidate_passed_all_gates"
    )
    scored["selection_method"] = (
        "composite_engineering_score_within_5pct_then_rmse_within_10pct_"
        "of_engineering_competitive_anchor_then_minimum_complexity"
    )
    scored.to_csv(component_dir / "candidate_validation_metrics.csv", index=False)
    selected["selection_pool_status"] = scored["selection_pool_status"].iloc[0]
    selected["selection_method"] = scored["selection_method"].iloc[0]
    pd.DataFrame([selected]).to_csv(component_dir / "selected_component_formula.csv", index=False)
    return selected


def _training_component_prediction(
    robust_matrix: np.ndarray,
    row: pd.Series,
    scaling: dict,
    features: Sequence[str],
) -> np.ndarray:
    function = _compiled_formula(row, scaling)
    return function(_component_matrix(robust_matrix, features))


def _component_checkpoint_signature(
    component: int,
    target_path: Path,
    scaling_path: Path,
    features: Sequence[str],
    config: RobustChunkedConfig,
) -> dict:
    """Identify every input that makes a selected component reusable."""

    return {
        "component": component,
        "target_sha256": _sha256_file(target_path),
        "scaling_sha256": _sha256_file(scaling_path),
        "features": list(features),
        "chunks_per_component": config.chunks_per_component,
        "niterations_per_chunk": config.niterations_per_chunk,
        "populations": config.populations,
        "population_size": config.population_size,
        "ncycles_per_iteration": config.ncycles_per_iteration,
        "batch_size": config.batch_size,
        "maxsize": config.maxsize,
        "maxdepth": config.maxdepth,
    }


def evaluate_fixed_composite(
    preflight: dict,
    components: Sequence[dict],
    case_ids: Sequence[str],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reused = preflight["reused"]
    mean_function = _compiled_formula(reused["mean_row"], reused["mean_scaling"])
    scale_function = _compiled_formula(reused["scale_row"], reused["scale_scaling"])
    compiled = [(record, _component_function(record)) for record in components]
    rows = []
    for position, case_id in enumerate(case_ids, start=1):
        print(f"[{position}/{len(case_ids)}] Frozen composite {split}: {case_id}", flush=True)
        frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float64)[None, :]
        predicted_mean = float(mean_function(context)[0])
        predicted_scale = float(np.exp(scale_function(context)[0]))
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        robust = build_robust_feature_matrix(build_model_matrix(frame, summary))
        shape = np.zeros(len(frame), dtype=np.float64)
        for record, function in compiled:
            shape += function(_component_matrix(robust, record["features"]))
        predicted = predicted_mean + predicted_scale * shape
        if not np.isfinite(predicted).all():
            raise FloatingPointError(f"{case_id}: frozen composite produced non-finite values")
        rows.append({
            "iteration": 1,
            "split": split,
            "model": "robust_chunked_hierarchical_symbolic",
            "case_id": case_id,
            "predicted_case_mean": predicted_mean,
            "predicted_case_scale": predicted_scale,
            **evaluate_prediction_arrays(actual, predicted),
        })
        del frame, actual, robust, shape, predicted
        gc.collect()
    case_metrics = pd.DataFrame(rows)
    aggregate = aggregate_case_metrics(case_metrics, ["model"]).iloc[0].to_dict()
    aggregate.update(_tail_adaptation(case_metrics))
    aggregate["split"] = split
    return pd.DataFrame([aggregate]), case_metrics


def _formula_record(preflight: dict, components: Sequence[dict]) -> dict:
    mean_formula = str(preflight["reused"]["mean_row"]["formula_original_variables"])
    scale_formula = str(preflight["reused"]["scale_row"]["formula_original_variables"])
    shape_formulas = [str(record["row"]["formula_original_variables"]) for record in components]
    shape_sum = " + ".join(f"({formula})" for formula in shape_formulas)
    composite = f"({mean_formula}) + exp({scale_formula}) * ({shape_sum})"
    return {
        "iteration": 1,
        "decomposition": "stress = case_mean + exp(case_log_scale) * (shape_1 + shape_2 + shape_3)",
        "case_mean_formula": mean_formula,
        "case_log_scale_formula": scale_formula,
        "shape_component_1": shape_formulas[0],
        "shape_component_2": shape_formulas[1],
        "shape_component_3": shape_formulas[2],
        "shape_formula": shape_sum,
        "composite_formula": composite,
        "shape_component_complexities_json": json.dumps([
            int(record["row"]["complexity"]) for record in components
        ]),
        "total_shape_complexity": int(sum(record["row"]["complexity"] for record in components)),
        "status": "Iteration-1 prototype; not the final four-rotation formula",
    }


def _save_formula_text(record: dict, output_dir: Path) -> None:
    text = (
        "Robust chunked hierarchical symbolic formula - Iteration 1\n"
        "==========================================================\n\n"
        f"mu = {record['case_mean_formula']}\n\n"
        f"log_scale = {record['case_log_scale_formula']}\n\n"
        f"shape_1 = {record['shape_component_1']}\n\n"
        f"shape_2 = {record['shape_component_2']}\n\n"
        f"shape_3 = {record['shape_component_3']}\n\n"
        f"sigma = {record['composite_formula']}\n\n"
        "Status: Iteration-1 development prototype. Formula selection used only "
        "the Iteration-1 validation cases. The internal test was evaluated once "
        "after freezing all components. The 50 final-test cases were not read.\n"
    )
    (output_dir / "selected_composite_formula.txt").write_text(text, encoding="utf-8")


def run_robust_iteration_one(package_root: Path, config: RobustChunkedConfig) -> dict:
    started = time.time()
    preflight = preflight_robust_iteration(package_root, config)
    output_dir = preflight["output_dir"]
    save_json(output_dir / "run_configuration.json", asdict(config))
    _feature_registry().to_csv(output_dir / "robust_feature_registry.csv", index=False)
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": "prepare_training_cache",
    })
    sleep_inhibitor = _start_sleep_inhibitor(output_dir)
    try:
        cache = prepare_training_cache(preflight, config)
        audit = audit_old_partial_shape(preflight, config)
        robust_matrix = cache["X"]
        original_target = np.asarray(cache["y_shape"], dtype=np.float64)
        weights = np.asarray(cache["weights"], dtype=np.float64)
        accumulated_prediction = np.zeros(len(original_target), dtype=np.float64)
        selected_components: list[dict] = []
        reference = _reference_metrics(preflight["inputs"], config.iteration, "validation")

        for component in range(1, config.components + 1):
            component_dir = output_dir / f"component_{component}"
            component_dir.mkdir(parents=True, exist_ok=True)
            target = original_target - accumulated_prediction
            target_path = component_dir / "training_residual_target.npy"
            np.save(target_path, np.asarray(target, dtype=np.float32))
            features = COMPONENT_FEATURES[component]
            X_component = _component_matrix(robust_matrix, features)
            scaling = _scaling_payload(X_component, target, features, weights)
            scaling_path = component_dir / "scaling.csv"
            _save_scaling(scaling, f"shape_component_{component}_residual", scaling_path)
            selected_path = component_dir / "selected_component_formula.csv"
            checkpoint_path = component_dir / "component_checkpoint.json"
            checkpoint_signature = _component_checkpoint_signature(
                component,
                target_path,
                scaling_path,
                features,
                config,
            )

            # A completed component is reconstructed from disk and skipped.
            # Recomputing its training correction is inexpensive and avoids
            # storing another large prediction array.  The hashes prevent a
            # formula selected for an older residual target from being reused.
            if checkpoint_path.exists() and not config.force_rerun_chunks:
                if not selected_path.exists():
                    raise RuntimeError(
                        f"Component {component} has a checkpoint but no selected formula"
                    )
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if checkpoint.get("input_signature") != checkpoint_signature:
                    raise RuntimeError(
                        f"Component {component} checkpoint does not match the current "
                        "residual target or configuration. Use a new output_subdir "
                        "for a changed experiment."
                    )
                selected = pd.read_csv(selected_path).iloc[0]
                record = {
                    "component": component,
                    "row": selected.to_dict(),
                    "scaling": scaling,
                    "features": list(features),
                }
                selected_components.append(record)
                correction = _training_component_prediction(
                    robust_matrix, selected, scaling, features
                )
                accumulated_prediction += correction
                print(
                    f"Component {component}: completed checkpoint reused; "
                    "search and validation skipped.",
                    flush=True,
                )
                del X_component, target, correction
                gc.collect()
                continue

            _atomic_json(output_dir / "run_status.json", {
                "status": "running",
                "stage": f"component_{component}_search",
                "component": component,
                "elapsed_seconds": time.time() - started,
            })

            chunk_frontiers = []
            for chunk in range(1, config.chunks_per_component + 1):
                chunk_frontiers.append(run_worker_chunk(
                    preflight,
                    config,
                    cache,
                    component,
                    chunk,
                    target_path,
                    scaling_path,
                ))
            merged = merge_component_frontiers(
                chunk_frontiers,
                target,
                weights,
                scaling,
                component,
                config.max_candidates_per_component,
            )
            merged.to_csv(component_dir / "merged_shortlisted_frontier.csv", index=False)
            metrics, case_metrics = evaluate_component_candidates(
                candidates=merged,
                candidate_scaling=scaling,
                candidate_features=features,
                accumulated=selected_components,
                preflight=preflight,
                case_ids=preflight["inputs"]["validation_ids"],
                split="validation",
                component=component,
            )
            selected = select_component_candidate(metrics, reference, component_dir)
            selected_index = int(selected["candidate_index"])
            case_metrics[case_metrics["candidate_index"] == selected_index].to_csv(
                component_dir / "selected_validation_case_metrics.csv", index=False
            )
            record = {
                "component": component,
                "row": selected.to_dict(),
                "scaling": scaling,
                "features": list(features),
            }
            selected_components.append(record)
            correction = _training_component_prediction(
                robust_matrix, selected, scaling, features
            )
            accumulated_prediction += correction
            pd.DataFrame({
                "component": [component],
                "weighted_residual_rmse_before": [math.sqrt(np.average(target**2, weights=weights))],
                "weighted_residual_rmse_after": [math.sqrt(np.average((target - correction) ** 2, weights=weights))],
                "selected_candidate_index": [selected_index],
                "selected_complexity": [int(selected["complexity"])],
            }).to_csv(component_dir / "training_residual_progress.csv", index=False)
            _atomic_json(checkpoint_path, {
                "status": "complete",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "input_signature": checkpoint_signature,
                "selected_candidate_index": selected_index,
                "selected_complexity": int(selected["complexity"]),
                "selected_formula_path": str(selected_path),
                "validation_case_metrics_path": str(
                    component_dir / "selected_validation_case_metrics.csv"
                ),
            })
            del X_component, target, correction
            gc.collect()

        validation_summary, validation_cases = evaluate_fixed_composite(
            preflight,
            selected_components,
            preflight["inputs"]["validation_ids"],
            "validation_frozen",
        )
        internal_summary, internal_cases = evaluate_fixed_composite(
            preflight,
            selected_components,
            preflight["inputs"]["internal_ids"],
            "internal_test_once",
        )
        split_metrics = pd.concat([validation_summary, internal_summary], ignore_index=True)
        split_metrics.to_csv(output_dir / "frozen_composite_split_metrics.csv", index=False)
        validation_cases.to_csv(output_dir / "frozen_validation_case_metrics.csv", index=False)
        internal_cases.to_csv(output_dir / "frozen_internal_test_case_metrics.csv", index=False)
        formula = _formula_record(preflight, selected_components)
        pd.DataFrame([formula]).to_csv(output_dir / "selected_composite_formula.csv", index=False)
        _save_formula_text(formula, output_dir)
        _atomic_json(output_dir / "run_status.json", {
            "status": "complete",
            "stage": "complete",
            "elapsed_seconds": time.time() - started,
            "components": 3,
            "old_partial_audit_rows": 0 if audit is None else len(audit),
            "final_test_cases_read": 0,
        })
        return {
            "preflight": preflight,
            "cache": cache,
            "selected_components": selected_components,
            "formula": formula,
            "split_metrics": split_metrics,
            "validation_case_metrics": validation_cases,
            "internal_case_metrics": internal_cases,
            "output_dir": output_dir,
        }
    except Exception as exc:
        _atomic_json(output_dir / "run_status.json", {
            "status": "failed",
            "stage": "exception",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
        })
        raise
    finally:
        _stop_sleep_inhibitor(sleep_inhibitor)
