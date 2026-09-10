"""Integrate four frozen CT3 rotations on all 149 development FEM cases.

This stage performs no new symbolic search and never parses final-test predictor
or stress values. It combines the four frozen V10 formulae, refits one compact
case-mean correction, and estimates one bounded gain for the stable V11 tail
consensus. Final files are checked only at the inventory/header level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from ct3_common import (  # noqa: E402
    TARGET_COL,
    aggregate_case_metrics,
    assert_no_final_cases,
    development_case_ids,
    evaluate_prediction_arrays,
    make_case_tail_weights,
    read_complete_case,
)
from hierarchical_feasibility import (  # noqa: E402
    FeasibilityConfig,
    _case_summary_row,
    load_inputs,
)
from signed_staged_residual_symbolic import (  # noqa: E402
    ALL_V10_FEATURES,
    PHYSICAL_CONTEXT_FEATURES,
    STAGE_A_FEATURES,
    STAGE_B_FEATURES,
    _feature_indices,
    _load_frozen_baseline,
    _predict_baseline_components,
)
from v11_tail_aware_localised_symbolic import (  # noqa: E402
    _fit_mean_calibration,
    _load_v10_state,
    _mean_correction,
)


OUTPUT_NAME = "18_149case_development_integration"
ROTATIONS = (1, 2, 3, 4)
CANDIDATE_ORDER = (
    "v10_rotation_consensus",
    "mean_calibrated_consensus",
    "full_consensus_tail",
    "frozen_rotation_selected_ensemble",
)


@dataclass(frozen=True)
class DevelopmentIntegrationConfig:
    output_subdir: str = "formal_149case_integration"
    tail_activity_std_threshold_mpa: float = 1e-6
    tail_gain_min: float = 0.0
    tail_gain_max: float = 1.0
    min_relative_rmse_improvement: float = 0.0025
    min_hotspot_overlap_gain: float = 0.005
    full_p99_ratio_to_mean_max: float = 0.98
    full_worst_rmse_ratio_to_mean_max: float = 1.03
    full_p95_ratio_to_base_max: float = 1.20
    mean_worst_rmse_ratio_to_base_max: float = 1.05
    mean_p99_ratio_to_base_max: float = 1.25
    mean_hotspot_drop_from_base_max: float = 0.02
    keep_case_cache_after_success: bool = False
    force_rebuild_case_cache: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.tail_gain_min < self.tail_gain_max <= 1.0:
            raise ValueError("The all-development tail gain must be bounded inside [0, 1]")
        if self.tail_activity_std_threshold_mpa <= 0:
            raise ValueError("Tail-activity threshold must be positive")
        if self.min_relative_rmse_improvement < 0:
            raise ValueError("RMSE improvement threshold cannot be negative")


def output_directory(package_root: Path, config: DevelopmentIntegrationConfig) -> Path:
    return Path(package_root) / "outputs" / OUTPUT_NAME / config.output_subdir


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-12))


def _compact_rotation_paths(package_root: Path, iteration: int) -> dict[str, Path]:
    return {
        "formal_completion": package_root
        / f"outputs/17_v11_four_rotation_formal/iteration_{iteration}/pipeline_complete.json",
        "v10_completion": package_root
        / f"outputs/10_signed_staged_residual_symbolic_pilot/iteration_{iteration}/pilot_complete.json",
        "v10_formula": package_root
        / f"outputs/10_signed_staged_residual_symbolic_pilot/iteration_{iteration}/selected_composite_formula.csv",
        "v11_completion": package_root
        / f"outputs/11_tail_aware_localised_symbolic/iteration_{iteration}/pilot_complete.json",
        "v11_formula": package_root
        / f"outputs/11_tail_aware_localised_symbolic/iteration_{iteration}/selected_composite_formula.csv",
        "v11_mean_model": package_root
        / f"outputs/11_tail_aware_localised_symbolic/iteration_{iteration}/training_cache/mean_calibration_model.json",
        "gain_completion": package_root
        / f"outputs/12_v11_gain_stability_audit/iteration_{iteration}/audit_complete.json",
        "selected_gain": package_root
        / f"outputs/12_v11_gain_stability_audit/iteration_{iteration}/selected_gain.json",
        "selected_gain_formula": package_root
        / f"outputs/12_v11_gain_stability_audit/iteration_{iteration}/selected_gain_formula.csv",
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_completion(path: Path, iteration: int, label: str) -> dict:
    record = _load_json(path)
    if record.get("status") != "complete":
        raise RuntimeError(f"{label} Iteration {iteration} is not complete")
    if int(record.get("iteration", -1)) != iteration:
        raise RuntimeError(f"{label} completion belongs to another iteration")
    if int(record.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError(f"{label} accessed sealed final-test element files")
    return record


def _compile_raw_formula(formula: str | float) -> tuple[Callable[[np.ndarray], np.ndarray], list[str], str]:
    expression = sp.sympify(
        str(formula),
        locals={"Abs": sp.Abs, "abs": sp.Abs, "exp": sp.exp, "tanh": sp.tanh},
    )
    symbols = sorted(str(symbol) for symbol in expression.free_symbols)
    unknown = sorted(set(symbols) - set(ALL_V10_FEATURES))
    if unknown:
        raise ValueError(f"Tail formula contains unregistered predictors: {unknown}")
    if not symbols:
        constant = float(expression)

        def constant_function(matrix: np.ndarray) -> np.ndarray:
            return np.full(len(matrix), constant, dtype=np.float64)

        return constant_function, symbols, str(expression)

    compiled = sp.lambdify(
        [sp.Symbol(name) for name in symbols],
        expression,
        modules=[{"Abs": np.abs}, "numpy"],
    )
    indices = [ALL_V10_FEATURES.index(name) for name in symbols]

    def evaluate(matrix: np.ndarray) -> np.ndarray:
        values = [np.asarray(matrix[:, index], dtype=np.float64) for index in indices]
        result = np.asarray(compiled(*values), dtype=np.float64)
        if result.ndim == 0:
            result = np.full(len(matrix), float(result), dtype=np.float64)
        result = result.reshape(-1)
        if len(result) != len(matrix) or not np.isfinite(result).all():
            raise FloatingPointError("Tail formula produced invalid complete-case values")
        return result

    return evaluate, symbols, str(expression)


def _load_prior_mean_model(path: Path) -> dict:
    model = _load_json(path)
    return {
        "features": list(model["features"]),
        "coefficients": np.asarray(model["coefficients"], dtype=np.float64),
        "intercept": float(model["intercept"]),
        "formula": str(model.get("formula_original_variables", "")),
    }


def _load_rotation_states(package_root: Path, baseline: dict) -> tuple[list[dict], pd.DataFrame]:
    states = []
    audit_rows = []
    for iteration in ROTATIONS:
        paths = _compact_rotation_paths(package_root, iteration)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing frozen rotation artifacts:\n" + "\n".join(missing))
        _assert_completion(paths["formal_completion"], iteration, "Formal pipeline")
        _assert_completion(paths["v10_completion"], iteration, "V10")
        _assert_completion(paths["v11_completion"], iteration, "V11")
        _assert_completion(paths["gain_completion"], iteration, "Gain audit")
        v10_state = _load_v10_state(package_root, baseline, iteration)
        v10_formula = pd.read_csv(paths["v10_formula"]).iloc[0]
        v11_formula = pd.read_csv(paths["v11_formula"]).iloc[0]
        gain_record = _load_json(paths["selected_gain"])
        tail_function, tail_symbols, canonical_tail = _compile_raw_formula(
            v11_formula["tail_raw_formula"]
        )
        state = {
            "iteration": iteration,
            "paths": paths,
            "v10": v10_state,
            "v10_formula": str(v10_formula["composite_stress_formula"]),
            "tail_formula": str(v11_formula["tail_raw_formula"]),
            "canonical_tail_formula": canonical_tail,
            "tail_complexity": int(v11_formula["tail_complexity"]),
            "tail_symbols": tail_symbols,
            "tail_function": tail_function,
            "symbolically_active": bool(tail_symbols),
            "prior_mean_model": _load_prior_mean_model(paths["v11_mean_model"]),
            "prior_selected_gain": float(gain_record["selected_gain"]),
            "prior_promotion_status": str(gain_record.get("promotion_status", "unknown")),
        }
        states.append(state)
        audit_rows.append({
            "iteration": iteration,
            "tail_complexity": state["tail_complexity"],
            "tail_predictors": ",".join(tail_symbols),
            "n_tail_predictors": len(tail_symbols),
            "symbolically_active": state["symbolically_active"],
            "prior_selected_gain": state["prior_selected_gain"],
            "prior_promotion_status": state["prior_promotion_status"],
        })
    return states, pd.DataFrame(audit_rows)


def _input_signature(
    package_root: Path,
    states: Sequence[dict],
    baseline: dict,
    inputs: dict,
    development_ids: Sequence[str],
) -> tuple[str, pd.DataFrame]:
    paths = [
        package_root / "shared/frozen_case_split_manifest_199cases.csv",
        package_root / "outputs/00_qc_sensitivity_ablation/development_case_summary.csv",
        package_root
        / "outputs/17_v11_four_rotation_formal/four_rotation_summary/aggregation_complete.json",
        package_root
        / "outputs/17_v11_four_rotation_formal/four_rotation_summary/all_rotation_split_metrics.csv",
    ]
    paths.extend(baseline["paths"].values())
    for state in states:
        paths.extend(state["paths"].values())
        paths.extend(
            state["v10"]["paths"][name]
            for name in (
                "geometry_row",
                "physical_row",
                "geometry_scaling",
                "physical_scaling",
                "cache_metadata",
            )
        )
    unique_paths = sorted(set(paths), key=str)
    rows = []
    for path in unique_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "path": str(path.relative_to(package_root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    # Full FEM files are too large to hash at every resume. Size and nanosecond
    # modification time still prevent silent reuse after a normal replacement.
    for case_id in development_ids:
        path = inputs["path_by_case"][case_id]
        stat = path.stat()
        rows.append({
            "path": str(path),
            "size_bytes": stat.st_size,
            "sha256": f"large_fem_file_mtime_ns_{stat.st_mtime_ns}",
        })
    table = pd.DataFrame(rows)
    digest = hashlib.sha256()
    for row in table.itertuples(index=False):
        digest.update(f"{row.path}|{row.size_bytes}|{row.sha256}\n".encode("utf-8"))
    return digest.hexdigest(), table


def preflight_development_integration(
    package_root: Path,
    config: DevelopmentIntegrationConfig,
) -> dict:
    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(
        package_root,
        FeasibilityConfig(
            iteration=1,
            rows_per_case=5_000,
            output_subdir="development_integration_input_check",
        ),
    )
    development_ids = development_case_ids(inputs["manifest"])
    assert_no_final_cases(development_ids, inputs["manifest"])
    if len(development_ids) != 149 or len(inputs["final_ids"]) != 50:
        raise RuntimeError("Expected exactly 149 development and 50 sealed final-test cases")

    aggregate_marker = _load_json(
        package_root
        / "outputs/17_v11_four_rotation_formal/four_rotation_summary/aggregation_complete.json"
    )
    if aggregate_marker.get("status") != "complete":
        raise RuntimeError("Four-rotation aggregation is not complete")
    if int(aggregate_marker.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError("Four-rotation aggregation accessed sealed final-test elements")

    baseline = _load_frozen_baseline(package_root)
    states, rotation_audit = _load_rotation_states(package_root, baseline)
    signature, input_hashes = _input_signature(
        package_root,
        states,
        baseline,
        inputs,
        development_ids,
    )
    input_hashes.to_csv(output_dir / "frozen_input_hash_audit.csv", index=False)
    rotation_audit.to_csv(output_dir / "rotation_formula_inventory.csv", index=False)

    checks = pd.DataFrame([
        {"check": "development_cases", "value": len(development_ids), "expected": 149},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "frozen_rotations", "value": len(states), "expected": 4},
        {
            "check": "constant_tail_rotations",
            "value": int((~rotation_audit["symbolically_active"]).sum()),
            "expected": 1,
        },
        {
            "check": "iteration_4_tail_is_constant",
            "value": bool(
                not rotation_audit.set_index("iteration").loc[4, "symbolically_active"]
            ),
            "expected": True,
        },
        {
            "check": "final_test_element_files_read",
            "value": 0,
            "expected": 0,
        },
    ])
    checks["pass"] = checks["value"] == checks["expected"]
    checks.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not checks["pass"].all():
        raise RuntimeError(
            f"Development-integration preflight failed: "
            f"{checks.loc[~checks['pass'], 'check'].tolist()}"
        )
    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "development_ids": development_ids,
        "baseline": baseline,
        "states": states,
        "rotation_audit": rotation_audit,
        "input_signature": signature,
        "input_hashes": input_hashes,
        "checks": checks,
    }


def _case_cache_path(cache_dir: Path, case_id: str) -> Path:
    return cache_dir / f"{case_id}.npz"


def _cache_is_valid(path: Path, case_id: str, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            return (
                str(cached["case_id"].item()) == case_id
                and str(cached["input_signature"].item()) == signature
                and cached["tail_corrections"].shape[0] == 4
                and len(cached["actual"]) == len(cached["base_consensus"])
            )
    except Exception:
        return False


def _build_case_cache(
    case_id: str,
    preflight: dict,
) -> dict:
    frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
    summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    first_state = preflight["states"][0]
    matrix, _, scale, _, baseline_stress = _predict_baseline_components(
        frame,
        summary,
        first_state["v10"]["baseline_functions"],
    )
    v10_predictions = []
    tail_corrections = []
    prior_selected_predictions = []
    for state in preflight["states"]:
        geometry = state["v10"]["geometry_function"](
            matrix[:, _feature_indices(STAGE_A_FEATURES)]
        )
        physical = state["v10"]["physical_function"](
            matrix[:, _feature_indices(STAGE_B_FEATURES)]
        )
        v10_prediction = baseline_stress + scale * (geometry + physical)
        raw_tail = state["tail_function"](matrix)
        tail_correction = scale * (raw_tail - float(raw_tail.mean()))
        prior_mean = _mean_correction(summary, state["prior_mean_model"])
        prior_selected = (
            v10_prediction
            + prior_mean
            + state["prior_selected_gain"] * tail_correction
        )
        for name, values in {
            "v10": v10_prediction,
            "tail": tail_correction,
            "prior_selected": prior_selected,
        }.items():
            if len(values) != len(actual) or not np.isfinite(values).all():
                raise FloatingPointError(f"{case_id} Iteration {state['iteration']} {name} is invalid")
        v10_predictions.append(v10_prediction)
        tail_corrections.append(tail_correction)
        prior_selected_predictions.append(prior_selected)

    v10_matrix = np.vstack(v10_predictions)
    tail_matrix = np.vstack(tail_corrections)
    return {
        "actual": actual.astype(np.float32),
        "base_consensus": v10_matrix.mean(axis=0).astype(np.float32),
        "tail_corrections": tail_matrix.astype(np.float32),
        "prior_selected_ensemble": np.vstack(prior_selected_predictions)
        .mean(axis=0)
        .astype(np.float32),
        "tail_std_by_rotation": tail_matrix.std(axis=1),
        "n_elements": len(actual),
    }


def prepare_development_case_cache(
    preflight: dict,
    config: DevelopmentIntegrationConfig,
) -> dict:
    output_dir = preflight["output_dir"]
    cache_dir = output_dir / "case_prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "case_cache_audit.csv"
    audit_rows = []
    started = time.time()
    for position, case_id in enumerate(preflight["development_ids"], start=1):
        path = _case_cache_path(cache_dir, case_id)
        reusable = (
            not config.force_rebuild_case_cache
            and _cache_is_valid(path, case_id, preflight["input_signature"])
        )
        print(
            f"[{position}/149] Development integration: {case_id} "
            f"({'reuse checkpoint' if reusable else 'build checkpoint'})",
            flush=True,
        )
        if not reusable:
            data = _build_case_cache(case_id, preflight)
            _atomic_npz(
                path,
                case_id=np.asarray(case_id),
                input_signature=np.asarray(preflight["input_signature"]),
                actual=data["actual"],
                base_consensus=data["base_consensus"],
                tail_corrections=data["tail_corrections"],
                prior_selected_ensemble=data["prior_selected_ensemble"],
            )
            tail_stds = data["tail_std_by_rotation"]
            n_elements = data["n_elements"]
            del data
        else:
            with np.load(path, allow_pickle=False) as cached:
                tail_stds = np.asarray(cached["tail_corrections"], dtype=np.float64).std(axis=1)
                n_elements = len(cached["actual"])
        row = {
            "case_id": case_id,
            "cache_path": str(path.relative_to(preflight["package_root"])),
            "cache_reused": reusable,
            "n_elements": n_elements,
            "size_mb": path.stat().st_size / 1_000_000,
        }
        for index, value in enumerate(tail_stds, start=1):
            row[f"iteration_{index}_tail_std_mpa"] = float(value)
        audit_rows.append(row)
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "running",
                "stage": "case_prediction_cache",
                "completed_cases": position,
                "total_development_cases": 149,
                "last_completed_case": case_id,
                "elapsed_seconds": time.time() - started,
                "final_test_cases_read": 0,
            },
        )
        gc.collect()
    return {"cache_dir": cache_dir, "audit": pd.DataFrame(audit_rows)}


def _tail_activity_audit(preflight: dict, cache: dict, config: DevelopmentIntegrationConfig) -> pd.DataFrame:
    rotation = preflight["rotation_audit"].set_index("iteration")
    rows = []
    for iteration in ROTATIONS:
        column = f"iteration_{iteration}_tail_std_mpa"
        values = cache["audit"][column].to_numpy(dtype=np.float64)
        symbolic = bool(rotation.loc[iteration, "symbolically_active"])
        numerical = bool(np.max(values) > config.tail_activity_std_threshold_mpa)
        rows.append({
            "iteration": iteration,
            "tail_complexity": int(rotation.loc[iteration, "tail_complexity"]),
            "tail_predictors": rotation.loc[iteration, "tail_predictors"],
            "symbolically_active": symbolic,
            "mean_complete_case_tail_std_mpa": float(values.mean()),
            "max_complete_case_tail_std_mpa": float(values.max()),
            "numerically_active": numerical,
            "included_in_tail_consensus": symbolic and numerical,
            "prior_selected_gain": float(rotation.loc[iteration, "prior_selected_gain"]),
            "interpretation": (
                "active spatial tail formula"
                if symbolic and numerical
                else "constant or numerically zero after complete-case centering"
            ),
        })
    table = pd.DataFrame(rows)
    if int(table["included_in_tail_consensus"].sum()) < 2:
        raise RuntimeError("Fewer than two rotations provide an active tail correction")
    return table


def _fit_all_development_mean(preflight: dict, cache: dict) -> dict:
    rows = []
    for case_id in preflight["development_ids"]:
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        with np.load(_case_cache_path(cache["cache_dir"], case_id), allow_pickle=False) as data:
            residual_mean = float(
                np.asarray(data["actual"], dtype=np.float64).mean()
                - np.asarray(data["base_consensus"], dtype=np.float64).mean()
            )
        row = {feature: float(summary[feature]) for feature in PHYSICAL_CONTEXT_FEATURES}
        row.update({
            "case_id": case_id,
            "estimated_v10_mean_residual_mpa": residual_mean,
        })
        rows.append(row)
    case_table = pd.DataFrame(rows)
    model = _fit_mean_calibration(case_table)
    return {"model": model, "case_table": case_table}


def _active_tail_indices(activity: pd.DataFrame) -> list[int]:
    return [
        int(iteration) - 1
        for iteration in activity.loc[
            activity["included_in_tail_consensus"], "iteration"
        ].tolist()
    ]


def _fit_all_development_tail_gain(
    preflight: dict,
    cache: dict,
    mean_model: dict,
    activity: pd.DataFrame,
    config: DevelopmentIntegrationConfig,
) -> dict:
    active = _active_tail_indices(activity)
    numerator = 0.0
    denominator = 0.0
    case_rows = []
    for position, case_id in enumerate(preflight["development_ids"], start=1):
        print(f"[{position}/149] Fit bounded tail gain: {case_id}", flush=True)
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        with np.load(_case_cache_path(cache["cache_dir"], case_id), allow_pickle=False) as data:
            actual = np.asarray(data["actual"], dtype=np.float64)
            base = np.asarray(data["base_consensus"], dtype=np.float64)
            tails = np.asarray(data["tail_corrections"], dtype=np.float64)
        tail = tails[active].mean(axis=0)
        mean_prediction = base + _mean_correction(summary, mean_model)
        residual = actual - mean_prediction
        weights, weight_audit = make_case_tail_weights(actual)
        weights = np.asarray(weights, dtype=np.float64)
        case_numerator = float(np.dot(weights * tail, residual))
        case_denominator = float(np.dot(weights * tail, tail))
        numerator += case_numerator
        denominator += case_denominator
        case_rows.append({
            "case_id": case_id,
            "weighted_numerator": case_numerator,
            "weighted_denominator": case_denominator,
            "unconstrained_case_gain": (
                case_numerator / case_denominator if case_denominator > 0 else np.nan
            ),
            **weight_audit,
        })
    if denominator <= 0 or not np.isfinite(denominator):
        raise RuntimeError("Tail consensus has no finite weighted variance")
    unconstrained = numerator / denominator
    selected = float(np.clip(unconstrained, config.tail_gain_min, config.tail_gain_max))
    return {
        "selected_gain": selected,
        "unconstrained_gain": float(unconstrained),
        "gain_was_clipped": not np.isclose(selected, unconstrained),
        "weighted_numerator": numerator,
        "weighted_denominator": denominator,
        "active_iterations": [index + 1 for index in active],
        "case_audit": pd.DataFrame(case_rows),
    }


def _heldout_tail_evidence(package_root: Path) -> tuple[pd.DataFrame, bool]:
    metrics = pd.read_csv(
        package_root
        / "outputs/17_v11_four_rotation_formal/four_rotation_summary/all_rotation_split_metrics.csv"
    )
    rows = []
    all_pass = True
    for split in ("validation", "internal_test"):
        block = metrics[metrics["split"].eq(split)]
        mean_row = block[block["model"].eq("v11_mean_calibrated_v10")]
        selected_row = block[block["model"].str.startswith("v11_selected_gain_")]
        if len(mean_row) != 4 or len(selected_row) != 4:
            raise RuntimeError(f"Incomplete four-rotation held-out evidence for {split}")
        mean_values = mean_row.mean(numeric_only=True)
        selected_values = selected_row.mean(numeric_only=True)
        gates = {
            "rmse_not_worse": selected_values["macro_rmse"] <= mean_values["macro_rmse"],
            "p99_not_worse": (
                selected_values["mean_p99_relative_error"]
                <= mean_values["mean_p99_relative_error"]
            ),
            "hotspot_not_worse": (
                selected_values["mean_top1pct_hotspot_overlap"]
                >= mean_values["mean_top1pct_hotspot_overlap"]
            ),
        }
        split_pass = bool(all(gates.values()))
        all_pass = all_pass and split_pass
        rows.append({
            "split": split,
            "mean_only_macro_rmse": float(mean_values["macro_rmse"]),
            "selected_macro_rmse": float(selected_values["macro_rmse"]),
            "selected_minus_mean_macro_rmse": float(
                selected_values["macro_rmse"] - mean_values["macro_rmse"]
            ),
            "mean_only_p99_relative_error": float(mean_values["mean_p99_relative_error"]),
            "selected_p99_relative_error": float(selected_values["mean_p99_relative_error"]),
            "selected_minus_mean_p99_relative_error": float(
                selected_values["mean_p99_relative_error"]
                - mean_values["mean_p99_relative_error"]
            ),
            "mean_only_top1_hotspot_overlap": float(
                mean_values["mean_top1pct_hotspot_overlap"]
            ),
            "selected_top1_hotspot_overlap": float(
                selected_values["mean_top1pct_hotspot_overlap"]
            ),
            "selected_minus_mean_top1_hotspot_overlap": float(
                selected_values["mean_top1pct_hotspot_overlap"]
                - mean_values["mean_top1pct_hotspot_overlap"]
            ),
            **gates,
            "all_split_gates_pass": split_pass,
        })
    return pd.DataFrame(rows), all_pass


def _evaluate_development_candidates(
    preflight: dict,
    cache: dict,
    mean_model: dict,
    activity: pd.DataFrame,
    tail_gain: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = _active_tail_indices(activity)
    rows = []
    for position, case_id in enumerate(preflight["development_ids"], start=1):
        print(f"[{position}/149] Evaluate development candidates: {case_id}", flush=True)
        summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
        with np.load(_case_cache_path(cache["cache_dir"], case_id), allow_pickle=False) as data:
            actual = np.asarray(data["actual"], dtype=np.float64)
            base = np.asarray(data["base_consensus"], dtype=np.float64)
            tails = np.asarray(data["tail_corrections"], dtype=np.float64)
            frozen = np.asarray(data["prior_selected_ensemble"], dtype=np.float64)
        mean_prediction = base + _mean_correction(summary, mean_model)
        tail_consensus = tails[active].mean(axis=0)
        predictions = {
            "v10_rotation_consensus": base,
            "mean_calibrated_consensus": mean_prediction,
            "full_consensus_tail": mean_prediction + tail_gain * tail_consensus,
            "frozen_rotation_selected_ensemble": frozen,
        }
        for model, predicted in predictions.items():
            rows.append({
                "split": "development_fit_diagnostic",
                "model": model,
                "case_id": case_id,
                **evaluate_prediction_arrays(actual, predicted),
            })
        gc.collect()
    case_metrics = pd.DataFrame(rows)
    summary = aggregate_case_metrics(case_metrics, ["split", "model"])
    summary["candidate_order"] = summary["model"].map(
        {name: index for index, name in enumerate(CANDIDATE_ORDER)}
    )
    summary = summary.sort_values("candidate_order").drop(columns="candidate_order")
    return case_metrics, summary


def _model_selection_decision(
    summary: pd.DataFrame,
    activity: pd.DataFrame,
    heldout_tail_pass: bool,
    config: DevelopmentIntegrationConfig,
) -> dict:
    indexed = summary.set_index("model")
    base = indexed.loc["v10_rotation_consensus"]
    mean = indexed.loc["mean_calibrated_consensus"]
    full = indexed.loc["full_consensus_tail"]
    active_count = int(activity["included_in_tail_consensus"].sum())

    full_gates = {
        "at_least_two_active_tail_rotations": active_count >= 2,
        "four_rotation_heldout_tail_evidence_pass": heldout_tail_pass,
        "development_macro_rmse_improves_mean": (
            full["macro_rmse"]
            <= mean["macro_rmse"] * (1.0 - config.min_relative_rmse_improvement)
        ),
        "development_p99_improves_mean": (
            full["mean_p99_relative_error"]
            <= mean["mean_p99_relative_error"] * config.full_p99_ratio_to_mean_max
        ),
        "development_top1_hotspot_improves_mean": (
            full["mean_top1pct_hotspot_overlap"]
            >= mean["mean_top1pct_hotspot_overlap"] + config.min_hotspot_overlap_gain
        ),
        "development_worst_rmse_controlled": (
            full["worst_case_rmse"]
            <= mean["worst_case_rmse"] * config.full_worst_rmse_ratio_to_mean_max
        ),
        "development_p95_guardrail": (
            full["mean_p95_relative_error"]
            <= base["mean_p95_relative_error"] * config.full_p95_ratio_to_base_max
        ),
    }
    mean_gates = {
        "development_macro_rmse_improves_base": (
            mean["macro_rmse"]
            <= base["macro_rmse"] * (1.0 - config.min_relative_rmse_improvement)
        ),
        "development_worst_rmse_controlled": (
            mean["worst_case_rmse"]
            <= base["worst_case_rmse"] * config.mean_worst_rmse_ratio_to_base_max
        ),
        "development_p99_guardrail": (
            mean["mean_p99_relative_error"]
            <= base["mean_p99_relative_error"] * config.mean_p99_ratio_to_base_max
        ),
        "development_hotspot_guardrail": (
            mean["mean_top1pct_hotspot_overlap"]
            >= base["mean_top1pct_hotspot_overlap"]
            - config.mean_hotspot_drop_from_base_max
        ),
    }
    if all(full_gates.values()):
        selected = "full_consensus_tail"
        reason = "Full consensus passed every predeclared development and held-out gate."
    elif all(mean_gates.values()):
        selected = "mean_calibrated_consensus"
        reason = "Tail consensus failed at least one gate; the stable all-development mean correction was retained."
    else:
        selected = "v10_rotation_consensus"
        reason = "Both refitted extensions failed their guardrails; the equal-weight V10 consensus was retained."
    return {
        "selected_model": selected,
        "selection_reason": reason,
        "full_consensus_tail_gates": full_gates,
        "full_consensus_tail_all_gates_pass": bool(all(full_gates.values())),
        "mean_calibrated_consensus_gates": mean_gates,
        "mean_calibrated_consensus_all_gates_pass": bool(all(mean_gates.values())),
        "active_tail_iterations": activity.loc[
            activity["included_in_tail_consensus"], "iteration"
        ].astype(int).tolist(),
        "inactive_tail_iterations": activity.loc[
            ~activity["included_in_tail_consensus"], "iteration"
        ].astype(int).tolist(),
        "selection_data_role": "149-case development fit diagnostic plus frozen four-rotation held-out evidence",
        "final_test_used_for_selection": False,
    }


def _formula_components(
    states: Sequence[dict],
    mean_model: dict,
    activity: pd.DataFrame,
    tail_gain: float,
    selected_model: str,
) -> dict:
    base_terms = [f"({state['v10_formula']})" for state in states]
    base_formula = "(1/4) * (" + " + ".join(base_terms) + ")"
    active_states = [
        state
        for state in states
        if int(state["iteration"])
        in set(
            activity.loc[activity["included_in_tail_consensus"], "iteration"].astype(int)
        )
    ]
    scale_formula = str(
        pd.read_csv(states[0]["paths"]["selected_gain_formula"]).iloc[0][
            "v10_log_scale_formula"
        ]
    )
    tail_terms = [
        f"exp({scale_formula}) * (({state['tail_formula']}) - case_mean({state['tail_formula']}))"
        for state in active_states
    ]
    tail_formula = (
        f"(1/{len(tail_terms)}) * (" + " + ".join(tail_terms) + ")"
    )
    mean_formula = str(mean_model["formula"])
    mean_consensus = f"({base_formula}) + ({mean_formula})"
    full_formula = f"({mean_consensus}) + ({tail_gain:.16g}) * ({tail_formula})"
    locked = {
        "v10_rotation_consensus": base_formula,
        "mean_calibrated_consensus": mean_consensus,
        "full_consensus_tail": full_formula,
    }[selected_model]
    return {
        "base_consensus_formula": base_formula,
        "all_development_mean_correction_formula": mean_formula,
        "active_tail_consensus_formula": tail_formula,
        "all_development_tail_gain": tail_gain,
        "locked_model_name": selected_model,
        "locked_model_formula": locked,
        "formula_representation": "componentised symbolic expression; algebraically deployable without FEM stress summaries",
    }


def _save_comparison_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("macro_rmse", "Macro RMSE", "lower is better"),
        ("mean_p99_relative_error", "Mean P99 relative error", "lower is better"),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap", "higher is better"),
        ("worst_case_rmse", "Worst-case RMSE", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [name.replace("_", "\n") for name in summary["model"]]
    for axis, (column, title, note) in zip(axes.ravel(), metrics):
        axis.bar(labels, summary[column], color=["#355070", "#6d9f71", "#c98239", "#747474"])
        axis.set_title(f"{title} ({note})")
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("149-case development integration diagnostics\nNot an independent final-test result")
    fig.tight_layout()
    fig.savefig(output_dir / "development_candidate_comparison.png", dpi=180)
    plt.close(fig)


def run_development_integration(
    package_root: Path,
    config: DevelopmentIntegrationConfig | None = None,
) -> dict:
    config = config or DevelopmentIntegrationConfig()
    config.validate()
    started = time.time()
    preflight = preflight_development_integration(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "development_integration_complete.json"
    if completion_path.exists():
        completion = _load_json(completion_path)
        if (
            completion.get("status") == "complete"
            and completion.get("input_signature_sha256") == preflight["input_signature"]
            and int(completion.get("final_test_cases_read", -1)) == 0
        ):
            print("Reusing completed 149-case development integration.", flush=True)
            return completion
        raise RuntimeError("An incompatible completion marker already exists; archive the output directory first")

    _atomic_json(output_dir / "locked_configuration.json", asdict(config))
    _atomic_json(
        output_dir / "run_status.json",
        {
            "status": "running",
            "stage": "preflight_complete",
            "input_signature_sha256": preflight["input_signature"],
            "development_cases": 149,
            "sealed_final_cases": 50,
            "final_test_cases_read": 0,
        },
    )
    cache = prepare_development_case_cache(preflight, config)
    activity = _tail_activity_audit(preflight, cache, config)
    activity.to_csv(output_dir / "tail_activity_audit.csv", index=False)

    mean_fit = _fit_all_development_mean(preflight, cache)
    mean_model = mean_fit["model"]
    mean_fit["case_table"].to_csv(output_dir / "mean_calibration_case_targets_149.csv", index=False)
    mean_model["audit"].to_csv(output_dir / "mean_calibration_fit_audit_149.csv", index=False)
    mean_payload = {
        "method": "StandardScaler_plus_RidgeCV_refit_on_all_149_development_case_means",
        "features": list(mean_model["features"]),
        "coefficients": np.asarray(mean_model["coefficients"]).tolist(),
        "intercept": float(mean_model["intercept"]),
        "alpha": float(mean_model["alpha"]),
        "formula_original_variables": str(mean_model["formula"]),
        "training_rmse": float(mean_model["training_rmse"]),
        "training_mae": float(mean_model["training_mae"]),
        "final_test_cases_read": 0,
    }
    _atomic_json(output_dir / "mean_calibration_model_149.json", mean_payload)

    gain_fit = _fit_all_development_tail_gain(
        preflight, cache, mean_model, activity, config
    )
    gain_fit["case_audit"].to_csv(output_dir / "tail_gain_case_audit_149.csv", index=False)
    gain_payload = {key: value for key, value in gain_fit.items() if key != "case_audit"}
    gain_payload.update({
        "method": "closed_form_complete_element_weighted_least_squares_clipped_to_0_1",
        "weighting": "within_each_case_P90_2x_P95_4x_P99_8x_then_mean_one",
        "all_elements_used": True,
        "development_cases": 149,
        "final_test_cases_read": 0,
    })
    _atomic_json(output_dir / "tail_gain_fit_149.json", gain_payload)

    heldout, heldout_pass = _heldout_tail_evidence(preflight["package_root"])
    heldout.to_csv(output_dir / "four_rotation_heldout_tail_evidence.csv", index=False)
    case_metrics, summary = _evaluate_development_candidates(
        preflight,
        cache,
        mean_model,
        activity,
        gain_fit["selected_gain"],
    )
    case_metrics.to_csv(
        output_dir / "development_candidate_case_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(output_dir / "development_candidate_summary.csv", index=False)
    _save_comparison_plot(summary, output_dir)

    decision = _model_selection_decision(
        summary, activity, heldout_pass, config
    )
    decision.update({
        "all_development_tail_gain": gain_fit["selected_gain"],
        "development_cases_used": 149,
        "development_elements_used": int(
            case_metrics.loc[
                case_metrics["model"].eq("v10_rotation_consensus"), "n_elements"
            ].sum()
        ),
        "final_test_cases_read": 0,
    })
    _atomic_json(output_dir / "final_model_decision.json", decision)
    formula = _formula_components(
        preflight["states"],
        mean_model,
        activity,
        gain_fit["selected_gain"],
        decision["selected_model"],
    )
    pd.DataFrame([formula]).to_csv(output_dir / "locked_model_formula.csv", index=False)
    formula_text = [
        "CT3 locked 149-case development formula",
        "=======================================",
        "",
        "IMPORTANT: This formula has not yet been evaluated on the sealed 50-case final test.",
        "The 149-case metrics are fit diagnostics, not an unbiased generalisation estimate.",
        "",
        f"Locked model: {formula['locked_model_name']}",
        f"Active tail rotations: {decision['active_tail_iterations']}",
        f"Inactive tail rotations: {decision['inactive_tail_iterations']}",
        "",
        "1. Equal-weight V10 rotation consensus",
        formula["base_consensus_formula"],
        "",
        "2. Mean correction refitted on all 149 development cases",
        formula["all_development_mean_correction_formula"],
        "",
        "3. Active V11 tail consensus",
        formula["active_tail_consensus_formula"],
        "",
        f"4. Bounded all-development tail gain = {formula['all_development_tail_gain']:.16g}",
        "",
        "5. Locked deployable expression",
        formula["locked_model_formula"],
        "",
        "All case_mean(...) terms are calculated from predictor-only values inside the new case.",
        "No measured or FEM-computed stress summary is required at deployment.",
    ]
    (output_dir / "locked_model_formula.txt").write_text(
        "\n".join(formula_text), encoding="utf-8"
    )

    locked_metrics = summary.loc[
        summary["model"].eq(decision["selected_model"])
    ].iloc[0].to_dict()
    manifest = {
        "model_id": "ct3_149case_development_locked_v1",
        "status": "frozen_pending_one_time_final_test",
        "model": decision["selected_model"],
        "base_rotation_weights": {str(i): 0.25 for i in ROTATIONS},
        "mean_calibration_features": list(mean_model["features"]),
        "active_tail_iterations": decision["active_tail_iterations"],
        "inactive_tail_iterations": decision["inactive_tail_iterations"],
        "tail_gain": gain_fit["selected_gain"],
        "development_fit_metrics": locked_metrics,
        "input_signature_sha256": preflight["input_signature"],
        "formula_file": "locked_model_formula.csv",
        "final_test_cases_read": 0,
    }
    _atomic_json(output_dir / "locked_model_manifest.json", manifest)
    release_gate = {
        "technical_status": "ready_for_human_review_before_one_time_final_test",
        "locked_model_exists": True,
        "locked_model": decision["selected_model"],
        "final_test_has_been_opened": False,
        "final_test_cases_read": 0,
        "human_approval_required": True,
        "approved": False,
        "next_stage": "review diagnostics, then create a separate one-time final-test notebook",
    }
    _atomic_json(output_dir / "final_test_release_gate.json", release_gate)

    completion = {
        "status": "complete",
        "method": "four_rotation_consensus_refit_on_all_149_development_cases",
        "selected_model": decision["selected_model"],
        "development_cases": 149,
        "development_elements": decision["development_elements_used"],
        "sealed_final_cases": 50,
        "final_test_cases_read": 0,
        "active_tail_iterations": decision["active_tail_iterations"],
        "inactive_tail_iterations": decision["inactive_tail_iterations"],
        "tail_gain": gain_fit["selected_gain"],
        "input_signature_sha256": preflight["input_signature"],
        "case_cache_retained": True,
        "elapsed_seconds": time.time() - started,
        "output_directory": str(output_dir),
    }
    _atomic_json(completion_path, completion)
    if not config.keep_case_cache_after_success:
        shutil.rmtree(cache["cache_dir"])
        completion["case_cache_retained"] = False
        _atomic_json(completion_path, completion)
    _atomic_json(output_dir / "run_status.json", completion)
    return completion


__all__ = [
    "DevelopmentIntegrationConfig",
    "output_directory",
    "preflight_development_integration",
    "run_development_integration",
]
