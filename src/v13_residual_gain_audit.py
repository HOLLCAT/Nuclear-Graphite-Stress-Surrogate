"""Validation-only gain audit for the frozen V13 structural residual.

The V13 symbolic expression is not searched again. This audit evaluates

    sigma(lambda) = sigma_V11 + lambda * scale_case * centered_r_V13

on complete FEM cases. Lambda is selected using only the 15 validation cases,
then frozen and checked on the 15 internal-development and all 149 development
cases. The 50 final-test element files are inventoried but never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import sys
import time
from typing import Sequence

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
    RANDOM_SEED,
    aggregate_case_metrics,
    assert_no_final_cases,
    evaluate_prediction_arrays,
)
from v13_structural_tail_residual import (  # noqa: E402
    V13_FEATURES,
    V13Config,
    _complete_v11_components,
    preflight_v13,
)


AUDIT_OUTPUT_NAME = "16_v13_residual_gain_audit"
PARENT_OUTPUT_NAME = "15_v13_structural_tail_residual"
PARENT_OUTPUT_SUBDIR = "iteration_1"


@dataclass(frozen=True)
class V13ResidualGainAuditConfig:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    gain_values: tuple[float, ...] = (
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    )
    primary_minimum_rmse_improvement_fraction: float = 0.005
    primary_maximum_p95_degradation_absolute: float = 0.005
    research_maximum_rmse_degradation_fraction: float = 0.010
    research_maximum_p95_degradation_absolute: float = 0.010
    research_minimum_p99_underprediction_improvement_absolute: float = 0.005
    maximum_p99_relative_error: float = 0.15
    maximum_p99_underprediction: float = 0.10
    maximum_top1_overlap_degradation_absolute: float = 0.002
    minimum_top1_recall: float = 0.80
    random_seed: int = RANDOM_SEED

    def validate(self) -> None:
        if self.iteration != 1:
            raise ValueError("The V13 residual gain audit is locked to Iteration 1")
        gains = np.asarray(self.gain_values, dtype=np.float64)
        if len(gains) != len(np.unique(gains)) or np.any(np.diff(gains) <= 0):
            raise ValueError("Gain grid must be unique and strictly increasing")
        if gains[0] != 0.0 or gains[-1] != 1.0:
            raise ValueError("Gain grid must include exactly lambda=0 and lambda=1 endpoints")
        if np.any((gains < 0.0) | (gains > 1.0)):
            raise ValueError("Residual gain must stay inside [0, 1]")

    @property
    def gain_grid(self) -> np.ndarray:
        return np.asarray(self.gain_values, dtype=np.float64)


def output_directory(package_root: Path, config: V13ResidualGainAuditConfig) -> Path:
    return Path(package_root) / "outputs" / AUDIT_OUTPUT_NAME / config.output_subdir


def parent_output_directory(package_root: Path) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / PARENT_OUTPUT_NAME
        / PARENT_OUTPUT_SUBDIR
    )


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


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parent_required_paths(package_root: Path) -> dict[str, Path]:
    root = parent_output_directory(package_root)
    return {
        "v13_completion": root / "v13_complete.json",
        "v13_decision": root / "v13_promotion_decision.json",
        "v13_selected_formula": root / "stages" / "stage_structural_residual" / "selected_formula.csv",
        "v13_candidate_formula": root / "v13_candidate_formula.csv",
        "v13_scope_metrics": root / "v13_scope_metrics.csv",
        "v13_case_metrics": root / "v13_case_metrics.csv.gz",
    }


def preflight_v13_residual_gain_audit(
    package_root: Path,
    config: V13ResidualGainAuditConfig,
) -> dict:
    """Freeze the exact V13 candidate and verify the development-only scope."""

    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    parent = preflight_v13(package_root, V13Config())
    inputs = parent["inputs"]
    development_ids = inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"]
    assert_no_final_cases(development_ids, inputs["manifest"])
    if len(development_ids) != 149 or len(set(development_ids)) != 149:
        raise ValueError("Expected exactly 149 unique development cases")

    paths = _parent_required_paths(package_root)
    paths["audit_source"] = package_root / "src" / "v13_residual_gain_audit.py"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("V13 residual-gain inputs are missing:\n" + "\n".join(missing))

    completion = json.loads(paths["v13_completion"].read_text(encoding="utf-8"))
    decision = json.loads(paths["v13_decision"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or int(completion.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError("Parent V13 run is incomplete or broke the final-test seal")
    if decision.get("promotion_status") != "retain_v11_global_gain_0_90":
        raise RuntimeError("This audit expects V13 to retain frozen V11")

    selected = pd.read_csv(paths["v13_selected_formula"])
    if len(selected) != 1:
        raise ValueError("Expected exactly one frozen V13 residual candidate")
    selected = selected.iloc[0]
    if int(selected["candidate_index"]) != int(completion["selected_candidate_index"]):
        raise ValueError("V13 completion marker and selected candidate disagree")

    expression_text = str(selected["formula_original_variables"])
    expression = sp.sympify(expression_text, locals={"Abs": sp.Abs, "exp": sp.exp})
    feature_symbols = [sp.Symbol(name) for name in V13_FEATURES]
    allowed = set(feature_symbols)
    if not expression.free_symbols.issubset(allowed):
        extra = sorted(str(symbol) for symbol in expression.free_symbols - allowed)
        raise ValueError(f"V13 expression contains unknown variables: {extra}")
    raw_function = sp.lambdify(feature_symbols, expression, modules="numpy")

    def residual_function(matrix: np.ndarray) -> np.ndarray:
        values = raw_function(*[matrix[:, index] for index in range(matrix.shape[1])])
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(len(matrix), float(values), dtype=np.float64)
        values = values.reshape(-1)
        if len(values) != len(matrix) or not np.isfinite(values).all():
            raise FloatingPointError("Frozen V13 residual produced invalid values")
        return values

    hash_rows = []
    for artifact, path in paths.items():
        hash_rows.append({
            "artifact": artifact,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    hash_audit = pd.DataFrame(hash_rows)
    hash_audit.to_csv(output_dir / "frozen_input_hash_audit.csv", index=False)

    checks = pd.DataFrame([
        {"check": "train_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "development_cases", "value": len(development_ids), "expected": 149},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "parent_v13_complete", "value": completion.get("status"), "expected": "complete"},
        {"check": "parent_final_cases_read", "value": completion.get("final_test_cases_read"), "expected": 0},
        {"check": "gain_grid_size", "value": len(config.gain_values), "expected": 22},
        {"check": "candidate_index", "value": int(selected["candidate_index"]), "expected": 2},
    ])
    checks["pass"] = checks["value"] == checks["expected"]
    checks.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not checks["pass"].all():
        failed = checks.loc[~checks["pass"], "check"].tolist()
        raise RuntimeError(f"V13 residual gain preflight failed: {failed}")

    signature = {
        "config": asdict(config),
        "candidate_index": int(selected["candidate_index"]),
        "formula_original_variables": expression_text,
        "frozen_input_sha256": dict(zip(hash_audit["artifact"], hash_audit["sha256"])),
        "validation_ids": inputs["validation_ids"],
        "internal_ids": inputs["internal_ids"],
        "train_ids": inputs["train_ids"],
        "final_ids_inventoried_not_read": inputs["final_ids"],
    }
    # Canonicalise tuples and NumPy-compatible scalar values before persisting
    # and comparing the signature. JSON serialises tuples as lists, so comparing
    # the raw in-memory dataclass payload with a reloaded JSON document would
    # otherwise report a false mismatch on the second preflight in one run.
    signature = json.loads(json.dumps(signature, default=str))
    signature["audit_signature_sha256"] = _canonical_hash(signature)
    signature_path = output_dir / "audit_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError("Existing gain audit has a different frozen signature")
    else:
        _atomic_json(signature_path, signature)

    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "parent": parent,
        "inputs": inputs,
        "development_ids": development_ids,
        "paths": paths,
        "completion": completion,
        "decision": decision,
        "selected": selected,
        "residual_function": residual_function,
        "hash_audit": hash_audit,
        "checks": checks,
        "signature": signature,
    }


def _gain_label(gain: float) -> str:
    return f"gain_{gain:.3f}"


def _case_cache_path(output_dir: Path, phase: str, split: str, case_id: str) -> Path:
    path = output_dir / "case_cache" / phase / split / f"{case_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache_valid(path: Path, signature: str, gains: Sequence[float]) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except Exception:
        return False
    required = {"audit_signature_sha256", "gain", "case_id", "split"}
    if not required.issubset(table.columns):
        return False
    observed = np.sort(table["gain"].to_numpy(dtype=np.float64))
    expected = np.sort(np.asarray(gains, dtype=np.float64))
    return (
        set(table["audit_signature_sha256"]) == {signature}
        and len(observed) == len(expected)
        and np.allclose(observed, expected, atol=1e-12, rtol=0.0)
    )


def _evaluate_case(
    preflight: dict,
    *,
    case_id: str,
    split: str,
    gains: Sequence[float],
    phase: str,
    config: V13ResidualGainAuditConfig,
) -> pd.DataFrame:
    output_dir = preflight["output_dir"]
    signature = preflight["signature"]["audit_signature_sha256"]
    path = _case_cache_path(output_dir, phase, split, case_id)
    if _cache_valid(path, signature, gains):
        return pd.read_csv(path)

    values = _complete_v11_components(preflight["parent"], case_id)
    actual = values["actual"]
    baseline = values["v11_prediction"]
    raw = preflight["residual_function"](values["v13_matrix"])
    correction = values["scale"] * (raw - float(np.mean(raw)))
    residual = actual - baseline
    denominator = float(np.dot(correction, correction))
    numerator = float(np.dot(residual, correction))
    analytic_gain = numerator / denominator if denominator > 1e-18 else 0.0
    common = {
        "iteration": config.iteration,
        "split": split,
        "case_id": case_id,
        "audit_signature_sha256": signature,
        "analytic_unconstrained_sse_gain": analytic_gain,
        "analytic_clipped_0_1_sse_gain": float(np.clip(analytic_gain, 0.0, 1.0)),
        "residual_dot_correction": numerator,
        "correction_squared_sum": denominator,
        "correction_mean_abs_numerical": float(abs(np.mean(correction))),
        "correction_std_mpa": float(np.std(correction, ddof=0)),
        "correction_abs_max_mpa": float(np.max(np.abs(correction))),
    }
    rows = []
    for gain in gains:
        gain = float(gain)
        predicted = baseline + gain * correction
        rows.append({
            **common,
            "gain": gain,
            "gain_label": _gain_label(gain),
            **evaluate_prediction_arrays(actual, predicted),
        })
    result = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, path)
    del values, actual, baseline, raw, correction, residual
    gc.collect()
    return result


def _evaluate_split(
    preflight: dict,
    *,
    split: str,
    case_ids: Sequence[str],
    gains: Sequence[float],
    phase: str,
    config: V13ResidualGainAuditConfig,
) -> pd.DataFrame:
    assert_no_final_cases(case_ids, preflight["inputs"]["manifest"])
    parts = []
    for position, case_id in enumerate(case_ids, start=1):
        print(
            f"[{position}/{len(case_ids)}] {split}: V13 residual gain audit for {case_id}",
            flush=True,
        )
        parts.append(
            _evaluate_case(
                preflight,
                case_id=case_id,
                split=split,
                gains=gains,
                phase=phase,
                config=config,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _aggregate_gain_metrics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    return aggregate_case_metrics(case_metrics, ["iteration", "split", "gain"]).sort_values(
        ["split", "gain"]
    ).reset_index(drop=True)


def _one_gain(summary: pd.DataFrame, gain: float) -> pd.Series:
    rows = summary[np.isclose(summary["gain"], gain, atol=1e-12, rtol=0.0)]
    if len(rows) != 1:
        raise ValueError(f"Expected one gain={gain} row, found {len(rows)}")
    return rows.iloc[0]


def select_validation_gains(
    summary: pd.DataFrame,
    config: V13ResidualGainAuditConfig,
) -> tuple[pd.DataFrame, dict]:
    """Select strict primary and optional research gains on validation only."""

    table = summary.copy()
    baseline = _one_gain(table, 0.0)
    table["macro_rmse_improvement_fraction"] = (
        float(baseline["macro_rmse"]) - table["macro_rmse"]
    ) / max(float(baseline["macro_rmse"]), 1e-12)
    table["macro_rmse_degradation_fraction"] = -table["macro_rmse_improvement_fraction"]
    table["p95_degradation_absolute"] = (
        table["mean_p95_relative_error"] - float(baseline["mean_p95_relative_error"])
    )
    table["p99_underprediction_improvement_absolute"] = (
        float(baseline["mean_p99_underprediction_fraction"])
        - table["mean_p99_underprediction_fraction"]
    )
    table["top1_overlap_degradation_absolute"] = (
        float(baseline["mean_top1pct_hotspot_overlap"])
        - table["mean_top1pct_hotspot_overlap"]
    )
    positive = table["gain"] > 0.0
    numerical_limit = max(1.60, 1.05 * float(baseline["max_prediction_abs_max_ratio"]))

    table["primary_gate_rmse"] = (
        table["macro_rmse_improvement_fraction"]
        >= config.primary_minimum_rmse_improvement_fraction
    )
    table["primary_gate_p95"] = (
        table["p95_degradation_absolute"]
        <= config.primary_maximum_p95_degradation_absolute
    )
    table["primary_gate_p99"] = (
        (table["mean_p99_relative_error"] <= config.maximum_p99_relative_error)
        & (table["mean_p99_underprediction_fraction"] <= config.maximum_p99_underprediction)
        & (table["mean_p99_underprediction_fraction"]
           <= float(baseline["mean_p99_underprediction_fraction"]))
    )
    table["primary_gate_hotspot"] = (
        table["top1_overlap_degradation_absolute"]
        <= config.maximum_top1_overlap_degradation_absolute
    )
    table["primary_gate_recall"] = (
        table["mean_top1_recall_in_predicted_top5"] >= config.minimum_top1_recall
    )
    table["primary_gate_numerical"] = table["max_prediction_abs_max_ratio"] <= numerical_limit
    primary_columns = [column for column in table.columns if column.startswith("primary_gate_")]
    table["primary_eligible"] = positive & table[primary_columns].all(axis=1)

    table["research_gate_rmse"] = (
        table["macro_rmse_degradation_fraction"]
        <= config.research_maximum_rmse_degradation_fraction
    )
    table["research_gate_p95"] = (
        table["p95_degradation_absolute"]
        <= config.research_maximum_p95_degradation_absolute
    )
    table["research_gate_p99_benefit"] = (
        table["p99_underprediction_improvement_absolute"]
        >= config.research_minimum_p99_underprediction_improvement_absolute
    )
    table["research_gate_p99_absolute"] = (
        (table["mean_p99_relative_error"] <= config.maximum_p99_relative_error)
        & (table["mean_p99_underprediction_fraction"] <= config.maximum_p99_underprediction)
    )
    table["research_gate_hotspot"] = (
        table["top1_overlap_degradation_absolute"]
        <= config.maximum_top1_overlap_degradation_absolute
    )
    table["research_gate_recall"] = (
        table["mean_top1_recall_in_predicted_top5"] >= config.minimum_top1_recall
    )
    table["research_gate_numerical"] = table["max_prediction_abs_max_ratio"] <= numerical_limit
    research_columns = [column for column in table.columns if column.startswith("research_gate_")]
    table["research_eligible"] = positive & table[research_columns].all(axis=1)

    primary_pool = table[table["primary_eligible"]]
    if primary_pool.empty:
        primary_gain = 0.0
        primary_status = "retain_v11_no_positive_gain_passes_primary_validation_gates"
    else:
        primary = primary_pool.sort_values(
            ["macro_rmse", "mean_p99_underprediction_fraction", "gain"]
        ).iloc[0]
        primary_gain = float(primary["gain"])
        primary_status = "positive_gain_passes_primary_validation_gates"

    research_pool = table[table["research_eligible"]].copy()
    if research_pool.empty:
        research_gain = 0.0
        research_status = "no_positive_gain_passes_research_validation_gates"
    else:
        # Validation-only Pareto preference: first minimise tail underprediction,
        # then retain P99 accuracy and finally minimise global cost and gain size.
        research = research_pool.sort_values(
            [
                "mean_p99_underprediction_fraction",
                "mean_p99_relative_error",
                "macro_rmse",
                "mean_p95_relative_error",
                "gain",
            ]
        ).iloc[0]
        research_gain = float(research["gain"])
        research_status = "positive_gain_selected_for_research_confirmation"

    table["selected_primary_gain"] = np.isclose(table["gain"], primary_gain)
    table["selected_research_gain"] = np.isclose(table["gain"], research_gain)
    selection = {
        "primary_gain": primary_gain,
        "primary_validation_status": primary_status,
        "research_gain": research_gain,
        "research_validation_status": research_status,
        "selection_scope": "15_complete_validation_cases_only",
        "baseline_gain": 0.0,
        "full_v13_gain": 1.0,
    }
    return table, selection


def _model_label(gain: float, selection: dict) -> str:
    if math.isclose(gain, 0.0, abs_tol=1e-12):
        return "v11_global_gain_0_90"
    if math.isclose(gain, 1.0, abs_tol=1e-12):
        return "v13_full_residual_gain_1_00"
    if math.isclose(gain, float(selection["primary_gain"]), abs_tol=1e-12):
        return f"v13_primary_residual_gain_{gain:.3f}"
    if math.isclose(gain, float(selection["research_gain"]), abs_tol=1e-12):
        return f"v13_research_residual_gain_{gain:.3f}"
    return f"v13_residual_gain_{gain:.3f}"


def _scope_metrics(case_metrics: pd.DataFrame, selection: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = case_metrics.copy()
    rows["model"] = [_model_label(float(gain), selection) for gain in rows["gain"]]
    split_metrics = aggregate_case_metrics(rows, ["iteration", "split", "model", "gain"])
    all_rows = rows.copy()
    all_rows["split"] = "all_149_development"
    all_metrics = aggregate_case_metrics(all_rows, ["iteration", "split", "model", "gain"])
    return rows, pd.concat([split_metrics, all_metrics], ignore_index=True, sort=False)


def _scope_row(scope: pd.DataFrame, split: str, gain: float) -> pd.Series:
    rows = scope[
        scope["split"].eq(split)
        & np.isclose(scope["gain"], gain, atol=1e-12, rtol=0.0)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one scope row for {split}/gain={gain}, found {len(rows)}")
    return rows.iloc[0]


def confirmation_decision(
    scope: pd.DataFrame,
    selection: dict,
    config: V13ResidualGainAuditConfig,
) -> tuple[pd.DataFrame, dict]:
    gates = []
    primary_gain = float(selection["primary_gain"])
    research_gain = float(selection["research_gain"])

    if primary_gain <= 0.0:
        gates.append({
            "decision_layer": "primary",
            "gate": "positive_primary_gain_selected_on_validation",
            "pass": False,
            "observed": primary_gain,
            "criterion": "> 0",
        })
        primary_confirmed = False
    else:
        primary_checks = []
        for split, minimum in [("internal_test", -0.005), ("all_149_development", 0.005)]:
            baseline = _scope_row(scope, split, 0.0)
            candidate = _scope_row(scope, split, primary_gain)
            improvement = (
                float(baseline["macro_rmse"]) - float(candidate["macro_rmse"])
            ) / max(float(baseline["macro_rmse"]), 1e-12)
            primary_checks.extend([
                (f"{split}_macro_rmse", improvement >= minimum, improvement, minimum),
                (
                    f"{split}_p95_preserved",
                    float(candidate["mean_p95_relative_error"])
                    <= float(baseline["mean_p95_relative_error"])
                    + config.primary_maximum_p95_degradation_absolute,
                    float(candidate["mean_p95_relative_error"]),
                    float(baseline["mean_p95_relative_error"])
                    + config.primary_maximum_p95_degradation_absolute,
                ),
                (
                    f"{split}_p99_underprediction",
                    float(candidate["mean_p99_underprediction_fraction"])
                    <= min(
                        config.maximum_p99_underprediction,
                        float(baseline["mean_p99_underprediction_fraction"]),
                    ),
                    float(candidate["mean_p99_underprediction_fraction"]),
                    min(
                        config.maximum_p99_underprediction,
                        float(baseline["mean_p99_underprediction_fraction"]),
                    ),
                ),
            ])
        for name, passed, observed, criterion in primary_checks:
            gates.append({
                "decision_layer": "primary",
                "gate": name,
                "pass": bool(passed),
                "observed": observed,
                "criterion": criterion,
            })
        primary_confirmed = all(item[1] for item in primary_checks)

    if research_gain <= 0.0:
        gates.append({
            "decision_layer": "research",
            "gate": "positive_research_gain_selected_on_validation",
            "pass": False,
            "observed": research_gain,
            "criterion": "> 0",
        })
        research_confirmed = False
    else:
        research_checks = []
        for split, rmse_allowance, p95_allowance in [
            ("internal_test", 0.015, 0.015),
            ("all_149_development", 0.010, 0.010),
        ]:
            baseline = _scope_row(scope, split, 0.0)
            candidate = _scope_row(scope, split, research_gain)
            degradation = (
                float(candidate["macro_rmse"]) - float(baseline["macro_rmse"])
            ) / max(float(baseline["macro_rmse"]), 1e-12)
            p99_benefit = (
                float(baseline["mean_p99_underprediction_fraction"])
                - float(candidate["mean_p99_underprediction_fraction"])
            )
            research_checks.extend([
                (f"{split}_rmse_cost", degradation <= rmse_allowance, degradation, rmse_allowance),
                (
                    f"{split}_p95_cost",
                    float(candidate["mean_p95_relative_error"])
                    <= float(baseline["mean_p95_relative_error"]) + p95_allowance,
                    float(candidate["mean_p95_relative_error"]),
                    float(baseline["mean_p95_relative_error"]) + p95_allowance,
                ),
                (
                    f"{split}_p99_benefit",
                    p99_benefit
                    >= config.research_minimum_p99_underprediction_improvement_absolute,
                    p99_benefit,
                    config.research_minimum_p99_underprediction_improvement_absolute,
                ),
                (
                    f"{split}_p99_absolute",
                    float(candidate["mean_p99_underprediction_fraction"])
                    <= config.maximum_p99_underprediction,
                    float(candidate["mean_p99_underprediction_fraction"]),
                    config.maximum_p99_underprediction,
                ),
                (
                    f"{split}_top1_overlap",
                    float(candidate["mean_top1pct_hotspot_overlap"])
                    >= float(baseline["mean_top1pct_hotspot_overlap"])
                    - config.maximum_top1_overlap_degradation_absolute,
                    float(candidate["mean_top1pct_hotspot_overlap"]),
                    float(baseline["mean_top1pct_hotspot_overlap"])
                    - config.maximum_top1_overlap_degradation_absolute,
                ),
            ])
        for name, passed, observed, criterion in research_checks:
            gates.append({
                "decision_layer": "research",
                "gate": name,
                "pass": bool(passed),
                "observed": observed,
                "criterion": criterion,
            })
        research_confirmed = all(item[1] for item in research_checks)

    decision = {
        "primary_model_decision": (
            f"promote_v13_residual_gain_{primary_gain:.3f}"
            if primary_confirmed
            else "freeze_v11_global_gain_0_90_as_primary_model"
        ),
        "primary_gain": primary_gain,
        "primary_gain_confirmed": bool(primary_confirmed),
        "research_correction_decision": (
            f"retain_v13_gain_{research_gain:.3f}_as_validated_tail_research_correction"
            if research_confirmed
            else "retain_v13_as_non_deployed_tail_correction_research"
        ),
        "research_gain": research_gain,
        "research_gain_confirmed": bool(research_confirmed),
        "proceed_to_iterations_2_4": True,
        "iterations_2_4_role": "V11_method_and_formula_structure_stability_only",
        "final_test_cases_read": 0,
    }
    return pd.DataFrame(gates), decision


def _analytic_gain_summary(validation_cases: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = [
        "case_id",
        "split",
        "analytic_unconstrained_sse_gain",
        "analytic_clipped_0_1_sse_gain",
        "residual_dot_correction",
        "correction_squared_sum",
    ]
    table = validation_cases.drop_duplicates("case_id")[columns].copy()
    numerator = float(table["residual_dot_correction"].sum())
    denominator = float(table["correction_squared_sum"].sum())
    global_gain = numerator / denominator if denominator > 1e-18 else 0.0
    summary = {
        "validation_micro_sse_optimal_gain_unconstrained": global_gain,
        "validation_micro_sse_optimal_gain_clipped_0_1": float(np.clip(global_gain, 0.0, 1.0)),
        "validation_cases_with_positive_optimal_gain": int(
            (table["analytic_unconstrained_sse_gain"] > 0.0).sum()
        ),
        "validation_cases_with_nonpositive_optimal_gain": int(
            (table["analytic_unconstrained_sse_gain"] <= 0.0).sum()
        ),
        "validation_median_unconstrained_gain": float(
            table["analytic_unconstrained_sse_gain"].median()
        ),
    }
    return table, summary


def _verify_parent_reproduction(scope: pd.DataFrame, preflight: dict) -> pd.DataFrame:
    parent = pd.read_csv(preflight["paths"]["v13_scope_metrics"])
    metrics = [
        "macro_rmse",
        "macro_r2",
        "mean_p95_relative_error",
        "mean_p99_relative_error",
        "mean_p99_underprediction_fraction",
        "mean_top1pct_hotspot_overlap",
    ]
    rows = []
    for split in ["validation", "internal_test", "train", "all_149_development"]:
        for gain, parent_model in [
            (0.0, "v11_global_gain_0_90"),
            (1.0, "v13_structural_tail_residual"),
        ]:
            current = _scope_row(scope, split, gain)
            expected = parent[
                parent["split"].eq(split) & parent["model"].eq(parent_model)
            ]
            if len(expected) != 1:
                raise ValueError(f"Missing parent reproduction row for {split}/{parent_model}")
            expected = expected.iloc[0]
            for metric in metrics:
                difference = abs(float(current[metric]) - float(expected[metric]))
                rows.append({
                    "split": split,
                    "gain": gain,
                    "metric": metric,
                    "current": float(current[metric]),
                    "parent": float(expected[metric]),
                    "absolute_difference": difference,
                    # Direct SymPy evaluation can differ from the parent PySR
                    # callable at roughly 1e-8 because of floating-point
                    # operation ordering. A 1e-7 tolerance remains many orders
                    # below any reported engineering metric precision.
                    "pass": difference <= 1e-7,
                })
    result = pd.DataFrame(rows)
    if not result["pass"].all():
        failed = result.loc[~result["pass"]]
        raise RuntimeError(f"V13 gain audit does not reproduce the parent metrics:\n{failed}")
    return result


def _save_formula(preflight: dict, selection: dict, decision: dict) -> None:
    output_dir = preflight["output_dir"]
    parent_formula = (parent_output_directory(preflight["package_root"]) / "selected_deployment_formula.txt").read_text(
        encoding="utf-8"
    )
    residual = str(preflight["selected"]["formula_original_variables"])
    research_gain = float(selection["research_gain"])
    lines = [
        "V13 frozen residual-gain audit",
        "================================",
        "",
        f"Primary decision: {decision['primary_model_decision']}",
        f"Research decision: {decision['research_correction_decision']}",
        "",
        "Primary stress model:",
        "sigma_primary = sigma_V11",
        "",
        "Frozen V13 raw residual:",
        f"r_raw = {residual}",
        "r_centered = r_raw - case_mean(r_raw)",
        "",
        "Optional research correction:",
        f"lambda_research = {research_gain:.6g}",
        "sigma_research = sigma_V11 + lambda_research * scale_case * r_centered",
        "",
        "The optional correction is not the primary stress surrogate unless all primary gates pass.",
        "The 50 final-test case element files were not read.",
        "",
        "Frozen parent formula record:",
        parent_formula,
    ]
    (output_dir / "selected_gain_formula.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame([{
        "primary_model_decision": decision["primary_model_decision"],
        "primary_gain": decision["primary_gain"],
        "research_correction_decision": decision["research_correction_decision"],
        "research_gain": research_gain,
        "residual_formula_original_variables": residual,
        "deployment_expression": "sigma_V11 + lambda * scale_case * (r_raw - case_mean(r_raw))",
    }]).to_csv(output_dir / "selected_gain_formula.csv", index=False)


def _save_plots(
    validation_summary: pd.DataFrame,
    analytic: pd.DataFrame,
    scope: pd.DataFrame,
    selection: dict,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(validation_summary["gain"], validation_summary["macro_rmse"], marker="o")
    axes[0].set_title("Validation macro RMSE")
    axes[1].plot(
        validation_summary["gain"],
        validation_summary["mean_p95_relative_error"],
        marker="o",
        label="P95 error",
    )
    axes[1].plot(
        validation_summary["gain"],
        validation_summary["mean_p99_relative_error"],
        marker="o",
        label="P99 error",
    )
    axes[1].legend()
    axes[1].set_title("Validation tail errors")
    axes[2].plot(
        validation_summary["gain"],
        validation_summary["mean_p99_underprediction_fraction"],
        marker="o",
        label="P99 underprediction",
    )
    axes[2].plot(
        validation_summary["gain"],
        validation_summary["mean_top1pct_hotspot_overlap"],
        marker="o",
        label="Top-1% overlap",
    )
    axes[2].legend()
    axes[2].set_title("Validation tail benefit")
    for axis in axes:
        axis.set_xlabel("V13 residual gain")
        axis.axvline(float(selection["research_gain"]), color="tab:orange", linestyle="--")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_residual_gain_tradeoff.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(analytic["analytic_unconstrained_sse_gain"], bins=12, color="tab:blue")
    axis.axvline(0.0, color="black", linestyle="--")
    axis.set_title("Validation case-wise analytic SSE-optimal residual gains")
    axis.set_xlabel("Unconstrained optimal gain")
    axis.set_ylabel("Cases")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_analytic_gain_distribution.png", dpi=180)
    plt.close(fig)

    gains = sorted({0.0, float(selection["research_gain"]), 1.0})
    all_scope = scope[scope["split"].eq("all_149_development") & scope["gain"].isin(gains)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric, title in [
        (axes[0], "macro_rmse", "All-149 macro RMSE"),
        (axes[1], "mean_p99_underprediction_fraction", "All-149 P99 underprediction"),
        (axes[2], "mean_top1pct_hotspot_overlap", "All-149 Top-1% overlap"),
    ]:
        axis.bar([f"{gain:.3f}" for gain in all_scope["gain"]], all_scope[metric])
        axis.set_title(title)
        axis.set_xlabel("Residual gain")
    fig.tight_layout()
    fig.savefig(output_dir / "all149_selected_gain_comparison.png", dpi=180)
    plt.close(fig)


def run_v13_residual_gain_audit(
    package_root: Path,
    config: V13ResidualGainAuditConfig | None = None,
) -> dict:
    config = config or V13ResidualGainAuditConfig()
    started = time.time()
    preflight = preflight_v13_residual_gain_audit(package_root, config)
    output_dir = preflight["output_dir"]
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "validation_gain_grid",
        "final_test_cases_read": 0,
    })

    validation_cases = _evaluate_split(
        preflight,
        split="validation",
        case_ids=preflight["inputs"]["validation_ids"],
        gains=config.gain_grid,
        phase="validation_grid",
        config=config,
    )
    validation_summary = _aggregate_gain_metrics(validation_cases)
    validation_ranked, selection = select_validation_gains(validation_summary, config)
    validation_cases.to_csv(output_dir / "validation_gain_case_metrics.csv.gz", index=False)
    validation_ranked.to_csv(output_dir / "validation_gain_grid_metrics.csv", index=False)
    _atomic_json(output_dir / "validation_gain_selection.json", selection)

    analytic, analytic_summary = _analytic_gain_summary(validation_cases)
    analytic.to_csv(output_dir / "validation_analytic_optimal_gain_by_case.csv", index=False)
    _atomic_json(output_dir / "validation_analytic_gain_summary.json", analytic_summary)

    confirmation_gains = sorted({
        0.0,
        1.0,
        float(selection["primary_gain"]),
        float(selection["research_gain"]),
    })
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "frozen_gain_development_confirmation",
        "confirmation_gains": confirmation_gains,
        "final_test_cases_read": 0,
    })
    train_cases = _evaluate_split(
        preflight,
        split="train",
        case_ids=preflight["inputs"]["train_ids"],
        gains=confirmation_gains,
        phase="confirmation",
        config=config,
    )
    internal_cases = _evaluate_split(
        preflight,
        split="internal_test",
        case_ids=preflight["inputs"]["internal_ids"],
        gains=confirmation_gains,
        phase="confirmation",
        config=config,
    )
    validation_confirmation = validation_cases[
        validation_cases["gain"].apply(
            lambda value: any(math.isclose(float(value), gain, abs_tol=1e-12) for gain in confirmation_gains)
        )
    ].copy()
    complete_cases = pd.concat(
        [train_cases, validation_confirmation, internal_cases],
        ignore_index=True,
        sort=False,
    )
    if complete_cases["case_id"].nunique() != 149:
        raise RuntimeError("Gain audit did not evaluate all 149 development cases")
    labelled_cases, scope = _scope_metrics(complete_cases, selection)
    labelled_cases.to_csv(output_dir / "selected_gain_case_metrics.csv.gz", index=False)
    scope.to_csv(output_dir / "selected_gain_scope_metrics.csv", index=False)

    reproduction = _verify_parent_reproduction(scope, preflight)
    reproduction.to_csv(output_dir / "parent_v13_reproduction_checks.csv", index=False)
    gates, decision = confirmation_decision(scope, selection, config)
    gates.to_csv(output_dir / "gain_confirmation_gates.csv", index=False)
    _atomic_json(output_dir / "residual_gain_decision.json", decision)
    _save_formula(preflight, selection, decision)
    _save_plots(validation_ranked, analytic, scope, selection, output_dir)

    complete = {
        "status": "complete",
        "iteration": 1,
        "parent_candidate_index": int(preflight["selected"]["candidate_index"]),
        "validation_gain_count": len(config.gain_values),
        "validation_cases": 15,
        "internal_cases": 15,
        "train_cases": 119,
        "development_cases_evaluated": 149,
        "primary_model_decision": decision["primary_model_decision"],
        "research_correction_decision": decision["research_correction_decision"],
        "primary_gain": decision["primary_gain"],
        "research_gain": decision["research_gain"],
        "proceed_to_iterations_2_4": decision["proceed_to_iterations_2_4"],
        "final_test_cases_read": 0,
        "elapsed_seconds": time.time() - started,
        "output_directory": str(output_dir),
    }
    _atomic_json(output_dir / "gain_audit_complete.json", complete)
    _atomic_json(output_dir / "run_status.json", {**complete, "stage": "complete"})
    return complete


__all__ = [
    "V13ResidualGainAuditConfig",
    "output_directory",
    "preflight_v13_residual_gain_audit",
    "run_v13_residual_gain_audit",
]
