"""Frozen V11 tail-gain and stability audit.

This supplement does not run PySR and does not change the formula discovered
by V11.  It evaluates a scalar gain on V11's centered local-tail correction:

    sigma(lambda) = sigma_v10 + delta_mu_case
                    + lambda * scale_v10 * centered_tail_v11

The gain is selected on all elements of the 15 validation cases.  It is then
evaluated once on all elements of the 15 internal-test cases.  The 50 final
test cases are inventoried by the frozen manifest but their element files are
never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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
    _tail_adaptation,
)
from signed_staged_residual_symbolic import _atomic_json  # noqa: E402
from v11_tail_aware_localised_symbolic import (  # noqa: E402
    V11PilotConfig,
    _full_case_components,
    _mean_correction,
    build_v11_feature_matrix,
    preflight_v11_pilot,
)


AUDIT_OUTPUT_NAME = "12_v11_gain_stability_audit"
V11_OUTPUT_NAME = "11_tail_aware_localised_symbolic"


@dataclass(frozen=True)
class V11GainAuditConfig:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    gain_min: float = 0.0
    gain_max: float = 1.20
    gain_step: float = 0.05
    max_selectable_gain: float = 1.0
    bootstrap_resamples: int = 20_000
    random_seed: int = RANDOM_SEED

    def validate(self) -> None:
        if self.iteration not in {1, 2, 3, 4}:
            raise ValueError("V11 gain-audit iteration must be one of 1, 2, 3 or 4")
        if self.gain_min < 0.0:
            raise ValueError("Tail gain must not be negative")
        if self.gain_max < self.gain_min:
            raise ValueError("gain_max must be at least gain_min")
        if self.gain_step <= 0.0:
            raise ValueError("gain_step must be positive")
        if not self.gain_min <= self.max_selectable_gain <= self.gain_max:
            raise ValueError("max_selectable_gain must lie inside the audit grid")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("Use at least 1,000 case-level bootstrap resamples")
        grid = self.gain_grid
        for required in (0.0, 1.0):
            if not np.any(np.isclose(grid, required, atol=1e-12, rtol=0.0)):
                raise ValueError(f"Gain grid must contain lambda={required:.1f}")

    @property
    def gain_grid(self) -> np.ndarray:
        count = int(round((self.gain_max - self.gain_min) / self.gain_step))
        values = self.gain_min + self.gain_step * np.arange(count + 1)
        return np.round(values, 10)


def output_directory(package_root: Path, config: V11GainAuditConfig) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / AUDIT_OUTPUT_NAME
        / config.output_subdir
    )


def v11_output_directory(package_root: Path, iteration: int = 1) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / V11_OUTPUT_NAME
        / f"iteration_{iteration}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v11_required_paths(package_root: Path, iteration: int = 1) -> dict[str, Path]:
    root = v11_output_directory(package_root, iteration)
    return {
        "v11_completion": root / "pilot_complete.json",
        "v11_composite": root / "selected_composite_formula.csv",
        "v11_case_metrics": root / "selected_models_case_metrics.csv.gz",
        "v11_tail_formula": root / "stages" / "stage_tail_localised" / "selected_formula.csv",
        "v11_tail_scaling": root / "training_cache" / "v11_tail_scaling.csv",
        "v11_rbf_centres": root / "training_cache" / "hotspot_rbf_centres.csv",
        "v11_mean_model": root / "training_cache" / "mean_calibration_model.json",
    }


def preflight_v11_gain_audit(
    package_root: Path,
    config: V11GainAuditConfig,
) -> dict:
    """Load and hash the exact frozen V11 state used by the audit."""

    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This performs the existing V10/V11 provenance and frozen-split checks.
    parent = preflight_v11_pilot(
        package_root,
        V11PilotConfig(
            iteration=config.iteration,
            output_subdir=f"iteration_{config.iteration}",
        ),
    )
    inputs = parent["inputs"]
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )

    paths = _v11_required_paths(package_root, config.iteration)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The frozen V11 audit inputs are incomplete:\n" + "\n".join(missing)
        )
    completion = json.loads(paths["v11_completion"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError("V11 pilot_complete.json is not marked complete")
    if int(completion.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError("The parent V11 run did not preserve the final-test seal")

    composite_table = pd.read_csv(paths["v11_composite"])
    tail_table = pd.read_csv(paths["v11_tail_formula"])
    if len(composite_table) != 1 or len(tail_table) != 1:
        raise ValueError("Expected exactly one frozen V11 composite and tail formula")
    selected_candidate = int(completion["selected_candidate_index"])
    tail_candidate = int(tail_table.iloc[0]["candidate_index"])
    if selected_candidate != tail_candidate:
        raise ValueError("V11 completion marker and selected tail formula disagree")

    scaling = _load_scaling(paths["v11_tail_scaling"])
    centres = pd.read_csv(paths["v11_rbf_centres"])
    mean_model = json.loads(paths["v11_mean_model"].read_text(encoding="utf-8"))
    mean_model["coefficients"] = np.asarray(mean_model["coefficients"], dtype=np.float64)
    tail_function = _compiled_formula(tail_table.iloc[0], scaling)

    hash_rows = []
    for artifact, path in paths.items():
        hash_rows.append({
            "artifact": artifact,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    hash_audit = pd.DataFrame(hash_rows)
    hash_audit.to_csv(output_dir / "frozen_v11_hash_audit.csv", index=False)

    checks = pd.DataFrame([
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "parent_v11_complete", "value": completion.get("status"), "expected": "complete"},
        {"check": "parent_final_cases_read", "value": completion.get("final_test_cases_read"), "expected": 0},
        {"check": "selected_candidate_match", "value": tail_candidate, "expected": selected_candidate},
        {"check": "gain_grid_values", "value": len(config.gain_grid), "expected": 25},
        {"check": "gain_one_selectable", "value": config.max_selectable_gain >= 1.0, "expected": True},
    ])
    checks["pass"] = checks["value"] == checks["expected"]
    checks.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not checks["pass"].all():
        failed = checks.loc[~checks["pass"], "check"].tolist()
        raise RuntimeError(f"V11 gain-audit preflight failed: {failed}")

    signature_payload = {
        "config": asdict(config),
        "frozen_v11_sha256": dict(zip(hash_audit["artifact"], hash_audit["sha256"])),
        "validation_ids": inputs["validation_ids"],
        "internal_test_ids": inputs["internal_ids"],
        "final_test_ids_inventoried_not_read": inputs["final_ids"],
    }
    signature_payload["audit_signature_sha256"] = _canonical_hash(signature_payload)
    signature_path = output_dir / "audit_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature_payload:
            raise RuntimeError(
                "Existing gain-audit output has a different input/configuration signature"
            )
    else:
        _atomic_json(signature_path, signature_payload)

    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "v10": parent["v10"],
        "completion": completion,
        "paths": paths,
        "composite": composite_table.iloc[0],
        "tail_row": tail_table.iloc[0],
        "tail_function": tail_function,
        "scaling": scaling,
        "centres": centres,
        "mean_model": mean_model,
        "hash_audit": hash_audit,
        "checks": checks,
        "signature": signature_payload,
    }


def _gain_label(gain: float) -> str:
    return f"gain_{gain:.2f}"


def _cache_path(output_dir: Path, split: str, case_id: str) -> Path:
    path = output_dir / "case_cache" / split / f"{case_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache_is_valid(
    path: Path,
    *,
    split: str,
    case_id: str,
    gains: Sequence[float],
    signature: str,
) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except Exception:
        return False
    required = {"split", "case_id", "model_family", "gain", "audit_signature_sha256"}
    if not required.issubset(table.columns):
        return False
    if set(table["split"]) != {split} or set(table["case_id"]) != {case_id}:
        return False
    if set(table["audit_signature_sha256"]) != {signature}:
        return False
    v10_count = int(table["model_family"].eq("v10_reference").sum())
    observed = np.sort(
        table.loc[table["model_family"].eq("v11_gain"), "gain"].to_numpy(dtype=float)
    )
    expected = np.sort(np.asarray(gains, dtype=float))
    return v10_count == 1 and len(observed) == len(expected) and np.allclose(
        observed, expected, atol=1e-12, rtol=0.0
    )


def _evaluate_case(
    preflight: dict,
    *,
    case_id: str,
    split: str,
    gains: Sequence[float],
    config: V11GainAuditConfig,
) -> pd.DataFrame:
    path = _cache_path(preflight["output_dir"], split, case_id)
    signature = preflight["signature"]["audit_signature_sha256"]
    if _cache_is_valid(
        path,
        split=split,
        case_id=case_id,
        gains=gains,
        signature=signature,
    ):
        return pd.read_csv(path)

    frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
    summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    matrix, scale, v10_prediction, components = _full_case_components(
        frame, summary, preflight["v10"]
    )
    delta_mean = _mean_correction(summary, preflight["mean_model"])
    mean_prediction = v10_prediction + delta_mean
    v11_matrix = build_v11_feature_matrix(
        matrix,
        preflight["v10"],
        preflight["centres"],
        components,
    )
    tail_raw = preflight["tail_function"](v11_matrix)
    centered_tail = tail_raw - float(np.mean(tail_raw))
    tail_correction_mpa = scale * centered_tail

    common = {
        "iteration": config.iteration,
        "split": split,
        "case_id": case_id,
        "audit_signature_sha256": signature,
        "case_mean_correction_mpa": delta_mean,
        "v10_case_scale": scale,
        "centered_tail_mean_abs_numerical": float(abs(np.mean(centered_tail))),
        "tail_correction_std_mpa": float(np.std(tail_correction_mpa, ddof=0)),
        "tail_correction_abs_max_mpa": float(np.max(np.abs(tail_correction_mpa))),
    }
    rows = [{
        **common,
        "model_family": "v10_reference",
        "gain": np.nan,
        "gain_label": "v10_reference",
        **evaluate_prediction_arrays(actual, v10_prediction),
    }]
    for gain in gains:
        gain = float(gain)
        predicted = mean_prediction + gain * tail_correction_mpa
        rows.append({
            **common,
            "model_family": "v11_gain",
            "gain": gain,
            "gain_label": _gain_label(gain),
            **evaluate_prediction_arrays(actual, predicted),
        })
    result = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, path)
    del (
        frame,
        actual,
        matrix,
        components,
        v10_prediction,
        mean_prediction,
        v11_matrix,
        tail_raw,
        centered_tail,
        tail_correction_mpa,
    )
    gc.collect()
    return result


def _evaluate_split(
    preflight: dict,
    *,
    split: str,
    case_ids: Sequence[str],
    gains: Sequence[float],
    config: V11GainAuditConfig,
) -> pd.DataFrame:
    assert_no_final_cases(case_ids, preflight["inputs"]["manifest"])
    parts = []
    for position, case_id in enumerate(case_ids, start=1):
        print(
            f"[{position}/{len(case_ids)}] {split}: full-case gain audit for {case_id}",
            flush=True,
        )
        parts.append(
            _evaluate_case(
                preflight,
                case_id=case_id,
                split=split,
                gains=gains,
                config=config,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _aggregate_gain_metrics(case_metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    gains = case_metrics[case_metrics["model_family"].eq("v11_gain")].copy()
    summary = aggregate_case_metrics(gains, ["iteration", "split", "gain"])
    extra = (
        gains.groupby(["iteration", "split", "gain"], observed=True)
        .agg(
            mean_max_relative_error=("max_relative_error", "mean"),
            worst_max_relative_error=("max_relative_error", "max"),
            mean_prediction_abs_max_ratio=("prediction_abs_max_ratio", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(extra, on=["iteration", "split", "gain"], how="left")
    adaptations = []
    for gain, group in gains.groupby("gain", observed=True):
        adaptations.append({"gain": float(gain), **_tail_adaptation(group)})
    summary = summary.merge(pd.DataFrame(adaptations), on="gain", how="left")
    summary["gain_label"] = summary["gain"].map(_gain_label)
    if set(summary["split"]) != {split}:
        raise AssertionError("Gain aggregation produced an unexpected split")
    return summary.sort_values("gain").reset_index(drop=True)


def _v10_aggregate(case_metrics: pd.DataFrame) -> pd.Series:
    v10 = case_metrics[case_metrics["model_family"].eq("v10_reference")]
    if v10["case_id"].nunique() != 15:
        raise ValueError("Expected one V10 reference row for each of 15 cases")
    return aggregate_case_metrics(v10, ["iteration", "split", "model_family"]).iloc[0]


def score_and_select_gain(
    validation_summary: pd.DataFrame,
    validation_cases: pd.DataFrame,
    config: V11GainAuditConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rank selectable gains on validation only; lower score is better."""

    table = validation_summary.copy()
    v10 = _v10_aggregate(validation_cases)
    table["macro_rmse_improvement_vs_v10_fraction"] = (
        float(v10["macro_rmse"]) - table["macro_rmse"]
    ) / max(float(v10["macro_rmse"]), 1e-12)
    table["max_ratio_deviation_from_one"] = abs(
        table["max_prediction_abs_max_ratio"] - 1.0
    )
    table["selectable_gain"] = table["gain"] <= config.max_selectable_gain + 1e-12

    table["gate_numerical_guardrail"] = table["max_prediction_abs_max_ratio"] <= 5.0
    table["gate_positive_macro_r2"] = table["macro_r2"] > 0.0
    table["gate_improves_v10_rmse"] = table["macro_rmse_improvement_vs_v10_fraction"] > 0.0
    table["gate_improves_v10_rmse_by_1pct"] = table[
        "macro_rmse_improvement_vs_v10_fraction"
    ] >= 0.01
    table["gate_p95_relative_error"] = table["mean_p95_relative_error"] <= 0.15
    table["gate_p99_relative_error"] = table["mean_p99_relative_error"] <= 0.15
    table["gate_p99_underprediction"] = (
        table["mean_p99_underprediction_fraction"] <= 0.10
    )
    table["gate_top1_hotspot_overlap"] = table["mean_top1pct_hotspot_overlap"] >= 0.60
    table["gate_top1_recall"] = table["mean_top1_recall_in_predicted_top5"] >= 0.80
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
    table["all_v11_promotion_gates_pass"] = table[gate_columns].all(axis=1)
    table["eligible_for_automatic_selection"] = (
        table["selectable_gain"] & table["gate_numerical_guardrail"]
    )

    selectable_index = table.index[table["selectable_gain"]]
    table["engineering_selection_score"] = np.nan
    error_metrics = [
        ("macro_rmse", 0.18),
        ("mean_top5_actual_rmse", 0.15),
        ("mean_p95_relative_error", 0.10),
        ("mean_p99_relative_error", 0.15),
        ("mean_p99_underprediction_fraction", 0.12),
        ("max_ratio_deviation_from_one", 0.10),
    ]
    benefit_metrics = [
        ("mean_top1pct_hotspot_overlap", 0.15),
        ("mean_top1_recall_in_predicted_top5", 0.05),
    ]
    score = pd.Series(0.0, index=selectable_index)
    for column, weight in error_metrics:
        score += weight * table.loc[selectable_index, column].rank(
            pct=True, ascending=True, method="average"
        )
    for column, weight in benefit_metrics:
        score += weight * table.loc[selectable_index, column].rank(
            pct=True, ascending=False, method="average"
        )
    table.loc[selectable_index, "engineering_selection_score"] = score

    full_gate_pool = table[
        table["selectable_gain"] & table["all_v11_promotion_gates_pass"]
    ]
    if not full_gate_pool.empty:
        pool = full_gate_pool
        pool_status = "all_v11_promotion_gates"
    else:
        improved = table[
            table["selectable_gain"]
            & table["gate_numerical_guardrail"]
            & table["gate_positive_macro_r2"]
            & table["gate_improves_v10_rmse"]
        ]
        if not improved.empty:
            pool = improved
            pool_status = "finite_positive_r2_and_improves_v10"
        else:
            pool = table[table["selectable_gain"] & table["gate_numerical_guardrail"]]
            if pool.empty:
                raise RuntimeError("No finite selectable V11 gain remains")
            pool_status = "diagnostic_selectable_gains_only"

    selected = pool.sort_values(
        [
            "engineering_selection_score",
            "mean_p99_underprediction_fraction",
            "mean_top1pct_hotspot_overlap",
            "max_ratio_deviation_from_one",
            "macro_rmse",
            "gain",
        ],
        ascending=[True, True, False, True, True, True],
    ).iloc[0].copy()
    selected["selection_pool_status"] = pool_status
    selected["selection_method"] = (
        "validation-only weighted engineering rank; lambda above 1.00 is "
        "diagnostic and cannot be selected"
    )
    selected["promotion_status"] = (
        "passes_all_v11_pilot_gates"
        if bool(selected["all_v11_promotion_gates_pass"])
        else "diagnostic_only_does_not_pass_all_v11_pilot_gates"
    )
    table["selected_gain"] = np.isclose(
        table["gain"], float(selected["gain"]), atol=1e-12, rtol=0.0
    )
    return table, selected


def _model_rows(
    case_metrics: pd.DataFrame,
    selected_gain: float,
) -> pd.DataFrame:
    """Create explicit reporting roles, including current V11 and selected gain."""

    parts = []
    v10 = case_metrics[case_metrics["model_family"].eq("v10_reference")].copy()
    v10["model"] = "v10_signed_staged_symbolic"
    v10["reporting_role"] = "parent_reference"
    parts.append(v10)

    reporting = [
        (0.0, "v11_mean_calibrated_v10", "mean_calibration_only"),
        (1.0, "v11_current_gain_1_00", "current_v11"),
        (selected_gain, f"v11_selected_gain_{selected_gain:.2f}", "validation_selected"),
    ]
    for gain, model, role in reporting:
        rows = case_metrics[
            case_metrics["model_family"].eq("v11_gain")
            & np.isclose(case_metrics["gain"], gain, atol=1e-12, rtol=0.0)
        ].copy()
        if rows.empty:
            raise ValueError(f"Missing reporting gain {gain:.2f}")
        rows["model"] = model
        rows["reporting_role"] = role
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def _verify_parent_v11_reproduction(
    preflight: dict,
    validation_cases: pd.DataFrame,
    internal_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Require lambda=1 to reproduce the frozen parent V11 case metrics."""

    current = pd.concat([validation_cases, internal_cases], ignore_index=True)
    current = current[
        current["model_family"].eq("v11_gain")
        & np.isclose(current["gain"], 1.0, atol=1e-12, rtol=0.0)
    ].copy()
    parent = pd.read_csv(
        preflight["paths"]["v11_case_metrics"],
        compression="gzip",
    )
    parent = parent[
        parent["model"].eq("v11_tail_aware_localised_symbolic")
        & parent["split"].isin(["validation", "internal_test"])
    ].copy()
    metrics = [
        "rmse",
        "r2",
        "predicted_mean",
        "predicted_p95",
        "predicted_p99",
        "predicted_max",
        "p95_relative_error",
        "p99_relative_error",
        "p99_underprediction_fraction",
        "prediction_abs_max_ratio",
        "top5pct_hotspot_overlap",
        "top1pct_hotspot_overlap",
        "top1_recall_in_predicted_top5",
    ]
    merged = current[["split", "case_id", *metrics]].merge(
        parent[["split", "case_id", *metrics]],
        on=["split", "case_id"],
        how="outer",
        suffixes=("_recomputed", "_parent"),
        indicator=True,
    )
    if len(merged) != 30 or not merged["_merge"].eq("both").all():
        raise RuntimeError("Could not pair all 30 frozen V11 case metrics")
    audit_rows = []
    for row in merged.itertuples(index=False):
        for metric in metrics:
            recomputed = float(getattr(row, f"{metric}_recomputed"))
            frozen = float(getattr(row, f"{metric}_parent"))
            absolute_difference = abs(recomputed - frozen)
            tolerance = 1e-6 * max(1.0, abs(frozen))
            audit_rows.append({
                "split": row.split,
                "case_id": row.case_id,
                "metric": metric,
                "recomputed_lambda_1": recomputed,
                "frozen_parent_v11": frozen,
                "absolute_difference": absolute_difference,
                "tolerance": tolerance,
                "pass": absolute_difference <= tolerance,
            })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(preflight["output_dir"] / "v11_reproduction_audit.csv", index=False)
    if not audit["pass"].all():
        failed = audit.loc[~audit["pass"], ["split", "case_id", "metric"]]
        raise RuntimeError(
            "Lambda=1 did not reproduce the frozen parent V11 metrics: "
            + failed.head(10).to_dict(orient="records").__repr__()
        )
    return audit


def _bootstrap_mean_delta(
    delta: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1_000
    cursor = 0
    while cursor < resamples:
        size = min(chunk, resamples - cursor)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[cursor : cursor + size] = values[indices].mean(axis=1)
        cursor += size
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_case_stability(
    reporting_cases: pd.DataFrame,
    selected_model: str,
    config: V11GainAuditConfig,
) -> pd.DataFrame:
    """Case-paired uncertainty summaries; descriptive, not a promotion rule."""

    comparisons = [
        ("v10_signed_staged_symbolic", "selected_vs_v10"),
        ("v11_current_gain_1_00", "selected_vs_current_v11"),
    ]
    metrics = [
        ("rmse", "lower_is_better"),
        ("top5_actual_rmse", "lower_is_better"),
        ("p95_relative_error", "lower_is_better"),
        ("p99_relative_error", "lower_is_better"),
        ("p99_underprediction_fraction", "lower_is_better"),
        ("top1pct_hotspot_overlap", "higher_is_better"),
        ("top1_recall_in_predicted_top5", "higher_is_better"),
        ("prediction_abs_max_ratio", "closer_to_one_is_better"),
    ]
    records = []
    for split in ["validation", "internal_test"]:
        split_table = reporting_cases[reporting_cases["split"].eq(split)]
        selected = split_table[split_table["model"].eq(selected_model)].set_index("case_id")
        for reference_model, comparison in comparisons:
            reference = split_table[split_table["model"].eq(reference_model)].set_index("case_id")
            common = selected.index.intersection(reference.index)
            if len(common) != 15:
                raise ValueError(f"Expected 15 paired cases for {split}/{comparison}")
            for metric_index, (metric, direction) in enumerate(metrics):
                delta = (
                    selected.loc[common, metric].to_numpy(dtype=np.float64)
                    - reference.loc[common, metric].to_numpy(dtype=np.float64)
                )
                ci_low, ci_high = _bootstrap_mean_delta(
                    delta,
                    resamples=config.bootstrap_resamples,
                    seed=config.random_seed + 100 * metric_index + (0 if split == "validation" else 10_000),
                )
                if np.allclose(delta, 0.0, atol=1e-15, rtol=0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                records.append({
                    "split": split,
                    "comparison": comparison,
                    "selected_model": selected_model,
                    "reference_model": reference_model,
                    "metric": metric,
                    "direction": direction,
                    "n_paired_cases": len(common),
                    "mean_delta_selected_minus_reference": float(np.mean(delta)),
                    "median_delta_selected_minus_reference": float(np.median(delta)),
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_two_sided_p": p_value,
                    "descriptive_only": True,
                })
    return pd.DataFrame(records)


def _save_worst_case_diagnostics(reporting_cases: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records = []
    for (split, model), group in reporting_cases.groupby(["split", "model"], observed=True):
        criteria = [
            ("rmse", False),
            ("p99_underprediction_fraction", False),
            ("max_relative_error", False),
            ("top1pct_hotspot_overlap", True),
        ]
        for metric, ascending in criteria:
            ranked = group.sort_values(metric, ascending=ascending).head(5)
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                records.append({
                    "split": split,
                    "model": model,
                    "criterion": metric,
                    "rank": rank,
                    "case_id": row.case_id,
                    "value": float(getattr(row, metric)),
                    "rmse": float(row.rmse),
                    "actual_p99": float(row.actual_p99),
                    "predicted_p99": float(row.predicted_p99),
                    "actual_max": float(row.actual_max),
                    "predicted_max": float(row.predicted_max),
                    "prediction_abs_max_ratio": float(row.prediction_abs_max_ratio),
                    "top1pct_hotspot_overlap": float(row.top1pct_hotspot_overlap),
                })
    result = pd.DataFrame(records)
    result.to_csv(output_dir / "worst_case_diagnostics.csv", index=False)
    return result


def _save_plots(
    validation_summary: pd.DataFrame,
    reporting_cases: pd.DataFrame,
    selected_gain: float,
    output_dir: Path,
) -> None:
    metrics = [
        ("macro_rmse", "Validation macro RMSE", None),
        ("mean_p95_relative_error", "Mean P95 relative error", 0.15),
        ("mean_p99_underprediction_fraction", "Mean P99 underprediction", 0.10),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap", 0.60),
        ("max_prediction_abs_max_ratio", "Worst maximum-stress ratio", None),
        ("worst_case_rmse", "Worst-case RMSE", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (column, title, threshold) in zip(axes.flat, metrics):
        axis.plot(validation_summary["gain"], validation_summary[column], "o-", color="#246b73")
        axis.axvline(1.0, color="#777777", linestyle="--", label="current V11")
        axis.axvline(selected_gain, color="#c44e35", linestyle="-", label="selected")
        if threshold is not None:
            axis.axhline(threshold, color="#b8860b", linestyle=":", label="gate")
        axis.axvspan(1.0, validation_summary["gain"].max(), color="#dddddd", alpha=0.35)
        axis.set_title(title)
        axis.set_xlabel("Tail gain lambda")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "validation_gain_tradeoff.png", dpi=180)
    plt.close(fig)

    selected_model = f"v11_selected_gain_{selected_gain:.2f}"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column_index, split in enumerate(["validation", "internal_test"]):
        subset = reporting_cases[reporting_cases["split"].eq(split)]
        selected = subset[subset["model"].eq(selected_model)].set_index("case_id")
        v10 = subset[subset["model"].eq("v10_signed_staged_symbolic")].set_index("case_id")
        common = selected.index.intersection(v10.index)
        rmse_delta = selected.loc[common, "rmse"] - v10.loc[common, "rmse"]
        overlap_delta = (
            selected.loc[common, "top1pct_hotspot_overlap"]
            - v10.loc[common, "top1pct_hotspot_overlap"]
        )
        axes[0, column_index].bar(common, rmse_delta, color="#246b73")
        axes[0, column_index].axhline(0.0, color="#555555", linewidth=1)
        axes[0, column_index].set_title(f"{split}: RMSE delta vs V10")
        axes[0, column_index].tick_params(axis="x", rotation=90)
        axes[1, column_index].bar(common, overlap_delta, color="#cf6b32")
        axes[1, column_index].axhline(0.0, color="#555555", linewidth=1)
        axes[1, column_index].set_title(f"{split}: top-1% overlap delta vs V10")
        axes[1, column_index].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_gain_case_deltas.png", dpi=180)
    plt.close(fig)


def _save_formula(preflight: dict, selected: pd.Series) -> dict:
    output_dir = preflight["output_dir"]
    gain = float(selected["gain"])
    v10_formula = str(preflight["v10"]["composite"]["composite_stress_formula"])
    log_scale = str(preflight["v10"]["composite"]["case_log_scale_formula"])
    mean_formula = str(preflight["mean_model"].get("formula_original_variables"))
    tail_formula = str(preflight["tail_row"]["formula_original_variables"])
    centered_tail = f"({tail_formula}) - case_mean({tail_formula})"
    combined = (
        f"({v10_formula}) + ({mean_formula}) + ({gain:.16g}) * "
        f"exp({log_scale}) * ({centered_tail})"
    )
    record = {
        "iteration": int(preflight["completion"]["iteration"]),
        "method": "frozen_v11_validation_selected_tail_gain",
        "selected_tail_gain": gain,
        "v10_formula": v10_formula,
        "case_mean_calibration_formula": mean_formula,
        "v10_log_scale_formula": log_scale,
        "v11_tail_raw_formula": tail_formula,
        "tail_centering_rule": "tail_raw_minus_complete_case_predictor_only_mean",
        "combined_stress_formula": combined,
        "selection_pool_status": selected["selection_pool_status"],
        "promotion_status": selected["promotion_status"],
    }
    pd.DataFrame([record]).to_csv(output_dir / "selected_gain_formula.csv", index=False)
    lines = [
        "CT3 frozen V11 gain-audited stress formula",
        "===========================================",
        "",
        f"Selected tail gain lambda = {gain:.2f}",
        "",
        f"sigma_v10 = {v10_formula}",
        f"delta_mu_case = {mean_formula}",
        f"log_scale_v10 = {log_scale}",
        f"tail_raw_v11 = {tail_formula}",
        "tail_centered_v11 = tail_raw_v11 - case_mean(tail_raw_v11)",
        "",
        "Combined expression",
        f"sigma = {combined}",
        "",
        "The selected lambda used only the 15 validation cases.",
        "The 15 internal-test cases were read only after lambda was frozen.",
        "The 50 final-test case element files were not read.",
        "Lambda values above 1.00 were diagnostic and were not selectable.",
    ]
    (output_dir / "selected_gain_formula.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return record


def run_v11_gain_audit(
    package_root: Path,
    config: V11GainAuditConfig,
    *,
    preflight: dict | None = None,
) -> dict:
    started = time.time()
    preflight = preflight or preflight_v11_gain_audit(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "audit_complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("audit_signature_sha256") != preflight["signature"][
            "audit_signature_sha256"
        ]:
            raise RuntimeError("Completed audit has a different input signature")
        return completion

    _atomic_json(output_dir / "run_configuration.json", asdict(config))
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "validation_gain_scan",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_test_cases_read": 0,
        "audit_signature_sha256": preflight["signature"]["audit_signature_sha256"],
    })
    try:
        validation_cases = _evaluate_split(
            preflight,
            split="validation",
            case_ids=preflight["inputs"]["validation_ids"],
            gains=config.gain_grid,
            config=config,
        )
        validation_cases.to_csv(
            output_dir / "validation_gain_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        validation_summary = _aggregate_gain_metrics(validation_cases, "validation")
        validation_summary, selected = score_and_select_gain(
            validation_summary, validation_cases, config
        )
        validation_summary.to_csv(output_dir / "validation_gain_summary.csv", index=False)
        pd.DataFrame([selected]).to_csv(output_dir / "selected_gain.csv", index=False)
        selected_gain = float(selected["gain"])
        _atomic_json(output_dir / "selected_gain.json", {
            "selected_gain": selected_gain,
            "selection_pool_status": selected["selection_pool_status"],
            "promotion_status": selected["promotion_status"],
            "all_v11_promotion_gates_pass": bool(selected["all_v11_promotion_gates_pass"]),
            "validation_only_selection": True,
            "gains_above_one_selectable": False,
            "final_test_cases_read": 0,
        })

        internal_gains = sorted({0.0, 1.0, selected_gain})
        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "internal_test_confirmation",
            "selected_gain": selected_gain,
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        internal_cases = _evaluate_split(
            preflight,
            split="internal_test",
            case_ids=preflight["inputs"]["internal_ids"],
            gains=internal_gains,
            config=config,
        )
        internal_cases.to_csv(
            output_dir / "internal_test_gain_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        reproduction_audit = _verify_parent_v11_reproduction(
            preflight,
            validation_cases,
            internal_cases,
        )

        validation_reporting = _model_rows(validation_cases, selected_gain)
        internal_reporting = _model_rows(internal_cases, selected_gain)
        reporting_cases = pd.concat(
            [validation_reporting, internal_reporting], ignore_index=True
        )
        reporting_cases.to_csv(
            output_dir / "selected_models_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        split_metrics = aggregate_case_metrics(
            reporting_cases,
            ["iteration", "split", "model", "reporting_role"],
        )
        split_metrics.to_csv(output_dir / "selected_models_split_metrics.csv", index=False)
        adaptation_rows = []
        for (split, model, role), group in reporting_cases.groupby(
            ["split", "model", "reporting_role"], observed=True
        ):
            adaptation_rows.append({
                "split": split,
                "model": model,
                "reporting_role": role,
                **_tail_adaptation(group),
            })
        pd.DataFrame(adaptation_rows).to_csv(
            output_dir / "selected_models_tail_adaptation.csv", index=False
        )

        selected_model = f"v11_selected_gain_{selected_gain:.2f}"
        stability = paired_case_stability(reporting_cases, selected_model, config)
        stability.to_csv(output_dir / "paired_case_stability_summary.csv", index=False)
        _save_worst_case_diagnostics(reporting_cases, output_dir)
        _save_formula(preflight, selected)
        _save_plots(validation_summary, reporting_cases, selected_gain, output_dir)

        result = {
            "status": "complete",
            "iteration": config.iteration,
            "supplement_to": "V11 tail-aware localised symbolic prototype",
            "pySR_search_run": False,
            "validation_cases": len(preflight["inputs"]["validation_ids"]),
            "internal_test_cases": len(preflight["inputs"]["internal_ids"]),
            "validation_gain_count": len(config.gain_grid),
            "selected_gain": selected_gain,
            "parent_v11_reproduction_checks": len(reproduction_audit),
            "parent_v11_reproduction_pass": bool(reproduction_audit["pass"].all()),
            "selection_pool_status": selected["selection_pool_status"],
            "promotion_status": selected["promotion_status"],
            "final_test_cases_read": 0,
            "audit_signature_sha256": preflight["signature"]["audit_signature_sha256"],
            "elapsed_seconds": time.time() - started,
            "output_directory": str(output_dir),
        }
        _atomic_json(completion_path, result)
        _atomic_json(output_dir / "run_status.json", {**result, "stage": "complete"})
        return result
    except Exception as exc:
        _atomic_json(output_dir / "run_status.json", {
            "status": "failed",
            "stage": "v11_gain_stability_audit",
            "error": repr(exc),
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        raise
