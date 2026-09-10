"""V12 bounded case-adaptive tail-gain calibration.

V12 keeps the V10/V11 symbolic structure frozen and replaces V11's single
global tail gain with a predictor-only, case-dependent coefficient:

    sigma = sigma_v10 + delta_mu_case
            + lambda_case(context) * scale_v10 * centered_tail_v11

The gain layer is deliberately small and transparent.  Oracle gains are
derived independently for the 149 development cases with a fixed tail-aware
Huber objective.  A bounded linear Ridge/Huber model predicts those gains from
case-level predictor summaries.  Candidate selection uses nested,
similarity-group-isolated cross-validation.  The 50 final-test element files
are never read by this module.

The original 119/15/15 split has already been inspected during V10/V11 model
development.  V12 therefore treats all 149 non-final cases as one development
set and labels its predictions as gain-layer OOF, not full-model OOF: the
frozen V11 base formula was originally fitted on 119 of those cases.
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
from typing import Iterable, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
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
)
from hierarchical_feasibility import _case_summary_row  # noqa: E402
from signed_staged_residual_symbolic import _atomic_json  # noqa: E402
from v11_gain_stability_audit import (  # noqa: E402
    V11GainAuditConfig,
    preflight_v11_gain_audit,
)
from v11_tail_aware_localised_symbolic import (  # noqa: E402
    _full_case_components,
    _mean_correction,
    build_v11_feature_matrix,
)


V12_OUTPUT_NAME = "13_v12_case_adaptive_tail_gain"
V11_AUDIT_OUTPUT = "12_v11_gain_stability_audit"
V11_AUDIT_SUBDIR = "iteration_1"

PHYSICAL_CONTEXT_FEATURES = [
    "fluence_rate_mean",
    "fluence_rate_std",
    "fluence_rate_p95",
    "temperature_mean",
    "temperature_std",
    "temperature_p95",
    "weight_loss_rate_mean",
    "weight_loss_rate_std",
    "weight_loss_rate_p95",
]

TAIL_CONTEXT_FEATURES = [
    "v10_case_scale",
    "case_mean_correction_mpa",
    "tail_correction_std_mpa",
    "tail_correction_abs_max_mpa",
]

FEATURE_SETS = {
    "physical_9": PHYSICAL_CONTEXT_FEATURES,
    "physical_plus_tail_13": PHYSICAL_CONTEXT_FEATURES + TAIL_CONTEXT_FEATURES,
}

ORACLE_TIER_LABELS = ["background_lt_p90", "p90_p95", "p95_p99", "p99_plus"]


@dataclass(frozen=True)
class V12Config:
    iteration: int = 1
    output_subdir: str = "iteration_1"
    global_reference_gain: float = 0.90
    oracle_gain_min: float = 0.00
    oracle_gain_max: float = 1.10
    oracle_gain_step: float = 0.025
    oracle_near_optimal_tolerance: float = 0.005
    oracle_huber_delta: float = 1.0
    oracle_tier_mass: tuple[float, float, float, float] = (0.55, 0.15, 0.20, 0.10)
    outer_folds: int = 5
    inner_folds: int = 4
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    huber_alphas: tuple[float, ...] = (0.001, 0.01, 0.1)
    huber_epsilon: float = 1.35
    shrinkage_values: tuple[float, ...] = (0.50, 0.75, 1.00)
    maximum_deployable_gain: float = 1.10
    minimum_deployable_gain: float = 0.00
    bootstrap_resamples: int = 10_000
    random_seed: int = RANDOM_SEED

    def validate(self) -> None:
        if self.iteration != 1:
            raise ValueError("V12 currently freezes the Iteration-1 V11 formula")
        if self.oracle_gain_step <= 0:
            raise ValueError("oracle_gain_step must be positive")
        if self.oracle_gain_max <= self.oracle_gain_min:
            raise ValueError("oracle gain bounds are invalid")
        if not self.oracle_gain_min <= self.global_reference_gain <= self.oracle_gain_max:
            raise ValueError("global_reference_gain must lie in the oracle grid")
        if not math.isclose(sum(self.oracle_tier_mass), 1.0, abs_tol=1e-12):
            raise ValueError("oracle_tier_mass must sum to one")
        if any(value <= 0 for value in self.oracle_tier_mass):
            raise ValueError("every oracle tier must have positive target mass")
        if self.oracle_near_optimal_tolerance < 0:
            raise ValueError("oracle_near_optimal_tolerance cannot be negative")
        if self.outer_folds < 3 or self.inner_folds < 3:
            raise ValueError("nested grouped CV requires at least three folds")
        if not 1.0 <= self.huber_epsilon:
            raise ValueError("Huber epsilon must be at least one")
        if self.minimum_deployable_gain < 0:
            raise ValueError("negative tail gains are not deployable")
        if self.maximum_deployable_gain > self.oracle_gain_max + 1e-12:
            raise ValueError("deployable gain cannot exceed the audited oracle grid")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("Use at least 1,000 case-level bootstrap resamples")
        if not np.any(np.isclose(self.oracle_gain_grid, self.global_reference_gain)):
            raise ValueError("The oracle grid must contain the global reference gain")

    @property
    def oracle_gain_grid(self) -> np.ndarray:
        count = int(round((self.oracle_gain_max - self.oracle_gain_min) / self.oracle_gain_step))
        values = self.oracle_gain_min + self.oracle_gain_step * np.arange(count + 1)
        return np.round(values, 10)


def output_directory(package_root: Path, config: V12Config) -> Path:
    return Path(package_root) / "outputs" / V12_OUTPUT_NAME / config.output_subdir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v11_audit_paths(package_root: Path) -> dict[str, Path]:
    root = Path(package_root) / "outputs" / V11_AUDIT_OUTPUT / V11_AUDIT_SUBDIR
    return {
        "completion": root / "audit_complete.json",
        "selected_gain": root / "selected_gain.json",
        "selected_formula": root / "selected_gain_formula.csv",
        "parent_signature": root / "audit_signature.json",
    }


def preflight_v12(package_root: Path, config: V12Config) -> dict:
    """Freeze V11 provenance, the 149-case development set and final-test seal."""

    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    parent = preflight_v11_gain_audit(package_root, V11GainAuditConfig())
    inputs = parent["inputs"]
    development_ids = sorted(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"]
    )
    assert_no_final_cases(development_ids, inputs["manifest"])
    if len(development_ids) != 149 or len(set(development_ids)) != 149:
        raise ValueError("V12 requires exactly 149 unique development cases")

    audit_paths = _v11_audit_paths(package_root)
    required = {
        **audit_paths,
        "manifest": package_root / "shared" / "frozen_case_split_manifest_199cases.csv",
        "similarity_map": package_root / "shared" / "frozen_similarity_group_map_199cases.csv",
        "development_summary": package_root
        / "outputs"
        / "00_qc_sensitivity_ablation"
        / "development_case_summary.csv",
        "v12_source": package_root / "src" / "v12_case_adaptive_tail_gain.py",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("V12 required inputs are missing:\n" + "\n".join(missing))

    completion = json.loads(audit_paths["completion"].read_text(encoding="utf-8"))
    selected_gain = json.loads(audit_paths["selected_gain"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or int(completion.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError("The V11 gain audit is incomplete or broke the final-test seal")
    if not math.isclose(
        float(selected_gain["selected_gain"]),
        config.global_reference_gain,
        abs_tol=1e-12,
    ):
        raise ValueError("V12 global_reference_gain differs from the frozen V11 audit")

    manifest = inputs["manifest"].copy()
    iteration_manifest = manifest[manifest["iteration"].eq(config.iteration)].copy()
    development_manifest = iteration_manifest[
        ~iteration_manifest["split"].eq("final_test")
    ].copy()
    if set(development_manifest["case_id"]) != set(development_ids):
        raise ValueError("Development manifest does not match the 149 non-final cases")
    if set(iteration_manifest.loc[iteration_manifest["split"].eq("final_test"), "case_id"]) != set(inputs["final_ids"]):
        raise ValueError("Final-test IDs drifted from the frozen manifest")

    summary = pd.read_csv(required["development_summary"])
    if set(summary["case_id"]) != set(development_ids) or len(summary) != 149:
        raise ValueError("Development-case predictor summary is incomplete")
    registered_context = sorted(set(sum(FEATURE_SETS.values(), [])))
    forbidden_context = [
        name
        for name in registered_context
        if name.startswith("stress_") or name == TARGET_COL
    ]
    if forbidden_context:
        raise AssertionError(f"Stress fields leaked into context registry: {forbidden_context}")

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
        {"check": "development_cases", "value": len(development_ids), "expected": 149},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "v11_gain_audit_complete", "value": completion.get("status"), "expected": "complete"},
        {"check": "parent_final_cases_read", "value": completion.get("final_test_cases_read"), "expected": 0},
        {"check": "frozen_global_gain", "value": float(selected_gain["selected_gain"]), "expected": config.global_reference_gain},
        {"check": "oracle_grid_values", "value": len(config.oracle_gain_grid), "expected": 45},
        {"check": "context_feature_sets", "value": len(FEATURE_SETS), "expected": 2},
    ])
    checks["pass"] = checks["value"] == checks["expected"]
    checks.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not checks["pass"].all():
        raise RuntimeError(
            "V12 preflight failed: "
            + ", ".join(checks.loc[~checks["pass"], "check"].tolist())
        )

    signature_payload = {
        "config": asdict(config),
        "frozen_input_sha256": dict(zip(hash_audit["artifact"], hash_audit["sha256"])),
        "development_ids": development_ids,
        "development_similarity_groups": dict(
            zip(development_manifest["case_id"], development_manifest["similarity_group"])
        ),
        "legacy_split": dict(zip(development_manifest["case_id"], development_manifest["split"])),
        "final_test_ids_inventoried_not_read": inputs["final_ids"],
    }
    signature_payload["v12_signature_sha256"] = _canonical_hash(signature_payload)
    signature_path = output_dir / "v12_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature_payload:
            raise RuntimeError("Existing V12 output has a different frozen signature")
    else:
        _atomic_json(signature_path, signature_payload)

    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "parent": parent,
        "inputs": inputs,
        "development_ids": development_ids,
        "development_manifest": development_manifest,
        "summary": summary,
        "completion": completion,
        "selected_gain": selected_gain,
        "required": required,
        "hash_audit": hash_audit,
        "checks": checks,
        "signature": signature_payload,
    }


def _oracle_cache_path(output_dir: Path, case_id: str) -> Path:
    path = output_dir / "case_cache" / "oracle_grid" / f"{case_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _adaptive_cache_path(output_dir: Path, case_id: str) -> Path:
    path = output_dir / "case_cache" / "adaptive_oof" / f"{case_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tier_weights(actual: np.ndarray, target_mass: Sequence[float]) -> tuple[np.ndarray, dict]:
    p90, p95, p99 = np.quantile(actual, [0.90, 0.95, 0.99])
    tiers = np.select(
        [actual < p90, actual < p95, actual < p99],
        [0, 1, 2],
        default=3,
    ).astype(np.int8)
    weights = np.zeros(len(actual), dtype=np.float64)
    counts = []
    for tier, mass in enumerate(target_mass):
        mask = tiers == tier
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Stress tier {tier} is empty")
        weights[mask] = float(mass) / count
        counts.append(count)
    weights /= weights.sum()
    return weights, {
        "actual_p90": float(p90),
        "actual_p95": float(p95),
        "actual_p99": float(p99),
        **{f"{label}_count": count for label, count in zip(ORACLE_TIER_LABELS, counts)},
    }


def _weighted_huber(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    delta: float,
) -> float:
    scale = max(float(np.std(actual, ddof=0)), 1e-12)
    residual = (predicted - actual) / scale
    absolute = np.abs(residual)
    loss = np.where(
        absolute <= delta,
        0.5 * residual ** 2,
        delta * (absolute - 0.5 * delta),
    )
    return float(np.sum(weights * loss))


def _context_record(
    summary: pd.Series,
    *,
    scale: float,
    delta_mean: float,
    tail_correction: np.ndarray,
) -> dict:
    record = {feature: float(summary[feature]) for feature in PHYSICAL_CONTEXT_FEATURES}
    record.update({
        "v10_case_scale": float(scale),
        "case_mean_correction_mpa": float(delta_mean),
        "tail_correction_std_mpa": float(np.std(tail_correction, ddof=0)),
        "tail_correction_abs_max_mpa": float(np.max(np.abs(tail_correction))),
    })
    if not np.isfinite(np.asarray(list(record.values()), dtype=np.float64)).all():
        raise ValueError("Non-finite predictor-only context feature")
    return record


def _oracle_cache_valid(
    path: Path,
    *,
    case_id: str,
    grid: np.ndarray,
    signature: str,
) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path)
    except Exception:
        return False
    required = {"case_id", "gain", "v12_signature_sha256", "tail_weighted_huber_loss"}
    if not required.issubset(table.columns):
        return False
    if set(table["case_id"]) != {case_id} or set(table["v12_signature_sha256"]) != {signature}:
        return False
    observed = np.sort(table["gain"].to_numpy(dtype=np.float64))
    return len(observed) == len(grid) and np.allclose(observed, np.sort(grid), atol=1e-12, rtol=0.0)


def _evaluate_oracle_grid_case(
    preflight: dict,
    case_id: str,
    config: V12Config,
) -> pd.DataFrame:
    output_dir = preflight["output_dir"]
    path = _oracle_cache_path(output_dir, case_id)
    signature = preflight["signature"]["v12_signature_sha256"]
    if _oracle_cache_valid(
        path,
        case_id=case_id,
        grid=config.oracle_gain_grid,
        signature=signature,
    ):
        return pd.read_csv(path)

    parent = preflight["parent"]
    frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
    summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    matrix, scale, v10_prediction, components = _full_case_components(
        frame, summary, parent["v10"]
    )
    delta_mean = _mean_correction(summary, parent["mean_model"])
    mean_prediction = v10_prediction + delta_mean
    v11_matrix = build_v11_feature_matrix(
        matrix,
        parent["v10"],
        parent["centres"],
        components,
    )
    tail_raw = parent["tail_function"](v11_matrix)
    centered_tail = tail_raw - float(np.mean(tail_raw))
    tail_correction = scale * centered_tail
    weights, tier_audit = _tier_weights(actual, config.oracle_tier_mass)
    context = _context_record(
        summary,
        scale=scale,
        delta_mean=delta_mean,
        tail_correction=tail_correction,
    )
    legacy_row = preflight["development_manifest"].set_index("case_id").loc[case_id]

    rows = []
    for gain in config.oracle_gain_grid:
        predicted = mean_prediction + float(gain) * tail_correction
        rows.append({
            "iteration": config.iteration,
            "case_id": case_id,
            "legacy_split": str(legacy_row["split"]),
            "similarity_group": str(legacy_row["similarity_group"]),
            "v12_signature_sha256": signature,
            "gain": float(gain),
            "tail_weighted_huber_loss": _weighted_huber(
                actual,
                predicted,
                weights,
                config.oracle_huber_delta,
            ),
            **tier_audit,
            **context,
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
        v10_prediction,
        components,
        mean_prediction,
        v11_matrix,
        tail_raw,
        centered_tail,
        tail_correction,
        weights,
    )
    gc.collect()
    return result


def evaluate_oracle_grid(preflight: dict, config: V12Config) -> pd.DataFrame:
    parts = []
    for position, case_id in enumerate(preflight["development_ids"], start=1):
        print(
            f"[{position}/{len(preflight['development_ids'])}] V12 oracle gain grid: {case_id}",
            flush=True,
        )
        parts.append(_evaluate_oracle_grid_case(preflight, case_id, config))
    return pd.concat(parts, ignore_index=True)


def select_oracle_targets(grid_metrics: pd.DataFrame, config: V12Config) -> pd.DataFrame:
    records = []
    context_features = sorted(set(sum(FEATURE_SETS.values(), [])))
    for case_id, group in grid_metrics.groupby("case_id", observed=True):
        group = group.sort_values("gain").copy()
        minimum = float(group["tail_weighted_huber_loss"].min())
        threshold = minimum * (1.0 + config.oracle_near_optimal_tolerance)
        near = group[group["tail_weighted_huber_loss"] <= threshold + 1e-15].copy()
        near["distance_to_global_reference"] = abs(
            near["gain"] - config.global_reference_gain
        )
        selected = near.sort_values(
            ["distance_to_global_reference", "gain"], ascending=[True, True]
        ).iloc[0]
        reference = group[
            np.isclose(group["gain"], config.global_reference_gain, atol=1e-12)
        ].iloc[0]
        improvement = max(
            float(reference["tail_weighted_huber_loss"] - selected["tail_weighted_huber_loss"]),
            0.0,
        )
        fraction = improvement / max(float(reference["tail_weighted_huber_loss"]), 1e-12)
        records.append({
            "case_id": case_id,
            "legacy_split": selected["legacy_split"],
            "similarity_group": selected["similarity_group"],
            "oracle_gain": float(selected["gain"]),
            "minimum_grid_loss": minimum,
            "selected_near_optimal_loss": float(selected["tail_weighted_huber_loss"]),
            "global_gain_loss": float(reference["tail_weighted_huber_loss"]),
            "oracle_loss_improvement_fraction": fraction,
            "oracle_target_weight": 1.0 + 4.0 * min(max(fraction, 0.0), 1.0),
            "near_optimal_gain_count": len(near),
            "oracle_at_lower_bound": bool(np.isclose(selected["gain"], config.oracle_gain_min)),
            "oracle_at_upper_bound": bool(np.isclose(selected["gain"], config.oracle_gain_max)),
            **{feature: float(selected[feature]) for feature in context_features},
        })
    result = pd.DataFrame(records).sort_values("case_id").reset_index(drop=True)
    if len(result) != 149 or result["case_id"].nunique() != 149:
        raise ValueError("Oracle target table is incomplete")
    if not result["oracle_gain"].between(
        config.oracle_gain_min, config.oracle_gain_max, inclusive="both"
    ).all():
        raise ValueError("Oracle gain escaped the configured bounds")
    return result


def candidate_configurations(config: V12Config) -> pd.DataFrame:
    """Create the small, auditable gain-model search space."""

    rows = [{
        "candidate_id": "constant_global_0_90",
        "model_family": "constant",
        "feature_set": "none",
        "alpha": 0.0,
        "shrinkage": 0.0,
        "n_features": 0,
        "model_complexity": 0,
    }]
    for feature_set, features in FEATURE_SETS.items():
        for alpha in config.ridge_alphas:
            for shrinkage in config.shrinkage_values:
                rows.append({
                    "candidate_id": (
                        f"ridge__{feature_set}__a{alpha:g}__s{shrinkage:.2f}"
                    ),
                    "model_family": "ridge",
                    "feature_set": feature_set,
                    "alpha": float(alpha),
                    "shrinkage": float(shrinkage),
                    "n_features": len(features),
                    "model_complexity": len(features) + 1,
                })
        for alpha in config.huber_alphas:
            for shrinkage in config.shrinkage_values:
                rows.append({
                    "candidate_id": (
                        f"huber__{feature_set}__a{alpha:g}__s{shrinkage:.2f}"
                    ),
                    "model_family": "huber",
                    "feature_set": feature_set,
                    "alpha": float(alpha),
                    "shrinkage": float(shrinkage),
                    "n_features": len(features),
                    "model_complexity": len(features) + 2,
                })
    result = pd.DataFrame(rows)
    if len(result) != 43 or result["candidate_id"].nunique() != 43:
        raise AssertionError("Expected exactly 43 unique V12 gain candidates")
    return result


def _balanced_group_folds(
    table: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Assign complete similarity groups to approximately balanced folds."""

    group_sizes = (
        table.groupby("similarity_group", observed=True)
        .size()
        .rename("n_cases")
        .reset_index()
    )
    if len(group_sizes) < n_folds:
        raise ValueError("There are fewer similarity groups than requested folds")
    rng = np.random.default_rng(seed)
    group_sizes["tie_breaker"] = rng.random(len(group_sizes))
    group_sizes = group_sizes.sort_values(
        ["n_cases", "tie_breaker"], ascending=[False, True]
    ).reset_index(drop=True)

    loads = np.zeros(n_folds, dtype=int)
    assignments = {}
    for row in group_sizes.itertuples(index=False):
        lightest = np.flatnonzero(loads == loads.min())
        fold = int(lightest[rng.integers(0, len(lightest))])
        assignments[str(row.similarity_group)] = fold
        loads[fold] += int(row.n_cases)

    result = table[["case_id", "similarity_group"]].copy()
    result["fold"] = result["similarity_group"].astype(str).map(assignments)
    if result["fold"].isna().any():
        raise AssertionError("A similarity group was not assigned to a fold")
    result["fold"] = result["fold"].astype(int)
    if result.groupby("similarity_group")["fold"].nunique().max() != 1:
        raise AssertionError("Similarity-group leakage detected in fold assignment")
    if result["fold"].nunique() != n_folds:
        raise AssertionError("Not every requested fold received at least one case")
    return result.sort_values("case_id").reset_index(drop=True)


def _candidate_features(candidate: pd.Series | dict) -> list[str]:
    feature_set = str(candidate["feature_set"])
    return [] if feature_set == "none" else list(FEATURE_SETS[feature_set])


def _fit_gain_state(
    training: pd.DataFrame,
    candidate: pd.Series | dict,
    config: V12Config,
) -> dict:
    family = str(candidate["model_family"])
    if family == "constant":
        return {
            "candidate_id": str(candidate["candidate_id"]),
            "model_family": family,
            "feature_set": "none",
            "features": [],
            "feature_mean": [],
            "feature_scale": [],
            "standardised_coefficients": [],
            "standardised_intercept": config.global_reference_gain,
            "alpha": 0.0,
            "shrinkage": 0.0,
        }

    features = _candidate_features(candidate)
    X = training[features].to_numpy(dtype=np.float64)
    y = training["oracle_gain"].to_numpy(dtype=np.float64)
    sample_weight = training["oracle_target_weight"].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    alpha = float(candidate["alpha"])
    if family == "ridge":
        model = Ridge(alpha=alpha)
    elif family == "huber":
        model = HuberRegressor(
            alpha=alpha,
            epsilon=config.huber_epsilon,
            max_iter=2_000,
            tol=1e-7,
        )
    else:
        raise ValueError(f"Unknown gain model family: {family}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y, sample_weight=sample_weight)

    shrinkage = float(candidate["shrinkage"])
    coefficients = shrinkage * np.asarray(model.coef_, dtype=np.float64)
    intercept = (
        (1.0 - shrinkage) * config.global_reference_gain
        + shrinkage * float(model.intercept_)
    )
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "model_family": family,
        "feature_set": str(candidate["feature_set"]),
        "features": features,
        "feature_mean": scaler.mean_.astype(float).tolist(),
        "feature_scale": scaler.scale_.astype(float).tolist(),
        "standardised_coefficients": coefficients.astype(float).tolist(),
        "standardised_intercept": float(intercept),
        "alpha": alpha,
        "shrinkage": shrinkage,
    }


def _predict_gain(state: dict, table: pd.DataFrame, config: V12Config) -> np.ndarray:
    if state["model_family"] == "constant":
        raw = np.full(len(table), config.global_reference_gain, dtype=np.float64)
    else:
        features = list(state["features"])
        X = table[features].to_numpy(dtype=np.float64)
        mean = np.asarray(state["feature_mean"], dtype=np.float64)
        scale = np.asarray(state["feature_scale"], dtype=np.float64)
        coefficients = np.asarray(
            state["standardised_coefficients"], dtype=np.float64
        )
        raw = float(state["standardised_intercept"]) + ((X - mean) / scale) @ coefficients
    if not np.isfinite(raw).all():
        raise FloatingPointError("Gain model produced non-finite values")
    return np.clip(
        raw,
        config.minimum_deployable_gain,
        config.maximum_deployable_gain,
    )


def _interpolated_oracle_loss(
    grid_metrics: pd.DataFrame,
    case_ids: Sequence[str],
    gains: np.ndarray,
) -> np.ndarray:
    lookup = {case_id: group.sort_values("gain") for case_id, group in grid_metrics.groupby("case_id")}
    losses = np.empty(len(case_ids), dtype=np.float64)
    for index, (case_id, gain) in enumerate(zip(case_ids, gains)):
        group = lookup[str(case_id)]
        losses[index] = np.interp(
            float(gain),
            group["gain"].to_numpy(dtype=np.float64),
            group["tail_weighted_huber_loss"].to_numpy(dtype=np.float64),
        )
    return losses


def _score_candidate_cv(
    table: pd.DataFrame,
    grid_metrics: pd.DataFrame,
    candidate: pd.Series,
    folds: pd.DataFrame,
    config: V12Config,
) -> dict:
    working = table.merge(folds[["case_id", "fold"]], on="case_id", validate="one_to_one")
    predictions = np.full(len(working), np.nan, dtype=np.float64)
    failure = ""
    try:
        for fold in sorted(working["fold"].unique()):
            train = working[~working["fold"].eq(fold)]
            held_out = working[working["fold"].eq(fold)]
            state = _fit_gain_state(train, candidate, config)
            predictions[held_out.index] = _predict_gain(state, held_out, config)
    except Exception as exc:
        failure = repr(exc)

    if failure or not np.isfinite(predictions).all():
        return {
            **candidate.to_dict(),
            "cv_selection_score": np.inf,
            "mean_oracle_loss_ratio": np.inf,
            "weighted_gain_mae": np.inf,
            "p90_absolute_gain_error": np.inf,
            "gain_saturation_fraction": 1.0,
            "fit_failure": failure or "non-finite cross-validated prediction",
        }

    case_ids = working["case_id"].astype(str).tolist()
    predicted_loss = _interpolated_oracle_loss(grid_metrics, case_ids, predictions)
    reference_loss = working["global_gain_loss"].to_numpy(dtype=np.float64)
    weights = working["oracle_target_weight"].to_numpy(dtype=np.float64)
    absolute_gain_error = np.abs(
        predictions - working["oracle_gain"].to_numpy(dtype=np.float64)
    )
    loss_ratio = predicted_loss / np.maximum(reference_loss, 1e-12)
    weighted_loss_ratio = float(np.average(loss_ratio, weights=weights))
    weighted_gain_mae = float(np.average(absolute_gain_error, weights=weights))
    p90_gain_error = float(np.quantile(absolute_gain_error, 0.90))
    saturated = (
        np.isclose(predictions, config.minimum_deployable_gain, atol=1e-12)
        | np.isclose(predictions, config.maximum_deployable_gain, atol=1e-12)
    )
    saturation_fraction = float(saturated.mean())
    return {
        **candidate.to_dict(),
        "cv_selection_score": (
            weighted_loss_ratio
            + 0.10 * weighted_gain_mae
            + 0.05 * p90_gain_error
            + 0.02 * saturation_fraction
        ),
        "mean_oracle_loss_ratio": weighted_loss_ratio,
        "weighted_gain_mae": weighted_gain_mae,
        "p90_absolute_gain_error": p90_gain_error,
        "gain_saturation_fraction": saturation_fraction,
        "fit_failure": "",
    }


def _select_candidate(scores: pd.DataFrame) -> pd.Series:
    finite = scores[np.isfinite(scores["cv_selection_score"])].copy()
    if finite.empty:
        raise RuntimeError("Every V12 gain candidate failed")
    return finite.sort_values(
        ["cv_selection_score", "model_complexity", "candidate_id"],
        ascending=[True, True, True],
    ).iloc[0]


def nested_group_gain_validation(
    targets: pd.DataFrame,
    grid_metrics: pd.DataFrame,
    config: V12Config,
) -> dict:
    """Nested grouped validation for the gain layer only."""

    candidates = candidate_configurations(config)
    outer_folds = _balanced_group_folds(
        targets,
        n_folds=config.outer_folds,
        seed=config.random_seed + 12_000,
    )
    oof_parts = []
    inner_score_parts = []
    outer_selection_rows = []

    for outer_fold in range(config.outer_folds):
        outer_train_ids = outer_folds.loc[
            ~outer_folds["fold"].eq(outer_fold), "case_id"
        ]
        outer_holdout_ids = outer_folds.loc[
            outer_folds["fold"].eq(outer_fold), "case_id"
        ]
        outer_train = targets[targets["case_id"].isin(outer_train_ids)].copy()
        outer_holdout = targets[targets["case_id"].isin(outer_holdout_ids)].copy()
        inner_folds = _balanced_group_folds(
            outer_train,
            n_folds=config.inner_folds,
            seed=config.random_seed + 13_000 + outer_fold,
        )
        score_rows = [
            _score_candidate_cv(
                outer_train,
                grid_metrics[grid_metrics["case_id"].isin(outer_train_ids)],
                candidate,
                inner_folds,
                config,
            )
            for _, candidate in candidates.iterrows()
        ]
        scores = pd.DataFrame(score_rows)
        selected = _select_candidate(scores)
        scores["outer_fold"] = outer_fold
        scores["selected_in_outer_fold"] = scores["candidate_id"].eq(
            selected["candidate_id"]
        )
        inner_score_parts.append(scores)

        state = _fit_gain_state(outer_train, selected, config)
        predicted = _predict_gain(state, outer_holdout, config)
        part = outer_holdout.copy()
        part["outer_fold"] = outer_fold
        part["selected_candidate_id"] = selected["candidate_id"]
        part["predicted_gain_oof"] = predicted
        part["absolute_gain_error"] = abs(part["predicted_gain_oof"] - part["oracle_gain"])
        part["predicted_oracle_loss"] = _interpolated_oracle_loss(
            grid_metrics,
            part["case_id"].tolist(),
            predicted,
        )
        part["predicted_oracle_loss_ratio"] = (
            part["predicted_oracle_loss"] / part["global_gain_loss"].clip(lower=1e-12)
        )
        oof_parts.append(part)
        outer_selection_rows.append({
            "outer_fold": outer_fold,
            "outer_train_cases": len(outer_train),
            "outer_holdout_cases": len(outer_holdout),
            "outer_train_groups": outer_train["similarity_group"].nunique(),
            "outer_holdout_groups": outer_holdout["similarity_group"].nunique(),
            **selected.to_dict(),
        })

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("case_id").reset_index(drop=True)
    if len(oof) != 149 or oof["case_id"].nunique() != 149:
        raise AssertionError("Nested gain-layer OOF predictions are incomplete")
    if not oof["predicted_gain_oof"].between(
        config.minimum_deployable_gain,
        config.maximum_deployable_gain,
        inclusive="both",
    ).all():
        raise AssertionError("OOF gain prediction escaped the deployable bounds")

    final_folds = _balanced_group_folds(
        targets,
        n_folds=config.outer_folds,
        seed=config.random_seed + 14_000,
    )
    final_scores = pd.DataFrame([
        _score_candidate_cv(targets, grid_metrics, candidate, final_folds, config)
        for _, candidate in candidates.iterrows()
    ])
    selected_final = _select_candidate(final_scores)
    final_scores["selected_for_149_case_refit"] = final_scores["candidate_id"].eq(
        selected_final["candidate_id"]
    )
    final_state = _fit_gain_state(targets, selected_final, config)

    return {
        "candidates": candidates,
        "outer_folds": outer_folds,
        "oof": oof,
        "inner_scores": pd.concat(inner_score_parts, ignore_index=True),
        "outer_selections": pd.DataFrame(outer_selection_rows),
        "final_folds": final_folds,
        "final_scores": final_scores,
        "selected_final": selected_final,
        "final_state": final_state,
    }


def _adaptive_cache_valid(path: Path, *, case_id: str, gain: float, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        row = pd.read_csv(path).iloc[0]
    except Exception:
        return False
    return (
        str(row.get("case_id")) == case_id
        and str(row.get("v12_signature_sha256")) == signature
        and math.isclose(float(row.get("gain")), float(gain), abs_tol=1e-12)
    )


def _evaluate_adaptive_case(
    preflight: dict,
    case_id: str,
    gain: float,
    outer_fold: int,
    config: V12Config,
) -> pd.DataFrame:
    path = _adaptive_cache_path(preflight["output_dir"], case_id)
    signature = preflight["signature"]["v12_signature_sha256"]
    if _adaptive_cache_valid(path, case_id=case_id, gain=gain, signature=signature):
        return pd.read_csv(path)

    parent = preflight["parent"]
    frame, _ = read_complete_case(preflight["inputs"]["path_by_case"][case_id])
    summary = _case_summary_row(preflight["inputs"]["case_summary"], case_id)
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    matrix, scale, v10_prediction, components = _full_case_components(
        frame, summary, parent["v10"]
    )
    delta_mean = _mean_correction(summary, parent["mean_model"])
    v11_matrix = build_v11_feature_matrix(
        matrix, parent["v10"], parent["centres"], components
    )
    tail_raw = parent["tail_function"](v11_matrix)
    tail_correction = scale * (tail_raw - float(np.mean(tail_raw)))
    predicted = v10_prediction + delta_mean + float(gain) * tail_correction
    legacy = preflight["development_manifest"].set_index("case_id").loc[case_id]
    result = pd.DataFrame([{
        "iteration": config.iteration,
        "case_id": case_id,
        "legacy_split": str(legacy["split"]),
        "similarity_group": str(legacy["similarity_group"]),
        "outer_fold": int(outer_fold),
        "model": "v12_adaptive_gain_nested_oof",
        "gain": float(gain),
        "v12_signature_sha256": signature,
        **evaluate_prediction_arrays(actual, predicted),
    }])
    temporary = path.with_suffix(path.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, path)
    del frame, actual, matrix, v10_prediction, components, v11_matrix, tail_raw, tail_correction, predicted
    gc.collect()
    return result


def evaluate_adaptive_oof(
    preflight: dict,
    oof: pd.DataFrame,
    config: V12Config,
) -> pd.DataFrame:
    parts = []
    for position, row in enumerate(oof.itertuples(index=False), start=1):
        print(
            f"[{position}/{len(oof)}] V12 complete-case OOF: {row.case_id} "
            f"gain={row.predicted_gain_oof:.4f}",
            flush=True,
        )
        parts.append(_evaluate_adaptive_case(
            preflight,
            row.case_id,
            float(row.predicted_gain_oof),
            int(row.outer_fold),
            config,
        ))
    return pd.concat(parts, ignore_index=True)


def build_comparison_case_metrics(
    grid_metrics: pd.DataFrame,
    targets: pd.DataFrame,
    adaptive: pd.DataFrame,
    config: V12Config,
) -> pd.DataFrame:
    metric_columns = list(evaluate_prediction_arrays(np.array([0.0, 1.0]), np.array([0.0, 1.0])).keys())
    identity_columns = ["case_id", "legacy_split", "similarity_group", "gain"]
    global_rows = grid_metrics[
        np.isclose(grid_metrics["gain"], config.global_reference_gain, atol=1e-12)
    ][identity_columns + metric_columns].copy()
    global_rows["model"] = "v11_global_gain_0_90"

    oracle_rows = grid_metrics.merge(
        targets[["case_id", "oracle_gain"]], on="case_id", validate="many_to_one"
    )
    oracle_rows = oracle_rows[
        np.isclose(oracle_rows["gain"], oracle_rows["oracle_gain"], atol=1e-12)
    ][identity_columns + metric_columns].copy()
    oracle_rows["model"] = "v12_oracle_gain_non_deployable"
    result = pd.concat([global_rows, adaptive, oracle_rows], ignore_index=True, sort=False)
    if result.groupby("model")["case_id"].nunique().to_dict() != {
        "v11_global_gain_0_90": 149,
        "v12_adaptive_gain_nested_oof": 149,
        "v12_oracle_gain_non_deployable": 149,
    }:
        raise AssertionError("V12 comparison case metrics are incomplete")
    return result


def aggregate_scopes(case_metrics: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_149_development": case_metrics["case_id"].notna(),
        "legacy_119_train": case_metrics["legacy_split"].eq("train"),
        "legacy_15_validation": case_metrics["legacy_split"].eq("validation"),
        "legacy_15_internal_test": case_metrics["legacy_split"].eq("internal_test"),
        "legacy_30_holdout": case_metrics["legacy_split"].isin(["validation", "internal_test"]),
    }
    parts = []
    for scope, mask in scopes.items():
        subset = case_metrics[mask].copy()
        subset["evaluation_scope"] = scope
        parts.append(aggregate_case_metrics(subset, ["evaluation_scope", "model"]))
    return pd.concat(parts, ignore_index=True)


def paired_group_bootstrap(
    case_metrics: pd.DataFrame,
    config: V12Config,
) -> pd.DataFrame:
    records = []
    scope_filters = {
        "all_149_development": case_metrics["case_id"].notna(),
        "legacy_30_holdout": case_metrics["legacy_split"].isin(["validation", "internal_test"]),
    }
    metrics = [
        ("rmse", "lower_is_better"),
        ("top5_actual_rmse", "lower_is_better"),
        ("p95_relative_error", "lower_is_better"),
        ("p99_relative_error", "lower_is_better"),
        ("p99_underprediction_fraction", "lower_is_better"),
        ("top1pct_hotspot_overlap", "higher_is_better"),
        ("top1_recall_in_predicted_top5", "higher_is_better"),
    ]
    rng = np.random.default_rng(config.random_seed + 15_000)
    for scope, mask in scope_filters.items():
        subset = case_metrics[mask]
        adaptive = subset[subset["model"].eq("v12_adaptive_gain_nested_oof")].set_index("case_id")
        reference = subset[subset["model"].eq("v11_global_gain_0_90")].set_index("case_id")
        common = adaptive.index.intersection(reference.index)
        group_labels = adaptive.loc[common, "similarity_group"].astype(str)
        groups = sorted(group_labels.unique())
        for metric, direction in metrics:
            delta = adaptive.loc[common, metric] - reference.loc[common, metric]
            group_delta = delta.groupby(group_labels).mean().reindex(groups).to_numpy(dtype=np.float64)
            bootstrap = np.empty(config.bootstrap_resamples, dtype=np.float64)
            for start in range(0, config.bootstrap_resamples, 1_000):
                size = min(1_000, config.bootstrap_resamples - start)
                indices = rng.integers(0, len(groups), size=(size, len(groups)))
                bootstrap[start:start + size] = group_delta[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            records.append({
                "evaluation_scope": scope,
                "metric": metric,
                "direction": direction,
                "n_cases": len(common),
                "n_similarity_groups": len(groups),
                "mean_delta_adaptive_minus_global": float(delta.mean()),
                "group_bootstrap_95ci_low": float(lower),
                "group_bootstrap_95ci_high": float(upper),
                "descriptive_only": True,
            })
    return pd.DataFrame(records)


def promotion_decision(
    split_metrics: pd.DataFrame,
    oof: pd.DataFrame,
    config: V12Config,
) -> tuple[pd.DataFrame, dict]:
    indexed = split_metrics.set_index(["evaluation_scope", "model"])

    def row(scope: str, model: str) -> pd.Series:
        return indexed.loc[(scope, model)]

    adaptive_all = row("all_149_development", "v12_adaptive_gain_nested_oof")
    global_all = row("all_149_development", "v11_global_gain_0_90")
    adaptive_holdout = row("legacy_30_holdout", "v12_adaptive_gain_nested_oof")
    global_holdout = row("legacy_30_holdout", "v11_global_gain_0_90")
    all_rmse_gain = (
        float(global_all["macro_rmse"]) - float(adaptive_all["macro_rmse"])
    ) / max(float(global_all["macro_rmse"]), 1e-12)
    holdout_rmse_gain = (
        float(global_holdout["macro_rmse"]) - float(adaptive_holdout["macro_rmse"])
    ) / max(float(global_holdout["macro_rmse"]), 1e-12)
    saturation_fraction = float(
        (
            np.isclose(oof["predicted_gain_oof"], config.minimum_deployable_gain)
            | np.isclose(oof["predicted_gain_oof"], config.maximum_deployable_gain)
        ).mean()
    )
    gates = [
        ("finite_bounded_gain_predictions", bool(np.isfinite(oof["predicted_gain_oof"]).all() and oof["predicted_gain_oof"].between(0.0, 1.10).all()), "all gains finite and in [0, 1.10]"),
        ("gain_layer_oracle_skill", float(np.average(oof["predicted_oracle_loss_ratio"], weights=oof["oracle_target_weight"])) < 0.995, "weighted oracle-loss ratio < 0.995"),
        ("all_case_macro_rmse_improves", all_rmse_gain >= 0.005, "all-149 macro RMSE improves by at least 0.5%"),
        ("legacy_holdout_macro_rmse_not_worse", holdout_rmse_gain >= 0.0, "legacy 30-case macro RMSE does not worsen"),
        ("all_case_p99_underprediction", float(adaptive_all["mean_p99_underprediction_fraction"]) <= min(float(global_all["mean_p99_underprediction_fraction"]), 0.10) + 1e-12, "all-149 mean P99 underprediction <= global and <= 10%"),
        ("legacy_holdout_p99_underprediction", float(adaptive_holdout["mean_p99_underprediction_fraction"]) <= float(global_holdout["mean_p99_underprediction_fraction"]) + 0.005, "legacy holdout mean P99 underprediction worsens by no more than 0.5 percentage points"),
        ("all_case_top1_overlap", float(adaptive_all["mean_top1pct_hotspot_overlap"]) >= float(global_all["mean_top1pct_hotspot_overlap"]) - 0.002, "all-149 top-1% overlap decreases by no more than 0.2 percentage points"),
        ("legacy_holdout_top1_overlap", float(adaptive_holdout["mean_top1pct_hotspot_overlap"]) >= float(global_holdout["mean_top1pct_hotspot_overlap"]) - 0.005, "legacy holdout top-1% overlap decreases by no more than 0.5 percentage points"),
        ("maximum_prediction_guardrail", float(adaptive_all["max_prediction_abs_max_ratio"]) <= min(float(global_all["max_prediction_abs_max_ratio"]), 1.50) + 1e-12, "worst absolute maximum ratio <= global and <= 1.50"),
        ("limited_gain_saturation", saturation_fraction <= 0.10, "no more than 10% of OOF gains at bounds"),
    ]
    gate_table = pd.DataFrame([
        {"gate": name, "pass": passed, "criterion": criterion}
        for name, passed, criterion in gates
    ])
    promoted = bool(gate_table["pass"].all())
    decision = {
        "promotion_status": "promote_v12_adaptive_gain" if promoted else "retain_v11_global_gain_0_90",
        "all_gates_pass": promoted,
        "failed_gates": gate_table.loc[~gate_table["pass"], "gate"].tolist(),
        "all_149_macro_rmse_improvement_fraction": all_rmse_gain,
        "legacy_30_macro_rmse_improvement_fraction": holdout_rmse_gain,
        "weighted_oof_oracle_loss_ratio": float(np.average(oof["predicted_oracle_loss_ratio"], weights=oof["oracle_target_weight"])),
        "oof_gain_min": float(oof["predicted_gain_oof"].min()),
        "oof_gain_max": float(oof["predicted_gain_oof"].max()),
        "oof_gain_saturation_fraction": saturation_fraction,
        "final_test_cases_read": 0,
    }
    return gate_table, decision


def _gain_formula(state: dict, config: V12Config) -> str:
    if state["model_family"] == "constant":
        return f"{config.global_reference_gain:.12g}"
    terms = [f"({float(state['standardised_intercept']):.16g})"]
    for feature, mean, scale, coefficient in zip(
        state["features"],
        state["feature_mean"],
        state["feature_scale"],
        state["standardised_coefficients"],
    ):
        terms.append(
            f"({float(coefficient):.16g})*(({feature} - ({float(mean):.16g}))/({float(scale):.16g}))"
        )
    raw = " + ".join(terms)
    return (
        f"Min({config.maximum_deployable_gain:.16g}, "
        f"Max({config.minimum_deployable_gain:.16g}, {raw}))"
    )


def save_formula_artifacts(
    preflight: dict,
    final_state: dict,
    decision: dict,
    config: V12Config,
) -> dict:
    output_dir = preflight["output_dir"]
    parent_formula = pd.read_csv(preflight["required"]["selected_formula"]).iloc[0]
    adaptive_gain_formula = _gain_formula(final_state, config)
    selected_gain_formula = (
        adaptive_gain_formula
        if decision["all_gates_pass"]
        else f"{config.global_reference_gain:.12g}"
    )

    def composite(gain_formula: str) -> str:
        return (
            f"({parent_formula['v10_formula']}) + "
            f"({parent_formula['case_mean_calibration_formula']}) + "
            f"({gain_formula}) * exp({parent_formula['v10_log_scale_formula']}) * "
            f"(({parent_formula['v11_tail_raw_formula']}) - "
            f"case_mean({parent_formula['v11_tail_raw_formula']}))"
        )

    candidate_record = {
        "iteration": config.iteration,
        "method": "bounded_predictor_only_case_adaptive_v11_tail_gain",
        "gain_formula": adaptive_gain_formula,
        "combined_stress_formula": composite(adaptive_gain_formula),
        "candidate_id": final_state["candidate_id"],
        "promotion_status": decision["promotion_status"],
    }
    selected_record = {
        **candidate_record,
        "method": "v12_adaptive_gain" if decision["all_gates_pass"] else "v11_global_gain_safe_fallback",
        "gain_formula": selected_gain_formula,
        "combined_stress_formula": composite(selected_gain_formula),
        "selected_for_deployment": True,
    }
    pd.DataFrame([candidate_record]).to_csv(output_dir / "v12_candidate_adaptive_formula.csv", index=False)
    pd.DataFrame([selected_record]).to_csv(output_dir / "selected_deployment_formula.csv", index=False)

    candidate_text = (
        "V12 candidate case-adaptive tail-gain formula\n"
        "=============================================\n\n"
        f"Gain model: {final_state['candidate_id']}\n"
        f"lambda_case = {adaptive_gain_formula}\n\n"
        f"sigma = {candidate_record['combined_stress_formula']}\n\n"
        "Every case summary used by lambda_case is calculated from predictor fields only.\n"
        "The case-specific oracle gains used during development require stress, but oracle gain is never a deployment input.\n"
    )
    selected_text = (
        "V12 formally selected deployment formula\n"
        "========================================\n\n"
        f"Decision: {decision['promotion_status']}\n"
        f"lambda_case = {selected_gain_formula}\n\n"
        f"sigma = {selected_record['combined_stress_formula']}\n\n"
        "The locked 50-case final set has not been read. This formula remains a development-stage selection.\n"
    )
    (output_dir / "v12_candidate_adaptive_formula.txt").write_text(candidate_text, encoding="utf-8")
    (output_dir / "selected_deployment_formula.txt").write_text(selected_text, encoding="utf-8")
    return {"candidate": candidate_record, "selected": selected_record}


def save_diagnostic_plots(
    targets: pd.DataFrame,
    oof: pd.DataFrame,
    case_metrics: pd.DataFrame,
    output_dir: Path,
    config: V12Config,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(targets["oracle_gain"], bins=np.arange(-0.0125, 1.125, 0.025), color="#2878B5", alpha=0.85)
    axes[0].axvline(config.global_reference_gain, color="#C23B22", linestyle="--", label="V11 gain 0.90")
    axes[0].set(title="Development-case oracle tail gains", xlabel="Oracle gain", ylabel="Cases")
    axes[0].legend()
    axes[1].scatter(oof["oracle_gain"], oof["predicted_gain_oof"], s=24, alpha=0.75, color="#228B22")
    axes[1].plot([0, 1.1], [0, 1.1], color="black", linestyle="--", linewidth=1)
    axes[1].axhline(config.global_reference_gain, color="#C23B22", linestyle=":")
    axes[1].set(title="Nested OOF gain prediction", xlabel="Oracle gain (stress-derived target)", ylabel="Predicted gain (predictor-only)", xlim=(0, 1.1), ylim=(0, 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "v12_oracle_and_oof_gain.png", dpi=180)
    plt.close(fig)

    pivot = case_metrics[case_metrics["model"].isin(["v11_global_gain_0_90", "v12_adaptive_gain_nested_oof"])].pivot(index="case_id", columns="model", values=["rmse", "p99_underprediction_fraction", "top1pct_hotspot_overlap"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specifications = [
        ("rmse", "RMSE delta", False),
        ("p99_underprediction_fraction", "P99 underprediction delta", False),
        ("top1pct_hotspot_overlap", "Top-1% overlap delta", True),
    ]
    for axis, (metric, label, higher_better) in zip(axes, specifications):
        delta = pivot[(metric, "v12_adaptive_gain_nested_oof")] - pivot[(metric, "v11_global_gain_0_90")]
        axis.hist(delta, bins=24, color="#6F4E7C", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(label + (" (positive is better)" if higher_better else " (negative is better)"))
        axis.set_xlabel("Adaptive minus global")
        axis.set_ylabel("Cases")
    fig.tight_layout()
    fig.savefig(output_dir / "v12_case_metric_deltas.png", dpi=180)
    plt.close(fig)


def run_v12(
    package_root: Path,
    config: V12Config | None = None,
    *,
    preflight: dict | None = None,
) -> dict:
    config = config or V12Config()
    config.validate()
    started = time.time()
    preflight = preflight or preflight_v12(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "v12_complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("status") == "complete"
            and completion.get("v12_signature_sha256") == preflight["signature"]["v12_signature_sha256"]
        ):
            print("V12 is already complete for this exact frozen signature.", flush=True)
            return completion

    _atomic_json(output_dir / "run_configuration.json", asdict(config))
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "oracle_gain_grid",
        "started_epoch_seconds": started,
        "final_test_cases_read": 0,
    })

    grid_metrics = evaluate_oracle_grid(preflight, config)
    grid_metrics.to_csv(output_dir / "oracle_gain_grid_case_metrics.csv.gz", index=False, compression="gzip")
    targets = select_oracle_targets(grid_metrics, config)
    targets.to_csv(output_dir / "oracle_gain_targets.csv", index=False)

    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "nested_group_gain_validation",
        "oracle_cases_complete": len(targets),
        "final_test_cases_read": 0,
    })
    nested = nested_group_gain_validation(targets, grid_metrics, config)
    nested["candidates"].to_csv(output_dir / "gain_candidate_registry.csv", index=False)
    nested["outer_folds"].to_csv(output_dir / "nested_outer_fold_manifest.csv", index=False)
    nested["inner_scores"].to_csv(output_dir / "nested_inner_candidate_scores.csv", index=False)
    nested["outer_selections"].to_csv(output_dir / "nested_outer_selected_candidates.csv", index=False)
    nested["final_folds"].to_csv(output_dir / "final_configuration_cv_fold_manifest.csv", index=False)
    nested["final_scores"].to_csv(output_dir / "final_configuration_candidate_scores.csv", index=False)
    nested["oof"].to_csv(output_dir / "gain_layer_nested_oof_predictions.csv", index=False)
    _atomic_json(output_dir / "gain_model_refit_on_149.json", nested["final_state"])

    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "complete_case_oof_evaluation",
        "selected_refit_candidate": nested["final_state"]["candidate_id"],
        "final_test_cases_read": 0,
    })
    adaptive = evaluate_adaptive_oof(preflight, nested["oof"], config)
    case_metrics = build_comparison_case_metrics(grid_metrics, targets, adaptive, config)
    split_metrics = aggregate_scopes(case_metrics)
    bootstrap = paired_group_bootstrap(case_metrics, config)
    gates, decision = promotion_decision(split_metrics, nested["oof"], config)
    formulas = save_formula_artifacts(preflight, nested["final_state"], decision, config)

    case_metrics.to_csv(output_dir / "v12_comparison_case_metrics.csv.gz", index=False, compression="gzip")
    split_metrics.to_csv(output_dir / "v12_comparison_scope_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "v12_paired_similarity_group_bootstrap.csv", index=False)
    gates.to_csv(output_dir / "v12_promotion_gates.csv", index=False)
    _atomic_json(output_dir / "v12_promotion_decision.json", decision)
    save_diagnostic_plots(targets, nested["oof"], case_metrics, output_dir, config)

    rationale = {
        "problem": "A single V11 gain improved average validation behaviour but could not resolve opposite case-level tail errors.",
        "v12_hypothesis": "A bounded gain predicted from predictor-only case context can adapt correction strength without changing the frozen V11 spatial formula.",
        "validation": "Five outer similarity-group folds with four-fold grouped inner model selection; all 149 non-final cases receive one gain-layer OOF prediction.",
        "methodological_limit": "The V11 base formula was developed on the legacy 119 training cases, so these are gain-layer OOF estimates, not full-pipeline OOF estimates.",
        "p95_p99_role": "P95/P99 are response-distribution evaluation metrics and oracle-objective diagnostics, not input clipping limits or confidence intervals.",
        "oracle_role": "Stress-derived oracle gains are training targets and a non-deployable upper bound; deployment gain uses predictor summaries only.",
        "safe_fallback": "If any formal V12 gate fails, the selected deployment artifact retains the frozen V11 global gain 0.90.",
        "final_test_policy": "The 50 locked final-case element files are inventoried but not read.",
        "references": [
            "Huber, P. J. (1964), Robust Estimation of a Location Parameter.",
            "Jacobs et al. (1991), Adaptive Mixtures of Local Experts.",
            "Hastie and Tibshirani (1993), Varying-Coefficient Models.",
            "Cawley and Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias.",
            "scikit-learn GroupKFold, HuberRegressor and Ridge documentation.",
        ],
    }
    _atomic_json(output_dir / "design_rationale.json", rationale)

    completion = {
        "status": "complete",
        "iteration": config.iteration,
        "method": "bounded_case_adaptive_tail_gain_with_nested_similarity_group_validation",
        "v12_signature_sha256": preflight["signature"]["v12_signature_sha256"],
        "development_cases": 149,
        "nested_outer_folds": config.outer_folds,
        "nested_inner_folds": config.inner_folds,
        "gain_candidates": len(nested["candidates"]),
        "oracle_grid_values_per_case": len(config.oracle_gain_grid),
        "selected_refit_candidate": nested["final_state"]["candidate_id"],
        "promotion_status": decision["promotion_status"],
        "failed_promotion_gates": decision["failed_gates"],
        "selected_formula_method": formulas["selected"]["method"],
        "final_test_cases_read": 0,
        "elapsed_seconds": time.time() - started,
        "output_directory": str(output_dir),
    }
    _atomic_json(completion_path, completion)
    _atomic_json(output_dir / "run_status.json", {**completion, "stage": "complete"})
    return completion
