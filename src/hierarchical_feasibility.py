"""Feasibility diagnostic for a hierarchical FEM stress surrogate.

This module tests whether separating case-level stress amplitude from the
within-case spatial shape is more learnable than one direct pointwise model.
The locked 50-case final test is never read.  It is intentionally a diagnostic
baseline, not the final symbolic-regression implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import math
import sys
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from ct3_common import (  # noqa: E402
    RANDOM_SEED,
    TARGET_COL,
    aggregate_case_metrics,
    assert_no_final_cases,
    build_paths,
    case_path_lookup,
    cases_for_role,
    development_case_ids,
    discover_case_files,
    evaluate_prediction_arrays,
    final_test_case_ids,
    load_frozen_manifest,
    make_case_tail_weights,
    read_complete_case,
)


LOCAL_FEATURES = [
    "fluence_rate",
    "temperature",
    "weight_loss_rate",
    "rho",
    "theta_sin",
    "theta_cos",
    "z",
]

PHYSICAL_CONTEXT_FEATURES = [
    f"{feature}_{stat}"
    for feature in ["fluence_rate", "temperature", "weight_loss_rate"]
    for stat in ["mean", "std", "min", "p95", "max"]
]

GEOMETRY_CONTEXT_FEATURES = [
    f"{feature}_{stat}"
    for feature in ["rho", "z"]
    for stat in ["mean", "std", "min", "max"]
] + [
    f"{feature}_{stat}"
    for feature in ["theta_sin", "theta_cos"]
    for stat in ["mean", "std"]
]

FULL_CONTEXT_FEATURES = PHYSICAL_CONTEXT_FEATURES + GEOMETRY_CONTEXT_FEATURES
STANDARDISED_LOCAL_FEATURES = [f"{feature}_within_case_z" for feature in LOCAL_FEATURES]
MODEL_FEATURES = LOCAL_FEATURES + STANDARDISED_LOCAL_FEATURES + FULL_CONTEXT_FEATURES


@dataclass(frozen=True)
class FeasibilityConfig:
    iteration: int = 2
    rows_per_case: int = 5_000
    hgb_max_iter: int = 180
    output_subdir: str = "iteration_2_pilot"
    force_rebuild_sample: bool = False


def _summary_paths(package_root: Path, iteration: int) -> tuple[Path, Path]:
    case_summary = (
        package_root
        / "outputs"
        / "00_qc_sensitivity_ablation"
        / "development_case_summary.csv"
    )
    discovery_manifest = (
        package_root
        / "outputs"
        / "02_symbolic_search_v4_context_interaction"
        / "formal_5000_per_case"
        / f"discovery_manifest_iteration_{iteration}.csv.gz"
    )
    return case_summary, discovery_manifest


def load_inputs(package_root: Path, config: FeasibilityConfig) -> dict:
    paths = build_paths(package_root)
    _, inventory = discover_case_files(paths.case_dir)
    path_by_case = case_path_lookup(inventory)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    development_ids = development_case_ids(manifest)
    final_ids = final_test_case_ids(manifest)
    train_ids = cases_for_role(manifest, config.iteration, "train")
    validation_ids = cases_for_role(manifest, config.iteration, "validation")
    internal_ids = cases_for_role(manifest, config.iteration, "internal_test")
    assert_no_final_cases(train_ids + validation_ids + internal_ids, manifest)

    case_summary_path, discovery_manifest_path = _summary_paths(
        package_root, config.iteration
    )
    if not case_summary_path.exists():
        raise FileNotFoundError(
            f"Missing development summary: {case_summary_path}. Run notebook 00 first."
        )
    case_summary = pd.read_csv(case_summary_path).set_index("case_id", drop=False)
    if set(case_summary.index) != set(development_ids):
        raise ValueError("Development case summary does not match the frozen 149-case pool")

    required_summary = {
        "stress_mean",
        "stress_p95",
        "stress_p99",
        *FULL_CONTEXT_FEATURES,
    }
    missing = sorted(required_summary - set(case_summary.columns))
    if missing:
        raise ValueError(f"Development summary is missing required columns: {missing}")
    if (case_summary["stress_p95"] <= case_summary["stress_mean"]).any():
        raise ValueError("At least one case has non-positive p95-minus-mean stress scale")

    output_dir = (
        package_root
        / "outputs"
        / "05_hierarchical_feasibility_v5"
        / config.output_subdir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "paths": paths,
        "inventory": inventory,
        "path_by_case": path_by_case,
        "manifest": manifest,
        "case_summary": case_summary,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "internal_ids": internal_ids,
        "final_ids": final_ids,
        "discovery_manifest_path": discovery_manifest_path,
        "output_dir": output_dir,
    }


def _fixed_width_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        return np.zeros(len(values), dtype=np.int16)
    scaled = np.clip((values - lower) / span, 0.0, 1.0)
    return np.minimum((scaled * n_bins).astype(np.int16), n_bins - 1)


def _joint_stratum_codes(local_matrix: np.ndarray) -> np.ndarray:
    feature_index = {name: index for index, name in enumerate(LOCAL_FEATURES)}
    theta = np.arctan2(
        local_matrix[:, feature_index["theta_sin"]],
        local_matrix[:, feature_index["theta_cos"]],
    )
    arrays_and_bins = [
        (local_matrix[:, feature_index["rho"]], 6),
        (theta, 8),
        (local_matrix[:, feature_index["z"]], 8),
        (local_matrix[:, feature_index["fluence_rate"]], 4),
        (local_matrix[:, feature_index["temperature"]], 4),
        (local_matrix[:, feature_index["weight_loss_rate"]], 4),
    ]
    codes = np.zeros(len(local_matrix), dtype=np.int64)
    for values, n_bins in arrays_and_bins:
        codes = codes * n_bins + _fixed_width_bins(values, n_bins)
    return codes


def _stratified_indices(
    pool_indices: np.ndarray,
    stratum_codes: np.ndarray,
    quota: int,
    seed: int,
) -> np.ndarray:
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    if quota <= 0 or len(pool_indices) == 0:
        return np.empty(0, dtype=np.int64)
    if len(pool_indices) <= quota:
        return np.sort(pool_indices)
    rng = np.random.default_rng(seed)
    random_key = rng.random(len(pool_indices))
    pool_codes = stratum_codes[pool_indices]
    order = np.lexsort((random_key, pool_codes))
    ordered = pool_indices[order]
    ordered_codes = pool_codes[order]
    first_in_stratum = np.r_[True, ordered_codes[1:] != ordered_codes[:-1]]
    representatives = ordered[first_in_stratum]
    if len(representatives) >= quota:
        positions = np.linspace(0, len(representatives) - 1, quota, dtype=np.int64)
        return np.sort(representatives[positions])
    selected_mask = np.zeros(len(stratum_codes), dtype=bool)
    selected_mask[representatives] = True
    remaining = pool_indices[~selected_mask[pool_indices]]
    needed = quota - len(representatives)
    selected = np.concatenate([
        representatives,
        remaining[rng.permutation(len(remaining))[:needed]],
    ])
    return np.sort(selected)


def _build_v4_discovery_selection(
    case_id: str,
    element_ids: np.ndarray,
    stress: np.ndarray,
    local_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the exact V4 tail and joint-field stratified design."""
    q90, q95, q99 = np.quantile(stress, [0.90, 0.95, 0.99])
    tier = np.zeros(len(stress), dtype=np.int8)
    tier[stress >= q90] = 1
    tier[stress >= q95] = 2
    tier[stress >= q99] = 3
    stratum_codes = _joint_stratum_codes(local_matrix)
    quotas = [3_500, 500, 500, 500]
    case_number = int(case_id.split("_")[-1])
    selected_parts = []
    for tier_index, quota in enumerate(quotas):
        pool = np.flatnonzero(tier == tier_index)
        chosen = _stratified_indices(
            pool,
            stratum_codes,
            quota,
            RANDOM_SEED + 10_000 * case_number + 101 * tier_index,
        )
        selected_parts.append(chosen)
    positions = np.sort(np.concatenate(selected_parts))
    if len(positions) < sum(quotas):
        selected_mask = np.zeros(len(stress), dtype=bool)
        selected_mask[positions] = True
        fill = _stratified_indices(
            np.flatnonzero(~selected_mask),
            stratum_codes,
            sum(quotas) - len(positions),
            RANDOM_SEED + 10_000 * case_number + 999,
        )
        positions = np.sort(np.concatenate([positions, fill]))
    if len(positions) != sum(quotas):
        raise AssertionError(f"{case_id}: expected {sum(quotas)} discovery rows")

    full_case_weights, _ = make_case_tail_weights(stress)
    selected_tiers = tier[positions]
    corrected_weights = np.empty(len(positions), dtype=np.float64)
    for tier_index in range(len(quotas)):
        selected_tier = selected_tiers == tier_index
        selected_count = int(selected_tier.sum())
        if selected_count == 0:
            raise AssertionError(f"{case_id}: selected stress tier {tier_index} is empty")
        full_tier_mass = float(full_case_weights[tier == tier_index].sum())
        corrected_weights[selected_tier] = full_tier_mass / selected_count
    corrected_weights /= corrected_weights.mean()
    return positions, corrected_weights


def _case_summary_row(case_summary: pd.DataFrame, case_id: str) -> pd.Series:
    row = case_summary.loc[case_id]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Duplicate summary rows for {case_id}")
    return row


def build_model_matrix(frame: pd.DataFrame, summary: pd.Series) -> np.ndarray:
    local = frame[LOCAL_FEATURES].to_numpy(dtype=np.float32)
    means = np.asarray([summary[f"{feature}_mean"] for feature in LOCAL_FEATURES], dtype=np.float32)
    stds = np.asarray([summary[f"{feature}_std"] for feature in LOCAL_FEATURES], dtype=np.float32)
    stds = np.maximum(stds, np.float32(1e-8))
    within_case = (local - means) / stds
    context = summary[FULL_CONTEXT_FEATURES].to_numpy(dtype=np.float32)
    context_matrix = np.broadcast_to(context, (len(frame), len(context)))
    return np.ascontiguousarray(
        np.concatenate([local, within_case, context_matrix], axis=1),
        dtype=np.float32,
    )


def assemble_training_sample(inputs: dict, config: FeasibilityConfig) -> dict:
    train_ids = inputs["train_ids"]
    n_rows = len(train_ids) * config.rows_per_case
    X = np.empty((n_rows, len(MODEL_FEATURES)), dtype=np.float32)
    y_direct = np.empty(n_rows, dtype=np.float32)
    y_shape = np.empty(n_rows, dtype=np.float32)
    weights = np.empty(n_rows, dtype=np.float32)
    sample_records = []
    sample_manifest_parts = []

    existing_manifest = None
    local_manifest_path = inputs["output_dir"] / "training_sample_manifest.csv.gz"
    manifest_candidates = [local_manifest_path, inputs["discovery_manifest_path"]]
    manifest_path = next((path for path in manifest_candidates if path.exists()), None)
    if manifest_path is not None and not config.force_rebuild_sample:
        existing_manifest = pd.read_csv(manifest_path)
        existing_manifest = existing_manifest[
            existing_manifest["case_id"].isin(train_ids)
        ].copy()
        counts = existing_manifest.groupby("case_id").size()
        if set(counts.index) != set(train_ids) or not (counts == config.rows_per_case).all():
            raise ValueError("Existing V4 discovery manifest does not match this training split")

    cursor = 0
    started = time.perf_counter()
    for index, case_id in enumerate(train_ids, start=1):
        print(f"[{index}/{len(train_ids)}] Assemble training sample: {case_id}", flush=True)
        frame, _ = read_complete_case(inputs["path_by_case"][case_id])
        summary = _case_summary_row(inputs["case_summary"], case_id)
        stress = frame[TARGET_COL].to_numpy(dtype=np.float64)
        element_ids = frame["element_id"].to_numpy(dtype=np.int64)

        if existing_manifest is not None:
            case_manifest = existing_manifest[existing_manifest["case_id"] == case_id]
            wanted_ids = case_manifest["element_id"].to_numpy(dtype=np.int64)
            positions = pd.Index(element_ids).get_indexer(wanted_ids)
            if (positions < 0).any():
                raise ValueError(f"V4 discovery manifest contains unknown ElementID for {case_id}")
            inclusion_weights = case_manifest["discovery_weight"].to_numpy(dtype=np.float64)
            sample_source = (
                "reused_local_feasibility_manifest"
                if manifest_path == local_manifest_path
                else "reused_v4_iteration_manifest"
            )
        else:
            positions, inclusion_weights = _build_v4_discovery_selection(
                case_id,
                element_ids,
                stress,
                frame[LOCAL_FEATURES].to_numpy(dtype=np.float32),
            )
            sample_source = "rebuilt_exact_v4_joint_stratified_sample"

        sampled = frame.iloc[positions]
        next_cursor = cursor + len(sampled)
        X[cursor:next_cursor] = build_model_matrix(sampled, summary)
        sampled_stress = sampled[TARGET_COL].to_numpy(dtype=np.float32)
        offset = np.float32(summary["stress_mean"])
        scale = np.float32(summary["stress_p95"] - summary["stress_mean"])
        y_direct[cursor:next_cursor] = sampled_stress
        y_shape[cursor:next_cursor] = (sampled_stress - offset) / scale

        _, tail_audit = make_case_tail_weights(stress)
        # Existing and newly built manifests both contain the exact V4 weights,
        # corrected for tail quotas and inclusion probability.
        combined = inclusion_weights.copy()
        combined /= combined.mean()
        weights[cursor:next_cursor] = combined.astype(np.float32)
        sample_manifest_parts.append(pd.DataFrame({
            "iteration": config.iteration,
            "case_id": case_id,
            "element_id": sampled["element_id"].to_numpy(dtype=np.int64),
            "discovery_weight": combined.astype(np.float32),
        }))
        sample_records.append({
            "iteration": config.iteration,
            "case_id": case_id,
            "n_rows": len(sampled),
            "sample_source": sample_source,
            "stress_mean_offset": float(offset),
            "stress_p95_scale": float(scale),
            "sample_stress_p95": float(np.quantile(sampled_stress, 0.95)),
            "weight_mean": float(combined.mean()),
            "weight_max": float(combined.max()),
            "tail_q99": float(tail_audit["q99"]),
        })
        cursor = next_cursor
        del frame, sampled, stress, element_ids
        gc.collect()

    if cursor != n_rows:
        raise AssertionError(f"Assembled {cursor} rows, expected {n_rows}")
    weights /= np.float32(weights.mean(dtype=np.float64))
    return {
        "X": X,
        "y_direct": y_direct,
        "y_shape": y_shape,
        "weights": weights,
        "audit": pd.DataFrame(sample_records),
        "manifest": pd.concat(sample_manifest_parts, ignore_index=True),
        "seconds": time.perf_counter() - started,
    }


def _case_target_matrix(summary: pd.DataFrame, case_ids: Sequence[str]) -> np.ndarray:
    selected = summary.loc[list(case_ids)]
    scale = selected["stress_p95"] - selected["stress_mean"]
    return np.column_stack([
        selected["stress_mean"].to_numpy(dtype=np.float64),
        np.log(scale.to_numpy(dtype=np.float64)),
    ])


def _decode_case_targets(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(encoded, dtype=np.float64)
    return encoded[:, 0], np.exp(encoded[:, 1])


def fit_case_level_models(inputs: dict, config: FeasibilityConfig) -> dict:
    summary = inputs["case_summary"]
    train_ids = inputs["train_ids"]
    validation_ids = inputs["validation_ids"]
    internal_ids = inputs["internal_ids"]
    candidates = []
    prediction_tables = []

    for context_name, context_features in [
        ("physical_context", PHYSICAL_CONTEXT_FEATURES),
        ("physical_plus_geometry_context", FULL_CONTEXT_FEATURES),
    ]:
        X_train = summary.loc[train_ids, context_features].to_numpy(dtype=np.float64)
        y_train_raw = _case_target_matrix(summary, train_ids)
        y_mean = y_train_raw.mean(axis=0)
        y_std = np.maximum(y_train_raw.std(axis=0), 1e-12)
        y_train = (y_train_raw - y_mean) / y_std

        models = {
            "RidgeCV": make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-5, 5, 31)),
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.75,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        }
        for model_name, model in models.items():
            fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_start
            for split, case_ids in [
                ("validation", validation_ids),
                ("internal_test", internal_ids),
            ]:
                X_split = summary.loc[case_ids, context_features].to_numpy(dtype=np.float64)
                encoded = model.predict(X_split) * y_std + y_mean
                predicted_offset, predicted_scale = _decode_case_targets(encoded)
                actual_offset = summary.loc[case_ids, "stress_mean"].to_numpy(dtype=np.float64)
                actual_scale = (
                    summary.loc[case_ids, "stress_p95"].to_numpy(dtype=np.float64)
                    - actual_offset
                )
                predicted_p95 = predicted_offset + predicted_scale
                actual_p95 = actual_offset + actual_scale
                rows = pd.DataFrame({
                    "iteration": config.iteration,
                    "context_design": context_name,
                    "case_model": model_name,
                    "split": split,
                    "case_id": case_ids,
                    "actual_offset_mean": actual_offset,
                    "predicted_offset_mean": predicted_offset,
                    "actual_scale_p95_minus_mean": actual_scale,
                    "predicted_scale_p95_minus_mean": predicted_scale,
                    "actual_p95": actual_p95,
                    "predicted_p95": predicted_p95,
                })
                rows["offset_absolute_error"] = np.abs(
                    rows["predicted_offset_mean"] - rows["actual_offset_mean"]
                )
                rows["scale_relative_error"] = np.abs(
                    rows["predicted_scale_p95_minus_mean"]
                    - rows["actual_scale_p95_minus_mean"]
                ) / np.maximum(np.abs(rows["actual_scale_p95_minus_mean"]), 1e-12)
                rows["p95_relative_error"] = np.abs(
                    rows["predicted_p95"] - rows["actual_p95"]
                ) / np.maximum(np.abs(rows["actual_p95"]), 1e-12)
                prediction_tables.append(rows)

                offset_rmse = float(np.sqrt(np.mean(
                    (predicted_offset - actual_offset) ** 2
                )))
                log_scale_actual = np.log(actual_scale)
                log_scale_predicted = np.log(predicted_scale)
                log_scale_rmse = float(np.sqrt(np.mean(
                    (log_scale_predicted - log_scale_actual) ** 2
                )))
                candidates.append({
                    "iteration": config.iteration,
                    "context_design": context_name,
                    "case_model": model_name,
                    "split": split,
                    "n_cases": len(case_ids),
                    "offset_rmse": offset_rmse,
                    "log_scale_rmse": log_scale_rmse,
                    "mean_scale_relative_error": float(rows["scale_relative_error"].mean()),
                    "mean_p95_relative_error": float(rows["p95_relative_error"].mean()),
                    "p95_spearman": float(rows["actual_p95"].corr(
                        rows["predicted_p95"], method="spearman"
                    )),
                    "fit_seconds": fit_seconds,
                })

            candidates[-2]["model_object"] = model
            candidates[-2]["target_mean"] = y_mean
            candidates[-2]["target_std"] = y_std
            candidates[-2]["context_features"] = context_features

    metrics = pd.DataFrame([
        {key: value for key, value in row.items() if key not in {
            "model_object", "target_mean", "target_std", "context_features"
        }}
        for row in candidates
    ])
    validation_metrics = metrics[metrics["split"] == "validation"].copy()
    train_offset_std = float(summary.loc[train_ids, "stress_mean"].std(ddof=0))
    train_log_scale_std = float(np.log(
        summary.loc[train_ids, "stress_p95"] - summary.loc[train_ids, "stress_mean"]
    ).std(ddof=0))
    validation_metrics["selection_score"] = (
        validation_metrics["offset_rmse"] / max(train_offset_std, 1e-12)
        + validation_metrics["log_scale_rmse"] / max(train_log_scale_std, 1e-12)
    )
    selected_key = validation_metrics.sort_values(
        ["selection_score", "mean_p95_relative_error", "context_design", "case_model"]
    ).iloc[0]
    selected_record = next(
        row for row in candidates
        if row["split"] == "validation"
        and row["context_design"] == selected_key["context_design"]
        and row["case_model"] == selected_key["case_model"]
    )
    return {
        "model": selected_record["model_object"],
        "target_mean": selected_record["target_mean"],
        "target_std": selected_record["target_std"],
        "context_features": list(selected_record["context_features"]),
        "context_design": selected_record["context_design"],
        "model_name": selected_record["case_model"],
        "metrics": metrics.merge(
            validation_metrics[["context_design", "case_model", "selection_score"]],
            on=["context_design", "case_model"],
            how="left",
        ),
        "predictions": pd.concat(prediction_tables, ignore_index=True),
    }


def _hgb(config: FeasibilityConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=config.hgb_max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=256,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def fit_element_models(sample: dict, config: FeasibilityConfig) -> dict:
    timings = []
    direct = _hgb(config)
    started = time.perf_counter()
    direct.fit(sample["X"], sample["y_direct"], sample_weight=sample["weights"])
    timings.append({"model": "direct_context_control", "fit_seconds": time.perf_counter() - started})

    shape = _hgb(config)
    started = time.perf_counter()
    shape.fit(sample["X"], sample["y_shape"], sample_weight=sample["weights"])
    timings.append({"model": "normalised_spatial_shape", "fit_seconds": time.perf_counter() - started})
    return {"direct": direct, "shape": shape, "timings": pd.DataFrame(timings)}


def predict_case_amplitude(case_model_bundle: dict, summary: pd.Series) -> tuple[float, float]:
    features = case_model_bundle["context_features"]
    X = summary[features].to_numpy(dtype=np.float64).reshape(1, -1)
    encoded = (
        case_model_bundle["model"].predict(X)
        * case_model_bundle["target_std"]
        + case_model_bundle["target_mean"]
    )
    offset, scale = _decode_case_targets(encoded)
    return float(offset[0]), float(scale[0])


def evaluate_models(
    inputs: dict,
    config: FeasibilityConfig,
    element_models: dict,
    case_model_bundle: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    tail_rows = []
    shape_rows = []
    evaluation_started = time.perf_counter()
    roles = [
        ("validation", inputs["validation_ids"]),
        ("internal_test", inputs["internal_ids"]),
    ]
    total_cases = sum(len(ids) for _, ids in roles)
    counter = 0
    for split, case_ids in roles:
        for case_id in case_ids:
            counter += 1
            print(f"[{counter}/{total_cases}] Full-case evaluation: {split} / {case_id}", flush=True)
            frame, _ = read_complete_case(inputs["path_by_case"][case_id])
            summary = _case_summary_row(inputs["case_summary"], case_id)
            y = frame[TARGET_COL].to_numpy(dtype=np.float64)
            X = build_model_matrix(frame, summary)
            direct_prediction = np.asarray(element_models["direct"].predict(X), dtype=np.float64)
            shape_prediction = np.asarray(element_models["shape"].predict(X), dtype=np.float64)

            actual_offset = float(summary["stress_mean"])
            actual_scale = float(summary["stress_p95"] - summary["stress_mean"])
            actual_shape = (y - actual_offset) / actual_scale
            predicted_offset, predicted_scale = predict_case_amplitude(
                case_model_bundle, summary
            )
            oracle_prediction = actual_offset + actual_scale * shape_prediction
            deployable_prediction = predicted_offset + predicted_scale * shape_prediction

            predictions = {
                "direct_context_control": direct_prediction,
                "oracle_hierarchical": oracle_prediction,
                "deployable_hierarchical": deployable_prediction,
            }
            for model_name, prediction in predictions.items():
                metric_rows.append({
                    "iteration": config.iteration,
                    "split": split,
                    "model": model_name,
                    "case_id": case_id,
                    **evaluate_prediction_arrays(y, prediction),
                })
                tail_rows.append({
                    "iteration": config.iteration,
                    "split": split,
                    "model": model_name,
                    "case_id": case_id,
                    "actual_p95": float(np.quantile(y, 0.95)),
                    "predicted_p95": float(np.quantile(prediction, 0.95)),
                    "actual_p99": float(np.quantile(y, 0.99)),
                    "predicted_p99": float(np.quantile(prediction, 0.99)),
                    "actual_mean": float(y.mean()),
                    "predicted_mean": float(prediction.mean()),
                })

            shape_error = shape_prediction - actual_shape
            shape_sst = float(np.sum((actual_shape - actual_shape.mean()) ** 2))
            shape_sse = float(np.sum(shape_error ** 2))
            direct_rank_metrics = evaluate_prediction_arrays(y, direct_prediction)
            shape_rank_metrics = evaluate_prediction_arrays(actual_shape, shape_prediction)
            shape_rows.append({
                "iteration": config.iteration,
                "split": split,
                "case_id": case_id,
                "shape_rmse": float(np.sqrt(np.mean(shape_error ** 2))),
                "shape_r2": 1.0 - shape_sse / shape_sst if shape_sst > 0 else np.nan,
                "shape_top1pct_hotspot_overlap": shape_rank_metrics["top1pct_hotspot_overlap"],
                "shape_top1_recall_in_predicted_top5": shape_rank_metrics["top1_recall_in_predicted_top5"],
                "direct_top1pct_hotspot_overlap": direct_rank_metrics["top1pct_hotspot_overlap"],
                "direct_top1_recall_in_predicted_top5": direct_rank_metrics["top1_recall_in_predicted_top5"],
                "actual_offset_mean": actual_offset,
                "predicted_offset_mean": predicted_offset,
                "actual_scale_p95_minus_mean": actual_scale,
                "predicted_scale_p95_minus_mean": predicted_scale,
            })
            del frame, X, y, direct_prediction, shape_prediction
            del oracle_prediction, deployable_prediction, actual_shape
            gc.collect()

    case_metrics = pd.DataFrame(metric_rows)
    split_metrics = aggregate_case_metrics(
        case_metrics, ["iteration", "split", "model"]
    )
    split_metrics["evaluation_seconds_total"] = time.perf_counter() - evaluation_started
    return case_metrics, split_metrics, pd.DataFrame(shape_rows), pd.DataFrame(tail_rows)


def build_decision_table(split_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in split_metrics.groupby("split"):
        indexed = group.set_index("model")
        direct = indexed.loc["direct_context_control"]
        oracle = indexed.loc["oracle_hierarchical"]
        deployable = indexed.loc["deployable_hierarchical"]
        rows.append({
            "split": split,
            "oracle_macro_rmse_ratio_to_direct": oracle["macro_rmse"] / direct["macro_rmse"],
            "deployable_macro_rmse_ratio_to_direct": deployable["macro_rmse"] / direct["macro_rmse"],
            "oracle_p95_error_ratio_to_direct": oracle["mean_p95_relative_error"] / max(direct["mean_p95_relative_error"], 1e-12),
            "deployable_p95_error_ratio_to_direct": deployable["mean_p95_relative_error"] / max(direct["mean_p95_relative_error"], 1e-12),
            "oracle_p99_error_ratio_to_direct": oracle["mean_p99_relative_error"] / max(direct["mean_p99_relative_error"], 1e-12),
            "deployable_p99_error_ratio_to_direct": deployable["mean_p99_relative_error"] / max(direct["mean_p99_relative_error"], 1e-12),
            "oracle_shape_supported": bool(
                oracle["macro_rmse"] <= 1.05 * direct["macro_rmse"]
                and oracle["mean_p99_relative_error"] <= 1.05 * direct["mean_p99_relative_error"]
            ),
            "deployable_hierarchy_supported": bool(
                deployable["macro_rmse"] <= 1.05 * direct["macro_rmse"]
                and deployable["mean_p99_relative_error"] <= 1.05 * direct["mean_p99_relative_error"]
            ),
        })
    return pd.DataFrame(rows)


def save_plots(
    output_dir: Path,
    split_metrics: pd.DataFrame,
    tail_predictions: pd.DataFrame,
    case_model_predictions: pd.DataFrame,
    case_model_bundle: dict,
) -> None:
    plot_table = split_metrics.copy()
    plot_table["p95_error_percent"] = 100 * plot_table["mean_p95_relative_error"]
    plot_table["p99_error_percent"] = 100 * plot_table["mean_p99_relative_error"]
    model_order = [
        "direct_context_control",
        "oracle_hierarchical",
        "deployable_hierarchical",
    ]
    colors = ["#2f6b9a", "#2a9d6f", "#d97732"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title in zip(
        axes,
        ["macro_rmse", "p95_error_percent", "p99_error_percent"],
        ["Macro RMSE", "Mean P95 relative error (%)", "Mean P99 relative error (%)"],
    ):
        pivot = plot_table.pivot(index="split", columns="model", values=metric)
        pivot = pivot.reindex(columns=model_order)
        pivot.plot(kind="bar", ax=axis, color=colors, width=0.75)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=0)
        if axis is not axes[0]:
            axis.get_legend().remove()
    axes[0].legend(title="Model", fontsize=8)
    fig.suptitle("Direct versus hierarchical stress prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "hierarchical_method_comparison.png", dpi=180)
    plt.close(fig)

    validation = tail_predictions[tail_predictions["split"] == "validation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, model_name, color in zip(axes, model_order, colors):
        rows = validation[validation["model"] == model_name]
        axis.scatter(rows["actual_p95"], rows["predicted_p95"], color=color, s=45)
        lower = min(rows["actual_p95"].min(), rows["predicted_p95"].min())
        upper = max(rows["actual_p95"].max(), rows["predicted_p95"].max())
        axis.plot([lower, upper], [lower, upper], color="#555555", linestyle="--")
        axis.set_title(model_name.replace("_", " "))
        axis.set_xlabel("Actual case P95 stress")
    axes[0].set_ylabel("Predicted case P95 stress")
    fig.suptitle("Validation case-level P95 adaptation")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_p95_adaptation.png", dpi=180)
    plt.close(fig)

    selected = case_model_predictions[
        (case_model_predictions["context_design"] == case_model_bundle["context_design"])
        & (case_model_predictions["case_model"] == case_model_bundle["model_name"])
        & (case_model_predictions["split"] == "validation")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, actual_col, predicted_col, title in [
        (axes[0], "actual_offset_mean", "predicted_offset_mean", "Case stress mean"),
        (
            axes[1],
            "actual_scale_p95_minus_mean",
            "predicted_scale_p95_minus_mean",
            "Case scale: P95 minus mean",
        ),
    ]:
        axis.scatter(selected[actual_col], selected[predicted_col], color="#7a4eab", s=45)
        lower = min(selected[actual_col].min(), selected[predicted_col].min())
        upper = max(selected[actual_col].max(), selected[predicted_col].max())
        axis.plot([lower, upper], [lower, upper], color="#555555", linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
    fig.suptitle(
        f"Selected case-level model: {case_model_bundle['context_design']} / "
        f"{case_model_bundle['model_name']}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "case_amplitude_prediction.png", dpi=180)
    plt.close(fig)


def run_feasibility(
    package_root: Path | None = None,
    config: FeasibilityConfig | None = None,
) -> dict:
    package_root = (package_root or MODULE_ROOT).resolve()
    config = config or FeasibilityConfig()
    total_started = time.perf_counter()
    inputs = load_inputs(package_root, config)
    output_dir = inputs["output_dir"]

    print("Hierarchical feasibility diagnostic", flush=True)
    print(f"Package root: {package_root}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f"Iteration {config.iteration}: train={len(inputs['train_ids'])}, "
        f"validation={len(inputs['validation_ids'])}, "
        f"internal_test={len(inputs['internal_ids'])}; final test unread={len(inputs['final_ids'])}",
        flush=True,
    )

    sample = assemble_training_sample(inputs, config)
    sample["audit"].to_csv(output_dir / "training_sample_audit.csv", index=False)
    sample["manifest"].to_csv(
        output_dir / "training_sample_manifest.csv.gz",
        index=False,
        compression="gzip",
    )

    case_model_bundle = fit_case_level_models(inputs, config)
    case_model_bundle["metrics"].to_csv(
        output_dir / "case_level_model_metrics.csv", index=False
    )
    case_model_bundle["predictions"].to_csv(
        output_dir / "case_level_model_predictions.csv", index=False
    )

    element_models = fit_element_models(sample, config)
    element_models["timings"].to_csv(output_dir / "element_model_fit_timing.csv", index=False)
    case_metrics, split_metrics, shape_metrics, tail_predictions = evaluate_models(
        inputs, config, element_models, case_model_bundle
    )
    decision_table = build_decision_table(split_metrics)

    case_metrics.to_csv(output_dir / "full_case_metrics.csv.gz", index=False, compression="gzip")
    split_metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    shape_metrics.to_csv(output_dir / "normalised_shape_case_metrics.csv", index=False)
    tail_predictions.to_csv(output_dir / "case_tail_predictions.csv", index=False)
    decision_table.to_csv(output_dir / "feasibility_decision_table.csv", index=False)
    save_plots(
        output_dir,
        split_metrics,
        tail_predictions,
        case_model_bundle["predictions"],
        case_model_bundle,
    )

    result = {
        "status": "complete",
        "diagnostic_only": True,
        "iteration": config.iteration,
        "random_seed": RANDOM_SEED,
        "rows_per_training_case": config.rows_per_case,
        "training_sample_rows": len(sample["y_direct"]),
        "full_case_evaluation": True,
        "final_test_read": False,
        "decomposition": "stress = case_mean + (case_p95 - case_mean) * normalised_shape",
        "selected_case_context_design": case_model_bundle["context_design"],
        "selected_case_model": case_model_bundle["model_name"],
        "sample_assembly_seconds": sample["seconds"],
        "total_seconds": time.perf_counter() - total_started,
        "output_directory": str(output_dir),
        "decision_records": decision_table.to_dict("records"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\nSplit metrics:", flush=True)
    print(split_metrics.to_string(index=False), flush=True)
    print("\nFeasibility decision:", flush=True)
    print(decision_table.to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {output_dir}", flush=True)
    return {
        "config": config,
        "inputs": inputs,
        "sample_audit": sample["audit"],
        "case_model_metrics": case_model_bundle["metrics"],
        "case_model_predictions": case_model_bundle["predictions"],
        "split_metrics": split_metrics,
        "case_metrics": case_metrics,
        "shape_metrics": shape_metrics,
        "tail_predictions": tail_predictions,
        "decision_table": decision_table,
        "run_summary": result,
    }


if __name__ == "__main__":
    run_feasibility()
