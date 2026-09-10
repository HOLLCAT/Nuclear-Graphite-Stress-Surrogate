"""One-time final evaluation of the frozen 149-case symbolic models.

The primary model is the frozen ``mean_calibrated_consensus``.  The
``full_consensus_tail`` is evaluated at the same time as a predeclared
secondary research model, never as a replacement selected from final-test
performance.  No fitting, gain selection, threshold tuning, or model
promotion is permitted in this module.

All metrics use every element in the 50 sealed FEM cases.  Fixed uniform
samples are retained only for publication plots.  An interrupted run can
resume exact-signature case caches; a completed run refuses re-execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import hashlib
import json
import os
import shutil
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ct3_common import (
    EXPECTED_ELEMENTS_PER_CASE,
    PHYSICAL_FEATURES,
    PROVISIONAL_PHYSICAL_BOUNDS,
    RANDOM_SEED,
    TARGET_COL,
    aggregate_case_metrics,
    build_paths,
    case_path_lookup,
    discover_case_files,
    evaluate_prediction_arrays,
    final_test_case_ids,
    load_frozen_manifest,
    read_complete_case,
)
from final_development_integration import (
    DevelopmentIntegrationConfig,
    _load_prior_mean_model,
    _load_rotation_states,
)
from signed_staged_residual_symbolic import (
    STAGE_A_FEATURES,
    STAGE_B_FEATURES,
    _feature_indices,
    _load_frozen_baseline,
    _predict_baseline_components,
)
from v11_tail_aware_localised_symbolic import _mean_correction


OUTPUT_NAME = "19_one_time_final_50case_locked_149"
PRIMARY_MODEL = "mean_calibrated_consensus"
SECONDARY_MODEL = "full_consensus_tail"
PRIMARY_ROLE = "predeclared_primary"
SECONDARY_ROLE = "predeclared_secondary_research_only"
SUMMARY_FEATURES = [
    "fluence_rate",
    "temperature",
    "weight_loss_rate",
    "rho",
    "theta",
    "z",
    "theta_sin",
    "theta_cos",
]
SUMMARY_STATS = ["mean", "std", "min", "p95", "max"]
BOOTSTRAP_METRICS = [
    "mae",
    "rmse",
    "r2",
    "top5_actual_rmse",
    "top5_actual_bias",
    "p95_relative_error",
    "p95_underprediction_fraction",
    "p99_relative_error",
    "p99_underprediction_fraction",
    "top5pct_hotspot_overlap",
    "top1pct_hotspot_overlap",
    "top1_recall_in_predicted_top5",
]


@dataclass(frozen=True)
class LockedFinal149Config:
    output_subdir: str = "formal_final_50case"
    bootstrap_resamples: int = 10_000
    plot_sample_rows_per_case: int = 5_000
    random_seed: int = RANDOM_SEED
    primary_macro_rmse_max: float = 2.0
    primary_macro_r2_min: float = 0.45
    primary_worst_case_rmse_max: float = 3.5
    primary_p95_relative_error_max: float = 0.10
    primary_p99_relative_error_max: float = 0.25
    primary_p99_underprediction_max: float = 0.25
    primary_top1_hotspot_overlap_min: float = 0.45
    primary_top1_recall_in_predicted_top5_min: float = 0.80
    primary_prediction_abs_max_ratio_max: float = 1.50

    def validate(self) -> None:
        if self.bootstrap_resamples < 1_000:
            raise ValueError("Use at least 1,000 similarity-group bootstrap resamples")
        if not 500 <= self.plot_sample_rows_per_case <= 20_000:
            raise ValueError("Plot sample rows per case must be between 500 and 20,000")
        if self.primary_macro_rmse_max <= 0 or self.primary_worst_case_rmse_max <= 0:
            raise ValueError("RMSE reporting thresholds must be positive")
        for value in (
            self.primary_p95_relative_error_max,
            self.primary_p99_relative_error_max,
            self.primary_p99_underprediction_max,
            self.primary_top1_hotspot_overlap_min,
            self.primary_top1_recall_in_predicted_top5_min,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Relative-error and hotspot thresholds must be inside [0, 1]")


def output_directory(package_root: Path, config: LockedFinal149Config) -> Path:
    return Path(package_root) / "outputs" / OUTPUT_NAME / config.output_subdir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _required_paths(package_root: Path) -> dict[str, Path]:
    integration = (
        package_root
        / "outputs/18_149case_development_integration/formal_149case_integration"
    )
    return {
        "split_manifest": package_root / "shared/frozen_case_split_manifest_199cases.csv",
        "approved_protocol": package_root / "shared/locked_149_final_test_protocol.json",
        "development_summary": package_root
        / "outputs/00_qc_sensitivity_ablation/development_case_summary.csv",
        "integration_complete": integration / "development_integration_complete.json",
        "model_decision": integration / "final_model_decision.json",
        "release_gate": integration / "final_test_release_gate.json",
        "locked_configuration": integration / "locked_configuration.json",
        "locked_manifest": integration / "locked_model_manifest.json",
        "locked_formula_csv": integration / "locked_model_formula.csv",
        "locked_formula_txt": integration / "locked_model_formula.txt",
        "mean_model": integration / "mean_calibration_model_149.json",
        "tail_activity": integration / "tail_activity_audit.csv",
        "development_metrics": integration / "development_candidate_summary.csv",
        "evaluator_source": package_root / "src/final_locked_149_evaluation.py",
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight_locked_final_149(
    package_root: Path,
    config: LockedFinal149Config | None = None,
) -> dict:
    """Validate the frozen contract without reading final predictor/target values."""

    config = config or LockedFinal149Config()
    config.validate()
    package_root = Path(package_root).resolve()
    paths = build_paths(package_root)
    _, inventory = discover_case_files(paths.case_dir)
    path_by_case = case_path_lookup(inventory)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    final_ids = final_test_case_ids(manifest)
    development_ids = sorted(
        manifest.loc[~manifest["split"].eq("final_test"), "case_id"].unique()
    )
    iteration_one = manifest[manifest["iteration"].eq(1)].copy()
    final_manifest = iteration_one[iteration_one["split"].eq("final_test")].copy()

    required = _required_paths(package_root)
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Locked-final prerequisites are missing:\n" + "\n".join(missing))

    protocol = _load_json(required["approved_protocol"])
    completion = _load_json(required["integration_complete"])
    decision = _load_json(required["model_decision"])
    release_gate = _load_json(required["release_gate"])
    locked_manifest = _load_json(required["locked_manifest"])
    tail_activity = pd.read_csv(required["tail_activity"])
    development_metrics = pd.read_csv(required["development_metrics"])

    checks = pd.DataFrame([
        {"check": "case_files", "value": len(inventory), "expected": 199},
        {"check": "development_cases", "value": len(development_ids), "expected": 149},
        {"check": "final_cases", "value": len(final_ids), "expected": 50},
        {
            "check": "final_manifest_matches",
            "value": set(final_manifest["case_id"]) == set(final_ids),
            "expected": True,
        },
        {"check": "integration_complete", "value": completion.get("status"), "expected": "complete"},
        {
            "check": "integration_final_values_unread",
            "value": int(completion.get("final_test_cases_read", -1)),
            "expected": 0,
        },
        {"check": "locked_primary", "value": decision.get("selected_model"), "expected": PRIMARY_MODEL},
        {
            "check": "locked_manifest_primary",
            "value": locked_manifest.get("model"),
            "expected": PRIMARY_MODEL,
        },
        {
            "check": "locked_manifest_final_values_unread",
            "value": int(locked_manifest.get("final_test_cases_read", -1)),
            "expected": 0,
        },
        {
            "check": "technical_release_ready",
            "value": release_gate.get("technical_status"),
            "expected": "ready_for_human_review_before_one_time_final_test",
        },
        {"check": "human_protocol_approved", "value": protocol.get("approved"), "expected": True},
        {"check": "protocol_primary", "value": protocol.get("primary_model"), "expected": PRIMARY_MODEL},
        {"check": "protocol_secondary", "value": protocol.get("secondary_model"), "expected": SECONDARY_MODEL},
        {
            "check": "protocol_forbids_reselection",
            "value": protocol.get("final_results_may_change_model"),
            "expected": False,
        },
        {
            "check": "protocol_locks_complete_evaluation_config",
            "value": protocol.get("locked_evaluation_config"),
            "expected": asdict(config),
        },
        {
            "check": "protocol_requires_all_final_elements",
            "value": protocol.get("all_final_elements_required"),
            "expected": True,
        },
        {
            "check": "protocol_limits_sampling_to_figures",
            "value": protocol.get("sampling_permitted_for"),
            "expected": ["publication_figures_only"],
        },
        {
            "check": "active_tail_rotations",
            "value": tail_activity.loc[
                tail_activity["included_in_tail_consensus"].astype(bool), "iteration"
            ].astype(int).tolist(),
            "expected": [1, 2, 3],
        },
    ])
    checks["pass"] = checks.apply(lambda row: row["value"] == row["expected"], axis=1)
    if not checks["pass"].all():
        failed = checks.loc[~checks["pass"], "check"].tolist()
        raise RuntimeError(f"Locked-final preflight failed: {failed}")

    baseline = _load_frozen_baseline(package_root)
    states, rotation_audit = _load_rotation_states(package_root, baseline)
    mean_model = _load_prior_mean_model(required["mean_model"])
    active_iterations = [1, 2, 3]
    tail_gain = float(locked_manifest["tail_gain"])
    if not np.isclose(tail_gain, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("The frozen secondary tail gain must remain 1.0")

    hash_paths = dict(required)
    hash_paths.update({f"baseline_{key}": value for key, value in baseline["paths"].items()})
    for state in states:
        iteration = int(state["iteration"])
        hash_paths.update(
            {f"rotation_{iteration}_{key}": value for key, value in state["paths"].items()}
        )
    hash_rows = []
    for name, path in sorted(hash_paths.items()):
        hash_rows.append({
            "artifact": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    hash_audit = pd.DataFrame(hash_rows)
    signature_payload = {
        "config": asdict(config),
        "primary_model": PRIMARY_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "primary_role": PRIMARY_ROLE,
        "secondary_role": SECONDARY_ROLE,
        "active_tail_iterations": active_iterations,
        "secondary_tail_gain": tail_gain,
        "final_case_ids": final_ids,
        "final_similarity_groups": dict(
            zip(final_manifest["case_id"], final_manifest["similarity_group"])
        ),
        "frozen_artifact_sha256": dict(zip(hash_audit["artifact"], hash_audit["sha256"])),
        "training_or_selection_permitted": False,
        "final_results_may_change_model": False,
    }
    signature_payload["final_evaluation_signature_sha256"] = _canonical_hash(signature_payload)

    output_dir = output_directory(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(checks, output_dir / "preflight_checks.csv")
    _atomic_csv(hash_audit, output_dir / "frozen_artifact_hash_audit.csv")
    signature_path = output_dir / "final_evaluation_signature.json"
    if signature_path.exists():
        existing = _load_json(signature_path)
        if existing != signature_payload:
            raise RuntimeError("Existing final signature differs from the frozen protocol")
    else:
        _atomic_json(signature_path, signature_payload)

    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inventory": inventory,
        "manifest": manifest,
        "final_manifest": final_manifest.set_index("case_id", drop=False),
        "final_ids": final_ids,
        "path_by_case": path_by_case,
        "required_paths": required,
        "protocol": protocol,
        "development_metrics": development_metrics,
        "baseline": baseline,
        "states": states,
        "rotation_audit": rotation_audit,
        "mean_model": mean_model,
        "active_iterations": active_iterations,
        "tail_gain": tail_gain,
        "checks": checks,
        "hash_audit": hash_audit,
        "signature": signature_payload,
    }


def predictor_only_case_summary(
    frame: pd.DataFrame,
    case_id: str,
    case_number: int,
) -> pd.Series:
    record: dict[str, float | int | str | bool] = {
        "case_id": case_id,
        "case_number": int(case_number),
        "n_elements": int(len(frame)),
        "predictor_only": True,
    }
    for feature in SUMMARY_FEATURES:
        values = frame[feature].to_numpy(dtype=np.float64)
        record.update({
            f"{feature}_mean": float(np.mean(values)),
            f"{feature}_std": float(np.std(values, ddof=0)),
            f"{feature}_min": float(np.min(values)),
            f"{feature}_p95": float(np.quantile(values, 0.95)),
            f"{feature}_max": float(np.max(values)),
        })
    return pd.Series(record)


def predict_frozen_models(
    frame: pd.DataFrame,
    summary: pd.Series,
    states: Sequence[dict],
    mean_model: dict,
    active_iterations: Sequence[int],
    tail_gain: float,
) -> dict[str, np.ndarray | float]:
    """Assemble both frozen predictions using predictor values only."""

    first_state = states[0]
    matrix, _, scale, _, baseline_stress = _predict_baseline_components(
        frame,
        summary,
        first_state["v10"]["baseline_functions"],
    )
    v10_predictions = []
    tail_corrections = []
    for state in states:
        geometry = state["v10"]["geometry_function"](
            matrix[:, _feature_indices(STAGE_A_FEATURES)]
        )
        physical = state["v10"]["physical_function"](
            matrix[:, _feature_indices(STAGE_B_FEATURES)]
        )
        v10 = baseline_stress + scale * (geometry + physical)
        raw_tail = state["tail_function"](matrix)
        tail = scale * (raw_tail - float(np.mean(raw_tail)))
        if not np.isfinite(v10).all() or not np.isfinite(tail).all():
            raise FloatingPointError(
                f"Iteration {state['iteration']} produced non-finite final prediction components"
            )
        v10_predictions.append(np.asarray(v10, dtype=np.float64))
        tail_corrections.append(np.asarray(tail, dtype=np.float64))

    base_consensus = np.vstack(v10_predictions).mean(axis=0)
    delta_mean = float(_mean_correction(summary, mean_model))
    primary = base_consensus + delta_mean
    active_indices = [int(iteration) - 1 for iteration in active_iterations]
    tail_consensus = np.vstack(tail_corrections)[active_indices].mean(axis=0)
    secondary = primary + float(tail_gain) * tail_consensus
    for name, values in {PRIMARY_MODEL: primary, SECONDARY_MODEL: secondary}.items():
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise FloatingPointError(f"{name} produced invalid final predictions")
    return {
        PRIMARY_MODEL: primary,
        SECONDARY_MODEL: secondary,
        "case_mean_correction": delta_mean,
        "tail_consensus_std": float(np.std(tail_consensus, ddof=0)),
    }


def _case_cache_paths(output_dir: Path, case_id: str) -> tuple[Path, Path, Path]:
    cache_dir = output_dir / "case_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{case_id}_metrics.csv",
        cache_dir / f"{case_id}_predictors.csv",
        cache_dir / f"{case_id}_plot_sample.npz",
    )


def _case_cache_valid(
    metrics_path: Path,
    predictors_path: Path,
    sample_path: Path,
    case_id: str,
    signature: str,
) -> bool:
    if not metrics_path.exists() or not predictors_path.exists() or not sample_path.exists():
        return False
    try:
        metrics = pd.read_csv(metrics_path)
        predictors = pd.read_csv(predictors_path)
        with np.load(sample_path, allow_pickle=False) as sample:
            sample_case = str(sample["case_id"].item())
            sample_signature = str(sample["signature"].item())
    except Exception:
        return False
    return (
        len(metrics) == 2
        and set(metrics["model"]) == {PRIMARY_MODEL, SECONDARY_MODEL}
        and len(predictors) == 1
        and str(metrics.iloc[0].get("case_id")) == case_id
        and str(predictors.iloc[0].get("case_id")) == case_id
        and metrics["final_evaluation_signature_sha256"].astype(str).eq(signature).all()
        and str(predictors.iloc[0].get("final_evaluation_signature_sha256")) == signature
        and sample_case == case_id
        and sample_signature == signature
    )


def _fixed_plot_indices(n_rows: int, case_number: int, config: LockedFinal149Config) -> np.ndarray:
    count = min(config.plot_sample_rows_per_case, n_rows)
    rng = np.random.default_rng(config.random_seed + 10_000 * int(case_number))
    return np.sort(rng.choice(n_rows, size=count, replace=False))


def _evaluate_final_case(
    preflight: dict,
    case_id: str,
    config: LockedFinal149Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = preflight["output_dir"]
    metrics_path, predictors_path, sample_path = _case_cache_paths(output_dir, case_id)
    signature = preflight["signature"]["final_evaluation_signature_sha256"]
    if _case_cache_valid(
        metrics_path, predictors_path, sample_path, case_id, signature
    ):
        return pd.read_csv(metrics_path), pd.read_csv(predictors_path)

    frame, audit = read_complete_case(preflight["path_by_case"][case_id])
    summary = predictor_only_case_summary(frame, case_id, audit["case_number"])

    # Assemble both frozen predictions before accessing the FEM stress target.
    predictions = predict_frozen_models(
        frame,
        summary,
        preflight["states"],
        preflight["mean_model"],
        preflight["active_iterations"],
        preflight["tail_gain"],
    )
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    manifest_row = preflight["final_manifest"].loc[case_id]
    rows = []
    for model, role in [
        (PRIMARY_MODEL, PRIMARY_ROLE),
        (SECONDARY_MODEL, SECONDARY_ROLE),
    ]:
        rows.append({
            "evaluation_scope": "locked_final_50",
            "model": model,
            "model_role": role,
            "case_id": case_id,
            "case_number": int(audit["case_number"]),
            "similarity_group": str(manifest_row["similarity_group"]),
            "case_mean_correction_mpa": float(predictions["case_mean_correction"]),
            "tail_consensus_std_mpa": float(predictions["tail_consensus_std"]),
            "tail_gain": 0.0 if model == PRIMARY_MODEL else preflight["tail_gain"],
            "final_evaluation_signature_sha256": signature,
            **evaluate_prediction_arrays(actual, np.asarray(predictions[model])),
        })
    metrics = pd.DataFrame(rows)
    predictor = pd.DataFrame([{
        **summary.to_dict(),
        "similarity_group": str(manifest_row["similarity_group"]),
        "final_evaluation_signature_sha256": signature,
    }])

    indices = _fixed_plot_indices(len(frame), audit["case_number"], config)
    x_values = (
        frame["x"].to_numpy(dtype=np.float64)
        if "x" in frame.columns
        else frame["rho"].to_numpy(dtype=np.float64) * np.cos(frame["theta"].to_numpy(dtype=np.float64))
    )
    y_values = (
        frame["y"].to_numpy(dtype=np.float64)
        if "y" in frame.columns
        else frame["rho"].to_numpy(dtype=np.float64) * np.sin(frame["theta"].to_numpy(dtype=np.float64))
    )
    _atomic_csv(metrics, metrics_path)
    _atomic_csv(predictor, predictors_path)
    _atomic_npz(
        sample_path,
        case_id=np.asarray(case_id),
        signature=np.asarray(signature),
        element_id=frame["element_id"].to_numpy()[indices],
        x=x_values[indices],
        y=y_values[indices],
        z=frame["z"].to_numpy(dtype=np.float64)[indices],
        rho=frame["rho"].to_numpy(dtype=np.float64)[indices],
        theta=frame["theta"].to_numpy(dtype=np.float64)[indices],
        actual=actual[indices],
        primary=np.asarray(predictions[PRIMARY_MODEL])[indices],
        secondary=np.asarray(predictions[SECONDARY_MODEL])[indices],
    )

    del frame, actual, predictions, x_values, y_values
    gc.collect()
    return metrics, predictor


def _predictor_coverage(
    final_summary: pd.DataFrame,
    development_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [f"{feature}_{stat}" for feature in SUMMARY_FEATURES for stat in SUMMARY_STATS]
    records = []
    flags = final_summary[["case_id", "case_number", "similarity_group"]].copy()
    flags["predictor_summary_values_outside_development_envelope"] = 0
    for column in columns:
        lower = float(development_summary[column].min())
        upper = float(development_summary[column].max())
        values = final_summary[column].to_numpy(dtype=np.float64)
        below = values < lower
        above = values > upper
        flags["predictor_summary_values_outside_development_envelope"] += below | above
        records.append({
            "predictor_summary": column,
            "development_case_min": lower,
            "development_case_max": upper,
            "final_case_min": float(values.min()),
            "final_case_max": float(values.max()),
            "n_final_cases_below_development": int(below.sum()),
            "n_final_cases_above_development": int(above.sum()),
            "n_final_cases_outside_development": int((below | above).sum()),
        })
    return pd.DataFrame(records), flags


def _physical_boundary_report(final_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in final_summary.itertuples(index=False):
        for feature in PHYSICAL_FEATURES:
            lower, upper = PROVISIONAL_PHYSICAL_BOUNDS[feature]
            observed_min = float(getattr(record, f"{feature}_min"))
            observed_max = float(getattr(record, f"{feature}_max"))
            rows.append({
                "case_id": record.case_id,
                "variable": feature,
                "provisional_lower": lower,
                "provisional_upper": upper,
                "observed_min": observed_min,
                "observed_max": observed_max,
                "below_provisional_boundary": observed_min < lower,
                "above_provisional_boundary": observed_max > upper,
                "action": "report_only_no_clipping_no_model_change",
            })
    return pd.DataFrame(rows)


def _group_bootstrap(case_metrics: pd.DataFrame, config: LockedFinal149Config) -> pd.DataFrame:
    rows = []
    for model_index, (model, block) in enumerate(case_metrics.groupby("model", sort=False)):
        groups = sorted(block["similarity_group"].astype(str).unique())
        rng = np.random.default_rng(config.random_seed + 30_000 + model_index)
        for metric in BOOTSTRAP_METRICS:
            grouped = block.groupby("similarity_group", observed=True)[metric].agg(["sum", "count"]).reindex(groups)
            sums = grouped["sum"].to_numpy(dtype=np.float64)
            counts = grouped["count"].to_numpy(dtype=np.float64)
            bootstrap = np.empty(config.bootstrap_resamples, dtype=np.float64)
            for start in range(0, config.bootstrap_resamples, 1_000):
                size = min(1_000, config.bootstrap_resamples - start)
                indices = rng.integers(0, len(groups), size=(size, len(groups)))
                bootstrap[start:start + size] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            rows.append({
                "model": model,
                "model_role": block["model_role"].iloc[0],
                "metric": metric,
                "point_estimate_case_mean": float(block[metric].mean()),
                "similarity_group_bootstrap_95ci_low": float(low),
                "similarity_group_bootstrap_95ci_high": float(high),
                "n_cases": int(block["case_id"].nunique()),
                "n_similarity_groups": len(groups),
                "bootstrap_resamples": config.bootstrap_resamples,
                "descriptive_uncertainty_only": True,
            })
    return pd.DataFrame(rows)


def _development_comparison(
    development: pd.DataFrame,
    final_summary: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "micro_mae",
        "micro_rmse",
        "micro_r2",
        "macro_mae",
        "macro_rmse",
        "macro_r2",
        "worst_case_rmse",
        "mean_top5_actual_rmse",
        "mean_top5_actual_bias",
        "mean_p95_relative_error",
        "mean_p95_underprediction_fraction",
        "mean_p99_relative_error",
        "mean_p99_underprediction_fraction",
        "mean_top5pct_hotspot_overlap",
        "mean_top1pct_hotspot_overlap",
        "mean_top1_recall_in_predicted_top5",
        "max_prediction_abs_max_ratio",
    ]
    rows = []
    for model in (PRIMARY_MODEL, SECONDARY_MODEL):
        development_row = development[development["model"].eq(model)]
        final_row = final_summary[final_summary["model"].eq(model)]
        if len(development_row) != 1 or len(final_row) != 1:
            raise ValueError(f"Could not match development/final metrics for {model}")
        development_row = development_row.iloc[0]
        final_row = final_row.iloc[0]
        for metric in metrics:
            development_value = float(development_row[metric])
            final_value = float(final_row[metric])
            rows.append({
                "model": model,
                "metric": metric,
                "development_149_fit_diagnostic": development_value,
                "locked_final_50": final_value,
                "final_minus_development": final_value - development_value,
                "relative_change_vs_development": (
                    (final_value - development_value) / abs(development_value)
                    if abs(development_value) > 1e-12
                    else np.nan
                ),
                "report_only_no_selection": True,
            })
    return pd.DataFrame(rows)


def _primary_reporting_gates(
    final_summary: pd.DataFrame,
    case_metrics: pd.DataFrame,
    config: LockedFinal149Config,
) -> pd.DataFrame:
    primary_summary = final_summary[final_summary["model"].eq(PRIMARY_MODEL)]
    primary_cases = case_metrics[case_metrics["model"].eq(PRIMARY_MODEL)]
    if len(primary_summary) != 1:
        raise ValueError("Expected exactly one primary final summary")
    row = primary_summary.iloc[0]
    gates = [
        ("complete_case_count", int(primary_cases["case_id"].nunique()), 50, "equal"),
        (
            "complete_element_count",
            int(primary_cases["n_elements"].sum()),
            50 * EXPECTED_ELEMENTS_PER_CASE,
            "equal",
        ),
        ("macro_rmse", float(row["macro_rmse"]), config.primary_macro_rmse_max, "less_equal"),
        ("macro_r2", float(row["macro_r2"]), config.primary_macro_r2_min, "greater_equal"),
        (
            "worst_case_rmse",
            float(row["worst_case_rmse"]),
            config.primary_worst_case_rmse_max,
            "less_equal",
        ),
        (
            "p95_relative_error",
            float(row["mean_p95_relative_error"]),
            config.primary_p95_relative_error_max,
            "less_equal",
        ),
        (
            "p99_relative_error",
            float(row["mean_p99_relative_error"]),
            config.primary_p99_relative_error_max,
            "less_equal",
        ),
        (
            "p99_underprediction",
            float(row["mean_p99_underprediction_fraction"]),
            config.primary_p99_underprediction_max,
            "less_equal",
        ),
        (
            "top1_hotspot_overlap",
            float(row["mean_top1pct_hotspot_overlap"]),
            config.primary_top1_hotspot_overlap_min,
            "greater_equal",
        ),
        (
            "top1_recall_in_predicted_top5",
            float(row["mean_top1_recall_in_predicted_top5"]),
            config.primary_top1_recall_in_predicted_top5_min,
            "greater_equal",
        ),
        (
            "prediction_abs_max_ratio",
            float(row["max_prediction_abs_max_ratio"]),
            config.primary_prediction_abs_max_ratio_max,
            "less_equal",
        ),
    ]
    records = []
    for gate, value, threshold, comparison in gates:
        if comparison == "equal":
            passed = value == threshold
        elif comparison == "greater_equal":
            passed = value >= threshold
        else:
            passed = value <= threshold
        records.append({
            "gate": gate,
            "model": PRIMARY_MODEL,
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "pass": bool(passed),
            "threshold_role": "predeclared_project_research_reporting_threshold_not_nuclear_safety_limit",
            "consequence": "report_result_only; never tune or reselect using final cases",
        })
    return pd.DataFrame(records)


def _load_plot_samples(output_dir: Path, final_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for case_id in final_ids:
        sample_path = _case_cache_paths(output_dir, case_id)[2]
        with np.load(sample_path, allow_pickle=False) as sample:
            n = len(sample["actual"])
            rows.append(pd.DataFrame({
                "case_id": np.repeat(case_id, n),
                "element_id": sample["element_id"],
                "x": sample["x"],
                "y": sample["y"],
                "z": sample["z"],
                "rho": sample["rho"],
                "theta": sample["theta"],
                "actual": sample["actual"],
                "primary": sample["primary"],
                "secondary": sample["secondary"],
            }))
    return pd.concat(rows, ignore_index=True)


def _style_axes(axes) -> None:
    for axis in np.atleast_1d(axes).ravel():
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)


def _save_summary_dashboard(
    summary: pd.DataFrame,
    config: LockedFinal149Config,
    figure_dir: Path,
) -> None:
    metrics = [
        ("macro_rmse", "Macro RMSE", config.primary_macro_rmse_max, "lower"),
        ("macro_r2", "Macro R2", config.primary_macro_r2_min, "higher"),
        ("worst_case_rmse", "Worst-case RMSE", config.primary_worst_case_rmse_max, "lower"),
        ("mean_p95_relative_error", "Mean P95 relative error", config.primary_p95_relative_error_max, "lower"),
        ("mean_p99_relative_error", "Mean P99 relative error", config.primary_p99_relative_error_max, "lower"),
        ("mean_p99_underprediction_fraction", "Mean P99 underprediction", config.primary_p99_underprediction_max, "lower"),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap", config.primary_top1_hotspot_overlap_min, "higher"),
        ("mean_top1_recall_in_predicted_top5", "Top-1% recall in predicted Top-5%", config.primary_top1_recall_in_predicted_top5_min, "higher"),
    ]
    order = [PRIMARY_MODEL, SECONDARY_MODEL]
    indexed = summary.set_index("model").loc[order]
    labels = ["Primary\nmean consensus", "Secondary\ntail-aware"]
    colors = ["#35618d", "#c06c3d"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for axis, (column, title, threshold, direction) in zip(axes.ravel(), metrics):
        values = indexed[column].to_numpy(dtype=float)
        axis.bar(labels, values, color=colors, width=0.64)
        axis.axhline(threshold, color="#222222", linestyle="--", linewidth=1.2)
        axis.set_title(f"{title}\n({direction} is better)")
        for position, value in enumerate(values):
            axis.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    _style_axes(axes)
    fig.suptitle("One-time 50-case final evaluation: frozen symbolic models")
    fig.tight_layout()
    fig.savefig(figure_dir / "01_final_model_metric_dashboard.png", dpi=220)
    plt.close(fig)


def _save_case_rankings(case_metrics: pd.DataFrame, figure_dir: Path) -> None:
    primary = case_metrics[case_metrics["model"].eq(PRIMARY_MODEL)].sort_values("rmse")
    order = primary["case_id"].tolist()
    pivot_rmse = case_metrics.pivot(index="case_id", columns="model", values="rmse").loc[order]
    pivot_r2 = case_metrics.pivot(index="case_id", columns="model", values="r2").loc[order]
    rank = np.arange(1, len(order) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for model, color, label in [
        (PRIMARY_MODEL, "#35618d", "Primary"),
        (SECONDARY_MODEL, "#c06c3d", "Secondary tail"),
    ]:
        axes[0].plot(rank, pivot_rmse[model], marker="o", markersize=3, linewidth=1.3, color=color, label=label)
        axes[1].plot(rank, pivot_r2[model], marker="o", markersize=3, linewidth=1.3, color=color, label=label)
    axes[0].set_title("Case RMSE ranked by primary model")
    axes[0].set_xlabel("Case rank")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("Case R2 in the same order")
    axes[1].set_xlabel("Case rank")
    axes[1].set_ylabel("R2")
    for axis in axes:
        axis.legend()
    _style_axes(axes)
    fig.tight_layout()
    fig.savefig(figure_dir / "02_final_case_rmse_r2_rankings.png", dpi=220)
    plt.close(fig)


def _save_quantile_calibration(case_metrics: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row_index, (model, role_label, color) in enumerate([
        (PRIMARY_MODEL, "Primary", "#35618d"),
        (SECONDARY_MODEL, "Secondary tail", "#c06c3d"),
    ]):
        block = case_metrics[case_metrics["model"].eq(model)]
        for axis, actual_column, predicted_column, label in [
            (axes[row_index, 0], "actual_p95", "predicted_p95", "P95 stress"),
            (axes[row_index, 1], "actual_p99", "predicted_p99", "P99 stress"),
            (axes[row_index, 2], "actual_max", "predicted_max", "Maximum stress"),
        ]:
            actual = block[actual_column].to_numpy(dtype=float)
            predicted = block[predicted_column].to_numpy(dtype=float)
            lower = float(min(actual.min(), predicted.min()))
            upper = float(max(actual.max(), predicted.max()))
            axis.scatter(actual, predicted, s=30, alpha=0.78, color=color)
            axis.plot([lower, upper], [lower, upper], "--", color="#222222")
            axis.set_xlabel(f"FEM {label}")
            axis.set_ylabel(f"Predicted {label}")
            axis.set_title(f"{role_label}: {label}")
    _style_axes(axes)
    fig.suptitle("Final stress-quantile calibration across 50 unseen cases")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_final_quantile_calibration.png", dpi=220)
    plt.close(fig)


def _save_tail_case_metrics(case_metrics: pd.DataFrame, figure_dir: Path) -> None:
    primary = case_metrics[case_metrics["model"].eq(PRIMARY_MODEL)].sort_values("p99_relative_error")
    order = primary["case_id"].tolist()
    p99 = case_metrics.pivot(index="case_id", columns="model", values="p99_relative_error").loc[order]
    overlap = case_metrics.pivot(index="case_id", columns="model", values="top1pct_hotspot_overlap").loc[order]
    rank = np.arange(1, len(order) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for model, color, label in [
        (PRIMARY_MODEL, "#35618d", "Primary"),
        (SECONDARY_MODEL, "#c06c3d", "Secondary tail"),
    ]:
        axes[0].plot(rank, p99[model], marker="o", markersize=3, linewidth=1.2, color=color, label=label)
        axes[1].plot(rank, overlap[model], marker="o", markersize=3, linewidth=1.2, color=color, label=label)
    axes[0].set_title("P99 relative error by primary-model rank")
    axes[0].set_xlabel("Case rank")
    axes[0].set_ylabel("Relative error")
    axes[1].set_title("Top-1% hotspot overlap in the same order")
    axes[1].set_xlabel("Case rank")
    axes[1].set_ylabel("Overlap")
    for axis in axes:
        axis.legend()
    _style_axes(axes)
    fig.tight_layout()
    fig.savefig(figure_dir / "04_final_tail_hotspot_case_metrics.png", dpi=220)
    plt.close(fig)


def _save_residual_diagnostics(samples: pd.DataFrame, figure_dir: Path) -> None:
    if len(samples) > 100_000:
        samples = samples.sample(100_000, random_state=RANDOM_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for row_index, (column, label) in enumerate([
        ("primary", "Primary"),
        ("secondary", "Secondary tail"),
    ]):
        actual = samples["actual"].to_numpy(dtype=float)
        predicted = samples[column].to_numpy(dtype=float)
        residual = predicted - actual
        lower = float(min(actual.min(), predicted.min()))
        upper = float(max(actual.max(), predicted.max()))
        density = axes[row_index, 0].hexbin(actual, predicted, gridsize=80, mincnt=1, cmap="viridis")
        axes[row_index, 0].plot([lower, upper], [lower, upper], "--", color="#d94841", linewidth=1.2)
        axes[row_index, 0].set_title(f"{label}: FEM vs predicted")
        axes[row_index, 0].set_xlabel("FEM maximum principal stress")
        axes[row_index, 0].set_ylabel("Predicted stress")
        fig.colorbar(density, ax=axes[row_index, 0], label="Sample count")
        axes[row_index, 1].hexbin(actual, residual, gridsize=80, mincnt=1, cmap="magma")
        axes[row_index, 1].axhline(0.0, color="#222222", linestyle="--")
        axes[row_index, 1].set_title(f"{label}: residual vs FEM stress")
        axes[row_index, 1].set_xlabel("FEM maximum principal stress")
        axes[row_index, 1].set_ylabel("Predicted - FEM")
    _style_axes(axes)
    fig.suptitle("Final residual diagnostics from fixed uniform plot samples")
    fig.tight_layout()
    fig.savefig(figure_dir / "05_final_residual_diagnostics.png", dpi=220)
    plt.close(fig)


def _save_generalisation_plot(comparison: pd.DataFrame, figure_dir: Path) -> None:
    metrics = [
        ("macro_rmse", "Macro RMSE"),
        ("macro_r2", "Macro R2"),
        ("worst_case_rmse", "Worst-case RMSE"),
        ("mean_p95_relative_error", "P95 relative error"),
        ("mean_p99_relative_error", "P99 relative error"),
        ("mean_top1pct_hotspot_overlap", "Top-1% hotspot overlap"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = ["#8da0b6", "#35618d", "#d7a27e", "#c06c3d"]
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        block = comparison[comparison["metric"].eq(metric)].set_index("model")
        values = [
            block.loc[PRIMARY_MODEL, "development_149_fit_diagnostic"],
            block.loc[PRIMARY_MODEL, "locked_final_50"],
            block.loc[SECONDARY_MODEL, "development_149_fit_diagnostic"],
            block.loc[SECONDARY_MODEL, "locked_final_50"],
        ]
        labels = ["Primary\ndev", "Primary\nfinal", "Secondary\ndev", "Secondary\nfinal"]
        axis.bar(labels, values, color=colors)
        axis.set_title(label)
    _style_axes(axes)
    fig.suptitle("Development diagnostics versus one-time final generalisation")
    fig.tight_layout()
    fig.savefig(figure_dir / "06_development_to_final_comparison.png", dpi=220)
    plt.close(fig)


def _save_spatial_maps(
    samples: pd.DataFrame,
    case_metrics: pd.DataFrame,
    figure_dir: Path,
) -> pd.DataFrame:
    primary = case_metrics[case_metrics["model"].eq(PRIMARY_MODEL)].sort_values("rmse")
    selected = {
        "best": str(primary.iloc[0]["case_id"]),
        "median": str(primary.iloc[len(primary) // 2]["case_id"]),
        "worst": str(primary.iloc[-1]["case_id"]),
    }
    records = []
    for category, case_id in selected.items():
        block = samples[samples["case_id"].eq(case_id)].copy()
        actual = block["actual"].to_numpy(dtype=float)
        primary_pred = block["primary"].to_numpy(dtype=float)
        secondary_pred = block["secondary"].to_numpy(dtype=float)
        stress_min = float(min(actual.min(), primary_pred.min(), secondary_pred.min()))
        stress_max = float(max(actual.max(), primary_pred.max(), secondary_pred.max()))
        residual_limit = float(max(
            np.max(np.abs(primary_pred - actual)),
            np.max(np.abs(secondary_pred - actual)),
            1e-8,
        ))
        actual_p99 = case_metrics.loc[
            case_metrics["case_id"].eq(case_id)
            & case_metrics["model"].eq(PRIMARY_MODEL),
            "actual_p99",
        ]
        if len(actual_p99) != 1:
            raise ValueError(f"Could not recover the full-case P99 for {case_id}")
        hotspot_threshold = float(actual_p99.iloc[0])
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
        scatter_actual = axes[0, 0].scatter(
            block["x"], block["y"], c=actual, s=5, cmap="viridis", vmin=stress_min, vmax=stress_max
        )
        axes[0, 0].set_title("FEM stress")
        axes[0, 1].scatter(
            block["x"], block["y"], c=primary_pred, s=5, cmap="viridis", vmin=stress_min, vmax=stress_max
        )
        axes[0, 1].set_title("Primary prediction")
        scatter_primary_residual = axes[0, 2].scatter(
            block["x"], block["y"], c=primary_pred - actual, s=5, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit
        )
        axes[0, 2].set_title("Primary residual")
        hotspot = (actual >= hotspot_threshold).astype(float)
        axes[1, 0].scatter(block["x"], block["y"], c=hotspot, s=5, cmap="binary", vmin=0, vmax=1)
        axes[1, 0].set_title("FEM Top-1% hotspot")
        axes[1, 1].scatter(
            block["x"], block["y"], c=secondary_pred, s=5, cmap="viridis", vmin=stress_min, vmax=stress_max
        )
        axes[1, 1].set_title("Secondary tail prediction")
        axes[1, 2].scatter(
            block["x"], block["y"], c=secondary_pred - actual, s=5, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit
        )
        axes[1, 2].set_title("Secondary residual")
        for axis in axes.ravel():
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_aspect("equal", adjustable="box")
        fig.colorbar(
            scatter_actual,
            ax=[axes[0, 0], axes[0, 1], axes[1, 1]],
            shrink=0.78,
            label="Maximum principal stress",
        )
        fig.colorbar(
            scatter_primary_residual,
            ax=[axes[0, 2], axes[1, 2]],
            shrink=0.78,
            label="Prediction - FEM",
        )
        fig.suptitle(
            f"{category.title()} primary-RMSE final case: {case_id}\n"
            "X-Y projection of fixed plot sample; metrics and P99 use all elements"
        )
        fig.subplots_adjust(left=0.07, right=0.91, bottom=0.07, top=0.90, wspace=0.22, hspace=0.20)
        filename = f"07_{category}_case_spatial_stress_and_residual.png"
        fig.savefig(figure_dir / filename, dpi=220)
        plt.close(fig)
        records.append({
            "category": category,
            "case_id": case_id,
            "primary_rmse": float(primary.set_index("case_id").loc[case_id, "rmse"]),
            "figure": filename,
            "plot_rows": len(block),
        })
    return pd.DataFrame(records)


def _figure_manifest(spatial: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("01_final_model_metric_dashboard.png", "Primary/secondary final metric dashboard with preregistered project thresholds", "all final elements", "Results overview"),
        ("02_final_case_rmse_r2_rankings.png", "Case-level stability and difficult-case ranking", "all final elements", "Generalisation stability"),
        ("03_final_quantile_calibration.png", "P95, P99 and maximum-stress calibration", "all final elements", "High-stress accuracy"),
        ("04_final_tail_hotspot_case_metrics.png", "Per-case P99 error and hotspot overlap", "all final elements", "Tail and hotspot analysis"),
        ("05_final_residual_diagnostics.png", "Global prediction and residual density", "fixed uniform plot sample only", "Residual diagnostics"),
        ("06_development_to_final_comparison.png", "Development-to-final generalisation shift", "all development/final metrics", "Generalisation discussion"),
    ]
    for record in spatial.itertuples(index=False):
        rows.append((
            record.figure,
            f"Spatial FEM/prediction/residual comparison for the {record.category} primary-RMSE case",
            "fixed uniform plot sample; case metrics use all elements",
            "Spatial case study",
        ))
    return pd.DataFrame(rows, columns=["figure", "purpose", "data_basis", "suggested_thesis_section"])


def run_locked_final_149(
    package_root: Path,
    config: LockedFinal149Config | None = None,
    preflight: dict | None = None,
) -> dict:
    """Run or resume the one-time final evaluation under its frozen signature."""

    config = config or LockedFinal149Config()
    config.validate()
    if os.environ.get("CT3_AUTHORISE_LOCKED_FINAL_TEST") != "YES":
        raise PermissionError(
            "The sealed final set remains closed. Set "
            "CT3_AUTHORISE_LOCKED_FINAL_TEST=YES only for the approved one-time run."
        )
    preflight = preflight or preflight_locked_final_149(package_root, config)
    output_dir = preflight["output_dir"]
    figure_dir = output_dir / "figures_for_thesis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    signature = preflight["signature"]["final_evaluation_signature_sha256"]
    started_path = output_dir / "final_evaluation_started.json"
    complete_path = output_dir / "final_evaluation_complete.json"
    if complete_path.exists():
        raise RuntimeError("The one-time final evaluation is complete and must not be rerun")

    started_payload = {
        "status": "started_or_resuming_exact_signature",
        "final_evaluation_signature_sha256": signature,
        "primary_model": PRIMARY_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "primary_role": PRIMARY_ROLE,
        "secondary_role": SECONDARY_ROLE,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "training_or_selection_performed": False,
        "final_results_may_change_model": False,
    }
    if started_path.exists():
        existing = _load_json(started_path)
        if existing.get("final_evaluation_signature_sha256") != signature:
            raise RuntimeError("Partial final run has a different signature and cannot resume")
    else:
        _atomic_json(started_path, started_payload)
        _atomic_json(
            output_dir / "final_test_opening_authorisation.json",
            {
                "approved": True,
                "approval_source": str(preflight["required_paths"]["approved_protocol"]),
                "signature": signature,
                "final_case_count": 50,
                "models_frozen_before_opening": [PRIMARY_MODEL, SECONDARY_MODEL],
                "model_reselection_after_opening_permitted": False,
            },
        )

    started = time.perf_counter()
    metric_parts = []
    predictor_parts = []
    for position, case_id in enumerate(preflight["final_ids"], start=1):
        metrics_path, predictors_path, sample_path = _case_cache_paths(output_dir, case_id)
        reused = _case_cache_valid(metrics_path, predictors_path, sample_path, case_id, signature)
        print(
            f"[{position}/50] LOCKED FINAL 149: {case_id} "
            f"({'reuse signed cache' if reused else 'evaluate all elements'})",
            flush=True,
        )
        metrics, predictor = _evaluate_final_case(preflight, case_id, config)
        metric_parts.append(metrics)
        predictor_parts.append(predictor)
        _atomic_json(
            output_dir / "run_status.json",
            {
                "status": "running",
                "completed_cases": position,
                "total_final_cases": 50,
                "last_completed_case": case_id,
                "signature": signature,
                "elapsed_seconds": time.perf_counter() - started,
                "model_selection_performed": False,
            },
        )

    case_metrics = pd.concat(metric_parts, ignore_index=True).sort_values(["case_number", "model"])
    predictor_summary = pd.concat(predictor_parts, ignore_index=True).sort_values("case_number")
    if case_metrics["case_id"].nunique() != 50 or len(case_metrics) != 100:
        raise AssertionError("The two-model 50-case final metrics are incomplete")
    if predictor_summary["case_id"].nunique() != 50:
        raise AssertionError("Final predictor summaries are incomplete")
    expected_elements = 50 * EXPECTED_ELEMENTS_PER_CASE
    for model, block in case_metrics.groupby("model"):
        if int(block["n_elements"].sum()) != expected_elements:
            raise AssertionError(f"{model} did not evaluate every final element")

    scope_metrics = aggregate_case_metrics(
        case_metrics,
        ["evaluation_scope", "model", "model_role"],
    )
    development_summary = pd.read_csv(preflight["required_paths"]["development_summary"])
    coverage, case_coverage = _predictor_coverage(predictor_summary, development_summary)
    boundaries = _physical_boundary_report(predictor_summary)
    bootstrap = _group_bootstrap(case_metrics, config)
    generalisation = _development_comparison(preflight["development_metrics"], scope_metrics)
    gates = _primary_reporting_gates(scope_metrics, case_metrics, config)

    case_metrics.to_csv(output_dir / "final_50case_case_metrics.csv.gz", index=False)
    scope_metrics.to_csv(output_dir / "final_50case_scope_metrics.csv", index=False)
    predictor_summary.to_csv(output_dir / "final_50case_predictor_summaries.csv", index=False)
    coverage.to_csv(output_dir / "final_predictor_coverage_against_development.csv", index=False)
    case_coverage.to_csv(output_dir / "final_case_predictor_coverage_flags.csv", index=False)
    boundaries.to_csv(output_dir / "final_provisional_physical_boundary_report.csv", index=False)
    bootstrap.to_csv(output_dir / "final_similarity_group_bootstrap_ci.csv", index=False)
    generalisation.to_csv(output_dir / "development_to_final_generalisation.csv", index=False)
    gates.to_csv(output_dir / "final_primary_research_reporting_gates.csv", index=False)
    case_metrics[case_metrics["model"].eq(PRIMARY_MODEL)].sort_values("rmse", ascending=False).head(15).to_csv(
        output_dir / "final_worst_15_primary_cases_by_rmse.csv", index=False
    )

    samples = _load_plot_samples(output_dir, preflight["final_ids"])
    samples.to_csv(output_dir / "final_fixed_uniform_plot_sample.csv.gz", index=False)
    _save_summary_dashboard(scope_metrics, config, figure_dir)
    _save_case_rankings(case_metrics, figure_dir)
    _save_quantile_calibration(case_metrics, figure_dir)
    _save_tail_case_metrics(case_metrics, figure_dir)
    _save_residual_diagnostics(samples, figure_dir)
    _save_generalisation_plot(generalisation, figure_dir)
    spatial = _save_spatial_maps(samples, case_metrics, figure_dir)
    spatial.to_csv(output_dir / "final_spatial_case_selection.csv", index=False)
    figure_manifest = _figure_manifest(spatial)
    figure_manifest.to_csv(output_dir / "figure_manifest_for_thesis.csv", index=False)

    shutil.copy2(
        preflight["required_paths"]["locked_formula_csv"],
        output_dir / "frozen_symbolic_formulas_evaluated.csv",
    )
    shutil.copy2(
        preflight["required_paths"]["locked_formula_txt"],
        output_dir / "frozen_symbolic_formulas_evaluated.txt",
    )
    shutil.copy2(
        preflight["required_paths"]["approved_protocol"],
        output_dir / "approved_final_test_protocol.json",
    )

    primary_gates_pass = bool(gates["pass"].all())
    result = {
        "status": "complete",
        "primary_model": PRIMARY_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "primary_role": PRIMARY_ROLE,
        "secondary_role": SECONDARY_ROLE,
        "final_test_cases_read": 50,
        "final_test_elements_evaluated_per_model": expected_elements,
        "models_evaluated": 2,
        "final_evaluation_signature_sha256": signature,
        "training_performed": False,
        "hyperparameter_selection_performed": False,
        "formula_selection_performed": False,
        "model_promotion_performed": False,
        "final_results_may_change_model": False,
        "primary_research_reporting_gates_passed": int(gates["pass"].sum()),
        "primary_research_reporting_gates_total": len(gates),
        "primary_passes_all_predeclared_project_thresholds": primary_gates_pass,
        "thresholds_are_nuclear_safety_limits": False,
        "figure_count": len(figure_manifest),
        "elapsed_seconds": time.perf_counter() - started,
        "output_directory": str(output_dir),
    }
    _atomic_json(complete_path, result)
    return result
