"""Export best, median, and worst locked-final cases for spatial comparison.

The three cases come from the formal final-test spatial selection record:
case_80 (best RMSE), case_178 (median RMSE), and case_31 (worst RMSE).
Every element is exported.  The frozen primary model is reproduced without
training, refitting, gain selection, formula selection, or model promotion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

import final_locked_149_evaluation as locked_final_module
from ct3_common import EXPECTED_ELEMENTS_PER_CASE, TARGET_COL, read_complete_case
from final_locked_149_evaluation import (
    LockedFinal149Config,
    PRIMARY_MODEL,
    evaluate_prediction_arrays,
    predict_frozen_models,
    predictor_only_case_summary,
    preflight_locked_final_149,
)


OUTPUT_NAME = "21_three_case_fem_vs_frozen_prediction"
SELECTED_CASES = {
    "best": "case_80",
    "median": "case_178",
    "worst": "case_31",
}
METRIC_REPRODUCTION_TOLERANCE = 2e-4
REPRODUCTION_METRICS = (
    "mae",
    "rmse",
    "r2",
    "bias_predicted_minus_actual",
    "top5_actual_rmse",
    "top5_actual_bias",
    "actual_mean",
    "predicted_mean",
    "actual_p95",
    "predicted_p95",
    "p95_relative_error",
    "actual_p99",
    "predicted_p99",
    "p99_relative_error",
    "actual_max",
    "predicted_max",
    "prediction_abs_max_ratio",
    "top5pct_hotspot_overlap",
    "top1pct_hotspot_overlap",
    "top1_recall_in_predicted_top5",
)


def output_directory(package_root: Path) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / OUTPUT_NAME
        / "full_400360_element_exports"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def _atomic_csv_gzip(frame: pd.DataFrame, path: Path, compression_level: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "compresslevel": compression_level, "mtime": 0},
    )
    os.replace(temporary, path)


def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    count = max(1, int(np.ceil(len(values) * fraction)))
    selected = np.argpartition(values, len(values) - count)[len(values) - count :]
    mask = np.zeros(len(values), dtype=np.uint8)
    mask[selected] = 1
    return mask


def _load_final_evidence(package_root: Path) -> dict:
    final_root = (
        package_root
        / "outputs/19_one_time_final_50case_locked_149/formal_final_50case"
    )
    paths = {
        "completion": final_root / "final_evaluation_complete.json",
        "signature": final_root / "final_evaluation_signature.json",
        "selection": final_root / "final_spatial_case_selection.csv",
        "metrics": final_root / "final_50case_case_metrics.csv.gz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Locked-final evidence is missing:\n" + "\n".join(missing))

    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    signature = json.loads(paths["signature"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("final_test_cases_read") != 50:
        raise RuntimeError("The locked 50-case final evaluation is not complete")
    if completion.get("primary_model") != PRIMARY_MODEL:
        raise RuntimeError("The frozen primary model role has changed")
    if (
        completion.get("final_evaluation_signature_sha256")
        != signature.get("final_evaluation_signature_sha256")
    ):
        raise RuntimeError("The final completion marker and signature disagree")

    selection = pd.read_csv(paths["selection"])
    found = dict(zip(selection["category"], selection["case_id"]))
    if found != SELECTED_CASES:
        raise RuntimeError(
            f"Formal spatial selections changed: {found} != {SELECTED_CASES}"
        )
    metrics = pd.read_csv(paths["metrics"])
    primary = metrics[
        metrics["model"].eq(PRIMARY_MODEL)
        & metrics["case_id"].isin(SELECTED_CASES.values())
    ].copy()
    if len(primary) != 3:
        raise RuntimeError("Primary metrics are missing for one or more selected cases")
    return {
        "completion": completion,
        "signature": signature,
        "selection": selection,
        "primary_metrics": primary.set_index("case_id", drop=False),
    }


def _post_evaluation_preflight(
    package_root: Path,
    original_signature: dict,
) -> tuple[dict, pd.DataFrame]:
    with tempfile.TemporaryDirectory(prefix="ct3-three-case-preflight-") as temporary:
        original_output_directory = locked_final_module.output_directory
        locked_final_module.output_directory = (
            lambda _package_root, _config: Path(temporary)
        )
        try:
            preflight = preflight_locked_final_149(
                package_root,
                LockedFinal149Config(),
            )
        finally:
            locked_final_module.output_directory = original_output_directory

    current_signature = preflight["signature"]
    ignored = {"frozen_artifact_sha256", "final_evaluation_signature_sha256"}
    for key in sorted(set(original_signature) | set(current_signature)):
        if key not in ignored and original_signature.get(key) != current_signature.get(key):
            raise RuntimeError(f"Frozen final contract changed after evaluation: {key}")

    original_hashes = original_signature["frozen_artifact_sha256"]
    current_hashes = current_signature["frozen_artifact_sha256"]
    allowed_metadata_differences = {"rotation_1_formal_completion"}
    rows = []
    for artifact in sorted(set(original_hashes) | set(current_hashes)):
        original_hash = original_hashes.get(artifact)
        current_hash = current_hashes.get(artifact)
        unchanged = original_hash == current_hash
        allowed = not unchanged and artifact in allowed_metadata_differences
        rows.append(
            {
                "artifact": artifact,
                "original_final_sha256": original_hash,
                "current_sha256": current_hash,
                "unchanged": unchanged,
                "allowed_non_predictive_metadata_change": allowed,
                "status": (
                    "unchanged"
                    if unchanged
                    else "allowed_non_predictive_completion_metadata_change"
                    if allowed
                    else "forbidden_frozen_artifact_change"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    forbidden = audit[audit["status"].eq("forbidden_frozen_artifact_change")]
    if not forbidden.empty:
        raise RuntimeError(
            "Model-bearing frozen artifacts changed after final evaluation: "
            f"{forbidden['artifact'].tolist()}"
        )
    return preflight, audit


def _metric_reproduction_audit(
    case_id: str,
    category: str,
    current: dict,
    prior: pd.Series,
) -> pd.DataFrame:
    rows = []
    for metric in REPRODUCTION_METRICS:
        difference = abs(float(current[metric]) - float(prior[metric]))
        rows.append(
            {
                "category": category,
                "case_id": case_id,
                "metric": metric,
                "recomputed_value": float(current[metric]),
                "prior_locked_final_value": float(prior[metric]),
                "absolute_difference": difference,
                "tolerance": METRIC_REPRODUCTION_TOLERANCE,
                "pass": difference <= METRIC_REPRODUCTION_TOLERANCE,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["pass"].all():
        raise RuntimeError(
            f"{case_id} failed metric reproduction: "
            f"{audit.loc[~audit['pass'], 'metric'].tolist()}"
        )
    return audit


def _export_one_case(
    output_dir: Path,
    preflight: dict,
    evidence: dict,
    category: str,
    case_id: str,
) -> dict:
    case_number = int(case_id.split("_")[-1])
    source_path = Path(preflight["path_by_case"][case_id])
    frame, read_audit = read_complete_case(source_path)
    if len(frame) != EXPECTED_ELEMENTS_PER_CASE:
        raise RuntimeError(f"{case_id} does not contain 400,360 elements")
    summary = predictor_only_case_summary(frame, case_id, case_number)
    predictions = predict_frozen_models(
        frame,
        summary,
        preflight["states"],
        preflight["mean_model"],
        preflight["active_iterations"],
        preflight["tail_gain"],
    )
    actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
    predicted = np.asarray(predictions[PRIMARY_MODEL], dtype=np.float64)
    metrics = evaluate_prediction_arrays(actual, predicted)
    reproduction = _metric_reproduction_audit(
        case_id,
        category,
        metrics,
        evidence["primary_metrics"].loc[case_id],
    )

    x = frame["x"].to_numpy(dtype=np.float64)
    y = frame["y"].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(f"{case_id} lacks finite exported X/Y coordinates")
    coordinates = {
        "ElementID": frame["element_id"].to_numpy(dtype=np.int64),
        "X": x,
        "Y": y,
        "Z": frame["z"].to_numpy(dtype=np.float64),
    }
    fem = pd.DataFrame({**coordinates, "MaxPrincipalStress": actual})
    prediction = pd.DataFrame({**coordinates, "MaxPrincipalStress": predicted})
    residual = predicted - actual
    comparison = pd.DataFrame(
        {
            "CaseID": case_id,
            "CaseNumber": case_number,
            "PerformanceCategory": category,
            **coordinates,
            "Rho": frame["rho"].to_numpy(dtype=np.float64),
            "Theta": frame["theta"].to_numpy(dtype=np.float64),
            "FluenceRate": frame["fluence_rate"].to_numpy(dtype=np.float64),
            "Temperature": frame["temperature"].to_numpy(dtype=np.float64),
            "WeightLossRate": frame["weight_loss_rate"].to_numpy(dtype=np.float64),
            "MaxPrincipalStress_FEM": actual,
            "MaxPrincipalStress_Predicted": predicted,
            "Residual_PredictedMinusFEM": residual,
            "AbsoluteError": np.abs(residual),
            "FEM_Top5pct": _top_fraction_mask(actual, 0.05),
            "FEM_Top1pct": _top_fraction_mask(actual, 0.01),
            "Predicted_Top5pct": _top_fraction_mask(predicted, 0.05),
            "Predicted_Top1pct": _top_fraction_mask(predicted, 0.01),
        }
    )

    prefix = f"{case_id}_{category}"
    fem_path = output_dir / f"{prefix}_FEM_field.csv.gz"
    prediction_path = output_dir / f"{prefix}_frozen_primary_prediction_field.csv.gz"
    comparison_path = output_dir / f"{prefix}_FEM_vs_prediction_full.csv.gz"
    metrics_path = output_dir / f"{prefix}_primary_metrics.csv"
    reproduction_path = output_dir / f"{prefix}_metric_reproduction_audit.csv"
    read_audit_path = output_dir / f"{prefix}_source_read_audit.csv"

    _atomic_csv_gzip(fem, fem_path)
    _atomic_csv_gzip(prediction, prediction_path)
    _atomic_csv_gzip(comparison, comparison_path)
    metric_row = {
        "category": category,
        "case_id": case_id,
        "case_number": case_number,
        "model": PRIMARY_MODEL,
        "model_role": "predeclared_primary",
        "case_mean_correction_mpa": float(predictions["case_mean_correction"]),
        "final_evaluation_signature_sha256": evidence["signature"][
            "final_evaluation_signature_sha256"
        ],
        **metrics,
    }
    _atomic_csv(pd.DataFrame([metric_row]), metrics_path)
    _atomic_csv(reproduction, reproduction_path)
    _atomic_csv(pd.DataFrame([read_audit]), read_audit_path)

    fem_check = pd.read_csv(fem_path, compression="gzip")
    prediction_check = pd.read_csv(prediction_path, compression="gzip")
    if len(fem_check) != EXPECTED_ELEMENTS_PER_CASE:
        raise RuntimeError(f"{case_id} saved FEM field has the wrong row count")
    if not fem_check[["ElementID", "X", "Y", "Z"]].equals(
        prediction_check[["ElementID", "X", "Y", "Z"]]
    ):
        raise RuntimeError(f"{case_id} FEM and prediction field rows are not aligned")

    paths = [
        fem_path,
        prediction_path,
        comparison_path,
        metrics_path,
        reproduction_path,
        read_audit_path,
    ]
    result = {
        "category": category,
        "case_id": case_id,
        "case_number": case_number,
        "source_case_file": str(source_path),
        "source_case_sha256": _sha256(source_path),
        "maximum_metric_reproduction_difference": float(
            reproduction["absolute_difference"].max()
        ),
        "metric_row": metric_row,
        "files": [
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    del frame, predictions, actual, predicted, fem, prediction, comparison
    del fem_check, prediction_check
    gc.collect()
    return result


def run_export(package_root: Path) -> dict:
    started = time.perf_counter()
    package_root = Path(package_root).resolve()
    output_dir = output_directory(package_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "three_case_export_complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "complete":
            print("The verified three-case export already exists; no prediction was repeated.")
            return completion
        raise RuntimeError("An invalid completion marker exists; inspect before rerunning")

    print("[1/3] Loading frozen evidence and model artifacts...", flush=True)
    evidence = _load_final_evidence(package_root)
    preflight, hash_audit = _post_evaluation_preflight(
        package_root,
        evidence["signature"],
    )
    if not set(SELECTED_CASES.values()).issubset(set(preflight["final_ids"])):
        raise RuntimeError("One or more spatial cases are outside final_test")
    _atomic_csv(hash_audit, output_dir / "three_case_frozen_artifact_hash_audit.csv")
    _atomic_csv(evidence["selection"], output_dir / "three_case_selection_record.csv")

    print("[2/3] Reproducing and exporting three complete 400,360-element cases...", flush=True)
    results = []
    for index, (category, case_id) in enumerate(SELECTED_CASES.items(), start=1):
        print(f"  [{index}/3] {category}: {case_id}", flush=True)
        results.append(
            _export_one_case(output_dir, preflight, evidence, category, case_id)
        )

    summary = pd.DataFrame([result["metric_row"] for result in results])
    summary_path = output_dir / "three_case_primary_metrics_summary.csv"
    _atomic_csv(summary, summary_path)
    manifest = {
        "purpose": "best, median, and worst final-test spatial comparison for thesis and professor review",
        "selection_source": "formal final_spatial_case_selection.csv",
        "selection_rule": "best, median, and worst per-case RMSE under the frozen primary model",
        "selected_cases": SELECTED_CASES,
        "formal_model": PRIMARY_MODEL,
        "model_role": "predeclared_primary",
        "elements_per_case": EXPECTED_ELEMENTS_PER_CASE,
        "total_elements_exported": 3 * EXPECTED_ELEMENTS_PER_CASE,
        "all_elements_exported": True,
        "element_sampling_used": False,
        "coordinates": "exported post-deformation X, Y, Z",
        "training_performed": False,
        "constant_refitting_performed": False,
        "formula_selection_performed": False,
        "gain_selection_performed": False,
        "model_promotion_performed": False,
        "final_evaluation_signature_sha256": evidence["signature"][
            "final_evaluation_signature_sha256"
        ],
        "case_exports": [
            {key: value for key, value in result.items() if key != "metric_row"}
            for result in results
        ],
        "summary_file": {
            "file": summary_path.name,
            "size_bytes": summary_path.stat().st_size,
            "sha256": _sha256(summary_path),
        },
    }
    _atomic_json(output_dir / "three_case_export_manifest.json", manifest)

    readme = output_dir / "README_three_case_spatial_comparison.txt"
    readme.write_text(
        "Three-case FEM versus frozen-primary spatial comparison\n"
        "======================================================\n\n"
        "Selected cases: case_80 (best RMSE), case_178 (median RMSE), and "
        "case_31 (worst RMSE) among the 50 sealed final-test cases.\n"
        "Each case contains all 400,360 elements; no element sampling was used.\n"
        "The formal frozen primary model is mean_calibrated_consensus.\n\n"
        "For every case, the FEM and prediction field files share the same "
        "ElementID/X/Y/Z/MaxPrincipalStress schema. Use identical camera settings "
        "and a common colour scale when comparing FEM with prediction. Use a common "
        "scale across all three cases only if cross-case stress magnitude is the intended comparison.\n\n"
        "Recommended thesis use: median case in the main results; best and worst cases "
        "together in the model-capability and limitation discussion or appendix.\n",
        encoding="utf-8",
    )

    print("[3/3] Recording completion and provenance...", flush=True)
    completion = {
        "status": "complete",
        "selected_cases": SELECTED_CASES,
        "model": PRIMARY_MODEL,
        "cases_exported": 3,
        "elements_per_case": EXPECTED_ELEMENTS_PER_CASE,
        "total_elements_exported": 3 * EXPECTED_ELEMENTS_PER_CASE,
        "all_elements_exported": True,
        "training_performed": False,
        "model_selection_performed": False,
        "maximum_metric_reproduction_difference": max(
            result["maximum_metric_reproduction_difference"] for result in results
        ),
        "final_evaluation_signature_sha256": evidence["signature"][
            "final_evaluation_signature_sha256"
        ],
        "output_directory": str(output_dir),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2), flush=True)
    return completion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_export(arguments.package_root)
