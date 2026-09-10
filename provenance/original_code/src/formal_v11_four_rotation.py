"""Formal four-rotation orchestration for the frozen V11 method."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import time

import pandas as pd

from signed_staged_residual_symbolic import (
    SignedStagedPilotConfig,
    run_signed_staged_residual_pilot,
)
from v11_tail_aware_localised_symbolic import V11PilotConfig, run_v11_pilot
from v11_gain_stability_audit import V11GainAuditConfig, run_v11_gain_audit


FORMAL_OUTPUT_NAME = "17_v11_four_rotation_formal"
FORMAL_ITERATIONS = (1, 2, 3, 4)


def output_directory(package_root: Path, iteration: int | None = None) -> Path:
    root = Path(package_root) / "outputs" / FORMAL_OUTPUT_NAME
    return root if iteration is None else root / f"iteration_{iteration}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_complete(record: dict, label: str, iteration: int) -> None:
    if record.get("status") != "complete":
        raise RuntimeError(f"{label} Iteration {iteration} is not complete")
    if int(record.get("iteration", -1)) != iteration:
        raise RuntimeError(f"{label} completion belongs to another iteration")
    if int(record.get("final_test_cases_read", -1)) != 0:
        raise RuntimeError(f"{label} accessed sealed final-test element files")


def register_completed_iteration_one(package_root: Path) -> dict:
    """Register the already completed Iteration-1 chain without rerunning PySR."""
    package_root = Path(package_root).resolve()
    out = output_directory(package_root, 1)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "v10": package_root
        / "outputs/10_signed_staged_residual_symbolic_pilot/iteration_1/pilot_complete.json",
        "v11": package_root
        / "outputs/11_tail_aware_localised_symbolic/iteration_1/pilot_complete.json",
        "gain_audit": package_root
        / "outputs/12_v11_gain_stability_audit/iteration_1/audit_complete.json",
        "selected_gain": package_root
        / "outputs/12_v11_gain_stability_audit/iteration_1/selected_gain.json",
        "formula": package_root
        / "outputs/12_v11_gain_stability_audit/iteration_1/selected_gain_formula.csv",
        "v13_decision": package_root
        / "outputs/16_v13_residual_gain_audit/iteration_1/residual_gain_decision.json",
    }
    records = {name: _load_json(path) for name, path in paths.items() if name != "formula"}
    _assert_complete(records["v10"], "V10", 1)
    _assert_complete(records["v11"], "V11", 1)
    _assert_complete(records["gain_audit"], "V11 gain audit", 1)
    if int(records["v13_decision"].get("final_test_cases_read", -1)) != 0:
        raise RuntimeError("V13 decision accessed sealed final-test elements")
    if records["v13_decision"].get("primary_model_decision") != (
        "freeze_v11_global_gain_0_90_as_primary_model"
    ):
        raise RuntimeError("V13 decision does not preserve V11 gain 0.90")
    selected_gain = float(records["selected_gain"]["selected_gain"])
    if abs(selected_gain - 0.90) > 1e-12:
        raise RuntimeError("Iteration-1 selected gain is not the reviewed value 0.90")

    evidence = pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        ]
    )
    evidence.to_csv(out / "registered_artifact_hashes.csv", index=False)
    completion = {
        "status": "complete",
        "iteration": 1,
        "execution_mode": "registered_existing_reviewed_result_no_rerun",
        "method": "v10_signed_staged_plus_v11_tail_plus_validation_gain",
        "selected_gain": selected_gain,
        "v10_population_iterations": records["v10"]["planned_population_iterations"],
        "v11_population_iterations": records["v11"]["planned_population_iterations"],
        "train_cases": records["v11"]["train_cases"],
        "validation_cases": records["v11"]["validation_cases"],
        "internal_test_cases": records["v11"]["internal_test_cases"],
        "final_test_cases_read": 0,
        "v13_role": "non_deployed_tail_correction_research",
        "formula_path": str(paths["formula"]),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(out / "pipeline_complete.json", completion)
    return completion


def _rotation_configs(iteration: int) -> tuple[
    SignedStagedPilotConfig, V11PilotConfig, V11GainAuditConfig
]:
    if iteration not in {2, 3, 4}:
        raise ValueError("Fresh formal runs are limited to Iterations 2, 3 and 4")
    output_subdir = f"iteration_{iteration}"
    return (
        SignedStagedPilotConfig(iteration=iteration, output_subdir=output_subdir),
        V11PilotConfig(iteration=iteration, output_subdir=output_subdir),
        V11GainAuditConfig(iteration=iteration, output_subdir=output_subdir),
    )


def run_formal_rotation(package_root: Path, iteration: int) -> dict:
    """Run or resume V10, V11 and gain audit for one frozen rotation."""
    package_root = Path(package_root).resolve()
    out = output_directory(package_root, iteration)
    out.mkdir(parents=True, exist_ok=True)
    completion_path = out / "pipeline_complete.json"
    if completion_path.exists():
        completion = _load_json(completion_path)
        _assert_complete(completion, "Formal pipeline", iteration)
        return completion

    v10_config, v11_config, gain_config = _rotation_configs(iteration)
    _atomic_json(
        out / "locked_configuration.json",
        {
            "iteration": iteration,
            "v10": asdict(v10_config),
            "v11": asdict(v11_config),
            "gain_audit": asdict(gain_config),
            "final_test_policy": "inventory_only_no_element_reads",
        },
    )
    started = time.time()
    _atomic_json(
        out / "run_status.json",
        {
            "status": "running",
            "iteration": iteration,
            "stage": "v10_signed_staged_search",
            "final_test_cases_read": 0,
        },
    )
    try:
        v10 = run_signed_staged_residual_pilot(package_root, v10_config)
        _assert_complete(v10, "V10", iteration)
        _atomic_json(
            out / "run_status.json",
            {
                "status": "running",
                "iteration": iteration,
                "stage": "v11_tail_localised_search",
                "elapsed_seconds": time.time() - started,
                "final_test_cases_read": 0,
            },
        )
        v11 = run_v11_pilot(package_root, v11_config)
        _assert_complete(v11, "V11", iteration)
        _atomic_json(
            out / "run_status.json",
            {
                "status": "running",
                "iteration": iteration,
                "stage": "validation_only_gain_audit",
                "elapsed_seconds": time.time() - started,
                "final_test_cases_read": 0,
            },
        )
        gain = run_v11_gain_audit(package_root, gain_config)
        _assert_complete(gain, "V11 gain audit", iteration)
        completion = {
            "status": "complete",
            "iteration": iteration,
            "execution_mode": "fresh_rotation_specific_training_and_audit",
            "method": "v10_signed_staged_plus_v11_tail_plus_validation_gain",
            "selected_gain": gain["selected_gain"],
            "v10_population_iterations": v10["planned_population_iterations"],
            "v11_population_iterations": v11["planned_population_iterations"],
            "train_cases": v11["train_cases"],
            "validation_cases": v11["validation_cases"],
            "internal_test_cases": v11["internal_test_cases"],
            "final_test_cases_read": 0,
            "elapsed_seconds": time.time() - started,
            "v10_output_directory": v10["output_directory"],
            "v11_output_directory": v11["output_directory"],
            "gain_output_directory": gain["output_directory"],
        }
        _atomic_json(completion_path, completion)
        _atomic_json(out / "run_status.json", completion)
        return completion
    except Exception as exc:
        _atomic_json(
            out / "run_status.json",
            {
                "status": "failed",
                "iteration": iteration,
                "error": repr(exc),
                "elapsed_seconds": time.time() - started,
                "final_test_cases_read": 0,
            },
        )
        raise


def aggregate_completed_rotations(package_root: Path) -> dict:
    """Aggregate four development rotations without reading final-test elements."""
    package_root = Path(package_root).resolve()
    root = output_directory(package_root)
    summary_dir = root / "four_rotation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    completion_rows = []
    metric_parts = []
    formula_rows = []
    for iteration in FORMAL_ITERATIONS:
        completion = _load_json(
            output_directory(package_root, iteration) / "pipeline_complete.json"
        )
        _assert_complete(completion, "Formal pipeline", iteration)
        completion_rows.append(completion)
        audit_root = (
            package_root
            / "outputs"
            / "12_v11_gain_stability_audit"
            / f"iteration_{iteration}"
        )
        metrics = pd.read_csv(audit_root / "selected_models_split_metrics.csv")
        metric_parts.append(metrics)
        formula = pd.read_csv(audit_root / "selected_gain_formula.csv").iloc[0].to_dict()
        formula["formula_file_sha256"] = _sha256(
            audit_root / "selected_gain_formula.csv"
        )
        formula_rows.append(formula)

    completions = pd.DataFrame(completion_rows)
    metrics = pd.concat(metric_parts, ignore_index=True)
    formulas = pd.DataFrame(formula_rows)
    selected = metrics[metrics["reporting_role"] == "validation_selected"].copy()
    stability = (
        selected.groupby("split", as_index=False)
        .agg(
            rotations=("iteration", "nunique"),
            macro_rmse_mean=("macro_rmse", "mean"),
            macro_rmse_std=("macro_rmse", "std"),
            macro_r2_mean=("macro_r2", "mean"),
            p95_relative_error_mean=("mean_p95_relative_error", "mean"),
            p99_relative_error_mean=("mean_p99_relative_error", "mean"),
            top1_hotspot_overlap_mean=("mean_top1pct_hotspot_overlap", "mean"),
            top1_hotspot_overlap_min=("mean_top1pct_hotspot_overlap", "min"),
        )
    )
    completions.to_csv(summary_dir / "rotation_completion_summary.csv", index=False)
    metrics.to_csv(summary_dir / "all_rotation_split_metrics.csv", index=False)
    formulas.to_csv(summary_dir / "all_rotation_selected_formulas.csv", index=False)
    stability.to_csv(summary_dir / "selected_model_stability_summary.csv", index=False)
    result = {
        "status": "complete",
        "iterations": list(FORMAL_ITERATIONS),
        "completed_rotations": int(completions["iteration"].nunique()),
        "selected_gains": completions.sort_values("iteration")["selected_gain"].tolist(),
        "final_test_cases_read": 0,
        "interpretation": "development_rotation_stability_not_external_validation",
        "summary_directory": str(summary_dir),
    }
    _atomic_json(summary_dir / "aggregation_complete.json", result)
    return result

