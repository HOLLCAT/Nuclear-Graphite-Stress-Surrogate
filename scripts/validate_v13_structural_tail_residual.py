#!/usr/bin/env python3
"""Fast structural validation for the V13 development-only package."""

from pathlib import Path
import ast
import json
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_ROOT / "src" / "v13_structural_tail_residual.py"
WORKER = PACKAGE_ROOT / "scripts" / "run_v13_structural_residual_segment.py"
BUILDER = PACKAGE_ROOT / "scripts" / "build_v13_structural_tail_residual_notebook.py"
NOTEBOOK = (
    PACKAGE_ROOT
    / "notebooks"
    / "15_V13_Structural_Tail_Residual"
    / "01_V13_Structural_Tail_Residual_Iteration_1.ipynb"
)
SLURM = PACKAGE_ROOT / "cluster" / "run_v13_structural_tail_residual.slurm"

if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ct3_common import build_paths, discover_case_files, load_frozen_manifest
from v13_structural_tail_residual import (
    V13_FEATURES,
    V13_TIER_LOSS_MASS,
    V13_TIER_QUOTAS,
    V13Config,
)


def main() -> None:
    print("[1/5] Checking Python and notebook syntax...", flush=True)
    for path in [SOURCE, WORKER, BUILDER]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parsed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4:
        raise ValueError("V13 notebook is not nbformat 4")
    for index, cell in enumerate(parsed.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")
    if not SLURM.exists():
        raise FileNotFoundError(SLURM)

    print("[2/5] Checking the 199-case frozen manifest...", flush=True)
    paths = build_paths(PACKAGE_ROOT)
    _, inventory = discover_case_files(paths.case_dir)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    iteration = manifest[manifest["iteration"].eq(1)].copy()
    counts = iteration["split"].value_counts().to_dict()
    expected = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    if counts != expected:
        raise ValueError(f"Unexpected Iteration-1 split counts: {counts}")

    print("[3/5] Checking frozen V11/V12 prerequisites...", flush=True)
    required = [
        PACKAGE_ROOT / "outputs" / "11_tail_aware_localised_symbolic" / "iteration_1" / "pilot_complete.json",
        PACKAGE_ROOT / "outputs" / "12_v11_gain_stability_audit" / "iteration_1" / "audit_complete.json",
        PACKAGE_ROOT / "outputs" / "12_v11_gain_stability_audit" / "iteration_1" / "selected_gain_formula.csv",
        PACKAGE_ROOT / "outputs" / "13_v12_case_adaptive_tail_gain" / "iteration_1" / "v12_complete.json",
        PACKAGE_ROOT / "outputs" / "13_v12_case_adaptive_tail_gain" / "iteration_1" / "v12_promotion_decision.json",
        PACKAGE_ROOT / "outputs" / "00_qc_sensitivity_ablation" / "development_case_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V13 prerequisites:\n" + "\n".join(missing))
    for path in [required[0], required[1], required[3]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("final_test_cases_read") != 0:
            raise RuntimeError(f"Incomplete parent or broken final seal: {path}")
    decision = json.loads(required[4].read_text(encoding="utf-8"))
    if decision.get("promotion_status") != "retain_v11_global_gain_0_90":
        raise RuntimeError("V13 expects V12 to retain V11 gain 0.90")
    formula = pd.read_csv(required[2])
    if len(formula) != 1 or float(formula.iloc[0]["selected_tail_gain"]) != 0.9:
        raise ValueError("Frozen V11 gain formula is not the reviewed 0.90 formula")
    summary = pd.read_csv(required[5])
    if len(summary) != 149 or summary["case_id"].nunique() != 149:
        raise ValueError("Expected 149 complete development summaries")

    print("[4/5] Checking V13 feature and search configuration...", flush=True)
    config = V13Config()
    config.validate()
    if len(V13_FEATURES) != 42 or len(set(V13_FEATURES)) != 42:
        raise AssertionError("V13 predictor feature registry drifted")
    if sum(V13_TIER_QUOTAS) != 5_000 or abs(sum(V13_TIER_LOSS_MASS) - 1.0) > 1e-12:
        raise AssertionError("V13 tier sampling or loss mass is invalid")
    forbidden = [name for name in V13_FEATURES if name.startswith("stress_")]
    if forbidden:
        raise AssertionError(f"Stress leakage in V13 deployable features: {forbidden}")

    print("[5/5] Checking final-test seal and planned work...", flush=True)
    final_ids = iteration.loc[iteration["split"].eq("final_test"), "case_id"]
    development_ids = iteration.loc[~iteration["split"].eq("final_test"), "case_id"]
    if len(final_ids) != 50 or len(development_ids) != 149:
        raise AssertionError("Frozen development/final partition is invalid")
    print("Case files:", len(inventory))
    print("Iteration-1 split:", counts)
    print("V13 features:", len(V13_FEATURES))
    print("Discovery rows:", 119 * config.rows_per_training_case)
    print("Recoverable segments:", config.n_segments)
    print("Population iterations:", config.population_iterations_total)
    print("Fast V13 validation passed. Full SHA-256 provenance runs inside the notebook.")


if __name__ == "__main__":
    main()
