#!/usr/bin/env python3
"""Fast structural validation for the formal V12 package."""

from pathlib import Path
import ast
import json
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src" / "v12_case_adaptive_tail_gain.py"
BUILDER = PACKAGE_ROOT / "scripts" / "build_v12_case_adaptive_tail_gain_notebook.py"
NOTEBOOK = PACKAGE_ROOT / "notebooks" / "13_V12_Case_Adaptive_Tail_Gain" / "01_V12_Formal_Case_Adaptive_Tail_Gain.ipynb"

if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ct3_common import build_paths, discover_case_files, load_frozen_manifest
from v12_case_adaptive_tail_gain import (
    FEATURE_SETS,
    PHYSICAL_CONTEXT_FEATURES,
    TAIL_CONTEXT_FEATURES,
    V12Config,
    _balanced_group_folds,
    candidate_configurations,
)


def main() -> None:
    print("[1/5] Checking Python and notebook syntax...", flush=True)
    ast.parse(SRC.read_text(encoding="utf-8"), filename=str(SRC))
    ast.parse(BUILDER.read_text(encoding="utf-8"), filename=str(BUILDER))
    parsed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4:
        raise ValueError("V12 notebook is not nbformat 4")
    for index, cell in enumerate(parsed.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")

    print("[2/5] Checking the 199-case frozen manifest...", flush=True)
    paths = build_paths(PACKAGE_ROOT)
    _, inventory = discover_case_files(paths.case_dir)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    iteration = manifest[manifest["iteration"].eq(1)].copy()
    counts = iteration["split"].value_counts().to_dict()
    expected = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    if counts != expected:
        raise ValueError(f"Unexpected Iteration-1 split counts: {counts}")

    print("[3/5] Checking V11 prerequisites and 149-case summaries...", flush=True)
    required = [
        PACKAGE_ROOT / "outputs" / "12_v11_gain_stability_audit" / "iteration_1" / "audit_complete.json",
        PACKAGE_ROOT / "outputs" / "12_v11_gain_stability_audit" / "iteration_1" / "selected_gain.json",
        PACKAGE_ROOT / "outputs" / "12_v11_gain_stability_audit" / "iteration_1" / "selected_gain_formula.csv",
        PACKAGE_ROOT / "outputs" / "00_qc_sensitivity_ablation" / "development_case_summary.csv",
        PACKAGE_ROOT / "shared" / "frozen_similarity_group_map_199cases.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V12 prerequisites:\n" + "\n".join(missing))
    completion = json.loads(required[0].read_text(encoding="utf-8"))
    selected_gain = json.loads(required[1].read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("final_test_cases_read") != 0:
        raise RuntimeError("V11 audit is incomplete or final-test seal is invalid")
    if float(selected_gain["selected_gain"]) != 0.9:
        raise ValueError("V12 expects the frozen V11 gain to be 0.90")
    summary = pd.read_csv(required[3])
    if len(summary) != 149 or summary["case_id"].nunique() != 149:
        raise ValueError("Expected 149 complete development summaries")

    print("[4/5] Checking predictor-only features and candidate registry...", flush=True)
    config = V12Config()
    config.validate()
    candidates = candidate_configurations(config)
    all_features = sorted(set(PHYSICAL_CONTEXT_FEATURES + TAIL_CONTEXT_FEATURES))
    if any(name.startswith("stress_") for name in all_features):
        raise AssertionError("Stress leakage in V12 gain context")
    if len(candidates) != 43 or set(FEATURE_SETS) != {"physical_9", "physical_plus_tail_13"}:
        raise AssertionError("V12 candidate registry drifted")

    print("[5/5] Checking similarity-group fold isolation...", flush=True)
    development = iteration[~iteration["split"].eq("final_test")][["case_id", "similarity_group"]]
    folds = _balanced_group_folds(development, n_folds=5, seed=42)
    if folds.groupby("similarity_group")["fold"].nunique().max() != 1:
        raise AssertionError("Similarity groups leak across V12 folds")
    print("Case files:", len(inventory))
    print("Iteration-1 split:", counts)
    print("Development similarity groups:", development["similarity_group"].nunique())
    print("Gain candidates:", len(candidates))
    print("Oracle grid values per case:", len(config.oracle_gain_grid))
    print("Fast V12 package validation passed. Full provenance checks run in the notebook.")


if __name__ == "__main__":
    main()
