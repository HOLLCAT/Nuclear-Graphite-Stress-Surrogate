#!/usr/bin/env python3
"""Fast static and frozen-parent validation for the V13 gain audit."""

from pathlib import Path
import ast
import json
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_ROOT / "src" / "v13_residual_gain_audit.py"
BUILDER = PACKAGE_ROOT / "scripts" / "build_v13_residual_gain_audit_notebook.py"
NOTEBOOK = (
    PACKAGE_ROOT
    / "notebooks"
    / "16_V13_Residual_Gain_Audit"
    / "01_V13_Residual_Gain_Audit_Iteration_1.ipynb"
)
SLURM = PACKAGE_ROOT / "cluster" / "run_v13_residual_gain_audit.slurm"

if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ct3_common import build_paths, discover_case_files, load_frozen_manifest
from v13_residual_gain_audit import V13ResidualGainAuditConfig


def main() -> None:
    print("[1/4] Checking Python, notebook and Slurm syntax...", flush=True)
    for path in [SOURCE, BUILDER]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")
    if not SLURM.exists():
        raise FileNotFoundError(SLURM)

    print("[2/4] Checking the frozen 199-case manifest...", flush=True)
    paths = build_paths(PACKAGE_ROOT)
    _, inventory = discover_case_files(paths.case_dir)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    iteration = manifest[manifest["iteration"].eq(1)]
    counts = iteration["split"].value_counts().to_dict()
    expected = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    if counts != expected:
        raise ValueError(f"Unexpected frozen split: {counts}")

    print("[3/4] Checking frozen V13 artifacts...", flush=True)
    root = PACKAGE_ROOT / "outputs" / "15_v13_structural_tail_residual" / "iteration_1"
    required = [
        root / "v13_complete.json",
        root / "v13_promotion_decision.json",
        root / "v13_scope_metrics.csv",
        root / "v13_case_metrics.csv.gz",
        root / "stages" / "stage_structural_residual" / "selected_formula.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V13 artifacts:\n" + "\n".join(missing))
    completion = json.loads(required[0].read_text(encoding="utf-8"))
    decision = json.loads(required[1].read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("final_test_cases_read") != 0:
        raise RuntimeError("V13 parent is incomplete or final-test seal is broken")
    if decision.get("promotion_status") != "retain_v11_global_gain_0_90":
        raise RuntimeError("Gain audit expects V13 to retain V11")
    formula = pd.read_csv(required[4])
    if len(formula) != 1 or int(formula.iloc[0]["candidate_index"]) != 2:
        raise ValueError("Expected frozen V13 candidate 2")

    print("[4/4] Checking gain policy...", flush=True)
    config = V13ResidualGainAuditConfig()
    config.validate()
    print("Case files:", len(inventory))
    print("Iteration-1 split:", counts)
    print("Validation gain count:", len(config.gain_grid))
    print("Final-test element files read: 0")
    print("Fast V13 residual-gain audit validation passed.")


if __name__ == "__main__":
    main()

