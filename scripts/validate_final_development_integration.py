#!/usr/bin/env python3
"""Validate the frozen 149-case development-integration package."""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import os
import subprocess
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "src"
NOTEBOOK = (
    PACKAGE_ROOT
    / "notebooks/18_149Case_Development_Integration/01_149Case_Development_Integration.ipynb"
)
BUILDER = PACKAGE_ROOT / "scripts/build_final_development_integration_notebook.py"
SLURM = PACKAGE_ROOT / "cluster/run_149case_development_integration.slurm"
SOURCE = SOURCE_DIR / "final_development_integration.py"
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from final_development_integration import DevelopmentIntegrationConfig  # noqa: E402


def validate_notebook(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        raise ValueError("Integration notebook is not nbformat 4")
    ids = []
    for index, cell in enumerate(payload.get("cells", [])):
        ids.append(cell.get("id"))
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")
    if len(ids) != len(set(ids)) or None in ids:
        raise ValueError("Notebook cell IDs are missing or duplicated")


def validate_static() -> None:
    print("[1/4] Checking source, builder, notebook and Slurm syntax...", flush=True)
    for path in (SOURCE, BUILDER, NOTEBOOK, SLURM):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (SOURCE, BUILDER):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    validate_notebook(NOTEBOOK)
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    DevelopmentIntegrationConfig().validate()


def validate_data_and_artifacts() -> None:
    print("[2/4] Checking the frozen 149/50 manifest...", flush=True)
    manifest = pd.read_csv(PACKAGE_ROOT / "shared/frozen_case_split_manifest_199cases.csv")
    if set(manifest["iteration"]) != {1, 2, 3, 4}:
        raise ValueError("Frozen manifest does not contain four rotations")
    final_ids = set(
        manifest.loc[manifest["split"].eq("final_test"), "case_id"].astype(str)
    )
    all_ids = set(manifest["case_id"].astype(str))
    if len(all_ids) != 199 or len(final_ids) != 50 or len(all_ids - final_ids) != 149:
        raise ValueError("Frozen manifest is not a 149-development/50-final division")

    print("[3/4] Checking case inventory and prerequisite summaries...", flush=True)
    case_directories = [
        PACKAGE_ROOT / "FE_Results_Cases_All",
        PACKAGE_ROOT / " Case Test Data/FE_Results_Cases_All",
        PACKAGE_ROOT / "Case Test Data/FE_Results_Cases_All",
    ]
    case_directory = next((path for path in case_directories if path.exists()), None)
    if case_directory is None:
        raise FileNotFoundError("Could not locate FE_Results_Cases_All")
    if len(list(case_directory.glob("FE_Results_Case_*.txt"))) != 199:
        raise ValueError("Expected 199 FEM case files")
    required = [
        PACKAGE_ROOT / "outputs/00_qc_sensitivity_ablation/development_case_summary.csv",
        PACKAGE_ROOT
        / "outputs/17_v11_four_rotation_formal/four_rotation_summary/aggregation_complete.json",
        PACKAGE_ROOT
        / "outputs/17_v11_four_rotation_formal/four_rotation_summary/all_rotation_split_metrics.csv",
    ]
    for iteration in (1, 2, 3, 4):
        required.extend([
            PACKAGE_ROOT
            / f"outputs/17_v11_four_rotation_formal/iteration_{iteration}/pipeline_complete.json",
            PACKAGE_ROOT
            / f"outputs/10_signed_staged_residual_symbolic_pilot/iteration_{iteration}/pilot_complete.json",
            PACKAGE_ROOT
            / f"outputs/10_signed_staged_residual_symbolic_pilot/iteration_{iteration}/selected_composite_formula.csv",
            PACKAGE_ROOT
            / f"outputs/11_tail_aware_localised_symbolic/iteration_{iteration}/pilot_complete.json",
            PACKAGE_ROOT
            / f"outputs/11_tail_aware_localised_symbolic/iteration_{iteration}/selected_composite_formula.csv",
            PACKAGE_ROOT
            / f"outputs/11_tail_aware_localised_symbolic/iteration_{iteration}/training_cache/mean_calibration_model.json",
            PACKAGE_ROOT
            / f"outputs/12_v11_gain_stability_audit/iteration_{iteration}/audit_complete.json",
            PACKAGE_ROOT
            / f"outputs/12_v11_gain_stability_audit/iteration_{iteration}/selected_gain.json",
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))

    print("[4/4] Checking frozen completion policies...", flush=True)
    for path in required:
        if path.suffix != ".json" or "complete" not in path.name:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            raise RuntimeError(f"Incomplete prerequisite: {path}")
        if int(record.get("final_test_cases_read", 0)) != 0:
            raise RuntimeError(f"Prerequisite accessed final-test elements: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="Skip data and frozen-artifact checks.",
    )
    args = parser.parse_args()
    validate_static()
    if not args.syntax_only:
        validate_data_and_artifacts()
    print("149-case development-integration package validation passed.", flush=True)


if __name__ == "__main__":
    main()

