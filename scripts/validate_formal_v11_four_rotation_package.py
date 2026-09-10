#!/usr/bin/env python3
"""Fast static and data validation for the formal V11 rotation package."""

from pathlib import Path
import ast
import json
import os
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / ".cache" / "matplotlib"))
(PACKAGE_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from formal_v11_four_rotation import (  # noqa: E402
    register_completed_iteration_one,
)
from signed_staged_residual_symbolic import SignedStagedPilotConfig  # noqa: E402
from v11_tail_aware_localised_symbolic import V11PilotConfig  # noqa: E402
from v11_gain_stability_audit import V11GainAuditConfig  # noqa: E402


NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks" / "17_Formal_V11_Four_Rotation"
NOTEBOOKS = [
    NOTEBOOK_DIR / "01_Register_Completed_Iteration_1.ipynb",
    NOTEBOOK_DIR / "02_Formal_V11_Iteration_2.ipynb",
    NOTEBOOK_DIR / "03_Formal_V11_Iteration_3.ipynb",
    NOTEBOOK_DIR / "04_Formal_V11_Iteration_4.ipynb",
    NOTEBOOK_DIR / "05_Aggregate_Four_Rotation_Stability.ipynb",
]
SLURMS = [
    PACKAGE_ROOT / "cluster" / f"run_formal_v11_iteration{iteration}.slurm"
    for iteration in (2, 3, 4)
]


def _validate_notebook(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        raise ValueError(f"Notebook is not nbformat 4: {path}")
    for index, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")


def main() -> None:
    print("[1/5] Checking source, notebooks and Slurm files...", flush=True)
    source_files = [
        SOURCE_DIR / "formal_v11_four_rotation.py",
        SOURCE_DIR / "signed_staged_residual_symbolic.py",
        SOURCE_DIR / "v11_tail_aware_localised_symbolic.py",
        SOURCE_DIR / "v11_gain_stability_audit.py",
    ]
    for path in [*source_files, *NOTEBOOKS, *SLURMS]:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in source_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in NOTEBOOKS:
        _validate_notebook(path)

    print("[2/5] Checking the frozen four-rotation manifest...", flush=True)
    manifest = pd.read_csv(PACKAGE_ROOT / "shared/frozen_case_split_manifest_199cases.csv")
    if set(manifest["iteration"]) != {1, 2, 3, 4}:
        raise ValueError("Frozen manifest does not contain exactly four rotations")
    expected = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    for iteration in (1, 2, 3, 4):
        counts = manifest.loc[manifest["iteration"] == iteration, "split"].value_counts()
        actual = {name: int(counts.get(name, 0)) for name in expected}
        if actual != expected:
            raise ValueError(f"Iteration {iteration} split mismatch: {actual}")
        print(f"Iteration {iteration}: {actual}")

    print("[3/5] Checking the 199 FEM files and prerequisite summaries...", flush=True)
    case_directories = [
        PACKAGE_ROOT / "FE_Results_Cases_All",
        PACKAGE_ROOT / " Case Test Data" / "FE_Results_Cases_All",
    ]
    case_directory = next((path for path in case_directories if path.exists()), None)
    if case_directory is None:
        raise FileNotFoundError("Could not locate FE_Results_Cases_All")
    case_files = list(case_directory.glob("FE_Results_Case_*.txt"))
    if len(case_files) != 199:
        raise ValueError(f"Expected 199 FEM files, found {len(case_files)}")
    required = [
        PACKAGE_ROOT / "outputs/00_qc_sensitivity_ablation/development_case_summary.csv",
        PACKAGE_ROOT / "shared/v10_authoritative_v8_candidate5/baseline_manifest.json",
        PACKAGE_ROOT / "scripts/run_signed_residual_stage_segment.py",
        PACKAGE_ROOT / "scripts/run_v11_tail_segment.py",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[4/5] Checking all rotation configurations...", flush=True)
    for iteration in (1, 2, 3, 4):
        SignedStagedPilotConfig(
            iteration=iteration, output_subdir=f"iteration_{iteration}"
        ).validate()
        V11PilotConfig(
            iteration=iteration, output_subdir=f"iteration_{iteration}"
        ).validate()
        V11GainAuditConfig(
            iteration=iteration, output_subdir=f"iteration_{iteration}"
        ).validate()

    print("[5/5] Registering and verifying completed Iteration 1...", flush=True)
    registered = register_completed_iteration_one(PACKAGE_ROOT)
    print(json.dumps(registered, indent=2))
    print("Formal V11 four-rotation package validation passed.", flush=True)


if __name__ == "__main__":
    main()
