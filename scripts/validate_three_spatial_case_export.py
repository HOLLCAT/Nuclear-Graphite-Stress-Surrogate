#!/usr/bin/env python3
"""Fast validation for the best/median/worst full-element export."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_ROOT / "src/export_three_spatial_comparison_cases.py"
SLURM = PACKAGE_ROOT / "cluster/run_three_spatial_case_export.slurm"
FINAL_ROOT = PACKAGE_ROOT / "outputs/19_one_time_final_50case_locked_149/formal_final_50case"

if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ct3_common import build_paths, discover_case_files, load_frozen_manifest
from export_three_spatial_comparison_cases import SELECTED_CASES
from final_locked_149_evaluation import PRIMARY_MODEL


def main() -> None:
    print("[1/5] Checking Python and Slurm syntax...", flush=True)
    ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    subprocess.run(["bash", "-n", str(SLURM)], check=True)

    print("[2/5] Checking the frozen inventory and final-test assignments...", flush=True)
    paths = build_paths(PACKAGE_ROOT)
    _, inventory = discover_case_files(paths.case_dir)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    iteration_one = manifest[manifest["iteration"].eq(1)]
    for category, case_id in SELECTED_CASES.items():
        split = iteration_one.loc[iteration_one["case_id"].eq(case_id), "split"].tolist()
        if split != ["final_test"]:
            raise RuntimeError(f"{category} {case_id} is not uniquely in final_test")

    print("[3/5] Checking completed final-test evidence...", flush=True)
    completion = json.loads(
        (FINAL_ROOT / "final_evaluation_complete.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "complete" or completion.get("final_test_cases_read") != 50:
        raise RuntimeError("Locked final-test evidence is incomplete")
    if completion.get("primary_model") != PRIMARY_MODEL:
        raise RuntimeError("The formal primary model has changed")

    print("[4/5] Checking the formal best/median/worst selection...", flush=True)
    selection = pd.read_csv(FINAL_ROOT / "final_spatial_case_selection.csv")
    found = dict(zip(selection["category"], selection["case_id"]))
    if found != SELECTED_CASES:
        raise RuntimeError(f"Spatial selection mismatch: {found}")
    metrics = pd.read_csv(FINAL_ROOT / "final_50case_case_metrics.csv.gz")
    primary = metrics[
        metrics["model"].eq(PRIMARY_MODEL)
        & metrics["case_id"].isin(SELECTED_CASES.values())
    ]
    if len(primary) != 3 or not primary["n_elements"].eq(400_360).all():
        raise RuntimeError("Full-element primary metrics are missing")

    print("[5/5] Checking all three raw FEM files...", flush=True)
    for category, case_id in SELECTED_CASES.items():
        case_number = int(case_id.split("_")[-1])
        row = inventory[inventory["case_number"].eq(case_number)]
        if len(row) != 1 or int(row.iloc[0]["n_elements_from_line_count"]) != 400_360:
            raise RuntimeError(f"{category} {case_id} raw file is incomplete")

    print("Case files:", len(inventory))
    print("Selected cases:", SELECTED_CASES)
    print("Frozen primary model:", PRIMARY_MODEL)
    print(primary[["case_id", "rmse", "r2"]].sort_values("rmse").to_string(index=False))
    print("Total element predictions to export:", 3 * 400_360)
    print("Training, fitting, and selection in export: none")
    print("Three-case full-element export package validation passed.")


if __name__ == "__main__":
    main()
