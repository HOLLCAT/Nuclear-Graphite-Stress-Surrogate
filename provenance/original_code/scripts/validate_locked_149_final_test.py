#!/usr/bin/env python3
"""Read-free structural validation for the locked 149-case final package."""

from pathlib import Path
import ast
import json
import subprocess
import sys

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_ROOT / "src" / "final_locked_149_evaluation.py"
BUILDER = PACKAGE_ROOT / "scripts" / "build_locked_149_final_test_notebook.py"
NOTEBOOK = (
    PACKAGE_ROOT
    / "notebooks/19_One_Time_Final_50Case_Locked_149"
    / "01_One_Time_Final_50Case_Locked_149.ipynb"
)
SLURM = PACKAGE_ROOT / "cluster" / "run_locked_149_final_test.slurm"

if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ct3_common import (
    build_paths,
    discover_case_files,
    evaluate_prediction_arrays,
    load_frozen_manifest,
    read_complete_case,
)
from final_locked_149_evaluation import (
    LockedFinal149Config,
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    output_directory,
    predict_frozen_models,
    predictor_only_case_summary,
    preflight_locked_final_149,
)


def main() -> None:
    print("[1/6] Checking source, builder, notebook and Slurm syntax...", flush=True)
    ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    ast.parse(BUILDER.read_text(encoding="utf-8"), filename=str(BUILDER))
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Final notebook is not nbformat 4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{NOTEBOOK}:cell-{index}")

    print("[2/6] Checking the frozen 199-case split without reading case values...", flush=True)
    paths = build_paths(PACKAGE_ROOT)
    _, inventory = discover_case_files(paths.case_dir)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    iteration_one = manifest[manifest["iteration"].eq(1)]
    counts = iteration_one["split"].value_counts().to_dict()
    expected = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    if counts != expected:
        raise ValueError(f"Unexpected frozen split: {counts}")
    development_cases = manifest.loc[~manifest["split"].eq("final_test"), "case_id"].nunique()
    if development_cases != 149:
        raise ValueError(f"Expected 149 development cases, found {development_cases}")

    print("[3/6] Checking the human-approved frozen protocol...", flush=True)
    protocol = json.loads(
        (PACKAGE_ROOT / "shared/locked_149_final_test_protocol.json").read_text(encoding="utf-8")
    )
    if not protocol.get("approved"):
        raise RuntimeError("The one-time final protocol is not approved")
    if protocol.get("primary_model") != PRIMARY_MODEL:
        raise RuntimeError("The formal primary role has changed")
    if protocol.get("secondary_model") != SECONDARY_MODEL:
        raise RuntimeError("The secondary research role has changed")

    print("[4/6] Running the read-free frozen-artifact preflight...", flush=True)
    config = LockedFinal149Config()
    preflight = preflight_locked_final_149(PACKAGE_ROOT, config)
    if not preflight["checks"]["pass"].all():
        raise RuntimeError("Locked final preflight did not pass")

    print("[5/6] Reproducing the frozen models on one development case...", flush=True)
    reproduction_case = "case_01"
    frame, audit = read_complete_case(preflight["path_by_case"][reproduction_case])
    summary = predictor_only_case_summary(
        frame,
        reproduction_case,
        audit["case_number"],
    )
    predictions = predict_frozen_models(
        frame,
        summary,
        preflight["states"],
        preflight["mean_model"],
        preflight["active_iterations"],
        preflight["tail_gain"],
    )
    actual = frame["sigma_max_principal"].to_numpy(dtype=np.float64)
    prior = pd.read_csv(
        PACKAGE_ROOT
        / "outputs/18_149case_development_integration/formal_149case_integration"
        / "development_candidate_case_metrics.csv.gz"
    )
    for model in (PRIMARY_MODEL, SECONDARY_MODEL):
        current = evaluate_prediction_arrays(actual, predictions[model])
        prior_row = prior[
            prior["case_id"].eq(reproduction_case) & prior["model"].eq(model)
        ]
        if len(prior_row) != 1:
            raise RuntimeError(f"Missing frozen development evidence for {model}")
        prior_row = prior_row.iloc[0]
        differences = [
            abs(float(current[metric]) - float(prior_row[metric]))
            for metric in ("rmse", "r2", "predicted_mean", "predicted_p95", "predicted_p99")
        ]
        if max(differences) > 2e-4:
            raise RuntimeError(
                f"{model} does not reproduce the frozen development prediction: {differences}"
            )
        print(f"{model}: maximum reproduction difference={max(differences):.3e}")

    print("[6/6] Checking one-time output state...", flush=True)
    output = output_directory(PACKAGE_ROOT, config)
    complete_path = output / "final_evaluation_complete.json"
    if complete_path.exists():
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete" or completion.get("final_test_cases_read") != 50:
            raise RuntimeError("Existing final completion marker is invalid")
        print("WARNING: final evaluation is complete and must not be submitted again")
    else:
        print("Final values remain unread by this validation; first run or exact-signature resume is permitted")
    print("Case files:", len(inventory))
    print("Development cases:", development_cases)
    print("Iteration-1 split:", counts)
    print("Final case element values read by validation: 0")
    print("Locked final-test package validation passed.")


if __name__ == "__main__":
    main()
