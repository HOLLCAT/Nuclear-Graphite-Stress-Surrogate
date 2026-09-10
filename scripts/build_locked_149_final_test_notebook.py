#!/usr/bin/env python3
"""Build the one-time final notebook for the locked 149-case symbolic model."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks" / "19_One_Time_Final_50Case_Locked_149"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_One_Time_Final_50Case_Locked_149.ipynb"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def build() -> Path:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "NotebookCT3",
                "language": "python",
                "name": "notebookct3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": [
            markdown(
                """
# One-Time Final Evaluation of the Locked 149-Case Symbolic Model

This notebook opens the sealed 50-case final set exactly once for reporting.

## Frozen model roles

- **Primary deployment model:** `mean_calibrated_consensus`.
- **Secondary research model:** `full_consensus_tail`, using the predeclared
  active tail rotations 1, 2 and 3 at fixed gain 1.0.
- The secondary result cannot replace the primary because of final-test
  performance.

## Irreversible evaluation contract

- No fitting, constant refit, formula search, gain selection, threshold tuning
  or model promotion is permitted.
- Every element in all 50 final cases is evaluated by both frozen models.
- A fixed sample is used only to render publication figures; it is never used
  to calculate reported metrics.
- The thresholds are project research-reporting thresholds, not nuclear safety
  or licensing limits.
- Failed thresholds are reported as limitations and do not reopen development.
- An interrupted identical run may resume signed per-case caches. A completed
  run refuses re-execution.
"""
            ),
            markdown("## 1. Resolve the package and import the locked evaluator"),
            code(
                """
from pathlib import Path
import json
import os
import sys

import pandas as pd

try:
    from IPython.display import Image, display
except Exception:
    display = print
    Image = None


def resolve_package_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "final_locked_149_evaluation.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from final_locked_149_evaluation import (
    LockedFinal149Config,
    PRIMARY_MODEL,
    PRIMARY_ROLE,
    SECONDARY_MODEL,
    SECONDARY_ROLE,
    output_directory,
    preflight_locked_final_149,
    run_locked_final_149,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
print("Explicit final-run authorisation:", os.environ.get("CT3_AUTHORISE_LOCKED_FINAL_TEST"))
"""
            ),
            markdown("## 2. Predeclared metrics, thresholds and read-free preflight"),
            code(
                """
CONFIG = LockedFinal149Config()
OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)
PREFLIGHT = preflight_locked_final_149(PACKAGE_ROOT, CONFIG)

print("Primary:", PRIMARY_MODEL, "-", PRIMARY_ROLE)
print("Secondary:", SECONDARY_MODEL, "-", SECONDARY_ROLE)
print("Locked final cases:", len(PREFLIGHT["final_ids"]))
print("Signature:", PREFLIGHT["signature"]["final_evaluation_signature_sha256"])
print("Output directory:", OUTPUT_DIR)
display(PREFLIGHT["checks"])
display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "locked_value"}))
"""
            ),
            markdown("## 3. Open the sealed set once, or resume the exact signed run"),
            code(
                """
if os.environ.get("CT3_AUTHORISE_LOCKED_FINAL_TEST") != "YES":
    raise PermissionError(
        "Final cases remain closed. Use the approved CSF3 submission command "
        "with CT3_AUTHORISE_LOCKED_FINAL_TEST=YES."
    )

RESULT = run_locked_final_149(PACKAGE_ROOT, CONFIG, preflight=PREFLIGHT)
display(pd.DataFrame([RESULT]).T.rename(columns={0: "value"}))
"""
            ),
            markdown("## 4. Final metrics and predeclared reporting gates"),
            code(
                """
artifacts = {
    "completion": OUTPUT_DIR / "final_evaluation_complete.json",
    "scope_metrics": OUTPUT_DIR / "final_50case_scope_metrics.csv",
    "case_metrics": OUTPUT_DIR / "final_50case_case_metrics.csv.gz",
    "reporting_gates": OUTPUT_DIR / "final_primary_research_reporting_gates.csv",
    "bootstrap": OUTPUT_DIR / "final_similarity_group_bootstrap_ci.csv",
    "generalisation": OUTPUT_DIR / "development_to_final_generalisation.csv",
    "coverage": OUTPUT_DIR / "final_predictor_coverage_against_development.csv",
    "boundaries": OUTPUT_DIR / "final_provisional_physical_boundary_report.csv",
    "worst_cases": OUTPUT_DIR / "final_worst_15_primary_cases_by_rmse.csv",
    "figures": OUTPUT_DIR / "figure_manifest_for_thesis.csv",
    "formula": OUTPUT_DIR / "frozen_symbolic_formulas_evaluated.txt",
}
for name, path in artifacts.items():
    print(f"{name:18s} exists={path.exists()}  {path}")

for key in [
    "scope_metrics",
    "reporting_gates",
    "bootstrap",
    "generalisation",
    "coverage",
    "worst_cases",
    "figures",
]:
    print(f"\\n{key}")
    display(pd.read_csv(artifacts[key]))
"""
            ),
            markdown("## 5. Thesis-ready figures"),
            code(
                """
figure_manifest = pd.read_csv(artifacts["figures"])
figure_dir = OUTPUT_DIR / "figures_for_thesis"
for record in figure_manifest.itertuples(index=False):
    path = figure_dir / record.figure
    print(f"\\n{record.figure}: {record.purpose}")
    if Image is not None:
        display(Image(filename=str(path)))
"""
            ),
            markdown(
                """
## Interpretation rule

The primary formula remains the formal result irrespective of the secondary
research model's final score. The final metrics are an unbiased estimate only
for this sealed 50-case distribution. They do not establish safety margins or
lifetimes without a separately justified strength, damage or failure model.
"""
            ),
        ],
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
