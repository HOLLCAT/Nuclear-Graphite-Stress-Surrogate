#!/usr/bin/env python3
"""Build the formal V11 four-rotation notebooks."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks" / "17_Formal_V11_Four_Rotation"


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


def notebook(cells: list[dict]) -> dict:
    return {
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
        "cells": cells,
    }


COMMON_IMPORTS = """
from pathlib import Path
import json
import sys

import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print


def resolve_package_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "formal_v11_four_rotation.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""


def build_registration() -> Path:
    path = NOTEBOOK_DIR / "01_Register_Completed_Iteration_1.ipynb"
    cells = [
        markdown(
            """
# Formal V11 Method: Register Completed Iteration 1

Iteration 1 is already complete. V10 performed 8,000 population iterations,
V11 performed 4,000 population iterations, and the validation-only gain audit
selected lambda=0.90 and passed the provisional V11 promotion gates. The V13
audit subsequently retained V11 gain 0.90 as the primary deployable model and
classified V13 as non-deployed research.

This notebook does not rerun PySR. It verifies and hashes the reviewed
Iteration-1 artifacts and creates the formal four-rotation registration record.
The 50 final-test element files remain unread.
"""
        ),
        markdown("## 1. Imports and package root"),
        code(COMMON_IMPORTS),
        code(
            """
from formal_v11_four_rotation import (
    output_directory,
    register_completed_iteration_one,
)

RESULT = register_completed_iteration_one(PACKAGE_ROOT)
OUTPUT_DIR = output_directory(PACKAGE_ROOT, 1)
display(pd.DataFrame([RESULT]))
display(pd.read_csv(OUTPUT_DIR / "registered_artifact_hashes.csv"))
print(json.dumps(RESULT, indent=2))
"""
        ),
        markdown(
            """
## Decision

No additional Iteration-1 symbolic search is required. Rerunning it would add
stochastic cost but would not create a new independent split. The registered
result is the first development rotation; Iterations 2-4 test method and
formula-structure stability under new case-level roles.
"""
        ),
    ]
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    return path


def build_rotation(iteration: int) -> Path:
    path = NOTEBOOK_DIR / f"0{iteration}_Formal_V11_Iteration_{iteration}.ipynb"
    cells = [
        markdown(
            f"""
# Formal V11 Symbolic Regression: Iteration {iteration}

This notebook runs one complete frozen development rotation:

1. V10 signed staged residual symbolic search, using the Iteration-{iteration}
   119 training cases and 595,000 stratified discovery rows;
2. V11 case-mean calibration and tail-aware localised symbolic search;
3. validation-only scalar gain selection, followed by one internal-test audit.

Every complete validation and internal-test case is evaluated. The 50 final
test case names are inventoried but their element files are not read. Each
PySR stage is segmented and recoverable, so a rerun resumes completed work.
"""
        ),
        markdown("## 1. Imports and package root"),
        code(COMMON_IMPORTS),
        markdown("## 2. Run or resume the locked rotation"),
        code(
            f"""
from formal_v11_four_rotation import output_directory, run_formal_rotation

ITERATION = {iteration}
RUN_FORMAL_ROTATION = True
OUTPUT_DIR = output_directory(PACKAGE_ROOT, ITERATION)

if RUN_FORMAL_ROTATION:
    RESULT = run_formal_rotation(PACKAGE_ROOT, ITERATION)
    display(pd.DataFrame([RESULT]))
else:
    RESULT = None
    print("Formal rotation skipped.")
"""
        ),
        markdown("## 3. Saved evidence"),
        code(
            f"""
roots = {{
    "pipeline": OUTPUT_DIR,
    "v10": PACKAGE_ROOT / "outputs/10_signed_staged_residual_symbolic_pilot/iteration_{iteration}",
    "v11": PACKAGE_ROOT / "outputs/11_tail_aware_localised_symbolic/iteration_{iteration}",
    "gain": PACKAGE_ROOT / "outputs/12_v11_gain_stability_audit/iteration_{iteration}",
}}

artifacts = {{
    "pipeline_completion": roots["pipeline"] / "pipeline_complete.json",
    "v10_completion": roots["v10"] / "pilot_complete.json",
    "v11_completion": roots["v11"] / "pilot_complete.json",
    "gain_completion": roots["gain"] / "audit_complete.json",
    "selected_gain": roots["gain"] / "selected_gain.json",
    "selected_formula": roots["gain"] / "selected_gain_formula.txt",
    "split_metrics": roots["gain"] / "selected_models_split_metrics.csv",
}}
for name, artifact in artifacts.items():
    print(f"{{name:22s}} exists={{artifact.exists()}}  {{artifact}}")

if artifacts["split_metrics"].exists():
    display(pd.read_csv(artifacts["split_metrics"]))
if artifacts["selected_formula"].exists():
    print("\\n" + artifacts["selected_formula"].read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## Interpretation

This is a development-rotation result, not external validation. Formula
coefficients are expected to change because the training cases change. The
primary questions are whether the same hierarchical method remains useful,
whether selected gain and feature families are reasonably stable, and whether
validation improvements survive the one-time internal-test audit.
"""
        ),
    ]
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    return path


def build_aggregation() -> Path:
    path = NOTEBOOK_DIR / "05_Aggregate_Four_Rotation_Stability.ipynb"
    cells = [
        markdown(
            """
# Aggregate Formal V11 Four-Rotation Stability

Run this notebook only after Iterations 1-4 each have a formal completion
record. It combines development validation/internal-test metrics, selected
gains and exact formula records. It does not read any final-test element file.
"""
        ),
        code(COMMON_IMPORTS),
        code(
            """
from formal_v11_four_rotation import (
    aggregate_completed_rotations,
    output_directory,
)

RESULT = aggregate_completed_rotations(PACKAGE_ROOT)
SUMMARY_DIR = output_directory(PACKAGE_ROOT) / "four_rotation_summary"
print(json.dumps(RESULT, indent=2))
display(pd.read_csv(SUMMARY_DIR / "rotation_completion_summary.csv"))
display(pd.read_csv(SUMMARY_DIR / "selected_model_stability_summary.csv"))
"""
        ),
        markdown(
            """
## Selection boundary

The four rotations support method and structure stability analysis. They do
not authorise choosing a formula with the final 50 cases. A later constant
refit may use all 149 development cases only after a structure is frozen; the
final 50 cases are then evaluated exactly once.
"""
        ),
    ]
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    return path


def build() -> list[Path]:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    paths = [build_registration()]
    paths.extend(build_rotation(iteration) for iteration in (2, 3, 4))
    paths.append(build_aggregation())
    return paths


if __name__ == "__main__":
    for built in build():
        print(built)
