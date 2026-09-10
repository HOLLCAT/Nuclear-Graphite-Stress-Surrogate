#!/usr/bin/env python3
"""Build the 149-case development-integration notebook."""

from __future__ import annotations

from pathlib import Path
import json


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PACKAGE_ROOT
    / "notebooks"
    / "18_149Case_Development_Integration"
)
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_149Case_Development_Integration.ipynb"


def markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """# CT3: Integration of Four Frozen Rotations on 149 Development Cases

## Purpose

This notebook is the **development-integration stage**, not another symbolic search. It combines the four independently searched V10/V11 rotations after their structures have been frozen.

The procedure is deliberately conservative:

1. Every one of the 149 development cases remains complete; no element is sampled.
2. The four V10 stress formulae receive fixed equal weights of 0.25.
3. One compact case-mean correction is refitted on all 149 development cases.
4. Only V11 tail formulae that remain non-zero after complete-case centring enter the tail consensus. A constant tail therefore contributes nothing and is excluded.
5. One global tail gain is fitted with complete-element P90/P95/P99 weighting and constrained to the interval 0 to 1.
6. A predeclared gate selects the full tail model, the mean-only model, or the unmodified V10 consensus.

The 50 final-test cases are checked only for inventory, header and row count. Their predictor and stress values are not parsed or used. The output is a locked candidate that still requires explicit human review before the one-time final test.
""",
        "purpose",
    ),
    code_cell(
        """from pathlib import Path
import json
import os
import sys

import pandas as pd

cwd = Path.cwd().resolve()
PACKAGE_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "src/ct3_common.py").is_file()), None)
if PACKAGE_ROOT is None:
    raise FileNotFoundError("Open the notebook from inside this project")
SRC_DIR = PACKAGE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from final_development_integration import (
    DevelopmentIntegrationConfig,
    output_directory,
    preflight_development_integration,
    run_development_integration,
)

CONFIG = DevelopmentIntegrationConfig(
    output_subdir="formal_149case_integration",
    keep_case_cache_after_success=False,
    force_rebuild_case_cache=False,
)
OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)

print("Package root:", PACKAGE_ROOT)
print("Output directory:", OUTPUT_DIR)
print("Final-test policy: inventory/header checks only; zero parsed predictor or stress rows")
print("Configuration:", CONFIG)
""",
        "setup",
    ),
    markdown_cell(
        """## Preflight

Preflight checks all four completion chains, formula artifacts, the 149/50 frozen division, and the input hashes. It also confirms that Rotation 4 has a constant tail expression, which becomes exactly zero after complete-case centring.
""",
        "preflight-note",
    ),
    code_cell(
        """preflight = preflight_development_integration(PACKAGE_ROOT, CONFIG)
display(preflight["checks"])
display(preflight["rotation_audit"])
print("Frozen input signature:", preflight["input_signature"])
""",
        "preflight",
    ),
    markdown_cell(
        """## Run or resume the 149-case integration

Each complete case is checkpointed after its four frozen symbolic predictions have been evaluated. If the scheduled job stops, submitting the same notebook again reuses valid case checkpoints. No PySR search is run here.
""",
        "run-note",
    ),
    code_cell(
        """completion = run_development_integration(PACKAGE_ROOT, CONFIG)
display(pd.DataFrame([completion]))
""",
        "run",
    ),
    markdown_cell(
        """## Locked result and audit

The table below is a **149-case fit diagnostic**. It is useful for checking whether the refit behaves as intended, but it is not an unbiased generalisation estimate because the mean coefficients and tail gain were fitted on these same development cases. The one-time 50-case result remains the only final performance statement.
""",
        "result-note",
    ),
    code_cell(
        """summary = pd.read_csv(OUTPUT_DIR / "development_candidate_summary.csv")
activity = pd.read_csv(OUTPUT_DIR / "tail_activity_audit.csv")
decision = json.loads((OUTPUT_DIR / "final_model_decision.json").read_text())
release = json.loads((OUTPUT_DIR / "final_test_release_gate.json").read_text())

display(summary)
display(activity)
display(pd.DataFrame([{
    "selected_model": decision["selected_model"],
    "selection_reason": decision["selection_reason"],
    "active_tail_iterations": decision["active_tail_iterations"],
    "inactive_tail_iterations": decision["inactive_tail_iterations"],
    "final_test_cases_read": decision["final_test_cases_read"],
}]))
display(pd.DataFrame([release]))

print("\\nLocked formula file:", OUTPUT_DIR / "locked_model_formula.txt")
print("\\n" + (OUTPUT_DIR / "locked_model_formula.txt").read_text())
""",
        "results",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "NotebookCT3",
            "language": "python",
            "name": "notebookct3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
