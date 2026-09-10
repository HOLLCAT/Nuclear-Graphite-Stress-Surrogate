#!/usr/bin/env python3
"""Build the frozen V13 residual-gain audit notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks" / "16_V13_Residual_Gain_Audit"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_V13_Residual_Gain_Audit_Iteration_1.ipynb"


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
    }
    notebook["cells"] = [
        markdown(
            """
# Frozen V13 Residual-Gain Audit: Iteration 1

V13 improved P99 underprediction and hotspot localisation but worsened RMSE
and P95 error on every development case at full gain. This supplement does
not run a new symbolic search. It freezes the V13 residual expression and
evaluates

\\[
\\sigma(\\lambda)=\\sigma_{V11}+\\lambda s_{case}(r_{V13}-\\bar r_{V13,case}).
\\]

The gain grid is selected on all elements of the 15 validation cases. The
selected primary and optional research gains are then frozen and checked on
the internal-development split and all 149 development cases. The 50 final
case element files remain sealed.
"""
        ),
        markdown("## 1. Imports and package root"),
        code(
            """
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
        if (candidate / "src" / "v13_residual_gain_audit.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from v13_residual_gain_audit import (
    V13ResidualGainAuditConfig,
    output_directory,
    preflight_v13_residual_gain_audit,
    run_v13_residual_gain_audit,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown("## 2. Locked audit configuration"),
        code(
            """
RUN_GAIN_AUDIT = True
CONFIG = V13ResidualGainAuditConfig()
OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)

display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "value"}))
print("Validation gain grid:", CONFIG.gain_grid.tolist())
print("Output directory:", OUTPUT_DIR)
"""
        ),
        markdown(
            """
## 3. Frozen-input preflight

This hashes the exact V13 candidate and verifies the parent V13 decision,
candidate index, 149-case development scope and untouched 50-case final seal.
"""
        ),
        code(
            """
PREFLIGHT = preflight_v13_residual_gain_audit(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
print("Frozen artifacts hashed:", len(PREFLIGHT["hash_audit"]))
print("Frozen V13 candidate:", int(PREFLIGHT["selected"]["candidate_index"]))
print("Final-test cases inventoried, not read:", len(PREFLIGHT["inputs"]["final_ids"]))
"""
        ),
        markdown(
            """
## 4. Run the complete-case gain audit

The validation grid is evaluated first. Only after the gains are frozen are
the train and internal-development cases evaluated. Per-case CSV caches allow
the audit to resume after interruption without re-reading completed cases.
"""
        ),
        code(
            """
RESULT = None
if RUN_GAIN_AUDIT:
    RESULT = run_v13_residual_gain_audit(PACKAGE_ROOT, CONFIG)
    display(pd.DataFrame([RESULT]))
else:
    print("Gain audit skipped because RUN_GAIN_AUDIT=False")
"""
        ),
        markdown("## 5. Saved decision evidence"),
        code(
            """
artifacts = {
    "completion": OUTPUT_DIR / "gain_audit_complete.json",
    "decision": OUTPUT_DIR / "residual_gain_decision.json",
    "validation_selection": OUTPUT_DIR / "validation_gain_selection.json",
    "analytic_gain": OUTPUT_DIR / "validation_analytic_gain_summary.json",
    "validation_grid": OUTPUT_DIR / "validation_gain_grid_metrics.csv",
    "scope_metrics": OUTPUT_DIR / "selected_gain_scope_metrics.csv",
    "gates": OUTPUT_DIR / "gain_confirmation_gates.csv",
    "formula": OUTPUT_DIR / "selected_gain_formula.txt",
}
for name, path in artifacts.items():
    print(f"{name:22s} exists={path.exists()}  {path}")

if artifacts["decision"].exists():
    print(json.dumps(json.loads(artifacts["decision"].read_text()), indent=2))
if artifacts["analytic_gain"].exists():
    print(json.dumps(json.loads(artifacts["analytic_gain"].read_text()), indent=2))
if artifacts["scope_metrics"].exists():
    display(pd.read_csv(artifacts["scope_metrics"]))
if artifacts["gates"].exists():
    display(pd.read_csv(artifacts["gates"]))
if artifacts["formula"].exists():
    print("\\n" + artifacts["formula"].read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## Decision boundary

A positive residual gain may replace V11 only if it passes the strict primary
validation and confirmation gates. A smaller gain may be retained as an
optional tail-risk research correction if it limits RMSE/P95 cost while
materially reducing P99 underprediction. Otherwise V11 remains the sole
deployable stress formula and V13 is reported as a non-deployed correction
study. No outcome in this notebook opens the final 50-case set.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
