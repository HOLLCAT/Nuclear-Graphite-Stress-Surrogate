#!/usr/bin/env python3
"""Build the V9 residual-shape pilot notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PACKAGE_ROOT
    / "notebooks"
    / "09_Residual_Shape_Symbolic_Regression_Pilot"
)
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_Residual_Shape_Symbolic_Regression_Pilot.ipynb"


def markdown(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def code(text: str):
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
# V9 Residual-Shape Symbolic Regression Pilot: Iteration 1

## Purpose

V8 established that a hierarchical surrogate is viable, but its single global
shape expression underfits local high-stress structure. V9 keeps the completed
V8 case-mean, positive case-scale and global-shape formulas fixed, then searches
for one additional symbolic formula for the unexplained normalised residual:

`stress = case_mean + exp(case_log_scale) * (fixed_global_shape + residual_shape)`

This is a development-only pilot. It uses the frozen 119 training cases, all
elements in the 15 validation cases for formula selection, and the 15 internal
test cases once after selection. The 50 final-test cases remain sealed.
"""
        ),
        markdown(
            """
## What Changed From V8

- Four dimensionless boundary proxies are added to the reviewed 27 predictors.
  They express relative radial/axial position and proximity to the nearest
  observed case boundary. They are labelled as proxies because named FEM
  surfaces have not been confirmed.
- Every training case still contributes 5,000 discovery elements, but the
  quotas are 2,500 below P90, 500 from P90-P95, 1,000 from P95-P99 and 1,000
  above P99.
- The optimisation loss mass is fixed at 50%, 10%, 20% and 20% across those
  tiers. Sampling frequency and loss importance are therefore explicit and
  separately auditable.
- A Gaussian localisation operator `exp(-x^2)` is added. `tanh` is deliberately
  not added in the same experiment so any improvement remains attributable.
- Candidate selection uses a validation-only engineering score across RMSE,
  P95/P99 error, underprediction and hotspot metrics. Complexity is only a late
  tie-breaker; the old "within 3% RMSE choose simplest" rule is not used.
- Before PySR, nonlinear residual diagnostics compare geometry/boundary,
  physical-field and all-feature models. These diagnose learnability only and
  do not enter the final formula.
"""
        ),
        markdown("## 1. Fresh Kernel and Imports"),
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
        if (candidate / "src" / "residual_shape_symbolic.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from residual_shape_symbolic import (
    ResidualShapePilotConfig,
    output_directory,
    preflight_residual_pilot,
    run_residual_shape_pilot,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown(
            """
## 2. Locked Pilot Configuration

The pilot budget is `500 x 8 = 4,000` population iterations, split into five
recoverable 100-iteration segments. Each segment writes a PySR checkpoint and
can resume after a stalled worker or interrupted notebook. A formal 16,000-step
residual search is justified only if this pilot improves validation behaviour.
"""
        ),
        code(
            """
RUN_RESIDUAL_PILOT = True

CONFIG = ResidualShapePilotConfig(
    iteration=1,
    output_subdir="iteration_1",
    rows_per_training_case=5_000,
    total_niterations=500,
    populations=8,
    segment_niterations=100,
    population_size=40,
    ncycles_per_iteration=100,
    batch_size=50_000,
    maxsize=28,
    maxdepth=10,
    julia_threads=8,
    no_activity_timeout_seconds=45 * 60,
    segment_wall_timeout_seconds=3 * 60 * 60,
    watchdog_poll_seconds=60,
    max_attempts_per_segment=3,
    max_candidates_for_full_validation=16,
    hgb_max_iter=100,
    force_rebuild_training_cache=False,
)

OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)
display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "value"}))
print("Output directory:", OUTPUT_DIR)
print("Recoverable segments:", CONFIG.n_segments)
print("Planned population iterations:", CONFIG.target_population_iterations)
"""
        ),
        markdown(
            """
## 3. Preflight

This verifies the frozen 119/15/15 development split, 50 sealed final cases,
199 available FEM files, completed V8 formula artifacts, 31 V9 predictors and
the dedicated residual worker. It does not read final-test element data.
"""
        ),
        code(
            """
PREFLIGHT = preflight_residual_pilot(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
print("Training cases:", len(PREFLIGHT["inputs"]["train_ids"]))
print("Validation cases:", len(PREFLIGHT["inputs"]["validation_ids"]))
print("Internal-test cases:", len(PREFLIGHT["inputs"]["internal_ids"]))
print("Sealed final-test cases:", len(PREFLIGHT["inputs"]["final_ids"]))
"""
        ),
        markdown(
            """
## 4. Run or Resume the Complete Pilot

The controller first builds/reuses the fixed training cache, runs the residual
learnability diagnostic, executes five recoverable PySR segments, validates the
candidate frontier on complete validation cases, selects one formula using
validation data only, and finally reports it once on the internal-test cases.

Rerunning this notebook reuses the cache and every completed segment. Detailed
worker logs are stored under `segments/segment_XX/attempt_Y/`.
"""
        ),
        code(
            """
RESULT = None
if RUN_RESIDUAL_PILOT:
    RESULT = run_residual_shape_pilot(PACKAGE_ROOT, CONFIG)
    display(pd.DataFrame([RESULT]))
else:
    print("Pilot skipped because RUN_RESIDUAL_PILOT=False.")
"""
        ),
        markdown("## 5. Saved Results and Decision Evidence"),
        code(
            """
progress_path = OUTPUT_DIR / "search_progress.json"
if progress_path.exists():
    display(pd.DataFrame([json.loads(progress_path.read_text(encoding="utf-8"))]))

attempt_path = OUTPUT_DIR / "segment_attempt_audit.csv"
if attempt_path.exists():
    display(pd.read_csv(attempt_path).tail(20))

artifacts = {
    "completion": OUTPUT_DIR / "pilot_complete.json",
    "formula_text": OUTPUT_DIR / "selected_corrected_composite_formula.txt",
    "split_metrics": OUTPUT_DIR / "selected_formula_split_metrics.csv",
    "candidate_metrics": OUTPUT_DIR / "residual_candidate_validation_metrics.csv",
    "learnability": OUTPUT_DIR / "residual_learnability_split_metrics.csv",
    "sampling_audit": OUTPUT_DIR / "training_cache" / "training_sample_audit.csv",
}
display(pd.DataFrame([
    {"artifact": name, "exists": path.exists(), "path": str(path)}
    for name, path in artifacts.items()
]))

if artifacts["completion"].exists():
    print(artifacts["formula_text"].read_text(encoding="utf-8"))
    display(pd.read_csv(artifacts["learnability"]))
    display(pd.read_csv(artifacts["split_metrics"]))
    candidates = pd.read_csv(artifacts["candidate_metrics"])
    display(candidates.sort_values("engineering_selection_score").head(16))
"""
        ),
        markdown(
            """
## Interpretation Boundary

This notebook tests whether a readable residual correction can improve the V8
stress surrogate, particularly at P95/P99 and hotspot locations. Passing all
pilot gates supports a longer residual search. Failing them is still a valid
result: it means the present inputs/search space cannot justify more symbolic
search without another modelling change.

The output is not yet a lifetime model. Lifetime conversion still requires a
professor-confirmed strength, damage or failure relationship. The sealed
50-case final test must remain unused until formula structure and constants are
frozen across the full development workflow.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
