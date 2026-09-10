#!/usr/bin/env python3
"""Build the V11 Iteration-1 tail-aware localised residual notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PACKAGE_ROOT
    / "notebooks"
    / "11_Tail_Aware_Localised_Symbolic_Regression"
)
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_V11_Tail_Aware_Localised_Iteration_1.ipynb"


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
# V11 Tail-Aware Localised Symbolic Regression: Iteration 1

## Purpose

V10 improved complete-case RMSE and P95/P99 prediction, but its selected
validation result still missed two engineering targets: P99 underprediction
was about 10.95% and top-1% hotspot overlap was about 0.518. It also allowed
the local residual formula to shift a case mean.

V11 keeps the completed V10 model fixed and adds two auditable corrections:

1. a small case-level ridge formula for the mean V10 residual; and
2. one PySR formula for the remaining local residual.

The PySR target is centered separately inside every training case. The same
predictor-only centering is applied on a complete case at deployment, so the
local formula cannot silently replace the case-mean model. Tail exposure is
55/15/20/10% for below-P90, P90-P95, P95-P99 and top-P99, and positive tail
residuals receive a further 1.75x weight.

Six smooth hotspot RBF features are fitted from training-case top-P99
locations. They are explicit analytic functions of case-relative rho, z and
theta, not a hidden tree model. Complete validation and internal-test cases
are evaluated; the 50 final-test cases stay sealed.
"""
        ),
        markdown("## 1. Imports and Package Root"),
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
        if (candidate / "src" / "v11_tail_aware_localised_symbolic.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from v11_tail_aware_localised_symbolic import (
    V11PilotConfig,
    output_directory,
    preflight_v11_pilot,
    run_v11_pilot,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown("## 2. Locked Iteration-1 Configuration"),
        code(
            """
RUN_V11_PILOT = True

CONFIG = V11PilotConfig(
    iteration=1,
    output_subdir="iteration_1",
    rows_per_training_case=5_000,
    total_niterations=500,
    populations=8,
    segment_niterations=100,
    population_size=40,
    ncycles_per_iteration=100,
    batch_size=50_000,
    maxsize=32,
    maxdepth=10,
    julia_threads=8,
    no_activity_timeout_seconds=45 * 60,
    segment_wall_timeout_seconds=3 * 60 * 60,
    watchdog_poll_seconds=60,
    max_attempts_per_segment=3,
    max_candidates_for_full_validation=20,
    force_rebuild_training_cache=False,
)

OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)
display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "value"}))
print("Output directory:", OUTPUT_DIR)
print("Recoverable segments:", CONFIG.n_segments)
print("Population iterations:", CONFIG.population_iterations_total)
"""
        ),
        markdown(
            """
## 3. Preflight

Preflight confirms the 199 FEM files and frozen 119/15/15/50 split, verifies
the completed V10 formulas and its exact 595,000-row cache by SHA-256, and
checks the dedicated V11 worker. It inventories final-test case names but does
not read their element data.
"""
        ),
        code(
            """
PREFLIGHT = preflight_v11_pilot(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
display(PREFLIGHT["hash_audit"])
print("Training cases:", len(PREFLIGHT["inputs"]["train_ids"]))
print("Validation cases:", len(PREFLIGHT["inputs"]["validation_ids"]))
print("Internal-test cases:", len(PREFLIGHT["inputs"]["internal_ids"]))
print("Sealed final-test cases:", len(PREFLIGHT["inputs"]["final_ids"]))
"""
        ),
        markdown(
            """
## 4. Run or Resume V11

The one symbolic stage is divided into five checkpointed segments. Re-running
this cell reuses the V11 cache and every completed segment. If a worker stops
producing file activity, the watchdog terminates it, rolls back to the last
stable checkpoint and retries without discarding earlier segments.
"""
        ),
        code(
            """
RESULT = None
if RUN_V11_PILOT:
    RESULT = run_v11_pilot(PACKAGE_ROOT, CONFIG)
    display(pd.DataFrame([RESULT]))
else:
    print("V11 pilot skipped because RUN_V11_PILOT=False.")
"""
        ),
        markdown("## 5. Saved Evidence"),
        code(
            """
artifacts = {
    "completion": OUTPUT_DIR / "pilot_complete.json",
    "formula": OUTPUT_DIR / "selected_composite_formula.txt",
    "split_metrics": OUTPUT_DIR / "selected_models_split_metrics.csv",
    "tail_adaptation": OUTPUT_DIR / "selected_models_tail_adaptation.csv",
    "candidate_metrics": OUTPUT_DIR / "stages" / "stage_tail_localised" / "candidate_validation_metrics.csv",
    "mean_model": OUTPUT_DIR / "training_cache" / "mean_calibration_model.json",
    "rbf_centres": OUTPUT_DIR / "training_cache" / "hotspot_rbf_centres.csv",
    "progress": OUTPUT_DIR / "stages" / "stage_tail_localised" / "search_progress.json",
}
for name, path in artifacts.items():
    print(f"{name:18s} exists={path.exists()}  {path}")

if artifacts["split_metrics"].exists():
    display(pd.read_csv(artifacts["split_metrics"]))
if artifacts["tail_adaptation"].exists():
    display(pd.read_csv(artifacts["tail_adaptation"]))
if artifacts["formula"].exists():
    print("\\n" + artifacts["formula"].read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## Interpretation Rule

This notebook is a development pilot, not final-test evidence. A candidate is
promoted only if validation simultaneously meets numerical stability, positive
R2, at least 1% RMSE improvement over V10, P95/P99 relative-error limits,
P99 underprediction at or below 10%, top-1% hotspot overlap at or above 0.60,
and top-1 recall in predicted top-5% at or above 0.80. If no candidate passes,
the best finite result is saved as diagnostic evidence without being labelled
an engineering-ready replacement.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
