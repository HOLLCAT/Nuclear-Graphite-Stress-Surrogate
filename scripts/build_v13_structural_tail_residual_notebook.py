#!/usr/bin/env python3
"""Build the V13 structural tail-residual Iteration-1 notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks" / "15_V13_Structural_Tail_Residual"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_V13_Structural_Tail_Residual_Iteration_1.ipynb"


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
# V13 Structural Tail Residual: Iteration 1

## Research question

V12 demonstrated that a predictor-only case-adaptive gain nearly reaches the
non-deployable oracle gain, but still does not pass the all-development P99
underprediction gate. This rules out scalar gain calibration as the main
remaining bottleneck.

V13 therefore freezes the V11 global-gain-0.90 predictor and searches for one
additional predictor-only, zero-case-mean local residual formula. The target
is the signed complete-case V11 residual divided by the frozen V10 stress
scale. The residual is centered over every complete case so it cannot silently
replace the existing case-mean model.

Search discovery uses 5,000 deterministic rows per training case with stronger
P95/P99 exposure. Formula selection uses all 400,360 elements of each of the 15
validation cases. The selected candidate is evaluated on all elements of the
15 internal cases and all 119 training cases for a 149-case development audit.
The 50 final-test files are inventoried but never read.
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
        if (candidate / "src" / "v13_structural_tail_residual.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from v13_structural_tail_residual import (
    V13Config,
    output_directory,
    preflight_v13,
    run_v13_pilot,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown("## 2. Locked V13 configuration"),
        code(
            """
RUN_V13_PILOT = True

CONFIG = V13Config(
    iteration=1,
    output_subdir="iteration_1",
    rows_per_training_case=5_000,
    total_niterations=1_000,
    populations=8,
    segment_niterations=100,
    population_size=40,
    ncycles_per_iteration=100,
    batch_size=50_000,
    maxsize=36,
    maxdepth=11,
    julia_threads=8,
    no_activity_timeout_seconds=45 * 60,
    segment_wall_timeout_seconds=3 * 60 * 60,
    watchdog_poll_seconds=60,
    max_attempts_per_segment=3,
    max_candidates_for_full_validation=24,
    force_rebuild_training_cache=False,
    minimum_validation_rmse_improvement_fraction=0.005,
    maximum_p95_degradation_absolute=0.010,
    maximum_all149_p95_degradation_absolute=0.005,
    maximum_p99_relative_error=0.15,
    maximum_p99_underprediction=0.10,
    minimum_top1_hotspot_overlap=0.60,
    minimum_top1_recall_in_predicted_top5=0.80,
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
## 3. Frozen-input preflight

Preflight hashes the exact V11 and V12 artifacts, confirms the V12 decision to
retain V11 gain 0.90, verifies the frozen 119/15/15/50 manifest and records the
50 final case identifiers without opening their element files.
"""
        ),
        code(
            """
PREFLIGHT = preflight_v13(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
print("Frozen artifacts hashed:", len(PREFLIGHT["hash_audit"]))
print("Training cases:", len(PREFLIGHT["inputs"]["train_ids"]))
print("Validation cases:", len(PREFLIGHT["inputs"]["validation_ids"]))
print("Internal development cases:", len(PREFLIGHT["inputs"]["internal_ids"]))
print("Sealed final-test cases:", len(PREFLIGHT["inputs"]["final_ids"]))
"""
        ),
        markdown(
            """
## 4. Run or resume V13

Training-case sample preparation is cached one complete case at a time. The
PySR search is divided into ten checkpointed segments. Re-running this cell
reuses every valid case cache and completed segment. A watchdog stops inactive
workers and resumes from the last stable checkpoint.
"""
        ),
        code(
            """
RESULT = None
if RUN_V13_PILOT:
    RESULT = run_v13_pilot(PACKAGE_ROOT, CONFIG)
    display(pd.DataFrame([RESULT]))
else:
    print("V13 pilot skipped because RUN_V13_PILOT=False.")
"""
        ),
        markdown("## 5. Saved development evidence"),
        code(
            """
artifacts = {
    "completion": OUTPUT_DIR / "v13_complete.json",
    "decision": OUTPUT_DIR / "v13_promotion_decision.json",
    "gates": OUTPUT_DIR / "v13_promotion_gates.csv",
    "scope_metrics": OUTPUT_DIR / "v13_scope_metrics.csv",
    "case_registry": OUTPUT_DIR / "v13_case_failure_registry.csv",
    "candidate_formula": OUTPUT_DIR / "v13_candidate_formula.txt",
    "selected_formula": OUTPUT_DIR / "selected_deployment_formula.txt",
    "candidate_metrics": OUTPUT_DIR / "stages" / "stage_structural_residual" / "candidate_validation_metrics.csv",
    "progress": OUTPUT_DIR / "stages" / "stage_structural_residual" / "search_progress.json",
}
for name, path in artifacts.items():
    print(f"{name:20s} exists={path.exists()}  {path}")

if artifacts["scope_metrics"].exists():
    display(pd.read_csv(artifacts["scope_metrics"]))
if artifacts["gates"].exists():
    display(pd.read_csv(artifacts["gates"]))
if artifacts["decision"].exists():
    print(json.dumps(json.loads(artifacts["decision"].read_text()), indent=2))
if artifacts["selected_formula"].exists():
    print("\\n" + artifacts["selected_formula"].read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## Interpretation boundary

V13 is promoted only if its validation candidate passes every locked gate and
the same frozen candidate is then confirmed on the internal-development and
all-149 scopes. A failed gate automatically retains V11 gain 0.90. Neither a
diagnostic V13 formula nor a small isolated metric improvement is sufficient
to open the final-test set.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
