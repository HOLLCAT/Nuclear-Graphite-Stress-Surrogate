#!/usr/bin/env python3
"""Build the V10 signed staged residual symbolic-regression pilot notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PACKAGE_ROOT
    / "notebooks"
    / "10_Signed_Staged_Residual_Symbolic_Regression_Pilot"
)
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_Signed_Staged_Residual_Symbolic_Regression_Pilot.ipynb"


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
# V10 Signed Staged Residual Symbolic Regression Pilot

## Purpose

V10 freezes the locally verified V8 Candidate-5 hierarchy and searches two
signed residual formulas:

`stress = mean + exp(log_scale) * (fixed_shape + geometry_residual + physical_residual)`

The geometry stage learns directional radial, axial and angular boundary
behaviour. The physical stage then learns the remaining local dependence on
fluence rate, temperature and weight-loss rate. Both stages are recoverable.

This is a development-only pilot. It uses 119 training cases, complete 15-case
validation, and one 15-case internal-test report. The 50 final-test cases stay
sealed.
"""
        ),
        markdown(
            """
## Why This Version Differs From V9

- The V8 baseline is copied into `shared/` and verified by SHA-256, preventing
  the local Candidate-5 and remote Candidate-11 baselines from being mixed.
- The residual target is computed against deployable V8 predictions, not true
  stress summaries unavailable at deployment.
- Discovery exposure/loss mass is 60/15/20/5% for below-P90, P90-P95,
  P95-P99 and top-P99. V9 exposed top-P99 at 20%, which encouraged broad
  overprediction.
- `HuberLoss(1.0)` reduces domination by extreme residuals while preserving
  tail emphasis through explicit weights.
- `square` is removed because V9 needed negative as well as positive
  corrections. `tanh` and constrained Gaussian localisation remain available.
- Formula selection requires complete-case validation and records explicit
  RMSE, P95/P99, underprediction and hotspot gates. A failed gate remains a
  valid diagnostic result and is not promoted as an engineering formula.

The search design follows the official
[PySR options](https://ai.damtp.cam.ac.uk/pysr/options/),
[tuning guidance](https://ai.damtp.cam.ac.uk/pysr/v2.0.0a2/tuning/) and
[SymbolicRegression loss reference](https://ai.damtp.cam.ac.uk/symbolicregression/dev/losses/)
for batching, weighting, robust loss and constrained operators. Recoverability
uses ordinary PySR checkpoints rather than `TemplateExpressionSpec`, whose
checkpoint serialization has a reported compatibility issue in
[PySR issue 941](https://github.com/MilesCranmer/PySR/issues/941).
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
        if (candidate / "src" / "signed_staged_residual_symbolic.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from signed_staged_residual_symbolic import (
    SignedStagedPilotConfig,
    output_directory,
    preflight_signed_staged_pilot,
    run_signed_staged_residual_pilot,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown("## 2. Locked Pilot Configuration"),
        code(
            """
RUN_V10_PILOT = True

CONFIG = SignedStagedPilotConfig(
    iteration=1,
    output_subdir="iteration_1",
    rows_per_training_case=5_000,
    total_niterations_per_stage=500,
    populations=8,
    segment_niterations=100,
    population_size=40,
    ncycles_per_iteration=100,
    batch_size=50_000,
    stage_a_maxsize=32,
    stage_a_maxdepth=10,
    stage_b_maxsize=26,
    stage_b_maxdepth=9,
    julia_threads=8,
    no_activity_timeout_seconds=45 * 60,
    segment_wall_timeout_seconds=3 * 60 * 60,
    watchdog_poll_seconds=60,
    max_attempts_per_segment=3,
    max_candidates_for_full_validation=16,
    force_rebuild_training_cache=False,
)

OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)
display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "value"}))
print("Output directory:", OUTPUT_DIR)
print("Segments per stage:", CONFIG.n_segments_per_stage)
print("Population iterations per stage:", CONFIG.population_iterations_per_stage)
print("Two-stage total population iterations:", CONFIG.total_population_iterations)
"""
        ),
        markdown(
            """
## 3. Preflight

Preflight verifies 199 FEM files, the frozen 119/15/15/50 split, all baseline
hashes, 40 total predictors, 28 geometry-stage predictors, 29 physical-stage
predictors and the dedicated worker. It does not read final-test element data.
"""
        ),
        code(
            """
PREFLIGHT = preflight_signed_staged_pilot(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
display(PREFLIGHT["baseline"]["hash_audit"])
print("Training cases:", len(PREFLIGHT["inputs"]["train_ids"]))
print("Validation cases:", len(PREFLIGHT["inputs"]["validation_ids"]))
print("Internal-test cases:", len(PREFLIGHT["inputs"]["internal_ids"]))
print("Sealed final-test cases:", len(PREFLIGHT["inputs"]["final_ids"]))
"""
        ),
        markdown(
            """
## 4. Run or Resume Both Stages

Rerunning this cell reuses the sample cache and every completed segment. Stage
B begins only after Stage A has been fully validated and selected. Search logs,
watchdog states and checkpoint snapshots are stored separately under
`stages/stage_geometry/` and `stages/stage_physical/`.
"""
        ),
        code(
            """
RESULT = None
if RUN_V10_PILOT:
    RESULT = run_signed_staged_residual_pilot(PACKAGE_ROOT, CONFIG)
    display(pd.DataFrame([RESULT]))
else:
    print("V10 pilot skipped because RUN_V10_PILOT=False.")
"""
        ),
        markdown("## 5. Saved Evidence"),
        code(
            """
artifacts = {
    "completion": OUTPUT_DIR / "pilot_complete.json",
    "formula": OUTPUT_DIR / "selected_composite_formula.txt",
    "split_metrics": OUTPUT_DIR / "selected_models_split_metrics.csv",
    "tail_metrics": OUTPUT_DIR / "selected_models_tail_adaptation.csv",
    "geometry_candidates": OUTPUT_DIR / "stages" / "stage_geometry" / "candidate_validation_metrics.csv",
    "physical_candidates": OUTPUT_DIR / "stages" / "stage_physical" / "candidate_validation_metrics.csv",
    "sampling_audit": OUTPUT_DIR / "training_cache" / "training_sample_audit.csv",
}
display(pd.DataFrame([
    {"artifact": name, "exists": path.exists(), "path": str(path)}
    for name, path in artifacts.items()
]))

if artifacts["completion"].exists():
    print(artifacts["formula"].read_text(encoding="utf-8"))
    display(pd.read_csv(artifacts["split_metrics"]))
    display(pd.read_csv(artifacts["tail_metrics"]))
    display(
        pd.read_csv(artifacts["physical_candidates"])
        .sort_values("engineering_selection_score")
        .head(16)
    )
"""
        ),
        markdown(
            """
## Interpretation Boundary

V10 tests whether a readable two-stage correction can beat the frozen V8
baseline without sacrificing P95/P99 or hotspot localisation. Passing every
validation gate supports a larger formal search. Failure means that this
symbolic search space has not justified promotion; it does not authorize use
of the diagnostic formula as an engineering replacement for FEM.

This remains a stress surrogate, not a lifetime model. Lifetime conversion
still requires a professor-confirmed strength, damage or failure relationship.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
