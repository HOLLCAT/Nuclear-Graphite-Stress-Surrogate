#!/usr/bin/env python3
"""Build the formal V12 bounded case-adaptive tail-gain notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks" / "13_V12_Case_Adaptive_Tail_Gain"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_V12_Formal_Case_Adaptive_Tail_Gain.ipynb"


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
# V12 Formal Case-Adaptive Tail-Gain Calibration

## Why V12 is needed

The frozen V11 formula uses one global tail gain, `lambda = 0.90`. Its audit
showed that different cases can fail in opposite directions: increasing one
global gain may reduce P99 underprediction in one case while increasing the
maximum-stress overshoot in another. V12 tests whether correction strength can
instead be predicted from the input fields available at deployment.

The frozen stress structure is unchanged:

`sigma = sigma_V10 + delta_mu_case + lambda_case * scale_V10 * centered_tail_V11`

Only `lambda_case` is modelled. It is bounded to `[0, 1.10]` and may use only
case-level summaries of coordinates, FluenceRate, Temperature,
WeightLossRate, and the already-frozen predictor-only formula components.

## Statistical contract

- The 50 final-test case files remain sealed and are not read.
- All 149 non-final cases are development data because the former 15-case
  internal set was already inspected while V11 was designed.
- Similarity groups remain together in five outer and four inner folds.
- Each development case receives one nested out-of-fold gain prediction.
- These are **gain-layer OOF** results, not full-pipeline OOF results, because
  the frozen V11 base formula was originally developed on 119 cases.
- P95 and P99 are response-distribution evaluation metrics, not confidence
  levels, clipping boundaries, or deployment inputs.
- If any formal promotion gate fails, the saved deployment formula falls back
  automatically to the audited V11 gain `0.90`.
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
        if (candidate / "src" / "v12_case_adaptive_tail_gain.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from v12_case_adaptive_tail_gain import (
    V12Config,
    candidate_configurations,
    output_directory,
    preflight_v12,
    run_v12,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown("## 2. Locked formal configuration"),
        code(
            """
RUN_V12 = True

CONFIG = V12Config(
    iteration=1,
    output_subdir="iteration_1",
    global_reference_gain=0.90,
    oracle_gain_min=0.00,
    oracle_gain_max=1.10,
    oracle_gain_step=0.025,
    oracle_near_optimal_tolerance=0.005,
    oracle_huber_delta=1.0,
    oracle_tier_mass=(0.55, 0.15, 0.20, 0.10),
    outer_folds=5,
    inner_folds=4,
    ridge_alphas=(0.1, 1.0, 10.0, 100.0),
    huber_alphas=(0.001, 0.01, 0.1),
    huber_epsilon=1.35,
    shrinkage_values=(0.50, 0.75, 1.00),
    minimum_deployable_gain=0.00,
    maximum_deployable_gain=1.10,
    bootstrap_resamples=10_000,
    random_seed=42,
)

OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)
display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "value"}))
display(candidate_configurations(CONFIG))
print("Output directory:", OUTPUT_DIR)
"""
        ),
        markdown(
            """
## 3. Frozen provenance and final-test seal

This cell verifies the exact V11 formula, V11 gain audit, 199-case manifest,
149-case development summary, similarity groups and V12 source hash. It checks
that the parent run read zero final-test cases and refuses to reuse an output
directory whose signature differs.
"""
        ),
        code(
            """
PREFLIGHT = preflight_v12(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
display(PREFLIGHT["hash_audit"])
print("Development cases:", len(PREFLIGHT["development_ids"]))
print("Sealed final cases:", len(PREFLIGHT["inputs"]["final_ids"]))
print("V12 signature:", PREFLIGHT["signature"]["v12_signature_sha256"])
"""
        ),
        markdown(
            """
## 4. Run or resume V12

The run is recoverable at case level. First, every development case is scanned
over 45 gains to define a tail-aware Huber oracle target. Then the bounded
Ridge/Huber gain layer is selected by nested similarity-group validation.
Finally, each OOF gain is checked against all 400,360 elements in its case.

Oracle gains use stress only as supervised development targets. The deployable
gain formula never receives stress or a stress-derived summary.
"""
        ),
        code(
            """
RESULT = None
if RUN_V12:
    RESULT = run_v12(PACKAGE_ROOT, CONFIG, preflight=PREFLIGHT)
    display(pd.DataFrame([RESULT]).T.rename(columns={0: "value"}))
else:
    print("V12 skipped because RUN_V12=False.")
"""
        ),
        markdown("## 5. Saved evidence and formal decision"),
        code(
            """
artifacts = {
    "completion": OUTPUT_DIR / "v12_complete.json",
    "decision": OUTPUT_DIR / "v12_promotion_decision.json",
    "gates": OUTPUT_DIR / "v12_promotion_gates.csv",
    "scope_metrics": OUTPUT_DIR / "v12_comparison_scope_metrics.csv",
    "oof_gains": OUTPUT_DIR / "gain_layer_nested_oof_predictions.csv",
    "outer_choices": OUTPUT_DIR / "nested_outer_selected_candidates.csv",
    "refit_model": OUTPUT_DIR / "gain_model_refit_on_149.json",
    "selected_formula": OUTPUT_DIR / "selected_deployment_formula.txt",
    "adaptive_candidate": OUTPUT_DIR / "v12_candidate_adaptive_formula.txt",
}
for name, path in artifacts.items():
    print(f"{name:20s} exists={path.exists()}  {path}")

if artifacts["decision"].exists():
    print("\\nPromotion decision")
    print(artifacts["decision"].read_text(encoding="utf-8"))
if artifacts["gates"].exists():
    display(pd.read_csv(artifacts["gates"]))
if artifacts["scope_metrics"].exists():
    display(pd.read_csv(artifacts["scope_metrics"]))
if artifacts["outer_choices"].exists():
    display(pd.read_csv(artifacts["outer_choices"]))
if artifacts["selected_formula"].exists():
    print("\\n" + artifacts["selected_formula"].read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## 6. How to interpret the result

Three models are reported over identical complete development cases:

1. `v11_global_gain_0_90`: the frozen V11 reference.
2. `v12_adaptive_gain_nested_oof`: each case's gain is predicted without using
   that case or any case in its similarity group for gain-layer fitting.
3. `v12_oracle_gain_non_deployable`: the best near-optimal grid gain found
   using the true case stress. This is an upper-bound diagnostic only.

If the oracle does not improve a case, gain adaptation cannot repair that
case's spatial shape and a new residual formula would be required. If the
oracle improves it but nested OOF does not, the available predictor context is
insufficient to generalise the required gain. Promotion requires improvement
in macro RMSE without degrading P99 underprediction, hotspot overlap or the
maximum-prediction guardrail. Failure leaves the selected formula at V11's
global gain `0.90`; it is a valid negative result, not a failed run.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
