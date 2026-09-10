#!/usr/bin/env python3
"""Build the frozen V11 gain and stability audit notebook."""

from pathlib import Path
import json
import uuid


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PACKAGE_ROOT
    / "notebooks"
    / "12_V11_Frozen_Gain_and_Stability_Audit"
)
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_V11_Frozen_Gain_and_Stability_Audit.ipynb"


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
# V11 Supplement: Frozen Tail-Gain and Stability Audit

## Purpose

V11 improved overall and high-stress prediction, but its validation top-1%
hotspot overlap was approximately 0.5997, just below the provisional 0.60
gate. It also increased maximum-stress prediction in a small number of cases.

This supplement therefore **does not run another symbolic search**. It freezes
the V10 formula, V11 mean calibration, V11 tail formula, scaling and RBF
centres, and evaluates only one transparent scalar:

`sigma(lambda) = sigma_v10 + delta_mu_case + lambda * scale_v10 * centered_tail_v11`

- `lambda = 0` is the V11 mean-calibrated V10 reference.
- `lambda = 1` is the current V11 formula.
- `lambda = 0.00...1.00` may be selected using validation cases only.
- `lambda = 1.05...1.20` is diagnostic only and cannot be selected.

Every element in every validation and internal-test case is evaluated. Each
case is checkpointed separately. The 50 final-test case files remain sealed.
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
        if (candidate / "src" / "v11_gain_stability_audit.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NotebookCT3 package root.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from v11_gain_stability_audit import (
    V11GainAuditConfig,
    output_directory,
    preflight_v11_gain_audit,
    run_v11_gain_audit,
)

print("Package root:", PACKAGE_ROOT)
print("Python:", sys.executable)
"""
        ),
        markdown("## 2. Locked audit configuration"),
        code(
            """
RUN_GAIN_AUDIT = True

CONFIG = V11GainAuditConfig(
    iteration=1,
    output_subdir="iteration_1",
    gain_min=0.00,
    gain_max=1.20,
    gain_step=0.05,
    max_selectable_gain=1.00,
    bootstrap_resamples=20_000,
    random_seed=42,
)

OUTPUT_DIR = output_directory(PACKAGE_ROOT, CONFIG)
display(pd.DataFrame([CONFIG.__dict__]).T.rename(columns={0: "value"}))
print("Gain grid:", CONFIG.gain_grid.tolist())
print("Output directory:", OUTPUT_DIR)
"""
        ),
        markdown(
            """
## 3. Preflight and frozen provenance

This checks the existing 119/15/15/50 case split, confirms that V11 completed
without reading the final-test elements, loads the exact selected V11 formula,
and hashes all compact formula, scale, mean-model and RBF-centre artifacts.
The audit refuses to resume if any frozen input or configuration changes.
"""
        ),
        code(
            """
PREFLIGHT = preflight_v11_gain_audit(PACKAGE_ROOT, CONFIG)
display(PREFLIGHT["checks"])
display(PREFLIGHT["hash_audit"])
print("Validation cases:", len(PREFLIGHT["inputs"]["validation_ids"]))
print("Internal-test cases:", len(PREFLIGHT["inputs"]["internal_ids"]))
print("Sealed final-test cases:", len(PREFLIGHT["inputs"]["final_ids"]))
print("Audit signature:", PREFLIGHT["signature"]["audit_signature_sha256"])
"""
        ),
        markdown(
            """
## 4. Run or resume the audit

The validation scan is completed first and selects one gain without using the
internal-test set. Only after selection is frozen are `lambda=0`, `lambda=1`
and the selected gain evaluated on the internal-test cases. A completed case
is loaded from its small CSV checkpoint when this cell is re-run.
"""
        ),
        code(
            """
RESULT = None
if RUN_GAIN_AUDIT:
    RESULT = run_v11_gain_audit(
        PACKAGE_ROOT,
        CONFIG,
        preflight=PREFLIGHT,
    )
    display(pd.DataFrame([RESULT]))
else:
    print("Gain audit skipped because RUN_GAIN_AUDIT=False.")
"""
        ),
        markdown("## 5. Saved evidence and interpretation"),
        code(
            """
artifacts = {
    "completion": OUTPUT_DIR / "audit_complete.json",
    "selected_gain": OUTPUT_DIR / "selected_gain.json",
    "formula": OUTPUT_DIR / "selected_gain_formula.txt",
    "validation_scan": OUTPUT_DIR / "validation_gain_summary.csv",
    "split_metrics": OUTPUT_DIR / "selected_models_split_metrics.csv",
    "tail_adaptation": OUTPUT_DIR / "selected_models_tail_adaptation.csv",
    "paired_stability": OUTPUT_DIR / "paired_case_stability_summary.csv",
    "v11_reproduction": OUTPUT_DIR / "v11_reproduction_audit.csv",
    "worst_cases": OUTPUT_DIR / "worst_case_diagnostics.csv",
}
for name, path in artifacts.items():
    print(f"{name:18s} exists={path.exists()}  {path}")

if artifacts["selected_gain"].exists():
    print("\\nSelected gain record")
    print(artifacts["selected_gain"].read_text(encoding="utf-8"))
if artifacts["validation_scan"].exists():
    scan = pd.read_csv(artifacts["validation_scan"])
    display(scan[
        [
            "gain",
            "selectable_gain",
            "macro_rmse",
            "mean_p95_relative_error",
            "mean_p99_underprediction_fraction",
            "mean_top1pct_hotspot_overlap",
            "max_prediction_abs_max_ratio",
            "all_v11_promotion_gates_pass",
            "engineering_selection_score",
            "selected_gain",
        ]
    ])
if artifacts["split_metrics"].exists():
    display(pd.read_csv(artifacts["split_metrics"]))
if artifacts["paired_stability"].exists():
    display(pd.read_csv(artifacts["paired_stability"]))
if artifacts["formula"].exists():
    print("\\n" + artifacts["formula"].read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## Decision rule for V12

The selected gain remains a development result. Validation determines the
gain; internal test only checks whether its direction and worst-case behaviour
remain stable. The paired bootstrap intervals and Wilcoxon values are
descriptive because there are only 15 cases in each split.

The V12 design should be chosen only after examining:

1. whether any selectable gain passes all provisional V11 gates;
2. whether a gain below 1 reduces the maximum-stress overshoot without losing
   P99 and hotspot improvements;
3. whether gains above 1 improve hotspot overlap only by worsening maximum or
   P95 behaviour; and
4. whether the same trade-off is visible on the untouched internal-test cases.

No result in this notebook is final-test evidence.
"""
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build())
