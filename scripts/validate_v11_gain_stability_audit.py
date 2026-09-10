#!/usr/bin/env python3
"""Fast static/package validation for the frozen V11 gain audit."""

from pathlib import Path
import ast
import json
import os
import sys

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "src"
CACHE_ROOT = PACKAGE_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ct3_common import build_paths, discover_case_files, load_frozen_manifest  # noqa: E402
from v11_gain_stability_audit import (  # noqa: E402
    V11GainAuditConfig,
    _v11_required_paths,
)


def main() -> None:
    print("[1/4] Checking source and notebook syntax...", flush=True)
    source = SOURCE_DIR / "v11_gain_stability_audit.py"
    builder = PACKAGE_ROOT / "scripts" / "build_v11_gain_stability_audit_notebook.py"
    notebook = (
        PACKAGE_ROOT
        / "notebooks"
        / "12_V11_Frozen_Gain_and_Stability_Audit"
        / "01_V11_Frozen_Gain_and_Stability_Audit.ipynb"
    )
    slurm = PACKAGE_ROOT / "cluster" / "run_v11_gain_stability_audit.slurm"
    runbook = PACKAGE_ROOT / "CSF3_V11_Gain_Audit_Runbook_CN.md"
    for path in [source, builder, notebook, slurm, runbook]:
        if not path.exists():
            raise FileNotFoundError(path)
    ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    ast.parse(builder.read_text(encoding="utf-8"), filename=str(builder))
    parsed = json.loads(notebook.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4:
        raise ValueError("Gain-audit notebook is not nbformat 4")
    for index, cell in enumerate(parsed.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{notebook}:cell-{index}")

    print("[2/4] Checking the 199-case inventory and frozen split...", flush=True)
    paths = build_paths(PACKAGE_ROOT)
    _, inventory = discover_case_files(paths.case_dir)
    manifest = load_frozen_manifest(paths.manifest_path, inventory)
    counts = manifest[manifest["iteration"].eq(1)]["split"].value_counts().to_dict()
    expected = {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}
    if counts != expected:
        raise ValueError(f"Unexpected Iteration-1 split counts: {counts}")

    print("[3/4] Checking frozen V11 artifacts...", flush=True)
    required = _v11_required_paths(PACKAGE_ROOT)
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V11 inputs:\n" + "\n".join(missing))
    completion = json.loads(required["v11_completion"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("final_test_cases_read") != 0:
        raise RuntimeError("V11 is incomplete or its final-test seal is not recorded")
    selected = pd.read_csv(required["v11_tail_formula"])
    if len(selected) != 1:
        raise ValueError("Expected one selected V11 tail formula")
    if int(selected.iloc[0]["candidate_index"]) != int(completion["selected_candidate_index"]):
        raise ValueError("V11 selected candidate does not match completion metadata")

    print("[4/4] Checking locked gain policy...", flush=True)
    config = V11GainAuditConfig()
    config.validate()
    if len(config.gain_grid) != 25:
        raise ValueError("Expected 25 audit gains from 0.00 to 1.20")
    print("Case files:", len(inventory))
    print("Iteration-1 split:", counts)
    print("Frozen V11 candidate:", int(selected.iloc[0]["candidate_index"]))
    print("Gain grid:", config.gain_grid.tolist())
    print("Maximum selectable gain:", config.max_selectable_gain)
    print("Fast package validation passed. Full SHA-256 preflight runs inside the notebook.")


if __name__ == "__main__":
    main()
