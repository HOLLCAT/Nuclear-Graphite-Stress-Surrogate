#!/usr/bin/env python3
"""Static and data preflight for V11 Iteration 1."""

from pathlib import Path
import ast
import json
import os
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "src"
CACHE_ROOT = PACKAGE_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

print("[1/3] Importing V11 validation modules...", flush=True)
from v11_tail_aware_localised_symbolic import (  # noqa: E402
    V11PilotConfig,
    preflight_v11_pilot,
)
print("[1/3] V11 modules imported.", flush=True)


def main() -> None:
    started = time.perf_counter()
    notebook = (
        PACKAGE_ROOT
        / "notebooks"
        / "11_Tail_Aware_Localised_Symbolic_Regression"
        / "01_V11_Tail_Aware_Localised_Iteration_1.ipynb"
    )
    slurm = PACKAGE_ROOT / "cluster" / "run_v11_iteration1_pilot.slurm"
    worker = PACKAGE_ROOT / "scripts" / "run_v11_tail_segment.py"
    for path in [notebook, slurm, worker]:
        if not path.exists():
            raise FileNotFoundError(path)
    parsed = json.loads(notebook.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4:
        raise ValueError("V11 notebook is not nbformat 4")
    for index, cell in enumerate(parsed.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"{notebook}:cell-{index}")
    print("[2/3] Notebook, worker and Slurm files validated.", flush=True)
    print("[3/3] Verifying V10 cache/formulas and frozen split...", flush=True)
    result = preflight_v11_pilot(PACKAGE_ROOT, V11PilotConfig())
    print(result["checks"].to_string(index=False))
    print("V10 input artifacts hashed:", len(result["hash_audit"]))
    print(
        f"V11 Iteration-1 package is ready "
        f"({time.perf_counter() - started:.1f} seconds)",
        flush=True,
    )


if __name__ == "__main__":
    main()
