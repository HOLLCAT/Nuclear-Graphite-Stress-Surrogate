#!/usr/bin/env python3
"""Static and data preflight for the V10 signed staged residual package."""

from pathlib import Path
import json
import os
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PACKAGE_ROOT / "src"
CACHE_ROOT = PACKAGE_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

print("[1/3] Importing validation modules...", flush=True)
from signed_staged_residual_symbolic import (
    SignedStagedPilotConfig,
    preflight_signed_staged_pilot,
)
print("[1/3] Validation modules imported.", flush=True)


def main() -> None:
    started = time.perf_counter()
    notebook = (
        PACKAGE_ROOT
        / "notebooks"
        / "10_Signed_Staged_Residual_Symbolic_Regression_Pilot"
        / "01_Signed_Staged_Residual_Symbolic_Regression_Pilot.ipynb"
    )
    slurm = PACKAGE_ROOT / "cluster" / "run_signed_staged_residual_pilot.slurm"
    if not notebook.exists():
        raise FileNotFoundError(notebook)
    if not slurm.exists():
        raise FileNotFoundError(slurm)
    parsed = json.loads(notebook.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4:
        raise ValueError("V10 notebook is not nbformat 4")
    print("[2/3] Notebook and Slurm files validated.", flush=True)
    print(
        "[3/3] Inventorying 199 case filenames and verifying split/baseline hashes...",
        flush=True,
    )
    config = SignedStagedPilotConfig()
    result = preflight_signed_staged_pilot(PACKAGE_ROOT, config)
    print(result["checks"].to_string(index=False))
    print("Frozen baseline hashes verified:", result["baseline"]["hash_audit"]["pass"].all())
    print(
        f"V10 signed staged residual package is ready "
        f"({time.perf_counter() - started:.1f} seconds)",
        flush=True,
    )


if __name__ == "__main__":
    main()
