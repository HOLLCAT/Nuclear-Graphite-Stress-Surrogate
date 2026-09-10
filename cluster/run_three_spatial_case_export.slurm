#!/bin/bash --login
#SBATCH --job-name=ct3-3case-export
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-ct3-3case-export-%j.out
#SBATCH --error=slurm-ct3-3case-export-%j.err

set -euo pipefail

PACKAGE_ROOT="${CT3_PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_DIR="$PACKAGE_ROOT/outputs/21_three_case_fem_vs_frozen_prediction/full_400360_element_exports"
VENV_PYTHON="$PACKAGE_ROOT/.venv/bin/python"
CASE_DIR="${CT3_CASE_DIR:-$PACKAGE_ROOT/FE_Results_Cases_All}"

echo "CT3 best/median/worst full FEM versus frozen-primary export"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Package root: $PACKAGE_ROOT"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-set}"

if [[ "${CT3_AUTHORISE_THREE_CASE_EXPORT:-}" != "YES" ]]; then
    echo "ERROR: explicit three-case export authorisation is absent." >&2
    echo "Submit with --export=ALL,CT3_AUTHORISE_THREE_CASE_EXPORT=YES" >&2
    exit 10
fi

cd "$PACKAGE_ROOT"
mkdir -p "$OUTPUT_DIR"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: virtual environment is missing: $VENV_PYTHON" >&2
    exit 11
fi
if [[ ! -d "$CASE_DIR" ]]; then
    echo "ERROR: FEM case directory is missing: $CASE_DIR" >&2
    exit 12
fi

CASE_COUNT=$(find "$CASE_DIR" -maxdepth 1 -type f -name 'FE_Results_Case_*.txt' | wc -l)
echo "Detected FEM case files: $CASE_COUNT"
if [[ "$CASE_COUNT" -ne 199 ]]; then
    echo "ERROR: expected 199 FEM case files but found $CASE_COUNT" >&2
    exit 13
fi

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source "$PACKAGE_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CT3_CASE_DIR="$CASE_DIR"
export CT3_AUTHORISE_THREE_CASE_EXPORT=YES

python --version
python -c "import numpy, pandas, scipy, sklearn, sympy; print('Scientific dependencies import successfully')"
python scripts/validate_three_spatial_case_export.py

if [[ -f "$OUTPUT_DIR/three_case_export_complete.json" ]]; then
    echo "Verified export already exists; no predictions were repeated."
    cat "$OUTPUT_DIR/three_case_export_complete.json"
    exit 0
fi

python src/export_three_spatial_comparison_cases.py --package-root "$PACKAGE_ROOT"

if [[ ! -f "$OUTPUT_DIR/three_case_export_complete.json" ]]; then
    echo "ERROR: export exited without three_case_export_complete.json" >&2
    exit 14
fi

echo "Completed: $(date --iso-8601=seconds)"
echo "Output directory: $OUTPUT_DIR"
echo "Metrics summary: $OUTPUT_DIR/three_case_primary_metrics_summary.csv"
echo "Manifest: $OUTPUT_DIR/three_case_export_manifest.json"
echo "Completion marker: $OUTPUT_DIR/three_case_export_complete.json"
