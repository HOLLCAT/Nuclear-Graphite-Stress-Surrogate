#!/bin/bash --login
#SBATCH --job-name=ct3-v13-gain
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-ct3-v13-gain-%j.out
#SBATCH --error=slurm-ct3-v13-gain-%j.err

set -euo pipefail

PACKAGE_ROOT="${CT3_PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
NOTEBOOK_REL="notebooks/16_V13_Residual_Gain_Audit/01_V13_Residual_Gain_Audit_Iteration_1.ipynb"
NOTEBOOK_PATH="$PACKAGE_ROOT/$NOTEBOOK_REL"
OUTPUT_DIR="$PACKAGE_ROOT/outputs/16_v13_residual_gain_audit/iteration_1"
EXECUTED_DIR="$OUTPUT_DIR/executed_notebooks"
VENV_PYTHON="$PACKAGE_ROOT/.venv/bin/python"
CASE_DIR="${CT3_CASE_DIR:-$PACKAGE_ROOT/FE_Results_Cases_All}"

echo "CT3 frozen V13 residual-gain audit"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Package root: $PACKAGE_ROOT"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-set}"

cd "$PACKAGE_ROOT"
mkdir -p "$OUTPUT_DIR" "$EXECUTED_DIR"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: virtual environment is missing: $VENV_PYTHON" >&2
    exit 11
fi
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
    echo "ERROR: audit notebook is missing: $NOTEBOOK_PATH" >&2
    exit 12
fi
if [[ ! -d "$CASE_DIR" ]]; then
    echo "ERROR: FEM case directory is missing: $CASE_DIR" >&2
    exit 13
fi

CASE_COUNT=$(find "$CASE_DIR" -maxdepth 1 -type f -name 'FE_Results_Case_*.txt' | wc -l)
echo "Detected FEM case files: $CASE_COUNT"
if [[ "$CASE_COUNT" -ne 199 ]]; then
    echo "ERROR: expected 199 FEM case files but found $CASE_COUNT" >&2
    exit 14
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
export MPLCONFIGDIR="$PACKAGE_ROOT/.cache/matplotlib"
export CT3_CASE_DIR="$CASE_DIR"
mkdir -p "$MPLCONFIGDIR"

python --version
python -c "import numpy, pandas, scipy, sklearn, sympy; print('Scientific dependencies import successfully')"
python scripts/validate_v13_residual_gain_audit.py

EXECUTED_NAME="v13_residual_gain_audit_executed_job_${SLURM_JOB_ID}.ipynb"
echo "Executing: $NOTEBOOK_PATH"

python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=notebookct3 \
    --output-dir="$EXECUTED_DIR" \
    --output="$EXECUTED_NAME" \
    "$NOTEBOOK_PATH"

if [[ ! -f "$OUTPUT_DIR/gain_audit_complete.json" ]]; then
    echo "ERROR: audit exited without gain_audit_complete.json" >&2
    exit 15
fi

echo "Completed: $(date --iso-8601=seconds)"
echo "Decision: $OUTPUT_DIR/residual_gain_decision.json"
echo "Scope metrics: $OUTPUT_DIR/selected_gain_scope_metrics.csv"
echo "Gates: $OUTPUT_DIR/gain_confirmation_gates.csv"
echo "Formula: $OUTPUT_DIR/selected_gain_formula.txt"

