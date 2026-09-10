#!/bin/bash --login
#SBATCH --job-name=ct3-final-149
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ct3-final-149-%j.out
#SBATCH --error=slurm-ct3-final-149-%j.err

set -euo pipefail

PACKAGE_ROOT="${CT3_PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
NOTEBOOK_REL="notebooks/19_One_Time_Final_50Case_Locked_149/01_One_Time_Final_50Case_Locked_149.ipynb"
NOTEBOOK_PATH="$PACKAGE_ROOT/$NOTEBOOK_REL"
OUTPUT_DIR="$PACKAGE_ROOT/outputs/19_one_time_final_50case_locked_149/formal_final_50case"
EXECUTED_DIR="$OUTPUT_DIR/executed_notebooks"
VENV_PYTHON="$PACKAGE_ROOT/.venv/bin/python"
CASE_DIR="${CT3_CASE_DIR:-$PACKAGE_ROOT/FE_Results_Cases_All}"

echo "CT3 one-time final evaluation of the locked 149-case symbolic model"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Package root: $PACKAGE_ROOT"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-set}"

if [[ "${CT3_AUTHORISE_LOCKED_FINAL_TEST:-}" != "YES" ]]; then
    echo "ERROR: explicit one-time final authorisation is absent." >&2
    echo "Submit with --export=ALL,CT3_AUTHORISE_LOCKED_FINAL_TEST=YES" >&2
    exit 10
fi

cd "$PACKAGE_ROOT"
mkdir -p "$OUTPUT_DIR" "$EXECUTED_DIR"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: virtual environment is missing: $VENV_PYTHON" >&2
    exit 11
fi
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
    echo "ERROR: final-evaluation notebook is missing: $NOTEBOOK_PATH" >&2
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
export CT3_AUTHORISE_LOCKED_FINAL_TEST=YES
mkdir -p "$MPLCONFIGDIR"

python --version
python -c "import numpy, pandas, scipy, sklearn, sympy; print('Scientific dependencies import successfully')"
python scripts/validate_locked_149_final_test.py

if [[ -f "$OUTPUT_DIR/final_evaluation_complete.json" ]]; then
    echo "ERROR: the one-time final evaluation is already complete; refusing a second run" >&2
    exit 16
fi

EXECUTED_NAME="locked_149_final_executed_job_${SLURM_JOB_ID}.ipynb"
echo "Executing: $NOTEBOOK_PATH"
echo "Executed notebook: $EXECUTED_DIR/$EXECUTED_NAME"

python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=notebookct3 \
    --output-dir="$EXECUTED_DIR" \
    --output="$EXECUTED_NAME" \
    "$NOTEBOOK_PATH"

if [[ ! -f "$OUTPUT_DIR/final_evaluation_complete.json" ]]; then
    echo "ERROR: notebook exited without final_evaluation_complete.json" >&2
    exit 17
fi

echo "Completed: $(date --iso-8601=seconds)"
echo "Final summary: $OUTPUT_DIR/final_50case_scope_metrics.csv"
echo "Reporting gates: $OUTPUT_DIR/final_primary_research_reporting_gates.csv"
echo "Thesis figures: $OUTPUT_DIR/figures_for_thesis"
echo "Figure manifest: $OUTPUT_DIR/figure_manifest_for_thesis.csv"
echo "Completion marker: $OUTPUT_DIR/final_evaluation_complete.json"
