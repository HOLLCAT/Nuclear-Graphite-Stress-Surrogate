#!/bin/bash --login
#SBATCH --job-name=ct3-sr-i1
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-ct3-sr-i1-%j.out
#SBATCH --error=slurm-ct3-sr-i1-%j.err

set -euo pipefail

PACKAGE_ROOT="${CT3_PACKAGE_ROOT:-/scratch/${USER}/NotebookCT3}"
NOTEBOOK_REL="notebooks/08_Recoverable_16000_Single_Formula_Symbolic_Regression/01_Recoverable_16000_Single_Formula_Iteration_1.ipynb"
NOTEBOOK_PATH="$PACKAGE_ROOT/$NOTEBOOK_REL"
OUTPUT_DIR="$PACKAGE_ROOT/outputs/08_recoverable_16000_single_formula/iteration_1"
EXECUTED_DIR="$OUTPUT_DIR/executed_notebooks"
VENV_PYTHON="$PACKAGE_ROOT/.venv/bin/python"
CASE_DIR="$PACKAGE_ROOT/FE_Results_Cases_All"

echo "CT3 recoverable Iteration-1 job"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Package root: $PACKAGE_ROOT"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-set}"

if [[ ! -d "$PACKAGE_ROOT" ]]; then
    echo "ERROR: package root does not exist: $PACKAGE_ROOT" >&2
    exit 10
fi
cd "$PACKAGE_ROOT"

mkdir -p "$OUTPUT_DIR" "$EXECUTED_DIR"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: virtual environment is missing: $VENV_PYTHON" >&2
    echo "Run cluster/setup_environment.sh from the package root before submitting." >&2
    exit 11
fi
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
    echo "ERROR: notebook is missing: $NOTEBOOK_PATH" >&2
    exit 12
fi
if [[ ! -d "$CASE_DIR" ]]; then
    echo "ERROR: FEM case directory is missing: $CASE_DIR" >&2
    exit 13
fi

CASE_COUNT=$(find "$CASE_DIR" -maxdepth 1 -type f -name 'FE_Results_Case_*.txt' | wc -l)
echo "Detected FEM case files: $CASE_COUNT"
if [[ "$CASE_COUNT" -ne 199 ]]; then
    echo "ERROR: expected 199 FEM case files but found $CASE_COUNT." >&2
    exit 14
fi

for REQUIRED_FILE in \
    "$PACKAGE_ROOT/shared/frozen_case_split_manifest_199cases.csv" \
    "$PACKAGE_ROOT/outputs/00_qc_sensitivity_ablation/development_case_summary.csv" \
    "$PACKAGE_ROOT/outputs/06_hierarchical_symbolic_regression/iteration_1/selected_case_mean_formula.csv" \
    "$PACKAGE_ROOT/outputs/06_hierarchical_symbolic_regression/iteration_1/selected_case_log_scale_formula.csv" \
    "$PACKAGE_ROOT/outputs/06_hierarchical_symbolic_regression/iteration_1/shape_scaling.csv" \
    "$PACKAGE_ROOT/scripts/run_recoverable_shape_segment.py"; do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "ERROR: required input is missing: $REQUIRED_FILE" >&2
        exit 15
    fi
done

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source "$PACKAGE_ROOT/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# PySR uses Julia threads. Keep Python/BLAS libraries single-threaded to avoid
# oversubscribing the eight cores allocated to this job.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$PACKAGE_ROOT/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python --version
python -c "from pysr import PySRRegressor; print('PySR import successful')"
python -c "import numpy, pandas, scipy, sklearn, sympy; print('Scientific dependencies import successfully')"

EXECUTED_NAME="iteration_1_executed_job_${SLURM_JOB_ID}.ipynb"
echo "Executing: $NOTEBOOK_PATH"
echo "Executed notebook will be saved as: $EXECUTED_DIR/$EXECUTED_NAME"

python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=notebookct3 \
    --output-dir="$EXECUTED_DIR" \
    --output="$EXECUTED_NAME" \
    "$NOTEBOOK_PATH"

if [[ ! -f "$OUTPUT_DIR/round_complete.json" ]]; then
    echo "ERROR: notebook exited without producing round_complete.json" >&2
    exit 16
fi

echo "Completed: $(date --iso-8601=seconds)"
echo "Formula: $OUTPUT_DIR/selected_composite_formula.txt"
echo "Completion marker: $OUTPUT_DIR/round_complete.json"
