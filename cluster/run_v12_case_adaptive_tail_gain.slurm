#!/bin/bash --login
#SBATCH --job-name=ct3-v12-gain
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-ct3-v12-gain-%j.out
#SBATCH --error=slurm-ct3-v12-gain-%j.err

set -euo pipefail

PACKAGE_ROOT="${CT3_PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
NOTEBOOK_REL="notebooks/13_V12_Case_Adaptive_Tail_Gain/01_V12_Formal_Case_Adaptive_Tail_Gain.ipynb"
NOTEBOOK_PATH="$PACKAGE_ROOT/$NOTEBOOK_REL"
OUTPUT_DIR="$PACKAGE_ROOT/outputs/13_v12_case_adaptive_tail_gain/iteration_1"
EXECUTED_DIR="$OUTPUT_DIR/executed_notebooks"
VENV_PYTHON="$PACKAGE_ROOT/.venv/bin/python"
CASE_DIR="${CT3_CASE_DIR:-$PACKAGE_ROOT/FE_Results_Cases_All}"

echo "CT3 V12 formal case-adaptive tail-gain calibration"
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
    echo "ERROR: V12 notebook is missing: $NOTEBOOK_PATH" >&2
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

for REQUIRED_FILE in \
    "$PACKAGE_ROOT/shared/frozen_case_split_manifest_199cases.csv" \
    "$PACKAGE_ROOT/shared/frozen_similarity_group_map_199cases.csv" \
    "$PACKAGE_ROOT/outputs/00_qc_sensitivity_ablation/development_case_summary.csv" \
    "$PACKAGE_ROOT/outputs/11_tail_aware_localised_symbolic/iteration_1/pilot_complete.json" \
    "$PACKAGE_ROOT/outputs/12_v11_gain_stability_audit/iteration_1/audit_complete.json" \
    "$PACKAGE_ROOT/outputs/12_v11_gain_stability_audit/iteration_1/selected_gain.json" \
    "$PACKAGE_ROOT/outputs/12_v11_gain_stability_audit/iteration_1/selected_gain_formula.csv" \
    "$PACKAGE_ROOT/src/v12_case_adaptive_tail_gain.py"; do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "ERROR: required V12 input is missing: $REQUIRED_FILE" >&2
        exit 15
    fi
done

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
python scripts/validate_v12_case_adaptive_tail_gain.py

EXECUTED_NAME="v12_case_adaptive_gain_executed_job_${SLURM_JOB_ID}.ipynb"
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

if [[ ! -f "$OUTPUT_DIR/v12_complete.json" ]]; then
    echo "ERROR: notebook exited without v12_complete.json" >&2
    exit 16
fi

echo "Completed: $(date --iso-8601=seconds)"
echo "Decision: $OUTPUT_DIR/v12_promotion_decision.json"
echo "Promotion gates: $OUTPUT_DIR/v12_promotion_gates.csv"
echo "Selected formula: $OUTPUT_DIR/selected_deployment_formula.txt"
echo "Completion marker: $OUTPUT_DIR/v12_complete.json"
