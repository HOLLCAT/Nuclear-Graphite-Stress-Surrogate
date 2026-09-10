#!/bin/bash --login
#SBATCH --job-name=ct3-v11-r2
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ct3-v11-r2-%j.out
#SBATCH --error=slurm-ct3-v11-r2-%j.err

set -euo pipefail

ITERATION=2
PACKAGE_ROOT="${CT3_PACKAGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
NOTEBOOK_PATH="$PACKAGE_ROOT/notebooks/17_Formal_V11_Four_Rotation/02_Formal_V11_Iteration_2.ipynb"
OUTPUT_DIR="$PACKAGE_ROOT/outputs/17_v11_four_rotation_formal/iteration_2"
EXECUTED_DIR="$OUTPUT_DIR/executed_notebooks"
CASE_DIR="${CT3_CASE_DIR:-$PACKAGE_ROOT/FE_Results_Cases_All}"

echo "CT3 formal V11 development rotation $ITERATION"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"

cd "$PACKAGE_ROOT"
mkdir -p "$OUTPUT_DIR" "$EXECUTED_DIR"
test -x "$PACKAGE_ROOT/.venv/bin/python"
test -f "$NOTEBOOK_PATH"
test -d "$CASE_DIR"
test "$(find "$CASE_DIR" -maxdepth 1 -type f -name 'FE_Results_Case_*.txt' | wc -l)" -eq 199

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source "$PACKAGE_ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$PACKAGE_ROOT/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python --version
python -c "from pysr import PySRRegressor; print('PySR import successful')"
python scripts/validate_formal_v11_four_rotation_package.py

EXECUTED_NAME="formal_v11_iteration_${ITERATION}_job_${SLURM_JOB_ID}.ipynb"
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=notebookct3 \
  --output-dir="$EXECUTED_DIR" --output="$EXECUTED_NAME" "$NOTEBOOK_PATH"

test -f "$OUTPUT_DIR/pipeline_complete.json"
echo "Completed: $(date --iso-8601=seconds)"
echo "Completion marker: $OUTPUT_DIR/pipeline_complete.json"

