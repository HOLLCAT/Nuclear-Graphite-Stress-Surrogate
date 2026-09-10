#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

# Import once on the login node so JuliaCall/PySR can initialise its Julia
# environment before a long scheduled job starts.
python -c "from pysr import PySRRegressor; print('PySR import successful')"
python -m ipykernel install --user --name notebookct3 --display-name "NotebookCT3"

echo "Environment ready at $PACKAGE_ROOT/.venv"

