#!/usr/bin/env bash
# Bootstrap a Python 3.10 venv named `recsys26-lora` and install pinned deps.
# Usage: ./setup_venv.sh [VENV_DIR]
set -euo pipefail

VENV_DIR="${1:-recsys26-lora}"
PY="${PYTHON:-python3.10}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: $PY not found on PATH. Install Python 3.10 first."
    echo "macOS:   brew install python@3.10"
    echo "Ubuntu:  sudo apt install python3.10 python3.10-venv"
    exit 1
fi

echo "==> Creating venv at $VENV_DIR using $($PY --version)"
"$PY" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Bootstrapping pip"
python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing requirements"
python -m pip install -r requirements.txt

echo "==> Verifying install"
python - <<'PY'
import torch, transformers, datasets, peft, trl, accelerate, bm25s
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"peft         {peft.__version__}")
print(f"trl          {trl.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"bm25s        {bm25s.__version__}")
print(f"MPS avail:   {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
print(f"CUDA avail:  {torch.cuda.is_available()}")
PY

echo
echo "==> Done. Activate with: source $VENV_DIR/bin/activate"
