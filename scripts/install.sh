#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY="$PWD/.venv/Scripts/python.exe"
"$PY" -m pip install --no-input "torch" --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-input -r requirements.txt
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
