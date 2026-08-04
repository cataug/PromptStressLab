#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u "$ROOT/scripts/status.py"
