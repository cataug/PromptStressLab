#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u \
    "$ROOT/scripts/evaluate_experiment.py" \
    --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/evaluate_experiment.log"
STATUS="${PIPESTATUS[0]}"
echo "[EVALUATION_STATUS] $STATUS"
cat "$ROOT/outputs/metrics/evaluation_summary.json" 2>/dev/null
