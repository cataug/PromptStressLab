#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

mkdir -p \
    "$ROOT/config" \
    "$ROOT/scripts" \
    "$ROOT/manifests" \
    "$ROOT/logs/experiments" \
    "$ROOT/outputs/generations" \
    "$ROOT/outputs/metrics"

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m py_compile \
    "$ROOT/scripts/psl_common.py" \
    "$ROOT/scripts/build_experiment_manifest.py" \
    "$ROOT/scripts/run_model_experiment.py" \
    "$ROOT/scripts/gpu_scheduler.py" \
    "$ROOT/scripts/evaluate_experiment.py" \
    "$ROOT/scripts/status.py"

COMPILE_STATUS="$?"

echo "[PY_COMPILE_STATUS] $COMPILE_STATUS"

if [ "$COMPILE_STATUS" -eq 0 ]; then
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/build_experiment_manifest.py" \
        --root "$ROOT" \
        2>&1 | tee "$ROOT/logs/build_experiment_manifest.log"
    BUILD_STATUS="${PIPESTATUS[0]}"
    echo "[MANIFEST_BUILD_STATUS] $BUILD_STATUS"
else
    echo "[MANIFEST_BUILD_SKIPPED] Python compilation failed"
fi

echo
cat "$ROOT/manifests/experiment_summary.json" 2>/dev/null
