#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export HF_HOME="/home/tahiti/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

export PSL_GPU_IDS="${PSL_GPU_IDS:-0}"
export PSL_MAX_PARALLEL_MODELS="${PSL_MAX_PARALLEL_MODELS:-1}"
export PSL_POLL_SECONDS="${PSL_POLL_SECONDS:-30}"
export PSL_MAX_START_UTILIZATION="${PSL_MAX_START_UTILIZATION:-20}"

mkdir -p "$ROOT/logs/experiments" "$ROOT/outputs/generations"

echo "=== LAUNCH CONFIGURATION ==="
echo "PYTHON_BIN=$PYTHON_BIN"
echo "PSL_GPU_IDS=$PSL_GPU_IDS"
echo "PSL_MAX_PARALLEL_MODELS=$PSL_MAX_PARALLEL_MODELS"
echo "PSL_POLL_SECONDS=$PSL_POLL_SECONDS"
echo "PSL_MAX_START_UTILIZATION=$PSL_MAX_START_UTILIZATION"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

echo
echo "=== CURRENT GPU ==="
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null

echo
echo "=== STARTING GPU-AWARE SCHEDULER ==="
"$PYTHON_BIN" -u \
    "$ROOT/scripts/gpu_scheduler.py" \
    --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/gpu_scheduler_console.log"

SCHEDULER_STATUS="${PIPESTATUS[0]}"
echo "[SCHEDULER_STATUS] $SCHEDULER_STATUS"
