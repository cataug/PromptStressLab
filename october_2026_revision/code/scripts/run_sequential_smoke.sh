#!/usr/bin/env bash

ROOT="/home/tahiti/PromptStressLab"

PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

export HF_HOME="/home/tahiti/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"

mkdir -p \
    "$ROOT/logs/smoke" \
    "$ROOT/outputs/smoke"

echo "=== SEQUENTIAL SMOKE TEST ==="
echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

echo
echo "=== ENVIRONMENT CHECK ==="

"$PYTHON_BIN" - <<'PY'
import sys

print("python:", sys.executable)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print("gpu_free_gib:", round(free / 1024**3, 3))
        print("gpu_total_gib:", round(total / 1024**3, 3))

except Exception as error:
    print("torch_error:", error)

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as error:
    print("transformers_error:", error)

try:
    import accelerate
    print("accelerate:", accelerate.__version__)
except Exception as error:
    print("accelerate_error:", error)
PY

run_one_model () {
    MODEL_KEY="$1"
    MODEL_PATH="$2"
    ARCHITECTURE="$3"

    LOG_PATH="$ROOT/logs/smoke/${MODEL_KEY}.log"

    echo
    echo "========================================"
    echo "[START] $MODEL_KEY"
    echo "[PATH]  $MODEL_PATH"
    echo "========================================"

    if [ ! -e "$MODEL_PATH" ]; then
        echo "[MISSING_MODEL] $MODEL_PATH"
        return 1
    fi

    echo
    echo "[NVIDIA_SMI_BEFORE]"

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free,memory.total \
        --format=csv,noheader 2>/dev/null

    "$PYTHON_BIN" -u \
        "$ROOT/scripts/smoke_local_model.py" \
        --root "$ROOT" \
        --model-key "$MODEL_KEY" \
        --model-path "$MODEL_PATH" \
        --architecture "$ARCHITECTURE" \
        --max-new-tokens 256 \
        2>&1 | tee "$LOG_PATH"

    STATUS="${PIPESTATUS[0]}"

    echo
    echo "[PYTHON_EXIT_CODE] $STATUS"

    echo "[NVIDIA_SMI_AFTER]"

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free,memory.total \
        --format=csv,noheader 2>/dev/null

    if [ "$STATUS" -eq 0 ]; then
        echo "[MODEL_SMOKE_OK] $MODEL_KEY"
    else
        echo "[MODEL_SMOKE_FAILED] $MODEL_KEY"
    fi

    sleep 3

    return "$STATUS"
}


QWEN_STATUS=99
MISTRAL_STATUS=99
GEMMA_STATUS=99


run_one_model \
    "qwen3_8b" \
    "$ROOT/models/Qwen3-8B" \
    "qwen3"

QWEN_STATUS="$?"


run_one_model \
    "mistral_7b_instruct_v03" \
    "$ROOT/models/Mistral-7B-Instruct-v0.3" \
    "mistral"

MISTRAL_STATUS="$?"


run_one_model \
    "gemma_3_12b_it" \
    "$ROOT/models/Gemma-3-12B-it" \
    "gemma3"

GEMMA_STATUS="$?"


echo
echo "========================================"
echo "=== FINAL SMOKE STATUS ==="
echo "========================================"
echo "Qwen3-8B:                  $QWEN_STATUS"
echo "Mistral-7B-Instruct-v0.3: $MISTRAL_STATUS"
echo "Gemma-3-12B-it:           $GEMMA_STATUS"

echo
echo "=== RESULT FILES ==="

find "$ROOT/outputs/smoke" \
    -maxdepth 2 \
    -type f \
    -printf "%p\n" \
    | sort

echo
echo "=== FINAL GPU STATUS ==="

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,memory.total \
    --format=csv,noheader 2>/dev/null

echo
echo "=== SEQUENTIAL_SMOKE_FINISHED ==="
