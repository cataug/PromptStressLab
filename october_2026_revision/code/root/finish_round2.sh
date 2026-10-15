#!/usr/bin/env bash
# Foreground, live output. Reuses the existing .venv_a100; no install, no nohup.
cd /home/tahiti/PromptStressLab || exit 1
source /home/tahiti/Malashin_Projects/.venv_a100/bin/activate || exit 1

# Optional: export PSL_TEXT_ENCODER_PATH=/absolute/path/to/local/text-only-encoder
EXTRA=()
if [ -n "${PSL_TEXT_ENCODER_PATH:-}" ]; then
  EXTRA+=(--text-encoder-path "$PSL_TEXT_ENCODER_PATH")
fi

python -u reviewer1_targeted_controls.py \
  --model-id qwen3_8b \
  --noise-documents 12 \
  --demo-documents 32 \
  --clip-path /home/tahiti/Forensics/models/clip_vit_base_patch32 \
  "${EXTRA[@]}"
CONTROL_EXIT=$?
if [ "$CONTROL_EXIT" -ne 0 ]; then
  echo "[ERROR] Targeted control failed with code $CONTROL_EXIT. Attempting standalone rebuild with available data."
fi

python -u build_promptstresslab_final.py
BUILD_EXIT=$?
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "[ERROR] Build failed with code $BUILD_EXIT"; exit "$BUILD_EXIT"
fi

python -u find_phase1_sources.py
printf '\n[READY] FULL SELF-CONTAINED LATEX:\n'
ls -lh /home/tahiti/PromptStressLab/PromptStressLab_R1_ROUND2_COMPLETE_INLINE.tex
printf '\n[NOTE] Historical Phase I remains subject to original raw logs and authentic dispatched P6 prompt.\n'
