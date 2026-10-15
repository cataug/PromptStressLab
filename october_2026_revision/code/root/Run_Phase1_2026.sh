#!/usr/bin/env bash
cd /home/tahiti/PromptStressLab || exit 1
PYTHON=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
if [ ! -x "$PYTHON" ]; then
  echo '[ERROR] Existing .venv_a100 not found';exit 1
fi
if [ ! -f rebuild_phase1_from_original.py ]; then
  echo '[ERROR] Put rebuild_phase1_from_original.py in /home/tahiti/PromptStressLab/';exit 1
fi
"$PYTHON" -u rebuild_phase1_from_original.py --root /home/tahiti/PromptStressLab
status=$?
if [ "$status" -ne 0 ]; then echo "[ERROR] Phase I failed: $status";exit "$status";fi
ls -lh outputs/phase1_reanalysis_2026/Figure*.pdf outputs/phase1_reanalysis_2026/phase1_recovered_tables.tex
