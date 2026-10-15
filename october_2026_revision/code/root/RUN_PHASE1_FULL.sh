#!/usr/bin/env bash
ROOT=/home/tahiti/PromptStressLab
PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
cd "$ROOT" || exit 1
if [ ! -x "$PY" ]; then echo '[ERROR] Existing .venv_a100 interpreter not found'; exit 1; fi
if [ ! -d "$ROOT/phase1_original/outputs/P1" ]; then
  echo '[ERROR] Original P1...P6 JSON folder missing; extract outputs.zip under phase1_original/'; exit 1
fi
if [ ! -f "$ROOT/legacy_PromptEng26/Prompting.ipynb" ]; then
  echo '[ERROR] Original cloned PromptEng26/Prompting.ipynb missing'; exit 1
fi
if [ ! -f "$ROOT/PromptStressLab_FINAL_R1_all_tables.tex" ]; then
  echo '[ERROR] Current manuscript PromptStressLab_FINAL_R1_all_tables.tex missing'; exit 1
fi
printf '\n===== [1/3] RECOMPUTE 530 x 6 FROM SAVED JSON =====\n'
"$PY" -u rebuild_phase1_from_original.py --root "$ROOT"
S=$?; if [ "$S" -ne 0 ]; then echo "[FAIL] Reanalysis exit=$S"; exit "$S"; fi
printf '\n===== [2/3] RECOVER EXACT 30-REPETITION P6 + INLINE TEX =====\n'
"$PY" -u phase1_complete_revision.py --root "$ROOT" --tex PromptStressLab_FINAL_R1_all_tables.tex
S=$?; if [ "$S" -ne 0 ]; then echo "[FAIL] Manuscript patch exit=$S"; exit "$S"; fi
F="$ROOT/outputs/phase1_reanalysis_2026/PromptStressLab_PHASE1_FINAL_INLINE.tex"
cp "$F" "$ROOT/PromptStressLab_PHASE1_FINAL_INLINE.tex"
printf '\n===== [3/3] VERIFY COUNTS AND OUTPUTS =====\n'
"$PY" - <<'PY'
from pathlib import Path
import re
root=Path('/home/tahiti/PromptStressLab')
a=(root/'PromptStressLab_FINAL_R1_all_tables.tex').read_text(encoding='utf8')
b=(root/'PromptStressLab_PHASE1_FINAL_INLINE.tex').read_text(encoding='utf8')
source=set(re.findall(r'\\cite\{([^}]+)\}',a))
final=set(re.findall(r'\\cite\{([^}]+)\}',b))
print('Citations (commands):',len(re.findall(r'\\cite\{',a)),'->',len(re.findall(r'\\cite\{',b)))
print('Lost citation groups:',source-final)
print('Figure environments:',a.count('\\begin{figure'),'->',b.count('\\begin{figure'))
assert not(source-final),'Citation was lost'
assert a.count('\\begin{figure')==b.count('\\begin{figure'),'A figure was removed'
print('[OK] Structural audit passed')
PY
S=$?; if [ "$S" -ne 0 ]; then echo "[FAIL] Final audit exit=$S"; exit "$S"; fi
ls -lh "$ROOT/PromptStressLab_PHASE1_FINAL_INLINE.tex" "$ROOT"/figs/Figure[345]*corrected.pdf
printf '\n[FINISHED] Original experiment preserved. No API calls.\n'
