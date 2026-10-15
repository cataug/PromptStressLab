#!/usr/bin/env bash
# Manuscript insertion stage ONLY: uses existing Phase I CSV/PDF outputs.
ROOT="${ROOT:-/home/tahiti/PromptStressLab}"
PY="${PYTHON_BIN:-/home/tahiti/Malashin_Projects/.venv_a100/bin/python}"
cd "$ROOT" || exit 1
if [ ! -x "$PY" ]; then echo "[ERROR] Python executable not found: $PY"; exit 1; fi
if [ ! -f "outputs/phase1_reanalysis_2026/phase1_psi_summary.csv" ]; then
  echo '[ERROR] Recomputed data missing; do not rerun as part of this hotfix'; exit 1
fi
printf '\n[1/2] INTEGRATE PHASE I INTO ARTICLE (no recalculation)\n'
"$PY" -u phase1_complete_revision_FIXED.py \
  --root "$ROOT" --tex PromptStressLab_FINAL_R1_all_tables.tex
code=$?
if [ "$code" -ne 0 ]; then echo "[FAIL] TeX integration exit=$code"; exit "$code"; fi
cp -v outputs/phase1_reanalysis_2026/PromptStressLab_PHASE1_FINAL_INLINE.tex \
  PromptStressLab_PHASE1_FINAL_INLINE.tex
printf '\n[2/2] CHECK STRUCTURE\n'
"$PY" - <<'PY'
from pathlib import Path
import re
src=Path('PromptStressLab_FINAL_R1_all_tables.tex').read_text(encoding='utf8')
res=Path('PromptStressLab_PHASE1_FINAL_INLINE.tex').read_text(encoding='utf8')
cite=lambda t:set(re.findall(r'\\cite\{([^}]+)\}',t))
labels=re.findall(r'\\label\{([^}]+)\}',res)
assert cite(src)<=cite(res),'Lost bibliography citations'
assert src.count(r'\begin{figure')==res.count(r'\begin{figure'),'A figure disappeared'
assert len(labels)==len(set(labels)),'Duplicated LaTeX labels'
assert r'\label{app:phase1-prompts-p6}' in res
assert 'Rule set repetition #30' in res
assert 'The available material still lacks the Phase~I record-level logs' not in res
print('[OK] Source citation commands:',src.count(r'\cite{'),'->',res.count(r'\cite{'))
print('[OK] Figure environments:',src.count(r'\begin{figure'),'->',res.count(r'\begin{figure'))
print('[OK] Figure 3--5 paths:',*[x for x in ['Figure3_PSI_ECDF_with_insets_corrected.pdf','Figure4_Field_Volatility_gradient_corrected.pdf','Figure5_Pareto_common_axes_local_insets_corrected.pdf'] if (Path('figs')/x).is_file()])
print('[OK] Unique labels:',len(labels))
print('[OK] Historical P6 30 repetitions embedded')
PY
code=$?
if [ "$code" -ne 0 ]; then echo "[FAIL] Structure audit exit=$code"; exit "$code"; fi
ls -lh PromptStressLab_PHASE1_FINAL_INLINE.tex
printf '\n[DONE] Manuscript assembled; original predictions untouched; no LLM calls.\n'
