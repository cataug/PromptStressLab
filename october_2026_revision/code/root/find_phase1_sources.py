#!/usr/bin/env python3
"""Locate, without changing any file, historical Phase-I pilot scripts and results."""
import os, csv, re
from pathlib import Path
ROOTS=[Path('/home/tahiti/PromptStressLab'),Path('/home/tahiti/Malashin_Projects'),Path('/home/tahiti')]
TARGET=re.compile(r'(phase.?1|pilot|P[1-6]|psi|volatil|run_summary|filtered|prompt|fig.?5|ecdf|gpt.4)',re.I)
BAN={'.git','.venv','.venv_a100','.venv_forensics','.cache','node_modules','site-packages','models','checkpoints','huggingface','wandb','datasets','data','outputs/statistics','venv'}
EXT={'.py','.ipynb','.json','.jsonl','.csv','.tsv','.xlsx','.xls','.txt','.pdf','.png','.md'}
seen=set();hits=[]
for base in ROOTS:
 if not base.exists():continue
 for dirname,dirs,files in os.walk(base):
  current=Path(dirname)
  depth=len(current.relative_to(base).parts)
  dirs[:]=[d for d in dirs if d not in BAN and not d.startswith('.venv') and depth<5]
  for name in files:
   p=current/name
   if p in seen or p.suffix.lower() not in EXT:continue
   seen.add(p)
   if TARGET.search(name) and (p.suffix.lower()!='.jsonl' or p.stat().st_size < 250000000):
    hits.append({'path':str(p),'size_bytes':p.stat().st_size,'modified_iso':__import__('datetime').datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds')})
print('[PHASE I CANDIDATES]',len(hits))
for r in sorted(hits,key=lambda d:d['path'])[:250]:print(r['modified_iso'], f"{r['size_bytes']:>10}",r['path'])
out=Path('/home/tahiti/PromptStressLab/outputs/statistics/reviewer1_round2/phase1_source_inventory.csv')
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=['path','size_bytes','modified_iso']);w.writeheader();w.writerows(hits)
print('[WROTE]',out)
