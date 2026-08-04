#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from psl_common import read_jsonl

ROOT = Path('/home/tahiti/PromptStressLab')


def ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output = set()
    for row in read_jsonl(path):
        if isinstance(row.get('job_id'), str):
            output.add(row['job_id'])
    return output

jobs = read_jsonl(ROOT / 'manifests' / 'experiment_jobs.jsonl')
totals = Counter(row['model_id'] for row in jobs)
print('=== EXPERIMENT STATUS ===')
for model_id in sorted(totals):
    out = ROOT / 'outputs' / 'generations' / model_id
    predictions = ids(out / 'predictions.jsonl')
    errors = ids(out / 'errors.jsonl')
    done = len(predictions)
    total = totals[model_id]
    print(
        f'{model_id:28s} total={total:4d} predictions={len(predictions):4d} '
        f'errors={len(errors):4d} remaining={total-done:4d} progress={100*done/total:6.2f}%'
    )
print('\n=== GPU STATUS ===')
subprocess.run([
    'nvidia-smi',
    '--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu',
    '--format=csv,noheader',
], check=False)
