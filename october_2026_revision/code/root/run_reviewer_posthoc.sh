#!/usr/bin/env bash

# PromptStressLab reviewer-requested post-hoc experiments.
# Foreground only: live output, no nohup/tmux/systemd-run.
# Intentionally no `set -e` so one failed stage does not silently close the workflow.

ROOT="/home/tahiti/PromptStressLab"
PY="/home/tahiti/Forensics/.venv_forensics/bin/python"
OUT="$ROOT/outputs/reviewer_posthoc"
SCRIPT="$ROOT/scripts/reviewer_posthoc.py"

export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$ROOT" || exit 1
mkdir -p "$ROOT/models" "$OUT"

echo "=== 0. Environment ==="
"$PY" - <<'PY'
import sys
print("python:", sys.executable)
mods = ["torch","transformers","accelerate","numpy","pandas","scipy","sklearn","statsmodels","tqdm","huggingface_hub"]
for m in mods:
    try:
        x=__import__(m)
        print(f"{m}: {getattr(x,'__version__','OK')}")
    except Exception as e:
        print(f"MISSING {m}: {e}")
PY

# Install only small analysis dependencies if missing.  Do NOT upgrade torch/transformers.
"$PY" - <<'PY'
import importlib.util, subprocess, sys
mods = {
    "numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "sklearn":"scikit-learn",
    "statsmodels":"statsmodels", "tqdm":"tqdm", "huggingface_hub":"huggingface_hub",
    "accelerate":"accelerate"
}
missing=[pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing missing analysis packages:", missing)
    subprocess.check_call([sys.executable,"-m","pip","install",*missing])
else:
    print("Analysis dependencies already present.")
PY

echo
echo "=== 1. Download/link two independent semantic encoders ==="
"$PY" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path('/home/tahiti/PromptStressLab/models')
root.mkdir(parents=True, exist_ok=True)
models={
    'all-mpnet-base-v2':'sentence-transformers/all-mpnet-base-v2',
    'allenai-specter':'allenai/specter',
}
for local_name, repo in models.items():
    link=root/local_name
    if link.exists() or link.is_symlink():
        print('EXISTS:', link, '->', link.resolve())
        continue
    print('DOWNLOADING:', repo)
    snap=Path(snapshot_download(repo_id=repo, cache_dir=str(Path.home()/'.cache/huggingface/hub')))
    link.symlink_to(snap, target_is_directory=True)
    print('LINKED:', link, '->', snap)
PY

echo
echo "=== 2. Verify the existing experiment artifacts BEFORE doing any analysis ==="
"$PY" "$SCRIPT" --root "$ROOT" --outdir "$OUT" audit
AUDIT_RC=$?
if [ "$AUDIT_RC" -ne 0 ]; then
    echo "AUDIT FAILED with code $AUDIT_RC"
    echo "Stop here and inspect the printed schema. No generations were launched."
    exit "$AUDIT_RC"
fi

echo
echo "=== 3. Multi-encoder PSI + embedding-free controls + ProSA + practical utility ==="
"$PY" "$SCRIPT" --root "$ROOT" --outdir "$OUT" analyze \
  --encoders clip_vit_b32,mpnet,specter \
  --device cuda \
  --embedding-batch-size 128 \
  --bootstrap 5000
ANALYZE_RC=$?
if [ "$ANALYZE_RC" -ne 0 ]; then
    echo "ANALYZE FAILED with code $ANALYZE_RC"
    exit "$ANALYZE_RC"
fi

echo
echo "=== 4. Exact POSIX from EXISTING responses (teacher forcing only; NO generate()) ==="
echo "This is the expensive stage. It scores saved y_j under all P1-P6 prompts."
echo "Resume is automatic through $OUT/posix_pair_logprobs.csv"
"$PY" "$SCRIPT" --root "$ROOT" --outdir "$OUT" posix \
  --batch-size 1 \
  --device-map auto
POSIX_RC=$?
if [ "$POSIX_RC" -ne 0 ]; then
    echo "POSIX stage stopped with code $POSIX_RC"
    echo "Inspect $OUT/posix_input_audit.csv. The earlier post-hoc analyses are still valid."
    exit "$POSIX_RC"
fi

echo
echo "=== DONE ==="
echo "Results: $OUT"
find "$OUT" -maxdepth 1 -type f -printf '%f\n' | sort
