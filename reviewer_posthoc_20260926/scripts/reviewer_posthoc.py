#!/usr/bin/env python3
"""
Reviewer-requested post-hoc analyses for PromptStressLab Phase II.

NO NEW LLM GENERATIONS are performed.

Implements:
  1) Multi-encoder robustness of output-semantic PSI
     - existing CLIP ViT-B/32
     - sentence-transformers/all-mpnet-base-v2 (mean pooled)
     - allenai/specter (CLS embedding)
  2) Encoder agreement / prompt-transition rank stability
  3) PSI <-> |Delta F1| correlations for every encoder, bootstrap CIs, Holm correction
  4) Practical PSI utility: risk stratification and AUROC/AUPRC for large F1 changes
  5) Embedding-free structured-output sensitivity baselines (set Jaccard / token Jaccard)
  6) ProSA-style PromptSensiScore (PSS) from existing per-document F1 values
  7) Exact POSIX from teacher-forced log-likelihood of EXISTING saved responses
     under alternative P1-P6 prompts (forward passes only; no generate()).
  8) CSV/JSON/LaTeX summaries and publication-ready PDF figures.

Designed for the existing repository:
  ROOT=/home/tahiti/PromptStressLab
  predictions: outputs/generations/*/predictions.jsonl
  metrics:     outputs/metrics/job_metrics_physical.csv

The loader intentionally accepts several historical column/key aliases.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Constants
# -----------------------------
PROMPTS = ["P1", "P2", "P3", "P4", "P5", "P6"]
ADJACENT = list(zip(PROMPTS[:-1], PROMPTS[1:]))

DATASET_FIELDS = {
    "scierc": [
        "Task", "Method", "Metric", "Material",
        "OtherScientificTerm", "Generic", "__RELATIONS__",
    ],
    "ebm-nlp": ["Participant", "Intervention", "Outcome"],
    "ebm_nlp": ["Participant", "Intervention", "Outcome"],
    "ebmnlp": ["Participant", "Intervention", "Outcome"],
    "scier": ["Dataset", "Method", "Task", "__RELATIONS__"],
}

MODEL_ALIASES = {
    "qwen": "Qwen3-8B",
    "qwen3-8b": "Qwen3-8B",
    "qwen/qwen3-8b": "Qwen3-8B",
    "gemma": "Gemma-3-12B-IT",
    "gemma-3-12b-it": "Gemma-3-12B-IT",
    "gemma-3-12b-it-it": "Gemma-3-12B-IT",
    "mistral": "Mistral-7B-Instruct-v0.3",
    "mistral-7b": "Mistral-7B-Instruct-v0.3",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct-v0.3",
}

DEFAULT_MODEL_PATHS = {
    "Qwen3-8B": "/home/tahiti/PromptStressLab/models/Qwen3-8B",
    "Gemma-3-12B-IT": "/home/tahiti/RIFT/models/Gemma-3-12B-it",
    "Mistral-7B-Instruct-v0.3": "/home/tahiti/TimeBound/models/mistralai__Mistral-7B-Instruct-v0.3",
}

DEFAULT_ENCODERS = {
    "clip_vit_b32": {
        "kind": "clip",
        "path": "/home/tahiti/Forensics/models/clip_vit_base_patch32",
        "max_length": 77,
    },
    "mpnet": {
        "kind": "meanpool",
        "path": "/home/tahiti/PromptStressLab/models/all-mpnet-base-v2",
        "max_length": 256,
    },
    "specter": {
        "kind": "cls",
        "path": "/home/tahiti/PromptStressLab/models/allenai-specter",
        "max_length": 256,
    },
}

KEY_ALIASES = {
    "model_id": ["model_id", "model", "model_name", "llm", "backbone"],
    "dataset": ["dataset", "dataset_name", "benchmark"],
    "record_id": ["record_id", "id", "doc_id", "sample_id", "example_id"],
    "condition": ["condition", "prompt_condition", "prompt_level", "variant"],
    "exact_f1": ["entity_exact_f1", "exact_entity_f1", "exact_f1"],
    "partial_f1": ["entity_partial_f1", "partial_entity_f1", "partial_f1"],
    "relation_f1": ["relation_exact_f1", "relation_f1", "rel_f1"],
}

PROMPT_FIELDS_RENDERED = [
    "rendered_prompt", "formatted_prompt", "model_input", "input_text",
    "prompt_text_rendered", "serialized_prompt",
]
PROMPT_FIELDS_USER = ["prompt", "user_prompt", "instruction", "input_prompt"]
PROMPT_FIELDS_MESSAGES = ["messages", "chat_messages"]
RESPONSE_FIELDS_RAW = [
    "raw_output", "raw_response", "generated_text", "generation",
    "response_text", "assistant_text", "raw_llm_text", "completion",
]
COMBINED_DECODE_FIELDS = ["full_text", "decoded_text", "decoded", "full_generation", "prediction"]

PARSED_FIELDS = [
    "parsed_prediction", "parsed_output", "prediction", "predicted",
    "output_json", "response_json", "structured_output", "output",
]


# -----------------------------
# Utility helpers
# -----------------------------
def norm_dataset(x: Any) -> str:
    s = str(x).strip()
    z = s.lower().replace(" ", "").replace("_", "-")
    if z in {"ebm-nlp", "ebmnlp"}:
        return "EBM-NLP"
    if z == "scierc":
        return "SciERC"
    if z == "scier":
        return "SciER"
    return s


def norm_model(x: Any) -> str:
    s = str(x).strip()
    key = s.lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    for k, v in MODEL_ALIASES.items():
        if k in key:
            return v
    return s


def norm_condition(x: Any) -> str:
    s = str(x).strip().upper()
    m = re.search(r"\b(P[1-6]|A[0-6])\b", s)
    return m.group(1) if m else s


def first_existing(mapping: Mapping[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            v = mapping[k]
            if not (isinstance(v, float) and np.isnan(v)):
                return v
    return default


def choose_col(df: pd.DataFrame, aliases: Sequence[str], required=True) -> Optional[str]:
    for c in aliases:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the expected columns found: {aliases}. Existing={list(df.columns)}")
    return None


def safe_json_loads(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Strip common markdown fences.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def stable_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def bootstrap_stat(x: np.ndarray, stat_fn, n_boot=5000, seed=42) -> Tuple[float, float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    obs = float(stat_fn(x))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(float(stat_fn(x[rng.integers(0, len(x), len(x))])))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return obs, float(lo), float(hi)


def bootstrap_corr(x, y, n_boot=3000, seed=42) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan, np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return rho, np.nan, np.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return rho, float(lo), float(hi)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
# Input loading
# -----------------------------
def load_jsonl_predictions(root: Path) -> pd.DataFrame:
    files = sorted((root / "outputs" / "generations").glob("*/predictions.jsonl"))
    if not files:
        files = sorted((root / "outputs" / "generations").glob("**/predictions.jsonl"))
    if not files:
        raise FileNotFoundError(f"No predictions.jsonl under {root/'outputs/generations'}")

    rows = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"WARN invalid JSONL {fp}:{ln}: {e}", file=sys.stderr)
                    continue
                obj["__source_file"] = str(fp)
                obj["__source_line"] = ln
                rows.append(obj)
    if not rows:
        raise RuntimeError("No prediction rows loaded")
    df = pd.DataFrame(rows)

    for canonical in ["model_id", "dataset", "record_id", "condition"]:
        c = choose_col(df, KEY_ALIASES[canonical])
        if c != canonical:
            df[canonical] = df[c]
    df["model_id"] = df["model_id"].map(norm_model)
    df["dataset"] = df["dataset"].map(norm_dataset)
    df["condition"] = df["condition"].map(norm_condition)
    df["record_id"] = df["record_id"].astype(str)

    # Main P1-P6 only for reviewer analyses.
    df = df[df["condition"].isin(PROMPTS)].copy()
    df = df.drop_duplicates(["model_id", "dataset", "record_id", "condition"], keep="last")
    return df



def augment_predictions_from_manifest(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort recovery of prompt-side fields from the original experiment manifest.

    The historical repository used manifests/experiment_jobs.jsonl.  If that file
    contains rendered prompts/messages, merge them by model/dataset/record/condition.
    Nothing is invented when the fields are absent.
    """
    fp = root / "manifests" / "experiment_jobs.jsonl"
    if not fp.exists():
        return df
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return df
    jobs = pd.DataFrame(rows)
    try:
        for canonical in ["model_id", "dataset", "record_id", "condition"]:
            c = choose_col(jobs, KEY_ALIASES[canonical])
            if c != canonical:
                jobs[canonical] = jobs[c]
        jobs["model_id"] = jobs["model_id"].map(norm_model)
        jobs["dataset"] = jobs["dataset"].map(norm_dataset)
        jobs["condition"] = jobs["condition"].map(norm_condition)
        jobs["record_id"] = jobs["record_id"].astype(str)
    except Exception as e:
        print(f"WARN could not normalize experiment_jobs.jsonl: {e}", file=sys.stderr)
        return df

    keys = ["model_id", "dataset", "record_id", "condition"]
    candidate_cols = [c for c in (PROMPT_FIELDS_RENDERED + PROMPT_FIELDS_USER + PROMPT_FIELDS_MESSAGES) if c in jobs.columns]
    if not candidate_cols:
        return df
    j = jobs[keys + candidate_cols].drop_duplicates(keys, keep="last")
    out = df.merge(j, on=keys, how="left", suffixes=("", "__job"))
    for c in candidate_cols:
        jc = c + "__job"
        if jc not in out.columns:
            continue
        if c not in out.columns:
            out[c] = out[jc]
        else:
            miss = out[c].isna() | out[c].astype(str).str.strip().isin(["", "nan", "None"])
            out.loc[miss, c] = out.loc[miss, jc]
        out.drop(columns=[jc], inplace=True)
    return out


def load_metrics(root: Path) -> pd.DataFrame:
    fp = root / "outputs" / "metrics" / "job_metrics_physical.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    for canonical in ["model_id", "dataset", "record_id", "condition"]:
        c = choose_col(df, KEY_ALIASES[canonical])
        if c != canonical:
            df[canonical] = df[c]
    for canonical in ["exact_f1", "partial_f1", "relation_f1"]:
        c = choose_col(df, KEY_ALIASES[canonical], required=False)
        if c:
            df[canonical] = pd.to_numeric(df[c], errors="coerce")
    if "partial_f1" not in df:
        raise KeyError("Could not find partial entity F1 in job_metrics_physical.csv")
    df["model_id"] = df["model_id"].map(norm_model)
    df["dataset"] = df["dataset"].map(norm_dataset)
    df["condition"] = df["condition"].map(norm_condition)
    df["record_id"] = df["record_id"].astype(str)
    return df[df["condition"].isin(PROMPTS)].drop_duplicates(
        ["model_id", "dataset", "record_id", "condition"], keep="last"
    ).copy()


def merge_inputs(pred: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "dataset", "record_id", "condition"]
    metric_cols = keys + [c for c in ["partial_f1", "exact_f1", "relation_f1"] if c in metrics.columns]
    out = pred.merge(metrics[metric_cols], on=keys, how="left", validate="one_to_one")
    missing = out["partial_f1"].isna().mean()
    if missing > 0:
        print(f"WARN: {missing:.1%} prediction rows lack partial_f1 after merge", file=sys.stderr)
    return out


# -----------------------------
# Structured output parsing
# -----------------------------
def locate_parsed_output(row: Mapping[str, Any]) -> Tuple[Any, str]:
    for k in PARSED_FIELDS:
        if k in row:
            obj = safe_json_loads(row[k])
            if isinstance(obj, (dict, list)):
                return obj, k
    if "entities" in row or "relations" in row:
        return {"entities": row.get("entities", []), "relations": row.get("relations", [])}, "row"
    # Last resort: try raw generation as JSON.
    for k in RESPONSE_FIELDS_RAW:
        if k in row:
            obj = safe_json_loads(row[k])
            if isinstance(obj, (dict, list)):
                return obj, k
    return {}, "missing"


def entity_text(e: Any) -> str:
    if isinstance(e, str):
        return e
    if isinstance(e, Mapping):
        return str(first_existing(e, ["text", "span", "mention", "surface", "value", "name"], ""))
    return str(e)


def entity_label(e: Any) -> str:
    if isinstance(e, Mapping):
        return str(first_existing(e, ["type", "label", "entity_type", "category", "class"], ""))
    return ""


def relation_text(r: Any, entities: Sequence[Any]) -> str:
    if isinstance(r, str):
        return r
    if not isinstance(r, Mapping):
        return str(r)

    rel = first_existing(r, ["type", "label", "relation", "relation_type", "predicate"], "REL")
    h = first_existing(r, ["head", "source", "subject", "arg1", "from", "head_text"], "")
    t = first_existing(r, ["tail", "target", "object", "arg2", "to", "tail_text"], "")

    def resolve(v):
        if isinstance(v, Mapping):
            return entity_text(v)
        if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
            idx = int(v)
            if 0 <= idx < len(entities):
                return entity_text(entities[idx])
        return str(v)

    return f"{resolve(h)} || {rel} || {resolve(t)}"


def fields_for_dataset(dataset: str) -> List[str]:
    k = dataset.lower().replace("_", "-")
    if k == "ebm-nlp":
        return DATASET_FIELDS["ebm-nlp"]
    if k == "scierc":
        return DATASET_FIELDS["scierc"]
    if k == "scier":
        return DATASET_FIELDS["scier"]
    raise KeyError(f"Unknown dataset for fields: {dataset}")


def extract_field_items(parsed: Any, dataset: str) -> Dict[str, List[str]]:
    fields = fields_for_dataset(dataset)
    out = {f: [] for f in fields}
    if not isinstance(parsed, Mapping):
        return out

    # Native entities + relations schema.
    entities = parsed.get("entities", []) or []
    relations = parsed.get("relations", []) or []

    if isinstance(entities, Mapping):
        # Some evaluators may store entities grouped by type.
        tmp = []
        for typ, vals in entities.items():
            vals = vals if isinstance(vals, list) else [vals]
            for v in vals:
                if isinstance(v, Mapping):
                    e = dict(v)
                    e.setdefault("type", typ)
                else:
                    e = {"text": v, "type": typ}
                tmp.append(e)
        entities = tmp

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    # Case-insensitive mapping to canonical field labels.
    cmap = {f.lower(): f for f in fields if f != "__RELATIONS__"}
    for e in entities:
        txt = normalize_text(entity_text(e))
        typ = entity_label(e).strip().lower()
        if not txt:
            continue
        if typ in cmap:
            out[cmap[typ]].append(txt)
        else:
            # Exact-ish normalized variants.
            typ2 = re.sub(r"[^a-z0-9]", "", typ)
            for low, canon in cmap.items():
                if re.sub(r"[^a-z0-9]", "", low) == typ2:
                    out[canon].append(txt)
                    break

    if "__RELATIONS__" in out:
        for r in relations:
            txt = normalize_text(relation_text(r, entities))
            if txt:
                out["__RELATIONS__"].append(txt)

    # Fallback for field-keyed outputs.
    for f in fields:
        if f == "__RELATIONS__" or out[f]:
            continue
        for key in [f, f.lower(), f.replace("Scientific", "_scientific_").lower()]:
            if key in parsed:
                vals = parsed[key]
                vals = vals if isinstance(vals, list) else [vals]
                out[f].extend([normalize_text(entity_text(v)) for v in vals if normalize_text(entity_text(v))])

    # Deduplicate without changing deterministic order.
    for f in out:
        out[f] = list(dict.fromkeys(out[f]))
    return out


def add_structured_fields(df: pd.DataFrame) -> pd.DataFrame:
    parsed_list, sources, fields_json = [], [], []
    for row in tqdm(df.to_dict("records"), desc="Parse structured outputs"):
        parsed, src = locate_parsed_output(row)
        parsed_list.append(parsed)
        sources.append(src)
        fields_json.append(extract_field_items(parsed, row["dataset"]))
    out = df.copy()
    out["parsed_output"] = parsed_list
    out["parsed_source"] = sources
    out["field_items"] = fields_json
    return out


# -----------------------------
# Encoder backends
# -----------------------------
@dataclass
class EncoderSpec:
    name: str
    kind: str
    path: str
    max_length: int


class TextEncoder:
    def __init__(self, spec: EncoderSpec, device: str = "cuda", batch_size: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPTokenizer

        self.torch = torch
        self.spec = spec
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.batch_size = batch_size
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if spec.kind == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained(spec.path, local_files_only=True)
            self.model = CLIPModel.from_pretrained(spec.path, local_files_only=True).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(spec.path, local_files_only=True, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                spec.path,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        all_vecs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Embed {self.spec.name}", leave=False):
            batch = list(texts[i:i + self.batch_size])
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.inference_mode():
                if self.spec.kind == "clip":
                    out = self.model.get_text_features(**toks, return_dict=True)
                    vec = out.pooler_output if hasattr(out, "pooler_output") else out
                else:
                    out = self.model(**toks)
                    h = out.last_hidden_state
                    if self.spec.kind == "cls":
                        vec = h[:, 0, :]
                    else:
                        mask = toks["attention_mask"].unsqueeze(-1).to(h.dtype)
                        vec = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
                vec = vec.float()
                vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_vecs.append(vec.cpu().numpy())
        return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 1), dtype=np.float32)

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_item_vocabulary(df: pd.DataFrame) -> List[str]:
    items = set()
    for d in df["field_items"]:
        for vals in d.values():
            items.update(v for v in vals if v)
    return sorted(items)


def load_or_compute_embeddings(
    df: pd.DataFrame,
    spec: EncoderSpec,
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    vocab = build_item_vocabulary(df)
    fingerprint = sha1_text("\n".join(vocab))[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"{spec.name}_{fingerprint}.npz"
    meta = cache_dir / f"{spec.name}_{fingerprint}.json"
    if npz.exists() and meta.exists():
        arr = np.load(npz, allow_pickle=False)
        texts = json.loads(meta.read_text(encoding="utf-8"))["texts"]
        if texts == vocab:
            print(f"Reuse embedding cache: {npz}")
            return {t: arr["emb"][i] for i, t in enumerate(texts)}

    enc = TextEncoder(spec, device=device, batch_size=batch_size)
    emb = enc.encode(vocab).astype(np.float32)
    enc.close()
    np.savez_compressed(npz, emb=emb)
    write_json(meta, {"encoder": spec.__dict__, "texts": vocab})
    return {t: emb[i] for i, t in enumerate(vocab)}


def centroid(items: Sequence[str], emb: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    vals = [emb[x] for x in items if x in emb]
    if not vals:
        return None
    c = np.mean(np.stack(vals), axis=0)
    n = np.linalg.norm(c)
    if n <= 1e-12:
        return None
    return c / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(1.0 - float(np.dot(a, b)), 0.0, 2.0))


def field_semantic_distance(a_items, b_items, emb) -> float:
    if not a_items and not b_items:
        return 0.0
    if not a_items or not b_items:
        return 1.0
    ca, cb = centroid(a_items, emb), centroid(b_items, emb)
    if ca is None and cb is None:
        return 0.0
    if ca is None or cb is None:
        return 1.0
    return cosine_distance(ca, cb)


def set_jaccard_distance(a_items, b_items) -> float:
    a, b = set(a_items), set(b_items)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def token_jaccard_distance(a_items, b_items) -> float:
    tok = lambda vals: set(re.findall(r"[a-z0-9]+", " ".join(vals).lower()))
    a, b = tok(a_items), tok(b_items)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


# -----------------------------
# Pair construction / PSI
# -----------------------------
def make_condition_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, Mapping[str, Any]]]:
    lookup = defaultdict(dict)
    for r in df.to_dict("records"):
        lookup[(r["model_id"], r["dataset"], r["record_id"])][r["condition"]] = r
    return lookup


def pairwise_sensitivity(
    df: pd.DataFrame,
    encoder_name: str,
    emb: Optional[Mapping[str, np.ndarray]],
    all_pairs: bool = False,
) -> pd.DataFrame:
    pairs = list(itertools.combinations(PROMPTS, 2)) if all_pairs else ADJACENT
    rows = []
    lookup = make_condition_lookup(df)
    for (model, dataset, rid), conds in tqdm(lookup.items(), desc=f"PSI {encoder_name} all={all_pairs}"):
        fields = fields_for_dataset(dataset)
        for p0, p1 in pairs:
            if p0 not in conds or p1 not in conds:
                continue
            a, b = conds[p0], conds[p1]
            field_d = {}
            for f in fields:
                ai = a["field_items"].get(f, [])
                bi = b["field_items"].get(f, [])
                if encoder_name == "set_jaccard":
                    d = set_jaccard_distance(ai, bi)
                elif encoder_name == "token_jaccard":
                    d = token_jaccard_distance(ai, bi)
                else:
                    assert emb is not None
                    d = field_semantic_distance(ai, bi, emb)
                field_d[f] = d
            row = {
                "model_id": model,
                "dataset": dataset,
                "record_id": rid,
                "from_condition": p0,
                "to_condition": p1,
                "transition": f"{p0}->{p1}",
                "encoder": encoder_name,
                "psi": float(np.mean(list(field_d.values()))),
                "partial_f1_from": a.get("partial_f1", np.nan),
                "partial_f1_to": b.get("partial_f1", np.nan),
                "exact_f1_from": a.get("exact_f1", np.nan),
                "exact_f1_to": b.get("exact_f1", np.nan),
            }
            row["delta_partial_f1"] = row["partial_f1_to"] - row["partial_f1_from"]
            row["abs_delta_partial_f1"] = abs(row["delta_partial_f1"])
            if np.isfinite(row["exact_f1_from"]) and np.isfinite(row["exact_f1_to"]):
                row["delta_exact_f1"] = row["exact_f1_to"] - row["exact_f1_from"]
                row["abs_delta_exact_f1"] = abs(row["delta_exact_f1"])
            for f, d in field_d.items():
                row[f"field::{f}"] = d
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Encoder robustness statistics
# -----------------------------
def encoder_agreement(pairwise: pd.DataFrame) -> pd.DataFrame:
    idx = ["model_id", "dataset", "record_id", "from_condition", "to_condition"]
    wide = pairwise.pivot_table(index=idx, columns="encoder", values="psi", aggfunc="first")
    encs = [e for e in wide.columns if e not in {"set_jaccard", "token_jaccard"}]
    rows = []
    for a, b in itertools.combinations(encs, 2):
        sub = wide[[a, b]].dropna()
        if len(sub) < 4:
            continue
        sr = spearmanr(sub[a], sub[b])
        kt = kendalltau(sub[a], sub[b])
        qa, qb = sub[a].quantile(.90), sub[b].quantile(.90)
        sa, sb = set(sub.index[sub[a] >= qa]), set(sub.index[sub[b] >= qb])
        jac = len(sa & sb) / max(1, len(sa | sb))
        rows.append({
            "encoder_a": a, "encoder_b": b, "n": len(sub),
            "spearman_rho": sr.statistic, "spearman_p": sr.pvalue,
            "kendall_tau": kt.statistic, "kendall_p": kt.pvalue,
            "top10pct_jaccard": jac,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["spearman_p_holm"] = holm_adjust(out["spearman_p"])
        out["kendall_p_holm"] = holm_adjust(out["kendall_p"])
    return out


def transition_rank_agreement(pairwise: pd.DataFrame) -> pd.DataFrame:
    # Compare ranking of the 5 Pk->Pk+1 transitions within each model x dataset.
    mean = pairwise.groupby(["encoder", "model_id", "dataset", "transition"], as_index=False)["psi"].mean()
    rows = []
    encs = sorted(e for e in mean["encoder"].unique() if e not in {"set_jaccard", "token_jaccard"})
    for (model, dataset), g in mean.groupby(["model_id", "dataset"]):
        w = g.pivot(index="transition", columns="encoder", values="psi")
        for a, b in itertools.combinations(encs, 2):
            if a not in w or b not in w:
                continue
            s = w[[a, b]].dropna()
            if len(s) < 3:
                continue
            rows.append({
                "model_id": model, "dataset": dataset,
                "encoder_a": a, "encoder_b": b, "n_transitions": len(s),
                "spearman_rho": spearmanr(s[a], s[b]).statistic,
                "kendall_tau": kendalltau(s[a], s[b]).statistic,
            })
    return pd.DataFrame(rows)



def field_volatility_tables(pairwise: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    field_cols = [c for c in pairwise.columns if c.startswith("field::")]
    rows = []
    for r in pairwise.to_dict("records"):
        for c in field_cols:
            rows.append({
                "encoder": r["encoder"], "model_id": r["model_id"], "dataset": r["dataset"],
                "record_id": r["record_id"], "transition": r["transition"],
                "field": c.split("::", 1)[1], "field_psi": r.get(c, np.nan),
            })
    long = pd.DataFrame(rows).dropna(subset=["field_psi"])
    doc = long.groupby(["encoder", "model_id", "dataset", "record_id", "field"], as_index=False).agg(
        volatility=("field_psi", "mean")
    )
    summary = doc.groupby(["encoder", "model_id", "dataset", "field"], as_index=False).agg(
        n=("volatility", "size"), mean_volatility=("volatility", "mean"),
        median_volatility=("volatility", "median")
    )
    # Agreement of field rankings across semantic encoders.
    semantic = [e for e in summary["encoder"].unique() if e not in {"set_jaccard", "token_jaccard"}]
    agree = []
    for (model, dataset), g in summary.groupby(["model_id", "dataset"]):
        w = g.pivot(index="field", columns="encoder", values="mean_volatility")
        for a, b in itertools.combinations(sorted(semantic), 2):
            if a not in w or b not in w:
                continue
            h = w[[a, b]].dropna()
            if len(h) < 3:
                continue
            agree.append({
                "model_id": model, "dataset": dataset, "encoder_a": a, "encoder_b": b,
                "n_fields": len(h), "spearman_rho": spearmanr(h[a], h[b]).statistic,
                "kendall_tau": kendalltau(h[a], h[b]).statistic,
            })
    return doc, summary, pd.DataFrame(agree)


def psi_f1_correlations(pairwise: pd.DataFrame, n_boot: int = 3000) -> pd.DataFrame:
    rows = []
    groupings = [
        ("overall", []),
        ("model", ["model_id"]),
        ("model_dataset", ["model_id", "dataset"]),
    ]
    for level, cols in groupings:
        iterator = [((), pairwise)] if not cols else pairwise.groupby(cols)
        for key, g in iterator:
            if not isinstance(key, tuple):
                key = (key,)
            meta = dict(zip(cols, key))
            for target in ["abs_delta_partial_f1", "abs_delta_exact_f1", "delta_partial_f1"]:
                if target not in g.columns:
                    continue
                for enc, h in g.groupby("encoder"):
                    sub = h[["psi", target]].dropna()
                    if len(sub) < 10:
                        continue
                    sr = spearmanr(sub["psi"], sub[target])
                    rho, lo, hi = bootstrap_corr(sub["psi"].values, sub[target].values, n_boot=n_boot)
                    rows.append({
                        "level": level, **meta, "encoder": enc, "target": target,
                        "n": len(sub), "spearman_rho": sr.statistic,
                        "p_value": sr.pvalue, "ci95_low": lo, "ci95_high": hi,
                    })
    out = pd.DataFrame(rows)
    if len(out):
        # Holm within each declared level x target family.
        out["p_holm"] = np.nan
        for _, ix in out.groupby(["level", "target"]).groups.items():
            out.loc[ix, "p_holm"] = holm_adjust(out.loc[ix, "p_value"])
    return out


# -----------------------------
# Practical PSI utility
# -----------------------------
def practical_risk(pairwise: pd.DataFrame, thresholds=(0.05, 0.10, 0.20)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, bins = [], []
    for (enc, model), g in pairwise.groupby(["encoder", "model_id"]):
        h = g[["psi", "abs_delta_partial_f1"]].dropna().copy()
        if len(h) < 30:
            continue
        # Quantile stratification, duplicates-safe.
        h["psi_quintile"] = pd.qcut(h["psi"].rank(method="first"), 5, labels=False) + 1
        for q, qg in h.groupby("psi_quintile"):
            bins.append({
                "encoder": enc, "model_id": model, "psi_quintile": int(q),
                "n": len(qg), "mean_psi": qg["psi"].mean(),
                "mean_abs_delta_partial_f1": qg["abs_delta_partial_f1"].mean(),
                "p_abs_delta_ge_0.10": (qg["abs_delta_partial_f1"] >= .10).mean(),
                "p_abs_delta_ge_0.20": (qg["abs_delta_partial_f1"] >= .20).mean(),
            })
        for t in thresholds:
            y = (h["abs_delta_partial_f1"] >= t).astype(int)
            if y.nunique() < 2:
                continue
            rows.append({
                "encoder": enc, "model_id": model, "threshold_abs_delta_f1": t,
                "n": len(h), "positive_rate": y.mean(),
                "auroc": roc_auc_score(y, h["psi"]),
                "auprc": average_precision_score(y, h["psi"]),
            })
    return pd.DataFrame(rows), pd.DataFrame(bins)


# -----------------------------
# ProSA PromptSensiScore
# -----------------------------
def compute_prosa_pss(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, dataset, rid), g in df.groupby(["model_id", "dataset", "record_id"]):
        by = {r.condition: r for r in g.itertuples(index=False) if r.condition in PROMPTS}
        if not all(p in by for p in PROMPTS):
            continue
        for metric in ["partial_f1", "exact_f1"]:
            if metric not in df.columns:
                continue
            vals = [getattr(by[p], metric, np.nan) for p in PROMPTS]
            if not np.all(np.isfinite(vals)):
                continue
            diffs = [abs(vals[i] - vals[j]) for i, j in itertools.combinations(range(6), 2)]
            rows.append({
                "model_id": model, "dataset": dataset, "record_id": rid,
                "metric": metric, "pss": float(np.mean(diffs)),
                "max_pairwise_difference": float(np.max(diffs)),
                "std_across_prompts": float(np.std(vals, ddof=0)),
            })
    doc = pd.DataFrame(rows)
    summary = doc.groupby(["model_id", "dataset", "metric"], as_index=False).agg(
        n=("pss", "size"), mean_pss=("pss", "mean"), median_pss=("pss", "median"),
        mean_max_pairwise_difference=("max_pairwise_difference", "mean"),
    ) if len(doc) else pd.DataFrame()
    return doc, summary


def compare_semantic_to_prosa(allpairs: pd.DataFrame, pss_doc: pd.DataFrame) -> pd.DataFrame:
    sem = allpairs.groupby(["encoder", "model_id", "dataset", "record_id"], as_index=False)["psi"].mean()
    p = pss_doc[pss_doc["metric"] == "partial_f1"]
    m = sem.merge(p[["model_id", "dataset", "record_id", "pss"]], on=["model_id", "dataset", "record_id"], how="inner")
    rows = []
    for (enc, model), g in m.groupby(["encoder", "model_id"]):
        if len(g) < 10:
            continue
        sr = spearmanr(g["psi"], g["pss"])
        rows.append({
            "encoder": enc, "model_id": model, "n": len(g),
            "spearman_rho_semantic_psi_vs_prosa_pss": sr.statistic,
            "p_value": sr.pvalue,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"])
    return out


# -----------------------------
# Exact POSIX: prompt / response extraction
# -----------------------------

def split_combined_decode(row: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """Recover (rendered_prefix, assistant_response) from a decoded full sequence.

    This is only used when a historical prediction row stored prompt+completion in
    one string. It recognizes the chat delimiters used by Qwen3, Gemma 3 and Mistral.
    """
    model = norm_model(row.get("model_id", row.get("model", "")))
    for k in COMBINED_DECODE_FIELDS:
        v = row.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        text = v
        candidates = []
        if model == "Qwen3-8B":
            candidates = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
        elif model == "Gemma-3-12B-IT":
            candidates = ["<start_of_turn>model\n", "<start_of_turn>model"]
        elif model == "Mistral-7B-Instruct-v0.3":
            candidates = ["[/INST]", "[/INST] "]
        candidates += ["assistant\n", "assistant:"]
        best = None
        for marker in candidates:
            pos = text.rfind(marker)
            if pos >= 0 and (best is None or pos > best[0]):
                best = (pos, marker)
        if best is None:
            continue
        pos, marker = best
        cut = pos + len(marker)
        prefix = text[:cut]
        resp = text[cut:].strip()
        # Strip common chat terminators from the response only.
        resp = re.sub(r"(?:<\|im_end\|>|<end_of_turn>|</s>)\s*$", "", resp).strip()
        if prefix.strip() and resp:
            return prefix, resp
    return None


def get_messages_from_row(row: Mapping[str, Any]) -> Optional[List[Dict[str, str]]]:
    for k in PROMPT_FIELDS_MESSAGES:
        if k in row:
            x = row[k]
            if isinstance(x, str):
                x = safe_json_loads(x)
            if isinstance(x, list) and x and isinstance(x[0], Mapping):
                return [dict(m) for m in x]
    return None


def get_prompt_from_row(row: Mapping[str, Any]) -> Tuple[Any, str]:
    msgs = get_messages_from_row(row)
    if msgs is not None:
        return msgs, "messages"
    for k in PROMPT_FIELDS_RENDERED:
        if k in row and str(row[k]).strip():
            return str(row[k]), "rendered"
    for k in PROMPT_FIELDS_USER:
        if k in row and str(row[k]).strip():
            return str(row[k]), "user"
    combined = split_combined_decode(row)
    if combined is not None:
        return combined[0], "rendered_from_combined_decode"
    return None, "missing"


def get_response_from_row(row: Mapping[str, Any]) -> Tuple[str, str]:
    for k in RESPONSE_FIELDS_RAW:
        if k in row:
            v = row[k]
            if isinstance(v, str) and v.strip():
                return v.strip(), f"raw:{k}"
    combined = split_combined_decode(row)
    if combined is not None:
        return combined[1], "raw:combined_decode"
    parsed, src = locate_parsed_output(row)
    if parsed:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), f"reconstructed:{src}"
    return "", "missing"


def apply_chat_template_safe(tokenizer, messages: List[Mapping[str, Any]], model_name: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if model_name == "Qwen3-8B":
        # Qwen3 supports this and original experiment disabled reasoning.
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_prefix(tokenizer, prompt_obj: Any, prompt_kind: str, model_name: str) -> str:
    if prompt_kind in {"rendered", "rendered_from_combined_decode"}:
        return str(prompt_obj)
    if prompt_kind == "messages":
        return apply_chat_template_safe(tokenizer, prompt_obj, model_name)
    if prompt_kind == "user":
        msgs = [{"role": "user", "content": str(prompt_obj)}]
        return apply_chat_template_safe(tokenizer, msgs, model_name)
    raise ValueError(f"Missing prompt for POSIX ({prompt_kind})")


class TeacherForcedScorer:
    """Computes average token log P(existing_response | prompt), without generation."""
    def __init__(self, model_name: str, model_path: str, device_map: str = "auto"):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        model = None
        errors = []
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception as e:
            errors.append(f"AutoModelForCausalLM: {e}")
        if model is None:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
            except Exception as e:
                errors.append(f"AutoModelForImageTextToText: {e}")
        if model is None:
            raise RuntimeError("Could not load model:\n" + "\n".join(errors))
        self.model = model.eval()
        # Input embedding device is reliable with device_map=auto.
        self.input_device = self.model.get_input_embeddings().weight.device

    def encode_pair(self, prompt_obj, prompt_kind: str, response: str) -> Tuple[List[int], int, int]:
        prefix = render_prefix(self.tokenizer, prompt_obj, prompt_kind, self.model_name)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=True)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        if not response_ids:
            return prefix_ids, len(prefix_ids), 0
        return prefix_ids + response_ids, len(prefix_ids), len(response_ids)

    def score_batch(self, batch: Sequence[Tuple[Any, str, str]]) -> List[Tuple[float, int]]:
        torch = self.torch
        encoded = [self.encode_pair(*x) for x in batch]
        maxlen = max(len(x[0]) for x in encoded)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(encoded), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(encoded), maxlen), dtype=torch.long)
        for i, (ids, _, _) in enumerate(encoded):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, :len(ids)] = 1
        input_ids = input_ids.to(self.input_device)
        attn = attn.to(self.input_device)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=False,
            ).logits

        # Do NOT materialize log_softmax(logits.float()) for the complete
        # sequence. For large-vocabulary models such as Gemma this can require
        # several additional GiB. POSIX only needs log-probabilities of the
        # observed response tokens, so compute FP32 log-softmax in small
        # sequence chunks.
        ans = []
        chunk_tokens = 64

        for i, (_, prefix_len, resp_len) in enumerate(encoded):
            if resp_len <= 0:
                ans.append((np.nan, 0))
                continue

            # Response tokens at positions
            # prefix_len ... prefix_len+resp_len-1
            # are predicted by logits at positions
            # prefix_len-1 ... prefix_len+resp_len-2.
            target = input_ids[
                i,
                prefix_len:prefix_len + resp_len
            ]

            pred_logits = logits[
                i,
                prefix_len - 1:prefix_len + resp_len - 1,
                :
            ]

            total_logprob = 0.0

            for start in range(0, resp_len, chunk_tokens):
                end = min(start + chunk_tokens, resp_len)

                chunk_logits = pred_logits[start:end, :].float()
                chunk_logp = torch.log_softmax(
                    chunk_logits,
                    dim=-1,
                )

                chunk_target = target[start:end]

                tok_lp = chunk_logp.gather(
                    -1,
                    chunk_target.unsqueeze(-1),
                ).squeeze(-1)

                total_logprob += float(tok_lp.sum().item())

                del chunk_logits, chunk_logp, chunk_target, tok_lp

            ans.append((total_logprob, int(resp_len)))

        del logits
        return ans

    def close(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_posix_tasks(df: pd.DataFrame, all_pairs=True) -> pd.DataFrame:
    rows = []
    lookup = make_condition_lookup(df)
    for (model, dataset, rid), conds in lookup.items():
        if not all(p in conds for p in PROMPTS):
            continue
        for target_cond in PROMPTS:
            response, response_source = get_response_from_row(conds[target_cond])
            if not response:
                continue
            sources = PROMPTS if all_pairs else [target_cond] + [p for p in PROMPTS if abs(int(p[1]) - int(target_cond[1])) == 1]
            for source_cond in sources:
                prompt_obj, prompt_kind = get_prompt_from_row(conds[source_cond])
                if prompt_obj is None:
                    continue
                rows.append({
                    "model_id": model, "dataset": dataset, "record_id": rid,
                    "source_condition": source_cond, "target_condition": target_cond,
                    "prompt_obj": prompt_obj, "prompt_kind": prompt_kind,
                    "response": response, "response_source": response_source,
                    "is_self": source_cond == target_cond,
                })
    return pd.DataFrame(rows)


def run_posix(
    df: pd.DataFrame,
    outdir: Path,
    model_paths: Mapping[str, str],
    batch_size: int = 2,
    device_map: str = "auto",
    all_pairs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task_df = build_posix_tasks(df, all_pairs=all_pairs)
    if task_df.empty:
        raise RuntimeError(
            "POSIX tasks are empty. predictions.jsonl must contain prompt/messages and raw or parsed response."
        )

    score_path = outdir / "posix_pair_logprobs.csv"
    done = set()
    prior = []
    keycols = ["model_id", "dataset", "record_id", "source_condition", "target_condition"]
    if score_path.exists():
        old = pd.read_csv(score_path)
        prior = old.to_dict("records")
        for r in prior:
            done.add(tuple(str(r[k]) for k in keycols))
        print(f"Resume POSIX: {len(done)} pair scores already present")

    new_rows = []
    for model_name in sorted(task_df["model_id"].unique()):
        if model_name not in model_paths:
            print(f"WARN no model path for {model_name}; skipping POSIX", file=sys.stderr)
            continue
        path = model_paths[model_name]
        if not Path(path).exists():
            print(f"WARN model path missing: {path}; skipping POSIX for {model_name}", file=sys.stderr)
            continue
        mdf = task_df[task_df["model_id"] == model_name].copy()
        pending = []
        pending_meta = []
        scorer = TeacherForcedScorer(model_name, path, device_map=device_map)
        try:
            for r in tqdm(mdf.to_dict("records"), desc=f"POSIX {model_name}"):
                key = tuple(str(r[k]) for k in keycols)
                if key in done:
                    continue
                pending.append((r["prompt_obj"], r["prompt_kind"], r["response"]))
                pending_meta.append(r)
                if len(pending) >= batch_size:
                    scores = scorer.score_batch(pending)
                    for meta, (sum_logp, n_tok) in zip(pending_meta, scores):
                        z = {k: meta[k] for k in keycols}
                        z.update({
                            "prompt_kind": meta["prompt_kind"],
                            "response_source": meta["response_source"],
                            "sum_logprob": sum_logp,
                            "response_tokens": n_tok,
                            "mean_logprob": sum_logp / n_tok if n_tok else np.nan,
                        })
                        new_rows.append(z)
                    pending, pending_meta = [], []
                    if len(new_rows) % 100 == 0:
                        pd.DataFrame(prior + new_rows).to_csv(score_path, index=False)
            if pending:
                scores = scorer.score_batch(pending)
                for meta, (sum_logp, n_tok) in zip(pending_meta, scores):
                    z = {k: meta[k] for k in keycols}
                    z.update({
                        "prompt_kind": meta["prompt_kind"],
                        "response_source": meta["response_source"],
                        "sum_logprob": sum_logp,
                        "response_tokens": n_tok,
                        "mean_logprob": sum_logp / n_tok if n_tok else np.nan,
                    })
                    new_rows.append(z)
        finally:
            scorer.close()
            pd.DataFrame(prior + new_rows).to_csv(score_path, index=False)

    scores = pd.DataFrame(prior + new_rows)
    if scores.empty:
        return scores, pd.DataFrame()

    # POSIX per document: average over i != j of | mean_logP(y_j|x_i) - mean_logP(y_j|x_j) |.
    self_lp = scores[scores["source_condition"] == scores["target_condition"]][
        ["model_id", "dataset", "record_id", "target_condition", "mean_logprob"]
    ].rename(columns={"mean_logprob": "self_mean_logprob"})
    cross = scores.merge(self_lp, on=["model_id", "dataset", "record_id", "target_condition"], how="left")
    cross = cross[cross["source_condition"] != cross["target_condition"]].copy()
    cross["posix_term"] = (cross["mean_logprob"] - cross["self_mean_logprob"]).abs()
    doc = cross.groupby(["model_id", "dataset", "record_id"], as_index=False).agg(
        posix=("posix_term", "mean"),
        n_cross_pairs=("posix_term", "size"),
        response_source=("response_source", lambda x: "raw" if all(str(v).startswith("raw:") for v in x) else "mixed_or_reconstructed"),
    )
    doc.to_csv(outdir / "posix_document.csv", index=False)
    summary = doc.groupby(["model_id", "dataset"], as_index=False).agg(
        n=("posix", "size"), mean_posix=("posix", "mean"), median_posix=("posix", "median")
    )
    summary.to_csv(outdir / "posix_summary.csv", index=False)
    return scores, doc


def compare_posix_with_others(posix_doc: pd.DataFrame, allpairs: pd.DataFrame, pss_doc: pd.DataFrame) -> pd.DataFrame:
    if posix_doc.empty:
        return pd.DataFrame()
    sem = allpairs.groupby(["encoder", "model_id", "dataset", "record_id"], as_index=False)["psi"].mean()
    pss = pss_doc[pss_doc["metric"] == "partial_f1"][["model_id", "dataset", "record_id", "pss"]]
    m = sem.merge(posix_doc, on=["model_id", "dataset", "record_id"], how="inner").merge(
        pss, on=["model_id", "dataset", "record_id"], how="left"
    )
    rows = []
    for (enc, model), g in m.groupby(["encoder", "model_id"]):
        if len(g) < 10:
            continue
        for target in ["psi", "pss"]:
            if target == "psi":
                sr = spearmanr(g["posix"], g["psi"])
                label = "semantic_psi"
            else:
                gg = g[["posix", "pss"]].dropna()
                if len(gg) < 10:
                    continue
                sr = spearmanr(gg["posix"], gg["pss"])
                label = "prosa_pss"
            rows.append({
                "encoder": enc, "model_id": model, "comparison": f"POSIX_vs_{label}",
                "n": len(g), "spearman_rho": sr.statistic, "p_value": sr.pvalue,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"])
    return out


# -----------------------------
# Reporting / plots
# -----------------------------
def latex_table(df: pd.DataFrame, path: Path, columns: Optional[List[str]] = None, float_format="%.3f"):
    if df.empty:
        path.write_text("% no rows\n", encoding="utf-8")
        return
    x = df[columns].copy() if columns else df.copy()
    path.write_text(x.to_latex(index=False, escape=True, float_format=lambda z: float_format % z), encoding="utf-8")


def make_plots(outdir: Path, pairwise: pd.DataFrame, agreement: pd.DataFrame, risk_bins: pd.DataFrame, prosa_cmp: pd.DataFrame):
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 1) PSI vs |Delta F1| by encoder (one figure, separate panels avoided by user preference;
    #    so create one figure per encoder).
    for enc, g in pairwise.groupby("encoder"):
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        ax.scatter(g["psi"], g["abs_delta_partial_f1"], s=14, alpha=.25)
        ax.set_xlabel("Sensitivity score")
        ax.set_ylabel(r"$|\Delta|$ partial entity F1")
        rho = spearmanr(g["psi"], g["abs_delta_partial_f1"], nan_policy="omit").statistic
        ax.set_title(f"{enc}: semantic/output drift vs extraction change (rho={rho:.3f})")
        fig.tight_layout()
        fig.savefig(outdir / f"fig_psi_f1_{enc}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 2) Practical quintile risk per encoder.
    for enc, g in risk_bins.groupby("encoder"):
        gg = g.groupby("psi_quintile", as_index=False)["p_abs_delta_ge_0.10"].mean()
        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        ax.plot(gg["psi_quintile"], gg["p_abs_delta_ge_0.10"], marker="o", linewidth=2)
        ax.set_xlabel("PSI quintile (1=lowest, 5=highest)")
        ax.set_ylabel(r"Pr($|\Delta F1| \geq 0.10$)")
        ax.set_xticks([1,2,3,4,5])
        ax.set_title(f"Operational risk stratification: {enc}")
        fig.tight_layout()
        fig.savefig(outdir / f"fig_practical_risk_{enc}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 3) Encoder pair agreement bars.
    if not agreement.empty:
        a = agreement.copy()
        a["pair"] = a["encoder_a"] + " vs " + a["encoder_b"]
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        ax.bar(a["pair"], a["spearman_rho"])
        ax.set_ylabel("Transition-level Spearman rho")
        ax.set_xlabel("Encoder pair")
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("Robustness of PSI to semantic encoder choice")
        fig.tight_layout()
        fig.savefig(outdir / "fig_encoder_agreement.pdf", bbox_inches="tight")
        plt.close(fig)


def build_summary(outdir: Path, inputs: pd.DataFrame, pairwise: pd.DataFrame, agreement: pd.DataFrame,
                  corr: pd.DataFrame, practical: pd.DataFrame, prosa_cmp: pd.DataFrame,
                  posix_cmp: Optional[pd.DataFrame]) -> Dict[str, Any]:
    return {
        "n_prediction_rows": int(len(inputs)),
        "models": sorted(inputs["model_id"].unique().tolist()),
        "datasets": sorted(inputs["dataset"].unique().tolist()),
        "conditions": sorted(inputs["condition"].unique().tolist()),
        "n_adjacent_transition_rows": int(len(pairwise)),
        "encoders": sorted(pairwise["encoder"].unique().tolist()),
        "encoder_agreement": agreement.to_dict("records"),
        "psi_f1_correlations_model_abs_partial": corr[
            (corr["level"] == "model") & (corr["target"] == "abs_delta_partial_f1")
        ].to_dict("records") if len(corr) else [],
        "practical_psi": practical.to_dict("records"),
        "semantic_vs_prosa": prosa_cmp.to_dict("records"),
        "posix_comparisons": [] if posix_cmp is None else posix_cmp.to_dict("records"),
        "notes": [
            "No new LLM generations were used.",
            "POSIX, when enabled, uses teacher-forced likelihood of already saved responses under P1-P6 prompts.",
            "The stochastic decoding / seed request cannot be answered without new generations and is therefore not fabricated here.",
        ],
    }


# -----------------------------
# Commands
# -----------------------------
def cmd_analyze(args):
    root = Path(args.root)
    outdir = Path(args.outdir or (root / "outputs" / "reviewer_posthoc"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"ROOT={root}")
    print(f"OUT ={outdir}")

    pred = augment_predictions_from_manifest(root, load_jsonl_predictions(root))
    metrics = load_metrics(root)
    df = merge_inputs(pred, metrics)
    df = add_structured_fields(df)
    print("Rows:", len(df), "models:", df.model_id.unique(), "datasets:", df.dataset.unique())

    # Parse coverage audit.
    audit = df.groupby(["model_id", "dataset", "parsed_source"], as_index=False).size()
    audit.to_csv(outdir / "parse_coverage.csv", index=False)

    specs = []
    for name in args.encoders.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in DEFAULT_ENCODERS:
            raise KeyError(f"Unknown encoder {name}; choices={list(DEFAULT_ENCODERS)}")
        z = DEFAULT_ENCODERS[name]
        if not Path(z["path"]).exists():
            raise FileNotFoundError(
                f"Encoder path missing: {z['path']}\nRun the supplied download_encoders command first."
            )
        specs.append(EncoderSpec(name=name, **z))

    adjacent_frames = []
    allpair_frames = []
    cache_dir = outdir / "embedding_cache"
    for spec in specs:
        emb = load_or_compute_embeddings(df, spec, cache_dir, args.device, args.embedding_batch_size)
        adjacent_frames.append(pairwise_sensitivity(df, spec.name, emb, all_pairs=False))
        allpair_frames.append(pairwise_sensitivity(df, spec.name, emb, all_pairs=True))
        del emb
        gc.collect()

    # Embedding-free controls.
    for baseline in ["set_jaccard", "token_jaccard"]:
        adjacent_frames.append(pairwise_sensitivity(df, baseline, None, all_pairs=False))
        allpair_frames.append(pairwise_sensitivity(df, baseline, None, all_pairs=True))

    pairwise = pd.concat(adjacent_frames, ignore_index=True)
    allpairs = pd.concat(allpair_frames, ignore_index=True)
    pairwise.to_csv(outdir / "sensitivity_adjacent_multi_encoder.csv", index=False)
    allpairs.to_csv(outdir / "sensitivity_allpairs_multi_encoder.csv", index=False)

    # Summary tables.
    summary = pairwise.groupby(["encoder", "model_id", "dataset"], as_index=False).agg(
        n=("psi", "size"), mean_psi=("psi", "mean"), median_psi=("psi", "median"),
        p95_psi=("psi", lambda x: x.quantile(.95)),
        mean_abs_delta_partial_f1=("abs_delta_partial_f1", "mean"),
    )
    summary.to_csv(outdir / "sensitivity_summary_multi_encoder.csv", index=False)

    agreement = encoder_agreement(pairwise)
    agreement.to_csv(outdir / "encoder_agreement.csv", index=False)
    rank_agree = transition_rank_agreement(pairwise)
    rank_agree.to_csv(outdir / "encoder_transition_rank_agreement.csv", index=False)

    field_doc, field_summary, field_agree = field_volatility_tables(pairwise)
    field_doc.to_csv(outdir / "field_volatility_multi_encoder.csv", index=False)
    field_summary.to_csv(outdir / "field_volatility_summary_multi_encoder.csv", index=False)
    field_agree.to_csv(outdir / "encoder_field_rank_agreement.csv", index=False)

    corr = psi_f1_correlations(pairwise, n_boot=args.bootstrap)
    corr.to_csv(outdir / "psi_f1_correlations_multi_encoder.csv", index=False)

    practical, risk_bins = practical_risk(pairwise)
    practical.to_csv(outdir / "psi_practical_utility.csv", index=False)
    risk_bins.to_csv(outdir / "psi_risk_by_quintile.csv", index=False)

    pss_doc, pss_summary = compute_prosa_pss(df)
    pss_doc.to_csv(outdir / "prosa_pss_document.csv", index=False)
    pss_summary.to_csv(outdir / "prosa_pss_summary.csv", index=False)
    prosa_cmp = compare_semantic_to_prosa(allpairs, pss_doc)
    prosa_cmp.to_csv(outdir / "semantic_psi_vs_prosa_pss.csv", index=False)

    # Compact LaTeX tables.
    latex_table(
        agreement,
        outdir / "table_encoder_robustness.tex",
        columns=["encoder_a", "encoder_b", "n", "spearman_rho", "kendall_tau", "top10pct_jaccard"],
    )
    latex_table(
        field_agree,
        outdir / "table_field_rank_robustness.tex",
        columns=["model_id", "dataset", "encoder_a", "encoder_b", "n_fields", "spearman_rho", "kendall_tau"],
    )
    model_corr = corr[(corr["level"] == "model") & (corr["target"] == "abs_delta_partial_f1")].copy()
    latex_table(
        model_corr,
        outdir / "table_psi_f1_by_encoder.tex",
        columns=["encoder", "model_id", "n", "spearman_rho", "ci95_low", "ci95_high", "p_holm"],
    )
    latex_table(
        practical,
        outdir / "table_psi_practical_utility.tex",
        columns=["encoder", "model_id", "threshold_abs_delta_f1", "n", "positive_rate", "auroc", "auprc"],
    )
    latex_table(
        prosa_cmp,
        outdir / "table_semantic_vs_prosa.tex",
        columns=["encoder", "model_id", "n", "spearman_rho_semantic_psi_vs_prosa_pss", "p_holm"],
    )

    if not args.no_plots:
        make_plots(outdir, pairwise, agreement, risk_bins, prosa_cmp)

    # Save cleaned index for exact POSIX command, but do not serialize nested objects to Parquet dependency.
    idx_cols = ["model_id", "dataset", "record_id", "condition", "partial_f1"]
    df[idx_cols].to_csv(outdir / "input_index.csv", index=False)

    summary_json = build_summary(outdir, df, pairwise, agreement, corr, practical, prosa_cmp, None)
    write_json(outdir / "reviewer_posthoc_summary.json", summary_json)
    print("\nDONE post-hoc analysis")
    print(outdir)


def cmd_posix(args):
    root = Path(args.root)
    outdir = Path(args.outdir or (root / "outputs" / "reviewer_posthoc"))
    outdir.mkdir(parents=True, exist_ok=True)

    pred = augment_predictions_from_manifest(root, load_jsonl_predictions(root))
    metrics = load_metrics(root)
    df = add_structured_fields(merge_inputs(pred, metrics))

    # ------------------------------------------------------------------
    # Exact original P1-P6 prompts reconstructed with the original
    # build_prompt() and independently verified 5526/5526 by SHA256
    # and prompt character count against experiment_jobs.jsonl.
    # This MUST happen before the POSIX recoverability audit below.
    # ------------------------------------------------------------------
    prompt_sidecar = outdir / "reconstructed_main_prompts.jsonl"
    if not prompt_sidecar.exists():
        raise FileNotFoundError(
            f"Exact prompt sidecar missing: {prompt_sidecar}"
        )

    exact_prompts = pd.read_json(prompt_sidecar, lines=True)

    exact_prompts["model_id"] = exact_prompts["model_id"].map(norm_model)
    exact_prompts["dataset"] = exact_prompts["dataset"].map(norm_dataset)
    exact_prompts["condition"] = exact_prompts["condition"].map(norm_condition)
    exact_prompts["record_id"] = exact_prompts["record_id"].astype(str)

    prompt_keys = ["model_id", "dataset", "record_id", "condition"]

    if len(exact_prompts) != 5526:
        raise RuntimeError(
            f"Expected 5526 exact prompts, got {len(exact_prompts)}"
        )

    if exact_prompts.duplicated(prompt_keys).any():
        raise RuntimeError("Duplicate keys in exact prompt sidecar")

    prompt_map = exact_prompts[
        prompt_keys + ["prompt"]
    ].rename(columns={"prompt": "user_prompt"})

    if "user_prompt" in df.columns:
        df = df.drop(columns=["user_prompt"])

    df = df.merge(
        prompt_map,
        on=prompt_keys,
        how="left",
        validate="one_to_one",
    )

    missing_exact = int(df["user_prompt"].isna().sum())
    empty_exact = int(
        df["user_prompt"].fillna("").astype(str).str.len().eq(0).sum()
    )

    print(
        f"Exact SHA256-audited prompts merged: "
        f"{len(df) - missing_exact}/{len(df)}; "
        f"missing={missing_exact}; empty={empty_exact}"
    )

    if missing_exact or empty_exact:
        raise RuntimeError(
            f"Exact prompt merge failed: missing={missing_exact}, "
            f"empty={empty_exact}"
        )

    model_paths = dict(DEFAULT_MODEL_PATHS)
    if args.model_path_json:
        model_paths.update(json.loads(Path(args.model_path_json).read_text(encoding="utf-8")))

    # Audit whether exact prompts/responses are recoverable before loading any 7-12B model.
    audit_rows = []
    for r in df.to_dict("records"):
        p, pk = get_prompt_from_row(r)
        resp, rs = get_response_from_row(r)
        audit_rows.append({
            "model_id": r["model_id"], "dataset": r["dataset"], "record_id": r["record_id"],
            "condition": r["condition"], "prompt_kind": pk, "has_prompt": p is not None,
            "response_source": rs, "has_response": bool(resp),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir / "posix_input_audit.csv", index=False)
    print(audit.groupby(["model_id", "prompt_kind", "response_source"]).size())
    if (~audit["has_prompt"]).any():
        bad = audit.loc[~audit["has_prompt"], ["model_id", "dataset", "record_id", "condition"]]
        raise RuntimeError(
            f"POSIX needs the original P1-P6 prompt/message text. Missing for {len(bad)} rows. "
            f"See {outdir/'posix_input_audit.csv'}. No fake reconstruction is attempted."
        )
    nonraw = ~audit["response_source"].astype(str).str.startswith("raw:")
    if nonraw.any() and not args.allow_reconstructed_response:
        raise RuntimeError(
            f"Exact POSIX also needs the original generated response token text. "
            f"{int(nonraw.sum())} rows only have reconstructed structured JSON. "
            f"See {outdir/'posix_input_audit.csv'}. Use --allow-reconstructed-response only for an explicitly labeled approximation."
        )

    if args.audit_only:
        print("\n" + "=" * 100)
        print("STRICT POSIX INPUT AUDIT PASSED")
        print("=" * 100)
        print(f"rows          : {len(audit)}")
        print(f"exact prompts : {int(audit['has_prompt'].sum())}")
        print(f"raw responses : {int(audit['response_source'].astype(str).str.startswith('raw:').sum())}")
        print("No LLM weights loaded.")
        return

    scores, posix_doc = run_posix(
        df, outdir, model_paths,
        batch_size=args.batch_size,
        device_map=args.device_map,
        all_pairs=not args.adjacent_only,
    )

    # Compare exact POSIX against previously computed semantic PSI and ProSA PSS.
    ap = outdir / "sensitivity_allpairs_multi_encoder.csv"
    pp = outdir / "prosa_pss_document.csv"
    if ap.exists() and pp.exists() and not posix_doc.empty:
        allpairs = pd.read_csv(ap)
        pss_doc = pd.read_csv(pp)
        cmp = compare_posix_with_others(posix_doc, allpairs, pss_doc)
        cmp.to_csv(outdir / "posix_vs_semantic_and_prosa.csv", index=False)
        latex_table(
            cmp, outdir / "table_posix_comparison.tex",
            columns=["encoder", "model_id", "comparison", "n", "spearman_rho", "p_holm"],
        )
    print("\nDONE exact teacher-forced POSIX")
    print(outdir)


def cmd_audit(args):
    root = Path(args.root)
    pred = augment_predictions_from_manifest(root, load_jsonl_predictions(root))
    metrics = load_metrics(root)
    df = merge_inputs(pred, metrics)
    print("PRED columns:")
    print("\n".join(map(str, pred.columns)))
    print("\nMETRIC columns:")
    print("\n".join(map(str, metrics.columns)))
    print("\nCounts main P1-P6:")
    print(df.groupby(["model_id", "dataset", "condition"]).size().to_string())
    print("\nPrompt/response recoverability:")
    a = []
    for r in df.to_dict("records"):
        _, pk = get_prompt_from_row(r)
        resp, rs = get_response_from_row(r)
        a.append((r["model_id"], pk, rs, bool(resp)))
    print(pd.DataFrame(a, columns=["model", "prompt_kind", "response_source", "has_response"]).groupby(
        ["model", "prompt_kind", "response_source", "has_response"]
    ).size().to_string())


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="/home/tahiti/PromptStressLab")
    p.add_argument("--outdir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="Inspect existing prediction/metric schema; no model loading")
    a.set_defaults(func=cmd_audit)

    a = sub.add_parser("analyze", help="All reviewer post-hoc analyses except exact POSIX")
    a.add_argument("--encoders", default="clip_vit_b32,mpnet,specter")
    a.add_argument("--device", default="cuda")
    a.add_argument("--embedding-batch-size", type=int, default=128)
    a.add_argument("--bootstrap", type=int, default=3000)
    a.add_argument("--no-plots", action="store_true")
    a.set_defaults(func=cmd_analyze)

    a = sub.add_parser("posix", help="Exact POSIX via teacher-forced likelihood; NO generate()")
    a.add_argument("--batch-size", type=int, default=2)
    a.add_argument("--device-map", default="auto")
    a.add_argument("--adjacent-only", action="store_true", help="Fast diagnostic; full reviewer result should omit this flag")
    a.add_argument("--model-path-json", default=None, help="Optional JSON mapping canonical model names to local paths")
    a.add_argument("--allow-reconstructed-response", action="store_true", help="Approximate POSIX only; default exact mode requires raw saved response text")
    a.add_argument("--audit-only", action="store_true", help="Verify exact POSIX inputs and exit before loading LLM weights")
    a.set_defaults(func=cmd_posix)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
