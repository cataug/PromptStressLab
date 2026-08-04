#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from psl_common import (
    DATASET_SPECS,
    MAIN_CONDITIONS,
    NEW_ABLATION_CONDITIONS,
    build_prompt,
    compact_gold,
    infer_relation_types,
    read_jsonl,
    stable_hash,
    write_jsonl,
)

MODEL_IDS = ["qwen3_8b", "mistral_7b_instruct_v03", "gemma_3_12b_it"]
EXPECTED_PHYSICAL_JOBS = 8526


def load_sources(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    normalized = root / "data" / "normalized"
    return {
        "SciERC": {
            "train": read_jsonl(normalized / "scierc" / "train.jsonl"),
            "dev": read_jsonl(normalized / "scierc" / "dev.jsonl"),
            "test": read_jsonl(normalized / "scierc" / "test.jsonl"),
        },
        "EBM-NLP": {
            "train": read_jsonl(normalized / "ebm_nlp" / "train.jsonl"),
            "test": read_jsonl(normalized / "ebm_nlp" / "test.jsonl"),
            "unspecified": read_jsonl(normalized / "ebm_nlp" / "unspecified.jsonl"),
        },
        "SciER": {
            "sentence_train": read_jsonl(normalized / "scier" / "sentence" / "train.jsonl"),
            "document_train": read_jsonl(normalized / "scier" / "document" / "train.jsonl"),
            "document_dev": read_jsonl(normalized / "scier" / "document" / "dev.jsonl"),
            "test": read_jsonl(normalized / "scier" / "document" / "test.jsonl"),
            "test_ood": read_jsonl(normalized / "scier" / "document" / "test_ood.jsonl"),
        },
    }


def choose_demos(records: list[dict[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for record in records:
        gold = compact_gold(record)
        if not gold["entities"]:
            continue
        eligible.append(
            {
                "record_id": record["record_id"],
                "input_text": record["input_text"],
                "output": gold,
                "chars": len(record["input_text"]),
                "hash": stable_hash(record["record_id"]),
            }
        )
    if len(eligible) < count:
        raise RuntimeError(f"Only {len(eligible)} eligible demonstrations found")
    # Keep examples compact, then use a stable hash to avoid arbitrary file order.
    pool = sorted(eligible, key=lambda item: (item["chars"], item["hash"]))[:100]
    chosen = sorted(pool, key=lambda item: item["hash"])[:count]
    for item in chosen:
        item.pop("chars", None)
        item.pop("hash", None)
    return chosen


def quartile_stratified_sample(records: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    if total % 4 != 0:
        raise ValueError("This helper expects a sample size divisible by four")
    ordered = sorted(
        records,
        key=lambda record: (
            int(record.get("statistics", {}).get("entities", 0)),
            stable_hash(record["record_id"]),
        ),
    )
    strata: list[list[dict[str, Any]]] = []
    n = len(ordered)
    for index in range(4):
        start = round(index * n / 4)
        end = round((index + 1) * n / 4)
        strata.append(ordered[start:end])
    per_stratum = total // 4
    sample: list[dict[str, Any]] = []
    for stratum in strata:
        selected = sorted(stratum, key=lambda record: stable_hash(record["record_id"]))[:per_stratum]
        sample.extend(selected)
    if len(sample) != total:
        raise RuntimeError(f"Expected {total} sampled records, got {len(sample)}")
    return sorted(sample, key=lambda record: record["record_id"])


def source_path_for(root: Path, dataset: str, split: str) -> Path:
    normalized = root / "data" / "normalized"
    mapping = {
        ("SciERC", "test"): normalized / "scierc" / "test.jsonl",
        ("EBM-NLP", "test"): normalized / "ebm_nlp" / "test.jsonl",
        ("SciER", "test"): normalized / "scier" / "document" / "test.jsonl",
        ("SciER", "test_ood"): normalized / "scier" / "document" / "test_ood.jsonl",
    }
    return mapping[(dataset, split)]


def max_new_tokens(dataset: str) -> int:
    return {"SciERC": 768, "EBM-NLP": 768, "SciER": 2048}[dataset]


def make_job(
    *,
    root: Path,
    model_id: str,
    dataset: str,
    split: str,
    record: dict[str, Any],
    phase: str,
    condition: str,
    demos: list[dict[str, Any]],
    relation_types: list[str],
) -> dict[str, Any]:
    prompt = build_prompt(
        dataset=dataset,
        record=record,
        condition=condition,
        demos=demos,
        relation_types=relation_types,
    )
    identity = "|".join(
        [model_id, dataset, split, record["record_id"], phase, condition]
    )
    return {
        "job_id": stable_hash(identity)[:24],
        "model_id": model_id,
        "dataset": dataset,
        "split": split,
        "record_id": record["record_id"],
        "phase": phase,
        "condition": condition,
        "source_file": str(source_path_for(root, dataset, split)),
        "max_new_tokens": max_new_tokens(dataset),
        "prompt_sha256": stable_hash(prompt),
        "prompt_characters": len(prompt),
        "input_characters": len(record["input_text"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/tahiti/PromptStressLab")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    sources = load_sources(root)

    relation_types = {
        "SciERC": list(DATASET_SPECS["SciERC"]["relation_types"]),
        "EBM-NLP": [],
        "SciER": infer_relation_types(
            sources["SciER"]["sentence_train"]
            + sources["SciER"]["document_dev"]
        ),
    }
    if not relation_types["SciER"]:
        raise RuntimeError("No SciER relation types were discovered from train/dev")

    demos = {
        "SciERC": choose_demos(sources["SciERC"]["train"]),
        "EBM-NLP": choose_demos(sources["EBM-NLP"]["train"]),
        "SciER": choose_demos(sources["SciER"]["sentence_train"]),
    }

    demo_manifest = {
        "datasets": demos,
        "relation_types": relation_types,
        "selection_policy": "three compact deterministic train examples per dataset",
    }
    (manifests / "demo_manifest.json").write_text(
        json.dumps(demo_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    main_records: list[tuple[str, str, dict[str, Any]]] = []
    main_records.extend(("SciERC", "test", record) for record in sources["SciERC"]["test"])
    main_records.extend(("EBM-NLP", "test", record) for record in sources["EBM-NLP"]["test"])
    main_records.extend(("SciER", "test", record) for record in sources["SciER"]["test"])
    main_records.extend(("SciER", "test_ood", record) for record in sources["SciER"]["test_ood"])

    ebm_ablation = quartile_stratified_sample(sources["EBM-NLP"]["test"], 84)
    ablation_records: list[tuple[str, str, dict[str, Any]]] = []
    ablation_records.extend(("SciERC", "test", record) for record in sources["SciERC"]["test"])
    ablation_records.extend(("EBM-NLP", "test", record) for record in ebm_ablation)
    ablation_records.extend(("SciER", "test", record) for record in sources["SciER"]["test"])
    ablation_records.extend(("SciER", "test_ood", record) for record in sources["SciER"]["test_ood"])

    if len(main_records) != 307:
        raise RuntimeError(f"Main set must contain 307 documents, found {len(main_records)}")
    if len(ablation_records) != 200:
        raise RuntimeError(f"Ablation set must contain 200 documents, found {len(ablation_records)}")

    ablation_rows = [
        {
            "dataset": dataset,
            "split": split,
            "record_id": record["record_id"],
            "entity_count": int(record.get("statistics", {}).get("entities", 0)),
            "text_sha256": record.get("text_sha256", ""),
        }
        for dataset, split, record in ablation_records
    ]
    write_csv(
        manifests / "ablation_sample.csv",
        ablation_rows,
        ["dataset", "split", "record_id", "entity_count", "text_sha256"],
    )

    jobs: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for dataset, split, record in main_records:
            for condition in MAIN_CONDITIONS:
                jobs.append(
                    make_job(
                        root=root,
                        model_id=model_id,
                        dataset=dataset,
                        split=split,
                        record=record,
                        phase="main",
                        condition=condition,
                        demos=demos[dataset],
                        relation_types=relation_types[dataset],
                    )
                )
        for dataset, split, record in ablation_records:
            for condition in NEW_ABLATION_CONDITIONS:
                jobs.append(
                    make_job(
                        root=root,
                        model_id=model_id,
                        dataset=dataset,
                        split=split,
                        record=record,
                        phase="ablation",
                        condition=condition,
                        demos=demos[dataset],
                        relation_types=relation_types[dataset],
                    )
                )

    duplicate_ids = [job_id for job_id, count in Counter(job["job_id"] for job in jobs).items() if count > 1]
    if duplicate_ids:
        raise RuntimeError(f"Duplicate job IDs found: {duplicate_ids[:5]}")
    if len(jobs) != EXPECTED_PHYSICAL_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_PHYSICAL_JOBS} physical jobs, generated {len(jobs)}"
        )

    write_jsonl(manifests / "experiment_jobs.jsonl", jobs)

    by_model = Counter(job["model_id"] for job in jobs)
    by_phase = Counter(job["phase"] for job in jobs)
    by_dataset = Counter(job["dataset"] for job in jobs)
    by_condition = Counter(job["condition"] for job in jobs)

    summary = {
        "physical_jobs": len(jobs),
        "logical_results_after_A0_A1_aliases": 9726,
        "main_documents": len(main_records),
        "ablation_documents": len(ablation_records),
        "models": MODEL_IDS,
        "by_model": dict(sorted(by_model.items())),
        "by_phase": dict(sorted(by_phase.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_condition": dict(sorted(by_condition.items())),
        "main_conditions": MAIN_CONDITIONS,
        "new_ablation_conditions": NEW_ABLATION_CONDITIONS,
        "alias_policy": {"A0": "P3", "A1": "P4"},
        "relation_types": relation_types,
    }
    (manifests / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== EXPERIMENT MANIFEST BUILT ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WROTE] {manifests / 'experiment_jobs.jsonl'}")
    print(f"[PASS] physical_jobs={len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
