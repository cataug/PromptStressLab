#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from psl_common import multiset_f1, normalize_key, read_jsonl


def latest_by_job(paths: list[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for record in read_jsonl(path):
            job_id = record.get("job_id")
            if isinstance(job_id, str):
                output[job_id] = record
    return output


def load_gold_index(jobs: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted({Path(job["source_file"]) for job in jobs}):
        for record in read_jsonl(path):
            index[(record["dataset"], record["split"], record["record_id"])] = record
    return index


def entity_tuples(value: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(value, dict) or not isinstance(value.get("entities"), list):
        return []
    output = []
    for item in value["entities"]:
        if not isinstance(item, dict):
            continue
        text = normalize_key(item.get("text", ""))
        entity_type = normalize_key(item.get("type", ""))
        if text and entity_type:
            output.append((text, entity_type))
    return output


def relation_tuples(value: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    if not isinstance(value, dict) or not isinstance(value.get("relations"), list):
        return []
    output = []
    for item in value["relations"]:
        if not isinstance(item, dict):
            continue
        head = normalize_key(item.get("head", ""))
        relation_type = normalize_key(item.get("type", ""))
        tail = normalize_key(item.get("tail", ""))
        if head and relation_type and tail:
            output.append((head, relation_type, tail))
    return output


def gold_entity_tuples(record: dict[str, Any]) -> list[tuple[str, str]]:
    output = []
    for item in record.get("gold", {}).get("entities", []):
        if not isinstance(item, dict):
            continue
        text = normalize_key(item.get("text", ""))
        entity_type = normalize_key(item.get("type", ""))
        if text and entity_type:
            output.append((text, entity_type))
    return output


def gold_relation_tuples(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    output = []
    for item in record.get("gold", {}).get("relations", []):
        if not isinstance(item, dict):
            continue
        head_value = item.get("head", {})
        tail_value = item.get("tail", {})
        head = normalize_key(head_value.get("text", "") if isinstance(head_value, dict) else head_value)
        tail = normalize_key(tail_value.get("text", "") if isinstance(tail_value, dict) else tail_value)
        relation_type = normalize_key(item.get("type", ""))
        if head and relation_type and tail:
            output.append((head, relation_type, tail))
    return output


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def token_jaccard(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def partial_entity_f1(predicted: list[tuple[str, str]], gold: list[tuple[str, str]], threshold: float = 0.5) -> dict[str, float | int]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, (pred_text, pred_type) in enumerate(predicted):
        for gold_index, (gold_text, gold_type) in enumerate(gold):
            if pred_type != gold_type:
                continue
            score = token_jaccard(pred_text, gold_text)
            if score >= threshold:
                candidates.append((score, pred_index, gold_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    true_positive = 0
    for score, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        true_positive += 1
    predicted_count = len(predicted)
    gold_count = len(gold)
    precision = true_positive / predicted_count if predicted_count else (1.0 if gold_count == 0 else 0.0)
    recall = true_positive / gold_count if gold_count else (1.0 if predicted_count == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "predicted": predicted_count,
        "gold": gold_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def substring_supported(entity_text: str, input_text: str) -> bool:
    entity = normalize_key(entity_text)
    source = normalize_key(input_text)
    return bool(entity and entity in source)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def aggregate(rows: list[dict[str, Any]], group_fields: list[str], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    metric_fields = [
        "json_valid", "schema_valid", "entity_exact_f1", "entity_partial_f1",
        "relation_exact_f1", "hallucinated_entity_rate", "truncated_at_limit",
        "prompt_tokens", "output_tokens", "seconds_per_item_in_batch",
    ]
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        result = {field: value for field, value in zip(group_fields, key)}
        result["documents"] = len(group)
        for metric in metric_fields:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            mean_value = sum(values) / len(values) if values else math.nan
            low, high = bootstrap_ci(values, seed=seed + abs(hash((key, metric))) % 100000)
            result[f"{metric}_mean"] = mean_value
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/tahiti/PromptStressLab")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    jobs = read_jsonl(root / "manifests" / "experiment_jobs.jsonl")
    job_by_id = {job["job_id"]: job for job in jobs}
    gold_index = load_gold_index(jobs)

    prediction_paths = sorted((root / "outputs" / "generations").glob("*/predictions.jsonl"))
    predictions = latest_by_job(prediction_paths)

    rows: list[dict[str, Any]] = []
    for job_id, prediction in sorted(predictions.items()):
        job = job_by_id.get(job_id)
        if job is None:
            continue
        gold_record = gold_index[(job["dataset"], job["split"], job["record_id"])]
        parsed = prediction.get("parsed_output")
        pred_entities = entity_tuples(parsed)
        gold_entities = gold_entity_tuples(gold_record)
        pred_relations = relation_tuples(parsed)
        gold_relations = gold_relation_tuples(gold_record)

        exact_entity = multiset_f1(pred_entities, gold_entities)
        partial_entity = partial_entity_f1(pred_entities, gold_entities)
        exact_relation = multiset_f1(pred_relations, gold_relations)

        unsupported = sum(
            1 for entity_text, entity_type in pred_entities
            if not substring_supported(entity_text, gold_record["input_text"])
        )
        hallucinated_rate = unsupported / len(pred_entities) if pred_entities else 0.0
        validation = prediction.get("validation", {})

        rows.append(
            {
                "job_id": job_id,
                "model_id": job["model_id"],
                "dataset": job["dataset"],
                "split": job["split"],
                "record_id": job["record_id"],
                "phase": job["phase"],
                "condition": job["condition"],
                "physical_or_alias": "physical",
                "json_valid": int(bool(validation.get("json_valid"))),
                "schema_valid": int(bool(validation.get("schema_valid"))),
                "entity_exact_precision": exact_entity["precision"],
                "entity_exact_recall": exact_entity["recall"],
                "entity_exact_f1": exact_entity["f1"],
                "entity_partial_precision": partial_entity["precision"],
                "entity_partial_recall": partial_entity["recall"],
                "entity_partial_f1": partial_entity["f1"],
                "relation_exact_precision": exact_relation["precision"],
                "relation_exact_recall": exact_relation["recall"],
                "relation_exact_f1": exact_relation["f1"],
                "predicted_entities": len(pred_entities),
                "gold_entities": len(gold_entities),
                "predicted_relations": len(pred_relations),
                "gold_relations": len(gold_relations),
                "unsupported_entities": unsupported,
                "hallucinated_entity_rate": hallucinated_rate,
                "truncated_at_limit": int(bool(prediction.get("truncated_at_limit"))),
                "prompt_tokens": prediction.get("prompt_tokens", 0),
                "output_tokens": prediction.get("output_tokens", 0),
                "seconds_per_item_in_batch": prediction.get("seconds_per_item_in_batch", 0.0),
            }
        )

    ablation_ids = {
        (row["dataset"], row["split"], row["record_id"])
        for row in csv.DictReader(
            (root / "manifests" / "ablation_sample.csv").open("r", encoding="utf-8")
        )
    }
    aliases: list[dict[str, Any]] = []
    for row in rows:
        key = (row["dataset"], row["split"], row["record_id"])
        if key not in ablation_ids:
            continue
        alias_condition = {"P3": "A0", "P4": "A1"}.get(row["condition"])
        if alias_condition is None:
            continue
        alias = dict(row)
        alias["job_id"] = f"alias::{row['job_id']}::{alias_condition}"
        alias["phase"] = "ablation_alias"
        alias["condition"] = alias_condition
        alias["physical_or_alias"] = "alias"
        aliases.append(alias)

    rows_with_aliases = rows + aliases
    metrics_dir = root / "outputs" / "metrics"
    fields = list(rows_with_aliases[0].keys()) if rows_with_aliases else ["job_id"]
    write_csv(metrics_dir / "job_metrics_physical.csv", rows, fields)
    write_csv(metrics_dir / "job_metrics_with_aliases.csv", rows_with_aliases, fields)

    aggregate_rows = aggregate(
        rows_with_aliases,
        group_fields=["model_id", "dataset", "phase", "condition"],
        seed=20260717,
    )
    aggregate_fields = list(aggregate_rows[0].keys()) if aggregate_rows else ["model_id"]
    write_csv(metrics_dir / "aggregate_metrics.csv", aggregate_rows, aggregate_fields)

    global_rows = aggregate(
        rows_with_aliases,
        group_fields=["model_id", "phase", "condition"],
        seed=20260717,
    )
    global_fields = list(global_rows[0].keys()) if global_rows else ["model_id"]
    write_csv(metrics_dir / "aggregate_metrics_global.csv", global_rows, global_fields)

    summary = {
        "physical_jobs_expected": 8526,
        "physical_predictions_found": len(rows),
        "aliases_materialized": len(aliases),
        "logical_rows": len(rows_with_aliases),
        "logical_rows_expected_when_complete": 9726,
        "prediction_files": [str(path) for path in prediction_paths],
    }
    (metrics_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== EVALUATION FINISHED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
