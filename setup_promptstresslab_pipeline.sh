#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"

mkdir -p \
    "$ROOT/config" \
    "$ROOT/scripts" \
    "$ROOT/manifests" \
    "$ROOT/logs/experiments" \
    "$ROOT/outputs/generations" \
    "$ROOT/outputs/metrics"

cat > "$ROOT/config/experiment.json" <<'JSONCFG'
{
  "project": "PromptStressLab",
  "seed": 20260717,
  "python_bin": "/home/tahiti/Forensics/.venv_forensics/bin/python",
  "poll_seconds": 30,
  "environment": {
    "HF_HOME": "/home/tahiti/.cache/huggingface",
    "HF_HUB_CACHE": "/home/tahiti/.cache/huggingface/hub",
    "OMP_NUM_THREADS": 4,
    "MKL_NUM_THREADS": 4
  },
  "models": [
    {
      "model_id": "mistral_7b_instruct_v03",
      "display_name": "Mistral-7B-Instruct-v0.3",
      "path": "/home/tahiti/PromptStressLab/models/Mistral-7B-Instruct-v0.3",
      "priority": 1,
      "min_free_gib": 34,
      "max_batch_size": 6,
      "batch_token_budget": 12000
    },
    {
      "model_id": "qwen3_8b",
      "display_name": "Qwen3-8B",
      "path": "/home/tahiti/PromptStressLab/models/Qwen3-8B",
      "priority": 2,
      "min_free_gib": 34,
      "max_batch_size": 4,
      "batch_token_budget": 10000
    },
    {
      "model_id": "gemma_3_12b_it",
      "display_name": "Gemma-3-12B-it",
      "path": "/home/tahiti/PromptStressLab/models/Gemma-3-12B-it",
      "priority": 3,
      "min_free_gib": 37,
      "max_batch_size": 2,
      "batch_token_budget": 7000
    }
  ]
}
JSONCFG

cat > "$ROOT/scripts/psl_common.py" <<'PYCOMMON'
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

MAIN_CONDITIONS = ["P1", "P2", "P3", "P4", "P5", "P6"]
NEW_ABLATION_CONDITIONS = ["A2", "A3", "A4", "A5", "A6"]
ALL_PHYSICAL_CONDITIONS = MAIN_CONDITIONS + NEW_ABLATION_CONDITIONS

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "SciERC": {
        "description": "scientific entity and relation extraction from an abstract",
        "entity_types": [
            "Task", "Method", "Metric", "Material",
            "OtherScientificTerm", "Generic",
        ],
        "relation_types": [
            "Compare", "Conjunction", "Evaluate-for", "Used-for",
            "Feature-of", "Part-of", "Hyponym-of",
        ],
        "entity_definitions": {
            "Task": "a research problem, objective, or task",
            "Method": "an algorithm, model, method, system, or procedure",
            "Metric": "an evaluation metric or quantitative criterion",
            "Material": "a dataset, corpus, language resource, sample, or other research material",
            "OtherScientificTerm": "a domain-specific scientific concept not covered by the other types",
            "Generic": "a generic scientific expression annotated by the benchmark",
        },
    },
    "EBM-NLP": {
        "description": "PICO-style span extraction from a biomedical abstract",
        "entity_types": ["Participant", "Intervention", "Outcome"],
        "relation_types": [],
        "entity_definitions": {
            "Participant": "the studied population, patients, eligibility group, or participant characteristics",
            "Intervention": "an intervention, treatment, comparator, exposure, or care strategy",
            "Outcome": "an outcome, endpoint, measurement, effect, or clinical result",
        },
    },
    "SciER": {
        "description": "scientific entity and relation extraction from a full scientific document",
        "entity_types": ["Dataset", "Method", "Task"],
        "relation_types": [],  # populated from train/dev by the manifest builder
        "entity_definitions": {
            "Dataset": "a named or described dataset, corpus, benchmark, or data collection",
            "Method": "a method, model, algorithm, architecture, system, or procedure",
            "Task": "a research problem, objective, prediction target, or evaluation task",
        },
    },
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_key(text: str) -> str:
    return normalize_space(text).casefold()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def compact_gold(record: dict[str, Any]) -> dict[str, Any]:
    gold = record.get("gold", {})
    entities_out: list[dict[str, str]] = []
    relations_out: list[dict[str, str]] = []

    for entity in gold.get("entities", []):
        if not isinstance(entity, dict):
            continue
        text = normalize_space(entity.get("text", ""))
        entity_type = normalize_space(entity.get("type", ""))
        if text and entity_type:
            entities_out.append({"text": text, "type": entity_type})

    for relation in gold.get("relations", []):
        if not isinstance(relation, dict):
            continue
        relation_type = normalize_space(relation.get("type", ""))
        head = relation.get("head", {})
        tail = relation.get("tail", {})
        head_text = normalize_space(head.get("text", "") if isinstance(head, dict) else head)
        tail_text = normalize_space(tail.get("text", "") if isinstance(tail, dict) else tail)
        if relation_type and head_text and tail_text:
            relations_out.append(
                {"head": head_text, "type": relation_type, "tail": tail_text}
            )

    return {"entities": entities_out, "relations": relations_out}


def infer_relation_types(records: Iterable[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for record in records:
        for relation in record.get("gold", {}).get("relations", []):
            if isinstance(relation, dict):
                relation_type = normalize_space(relation.get("type", ""))
                if relation_type:
                    values.add(relation_type)
    return sorted(values)


def render_output_schema(dataset: str, relation_types: list[str] | None = None) -> str:
    spec = DATASET_SPECS[dataset]
    entity_types = spec["entity_types"]
    relation_types = list(relation_types if relation_types is not None else spec["relation_types"])
    schema = {
        "entities": [
            {
                "text": "exact text span copied from the input",
                "type": " | ".join(entity_types),
            }
        ],
        "relations": [
            {
                "head": "exact extracted entity text",
                "type": " | ".join(relation_types) if relation_types else "",
                "tail": "exact extracted entity text",
            }
        ] if relation_types else [],
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def render_definitions(dataset: str, relation_types: list[str]) -> str:
    spec = DATASET_SPECS[dataset]
    lines = ["Entity-type definitions:"]
    for entity_type in spec["entity_types"]:
        lines.append(f"- {entity_type}: {spec['entity_definitions'][entity_type]}.")
    if relation_types:
        lines.append("Allowed relation types: " + ", ".join(relation_types) + ".")
        lines.append("A relation must connect two entities explicitly supported by the input.")
    return "\n".join(lines)


def render_demo(demo: dict[str, Any], index: int) -> str:
    output = demo["output"]
    return (
        f"Demonstration {index}\n"
        f"INPUT:\n{demo['input_text']}\n"
        f"OUTPUT:\n{json.dumps(output, ensure_ascii=False, separators=(',', ':'))}"
    )


def base_task(dataset: str, relation_types: list[str]) -> str:
    spec = DATASET_SPECS[dataset]
    entity_types = ", ".join(spec["entity_types"])
    relation_clause = (
        " Extract relations using only these relation types: " + ", ".join(relation_types) + "."
        if relation_types
        else " Return an empty relations list."
    )
    return (
        f"Perform {spec['description']}. Extract entities using only these types: {entity_types}."
        + relation_clause
    )


def grounding_rules() -> str:
    return "\n".join(
        [
            "Grounding rules:",
            "- Copy each entity wording from the input; do not paraphrase it.",
            "- Do not infer or invent entities or relations that are not explicitly supported.",
            "- Use an empty list when nothing is supported.",
            "- Every relation endpoint must correspond to an extracted entity.",
        ]
    )


def formatting_rules() -> str:
    return "\n".join(
        [
            "Output rules:",
            "- Return exactly one valid JSON object.",
            "- Use exactly the keys entities and relations.",
            "- Do not add markdown fences, prose, comments, or extra keys.",
            "- Preserve the requested array/object structure even when arrays are empty.",
        ]
    )


def negative_examples_and_checklist() -> str:
    return "\n".join(
        [
            "Negative examples:",
            "- Do not output a broad topic that is absent as a literal or clearly delimited span.",
            "- Do not convert an author interpretation into an entity.",
            "- Do not create a relation merely because two entities occur in the same sentence.",
            "Final checklist:",
            "1. Every entity text occurs in the input.",
            "2. Every entity type is allowed.",
            "3. Every relation type is allowed.",
            "4. Every relation endpoint appears in entities.",
            "5. The response parses as JSON and contains no surrounding text.",
        ]
    )


def compressed_unique_rules(dataset: str, relation_types: list[str]) -> str:
    spec = DATASET_SPECS[dataset]
    entity_types = ", ".join(spec["entity_types"])
    relation_text = ", ".join(relation_types) if relation_types else "none"
    return (
        "Return only JSON with entities and relations. Copy supported spans exactly; never infer missing "
        f"items. Entity types: {entity_types}. Relation types: {relation_text}. Empty categories are []. "
        "Relation endpoints must be extracted entities. Before answering, verify grounding, allowed labels, "
        "endpoint consistency, and JSON validity."
    )


def repeated_rules_to_length(seed_text: str, target_chars: int) -> str:
    rules = [
        "Only copy spans that occur in the input.",
        "Do not invent unsupported entities.",
        "Do not invent unsupported relations.",
        "Return an empty array rather than guessing.",
        "Use only the allowed labels.",
        "Return JSON without markdown or prose.",
        "Ensure relation endpoints are extracted entities.",
    ]
    pieces = [seed_text]
    cursor = 0
    while len("\n".join(pieces)) < target_chars:
        pieces.append(f"Repeated verification rule: {rules[cursor % len(rules)]}")
        cursor += 1
    return "\n".join(pieces)


def build_prompt(
    *,
    dataset: str,
    record: dict[str, Any],
    condition: str,
    demos: list[dict[str, Any]],
    relation_types: list[str],
) -> str:
    task = base_task(dataset, relation_types)
    schema = render_output_schema(dataset, relation_types)
    text = record["input_text"]

    minimal = f"{task}\nReturn JSON matching this schema:\n{schema}\nINPUT:\n{text}"
    p2 = f"{task}\n{formatting_rules()}\nSchema:\n{schema}\nINPUT:\n{text}"
    p3 = f"{task}\n{formatting_rules()}\n{grounding_rules()}\nSchema:\n{schema}\nINPUT:\n{text}"

    one_demo = render_demo(demos[0], 1) if demos else ""
    three_demo_text = "\n\n".join(render_demo(demo, i + 1) for i, demo in enumerate(demos[:3]))

    p4 = f"{task}\n{formatting_rules()}\n{grounding_rules()}\n{one_demo}\nSchema:\n{schema}\nINPUT:\n{text}"
    definitions = render_definitions(dataset, relation_types)
    p5 = (
        f"{task}\n{formatting_rules()}\n{grounding_rules()}\n{definitions}\n"
        f"{three_demo_text}\nSchema:\n{schema}\nINPUT:\n{text}"
    )
    p6 = (
        f"{task}\n{formatting_rules()}\n{grounding_rules()}\n{definitions}\n"
        f"{negative_examples_and_checklist()}\n"
        "Additional mandatory reminders:\n"
        "- Follow every rule literally.\n- Recheck all copied spans.\n- Recheck all labels.\n"
        "- Recheck all relation endpoints.\n- Recheck the JSON syntax.\n"
        "- Never include an explanation before or after the JSON.\n"
        f"{three_demo_text}\nSchema:\n{schema}\nINPUT:\n{text}"
    )

    if condition == "P1":
        return minimal
    if condition == "P2":
        return p2
    if condition == "P3":
        return p3
    if condition == "P4":
        return p4
    if condition == "P5":
        return p5
    if condition == "P6":
        return p6
    if condition == "A2":
        return (
            f"{task}\n{formatting_rules()}\n{grounding_rules()}\n{three_demo_text}\n"
            f"Schema:\n{schema}\nINPUT:\n{text}"
        )
    if condition == "A3":
        return (
            f"{task}\n{formatting_rules()}\n{grounding_rules()}\n{definitions}\n"
            f"Schema:\n{schema}\nINPUT:\n{text}"
        )
    if condition == "A4":
        return (
            f"{task}\n{formatting_rules()}\n{grounding_rules()}\n"
            f"{negative_examples_and_checklist()}\nSchema:\n{schema}\nINPUT:\n{text}"
        )
    if condition == "A5":
        repeated = repeated_rules_to_length(
            f"{task}\n{formatting_rules()}\n{grounding_rules()}",
            target_chars=max(len(p6) - len(text), 1),
        )
        return f"{repeated}\nSchema:\n{schema}\nINPUT:\n{text}"
    if condition == "A6":
        return (
            f"{compressed_unique_rules(dataset, relation_types)}\n{definitions}\n"
            f"{negative_examples_and_checklist()}\n{three_demo_text}\n"
            f"Schema:\n{schema}\nINPUT:\n{text}"
        )
    raise ValueError(f"Unknown condition: {condition}")


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value, cleaned
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(cleaned[position:])
        except Exception:
            continue
        if isinstance(value, dict):
            return value, cleaned[position : position + consumed]
    return None, None


def validate_prediction(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "json_valid": False,
            "schema_valid": False,
            "entities_valid": False,
            "relations_valid": False,
        }
    entities = value.get("entities")
    relations = value.get("relations")
    entities_valid = isinstance(entities, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and isinstance(item.get("type"), str)
        for item in entities
    )
    relations_valid = isinstance(relations, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("head"), str)
        and isinstance(item.get("type"), str)
        and isinstance(item.get("tail"), str)
        for item in relations
    )
    exact_keys = set(value.keys()) == {"entities", "relations"}
    return {
        "json_valid": True,
        "schema_valid": bool(exact_keys and entities_valid and relations_valid),
        "entities_valid": bool(entities_valid),
        "relations_valid": bool(relations_valid),
        "exact_keys": bool(exact_keys),
    }


def multiset_f1(predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]]) -> dict[str, float | int]:
    pred_counts = Counter(predicted)
    gold_counts = Counter(gold)
    true_positive = sum((pred_counts & gold_counts).values())
    predicted_count = sum(pred_counts.values())
    gold_count = sum(gold_counts.values())
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
PYCOMMON

cat > "$ROOT/scripts/build_experiment_manifest.py" <<'PYMANIFEST'
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
PYMANIFEST

cat > "$ROOT/scripts/run_model_experiment.py" <<'PYWORKER'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

from psl_common import (
    build_prompt,
    extract_json_object,
    read_jsonl,
    stable_hash,
    validate_prediction,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "free_gib": round(free_bytes / 1024**3, 3),
        "total_gib": round(total_bytes / 1024**3, 3),
        "allocated_gib": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_job_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            job_id = value.get("job_id")
            if isinstance(job_id, str):
                output.add(job_id)
    return output


def load_record_index(jobs: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    paths = sorted({Path(job["source_file"]) for job in jobs})
    for path in paths:
        for record in read_jsonl(path):
            dataset = record["dataset"]
            split = record["split"]
            record_id = record["record_id"]
            index[(dataset, split, record_id)] = record
    return index


def load_tokenizer(model_path: Path) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as fast_error:
        print(f"[TOKENIZER_FAST_FAILED] {fast_error}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def model_class_for(model_id: str) -> Any:
    if model_id == "gemma_3_12b_it":
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is not None:
            return cls
    return transformers.AutoModelForCausalLM


def load_model(model_id: str, model_path: Path) -> Any:
    model_class = model_class_for(model_id)
    common = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    try:
        model = model_class.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            **common,
        )
    except TypeError:
        model = model_class.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            **common,
        )
    model.eval()
    return model


def render_chat(tokenizer: Any, prompt: str, model_id: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    if model_id == "qwen3_8b":
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def context_limit(model: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    config = getattr(model, "config", None)
    for obj in [config, getattr(config, "text_config", None)]:
        if obj is None:
            continue
        for name in ["max_position_embeddings", "sliding_window", "n_positions"]:
            value = getattr(obj, name, None)
            if isinstance(value, int) and 1024 <= value < 10**7:
                candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 1024 <= tokenizer_limit < 10**7:
        candidates.append(tokenizer_limit)
    return min(candidates) if candidates else 32768


def make_batches(items: list[dict[str, Any]], max_batch_size: int, token_budget: int) -> list[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[int(item["job"]["max_new_tokens"])].append(item)

    batches: list[list[dict[str, Any]]] = []
    for max_new_tokens, group in sorted(groups.items()):
        group.sort(key=lambda item: (item["prompt_tokens"], item["job"]["job_id"]))
        current: list[dict[str, Any]] = []
        current_max = 0
        for item in group:
            proposed_size = len(current) + 1
            proposed_max = max(current_max, item["prompt_tokens"])
            padded_input_cost = proposed_max * proposed_size
            generation_cost = max_new_tokens * proposed_size
            fits = (
                proposed_size <= max_batch_size
                and padded_input_cost + generation_cost <= token_budget
            )
            if current and not fits:
                batches.append(current)
                current = []
                current_max = 0
            current.append(item)
            current_max = max(current_max, item["prompt_tokens"])
        if current:
            batches.append(current)
    return batches


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/tahiti/PromptStressLab")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--gpu-physical-id", default="0")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config = load_json(root / "config" / "experiment.json")
    model_config = next(
        model for model in config["models"] if model["model_id"] == args.model_id
    )
    model_path = Path(model_config["path"]).resolve()

    output_dir = root / "outputs" / "generations" / args.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    run_metadata_path = output_dir / "run_metadata.json"

    all_jobs = [
        job
        for job in read_jsonl(root / "manifests" / "experiment_jobs.jsonl")
        if job["model_id"] == args.model_id
    ]
    successful = completed_job_ids(predictions_path)
    failed = set() if args.retry_errors else completed_job_ids(errors_path)
    pending_jobs = [
        job for job in all_jobs if job["job_id"] not in successful and job["job_id"] not in failed
    ]

    print("=== MODEL EXPERIMENT WORKER ===", flush=True)
    print(f"[MODEL_ID] {args.model_id}", flush=True)
    print(f"[MODEL_PATH] {model_path}", flush=True)
    print(f"[GPU_PHYSICAL_ID] {args.gpu_physical_id}", flush=True)
    print(f"[PYTHON] {sys.executable}", flush=True)
    print(f"[TORCH] {torch.__version__}", flush=True)
    print(f"[TRANSFORMERS] {transformers.__version__}", flush=True)
    print(f"[TOTAL_JOBS] {len(all_jobs)}", flush=True)
    print(f"[ALREADY_SUCCESSFUL] {len(successful)}", flush=True)
    print(f"[SKIPPED_ERRORS] {len(failed)}", flush=True)
    print(f"[PENDING] {len(pending_jobs)}", flush=True)
    print(f"[GPU_BEFORE_LOAD] {json.dumps(gpu_status())}", flush=True)

    if not pending_jobs:
        print("[MODEL_ALREADY_COMPLETE]", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    demos_manifest = load_json(root / "manifests" / "demo_manifest.json")
    record_index = load_record_index(pending_jobs)

    print("[LOAD_TOKENIZER]", flush=True)
    tokenizer_started = time.perf_counter()
    tokenizer = load_tokenizer(model_path)
    print(f"[TOKENIZER_OK] {time.perf_counter() - tokenizer_started:.3f}s", flush=True)

    print("[LOAD_MODEL]", flush=True)
    model_started = time.perf_counter()
    model = load_model(args.model_id, model_path)
    load_seconds = time.perf_counter() - model_started
    print(f"[MODEL_OK] {load_seconds:.3f}s", flush=True)
    print(f"[GPU_AFTER_LOAD] {json.dumps(gpu_status())}", flush=True)

    max_batch_size = int(args.max_batch_size or model_config["max_batch_size"])
    token_budget = int(args.token_budget or model_config["batch_token_budget"])
    model_context = context_limit(model, tokenizer)
    print(f"[MAX_BATCH_SIZE] {max_batch_size}", flush=True)
    print(f"[TOKEN_BUDGET] {token_budget}", flush=True)
    print(f"[MODEL_CONTEXT] {model_context}", flush=True)

    prepared: list[dict[str, Any]] = []
    pre_errors: list[dict[str, Any]] = []
    for index, job in enumerate(pending_jobs, start=1):
        key = (job["dataset"], job["split"], job["record_id"])
        record = record_index.get(key)
        if record is None:
            pre_errors.append({
                "job_id": job["job_id"],
                "model_id": args.model_id,
                "status": "error",
                "error_type": "MissingRecord",
                "error": f"No record found for {key}",
                "job": job,
                "created_at": utc_now(),
            })
            continue
        prompt = build_prompt(
            dataset=job["dataset"],
            record=record,
            condition=job["condition"],
            demos=demos_manifest["datasets"][job["dataset"]],
            relation_types=demos_manifest["relation_types"][job["dataset"]],
        )
        prompt_hash = stable_hash(prompt)
        if prompt_hash != job["prompt_sha256"]:
            pre_errors.append({
                "job_id": job["job_id"],
                "model_id": args.model_id,
                "status": "error",
                "error_type": "PromptHashMismatch",
                "error": f"expected={job['prompt_sha256']} actual={prompt_hash}",
                "job": job,
                "created_at": utc_now(),
            })
            continue
        rendered = render_chat(tokenizer, prompt, args.model_id)
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        prompt_tokens = len(token_ids)
        if prompt_tokens + int(job["max_new_tokens"]) > model_context:
            pre_errors.append({
                "job_id": job["job_id"],
                "model_id": args.model_id,
                "status": "error",
                "error_type": "ContextOverflow",
                "error": (
                    f"prompt_tokens={prompt_tokens} max_new_tokens={job['max_new_tokens']} "
                    f"context={model_context}"
                ),
                "job": job,
                "created_at": utc_now(),
            })
            continue
        prepared.append({
            "job": job,
            "record": record,
            "prompt": prompt,
            "rendered_prompt": rendered,
            "prompt_tokens": prompt_tokens,
        })
        if index % 250 == 0 or index == len(pending_jobs):
            print(f"[PREPARED] {index}/{len(pending_jobs)}", flush=True)

    with errors_path.open("a", encoding="utf-8") as error_handle:
        for error in pre_errors:
            append_record(error_handle, error)
    print(f"[PREPARATION_ERRORS] {len(pre_errors)}", flush=True)

    batches = make_batches(prepared, max_batch_size=max_batch_size, token_budget=token_budget)
    print(f"[BATCHES] {len(batches)}", flush=True)

    counters = {"ok": 0, "invalid_json": 0, "errors": len(pre_errors)}
    started_at = utc_now()
    worker_start = time.perf_counter()

    prediction_handle = predictions_path.open("a", encoding="utf-8")
    error_handle = errors_path.open("a", encoding="utf-8")

    def process_batch(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        max_new_tokens = int(batch[0]["job"]["max_new_tokens"])
        prompts = [item["rendered_prompt"] for item in batch]
        try:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            batch_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            batch_seconds = time.perf_counter() - batch_started
            for row_index, item in enumerate(batch):
                generated_tokens = generated[row_index, input_width:]
                raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                parsed, extracted_json = extract_json_object(raw_output)
                validation = validate_prediction(parsed)
                output_tokens = int(generated_tokens.shape[-1])
                status = "ok" if validation["schema_valid"] else "invalid_json"
                record = {
                    **item["job"],
                    "status": status,
                    "gpu_physical_id": str(args.gpu_physical_id),
                    "prompt_tokens": item["prompt_tokens"],
                    "output_tokens": output_tokens,
                    "max_new_tokens": max_new_tokens,
                    "truncated_at_limit": output_tokens >= max_new_tokens,
                    "batch_size": len(batch),
                    "batch_seconds": round(batch_seconds, 6),
                    "seconds_per_item_in_batch": round(batch_seconds / len(batch), 6),
                    "raw_output": raw_output,
                    "extracted_json_text": extracted_json,
                    "parsed_output": parsed,
                    "validation": validation,
                    "created_at": utc_now(),
                }
                append_record(prediction_handle, record)
                counters[status] += 1
            del generated
            del encoded
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            gc.collect()
            if len(batch) > 1:
                midpoint = len(batch) // 2
                print(f"[OOM_SPLIT] size={len(batch)} -> {midpoint}+{len(batch)-midpoint}", flush=True)
                process_batch(batch[:midpoint])
                process_batch(batch[midpoint:])
                return
            item = batch[0]
            error_record = {
                **item["job"],
                "status": "error",
                "error_type": "CUDAOutOfMemoryError",
                "error": str(error),
                "prompt_tokens": item["prompt_tokens"],
                "gpu_status": gpu_status(),
                "traceback": traceback.format_exc(),
                "created_at": utc_now(),
            }
            append_record(error_handle, error_record)
            counters["errors"] += 1
        except Exception as error:
            if len(batch) > 1:
                print(
                    f"[BATCH_ERROR_SPLIT] {type(error).__name__}: {error}; size={len(batch)}",
                    flush=True,
                )
                midpoint = len(batch) // 2
                process_batch(batch[:midpoint])
                process_batch(batch[midpoint:])
                return
            item = batch[0]
            error_record = {
                **item["job"],
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "prompt_tokens": item["prompt_tokens"],
                "gpu_status": gpu_status(),
                "traceback": traceback.format_exc(),
                "created_at": utc_now(),
            }
            append_record(error_handle, error_record)
            counters["errors"] += 1

    try:
        for batch_index, batch in enumerate(batches, start=1):
            process_batch(batch)
            if batch_index % 10 == 0 or batch_index == len(batches):
                elapsed = time.perf_counter() - worker_start
                done = counters["ok"] + counters["invalid_json"] + counters["errors"]
                print(
                    f"[PROGRESS] batches={batch_index}/{len(batches)} done={done}/{len(pending_jobs)} "
                    f"ok={counters['ok']} invalid={counters['invalid_json']} errors={counters['errors']} "
                    f"elapsed_s={elapsed:.1f} gpu={json.dumps(gpu_status())}",
                    flush=True,
                )
    finally:
        prediction_handle.close()
        error_handle.close()
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    metadata = {
        "model_id": args.model_id,
        "model_path": str(model_path),
        "gpu_physical_id": str(args.gpu_physical_id),
        "started_at": started_at,
        "finished_at": utc_now(),
        "load_seconds": round(load_seconds, 6),
        "pending_at_start": len(pending_jobs),
        "prepared": len(prepared),
        "preparation_errors": len(pre_errors),
        "counters": counters,
        "max_batch_size": max_batch_size,
        "token_budget": token_budget,
        "model_context": model_context,
        "environment": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "PYTORCH_CUDA_ALLOC_CONF",
            ]
        },
    }
    run_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_success_ids = completed_job_ids(predictions_path)
    final_error_ids = completed_job_ids(errors_path)
    final_successful = len(final_success_ids)
    final_errors = len(final_error_ids)
    final_completed_unique = len(final_success_ids | final_error_ids)
    print("=== MODEL WORKER FINISHED ===", flush=True)
    print(f"[FINAL_SUCCESSFUL_OR_INVALID] {final_successful}/{len(all_jobs)}", flush=True)
    print(f"[FINAL_ERRORS] {final_errors}", flush=True)
    print(f"[GPU_AFTER_CLEANUP] {json.dumps(gpu_status())}", flush=True)
    return 0 if final_successful >= len(all_jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
PYWORKER

cat > "$ROOT/scripts/gpu_scheduler.py" <<'PYSCHED'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psl_common import read_jsonl


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_job_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            job_id = value.get("job_id")
            if isinstance(job_id, str):
                output.add(job_id)
    return output


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "free_mib": int(parts[2]),
                "total_mib": int(parts[3]),
                "utilization": int(parts[4]),
                "free_gib": int(parts[2]) / 1024,
            }
        )
    return gpus


def parse_gpu_ids(value: str | None, available: list[int]) -> list[int]:
    if not value:
        return available
    requested = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    missing = [gpu_id for gpu_id in requested if gpu_id not in available]
    if missing:
        raise RuntimeError(f"Requested GPUs are unavailable: {missing}; available={available}")
    return requested


@dataclass
class RunningWorker:
    model_id: str
    gpu_id: int
    process: subprocess.Popen[str]
    log_handle: Any
    thread: threading.Thread


def stream_output(
    process: subprocess.Popen[str],
    model_id: str,
    gpu_id: int,
    log_handle: Any,
    line_queue: queue.Queue[tuple[str, str]],
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log_handle.write(line)
        log_handle.flush()
        line_queue.put((f"{model_id}|gpu{gpu_id}", line.rstrip("\n")))


def model_progress(root: Path, model_id: str, total: int) -> dict[str, int]:
    output_dir = root / "outputs" / "generations" / model_id
    predictions = completed_job_ids(output_dir / "predictions.jsonl")
    errors = completed_job_ids(output_dir / "errors.jsonl")
    return {
        "total": total,
        "predictions": len(predictions),
        "errors": len(errors),
        "remaining": max(total - len(predictions), 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/tahiti/PromptStressLab")
    parser.add_argument("--poll-seconds", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config = load_json(root / "config" / "experiment.json")
    jobs = read_jsonl(root / "manifests" / "experiment_jobs.jsonl")
    totals: dict[str, int] = {}
    for job in jobs:
        totals[job["model_id"]] = totals.get(job["model_id"], 0) + 1

    detected = query_gpus()
    available_ids = [gpu["index"] for gpu in detected]
    allowed_gpu_ids = parse_gpu_ids(os.environ.get("PSL_GPU_IDS"), available_ids)
    max_parallel = int(
        os.environ.get("PSL_MAX_PARALLEL_MODELS", str(max(len(allowed_gpu_ids), 1)))
    )
    poll_seconds = int(
        args.poll_seconds or os.environ.get("PSL_POLL_SECONDS", config.get("poll_seconds", 30))
    )
    max_utilization = int(os.environ.get("PSL_MAX_START_UTILIZATION", "35"))

    python_bin = config["python_bin"]
    worker_script = root / "scripts" / "run_model_experiment.py"
    logs_dir = root / "logs" / "experiments"
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_configs = sorted(config["models"], key=lambda item: int(item["priority"]))
    config_by_id = {item["model_id"]: item for item in model_configs}

    print("=== GPU-AWARE EXPERIMENT SCHEDULER ===", flush=True)
    print(f"[ROOT] {root}", flush=True)
    print(f"[PYTHON_BIN] {python_bin}", flush=True)
    print(f"[ALLOWED_GPUS] {allowed_gpu_ids}", flush=True)
    print(f"[MAX_PARALLEL_MODELS] {max_parallel}", flush=True)
    print(f"[POLL_SECONDS] {poll_seconds}", flush=True)
    print(f"[MAX_START_UTILIZATION] {max_utilization}", flush=True)

    running: dict[str, RunningWorker] = {}
    occupied_gpus: set[int] = set()
    attempted_and_failed: set[str] = set()
    line_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    while True:
        while True:
            try:
                prefix, line = line_queue.get_nowait()
            except queue.Empty:
                break
            print(f"[{prefix}] {line}", flush=True)

        finished_models: list[str] = []
        for model_id, worker in list(running.items()):
            return_code = worker.process.poll()
            if return_code is None:
                continue
            worker.thread.join(timeout=5)
            worker.log_handle.close()
            occupied_gpus.discard(worker.gpu_id)
            finished_models.append(model_id)
            progress = model_progress(root, model_id, totals[model_id])
            print(
                f"[WORKER_FINISHED] model={model_id} gpu={worker.gpu_id} code={return_code} "
                f"progress={json.dumps(progress)}",
                flush=True,
            )
            if return_code != 0 and progress["remaining"] > 0:
                attempted_and_failed.add(model_id)
            del running[model_id]

        progress_by_model = {
            model_id: model_progress(root, model_id, total)
            for model_id, total in totals.items()
        }
        complete_models = {
            model_id
            for model_id, progress in progress_by_model.items()
            if progress["remaining"] == 0
        }

        if len(complete_models) == len(totals):
            print("=== ALL MODEL JOBS COMPLETE ===", flush=True)
            print(json.dumps(progress_by_model, indent=2), flush=True)
            return 0

        runnable_models = [
            item
            for item in model_configs
            if item["model_id"] not in complete_models
            and item["model_id"] not in running
            and item["model_id"] not in attempted_and_failed
        ]

        try:
            gpu_state = {gpu["index"]: gpu for gpu in query_gpus()}
        except Exception as error:
            print(f"[GPU_QUERY_ERROR] {error}", flush=True)
            time.sleep(poll_seconds)
            continue

        free_slots = max_parallel - len(running)
        if free_slots > 0 and runnable_models:
            for gpu_id in allowed_gpu_ids:
                if free_slots <= 0:
                    break
                if gpu_id in occupied_gpus:
                    continue
                gpu = gpu_state.get(gpu_id)
                if gpu is None:
                    continue

                selected = None
                for model_config in runnable_models:
                    required = float(model_config["min_free_gib"])
                    if gpu["free_gib"] >= required and gpu["utilization"] <= max_utilization:
                        selected = model_config
                        break
                if selected is None:
                    continue

                model_id = selected["model_id"]
                command = [
                    python_bin,
                    "-u",
                    str(worker_script),
                    "--root",
                    str(root),
                    "--model-id",
                    model_id,
                    "--gpu-physical-id",
                    str(gpu_id),
                    "--retry-errors",
                ]
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": str(gpu_id),
                        "HF_HOME": config["environment"]["HF_HOME"],
                        "HF_HUB_CACHE": config["environment"]["HF_HUB_CACHE"],
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONUNBUFFERED": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "OMP_NUM_THREADS": str(config["environment"].get("OMP_NUM_THREADS", 4)),
                        "MKL_NUM_THREADS": str(config["environment"].get("MKL_NUM_THREADS", 4)),
                        "PYTHONPATH": str(root / "scripts") + os.pathsep + env.get("PYTHONPATH", ""),
                    }
                )
                log_path = logs_dir / f"{model_id}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                thread = threading.Thread(
                    target=stream_output,
                    args=(process, model_id, gpu_id, log_handle, line_queue),
                    daemon=True,
                )
                thread.start()
                running[model_id] = RunningWorker(
                    model_id=model_id,
                    gpu_id=gpu_id,
                    process=process,
                    log_handle=log_handle,
                    thread=thread,
                )
                occupied_gpus.add(gpu_id)
                runnable_models = [item for item in runnable_models if item["model_id"] != model_id]
                free_slots -= 1
                print(
                    f"[WORKER_STARTED] model={model_id} gpu={gpu_id} free_gib={gpu['free_gib']:.2f} "
                    f"required_gib={selected['min_free_gib']} remaining={progress_by_model[model_id]['remaining']}",
                    flush=True,
                )

        if not running and not runnable_models:
            if attempted_and_failed:
                print("=== SCHEDULER STOPPED WITH FAILED MODELS ===", flush=True)
                print(f"[FAILED_MODELS] {sorted(attempted_and_failed)}", flush=True)
                print(json.dumps(progress_by_model, indent=2), flush=True)
                return 1

        state_text = ", ".join(
            f"gpu{gpu_id}:free={gpu_state[gpu_id]['free_gib']:.1f}GiB util={gpu_state[gpu_id]['utilization']}%"
            for gpu_id in allowed_gpu_ids
            if gpu_id in gpu_state
        )
        remaining_text = ", ".join(
            f"{model_id}={progress['remaining']}"
            for model_id, progress in sorted(progress_by_model.items())
        )
        print(
            f"[SCHEDULER_HEARTBEAT] running={list(running)} | {state_text} | remaining: {remaining_text}",
            flush=True,
        )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
PYSCHED

cat > "$ROOT/scripts/evaluate_experiment.py" <<'PYEVAL'
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
PYEVAL

cat > "$ROOT/scripts/status.py" <<'PYSTATUS'
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
PYSTATUS

cat > "$ROOT/scripts/prepare_experiment.sh" <<'SHPREP'
#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

mkdir -p \
    "$ROOT/config" \
    "$ROOT/scripts" \
    "$ROOT/manifests" \
    "$ROOT/logs/experiments" \
    "$ROOT/outputs/generations" \
    "$ROOT/outputs/metrics"

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m py_compile \
    "$ROOT/scripts/psl_common.py" \
    "$ROOT/scripts/build_experiment_manifest.py" \
    "$ROOT/scripts/run_model_experiment.py" \
    "$ROOT/scripts/gpu_scheduler.py" \
    "$ROOT/scripts/evaluate_experiment.py" \
    "$ROOT/scripts/status.py"

COMPILE_STATUS="$?"

echo "[PY_COMPILE_STATUS] $COMPILE_STATUS"

if [ "$COMPILE_STATUS" -eq 0 ]; then
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/build_experiment_manifest.py" \
        --root "$ROOT" \
        2>&1 | tee "$ROOT/logs/build_experiment_manifest.log"
    BUILD_STATUS="${PIPESTATUS[0]}"
    echo "[MANIFEST_BUILD_STATUS] $BUILD_STATUS"
else
    echo "[MANIFEST_BUILD_SKIPPED] Python compilation failed"
fi

echo
cat "$ROOT/manifests/experiment_summary.json" 2>/dev/null
SHPREP

cat > "$ROOT/scripts/run_all_experiments.sh" <<'SHRUN'
#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export HF_HOME="/home/tahiti/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

export PSL_GPU_IDS="${PSL_GPU_IDS:-0}"
export PSL_MAX_PARALLEL_MODELS="${PSL_MAX_PARALLEL_MODELS:-1}"
export PSL_POLL_SECONDS="${PSL_POLL_SECONDS:-30}"
export PSL_MAX_START_UTILIZATION="${PSL_MAX_START_UTILIZATION:-20}"

mkdir -p "$ROOT/logs/experiments" "$ROOT/outputs/generations"

echo "=== LAUNCH CONFIGURATION ==="
echo "PYTHON_BIN=$PYTHON_BIN"
echo "PSL_GPU_IDS=$PSL_GPU_IDS"
echo "PSL_MAX_PARALLEL_MODELS=$PSL_MAX_PARALLEL_MODELS"
echo "PSL_POLL_SECONDS=$PSL_POLL_SECONDS"
echo "PSL_MAX_START_UTILIZATION=$PSL_MAX_START_UTILIZATION"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

echo
echo "=== CURRENT GPU ==="
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null

echo
echo "=== STARTING GPU-AWARE SCHEDULER ==="
"$PYTHON_BIN" -u \
    "$ROOT/scripts/gpu_scheduler.py" \
    --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/gpu_scheduler_console.log"

SCHEDULER_STATUS="${PIPESTATUS[0]}"
echo "[SCHEDULER_STATUS] $SCHEDULER_STATUS"
SHRUN

cat > "$ROOT/scripts/show_status.sh" <<'SHSTATUS'
#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u "$ROOT/scripts/status.py"
SHSTATUS

cat > "$ROOT/scripts/evaluate_when_complete.sh" <<'SHEVAL'
#!/usr/bin/env bash
ROOT="/home/tahiti/PromptStressLab"
PYTHON_BIN="/home/tahiti/Forensics/.venv_forensics/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u \
    "$ROOT/scripts/evaluate_experiment.py" \
    --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/evaluate_experiment.log"
STATUS="${PIPESTATUS[0]}"
echo "[EVALUATION_STATUS] $STATUS"
cat "$ROOT/outputs/metrics/evaluation_summary.json" 2>/dev/null
SHEVAL

chmod +x \
    "$ROOT/scripts/prepare_experiment.sh" \
    "$ROOT/scripts/run_all_experiments.sh" \
    "$ROOT/scripts/show_status.sh" \
    "$ROOT/scripts/evaluate_when_complete.sh"

echo "[PIPELINE_FILES_INSTALLED]"
echo "ROOT=$ROOT"
echo "Next: bash $ROOT/scripts/prepare_experiment.sh"
