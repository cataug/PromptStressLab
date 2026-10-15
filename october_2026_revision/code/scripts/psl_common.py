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
