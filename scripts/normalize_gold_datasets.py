#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCIERC_EXPECTED_DOCS = 500
EBM_EXPECTED_DOCS = 4993
SCIER_EXPECTED_DOCS = 106
SCIER_EXPECTED_SENTENCE_ROWS = 7722


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="ignore")
    ).hexdigest()


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not raw:
        return []

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            return [
                item
                for item in parsed
                if isinstance(item, dict)
            ]

        if isinstance(parsed, dict):
            return [parsed]

    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        raw.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)

        except json.JSONDecodeError as error:
            print(
                f"[JSON_ERROR] {path}:{line_number}: {error}",
                flush=True,
            )
            continue

        if isinstance(item, dict):
            records.append(item)

    return records


def tokens_to_text(
    tokens: list[str],
) -> tuple[str, list[tuple[int, int]]]:
    """
    Reconstruct text with one space between tokens.

    Returns:
      text
      character offsets [start, end) for every token
    """
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for index, token in enumerate(tokens):
        token = str(token)

        if index > 0:
            pieces.append(" ")
            cursor += 1

        start = cursor
        pieces.append(token)
        cursor += len(token)
        offsets.append((start, cursor))

    return "".join(pieces), offsets


def span_from_tokens(
    tokens: list[str],
    token_offsets: list[tuple[int, int]],
    start_token: int,
    end_token_inclusive: int,
) -> dict[str, Any] | None:
    if start_token < 0:
        return None

    if end_token_inclusive < start_token:
        return None

    if end_token_inclusive >= len(tokens):
        return None

    char_start = token_offsets[start_token][0]
    char_end = token_offsets[end_token_inclusive][1]

    return {
        "start_token": start_token,
        "end_token": end_token_inclusive,
        "start_char": char_start,
        "end_char": char_end,
        "text": " ".join(
            tokens[start_token : end_token_inclusive + 1]
        ),
    }


def flatten_nested(items: Any) -> list[Any]:
    output: list[Any] = []

    if not isinstance(items, list):
        return output

    for group in items:
        if isinstance(group, list):
            output.extend(group)

    return output


def infer_split_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]

    for split in ("test_ood", "train", "dev", "test"):
        if split in parts or split in path.stem.lower():
            return split

    return "unspecified"


# ============================================================
# SciERC
# ============================================================

def find_scierc_directory(root: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []

    for train_file in root.rglob("train.json"):
        parent = train_file.parent

        if not (parent / "dev.json").exists():
            continue

        if not (parent / "test.json").exists():
            continue

        lower_parts = {
            part.lower()
            for part in parent.parts
        }

        score = 0

        if "processed_data" in lower_parts:
            score += 100

        if "json" in lower_parts:
            score += 20

        candidates.append((score, parent))

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (item[0], str(item[1])),
    )[1]


def normalize_scierc(
    raw_root: Path,
    output_root: Path,
    errors: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    Counter[str],
]:
    source_directory = find_scierc_directory(raw_root)

    if source_directory is None:
        errors.append(
            {
                "dataset": "SciERC",
                "split": "",
                "record_id": "",
                "error": "processed train/dev/test directory not found",
            }
        )
        return [], Counter()

    print(
        f"[SCIERC_SOURCE] {source_directory}",
        flush=True,
    )

    all_records: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        input_path = source_directory / f"{split}.json"
        source_records = read_json_lines(input_path)
        normalized_records: list[dict[str, Any]] = []

        for index, source in enumerate(source_records):
            doc_id = str(
                source.get(
                    "doc_key",
                    f"scierc_{split}_{index:05d}",
                )
            )

            sentences = source.get("sentences", [])

            if not isinstance(sentences, list):
                sentences = []

            flat_tokens: list[str] = []
            sentence_ranges: list[dict[str, int]] = []

            token_cursor = 0

            for sentence_index, sentence in enumerate(sentences):
                if not isinstance(sentence, list):
                    continue

                sentence_tokens = [
                    str(token)
                    for token in sentence
                ]

                start_token = token_cursor
                flat_tokens.extend(sentence_tokens)
                token_cursor += len(sentence_tokens)

                sentence_ranges.append(
                    {
                        "sentence_index": sentence_index,
                        "start_token": start_token,
                        "end_token": token_cursor - 1,
                    }
                )

            text, token_offsets = tokens_to_text(flat_tokens)

            entity_by_span: dict[
                tuple[int, int],
                list[str],
            ] = defaultdict(list)

            entities: list[dict[str, Any]] = []

            for entity_index, item in enumerate(
                flatten_nested(source.get("ner", []))
            ):
                if not isinstance(item, list) or len(item) < 3:
                    continue

                start_token = int(item[0])
                end_token = int(item[1])
                entity_type = str(item[2])

                span = span_from_tokens(
                    flat_tokens,
                    token_offsets,
                    start_token,
                    end_token,
                )

                if span is None:
                    errors.append(
                        {
                            "dataset": "SciERC",
                            "split": split,
                            "record_id": doc_id,
                            "error": (
                                "invalid entity token span: "
                                f"{start_token}-{end_token}"
                            ),
                        }
                    )
                    continue

                entity_id = f"E{entity_index:04d}"

                entity = {
                    "id": entity_id,
                    "type": entity_type,
                    **span,
                }

                entities.append(entity)

                entity_by_span[
                    (start_token, end_token)
                ].append(entity_id)

                label_counts[
                    f"entity:{entity_type}"
                ] += 1

            relations: list[dict[str, Any]] = []

            for relation_index, item in enumerate(
                flatten_nested(source.get("relations", []))
            ):
                if not isinstance(item, list) or len(item) < 5:
                    continue

                head_start = int(item[0])
                head_end = int(item[1])
                tail_start = int(item[2])
                tail_end = int(item[3])
                relation_type = str(item[4])

                head_span = span_from_tokens(
                    flat_tokens,
                    token_offsets,
                    head_start,
                    head_end,
                )

                tail_span = span_from_tokens(
                    flat_tokens,
                    token_offsets,
                    tail_start,
                    tail_end,
                )

                if head_span is None or tail_span is None:
                    errors.append(
                        {
                            "dataset": "SciERC",
                            "split": split,
                            "record_id": doc_id,
                            "error": (
                                "invalid relation token span: "
                                f"{item}"
                            ),
                        }
                    )
                    continue

                head_ids = entity_by_span.get(
                    (head_start, head_end),
                    [],
                )

                tail_ids = entity_by_span.get(
                    (tail_start, tail_end),
                    [],
                )

                relations.append(
                    {
                        "id": f"R{relation_index:04d}",
                        "type": relation_type,
                        "head_entity_id": (
                            head_ids[0]
                            if head_ids
                            else None
                        ),
                        "tail_entity_id": (
                            tail_ids[0]
                            if tail_ids
                            else None
                        ),
                        "head": head_span,
                        "tail": tail_span,
                    }
                )

                label_counts[
                    f"relation:{relation_type}"
                ] += 1

            coreference_clusters: list[list[dict[str, Any]]] = []

            raw_clusters = source.get("clusters", [])

            if isinstance(raw_clusters, list):
                for cluster in raw_clusters:
                    if not isinstance(cluster, list):
                        continue

                    normalized_cluster = []

                    for mention in cluster:
                        if (
                            not isinstance(mention, list)
                            or len(mention) < 2
                        ):
                            continue

                        span = span_from_tokens(
                            flat_tokens,
                            token_offsets,
                            int(mention[0]),
                            int(mention[1]),
                        )

                        if span is not None:
                            normalized_cluster.append(span)

                    if normalized_cluster:
                        coreference_clusters.append(
                            normalized_cluster
                        )

            record = {
                "dataset": "SciERC",
                "split": split,
                "record_id": doc_id,
                "input_unit": "abstract",
                "input_text": text,
                "tokens": flat_tokens,
                "token_offsets": [
                    {
                        "start_char": start,
                        "end_char": end,
                    }
                    for start, end in token_offsets
                ],
                "sentence_ranges": sentence_ranges,
                "task_schema": {
                    "entity_types": [
                        "Task",
                        "Method",
                        "Metric",
                        "Material",
                        "OtherScientificTerm",
                        "Generic",
                    ],
                    "relation_types": [
                        "Compare",
                        "Conjunction",
                        "Evaluate-for",
                        "Used-for",
                        "Feature-of",
                        "Part-of",
                        "Hyponym-of",
                    ],
                },
                "gold": {
                    "entities": entities,
                    "relations": relations,
                    "coreference_clusters": coreference_clusters,
                },
                "statistics": {
                    "characters": len(text),
                    "tokens": len(flat_tokens),
                    "sentences": len(sentence_ranges),
                    "entities": len(entities),
                    "relations": len(relations),
                },
                "text_sha256": sha256_text(text),
                "source": {
                    "file": str(input_path),
                    "source_doc_key": doc_id,
                },
            }

            normalized_records.append(record)
            all_records.append(record)

        write_jsonl(
            output_root / f"{split}.jsonl",
            normalized_records,
        )

        print(
            f"[SCIERC_{split.upper()}] "
            f"{len(normalized_records)} documents",
            flush=True,
        )

    write_jsonl(
        output_root / "all.jsonl",
        all_records,
    )

    return all_records, label_counts


# ============================================================
# EBM-NLP
# ============================================================

def extract_pmid(path: Path) -> str | None:
    match = re.match(r"(\d+)", path.name)
    return match.group(1) if match else None


def find_ebm_documents_directory(root: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []

    for directory in root.rglob("documents"):
        if not directory.is_dir():
            continue

        token_count = len(
            list(directory.glob("*.tokens"))
        )

        if token_count:
            candidates.append(
                (token_count, directory)
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (item[0], str(item[1])),
    )[1]


def read_ebm_tokens(path: Path) -> list[str]:
    raw_lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    while raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    return raw_lines


def read_integer_labels(path: Path) -> list[int]:
    labels: list[int] = []

    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            labels.append(int(line))

        except ValueError:
            continue

    return labels


def ebm_element_from_path(path: Path) -> str | None:
    parts = {
        part.lower()
        for part in path.parts
    }

    if "participants" in parts:
        return "Participant"

    if "interventions" in parts:
        return "Intervention"

    if "outcomes" in parts:
        return "Outcome"

    return None


def ebm_split_from_path(path: Path) -> str:
    parts = {
        part.lower()
        for part in path.parts
    }

    if "test" in parts:
        return "test"

    if "train" in parts:
        return "train"

    return "unspecified"


def find_ebm_annotations(
    root: Path,
) -> dict[
    str,
    dict[str, dict[str, Path]],
]:
    """
    Result:
      PMID -> split -> element -> annotation path
    """
    index: dict[
        str,
        dict[str, dict[str, Path]],
    ] = defaultdict(
        lambda: defaultdict(dict)
    )

    for path in root.rglob("*.ann"):
        lower_parts = {
            part.lower()
            for part in path.parts
        }

        if "aggregated" not in lower_parts:
            continue

        if "starting_spans" not in lower_parts:
            continue

        pmid = extract_pmid(path)
        element = ebm_element_from_path(path)
        split = ebm_split_from_path(path)

        if not pmid or not element:
            continue

        existing = index[pmid][split].get(element)

        # Prefer the explicit aggregated file if duplicates exist.
        if existing is None:
            index[pmid][split][element] = path

        elif "aggregated" in path.name.lower():
            index[pmid][split][element] = path

    return index


def condense_binary_spans(
    labels: list[int],
    tokens: list[str],
    token_offsets: list[tuple[int, int]],
    entity_type: str,
    start_entity_index: int,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    index = 0
    entity_index = start_entity_index

    while index < len(labels):
        if labels[index] == 0:
            index += 1
            continue

        start = index
        label_value = labels[index]

        while (
            index + 1 < len(labels)
            and labels[index + 1] == label_value
            and labels[index + 1] != 0
        ):
            index += 1

        end = index

        span = span_from_tokens(
            tokens,
            token_offsets,
            start,
            end,
        )

        if span is not None:
            entities.append(
                {
                    "id": f"E{entity_index:04d}",
                    "type": entity_type,
                    **span,
                }
            )
            entity_index += 1

        index += 1

    return entities


def normalize_ebm_nlp(
    raw_root: Path,
    output_root: Path,
    errors: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    Counter[str],
]:
    documents_directory = find_ebm_documents_directory(
        raw_root
    )

    if documents_directory is None:
        errors.append(
            {
                "dataset": "EBM-NLP",
                "split": "",
                "record_id": "",
                "error": "documents directory not found",
            }
        )
        return [], Counter()

    print(
        f"[EBM_DOCUMENTS] {documents_directory}",
        flush=True,
    )

    annotation_index = find_ebm_annotations(
        raw_root
    )

    print(
        f"[EBM_ANNOTATED_PMIDS] {len(annotation_index)}",
        flush=True,
    )

    all_records: list[dict[str, Any]] = []
    by_split: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    label_counts: Counter[str] = Counter()

    token_files = sorted(
        documents_directory.glob("*.tokens")
    )

    for token_file in token_files:
        pmid = token_file.stem
        tokens = read_ebm_tokens(token_file)

        if not tokens:
            errors.append(
                {
                    "dataset": "EBM-NLP",
                    "split": "",
                    "record_id": pmid,
                    "error": "empty token file",
                }
            )
            continue

        text, token_offsets = tokens_to_text(tokens)

        split_candidates = sorted(
            annotation_index.get(pmid, {}).keys()
        )

        if len(split_candidates) == 1:
            split = split_candidates[0]

        elif "test" in split_candidates:
            split = "test"

        elif "train" in split_candidates:
            split = "train"

        else:
            split = "unspecified"

        annotations = annotation_index.get(
            pmid,
            {},
        ).get(
            split,
            {},
        )

        entities: list[dict[str, Any]] = []
        valid_elements: list[str] = []
        invalid_elements: list[str] = []

        for element in (
            "Participant",
            "Intervention",
            "Outcome",
        ):
            annotation_path = annotations.get(element)

            if annotation_path is None:
                invalid_elements.append(element)
                continue

            labels = read_integer_labels(
                annotation_path
            )

            if len(labels) != len(tokens):
                # A trailing empty token can occasionally cause
                # a one-position mismatch.
                if (
                    len(tokens) == len(labels) + 1
                    and tokens[-1] == ""
                ):
                    tokens = tokens[:-1]
                    text, token_offsets = tokens_to_text(tokens)

                if len(labels) != len(tokens):
                    errors.append(
                        {
                            "dataset": "EBM-NLP",
                            "split": split,
                            "record_id": pmid,
                            "error": (
                                f"{element} label/token mismatch: "
                                f"labels={len(labels)}, "
                                f"tokens={len(tokens)}, "
                                f"file={annotation_path}"
                            ),
                        }
                    )
                    invalid_elements.append(element)
                    continue

            element_entities = condense_binary_spans(
                labels=labels,
                tokens=tokens,
                token_offsets=token_offsets,
                entity_type=element,
                start_entity_index=len(entities),
            )

            entities.extend(element_entities)
            valid_elements.append(element)

            label_counts[
                f"entity:{element}"
            ] += len(element_entities)

        record = {
            "dataset": "EBM-NLP",
            "split": split,
            "record_id": pmid,
            "input_unit": "abstract",
            "input_text": text,
            "tokens": tokens,
            "token_offsets": [
                {
                    "start_char": start,
                    "end_char": end,
                }
                for start, end in token_offsets
            ],
            "task_schema": {
                "entity_types": [
                    "Participant",
                    "Intervention",
                    "Outcome",
                ],
                "relation_types": [],
            },
            "gold": {
                "entities": entities,
                "relations": [],
            },
            "statistics": {
                "characters": len(text),
                "tokens": len(tokens),
                "entities": len(entities),
            },
            "annotation_status": {
                "valid_elements": valid_elements,
                "missing_or_invalid_elements": invalid_elements,
                "complete": (
                    len(valid_elements) == 3
                ),
            },
            "text_sha256": sha256_text(text),
            "source": {
                "token_file": str(token_file),
                "annotation_files": {
                    key: str(value)
                    for key, value in annotations.items()
                },
            },
        }

        all_records.append(record)
        by_split[split].append(record)

    for split, records in sorted(by_split.items()):
        write_jsonl(
            output_root / f"{split}.jsonl",
            records,
        )

        print(
            f"[EBM_{split.upper()}] "
            f"{len(records)} documents",
            flush=True,
        )

    write_jsonl(
        output_root / "all.jsonl",
        all_records,
    )

    return all_records, label_counts


# ============================================================
# SciER
# ============================================================

def parse_typed_entity(value: str) -> tuple[str, str | None]:
    value = str(value)

    if ":" not in value:
        return value, None

    text, entity_type = value.rsplit(":", 1)
    return text, entity_type


def normalize_scier_sentence_record(
    source: dict[str, Any],
    split: str,
    row_index: int,
    label_counts: Counter[str],
) -> dict[str, Any]:
    doc_id = str(
        source.get(
            "doc_id",
            f"scier_{split}_{row_index:06d}",
        )
    )

    sentence = normalize_space(
        str(source.get("sentence", ""))
    )

    entities: list[dict[str, Any]] = []

    raw_entities = source.get("ner", [])

    if isinstance(raw_entities, list):
        for entity_index, item in enumerate(raw_entities):
            if not isinstance(item, list) or len(item) < 2:
                continue

            entity_text = str(item[0])
            entity_type = str(item[1])

            entities.append(
                {
                    "id": f"E{entity_index:04d}",
                    "type": entity_type,
                    "text": entity_text,
                }
            )

            label_counts[
                f"entity:{entity_type}"
            ] += 1

    relations: list[dict[str, Any]] = []

    raw_relations = source.get("rel_plus")

    if not isinstance(raw_relations, list):
        raw_relations = source.get("rel", [])

    for relation_index, item in enumerate(raw_relations):
        if not isinstance(item, list) or len(item) < 3:
            continue

        head_text, head_type = parse_typed_entity(
            str(item[0])
        )

        relation_type = str(item[1])

        tail_text, tail_type = parse_typed_entity(
            str(item[2])
        )

        relations.append(
            {
                "id": f"R{relation_index:04d}",
                "type": relation_type,
                "head": {
                    "text": head_text,
                    "type": head_type,
                },
                "tail": {
                    "text": tail_text,
                    "type": tail_type,
                },
            }
        )

        label_counts[
            f"relation:{relation_type}"
        ] += 1

    record_id = f"{doc_id}::sent_{row_index:06d}"

    return {
        "dataset": "SciER",
        "split": split,
        "record_id": record_id,
        "document_id": doc_id,
        "input_unit": "sentence",
        "input_text": sentence,
        "task_schema": {
            "entity_types": [
                "Dataset",
                "Method",
                "Task",
            ],
            "relation_types": [],
        },
        "gold": {
            "entities": entities,
            "relations": relations,
        },
        "statistics": {
            "characters": len(sentence),
            "entities": len(entities),
            "relations": len(relations),
        },
        "text_sha256": sha256_text(sentence),
    }


def normalize_scier(
    raw_root: Path,
    sentence_output_root: Path,
    document_output_root: Path,
    errors: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    Counter[str],
]:
    llm_directory = raw_root / "LLM"

    if not llm_directory.exists():
        matches = [
            path
            for path in raw_root.rglob("LLM")
            if path.is_dir()
        ]

        if matches:
            llm_directory = matches[0]

    if not llm_directory.exists():
        errors.append(
            {
                "dataset": "SciER",
                "split": "",
                "record_id": "",
                "error": "LLM directory not found",
            }
        )
        return [], [], Counter()

    print(
        f"[SCIER_SOURCE] {llm_directory}",
        flush=True,
    )

    sentence_records_all: list[dict[str, Any]] = []
    document_records_all: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()

    split_files = {
        "train": llm_directory / "train.jsonl",
        "dev": llm_directory / "dev.jsonl",
        "test": llm_directory / "test.jsonl",
        "test_ood": llm_directory / "test_ood.jsonl",
    }

    for split, input_path in split_files.items():
        if not input_path.exists():
            errors.append(
                {
                    "dataset": "SciER",
                    "split": split,
                    "record_id": "",
                    "error": f"missing split file: {input_path}",
                }
            )
            continue

        source_records = read_json_lines(input_path)
        sentence_records: list[dict[str, Any]] = []

        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for row_index, source in enumerate(source_records):
            record = normalize_scier_sentence_record(
                source=source,
                split=split,
                row_index=row_index,
                label_counts=label_counts,
            )

            sentence_records.append(record)

            grouped[
                record["document_id"]
            ].append(record)

        document_records: list[dict[str, Any]] = []

        for document_id, sentences in sorted(grouped.items()):
            sentence_texts = [
                sentence["input_text"]
                for sentence in sentences
            ]

            input_text = "\n".join(sentence_texts)

            document_entities: list[dict[str, Any]] = []
            document_relations: list[dict[str, Any]] = []

            for sentence_index, sentence_record in enumerate(sentences):
                for entity in sentence_record["gold"]["entities"]:
                    document_entities.append(
                        {
                            **entity,
                            "sentence_index": sentence_index,
                            "source_record_id": sentence_record["record_id"],
                        }
                    )

                for relation in sentence_record["gold"]["relations"]:
                    document_relations.append(
                        {
                            **relation,
                            "sentence_index": sentence_index,
                            "source_record_id": sentence_record["record_id"],
                        }
                    )

            document_records.append(
                {
                    "dataset": "SciER",
                    "split": split,
                    "record_id": document_id,
                    "input_unit": "full_document",
                    "input_text": input_text,
                    "sentences": sentence_texts,
                    "task_schema": {
                        "entity_types": [
                            "Dataset",
                            "Method",
                            "Task",
                        ],
                        "relation_types": sorted(
                            {
                                relation["type"]
                                for relation
                                in document_relations
                            }
                        ),
                    },
                    "gold": {
                        "entities": document_entities,
                        "relations": document_relations,
                    },
                    "statistics": {
                        "characters": len(input_text),
                        "sentences": len(sentences),
                        "entities": len(document_entities),
                        "relations": len(document_relations),
                    },
                    "text_sha256": sha256_text(input_text),
                }
            )

        write_jsonl(
            sentence_output_root / f"{split}.jsonl",
            sentence_records,
        )

        write_jsonl(
            document_output_root / f"{split}.jsonl",
            document_records,
        )

        sentence_records_all.extend(sentence_records)
        document_records_all.extend(document_records)

        print(
            f"[SCIER_{split.upper()}] "
            f"sentences={len(sentence_records)} "
            f"documents={len(document_records)}",
            flush=True,
        )

    write_jsonl(
        sentence_output_root / "all.jsonl",
        sentence_records_all,
    )

    write_jsonl(
        document_output_root / "all.jsonl",
        document_records_all,
    )

    return (
        sentence_records_all,
        document_records_all,
        label_counts,
    )


# ============================================================
# Reports
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(
    dataset: str,
    level: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        groups[
            str(record.get("split", "unspecified"))
        ].append(record)

    rows: list[dict[str, Any]] = []

    for split, group in sorted(groups.items()):
        rows.append(
            {
                "dataset": dataset,
                "level": level,
                "split": split,
                "records": len(group),
                "characters": sum(
                    len(record.get("input_text", ""))
                    for record in group
                ),
                "entities": sum(
                    len(
                        record.get(
                            "gold",
                            {},
                        ).get(
                            "entities",
                            [],
                        )
                    )
                    for record in group
                ),
                "relations": sum(
                    len(
                        record.get(
                            "gold",
                            {},
                        ).get(
                            "relations",
                            [],
                        )
                    )
                    for record in group
                ),
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/home/tahiti/PromptStressLab",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    raw_root = root / "data" / "raw"
    normalized_root = root / "data" / "normalized"
    reports_root = normalized_root / "reports"

    errors: list[dict[str, Any]] = []

    print(
        "=== GOLD DATASET NORMALIZATION ===",
        flush=True,
    )
    print(f"root={root}", flush=True)

    scierc_records, scierc_labels = normalize_scierc(
        raw_root=raw_root / "scierc",
        output_root=normalized_root / "scierc",
        errors=errors,
    )

    ebm_records, ebm_labels = normalize_ebm_nlp(
        raw_root=raw_root / "ebm_nlp",
        output_root=normalized_root / "ebm_nlp",
        errors=errors,
    )

    (
        scier_sentence_records,
        scier_document_records,
        scier_labels,
    ) = normalize_scier(
        raw_root=raw_root / "scier" / "SciER",
        sentence_output_root=(
            normalized_root
            / "scier"
            / "sentence"
        ),
        document_output_root=(
            normalized_root
            / "scier"
            / "document"
        ),
        errors=errors,
    )

    summary_rows: list[dict[str, Any]] = []

    summary_rows.extend(
        summarize_records(
            "SciERC",
            "abstract",
            scierc_records,
        )
    )

    summary_rows.extend(
        summarize_records(
            "EBM-NLP",
            "abstract",
            ebm_records,
        )
    )

    summary_rows.extend(
        summarize_records(
            "SciER",
            "sentence",
            scier_sentence_records,
        )
    )

    summary_rows.extend(
        summarize_records(
            "SciER",
            "document",
            scier_document_records,
        )
    )

    write_csv(
        reports_root / "normalization_summary.csv",
        summary_rows,
        [
            "dataset",
            "level",
            "split",
            "records",
            "characters",
            "entities",
            "relations",
        ],
    )

    write_csv(
        reports_root / "normalization_errors.csv",
        errors,
        [
            "dataset",
            "split",
            "record_id",
            "error",
        ],
    )

    inventory_rows = []

    for dataset, counts in (
        ("SciERC", scierc_labels),
        ("EBM-NLP", ebm_labels),
        ("SciER", scier_labels),
    ):
        for label, count in sorted(counts.items()):
            inventory_rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "count": count,
                }
            )

    write_csv(
        reports_root / "label_inventory.csv",
        inventory_rows,
        [
            "dataset",
            "label",
            "count",
        ],
    )

    validation_rows = [
        {
            "dataset": "SciERC",
            "unit": "documents",
            "expected": SCIERC_EXPECTED_DOCS,
            "actual": len(scierc_records),
            "status": (
                "PASS"
                if len(scierc_records)
                == SCIERC_EXPECTED_DOCS
                else "WARN"
            ),
        },
        {
            "dataset": "EBM-NLP",
            "unit": "documents",
            "expected": EBM_EXPECTED_DOCS,
            "actual": len(ebm_records),
            "status": (
                "PASS"
                if len(ebm_records)
                == EBM_EXPECTED_DOCS
                else "WARN"
            ),
        },
        {
            "dataset": "SciER",
            "unit": "documents",
            "expected": SCIER_EXPECTED_DOCS,
            "actual": len(scier_document_records),
            "status": (
                "PASS"
                if len(scier_document_records)
                == SCIER_EXPECTED_DOCS
                else "WARN"
            ),
        },
        {
            "dataset": "SciER",
            "unit": "sentence_rows",
            "expected": SCIER_EXPECTED_SENTENCE_ROWS,
            "actual": len(scier_sentence_records),
            "status": (
                "PASS"
                if len(scier_sentence_records)
                == SCIER_EXPECTED_SENTENCE_ROWS
                else "WARN"
            ),
        },
        {
            "dataset": "ALL",
            "unit": "normalization_errors",
            "expected": 0,
            "actual": len(errors),
            "status": (
                "PASS"
                if len(errors) == 0
                else "WARN"
            ),
        },
    ]

    write_csv(
        reports_root / "normalization_validation.csv",
        validation_rows,
        [
            "dataset",
            "unit",
            "expected",
            "actual",
            "status",
        ],
    )

    metadata = {
        "root": str(root),
        "normalized_root": str(normalized_root),
        "counts": {
            "SciERC_documents": len(scierc_records),
            "EBM_NLP_documents": len(ebm_records),
            "SciER_sentence_rows": len(
                scier_sentence_records
            ),
            "SciER_documents": len(
                scier_document_records
            ),
            "normalization_errors": len(errors),
        },
    }

    (
        reports_root
        / "normalization_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== NORMALIZED COUNTS ===", flush=True)
    print(
        f"SciERC documents: {len(scierc_records)}",
        flush=True,
    )
    print(
        f"EBM-NLP documents: {len(ebm_records)}",
        flush=True,
    )
    print(
        f"SciER sentence rows: "
        f"{len(scier_sentence_records)}",
        flush=True,
    )
    print(
        f"SciER documents: "
        f"{len(scier_document_records)}",
        flush=True,
    )
    print(
        f"Normalization errors: {len(errors)}",
        flush=True,
    )
    print(
        "=== GOLD_NORMALIZATION_FINISHED ===",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
