#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TEXT_KEYS = (
    "text",
    "abstract",
    "content",
    "document",
    "article",
    "input",
    "sentences",
    "tokens",
    "paragraphs",
    "sections",
)

ID_KEYS = (
    "doc_id",
    "document_id",
    "paper_id",
    "pmid",
    "id",
    "doc_key",
    "filename",
)

ENTITY_KEYS = (
    "entities",
    "entity",
    "ner",
    "mentions",
    "spans",
)

RELATION_KEYS = (
    "relations",
    "relation",
    "rels",
)

SPLIT_WORDS = (
    "train",
    "dev",
    "valid",
    "validation",
    "test_ood",
    "test",
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def calculate_text_hash(text: str) -> str:
    normalized = normalize_space(text)
    return hashlib.sha256(
        normalized.encode("utf-8", errors="ignore")
    ).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def infer_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()

    for split in SPLIT_WORDS:
        if split in stem or split in parts:
            if split in {"valid", "validation"}:
                return "dev"
            return split

    return "unspecified"


def scalar_to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    return ""


def value_to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        if not value:
            return ""

        if all(isinstance(item, str) for item in value):
            return " ".join(value)

        chunks = []

        for item in value:
            if isinstance(item, dict):
                chunk = extract_text(item)
            else:
                chunk = value_to_text(item)

            if chunk:
                chunks.append(chunk)

        return "\n".join(chunks)

    if isinstance(value, dict):
        return extract_text(value)

    return ""


def extract_text(obj: Any) -> str:
    if isinstance(obj, str):
        return normalize_space(obj)

    if isinstance(obj, list):
        return normalize_space(value_to_text(obj))

    if not isinstance(obj, dict):
        return ""

    for key in TEXT_KEYS:
        if key not in obj:
            continue

        candidate = value_to_text(obj[key])

        if candidate:
            return normalize_space(candidate)

    for key in (
        "source",
        "example",
        "instance",
        "data",
        "payload",
    ):
        if key not in obj:
            continue

        candidate = extract_text(obj[key])

        if candidate:
            return candidate

    return ""


def extract_id(
    obj: Any,
    fallback: str,
) -> str:
    if isinstance(obj, dict):
        for key in ID_KEYS:
            if key not in obj:
                continue

            value = scalar_to_text(obj[key]).strip()

            if value:
                return value

    return fallback


def maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped:
        return value

    if stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except Exception:
        return value


def list_like_count(value: Any) -> int:
    value = maybe_parse_json(value)

    if isinstance(value, list):
        if value and all(
            isinstance(item, list)
            for item in value
        ):
            return sum(len(item) for item in value)

        return len(value)

    if isinstance(value, dict):
        return len(value)

    return 0


def recursive_named_count(
    obj: Any,
    names: Iterable[str],
) -> int:
    target_names = set(names)
    total = 0

    obj = maybe_parse_json(obj)

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in target_names:
                total += list_like_count(value)
            else:
                total += recursive_named_count(
                    value,
                    target_names,
                )

    elif isinstance(obj, list):
        for value in obj:
            total += recursive_named_count(
                value,
                target_names,
            )

    return total


def collect_labels_from_value(
    value: Any,
    labels: set[str],
) -> None:
    value = maybe_parse_json(value)

    if isinstance(value, dict):
        for key in (
            "type",
            "label",
            "relation",
            "relation_type",
            "entity_type",
            "category",
        ):
            label = value.get(key)

            if isinstance(label, str) and label.strip():
                labels.add(label.strip())

        for nested_value in value.values():
            collect_labels_from_value(
                nested_value,
                labels,
            )

    elif isinstance(value, list):
        # Common format:
        # [start_index, end_index, label]
        if (
            len(value) >= 3
            and isinstance(value[-1], str)
        ):
            labels.add(value[-1].strip())

        for nested_value in value:
            collect_labels_from_value(
                nested_value,
                labels,
            )


def collect_labels(obj: Any) -> list[str]:
    labels: set[str] = set()
    obj = maybe_parse_json(obj)

    if not isinstance(obj, dict):
        return []

    relevant_keys = set(
        ENTITY_KEYS
        + RELATION_KEYS
    )

    for key, value in obj.items():
        if key.lower() in relevant_keys:
            collect_labels_from_value(
                value,
                labels,
            )

        elif key.lower() in {
            "output",
            "gold",
            "target",
            "answer",
            "annotations",
        }:
            collect_labels_from_value(
                value,
                labels,
            )

    return sorted(
        label
        for label in labels
        if label
    )


def estimate_sentence_count(
    obj: Any,
    text: str,
) -> int:
    if (
        isinstance(obj, dict)
        and isinstance(obj.get("sentences"), list)
    ):
        return len(obj["sentences"])

    if not text:
        return 0

    pieces = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return max(
        1,
        len([piece for piece in pieces if piece]),
    )


def load_json_records(
    path: Path,
) -> list[Any]:
    raw = read_text(path).strip()

    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        records = []

        for line_number, line in enumerate(
            raw.splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except Exception as error:
                records.append(
                    {
                        "_parse_error": (
                            f"line {line_number}: {error}"
                        ),
                        "_raw": line,
                    }
                )

        return records

    try:
        parsed = json.loads(raw)

    except Exception:
        records = []

        for line_number, line in enumerate(
            raw.splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except Exception as error:
                records.append(
                    {
                        "_parse_error": (
                            f"line {line_number}: {error}"
                        ),
                        "_raw": line,
                    }
                )

        return records

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        for key in (
            "documents",
            "data",
            "records",
            "examples",
            "instances",
        ):
            if isinstance(parsed.get(key), list):
                return parsed[key]

        return [parsed]

    return []


def create_base_row(
    dataset: str,
    split: str,
    doc_id: str,
    source_path: Path,
    text: str,
    document_type: str,
    schema_name: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "doc_id": str(doc_id),
        "document_type": document_type,
        "schema_name": schema_name,
        "gold_available": 1,
        "source_path": str(source_path),
        "text": text,
        "char_count": len(text),
        "token_count_whitespace": (
            len(text.split())
            if text
            else 0
        ),
        "sentence_count": 0,
        "entity_count": 0,
        "relation_count": 0,
        "annotation_file_count": 0,
        "label_types": "",
        "text_sha256": (
            calculate_text_hash(text)
            if text
            else ""
        ),
        "parse_status": (
            "ok"
            if text
            else "missing_text"
        ),
        "eligible": (
            1
            if text
            else 0
        ),
        "length_bucket": "",
        "notes": "",
    }


def parse_structured_dataset(
    dataset: str,
    files: list[Path],
    root: Path,
    document_type: str,
    schema_name: str,
) -> list[dict[str, Any]]:
    rows = []
    seen_ids: Counter[str] = Counter()

    for path in sorted(files):
        records = load_json_records(path)
        split = infer_split(path)

        try:
            relative_path = path.relative_to(root)
        except ValueError:
            relative_path = path

        for index, obj in enumerate(records):
            fallback_id = (
                f"{path.stem}_{index:06d}"
            )

            doc_id = extract_id(
                obj,
                fallback_id,
            )

            seen_ids[doc_id] += 1

            if seen_ids[doc_id] > 1:
                doc_id = (
                    f"{doc_id}__"
                    f"{seen_ids[doc_id]}"
                )

            text = extract_text(obj)

            row = create_base_row(
                dataset=dataset,
                split=split,
                doc_id=doc_id,
                source_path=relative_path,
                text=text,
                document_type=document_type,
                schema_name=schema_name,
            )

            if (
                isinstance(obj, dict)
                and obj.get("_parse_error")
            ):
                row["parse_status"] = (
                    "json_parse_error"
                )
                row["eligible"] = 0
                row["notes"] = obj["_parse_error"]

            row["sentence_count"] = (
                estimate_sentence_count(
                    obj,
                    text,
                )
            )

            row["entity_count"] = (
                recursive_named_count(
                    obj,
                    ENTITY_KEYS,
                )
            )

            row["relation_count"] = (
                recursive_named_count(
                    obj,
                    RELATION_KEYS,
                )
            )

            row["label_types"] = "|".join(
                collect_labels(obj)
            )

            rows.append(row)

    return rows


def discover_scierc(
    root: Path,
) -> list[dict[str, Any]]:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {
            ".json",
            ".jsonl",
        }
        and any(
            split in path.stem.lower()
            for split in (
                "train",
                "dev",
                "test",
            )
        )
    ]

    if not candidates:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".json",
                ".jsonl",
            }
        ]

    return parse_structured_dataset(
        dataset="SciERC",
        files=candidates,
        root=root,
        document_type="abstract",
        schema_name="scierc_entity_relation",
    )


def discover_scier(
    root: Path,
) -> list[dict[str, Any]]:
    llm_directory = root / "LLM"

    candidates = [
        path
        for path in llm_directory.glob("*.jsonl")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    if not candidates:
        candidates = [
            path
            for path in root.rglob("*.jsonl")
            if "LLM" in path.parts
            and not path.name.startswith(".")
        ]

    return parse_structured_dataset(
        dataset="SciER",
        files=candidates,
        root=root,
        document_type="full_document",
        schema_name="scier_entity_relation",
    )


def document_id_from_filename(
    path: Path,
) -> str:
    name = path.name

    for suffix in (
        ".tokens",
        ".txt",
        ".text",
        ".abstract",
    ):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


def annotation_key(
    path: Path,
) -> str:
    match = re.search(
        r"\d{4,}",
        path.stem,
    )

    if match:
        return match.group(0)

    return path.stem


def count_positive_annotation_lines(
    path: Path,
) -> int:
    try:
        lines = [
            line.strip()
            for line in read_text(path).splitlines()
            if line.strip()
        ]
    except Exception:
        return 0

    positive_count = 0

    for line in lines:
        parts = re.split(
            r"[\s,;]+",
            line,
        )

        if any(
            part in {
                "1",
                "B",
                "I",
                "B-P",
                "I-P",
                "B-I",
                "I-I",
                "B-O",
                "I-O",
            }
            for part in parts
        ):
            positive_count += 1

    return positive_count


def discover_ebm_nlp(
    root: Path,
) -> list[dict[str, Any]]:
    all_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    document_files = [
        path
        for path in all_files
        if path.suffix.lower() in {
            ".tokens",
            ".txt",
            ".text",
            ".abstract",
        }
        and any(
            part.lower() in {
                "documents",
                "document",
                "texts",
                "abstracts",
            }
            for part in path.parts
        )
    ]

    if not document_files:
        document_files = [
            path
            for path in all_files
            if path.suffix.lower() == ".tokens"
        ]

    document_file_set = set(document_files)

    annotation_index: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    for path in all_files:
        if path in document_file_set:
            continue

        key = annotation_key(path)
        annotation_index[key].append(path)

    rows = []

    for path in sorted(document_files):
        raw_text = read_text(path)

        text = normalize_space(
            raw_text.replace("\n", " ")
        )

        doc_id = document_id_from_filename(path)
        split = infer_split(path)

        try:
            relative_path = path.relative_to(root)
        except ValueError:
            relative_path = path

        row = create_base_row(
            dataset="EBM-NLP",
            split=split,
            doc_id=doc_id,
            source_path=relative_path,
            text=text,
            document_type="abstract",
            schema_name="ebm_pico",
        )

        matching_annotations = (
            annotation_index.get(
                doc_id,
                [],
            )
        )

        if not matching_annotations:
            numeric_id = re.search(
                r"\d{4,}",
                doc_id,
            )

            if numeric_id:
                matching_annotations = (
                    annotation_index.get(
                        numeric_id.group(0),
                        [],
                    )
                )

        labels = set()
        positive_annotations = 0

        for annotation_path in matching_annotations:
            path_parts = [
                part.lower()
                for part in annotation_path.parts
            ]

            label_aliases = {
                "participants": "participant",
                "participant": "participant",
                "interventions": "intervention",
                "intervention": "intervention",
                "comparators": "comparator",
                "comparator": "comparator",
                "outcomes": "outcome",
                "outcome": "outcome",
            }

            for raw_label, normalized_label in (
                label_aliases.items()
            ):
                if raw_label in path_parts:
                    labels.add(normalized_label)

            positive_annotations += (
                count_positive_annotation_lines(
                    annotation_path
                )
            )

        row["annotation_file_count"] = (
            len(matching_annotations)
        )

        # For EBM-NLP this is an audit signal,
        # not yet the final normalized PICO span count.
        row["entity_count"] = (
            positive_annotations
        )

        row["label_types"] = "|".join(
            sorted(labels)
        )

        row["sentence_count"] = (
            estimate_sentence_count(
                {},
                text,
            )
        )

        if not matching_annotations:
            row["notes"] = (
                "No annotation files matched by "
                "document ID; inspect EBM-NLP layout."
            )

        rows.append(row)

    return rows


def assign_length_buckets(
    rows: list[dict[str, Any]],
) -> None:
    by_dataset = defaultdict(list)

    for row in rows:
        if row["eligible"]:
            by_dataset[row["dataset"]].append(row)

    for dataset_rows in by_dataset.values():
        ordered = sorted(
            dataset_rows,
            key=lambda row: int(
                row["char_count"]
            ),
        )

        total = len(ordered)

        for rank, row in enumerate(ordered):
            fraction = (
                (rank + 1)
                / max(total, 1)
            )

            if fraction <= 1 / 3:
                bucket = "short"
            elif fraction <= 2 / 3:
                bucket = "medium"
            else:
                bucket = "long"

            row["length_bucket"] = bucket


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        fieldnames = (
            list(rows[0].keys())
            if rows
            else []
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = defaultdict(list)

    for row in rows:
        groups[
            (
                row["dataset"],
                row["split"],
            )
        ].append(row)

    output_rows = []

    for (
        dataset,
        split,
    ), group in sorted(groups.items()):
        lengths = [
            int(row["char_count"])
            for row in group
            if row["eligible"]
        ]

        output_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "documents": len(group),
                "eligible_documents": sum(
                    int(row["eligible"])
                    for row in group
                ),
                "missing_text": sum(
                    not bool(row["text"])
                    for row in group
                ),
                "mean_chars": (
                    round(
                        statistics.mean(lengths),
                        2,
                    )
                    if lengths
                    else 0
                ),
                "median_chars": (
                    round(
                        statistics.median(lengths),
                        2,
                    )
                    if lengths
                    else 0
                ),
                "min_chars": (
                    min(lengths)
                    if lengths
                    else 0
                ),
                "max_chars": (
                    max(lengths)
                    if lengths
                    else 0
                ),
                "entities": sum(
                    int(row["entity_count"])
                    for row in group
                ),
                "relations": sum(
                    int(row["relation_count"])
                    for row in group
                ),
                "annotation_files": sum(
                    int(
                        row[
                            "annotation_file_count"
                        ]
                    )
                    for row in group
                ),
            }
        )

    return output_rows


def build_validation(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks = []

    datasets = sorted(
        {
            row["dataset"]
            for row in rows
        }
    )

    for dataset in datasets:
        group = [
            row
            for row in rows
            if row["dataset"] == dataset
        ]

        gold_signal_count = sum(
            (
                int(row["entity_count"])
                + int(row["relation_count"])
                + int(
                    row["annotation_file_count"]
                )
            )
            > 0
            for row in group
        )

        checks.extend(
            [
                {
                    "dataset": dataset,
                    "check": "nonempty_manifest",
                    "value": len(group),
                    "status": (
                        "PASS"
                        if group
                        else "FAIL"
                    ),
                },
                {
                    "dataset": dataset,
                    "check": "eligible_documents",
                    "value": sum(
                        int(row["eligible"])
                        for row in group
                    ),
                    "status": (
                        "PASS"
                        if any(
                            row["eligible"]
                            for row in group
                        )
                        else "FAIL"
                    ),
                },
                {
                    "dataset": dataset,
                    "check": "missing_text",
                    "value": sum(
                        not bool(row["text"])
                        for row in group
                    ),
                    "status": (
                        "PASS"
                        if all(
                            row["text"]
                            for row in group
                        )
                        else "WARN"
                    ),
                },
                {
                    "dataset": dataset,
                    "check": "gold_signal_present",
                    "value": gold_signal_count,
                    "status": (
                        "PASS"
                        if gold_signal_count > 0
                        else "WARN"
                    ),
                },
            ]
        )

    hashes = defaultdict(list)

    for row in rows:
        if row["text_sha256"]:
            hashes[
                row["text_sha256"]
            ].append(row)

    duplicate_rows = sum(
        len(group)
        for group in hashes.values()
        if len(group) > 1
    )

    checks.append(
        {
            "dataset": "ALL",
            "check": (
                "rows_in_duplicate_text_groups"
            ),
            "value": duplicate_rows,
            "status": (
                "PASS"
                if duplicate_rows == 0
                else "WARN"
            ),
        }
    )

    return checks


def build_duplicates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hashes = defaultdict(list)

    for row in rows:
        if row["text_sha256"]:
            hashes[
                row["text_sha256"]
            ].append(row)

    output_rows = []

    for digest, group in hashes.items():
        if len(group) < 2:
            continue

        for row in group:
            output_rows.append(
                {
                    "text_sha256": digest,
                    "group_size": len(group),
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "doc_id": row["doc_id"],
                    "source_path": (
                        row["source_path"]
                    ),
                }
            )

    return output_rows


def build_label_inventory(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[
        tuple[str, str]
    ] = Counter()

    for row in rows:
        labels = str(
            row["label_types"]
        ).split("|")

        for label in labels:
            if label:
                counts[
                    (
                        row["dataset"],
                        label,
                    )
                ] += 1

    return [
        {
            "dataset": dataset,
            "label": label,
            "documents_with_label": count,
        }
        for (
            dataset,
            label,
        ), count in sorted(counts.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build PromptStressLab dataset "
            "manifests and validation reports."
        )
    )

    parser.add_argument(
        "--root",
        default=(
            "/home/tahiti/"
            "PromptStressLab"
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).expanduser().resolve()

    raw_directory = (
        root
        / "data"
        / "raw"
    )

    manifests_directory = (
        root
        / "manifests"
    )

    dataset_jobs = [
        (
            "SciERC",
            raw_directory / "scierc",
            discover_scierc,
        ),
        (
            "EBM-NLP",
            (
                raw_directory
                / "ebm_nlp"
                / "v2_00"
            ),
            discover_ebm_nlp,
        ),
        (
            "SciER",
            (
                raw_directory
                / "scier"
                / "SciER"
            ),
            discover_scier,
        ),
    ]

    print(
        "=== PromptStressLab "
        "manifest builder ==="
    )
    print(f"root={root}")

    all_rows = []
    dataset_rows = {}

    for (
        dataset_name,
        dataset_root,
        loader,
    ) in dataset_jobs:
        print()
        print(
            f"[{dataset_name}] "
            f"root={dataset_root}"
        )

        if not dataset_root.exists():
            print(
                f"[{dataset_name}_MISSING]"
            )
            dataset_rows[dataset_name] = []
            continue

        try:
            rows = loader(dataset_root)

        except Exception as error:
            print(
                f"[{dataset_name}_ERROR] "
                f"{type(error).__name__}: "
                f"{error}"
            )
            rows = []

        dataset_rows[dataset_name] = rows
        all_rows.extend(rows)

        print(
            f"[{dataset_name}_ROWS] "
            f"{len(rows)}"
        )

        print(
            f"[{dataset_name}_ELIGIBLE] "
            f"{sum(int(row['eligible']) for row in rows)}"
        )

    assign_length_buckets(all_rows)

    fields = [
        "dataset",
        "split",
        "doc_id",
        "document_type",
        "schema_name",
        "gold_available",
        "source_path",
        "text",
        "char_count",
        "token_count_whitespace",
        "sentence_count",
        "entity_count",
        "relation_count",
        "annotation_file_count",
        "label_types",
        "text_sha256",
        "parse_status",
        "eligible",
        "length_bucket",
        "notes",
    ]

    output_names = {
        "SciERC": "scierc_manifest.csv",
        "EBM-NLP": "ebm_nlp_manifest.csv",
        "SciER": "scier_manifest.csv",
    }

    for dataset_name, rows in (
        dataset_rows.items()
    ):
        write_csv(
            manifests_directory
            / output_names[dataset_name],
            rows,
            fields,
        )

    write_csv(
        manifests_directory
        / "unified_manifest.csv",
        all_rows,
        fields,
    )

    write_csv(
        manifests_directory
        / "manifest_summary.csv",
        build_summary(all_rows),
    )

    write_csv(
        manifests_directory
        / "manifest_validation.csv",
        build_validation(all_rows),
    )

    write_csv(
        manifests_directory
        / "duplicate_texts.csv",
        build_duplicates(all_rows),
        [
            "text_sha256",
            "group_size",
            "dataset",
            "split",
            "doc_id",
            "source_path",
        ],
    )

    write_csv(
        manifests_directory
        / "label_inventory.csv",
        build_label_inventory(all_rows),
        [
            "dataset",
            "label",
            "documents_with_label",
        ],
    )

    metadata = {
        "root": str(root),
        "rows": len(all_rows),
        "datasets": {
            dataset_name: len(rows)
            for dataset_name, rows
            in dataset_rows.items()
        },
        "eligible": {
            dataset_name: sum(
                int(row["eligible"])
                for row in rows
            )
            for dataset_name, rows
            in dataset_rows.items()
        },
        "outputs": [
            "scierc_manifest.csv",
            "ebm_nlp_manifest.csv",
            "scier_manifest.csv",
            "unified_manifest.csv",
            "manifest_summary.csv",
            "manifest_validation.csv",
            "duplicate_texts.csv",
            "label_inventory.csv",
        ],
    }

    metadata_path = (
        manifests_directory
        / "manifest_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== MANIFEST OUTPUTS ===")

    for output_name in metadata["outputs"]:
        path = (
            manifests_directory
            / output_name
        )

        size = (
            path.stat().st_size
            if path.exists()
            else 0
        )

        print(
            f"[WROTE] {path} "
            f"({size} bytes)"
        )

    print(
        f"[TOTAL_ROWS] {len(all_rows)}"
    )
    print(
        "=== MANIFEST_BUILD_FINISHED ==="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
