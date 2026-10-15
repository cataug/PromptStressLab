#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


FIELDS = [
    "dataset",
    "split",
    "doc_id",
    "document_type",
    "schema_name",
    "source_file",
    "text",
    "char_count",
    "word_count",
    "sentence_count",
    "entity_count",
    "relation_count",
    "annotation_file_count",
    "label_types",
    "text_sha256",
    "eligible",
    "length_bucket",
    "notes",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(
        normalize_text(text).encode("utf-8", errors="ignore")
    ).hexdigest()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not raw:
        return []

    # Most SciERC/SciER files are JSONL even when
    # the extension is .json.
    records: list[dict[str, Any]] = []

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            return [
                item for item in parsed
                if isinstance(item, dict)
            ]

        if isinstance(parsed, dict):
            return [parsed]

    except json.JSONDecodeError:
        pass

    for line_number, line in enumerate(
        raw.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            obj = json.loads(line)

        except json.JSONDecodeError as error:
            print(
                f"[JSON_ERROR] {path}:{line_number}: {error}",
                flush=True,
            )
            continue

        if isinstance(obj, dict):
            records.append(obj)

    return records


def make_row(
    *,
    dataset: str,
    split: str,
    doc_id: str,
    document_type: str,
    schema_name: str,
    source_file: str,
    text: str,
    sentence_count: int,
    entity_count: int,
    relation_count: int,
    annotation_file_count: int,
    labels: set[str],
    notes: str = "",
) -> dict[str, Any]:
    text = normalize_text(text)
    eligible = int(bool(text))

    return {
        "dataset": dataset,
        "split": split,
        "doc_id": str(doc_id),
        "document_type": document_type,
        "schema_name": schema_name,
        "source_file": source_file,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count": int(sentence_count),
        "entity_count": int(entity_count),
        "relation_count": int(relation_count),
        "annotation_file_count": int(annotation_file_count),
        "label_types": "|".join(sorted(labels)),
        "text_sha256": text_hash(text) if text else "",
        "eligible": eligible,
        "length_bucket": "",
        "notes": notes,
    }


# ============================================================
# SciERC
# ============================================================

def find_scierc_json_dir(root: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []

    for train_file in root.rglob("train.json"):
        parent = train_file.parent

        if not (
            (parent / "dev.json").exists()
            and (parent / "test.json").exists()
        ):
            continue

        parts = {part.lower() for part in parent.parts}

        if "processed_data" in parts:
            priority = 30
        elif "normalized_data" in parts:
            priority = 20
        else:
            priority = 10

        candidates.append((priority, parent))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (item[0], str(item[1])),
        reverse=True,
    )

    return candidates[0][1]


def parse_scierc(root: Path) -> list[dict[str, Any]]:
    json_dir = find_scierc_json_dir(root)

    if json_dir is None:
        print(
            "[SCIERC_ERROR] processed train/dev/test JSON files not found",
            flush=True,
        )
        return []

    print(f"[SCIERC_SOURCE] {json_dir}", flush=True)

    rows: list[dict[str, Any]] = []

    for split in ("train", "dev", "test"):
        path = json_dir / f"{split}.json"
        records = load_json_records(path)

        print(
            f"[SCIERC_{split.upper()}_JSON_ROWS] {len(records)}",
            flush=True,
        )

        for index, obj in enumerate(records):
            sentences = obj.get("sentences", [])

            if not isinstance(sentences, list):
                sentences = []

            sentence_texts: list[str] = []

            for sentence in sentences:
                if isinstance(sentence, list):
                    sentence_texts.append(
                        " ".join(str(token) for token in sentence)
                    )
                elif isinstance(sentence, str):
                    sentence_texts.append(sentence)

            text = " ".join(sentence_texts)

            ner = obj.get("ner", [])
            relations = obj.get("relations", [])

            entity_count = (
                sum(len(items) for items in ner if isinstance(items, list))
                if isinstance(ner, list)
                else 0
            )

            relation_count = (
                sum(
                    len(items)
                    for items in relations
                    if isinstance(items, list)
                )
                if isinstance(relations, list)
                else 0
            )

            labels: set[str] = set()

            if isinstance(ner, list):
                for sentence_ner in ner:
                    if not isinstance(sentence_ner, list):
                        continue

                    for entity in sentence_ner:
                        if (
                            isinstance(entity, list)
                            and len(entity) >= 3
                        ):
                            labels.add(f"NER:{entity[-1]}")

            if isinstance(relations, list):
                for sentence_relations in relations:
                    if not isinstance(sentence_relations, list):
                        continue

                    for relation in sentence_relations:
                        if (
                            isinstance(relation, list)
                            and len(relation) >= 5
                        ):
                            labels.add(f"REL:{relation[-1]}")

            rows.append(
                make_row(
                    dataset="SciERC",
                    split=split,
                    doc_id=obj.get(
                        "doc_key",
                        f"scierc_{split}_{index:05d}",
                    ),
                    document_type="abstract",
                    schema_name="scierc_entity_relation",
                    source_file=str(path.relative_to(root)),
                    text=text,
                    sentence_count=len(sentences),
                    entity_count=entity_count,
                    relation_count=relation_count,
                    annotation_file_count=1,
                    labels=labels,
                )
            )

    return rows


# ============================================================
# EBM-NLP
# ============================================================

def find_ebm_documents_dir(root: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []

    for directory in root.rglob("documents"):
        if not directory.is_dir():
            continue

        text_count = len(list(directory.glob("*.text")))

        if text_count:
            candidates.append((text_count, directory))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (item[0], str(item[1])),
        reverse=True,
    )

    return candidates[0][1]


def extract_pmid(filename: str) -> str | None:
    match = re.search(r"(\d{5,})", filename)
    return match.group(1) if match else None


def annotation_category(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}

    mapping = {
        "participants": "Participant",
        "participant": "Participant",
        "interventions": "Intervention",
        "intervention": "Intervention",
        "outcomes": "Outcome",
        "outcome": "Outcome",
    }

    for key, value in mapping.items():
        if key in parts:
            return value

    return None


def read_binary_labels(path: Path) -> list[int]:
    raw = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    labels: list[int] = []

    for token in re.findall(r"(?<![\d.])[01](?![\d.])", raw):
        labels.append(int(token))

    return labels


def count_positive_spans(labels: list[int]) -> int:
    count = 0
    previous = 0

    for value in labels:
        current = int(value > 0)

        if current == 1 and previous == 0:
            count += 1

        previous = current

    return count


def build_ebm_annotation_index(
    root: Path,
) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        parts = {part.lower() for part in path.parts}

        # Use the recommended aggregated starting-span labels,
        # not every individual worker annotation.
        if not {
            "annotations",
            "aggregated",
            "starting_spans",
        }.issubset(parts):
            continue

        pmid = extract_pmid(path.name)

        if pmid:
            index[pmid].append(path)

    return index


def parse_ebm_nlp(root: Path) -> list[dict[str, Any]]:
    documents_dir = find_ebm_documents_dir(root)

    if documents_dir is None:
        print("[EBM_ERROR] documents directory not found", flush=True)
        return []

    print(f"[EBM_DOCUMENTS_SOURCE] {documents_dir}", flush=True)

    text_files = sorted(documents_dir.glob("*.text"))
    annotation_index = build_ebm_annotation_index(root)

    print(f"[EBM_TEXT_DOCUMENTS] {len(text_files)}", flush=True)
    print(
        f"[EBM_PMIDS_WITH_AGGREGATED_LABELS] {len(annotation_index)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []

    for path in text_files:
        pmid = path.stem
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        token_path = path.with_suffix(".tokens")
        annotations = annotation_index.get(pmid, [])

        entity_count = 0
        labels: set[str] = set()

        for annotation_path in annotations:
            category = annotation_category(annotation_path)

            if category:
                labels.add(f"PIO:{category}")

            binary_labels = read_binary_labels(annotation_path)
            entity_count += count_positive_spans(binary_labels)

        sentence_count = len(
            [
                part
                for part in re.split(
                    r"(?<=[.!?])\s+",
                    normalize_text(text),
                )
                if part.strip()
            ]
        )

        notes = ""

        if not token_path.exists():
            notes = "matching .tokens file missing"

        rows.append(
            make_row(
                dataset="EBM-NLP",
                split="all",
                doc_id=pmid,
                document_type="abstract",
                schema_name="ebm_pio_spans",
                source_file=str(path.relative_to(root)),
                text=text,
                sentence_count=max(sentence_count, 1),
                entity_count=entity_count,
                relation_count=0,
                annotation_file_count=len(annotations),
                labels=labels,
                notes=notes,
            )
        )

    return rows


# ============================================================
# SciER
# ============================================================

def parse_scier(root: Path) -> list[dict[str, Any]]:
    llm_dir = root / "LLM"

    if not llm_dir.exists():
        matches = [
            path
            for path in root.rglob("LLM")
            if path.is_dir()
        ]

        if matches:
            llm_dir = matches[0]

    if not llm_dir.exists():
        print("[SCIER_ERROR] LLM directory not found", flush=True)
        return []

    print(f"[SCIER_SOURCE] {llm_dir}", flush=True)

    grouped: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    split_files = [
        ("train", llm_dir / "train.jsonl"),
        ("dev", llm_dir / "dev.jsonl"),
        ("test", llm_dir / "test.jsonl"),
        ("test_ood", llm_dir / "test_ood.jsonl"),
    ]

    sentence_rows = 0

    for split, path in split_files:
        if not path.exists():
            print(f"[SCIER_MISSING] {path}", flush=True)
            continue

        records = load_json_records(path)

        print(
            f"[SCIER_{split.upper()}_SENTENCES] {len(records)}",
            flush=True,
        )

        sentence_rows += len(records)

        for index, obj in enumerate(records):
            doc_id = str(
                obj.get(
                    "doc_id",
                    f"{split}_unknown_{index:06d}",
                )
            )

            key = (split, doc_id)

            if key not in grouped:
                grouped[key] = {
                    "sentences": [],
                    "entity_count": 0,
                    "relation_count": 0,
                    "labels": set(),
                    "source_file": str(path.relative_to(root)),
                }

            group = grouped[key]

            sentence = obj.get("sentence", "")

            if isinstance(sentence, str) and sentence.strip():
                group["sentences"].append(sentence.strip())

            ner = obj.get("ner", [])

            if isinstance(ner, list):
                group["entity_count"] += len(ner)

                for entity in ner:
                    if (
                        isinstance(entity, list)
                        and len(entity) >= 2
                    ):
                        group["labels"].add(f"NER:{entity[-1]}")

            # rel_plus is the stricter end-to-end relation
            # annotation; fall back to rel when absent.
            relations = obj.get("rel_plus")

            if not isinstance(relations, list):
                relations = obj.get("rel", [])

            if isinstance(relations, list):
                group["relation_count"] += len(relations)

                for relation in relations:
                    if (
                        isinstance(relation, list)
                        and len(relation) >= 3
                    ):
                        group["labels"].add(f"REL:{relation[1]}")

    rows: list[dict[str, Any]] = []

    for (split, doc_id), group in sorted(grouped.items()):
        rows.append(
            make_row(
                dataset="SciER",
                split=split,
                doc_id=doc_id,
                document_type="full_document",
                schema_name="scier_entity_relation",
                source_file=group["source_file"],
                text=" ".join(group["sentences"]),
                sentence_count=len(group["sentences"]),
                entity_count=group["entity_count"],
                relation_count=group["relation_count"],
                annotation_file_count=1,
                labels=group["labels"],
            )
        )

    print(f"[SCIER_TOTAL_SENTENCE_ROWS] {sentence_rows}", flush=True)
    print(f"[SCIER_GROUPED_DOCUMENTS] {len(rows)}", flush=True)

    return rows


# ============================================================
# Outputs
# ============================================================

def assign_length_buckets(rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row["eligible"]:
            by_dataset[row["dataset"]].append(row)

    for dataset_rows in by_dataset.values():
        ordered = sorted(
            dataset_rows,
            key=lambda row: int(row["char_count"]),
        )

        total = len(ordered)

        for index, row in enumerate(ordered):
            fraction = (index + 1) / max(total, 1)

            if fraction <= 1 / 3:
                row["length_bucket"] = "short"
            elif fraction <= 2 / 3:
                row["length_bucket"] = "medium"
            else:
                row["length_bucket"] = "long"


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


def build_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        groups[(row["dataset"], row["split"])].append(row)

    summary: list[dict[str, Any]] = []

    for (dataset, split), group in sorted(groups.items()):
        lengths = [
            int(row["char_count"])
            for row in group
            if row["eligible"]
        ]

        summary.append(
            {
                "dataset": dataset,
                "split": split,
                "documents": len(group),
                "eligible": sum(int(row["eligible"]) for row in group),
                "sentences": sum(int(row["sentence_count"]) for row in group),
                "entities_or_spans": sum(
                    int(row["entity_count"]) for row in group
                ),
                "relations": sum(
                    int(row["relation_count"]) for row in group
                ),
                "mean_chars": round(mean(lengths), 2) if lengths else 0,
                "median_chars": round(median(lengths), 2) if lengths else 0,
                "min_chars": min(lengths) if lengths else 0,
                "max_chars": max(lengths) if lengths else 0,
            }
        )

    return summary


def build_label_inventory(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        for label in str(row["label_types"]).split("|"):
            if label:
                counts[(row["dataset"], label)] += 1

    return [
        {
            "dataset": dataset,
            "label": label,
            "documents_with_label": count,
        }
        for (dataset, label), count in sorted(counts.items())
    ]


def build_duplicates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row["text_sha256"]:
            groups[row["text_sha256"]].append(row)

    output: list[dict[str, Any]] = []

    for digest, group in groups.items():
        if len(group) < 2:
            continue

        for row in group:
            output.append(
                {
                    "text_sha256": digest,
                    "group_size": len(group),
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "doc_id": row["doc_id"],
                }
            )

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/tahiti/PromptStressLab",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    raw = root / "data" / "raw"
    manifests = root / "manifests"

    print("=== SPECIALIZED MANIFEST BUILDER V2 ===", flush=True)
    print(f"root={root}", flush=True)

    scierc_rows = parse_scierc(raw / "scierc")
    ebm_rows = parse_ebm_nlp(raw / "ebm_nlp" / "v2_00")
    scier_rows = parse_scier(raw / "scier" / "SciER")

    all_rows = scierc_rows + ebm_rows + scier_rows
    assign_length_buckets(all_rows)

    write_csv(
        manifests / "scierc_manifest.csv",
        scierc_rows,
        FIELDS,
    )
    write_csv(
        manifests / "ebm_nlp_manifest.csv",
        ebm_rows,
        FIELDS,
    )
    write_csv(
        manifests / "scier_manifest.csv",
        scier_rows,
        FIELDS,
    )
    write_csv(
        manifests / "unified_manifest.csv",
        all_rows,
        FIELDS,
    )

    summary = build_summary(all_rows)
    write_csv(
        manifests / "manifest_summary.csv",
        summary,
        list(summary[0].keys()) if summary else ["dataset"],
    )

    labels = build_label_inventory(all_rows)
    write_csv(
        manifests / "label_inventory.csv",
        labels,
        ["dataset", "label", "documents_with_label"],
    )

    duplicates = build_duplicates(all_rows)
    write_csv(
        manifests / "duplicate_texts.csv",
        duplicates,
        [
            "text_sha256",
            "group_size",
            "dataset",
            "split",
            "doc_id",
        ],
    )

    actual_counts = {
        "SciERC": len(scierc_rows),
        "EBM-NLP": len(ebm_rows),
        "SciER": len(scier_rows),
    }

    expected_counts = {
        "SciERC": 500,
        "EBM-NLP": 4993,
        "SciER": 106,
    }

    validation = []

    for dataset, expected in expected_counts.items():
        actual = actual_counts[dataset]

        validation.append(
            {
                "dataset": dataset,
                "check": "document_count",
                "expected": expected,
                "actual": actual,
                "status": "PASS" if actual == expected else "WARN",
            }
        )

    write_csv(
        manifests / "manifest_validation.csv",
        validation,
        ["dataset", "check", "expected", "actual", "status"],
    )

    metadata = {
        "root": str(root),
        "builder": "build_manifests_v2.py",
        "rows": len(all_rows),
        "datasets": actual_counts,
        "eligible": {
            "SciERC": sum(int(row["eligible"]) for row in scierc_rows),
            "EBM-NLP": sum(int(row["eligible"]) for row in ebm_rows),
            "SciER": sum(int(row["eligible"]) for row in scier_rows),
        },
        "expected": expected_counts,
    }

    (manifests / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=== FINAL COUNTS ===", flush=True)

    for dataset, actual in actual_counts.items():
        expected = expected_counts[dataset]
        status = "PASS" if actual == expected else "WARN"

        print(
            f"[{status}] {dataset}: actual={actual} expected={expected}",
            flush=True,
        )

    print(f"[TOTAL] {len(all_rows)}", flush=True)
    print("=== MANIFEST_V2_FINISHED ===", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
