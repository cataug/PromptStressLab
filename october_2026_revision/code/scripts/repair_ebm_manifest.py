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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(
        normalize(text).encode("utf-8", errors="ignore")
    ).hexdigest()


def extract_pmid(path: Path) -> str | None:
    match = re.search(r"(\d{5,})", path.name)
    return match.group(1) if match else None


def is_annotation_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}

    return bool(
        parts.intersection(
            {
                "annotations",
                "annotation",
                "aggregated",
                "individual",
                "starting_spans",
                "hierarchical_labels",
                "participants",
                "interventions",
                "outcomes",
            }
        )
    )


def document_path_score(path: Path) -> tuple[int, int, str]:
    parts = {part.lower() for part in path.parts}
    score = 0

    if "documents" in parts:
        score += 100

    if path.suffix.lower() == ".text":
        score += 20

    if path.suffix.lower() == ".tokens":
        score += 10

    if is_annotation_path(path):
        score -= 1000

    return score, -len(path.parts), str(path)


def select_unique_documents(
    paths: list[Path],
) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for path in paths:
        pmid = extract_pmid(path)

        if pmid and not is_annotation_path(path):
            grouped[pmid].append(path)

    selected = {}

    for pmid, candidates in grouped.items():
        selected[pmid] = max(
            candidates,
            key=document_path_score,
        )

    return selected


def load_document_text(path: Path) -> str:
    raw = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    # .tokens normally stores whitespace-separated
    # tokens, often one token per line.
    return normalize(raw)


def detect_annotation_label(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}

    if "participants" in parts or "participant" in parts:
        return "PIO:Participant"

    if "interventions" in parts or "intervention" in parts:
        return "PIO:Intervention"

    if "outcomes" in parts or "outcome" in parts:
        return "PIO:Outcome"

    return None


def build_annotation_index(
    root: Path,
) -> tuple[
    dict[str, int],
    dict[str, set[str]],
]:
    counts: Counter[str] = Counter()
    labels: dict[str, set[str]] = defaultdict(set)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if not is_annotation_path(path):
            continue

        pmid = extract_pmid(path)

        if not pmid:
            continue

        counts[pmid] += 1

        label = detect_annotation_label(path)

        if label:
            labels[pmid].add(label)

    return dict(counts), labels


def assign_length_buckets(
    rows: list[dict[str, Any]],
) -> None:
    ordered = sorted(
        rows,
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


def build_ebm_rows(
    raw_root: Path,
) -> list[dict[str, Any]]:
    text_files = [
        path
        for path in raw_root.rglob("*.text")
        if path.is_file()
        and not is_annotation_path(path)
    ]

    token_files = [
        path
        for path in raw_root.rglob("*.tokens")
        if path.is_file()
        and not is_annotation_path(path)
    ]

    selected_text = select_unique_documents(text_files)
    selected_tokens = select_unique_documents(token_files)

    print(f"[EBM_TEXT_FILES] {len(text_files)}", flush=True)
    print(f"[EBM_TOKEN_FILES] {len(token_files)}", flush=True)
    print(
        f"[EBM_UNIQUE_TEXT_PMIDS] {len(selected_text)}",
        flush=True,
    )
    print(
        f"[EBM_UNIQUE_TOKEN_PMIDS] {len(selected_tokens)}",
        flush=True,
    )

    if len(selected_text) >= 4990:
        documents = selected_text
        mode = "text"
    elif len(selected_tokens) >= 4990:
        documents = selected_tokens
        mode = "tokens"
    else:
        # Use the union and prefer raw text when both exist.
        documents = dict(selected_tokens)
        documents.update(selected_text)
        mode = "union"

    print(f"[EBM_DOCUMENT_MODE] {mode}", flush=True)
    print(f"[EBM_SELECTED_DOCUMENTS] {len(documents)}", flush=True)

    annotation_counts, annotation_labels = (
        build_annotation_index(raw_root)
    )

    print(
        "[EBM_PMIDS_WITH_ANNOTATIONS] "
        f"{len(annotation_counts)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []

    for pmid, path in sorted(documents.items()):
        text = load_document_text(path)

        sentence_count = len(
            [
                sentence
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    text,
                )
                if sentence.strip()
            ]
        )

        try:
            source_file = str(path.relative_to(raw_root))
        except ValueError:
            source_file = str(path)

        rows.append(
            {
                "dataset": "EBM-NLP",
                "split": "all",
                "doc_id": pmid,
                "document_type": "abstract",
                "schema_name": "ebm_pio_spans",
                "source_file": source_file,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": max(sentence_count, 1),
                # Exact PIO spans will be normalized in
                # the experiment-ready gold builder.
                "entity_count": 0,
                "relation_count": 0,
                "annotation_file_count": annotation_counts.get(
                    pmid,
                    0,
                ),
                "label_types": "|".join(
                    sorted(
                        annotation_labels.get(
                            pmid,
                            set(),
                        )
                    )
                ),
                "text_sha256": sha256(text) if text else "",
                "eligible": int(bool(text)),
                "length_bucket": "",
                "notes": (
                    "PIO gold available; exact spans "
                    "pending normalization"
                ),
            }
        )

    assign_length_buckets(rows)

    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
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
        groups[
            (
                row["dataset"],
                row["split"],
            )
        ].append(row)

    summary = []

    for (dataset, split), group in sorted(groups.items()):
        lengths = [
            int(row["char_count"])
            for row in group
            if int(row["eligible"])
        ]

        summary.append(
            {
                "dataset": dataset,
                "split": split,
                "documents": len(group),
                "eligible": sum(
                    int(row["eligible"])
                    for row in group
                ),
                "sentences": sum(
                    int(row["sentence_count"])
                    for row in group
                ),
                "entities_or_spans": sum(
                    int(row["entity_count"])
                    for row in group
                ),
                "relations": sum(
                    int(row["relation_count"])
                    for row in group
                ),
                "mean_chars": (
                    round(mean(lengths), 2)
                    if lengths
                    else 0
                ),
                "median_chars": (
                    round(median(lengths), 2)
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
            }
        )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/home/tahiti/PromptStressLab",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    manifests = root / "manifests"
    ebm_raw = root / "data" / "raw" / "ebm_nlp"

    print("=== EBM MANIFEST REPAIR ===", flush=True)
    print(f"[EBM_ROOT] {ebm_raw}", flush=True)

    ebm_rows = build_ebm_rows(ebm_raw)

    scierc_rows = read_csv(
        manifests / "scierc_manifest.csv"
    )

    scier_rows = read_csv(
        manifests / "scier_manifest.csv"
    )

    all_rows = scierc_rows + ebm_rows + scier_rows

    write_csv(
        manifests / "ebm_nlp_manifest.csv",
        ebm_rows,
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
        [
            "dataset",
            "split",
            "documents",
            "eligible",
            "sentences",
            "entities_or_spans",
            "relations",
            "mean_chars",
            "median_chars",
            "min_chars",
            "max_chars",
        ],
    )

    counts = {
        "SciERC": len(scierc_rows),
        "EBM-NLP": len(ebm_rows),
        "SciER": len(scier_rows),
    }

    expected = {
        "SciERC": 500,
        "EBM-NLP": 4993,
        "SciER": 106,
    }

    validation = []

    for dataset in ("SciERC", "EBM-NLP", "SciER"):
        actual = counts[dataset]
        wanted = expected[dataset]

        validation.append(
            {
                "dataset": dataset,
                "check": "document_count",
                "expected": wanted,
                "actual": actual,
                "status": (
                    "PASS"
                    if actual == wanted
                    else "WARN"
                ),
            }
        )

    write_csv(
        manifests / "manifest_validation.csv",
        validation,
        [
            "dataset",
            "check",
            "expected",
            "actual",
            "status",
        ],
    )

    metadata = {
        "root": str(root),
        "builder": "repair_ebm_manifest.py",
        "rows": len(all_rows),
        "datasets": counts,
        "eligible": {
            "SciERC": sum(
                int(row["eligible"])
                for row in scierc_rows
            ),
            "EBM-NLP": sum(
                int(row["eligible"])
                for row in ebm_rows
            ),
            "SciER": sum(
                int(row["eligible"])
                for row in scier_rows
            ),
        },
        "expected": expected,
    }

    (
        manifests
        / "manifest_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== FINAL COUNTS ===", flush=True)

    for dataset in ("SciERC", "EBM-NLP", "SciER"):
        actual = counts[dataset]
        wanted = expected[dataset]
        status = "PASS" if actual == wanted else "WARN"

        print(
            f"[{status}] {dataset}: "
            f"actual={actual} expected={wanted}",
            flush=True,
        )

    print(f"[TOTAL] {len(all_rows)}", flush=True)
    print("=== EBM_MANIFEST_REPAIR_FINISHED ===", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
