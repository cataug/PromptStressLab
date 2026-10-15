#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, CLIPModel


MAIN_CONDITIONS = ["P1", "P2", "P3", "P4", "P5", "P6"]

TRANSITIONS = list(
    zip(
        MAIN_CONDITIONS[:-1],
        MAIN_CONDITIONS[1:],
    )
)

DATASET_FIELDS = {
    "SciERC": [
        "Task",
        "Method",
        "Metric",
        "Material",
        "OtherScientificTerm",
        "Generic",
        "__RELATIONS__",
    ],
    "EBM-NLP": [
        "Participant",
        "Intervention",
        "Outcome",
    ],
    "SciER": [
        "Dataset",
        "Method",
        "Task",
        "__RELATIONS__",
    ],
}

ALIASES = {
    "scierc": {
        "task": "Task",
        "method": "Method",
        "metric": "Metric",
        "material": "Material",
        "otherscientificterm": "OtherScientificTerm",
        "other scientific term": "OtherScientificTerm",
        "other_scientific_term": "OtherScientificTerm",
        "generic": "Generic",
    },
    "ebm-nlp": {
        "participant": "Participant",
        "participants": "Participant",
        "population": "Participant",
        "intervention": "Intervention",
        "interventions": "Intervention",
        "outcome": "Outcome",
        "outcomes": "Outcome",
    },
    "scier": {
        "dataset": "Dataset",
        "method": "Method",
        "task": "Task",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except Exception:
                continue

            if isinstance(value, dict):
                rows.append(value)

    return rows


def latest_by_job(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    for path in paths:
        for row in read_jsonl(path):
            job_id = row.get("job_id")

            if isinstance(job_id, str) and job_id:
                rows[job_id] = row

    return rows


def clean_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def clean_key(value: Any) -> str:
    return clean_text(value).casefold()


def first_value(
    row: dict[str, Any],
    keys: list[str],
) -> str:
    for key in keys:
        value = row.get(key)

        if value is None:
            continue

        text = clean_text(value)

        if text:
            return text

    return ""


def canonical_type(
    dataset: str,
    value: Any,
) -> str | None:
    key = clean_key(value)
    aliases = ALIASES.get(
        dataset.casefold(),
        {},
    )

    if key in aliases:
        return aliases[key]

    compact = re.sub(
        r"[^a-z0-9]+",
        "",
        key,
    )

    if compact in aliases:
        return aliases[compact]

    for expected in DATASET_FIELDS.get(
        dataset,
        [],
    ):
        if expected.startswith("__"):
            continue

        if clean_key(expected) == key:
            return expected

    return None


def relation_to_text(relation: Any) -> str:
    if isinstance(relation, str):
        return clean_text(relation)

    if not isinstance(relation, dict):
        return clean_text(relation)

    relation_type = first_value(
        relation,
        [
            "type",
            "relation_type",
            "label",
            "predicate",
            "relation",
        ],
    )

    head = first_value(
        relation,
        [
            "head",
            "source",
            "subject",
            "from",
            "arg1",
            "entity1",
            "head_text",
            "source_text",
        ],
    )

    tail = first_value(
        relation,
        [
            "tail",
            "target",
            "object",
            "to",
            "arg2",
            "entity2",
            "tail_text",
            "target_text",
        ],
    )

    if relation_type or head or tail:
        return clean_text(
            f"{head} --[{relation_type or 'related_to'}]--> {tail}"
        )

    ignored = {
        "id",
        "relation_id",
        "confidence",
        "probability",
        "score",
    }

    parts: list[str] = []

    for key in sorted(relation):
        if key in ignored:
            continue

        value = relation[key]

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            text = clean_text(value)

            if text:
                parts.append(
                    f"{key}={text}"
                )

    return " | ".join(parts)


def normalize_values(value: Any) -> list[str]:
    values: list[str] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = first_value(
                    item,
                    [
                        "text",
                        "span",
                        "name",
                        "value",
                        "mention",
                    ],
                )
            else:
                text = clean_text(item)

            if text:
                values.append(text)

    elif isinstance(value, dict):
        text = first_value(
            value,
            [
                "text",
                "span",
                "name",
                "value",
                "mention",
            ],
        )

        if text:
            values.append(text)

    else:
        text = clean_text(value)

        if text:
            values.append(text)

    return values


def extract_fields(
    prediction: dict[str, Any],
    dataset: str,
) -> tuple[
    dict[str, list[str]],
    bool,
    bool,
]:
    fields = {
        field: []
        for field in DATASET_FIELDS[dataset]
    }

    validation = prediction.get(
        "validation",
        {},
    )

    if not isinstance(validation, dict):
        validation = {}

    json_valid = bool(
        validation.get(
            "json_valid",
            False,
        )
    )

    schema_valid = bool(
        validation.get(
            "schema_valid",
            False,
        )
    )

    parsed = prediction.get(
        "parsed_output"
    )

    if not isinstance(parsed, dict):
        return (
            fields,
            json_valid,
            schema_valid,
        )

    # Поддержка формата:
    # {"Task": [...], "Method": [...]}.
    for expected_field in fields:
        if expected_field.startswith("__"):
            continue

        for key, value in parsed.items():
            canonical = canonical_type(
                dataset,
                key,
            )

            if canonical == expected_field:
                fields[expected_field].extend(
                    normalize_values(value)
                )

    # Поддержка формата:
    # {"entities": [{"type": ..., "text": ...}]}.
    entities = parsed.get(
        "entities",
        [],
    )

    if isinstance(entities, dict):
        expanded = []

        for entity_type, values in entities.items():
            if isinstance(values, list):
                for value in values:
                    expanded.append(
                        {
                            "type": entity_type,
                            "text": value,
                        }
                    )
            else:
                expanded.append(
                    {
                        "type": entity_type,
                        "text": values,
                    }
                )

        entities = expanded

    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue

            entity_type = canonical_type(
                dataset,
                first_value(
                    entity,
                    [
                        "type",
                        "entity_type",
                        "label",
                        "category",
                    ],
                ),
            )

            if (
                entity_type is None
                or entity_type not in fields
            ):
                continue

            entity_text = first_value(
                entity,
                [
                    "text",
                    "span",
                    "name",
                    "value",
                    "mention",
                    "surface",
                ],
            )

            if entity_text:
                fields[entity_type].append(
                    entity_text
                )

    if "__RELATIONS__" in fields:
        relations = parsed.get(
            "relations",
            [],
        )

        if isinstance(relations, dict):
            relations = list(
                relations.values()
            )

        if isinstance(relations, list):
            for relation in relations:
                text = relation_to_text(
                    relation
                )

                if text:
                    fields[
                        "__RELATIONS__"
                    ].append(text)

    # Удаляем повторы, порядок делаем стабильным.
    for field, values in fields.items():
        unique: dict[str, str] = {}

        for value in values:
            text = clean_text(value)

            if text:
                unique.setdefault(
                    text.casefold(),
                    text,
                )

        fields[field] = sorted(
            unique.values(),
            key=lambda text: text.casefold(),
        )

    return (
        fields,
        json_valid,
        schema_valid,
    )


class ClipEmbedder:
    def __init__(
        self,
        model_path: Path,
        device: str,
        batch_size: int,
    ) -> None:
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.batch_size = batch_size

        print(
            f"[EMBEDDING_MODEL] {model_path}",
            flush=True,
        )

        print(
            f"[EMBEDDING_DEVICE] {self.device}",
            flush=True,
        )

        print(
            f"[EMBEDDING_BATCH_SIZE] {batch_size}",
            flush=True,
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True,
            )
        )

        self.model = CLIPModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        self.model.eval()
        self.model.to(
            self.device
        )

        if self.device.type == "cuda":
            self.model.half()

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, 512),
                dtype=np.float32,
            )

        output_batches: list[
            np.ndarray
        ] = []

        total = len(texts)

        for start in range(
            0,
            total,
            self.batch_size,
        ):
            batch = texts[
                start : start + self.batch_size
            ]

            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )

            tokens = {
                key: value.to(
                    self.device
                )
                for key, value
                in tokens.items()
            }

            with torch.inference_mode():
                features = (
                    self.model.get_text_features(
                        **tokens
                    )
                )

            if not isinstance(
                features,
                torch.Tensor,
            ):
                if hasattr(
                    features,
                    "text_embeds",
                ):
                    features = (
                        features.text_embeds
                    )

                elif hasattr(
                    features,
                    "pooler_output",
                ):
                    features = (
                        features.pooler_output
                    )

                else:
                    raise TypeError(
                        "Unknown CLIP output: "
                        f"{type(features)}"
                    )

            features = (
                torch.nn.functional.normalize(
                    features.float(),
                    p=2,
                    dim=-1,
                )
            )

            output_batches.append(
                features.cpu().numpy()
            )

            completed = min(
                start + self.batch_size,
                total,
            )

            if (
                completed == total
                or completed
                % (
                    self.batch_size * 20
                )
                == 0
            ):
                print(
                    "[EMBED_PROGRESS] "
                    f"{completed}/{total}",
                    flush=True,
                )

        return np.concatenate(
            output_batches,
            axis=0,
        )


def centroid(
    values: list[str],
    embeddings: dict[
        str,
        np.ndarray,
    ],
) -> np.ndarray | None:
    vectors = [
        embeddings[value]
        for value in values
        if value in embeddings
    ]

    if not vectors:
        return None

    matrix = np.stack(
        vectors,
        axis=0,
    ).astype(
        np.float32
    )

    vector = matrix.mean(
        axis=0
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0:
        return None

    return vector / norm


def field_drift(
    left_values: list[str],
    right_values: list[str],
    embeddings: dict[
        str,
        np.ndarray,
    ],
) -> float:
    if (
        not left_values
        and not right_values
    ):
        return 0.0

    if (
        not left_values
        or not right_values
    ):
        return 1.0

    left_vector = centroid(
        left_values,
        embeddings,
    )

    right_vector = centroid(
        right_values,
        embeddings,
    )

    if (
        left_vector is None
        and right_vector is None
    ):
        return 0.0

    if (
        left_vector is None
        or right_vector is None
    ):
        return 1.0

    similarity = float(
        np.dot(
            left_vector,
            right_vector,
        )
    )

    similarity = max(
        -1.0,
        min(
            1.0,
            similarity,
        ),
    )

    return max(
        0.0,
        min(
            2.0,
            1.0 - similarity,
        ),
    )


def stable_seed(
    *values: Any,
) -> int:
    text = "|".join(
        str(value)
        for value in values
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return (
            math.nan,
            math.nan,
        )

    if values.size == 1:
        value = float(
            values[0]
        )

        return (
            value,
            value,
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means: list[
        np.ndarray
    ] = []

    remaining = iterations

    # Делаем кусками, чтобы не раздувать RAM.
    while remaining > 0:
        current = min(
            remaining,
            500,
        )

        indices = rng.integers(
            0,
            values.size,
            size=(
                current,
                values.size,
            ),
        )

        bootstrap_means.append(
            values[indices].mean(
                axis=1
            )
        )

        remaining -= current

    means = np.concatenate(
        bootstrap_means
    )

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    iterations: int,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    grouped = frame.groupby(
        group_columns,
        dropna=False,
    )

    for keys, group in grouped:
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        values = pd.to_numeric(
            group[value_column],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if values.size == 0:
            continue

        ci_low, ci_high = (
            bootstrap_mean_ci(
                values,
                seed=stable_seed(
                    *keys,
                    value_column,
                ),
                iterations=iterations,
            )
        )

        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                keys,
            )
        }

        row.update(
            {
                "n": int(
                    values.size
                ),
                "mean": float(
                    np.mean(values)
                ),
                "median": float(
                    np.median(values)
                ),
                "std": (
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if values.size > 1
                    else 0.0
                ),
                "p90": float(
                    np.quantile(
                        values,
                        0.90,
                    )
                ),
                "p95": float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),
                "max": float(
                    np.max(values)
                ),
                "bootstrap_ci_low": (
                    ci_low
                ),
                "bootstrap_ci_high": (
                    ci_high
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def load_metrics(
    path: Path,
) -> dict[
    tuple[str, str, str, str],
    dict[str, float],
]:
    if not path.exists():
        print(
            f"[WARNING] Metrics not found: {path}",
            flush=True,
        )

        return {}

    frame = pd.read_csv(
        path
    )

    if "physical_or_alias" in frame.columns:
        frame = frame[
            frame[
                "physical_or_alias"
            ].astype(str)
            == "physical"
        ]

    frame = frame[
        frame["condition"].isin(
            MAIN_CONDITIONS
        )
    ]

    exact_column = find_column(
        frame,
        [
            "entity_exact_f1",
            "exact_entity_f1",
            "exact_f1",
        ],
    )

    partial_column = find_column(
        frame,
        [
            "entity_partial_f1",
            "partial_entity_f1",
            "partial_f1",
        ],
    )

    relation_column = find_column(
        frame,
        [
            "relation_exact_f1",
            "relation_f1",
        ],
    )

    lookup: dict[
        tuple[
            str,
            str,
            str,
            str,
        ],
        dict[str, float],
    ] = {}

    for _, row in frame.iterrows():
        key = (
            str(row["model_id"]),
            str(row["dataset"]),
            str(row["record_id"]),
            str(row["condition"]),
        )

        lookup[key] = {
            "entity_exact_f1": (
                float(
                    row[exact_column]
                )
                if (
                    exact_column
                    and pd.notna(
                        row[exact_column]
                    )
                )
                else math.nan
            ),
            "entity_partial_f1": (
                float(
                    row[partial_column]
                )
                if (
                    partial_column
                    and pd.notna(
                        row[partial_column]
                    )
                )
                else math.nan
            ),
            "relation_exact_f1": (
                float(
                    row[relation_column]
                )
                if (
                    relation_column
                    and pd.notna(
                        row[relation_column]
                    )
                )
                else math.nan
            ),
        }

    return lookup


def metric_delta(
    metrics: dict[
        tuple[
            str,
            str,
            str,
            str,
        ],
        dict[str, float],
    ],
    model_id: str,
    dataset: str,
    record_id: str,
    left: str,
    right: str,
    metric: str,
) -> float:
    left_value = metrics.get(
        (
            model_id,
            dataset,
            record_id,
            left,
        ),
        {},
    ).get(
        metric,
        math.nan,
    )

    right_value = metrics.get(
        (
            model_id,
            dataset,
            record_id,
            right,
        ),
        {},
    ).get(
        metric,
        math.nan,
    )

    if not (
        math.isfinite(
            left_value
        )
        and math.isfinite(
            right_value
        )
    ):
        return math.nan

    return (
        right_value
        - left_value
    )


def holm_adjust(
    values: np.ndarray,
) -> np.ndarray:
    adjusted = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    finite_indices = np.where(
        np.isfinite(values)
    )[0]

    if finite_indices.size == 0:
        return adjusted

    finite_values = values[
        finite_indices
    ]

    order = np.argsort(
        finite_values
    )

    corrected = np.ones(
        finite_values.shape,
        dtype=float,
    )

    running_max = 0.0
    total = len(
        finite_values
    )

    for rank, position in enumerate(
        order
    ):
        candidate = min(
            finite_values[position]
            * (
                total - rank
            ),
            1.0,
        )

        running_max = max(
            running_max,
            candidate,
        )

        corrected[position] = (
            running_max
        )

    adjusted[
        finite_indices
    ] = corrected

    return adjusted


def build_correlations(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    targets = [
        "delta_entity_exact_f1",
        "abs_delta_entity_exact_f1",
        "delta_entity_partial_f1",
        "abs_delta_entity_partial_f1",
        "delta_relation_exact_f1",
        "abs_delta_relation_exact_f1",
    ]

    grouping_schemes = [
        [
            "model_id",
            "dataset",
            "transition",
        ],
        [
            "model_id",
            "dataset",
        ],
        [
            "model_id",
        ],
    ]

    for group_columns in grouping_schemes:
        grouped = pairwise.groupby(
            group_columns,
            dropna=False,
        )

        for keys, group in grouped:
            if not isinstance(
                keys,
                tuple,
            ):
                keys = (keys,)

            for target in targets:
                subset = group[
                    [
                        "psi",
                        target,
                    ]
                ].replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                ).dropna()

                if len(subset) < 4:
                    continue

                if (
                    subset["psi"].nunique()
                    < 2
                    or subset[target].nunique()
                    < 2
                ):
                    rho = math.nan
                    p_value = math.nan

                else:
                    result = spearmanr(
                        subset[
                            "psi"
                        ].to_numpy(),
                        subset[
                            target
                        ].to_numpy(),
                    )

                    rho = float(
                        result.statistic
                    )

                    p_value = float(
                        result.pvalue
                    )

                row = {
                    column: value
                    for column, value
                    in zip(
                        group_columns,
                        keys,
                    )
                }

                row.update(
                    {
                        "grouping": (
                            "+".join(
                                group_columns
                            )
                        ),
                        "target": target,
                        "n": len(
                            subset
                        ),
                        "spearman_rho": (
                            rho
                        ),
                        "p_value": (
                            p_value
                        ),
                    }
                )

                rows.append(row)

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    output["p_holm"] = math.nan
    output[
        "significant_holm_005"
    ] = 0

    families = output.groupby(
        [
            "grouping",
            "target",
        ],
        dropna=False,
    ).groups

    for _, indices in families.items():
        indices = list(
            indices
        )

        p_values = output.loc[
            indices,
            "p_value",
        ].to_numpy(
            dtype=float
        )

        adjusted = holm_adjust(
            p_values
        )

        output.loc[
            indices,
            "p_holm",
        ] = adjusted

        output.loc[
            indices,
            "significant_holm_005",
        ] = (
            adjusted < 0.05
        ).astype(int)

    return output


def update_main_summary(
    path: Path,
    semantic_summary: dict[str, Any],
) -> None:
    current: dict[
        str,
        Any,
    ] = {}

    if path.exists():
        try:
            current = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            current = {}

    files = list(
        current.get(
            "statistics_files",
            [],
        )
    )

    for name in semantic_summary[
        "statistics_files"
    ]:
        if name not in files:
            files.append(name)

    current.update(
        {
            "psi_computed": True,
            "psi_embedding_backend": (
                semantic_summary[
                    "embedding_backend"
                ]
            ),
            "psi_embedding_model": (
                semantic_summary[
                    "embedding_model"
                ]
            ),
            "psi_document_transition_pairs": (
                semantic_summary[
                    "document_transition_pairs"
                ]
            ),
            "psi_field_transition_pairs": (
                semantic_summary[
                    "field_transition_pairs"
                ]
            ),
            "statistics_files": files,
            "note": (
                "PSI, field-wise semantic volatility, "
                "bootstrap confidence intervals, and "
                "PSI/F1 Spearman correlations were "
                "computed from the saved predictions."
            ),
        }
    )

    path.write_text(
        json.dumps(
            current,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=(
            "/home/tahiti/"
            "PromptStressLab"
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            "/home/tahiti/"
            "Forensics/models/"
            "clip_vit_base_patch32"
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).expanduser().resolve()

    embedding_model = Path(
        args.embedding_model
    ).expanduser().resolve()

    if not embedding_model.exists():
        raise FileNotFoundError(
            "Embedding model not found: "
            f"{embedding_model}"
        )

    jobs = read_jsonl(
        root
        / "manifests"
        / "experiment_jobs.jsonl"
    )

    prediction_paths = sorted(
        (
            root
            / "outputs"
            / "generations"
        ).glob(
            "*/predictions.jsonl"
        )
    )

    predictions = latest_by_job(
        prediction_paths
    )

    print(
        f"[JOBS] {len(jobs)}",
        flush=True,
    )

    print(
        f"[PREDICTIONS] "
        f"{len(predictions)}",
        flush=True,
    )

    main_jobs = {
        str(job["job_id"]): job
        for job in jobs
        if str(
            job.get(
                "condition"
            )
        )
        in MAIN_CONDITIONS
    }

    indexed: dict[
        tuple[
            str,
            str,
            str,
        ],
        dict[
            str,
            dict[str, Any],
        ],
    ] = defaultdict(dict)

    all_texts: set[
        str
    ] = set()

    for job_id, prediction in predictions.items():
        job = main_jobs.get(
            job_id
        )

        if job is None:
            continue

        model_id = str(
            job["model_id"]
        )

        dataset = str(
            job["dataset"]
        )

        record_id = str(
            job["record_id"]
        )

        condition = str(
            job["condition"]
        )

        if dataset not in DATASET_FIELDS:
            continue

        (
            fields,
            json_valid,
            schema_valid,
        ) = extract_fields(
            prediction,
            dataset,
        )

        indexed[
            (
                model_id,
                dataset,
                record_id,
            )
        ][condition] = {
            "fields": fields,
            "json_valid": (
                json_valid
            ),
            "schema_valid": (
                schema_valid
            ),
        }

        for values in fields.values():
            all_texts.update(
                values
            )

    texts = sorted(
        all_texts,
        key=lambda value: (
            value.casefold()
        ),
    )

    print(
        "[MAIN_PREDICTIONS_INDEXED] "
        f"{sum(len(x) for x in indexed.values())}",
        flush=True,
    )

    print(
        "[UNIQUE_FIELD_ITEMS] "
        f"{len(texts)}",
        flush=True,
    )

    embedder = ClipEmbedder(
        model_path=embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    embedding_matrix = (
        embedder.encode(
            texts
        )
    )

    embeddings = {
        text: embedding_matrix[index]
        for index, text
        in enumerate(texts)
    }

    metrics = load_metrics(
        root
        / "outputs"
        / "metrics"
        / "job_metrics_physical.csv"
    )

    pairwise_rows: list[
        dict[str, Any]
    ] = []

    field_rows: list[
        dict[str, Any]
    ] = []

    for (
        model_id,
        dataset,
        record_id,
    ), conditions in sorted(
        indexed.items()
    ):
        fields = DATASET_FIELDS[
            dataset
        ]

        for (
            left_condition,
            right_condition,
        ) in TRANSITIONS:
            if (
                left_condition
                not in conditions
                or right_condition
                not in conditions
            ):
                continue

            left = conditions[
                left_condition
            ]

            right = conditions[
                right_condition
            ]

            transition = (
                f"{left_condition}"
                f"->{right_condition}"
            )

            drifts: list[
                float
            ] = []

            for field in fields:
                left_values = (
                    left["fields"].get(
                        field,
                        [],
                    )
                )

                right_values = (
                    right["fields"].get(
                        field,
                        [],
                    )
                )

                drift = field_drift(
                    left_values,
                    right_values,
                    embeddings,
                )

                drifts.append(
                    drift
                )

                field_rows.append(
                    {
                        "model_id": (
                            model_id
                        ),
                        "dataset": (
                            dataset
                        ),
                        "record_id": (
                            record_id
                        ),
                        "transition": (
                            transition
                        ),
                        "left_condition": (
                            left_condition
                        ),
                        "right_condition": (
                            right_condition
                        ),
                        "field": field,
                        "left_item_count": (
                            len(
                                left_values
                            )
                        ),
                        "right_item_count": (
                            len(
                                right_values
                            )
                        ),
                        "left_empty": int(
                            not left_values
                        ),
                        "right_empty": int(
                            not right_values
                        ),
                        "semantic_drift": (
                            drift
                        ),
                        "left_schema_valid": int(
                            left[
                                "schema_valid"
                            ]
                        ),
                        "right_schema_valid": int(
                            right[
                                "schema_valid"
                            ]
                        ),
                        "both_schema_valid": int(
                            left[
                                "schema_valid"
                            ]
                            and right[
                                "schema_valid"
                            ]
                        ),
                    }
                )

            psi = float(
                np.mean(
                    drifts
                )
            )

            delta_exact = metric_delta(
                metrics,
                model_id,
                dataset,
                record_id,
                left_condition,
                right_condition,
                "entity_exact_f1",
            )

            delta_partial = metric_delta(
                metrics,
                model_id,
                dataset,
                record_id,
                left_condition,
                right_condition,
                "entity_partial_f1",
            )

            delta_relation = metric_delta(
                metrics,
                model_id,
                dataset,
                record_id,
                left_condition,
                right_condition,
                "relation_exact_f1",
            )

            pairwise_rows.append(
                {
                    "model_id": (
                        model_id
                    ),
                    "dataset": (
                        dataset
                    ),
                    "record_id": (
                        record_id
                    ),
                    "transition": (
                        transition
                    ),
                    "left_condition": (
                        left_condition
                    ),
                    "right_condition": (
                        right_condition
                    ),
                    "field_count": (
                        len(drifts)
                    ),
                    "psi": psi,
                    "left_json_valid": int(
                        left[
                            "json_valid"
                        ]
                    ),
                    "right_json_valid": int(
                        right[
                            "json_valid"
                        ]
                    ),
                    "left_schema_valid": int(
                        left[
                            "schema_valid"
                        ]
                    ),
                    "right_schema_valid": int(
                        right[
                            "schema_valid"
                        ]
                    ),
                    "both_schema_valid": int(
                        left[
                            "schema_valid"
                        ]
                        and right[
                            "schema_valid"
                        ]
                    ),
                    "schema_break": int(
                        left[
                            "schema_valid"
                        ]
                        and not right[
                            "schema_valid"
                        ]
                    ),
                    "delta_entity_exact_f1": (
                        delta_exact
                    ),
                    "abs_delta_entity_exact_f1": (
                        abs(
                            delta_exact
                        )
                        if math.isfinite(
                            delta_exact
                        )
                        else math.nan
                    ),
                    "delta_entity_partial_f1": (
                        delta_partial
                    ),
                    "abs_delta_entity_partial_f1": (
                        abs(
                            delta_partial
                        )
                        if math.isfinite(
                            delta_partial
                        )
                        else math.nan
                    ),
                    "delta_relation_exact_f1": (
                        delta_relation
                    ),
                    "abs_delta_relation_exact_f1": (
                        abs(
                            delta_relation
                        )
                        if math.isfinite(
                            delta_relation
                        )
                        else math.nan
                    ),
                }
            )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    field_pairwise = pd.DataFrame(
        field_rows
    )

    output_dir = (
        root
        / "outputs"
        / "statistics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairwise.to_csv(
        output_dir
        / "psi_pairwise.csv",
        index=False,
    )

    field_pairwise.to_csv(
        output_dir
        / "field_volatility_pairwise.csv",
        index=False,
    )

    psi_summary = summarize(
        pairwise,
        [
            "model_id",
            "dataset",
            "transition",
        ],
        "psi",
        args.bootstrap_iterations,
    )

    psi_summary.to_csv(
        output_dir
        / "psi_summary.csv",
        index=False,
    )

    psi_overall = summarize(
        pairwise,
        [
            "model_id",
            "dataset",
        ],
        "psi",
        args.bootstrap_iterations,
    )

    psi_overall.to_csv(
        output_dir
        / "psi_summary_overall.csv",
        index=False,
    )

    field_summary = summarize(
        field_pairwise,
        [
            "model_id",
            "dataset",
            "transition",
            "field",
        ],
        "semantic_drift",
        args.bootstrap_iterations,
    )

    field_summary.to_csv(
        output_dir
        / "field_volatility_summary.csv",
        index=False,
    )

    field_overall = summarize(
        field_pairwise,
        [
            "model_id",
            "dataset",
            "field",
        ],
        "semantic_drift",
        args.bootstrap_iterations,
    )

    field_overall.to_csv(
        output_dir
        / "field_volatility_summary_overall.csv",
        index=False,
    )

    document_volatility = (
        pairwise.groupby(
            [
                "model_id",
                "dataset",
                "record_id",
            ],
            dropna=False,
        )
        .agg(
            transitions=(
                "psi",
                "count",
            ),
            mean_psi=(
                "psi",
                "mean",
            ),
            median_psi=(
                "psi",
                "median",
            ),
            p95_psi=(
                "psi",
                lambda values: (
                    values.quantile(
                        0.95
                    )
                ),
            ),
            max_psi=(
                "psi",
                "max",
            ),
            schema_breaks=(
                "schema_break",
                "sum",
            ),
        )
        .reset_index()
    )

    document_volatility.to_csv(
        output_dir
        / "document_volatility.csv",
        index=False,
    )

    correlations = build_correlations(
        pairwise
    )

    correlations.to_csv(
        output_dir
        / "psi_f1_correlations.csv",
        index=False,
    )

    expected_pair_count = (
        307
        * 3
        * 5
    )

    expected_field_pair_count = (
        3
        * (
            100 * 5 * 7
            + 191 * 5 * 3
            + 16 * 5 * 4
        )
    )

    summary = {
        "embedding_backend": (
            "CLIP text encoder"
        ),
        "embedding_model": str(
            embedding_model
        ),
        "main_conditions": (
            MAIN_CONDITIONS
        ),
        "transitions": [
            f"{left}->{right}"
            for left, right
            in TRANSITIONS
        ],
        "document_transition_pairs": (
            int(
                len(pairwise)
            )
        ),
        "expected_document_transition_pairs": (
            expected_pair_count
        ),
        "field_transition_pairs": (
            int(
                len(
                    field_pairwise
                )
            )
        ),
        "expected_field_transition_pairs": (
            expected_field_pair_count
        ),
        "unique_field_items_embedded": (
            len(texts)
        ),
        "complete": (
            len(pairwise)
            == expected_pair_count
        ),
        "psi_definition": (
            "Mean semantic drift across dataset fields. "
            "Field drift = 1 - cosine similarity between "
            "normalized CLIP centroids. Empty-empty=0; "
            "empty-nonempty=1."
        ),
        "statistics_files": [
            "psi_pairwise.csv",
            "psi_summary.csv",
            "psi_summary_overall.csv",
            "field_volatility_pairwise.csv",
            "field_volatility_summary.csv",
            "field_volatility_summary_overall.csv",
            "document_volatility.csv",
            "psi_f1_correlations.csv",
            "semantic_statistics_summary.json",
        ],
    }

    (
        output_dir
        / "semantic_statistics_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    update_main_summary(
        output_dir
        / "statistics_summary.json",
        summary,
    )

    print()
    print(
        "=== PSI / VOLATILITY COMPLETE ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "=== OVERALL PSI ==="
    )

    print(
        psi_overall[
            [
                "model_id",
                "dataset",
                "n",
                "mean",
                "median",
                "p95",
                "max",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            ]
        ].sort_values(
            [
                "model_id",
                "dataset",
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "=== TOP VOLATILE FIELDS ==="
    )

    print(
        field_overall.sort_values(
            "mean",
            ascending=False,
        ).head(
            30
        )[
            [
                "model_id",
                "dataset",
                "field",
                "n",
                "mean",
                "p95",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=== SIGNIFICANT PSI/F1 CORRELATIONS ==="
    )

    significant = correlations[
        correlations[
            "significant_holm_005"
        ]
        == 1
    ].sort_values(
        "p_holm"
    )

    if significant.empty:
        print("none")
    else:
        print(
            significant.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
