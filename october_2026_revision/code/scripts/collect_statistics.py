#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


MAIN_CONDITIONS = ["P1", "P2", "P3", "P4", "P5", "P6"]
ABLATION_CONDITIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]

ENTITY_FIELDS = {
    "SciERC": [
        "Task",
        "Method",
        "Metric",
        "Material",
        "OtherScientificTerm",
        "Generic",
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
    ],
}

CONTINUOUS_METRICS = [
    "entity_exact_f1",
    "entity_partial_f1",
    "relation_exact_f1",
    "hallucinated_entity_rate",
    "prompt_tokens",
    "output_tokens",
    "seconds_per_item_in_batch",
]

BINARY_METRICS = [
    "json_valid",
    "schema_valid",
    "truncated_at_limit",
]


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
                # Writer may currently be appending the final line.
                continue

            if isinstance(value, dict):
                rows.append(value)

    return rows


def latest_by_job(paths: list[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    for path in paths:
        for row in read_jsonl(path):
            job_id = row.get("job_id")

            if isinstance(job_id, str):
                output[job_id] = row

    return output


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    count = len(p_values)

    if count == 0:
        return []

    order = np.argsort(
        np.asarray(p_values, dtype=float)
    )

    adjusted = np.ones(count, dtype=float)
    running_max = 0.0

    for rank, original_index in enumerate(order):
        multiplier = count - rank
        candidate = min(
            float(p_values[original_index]) * multiplier,
            1.0,
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[original_index] = running_max

    return adjusted.tolist()


def rank_biserial_from_differences(
    differences: np.ndarray,
) -> float:
    differences = differences[
        np.isfinite(differences)
    ]

    differences = differences[
        differences != 0
    ]

    if differences.size == 0:
        return 0.0

    positive = int(
        np.sum(differences > 0)
    )

    negative = int(
        np.sum(differences < 0)
    )

    return (
        positive - negative
    ) / max(
        positive + negative,
        1,
    )


def paired_continuous_test(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    family: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    group_columns = [
        "model_id",
        "dataset",
    ]

    for keys, group in frame.groupby(
        group_columns,
        dropna=False,
    ):
        pivot = group.pivot_table(
            index="record_id",
            columns="condition",
            values=metric,
            aggfunc="last",
        )

        if (
            left not in pivot.columns
            or right not in pivot.columns
        ):
            continue

        paired = pivot[
            [left, right]
        ].dropna()

        if paired.empty:
            continue

        left_values = paired[left].astype(float).to_numpy()
        right_values = paired[right].astype(float).to_numpy()
        differences = right_values - left_values

        nonzero = differences[
            differences != 0
        ]

        if nonzero.size == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            try:
                test = wilcoxon(
                    right_values,
                    left_values,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )

                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            except Exception:
                statistic = math.nan
                p_value = math.nan

        results.append(
            {
                "family": family,
                "model_id": keys[0],
                "dataset": keys[1],
                "left_condition": left,
                "right_condition": right,
                "metric": metric,
                "paired_documents": len(paired),
                "left_mean": float(
                    np.mean(left_values)
                ),
                "right_mean": float(
                    np.mean(right_values)
                ),
                "mean_delta_right_minus_left": float(
                    np.mean(differences)
                ),
                "median_delta_right_minus_left": float(
                    np.median(differences)
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_effect": (
                    rank_biserial_from_differences(
                        differences
                    )
                ),
                "right_better_count": int(
                    np.sum(differences > 0)
                ),
                "left_better_count": int(
                    np.sum(differences < 0)
                ),
                "ties": int(
                    np.sum(differences == 0)
                ),
            }
        )

    return results


def paired_binary_test(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    family: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for keys, group in frame.groupby(
        ["model_id", "dataset"],
        dropna=False,
    ):
        pivot = group.pivot_table(
            index="record_id",
            columns="condition",
            values=metric,
            aggfunc="last",
        )

        if (
            left not in pivot.columns
            or right not in pivot.columns
        ):
            continue

        paired = pivot[
            [left, right]
        ].dropna()

        if paired.empty:
            continue

        left_values = (
            paired[left]
            .astype(int)
            .to_numpy()
        )

        right_values = (
            paired[right]
            .astype(int)
            .to_numpy()
        )

        left_one_right_zero = int(
            np.sum(
                (left_values == 1)
                & (right_values == 0)
            )
        )

        left_zero_right_one = int(
            np.sum(
                (left_values == 0)
                & (right_values == 1)
            )
        )

        discordant = (
            left_one_right_zero
            + left_zero_right_one
        )

        if discordant == 0:
            p_value = 1.0
        else:
            p_value = float(
                binomtest(
                    min(
                        left_one_right_zero,
                        left_zero_right_one,
                    ),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

        results.append(
            {
                "family": family,
                "model_id": keys[0],
                "dataset": keys[1],
                "left_condition": left,
                "right_condition": right,
                "metric": metric,
                "paired_documents": len(paired),
                "left_rate": float(
                    np.mean(left_values)
                ),
                "right_rate": float(
                    np.mean(right_values)
                ),
                "rate_delta_right_minus_left": float(
                    np.mean(right_values)
                    - np.mean(left_values)
                ),
                "left_1_right_0": (
                    left_one_right_zero
                ),
                "left_0_right_1": (
                    left_zero_right_one
                ),
                "discordant_pairs": discordant,
                "mcnemar_exact_p": p_value,
            }
        )

    return results


def apply_holm(
    rows: list[dict[str, Any]],
    p_field: str,
    output_field: str,
) -> None:
    valid_indices: list[int] = []
    valid_p_values: list[float] = []

    for index, row in enumerate(rows):
        value = row.get(p_field)

        if value is None:
            continue

        try:
            numeric = float(value)
        except Exception:
            continue

        if not math.isfinite(numeric):
            continue

        valid_indices.append(index)
        valid_p_values.append(numeric)

    adjusted = holm_adjust(
        valid_p_values
    )

    for row in rows:
        row[output_field] = math.nan

    for index, value in zip(
        valid_indices,
        adjusted,
    ):
        rows[index][output_field] = value
        rows[index]["significant_holm_005"] = int(
            value < 0.05
        )


def entity_presence(
    prediction: dict[str, Any],
    dataset: str,
) -> tuple[
    dict[str, int],
    bool,
]:
    fields = ENTITY_FIELDS.get(
        dataset,
        [],
    )

    presence = {
        field: 0
        for field in fields
    }

    validation = prediction.get(
        "validation",
        {},
    )

    schema_valid = bool(
        validation.get("schema_valid")
    )

    parsed = prediction.get(
        "parsed_output"
    )

    if not isinstance(parsed, dict):
        return presence, schema_valid

    entities = parsed.get(
        "entities",
        [],
    )

    if not isinstance(entities, list):
        return presence, schema_valid

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_type = str(
            entity.get(
                "type",
                "",
            )
        ).strip()

        entity_text = str(
            entity.get(
                "text",
                "",
            )
        ).strip()

        if (
            entity_type in presence
            and entity_text
        ):
            presence[entity_type] = 1

    return presence, schema_valid


def bootstrap_binary_ci(
    values: list[int],
    seed: int,
    iterations: int = 5000,
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan

    array = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        array,
        size=(
            iterations,
            len(array),
        ),
        replace=True,
    )

    means = np.mean(
        samples,
        axis=1,
    )

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def calculate_idr(
    jobs: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    job_by_id = {
        job["job_id"]: job
        for job in jobs
    }

    indexed: dict[
        tuple[str, str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    for job_id, prediction in predictions.items():
        job = job_by_id.get(job_id)

        if job is None:
            continue

        condition = str(
            job.get("condition")
        )

        if condition not in MAIN_CONDITIONS:
            continue

        key = (
            str(job["model_id"]),
            str(job["dataset"]),
            str(job["record_id"]),
        )

        presence, schema_valid = entity_presence(
            prediction,
            str(job["dataset"]),
        )

        indexed[key][condition] = {
            "presence": presence,
            "schema_valid": schema_valid,
        }

    collected: dict[
        tuple[str, str, str, str],
        dict[str, list[int]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    transitions = list(
        zip(
            MAIN_CONDITIONS[:-1],
            MAIN_CONDITIONS[1:],
        )
    )

    for (
        model_id,
        dataset,
        record_id,
    ), conditions in indexed.items():
        fields = ENTITY_FIELDS.get(
            dataset,
            [],
        )

        for left, right in transitions:
            if (
                left not in conditions
                or right not in conditions
            ):
                continue

            previous = conditions[left]
            current = conditions[right]

            transition = f"{left}->{right}"

            field_events_inclusive = []
            field_events_valid = []

            for field in fields:
                previous_present = int(
                    previous["presence"].get(
                        field,
                        0,
                    )
                )

                current_present = int(
                    current["presence"].get(
                        field,
                        0,
                    )
                )

                inclusive_dropout = int(
                    previous_present == 1
                    and current_present == 0
                )

                collected[
                    (
                        model_id,
                        dataset,
                        transition,
                        field,
                    )
                ][
                    "inclusive"
                ].append(
                    inclusive_dropout
                )

                field_events_inclusive.append(
                    inclusive_dropout
                )

                if (
                    previous["schema_valid"]
                    and current["schema_valid"]
                ):
                    valid_dropout = (
                        inclusive_dropout
                    )

                    collected[
                        (
                            model_id,
                            dataset,
                            transition,
                            field,
                        )
                    ][
                        "valid_only"
                    ].append(
                        valid_dropout
                    )

                    field_events_valid.append(
                        valid_dropout
                    )

            any_inclusive = int(
                any(field_events_inclusive)
            )

            collected[
                (
                    model_id,
                    dataset,
                    transition,
                    "__ANY_FIELD__",
                )
            ][
                "inclusive"
            ].append(
                any_inclusive
            )

            if (
                previous["schema_valid"]
                and current["schema_valid"]
            ):
                any_valid = int(
                    any(field_events_valid)
                )

                collected[
                    (
                        model_id,
                        dataset,
                        transition,
                        "__ANY_FIELD__",
                    )
                ][
                    "valid_only"
                ].append(
                    any_valid
                )

    rows: list[dict[str, Any]] = []

    for key, variants in sorted(
        collected.items()
    ):
        model_id, dataset, transition, field = key

        for variant in (
            "valid_only",
            "inclusive",
        ):
            values = variants.get(
                variant,
                [],
            )

            if not values:
                continue

            seed = abs(
                hash(
                    (
                        model_id,
                        dataset,
                        transition,
                        field,
                        variant,
                    )
                )
            ) % 2**32

            low, high = bootstrap_binary_ci(
                values,
                seed=seed,
            )

            rows.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "transition": transition,
                    "field": field,
                    "variant": variant,
                    "paired_documents": len(values),
                    "dropout_events": int(
                        sum(values)
                    ),
                    "idr": float(
                        np.mean(values)
                    ),
                    "idr_ci_low": low,
                    "idr_ci_high": high,
                }
            )

    return rows


def progress_statistics(
    jobs: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    errors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs_by_model: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for job in jobs:
        jobs_by_model[
            str(job["model_id"])
        ].append(job)

    rows: list[dict[str, Any]] = []

    for model_id, model_jobs in sorted(
        jobs_by_model.items()
    ):
        expected_ids = {
            job["job_id"]
            for job in model_jobs
        }

        successful_ids = (
            expected_ids
            & set(predictions)
        )

        unresolved_error_ids = (
            expected_ids
            & set(errors)
        ) - successful_ids

        completed_predictions = [
            predictions[job_id]
            for job_id in successful_ids
        ]

        durations = [
            float(
                prediction.get(
                    "seconds_per_item_in_batch",
                    0.0,
                )
            )
            for prediction in completed_predictions
            if float(
                prediction.get(
                    "seconds_per_item_in_batch",
                    0.0,
                )
            ) > 0
        ]

        average_seconds = (
            float(np.mean(durations))
            if durations
            else math.nan
        )

        remaining = (
            len(expected_ids)
            - len(successful_ids)
        )

        projected_remaining_hours = (
            remaining * average_seconds / 3600
            if math.isfinite(average_seconds)
            else math.nan
        )

        json_valid = [
            int(
                bool(
                    prediction.get(
                        "validation",
                        {},
                    ).get(
                        "json_valid"
                    )
                )
            )
            for prediction in completed_predictions
        ]

        schema_valid = [
            int(
                bool(
                    prediction.get(
                        "validation",
                        {},
                    ).get(
                        "schema_valid"
                    )
                )
            )
            for prediction in completed_predictions
        ]

        rows.append(
            {
                "model_id": model_id,
                "expected_jobs": len(expected_ids),
                "predictions": len(successful_ids),
                "unresolved_errors": len(
                    unresolved_error_ids
                ),
                "remaining_without_prediction": remaining,
                "completion_percent": round(
                    100
                    * len(successful_ids)
                    / max(
                        len(expected_ids),
                        1,
                    ),
                    3,
                ),
                "json_valid_rate": (
                    float(np.mean(json_valid))
                    if json_valid
                    else math.nan
                ),
                "schema_valid_rate": (
                    float(np.mean(schema_valid))
                    if schema_valid
                    else math.nan
                ),
                "mean_seconds_per_item": (
                    average_seconds
                ),
                "projected_remaining_compute_hours": (
                    projected_remaining_hours
                ),
            }
        )

    return rows


def condition_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    numerical_metrics = [
        metric
        for metric in (
            CONTINUOUS_METRICS
            + BINARY_METRICS
        )
        if metric in metrics.columns
    ]

    grouped_rows: list[dict[str, Any]] = []

    for keys, group in metrics.groupby(
        [
            "model_id",
            "dataset",
            "phase",
            "condition",
        ],
        dropna=False,
    ):
        row: dict[str, Any] = {
            "model_id": keys[0],
            "dataset": keys[1],
            "phase": keys[2],
            "condition": keys[3],
            "documents": len(group),
        }

        for metric in numerical_metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            row[f"{metric}_mean"] = float(
                values.mean()
            )

            row[f"{metric}_median"] = float(
                values.median()
            )

            row[f"{metric}_std"] = float(
                values.std(
                    ddof=1,
                )
            ) if len(values) > 1 else 0.0

            row[f"{metric}_p95"] = float(
                values.quantile(0.95)
            )

        grouped_rows.append(row)

    return pd.DataFrame(
        grouped_rows
    )


def run_evaluator(
    root: Path,
    python_bin: Path,
) -> int:
    command = [
        str(python_bin),
        "-u",
        str(
            root
            / "scripts"
            / "evaluate_experiment.py"
        ),
        "--root",
        str(root),
    ]

    print(
        "=== RUN BASE EVALUATOR ===",
        flush=True,
    )

    process = subprocess.run(
        command,
        check=False,
    )

    print(
        f"[EVALUATOR_EXIT_CODE] "
        f"{process.returncode}",
        flush=True,
    )

    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/home/tahiti/PromptStressLab",
    )

    parser.add_argument(
        "--python-bin",
        default=(
            "/home/tahiti/Forensics/"
            ".venv_forensics/bin/python"
        ),
    )

    parser.add_argument(
        "--skip-evaluator",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).expanduser().resolve()

    python_bin = Path(
        args.python_bin
    ).expanduser().resolve()

    statistics_dir = (
        root
        / "outputs"
        / "statistics"
    )

    statistics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.skip_evaluator:
        run_evaluator(
            root,
            python_bin,
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

    error_paths = sorted(
        (
            root
            / "outputs"
            / "generations"
        ).glob(
            "*/errors.jsonl"
        )
    )

    predictions = latest_by_job(
        prediction_paths
    )

    errors = latest_by_job(
        error_paths
    )

    progress_rows = progress_statistics(
        jobs,
        predictions,
        errors,
    )

    write_csv(
        statistics_dir
        / "progress_by_model.csv",
        progress_rows,
    )

    metrics_path = (
        root
        / "outputs"
        / "metrics"
        / "job_metrics_with_aliases.csv"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Evaluator did not create {metrics_path}"
        )

    metrics = pd.read_csv(
        metrics_path
    )

    summary = condition_summary(
        metrics
    )

    summary.to_csv(
        statistics_dir
        / "condition_summary.csv",
        index=False,
    )

    main_frame = metrics[
        metrics["condition"].isin(
            MAIN_CONDITIONS
        )
        & (
            metrics["physical_or_alias"]
            == "physical"
        )
    ].copy()

    ablation_frame = metrics[
        metrics["condition"].isin(
            ABLATION_CONDITIONS
        )
    ].copy()

    continuous_tests: list[
        dict[str, Any]
    ] = []

    binary_tests: list[
        dict[str, Any]
    ] = []

    for left, right in zip(
        MAIN_CONDITIONS[:-1],
        MAIN_CONDITIONS[1:],
    ):
        for metric in CONTINUOUS_METRICS:
            if metric not in main_frame.columns:
                continue

            continuous_tests.extend(
                paired_continuous_test(
                    main_frame,
                    left,
                    right,
                    metric,
                    family="main_adjacent",
                )
            )

        for metric in BINARY_METRICS:
            if metric not in main_frame.columns:
                continue

            binary_tests.extend(
                paired_binary_test(
                    main_frame,
                    left,
                    right,
                    metric,
                    family="main_adjacent",
                )
            )

    for right in ABLATION_CONDITIONS[1:]:
        for metric in CONTINUOUS_METRICS:
            if metric not in ablation_frame.columns:
                continue

            continuous_tests.extend(
                paired_continuous_test(
                    ablation_frame,
                    "A0",
                    right,
                    metric,
                    family="ablation_vs_A0",
                )
            )

        for metric in BINARY_METRICS:
            if metric not in ablation_frame.columns:
                continue

            binary_tests.extend(
                paired_binary_test(
                    ablation_frame,
                    "A0",
                    right,
                    metric,
                    family="ablation_vs_A0",
                )
            )

    apply_holm(
        continuous_tests,
        p_field="p_value",
        output_field="p_holm",
    )

    apply_holm(
        binary_tests,
        p_field="mcnemar_exact_p",
        output_field="p_holm",
    )

    write_csv(
        statistics_dir
        / "paired_wilcoxon_tests.csv",
        continuous_tests,
    )

    write_csv(
        statistics_dir
        / "mcnemar_tests.csv",
        binary_tests,
    )

    idr_rows = calculate_idr(
        jobs,
        predictions,
    )

    write_csv(
        statistics_dir
        / "idr_summary.csv",
        idr_rows,
    )

    total_expected = len(jobs)
    total_predictions = len(
        set(predictions)
        & {
            job["job_id"]
            for job in jobs
        }
    )

    unresolved_errors = len(
        (
            set(errors)
            - set(predictions)
        )
        & {
            job["job_id"]
            for job in jobs
        }
    )

    result_summary = {
        "physical_jobs_expected": total_expected,
        "physical_predictions_found": (
            total_predictions
        ),
        "completion_percent": round(
            100
            * total_predictions
            / max(
                total_expected,
                1,
            ),
            4,
        ),
        "unresolved_errors": unresolved_errors,
        "prediction_files": [
            str(path)
            for path in prediction_paths
        ],
        "statistics_files": [
            "progress_by_model.csv",
            "condition_summary.csv",
            "paired_wilcoxon_tests.csv",
            "mcnemar_tests.csv",
            "idr_summary.csv",
        ],
        "note": (
            "PSI and field-wise volatility require "
            "an embedding backend and are not "
            "computed by this GPU-free pass."
        ),
    }

    (
        statistics_dir
        / "statistics_summary.json"
    ).write_text(
        json.dumps(
            result_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== STATISTICS COMPLETE ===")
    print(
        json.dumps(
            result_summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=== PROGRESS BY MODEL ===")

    if progress_rows:
        progress_frame = pd.DataFrame(
            progress_rows
        )

        display_columns = [
            "model_id",
            "predictions",
            "expected_jobs",
            "completion_percent",
            "unresolved_errors",
            "schema_valid_rate",
            "mean_seconds_per_item",
            "projected_remaining_compute_hours",
        ]

        print(
            progress_frame[
                display_columns
            ].to_string(
                index=False,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
