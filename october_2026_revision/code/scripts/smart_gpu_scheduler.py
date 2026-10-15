#!/usr/bin/env python3

from __future__ import annotations

import fcntl
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/home/tahiti/PromptStressLab")
PYTHON_BIN = Path(
    "/home/tahiti/Forensics/.venv_forensics/bin/python"
)

POLL_SECONDS = 15
MAX_PARALLEL = 2
MAX_RETRIES = 8

MODEL_PLAN = [
    {
        "model_id": "mistral_7b_instruct_v03",
        "priority": 1,
        "min_free_gib": 13.5,
        "max_batch_size": 1,
        "token_budget": 5000,
        "must_run_alone": False,
    },
    {
        "model_id": "qwen3_8b",
        "priority": 2,
        "min_free_gib": 23.0,
        "max_batch_size": 2,
        "token_budget": 7000,
        "must_run_alone": False,
    },
    {
        "model_id": "gemma_3_12b_it",
        "priority": 3,
        "min_free_gib": 35.0,
        "max_batch_size": 2,
        "token_budget": 7000,
        "must_run_alone": True,
    },
]


@dataclass
class Worker:
    model_id: str
    process: subprocess.Popen[str]
    log_handle: Any
    thread: threading.Thread


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
                row = json.loads(line)
            except Exception:
                continue

            if isinstance(row, dict):
                rows.append(row)

    return rows


def completed_ids(path: Path) -> set[str]:
    ids: set[str] = set()

    for row in read_jsonl(path):
        job_id = row.get("job_id")

        if isinstance(job_id, str):
            ids.add(job_id)

    return ids


def totals_by_model() -> dict[str, int]:
    totals: dict[str, int] = {}

    jobs = read_jsonl(
        ROOT / "manifests" / "experiment_jobs.jsonl"
    )

    for job in jobs:
        model_id = str(job["model_id"])
        totals[model_id] = totals.get(model_id, 0) + 1

    return totals


def progress(
    model_id: str,
    total: int,
) -> dict[str, int]:
    output_dir = (
        ROOT
        / "outputs"
        / "generations"
        / model_id
    )

    done = completed_ids(
        output_dir / "predictions.jsonl"
    )

    errors = completed_ids(
        output_dir / "errors.jsonl"
    )

    return {
        "total": total,
        "done": len(done),
        "errors_recorded": len(errors),
        "remaining": max(total - len(done), 0),
    }


def gpu_state() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,"
        "memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or "nvidia-smi failed"
        )

    first_line = result.stdout.strip().splitlines()[0]
    parts = [
        part.strip()
        for part in first_line.split(",")
    ]

    return {
        "index": int(parts[0]),
        "name": parts[1],
        "used_mib": int(parts[2]),
        "free_mib": int(parts[3]),
        "total_mib": int(parts[4]),
        "utilization": int(parts[5]),
        "free_gib": int(parts[3]) / 1024,
    }


def stream_worker(
    model_id: str,
    process: subprocess.Popen[str],
    log_handle: Any,
    output_queue: queue.Queue[tuple[str, str]],
) -> None:
    assert process.stdout is not None

    for line in process.stdout:
        log_handle.write(line)
        log_handle.flush()

        output_queue.put(
            (
                model_id,
                line.rstrip("\n"),
            )
        )


def worker_environment() -> dict[str, str]:
    env = os.environ.copy()

    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": (
                "/home/tahiti/.cache/huggingface"
            ),
            "HF_HUB_CACHE": (
                "/home/tahiti/.cache/huggingface/hub"
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": (
                "expandable_segments:True"
            ),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PSL_DEVICE_MAP": "auto",
            "PYTHONPATH": (
                str(ROOT / "scripts")
                + os.pathsep
                + env.get("PYTHONPATH", "")
            ),
        }
    )

    return env


def start_worker(
    config: dict[str, Any],
    output_queue: queue.Queue[tuple[str, str]],
) -> Worker:
    model_id = config["model_id"]

    command = [
        str(PYTHON_BIN),
        "-u",
        str(
            ROOT
            / "scripts"
            / "run_model_experiment.py"
        ),
        "--root",
        str(ROOT),
        "--model-id",
        model_id,
        "--gpu-physical-id",
        "0",
        "--max-batch-size",
        str(config["max_batch_size"]),
        "--token-budget",
        str(config["token_budget"]),
        "--retry-errors",
    ]

    log_dir = ROOT / "logs" / "experiments"
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / f"{model_id}.smart.log"
    )

    log_handle = log_path.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=worker_environment(),
    )

    thread = threading.Thread(
        target=stream_worker,
        args=(
            model_id,
            process,
            log_handle,
            output_queue,
        ),
        daemon=True,
    )

    thread.start()

    print(
        f"[WORKER_STARTED] model={model_id} "
        f"pid={process.pid} "
        f"batch={config['max_batch_size']} "
        f"token_budget={config['token_budget']}",
        flush=True,
    )

    return Worker(
        model_id=model_id,
        process=process,
        log_handle=log_handle,
        thread=thread,
    )


def run_evaluation() -> int:
    command = [
        str(PYTHON_BIN),
        "-u",
        str(
            ROOT
            / "scripts"
            / "evaluate_experiment.py"
        ),
        "--root",
        str(ROOT),
    ]

    print(
        "=== STARTING AUTOMATIC EVALUATION ===",
        flush=True,
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=worker_environment(),
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            f"[EVALUATION] {line.rstrip()}",
            flush=True,
        )

    return process.wait()


def main() -> int:
    lock_path = (
        ROOT
        / "outputs"
        / "smart_scheduler.lock"
    )

    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_handle = lock_path.open("w")

    try:
        fcntl.flock(
            lock_handle,
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        print(
            "[ERROR] Smart scheduler is already running.",
            flush=True,
        )
        return 2

    totals = totals_by_model()

    expected_models = {
        item["model_id"]
        for item in MODEL_PLAN
    }

    missing = expected_models - set(totals)

    if missing:
        print(
            f"[ERROR] Missing models in manifest: "
            f"{sorted(missing)}",
            flush=True,
        )
        return 2

    print(
        "=== SMART AUTOMATIC GPU SCHEDULER ===",
        flush=True,
    )

    print(f"[ROOT] {ROOT}", flush=True)
    print(f"[TOTALS] {totals}", flush=True)
    print(
        "[PLAN] Mistral now -> "
        "Qwen when RIFT frees memory -> "
        "Gemma alone",
        flush=True,
    )

    running: dict[str, Worker] = {}
    retries: dict[str, int] = {
        model_id: 0
        for model_id in totals
    }

    retry_after: dict[str, float] = {
        model_id: 0.0
        for model_id in totals
    }

    output_queue: queue.Queue[
        tuple[str, str]
    ] = queue.Queue()

    last_heartbeat = 0.0

    while True:
        while True:
            try:
                model_id, line = (
                    output_queue.get_nowait()
                )
            except queue.Empty:
                break

            print(
                f"[{model_id}] {line}",
                flush=True,
            )

        for model_id, worker in list(
            running.items()
        ):
            return_code = worker.process.poll()

            if return_code is None:
                continue

            worker.thread.join(timeout=5)
            worker.log_handle.close()

            del running[model_id]

            model_progress = progress(
                model_id,
                totals[model_id],
            )

            print(
                f"[WORKER_FINISHED] "
                f"model={model_id} "
                f"code={return_code} "
                f"progress={model_progress}",
                flush=True,
            )

            if model_progress["remaining"] > 0:
                retries[model_id] += 1
                retry_after[model_id] = (
                    time.time() + 60
                )

                print(
                    f"[AUTO_RETRY_SCHEDULED] "
                    f"model={model_id} "
                    f"attempt={retries[model_id]}/"
                    f"{MAX_RETRIES}",
                    flush=True,
                )

        all_progress = {
            model_id: progress(
                model_id,
                total,
            )
            for model_id, total in totals.items()
        }

        if all(
            item["remaining"] == 0
            for item in all_progress.values()
        ):
            print(
                "=== ALL 8526 GENERATIONS COMPLETE ===",
                flush=True,
            )

            print(
                json.dumps(
                    all_progress,
                    indent=2,
                ),
                flush=True,
            )

            evaluation_code = run_evaluation()

            print(
                f"[EVALUATION_EXIT_CODE] "
                f"{evaluation_code}",
                flush=True,
            )

            print(
                "=== PIPELINE_FINISHED ===",
                flush=True,
            )

            return evaluation_code

        failed_permanently = [
            model_id
            for model_id, count in retries.items()
            if (
                count >= MAX_RETRIES
                and all_progress[model_id][
                    "remaining"
                ] > 0
                and model_id not in running
            )
        ]

        if failed_permanently:
            print(
                "[FAILED_AFTER_MAX_RETRIES] "
                f"{failed_permanently}",
                flush=True,
            )

            return 1

        try:
            gpu = gpu_state()
        except Exception as error:
            print(
                f"[GPU_QUERY_ERROR] {error}",
                flush=True,
            )

            time.sleep(POLL_SECONDS)
            continue

        now = time.time()

        candidates = [
            config
            for config in MODEL_PLAN
            if (
                all_progress[
                    config["model_id"]
                ]["remaining"] > 0
                and config["model_id"]
                not in running
                and now
                >= retry_after[
                    config["model_id"]
                ]
            )
        ]

        candidates.sort(
            key=lambda item: item["priority"]
        )

        launched = False

        for config in candidates:
            model_id = config["model_id"]

            if len(running) >= MAX_PARALLEL:
                break

            if config["must_run_alone"]:
                if running:
                    continue

                if (
                    gpu["free_gib"]
                    < config["min_free_gib"]
                ):
                    continue

            else:
                if any(
                    MODEL_PLAN_ITEM[
                        "must_run_alone"
                    ]
                    for MODEL_PLAN_ITEM in MODEL_PLAN
                    if MODEL_PLAN_ITEM["model_id"]
                    in running
                ):
                    continue

                if (
                    gpu["free_gib"]
                    < config["min_free_gib"]
                ):
                    continue

            running[model_id] = start_worker(
                config,
                output_queue,
            )

            launched = True

            # После запуска одной модели ждём
            # обновления nvidia-smi перед следующей.
            break

        if (
            launched
            or now - last_heartbeat >= 30
        ):
            running_names = sorted(running)

            remaining_text = ", ".join(
                f"{model_id}="
                f"{item['remaining']}"
                for model_id, item
                in sorted(all_progress.items())
            )

            print(
                "[HEARTBEAT] "
                f"running={running_names} | "
                f"gpu_free={gpu['free_gib']:.1f}GiB "
                f"util={gpu['utilization']}% | "
                f"remaining: {remaining_text}",
                flush=True,
            )

            last_heartbeat = now

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
