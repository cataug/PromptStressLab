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
