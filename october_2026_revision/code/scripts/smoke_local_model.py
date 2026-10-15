#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)

    raise RuntimeError(f"No JSONL records found in {path}")


def gpu_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "free_gib": round(
            free_bytes / 1024**3,
            3,
        ),
        "total_gib": round(
            total_bytes / 1024**3,
            3,
        ),
        "allocated_gib": round(
            torch.cuda.memory_allocated(0) / 1024**3,
            3,
        ),
        "reserved_gib": round(
            torch.cuda.memory_reserved(0) / 1024**3,
            3,
        ),
    }


def build_prompt(record: dict[str, Any]) -> str:
    text = record["input_text"]

    return f"""You are a scientific information extraction system.

Extract entities from the scientific abstract and return ONLY one valid JSON object.

Use exactly these keys:
{{
  "Task": [],
  "Method": [],
  "Metric": [],
  "Material": [],
  "OtherScientificTerm": [],
  "Generic": []
}}

Rules:
- Copy entity wording from the abstract.
- Do not invent entities.
- Use an empty list if a category is absent.
- Do not add keys.
- Do not output markdown or explanations.
- Output JSON only.

Abstract:
{text}
"""


def render_chat(
    tokenizer: Any,
    prompt: str,
    model_key: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    if model_key == "qwen3_8b":
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


def load_tokenizer(
    model_path: Path,
) -> Any:
    try:
        return AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as fast_error:
        print(
            f"[TOKENIZER_FAST_FAILED] {fast_error}",
            flush=True,
        )

        return AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )


def load_model(
    model_path: Path,
    architecture: str,
) -> Any:
    common_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }

    model_class = None

    if architecture == "gemma3":
        model_class = getattr(
            transformers,
            "AutoModelForImageTextToText",
            None,
        )

    if model_class is None:
        model_class = transformers.AutoModelForCausalLM

    try:
        return model_class.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            **common_kwargs,
        )

    except TypeError:
        return model_class.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            **common_kwargs,
        )


def extract_json(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed, cleaned

    except Exception:
        pass

    start_positions = [
        index
        for index, char in enumerate(cleaned)
        if char == "{"
    ]

    decoder = json.JSONDecoder()

    for start in start_positions:
        try:
            parsed, consumed = decoder.raw_decode(
                cleaned[start:]
            )

            if isinstance(parsed, dict):
                json_text = cleaned[
                    start : start + consumed
                ]

                return parsed, json_text

        except Exception:
            continue

    return None, None


def validate_schema(
    parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_keys = [
        "Task",
        "Method",
        "Metric",
        "Material",
        "OtherScientificTerm",
        "Generic",
    ]

    if parsed is None:
        return {
            "json_valid": False,
            "exact_keys": False,
            "all_values_lists": False,
            "missing_keys": expected_keys,
            "extra_keys": [],
        }

    actual_keys = list(parsed.keys())

    return {
        "json_valid": True,
        "exact_keys": set(actual_keys) == set(expected_keys),
        "all_values_lists": all(
            isinstance(parsed.get(key), list)
            for key in expected_keys
        ),
        "missing_keys": sorted(
            set(expected_keys) - set(actual_keys)
        ),
        "extra_keys": sorted(
            set(actual_keys) - set(expected_keys)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--model-key",
        required=True,
    )

    parser.add_argument(
        "--model-path",
        required=True,
    )

    parser.add_argument(
        "--architecture",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_path = Path(args.model_path).resolve()

    output_directory = (
        root
        / "outputs"
        / "smoke"
        / args.model_key
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_directory
        / "smoke_result.json"
    )

    raw_output_path = (
        output_directory
        / "raw_output.txt"
    )

    print("=== LOCAL MODEL SMOKE TEST ===", flush=True)
    print(f"[MODEL_KEY] {args.model_key}", flush=True)
    print(f"[MODEL_PATH] {model_path}", flush=True)
    print(f"[ARCHITECTURE] {args.architecture}", flush=True)
    print(f"[PYTHON] {sys.executable}", flush=True)
    print(f"[TORCH] {torch.__version__}", flush=True)
    print(
        f"[TRANSFORMERS] {transformers.__version__}",
        flush=True,
    )

    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]

    print("=== ENVIRONMENT ===", flush=True)

    for key in env_keys:
        print(
            f"{key}={os.environ.get(key)}",
            flush=True,
        )

    before_status = gpu_status()

    print(
        f"[GPU_BEFORE] {json.dumps(before_status)}",
        flush=True,
    )

    result: dict[str, Any] = {
        "model_key": args.model_key,
        "model_path": str(model_path),
        "architecture": args.architecture,
        "status": "started",
        "gpu_before": before_status,
    }

    try:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this environment"
            )

        dataset_path = (
            root
            / "data"
            / "normalized"
            / "scierc"
            / "test.jsonl"
        )

        record = read_first_jsonl(dataset_path)
        prompt = build_prompt(record)

        result["record_id"] = record.get("record_id")
        result["prompt_characters"] = len(prompt)

        print(
            f"[RECORD_ID] {record.get('record_id')}",
            flush=True,
        )

        print(
            f"[PROMPT_CHARACTERS] {len(prompt)}",
            flush=True,
        )

        print("[LOAD_TOKENIZER]", flush=True)

        tokenizer_start = time.perf_counter()

        tokenizer = load_tokenizer(model_path)

        tokenizer_seconds = (
            time.perf_counter()
            - tokenizer_start
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print(
            f"[TOKENIZER_OK] {tokenizer_seconds:.3f} s",
            flush=True,
        )

        rendered_prompt = render_chat(
            tokenizer,
            prompt,
            args.model_key,
        )

        encoded = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        prompt_tokens = int(
            encoded["input_ids"].shape[-1]
        )

        result["prompt_tokens"] = prompt_tokens

        print(
            f"[PROMPT_TOKENS] {prompt_tokens}",
            flush=True,
        )

        print("[LOAD_MODEL]", flush=True)

        load_start = time.perf_counter()

        model = load_model(
            model_path,
            args.architecture,
        )

        model.eval()

        load_seconds = (
            time.perf_counter()
            - load_start
        )

        after_load_status = gpu_status()

        print(
            f"[MODEL_OK] {load_seconds:.3f} s",
            flush=True,
        )

        print(
            "[GPU_AFTER_LOAD] "
            f"{json.dumps(after_load_status)}",
            flush=True,
        )

        model_device = next(
            model.parameters()
        ).device

        encoded = {
            key: value.to(model_device)
            for key, value in encoded.items()
        }

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        print("[GENERATE]", flush=True)

        generation_start = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )

        generation_seconds = (
            time.perf_counter()
            - generation_start
        )

        generated_tokens = generated[
            0,
            encoded["input_ids"].shape[-1] :,
        ]

        output_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        output_token_count = int(
            generated_tokens.shape[-1]
        )

        raw_output_path.write_text(
            output_text,
            encoding="utf-8",
        )

        parsed, extracted_json_text = extract_json(
            output_text
        )

        validation = validate_schema(parsed)

        after_generation_status = gpu_status()

        result.update(
            {
                "status": (
                    "ok"
                    if validation["json_valid"]
                    else "invalid_json"
                ),
                "tokenizer_seconds": round(
                    tokenizer_seconds,
                    4,
                ),
                "model_load_seconds": round(
                    load_seconds,
                    4,
                ),
                "generation_seconds": round(
                    generation_seconds,
                    4,
                ),
                "output_tokens": output_token_count,
                "raw_output": output_text,
                "extracted_json_text": extracted_json_text,
                "parsed_output": parsed,
                "validation": validation,
                "gpu_after_load": after_load_status,
                "gpu_after_generation": (
                    after_generation_status
                ),
            }
        )

        print(
            f"[GENERATION_SECONDS] "
            f"{generation_seconds:.3f}",
            flush=True,
        )

        print(
            f"[OUTPUT_TOKENS] {output_token_count}",
            flush=True,
        )

        print(
            "[VALIDATION] "
            f"{json.dumps(validation)}",
            flush=True,
        )

        print("=== RAW OUTPUT ===", flush=True)
        print(output_text, flush=True)
        print("=== END RAW OUTPUT ===", flush=True)

        if validation["json_valid"]:
            print("[SMOKE_OK]", flush=True)
        else:
            print("[SMOKE_INVALID_JSON]", flush=True)

        del generated
        del encoded
        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as error:
        result.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "gpu_at_error": gpu_status(),
            }
        )

        print(
            f"[SMOKE_ERROR] "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        traceback.print_exc()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[RESULT] {result_path}", flush=True)
    print("=== SMOKE_PROCESS_FINISHED ===", flush=True)

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
