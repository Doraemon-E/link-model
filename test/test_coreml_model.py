#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


COREML_DIR = Path("models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml")
TOKENIZER_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
TARGET_LANGUAGE = "English"
SOURCE_TEXT = "今天下午三点半在5A会议室开会。"
SYSTEM_PROMPT = "You are a translation engine."
MAX_NEW_TOKENS = 64
CONTEXT_LENGTH = 256
COMPUTE_UNIT = "cpuAndNeuralEngine"
PRINT_GENERATED_TEXT = True


def _resolve_model_path(coreml_dir: Path) -> Path:
    candidates = [
        coreml_dir / "Compiled" / "causal_lm.mlmodelc",
        coreml_dir / "causal_lm.mlpackage",
        coreml_dir / "hy_mt_w8_from_torch.mlpackage",
        coreml_dir / "hy_mt_fp16.mlpackage",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"no coreml model found under: {coreml_dir}")


def _resolve_compute_unit(name: str) -> ct.ComputeUnit:
    if name == "cpuAndNeuralEngine":
        return ct.ComputeUnit.CPU_AND_NE
    if name == "all":
        return ct.ComputeUnit.ALL
    if name == "cpuAndGPU":
        return ct.ComputeUnit.CPU_AND_GPU
    if name == "cpuOnly":
        return ct.ComputeUnit.CPU_ONLY
    raise RuntimeError(f"unsupported compute unit: {name}")


def _build_prompt(target_language: str, source_text: str) -> str:
    return (
        f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
        f"{source_text}"
    )


def _load_eos_ids(tokenizer_dir: Path, tokenizer) -> set[int]:
    eos_ids: set[int] = set()
    generation_config_path = tokenizer_dir / "generation_config.json"
    if generation_config_path.is_file():
        payload = json.loads(generation_config_path.read_text(encoding="utf-8"))
        eos_raw = payload.get("eos_token_id")
        if isinstance(eos_raw, int):
            eos_ids.add(eos_raw)
        elif isinstance(eos_raw, list):
            eos_ids.update(int(v) for v in eos_raw if isinstance(v, int))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    return eos_ids


def _load_coreml_predictor(model_path: Path, compute_unit: ct.ComputeUnit):
    if model_path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(
            str(model_path),
            compute_units=compute_unit,
        )
    return ct.models.MLModel(
        str(model_path),
        compute_units=compute_unit,
    )


def _predict_with_optional_state(predictor, inputs: dict[str, np.ndarray], state):
    try:
        return predictor.predict(inputs, state=state)
    except TypeError:
        return predictor.predict(inputs)


def _run() -> dict[str, object]:
    coreml_dir = COREML_DIR.expanduser().resolve()
    tokenizer_dir = TOKENIZER_DIR.expanduser().resolve()
    model_path = _resolve_model_path(coreml_dir)
    compute_unit = _resolve_compute_unit(COMPUTE_UNIT)

    tmpdir = coreml_dir / "CoreMLTemp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = str(tmpdir)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    eos_ids = _load_eos_ids(tokenizer_dir, tokenizer)
    prompt = _build_prompt(TARGET_LANGUAGE, SOURCE_TEXT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = prompt_ids[-CONTEXT_LENGTH:]
    if not prompt_ids:
        raise RuntimeError("prompt ids is empty")

    load_start = time.perf_counter()
    predictor = _load_coreml_predictor(model_path, compute_unit)
    load_elapsed = time.perf_counter() - load_start

    state = predictor.make_state()
    if state is None:
        raise RuntimeError("failed to create CoreML state")

    current_input = np.asarray([prompt_ids], dtype=np.int32)
    generated_ids: list[int] = []
    step_times: list[float] = []

    for _ in range(MAX_NEW_TOKENS):
        step_start = time.perf_counter()
        output = _predict_with_optional_state(
            predictor,
            {"input_ids": current_input},
            state,
        )
        step_times.append(time.perf_counter() - step_start)

        if "logits" not in output:
            raise RuntimeError(f"output missing logits, keys={list(output.keys())}")
        logits = np.asarray(output["logits"])
        if logits.ndim == 3:
            next_scores = logits[0, -1, :]
        elif logits.ndim == 2:
            next_scores = logits[0, :]
        else:
            raise RuntimeError(f"unexpected logits shape: {logits.shape}")

        next_token = int(np.argmax(next_scores))
        if next_token in eos_ids:
            break
        generated_ids.append(next_token)
        current_input = np.asarray([[next_token]], dtype=np.int32)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("generated empty text")

    return {
        "status": "passed",
        "model_path": str(model_path),
        "tokenizer_dir": str(tokenizer_dir),
        "compute_unit": COMPUTE_UNIT,
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(generated_ids),
        "load_seconds": round(load_elapsed, 3),
        "first_token_latency_seconds": round(step_times[0], 3) if step_times else None,
        "generate_seconds": round(float(sum(step_times)), 3),
        "generated_text": text,
    }


def main() -> int:
    try:
        result = _run()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "inference_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "coreml_dir": str(COREML_DIR.expanduser().resolve()),
                    "tokenizer_dir": str(TOKENIZER_DIR.expanduser().resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    summary = {
        "status": "passed",
        "model_path": result["model_path"],
        "tokenizer_dir": result["tokenizer_dir"],
        "inference": {
            "compute_unit": result["compute_unit"],
            "load_seconds": result["load_seconds"],
            "first_token_latency_seconds": result["first_token_latency_seconds"],
            "generate_seconds": result["generate_seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "output_tokens": result["output_tokens"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if PRINT_GENERATED_TEXT:
        print("\n===== GENERATED TEXT =====")
        print(result["generated_text"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
