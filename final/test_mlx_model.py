#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler


def _apply_mlx_compat_patch() -> None:
    """
    mlx_lm 内部仍在调用已弃用的 mx.metal.device_info。
    这里在运行时替换为 mx.device_info，避免告警并兼容未来版本。
    """
    metal = getattr(mx, "metal", None)
    modern_device_info = getattr(mx, "device_info", None)
    if metal is None or modern_device_info is None:
        return
    try:
        setattr(metal, "device_info", modern_device_info)
    except Exception:
        pass


MODEL_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
TARGET_LANGUAGE = "English"
SOURCE_TEXT = "今天下午三点半在5A会议室开会。"
PROMPT = (
    f"将以下文本翻译为{TARGET_LANGUAGE}，注意字词、语法、语义语境，"
    "并只输出翻译结果：\n\n"
    f"{SOURCE_TEXT}"
)
MAX_TOKENS = 64
PRINT_GENERATED_TEXT = True
TOP_K = 20
TOP_P = 0.6
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.05


def _load_mlx_and_generate(
    model_dir: Path,
    prompt: str,
    max_tokens: int,
) -> dict[str, object]:

    load_start = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    load_elapsed = time.perf_counter() - load_start

    if getattr(tokenizer, "chat_template", None) is not None and hasattr(
        tokenizer, "apply_chat_template"
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt_for_generate = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    else:
        prompt_for_generate = prompt

    sampler = make_sampler(
        temp=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )
    logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)

    gen_start = time.perf_counter()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_for_generate,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    gen_elapsed = time.perf_counter() - gen_start

    text = output if isinstance(output, str) else str(output)
    if isinstance(prompt_for_generate, list):
        prompt_tokens = len(prompt_for_generate)
    elif isinstance(prompt_for_generate, str) and hasattr(tokenizer, "encode"):
        prompt_tokens = len(tokenizer.encode(prompt_for_generate))
    else:
        prompt_tokens = None
    output_tokens = (
        len(tokenizer.encode(text)) if hasattr(tokenizer, "encode") else None
    )

    return {
        "status": "passed" if text.strip() else "failed",
        "load_seconds": round(load_elapsed, 3),
        "generate_seconds": round(gen_elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_text": text,
    }


def main() -> int:
    _apply_mlx_compat_patch()
    model_dir = MODEL_DIR.expanduser().resolve()

    try:
        inference = _load_mlx_and_generate(
            model_dir=model_dir,
            prompt=PROMPT,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics only.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "inference_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "model_dir": str(model_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    summary = {
        "status": "passed" if inference["status"] == "passed" else "failed",
        "model_dir": str(model_dir),
        "inference": {
            key: value for key, value in inference.items() if key != "output_text"
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if PRINT_GENERATED_TEXT:
        print("\n===== GENERATED TEXT =====")
        print(inference["output_text"])

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
