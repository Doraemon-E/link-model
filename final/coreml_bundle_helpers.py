from __future__ import annotations

import json
import shutil
from pathlib import Path


DEFAULT_PREFERRED_COMPUTE_UNITS = ["cpuAndNeuralEngine", "cpuOnly", "all"]

REQUIRED_RUNTIME_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

OPTIONAL_RUNTIME_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def copy_runtime_files(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_required: list[str] = []

    for file_name in REQUIRED_RUNTIME_FILES:
        source = model_dir / file_name
        if not source.is_file():
            missing_required.append(file_name)
            continue
        shutil.copy2(source, output_dir / file_name)

    if missing_required:
        raise RuntimeError(
            f"missing required runtime files in model_dir={model_dir}: {missing_required}"
        )

    for file_name in OPTIONAL_RUNTIME_FILES:
        source = model_dir / file_name
        if source.is_file():
            shutil.copy2(source, output_dir / file_name)


def write_translation_manifest(
    output_dir: Path,
    *,
    model_file_name: str,
    context_length: int,
    preferred_compute_units: list[str] | None = None,
) -> Path:
    units = preferred_compute_units or DEFAULT_PREFERRED_COMPUTE_UNITS
    manifest = {
        "version": 1,
        "family": "coreml_causal_llm",
        "promptStyle": "hy_mt_coreml_chat_v1",
        "contextLength": context_length,
        "preferredComputeUnits": units,
        "coreml": {
            "kind": "mlpackage",
            "path": model_file_name,
            "inputName": "input_ids",
            "outputName": "logits",
        },
        "tokenizer": {
            "kind": "huggingface_tokenizer_json",
            "tokenizerJson": "tokenizer.json",
            "tokenizerConfig": "tokenizer_config.json",
            "generationConfig": "generation_config.json",
            "chatTemplate": "chat_template.jinja",
        },
    }
    manifest_path = output_dir / "translation-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path
