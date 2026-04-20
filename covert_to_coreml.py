from transformers import AutoModelForCausalLM
import coremltools as ct
import coremltools.optimize as cto
import torch
from pathlib import Path
import subprocess
from helper.stateful_hunyuan_for_coreml import StatefulHunYuanForCoreML
from torch.export import Dim
import numpy as np
from mlx_lm import convert

from helper.coreml_bundle_helpers import copy_runtime_files, write_translation_manifest


DEFAULT_MODEL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")

DEFAULT_COREML_OUTPUT_DIR = Path(
    "models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml"
)
DEFAULT_COREML_PACKAGED_ZIP = Path(
    "models/translation/packaged/hy-mt1.5-1.8b-coreml-int8.zip"
)

DEFAULT_MLX_OUTPUT_DIR = Path("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx")
DEFAULT_MLX_PACKAGED_ZIP = Path(
    "models/translation/packaged/hy-mt1.5-1.8b-mlx-int8.zip"
)

DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_MLX_Q_BITS = 8


def _load_base_model(model_dir: Path) -> torch.nn.Module:
    # 加载并固定PyTorch模型状态
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # 推理模式, 关闭dropout等训练行为
    model.eval()

    # 更稳的 attention 路径
    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"

    # 更保守的 rope 设置
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.rope_type = "default"

    return model


def _load_quantized_torch_model(model_dir: Path) -> torch.nn.Module:
    """
    先在 Torch 侧做 weight-only PTQ，然后返回“带压缩信息”的 torch model。
    这样 ct.convert() 时会自动尝试生成压缩后的 Core ML 表达。
    """
    model = _load_base_model(model_dir)

    # 只量化 Linear，先别碰 Embedding / LayerNorm / 输出头以外的特殊结构
    # 这一版先求稳，后面再逐步扩大覆盖范围。
    config = cto.torch.quantization.PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int8",
                # 文档示例里常见 per_block；而量化算法说明里提到 8-bit per-channel 通常精度更稳。
                # 对 LLM 你可以先试 per_channel；如果当前版本报配置不支持，再退到 per_block + block_size。
                "granularity": "per_channel",
            },
            # "module_type_configs": {
            #     torch.nn.Linear: None,
            # },
        }
    )

    quantizer = cto.torch.quantization.PostTrainingQuantizer(model, config)
    quantized_model = quantizer.compress()
    quantized_model.eval()
    return quantized_model


def _convert_coreml(model_dir: Path, output_dir: Path, context_length: int) -> Path:
    # 先在 Torch 侧量化/压缩
    quantized_torch_model = _load_quantized_torch_model(model_dir)

    wrapper = StatefulHunYuanForCoreML(
        model=quantized_torch_model, max_cache_len=context_length
    )
    wrapper.eval()
    wrapper.reset_cache()

    sample_input_ids = torch.ones((1, 8), dtype=torch.int32)
    dynamic_shapes = {"input_ids": {1: Dim("query_length", min=1, max=context_length)}}

    exported_program = torch.export.export(
        wrapper, args=(sample_input_ids,), dynamic_shapes=dynamic_shapes, strict=False
    )
    # 算子分解，转成CoreML兼容的算子
    exported_program = exported_program.run_decompositions({})

    # 构建输出路径
    output_dir.mkdir(parents=True, exist_ok=True)
    coreml_path = output_dir / "hy_mt_w8_from_torch.mlpackage"

    # 自定义CoreML 转换的 流水线
    default_passes = list(ct.PassPipeline.DEFAULT.passes)
    custom_pass_pipeline = ct.PassPipeline(
        pass_names=default_passes, pipeline_name="hy_mt_coreml_export"
    )
    # 防止 kv cache 被破坏
    custom_pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

    # 转换成CoreML
    coreml_model = ct.convert(
        exported_program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=custom_pass_pipeline,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(
                    1,
                    ct.RangeDim(lower_bound=1, upper_bound=context_length, default=1),
                ),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=_build_coreml_states(wrapper),
    )
    coreml_model.save(str(coreml_path))
    return coreml_path


def _convert_mlx(
    model_dir: Path,
    output_dir: Path,
    q_bits: int = 8,
) -> Path:
    """
    把 Hugging Face / 本地模型目录转成 MLX 量化版本。
    这里走 mlx-lm 的 Python API，不再用 subprocess。
    """
    # 官方公开示例是 convert(repo, quantize=True, ...)
    # 这里再补上本地输出目录和 q_bits。
    convert(
        str(model_dir),
        mlx_path=str(output_dir),
        quantize=True,
        q_bits=q_bits,
    )

    return output_dir


def _make_zip_with_parent(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(
        ["/usr/bin/ditto", "-c", "-k", "--keepParent", str(source_dir), str(zip_path)],
        check=True,
    )


# helper
def _make_state(name: str, shape: tuple[int, ...], dtype) -> ct.StateType:
    return ct.StateType(
        wrapped_type=ct.TensorType(
            shape=shape,
            dtype=dtype,
        ),
        name=name,
    )


def _build_coreml_states(
    wrapper,
    cache_dtype=np.float16,
    position_dtype=np.float16,
) -> list[ct.StateType]:
    states: list[ct.StateType] = []

    for layer_idx in range(wrapper.num_layers):
        key_cache = getattr(wrapper, f"key_cache_{layer_idx}")
        states.append(
            _make_state(
                name=f"key_cache_{layer_idx}",
                shape=tuple(key_cache.shape),
                dtype=cache_dtype,
            )
        )

    for layer_idx in range(wrapper.num_layers):
        value_cache = getattr(wrapper, f"value_cache_{layer_idx}")
        states.append(
            _make_state(
                name=f"value_cache_{layer_idx}",
                shape=tuple(value_cache.shape),
                dtype=cache_dtype,
            )
        )

    states.append(
        _make_state(
            name="cache_position",
            shape=tuple(wrapper.cache_position.shape),
            dtype=position_dtype,
        )
    )

    return states


def run():
    # 1. Core ML W8
    coreml_path = _convert_coreml(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_COREML_OUTPUT_DIR,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )
    copy_runtime_files(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_COREML_OUTPUT_DIR,
    )
    write_translation_manifest(
        output_dir=DEFAULT_COREML_OUTPUT_DIR,
        model_file_name=coreml_path.name,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )
    _make_zip_with_parent(
        source_dir=DEFAULT_COREML_OUTPUT_DIR,
        zip_path=DEFAULT_COREML_PACKAGED_ZIP,
    )

    # 2. MLX W8
    # mlx_path = _convert_mlx(
    #     model_dir=DEFAULT_MODEL_DIR,
    #     output_dir=DEFAULT_MLX_OUTPUT_DIR,
    #     q_bits=DEFAULT_MLX_Q_BITS,
    # )
    # _make_zip_with_parent(
    #     source_dir=mlx_path,
    #     zip_path=DEFAULT_MLX_PACKAGED_ZIP,
    # )


if __name__ == "__main__":
    run()
