from transformers import AutoModelForCausalLM, AutoTokenizer
import coremltools as ct
import coremltools.optimize as cto
import torch
from pathlib import Path
from stateful_hunyuan_for_coreml import StatefulHunYuanForCoreML
from torch.export import Dim
import numpy as np

DEFAULT_MODEL_DIR = Path("models/translation/downloaded/hy-mt1.5-1.8b")
DEFAULT_OUTPUT_DIR = Path(
    "models/translation/converted/coreml-int8/hy-mt1.5-1.8b-coreml"
)
DEFAULT_CONTEXT_LENGTH = 1024


def _convert_coreml(model_dir: Path, output_dir: Path, context_length: int):
    # 加载并固定PyTorch模型状态
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # 推理模式, 关闭dropout等训练行为
    model.eval()

    # 控制注意力机制（Attention）的实现方式，使用最原始，最通用的方式，也最稳定
    model.config._attn_implementation = "eager"

    # 最基础、最兼容的 RoPE 实现
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb.rope_type = "default"

    wrapper = StatefulHunYuanForCoreML(model=model, max_cache_len=context_length)
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
    fp16_mlpackage_path = output_dir / "hy_mt_fp16.mlpackage"

    # 自定义CoreML 转换的 流水线
    default_passes = list(ct.PassPipeline.DEFAULT.passes)
    custom_pass_pipeline = ct.PassPipeline(
        pass_names=default_passes, pipeline_name="hy_mt_coreml_export"
    )
    # 防止 kv cache 被破坏
    custom_pass_pipeline.remove_passes(["common::canonicalize_inplace_pattern"])

    # 转换成CoreML
    fp16_model = ct.convert(
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
    fp16_model.save(str(fp16_mlpackage_path))


def _quantize_coreml_w8(
    model_dir: Path,
    output_dir: Path,
):
    fp16_model = ct.models.MLModel(str(model_dir / "hy_mt_fp16.mlpackage"))
    w8_mlpackage_path = output_dir / "hy_mt_w8.mlpackage"
    # 对已经转换好的 Core ML mlprogram 做 W8 量化
    op_config = cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",  # 默认；先用这个最稳
        dtype=np.int8,  # W8
        granularity="per_channel",  # 8-bit 先用 per_channel
        weight_threshold=2048,  # 默认值；想覆盖更多小权重可调低到 512
    )
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    w8_model = cto.coreml.linear_quantize_weights(fp16_model, config=config)
    w8_model.save(str(w8_mlpackage_path))


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
    _convert_coreml(
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )
    _quantize_coreml_w8(
        model_dir=DEFAULT_OUTPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    )


if __name__ == "__main__":
    run()
