import torch
from transformers import AutoModelForCausalLM
from helper.slice_update_key_value_cache import SliceUpdateKeyValueCache


class StatefulHunYuanForCoreML(torch.nn.Module):
    def __init__(self, model: AutoModelForCausalLM, max_cache_len: int) -> None:
        super().__init__()
        self.model = model
        self.max_cache_len = max_cache_len
        config = model.config
        num_layers = int(config.num_hidden_layers)
        self.num_layers = num_layers
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(config.head_dim)
        layer_cache_shape = (1, num_kv_heads, max_cache_len, head_dim)

        # 每一层都增加kv cache
        for layer_idx in range(num_layers):
            self.register_buffer(
                f"key_cache_{layer_idx}",
                torch.zeros(layer_cache_shape, dtype=torch.float16),
            )
            self.register_buffer(
                f"value_cache_{layer_idx}",
                torch.zeros(layer_cache_shape, dtype=torch.float16),
            )

        self.register_buffer("cache_position", torch.zeros((1,), dtype=torch.float16))

    def _layer_key_caches(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"key_cache_{layer_idx}")
            for layer_idx in range(self.num_layers)
        ]

    def _layer_value_caches(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"value_cache_{layer_idx}")
            for layer_idx in range(self.num_layers)
        ]

    def reset_cache(self) -> None:
        for cache in self._layer_key_caches():
            cache.zero_()
        for cache in self._layer_value_caches():
            cache.zero_()
        self.cache_position.zero_()

    # 重写前向推理，利用kv缓存
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(torch.int64)

        # 获取kv cache的位置
        start_position = self.cache_position.to(torch.int64)

        # 获取 输入的 token 的长度
        local_positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.int64
        )

        cache_position = local_positions + start_position
        kv_positions = torch.arange(
            self.max_cache_len, device=input_ids.device, dtype=torch.int64
        )
        # mask attension, 只能看之前的数据
        allowed = kv_positions.unsqueeze(0) <= cache_position.unsqueeze(1)
        zero_mask = torch.zeros(
            allowed.shape, dtype=torch.float16, device=input_ids.device
        )
        neg_inf_mask = torch.full(
            allowed.shape, -1.0e4, dtype=torch.float16, device=input_ids.device
        )
        attention_mask = (
            torch.where(allowed, zero_mask, neg_inf_mask).unsqueeze(0).unsqueeze(0)
        )

        # 构造 kv_cache 对象
        past_key_values = SliceUpdateKeyValueCache(
            key_caches=self._layer_key_caches(),
            value_caches=self._layer_value_caches(),
            max_cache_len=self.max_cache_len,
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            return_dict=True,
        )

        next_position = torch.clamp(cache_position[-1:] + 1, max=self.max_cache_len)
        self.cache_position.copy_(next_position.to(self.cache_position.dtype))
        return outputs.logits.to(torch.float16)
