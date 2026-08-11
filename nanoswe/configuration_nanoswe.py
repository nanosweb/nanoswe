"""HuggingFace-style config for the nanoswe / nanoswe model, used by vLLM."""

from transformers import PretrainedConfig


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Mirrors nanoswe.gpt.has_ve — alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


class NanoChatConfig(PretrainedConfig):
    model_type = "nanoswe"

    def __init__(
        self,
        vocab_size: int = 32768,
        padded_vocab_size: int | None = None,
        hidden_size: int = 1536,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        max_position_embeddings: int = 32768,
        intermediate_size: int | None = None,
        rope_theta: float = 100000.0,
        tie_word_embeddings: bool = False,
        window_pattern: str = "L",
        smear_gate_channels: int = 24,
        ve_gate_channels: int = 12,
        qk_norm_scale: float = 1.2,
        logit_softcap: float = 15.0,
        pad_vocab_to_multiple: int = 64,
        n_experts: int = 0,
        n_experts_active: int = 2,
        n_shared_experts: int = 0,
        expert_hidden_mult: float = 0.25,
        **kwargs,
    ):
        if padded_vocab_size is None:
            padded_vocab_size = (
                (vocab_size + pad_vocab_to_multiple - 1)
                // pad_vocab_to_multiple
            ) * pad_vocab_to_multiple
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.window_pattern = window_pattern
        self.smear_gate_channels = smear_gate_channels
        self.ve_gate_channels = ve_gate_channels
        self.qk_norm_scale = qk_norm_scale
        self.logit_softcap = logit_softcap
        self.pad_vocab_to_multiple = pad_vocab_to_multiple
        self.head_dim = hidden_size // num_attention_heads
        self.value_embed_layers = [
            i for i in range(num_hidden_layers) if has_ve(i, num_hidden_layers)
        ]
        self.n_experts = n_experts
        self.n_experts_active = n_experts_active
        self.n_shared_experts = n_shared_experts
        self.expert_hidden_mult = expert_hidden_mult
        if n_experts > 0:
            # Mirror nanoswe.gpt.MoEMLP: h = round(4·d·m_h), rounded UP to multiple of 128.
            raw_h = int(round(4 * hidden_size * expert_hidden_mult))
            self.expert_hidden = ((raw_h + 127) // 128) * 128
        else:
            self.expert_hidden = 0

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @classmethod
    def from_gpt_config(cls, gpt_config, **overrides) -> "NanoChatConfig":
        """Build from a nanoswe.gpt.GPTConfig dataclass."""
        return cls(
            vocab_size=gpt_config.vocab_size,
            hidden_size=gpt_config.n_embd,
            num_hidden_layers=gpt_config.n_layer,
            num_attention_heads=gpt_config.n_head,
            num_key_value_heads=gpt_config.n_kv_head,
            max_position_embeddings=gpt_config.sequence_len,
            window_pattern=gpt_config.window_pattern,
            n_experts=getattr(gpt_config, "n_experts", 0),
            n_experts_active=getattr(gpt_config, "n_experts_active", 2),
            n_shared_experts=getattr(gpt_config, "n_shared_experts", 0),
            expert_hidden_mult=getattr(gpt_config, "expert_hidden_mult", 0.25),
            **overrides,
        )
