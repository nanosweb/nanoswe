"""vLLM-compatible NanoChatForCausalLM.

Mirrors nanoswe.gpt.GPT exactly so a converted state_dict (after stripping
`_orig_mod.`) loads with no key remapping.

Two execution modes:

* **Standalone** — call `forward(input_ids)` with shape `(B, T)` and the model
  runs the same forward pass as `nanoswe.gpt.GPT.forward(idx)` (no kv-cache).
  Used for the logit-equivalence test against nanoswe.

* **vLLM** — call `forward(input_ids, positions, ...)` with flat `(N,)` tensors
  and a populated `vllm.forward_context`. Uses `vllm.model_executor.layers.attention.Attention`
  for PagedAttention. Smear's per-sequence "previous embedding" is tracked via
  a per-batch-slot tensor reset each forward.

The 10 architectural quirks are preserved verbatim:
1. smear (previous-token mixing)        — `_apply_smear_*`
2. backout (mid-trunk residual subtract)— in `_run_trunk`
3. value embeddings on alternating layers — `value_embeds`, `NanoChatAttention`
4. per-layer resid_lambdas / x0_lambdas — in `_run_trunk`
5. QK-norm with 1.2 split scale         — `NanoChatAttention.forward_*`
6. RMSNorm with no learnable params     — `_norm`
7. relu² MLP                            — `NanoChatMLP`
8. logit softcap (15·tanh(·/15))        — `compute_logits`
9. untied wte / lm_head                 — separate `nn.Embedding` + `nn.Linear`
10. RoPE base=100000                    — `_precompute_rope`
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoswe.configuration_nanoswe import NanoChatConfig, has_ve

try:
    from vllm.compilation.decorators import support_torch_compile
    _HAVE_VLLM_COMPILE = True
except ImportError:
    _HAVE_VLLM_COMPILE = False
    def support_torch_compile(cls):
        return cls


# -----------------------------------------------------------------------------
# Helpers


def _norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm with no learnable weight (matches nanoswe.gpt.norm)."""
    return F.rms_norm(x, (x.size(-1),))


class _Linear(nn.Linear):
    """nn.Linear that casts its weight to the input dtype in forward.

    Matches `nanoswe.gpt.Linear` so that mixed-precision behavior (fp32 master
    weights, bf16 activations) is identical between the two implementations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.linear(x, self.weight.to(dtype=x.dtype))


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split rotary embedding (matches nanoswe.gpt.apply_rotary_emb).

    `x` is `(..., head_dim)`, `cos` and `sin` are broadcastable to `(..., head_dim/2)`.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def _precompute_rope(seq_len: int, head_dim: int, base: float, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (seq_len, head_dim/2) — flat over the time axis."""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


# -----------------------------------------------------------------------------
# Sub-modules


class NanoChatMLP(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        n = config.hidden_size
        self.c_fc = _Linear(n, 4 * n, bias=False)
        self.c_proj = _Linear(4 * n, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu²
        x = self.c_proj(x)
        return x


class NanoChatMoEMLP(nn.Module):
    """Inference port of nanoswe.gpt.MoEMLP (DeepSeekMoE-style, aux-loss-free routing).

    Eval-only — drops the training-time z-loss / load tracking. Mirrors the trained
    state_dict layout (router.weight, router_bias, expert_c_fc, expert_c_proj, optional
    shared_c_{fc,proj}) so weights load by name without remapping.

    Sized for the EP=1 saved checkpoint: each rank holds the full (E, h, d) and (E, d, h)
    stacks. vLLM serves single-GPU so EP is moot at inference.
    """

    def __init__(self, config: NanoChatConfig):
        super().__init__()
        d = config.hidden_size
        h = config.expert_hidden
        assert config.n_experts > 0 and h > 0, "NanoChatMoEMLP requires n_experts > 0 + expert_hidden > 0"
        self.d = d
        self.h = h
        self.n_experts = config.n_experts
        self.n_active = config.n_experts_active
        self.n_shared = config.n_shared_experts

        self.router = _Linear(d, self.n_experts, bias=False)
        # router_bias is a (non-persistent in training, but persistent in the on-disk ckpt)
        # buffer — DeepSeek-V3 aux-loss-free bias-shifted topk. Register as buffer so it loads
        # via load_weights' buffer pathway.
        self.register_buffer("router_bias", torch.zeros(self.n_experts), persistent=True)

        # Expert weight stacks. nn.Linear convention (out, in):
        #   expert_c_fc:   (E, h, d)
        #   expert_c_proj: (E, d, h)
        self.expert_c_fc = nn.Parameter(torch.empty(self.n_experts, h, d))
        self.expert_c_proj = nn.Parameter(torch.empty(self.n_experts, d, h))

        if self.n_shared > 0:
            shared_hidden = h * self.n_shared
            self.shared_c_fc = _Linear(d, shared_hidden, bias=False)
            self.shared_c_proj = _Linear(shared_hidden, d, bias=False)
        else:
            self.shared_c_fc = None
            self.shared_c_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts (B, T, d) or flat (N, d). Returns same leading shape."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d)
        N = x_flat.shape[0]

        # Routing in fp32 for stability — matches training.
        router_logits = self.router(x_flat).float()  # (N, E)
        biased = router_logits + self.router_bias.float()
        _, top_idx = biased.topk(self.n_active, dim=-1)  # (N, k) int64

        # Un-biased softmax for weights, then re-normalize over top-k.
        all_weights = F.softmax(router_logits, dim=-1)
        weights = all_weights.gather(1, top_idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Fan out: each token visits k experts. Sort by expert id so grouped_mm
        # sees contiguous per-expert blocks.
        flat_top = top_idx.reshape(-1)                                                  # (N*k,)
        flat_tok = torch.arange(N, device=x_flat.device).repeat_interleave(self.n_active)
        sort_perm = flat_top.argsort()
        sorted_expert = flat_top[sort_perm]
        sorted_token = flat_tok[sort_perm]
        sorted_w = weights.reshape(-1)[sort_perm].to(x_flat.dtype)

        permuted = x_flat.index_select(0, sorted_token)
        # scatter_add instead of bincount: bincount does an internal CPU sync
        # (size determination), which breaks cudagraph capture. scatter_add is
        # fully on-device and produces the same per-expert counts.
        counts = torch.zeros(self.n_experts, dtype=torch.int64, device=x_flat.device)
        counts.scatter_add_(0, sorted_expert.long(), torch.ones_like(sorted_expert, dtype=torch.int64))
        offsets = counts.cumsum(0).to(torch.int32)

        # grouped_mm wants (in, out) layout for the right operand.
        fc_w = self.expert_c_fc.to(dtype=permuted.dtype).transpose(-1, -2).contiguous()    # (E, d, h)
        proj_w = self.expert_c_proj.to(dtype=permuted.dtype).transpose(-1, -2).contiguous()  # (E, h, d)
        h_perm = torch._grouped_mm(permuted, fc_w, offsets)
        h_perm = F.relu(h_perm).square()
        out_perm = torch._grouped_mm(h_perm, proj_w, offsets)

        out_perm = out_perm * sorted_w.unsqueeze(-1)
        # Unpermute: scatter rows back to (token, k-slot) layout then sum over k.
        inv_perm = sort_perm.argsort()
        out_unperm = out_perm.index_select(0, inv_perm)
        out = out_unperm.view(N, self.n_active, self.d).sum(dim=1)

        if self.shared_c_fc is not None:
            shared = self.shared_c_proj(F.relu(self.shared_c_fc(x_flat)).square())
            out = out + shared

        return out.view(orig_shape)


class NanoChatAttention(nn.Module):
    """Self-attention block with QK-norm, RoPE, and value-embedding injection.

    Matches `nanoswe.gpt.CausalSelfAttention` parameter names exactly so the
    state_dict loads without remapping.
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.head_dim = config.head_dim
        n_embd = config.hidden_size
        self.qk_norm_scale = config.qk_norm_scale

        self.c_q = _Linear(n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = _Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = _Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = _Linear(n_embd, n_embd, bias=False)

        self.has_ve = has_ve(layer_idx, config.num_hidden_layers)
        self.ve_gate_channels = config.ve_gate_channels
        if self.has_ve:
            self.ve_gate = _Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

        # vLLM-mode attention layer is created lazily by NanoChatForCausalLM
        # when constructed via vllm_config (see `attach_vllm_attention`).
        self.attn = None

    # ---- shared helpers -------------------------------------------------

    def _qkv(self, x: torch.Tensor, ve: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to Q, K, V with shapes `(*lead, n_head, head_dim)` etc.

        `ve` is `(*lead, n_kv_head * head_dim)` or None.  Mixes value embeddings
        into V *before* attention so PagedAttention writes the modified V.
        """
        lead = x.shape[:-1]
        q = self.c_q(x).view(*lead, self.n_head, self.head_dim)
        k = self.c_k(x).view(*lead, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(*lead, self.n_kv_head, self.head_dim)
        if ve is not None and self.has_ve:
            ve_r = ve.view(*lead, self.n_kv_head, self.head_dim)
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve_r
        return q, k, v

    def _qk_finalize(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE → QK-norm → 1.2 split scale.  Caller supplies cos/sin already shaped."""
        q = _norm(q)
        k = _norm(k)
        q = q * self.qk_norm_scale
        k = k * self.qk_norm_scale
        return q, k

    # ---- standalone forward (for equivalence testing) ------------------

    def forward_standalone(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Standalone (B, T, C) forward — mirrors nanoswe.gpt CausalSelfAttention.

        nanoswe scales `q` and `k` each by 1.2 then calls flash_attn with the
        default scale of `1/sqrt(d)`.  We do the same with SDPA: pre-scale
        q,k and let SDPA's default scale apply.
        """
        B, T, _ = x.shape
        q, k, v = self._qkv(x, ve)  # (B, T, H, D)
        cos_b = cos.view(1, T, 1, -1)
        sin_b = sin.view(1, T, 1, -1)
        q = _apply_rotary_emb(q, cos_b, sin_b)
        k = _apply_rotary_emb(k, cos_b, sin_b)
        q, k = self._qk_finalize(q, k)
        # (B, T, H, D) → (B, H, T, D) for SDPA
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k_ = k_.repeat_interleave(repeat, dim=1)
            v_ = v_.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1).contiguous()
        return self.c_proj(y)

    # ---- vLLM forward (continuous batching) ----------------------------

    def forward_vllm(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """vLLM forward.  `x` is `(N, C)` flat, `cos`/`sin` are `(N, head_dim/2)`."""
        N = x.shape[0]
        q, k, v = self._qkv(x, ve)  # (N, H, D), (N, KH, D), (N, KH, D)
        cos_b = cos.view(N, 1, -1)
        sin_b = sin.view(N, 1, -1)
        q = _apply_rotary_emb(q, cos_b, sin_b)
        k = _apply_rotary_emb(k, cos_b, sin_b)
        q, k = self._qk_finalize(q, k)
        # vLLM Attention takes (N, H*D) layout
        q_flat = q.reshape(N, self.n_head * self.head_dim)
        k_flat = k.reshape(N, self.n_kv_head * self.head_dim)
        v_flat = v.reshape(N, self.n_kv_head * self.head_dim)
        out = self.attn(q_flat, k_flat, v_flat)  # (N, H*D)
        return self.c_proj(out)


class NanoChatBlock(nn.Module):
    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.attn = NanoChatAttention(config, layer_idx)
        self.mlp = NanoChatMoEMLP(config) if config.n_experts > 0 else NanoChatMLP(config)

    def forward_standalone(self, x, ve, cos, sin):
        x = x + self.attn.forward_standalone(_norm(x), ve, cos, sin)
        x = x + self.mlp(_norm(x))
        return x

    def forward_vllm(self, x, ve, cos, sin):
        x = x + self.attn.forward_vllm(_norm(x), ve, cos, sin)
        x = x + self.mlp(_norm(x))
        return x


# -----------------------------------------------------------------------------
# Top-level model


@support_torch_compile
class NanoChatForCausalLM(nn.Module):
    """vLLM-facing class.  Parameter names match nanoswe.gpt.GPT exactly."""

    def __init__(
        self,
        *,
        vllm_config=None,
        prefix: str = "",
        config: Optional[NanoChatConfig] = None,
    ):
        super().__init__()
        if vllm_config is not None and config is None:
            config = vllm_config.model_config.hf_config
        assert config is not None, "Either vllm_config or config must be provided"
        self.config = config
        self._vllm_config = vllm_config

        n = config.hidden_size
        n_layer = config.num_hidden_layers
        kv_dim = config.num_key_value_heads * config.head_dim
        padded_vocab_size = config.padded_vocab_size

        # Match nanoswe.gpt structure exactly so state_dict keys are identical.
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, n),
            "h": nn.ModuleList([NanoChatBlock(config, i) for i in range(n_layer)]),
        })
        self.lm_head = _Linear(n, padded_vocab_size, bias=False)

        # Per-layer scalars
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))

        # Smear
        self.smear_gate = _Linear(config.smear_gate_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Backout
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Value embeddings on alternating layers (last layer always included)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(n_layer)
            if has_ve(i, n_layer)
        })

        # RoPE buffers — recomputed deterministically; not persisted.
        # Over-compute by 10× to match nanoswe behavior (gpt.py:195).
        rotary_seq_len = config.max_position_embeddings * 10
        # Resolve rope_theta: vLLM 0.17 keeps it as a top-level attr, but
        # 0.20+/transformers v5 may move it into a `rope_parameters` dict
        # depending on which patch path ran.  Try both, fall back to the
        # nanoswe default (100000.0).
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(config, "rope_parameters", None) or {}
            rope_theta = rope_params.get("rope_theta", 100000.0)
        cos, sin = _precompute_rope(
            rotary_seq_len, config.head_dim, rope_theta,
            torch.float32, torch.device("cpu"),
        )
        # Register RoPE tables as buffers; vLLM's `model.to(device, dtype=bf16)`
        # then moves them to the GPU at the runtime dtype.  We rely on this
        # one-time conversion so the hot-path `self.cos[positions]` is just a
        # gather — no per-forward `.to()` allocation (which crashes cudagraph
        # capture, and which Inductor would otherwise materialize as a
        # `buf.copy_` inside the captured graph).
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Smear "previous embedding" cache for vLLM decode — indexed by the
        # physical block id of each request's position-0 KV slot (stable
        # per-request key).  Stale entries from finished requests are filtered
        # out by the `positions[start] > 0` mask on the read side: fresh
        # prefills always start at position 0, so they consult a zeroed slot
        # and overwrite it on the same call's save step.
        #
        # Registered as a buffer so vLLM's `model.to(device, dtype=bf16)`
        # places it correctly without per-forward `.to()` calls (which fail
        # inside cudagraph capture).  Sized to `_smear_cache_size`.  We use
        # 131072 as the default — large enough to cover the worst case we
        # serve on: d20 on B200-180GB has KV ≈ 153 GiB / 1.6 MiB per block
        # ≈ 96 K blocks (d34 needs ≤ 50 K, A100/H100 ≤ 50 K).  Memory cost is
        # ~320 MiB at hidden=1280 / bf16 — negligible compared to KV.  If a
        # future config exceeds 131072, override `smear_cache_size` in
        # config.json.  An undersized cache does NOT crash (the `clamp` in
        # `_apply_smear_vllm` keeps writes in-bounds) but silently corrupts
        # cross-forward smear state by collision, which would degrade eval.
        self._smear_cache_size = getattr(config, "smear_cache_size", 131072)
        # Allocate in bf16 from the start so model.to(bf16) is a no-op on the
        # buffer (avoiding any accidental dtype conversion that could
        # interfere with cudagraph capture's view of the tensor identity).
        self.register_buffer(
            "_smear_prev_cache",
            torch.zeros(self._smear_cache_size, n, dtype=torch.bfloat16),
            persistent=False,
        )

        # vLLM-mode integration: lazily attach Attention layers + RoPE if vllm_config provided.
        if vllm_config is not None:
            self._attach_vllm_attention(vllm_config, prefix)

    # ---- vLLM Attention attachment ------------------------------------

    def _attach_vllm_attention(self, vllm_config, prefix: str) -> None:
        from vllm.model_executor.layers.attention import Attention

        cache_config = vllm_config.cache_config
        head_dim = self.config.head_dim
        scale = head_dim ** -0.5  # SDPA-style; the 1.2 split scale is in q/k

        for i, block in enumerate(self.transformer.h):
            block.attn.attn = Attention(
                num_heads=self.config.num_attention_heads,
                head_size=head_dim,
                scale=scale,
                num_kv_heads=self.config.num_key_value_heads,
                cache_config=cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.transformer.h.{i}.attn.attn",
            )

        # vLLM's `model.to(device, dtype)` doesn't always move non-persistent
        # buffers; we move RoPE + smear cache to the runtime device/dtype
        # eagerly here so the hot-path forward never needs a `.to()` call
        # (Inductor would compile a `.to()` to a captured `buf.copy_` that
        # cudagraph capture rejects).
        device = getattr(getattr(vllm_config, "device_config", None), "device", None)
        model_cfg = getattr(vllm_config, "model_config", None)
        model_dtype = getattr(model_cfg, "dtype", None) if model_cfg is not None else None
        if device is not None and model_dtype is not None:
            self.cos = self.cos.to(device=device, dtype=model_dtype)
            self.sin = self.sin.to(device=device, dtype=model_dtype)
            self._smear_prev_cache = self._smear_prev_cache.to(device=device, dtype=model_dtype)

    # ---- standalone forward (for equivalence with nanoswe.gpt.GPT) ----

    def forward_standalone(self, idx: torch.Tensor) -> torch.Tensor:
        """Run a `(B, T)` forward identical to `nanoswe.gpt.GPT.forward(idx)`.

        Returns logits of shape `(B, T, vocab_size)` after softcap.
        """
        B, T = idx.shape
        assert T > 1, "Standalone forward expects T > 1 (matches nanoswe training path)"
        device = idx.device

        # Move RoPE to the right device on first use.
        cos_table = self.cos[:T].to(device=device)
        sin_table = self.sin[:T].to(device=device)

        x = self.transformer.wte(idx)               # (B, T, C)
        compute_dtype = x.dtype                      # follow embedding dtype
        x = x.to(compute_dtype)
        x = _norm(x)

        # Smear (training-style — first token unchanged)
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
            self.smear_gate(x[:, 1:, : self.config.smear_gate_channels])
        )
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x = self._run_trunk_standalone(x, idx, cos_table, sin_table)
        return self._compute_logits_softcap(x)

    def _run_trunk_standalone(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ) -> torch.Tensor:
        x0 = x
        n_layer = self.config.num_hidden_layers
        backout_layer = n_layer // 2
        x_backout: Optional[torch.Tensor] = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x = block.forward_standalone(x, ve, cos_table, sin_table)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = _norm(x)
        return x

    def _compute_logits_softcap(self, x: torch.Tensor) -> torch.Tensor:
        """lm_head + crop padding + 15·tanh(·/15) softcap in fp32."""
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]
        logits = logits.float()
        cap = self.config.logit_softcap
        logits = cap * torch.tanh(logits / cap)
        return logits

    # ---- vLLM forward (continuous batching) ----------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Dispatch:
        - `(B, T)` input_ids, no `positions`  → standalone (returns logits).
        - flat `(N,)` input_ids + `positions` → vLLM (returns hidden states).
        """
        if positions is None:
            assert input_ids is not None
            return self.forward_standalone(input_ids)
        return self._forward_vllm(input_ids, positions, inputs_embeds=inputs_embeds)

    def _forward_vllm(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = positions.device
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            assert input_ids is not None
            x = self.transformer.wte(input_ids)

        # RoPE indexed by positions.  `self.cos` was registered as a buffer in
        # fp32 on CPU and moved/cast by vLLM's `model.to(cuda, bf16)`, so a
        # plain gather suffices — no `.to()` on the hot path (which would
        # block cudagraph capture / Inductor compile).
        cos = self.cos[positions]
        sin = self.sin[positions]
        if cos.dtype != x.dtype:                  # standalone test path only
            cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        x = x.to(x.dtype)
        x = _norm(x)
        x = self._apply_smear_vllm(x, positions)
        x = self._run_trunk_vllm(x, input_ids, cos, sin)
        return x

    def _apply_smear_vllm(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Branchless smear with cross-forward state — graph-safe.

        Per request r in the packed batch (start_r = qsl[r], end_r = qsl[r+1]):

            prev[start_r]            = cache[bid_r]   if positions[start_r] > 0 else 0
            prev[start_r+1 : end_r]  = x[start_r : end_r-1]                  (in-batch shift)

        Then `x = x + gate * prev` and `cache[bid_r] = x[end_r-1]` (pre-smear).

        Implementation packs all per-request work into vectorized index
        ops — no Python loop, no dict lookup, no `.item()` / `.tolist()`
        sync — so cudagraph capture and torch.compile both succeed.

        Construction:
          1. `prev = pad(x[:-1], top=1)` does an in-batch right-shift with a
             zero in slot 0.  This is correct for every position EXCEPT the
             first token of each request (slots `qsl[:-1]`), where the shifted
             value belongs to a different request.
          2. We then `index_copy_(0, qsl[:-1], cached)` — overwriting those
             slots with `cache[block_id_r] * (positions[start_r] > 0)`.  The
             mask zeros out fresh prefills; the cache value covers
             continuations (decode + chunked-prefill follow-ups).
          3. After applying smear, `index_copy_` saves `x[ends-1]` into
             `cache[block_id_r]` for the next forward.
        """
        N = x.shape[0]
        if N == 0:
            return x

        # Smear gate — graph-safe (no state).
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
            self.smear_gate(x[..., : self.config.smear_gate_channels])
        )  # (N, 1)

        meta = self._get_attn_metadata()
        block_table = self._get_block_table(meta) if meta is not None else None
        qsl = self._get_query_start_loc(meta) if meta is not None else None

        # No metadata at all (vLLM profile/dummy run).  Use a positions-driven
        # mask to stitch in-batch boundaries; cross-forward state isn't
        # available here but profile runs don't need it.
        if qsl is None:
            if N <= 1:
                return x
            prev = F.pad(x[:-1], (0, 0, 1, 0))                               # (N, C)
            same_req = (positions[1:] == positions[:-1] + 1) & (positions[1:] > 0)
            mask = F.pad(same_req.to(x.dtype).unsqueeze(-1), (0, 0, 1, 0))   # (N, 1)
            return x + gate * (prev * mask)

        # Fast path — full per-request handling via vectorized index ops.
        starts = qsl[:-1].to(torch.long)                # (R,)
        ends = qsl[1:].to(torch.long) - 1               # (R,)

        # In-batch shift with implicit 0 at slot 0.
        prev = F.pad(x[:-1], (0, 0, 1, 0))              # (N, C)

        if block_table is None:
            # qsl available but no block_table (rare).  Just clear request
            # boundaries and skip cross-forward cache.
            zeros_at_starts = torch.zeros(starts.shape[0], x.shape[-1],
                                          device=x.device, dtype=x.dtype)
            prev = prev.index_copy(0, starts, zeros_at_starts)
            return x + gate * prev

        # Cross-forward cache path.  Cache is a registered buffer (sized once
        # at __init__) so its identity is fixed by cudagraph capture time.
        # Clamp `bids` to a guaranteed-valid range — padded slots in
        # full-cudagraph batches can carry stale block_table values; without
        # this clamp, an OOB index_copy_ asserts even for slots whose output
        # would be discarded.  Real block ids are bounded by num_gpu_blocks
        # (≪ cache size), so clamping doesn't lose any real entry.
        bids = block_table[:, 0].clamp(min=0, max=self._smear_cache_size - 1).to(torch.long)  # (R,)
        cached = self._smear_prev_cache.index_select(0, bids)          # (R, C)
        # Mask out fresh prefills (positions[start_r] == 0 → no prior context).
        pos_at_start = positions.index_select(0, starts).to(torch.long)
        is_cont = (pos_at_start > 0).to(x.dtype).unsqueeze(-1)         # (R, 1)
        cached = cached.to(dtype=x.dtype) * is_cont
        # Inject cache at request starts (overwrites the wrong shifted value).
        prev.index_copy_(0, starts, cached)

        x_smeared = x + gate * prev

        # Save the last pre-smear x of each request for the next forward.
        # vLLM's full-decode cudagraph captures this in-place scatter; the
        # cache buffer is a registered nn buffer, so its identity is stable
        # across capture and replay.  We mark the save with `torch.compiler
        # .disable` so Inductor does NOT functionalize the index_copy_
        # (its codegen inserts a `buf.copy_` that's illegal during cudagraph
        # capture) — it stays as an eager op inside the captured graph.
        last_pre_smear = x.index_select(0, ends)
        self._save_smear_state(bids, last_pre_smear)
        return x_smeared

    @torch.compiler.disable
    def _save_smear_state(self, bids: torch.Tensor, last_pre_smear: torch.Tensor) -> None:
        self._smear_prev_cache.index_copy_(0, bids, last_pre_smear)

    @staticmethod
    def _get_query_start_loc(meta) -> Optional[torch.Tensor]:
        qsl = getattr(meta, "query_start_loc", None)
        if qsl is None or not isinstance(qsl, torch.Tensor):
            return None
        return qsl

    @staticmethod
    def _get_block_table(meta) -> Optional[torch.Tensor]:
        bt = getattr(meta, "block_table", None)
        if bt is None:
            bt = getattr(meta, "block_table_tensor", None)
        if bt is None or not isinstance(bt, torch.Tensor):
            return None
        return bt

    def _get_attn_metadata(self):
        """Pull a single AttentionMetadata out of `vllm.forward_context`."""
        try:
            from vllm.forward_context import get_forward_context, is_forward_context_available

            if not is_forward_context_available():
                return None
            ctx = get_forward_context()
        except Exception:
            return None
        meta = ctx.attn_metadata
        if isinstance(meta, dict):
            for v in meta.values():
                if v is not None:
                    return v
            return None
        if isinstance(meta, list):
            for d in meta:
                if isinstance(d, dict):
                    for v in d.values():
                        if v is not None:
                            return v
            return None
        return meta

    def _run_trunk_vllm(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x0 = x
        n_layer = self.config.num_hidden_layers
        backout_layer = n_layer // 2
        x_backout: Optional[torch.Tensor] = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = None
            if str(i) in self.value_embeds:
                assert input_ids is not None, "value_embeds require input_ids in vLLM mode"
                ve = self.value_embeds[str(i)](input_ids).to(x.dtype)
            x = block.forward_vllm(x, ve, cos, sin)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = _norm(x)
        return x

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """vLLM hook — lm_head + crop padded vocab + 15·tanh(·/15) softcap.

        We bypass `LogitsProcessor` since our `lm_head` is a plain `_Linear`
        without the `quant_method` attribute that processor expects.  This is
        fine for single-GPU, unquantized serving — the regime we're targeting.
        """
        logits = F.linear(hidden_states, self.lm_head.weight.to(hidden_states.dtype))
        logits = logits[..., : self.config.vocab_size]
        cap = self.config.logit_softcap
        return cap * torch.tanh(logits.float() / cap)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM expects this method on text-generation models."""
        return self.transformer.wte(input_ids)

    # ---- weight loading ------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Direct name-for-name load.

        Names from a converted nanoswe checkpoint already match our parameter
        names (we deliberately mirrored the nanoswe module structure).  Any
        `_orig_mod.` prefix is stripped here defensively in case the converter
        was bypassed.
        """
        params = dict(self.named_parameters())
        # router_bias on every MoEMLP is a persistent buffer in the trained ckpt
        # but a non-parameter at our side; include it in the load map.
        buffers = {n: b for n, b in self.named_buffers() if n.endswith(".router_bias")}
        loaded: set[str] = set()
        for name, tensor in weights:
            name = name.removeprefix("_orig_mod.")
            if name in {"cos", "sin"}:
                continue  # buffers, recomputed
            target = params.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                # Tolerate the vLLM-internal Attention layer parameters not
                # being in the checkpoint (they're scale tensors etc.).
                continue
            with torch.no_grad():
                if target.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"target {tuple(target.shape)} vs ckpt {tuple(tensor.shape)}"
                    )
                target.copy_(tensor.to(dtype=target.dtype))
            loaded.add(name)
        return loaded


# -----------------------------------------------------------------------------
# vLLM registration. Importing this module registers NanoChatForCausalLM so that
# `vllm serve <export-dir>` dispatches on config.json architectures. Kept here
# (not in nanoswe/__init__.py) so that `import nanoswe.gpt` for TRAINING never
# pulls vLLM — only the export/serve/parity paths import this module.
def register() -> None:
    """Idempotently register NanoChatForCausalLM + the NanoChatConfig AutoConfig mapping."""
    try:
        from vllm import ModelRegistry
    except ImportError:
        return  # vLLM not installed; nothing to register
    if "NanoChatForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "NanoChatForCausalLM", "nanoswe.modeling_nanoswe:NanoChatForCausalLM"
        )
    try:
        from transformers import AutoConfig
        AutoConfig.register("nanoswe", NanoChatConfig)
    except Exception:
        pass  # already registered or transformers absent


register()
