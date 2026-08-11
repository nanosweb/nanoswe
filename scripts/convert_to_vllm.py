"""Package a nanoswe checkpoint as a vLLM-loadable model directory.

Output directory layout (compatible with `vllm serve <out>`):
    out/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json

Usage:
    # From a real checkpoint (real meta + tokenizer)
    python -m scripts.convert_to_vllm \\
        --ckpt-dir /fast/rolmedo/nanoswe/base_checkpoints/mini_coder_d24 \\
        --step 5568 \\
        --tokenizer /fast/rolmedo/nanoswe/tokenizer/tokenizer.pkl \\
        --out /fast/rolmedo/nanoswe/vllm_export/mini_coder_d24

    # From a random-init pt (for the equivalence pipeline test).  The model
    # config is inferred from the state-dict shapes; you must pass --depth /
    # --hidden / --vocab if they differ from the d24 defaults.
    python -m scripts.convert_to_vllm \\
        --random-pt /tmp/d24_random.pt \\
        --tokenizer /fast/rolmedo/nanoswe/tokenizer/tokenizer.pkl \\
        --out /tmp/nanoswe_vllm_random
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Avoid CUDA bf16 detection inside nanoswe.common when we just want to read meta.
os.environ.setdefault("NANOSWE_DTYPE", "float32")

from nanoswe.configuration_nanoswe import NanoChatConfig  # noqa: E402
from nanoswe.tokenizer_convert import convert as convert_tokenizer  # noqa: E402


_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _strip_compile_prefix(state: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in state.items()}


def _drop_buffers(state: dict) -> dict:
    """Drop non-persistent buffers (rotary cos/sin) that may have leaked in."""
    return {k: v for k, v in state.items() if not k.endswith(".cos") and not k.endswith(".sin")}


def _load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def _build_config_from_meta(meta: dict, *, max_position_embeddings: int | None = None) -> NanoChatConfig:
    mc = meta["model_config"]
    # Carry rope_theta + logit_softcap from the training config (meta is
    # asdict(GPTConfig)). Without rope_theta the export silently defaults to
    # NanoChatConfig's 100000 base, so a model trained at e.g. rope_theta=1e6 is
    # served with the wrong RoPE -> miscalibrated positions -> degraded outputs.
    # (auto_eval_pipeline.sh used to patch config.json post-export to work around
    # this; carrying it here makes the exported config correct on its own.)
    kw = {}
    if "rope_theta" in mc:
        kw["rope_theta"] = mc["rope_theta"]
    if "logit_softcap" in mc:
        kw["logit_softcap"] = mc["logit_softcap"]
    return NanoChatConfig(
        vocab_size=mc["vocab_size"],
        hidden_size=mc["n_embd"],
        num_hidden_layers=mc["n_layer"],
        num_attention_heads=mc["n_head"],
        num_key_value_heads=mc["n_kv_head"],
        max_position_embeddings=max_position_embeddings or mc["sequence_len"],
        window_pattern=mc.get("window_pattern", "L"),
        architectures=["NanoChatForCausalLM"],
        **kw,
    )


def _infer_config_from_state(state: dict) -> NanoChatConfig:
    """Best-effort config inference for random-pt mode (no meta.json)."""
    n_embd = state["transformer.wte.weight"].shape[1]
    padded_vocab = state["transformer.wte.weight"].shape[0]
    n_layer = max(
        int(k.split(".")[2]) for k in state if k.startswith("transformer.h.")
    ) + 1
    # Infer head dim from c_q (n_head * head_dim, n_embd)
    c_q = state[f"transformer.h.0.attn.c_q.weight"]
    n_kv = state[f"transformer.h.0.attn.c_k.weight"].shape[0]  # n_kv_head * head_dim
    # Standard: n_head = n_embd // head_dim; head_dim = c_q.shape[0] / n_head.
    # For nanoswe d24: n_head=12, head_dim=128, so c_q.shape[0] = 1536.
    # We can't disambiguate n_head from shape alone; assume head_dim = n_embd / n_head
    # by guessing n_head from kv (since n_kv_head * head_dim is known and head_dim
    # is usually n_embd / n_head).  Simplest: assume n_head divides n_embd evenly
    # and matches what the d24 run uses — 12.  Override via CLI if needed.
    n_head_guess = 12 if n_embd % 12 == 0 and c_q.shape[0] // (n_embd // 12) == 12 else None
    head_dim = c_q.shape[0] // (n_head_guess or 12)
    n_head = c_q.shape[0] // head_dim
    n_kv_head = n_kv // head_dim
    # padded_vocab is rounded up to multiple of 64; the "real" vocab size
    # cannot be recovered exactly, but for nanoswe d24 they are equal (32768).
    return NanoChatConfig(
        vocab_size=padded_vocab,
        padded_vocab_size=padded_vocab,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_kv_head,
        max_position_embeddings=32768,
        window_pattern="L",
        architectures=["NanoChatForCausalLM"],
    )


def _save_safetensors(state: dict, out_path: Path, dtype: torch.dtype) -> None:
    out_state: dict = {}
    for k, v in state.items():
        v = v.detach().cpu()
        # Cast floating-point tensors to the requested storage dtype; keep
        # int tensors as-is.
        if v.is_floating_point():
            v = v.to(dtype)
        out_state[k] = v.contiguous()
    save_file(out_state, str(out_path), metadata={"format": "pt"})


def _save_config(config: NanoChatConfig, out: Path) -> None:
    """Emit config.json with the architectures field for vLLM dispatch."""
    payload = config.to_dict()
    # PretrainedConfig.to_dict serializes our custom fields too.
    # The package renamed nanochat->nanoswe, but the eval stack (transformers 5.7 +
    # vLLM, e.g. the vllm0201 venv) only recognizes the upstream `nanochat` model_type;
    # the renamed `nanoswe` type isn't registered there, so `vllm serve` rejects the
    # config ("model type `nanoswe` ... not recognized"). The architecture class is
    # still NanoChatForCausalLM, so emit model_type=nanochat to load cleanly on eval.
    payload["model_type"] = "nanochat"
    (out / "config.json").write_text(json.dumps(payload, indent=2))


def _resolve_step(ckpt_dir: Path, step: int | None) -> int:
    if step is not None:
        return step
    candidates = sorted(ckpt_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no model_*.pt files found in {ckpt_dir}")
    return int(candidates[-1].name.removeprefix("model_").removesuffix(".pt"))


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ckpt-dir", type=Path,
        help="Directory containing model_<step>.pt + meta_<step>.json",
    )
    src.add_argument(
        "--random-pt", type=Path,
        help="A bare .pt state_dict (used to test the conversion pipeline)",
    )
    p.add_argument("--step", type=int, default=None,
                   help="(with --ckpt-dir) checkpoint step; default: latest")
    p.add_argument(
        "--tokenizer", type=Path,
        default=Path("/fast/rolmedo/nanoswe/tokenizer/tokenizer.pkl"),
        help="Path to tokenizer.pkl",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=_DTYPE_MAP.keys(), default="bf16",
                   help="Storage dtype for safetensors")
    p.add_argument("--max-len", type=int, default=None,
                   help="model_max_length to record in tokenizer_config.json "
                        "(default: config.max_position_embeddings)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.ckpt_dir is not None:
        step = _resolve_step(args.ckpt_dir, args.step)
        model_path = args.ckpt_dir / f"model_{step:06d}.pt"
        meta_path = args.ckpt_dir / f"meta_{step:06d}.json"
        print(f"loading {model_path}")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        meta = _load_meta(meta_path)
        config = _build_config_from_meta(meta)
    else:
        print(f"loading {args.random_pt}")
        state = torch.load(args.random_pt, map_location="cpu", weights_only=True)
        config = _infer_config_from_state(_strip_compile_prefix(state))

    state = _strip_compile_prefix(state)
    state = _drop_buffers(state)
    print(f"state has {len(state)} keys, {sum(v.numel() for v in state.values())/1e6:.1f}M params")

    # Validate that the state matches the inferred config by trying a build.
    # NanoChatForCausalLM is decorated with @support_torch_compile, which calls
    # get_current_vllm_config() in __init__ — must be inside a config context.
    from nanoswe.modeling_nanoswe import NanoChatForCausalLM
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        model = NanoChatForCausalLM(config=config)
    loaded = model.load_weights(state.items())
    expected = set(state)
    missing = expected - loaded
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(f"{len(missing)} keys not loaded by NanoChatForCausalLM, e.g. {sample}")

    target_dtype = _DTYPE_MAP[args.dtype]
    safetensors_path = args.out / "model.safetensors"
    print(f"writing {safetensors_path} ({args.dtype})")
    _save_safetensors(state, safetensors_path, target_dtype)

    print(f"writing {args.out / 'config.json'}")
    _save_config(config, args.out)

    max_len = args.max_len or config.max_position_embeddings
    print(f"converting tokenizer (max_len={max_len})")
    convert_tokenizer(args.tokenizer, args.out, model_max_length=max_len, validate=True)

    print()
    print(f"export complete: {args.out}")
    print(f"to serve: vllm serve {args.out} --max-model-len {min(max_len, 8192)}")


if __name__ == "__main__":
    main()
