"""Logit-equivalence test: nanoswe.gpt.GPT vs nanoswe.NanoChatForCausalLM.

Loads the same state_dict into both models, runs an identical input, asserts
their logits match within a tolerance.

Usage:
    python -m scripts.test_vllm_equivalence \
        [--depth 4] [--batch 2] [--seqlen 16] [--seed 42] \
        [--device cuda] [--dtype bf16] [--tol 5e-2]

By default uses a small (4-layer, 384-dim) random-init config so the test
runs fast on CPU.  Use `--depth 24 --device cuda --dtype bf16` for a
production-shape test on the dev A100.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo importable when run as `python -m scripts.test_vllm_equivalence`
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# nanoswe.common.COMPUTE_DTYPE is auto-detected at import time and the
# `assert self.cos.dtype == COMPUTE_DTYPE` in gpt.py forbids mixing.  Honor
# `--dtype` by setting NANOSWE_DTYPE *before* importing nanoswe.
_PRE_DTYPE = next(
    (a.split("=", 1)[1] if "=" in a else sys.argv[i + 1]
     for i, a in enumerate(sys.argv) if a == "--dtype" or a.startswith("--dtype=")),
    "fp32",
)
_NANO_DTYPE_NAME = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}[_PRE_DTYPE]
os.environ["NANOSWE_DTYPE"] = _NANO_DTYPE_NAME

import torch  # noqa: E402

from nanoswe.gpt import GPT, GPTConfig  # noqa: E402
from nanoswe.configuration_nanoswe import NanoChatConfig  # noqa: E402
from nanoswe.modeling_nanoswe import NanoChatForCausalLM  # noqa: E402


_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _build_models(depth: int, n_embd: int, n_head: int, seqlen: int, vocab_size: int, seed: int):
    torch.manual_seed(seed)
    gpt_cfg = GPTConfig(
        sequence_len=seqlen,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="L",
    )
    # nanoswe builds on meta then init_weights; do the same so the random
    # init is deterministic given the seed.
    with torch.device("meta"):
        gpt = GPT(gpt_cfg)
    gpt.to_empty(device="cpu")
    torch.manual_seed(seed)
    gpt.init_weights()

    nano_cfg = NanoChatConfig.from_gpt_config(gpt_cfg)
    ours = NanoChatForCausalLM(config=nano_cfg)

    state = gpt.state_dict()
    # Strip non-persistent buffers if they sneak in
    state = {k: v for k, v in state.items() if not k.endswith(".cos") and not k.endswith(".sin")}
    loaded = ours.load_weights(state.items())
    expected = {k for k in state if not k.endswith("cos") and not k.endswith("sin")}
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"Failed to load: {sorted(missing)[:10]} (total {len(missing)})")
    return gpt, ours, gpt_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="fp32", choices=_DTYPE_MAP.keys())
    p.add_argument("--tol", type=float, default=None,
                   help="Max-abs tolerance.  Default: 1e-4 fp32, 5e-2 bf16, 5e-2 fp16.")
    args = p.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    device = torch.device(args.device)
    if args.tol is None:
        args.tol = 1e-4 if dtype == torch.float32 else 5e-2

    gpt, ours, gpt_cfg = _build_models(
        depth=args.depth, n_embd=args.n_embd, n_head=args.n_head,
        seqlen=args.seqlen, vocab_size=args.vocab_size, seed=args.seed,
    )

    # Match the embedding/RoPE dtype on both sides — nanoswe's
    # `init_weights` casts wte and value_embeds to COMPUTE_DTYPE; on CPU
    # COMPUTE_DTYPE is fp32, on cuda Ampere+ it is bf16.  Force consistency
    # by moving everything to `dtype` on `device`.
    gpt = gpt.to(device=device, dtype=dtype).eval()
    ours = ours.to(device=device, dtype=dtype).eval()

    # nanoswe.gpt expects `self.cos` to match COMPUTE_DTYPE — reload the
    # rotary buffers in the requested dtype on the right device.
    gpt.cos, gpt.sin = gpt._precompute_rotary_embeddings(
        gpt.rotary_seq_len, gpt_cfg.n_embd // gpt_cfg.n_head, device=device,
    )
    gpt.cos = gpt.cos.to(dtype)
    gpt.sin = gpt.sin.to(dtype)

    torch.manual_seed(args.seed + 1)
    ids = torch.randint(0, args.vocab_size, (args.batch, args.seqlen), device=device)

    with torch.no_grad():
        a = gpt(ids)              # (B, T, vocab_size), fp32 (post-softcap)
        b = ours.forward_standalone(ids)
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = (diff.max() / a.abs().max().clamp(min=1e-6)).item()

    a_top = a.argmax(dim=-1)
    b_top = b.argmax(dim=-1)
    top1_match = (a_top == b_top).float().mean().item()

    print(f"depth={args.depth}, dim={args.n_embd}, dtype={args.dtype}, device={args.device}")
    print(f"max|Δ| = {max_diff:.3e}, mean|Δ| = {mean_diff:.3e}, rel = {rel:.3e}")
    print(f"top-1 argmax match = {top1_match:.4f}")
    print(f"tolerance = {args.tol:.1e}")
    if max_diff > args.tol:
        print("FAIL: logits diverge above tolerance.")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
