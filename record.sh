#!/bin/bash
# =============================================================================
# nanoswe speedrun record — 192 B200-hour track
#
# The record run (`nanoswe-192h-260812`): 11,295 steps in 182.2/192 GPU-h
# (22.7 h wall on 8x B200), SWE-bench Verified pass@1 11.0%.
#   weights: https://huggingface.co/nanoswe/nanoswe-192h-260812
#   data:    https://huggingface.co/datasets/nanoswe/nanoswe-trajs-260812
#   log:     speedrun.log (this branch; ends with the exported weights' sha256)
#
# RECIPE: depth-42 (5.67B params), 32k context, SSSL sliding-window attention,
# fp8, doc-mask, RoPE theta 1e6, per-token loss throughout; ~23.7B tokens over
# a 3-phase data curriculum with crossfade transitions, all in ONE process over
# one continuous global step (see nanoswe/phases.py). LR is one continuous
# piecewise-linear envelope (warmup 40 steps -> 1.0, annealed down to 0.05):
#   pI   4,406 it  broad mix: smith-extra(unverified) / zero / smith-short /
#                  zero-extra / hero-extra          LR 1.0    -> 0.94139
#   xf1    694 it  data crossfade pI -> pII         LR        -> 0.85159
#   pII  1,633 it  concentrate: smith-extra(verified) / zero
#                                                   LR        -> 0.64029
#   xf2    640 it  data crossfade pII -> pIII       LR        -> 0.55748
#   pIII 3,922 it  finish: mini-coder / zero        LR        -> 0.05
# Mixture weights set token shares, drawn with credit-SWRR; origins are
# (origin, verified) slices of the consolidated Hub dataset above.
#
# --max-gpu-hours=192 is the competition cutoff (rules: the clock starts after
# the first step; no step once the budget is spent). This recipe's horizon
# finished at 182.2 GPU-h on the reference node, inside the budget.
#
# Prereqs (see README.md): the uv-synced env, a tokenizer at
# $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl, and the corpus — set
# NANOSWE_TRAJS_DIR to a local copy, or it is snapshot-downloaded from the Hub
# dataset above (NANOSWE_TRAJS_REPO).
# =============================================================================
set -euo pipefail

# Run from the repo root so `-m scripts.base_train` (and the nanoswe package) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- environment ------------------------------------------------------------
export NANOSWE_BASE_DIR="${NANOSWE_BASE_DIR:?set NANOSWE_BASE_DIR (holds tokenizer/ + checkpoints)}"
export OMP_NUM_THREADS=1
export NANOSWE_FUSED_LCE="${NANOSWE_FUSED_LCE:-1}"     # fused linear cross-entropy (token loss)
export WANDB_MODE="${WANDB_MODE:-disabled}"            # "online" + WANDB_API_KEY to log
export NANOSWE_TRAJS_REPO="${NANOSWE_TRAJS_REPO:-nanoswe/nanoswe-trajs-260812}"
NPROC="${NPROC:-8}"
MAX_GPU_HOURS="${MAX_GPU_HOURS:-192}"                  # competition budget (-1 = uncapped)
TAG="${MODEL_TAG:-nanoswe-192h-260812}"

# ---- sanity ----------------------------------------------------------------
[ -f "$NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl" ] || { echo "ERROR: tokenizer missing at $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl"; exit 1; }
if [ -n "${NANOSWE_TRAJS_DIR:-}" ]; then
  ls "$NANOSWE_TRAJS_DIR"/*.parquet >/dev/null 2>&1 || { echo "ERROR: NANOSWE_TRAJS_DIR=$NANOSWE_TRAJS_DIR has no parquet shards"; exit 1; }
else
  echo "NOTE: NANOSWE_TRAJS_DIR unset; will snapshot-download $NANOSWE_TRAJS_REPO from the Hub"
fi

# ---- recipe: 3 phases + 2 crossfades, one continuous global step ------------
# num_iterations sum = 11,295; TBS 2,097,152 tok => ~23.7B tokens.
PHASES="$(cat <<'JSON'
[
  {"name":"pI","num_iterations":4406,"loss_norm":"token","lr_schedule":"wsd","warmup_steps":40,"lr_start_frac":1.0,"final_lr_frac":0.94139,"warmdown_ratio":0.10281,
   "mixture":[{"origin":"swe-smith-extra","verified":false,"weight":505.2308,"seed":1001},
              {"origin":"swe-zero","weight":221.256,"seed":1002},
              {"origin":"swe-smith-extra-short","weight":113.686,"seed":1003},
              {"origin":"swe-zero-extra","weight":91.8392,"seed":1004},
              {"origin":"swe-hero-extra","weight":67.988,"seed":1005}]},
  {"name":"xf1","num_iterations":694,"loss_norm":"token","lr_schedule":"wsd","warmup_steps":0,"lr_start_frac":0.94139,"final_lr_frac":0.85159,"warmdown_ratio":1.0,
   "transition_from":"pI","transition_to":"pII"},
  {"name":"pII","num_iterations":1633,"loss_norm":"token","lr_schedule":"wsd","warmup_steps":0,"lr_start_frac":0.85159,"final_lr_frac":0.64029,"warmdown_ratio":1.0,
   "mixture":[{"origin":"swe-smith-extra","verified":true,"weight":748.3131,"seed":2001},
              {"origin":"swe-zero","weight":251.6869,"seed":2002}]},
  {"name":"xf2","num_iterations":640,"loss_norm":"token","lr_schedule":"wsd","warmup_steps":0,"lr_start_frac":0.64029,"final_lr_frac":0.55748,"warmdown_ratio":1.0,
   "transition_from":"pII","transition_to":"pIII"},
  {"name":"pIII","num_iterations":3922,"loss_norm":"token","lr_schedule":"wsd","warmup_steps":0,"lr_start_frac":0.55748,"final_lr_frac":0.05,"warmdown_ratio":1.0,
   "mixture":[{"origin":"ricdomolm/mini-coder-trajs-400k","weight":563.0509,"seed":3001},
              {"origin":"swe-zero","weight":436.9491,"seed":3002}]}
]
JSON
)"

echo "=== nanoswe 192h record  tag=$TAG  start $(date '+%F %T')  budget=${MAX_GPU_HOURS} GPU-h on ${NPROC} GPUs ==="
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth=42 \
    --target-param-data-ratio=8 \
    --total-batch-size=2097152 \
    --device-batch-size=1 \
    --max-seq-len=32768 \
    --window-pattern=SSSL \
    --fp8 \
    --use-doc-mask \
    --rope-theta=1000000 \
    --logit-softcap=15 \
    --max-gpu-hours="$MAX_GPU_HOURS" \
    --phases="$PHASES" \
    --eval-every=-1 \
    --no-save-optimizer \
    --model-tag="$TAG" \
    --run="$TAG"

echo "=== nanoswe 192h record  tag=$TAG  done $(date '+%F %T') ==="
echo "checkpoint: $NANOSWE_BASE_DIR/base_checkpoints/$TAG"

# vLLM export (default ON): package <tag>/pt as a vLLM model dir (<tag>/vllm) and
# log the safetensors sha256 to <tag>/speedrun.log. Runs in VLLM_VENV; best-effort.
if [ "${NANOSWE_VLLM_EXPORT:-1}" = "1" ]; then
  scripts/export_vllm.sh "$TAG" || true
fi
