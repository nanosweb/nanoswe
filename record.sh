#!/bin/bash
# =============================================================================
# nanoswe speedrun record — 12 B200-hour track
#
# The record run (`nanoswe-12h-260812`): stopped by the budget at step 5,222
# of the 5,300-step horizon (12.001/12 GPU-h; 89.75 min training on 8x B200),
# SWE-bench Verified pass@1 5.0%.
#   weights: https://huggingface.co/nanoswe/nanoswe-12h-260812
#   data:    https://huggingface.co/datasets/nanoswe/nanoswe-trajs-260812
#   log:     speedrun.log (this branch; ends with the exported weights' sha256)
#
# RECIPE: depth-24 (1.38B params), 32k context, SSSL sliding-window attention,
# fp8, doc-mask, RoPE theta 1e6, per-token loss; single phase on a 2-source
# mixture (mini-coder + swe-zero, weights set token shares, drawn with
# credit-SWRR over the consolidated Hub dataset above). LR: WSD, warmup 40
# steps -> 1.0, warmdown over the last 65% of the horizon to 0.05.
#
# --max-gpu-hours=12 is the competition cutoff (rules: the clock starts after
# the first step; no step once the budget is spent). The 5,300-step horizon is
# sized just ABOVE what fits, so the budget is the binding stop and the WSD
# warmdown is ~fully annealed when it hits (~step 5,220 on the reference node).
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
MAX_GPU_HOURS="${MAX_GPU_HOURS:-12}"                   # competition budget (-1 = uncapped)
TAG="${MODEL_TAG:-nanoswe-12h-260812}"

# ---- sanity ----------------------------------------------------------------
[ -f "$NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl" ] || { echo "ERROR: tokenizer missing at $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl"; exit 1; }
if [ -n "${NANOSWE_TRAJS_DIR:-}" ]; then
  ls "$NANOSWE_TRAJS_DIR"/*.parquet >/dev/null 2>&1 || { echo "ERROR: NANOSWE_TRAJS_DIR=$NANOSWE_TRAJS_DIR has no parquet shards"; exit 1; }
else
  echo "NOTE: NANOSWE_TRAJS_DIR unset; will snapshot-download $NANOSWE_TRAJS_REPO from the Hub"
fi

# ---- recipe: one phase, 2-source mixture, WSD to 0.05 -----------------------
PHASES="$(cat <<'JSON'
[
  {"name":"pIIItok","num_iterations":5300,"loss_norm":"token","lr_schedule":"wsd","lr_start_frac":1.0,"final_lr_frac":0.05,"warmup_steps":40,"warmdown_ratio":0.65,
   "mixture":[{"origin":"ricdomolm/mini-coder-trajs-400k","weight":563.0509,"seed":3001},
              {"origin":"swe-zero","weight":436.9491,"seed":3002}]}
]
JSON
)"

echo "=== nanoswe 12h record  tag=$TAG  start $(date '+%F %T')  budget=${MAX_GPU_HOURS} GPU-h on ${NPROC} GPUs ==="
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --total-batch-size=1048576 \
    --device-batch-size=4 \
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

echo "=== nanoswe 12h record  tag=$TAG  done $(date '+%F %T') ==="
echo "checkpoint: $NANOSWE_BASE_DIR/base_checkpoints/$TAG"

# vLLM export (default ON): package <tag>/pt as a vLLM model dir (<tag>/vllm) and
# log the safetensors sha256 to <tag>/speedrun.log. Runs in VLLM_VENV; best-effort.
if [ "${NANOSWE_VLLM_EXPORT:-1}" = "1" ]; then
  scripts/export_vllm.sh "$TAG" || true
fi
