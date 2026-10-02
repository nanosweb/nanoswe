#!/bin/bash
# =============================================================================
# nanoswe speedrun record — 192 B200-hour track
#
# The record run (`nanoswe-192h-261002`): 13,024 steps in 174.6/192 GPU-h
# (21.8 h wall on 8x B200), SWE-bench Verified pass@1 15.65% (483-instance
# subset, 5 samples per problem; previous record re-graded on the same
# serving: 11.76%).
#   weights: https://huggingface.co/nanoswe/nanoswe-192h-261002
#   data:    https://huggingface.co/datasets/nanoswe/nanoswe-trajs-261002
#   log:     speedrun.log (this branch; ends with the exported weights' sha256)
#
# WHAT CHANGED vs nanoswe-192h-260812: the training data. Same idea (train a
# nanoswe transformer from scratch on agent trajectories only, no web text),
# but on SWE-smith trajectories from Nemotron-3.5-Lightning in the tool-call
# format: the model emits one JSON tool call (bash / file_editor) per turn
# inside <|python_start|> ... <|python_end|> and reads the result as the next
# user turn. 1,383,822 trajectories over 1,133 repositories (none of them a
# SWE-bench source repository or a fork/mirror of one). Evaluated with the
# matching tool-call agent in eval/ (eval/README.md).
#
# RECIPE: depth-40 (3.23B scaling params), 32k context, SSSL sliding-window
# attention, fp8, doc-mask, RoPE theta 1e6, softcap 15, per-token loss; ONE
# phase of 13,024 steps x 1,835,008 tokens = 23.9B tokens (7.4 tokens/param),
# WSD schedule (warmup 40 steps, flat, linear warmdown over the last 65% to
# 0.05). The corpus is read in its stored instance-stratified order: row groups
# are shuffled with seed 3001 and dealt round-robin to the 8 ranks
# ("rg_interleave": 1 = drain one row group at a time; 23.9B tokens is ~one
# pass). Weight decay 0.28 follows the T_epoch rule at the ratio-8 reference
# (=> 0.017858 at this depth and batch), cosine-decayed to 0.
#
# --max-gpu-hours=192 is the competition cutoff (rules: the clock starts after
# the first step; no step once the budget is spent). This recipe's horizon
# finished at 174.6 GPU-h on the reference node, inside the budget. (The
# original run was launched without the cap, i.e. --max-gpu-hours=-1; the cap
# never binds for this horizon and does not change the numerics.)
#
# Prereqs (see README.md): the uv-synced env, a tokenizer at
# $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl, and the corpus — set
# NANOSWE_TRAJS_DIR to a local copy of the Hub dataset above (train shards at
# its root, val/ and test/ next to them), or it is snapshot-downloaded
# (NANOSWE_TRAJS_REPO). val/ and test/ are only scored at step 0 and at the end
# (outside the training clock).
# =============================================================================
set -euo pipefail

# Run from the repo root so `-m scripts.base_train` (and the nanoswe package) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- environment ------------------------------------------------------------
export NANOSWE_BASE_DIR="${NANOSWE_BASE_DIR:?set NANOSWE_BASE_DIR (holds tokenizer/ + checkpoints)}"
export OMP_NUM_THREADS=1
export NANOSWE_FUSED_LCE="${NANOSWE_FUSED_LCE:-1}"     # fused linear cross-entropy (token loss)
export WANDB_MODE="${WANDB_MODE:-disabled}"            # "online" + WANDB_API_KEY to log
export NANOSWE_TRAJS_REPO="${NANOSWE_TRAJS_REPO:-nanoswe/nanoswe-trajs-261002}"
NPROC="${NPROC:-8}"
MAX_GPU_HOURS="${MAX_GPU_HOURS:-192}"                  # competition budget (-1 = uncapped)
TAG="${MODEL_TAG:-nanoswe-192h-261002}"

# ---- sanity + corpus ----------------------------------------------------------
[ -f "$NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl" ] || { echo "ERROR: tokenizer missing at $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl"; exit 1; }
if [ -z "${NANOSWE_TRAJS_DIR:-}" ]; then
  echo "NOTE: NANOSWE_TRAJS_DIR unset; snapshot-downloading $NANOSWE_TRAJS_REPO from the Hub"
  NANOSWE_TRAJS_DIR="$(python -c "from huggingface_hub import snapshot_download as s; print(s('$NANOSWE_TRAJS_REPO', repo_type='dataset'))")"
fi
export NANOSWE_TRAJS_DIR
n_train=$(ls "$NANOSWE_TRAJS_DIR"/train-*.parquet 2>/dev/null | wc -l)
[ "$n_train" -eq 168 ] || { echo "ERROR: expected 168 train shards at the root of $NANOSWE_TRAJS_DIR, found $n_train"; exit 1; }
ls "$NANOSWE_TRAJS_DIR"/val/*.parquet "$NANOSWE_TRAJS_DIR"/test/*.parquet >/dev/null || { echo "ERROR: $NANOSWE_TRAJS_DIR needs val/ and test/"; exit 1; }

# ---- recipe: one phase ------------------------------------------------------
# 13,024 it x TBS 1,835,008 tok => 23,899,144,192 tokens. "dir" = the train
# shards (every shard, split "all"; the held-out val/test live in subdirs).
PHASES="$(python - "$NANOSWE_TRAJS_DIR" <<'PY'
import json, sys
print(json.dumps([
  {"name": "pIII", "num_iterations": 13024, "loss_norm": "token", "lr_schedule": "wsd", "warmup_steps": 40,
   "lr_start_frac": 1.0, "final_lr_frac": 0.05, "warmdown_ratio": 0.65,
   "mixture": [{"origin": None, "dir": sys.argv[1], "split": "all", "weight": 1000.0, "seed": 3001, "rg_interleave": 1}]}
]))
PY
)"

echo "=== nanoswe 192h record  tag=$TAG  start $(date '+%F %T')  budget=${MAX_GPU_HOURS} GPU-h on ${NPROC} GPUs ==="
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth=40 \
    --target-param-data-ratio=8 \
    --total-batch-size=1835008 \
    --device-batch-size=1 \
    --max-seq-len=32768 \
    --window-pattern=SSSL \
    --fp8 \
    --use-doc-mask \
    --rope-theta=1000000 \
    --logit-softcap=15 \
    --max-gpu-hours="$MAX_GPU_HOURS" \
    --phases="$PHASES" \
    --eval-every=13024 \
    --val-chat-dir="$NANOSWE_TRAJS_DIR/val,$NANOSWE_TRAJS_DIR/test" \
    --eval-chat-tokens=40108032,29884416 \
    --val-sequential-pack \
    --no-save-optimizer \
    ${CHECKPOINT_STAGE_DIR:+--checkpoint-stage-dir="$CHECKPOINT_STAGE_DIR"} \
    --model-tag="$TAG" \
    --run="$TAG"

echo "=== nanoswe 192h record  tag=$TAG  done $(date '+%F %T') ==="
echo "checkpoint: $NANOSWE_BASE_DIR/base_checkpoints/$TAG"

# vLLM export (default ON): package <tag>/pt as a vLLM model dir (<tag>/vllm) and
# log the safetensors sha256 to <tag>/speedrun.log. Runs in VLLM_VENV; best-effort.
if [ "${NANOSWE_VLLM_EXPORT:-1}" = "1" ]; then
  scripts/export_vllm.sh "$TAG" || true
fi
