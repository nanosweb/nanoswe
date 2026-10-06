#!/bin/bash
# =============================================================================
# nanoswe speedrun record — 192 B200-hour track
#
# The record run (`nanoswe-192h-261006`): 192.0/192 GPU-h on one 8x B200 node
# (2.2 h web pretraining + 21.8 h SFT, wall-clock after each launch's first
# step), SWE-bench Verified pass@1 25.47% (483-instance subset, 5 samples per
# problem; previous record nanoswe-192h-261002: 15.65%).
#   weights: https://huggingface.co/nanoswe/nanoswe-192h-261006
#   data:    https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle (web)
#            https://huggingface.co/datasets/nanoswe/nanoswe-trajs-261002     (SFT)
#   log:     speedrun.log (this branch: the web launch's log, then the SFT
#            launch's log; ends with the exported weights' sha256)
#
# WHAT CHANGED vs nanoswe-192h-261002: how the budget is spent. Same SFT corpus
# (SWE-smith tool-call trajectories from Nemotron-3.5-Lightning) and the same
# eval, but a smaller model (d32 instead of d40) that first reads a short
# FineWeb-Edu web-text phase and then trains on 37.8B trajectory tokens (1.55
# epochs) instead of 23.9B. Depth, web ratio and SFT length were chosen with
# scaling-law fits of held-out bits-per-byte.
#
# RECIPE: depth-32 (1.68B scaling params, 2.82B total), SSSL sliding-window
# attention, fp8, RoPE theta 1e6, softcap 15, per-token loss; two launches:
#  1. web: FineWeb-Edu at 4 tokens/param = 6,400 steps x 1,048,576 tokens
#     (6.71B; 2k context, ALL tokens supervised, nanochat concat-and-chop
#     loader). Batch and weight decay follow the horizon (--batch-horizon /
#     --wd-horizon=actual: B=1,048,576, LR x sqrt(2), WD 0.051973, cosine to 0).
#     WSD: warmup 40, flat, linear warmdown over the last 65% to 0.05.
#  2. SFT: init from the web weights (fresh optimizer, fresh step counter), 32k
#     context, doc-mask, assistant-span loss, B=1,310,720, LR x 0.790569,
#     WD 0.00998. TIME-DRIVEN horizon (--time-budget-gpu-hours): the WSD
#     warmdown, Muon momentum warmdown and cosine WD run on the spent fraction
#     of a step-time budget, and the loop stops before the step that would cross
#     it; num_iterations=32,307 is only an upper bound. That budget is 192 minus
#     the web launch's step time minus a 0.30 GPU-h reserve for the non-step
#     wall-clock (checkpoint saves, logging). The reference run stopped after
#     28,859 steps = 37,826,068,480 tokens. Corpus order: row groups shuffled
#     with seed 3001, dealt to the ranks in interleaved blocks
#     ("rg_interleave": 56).
#
# BUDGET (rules: the clock starts after the first step; no step once the budget
# is spent). Each launch runs its own clock from its first step; the SFT launch
# gets --max-gpu-hours = 192 - (the web launch's wall-clock GPU-h, read from its
# speedrun.log), so the two clocks sum to at most 192. Reference run, from
# speedrun.log: web 03:35:43 -> 05:48:55 = 2.2200 h, SFT 05:54:49 -> 03:41:37
# (+1 day) = 21.7800 h; 24.0000 h x 8 = 192.00 GPU-h (191.70 GPU-h of summed
# step time). Between the launches (not on either clock): the web checkpoint
# save, the SFT launch's compile + first step, and its step-0 val eval.
# (The original run was launched with --max-gpu-hours=-1; its time-driven
# horizon ended it inside the budget, and the cap does not change numerics.)
#
# Prereqs (see README.md): the uv-synced env, a tokenizer at
# $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl, the FineWeb-Edu shards (shards
# 0-93 are read, the last partly; the first FINEWEB_SHARDS are downloaded into
# $NANOSWE_BASE_DIR/base_data_fineweb if missing) and the trajectory corpus
# (NANOSWE_TRAJS_DIR = a local copy of the Hub dataset above, train shards at
# its root and the held-out val/ next to them; else snapshot-downloaded). val/
# is only scored at SFT step 0 and at the end (outside the training clock).
# The original run also scored an internal test set of teacher trajectories on
# SWE-bench Verified instances (the bpb_base_data_smith_v3_test_trainfmt lines
# in speedrun.log); it is not released and never trained on, and scoring it
# does not touch training.
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
MAX_GPU_HOURS="${MAX_GPU_HOURS:-192}"                  # competition budget (web + SFT)
SFT_RESERVE_GPU_HOURS=0.30                             # non-step wall-clock reserve (see BUDGET above)
FINEWEB_SHARDS="${FINEWEB_SHARDS:-100}"                # the web launch reads shards 0-93
TAG="${MODEL_TAG:-nanoswe-192h-261006}"
WEB_TAG="${WEB_TAG:-$TAG-web}"

# ---- sanity + data ----------------------------------------------------------
[ -f "$NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl" ] || { echo "ERROR: tokenizer missing at $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl"; exit 1; }
WEB_DONE="$NANOSWE_BASE_DIR/base_checkpoints/$WEB_TAG/pt/model_006400.pt"
if [ ! -f "$WEB_DONE" ]; then
  n_web=$(ls "$NANOSWE_BASE_DIR"/base_data_fineweb/shard_*.parquet 2>/dev/null | wc -l)
  if [ "$n_web" -lt $((FINEWEB_SHARDS + 1)) ]; then
    python -m nanoswe.dataset -d fineweb -n "$FINEWEB_SHARDS"    # + the val shard (shard_01822), as nanochat
  fi
fi
if [ -z "${NANOSWE_TRAJS_DIR:-}" ]; then
  echo "NOTE: NANOSWE_TRAJS_DIR unset; snapshot-downloading $NANOSWE_TRAJS_REPO from the Hub"
  NANOSWE_TRAJS_DIR="$(python -c "from huggingface_hub import snapshot_download as s; print(s('$NANOSWE_TRAJS_REPO', repo_type='dataset'))")"
fi
export NANOSWE_TRAJS_DIR
n_train=$(ls "$NANOSWE_TRAJS_DIR"/train-*.parquet 2>/dev/null | wc -l)
[ "$n_train" -eq 168 ] || { echo "ERROR: expected 168 train shards at the root of $NANOSWE_TRAJS_DIR, found $n_train"; exit 1; }
ls "$NANOSWE_TRAJS_DIR"/val/*.parquet >/dev/null || { echo "ERROR: $NANOSWE_TRAJS_DIR needs val/"; exit 1; }

echo "=== nanoswe 192h record  tag=$TAG  start $(date '+%F %T')  budget=${MAX_GPU_HOURS} GPU-h on ${NPROC} GPUs ==="

# ---- launch 1: web pretraining (FineWeb-Edu, 4 tokens/param) -------------------
# 6,400 it x 1,048,576 tok (batch auto-sized from the horizon) => 6,710,886,400 tokens.
# Skipped when the web checkpoint already exists (resume after an SFT failure).
if [ ! -f "$WEB_DONE" ]; then
  WEB_PHASES='[{"name": "fineweb", "target_param_data_ratio": 4.0, "loss_norm": "token", "lr_schedule": "wsd", "warmup_steps": 40, "lr_start_frac": 1.0, "final_lr_frac": 0.05, "warmdown_ratio": 0.65}]'
  torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
      --depth=32 \
      --target-param-data-ratio=8 \
      --total-batch-size=-1 \
      --device-batch-size=16 \
      --wd-horizon=actual \
      --batch-horizon=actual \
      --max-seq-len=2048 \
      --window-pattern=SSSL \
      --fp8 \
      --rope-theta=1000000 \
      --logit-softcap=15 \
      --flat-data=fineweb \
      --max-gpu-hours="$MAX_GPU_HOURS" \
      --phases="$WEB_PHASES" \
      --eval-every=-1 \
      --no-save-optimizer \
      ${CHECKPOINT_STAGE_DIR:+--checkpoint-stage-dir="$CHECKPOINT_STAGE_DIR/$WEB_TAG"} \
      --model-tag="$WEB_TAG" \
      --run="$WEB_TAG"
fi
[ -f "$WEB_DONE" ] || { echo "ERROR: web launch did not produce $WEB_DONE"; exit 1; }

# ---- budget left for the SFT launch -------------------------------------------
# From the web launch's speedrun.log: wall-clock after its first step (the rules'
# clock) and summed step time (what the time-driven horizon is charged with).
read -r WEB_WALL_GPU_H WEB_STEP_GPU_H < <(python - "$NANOSWE_BASE_DIR/base_checkpoints/$WEB_TAG/speedrun.log" "$NPROC" <<'PY'
import re, sys, datetime as dt
pat = re.compile(r"^\[(\S+ \S+)\] step (\d+) \|.*\| dt_ms ([\d.]+)")
rows = [(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)), float(m.group(3)))
        for m in map(pat.match, open(sys.argv[1])) if m]
assert rows and rows[0][1] == 0 and [r[1] for r in rows] == list(range(len(rows))), "web speedrun.log: expected one uninterrupted run"
n = int(sys.argv[2])
print(f"{(rows[-1][0] - rows[0][0]).total_seconds() * n / 3600:.6f} {sum(r[2] for r in rows[1:]) * n / 3.6e6:.6f}")
PY
)
[ -n "${WEB_STEP_GPU_H:-}" ] || { echo "ERROR: could not read the web launch's clock from its speedrun.log"; exit 1; }
SFT_CAP=$(python -c "print(f'{$MAX_GPU_HOURS - $WEB_WALL_GPU_H:.6f}')")
SFT_TIME_BUDGET=$(python -c "print(f'{$MAX_GPU_HOURS - $WEB_STEP_GPU_H - $SFT_RESERVE_GPU_HOURS:.3f}')")
echo "web launch: ${WEB_WALL_GPU_H} GPU-h wall-clock (${WEB_STEP_GPU_H} of step time) => SFT cap ${SFT_CAP} GPU-h, time-driven horizon ${SFT_TIME_BUDGET} GPU-h of step time"

# ---- launch 2: SFT on the trajectory corpus ----------------------------------
# "dir" = the train shards (every shard, split "all"; the held-out val/ lives in a subdir).
SFT_PHASES="$(python - "$NANOSWE_TRAJS_DIR" <<'PY'
import json, sys
print(json.dumps([
  {"name": "agent", "max_seq_len": 32768, "device_batch_size": 1, "total_batch_size": 1310720, "loss_norm": "token",
   "lr_schedule": "wsd", "lr_scale": 0.790569, "weight_decay": 0.00998,
   "mixture": [{"origin": None, "dir": sys.argv[1], "split": "all", "weight": 1000.0, "seed": 3001, "rg_interleave": 56}],
   "num_iterations": 32307, "lr_start_frac": 1.0, "final_lr_frac": 0.05, "warmup_steps": 40, "warmdown_ratio": 0.65,
   "reset_optimizer": True}
]))
PY
)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth=32 \
    --target-param-data-ratio=8 \
    --total-batch-size=1310720 \
    --device-batch-size=1 \
    --max-seq-len=32768 \
    --window-pattern=SSSL \
    --rope-theta=1000000 \
    --logit-softcap=15 \
    --wd-schedule=cosine \
    --eval-every=32307 \
    --eval-chat-tokens=40108032 \
    --eval-flat-tokens=2097152 \
    --max-gpu-hours="$SFT_CAP" \
    --model-tag="$TAG" \
    --run="$TAG" \
    --phases="$SFT_PHASES" \
    --fp8 \
    --use-doc-mask \
    --init-from-tag="$WEB_TAG" \
    --init-from-step=6400 \
    --time-budget-gpu-hours="$SFT_TIME_BUDGET" \
    --save-every=3000 \
    ${CHECKPOINT_STAGE_DIR:+--checkpoint-stage-dir="$CHECKPOINT_STAGE_DIR/$TAG"} \
    --val-chat-dir="$NANOSWE_TRAJS_DIR/val"

echo "=== nanoswe 192h record  tag=$TAG  done $(date '+%F %T') ==="
echo "checkpoint: $NANOSWE_BASE_DIR/base_checkpoints/$TAG"

# vLLM export (default ON): package <tag>/pt as a vLLM model dir (<tag>/vllm) and
# log the safetensors sha256 to <tag>/speedrun.log. Runs in VLLM_VENV; best-effort.
if [ "${NANOSWE_VLLM_EXPORT:-1}" = "1" ]; then
  scripts/export_vllm.sh "$TAG" || true
fi
