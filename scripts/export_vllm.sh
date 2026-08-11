#!/bin/bash
# =============================================================================
# Package a finished nanoswe checkpoint as a vLLM model directory + hash it.
#
#   base_checkpoints/<tag>/pt/model_<step>.pt        (written by base_train)
#     -> base_checkpoints/<tag>/vllm/{model.safetensors, config.json, tokenizer*}
#      + base_checkpoints/<tag>/vllm/model.safetensors.sha256
#      + a "vllm export ... sha256=<h>" line appended to <tag>/speedrun.log
#
# The conversion runs in a SEPARATE vLLM+transformers env (VLLM_VENV) because the
# training env is lean (no vllm/transformers/safetensors). The speedrun scripts
# call this AFTER torchrun exits, so the GPUs are free (vLLM init touches CUDA).
# Best-effort: a SKIPPED/FAILED export always exits 0 — the trained checkpoint in
# <tag>/pt is intact regardless.
#
# Usage:  scripts/export_vllm.sh <model_tag>
# Env:    NANOSWE_BASE_DIR (required); VLLM_VENV (default /lustre/home/rolmedo/vllm0201);
#         NANOSWE_VLLM_DTYPE (default bf16)
# =============================================================================
set -uo pipefail   # deliberately NOT -e: this step is best-effort

TAG="${1:?usage: export_vllm.sh <model_tag>}"
: "${NANOSWE_BASE_DIR:?set NANOSWE_BASE_DIR}"
VLLM_VENV="${VLLM_VENV:-/lustre/home/rolmedo/vllm0201}"
DTYPE="${NANOSWE_VLLM_DTYPE:-bf16}"

# Run from the repo root so `-m scripts.convert_to_vllm` (+ the nanoswe package) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="$NANOSWE_BASE_DIR/base_checkpoints/$TAG"
LOG="$CKPT/speedrun.log"
note() { local line="[$(date '+%F %T')] $*"; echo "$line"; [ -d "$CKPT" ] && printf '%s\n' "$line" >> "$LOG" 2>/dev/null || true; }

if [ ! -x "$VLLM_VENV/bin/python" ]; then
  note "vllm export SKIPPED (no python at \$VLLM_VENV=$VLLM_VENV)"; exit 0
fi
if ! ls "$CKPT/pt"/model_*.pt >/dev/null 2>&1; then
  note "vllm export SKIPPED (no $CKPT/pt/model_*.pt)"; exit 0
fi

STAGE="/tmp/${TAG}_vllm.$$"
rm -rf "$STAGE"
echo "=== vLLM export ($VLLM_VENV): $CKPT/pt -> $CKPT/vllm ==="
if ! "$VLLM_VENV/bin/python" -m scripts.convert_to_vllm \
       --ckpt-dir "$CKPT/pt" \
       --tokenizer "$NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl" \
       --out "$STAGE" --dtype "$DTYPE"; then
  note "vllm export FAILED (convert_to_vllm error); checkpoint at $CKPT/pt is intact"
  rm -rf "$STAGE"; exit 0
fi

# dd-move the export onto Lustre: the safetensors is multi-GB and plain cp to
# Lustre is ~20 MB/s; dd oflag=direct is ~940 MB/s. Small files (<1 MB) use cp.
mkdir -p "$CKPT/vllm"
for f in "$STAGE"/*; do
  bn="$(basename "$f")"
  if [ "$(stat -c%s "$f")" -gt 1048576 ]; then
    dd if="$f" of="$CKPT/vllm/$bn" bs=64M oflag=direct status=none
  else
    cp -f "$f" "$CKPT/vllm/$bn"
  fi
done
rm -rf "$STAGE"

SAFETENSORS="$CKPT/vllm/model.safetensors"
if [ ! -f "$SAFETENSORS" ]; then
  note "vllm export FAILED (no model.safetensors after move)"; exit 0
fi
HASH="$(sha256sum "$SAFETENSORS" | cut -d' ' -f1)"
echo "$HASH  model.safetensors" > "$CKPT/vllm/model.safetensors.sha256"
note "vllm export model.safetensors sha256=$HASH -> $CKPT/vllm"
exit 0
