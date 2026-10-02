#!/bin/bash
set -euo pipefail
RELEASE=$(cd "$(dirname "$0")" && pwd)
REPO=/lustre/home/rolmedo/nanoswe-final
FIX_COMMIT=9d952fe
cd "$REPO"
export NANOSWE_BASE_DIR=/fast/rolmedo/nanoswe HF_HOME=/fast/rolmedo/nanoswe/hf_cache HF_HUB_OFFLINE=1
export MSWEA_EVICT_SANDBOX=1 AUTO_CONCURRENCY=0 TOKEN_SCHEDULER=0
export VLLM_EXTRA_ARGS=''
unset NANOCHAT_NO_SMEAR_CACHE NANOCHAT_SMEAR_GUARD NANOCHAT_SMEAR_STATS
export AGENT_CFG="$RELEASE/agent.yaml"
export INSTANCE_IDS_FILE="$RELEASE/ids/s${SHARDS}_${SHARD}.json"
export EXPORT_DIR="$NANOSWE_BASE_DIR/base_checkpoints/$TAG/vllm"
mkdir -p "$OUT_DIR"
RUNTIME="${_CONDOR_SCRATCH_DIR:?}/compile_fix"
mkdir -p "$RUNTIME"
git archive "$FIX_COMMIT" eval/compile_fix/plugin | tar -x -C "$RUNTIME"
export PYTHONPATH="$RUNTIME/eval/compile_fix/plugin"
# Fail closed if this run does not use the exact GPU-tested serving source.
EXPECTED=4a467b7c351dc6b7dc4dde57b2c025eed389f6de7fc5f67bc1d3405408fcbd83
ACTUAL=$(sha256sum "$PYTHONPATH/nanochat_vllm/modeling_nanochat.py" | cut -d' ' -f1)
test "$ACTUAL" = "$EXPECTED"
test -s "$EXPORT_DIR/model.safetensors.sha256"
printf 'fix_commit=%s\nserving_sha256=%s\nrelease=%s\ncompile=default_on\n' "$FIX_COMMIT" "$ACTUAL" "$RELEASE" > "$OUT_DIR/serving_identity.txt"
bash "$RELEASE/run_v483.sh"
/home/rolmedo/mini-swe-agent-v2-eval/.venv/bin/python "$RELEASE/check_shard.py" "$OUT_DIR" "$INSTANCE_IDS_FILE"
