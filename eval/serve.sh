#!/bin/bash
# Serve a converted nanoswe checkpoint (scripts/convert_to_vllm.py output) with vLLM,
# exactly as the record evals were served:
#   * the repo's vLLM model, nanoswe/modeling_nanoswe.py (sliding windows + compiled-decode
#     smear-state repair), registered through eval/vllm_nanoswe_plugin;
#   * --max-model-len 34816 (32k context + one 2k turn; vLLM rejects a request whose prompt
#     exceeds 34816 - max_tokens, which the agent turns into its context-limit submit);
#   * default vLLM compilation, CUDA graphs and prefix caching.
#
# Usage:  eval/serve.sh <vllm_export_dir> [PORT]
# Env:    VLLM_VENV   a vLLM 0.20.1 env with `pip install -e eval/vllm_nanoswe_plugin` (default: internal path)
#         SERVED_NAME OpenAI model name (default nanoswe)
set -euo pipefail

EXPORT_DIR="${1:?usage: serve.sh <vllm_export_dir> [PORT]}"
PORT="${2:-8000}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_VENV="${VLLM_VENV:-/lustre/home/rolmedo/vllm0201}"

[ -f "$EXPORT_DIR/config.json" ] || { echo "ERROR: $EXPORT_DIR/config.json missing (run scripts/convert_to_vllm.py first)"; exit 1; }
source "$VLLM_VENV/bin/activate"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export VLLM_DEEP_GEMM_WARMUP=skip VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Load ONLY this repo's plugin: another nanochat/nanoswe plugin installed in the same env
# (e.g. an older nanoswe-vllm without the sliding-window fix) would otherwise register
# NanoChatForCausalLM first and silently serve different code.
export VLLM_PLUGINS="${VLLM_PLUGINS:-nanoswe}"

python - <<'PY' || { echo "ERROR: vLLM cannot see this repo's NanoChatForCausalLM — install the plugin: pip install -e eval/vllm_nanoswe_plugin"; exit 1; }
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm import ModelRegistry
assert "NanoChatForCausalLM" in ModelRegistry.get_supported_archs(), "plugin not registered"
import nanoswe.modeling_nanoswe as m
assert hasattr(m, "compute_window_sizes"), "nanoswe.modeling_nanoswe lacks the sliding-window fix"
print(f"[serve] NanoChatForCausalLM registered with vLLM ({m.__file__})")
PY

echo "=== serving $EXPORT_DIR on :$PORT (vLLM venv: $VLLM_VENV) ==="
exec vllm serve "$EXPORT_DIR" \
    --port "$PORT" \
    --served-model-name "${SERVED_NAME:-nanoswe}" \
    --max-model-len 34816
