#!/bin/bash
# Single-node SWE-bench pass@1 for a converted nanoswe checkpoint:
#   serve (vLLM) -> mini-swe-agent rollouts (inline-graded) -> aggregate pass@k.
#
# This is the portable driver for the record protocol (eval/README.md):
#   tool-call agent (configs/toolcall_agent.yaml: bash + file_editor tools, 120 s command
#   timeout, 100 steps), temperature 0.7, max 8,192 tokens per assistant turn, K=5 samples.
# On a cluster you typically wrap it in one GPU slot per shard (eval/cluster/ has the exact
# HTCondor runner the record evals were produced with).
#
# Usage:  eval/run_eval.sh <vllm_export_dir> <out_dir> [SUBSET] [K] [WORKERS]
#   SUBSET: dataset key — verified (canonical SWE-bench Verified, all 500, official
#           swebench/sweb.eval.* images; the default) | verified_cluster_483 (internal mirror).
#   Env:  INSTANCE_IDS=<ids.json>      restrict to those ids (e.g. ids/v483_ids.json = the record's 483)
#         AGENT_CFG=<yaml>             agent config (default configs/toolcall_agent.yaml; configs/bash_agent.yaml
#                                      is the ```bash protocol for models trained on bash trajectories, with
#                                      EVAL_MAX_TOKENS=2048)
#         EVAL_MAX_TOKENS (8192)  EVAL_TEMPERATURE (0.7)  STEP_LIMIT (100)
#         ENV_CLASS (docker)           docker | singularity (needs the config's image_sif_dir) | singularity-kernel
#         NANOSWE_TEST_SPEC_CACHE=<path>  offline grading cache (must cover the eval set)
#         PORT / AGENT_VENV / VLLM_VENV / REPO_DIR
set -euo pipefail
EXPORT_DIR="${1:?usage: run_eval.sh <vllm_export_dir> <out_dir> [SUBSET] [K] [WORKERS]}"
OUT_DIR="${2:?out_dir required}"
SUBSET="${3:-verified}"
K="${4:-5}"
WORKERS="${5:-12}"
PORT="${PORT:-8000}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_DIR="$REPO_DIR/eval"
AGENT_VENV="${AGENT_VENV:-/home/rolmedo/mini-swe-agent-v2-eval/.venv}"   # jinja2/datasets/swebench/httpx/litellm/pydantic/rich/typer...
AGENT_CFG="${AGENT_CFG:-$EVAL_DIR/configs/toolcall_agent.yaml}"
_abs() { case "$1" in /*|"") printf %s "$1" ;; *) printf %s "$PWD/$1" ;; esac; }   # resolve user paths before the cd below
EXPORT_DIR="$(_abs "$EXPORT_DIR")"; OUT_DIR="$(_abs "$OUT_DIR")"; AGENT_CFG="$(_abs "$AGENT_CFG")"
INSTANCE_IDS="$(_abs "${INSTANCE_IDS:-}")"; [ -n "${NANOSWE_TEST_SPEC_CACHE:-}" ] && export NANOSWE_TEST_SPEC_CACHE="$(_abs "$NANOSWE_TEST_SPEC_CACHE")"
cd "$REPO_DIR"   # cwd is on sys.path for `python -`/`-m`: never let another checkout's nanoswe/ or minisweagent/ shadow this one
mkdir -p "$OUT_DIR"

# 1) serve in the background (its own vLLM env), wait until the endpoint is live.
SERVED_NAME="nanoswe-$(basename "$OUT_DIR")"
REPO_DIR="$REPO_DIR" SERVED_NAME="$SERVED_NAME" "$EVAL_DIR/serve.sh" "$EXPORT_DIR" "$PORT" >"$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
echo "[run_eval] waiting for vLLM on :$PORT ..."
for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $SERVE_PID 2>/dev/null || { echo "ERROR: serve died — see $OUT_DIR/serve.log"; tail -20 "$OUT_DIR/serve.log"; exit 1; }
    sleep 10
done
echo "[run_eval] endpoint up."

# 2) agent config = the protocol file (copied verbatim) + a separate model block (endpoint + sampling),
#    merged by the runner (-c base -c model): v2 keeps the observation/format templates under `model:`,
#    so a second top-level `model:` key in the same file would clobber them.
CFG="$OUT_DIR/agent_config.yaml"; cp "$AGENT_CFG" "$CFG"
sed -i "s/^  step_limit: .*/  step_limit: ${STEP_LIMIT:-100}/" "$CFG"
sed -i "s/^  environment_class: .*/  environment_class: ${ENV_CLASS:-docker}/" "$CFG"   # as the cluster runner does
MCFG="$OUT_DIR/agent_config.model.yaml"
cat > "$MCFG" <<YAML
model:
  model_name: "$SERVED_NAME"
  model_class: "vllm"
  model_kwargs: {api_base: "http://localhost:$PORT/v1", temperature: ${EVAL_TEMPERATURE:-0.7}, max_tokens: ${EVAL_MAX_TOKENS:-8192}}
YAML

# 3) rollouts (mini-swe-agent v2, vendored in eval/minisweagent) with inline grading; uses the agent env.
source "$AGENT_VENV/bin/activate"
export PYTHONPATH="$EVAL_DIR:${PYTHONPATH:-}"   # shadow any installed mini-swe-agent with the vendored copy
export MSWEA_ROBUST_SUBMIT=1                     # part of the protocol: re-stage + re-diff when the submit captured nothing
export MSWEA_SILENT_STARTUP=1 MSWEA_ALLOW_REGISTRY_PULL="${MSWEA_ALLOW_REGISTRY_PULL:-1}"
export GRADE_KERNEL_OVERLAY=0                    # never the kernel-overlay grade path (see README)
export NO_PROXY="*" no_proxy="*"                 # the localhost endpoint must bypass any HTTP proxy
NSAMP=(); [ "$K" -ge 2 ] && NSAMP=(--num-samples "$K")
IIDS=(); [ -n "${INSTANCE_IDS:-}" ] && IIDS=(--instance-ids "@$INSTANCE_IDS")
python -m minisweagent.run.benchmarks.swebench \
    --subset "$SUBSET" --split test \
    --workers "$WORKERS" \
    --config "$CFG" --config "$MCFG" \
    --output "$OUT_DIR" \
    "${NSAMP[@]}" "${IIDS[@]}"

# 4) aggregate -> pass_at_k.json (per_sample_resolved_rate = pass@1).
python "$EVAL_DIR/aggregate_pass_at_k.py" --base "$(dirname "$OUT_DIR")" --tag "$(basename "$OUT_DIR")"
echo "[run_eval] done -> $OUT_DIR/pass_at_k.json"
