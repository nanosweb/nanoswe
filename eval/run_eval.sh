#!/bin/bash
# Single-node SWE-bench pass@1 for a converted nanoswe checkpoint:
#   serve (vLLM) -> mini-swe-agent rollouts (inline-graded) -> aggregate pass@k.
#
# This is the portable driver. On a cluster you typically wrap it in one GPU
# slot (see eval/submit_full483.py). Needs: a GPU, the vLLM env (serve.sh), the agent
# env (jinja2/datasets/swebench/httpx/rich/tenacity/typer/pyyaml), a container
# runtime (apptainer or docker) + per-instance images, and the SWE-bench dataset.
#
# Usage:  eval/run_eval.sh <vllm_export_dir> <out_dir> [SUBSET] [K] [WORKERS]
#   SUBSET: dataset key — verified (canonical SWE-bench Verified, all 500; the default)
#           | verified_cluster (446) / verified_cluster_483 (internal-cluster mirrors).
#   Env:  INSTANCE_IDS=<ids.json>  restrict to those ids (e.g. ids/v091_ids.json = subset91)
#         NANOSWE_TEST_SPEC_CACHE=<path>  offline grading cache (must cover the eval set)
#         PORT / AGENT_VENV / REPO_DIR
set -euo pipefail
EXPORT_DIR="${1:?usage: run_eval.sh <vllm_export_dir> <out_dir> [SUBSET] [K] [WORKERS]}"
OUT_DIR="${2:?out_dir required}"
SUBSET="${3:-verified}"
K="${4:-3}"
WORKERS="${5:-48}"
PORT="${PORT:-8000}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_DIR="$REPO_DIR/eval"
AGENT_VENV="${AGENT_VENV:-/lustre/home/rolmedo/miniswa}"     # has jinja2/datasets/swebench/httpx/rich...
mkdir -p "$OUT_DIR"

# 1) serve in the background (its own vLLM env), wait until the endpoint is live.
REPO_DIR="$REPO_DIR" "$EVAL_DIR/serve.sh" "$EXPORT_DIR" "$PORT" >"$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
echo "[run_eval] waiting for vLLM on :$PORT ..."
for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $SERVE_PID 2>/dev/null || { echo "ERROR: serve died — see $OUT_DIR/serve.log"; tail -20 "$OUT_DIR/serve.log"; exit 1; }
    sleep 10
done
echo "[run_eval] endpoint up."

# 2) render the agent config: base stripped config + model section (api_base).
DEST_CFG="$OUT_DIR/agent_config.yaml"
cp "$EVAL_DIR/configs/stripped_agent.yaml" "$DEST_CFG"
cat >> "$DEST_CFG" <<YAML
model:
  model_name: "nanoswe"
  model_class: "vllm"
  model_kwargs:
    api_base: "http://localhost:$PORT/v1"
    temperature: 0.7
    max_tokens: 2048
YAML

# 3) rollouts (mini-swe-agent, vendored) with inline grading; uses the agent env.
source "$AGENT_VENV/bin/activate"
export PYTHONPATH="$EVAL_DIR:${PYTHONPATH:-}"   # shadow any installed mini-swe-agent with the vendored copy
NSAMP=(); [ "$K" -ge 2 ] && NSAMP=(--num-samples "$K")
IIDS=(); [ -n "${INSTANCE_IDS:-}" ] && IIDS=(--instance-ids "@$INSTANCE_IDS")
python -m minisweagent.run.extra.swebench \
    --subset "$SUBSET" --split test \
    --workers "$WORKERS" \
    --config "$DEST_CFG" \
    --output "$OUT_DIR" \
    "${NSAMP[@]}" "${IIDS[@]}"

# 4) aggregate -> pass_at_k.json (per-sample resolved rate = pass@1 estimate).
python "$EVAL_DIR/aggregate_pass_at_k.py" --base "$(dirname "$OUT_DIR")" --tag "$(basename "$OUT_DIR")"
echo "[run_eval] done -> $OUT_DIR/pass_at_k.json"
