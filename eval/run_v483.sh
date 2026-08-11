#!/bin/bash
# In-slot v483 verification: serve a d24 export (vLLM) + run the VENDORED eval/
# agent+grader against verified_cluster, inline-graded, then aggregate.
# Serves via the vllm0201 env (its installed plugin registers the identical arch);
# rollouts run in the miniswa env with PYTHONPATH=eval/ shadowing any installed agent.
set -uo pipefail
export OMP_NUM_THREADS=4
EXPORT_DIR="${EXPORT_DIR:?}"; OUT_DIR="${OUT_DIR:?}"; K="${K:-1}"; WORKERS="${WORKERS:-60}"; PORT="${PORT:-8123}"
REPO_DIR=/home/rolmedo/nanoswe-final ; EVAL_DIR="$REPO_DIR/eval"
# Unique served-model name per run: a foreign server sharing our port would 404 our model
# name (and ours would 404 theirs) -> contamination becomes a loud failure, never silent.
SERVED_NAME="nanoswe-$(basename "$OUT_DIR")"
# PORT ISOLATION (root cause of the 06-24..07-02 eval anomalies): a server leaked by an
# evicted job keeps listening on the fixed port; vLLM binds SO_REUSEPORT so the new server
# shares it and the kernel splits connections between processes -> ~50% of completions
# silently answered by a DIFFERENT checkpoint. Always serve on a fresh free port unless
# PORT_STRICT=1 (debug). Old colocated pipeline was immune (random port per job).
if [ "${PORT_STRICT:-0}" != "1" ]; then
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
  echo "[v483] using fresh port $PORT"
fi
mkdir -p "$OUT_DIR"

# --- caches / offline grading ---
export HF_HOME=/fast/rolmedo/nanoswe/hf_cache HF_DATASETS_CACHE=/fast/rolmedo/nanoswe/hf_cache/datasets HF_HUB_CACHE=/fast/rolmedo/nanoswe/hf_cache/hub HF_HUB_OFFLINE=1
export NANOSWE_TEST_SPEC_CACHE=${NANOSWE_TEST_SPEC_CACHE:-/fast/rolmedo/nanoswe/test_spec_cache.json}
export NO_PROXY="*" no_proxy="*" HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy=""  # localhost endpoint must bypass the cluster proxy
export MSWEA_ALLOW_REGISTRY_PULL=0 MSWEA_ROBUST_SUBMIT=1
# Grade with the apptainer-overlay grader (robust under concurrency). Do NOT enable
# the kernel-overlay GRADE path: GRADE_KERNEL_OVERLAY=1 errors on ~75% of grades at
# eval concurrency (RuntimeError/OSError mount+fd exhaustion), counted as unresolved
# -> artifactual ~0% pass@1 (measured: v3_sssl's 3.73% patches read 0.5% under it).
# This is the GRADE path only; the kernel-overlay ROLLOUT path below is fine — keep it.
export GRADE_KERNEL_OVERLAY="${GRADE_KERNEL_OVERLAY:-0}"
[ -e /tmp/singularity_images ] || ln -sfn /lustre/fast/fast/rolmedo/swesmith/singularity_images /tmp/singularity_images

# --- route heavy scratch OFF /tmp: on some nodes /tmp is tmpfs (RAM -> counts against
#     request_memory, HoldReasonCode 34); under grading concurrency each grade builds a
#     ~2GB apptainer overlay.img in tempfile.gettempdir()/nanoswe_grade, so ~WORKERS*2GB
#     can fill /tmp -> 'overlay_failed' counted as unresolved (false negatives, observed
#     on lc/shard3 + freshlc/shard1). Use the Condor scratch dir: real LOCAL disk sized
#     to request_disk. tempfile.gettempdir() honors $TMPDIR, so this also relocates the
#     grader's per-grade scratch_root off /tmp. ---
SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}"
export TMPDIR="$SCRATCH/jobtmp"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache" SINGULARITY_CACHEDIR="$SCRATCH/apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"     SINGULARITY_TMPDIR="$SCRATCH/apptainer_tmp"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
# --- optional: force a FRESH per-job node-local vLLM compile cache (bypass the shared
#     Lustre ~/.cache/vllm -> /fast/rolmedo/caches/vllm). Tests whether a compile cache
#     baked on one H100 variant (96GB vs 80GB) is reused cross-variant and shifts numerics. ---
if [ -n "${FRESH_VLLM_CACHE:-}" ]; then export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"; mkdir -p "$VLLM_CACHE_ROOT"; echo "FRESH_VLLM_CACHE -> VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT" >> "$OUT_DIR/node.txt" 2>/dev/null; fi

# --- per-shard node / serving identity (so any future anomaly is shard-localizable) ---
{ echo "host=$(hostname)"; echo "date=$(date -Is)"; echo "scratch=$SCRATCH"; echo "cpus=$(nproc)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader 2>/dev/null
  # GPU telemetry for transient-regime correlation (clocks/power/temp/throttle/pstate + Xid via -q)
  echo "gpu_telemetry=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,pstate,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1)"
  echo "gpu_xid=$(nvidia-smi -q 2>/dev/null | grep -iE 'Xid|Pending Page|Retired|Remapped Rows|Uncorrectable' | head -6 | tr '\n' '|')"
  echo "nvidia_driver+cuda=$(nvidia-smi 2>/dev/null | grep -oE 'Driver Version: [0-9.]+ +CUDA Version: [0-9.]+' | head -1)"
  python -c "import torch;print(f'torch={torch.__version__} torch_cuda={torch.version.cuda}')" 2>/dev/null
} > "$OUT_DIR/node.txt" 2>&1

# --- preflight: assert the served weights match the recorded digest (catch silent drift) ---
if [ -f "$EXPORT_DIR/model.safetensors.sha256" ]; then
  want=$(awk '{print $1}' "$EXPORT_DIR/model.safetensors.sha256")
  got=$(sha256sum "$EXPORT_DIR/model.safetensors" | awk '{print $1}')
  { echo "weights_sha256_want=$want"; echo "weights_sha256_got=$got"; } >> "$OUT_DIR/node.txt"
  if [ "$want" != "$got" ]; then
    echo "[v483] FATAL: weights sha256 mismatch ($got != $want) — refusing to eval a drifted checkpoint" | tee -a "$OUT_DIR/node.txt"
    exit 1
  fi
fi
[ -s "${NANOSWE_TEST_SPEC_CACHE:-/nonexistent}" ] || echo "[v483] WARN: test-spec cache missing/empty: ${NANOSWE_TEST_SPEC_CACHE:-unset}" | tee -a "$OUT_DIR/node.txt"

# --- 1) serve (vllm0201 has the NanoChatForCausalLM plugin already) ---
# squatter visibility: leaked servers/agents from evicted jobs are how port contamination happened
{ echo "stray_vllm_procs=$(pgrep -u $(id -u) -f 'vllm serve' | wc -l)"
  echo "port_listeners_pre=$(ss -ltn 2>/dev/null | grep -c ":$PORT ")"; } >> "$OUT_DIR/node.txt"
setsid bash -c 'source /lustre/home/rolmedo/vllm0201/bin/activate
  export VLLM_DEEP_GEMM_WARMUP=skip VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  exec vllm serve "$0" --port "$1" --served-model-name "$2" \
      --max-model-len 34816 ${VLLM_EXTRA_ARGS:-}' "$EXPORT_DIR" "$PORT" "$SERVED_NAME" >"$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
# teardown must kill the WHOLE tree (APIServer + EngineCore re-parent and outlive a plain kill)
trap 'kill -- -$SERVE_PID 2>/dev/null; kill $SERVE_PID 2>/dev/null; fuser -k "$PORT/tcp" 2>/dev/null' EXIT
echo "[v483] waiting for vLLM :$PORT ..."
for _ in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && grep -q "Application startup complete" "$OUT_DIR/serve.log" 2>/dev/null && break
  kill -0 $SERVE_PID 2>/dev/null || { echo "serve died"; tail -30 "$OUT_DIR/serve.log"; exit 1; }; sleep 10; done
# identity check: /v1/models must report OUR checkpoint path (a foreign server on a shared
# port passes a bare curl but fails this — the exact contamination mode found 2026-07-02)
MID=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0].get('root',d['data'][0]['id']))" 2>/dev/null)
if [ "$MID" != "$EXPORT_DIR" ] && [ "$MID" != "$SERVED_NAME" ]; then
  echo "[v483] FATAL: endpoint identity mismatch: serving '$MID' != '$EXPORT_DIR' (foreign server on port $PORT?)" | tee -a "$OUT_DIR/node.txt"; exit 1
fi
echo "[v483] endpoint up (identity=$MID)."
# background GPU trace under load (30s cadence) — correlate high-windows w/ actual GPU state
( while true; do nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null; sleep 30; done ) > "$OUT_DIR/gpu_trace.csv" 2>&1 &
GPUTRACE_PID=$!; trap 'kill $SERVE_PID $GPUTRACE_PID 2>/dev/null' EXIT

# --- 2) render agent config + rollouts (vendored mini-swe-agent) with inline grading ---
CFG="$OUT_DIR/agent_config.yaml"; cp "$EVAL_DIR/configs/stripped_agent.yaml" "$CFG"
if [ "${MODEL_CLASS:-vllm}" = "litellm" ]; then
cat >> "$CFG" <<YAML
model:
  model_name: "hosted_vllm/$SERVED_NAME"
  model_class: "litellm"
  litellm_model_registry: "/home/rolmedo/mini-swe-agent/model_prices_and_context_window.json"
  model_kwargs: {api_base: "http://localhost:$PORT/v1", temperature: ${EVAL_TEMPERATURE:-0.7}, max_tokens: 2048}
YAML
else
cat >> "$CFG" <<YAML
model:
  model_name: "$SERVED_NAME"
  model_class: "vllm"
  model_kwargs: {api_base: "http://localhost:$PORT/v1", temperature: ${EVAL_TEMPERATURE:-0.7}, max_tokens: 2048}
YAML
fi
# KV-aware client admission control (parity with the old colocated pipeline, which
# runs TokenScheduler admit=0.92/pause=0.97/growth=2000 and measures ~4x tighter
# pass@1; the cleanup dropped this block while VLLMModel still supports it).
# TOKEN_SCHEDULER=1 enables; default off preserves faithful-replay behavior.
if [ "${TOKEN_SCHEDULER:-0}" = "1" ]; then
cat >> "$CFG" <<YAML
  token_scheduler:
    admit_threshold: ${ADMIT_THRESHOLD:-0.92}
    pause_threshold: ${PAUSE_THRESHOLD:-0.97}
    per_turn_growth: ${PER_TURN_GROWTH:-2000}
YAML
fi
sed -i "s/^  step_limit: .*/  step_limit: ${STEP_LIMIT:-100}/" "$CFG"
# Cluster fast path for the agent's bash exec: kernel-native overlayfs + chroot per
# command (no apptainer/FUSE), ~Nx faster at concurrency. Our nodes allow unprivileged
# userns. EVAL_ENV_CLASS=singularity falls back to the portable apptainer path.
sed -i "s/^  environment_class: .*/  environment_class: ${EVAL_ENV_CLASS:-singularity-kernel}/" "$CFG"
echo "agent_config_sha256=$(sha256sum "$CFG" | awk '{print $1}')" >> "$OUT_DIR/node.txt"
source /lustre/home/rolmedo/miniswa/bin/activate
export PYTHONPATH="$EVAL_DIR:${PYTHONPATH:-}"
NS=(); [ "$K" -ge 2 ] && NS=(--num-samples "$K")
SHARD=(); [ "${NINST:-0}" -gt 0 ] && SHARD=(--shuffle --slice "0:$NINST")
IID=();   [ -n "${INSTANCE_IDS_FILE:-}" ] && IID=(--instance-ids "@$INSTANCE_IDS_FILE")
# --- post-run contamination self-audit: client-side calls (traj api_calls) must match
# server-side 200s; >5% mismatch means another server answered part of the run.
_selfaudit() {
  python3 - "$OUT_DIR" <<'PYEOF'
import json, glob, re, sys, os
out=sys.argv[1]
calls=0
for f in set(glob.glob(out+"/*/*.traj.json")):
    try: calls+=int(json.load(open(f)).get("info",{}).get("model_stats",{}).get("api_calls",0))
    except: pass
srv=0
try:
    for line in open(out+"/serve.log",errors="replace"):
        if "/chat/completions" in line and " 200 " in line: srv+=1
except: pass
mismatch = (calls-srv)/calls if calls else 0.0
line=f"selfaudit_client_calls={calls} selfaudit_server_200s={srv} mismatch={mismatch:.3f}"
open(out+"/node.txt","a").write(line+"\n")
if calls and abs(mismatch)>0.05:
    open(out+"/CONTAMINATION_SUSPECT","w").write(line+"\n")
    print("[v483] WARNING: "+line+" -> CONTAMINATION_SUSPECT flagged")
else:
    print("[v483] selfaudit OK: "+line)
PYEOF
}
python -m minisweagent.run.extra.swebench --subset "${SUBSET:-verified_cluster}" --split test \
    --workers "$WORKERS" --config "$CFG" --output "$OUT_DIR" "${NS[@]}" "${SHARD[@]}" "${IID[@]}"
echo "[v483] rollouts done; aggregating"
python "$EVAL_DIR/aggregate_pass_at_k.py" --base "$(dirname "$OUT_DIR")" --tag "$(basename "$OUT_DIR")"
echo "[v483] DONE -> $OUT_DIR/pass_at_k.json"; cat "$OUT_DIR/pass_at_k.json" 2>/dev/null | head -20

_selfaudit
