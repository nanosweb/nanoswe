# eval — SWE-bench pass@1 for nanoswe

Self-contained harness to take a trained nanoswe checkpoint to a SWE-bench
Verified **pass@1** number: **export → serve → agent rollouts → grade → score**.

The agent is a **vendored** copy of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
v2 (MIT — see `minisweagent/LICENSE.md`) with nanoswe's additions (tool-call action parser,
inline grader, offline test specs, singularity sandboxes), byte-identical to the harness the
current record was evaluated with.

## What's here
```
serve.sh                 vllm serve <export>  (registers this repo's model via the plugin below)
run_eval.sh              portable single-node driver: serve → rollouts (inline-graded) → aggregate
vllm_nanoswe_plugin/     tiny vLLM plugin so `vllm serve` finds NanoChatForCausalLM = nanoswe/modeling_nanoswe.py
minisweagent/            vendored mini-swe-agent v2 (agent loop, vLLM client + tool-call parser, docker/singularity envs, swebench runner + inline grader)
configs/toolcall_agent.yaml   THE record protocol: tool-call agent (bash + file_editor), 120 s timeout, 100 steps
configs/bash_agent.yaml       the ```bash protocol, for models trained on bash trajectories (e.g. nanoswe-192h-260812)
ids/v483_ids.json, v091_ids.json   the 483 instances the records are scored on, and the old cheap subset91
cache_test_specs.py      one-time builder of the offline test_spec cache (no network at grade time)
aggregate_pass_at_k.py   run dir → pass_at_k.json (per_sample_resolved_rate = pass@1)
cluster/                 the exact HTCondor runner + shard lists the record evals were produced with (internal paths)
```

## The record protocol

| setting | value |
|---|---|
| instances | the 483 SWE-bench Verified instances in `ids/v483_ids.json` (17 of the 500 do not run in our sandboxes; both records are scored on these 483) |
| samples | **K=5 independent trajectories per instance**; pass@1 = resolved / (5 × 483) |
| sampling | temperature **0.7**, `max_tokens` **8,192** per assistant turn, no stop strings |
| agent | `configs/toolcall_agent.yaml`: no system message, the problem statement `{{task \| trim}}`, one JSON tool call per turn between `<\|python_start\|>` / `<\|python_end\|>` (tools `bash`, `file_editor`), `skip_special_tokens: false`, **120 s** command timeout, **100** steps, at most 3 consecutive format errors, `MSWEA_ROBUST_SUBMIT=1` |
| serving | vLLM 0.20.1, `--max-model-len 34816`, this repo's `nanoswe/modeling_nanoswe.py` (below), default compilation / CUDA graphs / prefix caching |
| grading | inline, apptainer overlay of the instance image, offline test specs (`NANOSWE_TEST_SPEC_CACHE`) |

The configs are the protocol: copy them verbatim. Do not add stop strings, change the
sampling, or edit the observation / format-error templates if you want comparable numbers —
`configs/toolcall_agent.yaml` reproduces, byte for byte, the observations the teacher saw when
the training trajectories were generated (checked on 550k observations).

### Serving matters: use this repo's model file
`nanoswe/modeling_nanoswe.py` contains two repairs that change pass@1 materially:
- **sliding windows** — the model trains with `window_pattern="SSSL"` (3 of every 4 layers attend
  to the last 8,192 tokens at 32k context). Earlier serving code built every layer as full
  attention, a train/serve mismatch on every turn past 8k tokens of context;
- **compiled decode** — the previous-token "smear" state was dropped from the compiled decode
  graph, and prefix-sharing requests could read each other's state.

`serve.sh` loads only this repo's plugin (`VLLM_PLUGINS=nanoswe`) and refuses to start if the
registered model lacks the window repair. Effect on the record models (same weights, same
harness, 483 × 5): `sv3_d40_r7p4` (nanoswe-192h-261002) 11.93% → 15.65%; the bash-format
`nanoswe-192h-260812` 11.01% → 11.76%.

## How to run an evaluation
Run from the repo root.

### 0. One-time setup
```bash
# (a) Export the checkpoint to a vLLM dir. Needs a GPU: the converter instantiates
#     the vLLM model to validate the weight mapping (see convert_to_vllm.py).
python -m scripts.convert_to_vllm \
    --ckpt-dir $NANOSWE_BASE_DIR/base_checkpoints/<run> --step <STEP> \
    --tokenizer $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl \
    --out /path/to/export/<run>

# (b) Register THIS repo's NanoChatForCausalLM with your vLLM 0.20.1 env (once per env):
pip install -e eval/vllm_nanoswe_plugin

# (c) Agent env: python>=3.10 with pyyaml requests jinja2 "pydantic>=2" litellm tenacity rich
#     python-dotenv typer platformdirs datasets swebench httpx (eval/minisweagent is put on
#     PYTHONPATH by run_eval.sh; nothing to install from it).

# (d) Build the offline grading cache for the eval set (one-time; needs network for
#     a few repos' requirements.txt, then grading is fully offline):
python eval/cache_test_specs.py --all \
    --dataset princeton-nlp/SWE-Bench_Verified \
    --out /path/to/test_spec_cache.json
export NANOSWE_TEST_SPEC_CACHE=/path/to/test_spec_cache.json
```

### 1. Single node — `run_eval.sh`
Serves, runs K rollouts per instance (inline-graded), aggregates → `<out>/pass_at_k.json`.
```bash
# the record protocol on the record's 483 instances, K=5, docker sandboxes (official images):
INSTANCE_IDS=eval/ids/v483_ids.json AGENT_VENV=/path/to/agent-venv VLLM_VENV=/path/to/vllm-venv \
  eval/run_eval.sh /path/to/export/<run>  /path/to/out  verified  5  12

cat /path/to/out/pass_at_k.json    # per_sample_resolved_rate = pass@1, plus pass@k + counts
```
Workers: the KV cache, not the GPU's compute, limits concurrency. On one H100 we use 12
concurrent agents for depth ≤ 32 and 8 for depth 40-42; more thrashes the prefix cache.

A bash-format model (trained on ```bash trajectories, e.g. `nanoswe-192h-260812`) is evaluated
with `AGENT_CFG=eval/configs/bash_agent.yaml EVAL_MAX_TOKENS=2048` (its 60 s timeout and
templates are in the config).

### 2. Our HTCondor cluster — `cluster/` (as run, not portable)
`cluster/run.sh` + `cluster/run_v483.sh` are the exact wrapper and runner the record evals used
(fresh random port + endpoint identity check, weights sha256 preflight, sandbox eviction,
singularity-kernel rollouts, per-shard completion audit `cluster/check_shard.py`);
`cluster/full483_toolcall.sub` is the Condor submit file (10 shards × 8 workers, 1 H100 each)
with the shard lists in `cluster/ids/`. They hardcode internal paths and extract the serving
code from the internal repo's commit `9d952fe` (the same code as `nanoswe/modeling_nanoswe.py`
here, sha256 `4a467b7c…` of the file then named `modeling_nanochat.py`). Kept for auditing.

A job leaving the queue is not proof of a complete shard: check `check_shard.py`'s
`completion_audit.json` (every instance × sample present, graded, no unsalvaged
infrastructure traceback, no `overlay_failed`) before aggregating.

## Grading runtime (apptainer-overlay; docker fallback)
`minisweagent/run/benchmarks/grading.py` uses `swebench` only for spec-build + report-parse;
the eval script runs in an **apptainer-overlay** container (or docker as fallback). Grading is
offline via the test_spec cache (`NANOSWE_TEST_SPEC_CACHE`).

> ⚠️ Do **not** set `GRADE_KERNEL_OVERLAY=1`. The kernel-overlay *grade* path errors
> (RuntimeError/OSError, mount+fd exhaustion) on ~75% of grades at eval concurrency,
> silently counting them unresolved → artifactual ~0% pass@1. It is unrelated to the
> kernel-overlay *rollout* path (`environment_class: singularity-kernel`), which is fine.

Known grading caveats (apply to every model equally): 4 instances need network at test time
(pylint-4661, sphinx-10435, sphinx-7985, matplotlib-20488) and cannot resolve offline. The
agent config's startup step commits the image's own uncommitted setup edits before the agent
starts, so submissions are agent-only diffs (without it every sphinx submission graded as
unresolved).
