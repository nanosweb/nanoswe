# eval — SWE-bench pass@1 for nanoswe

Self-contained harness to take a trained nanoswe checkpoint to a SWE-bench
Verified **pass@1** number: **export → serve → agent rollouts → grade → score**.

The agent is a **vendored** subset of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
(MIT — see `minisweagent/LICENSE.md`), so the agent loop + grader live in-repo.
It's vendored as *an* agent, not wired in as the only one — `run_eval.sh` calls it
behind `mini-extra swebench`; a different agent is a drop-in replacement.

## What's here
```
serve.sh                 vllm serve <export>  (OpenAI endpoint; registers the arch via the plugin below)
run_eval.sh              portable single-node driver: serve → rollouts (inline-graded) → aggregate
run_v483.sh              cluster (HTCondor) runner: serve + sharded rollouts + inline grade + aggregate
submit_full483.py        HTCondor submitter — all 483, K=3, sharded (drives run_v483.sh)
submit_subset91_parity.py  HTCondor submitter — subset91, K=10, sharded
vllm_nanoswe_plugin/     tiny vLLM plugin so `vllm serve` finds NanoChatForCausalLM without pip-installing the repo
minisweagent/            vendored mini-swe-agent subset (agent loop, vLLM client, docker+singularity envs, swebench runner + inline grader)
configs/stripped_agent.yaml   the prompt/tool config that defines pass@1 behavior
ids/v091_ids.json, v483_ids.json   the canonical instance sets (subset91 + v483); ids/*_shards/ pre-split for the submitters
cache_test_specs.py      one-time builder of the offline test_spec cache (no network at grade time)
aggregate_pass_at_k.py   run dir → pass_at_k.json (per-sample resolved rate = pass@1)
```

> **Internal-cluster runners, kept for reproducibility.** `run_v483.sh`,
> `submit_full483.py`, and `submit_subset91_parity.py` are specific to the internal
> HTCondor cluster the record evals were produced on (hardcoded repo/venv/cache paths,
> node blacklists, condor knobs). They are included so the record numbers are
> reproducible/auditable as-run, not as a portable entry point — outside this cluster,
> use `run_eval.sh` (with `serve.sh`), which is the same serve → rollout → grade →
> aggregate pipeline on a single node. The portable scripts default `VLLM_VENV` /
> `AGENT_VENV` to internal paths too — override them for your environment.

## External dependencies (NOT vendored)
- **vLLM** (for serving) + the model: `nanoswe/modeling_nanoswe.py` is the vLLM-side
  model and **must mirror `nanoswe/gpt.py`** — run `scripts/test_vllm_equivalence.py`
  after any architecture change. Register it for serving once per vLLM env:
  `pip install -e eval/vllm_nanoswe_plugin` (keeps the main repo non-package).
- **`swebench`** harness + **`datasets`** + the agent deps (jinja2, httpx, rich, tenacity,
  typer, pyyaml, python-dotenv, platformdirs). The model client is a lean in-repo
  httpx wrapper (`models/vllm_model.py`) — **no litellm**.
- A **container runtime** — apptainer/singularity **or** docker — plus the per-instance
  images, and a **SWE-bench Verified dataset**, selected by `SUBSET`. The canonical,
  public choice is `verified` (`princeton-nlp/SWE-Bench_Verified`, all 500 instances,
  official `swebench/sweb.eval.*` images from Docker Hub). The `verified_cluster*`
  subsets (`ricdomolm/SWE-bench_Verified-Cluster483` etc.) are **internal-cluster
  mirrors** — their image names point at our org's registry and won't resolve outside.

## Grading runtime (apptainer-overlay; docker fallback)
`minisweagent/run/extra/grading.py` uses `swebench` only for spec-build + report-parse;
the eval script runs in an **apptainer-overlay** container (or docker as fallback).
Grading is offline via the test_spec cache (`NANOSWE_TEST_SPEC_CACHE`).

> ⚠️ Do **not** set `GRADE_KERNEL_OVERLAY=1`. The kernel-overlay *grade* path errors
> (RuntimeError/OSError, mount+fd exhaustion) on ~75% of grades at eval concurrency,
> silently counting them unresolved → artifactual ~0% pass@1 (measured: patches that
> grade to 3.73% on apptainer read 0.5% under it). It is unrelated to the kernel-overlay
> *rollout* path (`environment_class: singularity-kernel`), which is fine and fast.

## How to run an evaluation

End-to-end is **export → serve → agent rollouts → grade → aggregate**. Run from the
repo root. `pass@1` = per-sample resolved rate (`per_sample_resolved_rate` in the output).

### Rulers / protocols
| ruler | `SUBSET` (dataset) | ids restriction | K |
|---|---|---|---|
| **verified** — record submissions | `verified` (canonical, all 500) | — | **5** |
| **v483** — internal headline eval | `verified_cluster_483` (483 mirror) | — | 3 |
| **subset91** — cheap iteration | `verified_cluster_483` | `INSTANCE_IDS=eval/ids/v091_ids.json` | 10 |

> **Record submissions** use the canonical `verified` set with **K=5** (5 independent
> samples per problem, per the [rules](https://www.nanoswe.com/rules.html)). The
> `verified_cluster*` rulers only work on our internal cluster (see above). Sampling is
> temperature 0.7 / max_tokens 2048, pinned in the runners — don't change it if you want
> comparable numbers.

### 0. One-time setup
```bash
# (a) Export the checkpoint to a vLLM dir. Needs a GPU: the converter instantiates
#     the vLLM model to validate the weight mapping (see convert_to_vllm.py).
python -m scripts.convert_to_vllm \
    --ckpt-dir $NANOSWE_BASE_DIR/base_checkpoints/<run> --step <STEP> \
    --tokenizer $NANOSWE_BASE_DIR/tokenizer/tokenizer.pkl \
    --out /path/to/export/<run>

# (b) Register THIS repo's NanoChatForCausalLM with your vLLM env (once per env):
pip install -e eval/vllm_nanoswe_plugin

# (c) Build the offline grading cache for the eval set (one-time; needs network for
#     a few repos' requirements.txt, then grading is fully offline):
python eval/cache_test_specs.py --all \
    --dataset princeton-nlp/SWE-Bench_Verified \
    --out /path/to/test_spec_cache.json
export NANOSWE_TEST_SPEC_CACHE=/path/to/test_spec_cache.json
```

### 1. Single node — `run_eval.sh` (portable driver)
Serves, runs K rollouts/instance (inline-graded), aggregates → `<out>/pass_at_k.json`.
```bash
# record submission — canonical SWE-bench Verified, all 500, K=5:
eval/run_eval.sh /path/to/export/<run>  /path/to/out  verified  5  48

# subset91 — cheap iteration, K=10:
INSTANCE_IDS=eval/ids/v091_ids.json \
  eval/run_eval.sh /path/to/export/<run>  /path/to/out  verified  10  48

cat /path/to/out/pass_at_k.json    # per_sample_resolved_rate = pass@1, plus pass@k + counts
```

### 2. Our HTCondor cluster — sharded submitters
`run_v483.sh` (the condor executable) shards the set across GPU slots, inline-grades,
and aggregates per shard. The submitters set EXPORT/TAG/K/shard-count at the top:
```bash
python eval/submit_full483.py          # all 483, K=3, 5 shards
python eval/submit_subset91_parity.py  # subset91, K=10, 3 shards
```
Pool the per-shard `pass_at_k.json`s (or re-aggregate over the inline grades) for the
final number.

### Which code actually runs
The agent loop + grader + aggregator are always **this repo's** vendored code
(`PYTHONPATH=eval/` shadows any installed mini-swe-agent). For the **model** to be this
repo's `nanoswe/modeling_nanoswe.py`, the serve env must have `eval/vllm_nanoswe_plugin`
installed (step 0b) — `serve.sh`/`run_eval.sh` ensure this via `PYTHONPATH=$REPO_DIR`.
(The cluster `run_v483.sh` reuses a prebuilt, behaviorally-identical plugin in the
`vllm0201` venv instead.)
