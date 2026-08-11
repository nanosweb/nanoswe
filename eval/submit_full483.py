#!/usr/bin/env python3
"""Submit a TRUE full-483 eval (all 483 instances, not the 446 subset) of the
VENDORED eval/ harness, sharded 5 ways, K=3, inline-graded.

Why this exists: SUBSET=verified_cluster maps to the Working-Harbor dataset and
the shared test_spec_cache only covers 446 -> the 37 extras silently never get
graded. This run uses SUBSET=verified_cluster_483 (Cluster483 dataset, all 483)
plus the extended NANOSWE_TEST_SPEC_CACHE (483 entries, built via
runs/swe_eval/cache_test_specs.py). 5 shards x ~97 instances x K=3 ~= 290
trajs/slot (under the proven ~335/slot of release_v446_steplimit250).
"""
import os, subprocess, textwrap
from pathlib import Path

EVAL = Path("/home/rolmedo/nanoswe-final/eval")
EXPORT = os.environ.get("EXPORT", "/fast/rolmedo/nanoswe/vllm_export/combined_d24_v3_sssl")
TAG = os.environ.get("TAG", "release_full483_v3sssl_k3")
BASE_OUT = "/fast/rolmedo/nanoswe/swe_eval"
CACHE_483 = "/fast/rolmedo/nanoswe/test_spec_cache_483.json"
K, WORKERS, STEP_LIMIT, PORT = int(os.environ.get("K", "3")), int(os.environ.get("WORKERS", "48")), 100, 8123
EVAL_TEMPERATURE = os.environ.get("EVAL_TEMPERATURE", "0.7")   # sampling temperature; 0 = greedy (use K=1). NOT 'TEMP' (collides with system $TEMP=/tmp)
VLLM_EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")        # extra `vllm serve` flags, e.g. --enforce-eager (no torch.compile)
VLLM_CACHE_ROOT = os.environ.get("VLLM_CACHE_ROOT", "")       # set to a unique empty path to force a FRESH torch.compile (no cache reuse)
N_SHARDS, BID = int(os.environ.get("N_SHARDS", "5")), "61"
SHARD_DIR = "v483_shards12" if N_SHARDS == 12 else "v483_shards"

for s in range(N_SHARDS):
    out_dir = f"{BASE_OUT}/{TAG}/shard{s}"
    shard_file = f"{EVAL}/ids/{SHARD_DIR}/shard_{s}.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = (
        f"HOME=/home/rolmedo EXPORT_DIR={EXPORT} OUT_DIR={out_dir} "
        f"K={K} WORKERS={WORKERS} STEP_LIMIT={STEP_LIMIT} PORT={PORT} EVAL_TEMPERATURE={EVAL_TEMPERATURE} "
        f"VLLM_EXTRA_ARGS='{VLLM_EXTRA_ARGS}' "
        f"TOKEN_SCHEDULER={os.environ.get('TOKEN_SCHEDULER','0')} "
        f"SUBSET=verified_cluster_483 INSTANCE_IDS_FILE={shard_file} "
        f"NANOSWE_TEST_SPEC_CACHE={CACHE_483} "
        f"APPTAINER_CACHEDIR=/tmp/apptainer_cache SINGULARITY_CACHEDIR=/tmp/apptainer_cache "
        f"APPTAINER_TMPDIR=/tmp/apptainer_tmp SINGULARITY_TMPDIR=/tmp/apptainer_tmp"
    )
    if VLLM_CACHE_ROOT:
        env += f" VLLM_CACHE_ROOT={VLLM_CACHE_ROOT}/shard{s}"
    sub = textwrap.dedent(f"""\
        universe       = vanilla
        executable     = {EVAL}/run_v483.sh
        environment    = "{env}"
        output         = {out_dir}/condor.out
        error          = {out_dir}/condor.err
        log            = {out_dir}/condor.log
        request_cpus   = 24
        request_gpus   = 1
        request_memory = 512GB
        request_disk   = 304GB
        requirements   = (TARGET.CUDACapability == 9.0) && (TARGET.UtsnameNodename != "i307") && (TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105") && (TARGET.UtsnameNodename != "g174") && (TARGET.UtsnameNodename != "g205") && (TARGET.UtsnameNodename != "g204") && (TARGET.UtsnameNodename != "i105") && (TARGET.UtsnameNodename != "i107")
        +BypassLXCfs   = true
        queue
    """)
    sub_path = f"{out_dir}/submit.sub"
    Path(sub_path).write_text(sub)
    print(f"[submit] shard{s}: {shard_file} -> {out_dir}")
    subprocess.run(["condor_submit_bid", BID, sub_path], check=True)
print(f"\nsubmitted {N_SHARDS} full-483 shards under {BASE_OUT}/{TAG}/")
