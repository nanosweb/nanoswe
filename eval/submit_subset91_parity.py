#!/usr/bin/env python3
"""Submit the VENDORED eval/ harness on subset91 at K=10, sharded 3 ways.

Head-to-head partner of the ORIGINAL colocated harness (launch_swe_colocated.py)
for the eval-parity check on combined_d24_theta1m_gnorm. Mirrors the proven v483
vendored sub (512GB/304GB, CC==9.0, exclude bad nodes, run_v483.sh executable).
Each shard is its own condor slot: serve vLLM + run vendored agents+grader on
~30 instances x K=10 (~300 trajs/slot, matches the v483 shard size that completed
cleanly). pass@k aggregated per shard via aggregate_pass_at_k.py; union later.
"""
import subprocess, textwrap
from pathlib import Path

EVAL = Path("/home/rolmedo/nanoswe-final/eval")
EXPORT = "/fast/rolmedo/nanoswe/vllm_export/combined_d24_theta1m_gnorm"
TAG = "theta1m_gnorm_parity_vendored_subset91"
BASE_OUT = "/fast/rolmedo/nanoswe/swe_eval"
K, WORKERS, STEP_LIMIT, PORT = 10, 48, 100, 8123
N_SHARDS, BID = 3, "61"

for s in range(N_SHARDS):
    out_dir = f"{BASE_OUT}/{TAG}/shard{s}"
    shard_file = f"{EVAL}/ids/subset91_shards/shard_{s}.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = (
        f"HOME=/home/rolmedo EXPORT_DIR={EXPORT} OUT_DIR={out_dir} "
        f"K={K} WORKERS={WORKERS} STEP_LIMIT={STEP_LIMIT} PORT={PORT} "
        f"SUBSET=verified_cluster INSTANCE_IDS_FILE={shard_file} "
        f"APPTAINER_CACHEDIR=/tmp/apptainer_cache SINGULARITY_CACHEDIR=/tmp/apptainer_cache "
        f"APPTAINER_TMPDIR=/tmp/apptainer_tmp SINGULARITY_TMPDIR=/tmp/apptainer_tmp"
    )
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
        requirements   = (TARGET.CUDACapability == 9.0) && (TARGET.UtsnameNodename != "i307") && (TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105") && (TARGET.UtsnameNodename != "g174")
        +BypassLXCfs   = true
        queue
    """)
    sub_path = f"{out_dir}/submit.sub"
    Path(sub_path).write_text(sub)
    print(f"[submit] shard{s}: {shard_file} -> {out_dir}")
    subprocess.run(["condor_submit_bid", BID, sub_path], check=True)
print(f"\nsubmitted {N_SHARDS} vendored shards under {BASE_OUT}/{TAG}/")
