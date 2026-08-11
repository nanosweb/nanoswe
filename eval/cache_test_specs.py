"""Pre-build SWE-bench TestSpec objects and cache the fields the grader needs.

Run ONCE on the submit node (where HTTP proxy works). Writes a JSON file with
{iid: {eval_script, instance_image_key, instance_id}} entries. The grader on
the compute slot reads from this cache and skips network calls entirely.

Why this is necessary: swebench.harness.test_spec.make_test_spec() fetches
requirements.txt / environment.yml from raw.githubusercontent.com for repos
(django, xarray, flask, pylint, ...) whose specs aren't bundled in the
swebench package. At any scale this is a reliability + rate-limit hazard.

Cache is per (instance_id) — invariant across model / sample / pass-run.
Safe to share across all evals.

Usage:
    python cache_test_specs.py --instance-ids expand500/v091_ids.json \
        --out /lustre/home/rolmedo/nanoswe/runs/swe_eval/test_spec_cache.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

DEFAULT_OUT = Path("/fast/rolmedo/nanoswe/test_spec_cache.json")
DEFAULT_IDS = Path("/lustre/home/rolmedo/nanoswe/runs/swe_eval/expand500/v091_ids.json")
DEFAULT_DATASET = "ricdomolm/SWE-bench_Verified-Working-Harbor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-ids", default=str(DEFAULT_IDS),
                    help="JSON with {instance_ids: [...]} OR a list.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all", action="store_true",
                    help="Cache the full dataset, ignoring --instance-ids.")
    args = ap.parse_args()

    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec
    import os
    os.environ.setdefault("HF_HOME", "/fast/rolmedo/nanoswe/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/fast/rolmedo/nanoswe/hf_cache/datasets")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    print(f"[cache] loading dataset {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    by_id = {r["instance_id"]: r for r in ds}

    if args.all:
        ids = list(by_id.keys())
    else:
        payload = json.loads(Path(args.instance_ids).read_text())
        ids = payload.get("instance_ids", payload if isinstance(payload, list) else [])
    ids = [i for i in ids if i in by_id]
    print(f"[cache] {len(ids)} instances to spec")

    # Load existing cache so re-runs are incremental.
    out_path = Path(args.out)
    cache = {}
    if out_path.exists():
        try:
            cache = json.loads(out_path.read_text())
            print(f"[cache] loaded existing cache: {len(cache)} entries")
        except Exception as e:
            print(f"[cache] existing cache unreadable ({e!r}); starting fresh")

    n_built = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.time()
    for i, iid in enumerate(ids):
        if iid in cache and cache[iid].get("eval_script"):
            n_skipped += 1
            continue
        try:
            ts = make_test_spec(by_id[iid])
            cache[iid] = {
                "instance_id": ts.instance_id,
                "instance_image_key": ts.instance_image_key,
                "eval_script": ts.eval_script,
                # Required by swebench.harness.grading.get_eval_report:
                "repo": ts.repo,
                "version": ts.version,
                "FAIL_TO_PASS": list(ts.FAIL_TO_PASS),
                "PASS_TO_PASS": list(ts.PASS_TO_PASS),
            }
            n_built += 1
        except Exception as e:
            n_failed += 1
            print(f"  [{iid}] FAILED to build test_spec: {e!r}")
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(ids)}  built={n_built} skipped={n_skipped} failed={n_failed}")

    out_path.write_text(json.dumps(cache, indent=2))
    elapsed = time.time() - t0
    print(f"[cache] wrote {out_path}  total={len(cache)}  built={n_built}  skipped={n_skipped}  failed={n_failed}  wallclock={elapsed:.1f}s")


if __name__ == "__main__":
    main()
