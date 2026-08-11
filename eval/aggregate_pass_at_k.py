"""Aggregate inline-graded trajectories into a pass_at_k.json summary.

Walks `<base>/<tag>/slice_*/agents/<iid>/sample_*.traj.json` files, reads
`info.grading.resolved` from each (set by the inline grader), aggregates
to per-instance pass@k stats.

Usage:
    python aggregate_pass_at_k.py --base /fast/rolmedo/nanoswe/swe_eval \
        --tag combined_d24_v3_sssl_rerun_k10_subset91_strrepl \
        --out /fast/rolmedo/nanoswe/swe_eval/<tag>/pass_at_k.json
"""
from __future__ import annotations
import argparse, glob, json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/fast/rolmedo/nanoswe/swe_eval")
    ap.add_argument("--tag", required=True, help="e.g. combined_d24_v3_sssl_rerun_k10_subset91_strrepl")
    ap.add_argument("--out", help="output JSON path (default: <base>/<tag>/pass_at_k.json)")
    args = ap.parse_args()

    base = Path(args.base) / args.tag
    out_path = Path(args.out) if args.out else base / "pass_at_k.json"

    # Layouts: N_SLICES>1 ({base}/slice_*/agents/...), N_SLICES=1
    # ({base}/agents/...), and the direct runner --output layout
    # ({base}/<iid>/sample_*.traj.json, e.g. the setv400 shard runs).
    paths = sorted(glob.glob(str(base / "slice_*/agents/*/sample_*.traj.json")))
    if not paths:
        paths = sorted(glob.glob(str(base / "agents/*/sample_*.traj.json")))
    if not paths:
        paths = sorted(glob.glob(str(base / "*/sample_*.traj.json")))
    if not paths:
        raise SystemExit(f"no trajectory files found under {base}")
    print(f"[aggregate] found {len(paths)} traj files for {args.tag}")

    by_iid: dict[str, dict[int, dict]] = {}
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  parse error {p}: {e!r}")
            continue
        pp = Path(p)
        iid = pp.parent.name
        sample_n = int(pp.name.removeprefix("sample_").removesuffix(".traj.json"))
        g = d.get("info", {}).get("grading") or {}
        by_iid.setdefault(iid, {})[sample_n] = {
            "resolved": bool(g.get("resolved")),
            "status": g.get("status"),
            "f2p_pass": g.get("f2p_pass"),
            "f2p_fail": g.get("f2p_fail"),
            "p2p_pass": g.get("p2p_pass"),
            "p2p_fail": g.get("p2p_fail"),
            "patch_successfully_applied": g.get("patch_successfully_applied"),
        }

    # Per-instance summary + global pass@k
    per_iid = {}
    n_total_trajs = 0
    n_resolved_trajs = 0
    n_pass_at_1 = 0
    n_pass_at_5 = 0
    samples_per_iid_max = 0
    res_counts = Counter()
    status_counts = Counter()

    for iid, samples in by_iid.items():
        n_samples = len(samples)
        n_resolved = sum(1 for s in samples.values() if s["resolved"])
        any_resolved = n_resolved > 0
        first5_resolved = any(samples[s]["resolved"] for s in sorted(samples)[:5] if s in samples)
        per_iid[iid] = {
            "n_samples": n_samples,
            "n_resolved": n_resolved,
            "any_resolved": any_resolved,
            "first5_resolved": first5_resolved,
        }
        n_total_trajs += n_samples
        n_resolved_trajs += n_resolved
        if any_resolved:
            n_pass_at_1 += 1
        if first5_resolved:
            n_pass_at_5 += 1
        res_counts[n_resolved] += 1
        samples_per_iid_max = max(samples_per_iid_max, n_samples)
        for s in samples.values():
            status_counts[s["status"]] += 1

    summary = {
        "tag": args.tag,
        "base": str(base),
        "total_iids": len(by_iid),
        "samples_per_iid_max": samples_per_iid_max,
        "total_trajs": n_total_trajs,
        "resolved_trajs": n_resolved_trajs,
        "per_sample_resolved_rate": n_resolved_trajs / max(n_total_trajs, 1),
        "pass_at_1_unique_iids": n_pass_at_1,
        "pass_at_1_rate": n_pass_at_1 / max(len(by_iid), 1),
        "pass_at_5_unique_iids": n_pass_at_5,
        "pass_at_5_rate": n_pass_at_5 / max(len(by_iid), 1),
        "status_distribution": dict(status_counts),
        "resolved_count_distribution": {str(k): v for k, v in sorted(res_counts.items())},
        "by_iid": per_iid,
    }

    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[aggregate] {args.tag}")
    print(f"  trajs              : {n_total_trajs} ({n_resolved_trajs} resolved)")
    print(f"  per-sample resolved: {100*summary['per_sample_resolved_rate']:.2f}%")
    print(f"  pass@1 (any K)     : {n_pass_at_1}/{len(by_iid)} = {100*summary['pass_at_1_rate']:.1f}%")
    print(f"  pass@5 (first 5)   : {n_pass_at_5}/{len(by_iid)} = {100*summary['pass_at_5_rate']:.1f}%")
    print(f"  status: {dict(status_counts)}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
