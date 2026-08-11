"""
Mechanistic validation of nanoswe.phases (no torch/GPU; behavior, not performance).
Run: python -m scripts.validate_phases
"""
import math
from nanoswe.phases import Phase, Plan, resolve_phases

def approx(a, b, tol=1e-9): return abs(a - b) <= tol

# ----- legacy single-phase formulas (verbatim from scripts/base_train.py) -----
def legacy_lr(it, warm, sched, start, final, N, wdr):
    if it < warm:
        return (it + 1) / warm * start
    if sched == "cosine":
        prog = min(1.0, (it - warm) / max(1, N - warm))
        return final + (start - final) * 0.5 * (1 + math.cos(math.pi * prog))
    wd = round(wdr * N)
    if it <= N - wd:
        return start
    prog = (N - it) / wd
    return prog * start + (1 - prog) * final

def legacy_mom(it, sched, N, wdr):
    if it < 400:
        f = it / 400; return (1 - f) * 0.85 + f * 0.97
    if sched == "cosine":
        prog = min(1.0, (it - 400) / max(1, N - 400)); return 0.97 * (1 - prog) + 0.90 * prog
    wd = round(wdr * N); start = N - wd
    if it >= start:
        f = (it - start) / wd; return 0.97 * (1 - f) + 0.90 * f
    return 0.97

def legacy_wd(it, N): return 0.5 * (1 + math.cos(math.pi * it / N))

# ----------------------------- the d40 two-phase plan -----------------------------
N1, N2 = 10218, 4687
d40 = Plan([
    Phase("base", N1, "cosine", 1.0, 0.30, 40, 0.0, "base", "token"),
    Phase("ft",   N2, "wsd",    0.30, 0.05, 0, 0.4, "ft",   "example"),
], wd_base=1.0)
B = N1

fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not cond: fails.append(name)

print("=== LR continuity (no re-warm dip) ===")
check("phase-1 floor == phase-2 plateau == 0.30", approx(d40.lr_mult(B-1), 0.30, 2e-3) and approx(d40.lr_mult(B), 0.30))
check("no dip at boundary", d40.lr_mult(B) >= 0.99 * d40.lr_mult(B-1),
      f"lr {d40.lr_mult(B-1):.4f}->{d40.lr_mult(B):.4f} (a warmup=100 re-warm would give ~0.0030)")
check("warmup only at global start; ends at 0.05", d40.lr_mult(0) < 0.05 and approx(d40.lr_mult(d40.total-1), 0.05, 1e-3))

print("=== Momentum (warm once, no re-warm, settle at true end) ===")
check("no re-warm at boundary (~0.97)", approx(d40.muon_momentum(B-1), 0.97) and approx(d40.muon_momentum(B), 0.97))
check("held 0.97 through phase 1", approx(d40.muon_momentum(N1//2), 0.97))
check("warmdown to 0.90 at the end", approx(d40.muon_momentum(d40.total-1), 0.90, 1e-3))

print("=== Weight decay (single base, full horizon, no jump) ===")
mono = all(d40.weight_decay(it) <= d40.weight_decay(it-50) + 1e-12 for it in range(50, d40.total, 50))
check("monotone over full horizon", mono)
check("no jump at boundary", abs(d40.weight_decay(B) - d40.weight_decay(B-1)) < 1e-3)
check("base at start, ~0 at end", approx(d40.weight_decay(0), 1.0) and d40.weight_decay(d40.total) < 1e-9)

print("=== Phase routing ===")
check("data/loss switch exactly at boundary",
      d40.data_source(B-1) == "base" and d40.data_source(B) == "ft"
      and d40.loss_norm(B-1) == "token" and d40.loss_norm(B) == "example")
check("boundaries() reports the FT start", d40.boundaries() == [N1])

print("=== 1-phase identity (single-phase WSD == legacy base_train) ===")
Nd = 5568
one = Plan([Phase("d24", Nd, "wsd", 1.0, 0.05, 40, 0.65, "", "example")], wd_base=1.0)
check("LR identical",  all(approx(one.lr_mult(it),       legacy_lr(it, 40, "wsd", 1.0, 0.05, Nd, 0.65)) for it in range(0, Nd, 7)))
check("momentum identical", all(approx(one.muon_momentum(it), legacy_mom(it, "wsd", Nd, 0.65)) for it in range(0, Nd, 7)))
check("weight decay identical", all(approx(one.weight_decay(it), legacy_wd(it, Nd)) for it in range(0, Nd, 7)))

print("=== resolve_phases (horizon from ratio or explicit iters) ===")
plan = resolve_phases(
    [{"data_source": "a", "target_param_data_ratio": 5.806, "lr_schedule": "cosine", "final_lr_frac": 0.30, "warmup_steps": 40, "loss_norm": "token"},
     {"data_source": "b", "num_iterations": 4687, "lr_schedule": "wsd", "lr_start_frac": 0.30, "final_lr_frac": 0.05, "warmup_steps": 0, "warmdown_ratio": 0.4}],
    scaling_params=3_230_000_000, total_batch_size=1_835_008, wd_base=0.123)
check("ratio resolves to a phase-1 horizon", plan.phases[0].num_iterations == int(5.806 * 3_230_000_000) // 1_835_008,
      f"n1={plan.phases[0].num_iterations}")
check("explicit iters honored", plan.phases[1].num_iterations == 4687)
check("wd_base threaded through", approx(plan.weight_decay(0), 0.123))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
import sys; sys.exit(1 if fails else 0)
