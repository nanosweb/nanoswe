"""
Mechanistic validation of the data-mixture sampler (nanoswe.phases): the
deterministic credit-SWRR must draw sources proportionally for constant weights
AND track a linear fade faithfully. Pure logic, no data. Run:
    python -m scripts.validate_mixture
"""
from collections import Counter
from nanoswe.phases import Mixture, CreditRoundRobin, make_transition, lerp_weight

fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not cond: fails.append(name)

def realized(mixture, draws):
    """Run the sampler for `draws` steps with f = step/draws; return list of picks."""
    rr = CreditRoundRobin(len(mixture.sources))
    picks = []
    for t in range(draws):
        f = t / draws
        picks.append(rr.select(mixture.weights(f)))
    return picks

def window_frac(picks, mixture, lo, hi):
    """Realized source fractions over the f-window [lo, hi)."""
    n = len(picks)
    sel = [picks[t] for t in range(n) if lo <= t / n < hi]
    c = Counter(sel)
    return [c.get(i, 0) / max(1, len(sel)) for i in range(len(mixture.sources))]

def expected_frac(mixture, lo, hi, samples=200):
    """Mean normalized weights over the window (the target the sampler tracks)."""
    acc = [0.0] * len(mixture.sources)
    for s in range(samples):
        f = lo + (hi - lo) * (s + 0.5) / samples
        w = mixture.weights(f); W = sum(w)
        for i in range(len(w)): acc[i] += w[i] / W
    return [a / samples for a in acc]

print("=== constant mixture draws proportionally ===")
const = Mixture([{"origin": "a", "weight": 5.0, "seed": 1},
                 {"origin": "b", "weight": 3.0, "seed": 2},
                 {"origin": "c", "weight": 2.0, "seed": 3}])
fr = window_frac(realized(const, 100000), const, 0.0, 1.0)
check("proportions ~ [0.5, 0.3, 0.2]", all(abs(fr[i] - e) < 5e-3 for i, e in enumerate([0.5, 0.3, 0.2])),
      f"realized={[round(x,3) for x in fr]}")

print("=== transition: realized mix tracks the linear fade at f=0.05/0.5/0.95 ===")
src_from = [{"origin": "A", "weight": 8}, {"origin": "Z", "weight": 4}]   # Z shared
src_to   = [{"origin": "Z", "weight": 2}, {"origin": "B", "weight": 6}]
mix = make_transition(src_from, src_to)
print(f"  union sources (origin -> (start,end)): {[(s['origin'], s['weight']) for s in mix.sources]}")
picks = realized(mix, 120000)
for lo, hi in [(0.0, 0.1), (0.45, 0.55), (0.9, 1.0)]:
    got = window_frac(picks, mix, lo, hi)
    exp = expected_frac(mix, lo, hi)
    ok = all(abs(got[i] - exp[i]) < 0.02 for i in range(len(got)))
    check(f"window f∈[{lo},{hi}) matches integrated weights", ok,
          f"got={[round(x,3) for x in got]} exp={[round(x,3) for x in exp]}")

print("=== make_transition builds the right union ===")
by = {s["origin"]: s["weight"] for s in mix.sources}
check("A fades out (8 -> 0)", by["A"] == (8.0, 0.0))
check("Z interpolates (4 -> 2)", by["Z"] == (4.0, 2.0))
check("B fades in (0 -> 6)", by["B"] == (0.0, 6.0))
check("distinct seeds", len({s["seed"] for s in mix.sources}) == len(mix.sources))
check("weights(0)=start, weights(1)=end",
      mix.weights(0.0) == [8.0, 4.0, 0.0] and mix.weights(1.0) == [0.0, 2.0, 6.0])

print("=== faithfulness: a ramping source's cumulative draws ~ ∫w ===")
# Z ramps 4->2 (avg 3) out of a total that also changes; just sanity-check the
# fade-in source B never appears in the first 1% and dominates the last 1%.
b_idx = next(i for i, s in enumerate(mix.sources) if s["origin"] == "B")
early_frac = sum(1 for p in picks[:1200] if p == b_idx) / 1200  # first 1% (f<0.01)
late_frac = sum(1 for p in picks[-1200:] if p == b_idx) / 1200
# Smooth fade => the fade-in source is RARE (not zero) at the start, drawn in
# exact proportion to its tiny rising weight, and dominant at the end.
check("fade-in source rare at the very start", early_frac < 0.01, f"early B frac={early_frac:.4f}")
check("fade-in source dominant at the very end", late_frac > 0.6, f"late B frac={late_frac:.2f}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
import sys; sys.exit(1 if fails else 0)
