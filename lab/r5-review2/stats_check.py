import numpy as np
from optiforge.benchmarks.runner import holm_adjust, paired_comparison
p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": None, "e": 0.5}
print(holm_adjust(p))  # expected a .04, c .09, b .09, e .5 (m=4)
try:
    from statsmodels.stats.multitest import multipletests
    print(multipletests([0.01, 0.04, 0.03, 0.5], method="holm")[1])
except ImportError:
    print("no statsmodels")
rng = np.random.default_rng(0)
rows = []
for i in range(10):
    for s in range(3):
        sb = rng.uniform(0.2, 0.4)
        rows += [{"solver": "A", "instance": i, "seed": s, "score": sb - 0.05}, {"solver": "B", "instance": i, "seed": s, "score": sb}]
c = paired_comparison(rows, "A", "B")
print({k: c[k] for k in ("pairs", "mean_diff", "ci95_mean_diff", "rank_biserial", "practical_wins", "p_value_a_better")})
# tiny differences: wilcoxon significant, CI below zero but practically null
rows2 = [dict(r) for r in rows]
for r in rows2:
    if r["solver"] == "A": r["score"] += 0.05 - 1e-6
c2 = paired_comparison(rows2, "A", "B")
print({k: c2[k] for k in ("mean_diff", "ci95_mean_diff", "rank_biserial", "practical_wins", "practical_ties", "p_value_a_better")})
# nan p from wilcoxon?
rows3 = [{"solver": "A", "instance": i, "seed": 0, "score": 0.1} for i in range(6)] + [{"solver": "B", "instance": i, "seed": 0, "score": 0.1 + (1e-11 if i == 0 else 0)} for i in range(6)]
print(paired_comparison(rows3, "A", "B")["p_value_a_better"])
