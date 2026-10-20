"""Round 16: pre-registered confirmation of host_ga vs genetic, and of the rule option ga=host_ga, on fresh instances.

Usage: python confirm.py OUT_DIR   (everything fixed in PREREG.md)
"""
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

from optiforge.benchmarks.runner import builtin, holm_adjust, paired_comparison, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import InstanceSpec
from optiforge.core.budget import Budget

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
FAMILIES = [("knapsack_correlated", {"n": 50}), ("knapsack_correlated", {"n": 100}), ("multi_knapsack", {"n": 40, "m": 5}),
            ("independent_set", {"n": 40}), ("set_cover", {"n_sets": 40, "n_elements": 30}),
            ("integer_quadratic", {"n": 6, "radius": 7}), ("knapsack", {"n": 25}), ("maxcut", {"n": 14}),
            ("tsp", {"n": 10}), ("mixed_rastrigin", {}), ("mixed_constrained", {"n": 4})]
inst = [InstanceSpec(f, s, tuple(sorted(kw.items()))) for f, kw in FAMILIES for s in range(5000, 5005)]
specs = [builtin("random_search"), builtin("tabu"), builtin("simulated_annealing"), builtin("genetic"), builtin("host_ga"),
         builtin("auto"), builtin("auto", "auto[ga=host_ga]", ga="host_ga")]
t = time.time()
rows = run_matrix(specs, inst, [900, 901, 902], Budget(max_evaluations=5000))
wall = time.time() - t
base = ["random_search", "tabu", "simulated_annealing", "genetic"]
summ = summarize(rows, reference_table([r for r in rows if r["solver"] in base]))
comps = {"host_ga vs genetic": paired_comparison(rows, "host_ga", "genetic"),
         "auto[ga=host_ga] vs auto": paired_comparison(rows, "auto[ga=host_ga]", "auto")}
adj = holm_adjust({k: v["p_value_a_better"] for k, v in comps.items()})
for k, v in comps.items():
    v["p_holm"] = adj[k]
    v["confirmed"] = bool(adj[k] < 0.05 and v["mean_diff"] <= -0.01 and v["ci95_mean_diff"][1] < 0)
fam = {}
for r in rows:
    fam.setdefault(r["instance"].split("#")[0], {}).setdefault(r["solver"], []).append(r["score"])
table = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in fam.items()}
meta = {"instances": [i.id for i in inst], "seeds": [900, 901, 902], "budget": 5000, "wall_s": round(wall, 1),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "src_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())}
json.dump({"meta": meta, "summary": summ, "by_family": table, "comparisons": comps},
          open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
names = [s.label for s in specs]
print("| family | " + " | ".join(names) + " |")
for k, d in table.items():
    print(f"| {k} | " + " | ".join(f"{d[s]:.3f}" for s in names) + " |")
for k, v in comps.items():
    print(k, {f: v.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses",
                                    "p_value_a_better", "p_holm", "rank_biserial", "confirmed")})
