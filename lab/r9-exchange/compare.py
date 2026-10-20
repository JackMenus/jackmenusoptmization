"""Round 9: binary exchange moves for tabu and simulated annealing on discrete_general v2.

Usage: python compare.py OUT_DIR --split validation|test [--budget 5000] [--seeds 0 1 2 3 4]
Variant settings were fixed on one train instance (knapsack_correlated etc., seed 2010) before this script ran.
"""
import argparse
import gzip
import json
import os
import subprocess
import time

import numpy as np

from optiforge.benchmarks.runner import builtin, holm_adjust, paired_comparison, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import get_suite
from optiforge.core.budget import Budget

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--split", required=True, choices=["validation", "test"])
ap.add_argument("--budget", type=int, default=5000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
suite = get_suite("discrete_general")
inst = suite.instances(a.split)
specs = [builtin("random_search"), builtin("genetic"), builtin("tabu"), builtin("tabu", "tabu+exchange", exchange=True),
         builtin("simulated_annealing"), builtin("simulated_annealing", "sa+exchange", exchange_prob=0.5)]
t = time.time()
rows = run_matrix(specs, inst, a.seeds, Budget(max_evaluations=a.budget))
wall = time.time() - t
ref = reference_table([r for r in rows if r["solver"] in ("random_search", "genetic", "tabu", "simulated_annealing")])
summ = summarize(rows, ref)
pairs = [("tabu+exchange", "tabu"), ("sa+exchange", "simulated_annealing")]
comps = {f"{x} vs {y}": paired_comparison(rows, x, y) for x, y in pairs}
adj = holm_adjust({k: c["p_value_a_better"] for k, c in comps.items()})
for k, c in comps.items():
    c["p_holm"] = adj.get(k)
extra = {f"{x} vs genetic": paired_comparison(rows, x, "genetic") for x in ("tabu+exchange", "sa+exchange")}
fam = {}
for r in rows:
    fam.setdefault(r["instance"].split("#")[0], {}).setdefault(r["solver"], []).append(r)
table = {k: {s: {"mean_score": float(np.mean([x["score"] for x in v])), "solved": float(np.mean([x.get("solved", False) for x in v])),
                 "mean_gap": float(np.mean([x.get("gap", np.nan) for x in v]))} for s, v in d.items()} for k, d in fam.items()}
meta = {"suite": suite.to_dict(), "split": a.split, "budget": a.budget, "seeds": a.seeds, "wall_s": round(wall, 1),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "src_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())}
json.dump({"meta": meta, "summary": summ, "by_family": table, "primary": comps, "secondary": extra},
          open(os.path.join(a.out, f"{a.split}_summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(a.out, f"{a.split}_rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
print(json.dumps(meta, default=str))
names = [s.label for s in specs]
print("| family | " + " | ".join(names) + " |")
for k, d in table.items():
    print(f"| {k} | " + " | ".join(f"{d[s]['mean_score']:.3f}" for s in names) + " |")
print("| mean | " + " | ".join(f"{summ[s]['mean_score']:.3f}" for s in names) + " |")
for k, c in {**comps, **extra}.items():
    print(k, {f: c.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses", "p_value_a_better", "p_holm", "rank_biserial")})
