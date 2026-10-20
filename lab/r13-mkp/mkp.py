"""Round 13: round-10 candidates vs genetic on new multidimensional knapsack instances (see PREREG.md)."""
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

from optiforge.benchmarks.runner import (SolverSpec, builtin, holm_adjust, paired_comparison, reference_table,
                                         run_matrix, summarize)
from optiforge.benchmarks.suites import InstanceSpec
from optiforge.core.budget import Budget
from optiforge.registry.store import Registry

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
CANDS = ["alg-dsl-f18700ed7f", "ope-dsl-d23ef2a66f"]
SIZES = [(40, 5), (40, 10), (60, 5), (60, 10)]
inst = [InstanceSpec("multi_knapsack", s, (("m", m), ("n", n))) for n, m in SIZES for s in range(4000, 4005)]
reg = Registry("registry")
base = ["random_search", "simulated_annealing", "tabu", "genetic"]
specs = [builtin(b) for b in base] + [SolverSpec(c, record=reg.get(c)) for c in CANDS]
t = time.time()
rows = run_matrix(specs, inst, [800, 801, 802, 803, 804], Budget(max_evaluations=5000))
wall = time.time() - t
summ = summarize(rows, reference_table([r for r in rows if r["solver"] in base]))
comps = {c: paired_comparison(rows, c, "genetic") for c in CANDS}
adj = holm_adjust({c: v["p_value_a_better"] for c, v in comps.items()})
for c, v in comps.items():
    v["p_holm"] = adj[c]
    v["supported"] = bool(adj[c] < 0.05 and v["mean_diff"] <= -0.01 and v["ci95_mean_diff"][1] < 0)
by = {}
for r in rows:
    by.setdefault(r["instance"].split("#")[0], {}).setdefault(r["solver"], []).append(r)
table = {k: {s: {"mean_score": float(np.mean([x["score"] for x in v])),
                 "mean_rel_gap": float(np.mean([x["gap"] / abs(x["known_internal"]) for x in v if "gap" in x]))}
             for s, v in d.items()} for k, d in by.items()}
meta = {"instances": [i.id for i in inst], "budget": 5000, "seeds": [800, 801, 802, 803, 804], "wall_s": round(wall, 1),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "src_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())}
json.dump({"meta": meta, "summary": summ, "by_size": table, "comparisons": comps},
          open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
names = [s.label for s in specs]
print("| size | " + " | ".join(names) + " |")
for k, d in table.items():
    print(f"| {k} | " + " | ".join(f"{d[s]['mean_score']:.3f} ({100 * d[s]['mean_rel_gap']:.2f}%)" for s in names) + " |")
for c, v in comps.items():
    print(c, {f: v.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses",
                                    "p_value_a_better", "p_holm", "rank_biserial", "supported")})
