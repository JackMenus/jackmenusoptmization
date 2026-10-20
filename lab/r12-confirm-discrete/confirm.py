"""Round 12: confirmatory comparison of candidates frozen in round 10 on discrete_general_v3 (fresh test instances).

Usage: python confirm.py OUT_DIR   (all settings fixed in PREREG.md; nothing is tuned here)
"""
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

from optiforge.benchmarks.runner import (SolverSpec, builtin, holm_adjust, paired_comparison, reference_table,
                                         run_matrix, summarize)
from optiforge.benchmarks.suites import get_suite
from optiforge.core.budget import Budget
from optiforge.registry.store import Registry

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
BUDGET, SEEDS = 5000, [700, 701, 702, 703, 704]
PRIMARY = "alg-dsl-f18700ed7f"
SECONDARY = "ope-dsl-d23ef2a66f"
suite = get_suite("discrete_general_v3")
inst = suite.instances("test")
reg = Registry("registry")
baselines = ["random_search", "simulated_annealing", "genetic", "tabu", "greedy_ls"]
specs = [builtin(b) for b in baselines] + [SolverSpec(c, record=reg.get(c)) for c in (PRIMARY, SECONDARY)]
t = time.time()
rows = run_matrix(specs, inst, SEEDS, Budget(max_evaluations=BUDGET))
wall = time.time() - t
ref = reference_table([r for r in rows if r["solver"] in baselines])
summ = summarize(rows, ref)
comps = {c: paired_comparison(rows, c, "genetic") for c in (PRIMARY, SECONDARY)}
adj = holm_adjust({c: v["p_value_a_better"] for c, v in comps.items()})
for c in comps:
    comps[c]["p_holm"] = adj[c]
    v = comps[c]
    v["confirmed"] = bool(v["p_holm"] < 0.05 and v["mean_diff"] <= -0.01 and v["ci95_mean_diff"][1] < 0)
fam = {}
for r in rows:
    fam.setdefault(r["instance"].split("#")[0], {}).setdefault(r["solver"], []).append(r)
table = {k: {s: {"mean_score": float(np.mean([x["score"] for x in v])), "solved": float(np.mean([x.get("solved", False) for x in v])),
                 "mean_gap": float(np.mean([x.get("gap", np.nan) for x in v]))} for s, v in d.items()} for k, d in fam.items()}
meta = {"suite": suite.to_dict(), "split": "test", "budget": BUDGET, "seeds": SEEDS, "wall_s": round(wall, 1),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "src_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())}
json.dump({"meta": meta, "summary": summ, "by_family": table, "comparisons": comps},
          open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
names = [s.label for s in specs]
print("| family | " + " | ".join(names) + " |")
for k, d in table.items():
    print(f"| {k} | " + " | ".join(f"{d[s]['mean_score']:.3f}" for s in names) + " |")
print("| mean | " + " | ".join(f"{summ[s]['mean_score']:.3f}" for s in names) + " |")
for c, v in comps.items():
    print(c, {f: v.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses",
                                    "p_value_a_better", "p_holm", "rank_biserial", "confirmed")})
