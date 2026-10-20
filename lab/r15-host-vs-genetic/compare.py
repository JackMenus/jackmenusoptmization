"""Round 15 (development data): host_ga vs genetic across discrete and mixed families.

Instances: train + validation of discrete_general_v3, discrete_combinatorial, mixed and permutation (v1 test splits
are historical, train/validation are development data). Usage: python compare.py OUT_DIR
"""
import gzip
import json
import os
import sys

import numpy as np

from optiforge.benchmarks.runner import builtin, paired_comparison, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import get_suite
from optiforge.core.budget import Budget

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
inst = []
for s in ("discrete_general_v3", "discrete_combinatorial", "mixed", "permutation"):
    su = get_suite(s)
    inst += su.instances("train") + su.instances("validation")
solvers = ["random_search", "genetic", "host_ga", "tabu", "simulated_annealing"]
rows = run_matrix([builtin(s) for s in solvers], inst, [0, 1, 2], Budget(max_evaluations=5000))
summ = summarize(rows, reference_table(rows))
overall = paired_comparison(rows, "host_ga", "genetic")
fam = {}
for r in rows:
    fam.setdefault(r["instance"].split("#")[0], []).append(r)
per = {}
for k, rs in fam.items():
    c = paired_comparison(rs, "host_ga", "genetic")
    per[k] = {"host_ga": float(np.mean([r["score"] for r in rs if r["solver"] == "host_ga"])),
              "genetic": float(np.mean([r["score"] for r in rs if r["solver"] == "genetic"])),
              "mean_diff": c.get("mean_diff"), "ci95": c.get("ci95_mean_diff"), "wins": c.get("practical_wins"),
              "losses": c.get("practical_losses")}
json.dump({"instances": [i.id for i in inst], "summary": summ, "overall": overall, "per_family": per},
          open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
for k, v in per.items():
    print(f"{k:40s} host {v['host_ga']:.3f} genetic {v['genetic']:.3f} diff {v['mean_diff']:+.3f} CI {np.round(v['ci95'], 3).tolist()} W/L {v['wins']}/{v['losses']}")
print("overall", {f: overall.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses", "p_value_a_better")})
print({s: round(summ[s]["mean_score"], 3) for s in solvers})
