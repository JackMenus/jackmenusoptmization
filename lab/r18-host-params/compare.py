"""Round 18 (development data): host_ga parameter variants vs default host_ga across discrete and mixed families.

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
specs = [builtin("random_search"), builtin("genetic"), builtin("tabu"), builtin("simulated_annealing"), builtin("host_ga"),
         builtin("host_ga", "host_ga[sigma0.3]", sigma=0.3), builtin("host_ga", "host_ga[pop40]", population=40),
         builtin("host_ga", "host_ga[pop40,sigma0.3]", population=40, sigma=0.3)]
solvers = [s.label for s in specs]
rows = run_matrix(specs, inst, [0, 1, 2], Budget(max_evaluations=5000))
summ = summarize(rows, reference_table([r for r in rows if r["solver"] in ("random_search", "genetic", "tabu", "simulated_annealing")]))
VARIANTS = ["host_ga[sigma0.3]", "host_ga[pop40]", "host_ga[pop40,sigma0.3]"]
overall = {v: paired_comparison(rows, v, "host_ga") for v in VARIANTS}
fam = {}
for r in rows:
    fam.setdefault(r["instance"].split("#")[0], []).append(r)
per = {k: {s: float(np.mean([r["score"] for r in rs if r["solver"] == s])) for s in solvers} for k, rs in fam.items()}
json.dump({"instances": [i.id for i in inst], "summary": summ, "vs_host_ga": overall, "per_family": per},
          open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
print("| family | " + " | ".join(solvers[1:]) + " |")
for k, v in per.items():
    print(f"| {k} | " + " | ".join(f"{v[s]:.3f}" for s in solvers[1:]) + " |")
for v, c in overall.items():
    print(v, "vs host_ga", {f: c.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses", "p_value_a_better")})
print({s: round(summ[s]["mean_score"], 3) for s in solvers})
