"""Round 8 pilot: calibrate instance sizes for a version-2 discrete suite on pilot seeds (1100, 1101) that the suite
will NOT use. Usage: python pilot.py OUT_DIR --budgets 3000 10000 --seeds 0 1 2"""
import argparse
import gzip
import json
import os
import time

import numpy as np

from optiforge.benchmarks.runner import builtin, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import InstanceSpec
from optiforge.core.budget import Budget

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--budgets", type=int, nargs="+", default=[3000, 10000])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
CANDIDATES = [("knapsack_correlated", {"n": 50}), ("knapsack_correlated", {"n": 100}),
              ("multi_knapsack", {"n": 40, "m": 5}), ("maxcut", {"n": 20}),
              ("integer_quadratic", {"n": 6, "radius": 7}), ("independent_set", {"n": 40}),
              ("independent_set", {"n": 60})]
inst = [InstanceSpec(f, s, tuple(sorted(kw.items()))) for f, kw in CANDIDATES for s in (1100, 1101)]
solvers = ["random_search", "simulated_annealing", "genetic", "tabu", "greedy_ls", "auto"]
out = {}
for B in a.budgets:
    t = time.time()
    rows = run_matrix([builtin(s) for s in solvers], inst, a.seeds, Budget(max_evaluations=B))
    ref = reference_table(rows)
    summarize(rows, ref)  # adds score / gap / solved to rows
    fam = {}
    for r in rows:
        fam.setdefault(r["instance"].split("#")[0], {}).setdefault(r["solver"], []).append(r)
    out[B] = {"wall_s": round(time.time() - t, 1),
              "by_family": {k: {s: {"solved": float(np.mean([x.get("solved", False) for x in v])),
                                    "feasible": float(np.mean([x["feasible"] for x in v])),
                                    "median_gap": float(np.median([x.get("gap", np.inf) for x in v])),
                                    "mean_score": float(np.mean([x["score"] for x in v]))} for s, v in d.items()}
                            for k, d in fam.items()}}
    with gzip.open(os.path.join(a.out, f"rows_B{B}.json.gz"), "wt") as fh:
        json.dump(rows, fh, default=str)
json.dump({"instances": [i.id for i in inst], "solvers": solvers, "seeds": a.seeds, "results": out},
          open(os.path.join(a.out, "pilot.json"), "w"), indent=1)
for B, res in out.items():
    print(f"\n## budget {B} (wall {res['wall_s']} s): solved rate / median gap")
    print("| family | " + " | ".join(solvers) + " |")
    for k, d in res["by_family"].items():
        print(f"| {k} | " + " | ".join(f"{d[s]['solved']:.2f}/{d[s]['median_gap']:.3g}" for s in solvers) + " |")
