"""Round 3: where do the baselines and the 0.1.0 generated continuous algorithm work, and where do they fail?

Uses ONLY the train and validation splits of suite `continuous_general` (v2). The test split is not touched.
Usage: python lab/r3-continuous/characterize.py OUT_DIR [--budget N] [--seeds 0 1 2]
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

from optiforge.benchmarks.runner import SolverSpec, builtin, paired_comparison, per_family, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import get_suite
from optiforge.core.budget import Budget
from optiforge.registry.store import Registry

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--budget", type=int, default=5000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--registry", default="registry")
ap.add_argument("--candidates", nargs="*", default=["alg-dsl-ae8b1f11f2"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

suite = get_suite("continuous_general")
inst = suite.instances("train") + suite.instances("validation")
baselines = ["random_search", "cmaes", "de", "nelder_mead", "pattern_search", "multistart", "hybrid", "auto"]
reg = Registry(a.registry)
specs = [builtin(b) for b in baselines] + [SolverSpec(c, record=reg.get(c)) for c in a.candidates]
t = time.time()
rows = run_matrix(specs, inst, a.seeds, Budget(max_evaluations=a.budget))
wall = time.time() - t
ref = reference_table([r for r in rows if r["solver"] in baselines])
summ = summarize(rows, ref)
fam = {}
for r in rows:  # per family+dimension
    key = r["instance"].split("#")[0]
    fam.setdefault(key, {}).setdefault(r["solver"], []).append(r)
table = {k: {s: {"mean_score": float(np.mean([x["score"] for x in v])),
                 "solved": float(np.mean([x.get("solved", False) for x in v])),
                 "median_gap": float(np.median([x.get("gap", np.inf) for x in v]))} for s, v in d.items()} for k, d in fam.items()}
comps = {c: {b: paired_comparison(rows, c, b) for b in ["cmaes", "hybrid", "auto"]} for c in a.candidates}
comps["auto"] = {b: paired_comparison(rows, "auto", b) for b in ["cmaes", "hybrid"]}
rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())
meta = {"suite": suite.to_dict(), "splits_used": ["train", "validation"], "budget": a.budget, "seeds": a.seeds,
        "solvers": [s.label for s in specs], "git": rev, "src_dirty": dirty, "wall_s": round(wall, 1),
        "python": sys.version, "numpy": np.__version__}
json.dump({"meta": meta, "summary": summ, "by_family": table, "comparisons": comps}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(a.out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
print(json.dumps(meta, default=str))
names = [s.label for s in specs]
print("| family | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for k in sorted(table):
    print(f"| {k} | " + " | ".join(f"{table[k][n]['mean_score']:.3f}" for n in names) + " |")
print("| **mean** | " + " | ".join(f"{summ[n]['mean_score']:.3f}" for n in names) + " |")
for c, d in comps.items():
    for b, r in d.items():
        print(c, "vs", b, {k: r[k] for k in ("pairs", "mean_diff", "wins", "losses", "ties", "p_value_a_better")})
