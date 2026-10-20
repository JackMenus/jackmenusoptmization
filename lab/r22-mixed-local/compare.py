"""Round 22 (development data): local stage of rule R13's hybrid on mixed problems: pattern_search (current) vs
greedy_ls (single-move neighbourhood search incl. bit flips). Instances: mixed suite train+validation and
mixed_constrained seeds 5100-5109 (round-17 development instances). Usage: python compare.py OUT_DIR"""
import gzip
import json
import os
import sys

import numpy as np

from optiforge.benchmarks.runner import builtin, paired_comparison, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import InstanceSpec, get_suite
from optiforge.core.budget import Budget

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
su = get_suite("mixed")
inst = su.instances("train") + su.instances("validation") + [InstanceSpec("mixed_constrained", s, (("n", 4),)) for s in range(5100, 5110)]
H = dict(global_solver="host_ga", global_fraction=0.8)
specs = [builtin("random_search"), builtin("genetic"),
         builtin("hybrid", "hybrid[pattern_search]", local="pattern_search", **H),
         builtin("hybrid", "hybrid[greedy_ls]", local="greedy_ls", **H)]
rows = run_matrix(specs, inst, [0, 1, 2], Budget(max_evaluations=5000))
summ = summarize(rows, reference_table([r for r in rows if r["solver"] in ("random_search", "genetic")]))
c = paired_comparison(rows, "hybrid[greedy_ls]", "hybrid[pattern_search]")
fam = {}
for r in rows:
    fam.setdefault(r["instance"].split("#")[0], {}).setdefault(r["solver"], []).append(r["score"])
json.dump({"summary": summ, "comparison": c, "by_family": {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in fam.items()}},
          open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
for k, d in fam.items():
    print(k, {s: round(float(np.mean(v)), 3) for s, v in d.items()})
print({f: c.get(f) for f in ("pairs", "mean_diff", "ci95_mean_diff", "practical_wins", "practical_losses", "p_value_a_better")})
