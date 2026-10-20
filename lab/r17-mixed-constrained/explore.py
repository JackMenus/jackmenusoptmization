"""Round 17 (exploratory, development instances mixed_constrained seeds 5100-5109): why does host_ga lose to
genetic on mixed_constrained? Usage: python explore.py OUT_DIR"""
import gzip
import json
import os
import sys

import numpy as np

from optiforge.benchmarks.runner import SolverSpec, builtin, paired_comparison, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import InstanceSpec
from optiforge.core.budget import Budget
from optiforge.dsl.spec import DEFAULT_OPERATOR, host_algorithm

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def host(cid, variation=DEFAULT_OPERATOR, population=20, **changes):
    h = host_algorithm(variation, population=population, name=cid)
    for k, v in changes.items():
        if k in ("repair", "restart", "init"):
            h[k] = v
        else:
            h["stages"][0][k] = v
    return SolverSpec(cid, record={"id": cid, "kind": "algorithm", "representation": "dsl", "source": h, "status": "validated"})


specs = [builtin("random_search"), builtin("genetic"), builtin("host_ga"),
         builtin("hybrid", "hybrid[genetic]", global_solver="genetic", local="pattern_search", global_fraction=0.8),
         builtin("hybrid", "hybrid[host_ga]", global_solver="host_ga", local="pattern_search", global_fraction=0.8),
         builtin("host_ga", "host_ga[pop40]", population=40),
         host("host,no_restart", restart=None),
         host("host,repair=resample", repair={"op": "resample"}),
         host("host,sigma0.3", variation={"op": "seq", "children": [{"op": "uniform_cx"}, {"op": "type_aware", "sigma": 0.3}]}),
         host("host,no_step", step=None)]
inst = [InstanceSpec("mixed_constrained", s, (("n", 4),)) for s in range(5100, 5110)]
rows = run_matrix(specs, inst, [0, 1, 2], Budget(max_evaluations=5000))
summ = summarize(rows, reference_table([r for r in rows if r["solver"] in ("random_search", "genetic")]))
comps = {s.label: paired_comparison(rows, s.label, "genetic") for s in specs if s.label != "genetic"}
json.dump({"summary": summ, "vs_genetic": comps}, open(os.path.join(out, "explore.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
for s in specs:
    c = comps.get(s.label, {})
    sm = summ[s.label]
    print(f"{s.label:22s} mean {sm['mean_score']:.3f} median {sm['median_score']:.3f} feas {sm['feasibility_rate']:.2f} "
          f"vs genetic {c.get('mean_diff', 0):+.3f} CI {np.round(c.get('ci95_mean_diff', [0, 0]), 3).tolist()}")
