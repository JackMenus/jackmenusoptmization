"""Round 14 (exploratory): where does the multi-knapsack gain of the round-10 GA variants come from?

Development instances only (multi_knapsack seeds 4200-4204, not used before or in any confirmatory test).
Usage: python ablate.py OUT_DIR
"""
import copy
import gzip
import json
import os
import sys

import numpy as np

from optiforge.benchmarks.runner import SolverSpec, builtin, paired_comparison, reference_table, run_matrix, summarize
from optiforge.benchmarks.suites import InstanceSpec
from optiforge.core.budget import Budget
from optiforge.dsl.spec import DEFAULT_OPERATOR, host_algorithm
from optiforge.registry.store import Registry

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
reg = Registry("registry")
champ = reg.get("ope-dsl-d23ef2a66f")
early = champ["source"]["else"]["else"]  # the variation used for the first 76% of the budget


def rec(cid, kind, source):
    return {"id": cid, "kind": kind, "representation": "dsl", "source": source, "status": "validated"}


def host_variant(cid, variation, **changes):
    h = host_algorithm(variation, name=cid)
    st = h["stages"][0]
    for k, v in changes.items():
        if k in ("population", "offspring"):
            h[k] = v
        elif k == "restart":
            h["restart"] = v
        else:
            st[k] = v
    return SolverSpec(cid, record=rec(cid, "algorithm", h))


specs = [
    builtin("genetic"),
    builtin("genetic", "genetic_pop20_t2", pop_size=20, tournament=2),
    SolverSpec("champion", record=champ),
    host_variant("host+default_op", DEFAULT_OPERATOR),
    host_variant("host+early_op", early),
    host_variant("host+default_op,no_restart", DEFAULT_OPERATOR, restart=None),
    host_variant("host+default_op,comma", DEFAULT_OPERATOR, survival={"op": "comma"}),
    builtin("random_search"), builtin("tabu"),
]
inst = [InstanceSpec("multi_knapsack", s, (("m", m), ("n", n))) for n, m in [(40, 5), (60, 10)] for s in range(4200, 4205)]
rows = run_matrix(specs, inst, [0, 1, 2], Budget(max_evaluations=5000))
summ = summarize(rows, reference_table([r for r in rows if r["solver"] in ("random_search", "tabu", "genetic")]))
comps = {s.label: paired_comparison(rows, s.label, "genetic") for s in specs if s.label not in ("genetic",)}
json.dump({"summary": summ, "vs_genetic": comps, "instances": [i.id for i in inst]},
          open(os.path.join(out, "ablation.json"), "w"), indent=1, default=str)
with gzip.open(os.path.join(out, "rows.json.gz"), "wt") as fh:
    json.dump(rows, fh, default=str)
for s in specs:
    c = comps.get(s.label, {})
    print(f"{s.label:32s} mean {summ[s.label]['mean_score']:.3f}  vs genetic {c.get('mean_diff', 0):+.3f} "
          f"CI {np.round(c.get('ci95_mean_diff', [0, 0]), 3).tolist()}")
