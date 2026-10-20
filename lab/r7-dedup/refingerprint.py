"""Round 7: fingerprint every candidate benchmarked in round 6 (train-split probes only) and check whether equal
fingerprints coincide with equal round-6 train results. Usage: python refingerprint.py <out.json>"""
import json
import sys
from collections import defaultdict

from optiforge.benchmarks.runner import builtin
from optiforge.benchmarks.suites import get_suite
from optiforge.generation.fingerprint import behaviour_fingerprint, probe_instances
from optiforge.generation.materialize import build_solver
from optiforge.registry.store import Registry

reg = Registry("registry")
probes = probe_instances(get_suite("continuous_general").instances("train"))
R6_START = "2026-10-20T12:16"
rows = []
for meta in reg.list():
    rec = reg.get(meta["id"]) if isinstance(meta, dict) and "source" not in meta else meta
    if rec.get("provider") == "baseline" or rec["created"] < R6_START:
        continue
    tr = [b for b in rec.get("benchmarks", []) if b.get("split") == "train" and b.get("reference_baseline")]
    if not tr:
        continue
    fp, used = behaviour_fingerprint(lambda r=rec: build_solver(r, allow_unvalidated=True), probes)
    rows.append({"id": rec["id"], "provider": rec["provider"], "fingerprint": fp, "evals": used,
                 "train_score": rec.get("train_score"), "train_mean": tr[-1]["summary"].get("mean_score")})
for b in ["random_search", "de", "nelder_mead", "cmaes", "hybrid"]:
    fp, used = behaviour_fingerprint(builtin(b).build, probes)
    rows.append({"id": b, "provider": "baseline", "fingerprint": fp, "evals": used})
groups = defaultdict(list)
for r in rows:
    groups[r["fingerprint"]].append(r["id"])
dup = {k: v for k, v in groups.items() if len(v) > 1}
by_id = {r["id"]: r for r in rows}
consistent = all(len({by_id[i].get("train_mean") for i in v}) == 1 for v in dup.values())
out = {"probes": [i.id for i in probes], "n_candidates": sum(r["provider"] != "baseline" for r in rows),
       "n_fingerprints": len(groups), "duplicate_groups": dup,
       "duplicates_have_identical_train_means": consistent,
       "distinct_fingerprints_with_equal_train_mean": [
           m for m in {by_id[i].get("train_mean") for i in by_id if by_id[i]["provider"] != "baseline"}
           if len({by_id[i]["fingerprint"] for i in by_id if by_id[i].get("train_mean") == m}) > 1],
       "rows": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
