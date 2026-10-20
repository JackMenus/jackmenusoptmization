"""Reuse of one DSLAlgorithm instance across solve() calls: stale Cholesky factor via id(X) reuse."""
import sys; sys.path.insert(0, "/home/claude/dev/optiforge")
import numpy as np
from optiforge import Budget
from optiforge.benchmarks.problems import make_problem
from optiforge.dsl import interpreter as I
CORR_SPEC = {"dsl_version": 1, "kind": "algorithm", "name": "corr", "population": 16, "offspring": 16,
             "init": {"op": "lhs"}, "repair": {"op": "reflect"},
             "stages": [{"name": "s", "parent_selection": {"op": "tournament", "k": 2},
                         "variation": {"op": "seq", "children": [{"op": "arithmetic_cx"},
                                                                 {"op": "correlated_gaussian", "scale": 0.8}]},
                         "survival": {"op": "plus"}, "step": {"op": "one_fifth"}}]}

stats = {"calls": 0, "stale": 0, "shape_mismatch": 0}
orig = I._Env.elite_factor
def checked(self, elite):
    L = orig(self, elite)
    saved = dict(self.algo.__dict__.get("_chol_cache", {}))
    self.algo.__dict__["_chol_cache"] = {}
    fresh = orig(self, elite)
    self.algo.__dict__["_chol_cache"] = saved
    stats["calls"] += 1
    if (L is None) != (fresh is None) or (L is not None and (L.shape != fresh.shape or not np.allclose(L, fresh))):
        stats["stale"] += 1
        if L is not None and fresh is not None and L.shape != fresh.shape:
            stats["shape_mismatch"] += 1
    return L
I._Env.elite_factor = checked

algo = I.DSLAlgorithm(CORR_SPEC)
pop = CORR_SPEC["population"]
errs = []
for t in range(200):
    dim = 6
    p = make_problem("ellipsoid" if t % 2 else "rastrigin_rotated", t, dim=dim)
    r = algo.solve(p, Budget(pop + 3), seed=t)   # ends inside generation 0
    if r.termination_reason == "error": errs.append(r.messages[-1])
print(stats, "errors:", errs[:2])
