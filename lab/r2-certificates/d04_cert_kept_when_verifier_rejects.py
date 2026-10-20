# Exhaustive certificate kept while independent verifier rejects the point
from common import *
p = Problem(Space([Integer("x", 0, 5)]), lambda x: float(x[0]),
            verifier=lambda x: {"feasible": False, "scope": "dense"})
r = get_solver("exhaustive").solve(p, seed=0); show(r)
print("guarantees:", [g.kind for g in r.guarantees])
