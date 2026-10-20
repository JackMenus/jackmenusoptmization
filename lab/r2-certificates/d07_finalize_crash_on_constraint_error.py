# Constraint function raising during finalize's _named_violations crashes solve()
from common import *
calls = {"n": 0}
def g(x):
    calls["n"] += 1
    if calls["n"] > 31: raise RuntimeError("transient failure")
    return x[0] - 0.5
p = Problem(Space([Real("x", 0, 1)]), lambda x: float(x[0]), constraints=[ineq(g)])
try:
    r = get_solver("random_search").solve(p, Budget(max_evaluations=30), seed=0); show(r)
except Exception as e:
    print("solve() raised:", type(e).__name__, e)
