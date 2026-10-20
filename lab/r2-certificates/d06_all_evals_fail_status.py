# Every objective call raises -> status no_feasible_found (not error); termination budget
from common import *
def f(x): raise RuntimeError("simulator crashed")
p = Problem(Space([Real("x", 0, 1)]), f)
r = get_solver("random_search").solve(p, Budget(max_evaluations=100), seed=0); show(r)
print("diag errors:", r.diagnostics["n_errors"])
p2 = Problem(Space([Integer("x", 0, 5)]), f)
r2 = get_solver("exhaustive").solve(p2, seed=0); show(r2)
# flaky: best found feasible during search, verification call raises -> "No feasible solution was found"
calls = {"n": 0}
def g(x):
    calls["n"] += 1
    if calls["n"] > 50: raise RuntimeError("license server down")
    return float((x[0]-0.3)**2)
p3 = Problem(Space([Real("x", 0, 1)]), g)
r3 = get_solver("random_search").solve(p3, Budget(max_evaluations=50), seed=0); show(r3)
print("guarantees:", [(gg.kind, gg.statement[:60]) for gg in r3.guarantees])
