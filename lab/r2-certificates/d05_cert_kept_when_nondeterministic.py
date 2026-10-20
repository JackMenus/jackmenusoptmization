# Non-determinism detected by finalize, certificate still issued
from common import *
calls = {"n": 0}
def f(x):
    calls["n"] += 1
    # values drift with call count: the enumerated "minimum" is not reproducible
    return float(x[0]) + (0.0 if calls["n"] <= 6 else 100.0)
p = Problem(Space([Integer("x", 0, 5)]), f)
r = get_solver("exhaustive").solve(p, seed=0); show(r)
