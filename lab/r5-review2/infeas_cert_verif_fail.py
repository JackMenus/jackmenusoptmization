import numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Integer
from optiforge.core.constraints import ineq
calls = {"n": 0}
def f(x):
    calls["n"] += 1
    if calls["n"] > 4:          # behaves differently at verification time (declared deterministic)
        raise RuntimeError("flaky")
    return float(x[0])
p = Problem(Space([Integer("a", 0, 3)]), f, constraints=[ineq(lambda x: 1.0, "never")])
r = solve(p, solver="exhaustive", budget=100)
print(r.status, [g.kind for g in r.guarantees]); print(r.messages)
