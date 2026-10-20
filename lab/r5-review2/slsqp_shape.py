import numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Real
from optiforge.core.constraints import ineq
def f(x):
    if abs(x[0] - 0.5) < 1e-12 and abs(x[1] - 0.5) < 1e-12:
        raise RuntimeError("fails at the start point")
    return float((x[0] - 0.2) ** 2 + (x[1] - 0.3) ** 2)
g = ineq(lambda x: np.array([x[0] + x[1] - 1.0, -x[0], x[1] - 0.9]), "g3")
p = Problem(Space([Real("a", 0, 1), Real("b", 0, 1)]), f, constraints=[g])
try:
    r = solve(p, solver="slsqp", budget=300, warm_start=[np.array([0.5, 0.5])])
    print(r.status, r.objective, r.x, r.termination_reason, r.messages)
except Exception as e:
    print("CRASH", type(e).__name__, e)
