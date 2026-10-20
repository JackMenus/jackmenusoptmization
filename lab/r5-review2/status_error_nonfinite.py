import math, numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Real
from optiforge.core.constraints import ineq
# constraint returns +inf ("infinitely violated") outside a tiny feasible region that random search does not hit
g = ineq(lambda x: 0.0 if abs(x[0] - 0.123456) < 1e-9 else math.inf, "pin")
p = Problem(Space([Real("a", 0, 1)]), lambda x: float(x[0]), constraints=[g])
r = solve(p, solver="random_search", budget=50)
print(r.status, r.termination_reason, r.messages[-1:])
# objective returning inf on the infeasible side (common penalty convention)
p2 = Problem(Space([Real("a", 0, 1)]), lambda x: float(x[0]) if x[0] > 0.9999999 else math.inf)
r2 = solve(p2, solver="random_search", budget=50)
print(r2.status, r2.messages[-1:])
