import numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Real
from optiforge.core.constraints import ineq, eq
n = {"c": 0}
def g(x):
    n["c"] += 1; return x[0] + x[1] - 1.0
p = Problem(Space([Real("a", 0, 1), Real("b", 0, 1)]), lambda x: float((x[0]-0.9)**2+(x[1]-0.6)**2), constraints=[ineq(g, "g")])
for b in (3, 7, 50, 200):
    n["c"] = 0
    r = solve(p, solver="slsqp", budget=b)
    print(b, r.status, r.termination_reason, r.n_evaluations, "constraint calls", n["c"], r.objective)
# declared jacobian with vector constraint and eq
p2 = Problem(Space([Real("a", -2, 2), Real("b", -2, 2)]), lambda x: float(x[0]**2 + 2*x[1]**2),
             constraints=[ineq(lambda x: np.array([1 - x[0], 0.5 - x[1]]), "g2", gradient=lambda x: np.array([[-1.0, 0], [0, -1.0]])),
                          eq(lambda x: x[0] - 2*x[1] - 0.5, "h", gradient=lambda x: np.array([1.0, -2.0]))])
r = solve(p2, solver="slsqp", budget=500); print(r.status, r.x, r.objective, r.n_evaluations)
