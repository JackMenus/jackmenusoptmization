import sys
SCALE=float(sys.argv[1]); TOL=float(sys.argv[2])
import numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Real, Integer
from optiforge.core.constraints import LinearStructure
from collections import Counter
rng = np.random.default_rng(0)
stats = Counter(); ex=None
for t in range(200):
    n = 6
    vars_ = [Integer(f"i{k}", 0, 20) for k in range(3)] + [Real(f"r{k}", 0, 50) for k in range(3)]
    A_eq = rng.normal(size=(2, n)) * SCALE
    x0 = np.concatenate([rng.integers(0, 20, 3), rng.uniform(0, 50, 3)])
    b_eq = A_eq @ x0
    A_ub = rng.normal(size=(3, n)); b_ub = A_ub @ x0 + rng.uniform(0, 5, 3)
    c = rng.normal(size=n)
    m = LinearStructure(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None)
    p = Problem.from_linear(Space(vars_), m, sense=rng.choice(["min","max"]), tol=TOL)
    r = solve(p, solver="milp", budget=200, seed=t)
    stats[r.status] += 1
    if r.status != "optimal_certified" and ex is None:
        ex = (t, r.status, r.max_violation, r.messages[-3:])
print(stats); print(ex)
