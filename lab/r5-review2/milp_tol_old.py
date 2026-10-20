import sys, numpy as np
from collections import Counter
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Real, Integer
from optiforge.core.constraints import LinearStructure, ineq, eq
rng = np.random.default_rng(0); stats = Counter()
for t in range(100):
    n = 6
    vars_ = [Integer(f"i{k}", 0, 20) for k in range(3)] + [Real(f"r{k}", 0, 50) for k in range(3)]
    A_eq = rng.normal(size=(2, n)); x0 = np.concatenate([rng.integers(0, 20, 3), rng.uniform(0, 50, 3)])
    b_eq = A_eq @ x0; A_ub = rng.normal(size=(3, n)); b_ub = A_ub @ x0 + rng.uniform(0, 5, 3); c = rng.normal(size=n)
    m = LinearStructure(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    p = Problem(Space(vars_), lambda x, c=c: float(c @ x), constraints=[ineq(lambda x, A=A_ub, b=b_ub: A @ x - b, "u", 1e-9), eq(lambda x, A=A_eq, b=b_eq: A @ x - b, "e", 1e-9)], linear=m)
    r = solve(p, solver="milp", budget=200, seed=t); stats[r.status] += 1
print(stats)
