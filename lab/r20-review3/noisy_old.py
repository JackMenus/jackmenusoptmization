import numpy as np
from optiforge.solvers import get_solver
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Binary, Real, Integer
from optiforge.core.constraints import ineq
from optiforge.core.budget import Budget
rng = np.random.default_rng(0)
p = Problem(Space([Binary(f"b{i}") for i in range(30)]), lambda x: float(np.sum(x) + rng.normal()*0.1), constraints=[ineq(lambda x: float(3 - np.sum(x)), "c")], noisy=True)
q = Problem(Space([Real("r",0,1)]+[Integer(f"i{k}",0,20) for k in range(5)]), lambda x: float(np.sum(x)+rng.normal()), noisy=True)
print(get_solver("genetic").solve(p, Budget(max_evaluations=2000)).status)
print(get_solver("hybrid", global_solver="genetic", local="pattern_search", global_fraction=0.8).solve(q, Budget(max_evaluations=2000)).status)
