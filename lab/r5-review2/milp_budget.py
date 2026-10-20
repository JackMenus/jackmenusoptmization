import numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Real, Integer
from optiforge.core.constraints import LinearStructure
m = LinearStructure(c=np.array([1.0, 2.0]), A_ub=np.array([[-1.0, -1.0]]), b_ub=np.array([-3.0]))
p = Problem.from_linear(Space([Integer("a", 0, 5), Integer("b", 0, 5)]), m)
for b in (5, 30, 60):
    r = solve(p, solver="milp", budget=b)
    print(b, r.status, r.objective, r.termination_reason, r.n_evaluations, r.messages[-2:])
pm = Problem.from_linear(Space([Integer("a", 0, 5), Integer("b", 0, 5)]), m, sense="max")
r = solve(pm, solver="milp", budget=100); print("max", r.status, r.objective, r.x)
