import numpy as np
from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Integer
from optiforge.core.constraints import LinearStructure
m = LinearStructure(c=np.array([1.0, 1.0]), A_ub=np.array([[-1.0, -1.0]]), b_ub=np.array([-3.0]))
p = Problem.from_linear(Space([Integer("a", 0, 5), Integer("b", 0, 5)]), m)
from collections import Counter
c = Counter()
for s in range(20):
    r = solve(p, solver="milp", budget=200, seed=s); c[(r.status, r.objective)] += 1
print(c); print([m_ for m_ in r.messages if "dropped" in m_])
