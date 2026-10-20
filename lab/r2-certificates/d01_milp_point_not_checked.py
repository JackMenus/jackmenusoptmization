# MILP certifies a point whose real objective disagrees with the declared linear objective
from common import *
f = lambda x: float(x[0]) + (1000.0 if x[0] == 0 else 0.0)   # linear except at x=0
p = Problem(Space([Integer("x", 0, 10**6)]), f, linear=LinearStructure(c=np.array([1.0])))
r = get_solver("milp").solve(p, seed=0); show(r)
print("objective at certified point:", r.objective, "  f(1) =", f(np.array([1.0])))
