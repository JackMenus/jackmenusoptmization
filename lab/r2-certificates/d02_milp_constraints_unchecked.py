# Declared linear constraints are never checked when the problem has no constraint functions
from common import *
p = Problem(Space([Integer("x", 0, 10**6)]), lambda x: float(x[0]),
            linear=LinearStructure(c=np.array([1.0]), A_ub=np.array([[-1.0]]), b_ub=np.array([-1.0])))  # declares x>=1
r = get_solver("milp").solve(p, seed=0); show(r)
print("true optimum: x=0, f=0 (unconstrained problem)")
