# spot_checks=0 is accepted: declaration is never compared to the real functions, yet certificates are issued
from common import *
# real problem: min x s.t. x >= 2 ; declaration (wrong): x >= 20 and x <= 10 -> HiGHS infeasible
p = Problem(Space([Integer("x", 0, 100)]), lambda x: float(x[0]), constraints=[ineq(lambda x: 2 - x[0])],
            linear=LinearStructure(c=np.array([1.0]), A_ub=np.array([[-1.0], [1.0]]), b_ub=np.array([-20.0, 10.0])))
r = get_solver("milp", spot_checks=0).solve(p, seed=0); show(r)
p2 = Problem(Space([Integer("x", 0, 100)]), lambda x: float(x[0]), constraints=[ineq(lambda x: 2 - x[0])],
             linear=LinearStructure(c=np.array([1.0]), A_ub=np.array([[-1.0]]), b_ub=np.array([-7.0])))
r2 = get_solver("milp", spot_checks=0).solve(p2, seed=0); show(r2); print("true optimum x=2, f=2")
