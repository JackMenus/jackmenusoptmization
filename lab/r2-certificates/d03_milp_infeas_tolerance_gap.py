# MILP certifies infeasibility although a point feasible under the problem's own tolerance exists
from common import *
from optiforge.core.evaluator import Evaluator
b = 3.000005
p = Problem(Space([Integer("x", 0, 10**6)]), lambda x: float(x[0]),
            constraints=[eq(lambda x: x[0] - b, tol=1e-5)],
            linear=LinearStructure(c=np.array([1.0]), A_eq=np.array([[1.0]]), b_eq=np.array([b])))
r = get_solver("milp").solve(p, seed=0); show(r)
ev = Evaluator(p).evaluate(np.array([3.0]))
print("x=3 evaluates feasible:", ev.feasible, "violations", ev.violations)
