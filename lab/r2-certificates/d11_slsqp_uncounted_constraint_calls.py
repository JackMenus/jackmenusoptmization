# SLSQP calls constraint functions outside the Evaluator: not counted, not time-gated, exceptions abort the solve
from common import *
calls = {"g": 0, "f": 0}
def f(x): calls["f"] += 1; return float(np.sum((x - 0.7) ** 2))
def g(x): calls["g"] += 1; return float(np.sum(x) - 1.0)
p = Problem(Space([Real(f"x{i}", 0, 1) for i in range(5)]), f, constraints=[ineq(g)])
r = get_solver("slsqp").solve(p, Budget(max_evaluations=40), seed=0); show(r)
print("objective calls:", calls["f"], " constraint calls:", calls["g"], " reported n_evaluations:", r.n_evaluations)
