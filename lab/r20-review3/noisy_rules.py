import numpy as np, traceback
import optiforge as of
from optiforge.api import solve, plan
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Binary, Real, Integer
from optiforge.core.constraints import ineq
from optiforge.core.budget import Budget
rng = np.random.default_rng(0)
sp = Space([Binary(f"b{i}") for i in range(30)])
p = Problem(sp, lambda x: float(np.sum(x) + rng.normal()*0.1), constraints=[ineq(lambda x: float(3 - np.sum(x)), "c")], noisy=True)
for label, prob in [("noisy constrained discrete", p),
    ("noisy mixed", Problem(Space([Real("r",0,1)]+[Integer(f"i{k}",0,20) for k in range(5)]), lambda x: float(np.sum(x)+rng.normal()), noisy=True))]:
    print(label, "->", plan(prob, Budget(max_evaluations=2000), probe_cost=False).solver, plan(prob, Budget(max_evaluations=2000), probe_cost=False).params)
    try:
        r = solve(prob, budget=2000, seed=0)
        print("  ok", r.status)
    except Exception as e:
        print("  FAIL", type(e).__name__, e)
