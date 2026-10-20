# sense="max" + convex=True: L-BFGS-B issues "global optimum conditional on assumptions" at the global MINIMUM
from common import *
p = Problem(Space([Real("x", -1, 1)]), lambda x: float(x[0] ** 2), sense="max", convex=True,
            gradient=lambda x: 2 * x)
r = get_solver("lbfgsb").solve(p, seed=0); show(r)
print("guarantees:", [g.kind for g in r.guarantees], " true max = 1 at x=+-1")
