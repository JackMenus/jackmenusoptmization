from optiforge import solve
from optiforge.core.problem import Problem
from optiforge.core.space import Space, Integer
def ver(x): raise RuntimeError("verifier crashed")
p = Problem(Space([Integer("a", 0, 3)]), lambda x: float(x[0]), verifier=ver)
r = solve(p, solver="exhaustive", budget=10)
print(r.status, [ (g.kind, g.statement[:90]) for g in r.guarantees])
