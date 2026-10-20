# MILP ignores Budget.max_time (uses its own 60 s default) and labels a HiGHS time-out "exact_solver_finished"
import time
from common import *
rng = np.random.default_rng(1)
m, n = 4, 40                                   # market-split instance (hard for branch and bound)
A = rng.integers(0, 100, size=(m, n)).astype(float)
b = np.floor(A.sum(1) / 2)
# objective: min sum of slacks; we use slack-free equality form with a dummy objective -> feasibility search
c = np.zeros(n)
p = Problem(Space([Binary(f"x{i}") for i in range(n)]), lambda x: 0.0,
            constraints=[eq(lambda x: A @ x - b, tol=1e-9)],
            linear=LinearStructure(c=c, A_eq=A, b_eq=b))
t = time.perf_counter()
r = get_solver("milp", time_limit=8.0).solve(p, Budget(max_time=1.0), seed=0)
print("wall time %.2fs with Budget(max_time=1.0)" % (time.perf_counter() - t))
show(r); print("highs:", r.diagnostics.get("highs_status"), r.diagnostics.get("highs_message"))
