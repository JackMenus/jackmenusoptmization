import os, tempfile, numpy as np
from optiforge.solvers import get_solver, SOLVERS
from optiforge.benchmarks.problems import make_problem
from optiforge.core.budget import Budget
p = lambda: make_problem("multi_knapsack", 0, n=20, m=3)
s = get_solver("host_ga", population=10, sigma=0.3)
print(s.name, s.version, s.params, SOLVERS["host_ga"].default_params())
r_full = get_solver("host_ga", population=10, sigma=0.3).solve(p(), Budget(max_evaluations=1500), seed=1)
d = tempfile.mkdtemp(dir="/tmp/claude-0/review3"); ck = os.path.join(d, "c.json")
try:
    r1 = get_solver("host_ga", population=10, sigma=0.3).solve(p(), Budget(max_evaluations=700), seed=1, checkpoint_path=ck, checkpoint_every=5)
    print("partial", r1.status, r1.n_evaluations)
    r2 = get_solver("host_ga", population=10, sigma=0.3).solve(p(), Budget(max_evaluations=1500), seed=1, checkpoint_path=ck, checkpoint_every=5, resume=True)
    print("full", r_full.objective, r_full.n_evaluations, "resumed", r2.objective, r2.n_evaluations)
    # resume with different params
    r3 = get_solver("host_ga", population=30, sigma=0.05).solve(p(), Budget(max_evaluations=1500), seed=1, checkpoint_path=ck, checkpoint_every=5, resume=True)
    print("resumed with different params accepted:", r3.status, r3.messages[:2] if hasattr(r3,'messages') else '')
except Exception as e:
    import traceback; traceback.print_exc()
print(get_solver("host_ga").info())
