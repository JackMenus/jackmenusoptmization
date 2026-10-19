"""Post-hoc diagnostic (not part of selection): does the generated continuous algorithm's advantage on Rastrigin
survive a random rotation, i.e. is it exploiting separability?  Same budget (4000) and seeds 0-9 for every solver."""
import json

import numpy as np

from optiforge import Budget, Problem, Real, Space
from optiforge.generation.materialize import build_solver
from optiforge.registry.store import Registry
from optiforge.solvers import get_solver


def rastrigin(seed, rotate, dim=5):
    r = np.random.default_rng([777, seed])
    o = r.uniform(-2, 2, dim)
    Q = np.linalg.qr(r.standard_normal((dim, dim)))[0] if rotate else np.eye(dim)
    f = lambda x: float(10 * dim + np.sum((Q @ (x - o)) ** 2 - 10 * np.cos(2 * np.pi * (Q @ (x - o)))))
    return Problem(Space([Real(f"x{i}", -5.12, 5.12) for i in range(dim)]), f, name=f"rastrigin-rot{rotate}-{seed}")


reg = Registry("registry")
out = {}
for rotate in (False, True):
    for label in ("alg-dsl-ae8b1f11f2", "cmaes", "de", "random_search"):
        vals = []
        for seed in range(10):
            s = build_solver(reg.get(label)) if label.startswith("alg-") else get_solver(label)
            vals.append(s.solve(rastrigin(seed, rotate), Budget(4000), seed=seed).objective)
        out[f"{'rotated' if rotate else 'separable'}/{label}"] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                                                                   "solved(<1e-6)": int(np.sum(np.array(vals) < 1e-6))}
print(json.dumps(out, indent=1))
