"""User-defined problems used by the example configurations.

These are ordinary trusted Python factories returning ``optiforge.Problem``.
"""
import numpy as np

from optiforge import Binary, Categorical, Integer, Problem, Real, Space, ineq
from optiforge.core.constraints import LinearStructure


def pressure_vessel():
    """Classic mixed-variable pressure-vessel design (Sandgren 1990).

    Shell/head thicknesses are integer multiples of 0.0625 in (encoded as integers 1..99);
    inner radius and length are continuous. Minimise material + forming + welding cost.
    The best value reported in the literature is about 6059.71; it is a reference, not a verified optimum.
    """
    sp = Space([Integer("ts_units", 1, 99), Integer("th_units", 1, 99), Real("R", 10.0, 200.0), Real("L", 10.0, 200.0)])

    def unpack(x):
        return 0.0625 * x[0], 0.0625 * x[1], x[2], x[3]

    def cost(x):
        ts, th, R, L = unpack(x)
        return 0.6224 * ts * R * L + 1.7781 * th * R ** 2 + 3.1661 * ts ** 2 * L + 19.84 * ts ** 2 * R

    cons = [
        ineq(lambda x: -unpack(x)[0] + 0.0193 * unpack(x)[2], "shell_thickness"),
        ineq(lambda x: -unpack(x)[1] + 0.00954 * unpack(x)[2], "head_thickness"),
        ineq(lambda x: (-np.pi * x[2] ** 2 * x[3] - 4.0 / 3.0 * np.pi * x[2] ** 3 + 1296000.0) / 1296000.0, "volume"),
        ineq(lambda x: x[3] - 240.0, "length"),
    ]
    return Problem(sp, cost, constraints=cons, name="pressure_vessel",
                   metadata={"literature_best": 6059.71, "reference_note": "literature value, not verified here"})


def production_planning():
    """Small integer production plan with a declared linear structure (enables the exact MILP adapter)."""
    profit = np.array([20.0, 30.0, 25.0, 18.0])
    A = np.array([[2.0, 3.0, 2.5, 1.0],    # machine hours
                  [1.0, 2.0, 1.5, 2.0],    # labour hours
                  [3.0, 1.0, 2.0, 1.5]])   # material
    b = np.array([100.0, 80.0, 120.0])
    sp = Space([Integer(f"q{i}", 0, 50) for i in range(4)])
    cons = [ineq(lambda x, k=k: float(A[k] @ x - b[k]), f"resource{k}", tol=1e-9) for k in range(3)]
    return Problem(sp, lambda x: float(profit @ x), sense="max", constraints=cons, name="production_planning",
                   linear=LinearStructure(c=profit, A_ub=A, b_ub=b))


def noisy_quadratic(sigma: float = 0.05, dim: int = 4, seed: int = 0):
    """Noisy objective: repeated evaluations differ; results report the mean of verification repeats."""
    rng = np.random.default_rng(seed)
    o = np.linspace(-1, 1, dim)
    return Problem(Space([Real(f"x{i}", -3, 3) for i in range(dim)]),
                   lambda x: float(np.sum((x - o) ** 2) + rng.normal(0, sigma)), noisy=True, name="noisy_quadratic")


def facility_mixed():
    """Mixed binary/categorical/continuous example: open facilities, choose a technology, set capacities."""
    demand = np.array([30.0, 45.0, 25.0])
    tech_cost = {"small": (1.0, 60.0), "medium": (0.8, 90.0), "large": (0.65, 130.0)}  # (unit cost, fixed cost)
    sp = Space([Binary("open0"), Binary("open1"), Binary("open2"), Categorical("tech", list(tech_cost)),
                Real("cap0", 0, 80), Real("cap1", 0, 80), Real("cap2", 0, 80)])

    def f(v):
        u, fixed = tech_cost[v["tech"]]
        opened = np.array([v["open0"], v["open1"], v["open2"]], dtype=float)
        caps = np.array([v["cap0"], v["cap1"], v["cap2"]])
        return float(fixed * opened.sum() + u * caps @ opened + 0.002 * (caps @ caps))

    cons = [ineq(lambda v: float(demand.sum() - sum(v[f"cap{i}"] * v[f"open{i}"] for i in range(3))), "meet_demand"),
            ineq(lambda v: float(np.max([v[f"cap{i}"] for i in range(3)]) - 70.0), "max_capacity")]
    return Problem(sp, f, constraints=cons, name="facility_mixed", input_format="dict")
