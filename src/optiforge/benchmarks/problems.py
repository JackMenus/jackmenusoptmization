"""Benchmark problem families with seeded instances.

Every family is a function ``make(seed, **kwargs) -> Problem``.  Where a
verified optimum is available it is attached as ``problem.known_optimum`` with
the method that established it (analytic argument, dynamic programming,
complete enumeration, or a published reference value).  Solvers never read it;
only the benchmark scoring does.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.constraints import eq, ineq
from ..core.problem import KnownOptimum, Problem
from ..core.space import Binary, Categorical, Integer, Permutation, Real, Space
from ..dynamics.systems import ContinuousTimeSystem, DiscreteTimeSystem, simulation_problem


@dataclass(frozen=True)
class Family:
    name: str
    make: Callable[..., Problem]
    tags: frozenset
    description: str


FAMILIES: dict[str, Family] = {}


def family(name: str, tags: set, description: str):
    def deco(fn):
        FAMILIES[name] = Family(name, fn, frozenset(tags), description)
        return fn
    return deco


def make_problem(name: str, seed: int = 0, **kw) -> Problem:
    if name not in FAMILIES:
        raise KeyError(f"unknown benchmark family '{name}'. Available: {sorted(FAMILIES)}")
    p = FAMILIES[name].make(seed, **kw)
    p.metadata.setdefault("family", name)
    p.metadata.setdefault("instance_seed", seed)
    p.metadata.setdefault("tags", sorted(FAMILIES[name].tags))
    return p


def _rng(name: str, seed: int) -> np.random.Generator:
    """Deterministic per-family, per-instance generator (independent of PYTHONHASHSEED)."""
    return np.random.default_rng([sum((i + 1) * ord(ch) for i, ch in enumerate(name)), int(seed)])


def _box(dim, lo, hi):
    return Space([Real(f"x{i}", lo, hi) for i in range(dim)])


# ---------------------------------------------------------------------------
# continuous, unconstrained
# ---------------------------------------------------------------------------

@family("sphere", {"continuous", "convex", "unimodal"}, "Shifted sphere; convex, analytic gradient.")
def sphere(seed=0, dim=5):
    o = _rng("sphere", seed).uniform(-3, 3, dim)
    f = lambda x: float(np.sum((x - o) ** 2))
    return Problem(_box(dim, -5, 5), f, name=f"sphere-d{dim}-s{seed}", gradient=lambda x: 2 * (x - o), convex=True,
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


@family("ellipsoid", {"continuous", "convex", "ill_conditioned"}, "Rotated ellipsoid, condition number 1e4 (log10_cond).")
def ellipsoid(seed=0, dim=5, log10_cond=4):
    r = _rng("ellipsoid", seed)
    o = r.uniform(-3, 3, dim)
    Q, _ = np.linalg.qr(r.standard_normal((dim, dim)))
    w = 10 ** (log10_cond * np.arange(dim) / max(dim - 1, 1))
    f = lambda x: float(np.sum(w * (Q @ (x - o)) ** 2))
    grad = lambda x: 2 * Q.T @ (w * (Q @ (x - o)))
    return Problem(_box(dim, -5, 5), f, name=f"ellipsoid-d{dim}-s{seed}", gradient=grad, convex=True,
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


@family("rosenbrock", {"continuous", "nonconvex", "unimodal"}, "Shifted Rosenbrock valley.")
def rosenbrock(seed=0, dim=5):
    o = _rng("rosenbrock", seed).uniform(-2, 2, dim)

    def f(x):
        z = x - o + 1.0
        return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2))

    def g(x):
        z = x - o + 1.0
        gr = np.zeros_like(z)
        gr[:-1] = -400 * z[:-1] * (z[1:] - z[:-1] ** 2) - 2 * (1 - z[:-1])
        gr[1:] += 200 * (z[1:] - z[:-1] ** 2)
        return gr
    return Problem(_box(dim, -5, 5), f, name=f"rosenbrock-d{dim}-s{seed}", gradient=g,
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


@family("rastrigin", {"continuous", "nonconvex", "multimodal"}, "Shifted Rastrigin; highly multimodal.")
def rastrigin(seed=0, dim=5):
    o = _rng("rastrigin", seed).uniform(-2, 2, dim)
    f = lambda x: float(10 * dim + np.sum((x - o) ** 2 - 10 * np.cos(2 * np.pi * (x - o))))
    g = lambda x: 2 * (x - o) + 20 * np.pi * np.sin(2 * np.pi * (x - o))
    return Problem(_box(dim, -5.12, 5.12), f, name=f"rastrigin-d{dim}-s{seed}", gradient=g,
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


@family("ackley", {"continuous", "nonconvex", "multimodal"}, "Shifted Ackley.")
def ackley(seed=0, dim=5):
    o = _rng("ackley", seed).uniform(-5, 5, dim)

    def f(x):
        z = x - o
        return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + math.e)
    return Problem(_box(dim, -10, 10), f, name=f"ackley-d{dim}-s{seed}",
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


@family("styblinski_tang", {"continuous", "nonconvex", "multimodal"}, "Styblinski-Tang with random signs (2^d local minima).")
def styblinski_tang(seed=0, dim=5):
    s = _rng("styblinski_tang", seed).choice([-1.0, 1.0], dim)
    f = lambda x: float(0.5 * np.sum((s * x) ** 4 - 16 * (s * x) ** 2 + 5 * s * x))
    xstar = -2.903534027771177 * s
    fstar = float(f(xstar))
    return Problem(_box(dim, -5, 5), f, name=f"styblinski_tang-d{dim}-s{seed}",
                   known_optimum=KnownOptimum(fstar, xstar, "analytic (separable; 1-D minimiser -2.9035340277711770)", 1e-9))


# ---------------------------------------------------------------------------
# continuous, rotated / non-separable (added 2026-10-20: the original multimodal families are all separable)
# ---------------------------------------------------------------------------

def _haar_rotation(r: np.random.Generator, dim: int) -> np.ndarray:
    """Uniformly distributed random orthogonal matrix (QR of a Gaussian matrix with sign correction)."""
    Q, R = np.linalg.qr(r.standard_normal((dim, dim)))
    return Q * np.sign(np.diag(R))


@family("rastrigin_rotated", {"continuous", "nonconvex", "multimodal", "nonseparable", "rotated"},
        "Shifted Rastrigin composed with a random rotation; coordinate-wise moves no longer decompose it.")
def rastrigin_rotated(seed=0, dim=5):
    r = _rng("rastrigin_rotated", seed)
    o = r.uniform(-2, 2, dim)
    Q = _haar_rotation(r, dim)
    f = lambda x: float(10 * dim + np.sum((z := Q @ (x - o)) ** 2 - 10 * np.cos(2 * np.pi * z)))
    g = lambda x: Q.T @ (2 * (z := Q @ (x - o)) + 20 * np.pi * np.sin(2 * np.pi * z))
    return Problem(_box(dim, -5.12, 5.12), f, name=f"rastrigin_rotated-d{dim}-s{seed}", gradient=g,
                   known_optimum=KnownOptimum(0.0, o, "analytic (rotation preserves the minimiser at the shift)"))


@family("ackley_rotated", {"continuous", "nonconvex", "multimodal", "nonseparable", "rotated"},
        "Shifted Ackley composed with a random rotation.")
def ackley_rotated(seed=0, dim=5):
    r = _rng("ackley_rotated", seed)
    o = r.uniform(-5, 5, dim)
    Q = _haar_rotation(r, dim)

    def f(x):
        z = Q @ (x - o)
        return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + math.e)
    return Problem(_box(dim, -10, 10), f, name=f"ackley_rotated-d{dim}-s{seed}",
                   known_optimum=KnownOptimum(0.0, o, "analytic (rotation preserves the minimiser at the shift)"))


@family("schwefel_1_2", {"continuous", "convex", "unimodal", "nonseparable", "ill_conditioned"},
        "Shifted Schwefel 1.2 (sum of squared prefix sums): convex, fully non-separable, ill-conditioned.")
def schwefel_1_2(seed=0, dim=5):
    o = _rng("schwefel_1_2", seed).uniform(-3, 3, dim)
    f = lambda x: float(np.sum(np.cumsum(x - o) ** 2))
    g = lambda x: np.cumsum(np.cumsum(x - o)[::-1])[::-1] * 2
    return Problem(_box(dim, -5, 5), f, name=f"schwefel_1_2-d{dim}-s{seed}", gradient=g, convex=True,
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


@family("bent_cigar_rotated", {"continuous", "convex", "unimodal", "nonseparable", "ill_conditioned", "rotated"},
        "Rotated bent cigar: one direction 1e6 times flatter than all others.")
def bent_cigar_rotated(seed=0, dim=5):
    r = _rng("bent_cigar_rotated", seed)
    o = r.uniform(-3, 3, dim)
    Q = _haar_rotation(r, dim)
    w = np.full(dim, 1e6)
    w[0] = 1.0
    f = lambda x: float(np.sum(w * (Q @ (x - o)) ** 2))
    g = lambda x: 2 * Q.T @ (w * (Q @ (x - o)))
    return Problem(_box(dim, -5, 5), f, name=f"bent_cigar_rotated-d{dim}-s{seed}", gradient=g, convex=True,
                   known_optimum=KnownOptimum(0.0, o, "analytic"))


# ---------------------------------------------------------------------------
# continuous, constrained
# ---------------------------------------------------------------------------

@family("halfspace_sphere", {"continuous", "convex", "constrained"}, "Sphere centred outside a half-space; optimum = projection.")
def halfspace_sphere(seed=0, dim=5):
    r = _rng("halfspace_sphere", seed)
    c = r.uniform(-2, 2, dim)
    a = r.standard_normal(dim)
    a /= np.linalg.norm(a)
    b = float(a @ c) - r.uniform(0.5, 1.5)  # c violates a.x <= b
    xs = c - (a @ c - b) * a
    f = lambda x: float(np.sum((x - c) ** 2))
    return Problem(_box(dim, -5, 5), f, constraints=[ineq(lambda x: float(a @ x - b), "halfspace")],
                   name=f"halfspace_sphere-d{dim}-s{seed}", gradient=lambda x: 2 * (x - c), convex=True,
                   known_optimum=KnownOptimum(float((a @ c - b) ** 2), xs, "analytic (Euclidean projection)"))


@family("hyperplane_sphere", {"continuous", "convex", "constrained", "equality"}, "Sphere with one linear equality.")
def hyperplane_sphere(seed=0, dim=5):
    r = _rng("hyperplane_sphere", seed)
    c = r.uniform(-2, 2, dim)
    s = float(np.sum(c) + r.uniform(1, 2))
    xs = c + (s - np.sum(c)) / dim
    f = lambda x: float(np.sum((x - c) ** 2))
    return Problem(_box(dim, -5, 5), f, constraints=[eq(lambda x: float(np.sum(x) - s), "sum", tol=1e-6)],
                   name=f"hyperplane_sphere-d{dim}-s{seed}", gradient=lambda x: 2 * (x - c), convex=True,
                   known_optimum=KnownOptimum(float((s - np.sum(c)) ** 2 / dim), xs, "analytic (projection onto hyperplane)"))


@family("disk_rosenbrock", {"continuous", "nonconvex", "constrained"}, "2-D Rosenbrock restricted to the disk x^2+y^2 <= 2.")
def disk_rosenbrock(seed=0):
    f = lambda x: float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
    return Problem(_box(2, -1.5, 1.5), f, constraints=[ineq(lambda x: float(x[0] ** 2 + x[1] ** 2 - 2), "disk")],
                   name="disk_rosenbrock", known_optimum=KnownOptimum(0.0, np.ones(2), "analytic (unconstrained minimiser is feasible)"))


@family("g06", {"continuous", "nonconvex", "constrained"}, "CEC2006 g06: narrow nonconvex feasible region.")
def g06(seed=0):
    f = lambda x: float((x[0] - 10) ** 3 + (x[1] - 20) ** 3)
    cons = [ineq(lambda x: float(-(x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100), "g1"),
            ineq(lambda x: float((x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81), "g2")]
    sp = Space([Real("x0", 13, 100), Real("x1", 0, 100)])
    return Problem(sp, f, constraints=cons, name="g06",
                   known_optimum=KnownOptimum(-6961.81387558015, np.array([14.09500000000000064, 0.8429607892154795668]),
                                              "published reference value (Liang et al. 2006, CEC2006 suite)", 1e-4))


# ---------------------------------------------------------------------------
# discrete
# ---------------------------------------------------------------------------

@family("knapsack", {"discrete", "binary", "constrained", "combinatorial"}, "0/1 knapsack (maximise); optimum by DP.")
def knapsack(seed=0, n=20):
    r = _rng("knapsack", seed)
    w = r.integers(5, 40, n)
    v = r.integers(5, 60, n).astype(float)
    cap = int(0.4 * w.sum())
    # dynamic programming over capacity gives the exact optimum
    best = np.zeros(cap + 1)
    for i in range(n):
        best[w[i]:] = np.maximum(best[w[i]:], best[: cap + 1 - w[i]] + v[i])
    fstar = float(best[cap])
    sp = Space([Binary(f"b{i}") for i in range(n)])
    return Problem(sp, lambda x: float(v @ x), sense="max",
                   constraints=[ineq(lambda x: float(w @ x - cap), "capacity", tol=0.0)],
                   name=f"knapsack-n{n}-s{seed}", known_optimum=KnownOptimum(fstar, None, "dynamic programming"),
                   metadata={"weights": w.tolist(), "values": v.tolist(), "capacity": cap})


@family("tsp", {"discrete", "permutation", "combinatorial"}, "Euclidean TSP on random points; optimum by Held-Karp.")
def tsp(seed=0, n=9):
    r = _rng("tsp", seed)
    pts = r.random((n, 2))
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)

    def length(x):
        p = np.asarray(np.round(x), dtype=int)
        return float(D[p, np.roll(p, -1)].sum())
    fstar = _held_karp(D)
    return Problem(Space([Permutation("tour", n)]), length, name=f"tsp-n{n}-s{seed}",
                   known_optimum=KnownOptimum(fstar, None, "Held-Karp dynamic programming", 1e-9),
                   metadata={"points": pts.tolist()})


def _held_karp(D) -> float:
    n = len(D)
    full = 1 << (n - 1)
    C = np.full((full, n - 1), np.inf)
    for k in range(n - 1):
        C[1 << k, k] = D[0, k + 1]
    for S in range(1, full):
        for k in range(n - 1):
            if not (S >> k) & 1 or C[S, k] == np.inf:
                continue
            for j in range(n - 1):
                if (S >> j) & 1:
                    continue
                T = S | (1 << j)
                c = C[S, k] + D[k + 1, j + 1]
                if c < C[T, j]:
                    C[T, j] = c
    return float(min(C[full - 1, k] + D[k + 1, 0] for k in range(n - 1)))


@family("maxcut", {"discrete", "binary", "combinatorial"}, "Weighted max-cut (maximise); optimum by enumeration.")
def maxcut(seed=0, n=14):
    r = _rng("maxcut", seed)
    W = np.triu(r.random((n, n)) * (r.random((n, n)) < 0.5), 1)
    W = W + W.T

    def cut(x):
        s = np.asarray(x)
        return float(0.5 * np.sum(W * (s[:, None] != s[None, :])))
    # vectorised complete enumeration (fix node 0 by symmetry)
    best = 0.0
    for chunk in range(0, 1 << (n - 1), 4096):
        ids = np.arange(chunk, min(chunk + 4096, 1 << (n - 1)))
        S = ((ids[:, None] >> np.arange(n - 1)) & 1).astype(float)
        S = np.hstack([np.zeros((len(ids), 1)), S])
        diff = (S[:, :, None] != S[:, None, :])
        vals = 0.5 * np.einsum("bij,ij->b", diff, W)
        best = max(best, float(vals.max()))
    return Problem(Space([Binary(f"s{i}") for i in range(n)]), cut, sense="max", name=f"maxcut-n{n}-s{seed}",
                   known_optimum=KnownOptimum(best, None, "complete enumeration (2^(n-1) cuts)", 1e-9))


@family("integer_quadratic", {"discrete", "integer"}, "Convex quadratic over an integer box; optimum by enumeration.")
def integer_quadratic(seed=0, n=4, radius=6):
    r = _rng("integer_quadratic", seed)
    A = r.standard_normal((n, n))
    Q = A @ A.T + 0.5 * np.eye(n)
    a = r.uniform(-radius + 1, radius - 1, n)
    f = lambda z: float((z - a) @ Q @ (z - a))
    grid = np.array(list(itertools.product(range(-radius, radius + 1), repeat=n)), dtype=float)
    d = grid - a
    vals = np.einsum("bi,ij,bj->b", d, Q, d)
    k = int(np.argmin(vals))
    return Problem(Space([Integer(f"z{i}", -radius, radius) for i in range(n)]), f, name=f"integer_quadratic-n{n}-s{seed}",
                   known_optimum=KnownOptimum(float(vals[k]), grid[k], "complete enumeration", 1e-9))


# ---------------------------------------------------------------------------
# mixed
# ---------------------------------------------------------------------------

@family("mixed_rastrigin", {"mixed", "multimodal", "categorical"},
        "Reals + integers + a categorical choice of offset; separable, optimum analytic.")
def mixed_rastrigin(seed=0, n_real=4, n_int=3, n_cat=4):
    r = _rng("mixed_rastrigin", seed)
    offsets = r.uniform(-2, 2, (n_cat, n_real))
    bvals = r.integers(-4, 5, n_int).astype(float)
    cat_cost = r.uniform(0, 5, n_cat)
    cat_cost[r.integers(n_cat)] = 0.0
    choices = [f"mode{k}" for k in range(n_cat)]
    sp = Space([Real(f"x{i}", -5.12, 5.12) for i in range(n_real)] + [Integer(f"z{j}", -8, 8) for j in range(n_int)]
               + [Categorical("mode", choices)])

    def f(v):
        x = v[:n_real]
        z = v[n_real:n_real + n_int]
        c = int(round(v[-1]))
        d = x - offsets[c]
        return float(10 * n_real + np.sum(d ** 2 - 10 * np.cos(2 * np.pi * d)) + 3 * np.sum((z - bvals) ** 2) + cat_cost[c])
    return Problem(sp, f, name=f"mixed_rastrigin-s{seed}",
                   known_optimum=KnownOptimum(float(cat_cost.min()), None,
                                              "analytic (separable: Rastrigin min 0 at the offset, integer part 0 at b)"))


@family("mixed_constrained", {"mixed", "constrained", "integer", "binary"},
        "Mixed-integer quadratic with linking constraints x_i <= U*y_i and a budget; optimum by enumeration + projection.")
def mixed_constrained(seed=0, n=4):
    r = _rng("mixed_constrained", seed)
    t = r.uniform(0.5, 3, n)  # targets for continuous amounts
    gain = r.uniform(1, 3, n)  # linear reward per unit
    fixed = r.uniform(1, 6, n)  # cost of opening y_i
    cap = float(0.6 * (t + gain / 2).sum())
    sp = Space([Real(f"x{i}", 0, 5) for i in range(n)] + [Binary(f"y{i}") for i in range(n)])

    def f(v):
        x, y = v[:n], v[n:]
        return float(np.sum((x - t) ** 2) - gain @ x + fixed @ y)
    cons = [ineq(lambda v: v[:n] - 5.0 * v[n:], "link"), ineq(lambda v: float(np.sum(v[:n]) - cap), "budget")]
    # exact optimum: enumerate y, solve each convex QP subproblem with SLSQP to high accuracy
    from scipy.optimize import minimize
    best = math.inf
    for ys in itertools.product([0, 1], repeat=n):
        ys = np.array(ys, float)
        ub = 5.0 * ys
        obj = lambda x: float(np.sum((x - t) ** 2) - gain @ x + fixed @ ys)
        jac = lambda x: 2 * (x - t) - gain
        res = minimize(obj, ub * 0.1, jac=jac, method="SLSQP", bounds=list(zip(np.zeros(n), ub)),
                       constraints=[{"type": "ineq", "fun": lambda x: cap - np.sum(x), "jac": lambda x: -np.ones(n)}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        if res.success:
            best = min(best, float(res.fun))
    p = Problem(sp, f, constraints=cons, name=f"mixed_constrained-n{n}-s{seed}",
                known_optimum=KnownOptimum(best, None, "enumeration of binaries + SLSQP on each convex QP subproblem (numerical, tol 1e-6)", 1e-6))
    return p


# ---------------------------------------------------------------------------
# simulation-based system optimisation
# ---------------------------------------------------------------------------

@family("pd_control", {"continuous", "simulation", "constrained", "continuous_time"},
        "Tune PD gains of a mass-spring-damper (ODE) for setpoint tracking; overshoot and actuator limits.")
def pd_control(seed=0):
    r = _rng("pd_control", seed)
    m, c, k = r.uniform(0.5, 2.0), r.uniform(0.1, 0.6), r.uniform(1.0, 4.0)
    ref, umax = 1.0, 25.0

    def rhs(t, s, th):
        kp, kd = th
        u = np.clip(kp * (ref - s[0]) - kd * s[1], -umax * 2, umax * 2)
        return [s[1], (u - c * s[1] - k * s[0]) / m]
    sys = ContinuousTimeSystem(rhs, np.zeros(2), 6.0, rtol=1e-5, atol=1e-7)
    sp = Space([Real("kp", 0.0, 60.0), Real("kd", 0.0, 20.0)])

    def ise(t, X, th):
        e = ref - X[:, 0]
        return float(np.trapezoid(e ** 2, t))
    cons = [("overshoot", lambda t, X, th: X[:, 0] - 1.05 * ref),
            ("actuator", lambda t, X, th: np.abs(th[0] * (ref - X[:, 0]) - th[1] * X[:, 1]) - umax)]
    p = simulation_problem(sys, sp, ise, cons, name=f"pd_control-s{seed}", n_samples=120)
    p.metadata.update(mass=m, damping=c, stiffness=k)
    return p


@family("inventory", {"discrete", "integer", "simulation", "discrete_time", "constrained"},
        "(s, S) inventory policy in a discrete-time simulation with a fill-rate constraint.")
def inventory(seed=0, horizon=120):
    r = _rng("inventory", seed)
    demand = r.poisson(r.uniform(8, 14), horizon).astype(float)
    hold, back, order_cost = 1.0, 6.0, 40.0

    def step(x, kk, th):
        # state: [inventory, cumulative cost, cumulative lost sales, cumulative demand]; lost-sales model
        s, S = th
        inv, cost, lost, dem = x
        q = (S - inv) if inv < s else 0.0
        avail = inv + q
        served = min(avail, demand[kk])
        inv2 = avail - served
        unmet = demand[kk] - served
        stage = hold * inv2 + back * unmet + (order_cost if q > 0 else 0.0)
        return np.array([inv2, cost + stage, lost + unmet, dem + demand[kk]])
    sys = DiscreteTimeSystem(step, np.array([20.0, 0.0, 0.0, 0.0]), horizon)
    sp = Space([Integer("s", 0, 60), Integer("S", 0, 120)])
    cost = lambda t, X, th: float(X[-1, 1] / horizon)
    cons = [("order_up_to", lambda t, X, th: np.array([th[0] - th[1]])),
            ("fill_rate", lambda t, X, th: np.array([X[-1, 2] / max(X[-1, 3], 1.0) - 0.05]))]
    return simulation_problem(sys, sp, cost, cons, name=f"inventory-s{seed}")
