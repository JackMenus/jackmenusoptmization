"""Continuous solvers: gradient-based local, derivative-free local, population-based global.

* ``lbfgsb``      - L-BFGS-B (SciPy) with analytic or finite-difference gradients; bounds only.
* ``slsqp``       - SLSQP (SciPy), gradient-based local method with equality/inequality constraints.
* ``nelder_mead`` - Nelder-Mead (SciPy) with bounds; constraints through an internal penalty.
* ``de``          - differential evolution (own implementation, feasibility-first selection).
* ``cmaes``       - CMA-ES with IPOP restarts (own implementation, feasibility-first ranking).

Repeated local runs ("restarts") make the SciPy methods multi-start; the
dedicated ``multistart`` and ``hybrid`` solvers live in :mod:`.composite`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from ..core.context import SearchContext
from ..core.errors import BudgetExhausted
from ..core.result import GLOBAL_CONDITIONAL, LOCAL_CONVERGED, STATIONARY, Guarantee
from ..operators.primitives import rank_keys
from .base import IterativeSolver, Solver, register_solver

BIG = 1e30


def _starting_points(ctx: SearchContext, n: int) -> list[np.ndarray]:
    """Warm starts first, then the box centre, then Latin-hypercube points."""
    sp = ctx.space
    pts = list(ctx.warm_start)
    if not pts:
        pts.append(sp.lower + 0.5 * (sp.upper - sp.lower))
    if len(pts) < n:
        pts += list(sp.latin_hypercube(ctx.rng, n - len(pts)))
    return pts[:n]


def projected_gradient_norm(x, g, lower, upper) -> float:
    """Infinity norm of the projected-gradient step; zero iff x is first-order stationary on the box."""
    p = np.clip(x - g, lower, upper) - x
    return float(np.max(np.abs(p))) if p.size else 0.0


class _ScipyLocal(Solver):
    family = "local"
    space_kinds = frozenset({"continuous"})
    supports_permutations = False
    supports_categorical = False
    method = ""

    @classmethod
    def default_params(cls) -> dict:
        return {"restarts": 0, "tol": 1e-8, "stationarity_tol": 1e-5}

    def _fun(self, ctx: SearchContext, penalty: float | None):
        def f(z):
            ev = ctx.evaluate(ctx.space.project(z))
            if not ev.valid:
                return BIG
            if penalty is None:
                return ev.fmin
            return ev.fmin + penalty * ev.total_violation
        return f

    def _run(self, ctx: SearchContext):
        n_runs = 1 + int(self.params["restarts"])
        reason = None
        for k, x0 in enumerate(_starting_points(ctx, n_runs)):
            if ctx.evaluator.remaining() <= 0:
                break
            res = self._local(ctx, x0)
            ctx.diagnostics.setdefault("local_runs", []).append(
                {"start": k, "success": bool(res.success), "message": str(res.message), "nfev": int(res.nfev)})
            reason = "converged" if res.success else None
            self._hints(ctx, res)
        return reason

    def _local(self, ctx, x0):  # pragma: no cover - abstract
        raise NotImplementedError

    def _hints(self, ctx, res) -> None:
        pass


@register_solver
class LBFGSB(_ScipyLocal):
    name = "lbfgsb"
    version = "1.0.0"
    supports_constraints = False
    supports_noise = False
    description = "L-BFGS-B quasi-Newton (SciPy); analytic gradient if provided, else finite differences (counted as evaluations)."

    def _local(self, ctx, x0):
        sp = ctx.space
        jac = (lambda z: ctx.evaluator.gradient(sp.project(z))) if ctx.problem.gradient is not None else "2-point"
        maxfun = max(ctx.evaluator.remaining(), 1)
        res = optimize.minimize(self._fun(ctx, None), x0, method="L-BFGS-B", jac=jac,
                                bounds=list(zip(sp.lower, sp.upper)),
                                options={"maxfun": maxfun, "maxiter": maxfun, "ftol": self.params["tol"], "gtol": 1e-10})
        ctx.diagnostics["last_gradient"] = np.asarray(res.jac).tolist() if getattr(res, "jac", None) is not None else None
        ctx.diagnostics["last_x"] = np.asarray(res.x).tolist()
        return res

    def _hints(self, ctx, res) -> None:
        best = ctx.evaluator.best
        if best is None or not best.valid or getattr(res, "jac", None) is None:
            return
        if not np.allclose(best.x, ctx.space.project(res.x)):
            return  # the incumbent came from a different local run
        sp = ctx.space
        analytic = ctx.problem.gradient is not None
        g = ctx.evaluator.gradient(best.x) if analytic else np.asarray(res.jac)
        pg = projected_gradient_norm(best.x, g, sp.lower, sp.upper)
        tol = self.params["stationarity_tol"]
        ctx.diagnostics["projected_gradient_norm"] = pg
        ctx.guarantee_hints = [h for h in ctx.guarantee_hints if h.kind not in (STATIONARY, GLOBAL_CONDITIONAL)]
        if pg <= tol:
            how = "analytic gradient" if analytic else "finite-difference gradient estimate"
            ctx.guarantee_hints.append(Guarantee(
                STATIONARY,
                f"First-order stationarity on the box: ||P(x - grad f) - x||_inf = {pg:.3g} <= {tol:g} ({how}). "
                "Stationarity is necessary, not sufficient, for a local minimum; no local optimality is proven.",
                method="projected gradient norm", tolerance=tol, point=best.x.tolist()))
            if ctx.problem.convex and analytic:
                ctx.guarantee_hints.append(Guarantee(
                    GLOBAL_CONDITIONAL,
                    "Under the user's declaration that the objective is convex, a stationary point of the box-constrained "
                    f"problem is a global minimiser up to the stationarity tolerance {tol:g}.",
                    method="convexity + first-order optimality", tolerance=tol,
                    assumptions=["objective convex (declared, not verified)", "analytic gradient correct"],
                    point=best.x.tolist()))


@register_solver
class SLSQP(_ScipyLocal):
    name = "slsqp"
    version = "1.0.0"
    supports_noise = False
    description = "SLSQP sequential quadratic programming (SciPy); handles equality and inequality constraints."

    def _local(self, ctx, x0):
        sp, p = ctx.space, ctx.problem
        cons = []
        for c in p.constraints:
            if c.kind == "ineq":  # SciPy convention: fun(x) >= 0
                cons.append({"type": "ineq", "fun": (lambda z, c=c: -p.call_constraint(c, sp.project(z)))})
            else:
                cons.append({"type": "eq", "fun": (lambda z, c=c: p.call_constraint(c, sp.project(z)))})
        jac = (lambda z: ctx.evaluator.gradient(sp.project(z))) if p.gradient is not None else None
        maxiter = max(10, ctx.evaluator.remaining())
        return optimize.minimize(self._fun(ctx, None), x0, method="SLSQP", jac=jac, bounds=list(zip(sp.lower, sp.upper)),
                                 constraints=cons, options={"maxiter": maxiter, "ftol": self.params["tol"]})

    def _hints(self, ctx, res) -> None:
        best = ctx.evaluator.best
        if res.success and best is not None and best.feasible and np.allclose(best.x, ctx.space.project(res.x)):
            ctx.guarantee_hints = [h for h in ctx.guarantee_hints if h.kind != LOCAL_CONVERGED]
            ctx.guarantee_hints.append(Guarantee(
                LOCAL_CONVERGED,
                "SLSQP reported successful termination (its KKT-based test met to ftol). This indicates local "
                "convergence of the method; it is not a proof of local optimality.",
                method="SLSQP termination test", tolerance=self.params["tol"], point=best.x.tolist()))


@register_solver
class NelderMead(_ScipyLocal):
    name = "nelder_mead"
    version = "1.0.0"
    description = "Nelder-Mead simplex (SciPy) with bounds; constraints via an internal penalty used only for search."

    @classmethod
    def default_params(cls) -> dict:
        return {"restarts": 0, "tol": 1e-10, "penalty": 1e6, "initial_simplex_scale": 0.1}

    def _local(self, ctx, x0):
        sp = ctx.space
        maxfev = max(ctx.evaluator.remaining(), 1)
        n = sp.dim
        simplex = np.vstack([x0] + [x0 + np.eye(n)[i] * self.params["initial_simplex_scale"] * sp.width for i in range(n)])
        simplex = np.clip(simplex, sp.lower, sp.upper)
        pen = self.params["penalty"] if ctx.problem.has_constraints else None
        return optimize.minimize(self._fun(ctx, pen), x0, method="Nelder-Mead", bounds=list(zip(sp.lower, sp.upper)),
                                 options={"maxfev": maxfev, "maxiter": maxfev, "xatol": self.params["tol"],
                                          "fatol": self.params["tol"], "initial_simplex": simplex})

    def _hints(self, ctx, res) -> None:
        if res.success:
            ctx.guarantee_hints = [h for h in ctx.guarantee_hints if h.kind != LOCAL_CONVERGED]
            ctx.guarantee_hints.append(Guarantee(
                LOCAL_CONVERGED,
                "Nelder-Mead simplex collapsed below xatol/fatol. Nelder-Mead can stall at non-stationary points; "
                "no stationarity or local optimality is implied.",
                method="simplex size test", tolerance=self.params["tol"],
                point=ctx.space.project(res.x).tolist()))


# ---------------------------------------------------------------------------
# Differential evolution
# ---------------------------------------------------------------------------

@register_solver
class DifferentialEvolution(IterativeSolver):
    name = "de"
    version = "1.1.0"
    family = "global"
    space_kinds = frozenset({"continuous", "mixed", "discrete"})
    supports_permutations = False
    description = "DE/rand/1/bin or DE/current-to-pbest/1 with feasibility-first (Deb) selection; discrete coordinates rounded."

    @classmethod
    def default_params(cls) -> dict:
        return {"pop_size": None, "F": 0.6, "CR": 0.9, "strategy": "rand1", "p_best": 0.1}

    def initialize(self, ctx):
        sp = ctx.space
        n = self.params["pop_size"] or int(min(max(10, 8 * sp.dim), 80))
        X = sp.latin_hypercube(ctx.rng, n)
        for i, w in enumerate(ctx.warm_start[:n]):
            X[i] = w
        X = np.array([ctx.repair(x) for x in X])
        evs = ctx.evaluate_many(X)
        return {"X": X, "keys": _keys(evs), "gen": 0}

    def iterate(self, ctx, st):
        sp = ctx.space
        X, keys = st["X"], st["keys"]
        n = len(X)
        F, CR = self.params["F"], self.params["CR"]
        order = sorted(range(n), key=lambda i: tuple(keys[i]))
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            r1, r2, r3 = ctx.rng.choice(idx, size=3, replace=False)
            if self.params["strategy"] == "current_to_pbest":
                pb = order[int(ctx.rng.integers(max(1, int(self.params["p_best"] * n))))]
                donor = X[i] + F * (X[pb] - X[i]) + F * (X[r1] - X[r2])
            else:
                donor = X[r1] + F * (X[r2] - X[r3])
            cross = ctx.rng.random(sp.dim) < CR
            cross[ctx.rng.integers(sp.dim)] = True
            trial = np.where(cross, donor, X[i])
            # bounce-back boundary handling for reals
            r = sp.real_mask
            lo, hi = sp.lower[r], sp.upper[r]
            t = trial[r]
            t = np.where(t < lo, lo + ctx.rng.random(t.size) * (X[i][r] - lo), t)
            t = np.where(t > hi, hi - ctx.rng.random(t.size) * (hi - X[i][r]), t)
            trial[r] = t
            trial = ctx.repair(trial)
            ev = ctx.evaluate(trial)
            k = list(ev.rank_key())
            if tuple(k) <= tuple(keys[i]):
                X[i] = trial
                keys[i] = k
        st["gen"] += 1
        return None


def _keys(evs):
    return [list(e.rank_key()) for e in evs]


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

@register_solver
class CMAES(IterativeSolver):
    name = "cmaes"
    version = "1.0.0"
    family = "global"
    space_kinds = frozenset({"continuous"})
    supports_permutations = False
    supports_categorical = False
    description = ("CMA-ES (Hansen 2016 tutorial update rules) in normalised coordinates, feasibility-first ranking, "
                   "reflection at the bounds (repaired samples enter the update), IPOP restarts.")

    @classmethod
    def default_params(cls) -> dict:
        return {"sigma0": 0.3, "popsize": None, "restarts": True, "tol_sigma": 1e-11, "stagnation_gens": 100}

    def _fresh(self, ctx, lam: int, mean=None):
        n = ctx.space.dim
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        return {
            "lam": lam, "mu": mu, "w": w, "mueff": mueff,
            "mean": ctx.rng.random(n) if mean is None else mean,
            "sigma": self.params["sigma0"], "C": np.eye(n), "pc": np.zeros(n), "ps": np.zeros(n),
            "B": np.eye(n), "D": np.ones(n), "eigen_gen": 0, "gen": 0, "best_hist": [],
        }

    def initialize(self, ctx):
        n = ctx.space.dim
        lam0 = self.params["popsize"] or 4 + int(3 * math.log(n))
        mean = None
        if ctx.warm_start:
            mean = (ctx.warm_start[0] - ctx.space.lower) / ctx.space.width
        st = self._fresh(ctx, lam0, mean)
        st["lam0"] = lam0
        st["restarts"] = 0
        return st

    def iterate(self, ctx, st):
        sp = ctx.space
        n = sp.dim
        lam, mu, w, mueff = st["lam"], st["mu"], st["w"], st["mueff"]
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        B, D, mean, sigma = st["B"], st["D"], st["mean"], st["sigma"]
        Z = ctx.rng.standard_normal((lam, n))
        Y = Z @ (B * D).T
        U = mean + sigma * Y
        # boundary handling: reflect into the unit box and use the repaired sample in the update
        U = np.abs(U) % 2.0
        U = np.where(U > 1.0, 2.0 - U, U)
        Y = (U - mean) / sigma
        keys = []
        for u in U:
            ev = ctx.evaluate(ctx.repair(sp.lower + u * sp.width))
            keys.append(ev.rank_key())
        order = sorted(range(lam), key=lambda i: keys[i])
        sel = order[:mu]
        yw = np.sum(w[:, None] * Y[sel], axis=0)
        mean = mean + sigma * yw
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * st["ps"] + math.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (st["gen"] + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * st["pc"] + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        C = st["C"]
        rank_mu = (w[:, None, None] * np.einsum("ki,kj->kij", Y[sel], Y[sel])).sum(0)
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * rank_mu
        sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, 2.0)
        st["gen"] += 1
        if st["gen"] - st["eigen_gen"] > lam / (c1 + cmu) / n / 10:
            st["eigen_gen"] = st["gen"]
            C = np.triu(C) + np.triu(C, 1).T
            evals, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(evals, 1e-20))
        st.update(mean=mean, ps=ps, pc=pc, C=C, sigma=sigma, B=B, D=D)
        best_key = keys[order[0]]
        st["best_hist"] = (st["best_hist"] + [list(best_key)])[-self.params["stagnation_gens"]:]
        # restart tests
        stop = sigma * np.max(D) < self.params["tol_sigma"] or np.max(D) / np.min(D) > 1e7
        h = st["best_hist"]
        if len(h) >= self.params["stagnation_gens"] and tuple(h[-1]) >= tuple(h[0]):
            stop = True
        if stop:
            if not self.params["restarts"]:
                return "converged"
            st["restarts"] += 1
            new = self._fresh(ctx, int(st["lam"] * 2))
            new["lam0"], new["restarts"] = st["lam0"], st["restarts"]
            st.clear()
            st.update(new)
            ctx.diagnostics["cmaes_restarts"] = new["restarts"]
        return None
