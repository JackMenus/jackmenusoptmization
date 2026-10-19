"""Composite strategies: multi-start, global-plus-local hybrids, and compass search.

Composite solvers run sub-solvers on the *same* search context, so all
evaluations are counted once against one budget and the incumbent is shared.
A stage's share of the budget is enforced with ``Evaluator.stage_limit``.
"""
from __future__ import annotations

import numpy as np

from ..core.context import SearchContext
from ..core.errors import BudgetExhausted
from ..core.result import LOCAL_CONVERGED, Guarantee
from .base import IterativeSolver, Solver, get_solver, register_solver


def run_stage(ctx: SearchContext, solver: Solver, max_evals: int, warm_start=None) -> str:
    """Run ``solver`` inside ``ctx`` for at most ``max_evals`` more evaluations."""
    ev = ctx.evaluator
    prev_limit, prev_ws = ev.stage_limit, ctx.warm_start
    ev.stage_limit = ev.n_evals + max(int(max_evals), 0)
    if prev_limit is not None:
        ev.stage_limit = min(ev.stage_limit, prev_limit)
    if warm_start is not None:
        ctx.warm_start = [np.asarray(w, dtype=float) for w in warm_start]
    ctx.operator_versions[solver.name] = solver.version
    try:
        reason = solver._run(ctx)
        return reason or "completed"
    except BudgetExhausted as e:
        if e.kind != "stage_budget":
            raise
        return "stage_budget"
    finally:
        ev.stage_limit = prev_limit
        ctx.warm_start = prev_ws


def default_local_for(problem) -> str:
    if problem.space.kind != "continuous" or problem.noisy:
        return "pattern_search"
    if problem.has_constraints:
        return "slsqp"
    return "lbfgsb"


@register_solver
class PatternSearch(IterativeSolver):
    name = "pattern_search"
    version = "1.0.0"
    family = "local"
    description = ("Compass (coordinate pattern) search on real and integer coordinates with step halving; "
                   "other discrete coordinates are held fixed. Usable as a local refiner for mixed problems.")

    @classmethod
    def default_params(cls) -> dict:
        return {"step0": 0.1, "min_step": 1e-7, "shrink": 0.5, "expand": 1.0}

    def initialize(self, ctx):
        sp = ctx.space
        x = ctx.warm_start[0].copy() if ctx.warm_start else sp.lower + 0.5 * (sp.upper - sp.lower)
        x = ctx.repair(x)
        ev = ctx.evaluate(x)
        return {"x": x, "key": list(ev.rank_key()), "step": float(self.params["step0"])}

    def iterate(self, ctx, st):
        sp = ctx.space
        coords = np.flatnonzero(sp.real_mask | sp.int_mask)
        if coords.size == 0:
            return "converged"
        x, key = st["x"], tuple(st["key"])
        improved = False
        for i in ctx.rng.permutation(coords):
            h = st["step"] * sp.width[i]
            if sp.int_mask[i]:
                h = max(1.0, round(h))
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] += sgn * h
                y = ctx.repair(y)
                if np.array_equal(y, x):
                    continue
                ev = ctx.evaluate(y)
                if ev.rank_key() < key:
                    x, key, improved = y, ev.rank_key(), True
                    break
        st["x"], st["key"] = x, list(key)
        if improved:
            st["step"] *= self.params["expand"]
        else:
            st["step"] *= self.params["shrink"]
            int_only_done = not sp.real_mask.any() and st["step"] * np.max(sp.width) < 0.5
            if st["step"] < self.params["min_step"] or int_only_done:
                best = ctx.evaluator.best
                if best is not None and np.array_equal(best.x, x):
                    ctx.guarantee_hints = [g for g in ctx.guarantee_hints if g.kind != LOCAL_CONVERGED]
                    ctx.guarantee_hints.append(Guarantee(
                        LOCAL_CONVERGED,
                        f"Compass search mesh size fell below {self.params['min_step']:g} (relative) with no improving "
                        "poll step. For smooth objectives such limit points are stationary in theory; this finite run "
                        "provides no proof of stationarity or local optimality.",
                        method="compass search", tolerance=self.params["min_step"], point=x.tolist()))
                return "converged"
        return None


@register_solver
class MultiStart(Solver):
    name = "multistart"
    version = "1.0.0"
    family = "hybrid"
    description = "Repeated local searches from warm starts, then Latin-hypercube starting points, with equal budget shares."

    @classmethod
    def default_params(cls) -> dict:
        return {"local": None, "n_starts": 8, "local_params": {}}

    def incompatibilities(self, problem):
        local = self.params["local"] or default_local_for(problem)
        return get_solver(local, **self.params["local_params"]).incompatibilities(problem)

    def _run(self, ctx):
        local = self.params["local"] or default_local_for(ctx.problem)
        n = int(self.params["n_starts"])
        sp = ctx.space
        starts = list(ctx.warm_start) + list(sp.latin_hypercube(ctx.rng, max(n - len(ctx.warm_start), 0)))
        reasons = []
        for k, x0 in enumerate(starts[:n]):
            share = ctx.evaluator.remaining() // max(n - k, 1)
            if share <= 0:
                break
            reasons.append(run_stage(ctx, get_solver(local, **self.params["local_params"]), share, [x0]))
        ctx.diagnostics["multistart_stage_reasons"] = reasons
        return "completed"


@register_solver
class Hybrid(Solver):
    name = "hybrid"
    version = "1.0.0"
    family = "hybrid"
    description = ("Global exploration on a share of the budget, then local refinement from the best distinct points "
                   "found (global-search-plus-local-refinement).")

    @classmethod
    def default_params(cls) -> dict:
        return {"global_solver": None, "local": None, "global_fraction": 0.7, "n_refine": 2,
                "global_params": {}, "local_params": {}}

    def _choose(self, problem):
        g = self.params["global_solver"]
        if g is None:
            g = "cmaes" if problem.space.kind == "continuous" else ("de" if not problem.space.perm_blocks and problem.space.kind == "mixed" else "genetic")
        return g, self.params["local"] or default_local_for(problem)

    def incompatibilities(self, problem):
        g, l = self._choose(problem)
        return (get_solver(g, **self.params["global_params"]).incompatibilities(problem)
                + get_solver(l, **self.params["local_params"]).incompatibilities(problem))

    def _run(self, ctx):
        g, l = self._choose(ctx.problem)
        ev = ctx.evaluator
        if ev.trace is None:
            ev.trace = []  # needed to pick several distinct refinement starts
        total = ev.remaining()
        r1 = run_stage(ctx, get_solver(g, **self.params["global_params"]), int(total * self.params["global_fraction"]))
        # distinct good points: incumbent plus the best evaluated points far from it (from the trace if present)
        starts = [ev.best.x.copy()] if ev.best is not None else []
        if ev.trace:
            pts = sorted(ev.trace, key=lambda t: t[1])
            for x, _ in pts:
                if len(starts) >= self.params["n_refine"]:
                    break
                if all(np.linalg.norm((x - s) / ctx.space.width) > 0.1 for s in starts):
                    starts.append(x.copy())
        reasons = [r1]
        for k, x0 in enumerate(starts):
            share = ev.remaining() // max(len(starts) - k, 1)
            if share <= 0:
                break
            reasons.append(run_stage(ctx, get_solver(l, **self.params["local_params"]), share, [x0]))
        ctx.diagnostics["hybrid_stages"] = {"global": g, "local": l, "reasons": reasons}
        return "completed"
