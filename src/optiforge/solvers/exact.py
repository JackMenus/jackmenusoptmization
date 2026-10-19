"""Exact methods: complete enumeration and a MILP adapter (SciPy/HiGHS).

These are the only built-in components that may emit ``global_optimum_certified``
or ``infeasibility_certified`` statements, and only under the conditions spelled
out in each statement.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ..core.errors import UnsupportedProblemError
from ..core.result import GLOBAL_CERTIFIED, INFEASIBILITY_CERTIFIED, Guarantee
from .base import Solver, register_solver


@register_solver
class ExhaustiveEnumeration(Solver):
    name = "exhaustive"
    version = "1.0.0"
    family = "exact"
    space_kinds = frozenset({"discrete"})
    supports_noise = False
    description = "Complete enumeration of a finite domain; certifies the optimum (or infeasibility) when it finishes."

    @classmethod
    def default_params(cls) -> dict:
        return {"max_points": 2_000_000}

    def incompatibilities(self, problem):
        r = super().incompatibilities(problem)
        card = problem.space.cardinality()
        if card is not None and card > self.params["max_points"]:
            r.append(f"domain has {card} points > max_points={self.params['max_points']}")
        return r

    def _run(self, ctx):
        p = ctx.problem
        card = p.space.cardinality()
        if ctx.evaluator.remaining() < card:
            ctx.messages.append(f"evaluation budget ({ctx.evaluator.remaining()}) is smaller than the domain ({card}); "
                                "enumeration will be incomplete and no certificate will be issued")
        for x in p.space.enumerate():
            ctx.evaluate(x)
        s = ctx.evaluator.stats
        tols = sorted({c.tol for c in p.constraints})
        if s.n_errors or s.n_nonfinite:
            ctx.messages.append(f"{s.n_errors + s.n_nonfinite} points failed to evaluate; no certificate issued")
            return "search_space_exhausted"
        best = ctx.evaluator.best
        if best is not None and best.feasible:
            ctx.guarantee_hints.append(Guarantee(
                GLOBAL_CERTIFIED,
                f"Global optimum over the finite domain: all {card} points were evaluated exactly and this point is "
                "the best feasible one (ties broken by enumeration order).",
                method="complete enumeration", tolerance=tols[-1] if tols else 0.0,
                assumptions=["objective and constraints deterministic", "constraint tolerances as declared"],
                point=best.x.tolist()))
        else:
            ctx.guarantee_hints.append(Guarantee(
                INFEASIBILITY_CERTIFIED,
                f"No point of the finite domain ({card} points, all evaluated) satisfies the constraints within the "
                "declared tolerances.", method="complete enumeration",
                assumptions=["objective and constraints deterministic"]))
        return "search_space_exhausted"


@register_solver
class MILPSolver(Solver):
    name = "milp"
    version = "1.0.0"
    family = "exact"
    space_kinds = frozenset({"continuous", "discrete", "mixed"})
    supports_permutations = False
    supports_categorical = False
    supports_noise = False
    description = ("Adapter to scipy.optimize.milp (HiGHS) for problems that declare a LinearStructure. The declaration "
                   "is spot-checked against the actual objective/constraint functions before solving.")

    @classmethod
    def default_params(cls) -> dict:
        return {"spot_checks": 20, "check_tol": 1e-7, "time_limit": 60.0, "mip_rel_gap": 1e-9}

    def incompatibilities(self, problem):
        r = super().incompatibilities(problem)
        if problem.linear is None:
            r.append("milp requires problem.linear (a declared LinearStructure)")
        return r

    def _spot_check(self, ctx) -> None:
        p, L = ctx.problem, ctx.problem.linear
        c = np.asarray(L.c, dtype=float)
        for x in p.space.sample(ctx.rng, self.params["spot_checks"]):
            ev = ctx.evaluate(x)
            if not ev.valid:
                raise UnsupportedProblemError(f"spot check evaluation failed: {ev.error}")
            lin_f = float(c @ x + L.offset)
            if abs(lin_f - ev.f) > self.params["check_tol"] * (1 + abs(ev.f)):
                raise UnsupportedProblemError(f"declared linear objective {lin_f} != objective {ev.f} at a sample point")
            lin_feas = True
            if L.A_ub is not None:
                lin_feas &= bool(np.all(np.asarray(L.A_ub) @ x <= np.asarray(L.b_ub) + 1e-9))
            if L.A_eq is not None:
                lin_feas &= bool(np.all(np.abs(np.asarray(L.A_eq) @ x - np.asarray(L.b_eq)) <= 1e-9))
            if p.has_constraints and lin_feas != ev.feasible:
                raise UnsupportedProblemError("declared linear constraints disagree with the constraint functions at a sample point")

    def _run(self, ctx):
        p, L, sp = ctx.problem, ctx.problem.linear, ctx.problem.space
        self._spot_check(ctx)
        c = np.asarray(L.c, dtype=float) * (1 if p.sense == "min" else -1)
        integrality = np.where(sp.real_mask, 0, 1)
        cons = []
        if L.A_ub is not None:
            cons.append(optimize.LinearConstraint(np.asarray(L.A_ub, float), -np.inf, np.asarray(L.b_ub, float)))
        if L.A_eq is not None:
            beq = np.asarray(L.b_eq, float)
            cons.append(optimize.LinearConstraint(np.asarray(L.A_eq, float), beq, beq))
        res = optimize.milp(c, integrality=integrality, bounds=optimize.Bounds(sp.lower, sp.upper), constraints=cons,
                            options={"time_limit": self.params["time_limit"], "mip_rel_gap": self.params["mip_rel_gap"],
                                     "disp": False})
        ctx.diagnostics["highs_status"] = int(res.status)
        ctx.diagnostics["highs_message"] = str(res.message)
        gap = getattr(res, "mip_gap", None)
        ctx.diagnostics["mip_gap"] = None if gap is None else float(gap)
        if res.status == 2:
            ctx.guarantee_hints.append(Guarantee(
                INFEASIBILITY_CERTIFIED, "HiGHS reported the declared MILP infeasible (subject to its feasibility "
                "tolerances, default 1e-7) and the declaration agreed with the constraint functions at the spot checks.",
                method="HiGHS via scipy.optimize.milp",
                assumptions=["declared linear structure is exact (spot-checked, not proven)"]))
            return "exact_solver_finished"
        if res.x is None:
            ctx.messages.append(f"MILP solver returned no solution: {res.message}")
            return "exact_solver_finished"
        x = sp.project(res.x)
        ev = ctx.evaluate(x)
        if res.status == 0 and ev.feasible:
            ctx.guarantee_hints.append(Guarantee(
                GLOBAL_CERTIFIED,
                f"HiGHS reported optimality (relative MIP gap {0.0 if gap is None else gap:.2g} <= "
                f"{self.params['mip_rel_gap']:g}); the solution was re-evaluated on the original functions.",
                method="HiGHS branch-and-bound via scipy.optimize.milp", tolerance=self.params["mip_rel_gap"],
                assumptions=["declared linear structure is exact (spot-checked at random points, not proven)",
                             "HiGHS default feasibility/integrality tolerances"],
                point=x.tolist()))
        return "exact_solver_finished"
