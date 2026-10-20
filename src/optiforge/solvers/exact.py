"""Exact methods: complete enumeration and a MILP adapter (SciPy/HiGHS).

These are the only built-in components that may emit ``global_optimum_certified``
or ``infeasibility_certified`` statements, and only under the conditions spelled
out in each statement.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ..core.errors import UnsupportedProblemError
from ..core.result import GLOBAL_CERTIFIED, GLOBAL_CONDITIONAL, INFEASIBILITY_CERTIFIED, Guarantee
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

    MIN_SPOT_CHECKS = 20

    @classmethod
    def default_params(cls) -> dict:
        return {"spot_checks": 50, "check_tol": 1e-7, "time_limit": 60.0, "mip_rel_gap": 1e-9, "certificate_tol": 1e-6}

    def incompatibilities(self, problem):
        r = super().incompatibilities(problem)
        if problem.linear is None:
            r.append("milp requires problem.linear (a declared LinearStructure)")
        if int(self.params["spot_checks"]) < self.MIN_SPOT_CHECKS:
            r.append(f"spot_checks must be at least {self.MIN_SPOT_CHECKS}")
        return r

    # ------------------------------------------------------------------
    def _model(self, p):
        """(c, offset, A_ub, b_ub, A_eq, b_eq, tol_ub, tol_eq) of the model that will be solved."""
        L = p.linear
        if p.linear_is_exact:  # use the private copies held by the generated functions
            c, offset = p.objective.M[0], -float(p.objective.b[0])
            ub = next((k for k in p.constraints if k.fn.part == "ub"), None)
            eq_ = next((k for k in p.constraints if k.fn.part == "eq"), None)
            return (c, offset, None if ub is None else ub.fn.M, None if ub is None else ub.fn.b,
                    None if eq_ is None else eq_.fn.M, None if eq_ is None else eq_.fn.b,
                    0.0 if ub is None else ub.tol, 0.0 if eq_ is None else eq_.tol)
        f = lambda a, nd: None if a is None else np.array(a, dtype=float, ndmin=nd)
        return (np.asarray(L.c, float), float(L.offset), f(L.A_ub, 2), f(L.b_ub, 1), f(L.A_eq, 2), f(L.b_eq, 1), 0.0, 0.0)

    def _agrees(self, ctx, model, x, ev) -> str | None:
        """Why the declared model disagrees with the problem's functions at x (None when it agrees)."""
        c, offset, A_ub, b_ub, A_eq, b_eq, _, _ = model
        lin_f = float(c @ x + offset)
        if abs(lin_f - ev.f) > self.params["check_tol"] * (1 + abs(ev.f)):
            return f"declared linear objective {lin_f} != objective {ev.f}"
        tol = max([k.tol for k in ctx.problem.constraints], default=0.0)
        margin = self.params["check_tol"] * (1 + float(np.max(np.abs(x)) if x.size else 0.0))
        lin_viol = 0.0
        if A_ub is not None:
            lin_viol = max(lin_viol, float(np.max(A_ub @ x - b_ub, initial=-np.inf)))
        if A_eq is not None:
            lin_viol = max(lin_viol, float(np.max(np.abs(A_eq @ x - b_eq), initial=0.0)))
        # points within a margin of the tolerance boundary are inconclusive and skipped
        if abs(lin_viol - tol) <= margin:
            return None
        if (lin_viol <= tol) != bool(ev.feasible):
            return ("declared linear constraints disagree with the constraint functions "
                    f"(declared violation {lin_viol:.3g}, functions feasible={ev.feasible})")
        return None

    def _spot_check(self, ctx, model) -> None:
        p = ctx.problem
        corners = [p.space.project(p.space.lower.copy()), p.space.project(p.space.upper.copy())]
        for x in corners + list(p.space.sample(ctx.rng, int(self.params["spot_checks"]))):
            ev = ctx.evaluate(x)
            if not ev.valid:
                raise UnsupportedProblemError(f"spot check evaluation failed: {ev.error}")
            why = self._agrees(ctx, model, x, ev)
            if why:
                raise UnsupportedProblemError(why + " at a sample point")

    def _highs(self, ctx, cmin, model, relax: float, tighten: float):
        """Solve min cmin@x over the box with every constraint shifted outward by ``relax`` x its tolerance and
        inward by ``tighten`` (relative to 1 + |rhs|)."""
        sp = ctx.problem.space
        _, _, A_ub, b_ub, A_eq, b_eq, tol_ub, tol_eq = model
        cons = []
        if A_ub is not None:
            cons.append(optimize.LinearConstraint(A_ub, -np.inf, b_ub + relax * tol_ub - tighten * (1 + np.abs(b_ub))))
        if A_eq is not None:
            w = relax * tol_eq - tighten * (1 + np.abs(b_eq))
            lo, hi = b_eq - w, b_eq + w
            if np.any(lo > hi):  # an equality cannot be tightened inward
                lo, hi = b_eq.copy(), b_eq.copy()
            cons.append(optimize.LinearConstraint(A_eq, lo, hi))
        time_limit = float(self.params["time_limit"])
        if ctx.evaluator.budget.max_time is not None:
            time_limit = max(0.01, min(time_limit, ctx.evaluator.budget.max_time - ctx.evaluator.elapsed()))
        ctx.diagnostics["highs_time_limit"] = time_limit
        res = optimize.milp(cmin, integrality=np.where(sp.real_mask, 0, 1), bounds=optimize.Bounds(sp.lower, sp.upper),
                            constraints=cons, options={"time_limit": time_limit, "mip_rel_gap": self.params["mip_rel_gap"],
                                                       "disp": False})
        ctx.diagnostics.setdefault("highs_solves", []).append(
            {"relax": relax, "tighten": tighten, "status": int(res.status), "message": str(res.message),
             "fun": None if res.fun is None else float(res.fun), "mip_gap": _f(getattr(res, "mip_gap", None)),
             "dual_bound": _f(getattr(res, "mip_dual_bound", None))})
        ctx.diagnostics["highs_status"] = int(res.status)
        ctx.diagnostics["highs_message"] = str(res.message)
        return res

    def _run(self, ctx):
        p, sp = ctx.problem, ctx.problem.space
        exact = p.linear_is_exact
        model = self._model(p)
        c, offset = model[0], model[1]
        ctx.diagnostics["linear_model_exact"] = exact
        sgn = 1 if p.sense == "min" else -1
        cmin = c * sgn
        if exact:
            return self._run_exact(ctx, cmin, model, sgn * offset)
        self._spot_check(ctx, model)  # functions written by the user: compare them with the declaration first
        res = self._highs(ctx, cmin, model, 0.0, 0.0)
        if res.status == 1:
            return "budget_time" if res.x is None else self._timed_out(ctx, sp, res)
        if res.status == 2:
            ctx.messages.append(
                "HiGHS reported the declared linear model infeasible. The problem's own functions were only "
                "spot-checked against the declaration, so infeasibility is NOT certified (build the problem with "
                "Problem.from_linear for a certificate).")
            return "exact_solver_finished"
        if res.x is None:
            ctx.messages.append(f"MILP solver returned no solution: {res.message}")
            return "exact_solver_finished"
        x = sp.project(res.x)
        ev = ctx.evaluate(x)
        if res.status != 0:
            ctx.messages.append(f"HiGHS did not prove optimality ({res.message}); no optimality statement is made")
            return "exact_solver_finished"
        if not ev.valid or not ev.feasible:
            ctx.messages.append("the MILP solution does not evaluate as feasible on the problem's functions")
            return "exact_solver_finished"
        why = self._agrees(ctx, model, x, ev)
        if why:
            ctx.messages.append(f"{why} at the MILP solution; the declaration is wrong, no optimality statement is made")
            return "exact_solver_finished"
        gap = getattr(res, "mip_gap", None)
        ctx.guarantee_hints.append(Guarantee(
            GLOBAL_CONDITIONAL,
            f"HiGHS proved optimality of the DECLARED linear model (relative MIP gap {0.0 if gap is None else gap:.2g} "
            f"<= {self.params['mip_rel_gap']:g}). The problem's own functions agreed with the declaration at the two "
            f"box corners, {int(self.params['spot_checks'])} random points and the solution, but agreement elsewhere "
            "is not proven; the point is globally optimal only if the declaration is exact.",
            method="HiGHS branch-and-bound via scipy.optimize.milp", tolerance=self.params["mip_rel_gap"],
            assumptions=["declared linear structure is exact (spot-checked, not proven)",
                         "HiGHS feasibility/integrality tolerances (default 1e-7 / 1e-6)"],
            point=x.tolist()))
        return "exact_solver_finished"

    def _run_exact(self, ctx, cmin, model, int_offset):
        """Problem.from_linear: the model is the problem, so HiGHS can certify optimality or infeasibility."""
        sp = ctx.problem.space
        # 1. the problem as defined: every constraint relaxed by its declared tolerance
        resR = self._highs(ctx, cmin, model, 1.0, 0.0)
        if resR.status == 2:
            ctx.guarantee_hints.append(Guarantee(
                INFEASIBILITY_CERTIFIED,
                "HiGHS reported the linear model infeasible with every constraint relaxed by its declared tolerance. "
                "The problem's functions are generated from that model (Problem.from_linear), so the statement is about "
                "the problem as defined.",
                method="HiGHS via scipy.optimize.milp",
                assumptions=["HiGHS feasibility/integrality tolerances (default 1e-7 / 1e-6)"]))
            return "exact_solver_finished"
        if resR.status == 1:
            return "budget_time" if resR.x is None else self._timed_out(ctx, sp, resR)
        if resR.status != 0 or resR.x is None:
            ctx.messages.append(f"HiGHS did not solve the model ({resR.message}); no statement is made")
            return "exact_solver_finished"
        bound = getattr(resR, "mip_dual_bound", None)
        bound = float(resR.fun if bound is None or not np.isfinite(bound) else min(bound, resR.fun))
        lower = bound + int_offset  # lower bound on the internal (minimisation) objective over the feasible set
        gap = getattr(resR, "mip_gap", None)
        # 2. a point that verifies: the relaxed optimum itself, else the strict model, tightened inward if HiGHS's own
        #    tolerances leave its point slightly outside the declared ones
        cert = None
        ev = ctx.evaluate(sp.project(resR.x))
        if ev.valid and ev.feasible:
            cert = ev
        else:
            for tighten in (0.0, 1e-7, 1e-6):
                r = self._highs(ctx, cmin, model, 0.0, tighten)
                if r.status == 1:
                    return "budget_time" if r.x is None else self._timed_out(ctx, sp, r)
                if r.status != 0 or r.x is None:
                    break
                ev = ctx.evaluate(sp.project(r.x))
                if ev.valid and ev.feasible:
                    cert = ev
                    break
        if cert is None:
            ctx.messages.append("no MILP solution verified as feasible within the declared tolerances; "
                                "no optimality statement is made")
            return "exact_solver_finished"
        best = ctx.evaluator.best
        if best is not None and best.feasible and best.fmin <= cert.fmin:
            cert = best  # ties and earlier equal points: certify the incumbent that will be reported
        slack = max(0.0, cert.fmin - lower)
        ctx.diagnostics["optimality_gap_vs_bound"] = slack
        allowed = float(self.params["certificate_tol"]) * (1 + abs(cert.fmin))
        if slack > allowed:
            ctx.messages.append(f"the verified point is {slack:.3g} worse than HiGHS's lower bound for the problem as "
                                f"defined (allowed {allowed:.3g}); no optimality statement is made")
            return "exact_solver_finished"
        ctx.guarantee_hints.append(Guarantee(
            GLOBAL_CERTIFIED,
            f"Global optimum of the problem as defined up to an absolute objective tolerance of {allowed:.3g}: the point "
            f"is feasible on re-evaluation and is within {slack:.3g} of HiGHS's lower bound over all points that satisfy "
            f"the constraints within their declared tolerances (relative MIP gap "
            f"{0.0 if gap is None else gap:.2g} <= {self.params['mip_rel_gap']:g}). The problem's functions are "
            "generated from the model (Problem.from_linear).",
            method="HiGHS branch-and-bound via scipy.optimize.milp", tolerance=allowed,
            assumptions=["HiGHS feasibility/integrality tolerances (default 1e-7 / 1e-6)"],
            point=cert.x.tolist()))
        return "exact_solver_finished"

    def _timed_out(self, ctx, sp, res):
        ctx.evaluate(sp.project(res.x))
        ctx.messages.append(f"HiGHS stopped at its time limit ({res.message}); no optimality statement is made")
        return "budget_time"


def _f(v):
    return None if v is None else float(v)
