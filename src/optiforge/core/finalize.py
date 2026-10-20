"""Turning a finished search into an honest :class:`OptimizationResult`.

Rules enforced here (independent of the solver that ran):

1. The reported point is re-evaluated with a fresh evaluator against the
   original objective and constraints; the reported objective, feasibility
   and violations come from that verification, never from a solver's
   internal (possibly penalised) fitness.
2. Optimality claims from solvers are *hints* and are only kept when they are
   consistent with the verification (a "certified optimum" that verifies as
   infeasible is dropped and reported).
3. Without a certificate the result says "best feasible found" and makes no
   optimality claim.  Failing to find a feasible point is never reported as
   infeasibility unless a complete method certified it.
"""
from __future__ import annotations

import math

import numpy as np

from .budget import Budget
from .context import SearchContext
from .evaluator import Evaluator
from .result import (FEASIBLE_SAMPLED, FEASIBLE_VERIFIED, GLOBAL_CERTIFIED, GLOBAL_CONDITIONAL, HEURISTIC,
                     INFEASIBILITY_CERTIFIED, NO_FEASIBLE_FOUND, Guarantee, OptimizationResult)

NOISY_VERIFICATION_REPEATS = 5


def finalize(ctx: SearchContext, solver_name: str, solver_version: str, termination: str,
             extra_messages: list[str] | None = None) -> OptimizationResult:
    p = ctx.problem
    ev = ctx.evaluator
    best = ev.best
    messages = list(ctx.messages) + list(extra_messages or [])
    assumptions = _problem_assumptions(p)
    guarantees: list[Guarantee] = []
    noise = None
    n_verif = 0
    hints = list(ctx.guarantee_hints)

    if best is None or not best.valid:
        infeas = [g for g in hints if g.kind == INFEASIBILITY_CERTIFIED]
        if infeas:
            guarantees += infeas
            status = "infeasible_certified"
        else:
            guarantees.append(Guarantee(
                NO_FEASIBLE_FOUND,
                "No valid evaluation was obtained within the budget. This is not evidence that the problem is infeasible.",
            ))
            st = ev.stats
            failed = st.n_errors + st.n_nonfinite
            if failed > 0 and failed >= ev.n_evals:
                # every evaluation that ran failed: an execution failure, not an unsuccessful search
                status = "error"
                messages.append(f"execution failure: all {ev.n_evals} evaluations failed "
                                f"({st.n_errors} raised, {st.n_nonfinite} non-finite)")
            elif termination == "error":
                status = "error"
                messages.append(f"execution failure: the solver stopped with an error after {ev.n_evals} evaluations")
            elif ev.n_evals == 0:
                status = "error"
                messages.append("no evaluation was performed")
            else:
                status = "no_feasible_found"
        return OptimizationResult(
            problem=p.name, solver=solver_name, solver_version=solver_version, status=status, x=None,
            x_decoded=None, objective=None, feasible=False, constraint_violations={}, max_violation=math.inf,
            n_evaluations=ev.n_evals, n_gradient_evaluations=ev.stats.n_gradient_evaluations,
            n_verification_evaluations=0, runtime_s=ev.elapsed(), seed=ctx.seed, termination_reason=termination,
            history=[list(h) for h in ev.history], guarantees=guarantees, assumptions=assumptions,
            operator_versions=dict(ctx.operator_versions), sense=p.sense, diagnostics=_diag(ctx),
            messages=messages,
        )

    # -- verification against the original problem ---------------------------
    verifier = Evaluator(p, Budget(max_evaluations=10**9))
    reps = NOISY_VERIFICATION_REPEATS if p.noisy else 1
    vals, last, repeats = [], None, []
    for _ in range(reps):
        last = verifier.evaluate(best.x)
        n_verif += 1
        repeats.append(last)
        if last.valid:
            vals.append(last.f)
    verification_failed = False
    if last is None or not last.valid or not vals:
        messages.append(f"verification evaluation failed: {None if last is None else last.error}")
        feasible, objective, viol = False, None, {}
        max_v = math.inf
        verification_failed = True
    else:
        objective = float(np.mean(vals))
        # noisy problems: the point counts as feasible only if every repeat was valid and feasible; the violation
        # report uses the worst repeat (a single lucky repeat must not decide feasibility)
        feasible = all(r.valid and r.feasible for r in repeats)
        if p.noisy and len(repeats) > 1:
            n_feas = sum(bool(r.valid and r.feasible) for r in repeats)
            if not feasible:
                messages.append(f"noisy verification: the point was feasible in {n_feas} of {len(repeats)} repeats, "
                                "so it is reported as infeasible")
            last = max((r for r in repeats if r.valid), key=lambda r: r.total_violation)
        try:
            viol = _named_violations(p, best.x, last.raw_constraints)
            max_v = float(max(viol.values())) if viol else 0.0
        except Exception as exc:  # a constraint that raises here must not lose the result
            messages.append(f"per-constraint violation report failed: {type(exc).__name__}: {exc}")
            viol, max_v = {}, (0.0 if feasible else math.inf)
    if p.noisy and vals:
        noise = {"verification_repeats": reps, "feasible_repeats": sum(bool(r.valid and r.feasible) for r in repeats), "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                 "search_value_of_reported_point": best.f,
                 "note": "objective is noisy; reported value is the mean of repeated verification evaluations"}
    nondeterministic = False
    if not p.noisy and objective is not None and best.valid and (
            not math.isclose(objective, best.f, rel_tol=1e-9, abs_tol=1e-12) or bool(last.feasible) != bool(best.feasible)):
        nondeterministic = True
        messages.append(
            f"objective or feasibility at reported point changed between search ({best.f}, feasible={best.feasible}) "
            f"and verification ({objective}, feasible={last.feasible}); the function may be non-deterministic despite "
            "being declared deterministic, so no certificate is kept"
        )

    # -- optional independent verifier (e.g. dense re-simulation) ------------
    verifier_report = None
    dense_rejected = False
    verifier_failed = False
    if p.verifier is not None and objective is not None:
        try:
            verifier_report = dict(p.verifier(best.x))
        except Exception as exc:
            verifier_report = {"feasible": None, "scope": "verifier_error", "error": repr(exc)}
            verifier_failed = True
            messages.append(f"independent verifier raised {type(exc).__name__}: {exc}; its check is missing, so no "
                            "global certificate is kept")
        if feasible and not verifier_failed and not verifier_report.get("feasible", False):
            messages.append("independent verifier rejected the reported point: " + str(verifier_report))
            dense_rejected = True

    # -- reconcile solver hints with the verification -------------------------
    certified = False
    for g in hints:
        if g.point is not None and not np.allclose(np.asarray(g.point, dtype=float), best.x, rtol=0, atol=1e-12):
            messages.append(f"dropped guarantee '{g.kind}': it refers to a point other than the reported one")
            continue
        if g.kind in (GLOBAL_CERTIFIED, GLOBAL_CONDITIONAL, INFEASIBILITY_CERTIFIED) and nondeterministic:
            messages.append(f"dropped guarantee '{g.kind}': the problem did not behave deterministically")
            continue
        if g.kind in (GLOBAL_CERTIFIED, GLOBAL_CONDITIONAL):
            if not feasible:
                messages.append(f"dropped guarantee '{g.kind}': reported point does not verify as feasible")
                continue
            if p.noisy:
                messages.append(f"dropped guarantee '{g.kind}': objective is noisy")
                continue
            if dense_rejected or verifier_failed:
                messages.append(f"dropped guarantee '{g.kind}': the independent verifier "
                                f"{'raised' if verifier_failed else 'rejected the reported point'}")
                continue
            certified = certified or g.kind == GLOBAL_CERTIFIED
        guarantees.append(g)
    if feasible and any(g.kind == INFEASIBILITY_CERTIFIED for g in guarantees):
        messages.append("dropped guarantee 'infeasibility_certified': the reported point verifies as feasible")
        guarantees = [g for g in guarantees if g.kind != INFEASIBILITY_CERTIFIED]
    if feasible:
        scope = (None if verifier_failed else (verifier_report or {}).get("scope")) or p.metadata.get("feasibility_scope", "exact")
        if dense_rejected:
            guarantees.append(Guarantee(
                FEASIBLE_SAMPLED,
                "Feasible for the problem as defined (constraints evaluated on the simulation's sample grid), but the "
                "independent re-simulation on a denser grid with tighter tolerances found constraint violations. "
                "Feasibility of the underlying continuous-time system is NOT established.",
                method=str(scope), assumptions=[str(verifier_report)]))
        elif scope == "sampled" or str(scope).startswith("sampled"):
            guarantees.append(Guarantee(
                FEASIBLE_SAMPLED,
                "Constraints hold at the simulated/sample points that were checked; satisfaction between "
                "samples or in the underlying continuous-time system is not proven.",
                method=str(scope), assumptions=[str(verifier_report)] if verifier_report else []))
        else:
            guarantees.append(Guarantee(
                FEASIBLE_VERIFIED,
                "The reported point satisfies every original constraint within its stated tolerance at verification.",
                method="re-evaluation of original constraints",
                tolerance=max([c.tol for c in p.constraints], default=None)))
        if not certified and not any(g.kind == GLOBAL_CONDITIONAL for g in guarantees):
            guarantees.append(Guarantee(
                HEURISTIC,
                "Best feasible solution found within the budget. No global (or proven local) optimality is claimed.",
            ))
        status = "optimal_certified" if certified else "feasible"
        if dense_rejected:
            status = "feasible_discretized_only"
    else:
        guarantees = [g for g in guarantees if g.kind not in (GLOBAL_CERTIFIED, GLOBAL_CONDITIONAL)]
        if verification_failed and any(g.kind == INFEASIBILITY_CERTIFIED for g in guarantees):
            # the problem changed behaviour between search and verification: the enumeration is not trustworthy
            messages.append("dropped guarantee 'infeasibility_certified': the verification evaluation failed")
            guarantees = [g for g in guarantees if g.kind != INFEASIBILITY_CERTIFIED]
        if any(g.kind == INFEASIBILITY_CERTIFIED for g in guarantees) and not p.noisy:
            status = "infeasible_certified"
        else:
            guarantees = [g for g in guarantees if g.kind != INFEASIBILITY_CERTIFIED]
            if verification_failed:
                guarantees.append(Guarantee(
                    NO_FEASIBLE_FOUND,
                    "The verification evaluation of the reported point failed, so neither its objective nor its "
                    "feasibility could be confirmed. This is not evidence that the problem is infeasible.",
                ))
                status = "error"
            else:
                guarantees.append(Guarantee(
                    NO_FEASIBLE_FOUND,
                    "No feasible solution was found; the reported point minimises constraint violation among the points "
                    "evaluated. This is not evidence that the problem is infeasible.",
                ))
                status = "no_feasible_found"

    diag = _diag(ctx)
    if verifier_report is not None:
        diag["verifier"] = verifier_report
    return OptimizationResult(
        problem=p.name, solver=solver_name, solver_version=solver_version, status=status,
        x=best.x.tolist(), x_decoded=p.space.decode(best.x), objective=objective, feasible=feasible,
        constraint_violations=viol, max_violation=max_v, n_evaluations=ev.n_evals,
        n_gradient_evaluations=ev.stats.n_gradient_evaluations, n_verification_evaluations=n_verif,
        runtime_s=ev.elapsed(), seed=ctx.seed, termination_reason=termination,
        history=[list(h) for h in ev.history], guarantees=guarantees, assumptions=assumptions,
        operator_versions=dict(ctx.operator_versions), sense=p.sense, noise=noise, diagnostics=diag,
        messages=messages,
    )


def _named_violations(p, x: np.ndarray, raw: list | None = None) -> dict[str, float]:
    """Per-constraint violation amounts from the verification evaluation (recomputed only if it kept no values)."""
    out: dict[str, float] = {}
    if raw is None or len(raw) != len(p.constraints):
        raw = [p.call_constraint(c, x) for c in p.constraints]
    for c, r in zip(p.constraints, raw):
        v = c.violation(r)
        if v.size == 1:
            out[c.name] = float(v[0])
        else:
            for k, vk in enumerate(v):
                out[f"{c.name}[{k}]"] = float(vk)
    return out


def _problem_assumptions(p) -> list[str]:
    a = []
    a.append("objective is noisy; values are estimates" if p.noisy else "objective declared deterministic")
    for c in p.constraints:
        a.append(f"constraint '{c.name}' ({c.kind}) satisfied when {'g(x) <= ' if c.kind == 'ineq' else '|h(x)| <= '}{c.tol:g}")
    if p.convex:
        a.append("problem declared convex by the user (not verified by the platform)")
    if p.evaluation == "simulation":
        a.append("objective/constraints computed by numerical simulation; accuracy limited by the integrator/discretisation")
    return a


def _diag(ctx: SearchContext) -> dict:
    s = ctx.evaluator.stats
    d = dict(ctx.diagnostics)
    d.update(n_errors=s.n_errors, n_nonfinite=s.n_nonfinite, n_domain_rejections=s.n_domain_rejections,
             n_feasible_evaluations=s.n_feasible)
    return d
