"""Validation gates a generated candidate must pass before ordinary use.

The validation problems, budgets and pass criteria live here, outside any
candidate's reach (DSL candidates cannot perform I/O; code candidates run in a
sandbox whose filesystem view does not contain this package).

Checks
------
1. ``interface``            - static type check (DSL) or sandboxed load + call-contract test (code)
2. ``bounds_and_types``     - no out-of-domain point is ever proposed for evaluation
                              (code: raw outputs must already lie in the domain)
3. ``numerical_validity``   - no non-finite offspring; survives an objective with a NaN region
4. ``constraint_handling``  - finds a feasible point on a constrained problem with a large feasible region
5. ``reproducibility``      - identical seed gives an identical trajectory and result
6. ``termination``          - evaluation and wall-clock budgets are respected
7. ``failure_handling``     - an objective that raises on some points does not crash the run
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..core.budget import Budget, BudgetLedger
from ..core.constraints import ineq
from ..core.errors import CandidateRejected, IsolationUnavailable
from ..core.problem import Problem
from ..core.space import Binary, Categorical, Integer, Permutation, Real, Space
from ..dsl.spec import check_algorithm, check_operator, host_algorithm

VALIDATION_BUDGET = 600


@dataclass
class Check:
    name: str
    passed: bool
    details: str = ""


@dataclass
class ValidationReport:
    passed: bool
    checks: list[Check] = field(default_factory=list)
    evaluations: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks], "evaluations": self.evaluations,
                "seconds": round(self.seconds, 3)}


def validation_space(types: set) -> Space:
    vs = []
    if "real" in types:
        vs += [Real("r0", -2.0, 3.0), Real("r1", -1.0, 1.0)]
    if "integer" in types:
        vs += [Integer("i0", -3, 4)]
    if "binary" in types:
        vs += [Binary("b0"), Binary("b1"), Binary("b2")]
    if "categorical" in types:
        vs += [Categorical("c0", ["a", "b", "c"])]
    if "permutation" in types:
        vs += [Permutation("p0", 5)]
    return Space(vs)


def validation_problem(types: set, variant: str = "plain") -> Problem:
    sp = validation_space(types)
    target = sp.project(sp.lower + 0.37 * (sp.upper - sp.lower))
    perm = sp.perm_mask

    def base(x):
        z = (x - target) / sp.width
        val = float(np.sum(z[~perm] ** 2))
        if perm.any():
            val += float(np.sum(x[perm] != np.arange(perm.sum()))) / perm.sum()
        return val

    cons = []
    if variant == "constrained":
        lo, w = sp.lower[0], sp.width[0]
        cons = [ineq(lambda x: float(0.3 - (x[0] - lo) / w) if not perm[0] else float(0.5 - x[0]), "region")]
        f = base
    elif variant == "nan":
        def f(x):
            return math.nan if (x[0] - sp.lower[0]) / sp.width[0] < 0.2 else base(x)
    elif variant == "raise":
        def f(x):
            if int(abs(np.sum(x)) * 1000) % 7 == 0:
                raise ValueError("simulated evaluation failure")
            return base(x)
    elif variant == "slow":
        def f(x):
            time.sleep(0.002)
            return base(x)
    else:
        f = base
    return Problem(sp, f, constraints=cons, name=f"validation-{variant}-{'+'.join(sorted(types))}")


def _run(solver, problem, budget, seed, ledger):
    return solver.solve(problem, budget, seed=seed, ledger=ledger, ledger_category="validation")


def validate(rec: dict, types: set, make_solver, ledger: BudgetLedger | None = None) -> ValidationReport:
    """Run every gate.  ``make_solver()`` must return a fresh solver for ``rec``."""
    t0 = time.time()
    checks: list[Check] = []
    used = [0]

    def run(problem, budget, seed):
        s = make_solver()
        r = _run(s, problem, budget, seed, ledger)
        used[0] += r.n_evaluations
        return r, s

    # 1. interface ---------------------------------------------------------
    if rec["representation"] == "dsl":
        rep = (check_algorithm(rec["source"], types) if rec["kind"] == "algorithm"
               else check_algorithm(host_algorithm(rec["source"]), types))
        checks.append(Check("interface", rep.ok, "; ".join(rep.errors) or f"static type check ok {rep.info}"))
        if not rep.ok:
            return ValidationReport(False, checks, used[0], time.time() - t0)
    else:
        ok, det = _code_contract(rec, types)
        checks.append(Check("interface", ok, det))
        if not ok:
            return ValidationReport(False, checks, used[0], time.time() - t0)
    try:
        plain = validation_problem(types, "plain")
        # 2. bounds and types --------------------------------------------------
        r1, s1 = run(plain, Budget(VALIDATION_BUDGET), 11)
        dom = r1.diagnostics.get("n_domain_rejections", 0)
        raw = r1.diagnostics.get("code_raw_domain_violations", 0)
        ok = dom == 0 and raw == 0 and r1.termination_reason != "error"
        checks.append(Check("bounds_and_types", ok, f"out-of-domain evaluations={dom}, raw code outputs outside domain={raw}, "
                                                    f"termination={r1.termination_reason} {r1.messages[:1]}"))
        # 3. numerical validity -------------------------------------------------
        nonfin = r1.diagnostics.get("nonfinite_offspring", 0) + r1.diagnostics.get("code_invalid_outputs", 0)
        rn, _ = run(validation_problem(types, "nan"), Budget(VALIDATION_BUDGET), 12)
        ok = nonfin == 0 and rn.termination_reason != "error" and rn.objective is not None and math.isfinite(rn.objective)
        checks.append(Check("numerical_validity", ok, f"non-finite offspring={nonfin}; NaN-region run: termination="
                                                      f"{rn.termination_reason}, objective={rn.objective}"))
        # 4. constraint handling ------------------------------------------------
        rc, _ = run(validation_problem(types, "constrained"), Budget(VALIDATION_BUDGET), 13)
        ok = rc.feasible and rc.termination_reason != "error"
        checks.append(Check("constraint_handling", ok, f"status={rc.status}, max_violation={rc.max_violation}, "
                                                       f"termination={rc.termination_reason}"))
        # 5. reproducibility ----------------------------------------------------
        r2, _ = run(plain, Budget(VALIDATION_BUDGET), 11)
        same = r1.history == r2.history and r1.x == r2.x and r1.n_evaluations == r2.n_evaluations
        checks.append(Check("reproducibility", same, "identical history/result for repeated seed" if same else
                            "runs with the same seed diverged"))
        # 6. termination and resource limits -------------------------------------
        ok_e = r1.n_evaluations <= VALIDATION_BUDGET
        rs, _ = run(validation_problem(types, "slow"), Budget(10**6, max_time=0.3), 14)
        ok_t = rs.termination_reason == "budget_time" and rs.runtime_s <= 0.3 + 0.5
        checks.append(Check("termination", ok_e and ok_t, f"evaluations {r1.n_evaluations}/{VALIDATION_BUDGET}; time-limited run "
                                                          f"stopped after {rs.runtime_s:.3f}s (limit 0.3s, reason {rs.termination_reason})"))
        # 7. failure handling ---------------------------------------------------
        rf, _ = run(validation_problem(types, "raise"), Budget(VALIDATION_BUDGET), 15)
        ok = rf.termination_reason != "error" and rf.objective is not None
        checks.append(Check("failure_handling", ok, f"objective errors absorbed={rf.diagnostics.get('n_errors')}, "
                                                    f"termination={rf.termination_reason}"))
    except (CandidateRejected, IsolationUnavailable) as exc:
        checks.append(Check("execution", False, str(exc)))
    passed = all(c.passed for c in checks)
    return ValidationReport(passed, checks, used[0], time.time() - t0)


def _code_contract(rec: dict, types: set) -> tuple[bool, str]:
    """Load the source in the sandbox and check the call contract on random inputs (strict domain check)."""
    from ..generation.code_operators import SandboxedCodeOperator
    sp = validation_space(types)
    declared = set(rec.get("meta", {}).get("types", []))
    missing = types - declared
    if missing:
        return False, f"code operator declares no moves for variable types {sorted(missing)}"
    try:
        op = SandboxedCodeOperator(rec["id"], rec["source"], sorted(declared))
    except IsolationUnavailable as exc:
        return False, str(exc)
    rng = np.random.default_rng(0)
    bad, changed = 0, 0
    try:
        pts = list(sp.sample(rng, 30)) + [sp.project(sp.lower), sp.project(sp.upper)]
        for i, x in enumerate(pts):
            y = op.transform(x, sp, rng, 1.0 if i % 2 else 3.0)
            if not sp.contains(y):
                bad += 1
            if not np.array_equal(x, y):
                changed += 1
    except CandidateRejected as exc:
        op.close()
        return False, f"sandboxed call failed: {exc}"
    finally:
        pass
    inv = op.stats["invalid_outputs"]
    op.close()
    ok = bad == 0 and inv == 0 and changed > 0
    return ok, (f"sandboxed contract test on {len(pts)} inputs: outside-domain outputs={bad}, invalid outputs={inv}, "
                f"inputs changed={changed}")
