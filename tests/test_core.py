"""Space, problem validation, evaluator, budgets and result honesty."""
import math

import numpy as np
import pytest

from optiforge import Binary, Budget, Categorical, Integer, Permutation, Problem, Real, Space
from optiforge.core.budget import BudgetLedger, CancelToken
from optiforge.core.constraints import eq, ineq
from optiforge.core.errors import BudgetExhausted, ProblemDefinitionError, UnsupportedProblemError
from optiforge.core.evaluator import Evaluator
from optiforge.core.result import GLOBAL_CERTIFIED, HEURISTIC, NO_FEASIBLE_FOUND
from optiforge.solvers import get_solver

from conftest import sphere_problem


def mixed_space():
    return Space([Real("r", -1, 1), Integer("i", 0, 4), Binary("b"), Categorical("c", ["a", "b", "c"]),
                  Permutation("p", 4)])


def test_space_layout_and_kind():
    sp = mixed_space()
    assert sp.dim == 1 + 1 + 1 + 1 + 4
    assert sp.kind == "mixed"
    assert sp.types_present == {"real", "integer", "binary", "categorical", "permutation"}
    assert Space([Integer("i", 0, 3), Binary("b")]).cardinality() == 8
    assert sp.cardinality() is None  # real variable -> infinite


def test_space_sample_project_contains():
    sp = mixed_space()
    rng = np.random.default_rng(0)
    X = sp.sample(rng, 50)
    assert all(sp.contains(x) for x in X)
    bad = X[0].copy()
    bad[1] = 2.4  # non-integer integer
    bad[0] = 3.0  # out of bounds
    bad[4:8] = [0, 0, 1, 2]  # not a permutation
    assert not sp.contains(bad)
    assert len(sp.domain_violations(bad)) >= 3
    assert sp.contains(sp.project(bad))


def test_decode_encode_roundtrip():
    sp = mixed_space()
    x = sp.sample(np.random.default_rng(1))
    d = sp.decode(x)
    assert d["c"] in ("a", "b", "c")
    assert sorted(d["p"]) == [0, 1, 2, 3]
    assert np.allclose(sp.encode(d), x)


def test_enumerate_covers_cardinality():
    sp = Space([Integer("i", 0, 2), Binary("b"), Permutation("p", 3)])
    pts = list(sp.enumerate())
    assert len(pts) == sp.cardinality() == 3 * 2 * 6
    assert len({tuple(p) for p in pts}) == len(pts)


@pytest.mark.parametrize("bad", [
    dict(sense="maximize"),
    dict(evaluation="quantum"),
    dict(n_objectives=2),
])
def test_problem_rejects_bad_config(bad):
    sp = Space([Real("x", 0, 1)])
    with pytest.raises((ProblemDefinitionError, UnsupportedProblemError)):
        Problem(sp, lambda x: 0.0, **bad)


def test_problem_rejects_gradient_on_discrete_and_bad_bounds():
    with pytest.raises(UnsupportedProblemError):
        Problem(Space([Integer("i", 0, 3)]), lambda x: 0.0, gradient=lambda x: x)
    with pytest.raises(UnsupportedProblemError):
        Problem(Space([Integer("i", 0, 3)]), lambda x: 0.0, convex=True)
    with pytest.raises(ProblemDefinitionError):
        Real("x", 1.0, 0.0)
    with pytest.raises(ProblemDefinitionError):
        Problem(Space([Real("x", 0, 1)]), lambda x: 0.0,
                constraints=[ineq(lambda x: x[0], name="a"), ineq(lambda x: x[0], name="a")])


def test_solver_rejects_incompatible_problem():
    p = Problem(Space([Permutation("p", 4)]), lambda x: 0.0)
    with pytest.raises(UnsupportedProblemError):
        get_solver("lbfgsb").solve(p, Budget(100))


def test_evaluator_captures_failures_and_domain():
    sp = Space([Real("x", -1, 1)])

    def f(x):
        if x[0] > 0.5:
            raise RuntimeError("boom")
        if x[0] < -0.5:
            return math.nan
        return float(x[0] ** 2)

    ev = Evaluator(Problem(sp, f), Budget(10))
    assert ev.evaluate([0.9]).error.startswith("objective raised")
    assert ev.evaluate([-0.9]).error == "objective non-finite"
    assert ev.evaluate([5.0]).error.startswith("domain")
    good = ev.evaluate([0.1])
    assert good.feasible and good.valid and abs(good.f - 0.01) < 1e-12
    assert ev.stats.n_errors == 1 and ev.stats.n_nonfinite == 1 and ev.stats.n_domain_rejections == 1
    assert ev.best is good


def test_evaluator_budget_and_cancel():
    p = sphere_problem()
    ev = Evaluator(p, Budget(2))
    ev.evaluate(np.zeros(3))
    ev.evaluate(np.zeros(3))
    with pytest.raises(BudgetExhausted):
        ev.evaluate(np.zeros(3))
    tok = CancelToken()
    r = get_solver("de").solve(p, Budget(10_000), seed=0, cancel=tok,
                               callback=lambda ctx: tok.cancel() if ctx.evaluator.n_evals > 200 else None)
    assert r.termination_reason == "cancelled"
    assert r.n_evaluations < 10_000


def test_maximisation_sense():
    sp = Space([Real("x", 0, 2)])
    p = Problem(sp, lambda x: float(-(x[0] - 1.5) ** 2 + 3), sense="max")
    r = get_solver("nelder_mead").solve(p, Budget(500), seed=0)
    assert r.feasible and abs(r.objective - 3.0) < 1e-5
    assert r.history[-1][1] == pytest.approx(r.objective, abs=1e-6)


def test_ledger_categories_and_reserve():
    led = BudgetLedger(total=100, limits={"test": 10})
    led.charge("train", 50)
    with pytest.raises(BudgetExhausted):
        led.charge("test", 11)
    led.reserve("bench", 30)
    assert led.remaining() == 20
    led.settle("bench", 30, 12)
    assert led.used["bench"] == 12 and led.remaining() == 38


def test_result_reevaluated_and_honest():
    """Reported objective comes from a fresh evaluation of the original problem, and heuristic
    results never claim global optimality."""
    p = sphere_problem(dim=4)
    r = get_solver("de").solve(p, Budget(2000), seed=3)
    assert r.n_verification_evaluations >= 1
    assert r.objective == pytest.approx(float(np.sum(np.asarray(r.x) ** 2)))
    kinds = {g.kind for g in r.guarantees}
    assert GLOBAL_CERTIFIED not in kinds and HEURISTIC in kinds
    assert set(r.to_dict()) >= {"x", "objective", "feasible", "constraint_violations", "n_evaluations", "runtime_s",
                               "seed", "solver_version", "history", "termination_reason", "guarantees", "assumptions"}


def test_no_feasible_found_is_not_infeasibility_proof():
    sp = Space([Real("x", 0, 1), Real("y", 0, 1)])
    p = Problem(sp, lambda x: 0.0, constraints=[ineq(lambda x: 2.5 - x[0] - x[1], name="impossible")])
    r = get_solver("random_search").solve(p, Budget(200), seed=0)
    assert not r.feasible and r.status == "no_feasible_found"
    g = [g for g in r.guarantees if g.kind == NO_FEASIBLE_FOUND]
    assert g and "not evidence that the problem is infeasible" in g[0].statement
    assert r.constraint_violations["impossible"] > 0


def test_penalty_solver_reports_original_violation():
    """Nelder-Mead optimises a penalised surrogate; the report must use the original constraint."""
    sp = Space([Real("x", -3, 3), Real("y", -3, 3)])
    p = Problem(sp, lambda x: float(x[0] + x[1]), constraints=[eq(lambda x: x[0] ** 2 + x[1] ** 2 - 1, name="circle", tol=1e-6)])
    r = get_solver("nelder_mead").solve(p, Budget(3000), seed=0)
    v = abs(float(r.x[0]) ** 2 + float(r.x[1]) ** 2 - 1)
    assert r.feasible == (v <= 1e-6)
    assert r.constraint_violations["circle"] == pytest.approx(max(v - 1e-6, 0.0), abs=1e-9)


def test_problem_repair_does_not_redefine_problem():
    sp = Space([Real("x", 0, 1)])
    # a "repair" that moves points to an infeasible spot must not make them count as feasible
    p = Problem(sp, lambda x: float(x[0]), constraints=[ineq(lambda x: 0.5 - x[0], name="x>=0.5")],
                repair=lambda x: np.array([0.1]))
    r = get_solver("genetic").solve(p, Budget(300), seed=0)
    assert not r.feasible or r.x[0] >= 0.5 - 1e-9


def test_noisy_reports_repeat_statistics():
    rng = np.random.default_rng(0)
    sp = Space([Real("x", -2, 2)])
    p = Problem(sp, lambda x: float(x[0] ** 2 + 0.01 * rng.normal()), noisy=True)
    r = get_solver("cmaes").solve(p, Budget(600), seed=0)
    assert r.noise is not None and r.noise["verification_repeats"] >= 2
