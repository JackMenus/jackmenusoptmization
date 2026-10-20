"""Regression tests for certificate and status defects found in the 2026-10-20 correctness audit.

Each test pins one rule from docs/guarantees.md: a certificate needs exhaustive enumeration or an exact solve of a
model that *is* the problem, and execution failures are not reported as unsuccessful searches.
"""
import time

import numpy as np
import pytest

from optiforge import Budget, Problem, Space
from optiforge.core.constraints import LinearStructure, eq, ineq
from optiforge.core.space import Categorical, Integer, Real
from optiforge.solvers import get_solver


def kinds(r):
    return [g.kind for g in r.guarantees]


# --- MILP ---------------------------------------------------------------------------------

def test_milp_refuses_when_the_solution_point_disagrees_with_the_declaration():
    f = lambda x: float(x[0]) + (1000.0 if x[0] == 7 else 0.0)  # linear everywhere except the MILP optimum x = 7
    p = Problem(Space([Integer("x", 0, 10 ** 6)]), f, constraints=[ineq(lambda x: 7.0 - x[0], "x>=7", tol=0.0)],
                linear=LinearStructure(c=np.array([1.0]), A_ub=np.array([[-1.0]]), b_ub=np.array([-7.0])))
    r = get_solver("milp").solve(p, seed=0)
    assert not r.certified_optimal
    assert "global_optimum_conditional_on_assumptions" not in kinds(r)
    assert any("declaration is wrong" in m for m in r.messages)


def test_milp_checks_declared_constraints_without_constraint_functions():
    p = Problem(Space([Integer("x", 0, 10 ** 6)]), lambda x: float(x[0]),
                linear=LinearStructure(c=np.array([1.0]), A_ub=np.array([[-1.0]]), b_ub=np.array([-1.0])))
    r = get_solver("milp").solve(p, seed=0)
    assert not r.certified_optimal and "global_optimum_conditional_on_assumptions" not in kinds(r)


def test_milp_declared_infeasibility_is_not_certified():
    # x = 3 is feasible within the constraint's own tolerance; HiGHS on the exact declaration says infeasible
    p = Problem(Space([Integer("x", 0, 10)]), lambda x: float(x[0]), constraints=[eq(lambda x: x[0] - 3.000005, tol=1e-5)],
                linear=LinearStructure(c=np.array([1.0]), A_eq=np.array([[1.0]]), b_eq=np.array([3.000005])))
    r = get_solver("milp").solve(p, seed=0)
    assert r.status != "infeasible_certified"


def test_milp_model_problem_uses_constraint_tolerance():
    model = LinearStructure(c=np.array([1.0]), A_eq=np.array([[1.0]]), b_eq=np.array([3.000005]))
    p = Problem.from_linear(Space([Integer("x", 0, 10)]), model, tol=1e-5)
    r = get_solver("milp").solve(p, seed=0)
    assert r.status == "optimal_certified" and r.x == [3.0]
    p = Problem.from_linear(Space([Integer("x", 0, 10)]), model, tol=1e-9)
    r = get_solver("milp").solve(p, seed=0)
    assert r.status == "infeasible_certified"


def test_milp_requires_minimum_spot_checks():
    p = Problem.from_linear(Space([Integer("x", 0, 10)]), LinearStructure(c=np.array([1.0])))
    with pytest.raises(Exception, match="spot_checks"):
        get_solver("milp", spot_checks=0).solve(p, seed=0)


def test_model_problem_ignores_later_edits_of_the_declaration():
    c = np.array([1.0, 1.0])
    p = Problem.from_linear(Space([Integer("a", 0, 5), Integer("b", 0, 5)]), LinearStructure(c=c))
    c[0] = -100.0  # in-place edit after construction must not split model and functions
    r = get_solver("milp").solve(p, seed=0)
    assert r.status == "optimal_certified" and r.objective == 0.0


def test_editing_a_model_problem_removes_exactness():
    p = Problem.from_linear(Space([Integer("a", 0, 5)]), LinearStructure(c=np.array([1.0])))
    assert p.linear_is_exact
    p.objective = lambda x: float(x[0])
    assert not p.linear_is_exact


def test_milp_respects_budget_time():
    # market split instance: hard for branch and bound
    rng = np.random.default_rng(0)
    A = rng.integers(0, 100, (4, 30)).astype(float)
    b = np.floor(A.sum(axis=1) / 2)
    p = Problem.from_linear(Space([Integer(f"x{i}", 0, 1) for i in range(30)]),
                            LinearStructure(c=np.zeros(30), A_eq=A, b_eq=b))
    t = time.perf_counter()
    r = get_solver("milp").solve(p, Budget(max_time=1.0), seed=0)
    assert time.perf_counter() - t < 5.0
    if r.diagnostics.get("highs_status") == 1:
        assert r.termination_reason == "budget_time" and not r.certified_optimal and r.status != "infeasible_certified"


# --- finalize -----------------------------------------------------------------------------

def test_certificate_dropped_when_independent_verifier_rejects():
    p = Problem(Space([Integer("x", 0, 5)]), lambda x: float(x[0]), verifier=lambda x: {"feasible": False, "scope": "dense"})
    r = get_solver("exhaustive").solve(p, seed=0)
    assert r.status == "feasible_discretized_only" and not r.certified_optimal


def test_certificate_dropped_when_verifier_raises():
    def bad(x):
        raise RuntimeError("verifier crashed")
    p = Problem(Space([Integer("x", 0, 5)]), lambda x: float(x[0]), verifier=bad)
    r = get_solver("exhaustive").solve(p, seed=0)
    assert not r.certified_optimal


def test_certificate_dropped_when_problem_is_nondeterministic():
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return 100.0 if calls["n"] > 6 else float(x[0])
    p = Problem(Space([Integer("x", 0, 5)]), f)
    r = get_solver("exhaustive").solve(p, seed=0)
    assert not r.certified_optimal and r.status == "feasible"


def test_all_evaluations_failing_is_an_error_not_an_unsuccessful_search():
    def f(x):
        raise RuntimeError("simulator crashed")
    r = get_solver("random_search").solve(Problem(Space([Real("x", 0, 1)]), f), Budget(50), seed=0)
    assert r.status == "error"
    r = get_solver("exhaustive").solve(Problem(Space([Integer("x", 0, 5)]), f), seed=0)
    assert r.status == "error" and "infeasibility_certified" not in kinds(r)


def test_failed_verification_is_an_error():
    calls = {"n": 0}

    def g(x):
        calls["n"] += 1
        if calls["n"] > 50:
            raise RuntimeError("license server down")
        return float((x[0] - 0.3) ** 2)
    r = get_solver("random_search").solve(Problem(Space([Real("x", 0, 1)]), g), Budget(50), seed=0)
    assert r.status == "error"
    assert not any("minimises constraint violation" in g.statement for g in r.guarantees)


def test_raising_constraint_during_finalize_does_not_lose_the_result():
    calls = {"n": 0}

    def c(x):
        calls["n"] += 1
        if calls["n"] > 31:
            raise RuntimeError("transient failure")
        return float(x[0] - 0.5)
    p = Problem(Space([Real("x", 0, 1)]), lambda x: float(x[0] ** 2), constraints=[ineq(c, "c")])
    r = get_solver("random_search").solve(p, Budget(30), seed=0)
    assert r.x is not None and r.objective is not None


def test_convex_declaration_gives_no_conditional_claim_for_maximisation():
    p = Problem(Space([Real("x", -1, 1)]), lambda x: float(x[0] ** 2), sense="max", convex=True,
                gradient=lambda x: 2 * np.asarray(x))
    r = get_solver("lbfgsb").solve(p, Budget(200), seed=0)
    assert "global_optimum_conditional_on_assumptions" not in kinds(r)


# --- space --------------------------------------------------------------------------------

@pytest.mark.parametrize("choices,value", [([1, True], True), ([1, True], 1), ([0, False], False), (["a", "b"], "b")])
def test_categorical_roundtrip_distinguishes_equal_but_different_choices(choices, value):
    sp = Space([Categorical("c", choices)])
    out = sp.decode(sp.encode({"c": value}))["c"]
    assert out == value and type(out) is type(value)
