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


def test_slsqp_constraint_calls_go_through_the_evaluator():
    calls = {"f": 0, "c": 0}

    def f(x):
        calls["f"] += 1
        return float(np.sum((x - 1.0) ** 2))

    def c(x):
        calls["c"] += 1
        return float(np.sum(x) - 1.0)
    p = Problem(Space([Real(f"x{i}", -2, 2) for i in range(5)]), f, constraints=[ineq(c, "sum")])
    r = get_solver("slsqp").solve(p, Budget(500), seed=0)
    assert r.feasible
    # every constraint call belongs to a counted evaluation (+1 for verification)
    assert calls["c"] == calls["f"] == r.n_evaluations + r.n_verification_evaluations


def test_slsqp_respects_evaluation_budget_with_constraints():
    calls = {"c": 0}

    def c(x):
        calls["c"] += 1
        return float(x[0] ** 2 + x[1] ** 2 - 1.0)
    p = Problem(Space([Real("x", -2, 2), Real("y", -2, 2)]), lambda x: float(x[0] + x[1]), constraints=[ineq(c, "disk")])
    r = get_solver("slsqp").solve(p, Budget(7), seed=0)
    assert r.n_evaluations <= 7 and calls["c"] <= 7 + r.n_verification_evaluations


def test_slsqp_uses_declared_constraint_jacobian():
    calls = {"c": 0}

    def c(x):
        calls["c"] += 1
        return float(np.sum(x) - 1.0)
    p = Problem(Space([Real(f"x{i}", -2, 2) for i in range(5)]), lambda x: float(np.sum((x - 1.0) ** 2)),
                gradient=lambda x: 2 * (np.asarray(x) - 1.0),
                constraints=[ineq(c, "sum", gradient=lambda x: np.ones((1, 5)))])
    r = get_solver("slsqp").solve(p, Budget(500), seed=0)
    assert r.feasible and r.objective == pytest.approx(0.8 * 5 * 0.0 + 5 * 0.8 ** 2, rel=1e-6)
    assert r.n_evaluations < 40


# --- review of 2026-10-20 (rounds 2-6) ----------------------------------------------------

def test_milp_certifies_mixed_integer_model_problems_with_real_variables():
    rng = np.random.default_rng(0)
    certified = 0
    for t in range(20):
        A_eq = rng.normal(size=(2, 6))
        x0 = np.concatenate([rng.integers(0, 20, 3), rng.uniform(0, 50, 3)])
        A_ub = rng.normal(size=(3, 6))
        m = LinearStructure(c=rng.normal(size=6), A_ub=A_ub, b_ub=A_ub @ x0 + rng.uniform(0, 5, 3), A_eq=A_eq, b_eq=A_eq @ x0)
        sp = Space([Integer(f"i{k}", 0, 20) for k in range(3)] + [Real(f"r{k}", 0, 50) for k in range(3)])
        r = get_solver("milp").solve(Problem.from_linear(sp, m, sense=["min", "max"][t % 2], tol=1e-9), Budget(200), seed=t)
        assert r.feasible
        certified += r.certified_optimal
    assert certified == 20


def test_milp_certificate_survives_tied_optima():
    m = LinearStructure(c=np.array([1.0, 1.0]), A_ub=np.array([[-1.0, -1.0]]), b_ub=np.array([-3.0]))
    p = Problem.from_linear(Space([Integer("a", 0, 5), Integer("b", 0, 5)]), m)
    for seed in range(10):
        r = get_solver("milp").solve(p, Budget(200), seed=seed)
        assert r.status == "optimal_certified" and r.objective == 3.0


def test_milp_model_problem_needs_only_a_few_evaluations():
    m = LinearStructure(c=np.array([1.0, 1.0]), A_ub=np.array([[-1.0, -1.0]]), b_ub=np.array([-3.0]))
    p = Problem.from_linear(Space([Integer("a", 0, 5), Integer("b", 0, 5)]), m)
    r = get_solver("milp").solve(p, Budget(5), seed=0)
    assert r.status == "optimal_certified" and r.n_evaluations <= 4


def test_infeasibility_certificate_dropped_when_verification_fails():
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        if calls["n"] > 4:
            raise RuntimeError("flaky")
        return float(x[0])
    p = Problem(Space([Integer("x", 0, 3)]), f, constraints=[ineq(lambda x: 1.0, "never")])
    r = get_solver("exhaustive").solve(p, seed=0)
    assert r.status == "error" and "infeasibility_certified" not in kinds(r)


def test_slsqp_skips_a_start_whose_vector_constraint_fails():
    calls = {"n": 0}

    def c(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first call fails")
        return np.array([x[0] - 1.0, x[1] - 1.0, x[0] + x[1] - 1.5])
    p = Problem(Space([Real("a", -2, 2), Real("b", -2, 2)]), lambda x: float(-x[0] - x[1]), constraints=[ineq(c, "v")])
    r = get_solver("slsqp", restarts=2).solve(p, Budget(300), seed=0)
    assert r.feasible and r.objective == pytest.approx(-1.5, abs=1e-6)
    assert not any("broadcast" in m for m in r.messages)


def test_raising_verifier_is_not_reported_as_dense_rejection():
    def bad(x):
        raise RuntimeError("verifier crashed")
    p = Problem(Space([Integer("x", 0, 5)]), lambda x: float(x[0]), verifier=bad)
    r = get_solver("exhaustive").solve(p, seed=0)
    assert r.status == "feasible" and not r.certified_optimal
    assert not any("denser grid" in g.statement for g in r.guarantees)


def test_user_code_receives_exact_in_domain_point():
    from optiforge import Budget, Integer, Problem, Real, Space
    from optiforge.core.evaluator import Evaluator
    seen = []
    p = Problem(Space([Integer("k", 0, 5), Real("r", 0.0, 1.0)]), lambda x: seen.append(np.array(x)) or 0.0)
    ev = Evaluator(p, Budget(max_evaluations=5))
    e = ev.evaluate(np.array([3.0 + 5e-10, 1.0 + 5e-10]))
    assert e.feasible
    assert seen[0][0] == 3.0 and seen[0][1] == 1.0
    assert e.x[0] == 3.0 and e.x[1] == 1.0


def test_categorical_decode_rejects_out_of_range_codes():
    from optiforge import Categorical, Space
    sp = Space([Categorical("c", ["a", "b", "c"])])
    assert sp.decode(np.array([2.0]))["c"] == "c"
    for bad in (-0.6, 2.6):
        with pytest.raises(ValueError):
            sp.decode(np.array([bad]))
