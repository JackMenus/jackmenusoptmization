"""Numerical tests of the solver portfolio, certificates, checkpoints and warm starts."""
import numpy as np
import pytest

from optiforge import Budget, Problem, Space, Real, Integer, Binary, Permutation
from optiforge.benchmarks.problems import make_problem
from optiforge.core.constraints import LinearStructure, ineq
from optiforge.core.result import GLOBAL_CERTIFIED, INFEASIBILITY_CERTIFIED, STATIONARY, GLOBAL_CONDITIONAL
from optiforge.solvers import SOLVERS, get_solver

from conftest import sphere_problem

CONTINUOUS = ["lbfgsb", "slsqp", "nelder_mead", "de", "cmaes", "pattern_search", "multistart", "hybrid"]
DISCRETE = ["random_search", "greedy_ls", "simulated_annealing", "tabu", "genetic"]


@pytest.mark.parametrize("name", CONTINUOUS)
def test_continuous_solvers_reach_sphere_minimum(name):
    r = get_solver(name).solve(sphere_problem(dim=3), Budget(4000), seed=0)
    assert r.feasible
    assert r.objective < 1e-4, r.summary()


def test_random_search_is_weaker_than_cmaes():
    p = make_problem("rosenbrock", 0)
    rs = get_solver("random_search").solve(p, Budget(3000), seed=0)
    cm = get_solver("cmaes").solve(p, Budget(3000), seed=0)
    assert cm.objective < rs.objective


def test_lbfgsb_stationarity_and_convex_conditional():
    r = get_solver("lbfgsb").solve(sphere_problem(dim=3), Budget(500), seed=0)
    assert STATIONARY in {g.kind for g in r.guarantees}
    assert GLOBAL_CONDITIONAL not in {g.kind for g in r.guarantees}  # convexity not declared
    rc = get_solver("lbfgsb").solve(sphere_problem(dim=3, convex=True), Budget(500), seed=0)
    g = [g for g in rc.guarantees if g.kind == GLOBAL_CONDITIONAL]
    assert g and g[0].assumptions
    assert GLOBAL_CERTIFIED not in {g.kind for g in rc.guarantees}


def test_slsqp_handles_inequality():
    sp = Space([Real("x", -2, 2), Real("y", -2, 2)])
    p = Problem(sp, lambda x: float(x[0] ** 2 + x[1] ** 2), constraints=[ineq(lambda x: 1 - x[0] - x[1], name="sum>=1")])
    r = get_solver("slsqp").solve(p, Budget(1000), seed=0)
    assert r.feasible and r.objective == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("name", DISCRETE)
def test_discrete_solvers_on_onemax(name):
    sp = Space([Binary(f"b{i}") for i in range(12)])
    p = Problem(sp, lambda x: float(np.sum(x)), sense="max")
    r = get_solver(name).solve(p, Budget(3000), seed=1)
    assert r.feasible and r.objective >= (12 if name != "random_search" else 9)


@pytest.mark.parametrize("name", ["simulated_annealing", "genetic", "tabu"])
def test_permutation_solvers_on_tsp(name):
    p = make_problem("tsp", 0, n=7)
    r = get_solver(name).solve(p, Budget(4000), seed=0)
    assert r.feasible
    assert sorted(r.x) == list(range(7))
    assert r.objective <= p.known_optimum.value * 1.1


def test_exhaustive_certifies_global_optimum():
    p = make_problem("knapsack", 0, n=10)
    r = get_solver("exhaustive").solve(p, Budget(2000), seed=0)
    assert r.status == "optimal_certified" and r.certified_optimal
    assert r.objective == pytest.approx(p.known_optimum.value)


def test_exhaustive_certifies_infeasibility():
    sp = Space([Integer("i", 0, 3), Binary("b")])
    p = Problem(sp, lambda x: 0.0, constraints=[ineq(lambda x: 10 - x[0] - x[1], name="impossible")])
    r = get_solver("exhaustive").solve(p, Budget(100), seed=0)
    assert r.status == "infeasible_certified"
    assert INFEASIBILITY_CERTIFIED in {g.kind for g in r.guarantees}


def test_exhaustive_refuses_when_budget_too_small():
    p = make_problem("knapsack", 0, n=10)
    r = get_solver("exhaustive").solve(p, Budget(100), seed=0)
    assert not r.certified_optimal


def test_milp_certified():
    sp = Space([Integer("a", 0, 10), Integer("b", 0, 10)])
    lin = LinearStructure(c=np.array([-3.0, -2.0]), A_ub=np.array([[1.0, 1.0], [1.0, 3.0]]), b_ub=np.array([4.0, 6.0]))
    p = Problem.from_linear(sp, lin)
    r = get_solver("milp").solve(p, Budget(100), seed=0)
    assert r.status == "optimal_certified" and r.objective == pytest.approx(-12.0)


def test_milp_on_declared_black_box_functions_is_only_conditional():
    sp = Space([Integer("a", 0, 10), Integer("b", 0, 10)])
    lin = LinearStructure(c=np.array([-3.0, -2.0]), A_ub=np.array([[1.0, 1.0], [1.0, 3.0]]), b_ub=np.array([4.0, 6.0]))
    p = Problem(sp, lambda x: float(-3 * x[0] - 2 * x[1]),
                constraints=[ineq(lambda x: x[0] + x[1] - 4, name="c1"), ineq(lambda x: x[0] + 3 * x[1] - 6, name="c2")],
                linear=lin)
    r = get_solver("milp").solve(p, Budget(100), seed=0)
    assert r.status == "feasible" and not r.certified_optimal and r.objective == pytest.approx(-12.0)
    assert "global_optimum_conditional_on_assumptions" in [g.kind for g in r.guarantees]


def test_milp_detects_mismatched_linear_declaration():
    sp = Space([Integer("a", 0, 10)])
    p = Problem(sp, lambda x: float(x[0] ** 2), linear=LinearStructure(c=np.array([1.0])))
    r = get_solver("milp").solve(p, Budget(100), seed=0)
    assert not r.certified_optimal


def test_mixed_hybrid_on_mixed_rastrigin():
    p = make_problem("mixed_rastrigin", 0)
    r = get_solver("hybrid", global_solver="genetic", local="pattern_search").solve(p, Budget(3000), seed=0)
    rs = get_solver("random_search").solve(p, Budget(3000), seed=0)
    assert r.feasible and r.objective < rs.objective


@pytest.mark.parametrize("name", [n for n, c in sorted(SOLVERS.items()) if c.supports_checkpoint])
def test_checkpoint_resume_reproduces_uninterrupted_run(name, tmp_path):
    p = make_problem("rastrigin", 0) if name not in DISCRETE else make_problem("maxcut", 0, n=10)
    full = get_solver(name).solve(p, Budget(1500), seed=4)
    ck = str(tmp_path / "ck.json")
    part = get_solver(name).solve(p, Budget(1500), seed=4, checkpoint_path=ck, checkpoint_every=3,
                                  cancel=None, callback=_stop_after(full.n_evaluations // 2))
    assert part.termination_reason == "cancelled"
    resumed = get_solver(name).solve(p, Budget(1500), seed=4, checkpoint_path=ck, checkpoint_every=3, resume=True)
    assert resumed.n_evaluations == full.n_evaluations
    assert resumed.x == full.x
    assert resumed.objective == full.objective


def _stop_after(n):
    from optiforge.core.errors import SearchCancelled

    def cb(ctx):
        if ctx.evaluator.n_evals >= n:
            raise SearchCancelled("test interruption")
    return cb


def test_warm_start_is_used():
    p = sphere_problem(dim=5)
    r = get_solver("random_search").solve(p, Budget(5), seed=0, warm_start=np.zeros(5))
    assert r.objective == 0.0


def test_reproducible_with_seed():
    p = make_problem("ackley", 0)
    a = get_solver("de").solve(p, Budget(800), seed=7)
    b = get_solver("de").solve(p, Budget(800), seed=7)
    assert a.x == b.x and a.history == b.history


def test_budget_never_exceeded():
    p = make_problem("rastrigin", 0)
    for name in CONTINUOUS + ["random_search", "genetic"]:
        r = get_solver(name).solve(p, Budget(333), seed=0)
        assert r.n_evaluations <= 333, name


def test_time_budget():
    import time
    sp = Space([Real("x", -1, 1)])
    p = Problem(sp, lambda x: (time.sleep(0.002), float(x[0] ** 2))[1])
    r = get_solver("de").solve(p, Budget(10**6, max_time=0.3), seed=0)
    assert r.termination_reason == "budget_time" and r.runtime_s < 1.0
