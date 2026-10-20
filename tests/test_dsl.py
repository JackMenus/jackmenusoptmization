"""Operator language: static checks, interpreter, grammar and evolution."""
import copy

import numpy as np
import pytest

from optiforge import Budget
from optiforge.benchmarks.problems import make_problem
from optiforge.dsl.interpreter import DSLAlgorithm
from optiforge.dsl.spec import DEFAULT_OPERATOR, check_algorithm, check_operator, host_algorithm, spec_hash
from optiforge.generation.grammar import (OperatorGrammar, crossover_algorithms, mutate_algorithm, mutate_operator,
                                          valid_algorithm, valid_operator)

from conftest import sphere_problem

REAL = {"real"}


def test_default_operator_and_host_are_valid():
    assert check_operator(DEFAULT_OPERATOR, {"real", "integer", "binary"}).ok
    rep = check_algorithm(host_algorithm(DEFAULT_OPERATOR), REAL)
    assert rep.ok, rep.errors


@pytest.mark.parametrize("expr,needle", [
    ({"op": "no_such_op"}, "unknown"),
    ({"op": "gaussian", "sigma": 5.0}, "sigma"),
    ({"op": "seq", "children": [{"op": "gaussian"}, {"op": "uniform_cx"}]}, "producer"),
    ({"op": "bit_flip"}, "real"),  # cannot change real variables
    ({"op": "prob", "p": 0.5, "child": {"op": "gaussian"}, "evil": 1}, "unknown"),
])
def test_check_operator_rejects(expr, needle):
    rep = check_operator(expr, REAL)
    assert not rep.ok
    assert any(needle in e for e in rep.errors), rep.errors


def test_check_algorithm_rejects_structural_errors():
    a = host_algorithm(DEFAULT_OPERATOR)
    bad = copy.deepcopy(a)
    bad["population"] = 0
    bad["stages"].append(copy.deepcopy(bad["stages"][0]))  # first stage now lacks 'until'
    bad["exec"] = "import os"
    rep = check_algorithm(bad, REAL)
    assert not rep.ok
    joined = " ".join(rep.errors)
    assert "population" in joined and "until" in joined and "unknown top-level" in joined


def test_permutation_coverage_required():
    rep = check_algorithm(host_algorithm({"op": "seq", "children": [{"op": "copy"}, {"op": "gaussian"}]}), {"permutation"})
    assert not rep.ok and any("permutation" in e for e in rep.errors)


def test_interpreter_solves_sphere_and_is_reproducible():
    a = host_algorithm(DEFAULT_OPERATOR)
    r1 = DSLAlgorithm(a).solve(sphere_problem(dim=4), Budget(3000), seed=0)
    r2 = DSLAlgorithm(a).solve(sphere_problem(dim=4), Budget(3000), seed=0)
    assert r1.feasible and r1.objective < 1e-2
    assert r1.x == r2.x and r1.history == r2.history


def test_multistage_conditional_algorithm_runs():
    spec = {
        "dsl_version": 1, "kind": "algorithm", "name": "two-stage", "population": 16, "offspring": 16,
        "init": {"op": "lhs"}, "repair": {"op": "reflect"},
        "stages": [
            {"name": "explore", "until": {"pred": "evals_fraction", "ge": 0.5},
             "parent_selection": {"op": "random"},
             "variation": {"op": "de_rand1", "F": 0.7, "CR": 0.9}, "survival": {"op": "pairwise"}},
            {"name": "exploit", "parent_selection": {"op": "tournament", "k": 3},
             "variation": {"op": "if", "cond": {"pred": "stagnation", "ge": 5},
                           "then": {"op": "seq", "children": [{"op": "copy"}, {"op": "cauchy", "sigma": 0.2}]},
                           "else": {"op": "seq", "children": [{"op": "sbx"}, {"op": "gaussian", "sigma": 0.05}]}},
             "survival": {"op": "plus"}, "step": {"op": "one_fifth"}},
        ],
        "restart": {"op": "stagnation", "patience": 40, "keep_best": 1, "reinit": "around_best", "scale": 0.1},
        "refine": {"op": "pattern", "evals_fraction": 0.1, "step0": 0.05},
    }
    rep = check_algorithm(spec, REAL)
    assert rep.ok, rep.errors
    r = DSLAlgorithm(spec).solve(make_problem("rastrigin", 0), Budget(3000), seed=0)
    assert r.feasible and r.termination_reason in ("budget_evaluations", "completed", "converged")


def test_mixed_per_type_operator():
    expr = {"op": "seq", "children": [{"op": "uniform_cx"}, {"op": "per_type",
            "real": {"op": "gaussian", "sigma": 0.1}, "integer": {"op": "integer_step"},
            "categorical": {"op": "categorical_reset"}}]}
    p = make_problem("mixed_rastrigin", 0)
    assert check_operator(expr, p.space.types_present).ok
    r = DSLAlgorithm(host_algorithm(expr)).solve(p, Budget(1500), seed=0)
    assert r.feasible and p.space.contains(np.array(r.x))


def test_grammar_generates_valid_specs():
    rng = np.random.default_rng(0)
    for types in [REAL, {"binary"}, {"permutation"}, {"real", "integer", "categorical"}]:
        g = OperatorGrammar(types)
        for _ in range(15):
            a = g.algorithm(rng)
            assert valid_algorithm(a, types), check_algorithm(a, types).errors
            o = g.operator(rng)
            assert valid_operator(o, types), check_operator(o, types).errors


def test_mutation_and_crossover_preserve_validity_and_change_hash():
    rng = np.random.default_rng(1)
    g = OperatorGrammar(REAL)
    a, b = g.algorithm(rng), g.algorithm(rng)
    changed = 0
    for _ in range(10):
        m, desc = mutate_algorithm(a, rng, g)
        if valid_algorithm(m, REAL):
            changed += spec_hash(m) != spec_hash(a)
        c, _ = crossover_algorithms(a, b, rng)
        o, _ = mutate_operator(g.operator(rng), rng, g)
    assert changed >= 5


CORR_SPEC = {"dsl_version": 1, "kind": "algorithm", "name": "corr", "population": 16, "offspring": 16,
             "init": {"op": "lhs"}, "repair": {"op": "reflect"},
             "stages": [{"name": "s", "parent_selection": {"op": "tournament", "k": 2},
                         "variation": {"op": "seq", "children": [{"op": "arithmetic_cx"},
                                                                 {"op": "correlated_gaussian", "scale": 0.8}]},
                         "survival": {"op": "plus"}, "step": {"op": "one_fifth"}}]}


def test_correlated_gaussian_is_checked_reproducible_and_resumable(tmp_path):
    from optiforge import Budget
    from optiforge.benchmarks.problems import make_problem
    from optiforge.dsl.interpreter import DSLAlgorithm
    from optiforge.dsl.spec import check_algorithm
    assert check_algorithm(CORR_SPEC, {"real"}).ok
    from optiforge.dsl.spec import check_operator
    assert check_operator({"op": "correlated_gaussian"}, {"real"}).ok
    rep = check_operator({"op": "correlated_gaussian"}, {"real", "binary"})
    assert not rep.ok and "binary" in rep.errors[0]  # cannot change binary variables on its own
    p = make_problem("ellipsoid", 1, dim=6)
    a = DSLAlgorithm(CORR_SPEC).solve(p, Budget(1200), seed=3)
    b = DSLAlgorithm(CORR_SPEC).solve(p, Budget(1200), seed=3)
    assert a.x == b.x and a.objective < make_problem("ellipsoid", 1, dim=6).objective(p.space.lower)
    ck = str(tmp_path / "ck.json")
    full = a
    part = DSLAlgorithm(CORR_SPEC).solve(p, Budget(1200), seed=3, checkpoint_path=ck, checkpoint_every=2,
                                         callback=_stop_after(600))
    assert part.termination_reason == "cancelled"
    res = DSLAlgorithm(CORR_SPEC).solve(p, Budget(1200), seed=3, checkpoint_path=ck, checkpoint_every=2, resume=True)
    assert res.x == full.x and res.n_evaluations == full.n_evaluations


def _stop_after(n):
    from tests.test_solvers import _stop_after as s
    return s(n)
