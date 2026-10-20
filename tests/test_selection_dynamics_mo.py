"""Rule/evidence selection, simulation adapters, multi-objective utilities and the high-level API."""
import numpy as np
import pytest

from optiforge import Budget, Problem, Real, Space, Integer
from optiforge.api import plan, solve
from optiforge.benchmarks.problems import make_problem
from optiforge.core.result import FEASIBLE_SAMPLED, FEASIBLE_VERIFIED
from optiforge.dynamics.systems import ContinuousTimeSystem, DiscreteTimeSystem, simulation_problem
from optiforge.multiobjective import MultiObjectiveProblem, ScalarizedProblem, pareto_front, sweep
from optiforge.registry.store import Registry
from optiforge.selection.evidence import EvidenceSelector
from optiforge.selection.rules import select_by_rules
from optiforge.solvers import get_solver

from conftest import sphere_problem


@pytest.mark.parametrize("family,kw,expected", [
    ("knapsack", {"n": 10}, "exhaustive"),
    ("tsp", {"n": 9}, "simulated_annealing"),
    ("rastrigin", {}, "hybrid"),
    ("g06", {}, "multistart"),
    ("mixed_rastrigin", {}, "hybrid"),
])
def test_rules(family, kw, expected):
    p = make_problem(family, 0, **kw)
    pl = select_by_rules(p, Budget(5000))
    assert pl.solver == expected
    assert pl.rationale and any(r.startswith("R") for r in pl.rationale)


def test_mode_restricts_strategy():
    p = sphere_problem()
    assert select_by_rules(p, Budget(5000), mode="local").strategy == "local"
    assert select_by_rules(p, Budget(5000), mode="global").solver == "cmaes"


def _champion(reg, pairs, p_value, kind="continuous", dims=(5,), legacy=False):
    rec = reg.add(kind="algorithm", representation="dsl", source={"name": f"c{pairs}{p_value}{legacy}"}, provider="test")
    comp = {"pairs": pairs, "p_value_a_better": p_value, "baseline": "cmaes", "mean_diff": -0.2}
    if not legacy:
        comp.update(p_value_adjusted=p_value, ci95_mean_diff=[-0.3, -0.1], min_effect=0.01)
    entry = {"split": "validation", "budget": 3000, "summary": {"mean_score": 0.1},
             "problem_profile": {"kind": kind, "types": ["real"], "dims": list(dims), "constrained": False},
             "vs_best_baseline": comp}
    reg.update(rec["id"], status="champion", benchmarks_append=entry)
    return rec["id"]


def test_evidence_requires_enough_significant_runs(tmp_path):
    reg = Registry(str(tmp_path))
    p = make_problem("rastrigin", 0)
    base = select_by_rules(p, Budget(3000))
    _champion(reg, pairs=6, p_value=0.001)
    _champion(reg, pairs=20, p_value=0.3)
    _champion(reg, pairs=40, p_value=1e-6, legacy=True)  # raw p only (pre-2026-10-20 record): not used
    kept = EvidenceSelector(reg).refine(p, Budget(3000), base)
    assert kept.solver == base.solver and kept.source == "rules"
    cid = _champion(reg, pairs=20, p_value=0.001)
    got = EvidenceSelector(reg).refine(p, Budget(3000), select_by_rules(p, Budget(3000)))
    assert got.solver == f"candidate:{cid}" and got.source == "evidence"
    far = EvidenceSelector(reg).refine(p, Budget(50_000), select_by_rules(p, Budget(50_000)))
    assert far.source == "rules"  # budget outside x2 of the evidence


def test_api_solve_auto_records_plan():
    r = solve(make_problem("knapsack", 0, n=8), "auto", 1000)
    assert r.certified_optimal and r.diagnostics["plan"]["solver"] == "exhaustive"


def _decay_system():
    # x' = -theta x, x(0) = 1; path constraint x(t) <= 0.5 for t >= 1  ->  theta >= ln 2
    sys_ = ContinuousTimeSystem(lambda t, x, th: -th[0] * x, np.array([1.0]), t_final=2.0)
    cons = [("settle", lambda t, X, th: np.where(t >= 1.0, X[:, 0] - 0.5, -1.0))]
    return simulation_problem(sys_, Space([Real("theta", 0.0, 3.0)]), lambda t, X, th: float(th[0]),
                              cons, name="decay", n_samples=50)


def test_continuous_time_feasibility_is_sampled_only():
    p = _decay_system()
    r = get_solver("de").solve(p, Budget(400), seed=0)
    assert r.feasible
    assert r.objective == pytest.approx(np.log(2), abs=0.02)
    kinds = {g.kind for g in r.guarantees}
    assert FEASIBLE_SAMPLED in kinds and FEASIBLE_VERIFIED not in kinds
    assert r.diagnostics["verifier"]["scope"].startswith("sampled")


def test_discrete_time_feasibility_is_exact():
    sys_ = DiscreteTimeSystem(lambda x, k, th: 0.9 * x + th[0], np.array([0.0]), horizon=30)
    p = simulation_problem(sys_, Space([Real("u", 0, 1)]), lambda t, X, th: -float(X[-1, 0]),
                           [("cap", lambda t, X, th: X[:, 0] - 5.0)], name="dt")
    r = get_solver("de").solve(p, Budget(400), seed=0)
    assert r.feasible and FEASIBLE_VERIFIED in {g.kind for g in r.guarantees}


def test_pd_control_benchmark_runs():
    p = make_problem("pd_control", 0)
    r = get_solver("cmaes").solve(p, Budget(150), seed=0)
    assert r.x is not None and p.metadata["system_kind"] == "continuous_time"
    assert r.status in ("feasible", "feasible_discretized_only", "no_feasible_found")


def test_multiobjective_scalarisation_and_front():
    sp = Space([Real("x", 0, 1)])
    mo = MultiObjectiveProblem(sp, [lambda x: float(x[0] ** 2), lambda x: float((x[0] - 1) ** 2)])
    p = ScalarizedProblem(mo, [0.5, 0.5])
    r = get_solver("nelder_mead").solve(p, Budget(300), seed=0)
    assert r.x[0] == pytest.approx(0.5, abs=1e-3)
    F = np.array([[0, 1], [1, 0], [0.5, 0.5], [1, 1]])
    assert sorted(pareto_front(F).tolist()) == [0, 1, 2]
    pts, vals, _ = sweep(mo, lambda: get_solver("nelder_mead"), [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]], Budget(200))
    assert len(pts) == 3 and np.all(np.diff(vals[np.argsort(vals[:, 0]), 1]) <= 1e-9)
    with pytest.raises(ValueError):
        ScalarizedProblem(mo, [-1, 2])


def test_scoring_excludes_not_applicable_runs():
    from optiforge.benchmarks.runner import paired_comparison, per_family, reference_table, summarize
    rows = []
    for seed in range(6):
        rows.append({"solver": "random_search", "instance": "i", "family": "f", "seed": seed, "status": "feasible",
                     "feasible": True, "internal": 10.0 + seed, "known_internal": 0.0, "known_method": "analytic"})
        rows.append({"solver": "a", "instance": "i", "family": "f", "seed": seed, "status": "feasible",
                     "feasible": True, "internal": 1.0, "known_internal": 0.0, "known_method": "analytic"})
        rows.append({"solver": "b", "instance": "i", "family": "f", "seed": seed, "status": "incompatible",
                     "feasible": False, "internal": None})
    ref = reference_table(rows)
    assert ref["i"]["ref_kind"] == "known optimum (analytic)"
    s = summarize(rows, ref)
    assert "b" not in s and s["a"]["feasibility_rate"] == 1.0
    assert "b" not in per_family(rows)["f"]
    assert paired_comparison(rows, "b", "random_search")["pairs"] == 0
    assert paired_comparison(rows, "a", "random_search")["wins"] == 6


def test_paired_comparison_reports_effect_sizes_and_practical_ties():
    from optiforge.benchmarks.runner import holm_adjust, paired_comparison
    rows = []
    for i in range(20):
        rows.append({"solver": "a", "instance": f"i{i}", "seed": 0, "score": 0.1 - 1e-10})  # tiny, consistent edge
        rows.append({"solver": "b", "instance": f"i{i}", "seed": 0, "score": 0.1})
    c = paired_comparison(rows, "a", "b")
    assert c["practical_ties"] == 20 and c["practical_wins"] == 0
    lo, hi = c["ci95_mean_diff"]
    assert lo <= c["mean_diff"] <= hi and abs(hi) < 1e-6
    assert c["rank_biserial"] == 0.0  # 1e-10 differences are ties for the effect size too
    adj = holm_adjust({"x": 0.01, "y": 0.04, "z": None})
    assert adj == {"x": 0.02, "y": 0.04, "z": None}
