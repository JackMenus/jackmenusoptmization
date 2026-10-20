"""Registry, validation gates, providers (incl. scripted LLM), materialisation and the development loop."""
import json
import os

import numpy as np
import pytest

from optiforge import Budget
from optiforge.benchmarks.suites import SUITES, get_suite
from optiforge.core.budget import BudgetLedger
from optiforge.core.errors import BudgetExhausted, CandidateRejected
from optiforge.dsl.spec import DEFAULT_OPERATOR, host_algorithm
from optiforge.generation import loop as loop_mod
from optiforge.generation.code_operators import CodeTemplateGenerator, render_mutation
from optiforge.generation.grammar import Bias
from optiforge.generation.loop import DevelopmentConfig, DevelopmentLoop
from optiforge.generation.materialize import build_solver
from optiforge.generation.providers import (GrammarProvider, LLMProvider, ProposalContext, ScriptedLLMClient,
                                            default_providers)
from optiforge.registry.store import Registry
from optiforge.validation.gates import validate

REAL = {"real"}
GOOD_CODE = {"select": "rate", "dist": "gauss", "scale": "sigma_range", "boundary": "reflect", "integer": "round",
             "discrete": "keep", "perm": "keep", "rate": 0.3, "p_discrete": 0.1}


def add_dsl(reg, source, kind="operator"):
    return reg.add(kind=kind, representation="dsl", source=source, provider="test")


def test_registry_versioning_and_history(tmp_path):
    reg = Registry(str(tmp_path / "reg"))
    rec = add_dsl(reg, DEFAULT_OPERATOR)
    assert rec["version"] == 1 and rec["status"] == "proposed"
    reg.update(rec["id"], status="validated", validation={"passed": True})
    reg.update(rec["id"], benchmarks_append={"split": "train", "summary": {"mean_score": 0.3}})
    latest = reg.get(rec["id"])
    assert latest["version"] == 3 and len(latest["benchmarks"]) == 1 and len(latest["history"]) == 3
    assert reg.get(rec["id"], 1)["status"] == "proposed"  # earlier versions are kept unchanged
    again = add_dsl(reg, DEFAULT_OPERATOR)  # same canonical source -> same id, re-proposal recorded
    assert again["id"] == rec["id"] and again["history"][-1]["operation"] == "re-proposed"
    assert [r["id"] for r in reg.list(status="validated")] == [rec["id"]]


def test_unvalidated_candidates_cannot_be_used(tmp_path):
    reg = Registry(str(tmp_path / "reg"))
    rec = add_dsl(reg, DEFAULT_OPERATOR)
    with pytest.raises(CandidateRejected):
        build_solver(reg.get(rec["id"]))
    reg.update(rec["id"], status="validated")
    assert build_solver(reg.get(rec["id"])) is not None


def test_gates_pass_valid_dsl_operator(tmp_path):
    reg = Registry(str(tmp_path / "reg"))
    rec = add_dsl(reg, DEFAULT_OPERATOR)
    rep = validate(rec, REAL, lambda: build_solver(rec, allow_unvalidated=True), BudgetLedger(10**6))
    assert rep.passed, [(c.name, c.detail) for c in rep.checks if not c.passed]
    assert {c.name for c in rep.checks} == {"interface", "bounds_and_types", "numerical_validity", "constraint_handling",
                                            "reproducibility", "termination", "failure_handling"}


def test_gates_reject_statically_invalid_dsl(tmp_path):
    reg = Registry(str(tmp_path / "reg"))
    rec = add_dsl(reg, {"op": "bit_flip"})
    rep = validate(rec, REAL, lambda: build_solver(rec, allow_unvalidated=True))
    assert not rep.passed and rep.checks[0].name == "interface"


def _code_rec(reg, g):
    src = render_mutation(g)
    return reg.add(kind="operator", representation="python", source=src, provider="test",
                   extra={"meta": {"types": CodeTemplateGenerator.types_changed(g), "grammar": g}})


def test_gates_on_code_operators(tmp_path, needs_isolation):
    reg = Registry(str(tmp_path / "reg"))
    good = _code_rec(reg, GOOD_CODE)
    rep = validate(good, REAL, lambda: build_solver(good, allow_unvalidated=True))
    assert rep.passed, [(c.name, c.detail) for c in rep.checks if not c.passed]
    bad = _code_rec(reg, {**GOOD_CODE, "boundary": "none", "dist": "cauchy"})  # leaves the box
    rep = validate(bad, REAL, lambda: build_solver(bad, allow_unvalidated=True))
    assert not rep.passed
    assert not next(c for c in rep.checks if c.name == "interface").passed


def test_code_operator_raising_is_rejected(tmp_path, needs_isolation):
    reg = Registry(str(tmp_path / "reg"))
    rec = reg.add(kind="operator", representation="python", provider="test",
                  source="def mutate(x, info, rng):\n    raise RuntimeError('bad')\n", extra={"meta": {"types": ["real", "integer"]}})
    rep = validate(rec, REAL, lambda: build_solver(rec, allow_unvalidated=True))
    assert not rep.passed


def _ctx(ledger=None):
    return ProposalContext(REAL, Bias(), [], {}, 0, ledger)


def test_scripted_llm_provider_charges_and_checks():
    good = json.dumps(host_algorithm(DEFAULT_OPERATOR, name="llm-alg"))
    client = ScriptedLLMClient(["Here you go:\n" + good, "no json here", '{"op": "bit_flip"}'])
    led = BudgetLedger(total=100, limits={"model_calls": 3})
    prov = LLMProvider(client)
    props = prov.propose(3, _ctx(led), np.random.default_rng(0))
    assert led.used["model_calls"] == 3 and client.calls == 3
    assert len(props) == 2 and props[0].kind == "algorithm"
    assert "fails static checks" in props[1].rationale  # invalid proposals are kept for the record, then rejected
    assert prov.failures == ["no JSON object in model output"]
    with pytest.raises(BudgetExhausted):
        prov.propose(1, _ctx(led), np.random.default_rng(0))


def test_code_generation_disabled_without_isolation():
    provs, notes = default_providers({"code_generation": True}, isolation_available=False)
    assert all(p.name != "code_template" for p in provs)
    assert any("isolation unavailable" in n for n in notes)


def test_grammar_provider_proposals_are_valid():
    props = GrammarProvider().propose(6, _ctx(), np.random.default_rng(3))
    assert len(props) == 6 and all(p.representation == "dsl" for p in props)


def test_suite_splits_are_disjoint():
    for s in SUITES.values():
        s.check_disjoint()
        ids = {sp: {i.id for i in s.instances(sp)} for sp in ("train", "validation", "test")}
        assert not (ids["train"] & ids["test"]) and not (ids["validation"] & ids["test"])


@pytest.mark.integration
def test_development_loop_end_to_end(tmp_path, monkeypatch):
    calls = []
    real = loop_mod.run_matrix

    def spy(specs, instances, seeds, budget, ledger=None, category="benchmark", workers=None):
        calls.append((category, {i.id for i in instances}))
        return real(specs, instances, seeds, budget, ledger, category, workers)

    monkeypatch.setattr(loop_mod, "run_matrix", spy)
    cfg = DevelopmentConfig(suite="continuous_multimodal", run_budget=400, run_seeds=[0], test_run_seeds=[0, 1],
                            generations=2, proposals_per_generation=4, retain=2, total_evaluations=800_000,
                            max_wall_time=600, registry=str(tmp_path / "reg"), workers=4)
    rep = DevelopmentLoop(cfg, str(tmp_path / "out")).run()
    test_ids = {i.id for i in get_suite("continuous_multimodal").instances("test")}
    cats = [c for c, _ in calls]
    first_test = cats.index("test")
    assert all(c == "test" for c in cats[first_test:])
    assert all(not (ids & test_ids) for c, ids in calls if c != "test")
    assert os.path.exists(tmp_path / "out" / "development_report.md")
    assert rep["ledger"]["used_total"] <= cfg.total_evaluations
    assert rep["selection"]["decision"]
    assert all("behavioural_duplicates" in g for g in rep["generations"])
    assert rep["ledger"]["used_by_category"].get("fingerprint", 0) > 0
    assert Registry(str(tmp_path / "reg")).list()


def test_v2_suite_has_fresh_seeds_and_test_only_families():
    from optiforge.benchmarks.suites import SUITES, get_suite
    s = get_suite("continuous_general")
    used = set()
    for name, other in SUITES.items():
        if name != s.name:
            for split in other.splits.values():
                used |= set(split)
    for split in ("train", "validation", "test"):
        assert not set(s.splits[split]) & used
    fam = lambda split: {i.family for i in s.instances(split)}
    held_out = {f for f, _ in s.test_only}
    assert held_out and held_out <= fam("test")
    assert not held_out & (fam("train") | fam("validation"))
    elsewhere = {f for name, other in SUITES.items() if name != s.name for f, _ in other.families + other.test_only}
    assert not held_out & elsewhere  # never seen by any earlier development run either
    assert s.test_status.startswith("historical")  # test split run once in round 6


def test_rotated_families_have_verified_optimum_inside_the_box():
    from optiforge.benchmarks.problems import make_problem
    for fam in ("rastrigin_rotated", "ackley_rotated", "schwefel_1_2", "bent_cigar_rotated", "griewank_rotated"):
        for dim in (5, 10):
            p = make_problem(fam, 3, dim=dim)
            x = p.known_optimum.x
            assert p.space.contains(x) and abs(p.objective(x) - p.known_optimum.value) < 1e-12


def test_behaviour_fingerprint_merges_equivalent_code_operators(tmp_path, needs_isolation):
    from optiforge.generation.fingerprint import behaviour_fingerprint, probe_instances
    reg = Registry(str(tmp_path / "reg"))
    probes = probe_instances(get_suite("continuous_general").instances("train"))[:2]
    allc = {**GOOD_CODE, "select": "all", "dist": "uniform"}
    a = _code_rec(reg, {**allc, "rate": 0.59})
    b = _code_rec(reg, {**allc, "rate": 0.28, "discrete": "resample"})  # rate unused with select=all; no discrete vars
    c = _code_rec(reg, {**allc, "dist": "gauss"})
    fp = {k: behaviour_fingerprint(lambda r=r: build_solver(r, allow_unvalidated=True), probes, budget=150)
          for k, r in {"a": a, "b": b, "c": c}.items()}
    assert a["source"] != b["source"]
    assert fp["a"][0] == fp["b"][0]
    assert fp["a"][0] != fp["c"][0]
    assert 0 < fp["a"][1] <= 2 * 150


def test_behaviour_fingerprint_of_builtins_is_stable_and_distinguishes(tmp_path):
    from optiforge.benchmarks.runner import builtin
    from optiforge.generation.fingerprint import behaviour_fingerprint, probe_instances
    probes = probe_instances(get_suite("continuous_general").instances("train"))
    assert len(probes) == len({(i.family, i.params) for i in probes}) == 8
    cm, de = builtin("cmaes"), builtin("de")
    assert behaviour_fingerprint(cm.build, probes[:2])[0] == behaviour_fingerprint(cm.build, probes[:2])[0]
    assert behaviour_fingerprint(cm.build, probes[:2])[0] != behaviour_fingerprint(de.build, probes[:2])[0]


@pytest.mark.parametrize("family,kw", [("knapsack_correlated", {"n": 12}), ("multi_knapsack", {"n": 12, "m": 3}),
                                       ("independent_set", {"n": 13, "density": 0.3})])
def test_discrete_v2_optima_match_brute_force(family, kw):
    import itertools
    from optiforge.benchmarks.problems import make_problem
    for seed in (0, 1):
        p = make_problem(family, seed, **kw)
        n = p.space.dim
        best = -np.inf
        for bits in itertools.product((0, 1), repeat=n):
            x = np.array(bits, dtype=float)
            if all(c.fn(x) <= 0 for c in p.constraints):
                best = max(best, p.objective(x))
        assert p.known_optimum.value == pytest.approx(best, abs=1e-9)
        if p.known_optimum.x is not None:
            assert p.objective(p.known_optimum.x) == pytest.approx(best, abs=1e-9)
            assert all(c.fn(p.known_optimum.x) <= 0 for c in p.constraints)


def test_integer_quadratic_chunked_enumeration_matches_full_grid():
    import itertools
    from optiforge.benchmarks.problems import _enumerate_quadratic
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    Q, a = A @ A.T + 0.5 * np.eye(3), rng.uniform(-3, 3, 3)
    grid = np.array(list(itertools.product(range(-4, 5), repeat=3)), dtype=float)
    vals = np.einsum("bi,ij,bj->b", grid - a, Q, grid - a)
    f, z = _enumerate_quadratic(Q, a, 3, 4, chunk=50)
    assert f == vals.min() and np.array_equal(z, grid[np.argmin(vals)])


def test_set_cover_optimum_matches_brute_force():
    import itertools
    from optiforge.benchmarks.problems import make_problem
    p = make_problem("set_cover", 0, n_sets=12, n_elements=10, density=0.25)
    best = min(p.objective(np.array(b, float)) for b in itertools.product((0, 1), repeat=12)
               if p.constraints[0].fn(np.array(b, float)) <= 0)
    assert p.known_optimum.value == pytest.approx(best, abs=1e-9)


def test_discrete_general_v2_uses_fresh_seeds_and_a_held_out_family():
    s = get_suite("discrete_general")
    other_seeds, other_fams = set(), set()
    for name, o in SUITES.items():
        if not name.startswith("discrete_general"):  # later versions of this suite reuse its families
            other_fams |= {f for f, _ in o.families} | {f for f, _ in o.test_only}
            for split in o.splits.values():
                other_seeds |= set(split)
    assert not other_seeds & {x for v in s.splits.values() for x in v}
    assert not {f for f, _ in s.test_only} & (other_fams | {f for f, _ in s.families})
    assert s.test_status.startswith("historical")  # test split run once in round 10
    for split in ("train", "validation", "test"):
        for inst in s.instances(split):
            assert inst.make().known_optimum is not None


def test_discrete_general_v3_test_instances_are_fresh():
    s = get_suite("discrete_general_v3")
    used = set()
    for name, o in SUITES.items():
        if name != s.name:
            for split in o.splits.values():
                used |= set(split)
    assert not used & {x for v in s.splits.values() for x in v}
    assert len(s.instances("test")) == 6 * 5
