"""The autonomous operator / algorithm development loop.

    analyse problem characteristics (training instances)
      -> propose candidates (grammar, evolution of retained candidates, code templates, optional LLM)
      -> generate executable implementations (DSL interpreter / sandboxed Python)
      -> validate (static checks + seven validation gates)
      -> benchmark on TRAINING instances at the baselines' evaluation budget
      -> retain the best candidates (archive of size k) and iterate within the budget
    then
      -> select on VALIDATION instances against the baselines (paired Wilcoxon, alpha)
      -> evaluate once on held-out TEST instances (after selection is frozen; never fed back)

Every evaluation, validation run and model call is charged to one
:class:`BudgetLedger`; the test phase has a reserved share of the total.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..core.budget import Budget, BudgetLedger
from ..core.errors import BudgetExhausted, CandidateRejected
from ..dsl.spec import check_algorithm, check_operator, compatible_kinds, host_algorithm
from ..isolation.sandbox import probe
from ..registry.store import Registry
from ..benchmarks.runner import (SolverSpec, builtin, holm_adjust, paired_comparison, per_family, reference_table,
                                 run_matrix, summarize)
from ..benchmarks.suites import get_suite
from ..validation.gates import validate
from .features import analyze
from .fingerprint import behaviour_fingerprint, probe_instances
from .grammar import bias_from_features
from .materialize import build_solver
from .providers import ProposalContext, default_providers

log = logging.getLogger("optiforge.development")

DEFAULT_BASELINES = {
    "continuous": ["random_search", "cmaes", "de", "nelder_mead"],
    "discrete": ["random_search", "simulated_annealing", "genetic", "tabu"],
    "mixed": ["random_search", "genetic", "de", "hybrid"],
}


@dataclass
class DevelopmentConfig:
    suite: str = "continuous_multimodal"
    run_budget: int = 3000  # objective evaluations per benchmark run (matched for every solver)
    run_seeds: list[int] = field(default_factory=lambda: [0, 1])
    test_run_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    generations: int = 5
    proposals_per_generation: int = 8
    retain: int = 4
    total_evaluations: int = 3_000_000
    max_model_calls: int = 0
    max_wall_time: float | None = 3600.0
    test_share: float = 0.15  # fraction of the total budget reserved for the held-out test phase
    alpha: float = 0.05
    # selection rule (fixed before any run): Holm-adjusted p < alpha over the archive AND mean score improvement of at
    # least min_effect AND the bootstrap 95% CI of the paired mean difference entirely below zero
    min_effect: float = 0.01
    multiple_testing: str = "holm"  # "holm" or "none" (the 0.1.0 rule)
    # behavioural de-duplication: a validated candidate whose evaluation sequence on one train instance per family
    # (fixed seed, fingerprint_budget evaluations each) equals that of an earlier candidate or a baseline is not
    # benchmarked (status "duplicate"). 0 disables it.
    fingerprint_budget: int = 200
    baselines: list[str] | None = None
    operator_share: float = 0.4
    code_generation: bool = True
    llm: dict | None = None
    workers: int | None = None
    seed: int = 0
    registry: str = "registry"

    @classmethod
    def from_dict(cls, d: dict) -> "DevelopmentConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class DevelopmentLoop:
    def __init__(self, cfg: DevelopmentConfig, out_dir: str, event=None) -> None:
        self.cfg = cfg
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.registry = Registry(cfg.registry)
        self.event = event or (lambda name, **kw: log.info("%s %s", name, kw))
        test_units = int(cfg.total_evaluations * cfg.test_share)
        self.ledger = BudgetLedger(
            total=cfg.total_evaluations + cfg.max_model_calls,
            limits={"test": test_units, "model_calls": cfg.max_model_calls,
                    "analysis": None, "validation": None, "train": None, "selection": None, "train_baselines": None},
            max_wall_time=cfg.max_wall_time)
        self.dev_limit = cfg.total_evaluations + cfg.max_model_calls - test_units  # everything except test
        self._selection_frozen = False
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    def _dev_remaining(self) -> int:
        used = sum(v for k, v in self.ledger.used.items() if k != "test") + sum(
            v for k, v in self.ledger.reserved.items() if k != "test")
        return self.dev_limit - used

    def _fingerprint(self, make_solver, probes) -> str:
        reserve = len(probes) * self.cfg.fingerprint_budget
        try:
            self.ledger.reserve("fingerprint", reserve)
        except BudgetExhausted:
            return f"unfingerprinted-{self.rng.integers(1 << 62)}"  # never matches: no de-duplication without budget
        used = reserve
        try:
            fp, used = behaviour_fingerprint(make_solver, probes, seed=0, budget=self.cfg.fingerprint_budget)
        finally:
            self.ledger.settle("fingerprint", reserve, used)
        return fp

    def _can_afford(self, units: int) -> bool:
        return units <= self._dev_remaining()

    def _register_baselines(self, names):
        recs = {}
        for n in names:
            recs[n] = self.registry.add(kind="algorithm", representation="builtin", source={"builtin": n},
                                        provider="baseline", status="baseline",
                                        rationale="established baseline solver (not generated)")
        return recs

    # ------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        suite = get_suite(cfg.suite)
        train, val, test = suite.instances("train"), suite.instances("validation"), suite.instances("test")
        train_p = [i.make() for i in train]
        feats = analyze(train_p, self.rng, self.ledger)
        types = set(feats["types"])
        kind = feats["kind"] if feats["kind"] in DEFAULT_BASELINES else "mixed"
        baselines = cfg.baselines or DEFAULT_BASELINES[kind]
        if "random_search" not in baselines:
            baselines = ["random_search"] + list(baselines)
        bias = bias_from_features(feats)
        iso = probe()
        providers, notes = default_providers({"operator_share": cfg.operator_share, "code_generation": cfg.code_generation,
                                              "llm": cfg.llm}, iso.available)
        self.event("development_started", suite=suite.name, types=sorted(types), baselines=baselines,
                   isolation=iso.to_dict(), notes=notes, features={k: v for k, v in feats.items() if k != "per_problem"})
        self._register_baselines(baselines)
        budget = Budget(max_evaluations=cfg.run_budget)
        base_specs = [builtin(b) for b in baselines]
        base_rows = run_matrix(base_specs, train, cfg.run_seeds, budget, self.ledger, "train_baselines", cfg.workers)
        ref = reference_table(base_rows)
        base_summary = summarize(base_rows, ref)
        best_base_train = min(base_summary, key=lambda s: base_summary[s]["mean_score"])
        self.event("train_baselines", summary=base_summary)

        probes = probe_instances(train)
        fingerprints: dict[str, str] = {}  # fingerprint -> first candidate or baseline that produced it
        if cfg.fingerprint_budget > 0:
            for b in base_specs:
                fp = self._fingerprint(b.build, probes)
                fingerprints.setdefault(fp, b.label)

        archive: list[dict] = []  # retained registry records with train scores
        generations = []
        stop_reason = "generations_completed"
        all_cands: list[dict] = []
        per_run_cost = len(train) * len(cfg.run_seeds) * cfg.run_budget
        for gen in range(cfg.generations):
            gctx = ProposalContext(types, bias, archive, feats, gen, self.ledger)
            proposals = []
            # allocate proposals: evolution only once something is retained
            n = cfg.proposals_per_generation
            shares = {"grammar": n // 2 if archive else n, "evolution": n // 2 if archive else 0}
            if any(p.name == "code_template" for p in providers):
                shares["code_template"] = max(1, n // 4)
                shares["grammar"] = max(0, shares["grammar"] - shares["code_template"])
            for p in providers:
                k = shares.get(p.name, 1 if p.name == "llm" else 0)
                if k <= 0:
                    continue
                try:
                    proposals += p.propose(k, gctx, self.rng)
                except BudgetExhausted:
                    notes.append(f"provider {p.name} stopped: model-call budget exhausted")
            gen_info = {"generation": gen, "proposed": len(proposals), "rejected": 0, "validated": 0, "benchmarked": 0,
                        "duplicates": 0, "behavioural_duplicates": 0}
            to_bench = []
            for prop in proposals:
                rec = self._register(prop)
                if rec["status"] != "proposed":
                    gen_info["duplicates"] += 1
                    continue
                if not self._can_afford(8 * 600 + per_run_cost):
                    stop_reason = "development_budget_exhausted"
                    break
                rep = self._validate(rec, types)
                if not rep.passed:
                    gen_info["rejected"] += 1
                    continue
                gen_info["validated"] += 1
                if cfg.fingerprint_budget > 0:
                    r = self.registry.get(rec["id"])
                    fp = self._fingerprint(lambda r=r: build_solver(r), probes)
                    first = fingerprints.setdefault(fp, rec["id"])
                    if first != rec["id"]:
                        gen_info["behavioural_duplicates"] += 1
                        self.registry.update(rec["id"], status="duplicate", fingerprint=fp, behavioural_duplicate_of=first,
                                             history_entry={"operation": f"behavioural duplicate of {first}"})
                        self.event("behavioural_duplicate", candidate=rec["id"], duplicate_of=first, fingerprint=fp)
                        continue
                    self.registry.update(rec["id"], fingerprint=fp)
                to_bench.append(self.registry.get(rec["id"]))
            if to_bench and self._can_afford(len(to_bench) * per_run_cost):
                specs = [SolverSpec(r["id"], record=r) for r in to_bench]
                rows = run_matrix(specs, train, cfg.run_seeds, budget, self.ledger, "train", cfg.workers)
                summ = summarize(rows, ref)
                for r in to_bench:
                    s = summ.get(r["id"], {})
                    rec = self.registry.update(r["id"], benchmarks_append={"split": "train", "budget": cfg.run_budget,
                                                                           "seeds": cfg.run_seeds, "summary": s,
                                                                           "reference_baseline": best_base_train,
                                                                           "per_family": per_family([x for x in rows if x["solver"] == r["id"]])},
                                               train_score=s.get("mean_score"),
                                               history_entry={"operation": "benchmarked", "split": "train"})
                    all_cands.append(rec)
                    gen_info["benchmarked"] += 1
            elif to_bench:
                stop_reason = "development_budget_exhausted"
            # retention: best k validated+benchmarked candidates so far
            scored = [self.registry.get(c["id"]) for c in {c["id"]: c for c in all_cands}.values()]
            scored = [c for c in scored if c.get("train_score") is not None]
            scored.sort(key=lambda c: c["train_score"])
            keep = {c["id"] for c in scored[: cfg.retain]}
            for c in scored:
                want = "retained" if c["id"] in keep else "discarded"
                if c["status"] != want:
                    self.registry.update(c["id"], status=want, history_entry={"operation": f"status -> {want}", "generation": gen})
            archive = [self.registry.get(i) for i in [c["id"] for c in scored[: cfg.retain]]]
            gen_info["best_train_score"] = scored[0]["train_score"] if scored else None
            gen_info["best_candidate"] = scored[0]["id"] if scored else None
            gen_info["best_baseline_train_score"] = base_summary[best_base_train]["mean_score"]
            generations.append(gen_info)
            self.event("generation", **gen_info, ledger=self.ledger.summary())
            if stop_reason != "generations_completed":
                break

        # ---- selection on the validation split ----------------------------------
        sel = self._select(archive, baselines, val, budget)
        self._selection_frozen = True
        # ---- held-out test (never used for any decision above) -----------------
        demos = self._demonstrations(all_cands)
        previous = self._test_accesses(suite)
        if previous:
            self.event("test_split_previously_used", accesses=previous)
        test_res = self._final_test(sel, demos, baselines, test, budget)
        if "summary" in test_res:
            self._record_test_access(suite)
        report = {
            "config": asdict(cfg), "suite": suite.to_dict(), "features": {k: v for k, v in feats.items() if k != "per_problem"},
            "bias_notes": bias.notes, "isolation": iso.to_dict(), "provider_notes": notes,
            "baselines": baselines, "train_baselines": base_summary, "train_reference": ref,
            "generations": generations, "stop_reason": stop_reason,
            "archive": [{"id": a["id"], "kind": a["kind"], "representation": a["representation"], "train_score": a.get("train_score"),
                         "provider": a.get("provider"), "parents": a.get("parents")} for a in archive],
            "selection": sel, "demonstrations": demos, "test": test_res, "ledger": self.ledger.summary(),
            "test_split_previous_accesses": previous,
            "wall_time_s": round(time.time() - t0, 1),
            "candidate_counts": self._counts(),
        }
        with open(os.path.join(self.out, "development_report.json"), "w") as fh:
            json.dump(_clean(report), fh, indent=2)
        with open(os.path.join(self.out, "development_report.md"), "w") as fh:
            fh.write(render_markdown(report))
        return report

    # ------------------------------------------------------------------
    def _register(self, prop) -> dict:
        types_changed = []
        if prop.representation == "dsl":
            rep = check_algorithm(prop.source) if prop.kind == "algorithm" else check_operator(prop.source)
            types_changed = rep.info.get("types_changed", [])
        else:
            types_changed = prop.meta.get("types", [])
        hp = _hyperparameters(prop.source) if prop.representation == "dsl" else prop.meta.get("grammar", {})
        rec = self.registry.add(kind=prop.kind, representation=prop.representation, source=prop.source,
                                parents=prop.parents, provider=prop.provider, rationale=prop.rationale,
                                operation=prop.operation, hyperparameters=hp,
                                compatible_kinds=compatible_kinds(set(types_changed)),
                                extra={"meta": prop.meta} if prop.meta else None)
        return rec

    def _validate(self, rec: dict, types: set):
        try:
            rep = validate(rec, types, lambda: build_solver(rec, allow_unvalidated=True), self.ledger)
        except CandidateRejected as exc:
            from ..validation.gates import Check, ValidationReport
            rep = ValidationReport(False, [Check("materialize", False, str(exc))])
        lim = [f"{c.name}: {c.details}" for c in rep.checks if not c.passed]
        self.registry.update(rec["id"], validation=rep.to_dict(), status="validated" if rep.passed else "rejected",
                             limitations=lim + (["benchmarked only on suite " + self.cfg.suite] if rep.passed else []),
                             history_entry={"operation": "validated" if rep.passed else "rejected by validation"})
        self.event("validation", candidate=rec["id"], passed=rep.passed, failed=[c.name for c in rep.checks if not c.passed])
        return rep

    def _select(self, archive, baselines, val, budget) -> dict:
        cfg = self.cfg
        if not archive:
            return {"champion": None, "decision": "no candidate survived validation and training; baselines retained",
                    "comparisons": {}}
        specs = [builtin(b) for b in baselines] + [SolverSpec(a["id"], record=a) for a in archive]
        cost = len(specs) * len(val) * len(cfg.run_seeds) * cfg.run_budget
        if not self._can_afford(cost):
            return {"champion": None, "decision": f"development budget left ({self._dev_remaining()}) cannot cover the "
                    f"validation-split comparison ({cost}); no candidate selected, baselines retained", "comparisons": {}}
        rows = run_matrix(specs, val, cfg.run_seeds, budget, self.ledger, "selection", cfg.workers)
        ref = reference_table([r for r in rows if r["solver"] in baselines])
        summ = summarize(rows, ref)
        best_base = min(baselines, key=lambda b: summ[b]["mean_score"])
        comps = {a["id"]: paired_comparison(rows, a["id"], best_base, cfg.min_effect) for a in archive}
        raw_p = {k: c["p_value_a_better"] for k, c in comps.items()}
        adj = holm_adjust(raw_p) if cfg.multiple_testing == "holm" else raw_p
        for k in comps:
            comps[k]["p_value_adjusted"] = adj[k]
            comps[k]["multiple_testing"] = cfg.multiple_testing
        passing = [a["id"] for a in archive
                   if adj[a["id"]] is not None and adj[a["id"]] < cfg.alpha
                   and comps[a["id"]]["mean_diff"] is not None and comps[a["id"]]["mean_diff"] <= -cfg.min_effect
                   and comps[a["id"]]["ci95_mean_diff"] is not None and comps[a["id"]]["ci95_mean_diff"][1] < 0
                   and summ[a["id"]]["feasibility_rate"] >= summ[best_base]["feasibility_rate"]]
        champion = min(passing, key=lambda c: summ[c]["mean_score"]) if passing else None
        vp = [i.make() for i in val]
        profile = {"kind": vp[0].space.kind if len({q.space.kind for q in vp}) == 1 else "heterogeneous",
                   "types": sorted(set().union(*[q.space.types_present for q in vp])),
                   "dims": sorted({q.space.dim for q in vp}), "constrained": any(q.has_constraints for q in vp),
                   "suite": cfg.suite}
        for a in archive:
            self.registry.update(a["id"], benchmarks_append={"split": "validation", "budget": cfg.run_budget, "summary": summ[a["id"]],
                                                             "problem_profile": profile,
                                                             "vs_best_baseline": {"baseline": best_base, **comps[a["id"]]}},
                                 history_entry={"operation": "benchmarked", "split": "validation"})
        if champion:
            self.registry.update(champion, status="champion", history_entry={"operation": "selected as champion on validation split"})
            ci = comps[champion]["ci95_mean_diff"]
            decision = (f"{champion} beat the best baseline '{best_base}' on the validation split "
                        f"(mean score {summ[champion]['mean_score']:.4f} vs {summ[best_base]['mean_score']:.4f}, "
                        f"95% CI of paired difference [{ci[0]:+.4f}, {ci[1]:+.4f}], one-sided Wilcoxon "
                        f"p={comps[champion]['p_value_a_better']:.3g}, {cfg.multiple_testing}-adjusted "
                        f"{comps[champion]['p_value_adjusted']:.3g} < {cfg.alpha}, required improvement {cfg.min_effect})")
        else:
            decision = (f"no retained candidate beat the best baseline '{best_base}' on the validation split "
                        f"(alpha={cfg.alpha}, {cfg.multiple_testing} adjustment, minimum improvement {cfg.min_effect}, "
                        "CI below zero); the baselines are retained for ordinary use")
        self.event("selection", decision=decision)
        return {"champion": champion, "best_baseline": best_base, "decision": decision, "summary": summ,
                "comparisons": comps, "per_family": per_family(rows)}

    def _demonstrations(self, cands: list[dict]) -> dict:
        """Best generated operator and best generated algorithm composition (by training score)."""
        latest = {c["id"]: self.registry.get(c["id"]) for c in cands}
        out = {}
        for kind in ("operator", "algorithm"):
            ks = [c for c in latest.values() if c["kind"] == kind and c.get("train_score") is not None]
            if ks:
                out[kind] = min(ks, key=lambda c: c["train_score"])["id"]
        code = [c for c in latest.values() if c["representation"] == "python" and c.get("train_score") is not None]
        if code:
            out["code_operator"] = min(code, key=lambda c: c["train_score"])["id"]
        return out

    def _final_test(self, sel, demos, baselines, test, budget) -> dict:
        if not self._selection_frozen:
            raise RuntimeError("held-out test evaluation requested before selection was frozen")
        cfg = self.cfg
        ids = []
        if sel.get("champion"):
            ids.append(sel["champion"])
        for v in demos.values():  # the same candidate can be both the best operator and the best code operator
            if v not in ids:
                ids.append(v)
        specs = [builtin(b) for b in baselines] + [SolverSpec(i, record=self.registry.get(i)) for i in ids]
        try:
            rows = run_matrix(specs, test, cfg.test_run_seeds, budget, self.ledger, "test", cfg.workers)
        except BudgetExhausted as exc:
            return {"error": f"test budget insufficient: {exc}"}
        ref = reference_table([r for r in rows if r["solver"] in baselines])
        summ = summarize(rows, ref)
        best_base = min(baselines, key=lambda b: summ[b]["mean_score"])
        comps = {i: paired_comparison(rows, i, best_base, cfg.min_effect) for i in ids}
        for i in ids:
            self.registry.update(i, benchmarks_append={"split": "test", "budget": cfg.run_budget, "summary": summ[i],
                                                       "used_for_selection": False,
                                                       "vs_best_test_baseline": {"baseline": best_base, **comps[i]}},
                                 history_entry={"operation": "evaluated on held-out test split (not used for selection)"})
        return {"solvers": [s.label for s in specs], "summary": summ, "best_baseline": best_base, "comparisons": comps,
                "per_family": per_family(rows), "rows": rows}

    def _ledger_path(self) -> str:
        return os.path.join(self.cfg.registry, "test_access.jsonl")

    def _test_accesses(self, suite) -> list[dict]:
        """Earlier runs of this suite version's test split recorded next to the registry."""
        out = []
        try:
            with open(self._ledger_path()) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("suite") == suite.name and str(r.get("version")) == str(suite.version):
                        out.append(r)
        except (OSError, ValueError):
            pass
        return out

    def _record_test_access(self, suite) -> None:
        os.makedirs(self.cfg.registry, exist_ok=True)
        with open(self._ledger_path(), "a") as fh:
            fh.write(json.dumps({"suite": suite.name, "version": suite.version, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "run_dir": os.path.abspath(self.out), "seed": self.cfg.seed}) + "\n")

    def _counts(self) -> dict:
        out: dict[str, int] = {}
        for r in self.registry.list():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


def _hyperparameters(spec) -> dict:
    hp = {}

    def walk(e, path):
        if isinstance(e, dict):
            for k, v in e.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("dsl_version",):
                    hp[f"{path}{k}"] = v
                elif isinstance(v, (dict, list)):
                    walk(v, f"{path}{k}.")
        elif isinstance(e, list):
            for i, v in enumerate(e):
                walk(v, f"{path}{i}.")
    walk(spec, "")
    return hp


def _clean(o):
    from ..core.result import _jsonable
    return _jsonable(o)


def render_markdown(rep: dict) -> str:
    L = [f"# Development run report: suite `{rep['suite']['name']}`", ""]
    L.append(f"Wall time {rep['wall_time_s']} s. Stop reason: {rep['stop_reason']}. Ledger: "
             f"{rep['ledger']['used_total']} of {rep['ledger']['total_limit']} units used "
             f"({rep['ledger']['used_by_category']}).")
    L.append("")
    L.append(f"Isolation available: {rep['isolation']['available']} ({rep['isolation'].get('reason') or rep['isolation']['mechanism']}).")
    if rep["provider_notes"]:
        L.append("Provider notes: " + "; ".join(rep["provider_notes"]))
    L.append(f"Candidates by final status: {rep['candidate_counts']}")
    L += ["", "## Generations", "", "| gen | proposed | duplicates (source) | rejected | validated | duplicates (behaviour) | benchmarked | best candidate train score | best baseline train score |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in rep["generations"]:
        bs = g.get("best_train_score")
        L.append(f"| {g['generation']} | {g['proposed']} | {g['duplicates']} | {g['rejected']} | {g['validated']} | {g.get('behavioural_duplicates', 0)} | {g['benchmarked']} | "
                 f"{'n/a' if bs is None else f'{bs:.4f}'} | {g['best_baseline_train_score']:.4f} |")
    L += ["", "Scores: 0 = known optimum (or best baseline value), 1 = random-search median at the same budget, 2 = infeasible. Lower is better.", ""]
    L += ["## Selection (validation split)", "", rep["selection"]["decision"], ""]
    if rep["selection"].get("summary"):
        L += _table(rep["selection"]["summary"])
    t = rep["test"]
    L += ["", "## Held-out test split (evaluated once, after selection)", "",
          f"Suite version {rep['suite'].get('version', '1')}; test-split status before this run: "
          f"{rep['suite'].get('test_status', 'unknown')}."
          + (f" WARNING: this test split was already run {len(rep.get('test_split_previous_accesses') or [])} time(s) "
             "(see test_access.jsonl next to the registry); its results are not confirmatory."
             if rep.get("test_split_previous_accesses") else ""), ""]
    if "summary" in t:
        L += _table(t["summary"])
        L += ["", f"Best baseline on test: `{t['best_baseline']}`.", ""]
        for cid, c in t["comparisons"].items():
            p = c.get("p_value_a_better")
            ci = c.get("ci95_mean_diff")
            L.append(f"- `{cid}` vs `{t['best_baseline']}`: mean score difference {c['mean_diff']:+.4f} over {c['pairs']} paired runs "
                     f"(95% CI {'n/a' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'}; wins {c['wins']}, losses {c['losses']}, "
                     f"ties {c['ties']}; beyond {c.get('min_effect')}: {c.get('practical_wins')} better, "
                     f"{c.get('practical_losses')} worse; one-sided Wilcoxon p={'n/a' if p is None else f'{p:.3g}'})")
    else:
        L.append(str(t))
    L += ["", f"Demonstration candidates: {rep['demonstrations']}"]
    return "\n".join(L) + "\n"


def _table(summ: dict) -> list[str]:
    L = ["| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |", "|---|---|---|---|---|---|---|---|---|"]
    for s, v in sorted(summ.items(), key=lambda kv: (kv[1]["mean_score"] if kv[1]["mean_score"] is not None else 9)):
        sr = v["solved_rate"]
        L.append(f"| `{s}` | {v['runs']} | {v['mean_score']:.4f} | {v['median_score']:.4f} | {v['std_score']:.4f} | "
                 f"{v['feasibility_rate']:.2f} | {v['failure_rate']:.2f} | {'n/a' if sr is None else f'{sr:.2f}'} | {v['runtime_per_eval_ms']:.3f} |")
    return L
