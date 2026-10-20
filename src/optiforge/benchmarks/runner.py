"""Benchmark execution and scoring.

Scoring rules (fixed here, outside candidate control)
-----------------------------------------------------
For an instance with reference value ``f_ref`` (verified optimum when known,
otherwise the best value any *baseline* reached in the reference runs, marked
as unverified) and random-search median ``f_rand`` at the same evaluation
budget, a run that ends with internal (minimisation) value ``f`` scores

    s = clip((f - f_ref) / (f_rand - f_ref), -1, 2)

so 0 = reference reached, 1 = random-search level; runs without a feasible
result score 2.  Lower is better.  Everything is computed on the problem's
original objective and constraints as re-evaluated by ``finalize``.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..core.budget import Budget, BudgetLedger
from ..core.errors import BudgetExhausted
from .suites import InstanceSpec

INFEASIBLE_SCORE = 2.0


@dataclass(frozen=True)
class SolverSpec:
    """Picklable description of a solver: a built-in by name, or a registry record."""

    label: str
    builtin: str | None = None
    params: tuple = ()
    record: dict | None = None

    def build(self):
        if self.builtin is not None:
            from ..solvers import get_solver
            return get_solver(self.builtin, **dict(self.params))
        from ..generation.materialize import build_solver
        return build_solver(self.record, allow_unvalidated=True)


def builtin(name: str, label: str | None = None, **params) -> SolverSpec:
    return SolverSpec(label or name, builtin=name, params=tuple(sorted(params.items())))


def _one(args) -> dict:
    spec, inst, seed, budget = args
    t = time.perf_counter()
    try:
        p = inst.make()
        if spec.builtin == "auto":  # benchmark the rule-based selection layer itself
            from ..selection.rules import select_by_rules
            from ..solvers import get_solver
            plan = select_by_rules(p, budget)
            s = get_solver(plan.solver, **plan.params)
        else:
            s = spec.build()
        inc = s.incompatibilities(p)
        if inc:
            return {"solver": spec.label, "instance": inst.id, "family": inst.family, "seed": seed, "status": "incompatible",
                    "error": "; ".join(inc), "n_evaluations": 0, "runtime_s": 0.0, "feasible": False, "objective": None}
        r = s.solve(p, budget, seed=seed)
        ko = p.known_optimum
        return {
            "solver": spec.label, "instance": inst.id, "family": inst.family, "seed": seed, "status": r.status,
            "termination": r.termination_reason, "objective": r.objective, "feasible": bool(r.feasible),
            "internal": None if r.objective is None else p.to_internal(r.objective), "n_evaluations": r.n_evaluations,
            "runtime_s": r.runtime_s, "wall_s": time.perf_counter() - t, "sense": p.sense,
            "known_optimum": None if ko is None else ko.value, "known_internal": None if ko is None else p.to_internal(ko.value),
            "known_method": None if ko is None else ko.method, "error": next((m for m in r.messages if "solver error" in m), None),
            "guarantees": [g.kind for g in r.guarantees], "solver_version": r.solver_version,
        }
    except Exception as exc:  # infrastructure failure for this run (counted as a failure)
        return {"solver": spec.label, "instance": inst.id, "family": inst.family, "seed": seed, "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "n_evaluations": 0, "runtime_s": time.perf_counter() - t,
                "feasible": False, "objective": None}


def run_matrix(solvers: list[SolverSpec], instances: list[InstanceSpec], seeds: list[int], budget: Budget,
               ledger: BudgetLedger | None = None, category: str = "benchmark", workers: int | None = None) -> list[dict]:
    jobs = [(s, i, sd, budget) for s in solvers for i in instances for sd in seeds]
    reserve = len(jobs) * (budget.max_evaluations or 0)
    if ledger is not None:
        ledger.reserve(category, reserve)
    try:
        workers = workers if workers is not None else min(len(jobs), os.cpu_count() or 1)
        if workers <= 1 or len(jobs) <= 1:
            rows = [_one(j) for j in jobs]
        else:
            with mp.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_one, jobs, chunksize=1)
    finally:
        if ledger is not None:
            used = sum(r.get("n_evaluations", 0) for r in rows) if "rows" in locals() else reserve
            ledger.settle(category, reserve, used)
    return rows


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def reference_table(baseline_rows: list[dict]) -> dict:
    """Per-instance reference values from baseline runs (must include random_search)."""
    ref: dict[str, dict] = {}
    by_inst: dict[str, list] = {}
    for r in baseline_rows:
        by_inst.setdefault(r["instance"], []).append(r)
    for inst, rows in by_inst.items():
        rand = [r["internal"] for r in rows if r["solver"] == "random_search" and r.get("feasible")]
        feas = [r["internal"] for r in rows if r.get("feasible")]
        known = next((r for r in rows if r.get("known_internal") is not None), None)
        if known is not None:
            f_ref, ref_kind = known["known_internal"], f"known optimum ({known.get('known_method') or 'unspecified'})"
        elif feas:
            f_ref, ref_kind = min(feas), "best baseline value (not verified optimal)"
        else:
            f_ref, ref_kind = None, "none"
        ref[inst] = {"f_ref": f_ref, "ref_kind": ref_kind,
                     "f_rand": float(np.median(rand)) if rand else (max(feas) if feas else None)}
    return ref


def score_row(r: dict, ref: dict) -> float:
    if not r.get("feasible") or r.get("internal") is None:
        return INFEASIBLE_SCORE
    rf = ref.get(r["instance"])
    if rf is None or rf["f_ref"] is None or rf["f_rand"] is None:
        return math.nan
    d = rf["f_rand"] - rf["f_ref"]
    if d <= 1e-12 * (1 + abs(rf["f_ref"])):
        return 0.0 if r["internal"] <= rf["f_ref"] + 1e-9 * (1 + abs(rf["f_ref"])) else 1.0
    return float(np.clip((r["internal"] - rf["f_ref"]) / d, -1.0, INFEASIBLE_SCORE))


def summarize(rows: list[dict], ref: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        r["score"] = score_row(r, ref)
        if r.get("known_internal") is not None and r.get("internal") is not None and r.get("feasible"):
            g = r["internal"] - r["known_internal"]
            r["gap"] = g
            r["solved"] = bool(g <= 1e-6 * (1 + abs(r["known_internal"])))
    by: dict[str, list] = {}
    na: dict[str, int] = {}
    for r in rows:
        if r.get("status") == "incompatible":  # solver not applicable to this problem class
            na[r["solver"]] = na.get(r["solver"], 0) + 1
            continue
        by.setdefault(r["solver"], []).append(r)
    for s, rs in by.items():
        sc = np.array([r["score"] for r in rs if not math.isnan(r["score"])])
        gaps = [r["gap"] for r in rs if "gap" in r]
        solved = [r["solved"] for r in rs if "solved" in r]
        out[s] = {
            "runs": len(rs),
            "mean_score": float(sc.mean()) if sc.size else None,
            "median_score": float(np.median(sc)) if sc.size else None,
            "std_score": float(sc.std(ddof=1)) if sc.size > 1 else 0.0,
            "iqr_score": float(np.subtract(*np.percentile(sc, [75, 25]))) if sc.size else None,
            "feasibility_rate": float(np.mean([r.get("feasible", False) for r in rs])),
            "failure_rate": float(np.mean([r.get("status") in ("error", "incompatible") or r.get("termination") == "error" for r in rs])),
            "solved_rate": float(np.mean(solved)) if solved else None,
            "median_gap": float(np.median(gaps)) if gaps else None,
            "mean_runtime_s": float(np.mean([r.get("runtime_s", 0.0) for r in rs])),
            "mean_evaluations": float(np.mean([r.get("n_evaluations", 0) for r in rs])),
            "runtime_per_eval_ms": float(1e3 * np.sum([r.get("runtime_s", 0.0) for r in rs]) / max(1, np.sum([r.get("n_evaluations", 0) for r in rs]))),
            "not_applicable_runs": na.get(s, 0),
        }
    return out


def per_family(rows: list[dict]) -> dict[str, dict[str, float]]:
    t: dict[str, dict[str, list]] = {}
    for r in rows:
        if "score" not in r or r.get("status") == "incompatible":  # not applicable: shown as a dash, not as a failure
            continue
        t.setdefault(r["family"], {}).setdefault(r["solver"], []).append(r["score"])
    return {f: {s: float(np.nanmean(v)) for s, v in d.items()} for f, d in t.items()}


PRACTICAL_EFFECT = 0.01  # score units (1 = random-search level); smaller paired differences count as practical ties


def paired_comparison(rows: list[dict], a: str, b: str, min_effect: float = PRACTICAL_EFFECT,
                      n_boot: int = 2000) -> dict:
    """Paired comparison of solver a vs b on matched (instance, seed) pairs; one-sided Wilcoxon (a better).

    Besides the test it reports effect sizes, so statistical and practical significance are kept apart:
    ``ci95_mean_diff`` (percentile bootstrap over pairs, fixed seed), ``rank_biserial`` (matched-pairs rank-biserial
    correlation, -1 = a always better) and practical wins/losses/ties at ``min_effect`` score units.
    """
    # not-applicable runs are excluded rather than counted as failures
    sa = {(r["instance"], r["seed"]): r.get("score", math.nan) for r in rows if r["solver"] == a and r.get("status") != "incompatible"}
    sb = {(r["instance"], r["seed"]): r.get("score", math.nan) for r in rows if r["solver"] == b and r.get("status") != "incompatible"}
    keys = sorted(set(sa) & set(sb))
    da = np.array([sa[k] for k in keys])
    db = np.array([sb[k] for k in keys])
    ok = ~(np.isnan(da) | np.isnan(db))
    da, db = da[ok], db[ok]
    diff = da - db
    res = {"pairs": int(len(diff)), "mean_diff": float(diff.mean()) if len(diff) else None,
           "wins": int(np.sum(diff < -1e-9)), "losses": int(np.sum(diff > 1e-9)), "ties": int(np.sum(np.abs(diff) <= 1e-9))}
    res["min_effect"] = min_effect
    res["practical_wins"] = int(np.sum(diff < -min_effect))
    res["practical_losses"] = int(np.sum(diff > min_effect))
    res["practical_ties"] = int(np.sum(np.abs(diff) <= min_effect))
    res["median_diff"] = float(np.median(diff)) if len(diff) else None
    if len(diff) >= 2:
        boot_rng = np.random.default_rng(12345)
        idx = boot_rng.integers(0, len(diff), (n_boot, len(diff)))
        means = diff[idx].mean(axis=1)
        res["ci95_mean_diff"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    else:
        res["ci95_mean_diff"] = None
    nz = diff[np.abs(diff) > 1e-12]
    if nz.size:
        ranks = stats.rankdata(np.abs(nz))
        res["rank_biserial"] = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    else:
        res["rank_biserial"] = 0.0
    if len(diff) >= 5 and np.any(np.abs(diff) > 1e-12):
        try:
            w = stats.wilcoxon(da, db, alternative="less", zero_method="zsplit")
            res["p_value_a_better"] = float(w.pvalue)
        except ValueError:
            res["p_value_a_better"] = None
    else:
        res["p_value_a_better"] = None
    return res


def holm_adjust(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (family-wise error control); None entries stay None."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m = len(items)
    out, running = {k: None for k in pvalues}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
