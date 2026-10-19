"""Problem-characteristic analysis used to steer candidate proposals and solver selection.

Static features come from the problem declaration.  Probe features use a few
objective evaluations (random samples and a short random walk); those
evaluations are charged to the development ledger under ``analysis``.
"""
from __future__ import annotations

import time

import numpy as np

from ..core.budget import Budget, BudgetLedger
from ..core.evaluator import Evaluator
from ..core.problem import Problem
from ..operators.primitives import random_neighbor


def static_features(p: Problem) -> dict:
    d = p.describe()
    d["kind"] = d.pop("space_kind")
    d["has_repair"] = p.repair is not None
    return d


def probe_features(p: Problem, rng: np.random.Generator, n_samples: int = 40, walk: int = 40,
                   ledger: BudgetLedger | None = None) -> dict:
    ev = Evaluator(p, Budget(max_evaluations=n_samples + walk + 1), ledger=ledger, ledger_category="analysis")
    t0 = time.perf_counter()
    samples = [ev.evaluate(x) for x in p.space.sample(rng, n_samples)]
    cost = (time.perf_counter() - t0) / max(n_samples, 1)
    vals = np.array([s.fmin for s in samples if s.valid and s.feasible])
    feas = float(np.mean([s.feasible for s in samples]))
    x = p.space.sample(rng)
    seq = []
    for _ in range(walk):
        e = ev.evaluate(x)
        if e.valid:
            seq.append(e.fmin + 0.0 * e.total_violation)
        x = random_neighbor(x, p.space, rng)
    seq = np.array(seq)
    rugged = None
    if len(seq) > 5 and np.std(seq) > 0:
        r1 = np.corrcoef(seq[:-1], seq[1:])[0, 1]
        rugged = float(np.clip(1 - r1, 0, 1)) if np.isfinite(r1) else None
    return {
        "feasible_fraction": feas,
        "objective_spread": float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)) if len(vals) > 1 else None,
        "ruggedness": rugged,
        "smoothness": None if rugged is None else 1 - rugged,
        "eval_cost_s": cost,
        "error_fraction": float(np.mean([not s.valid for s in samples])),
        "probe_evaluations": ev.n_evals,
    }


def analyze(problems: list[Problem], rng: np.random.Generator, ledger: BudgetLedger | None = None,
            n_samples: int = 40, walk: int = 40) -> dict:
    """Aggregate features over a set of (training) problems."""
    stat = [static_features(p) for p in problems]
    probes = [probe_features(p, rng, n_samples, walk, ledger) for p in problems]
    types = set()
    for p in problems:
        types |= p.space.types_present
    agg = {"n_problems": len(problems), "types": sorted(types),
           "kinds": sorted({s["kind"] for s in stat}),
           "kind": stat[0]["kind"] if len({s["kind"] for s in stat}) == 1 else "heterogeneous",
           "max_dim": max(s["dim"] for s in stat), "any_constraints": any(s["n_constraints"] for s in stat),
           "any_noise": any(s["noisy"] for s in stat), "has_repair": any(s["has_repair"] for s in stat)}
    for k in ("feasible_fraction", "ruggedness", "smoothness", "eval_cost_s", "error_fraction"):
        v = [q[k] for q in probes if q[k] is not None]
        agg[k] = float(np.mean(v)) if v else None
    agg["per_problem"] = [{**s, **q} for s, q in zip(stat, probes)]
    return agg
