"""Objective/constraint evaluation with budget accounting.

The evaluator is the single gateway between search code and the user's
problem.  It

* rejects points outside the declared domain (counted, never passed to the
  user's functions, and reported as invalid),
* evaluates the *original* objective and every constraint,
* catches exceptions and non-finite values and records them as failed
  evaluations instead of crashing the search,
* counts objective and gradient evaluations and enforces the solve budget,
* tracks the incumbent using feasibility-first ordering (Deb's rules).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .budget import Budget, BudgetLedger, CancelToken
from .errors import BudgetExhausted, SearchCancelled
from .problem import Problem


@dataclass
class Evaluation:
    x: np.ndarray
    f: float  # objective in the problem's original sense (nan if unavailable)
    fmin: float  # internal minimisation value (inf if unavailable)
    violations: np.ndarray  # one non-negative entry per scalar constraint
    total_violation: float
    feasible: bool
    domain_ok: bool = True
    error: str | None = None
    index: int = -1

    @property
    def valid(self) -> bool:
        return self.domain_ok and self.error is None

    def rank_key(self) -> tuple:
        """Feasibility-first ordering: feasible by value, then infeasible by violation, then invalid."""
        if not self.valid:
            return (2, math.inf)
        if self.feasible:
            return (0, self.fmin)
        return (1, self.total_violation)

    def better_than(self, other: "Evaluation | None") -> bool:
        return other is None or self.rank_key() < other.rank_key()


@dataclass
class EvaluatorStats:
    n_evaluations: int = 0
    n_gradient_evaluations: int = 0
    n_errors: int = 0
    n_nonfinite: int = 0
    n_domain_rejections: int = 0
    n_feasible: int = 0


class Evaluator:
    def __init__(
        self,
        problem: Problem,
        budget: Budget | None = None,
        ledger: BudgetLedger | None = None,
        ledger_category: str = "evaluations",
        cancel: CancelToken | None = None,
        record_trace: bool = False,
    ) -> None:
        self.problem = problem
        self.budget = budget or Budget()
        self.ledger = ledger
        self.ledger_category = ledger_category
        self.cancel = cancel
        self.stats = EvaluatorStats()
        self.best: Evaluation | None = None
        self.history: list[tuple[int, float, bool]] = []  # (eval count, best value (orig sense), feasible)
        self.trace: list[tuple[np.ndarray, float]] | None = [] if record_trace else None
        self.t0 = time.perf_counter()
        self.time_offset = 0.0  # elapsed time carried over from a checkpoint
        self._target_hit = False
        # Optional lower evaluation cap used by composite solvers to run one stage
        # (e.g. global exploration) on a share of the budget; raises "stage_budget".
        self.stage_limit: int | None = None

    # ------------------------------------------------------------------
    @property
    def n_evals(self) -> int:
        return self.stats.n_evaluations

    def elapsed(self) -> float:
        return self.time_offset + time.perf_counter() - self.t0

    def remaining(self) -> int:
        cap = 2**62 if self.budget.max_evaluations is None else self.budget.max_evaluations
        if self.stage_limit is not None:
            cap = min(cap, self.stage_limit)
        return max(cap - self.stats.n_evaluations, 0)

    def exhausted_reason(self) -> str | None:
        if self.cancel is not None and self.cancel.cancelled:
            return "cancelled"
        if self.budget.max_evaluations is not None and self.stats.n_evaluations >= self.budget.max_evaluations:
            return "budget_evaluations"
        if self.budget.max_time is not None and self.elapsed() >= self.budget.max_time:
            return "budget_time"
        if self._target_hit:
            return "target_reached"
        if self.stage_limit is not None and self.stats.n_evaluations >= self.stage_limit:
            return "stage_budget"
        return None

    def _gate(self) -> None:
        reason = self.exhausted_reason()
        if reason == "cancelled":
            raise SearchCancelled("search cancelled")
        if reason is not None:
            raise BudgetExhausted(reason)
        if self.ledger is not None:
            self.ledger.charge(self.ledger_category, 1)

    # ------------------------------------------------------------------
    def evaluate(self, x) -> Evaluation:
        """Evaluate one encoded point (raises BudgetExhausted / SearchCancelled when out of budget)."""
        self._gate()
        x = np.array(x, dtype=float, copy=True)
        self.stats.n_evaluations += 1
        idx = self.stats.n_evaluations
        p = self.problem
        n_c = sum(1 for _ in p.constraints)
        dom = p.space.domain_violations(x) if x.shape == (p.space.dim,) else ["bad shape"]
        if dom:
            self.stats.n_domain_rejections += 1
            ev = Evaluation(x, math.nan, math.inf, np.full(max(n_c, 0), math.nan), math.inf, False, False,
                            "domain: " + "; ".join(dom[:3]), idx)
            self._record(ev)
            return ev
        try:
            f = p.call_objective(x)
        except Exception as exc:  # user code failure is data, not a crash
            self.stats.n_errors += 1
            ev = Evaluation(x, math.nan, math.inf, np.zeros(0), math.inf, False, True,
                            f"objective raised {type(exc).__name__}: {exc}", idx)
            self._record(ev)
            return ev
        if not math.isfinite(f):
            self.stats.n_nonfinite += 1
            ev = Evaluation(x, f, math.inf, np.zeros(0), math.inf, False, True, "objective non-finite", idx)
            self._record(ev)
            return ev
        viols = []
        try:
            for c in p.constraints:
                viols.append(c.violation(p.call_constraint(c, x)))
            v = np.concatenate(viols) if viols else np.zeros(0)
        except Exception as exc:
            self.stats.n_errors += 1
            ev = Evaluation(x, f, math.inf, np.zeros(0), math.inf, False, True,
                            f"constraint raised {type(exc).__name__}: {exc}", idx)
            self._record(ev)
            return ev
        if not np.all(np.isfinite(v)):
            ev = Evaluation(x, f, math.inf, v, math.inf, False, True, "constraint non-finite", idx)
            self.stats.n_nonfinite += 1
            self._record(ev)
            return ev
        tv = float(v.sum())
        feasible = bool(tv == 0.0)
        ev = Evaluation(x, f, p.to_internal(f), v, tv, feasible, True, None, idx)
        if feasible:
            self.stats.n_feasible += 1
        self._record(ev)
        return ev

    def _record(self, ev: Evaluation) -> None:
        if self.trace is not None:
            self.trace.append((ev.x, ev.fmin if ev.valid else math.inf))
        if ev.better_than(self.best):
            self.best = ev
            self.history.append((ev.index, ev.f if ev.valid else math.nan, ev.feasible))
            t = self.budget.target
            if t is not None and ev.feasible:
                if (self.problem.sense == "min" and ev.f <= t) or (self.problem.sense == "max" and ev.f >= t):
                    self._target_hit = True

    def gradient(self, x) -> np.ndarray:
        """Analytic gradient of the *internal* (minimisation) objective."""
        if self.problem.gradient is None:
            raise ValueError("problem has no analytic gradient")
        self.stats.n_gradient_evaluations += 1
        g = np.asarray(self.problem.gradient(np.asarray(x, dtype=float)), dtype=float)
        return g if self.problem.sense == "min" else -g

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Serializable state for checkpoints."""
        return {
            "stats": self.stats.__dict__.copy(),
            "best": None if self.best is None else _ev_to_dict(self.best),
            "history": [list(h) for h in self.history],
            "elapsed": self.elapsed(),
        }

    def restore(self, snap: dict) -> None:
        self.stats = EvaluatorStats(**snap["stats"])
        self.best = None if snap["best"] is None else _ev_from_dict(snap["best"])
        self.history = [tuple(h) for h in snap["history"]]
        self.time_offset = snap["elapsed"]
        self.t0 = time.perf_counter()


def _ev_to_dict(e: Evaluation) -> dict:
    return {"x": e.x.tolist(), "f": e.f, "fmin": e.fmin, "violations": e.violations.tolist(),
            "total_violation": e.total_violation, "feasible": e.feasible, "domain_ok": e.domain_ok,
            "error": e.error, "index": e.index}


def _ev_from_dict(d: dict) -> Evaluation:
    return Evaluation(np.array(d["x"]), d["f"], d["fmin"], np.array(d["violations"]), d["total_violation"],
                      d["feasible"], d["domain_ok"], d["error"], d["index"])
