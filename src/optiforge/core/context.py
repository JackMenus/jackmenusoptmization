"""Per-run search context handed to solvers."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .evaluator import Evaluation, Evaluator
from .problem import Problem
from .result import Guarantee

log = logging.getLogger("optiforge.search")


class StagnationTracker:
    """Counts iterations without a meaningful improvement of the incumbent."""

    def __init__(self, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> None:
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.best_key: tuple | None = None
        self.since_improvement = 0

    def update(self, best: Evaluation | None) -> bool:
        """Register the incumbent after an iteration; returns True if it improved."""
        if best is None:
            self.since_improvement += 1
            return False
        key = best.rank_key()
        if self.best_key is None or key[0] < self.best_key[0]:
            improved = True
        elif key[0] == self.best_key[0]:
            old, new = self.best_key[1], key[1]
            improved = new < old - max(self.abs_tol, self.rel_tol * abs(old))
        else:
            improved = False
        if improved:
            self.best_key = key
            self.since_improvement = 0
        else:
            self.since_improvement += 1
        return improved

    def state(self) -> dict:
        return {"best_key": None if self.best_key is None else list(self.best_key),
                "since_improvement": self.since_improvement}

    def restore(self, d: dict) -> None:
        self.best_key = None if d["best_key"] is None else tuple(d["best_key"])
        self.since_improvement = d["since_improvement"]


@dataclass
class SearchContext:
    problem: Problem
    evaluator: Evaluator
    rng: np.random.Generator
    seed: int | None
    warm_start: list[np.ndarray] = field(default_factory=list)
    callback: Callable[["SearchContext"], None] | None = None
    guarantee_hints: list[Guarantee] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    operator_versions: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    stagnation: StagnationTracker = field(default_factory=StagnationTracker)

    @property
    def space(self):
        return self.problem.space

    def evaluate(self, x) -> Evaluation:
        """Evaluate exactly this point on the original problem (no implicit repair)."""
        return self.evaluator.evaluate(x)

    def evaluate_many(self, X) -> list[Evaluation]:
        return [self.evaluator.evaluate(x) for x in X]

    def repair(self, x: np.ndarray) -> np.ndarray:
        """Apply domain projection and the problem's optional repair operator.

        Repair produces a *different point*; that point is what gets evaluated
        and reported, so the problem definition is never altered.
        """
        x = self.problem.space.project(x)
        if self.problem.repair is not None:
            try:
                x = self.problem.space.project(np.asarray(self.problem.repair(x), dtype=float))
            except Exception as exc:  # a failing repair must not hide the original point
                self.diagnostics["repair_errors"] = self.diagnostics.get("repair_errors", 0) + 1
                log.debug("repair failed: %s", exc)
        return x

    def notify(self) -> None:
        if self.callback is not None:
            self.callback(self)
