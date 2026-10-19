"""High-level Python API."""
from __future__ import annotations

import time

import numpy as np

from .core.budget import Budget, CancelToken
from .core.evaluator import Evaluator
from .core.problem import Problem
from .core.result import OptimizationResult
from .selection.rules import SolverPlan, select_by_rules
from .solvers import get_solver


def estimate_eval_cost(problem: Problem, n: int = 3, seed: int = 0) -> float:
    """Time a few evaluations at random points (these are *not* charged to a solve budget)."""
    ev = Evaluator(problem, Budget(max_evaluations=n))
    rng = np.random.default_rng(seed)
    t = time.perf_counter()
    for x in problem.space.sample(rng, n):
        ev.evaluate(x)
    return (time.perf_counter() - t) / n


def plan(problem: Problem, budget: Budget | None = None, mode: str = "auto", registry=None,
         probe_cost: bool = True) -> SolverPlan:
    budget = budget or Budget()
    cost = estimate_eval_cost(problem) if probe_cost else None
    p = select_by_rules(problem, budget, cost, mode=mode)
    if registry is not None:
        from .selection.evidence import EvidenceSelector
        p = EvidenceSelector(registry).refine(problem, budget, p)
    return p


def solve(problem: Problem, solver: str = "auto", budget: Budget | int | None = None, seed: int | None = 0,
          mode: str = "auto", solver_params: dict | None = None, registry=None, **kw) -> OptimizationResult:
    """Solve ``problem``.  ``solver='auto'`` uses the transparent selection rules (and, if a
    registry is passed, validated benchmark evidence).  Extra keyword arguments are passed
    to :meth:`Solver.solve` (``warm_start``, ``cancel``, ``checkpoint_path``, ``resume``...)."""
    if isinstance(budget, int):
        budget = Budget(max_evaluations=budget)
    budget = budget or Budget()
    chosen = None
    if solver == "auto":
        chosen = plan(problem, budget, mode=mode, registry=registry)
        solver, params = chosen.solver, {**chosen.params, **(solver_params or {})}
    else:
        params = solver_params or {}
    if solver.startswith("candidate:"):
        from .generation.materialize import solver_from_registry
        s = solver_from_registry(registry, solver.split(":", 1)[1])
    else:
        s = get_solver(solver, **params)
    res = s.solve(problem, budget, seed=seed, **kw)
    if chosen is not None:
        res.diagnostics["plan"] = chosen.to_dict()
    return res


__all__ = ["solve", "plan", "estimate_eval_cost", "Budget", "CancelToken"]
