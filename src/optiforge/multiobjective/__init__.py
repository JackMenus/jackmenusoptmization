"""Extension point for multi-objective optimisation.

The core solvers are single-objective.  This module provides the pieces a
multi-objective layer builds on, without pretending the core is
multi-objective:

* :class:`MultiObjectiveProblem` - several objectives over one space/constraint set,
* :class:`ScalarizedProblem` - weighted-sum or augmented Chebyshev scalarisation
  producing an ordinary :class:`Problem` (each scalarisation yields one point;
  weighted sums cannot reach non-convex parts of a Pareto front),
* :func:`pareto_front` - non-dominated filtering of evaluated objective vectors,
* :func:`sweep` - run a solver over several weight vectors and collect the
  non-dominated set (an approximation, with no completeness claim).

Native multi-objective solvers (NSGA-II, MOEA/D, hypervolume indicators) are
not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.constraints import Constraint
from ..core.problem import Problem
from ..core.space import Space


@dataclass
class MultiObjectiveProblem:
    space: Space
    objectives: Sequence[Callable]
    senses: Sequence[str] | None = None
    constraints: Sequence[Constraint] = ()
    name: str = "mo-problem"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.objectives) < 2:
            raise ValueError("a multi-objective problem needs at least two objectives")
        self.senses = list(self.senses or ["min"] * len(self.objectives))

    def values(self, x) -> np.ndarray:
        """Objective vector in minimisation form."""
        return np.array([f(x) if s == "min" else -f(x) for f, s in zip(self.objectives, self.senses)], dtype=float)


def ScalarizedProblem(mo: MultiObjectiveProblem, weights, method: str = "weighted_sum", ideal=None,
                      rho: float = 1e-4) -> Problem:
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(mo.objectives),) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per objective, with positive sum")
    w = w / w.sum()
    z = np.zeros(len(w)) if ideal is None else np.asarray(ideal, dtype=float)
    if method == "weighted_sum":
        f = lambda x: float(w @ mo.values(x))
    elif method == "chebyshev":
        f = lambda x: float(np.max(w * (mo.values(x) - z)) + rho * np.sum(w * (mo.values(x) - z)))
    else:
        raise ValueError("method must be 'weighted_sum' or 'chebyshev'")
    return Problem(mo.space, f, constraints=mo.constraints, name=f"{mo.name}[{method}:{np.round(w, 3).tolist()}]",
                   metadata={"scalarization": method, "weights": w.tolist(), "multiobjective_source": mo.name})


def pareto_front(F: np.ndarray) -> np.ndarray:
    """Indices of non-dominated rows of F (minimisation)."""
    F = np.asarray(F, dtype=float)
    keep = []
    for i in range(len(F)):
        dominated = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if not dominated:
            keep.append(i)
    return np.array(keep, dtype=int)


def sweep(mo: MultiObjectiveProblem, solver_factory, weight_vectors, budget, seed: int = 0, method: str = "chebyshev"):
    points, values, results = [], [], []
    for k, w in enumerate(weight_vectors):
        r = solver_factory().solve(ScalarizedProblem(mo, w, method), budget, seed=seed + k)
        results.append(r)
        if r.x is not None and r.feasible:
            points.append(np.array(r.x))
            values.append(mo.values(np.array(r.x)))
    if not values:
        return [], np.zeros((0, len(mo.objectives))), results
    V = np.array(values)
    idx = pareto_front(V)
    return [points[i] for i in idx], V[idx], results
