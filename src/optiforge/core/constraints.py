"""Constraint definitions.

Inequality constraints use the convention ``g(x) <= 0``; equality constraints
``h(x) = 0`` are satisfied when ``|h(x)| <= tol``.  A constraint function may
return a scalar or an array (several constraints of the same kind at once).
Violations are always measured against these original definitions; solvers
that use penalties do so on their own internal copy of the fitness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .errors import ProblemDefinitionError


@dataclass(frozen=True)
class Constraint:
    fn: Callable
    kind: str = "ineq"  # "ineq" (g(x) <= 0) or "eq" (|h(x)| <= tol)
    name: str = "c"
    tol: float = 1e-6
    gradient: Callable | None = None  # optional Jacobian of fn

    def __post_init__(self) -> None:
        if self.kind not in ("ineq", "eq"):
            raise ProblemDefinitionError(f"constraint '{self.name}': kind must be 'ineq' or 'eq'")
        if not callable(self.fn):
            raise ProblemDefinitionError(f"constraint '{self.name}': fn must be callable")
        if not (self.tol >= 0):
            raise ProblemDefinitionError(f"constraint '{self.name}': tol must be >= 0")

    def raw(self, x) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.fn(x), dtype=float)).ravel()

    def violation(self, values: np.ndarray) -> np.ndarray:
        """Non-negative violation amounts (0 when satisfied within tolerance)."""
        if self.kind == "ineq":
            v = np.maximum(values, 0.0)
            return np.where(v <= self.tol, 0.0, v)
        a = np.abs(values)
        return np.where(a <= self.tol, 0.0, a)


def ineq(fn: Callable, name: str = "g", tol: float = 1e-6, gradient: Callable | None = None) -> Constraint:
    return Constraint(fn, "ineq", name, tol, gradient)


def eq(fn: Callable, name: str = "h", tol: float = 1e-6, gradient: Callable | None = None) -> Constraint:
    return Constraint(fn, "eq", name, tol, gradient)


@dataclass(frozen=True)
class LinearStructure:
    """Optional declaration that objective and constraints are linear.

    ``minimize c @ x`` (in the problem's original sense) subject to
    ``A_ub @ x <= b_ub`` and ``A_eq @ x == b_eq``.  This enables the MILP
    adapter.  The platform cross-checks the declaration against the actual
    objective/constraint functions before trusting it.
    """

    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    offset: float = 0.0
