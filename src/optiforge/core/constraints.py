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


class LinearFunction:
    """``x -> c @ x + offset`` built from a :class:`LinearStructure`.

    Problems made with :meth:`Problem.from_linear` use these objects as their
    objective and constraints, so the declared model *is* the problem. Only then
    can an exact MILP solve certify optimality or infeasibility.
    """

    def __init__(self, model: LinearStructure, part: str) -> None:
        self.model, self.part = model, part
        if part == "objective":
            self.M, self.b = np.array(model.c, dtype=float)[None, :], np.array([-float(model.offset)])
        elif part == "ub":
            self.M, self.b = np.array(model.A_ub, dtype=float, ndmin=2), np.array(model.b_ub, dtype=float, ndmin=1)
        elif part == "eq":
            self.M, self.b = np.array(model.A_eq, dtype=float, ndmin=2), np.array(model.b_eq, dtype=float, ndmin=1)
        else:
            raise ValueError(part)
        # private copies: later in-place edits of the declaration cannot make the model and the functions disagree
        self.M.setflags(write=False)
        self.b.setflags(write=False)

    def __call__(self, x):
        v = self.M @ np.asarray(x, dtype=float) - self.b
        return float(v[0]) if self.part == "objective" else v
