"""The unified single-objective Problem interface.

A :class:`Problem` bundles *what* is optimized and nothing about *how*:

* the variable :class:`~optiforge.core.space.Space` (types, bounds, encodings),
* the objective callable and its sense (``min``/``max``),
* constraints (inequality ``g(x) <= 0`` and equality ``|h(x)| <= tol``),
* optional analytic gradient, optional repair operator,
* declared properties (noise, evaluation kind, convexity, linear structure).

Solvers never modify a problem.  Penalties and repairs live in solvers or
operators, and every reported solution is re-evaluated against the original
objective and constraints (see :mod:`optiforge.core.finalize`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .constraints import Constraint, LinearStructure
from .errors import ProblemDefinitionError, UnsupportedProblemError
from .space import Space


EVALUATION_KINDS = ("analytic", "blackbox", "simulation")


@dataclass(frozen=True)
class KnownOptimum:
    """A verified reference optimum (used for benchmark gaps, never by solvers)."""

    value: float
    x: np.ndarray | None = None
    method: str = "unspecified"  # e.g. "analytic", "enumeration", "dynamic programming"
    tolerance: float = 0.0


@dataclass
class Problem:
    space: Space
    objective: Callable[[Any], float]
    sense: str = "min"
    constraints: Sequence[Constraint] = ()
    name: str = "problem"
    gradient: Callable[[np.ndarray], np.ndarray] | None = None
    repair: Callable[[np.ndarray], np.ndarray] | None = None
    # Optional independent check of a reported solution (e.g. a dense re-simulation).
    # Returns a dict with at least {"feasible": bool, "scope": str}.
    verifier: Callable[[np.ndarray], dict] | None = None
    noisy: bool = False
    evaluation: str = "analytic"
    input_format: str = "vector"  # "vector" (encoded ndarray) or "dict" (decoded values)
    convex: bool = False  # declared by the user; enables conditional statements only
    linear: LinearStructure | None = None
    known_optimum: KnownOptimum | None = None
    n_objectives: int = 1
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_linear(cls, space: Space, model: LinearStructure, sense: str = "min", name: str = "linear_model",
                    tol: float = 1e-9, **kw) -> "Problem":
        """A problem whose objective and constraints are generated from ``model``.

        Because the functions are the model, an exact MILP solve of such a problem can certify global optimality
        (or infeasibility) of the problem as defined. A ``linear=`` declaration on a problem with user-written
        functions is only spot-checked, and the MILP adapter then makes conditional statements only.
        """
        from .constraints import Constraint, LinearFunction
        cons = []
        if model.A_ub is not None:
            cons.append(Constraint(LinearFunction(model, "ub"), "ineq", "linear_ub", tol))
        if model.A_eq is not None:
            cons.append(Constraint(LinearFunction(model, "eq"), "eq", "linear_eq", tol))
        return cls(space, LinearFunction(model, "objective"), sense=sense, constraints=cons, name=name, linear=model,
                   **kw)

    @property
    def linear_is_exact(self) -> bool:
        """True when the objective and every constraint are generated from ``self.linear`` (see ``from_linear``)."""
        from .constraints import LinearFunction
        L = self.linear
        if L is None or self.input_format != "vector" or self.noisy:
            return False
        if not (isinstance(self.objective, LinearFunction) and self.objective.model is L
                and self.objective.part == "objective"):
            return False
        parts = []
        for c in self.constraints:
            if not (isinstance(c.fn, LinearFunction) and c.fn.model is L):
                return False
            if (c.fn.part == "ub") != (c.kind == "ineq"):
                return False
            parts.append(c.fn.part)
        return sorted(parts) == sorted((["ub"] if L.A_ub is not None else []) + (["eq"] if L.A_eq is not None else []))

    def __post_init__(self) -> None:
        self.constraints = tuple(self.constraints)
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if not isinstance(self.space, Space):
            raise ProblemDefinitionError("space must be an optiforge Space")
        if not callable(self.objective):
            raise ProblemDefinitionError("objective must be callable")
        if self.sense not in ("min", "max"):
            raise ProblemDefinitionError("sense must be 'min' or 'max'")
        if self.evaluation not in EVALUATION_KINDS:
            raise ProblemDefinitionError(f"evaluation must be one of {EVALUATION_KINDS}")
        if self.input_format not in ("vector", "dict"):
            raise ProblemDefinitionError("input_format must be 'vector' or 'dict'")
        if self.n_objectives != 1:
            raise UnsupportedProblemError(
                "the core solvers are single-objective; use optiforge.multiobjective.ScalarizedProblem "
                "or the multi-objective extension point"
            )
        for c in self.constraints:
            if not isinstance(c, Constraint):
                raise ProblemDefinitionError("constraints must be Constraint instances")
        names = [c.name for c in self.constraints]
        if len(set(names)) != len(names):
            raise ProblemDefinitionError(f"duplicate constraint names: {names}")
        if self.repair is not None and not callable(self.repair):
            raise ProblemDefinitionError("repair must be callable")
        if self.verifier is not None and not callable(self.verifier):
            raise ProblemDefinitionError("verifier must be callable")
        if self.gradient is not None:
            if not callable(self.gradient):
                raise ProblemDefinitionError("gradient must be callable")
            if self.space.kind != "continuous":
                raise UnsupportedProblemError(
                    "analytic gradients are only meaningful for purely continuous spaces"
                )
            if self.input_format != "vector":
                raise UnsupportedProblemError("gradients require input_format='vector'")
        if self.convex and self.space.kind != "continuous":
            raise UnsupportedProblemError("convexity declarations are only supported for continuous spaces")
        if self.linear is not None:
            c = np.asarray(self.linear.c, dtype=float)
            if c.shape != (self.space.dim,):
                raise ProblemDefinitionError("linear.c must have one entry per encoded position")
            if self.space.perm_blocks or self.space.cat_mask.any():
                raise UnsupportedProblemError("linear structure is not supported with permutation/categorical variables")

    # ------------------------------------------------------------------
    @property
    def has_constraints(self) -> bool:
        return len(self.constraints) > 0

    @property
    def has_equality(self) -> bool:
        return any(c.kind == "eq" for c in self.constraints)

    @property
    def is_deterministic(self) -> bool:
        return not self.noisy

    def to_internal(self, f: float) -> float:
        """Map an original-sense objective value to the internal minimisation value."""
        return f if self.sense == "min" else -f

    def from_internal(self, v: float) -> float:
        return v if self.sense == "min" else -v

    def call_objective(self, x: np.ndarray) -> float:
        arg = self.space.decode(x) if self.input_format == "dict" else x
        return float(self.objective(arg))

    def call_constraint(self, c: Constraint, x: np.ndarray) -> np.ndarray:
        arg = self.space.decode(x) if self.input_format == "dict" else x
        return c.raw(arg)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "sense": self.sense,
            "space_kind": self.space.kind,
            "dim": self.space.dim,
            "types": self.space.type_counts,
            "n_constraints": len(self.constraints),
            "n_equality": sum(c.kind == "eq" for c in self.constraints),
            "gradient": self.gradient is not None,
            "noisy": self.noisy,
            "evaluation": self.evaluation,
            "convex_declared": self.convex,
            "linear_declared": self.linear is not None,
            "cardinality": self.space.cardinality(),
        }
