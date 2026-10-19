"""Structured, honest optimisation results."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


# Guarantee kinds, ordered from strongest to weakest.
GLOBAL_CERTIFIED = "global_optimum_certified"
GLOBAL_CONDITIONAL = "global_optimum_conditional_on_assumptions"
INFEASIBILITY_CERTIFIED = "infeasibility_certified"
STATIONARY = "first_order_stationary_point"
LOCAL_CONVERGED = "local_method_converged"
FEASIBLE_VERIFIED = "feasibility_verified"
FEASIBLE_SAMPLED = "feasibility_verified_at_sample_points_only"
HEURISTIC = "heuristic_best_found"
NO_FEASIBLE_FOUND = "no_feasible_solution_found"


@dataclass
class Guarantee:
    kind: str
    statement: str
    method: str = ""
    tolerance: float | None = None
    assumptions: list[str] = field(default_factory=list)
    point: list | None = None  # the encoded point this statement is about (checked by finalize)


TERMINATION_REASONS = (
    "budget_evaluations",
    "budget_time",
    "target_reached",
    "converged",
    "stagnation",
    "search_space_exhausted",
    "exact_solver_finished",
    "cancelled",
    "error",
    "completed",
)


@dataclass
class OptimizationResult:
    problem: str
    solver: str
    solver_version: str
    status: str  # "optimal_certified" | "feasible" | "infeasible_certified" | "no_feasible_found" | "error"
    x: list | None
    x_decoded: dict | None
    objective: float | None  # original sense, from the verification evaluation
    feasible: bool
    constraint_violations: dict[str, float]
    max_violation: float
    n_evaluations: int
    n_gradient_evaluations: int
    n_verification_evaluations: int
    runtime_s: float
    seed: int | None
    termination_reason: str
    history: list[list]  # [evaluation index, best objective (orig sense), feasible]
    guarantees: list[Guarantee]
    assumptions: list[str]
    operator_versions: dict[str, str] = field(default_factory=dict)
    sense: str = "min"
    noise: dict | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def certified_optimal(self) -> bool:
        return any(g.kind == GLOBAL_CERTIFIED for g in self.guarantees)

    def to_dict(self) -> dict:
        d = asdict(self)
        return _jsonable(d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    def summary(self) -> str:
        g = ", ".join(x.kind for x in self.guarantees)
        val = "n/a" if self.objective is None else f"{self.objective:.10g}"
        return (
            f"[{self.solver} {self.solver_version}] {self.problem}: status={self.status} objective={val} "
            f"feasible={self.feasible} max_violation={self.max_violation:.3g} evals={self.n_evaluations} "
            f"time={self.runtime_s:.3f}s termination={self.termination_reason} guarantees=[{g}]"
        )


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    return o


def result_from_dict(d: dict) -> OptimizationResult:
    d = dict(d)
    d["guarantees"] = [Guarantee(**g) for g in d.get("guarantees", [])]
    return OptimizationResult(**d)
