"""Solver interface, iterative-solver template, and the solver registry."""
from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..core.budget import Budget, BudgetLedger, CancelToken
from ..core.checkpoint import CheckpointManager
from ..core.context import SearchContext
from ..core.errors import BudgetExhausted, SearchCancelled, UnsupportedProblemError
from ..core.evaluator import Evaluator
from ..core.finalize import finalize
from ..core.problem import Problem
from ..core.result import Guarantee, OptimizationResult

log = logging.getLogger("optiforge.solvers")

SOLVERS: dict[str, type["Solver"]] = {}


def _plain(obj):
    """JSON-normalised copy, so parameters compare equal after a checkpoint round trip."""
    import json
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def register_solver(cls):
    SOLVERS[cls.name] = cls
    return cls


def get_solver(name: str, **params) -> "Solver":
    if name not in SOLVERS:
        raise KeyError(f"unknown solver '{name}'. Available: {sorted(SOLVERS)}")
    return SOLVERS[name](**params)


class Solver(ABC):
    """Base class.  Subclasses implement :meth:`_run` and declare capabilities."""

    name = "base"
    version = "1.0.0"
    family = "heuristic"  # "local" | "global" | "hybrid" | "exact" | "heuristic"
    space_kinds: frozenset = frozenset({"continuous", "discrete", "mixed"})
    supports_permutations = True
    supports_categorical = True
    supports_constraints = True
    supports_equality = True
    requires_gradient = False
    supports_noise = True
    supports_checkpoint = False
    description = ""

    def __init__(self, **params) -> None:
        defaults = self.default_params()
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}; valid: {sorted(defaults)}")
        self.params = {**defaults, **params}

    @classmethod
    def default_params(cls) -> dict:
        return {}

    # ------------------------------------------------------------------
    def incompatibilities(self, problem: Problem) -> list[str]:
        r = []
        sp = problem.space
        if sp.kind not in self.space_kinds:
            r.append(f"{self.name} does not support {sp.kind} spaces")
        if sp.perm_blocks and not self.supports_permutations:
            r.append(f"{self.name} does not support permutation variables")
        if sp.cat_mask.any() and not self.supports_categorical:
            r.append(f"{self.name} does not support categorical variables")
        if problem.has_constraints and not self.supports_constraints:
            r.append(f"{self.name} does not support constraints")
        if problem.has_equality and not self.supports_equality:
            r.append(f"{self.name} does not support equality constraints")
        if self.requires_gradient and problem.gradient is None:
            r.append(f"{self.name} requires an analytic gradient")
        if problem.noisy and not self.supports_noise:
            r.append(f"{self.name} assumes a deterministic objective")
        return r

    def check(self, problem: Problem) -> None:
        r = self.incompatibilities(problem)
        if r:
            raise UnsupportedProblemError("; ".join(r))

    # ------------------------------------------------------------------
    def solve(
        self,
        problem: Problem,
        budget: Budget | None = None,
        seed: int | None = 0,
        warm_start=None,
        cancel: CancelToken | None = None,
        callback: Callable[[SearchContext], None] | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 10,
        resume: bool = False,
        ledger: BudgetLedger | None = None,
        ledger_category: str = "evaluations",
        record_trace: bool = False,
    ) -> OptimizationResult:
        self.check(problem)
        budget = budget or Budget()
        evaluator = Evaluator(problem, budget, ledger=ledger, ledger_category=ledger_category, cancel=cancel,
                              record_trace=record_trace)
        ws = []
        if warm_start is not None:
            arr = np.atleast_2d(np.asarray(warm_start, dtype=float))
            for w in arr:
                if w.shape != (problem.space.dim,):
                    raise ValueError("warm start has the wrong dimension")
                ws.append(problem.space.project(w))
        ctx = SearchContext(problem, evaluator, np.random.default_rng(seed), seed, ws, callback)
        ctx.operator_versions[self.name] = self.version
        ckpt = CheckpointManager(checkpoint_path, checkpoint_every) if checkpoint_path else None
        if ckpt is not None and not self.supports_checkpoint:
            raise UnsupportedProblemError(f"{self.name} does not support checkpointing")
        termination = "completed"
        try:
            reason = self._run(ctx, ckpt, resume) if self.supports_checkpoint else self._run(ctx)
            termination = reason or "completed"
        except BudgetExhausted as e:
            termination = e.kind
        except SearchCancelled:
            termination = "cancelled"
        except Exception as e:  # solver bug: report rather than crash, keep the incumbent
            termination = "error"
            ctx.messages.append(f"solver error: {type(e).__name__}: {e}")
            ctx.diagnostics["traceback"] = traceback.format_exc(limit=8)
            log.exception("solver %s failed", self.name)
        finally:
            self.close()
        if termination == "completed" and evaluator.exhausted_reason():
            termination = evaluator.exhausted_reason()
        return finalize(ctx, self.name, self.version, termination)

    def close(self) -> None:
        """Release external resources (e.g. sandbox processes) after a solve."""

    @abstractmethod
    def _run(self, ctx: SearchContext, *args):
        """Run the search; return a termination reason or None. Budget exhaustion raises."""

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "family": self.family,
                "space_kinds": sorted(self.space_kinds), "constraints": self.supports_constraints,
                "gradient_required": self.requires_gradient, "checkpoint": self.supports_checkpoint,
                "params": self.default_params(), "description": self.description}


class IterativeSolver(Solver):
    """Template for solvers with an explicit, serializable iteration state.

    Subclasses implement ``initialize(ctx) -> state`` and
    ``iterate(ctx, state) -> reason | None``.  ``state`` must be a dict of
    JSON-friendly values and numpy arrays.  Checkpoints are written between
    iterations, so resuming reproduces an uninterrupted run exactly for
    deterministic problems.
    """

    supports_checkpoint = True
    max_iterations = 10**9

    @abstractmethod
    def initialize(self, ctx: SearchContext) -> dict: ...

    @abstractmethod
    def iterate(self, ctx: SearchContext, state: dict) -> str | None: ...

    def _run(self, ctx: SearchContext, ckpt: CheckpointManager | None = None, resume: bool = False):
        it = 0
        if resume and ckpt is not None and ckpt.exists():
            payload = ckpt.load()
            if payload["solver"] != self.name:
                raise ValueError(f"checkpoint was written by {payload['solver']}, not {self.name}")
            if payload.get("version") != self.version or _plain(payload.get("params")) != _plain(self.params):
                raise ValueError(f"checkpoint was written by {self.name} {payload.get('version')} with parameters "
                                 f"{payload.get('params')}; this run is {self.version} with {self.params}")
            state = payload["state"]
            it = payload["iteration"]
            ctx.rng.bit_generator.state = payload["rng"]
            ctx.evaluator.restore(payload["evaluator"])
            ctx.stagnation.restore(payload["stagnation"])
            ctx.guarantee_hints = [Guarantee(**g) for g in payload.get("guarantee_hints", [])]
            ctx.diagnostics.update(payload.get("diagnostics", {}))
            ctx.messages.append(f"resumed from checkpoint at iteration {it}")
            if payload.get("reason"):  # the checkpointed iteration had already terminated the search
                return payload["reason"]
        else:
            state = self.initialize(ctx)
            ctx.stagnation.update(ctx.evaluator.best)
        while it < self.max_iterations:
            reason = self.iterate(ctx, state)
            it += 1
            ctx.stagnation.update(ctx.evaluator.best)
            ctx.diagnostics["iterations"] = it
            ctx.notify()
            if ckpt is not None and ckpt.due(it):
                ckpt.save({"solver": self.name, "version": self.version, "iteration": it, "state": state,
                           "rng": ctx.rng.bit_generator.state, "evaluator": ctx.evaluator.snapshot(),
                           "stagnation": ctx.stagnation.state(), "params": self.params, "reason": reason,
                           "guarantee_hints": [dict(vars(g)) for g in ctx.guarantee_hints],
                           "diagnostics": dict(ctx.diagnostics)})
            if reason:
                return reason
        return "completed"
