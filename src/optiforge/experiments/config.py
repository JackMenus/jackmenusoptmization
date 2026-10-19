"""Configuration loading and problem resolution for configuration-driven experiments."""
from __future__ import annotations

import importlib
import json
import os
from typing import Any

import yaml

from ..benchmarks.problems import make_problem
from ..core.budget import Budget
from ..core.errors import ProblemDefinitionError
from ..core.problem import Problem


def load_config(path: str) -> dict:
    with open(path) as fh:
        if path.endswith(".json"):
            cfg = json.load(fh)
        else:
            cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ProblemDefinitionError(f"{path}: configuration must be a mapping")
    cfg.setdefault("_source", os.path.abspath(path))
    return cfg


def resolve_problem(spec: dict) -> Problem:
    """``{benchmark: name, seed: 0, params: {...}}`` or ``{factory: "module:function", params: {...}}``.

    Factories are trusted user code (they define the problem); they are not
    generated candidates and are not sandboxed.
    """
    if not isinstance(spec, dict):
        raise ProblemDefinitionError("problem spec must be a mapping")
    params = spec.get("params", {}) or {}
    if "benchmark" in spec:
        return make_problem(spec["benchmark"], int(spec.get("seed", 0)), **params)
    if "factory" in spec:
        mod, _, fn = spec["factory"].partition(":")
        import sys
        if os.getcwd() not in sys.path:  # factories are trusted user modules relative to the working directory
            sys.path.insert(0, os.getcwd())
        if not fn:
            raise ProblemDefinitionError("factory must look like 'package.module:function'")
        obj = getattr(importlib.import_module(mod), fn)(**params)
        if not isinstance(obj, Problem):
            raise ProblemDefinitionError(f"factory {spec['factory']} did not return a Problem")
        return obj
    raise ProblemDefinitionError("problem spec needs 'benchmark' or 'factory'")


def resolve_budget(spec: Any) -> Budget:
    spec = spec or {}
    if isinstance(spec, int):
        return Budget(max_evaluations=spec)
    return Budget(max_evaluations=spec.get("max_evaluations", 10_000), max_time=spec.get("max_time"),
                  target=spec.get("target"))
