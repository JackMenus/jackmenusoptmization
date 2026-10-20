"""Behavioural fingerprints of candidate solvers.

Two candidates with different source can behave identically on the problems a development run cares about: a
code-template operator whose mutation rate is ignored because it mutates every coordinate, or one whose discrete
branch never fires on a purely continuous suite. Such candidates waste benchmark budget and fill the archive with
copies (round 6: four retained operators with identical scores).

A fingerprint runs the solver on a few probe problems with a fixed seed and a small evaluation budget and hashes
every point it asks to evaluate, in order, plus the final status and evaluation count. Equal fingerprints mean the
two solvers produced the same evaluation sequence on those probes; that is evidence of equivalence *on this suite's
variable types*, not a proof of equivalence in general. A non-deterministic solver gets a fresh fingerprint on each
call and is therefore never merged (the safe direction).
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from ..core.budget import Budget


def _canon(x) -> bytes:
    if isinstance(x, dict):
        return repr(sorted((k, _canon(v)) for k, v in x.items())).encode()
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return repr(x).encode()
    return np.round(a, 12).tobytes()


def behaviour_fingerprint(make_solver: Callable, instances, seed: int = 0, budget: int = 200) -> tuple[str, int]:
    """Return ``(fingerprint, evaluations_used)``.

    ``instances`` are benchmark instance specs (``.make()`` gives a fresh problem). Errors are part of the
    fingerprint, so two candidates that fail the same way also match.
    """
    h = hashlib.sha256()
    used = 0
    for inst in instances:
        p = inst.make()
        f = p.objective

        def recorded(x, f=f):
            h.update(_canon(x))
            return f(x)

        p.objective = recorded
        h.update(b"|" + inst.id.encode() + b"|")
        try:
            r = make_solver().solve(p, Budget(max_evaluations=budget), seed=seed)
            h.update(repr((r.status, r.n_evaluations)).encode())
            used += r.n_evaluations
        except Exception as exc:  # noqa: BLE001 - a failure is behaviour too
            h.update(f"error:{type(exc).__name__}".encode())
            used += budget
    return h.hexdigest()[:20], used


def probe_instances(instances) -> list:
    """One instance per family (the first in split order), so every family's variable types are exercised."""
    seen, out = set(), []
    for i in instances:
        key = (i.family, i.params)
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


_HORIZON_KEYS = ("restart", "until", "cond", "step", "refine", "population", "offspring")


def horizon_signature(source) -> str:
    """Hash of the settings whose effect may only appear after the probe budget (restarts, stage switches,
    conditions, step-size rules, refinement, population sizes). Empty for non-DSL sources (Python operators run in a
    fixed host, so their behaviour is fully exercised by the probes' evaluation sequence)."""
    if not isinstance(source, dict):
        return ""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in sorted(node.items()):
                if k in _HORIZON_KEYS:
                    found.append((path + "/" + k, json.dumps(v, sort_keys=True, default=str)))
                walk(v, path + "/" + k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(source, "")
    return ":" + hashlib.sha256(repr(found).encode()).hexdigest()[:12] if found else ""
