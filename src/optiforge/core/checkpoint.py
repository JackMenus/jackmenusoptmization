"""JSON checkpoints for iterative solvers.

A checkpoint stores the solver's iteration state, the random generator state,
the evaluator's counters/incumbent and the stagnation tracker, so a resumed run
continues exactly where the saved one stood (bit-identical for deterministic
problems).  Checkpoints are plain JSON (no pickle) and are written atomically.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np

FORMAT_VERSION = 1


def encode_state(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return {"__ndarray__": o.tolist(), "dtype": str(o.dtype)}
    if isinstance(o, dict):
        return {k: encode_state(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [encode_state(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return {"__float__": repr(o)}
    return o


def decode_state(o: Any) -> Any:
    if isinstance(o, dict):
        if "__ndarray__" in o:
            return np.array(o["__ndarray__"], dtype=o["dtype"])
        if "__float__" in o:
            return float(o["__float__"])
        return {k: decode_state(v) for k, v in o.items()}
    if isinstance(o, list):
        return [decode_state(v) for v in o]
    return o


class CheckpointManager:
    def __init__(self, path: str, every: int = 10) -> None:
        if every < 1:
            raise ValueError("checkpoint interval must be >= 1")
        self.path = path
        self.every = every
        self.n_saved = 0

    def due(self, iteration: int) -> bool:
        return iteration % self.every == 0

    def save(self, payload: dict) -> None:
        payload = {"format": FORMAT_VERSION, **payload}
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".ckpt-")
        with os.fdopen(fd, "w") as fh:
            json.dump(encode_state(payload), fh)
        os.replace(tmp, self.path)
        self.n_saved += 1

    def load(self) -> dict:
        with open(self.path) as fh:
            payload = decode_state(json.load(fh))
        if payload.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format {payload.get('format')}")
        return payload

    def exists(self) -> bool:
        return os.path.exists(self.path)
