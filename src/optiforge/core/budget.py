"""Budgets for solution search and for algorithm development.

Two separate notions are tracked:

* :class:`Budget` - the budget of a single solve (objective evaluations,
  wall-clock seconds, optional target value).
* :class:`BudgetLedger` - a total budget for algorithm development, split into
  named categories (feature probing, validation, training benchmarks,
  validation benchmarks, held-out testing, model calls).  Every evaluation made
  on behalf of development is charged here, and a ``total`` cap covers the sum.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .errors import BudgetExhausted


@dataclass(frozen=True)
class Budget:
    max_evaluations: int = 10_000
    max_time: float | None = None  # seconds
    target: float | None = None  # stop when a feasible value at least this good (original sense) is found

    def __post_init__(self) -> None:
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        if self.max_time is not None and self.max_time <= 0:
            raise ValueError("max_time must be > 0")

    def to_dict(self) -> dict:
        return {"max_evaluations": self.max_evaluations, "max_time": self.max_time, "target": self.target}


class CancelToken:
    """Cooperative cancellation shared between a caller and a running solver."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class BudgetLedger:
    """Total development budget with per-category caps.

    ``limits`` maps category -> max units (``None`` = only the total applies).
    ``reserve`` lets callers set aside units before launching work (e.g. a
    benchmark run in another process) and ``settle`` returns unused units.
    """

    total: int
    limits: dict[str, int | None] = field(default_factory=dict)
    max_wall_time: float | None = None
    used: dict[str, int] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def _committed(self, cat: str | None = None) -> int:
        if cat is None:
            return sum(self.used.values()) + sum(self.reserved.values())
        return self.used.get(cat, 0) + self.reserved.get(cat, 0)

    def remaining(self, cat: str | None = None) -> int:
        rem = self.total - self._committed()
        if cat is not None and self.limits.get(cat) is not None:
            rem = min(rem, self.limits[cat] - self._committed(cat))
        return max(rem, 0)

    def check_time(self) -> None:
        if self.max_wall_time is not None and time.time() - self.started > self.max_wall_time:
            raise BudgetExhausted("development_wall_time")

    def charge(self, cat: str, n: int = 1) -> None:
        with self._lock:
            self.check_time()
            if n > self.remaining(cat):
                raise BudgetExhausted(cat, f"development budget exhausted in category '{cat}'")
            self.used[cat] = self.used.get(cat, 0) + n

    def reserve(self, cat: str, n: int) -> None:
        with self._lock:
            self.check_time()
            if n > self.remaining(cat):
                raise BudgetExhausted(cat, f"cannot reserve {n} units in '{cat}' (remaining {self.remaining(cat)})")
            self.reserved[cat] = self.reserved.get(cat, 0) + n

    def settle(self, cat: str, reserved: int, actually_used: int) -> None:
        with self._lock:
            self.reserved[cat] = self.reserved.get(cat, 0) - reserved
            self.used[cat] = self.used.get(cat, 0) + min(actually_used, reserved)

    def summary(self) -> dict:
        return {
            "total_limit": self.total,
            "used_total": sum(self.used.values()),
            "used_by_category": dict(self.used),
            "category_limits": dict(self.limits),
            "wall_time_s": round(time.time() - self.started, 3),
            "max_wall_time": self.max_wall_time,
        }
