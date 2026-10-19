"""Decision-variable definitions and the variable space.

Every solution is held internally as a flat ``float64`` vector (the *encoded*
representation).  The space knows, for every position of that vector, which
variable type it belongs to:

* ``real``         - continuous value in ``[lower, upper]``
* ``integer``      - integer value in ``[lower, upper]``
* ``binary``       - 0 or 1
* ``categorical``  - index ``0 .. k-1`` into a list of choices
* ``permutation``  - a block of ``n`` positions holding a permutation of ``0 .. n-1``

Discrete types are stored as integer-valued floats.  ``decode`` maps an encoded
vector to user-facing values (choices, permutation lists) and ``encode`` maps
back.  Projection (``project``) and membership (``contains``) are explicit and
separate: the space never silently changes a point unless asked to.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Sequence

import numpy as np

from .errors import ProblemDefinitionError


class VarType(str, Enum):
    REAL = "real"
    INTEGER = "integer"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    PERMUTATION = "permutation"

    @property
    def is_discrete(self) -> bool:
        return self is not VarType.REAL


@dataclass(frozen=True)
class Variable:
    """A single decision variable (or, for permutations, a block of positions)."""

    name: str
    type: VarType
    lower: float = 0.0
    upper: float = 1.0
    choices: tuple = ()
    size: int = 1  # number of encoded positions (``n`` for permutations)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ProblemDefinitionError("variable name must be a non-empty string")
        t = self.type
        if t is VarType.REAL:
            if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
                raise ProblemDefinitionError(
                    f"real variable '{self.name}' needs finite bounds; unbounded reals are not supported"
                )
            if not self.lower < self.upper:
                raise ProblemDefinitionError(f"real variable '{self.name}': lower must be < upper")
        elif t is VarType.INTEGER:
            if not (float(self.lower).is_integer() and float(self.upper).is_integer()):
                raise ProblemDefinitionError(f"integer variable '{self.name}' needs integer bounds")
            if self.lower > self.upper:
                raise ProblemDefinitionError(f"integer variable '{self.name}': lower must be <= upper")
        elif t is VarType.BINARY:
            if (self.lower, self.upper) != (0.0, 1.0):
                raise ProblemDefinitionError("binary variables have fixed bounds 0..1")
        elif t is VarType.CATEGORICAL:
            if len(self.choices) < 1:
                raise ProblemDefinitionError(f"categorical variable '{self.name}' needs at least one choice")
            if len(set(map(repr, self.choices))) != len(self.choices):
                raise ProblemDefinitionError(f"categorical variable '{self.name}' has duplicate choices")
        elif t is VarType.PERMUTATION:
            if self.size < 2:
                raise ProblemDefinitionError(f"permutation '{self.name}' needs size >= 2")
        if t is not VarType.PERMUTATION and self.size != 1:
            raise ProblemDefinitionError("only permutation variables may span several positions")

    @property
    def cardinality(self) -> int | None:
        if self.type is VarType.REAL:
            return None
        if self.type is VarType.INTEGER:
            return int(self.upper - self.lower) + 1
        if self.type is VarType.BINARY:
            return 2
        if self.type is VarType.CATEGORICAL:
            return len(self.choices)
        return math.factorial(self.size)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "type": self.type.value}
        if self.type in (VarType.REAL, VarType.INTEGER):
            d.update(lower=self.lower, upper=self.upper)
        if self.type is VarType.CATEGORICAL:
            d["choices"] = list(self.choices)
        if self.type is VarType.PERMUTATION:
            d["size"] = self.size
        return d


# Convenience constructors ---------------------------------------------------

def Real(name: str, lower: float, upper: float) -> Variable:
    return Variable(name, VarType.REAL, float(lower), float(upper))


def Integer(name: str, lower: int, upper: int) -> Variable:
    return Variable(name, VarType.INTEGER, float(lower), float(upper))


def Binary(name: str) -> Variable:
    return Variable(name, VarType.BINARY, 0.0, 1.0)


def Categorical(name: str, choices: Sequence[Any]) -> Variable:
    return Variable(name, VarType.CATEGORICAL, 0.0, float(len(choices) - 1), tuple(choices))


def Permutation(name: str, n: int) -> Variable:
    return Variable(name, VarType.PERMUTATION, 0.0, float(n - 1), (), int(n))


def variable_from_dict(d: dict) -> Variable:
    t = VarType(d["type"])
    if t is VarType.REAL:
        return Real(d["name"], d["lower"], d["upper"])
    if t is VarType.INTEGER:
        return Integer(d["name"], d["lower"], d["upper"])
    if t is VarType.BINARY:
        return Binary(d["name"])
    if t is VarType.CATEGORICAL:
        return Categorical(d["name"], d["choices"])
    return Permutation(d["name"], d["size"])


# Type codes used in the flat per-position arrays.
CODE = {VarType.REAL: 0, VarType.INTEGER: 1, VarType.BINARY: 2, VarType.CATEGORICAL: 3, VarType.PERMUTATION: 4}


@dataclass
class Space:
    """An ordered collection of variables with a flat encoded representation."""

    variables: list[Variable]
    _slices: list[slice] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.variables = list(self.variables)
        if not self.variables:
            raise ProblemDefinitionError("a space needs at least one variable")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ProblemDefinitionError(f"duplicate variable names: {names}")
        self._slices = []
        pos = 0
        codes, lo, hi = [], [], []
        for v in self.variables:
            self._slices.append(slice(pos, pos + v.size))
            pos += v.size
            codes += [CODE[v.type]] * v.size
            lo += [v.lower] * v.size
            hi += [v.upper] * v.size
        self.dim = pos
        self.codes = np.array(codes, dtype=np.int8)
        self.lower = np.array(lo, dtype=float)
        self.upper = np.array(hi, dtype=float)
        self.real_mask = self.codes == 0
        self.int_mask = self.codes == 1
        self.bin_mask = self.codes == 2
        self.cat_mask = self.codes == 3
        self.perm_mask = self.codes == 4
        self.scalar_discrete_mask = self.int_mask | self.bin_mask | self.cat_mask
        self.perm_blocks = [s for v, s in zip(self.variables, self._slices) if v.type is VarType.PERMUTATION]
        self.width = np.where(self.real_mask | self.int_mask | self.cat_mask, self.upper - self.lower, 1.0)
        self.width[self.width == 0] = 1.0

    # -- descriptive --------------------------------------------------------
    @property
    def kind(self) -> str:
        """``continuous``, ``discrete`` or ``mixed``."""
        has_real = bool(self.real_mask.any())
        has_disc = bool((~self.real_mask).any())
        if has_real and has_disc:
            return "mixed"
        return "continuous" if has_real else "discrete"

    @property
    def type_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.variables:
            out[v.type.value] = out.get(v.type.value, 0) + 1
        return out

    @property
    def types_present(self) -> set[str]:
        return {v.type.value for v in self.variables}

    def cardinality(self) -> int | None:
        """Number of points in a finite space, or ``None`` if any variable is real."""
        total = 1
        for v in self.variables:
            c = v.cardinality
            if c is None:
                return None
            total *= c
        return total

    def to_dict(self) -> dict:
        return {"variables": [v.to_dict() for v in self.variables]}

    @classmethod
    def from_dict(cls, d: dict) -> "Space":
        return cls([variable_from_dict(v) for v in d["variables"]])

    # -- membership and projection -----------------------------------------
    def domain_violations(self, x: np.ndarray, tol: float = 1e-9) -> list[str]:
        """Human-readable list of domain violations (empty when ``x`` is in the domain)."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            return [f"shape {x.shape} != ({self.dim},)"]
        if not np.all(np.isfinite(x)):
            return ["non-finite entries"]
        # vectorised fast path for the common case of a valid point
        np_ = ~self.perm_mask
        ok = bool(np.all(x[np_] >= self.lower[np_] - tol) and np.all(x[np_] <= self.upper[np_] + tol))
        d = self.scalar_discrete_mask
        ok = ok and bool(np.all(np.abs(x[d] - np.round(x[d])) <= tol))
        if ok:
            for s in self.perm_blocks:
                b = x[s]
                if not (np.all(b == np.round(b)) and np.array_equal(np.sort(b), np.arange(len(b)))):
                    ok = False
                    break
        if ok:
            return []
        out = []
        for v, s in zip(self.variables, self._slices):
            xv = x[s]
            if v.type is VarType.PERMUTATION:
                if not np.allclose(xv, np.round(xv)) or sorted(np.round(xv).astype(int)) != list(range(v.size)):
                    out.append(f"{v.name}: not a permutation")
                continue
            val = xv[0]
            if val < v.lower - tol or val > v.upper + tol:
                out.append(f"{v.name}={val} outside [{v.lower}, {v.upper}]")
            if v.type.is_discrete and abs(val - round(val)) > tol:
                out.append(f"{v.name}={val} is not integral")
        return out

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        return not self.domain_violations(x, tol)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Nearest-in-type projection: clip reals, round+clip discrete, repair permutations.

        Permutation blocks are repaired by ranking (argsort of the values), which
        maps any real vector to a valid permutation.
        """
        x = np.array(x, dtype=float, copy=True)
        x[~np.isfinite(x)] = (self.lower[~np.isfinite(x)] + self.upper[~np.isfinite(x)]) / 2
        nonperm = ~self.perm_mask
        x[nonperm] = np.clip(x[nonperm], self.lower[nonperm], self.upper[nonperm])
        d = self.scalar_discrete_mask
        x[d] = np.clip(np.round(x[d]), self.lower[d], self.upper[d])
        for s in self.perm_blocks:
            block = x[s]
            if not (np.allclose(block, np.round(block)) and sorted(np.round(block).astype(int)) == list(range(len(block)))):
                x[s] = np.argsort(np.argsort(block, kind="stable"), kind="stable").astype(float)
            else:
                x[s] = np.round(block)
        return x

    # -- sampling -----------------------------------------------------------
    def sample(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        """Uniform type-aware sample (a single vector when ``n`` is None)."""
        m = 1 if n is None else int(n)
        X = np.empty((m, self.dim))
        r = self.real_mask
        X[:, r] = rng.uniform(self.lower[r], self.upper[r], size=(m, int(r.sum())))
        d = self.scalar_discrete_mask
        if d.any():
            X[:, d] = rng.integers(self.lower[d].astype(int), self.upper[d].astype(int) + 1, size=(m, int(d.sum())))
        for s in self.perm_blocks:
            n_ = s.stop - s.start
            for i in range(m):
                X[i, s] = rng.permutation(n_)
        return X[0] if n is None else X

    def latin_hypercube(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Latin hypercube over the real/integer/categorical coordinates (stratified per axis)."""
        X = self.sample(rng, n)
        strata = (np.argsort(rng.random((self.dim, n)), axis=1).T + rng.random((n, self.dim))) / n
        nonperm = ~self.perm_mask
        span = np.where(self.scalar_discrete_mask, self.upper - self.lower + 1, self.upper - self.lower)
        vals = self.lower + strata * span
        vals = np.where(self.scalar_discrete_mask, np.floor(vals), vals)
        X[:, nonperm] = vals[:, nonperm]
        return np.array([self.project(x) for x in X])

    def enumerate(self) -> Iterator[np.ndarray]:
        """Iterate over every point of a finite space (raises for real variables)."""
        if self.cardinality() is None:
            raise ProblemDefinitionError("cannot enumerate a space containing real variables")
        pools: list[list[tuple]] = []
        for v in self.variables:
            if v.type is VarType.PERMUTATION:
                pools.append([tuple(p) for p in itertools.permutations(range(v.size))])
            else:
                pools.append([(float(k),) for k in range(int(v.lower), int(v.upper) + 1)])
        for combo in itertools.product(*pools):
            yield np.array([c for part in combo for c in part], dtype=float)

    # -- encoding -----------------------------------------------------------
    def decode(self, x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x, dtype=float)
        out: dict[str, Any] = {}
        for v, s in zip(self.variables, self._slices):
            if v.type is VarType.REAL:
                out[v.name] = float(x[s][0])
            elif v.type in (VarType.INTEGER, VarType.BINARY):
                out[v.name] = int(round(x[s][0]))
            elif v.type is VarType.CATEGORICAL:
                out[v.name] = v.choices[int(round(x[s][0]))]
            else:
                out[v.name] = [int(round(t)) for t in x[s]]
        return out

    def encode(self, values: dict[str, Any]) -> np.ndarray:
        x = np.empty(self.dim)
        missing = [v.name for v in self.variables if v.name not in values]
        if missing:
            raise ProblemDefinitionError(f"missing values for variables {missing}")
        for v, s in zip(self.variables, self._slices):
            val = values[v.name]
            if v.type is VarType.CATEGORICAL:
                try:
                    x[s] = v.choices.index(val)
                except ValueError as exc:
                    raise ProblemDefinitionError(f"{val!r} is not a choice of {v.name}") from exc
            elif v.type is VarType.PERMUTATION:
                x[s] = np.asarray(val, dtype=float)
            else:
                x[s] = float(val)
        return x

    def slice_of(self, name: str) -> slice:
        for v, s in zip(self.variables, self._slices):
            if v.name == name:
                return s
        raise KeyError(name)
