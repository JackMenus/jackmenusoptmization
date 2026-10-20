"""Type-aware search primitives shared by the baseline solvers and the DSL.

All functions take and return encoded vectors (``np.ndarray``) and never
evaluate the objective.  Unless stated otherwise outputs may leave the domain
only in the continuous sense (e.g. a Gaussian step past a bound); callers are
expected to apply :meth:`Space.project` or another explicit repair/boundary
policy, which keeps boundary handling a visible, separate decision.
"""
from __future__ import annotations

import numpy as np

from ..core.space import Space


# ---------------------------------------------------------------------------
# rank helpers
# ---------------------------------------------------------------------------

def rank_keys(evals) -> np.ndarray:
    """Return 0-based ranks (0 = best) under feasibility-first ordering."""
    keys = [e.rank_key() for e in evals]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = np.empty(len(keys), dtype=int)
    ranks[order] = np.arange(len(keys))
    return ranks


def tournament(ranks: np.ndarray, rng: np.random.Generator, k: int = 2) -> int:
    idx = rng.integers(0, len(ranks), size=max(1, k))
    return int(idx[np.argmin(ranks[idx])])


# ---------------------------------------------------------------------------
# boundary handling for reals
# ---------------------------------------------------------------------------

def reflect(x: np.ndarray, space: Space) -> np.ndarray:
    """Reflect real coordinates at the bounds (then project everything else)."""
    x = np.array(x, dtype=float, copy=True)
    r = space.real_mask
    lo, hi = space.lower[r], space.upper[r]
    w = hi - lo
    y = (x[r] - lo) % (2 * w)
    y = np.where(y > w, 2 * w - y, y)
    x[r] = lo + y
    return space.project(x)


def resample_out_of_bounds(x: np.ndarray, space: Space, rng: np.random.Generator) -> np.ndarray:
    x = np.array(x, dtype=float, copy=True)
    bad = space.real_mask & ((x < space.lower) | (x > space.upper))
    if bad.any():
        x[bad] = rng.uniform(space.lower[bad], space.upper[bad])
    return space.project(x)


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------

def _rate_mask(space: Space, rng, rate: float | None, mask: np.ndarray) -> np.ndarray:
    n = int(mask.sum())
    if n == 0:
        return mask & False
    r = (1.0 / n) if rate is None else rate
    m = mask & (rng.random(space.dim) < r)
    if not m.any():
        m = np.zeros(space.dim, dtype=bool)
        m[rng.choice(np.flatnonzero(mask))] = True
    return m


def gaussian(x, space: Space, rng, sigma: float = 0.1, rate: float | None = 1.0) -> np.ndarray:
    """Gaussian step on real and integer coordinates (scale relative to the variable range)."""
    y = np.array(x, dtype=float, copy=True)
    mask = space.real_mask | space.int_mask
    if not mask.any():
        return y
    m = _rate_mask(space, rng, rate, mask)
    step = rng.normal(0.0, 1.0, space.dim) * sigma * space.width
    ints = m & space.int_mask
    step[ints] = np.where(np.abs(step[ints]) < 0.5, np.sign(step[ints] + 1e-12), np.round(step[ints]))
    y[m] += step[m]
    return y


def cauchy(x, space: Space, rng, sigma: float = 0.05, rate: float | None = None) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    mask = space.real_mask | space.int_mask
    if not mask.any():
        return y
    m = _rate_mask(space, rng, rate, mask)
    step = rng.standard_cauchy(space.dim) * sigma * space.width
    ints = m & space.int_mask
    step[ints] = np.where(np.abs(step[ints]) < 0.5, np.sign(step[ints] + 1e-12), np.round(step[ints]))
    y[m] += np.clip(step[m], -10 * space.width[m], 10 * space.width[m])
    return y


def uniform_reset(x, space: Space, rng, rate: float | None = None) -> np.ndarray:
    """Resample selected scalar coordinates uniformly in their domain."""
    y = np.array(x, dtype=float, copy=True)
    mask = ~space.perm_mask
    if not mask.any():
        return y
    m = _rate_mask(space, rng, rate, mask)
    fresh = space.sample(rng)
    y[m] = fresh[m]
    return y


def bit_flip(x, space: Space, rng, rate: float | None = None) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    if not space.bin_mask.any():
        return y
    m = _rate_mask(space, rng, rate, space.bin_mask)
    y[m] = 1.0 - y[m]
    return y


def categorical_reset(x, space: Space, rng, rate: float | None = None) -> np.ndarray:
    """Change selected categorical coordinates to a *different* category."""
    y = np.array(x, dtype=float, copy=True)
    if not space.cat_mask.any():
        return y
    m = _rate_mask(space, rng, rate, space.cat_mask)
    for i in np.flatnonzero(m):
        k = int(space.upper[i]) + 1
        if k > 1:
            y[i] = float((int(y[i]) + rng.integers(1, k)) % k)
    return y


def integer_step(x, space: Space, rng, rate: float | None = None, max_step: int = 1) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    if not space.int_mask.any():
        return y
    m = _rate_mask(space, rng, rate, space.int_mask)
    steps = rng.integers(1, max_step + 1, size=space.dim) * rng.choice([-1, 1], size=space.dim)
    y[m] += steps[m]
    return y


def _perm_block(space: Space, rng):
    if not space.perm_blocks:
        return None
    return space.perm_blocks[rng.integers(len(space.perm_blocks))]


def perm_swap(x, space: Space, rng) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    s = _perm_block(space, rng)
    if s is None:
        return y
    i, j = rng.choice(s.stop - s.start, size=2, replace=False) + s.start
    y[i], y[j] = y[j], y[i]
    return y


def perm_insert(x, space: Space, rng) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    s = _perm_block(space, rng)
    if s is None:
        return y
    blk = list(y[s])
    i, j = rng.choice(len(blk), size=2, replace=False)
    v = blk.pop(i)
    blk.insert(j, v)
    y[s] = blk
    return y


def perm_reverse(x, space: Space, rng) -> np.ndarray:
    """2-opt style segment reversal."""
    y = np.array(x, dtype=float, copy=True)
    s = _perm_block(space, rng)
    if s is None:
        return y
    i, j = sorted(rng.choice(s.stop - s.start, size=2, replace=False))
    seg = y[s.start + i: s.start + j + 1].copy()
    y[s.start + i: s.start + j + 1] = seg[::-1]
    return y


def perm_scramble(x, space: Space, rng, max_len: int = 4) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    s = _perm_block(space, rng)
    if s is None:
        return y
    n = s.stop - s.start
    L = int(rng.integers(2, min(max_len, n) + 1))
    i = int(rng.integers(0, n - L + 1))
    seg = y[s.start + i: s.start + i + L].copy()
    rng.shuffle(seg)
    y[s.start + i: s.start + i + L] = seg
    return y


def type_aware_mutation(x, space: Space, rng, sigma: float = 0.1, rate: float | None = None) -> np.ndarray:
    """Default mixed-variable mutation: each present type gets its natural move."""
    y = np.array(x, dtype=float, copy=True)
    changed = False
    if space.real_mask.any():
        y = gaussian(y, space, rng, sigma=sigma, rate=rate if rate is not None else min(1.0, 2.0 / max(1, space.real_mask.sum())))
        changed = True
    for mask, fn in ((space.int_mask, integer_step), (space.bin_mask, bit_flip), (space.cat_mask, categorical_reset)):
        if mask.any() and (not changed or rng.random() < 0.5):
            y = fn(y, space, rng, rate=rate)
            changed = True
    if space.perm_blocks and (not changed or rng.random() < 0.5):
        y = [perm_swap, perm_insert, perm_reverse][rng.integers(3)](y, space, rng)
    return y


# ---------------------------------------------------------------------------
# recombination
# ---------------------------------------------------------------------------

def uniform_crossover(a, b, space: Space, rng, p: float = 0.5) -> np.ndarray:
    """Uniform crossover on scalar positions, order crossover (OX) on permutation blocks."""
    m = rng.random(space.dim) < p
    y = np.where(m, a, b).astype(float)
    for s in space.perm_blocks:
        y[s] = order_crossover(np.asarray(a)[s], np.asarray(b)[s], rng)
    return y


def blend_crossover(a, b, space: Space, rng, alpha: float = 0.3) -> np.ndarray:
    """BLX-alpha on reals, uniform on discrete scalars, OX on permutations."""
    y = uniform_crossover(a, b, space, rng)
    r = space.real_mask
    lo = np.minimum(a[r], b[r])
    hi = np.maximum(a[r], b[r])
    d = hi - lo
    y[r] = rng.uniform(lo - alpha * d, hi + alpha * d + 1e-300)
    return y


def arithmetic_crossover(a, b, space: Space, rng) -> np.ndarray:
    y = uniform_crossover(a, b, space, rng)
    r = space.real_mask
    w = rng.random()
    y[r] = w * a[r] + (1 - w) * b[r]
    return y


def sbx_crossover(a, b, space: Space, rng, eta: float = 15.0) -> np.ndarray:
    y = uniform_crossover(a, b, space, rng)
    r = space.real_mask
    u = rng.random(int(r.sum()))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u) + 1e-300)) ** (1 / (eta + 1)))
    c = 0.5 * ((1 + beta) * a[r] + (1 - beta) * b[r])
    y[r] = c
    return y


def order_crossover(pa, pb, rng) -> np.ndarray:
    """OX crossover for one permutation block."""
    n = len(pa)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1.0)
    child[i:j + 1] = pa[i:j + 1]
    used = set(int(v) for v in pa[i:j + 1])
    fill = [v for v in pb if int(v) not in used]
    k = 0
    for pos in list(range(j + 1, n)) + list(range(0, i)):
        child[pos] = fill[k]
        k += 1
    return child


def differential(target, r1, r2, r3, space: Space, rng, F: float = 0.5, CR: float = 0.9) -> np.ndarray:
    """DE/rand/1/bin mutant+crossover; discrete scalars inherit from the donor with probability CR."""
    donor = r1 + F * (r2 - r3)
    j = rng.integers(space.dim)
    cross = rng.random(space.dim) < CR
    cross[j] = True
    y = np.where(cross, donor, target).astype(float)
    for s in space.perm_blocks:  # permutations: fall back to OX between target and r1
        y[s] = order_crossover(np.asarray(target)[s], np.asarray(r1)[s], rng)
    return y


# ---------------------------------------------------------------------------
# neighbourhoods (for local search / tabu)
# ---------------------------------------------------------------------------

def neighborhood(x, space: Space, max_values_per_var: int = 6, rng=None, exchange: bool = False):
    """Enumerate single-change neighbours as (move, y).

    move = ("set", i, value) for scalar coordinates (integers: +-1 steps;
    categorical: every other category up to ``max_values_per_var``; binary:
    flip; real: +-5% of the range) or ("swap", i, j) for permutation pairs.
    With ``exchange=True`` also ("exchange", i, j): binary i goes 1 -> 0 and binary j goes 0 -> 1 in one move
    (keeps the number of selected items, e.g. swapping a heavy knapsack item for a light one).
    """
    x = np.asarray(x, dtype=float)
    if exchange:
        yield from binary_exchanges(x, space)
    for i in range(space.dim):
        c = space.codes[i]
        if c == 4:
            continue
        cand = []
        if c == 0:
            step = 0.05 * space.width[i]
            cand = [x[i] - step, x[i] + step]
        elif c == 1:
            cand = [x[i] - 1, x[i] + 1]
        elif c == 2:
            cand = [1 - x[i]]
        elif c == 3:
            k = int(space.upper[i]) + 1
            others = [v for v in range(k) if v != int(x[i])]
            if rng is not None and len(others) > max_values_per_var:
                others = list(rng.choice(others, size=max_values_per_var, replace=False))
            cand = others[:max_values_per_var] if rng is None else others
        for v in cand:
            if space.lower[i] <= v <= space.upper[i]:
                y = x.copy()
                y[i] = float(v)
                yield ("set", i, float(v)), y
    for s in space.perm_blocks:
        n = s.stop - s.start
        for a in range(n - 1):
            for b in range(a + 1, n):
                y = x.copy()
                y[s.start + a], y[s.start + b] = y[s.start + b], y[s.start + a]
                yield ("swap", s.start + a, s.start + b), y


def binary_exchanges(x, space: Space):
    """All ("exchange", i, j) moves: binary coordinate i from 1 to 0 and binary coordinate j from 0 to 1."""
    x = np.asarray(x, dtype=float)
    ones = np.flatnonzero(space.bin_mask & (x == 1))
    zeros = np.flatnonzero(space.bin_mask & (x == 0))
    for i in ones:
        for j in zeros:
            y = x.copy()
            y[i], y[j] = 0.0, 1.0
            yield ("exchange", int(i), int(j)), y


def sample_binary_exchanges(x, space: Space, k: int, rng) -> list:
    """Up to ``k`` distinct ("exchange", i, j) moves drawn uniformly without replacement from all 1 -> 0 / 0 -> 1
    pairs, without materialising the full neighbourhood."""
    x = np.asarray(x, dtype=float)
    ones = np.flatnonzero(space.bin_mask & (x == 1))
    zeros = np.flatnonzero(space.bin_mask & (x == 0))
    total = len(ones) * len(zeros)
    k = min(int(k), total)
    if k <= 0:
        return []
    flat = rng.choice(total, size=k, replace=False) if total <= 10**7 else np.unique(rng.integers(0, total, 2 * k))[:k]
    out = []
    for f in flat:
        i, j = ones[f // len(zeros)], zeros[f % len(zeros)]
        y = x.copy()
        y[i], y[j] = 0.0, 1.0
        out.append((("exchange", int(i), int(j)), y))
    return out


def random_exchange(x, space: Space, rng) -> np.ndarray | None:
    """One random binary exchange (1 -> 0 and 0 -> 1), or None if no such pair exists."""
    x = np.asarray(x, dtype=float)
    ones = np.flatnonzero(space.bin_mask & (x == 1))
    zeros = np.flatnonzero(space.bin_mask & (x == 0))
    if not len(ones) or not len(zeros):
        return None
    y = x.copy()
    y[rng.choice(ones)], y[rng.choice(zeros)] = 0.0, 1.0
    return y


def random_neighbor(x, space: Space, rng) -> np.ndarray:
    """One random single-coordinate (or single-swap) move, always inside the domain."""
    x = np.asarray(x, dtype=float)
    n_scalar = int((~space.perm_mask).sum())
    n_perm = len(space.perm_blocks)
    if n_perm and (n_scalar == 0 or rng.random() < n_perm / (n_perm + 1)):
        y = [perm_swap, perm_insert, perm_reverse][rng.integers(3)](x, space, rng)
        return space.project(y)
    i = int(rng.choice(np.flatnonzero(~space.perm_mask)))
    y = x.copy()
    c = space.codes[i]
    if c == 0:
        y[i] += rng.normal(0, 0.1 * space.width[i])
    elif c == 1:
        y[i] += rng.choice([-1, 1]) * rng.integers(1, max(2, int(0.1 * space.width[i]) + 1))
    elif c == 2:
        y[i] = 1 - y[i]
    else:
        k = int(space.upper[i]) + 1
        if k > 1:
            y[i] = float((int(y[i]) + rng.integers(1, k)) % k)
    return space.project(y)
