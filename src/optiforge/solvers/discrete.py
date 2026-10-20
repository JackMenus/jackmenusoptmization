"""Solvers for discrete and mixed spaces (most also run on continuous spaces).

* ``random_search``       - uniform type-aware sampling (reference baseline for every space).
* ``greedy_ls``           - greedy coordinate construction + best-improvement neighbourhood search with
                            perturbation restarts (iterated local search).
* ``simulated_annealing`` - Metropolis acceptance, geometric cooling, reheating on stagnation.
* ``tabu``                - tabu search over sampled single-move and binary-exchange neighbourhoods with aspiration.
* ``genetic``             - generational GA with elitism, tournament selection, type-aware variation
                            (also the mixed-variable baseline).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.result import LOCAL_CONVERGED, Guarantee
from ..operators import primitives as P
from .base import IterativeSolver, register_solver


def penalized(ev, rho: float) -> float:
    """Internal acceptance score (never reported): objective + rho * violation; invalid -> inf."""
    if not ev.valid:
        return math.inf
    return ev.fmin + rho * ev.total_violation


@register_solver
class RandomSearch(IterativeSolver):
    name = "random_search"
    version = "1.0.0"
    family = "global"
    description = "Uniform type-aware random sampling; the reference baseline."

    @classmethod
    def default_params(cls) -> dict:
        return {"batch": 50}

    def initialize(self, ctx):
        for w in ctx.warm_start:
            ctx.evaluate(w)
        return {"n": 0}

    def iterate(self, ctx, st):
        for x in ctx.space.sample(ctx.rng, self.params["batch"]):
            ctx.evaluate(x)
        st["n"] += self.params["batch"]
        return None


@register_solver
class GreedyLocalSearch(IterativeSolver):
    name = "greedy_ls"
    version = "1.0.0"
    family = "local"
    supports_noise = False
    description = ("Greedy coordinate-wise construction, then best-improvement search over the full single-move "
                   "neighbourhood; perturbation restarts (iterated local search) at local optima.")

    @classmethod
    def default_params(cls) -> dict:
        return {"values_per_var": 8, "perturb_strength": 3, "construct": True}

    def initialize(self, ctx):
        sp = ctx.space
        x = ctx.warm_start[0].copy() if ctx.warm_start else sp.sample(ctx.rng)
        cur = ctx.evaluate(x)
        if self.params["construct"] and not ctx.warm_start:
            for i in ctx.rng.permutation(sp.dim):
                c = sp.codes[i]
                if c == 4:
                    continue  # permutation blocks are constructed randomly (no partial evaluation available)
                if c == 0:
                    vals = sp.lower[i] + (np.arange(self.params["values_per_var"]) + 0.5) / self.params["values_per_var"] * sp.width[i]
                else:
                    allv = np.arange(int(sp.lower[i]), int(sp.upper[i]) + 1, dtype=float)
                    vals = allv if len(allv) <= self.params["values_per_var"] else ctx.rng.choice(allv, self.params["values_per_var"], replace=False)
                for v in vals:
                    if v == x[i]:
                        continue
                    y = x.copy()
                    y[i] = v
                    ev = ctx.evaluate(y)
                    if ev.better_than(cur):
                        x, cur = y, ev
        return {"x": x, "key": list(cur.rank_key()), "local_optima": 0}

    def iterate(self, ctx, st):
        sp = ctx.space
        x = st["x"]
        cur_key = tuple(st["key"])
        best_y, best_key = None, cur_key
        full = True
        for move, y in P.neighborhood(x, sp, max_values_per_var=10**9):
            ev = ctx.evaluate(y)
            if ev.rank_key() < best_key:
                best_y, best_key = y, ev.rank_key()
        if best_y is not None:
            st["x"], st["key"] = best_y, list(best_key)
            return None
        # local optimum w.r.t. the full single-move neighbourhood
        st["local_optima"] += 1
        inc = ctx.evaluator.best
        if full and not sp.real_mask.any() and inc is not None and np.array_equal(inc.x, x):
            ctx.guarantee_hints = [h for h in ctx.guarantee_hints if h.kind != LOCAL_CONVERGED]
            ctx.guarantee_hints.append(Guarantee(
                LOCAL_CONVERGED,
                "The incumbent is a local optimum with respect to the complete single-move neighbourhood "
                "(+-1 integer steps, bit flips, every other category, all pairwise swaps): every neighbour was "
                "evaluated and none is better under feasibility-first ordering. No global claim is made.",
                method="exhaustive neighbourhood check", point=x.tolist()))
        y = x.copy()
        for _ in range(self.params["perturb_strength"]):
            y = P.random_neighbor(y, sp, ctx.rng)
        ev = ctx.evaluate(y)
        st["x"], st["key"] = y, list(ev.rank_key())
        return None


@register_solver
class SimulatedAnnealing(IterativeSolver):
    name = "simulated_annealing"
    version = "1.0.0"
    family = "global"
    description = "Simulated annealing with random single moves, geometric cooling and reheating on stagnation."

    @classmethod
    def default_params(cls) -> dict:
        return {"t0": None, "cooling": 0.995, "moves_per_iter": 20, "penalty": 1e4, "reheat_after": 50, "sigma": 0.1,
                "exchange_prob": 0.0}

    def initialize(self, ctx):
        sp = ctx.space
        x = ctx.warm_start[0].copy() if ctx.warm_start else sp.sample(ctx.rng)
        ev = ctx.evaluate(x)
        f = penalized(ev, self.params["penalty"])
        t0 = self.params["t0"]
        if t0 is None:  # estimate from random move deltas
            deltas = []
            for _ in range(10):
                e2 = ctx.evaluate(self._move(ctx, x))
                f2 = penalized(e2, self.params["penalty"])
                if math.isfinite(f2) and math.isfinite(f):
                    deltas.append(abs(f2 - f))
            t0 = (np.mean(deltas) if deltas and np.mean(deltas) > 0 else 1.0)
        return {"x": x, "f": f, "T": float(t0), "T0": float(t0), "stale": 0}

    def _move(self, ctx, x):
        sp = ctx.space
        if sp.real_mask.any() and ctx.rng.random() < 0.5:
            return ctx.repair(P.gaussian(x, sp, ctx.rng, sigma=self.params["sigma"], rate=None))
        if self.params["exchange_prob"] > 0 and ctx.rng.random() < self.params["exchange_prob"]:
            y = P.random_exchange(x, sp, ctx.rng)
            if y is not None:
                return ctx.repair(y)
        return ctx.repair(P.random_neighbor(x, sp, ctx.rng))

    def iterate(self, ctx, st):
        before = ctx.evaluator.best.rank_key() if ctx.evaluator.best else None
        for _ in range(self.params["moves_per_iter"]):
            y = self._move(ctx, st["x"])
            ev = ctx.evaluate(y)
            fy = penalized(ev, self.params["penalty"])
            d = fy - st["f"]
            if d <= 0 or (math.isfinite(d) and ctx.rng.random() < math.exp(-d / max(st["T"], 1e-300))):
                st["x"], st["f"] = y, fy
        st["T"] *= self.params["cooling"]
        after = ctx.evaluator.best.rank_key() if ctx.evaluator.best else None
        st["stale"] = 0 if after != before else st["stale"] + 1
        if st["stale"] >= self.params["reheat_after"]:
            st["T"] = st["T0"]
            st["stale"] = 0
            ctx.diagnostics["reheats"] = ctx.diagnostics.get("reheats", 0) + 1
            st["x"] = ctx.evaluator.best.x.copy()
            st["f"] = penalized(ctx.evaluator.best, self.params["penalty"])
        return None


@register_solver
class TabuSearch(IterativeSolver):
    name = "tabu"
    # 1.1.0 (round 10): binary exchange moves on by default, after a pre-registered test-split confirmation
    # (lab/r10-develop-discrete); exchange=False restores 1.0.0 behaviour exactly.
    # 1.1.1 (round 20): exchanges are sampled as index pairs instead of enumerating all of them (O(k) instead of
    # O(n^2) arrays per iteration; same random draws and results as 1.1.0 while ones x zeros <= 10^7)
    version = "1.1.1"
    family = "local"
    space_kinds = frozenset({"discrete"})
    description = ("Tabu search: best non-tabu move from a sampled neighbourhood of single moves and (for binary "
                   "variables) 1-0 exchanges, aspiration by incumbent.")

    @classmethod
    def default_params(cls) -> dict:
        return {"tenure": 7, "sample": 40, "exchange": True}

    def initialize(self, ctx):
        x = ctx.warm_start[0].copy() if ctx.warm_start else ctx.space.sample(ctx.rng)
        ev = ctx.evaluate(x)
        return {"x": x, "key": list(ev.rank_key()), "tabu": {}, "it": 0}

    def iterate(self, ctx, st):
        sp = ctx.space
        st["it"] += 1
        k = self.params["sample"]
        moves = list(P.neighborhood(st["x"], sp, max_values_per_var=4, rng=ctx.rng))
        swaps = P.sample_binary_exchanges(st["x"], sp, k // 2, ctx.rng) if self.params["exchange"] else []
        k -= len(swaps)  # half of the sample from exchanges, so they do not crowd out the single moves
        if len(moves) > k:
            pick = ctx.rng.choice(len(moves), size=k, replace=False)
            moves = [moves[i] for i in pick]
        moves += swaps
        best = None
        inc = ctx.evaluator.best.rank_key() if ctx.evaluator.best else (3, math.inf)
        for move, y in moves:
            # tabu attributes: the changed coordinate(s); an exchange touches two coordinates
            attrs = ([str(move[1])] if move[0] == "set" else [str(move[1]), str(move[2])] if move[0] == "exchange"
                     else [f"{move[1]}-{move[2]}"])
            ev = ctx.evaluate(y)
            k = ev.rank_key()
            is_tabu = any(st["tabu"].get(a, 0) > st["it"] for a in attrs)
            if is_tabu and not k < inc:
                continue
            if best is None or k < best[0]:
                best = (k, y, attrs)
        if best is not None:
            st["x"], st["key"] = best[1], list(best[0])
            for a in best[2]:
                st["tabu"][a] = st["it"] + self.params["tenure"]
        st["tabu"] = {a: t for a, t in st["tabu"].items() if t > st["it"]}
        return None


@register_solver
class GeneticAlgorithm(IterativeSolver):
    name = "genetic"
    version = "1.0.0"
    family = "global"
    description = ("Generational GA: tournament selection, type-aware crossover (uniform/BLX/OX), type-aware mutation, "
                   "elitism, feasibility-first ranking. Works for discrete, continuous and mixed spaces.")

    @classmethod
    def default_params(cls) -> dict:
        return {"pop_size": 40, "elite": 2, "p_cx": 0.9, "tournament": 3, "sigma": 0.1, "mutation_rate": None}

    def initialize(self, ctx):
        n = self.params["pop_size"]
        X = ctx.space.sample(ctx.rng, n)
        for i, w in enumerate(ctx.warm_start[:n]):
            X[i] = w
        X = np.array([ctx.repair(x) for x in X])
        evs = ctx.evaluate_many(X)
        return {"X": X, "keys": [list(e.rank_key()) for e in evs]}

    def iterate(self, ctx, st):
        sp = ctx.space
        X, keys = st["X"], st["keys"]
        n = len(X)
        order = sorted(range(n), key=lambda i: tuple(keys[i]))
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        newX = [X[i].copy() for i in order[: self.params["elite"]]]
        newK = [keys[i] for i in order[: self.params["elite"]]]
        while len(newX) < n:
            a = X[P.tournament(ranks, ctx.rng, self.params["tournament"])]
            b = X[P.tournament(ranks, ctx.rng, self.params["tournament"])]
            if ctx.rng.random() < self.params["p_cx"]:
                child = P.blend_crossover(a, b, sp, ctx.rng) if sp.real_mask.any() else P.uniform_crossover(a, b, sp, ctx.rng)
            else:
                child = a.copy()
            child = P.type_aware_mutation(child, sp, ctx.rng, sigma=self.params["sigma"], rate=self.params["mutation_rate"])
            child = ctx.repair(child)
            ev = ctx.evaluate(child)
            newX.append(child)
            newK.append(list(ev.rank_key()))
        st["X"], st["keys"] = np.array(newX), newK
        return None
