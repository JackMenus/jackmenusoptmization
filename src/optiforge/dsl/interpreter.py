"""Interpreter that turns a checked DSL algorithm spec into a runnable solver."""
from __future__ import annotations

import math

import numpy as np

from ..core.context import SearchContext
from ..core.errors import CandidateRejected
from ..operators import primitives as P
from ..solvers.base import IterativeSolver
from ..solvers.composite import PatternSearch, run_stage
from .spec import OPS, check_algorithm, spec_hash

TYPE_MASK = {"real": "real_mask", "integer": "int_mask", "binary": "bin_mask", "categorical": "cat_mask",
             "permutation": "perm_mask"}


def _default(op: str, key: str):
    return OPS[op][2][key][3]


class _Env:
    """Per-offspring evaluation environment for operator expressions."""

    def __init__(self, algo: "DSLAlgorithm", ctx: SearchContext, st: dict, ranks: np.ndarray, target: int,
                 parent_sel: dict):
        self.algo, self.ctx, self.st, self.ranks, self.target, self.sel = algo, ctx, st, ranks, target, parent_sel
        self.X = st["X"]
        self.space = ctx.space
        self.rng = ctx.rng
        self.best = int(np.argmin(ranks))

    def draw(self) -> np.ndarray:
        op = self.sel.get("op", "tournament")
        n = len(self.X)
        if op == "random":
            return self.X[int(self.rng.integers(n))]
        if op == "best":
            return self.X[self.best]
        return self.X[P.tournament(self.ranks, self.rng, self.sel.get("k", 2))]

    def distinct(self, k: int, exclude: int) -> list[int]:
        n = len(self.X)
        pool = [j for j in range(n) if j != exclude]
        if len(pool) >= k:
            return list(self.rng.choice(pool, size=k, replace=False))
        return list(self.rng.integers(0, n, size=k))


class DSLAlgorithm(IterativeSolver):
    family = "generated"
    space_kinds = frozenset({"continuous", "discrete", "mixed"})

    def __init__(self, spec: dict, candidate_id: str | None = None, code_ops: dict | None = None) -> None:
        self.spec = spec
        self.params = {}
        self.code_ops = code_ops or {}
        h = spec_hash(spec)
        self.candidate_id = candidate_id or h
        self.name = f"dsl:{self.candidate_id}"
        self.version = f"dsl1-{h}"
        self.description = spec.get("name", "")
        report = check_algorithm(spec, None, self._code_info())
        if not report.ok:
            raise CandidateRejected("; ".join(report.errors))

    def close(self) -> None:
        for op in self.code_ops.values():
            op.close()

    def _code_info(self) -> dict:
        return {cid: {"types": op.types} for cid, op in self.code_ops.items()}

    def incompatibilities(self, problem):
        r = super().incompatibilities(problem)
        rep = check_algorithm(self.spec, problem.space.types_present, self._code_info())
        return r + rep.errors

    # ------------------------------------------------------------------
    def _init_pop(self, ctx: SearchContext, n: int, op: str, around: np.ndarray | None = None, scale: float = 0.2):
        sp = ctx.space
        if n <= 0:
            return np.empty((0, sp.dim))
        if around is not None:
            X = np.array([ctx.repair(P.type_aware_mutation(around, sp, ctx.rng, sigma=scale, rate=0.5)) for _ in range(n)])
        elif op == "lhs":
            X = sp.latin_hypercube(ctx.rng, n)
        elif op == "opposition":
            half = sp.sample(ctx.rng, (n + 1) // 2)
            opp = half.copy()
            r = sp.real_mask | sp.int_mask
            opp[:, r] = sp.lower[r] + sp.upper[r] - half[:, r]
            X = np.vstack([half, opp])[:n]
        else:
            X = sp.sample(ctx.rng, n)
        return np.array([sp.project(x) for x in X])

    def initialize(self, ctx: SearchContext) -> dict:
        n = self.spec["population"]
        X = self._init_pop(ctx, n, self.spec.get("init", {"op": "lhs"})["op"])
        for i, w in enumerate(ctx.warm_start[:n]):
            X[i] = w
        evs = ctx.evaluate_many(X)
        ctx.operator_versions[self.name] = self.version
        for cid, op in self.code_ops.items():
            ctx.operator_versions[f"code:{cid}"] = op.version
        return {"X": X, "keys": [list(e.rank_key()) for e in evs], "vals": [_scalar(e) for e in evs], "gen": 0,
                "stage": 0, "mult": 1.0, "succ": 0.0, "T": None, "restarts": 0, "refined": False}

    # ------------------------------------------------------------------
    def _pred(self, p: dict, ctx: SearchContext, st: dict) -> bool:
        name = p["pred"]
        if name == "and":
            return all(self._pred(a, ctx, st) for a in p["args"])
        if name == "or":
            return any(self._pred(a, ctx, st) for a in p["args"])
        if name == "not":
            return not self._pred(p["arg"], ctx, st)
        if name == "random":
            return bool(ctx.rng.random() < p["p"])
        if name == "stagnation":
            v = ctx.stagnation.since_improvement
        elif name == "evals_fraction":
            m = ctx.evaluator.budget.max_evaluations
            v = ctx.evaluator.n_evals / m if m else 0.0
        elif name == "success_rate":
            v = st["succ"]
        elif name == "feasible_fraction":
            v = float(np.mean([k[0] == 0 for k in st["keys"]]))
        else:  # diversity
            sp = ctx.space
            m = ~sp.perm_mask
            v = float(np.mean(np.std(st["X"][:, m], axis=0) / sp.width[m])) if m.any() else 0.0
        return v >= p["ge"] if "ge" in p else v < p["lt"]

    def _apply(self, e: dict, x: np.ndarray | None, env: _Env) -> np.ndarray:
        op = e["op"]
        sp, rng = env.space, env.rng
        mult = env.st["mult"]
        if op == "seq":
            for c in e["children"]:
                x = self._apply(c, x, env)
            return x
        if op == "mix":
            w = np.asarray(e.get("weights") or [1.0] * len(e["children"]), dtype=float)
            k = int(rng.choice(len(w), p=w / w.sum()))
            return self._apply(e["children"][k], x, env)
        if op == "if":
            return self._apply(e["then"] if self._pred(e["cond"], env.ctx, env.st) else e["else"], x, env)
        if op == "prob":
            return self._apply(e["child"], x, env) if rng.random() < e["p"] else x
        if op == "repeat":
            for _ in range(e["n"]):
                x = self._apply(e["child"], x, env)
            return x
        if op == "per_type":
            y = x.copy()
            for t, branch in e.items():
                if t == "op":
                    continue
                m = getattr(sp, TYPE_MASK[t])
                if m.any():
                    z = self._apply(branch, y, env)
                    y = np.where(m, z, y)
            return y
        if op == "code":
            return self.code_ops[e["candidate"]].transform(x, sp, rng, mult, env.ctx)
        # producers ----------------------------------------------------
        if op == "copy":
            return env.X[env.target].copy()
        if op in ("uniform_cx", "blend_cx", "sbx", "arithmetic_cx"):
            a, b = env.draw(), env.draw()
            if op == "uniform_cx":
                return P.uniform_crossover(a, b, sp, rng, e.get("p", 0.5))
            if op == "blend_cx":
                return P.blend_crossover(a, b, sp, rng, e.get("alpha", 0.3))
            if op == "sbx":
                return P.sbx_crossover(a, b, sp, rng, e.get("eta", 15.0))
            return P.arithmetic_crossover(a, b, sp, rng)
        if op.startswith("de_"):
            F, CR = e.get("F", 0.6), e.get("CR", 0.9)
            i = env.target
            r1, r2, r3 = env.distinct(3, i)
            X = env.X
            if op == "de_rand1":
                donor = X[r1] + F * (X[r2] - X[r3])
            elif op == "de_best1":
                donor = X[env.best] + F * (X[r1] - X[r2])
            else:
                k = max(1, int(e.get("p", 0.1) * len(X)))
                pb = int(rng.choice(np.argsort(env.ranks)[:k]))
                donor = X[i] + F * (X[pb] - X[i]) + F * (X[r1] - X[r2])
            cross = rng.random(sp.dim) < CR
            cross[rng.integers(sp.dim)] = True
            y = np.where(cross, donor, X[i]).astype(float)
            for s in sp.perm_blocks:
                y[s] = P.order_crossover(X[i][s], X[r1][s], rng)
            return y
        # transformers ---------------------------------------------------
        if x is None:
            x = env.X[env.target].copy()
        if op == "gaussian":
            s = e.get("sigma", 0.1) * (mult if e.get("adaptive", True) else 1.0)
            return P.gaussian(x, sp, rng, sigma=s, rate=e.get("rate"))
        if op == "cauchy":
            s = e.get("sigma", 0.05) * (mult if e.get("adaptive", True) else 1.0)
            return P.cauchy(x, sp, rng, sigma=s, rate=e.get("rate"))
        if op == "uniform_reset":
            return P.uniform_reset(x, sp, rng, rate=e.get("rate"))
        if op == "bit_flip":
            return P.bit_flip(x, sp, rng, rate=e.get("rate"))
        if op == "categorical_reset":
            return P.categorical_reset(x, sp, rng, rate=e.get("rate"))
        if op == "integer_step":
            return P.integer_step(x, sp, rng, rate=e.get("rate"), max_step=e.get("max_step", 1))
        if op == "perm_swap":
            return P.perm_swap(x, sp, rng)
        if op == "perm_insert":
            return P.perm_insert(x, sp, rng)
        if op == "perm_reverse":
            return P.perm_reverse(x, sp, rng)
        if op == "perm_scramble":
            return P.perm_scramble(x, sp, rng, e.get("max_len", 4))
        if op == "type_aware":
            s = e.get("sigma", 0.1) * (mult if e.get("adaptive", True) else 1.0)
            return P.type_aware_mutation(x, sp, rng, sigma=s)
        if op == "toward_best":
            y = x.copy()
            m = sp.real_mask | sp.int_mask
            y[m] = x[m] + e.get("w", 0.3) * (env.X[env.best][m] - x[m])
            return y
        raise CandidateRejected(f"interpreter: unsupported op {op}")

    def _repair(self, ctx: SearchContext, x: np.ndarray) -> np.ndarray:
        op = self.spec.get("repair", {"op": "project"})["op"]
        if not np.all(np.isfinite(x)):
            ctx.diagnostics["nonfinite_offspring"] = ctx.diagnostics.get("nonfinite_offspring", 0) + 1
        if op == "reflect":
            return P.reflect(np.nan_to_num(x), ctx.space)
        if op == "resample":
            return P.resample_out_of_bounds(np.nan_to_num(x), ctx.space, ctx.rng)
        if op == "problem_repair":
            return ctx.repair(x)
        return ctx.space.project(x)

    # ------------------------------------------------------------------
    def iterate(self, ctx: SearchContext, st: dict):
        spec = self.spec
        stages = spec["stages"]
        while st["stage"] < len(stages) - 1 and self._pred(stages[st["stage"]]["until"], ctx, st):
            st["stage"] += 1
            st["mult"] = 1.0
            ctx.diagnostics.setdefault("stage_switches", []).append([ctx.evaluator.n_evals, st["stage"]])
        stage = stages[st["stage"]]
        self._maybe_refine(ctx, st)
        X, keys = st["X"], st["keys"]
        n = len(X)
        lam = spec.get("offspring", n)
        ranks = _ranks(keys)
        sel = stage.get("parent_selection", {"op": "tournament", "k": 2})
        var = stage["variation"]
        children, ckeys, cvals, succ = [], [], [], 0
        for j in range(lam):
            env = _Env(self, ctx, st, ranks, j % n, sel)
            y = self._apply(var, None, env)
            if y is None:
                y = X[j % n].copy()
            y = self._repair(ctx, y)
            ev = ctx.evaluate(y)
            k = list(ev.rank_key())
            if tuple(k) < tuple(keys[j % n]):
                succ += 1
            children.append(y)
            ckeys.append(k)
            cvals.append(_scalar(ev))
        st["succ"] = succ / lam
        self._survive(ctx, st, stage.get("survival", {"op": "plus"}), np.array(children), ckeys, cvals)
        self._step(st, stage.get("step", {"op": "constant"}))
        st["gen"] += 1
        self._maybe_restart(ctx, st)
        return None

    def _survive(self, ctx, st, sv, C, ckeys, cvals):
        op = sv["op"]
        X, keys, vals = st["X"], st["keys"], st["vals"]
        n = len(X)
        if op in ("plus", "comma"):
            if op == "plus":
                allX = np.vstack([X, C])
                allK, allV = keys + ckeys, vals + cvals
            else:
                allX, allK, allV = C, ckeys, cvals
            order = sorted(range(len(allK)), key=lambda i: tuple(allK[i]))[:n]
            st["X"] = allX[order]
            st["keys"] = [allK[i] for i in order]
            st["vals"] = [allV[i] for i in order]
            return
        if st["T"] is None and op == "metropolis":
            st["T"] = float(sv.get("t0", 1.0))
        for j in range(len(C)):
            i = j % n
            accept = tuple(ckeys[j]) <= tuple(keys[i])
            if not accept and op == "metropolis":
                d = cvals[j] - vals[i]
                accept = math.isfinite(d) and ctx.rng.random() < math.exp(-d / max(st["T"], 1e-300))
            elif not accept and op == "threshold":
                accept = math.isfinite(cvals[j]) and cvals[j] <= vals[i] + sv.get("tau", 0.01) * (abs(vals[i]) + 1e-12)
            if accept:
                X[i] = C[j]
                keys[i] = ckeys[j]
                vals[i] = cvals[j]
        if op == "metropolis":
            st["T"] *= float(sv.get("cooling", 0.995))

    @staticmethod
    def _step(st, stp):
        op = stp["op"]
        if op == "one_fifth":
            f, tgt = stp.get("factor", 1.2), stp.get("target", 0.2)
            st["mult"] *= f if st["succ"] > tgt else f ** (-tgt / (1 - tgt))
        elif op == "decay":
            st["mult"] = max(stp.get("floor", 1e-3), st["mult"] * stp.get("rate", 0.995))
        st["mult"] = float(min(max(st["mult"], 1e-6), 10.0))

    def _maybe_restart(self, ctx, st):
        rs = self.spec.get("restart")
        if not rs or ctx.stagnation.since_improvement + 1 < rs.get("patience", 30):
            return
        if st.get("last_restart_gen", -1) >= st["gen"] - rs.get("patience", 30):
            return
        n = len(st["X"])
        keep = min(rs.get("keep_best", 1), n - 1)  # the incumbent is always retained by the evaluator
        order = sorted(range(len(st["keys"])), key=lambda i: tuple(st["keys"][i]))
        kept = order[:keep]
        reinit = rs.get("reinit", "lhs")
        around = ctx.evaluator.best.x if reinit == "around_best" and ctx.evaluator.best is not None else None
        newX = self._init_pop(ctx, n - keep, "uniform" if reinit == "uniform" else "lhs", around, rs.get("scale", 0.2))
        evs = ctx.evaluate_many(newX)
        st["X"] = np.vstack([st["X"][kept], newX]) if keep else newX
        st["keys"] = [st["keys"][i] for i in kept] + [list(e.rank_key()) for e in evs]
        st["vals"] = [st["vals"][i] for i in kept] + [_scalar(e) for e in evs]
        st["mult"] = 1.0
        st["T"] = None
        st["restarts"] += 1
        st["last_restart_gen"] = st["gen"]
        ctx.diagnostics["restarts"] = st["restarts"]

    def _maybe_refine(self, ctx, st):
        rf = self.spec.get("refine")
        if not rf or st["refined"]:
            return
        m = ctx.evaluator.budget.max_evaluations
        frac = rf.get("evals_fraction", 0.1)
        if m is None or frac <= 0 or ctx.evaluator.n_evals < (1 - frac) * m:
            return
        st["refined"] = True
        best = ctx.evaluator.best
        if best is None:
            return
        run_stage(ctx, PatternSearch(step0=rf.get("step0", 0.05)), ctx.evaluator.remaining(), [best.x])
        # refresh the worst member with the refined incumbent so the population benefits
        worst = max(range(len(st["keys"])), key=lambda i: tuple(st["keys"][i]))
        b = ctx.evaluator.best
        st["X"][worst] = b.x
        st["keys"][worst] = list(b.rank_key())
        st["vals"][worst] = _scalar(b)


def _ranks(keys) -> np.ndarray:
    order = sorted(range(len(keys)), key=lambda i: tuple(keys[i]))
    r = np.empty(len(keys), dtype=int)
    r[order] = np.arange(len(keys))
    return r


def _scalar(ev) -> float:
    """Scalar acceptance value used by metropolis/threshold survival (internal only)."""
    if not ev.valid:
        return math.inf
    return ev.fmin + 1e6 * ev.total_violation
