"""Offline generation and evolution of DSL operators and algorithm compositions.

The grammar builds well-typed expressions directly (producers first, then
transformers; ``per_type`` branches only for types present in the target
problems), so most generated candidates pass static checking; the checker in
:mod:`optiforge.dsl.spec` remains the authority.  ``Bias`` weights derived
from problem features steer the proposal distribution (e.g. a rugged
landscape makes heavy-tailed moves and restarts more likely), but nothing
here assumes a proposal is better than the baselines.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from ..dsl.spec import DSL_VERSION, OPS, STEP_OPS, SURVIVAL_OPS, check_algorithm, check_operator

LEAVES_BY_TYPE = {
    "real": ["gaussian", "cauchy", "uniform_reset", "toward_best"],
    "integer": ["integer_step", "gaussian", "uniform_reset"],
    "binary": ["bit_flip", "uniform_reset"],
    "categorical": ["categorical_reset", "uniform_reset"],
    "permutation": ["perm_swap", "perm_insert", "perm_reverse", "perm_scramble"],
}


@dataclass
class Bias:
    """Relative preferences for grammar choices (all default to neutral)."""

    heavy_tail: float = 1.0
    restart: float = 0.5
    refine: float = 0.4
    conditional: float = 1.0
    repair_problem: float = 0.0
    notes: list[str] = field(default_factory=list)


def bias_from_features(f: dict) -> Bias:
    b = Bias()
    if f.get("ruggedness", 0) > 0.5:
        b.heavy_tail, b.restart, b.conditional = 2.0, 0.8, 1.5
        b.notes.append("rugged landscape estimate -> favour heavy-tailed moves, restarts, conditional switching")
    if f.get("feasible_fraction", 1.0) < 0.2:
        b.repair_problem = 0.5 if f.get("has_repair") else 0.0
        b.conditional = max(b.conditional, 1.5)
        b.notes.append("few feasible random samples -> favour feasibility-conditioned operators")
    if f.get("kind") == "continuous" and f.get("smoothness", 0) > 0.8:
        b.refine = 0.7
        b.notes.append("smooth landscape estimate -> favour local refinement stage")
    return b


def _num(rng, lo, hi, log=False):
    if log and lo > 0:
        return float(np.round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 4))
    return float(np.round(rng.uniform(lo, hi), 3))


def _leaf(rng, op: str) -> dict:
    node = {"op": op}
    for k, (kind, lo, hi, default) in OPS[op][2].items():
        if kind == "float" and rng.random() < 0.7:
            node[k] = _num(rng, max(lo, 1e-3) if k == "sigma" else lo, min(hi, 0.5) if k == "sigma" else hi, log=(k == "sigma"))
        elif kind == "int" and rng.random() < 0.5:
            node[k] = int(rng.integers(lo, min(hi, 6) + 1))
        elif kind == "rate" and rng.random() < 0.3:
            node[k] = float(np.round(rng.uniform(0.05, 0.6), 2))
    return node


class OperatorGrammar:
    def __init__(self, types: set[str], bias: Bias | None = None, max_depth: int = 3) -> None:
        self.types = set(types)
        self.bias = bias or Bias()
        self.max_depth = max_depth

    # -- transformers --------------------------------------------------
    def leaf_for(self, rng, t: str) -> dict:
        cands = list(LEAVES_BY_TYPE[t])
        w = np.array([self.bias.heavy_tail if c == "cauchy" else 1.0 for c in cands])
        return _leaf(rng, str(rng.choice(cands, p=w / w.sum())))

    def per_type(self, rng, depth: int) -> dict:
        node = {"op": "per_type"}
        for t in sorted(self.types):
            node[t] = self.transformer_for(rng, t, depth + 1)
        return node

    def transformer_for(self, rng, t: str, depth: int) -> dict:
        """A transformer guaranteed to be able to change type ``t``."""
        if depth >= self.max_depth or rng.random() < 0.55:
            return self.leaf_for(rng, t)
        r = rng.random()
        if r < 0.3:
            return {"op": "mix", "children": [self.leaf_for(rng, t), self.leaf_for(rng, t)],
                    "weights": [float(np.round(rng.uniform(0.2, 1), 2)), float(np.round(rng.uniform(0.2, 1), 2))]}
        if r < 0.3 + 0.25 * self.bias.conditional:
            return {"op": "if", "cond": self.predicate(rng), "then": self.leaf_for(rng, t), "else": self.leaf_for(rng, t)}
        if r < 0.85:
            return {"op": "seq", "children": [self.leaf_for(rng, t),
                                              {"op": "prob", "p": float(np.round(rng.uniform(0.1, 0.6), 2)),
                                               "child": self.leaf_for(rng, t)}]}
        return {"op": "repeat", "n": int(rng.integers(2, 4)), "child": self.leaf_for(rng, t)}

    def transformer(self, rng, depth: int = 0) -> dict:
        if len(self.types) == 1:
            return self.transformer_for(rng, next(iter(self.types)), depth)
        if rng.random() < 0.3:
            return {"op": "type_aware", "sigma": _num(rng, 0.01, 0.4, log=True)}
        return self.per_type(rng, depth)

    # -- producers --------------------------------------------------------
    def producer(self, rng) -> dict:
        cands = ["copy", "uniform_cx"]
        if "real" in self.types:
            cands += ["blend_cx", "sbx", "arithmetic_cx"]
        if "permutation" not in self.types and self.types & {"real", "integer"}:
            cands += ["de_rand1", "de_best1", "de_current_to_pbest"]
        return _leaf(rng, str(rng.choice(cands)))

    def predicate(self, rng) -> dict:
        r = rng.random()
        if r < 0.35:
            return {"pred": "stagnation", "ge": int(rng.integers(3, 40))}
        if r < 0.55:
            return {"pred": "success_rate", "lt": float(np.round(rng.uniform(0.02, 0.3), 2))}
        if r < 0.7:
            return {"pred": "diversity", "lt": float(np.round(rng.uniform(0.01, 0.2), 3))}
        if r < 0.85:
            return {"pred": "evals_fraction", "ge": float(np.round(rng.uniform(0.2, 0.9), 2))}
        return {"pred": "feasible_fraction", "lt": float(np.round(rng.uniform(0.1, 0.9), 2))}

    def operator(self, rng) -> dict:
        """A full variation operator (producer followed by a transformer), possibly conditional."""
        base = {"op": "seq", "children": [self.producer(rng), self.transformer(rng)]}
        r = rng.random()
        if r < 0.25 * self.bias.conditional:
            alt = {"op": "seq", "children": [self.producer(rng), self.transformer(rng)]}
            return {"op": "if", "cond": self.predicate(rng), "then": alt, "else": base}
        if r < 0.45:
            alt = {"op": "seq", "children": [self.producer(rng), self.transformer(rng)]}
            return {"op": "mix", "children": [base, alt], "weights": [0.5, 0.5]}
        return base

    # -- algorithms -------------------------------------------------------
    def stage(self, rng, pop: int, off: int, name: str, until: dict | None) -> dict:
        surv_opts = ["plus"]
        if off == pop:
            surv_opts.append("pairwise")
        if off >= pop and pop > 1:
            surv_opts.append("comma")
        if off == pop:
            surv_opts += ["metropolis", "threshold"]
        surv = str(rng.choice(surv_opts))
        sv = {"op": surv}
        if surv == "metropolis":
            sv.update(t0=_num(rng, 1e-3, 10.0, log=True), cooling=float(np.round(rng.uniform(0.98, 0.9995), 4)))
        if surv == "threshold":
            sv["tau"] = _num(rng, 0.001, 0.1, log=True)
        step = str(rng.choice(list(STEP_OPS)))
        st = {"op": step}
        if step == "one_fifth":
            st.update(factor=float(np.round(rng.uniform(1.05, 1.5), 2)), target=float(np.round(rng.uniform(0.1, 0.3), 2)))
        if step == "decay":
            st.update(rate=float(np.round(rng.uniform(0.98, 0.999), 4)), floor=1e-3)
        sel = {"op": str(rng.choice(["tournament", "random", "best"], p=[0.6, 0.3, 0.1]))}
        if sel["op"] == "tournament":
            sel["k"] = int(rng.integers(2, 5))
        out = {"name": name, "parent_selection": sel, "variation": self.operator(rng), "survival": sv, "step": st}
        if until is not None:
            out["until"] = until
        return out

    def algorithm(self, rng, name: str = "generated") -> dict:
        pop = int(rng.choice([1, 4, 10, 20, 30, 50], p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]))
        off = pop if pop > 1 and rng.random() < 0.7 else int(rng.choice([pop, 2 * pop, max(1, pop // 2)]))
        n_st = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
        cuts = sorted(float(np.round(c, 2)) for c in rng.uniform(0.2, 0.9, n_st - 1))
        stages = []
        for i in range(n_st):
            until = None
            if i < n_st - 1:
                until = ({"pred": "evals_fraction", "ge": cuts[i]} if rng.random() < 0.7
                         else {"pred": "stagnation", "ge": int(rng.integers(10, 60))})
            stages.append(self.stage(rng, pop, off, f"stage{i}", until))
        spec = {"dsl_version": DSL_VERSION, "kind": "algorithm", "name": name, "population": pop, "offspring": off,
                "init": {"op": str(rng.choice(["lhs", "uniform", "opposition"]))},
                "repair": {"op": "problem_repair" if rng.random() < self.bias.repair_problem else
                           str(rng.choice(["project", "reflect", "resample"]))},
                "stages": stages, "restart": None, "refine": None}
        if rng.random() < self.bias.restart:
            spec["restart"] = {"op": "stagnation", "patience": int(rng.integers(15, 120)), "keep_best": int(min(pop - 1, rng.integers(0, 3))) if pop > 1 else 0,
                               "reinit": str(rng.choice(["lhs", "uniform", "around_best"])), "scale": _num(rng, 0.05, 0.5)}
        if self.types & {"real", "integer"} and rng.random() < self.bias.refine:
            spec["refine"] = {"op": "pattern", "evals_fraction": float(np.round(rng.uniform(0.05, 0.25), 2)),
                              "step0": _num(rng, 0.005, 0.1, log=True)}
        return spec


# ---------------------------------------------------------------------------
# evolutionary modification of existing representations
# ---------------------------------------------------------------------------

def _nodes(e, path=()):
    """Yield (path, node) for every expression node."""
    yield path, e
    if not isinstance(e, dict):
        return
    for k, v in e.items():
        if k in ("children",) and isinstance(v, list):
            for i, c in enumerate(v):
                yield from _nodes(c, path + (k, i))
        elif k in ("then", "else", "child") or (e.get("op") == "per_type" and k != "op"):
            yield from _nodes(v, path + (k,))


def _get(e, path):
    for p in path:
        e = e[p]
    return e


def _set(e, path, value):
    if not path:
        return value
    tgt = _get(e, path[:-1])
    tgt[path[-1]] = value
    return e


def _node_type(n) -> str | None:
    rep = check_operator(n)
    return rep.info.get("node_type") if rep.ok or rep.info.get("node_type") else None


def mutate_operator(expr: dict, rng, grammar: OperatorGrammar) -> tuple[dict, str]:
    e = copy.deepcopy(expr)
    nodes = list(_nodes(e))
    r = rng.random()
    with_params = [(p, n) for p, n in nodes if isinstance(n, dict) and n.get("op") in OPS and OPS[n["op"]][2]]
    if r < 0.4 and with_params:
        path, n = with_params[int(rng.integers(len(with_params)))]
        k = str(rng.choice(list(OPS[n["op"]][2])))
        kind, lo, hi, default = OPS[n["op"]][2][k]
        cur = n.get(k, default)
        if kind == "float":
            v = float(np.clip((cur if cur else (lo + hi) / 2) * math.exp(rng.normal(0, 0.35)), lo, hi))
            n[k] = float(np.round(v, 4))
        elif kind == "int":
            n[k] = int(np.clip((cur or lo) + rng.choice([-1, 1]), lo, hi))
        elif kind == "rate":
            n[k] = None if cur is not None and rng.random() < 0.3 else float(np.round(rng.uniform(0.05, 0.6), 2))
        elif kind == "bool":
            n[k] = not bool(cur)
        return e, f"perturbed {n['op']}.{k} at {'/'.join(map(str, path)) or 'root'}"
    if r < 0.75:
        path, n = nodes[int(rng.integers(len(nodes)))]
        t = _node_type(n)
        if t == "producer":
            new = grammar.producer(rng) if n.get("op") != "seq" else grammar.operator(rng)
        else:
            new = grammar.transformer(rng)
        return _set(e, path, new), f"replaced subtree at {'/'.join(map(str, path)) or 'root'} ({n.get('op')} -> {new.get('op')})"
    alt = grammar.operator(rng)
    return ({"op": "if", "cond": grammar.predicate(rng), "then": alt, "else": e},
            "wrapped operator in a conditional with a new alternative branch")


def mutate_algorithm(spec: dict, rng, grammar: OperatorGrammar) -> tuple[dict, str]:
    s = copy.deepcopy(spec)
    ops = ["variation", "variation", "survival", "step", "selection", "population", "restart", "refine", "stage",
           "repair", "init"]
    what = str(rng.choice(ops))
    i = int(rng.integers(len(s["stages"])))
    st = s["stages"][i]
    if what == "variation":
        st["variation"], d = mutate_operator(st["variation"], rng, grammar)
        return s, f"stage {i}: {d}"
    if what in ("survival", "step", "selection"):
        fresh = grammar.stage(rng, s["population"], s.get("offspring", s["population"]), st.get("name", f"stage{i}"), None)
        key = {"survival": "survival", "step": "step", "selection": "parent_selection"}[what]
        st[key] = fresh[key]
        return s, f"stage {i}: new {key} {fresh[key]['op']}"
    if what == "population":
        pop = int(np.clip(round(s["population"] * float(rng.choice([0.5, 2.0]))), 1, 200))
        s["population"] = pop
        s["offspring"] = pop
        for stg in s["stages"]:
            if stg.get("survival", {}).get("op") == "comma" and pop == 1:
                stg["survival"] = {"op": "plus"}
        if s.get("restart") and pop > 1:
            s["restart"]["keep_best"] = min(s["restart"].get("keep_best", 1), pop - 1)
        return s, f"population/offspring -> {pop}"
    if what == "restart":
        if s.get("restart"):
            s["restart"] = None
            return s, "removed restart policy"
        s["restart"] = {"op": "stagnation", "patience": int(rng.integers(15, 120)), "keep_best": 1 if s["population"] > 1 else 0,
                        "reinit": str(rng.choice(["lhs", "around_best"])), "scale": 0.2}
        return s, "added stagnation restart policy"
    if what == "refine":
        if s.get("refine"):
            s["refine"] = None
            return s, "removed local refinement"
        if grammar.types & {"real", "integer"}:
            s["refine"] = {"op": "pattern", "evals_fraction": 0.1, "step0": 0.02}
            return s, "added final pattern-search refinement (10% of budget)"
        return s, "no-op (refinement needs real/integer variables)"
    if what == "stage":
        if len(s["stages"]) > 1 and rng.random() < 0.5:
            s["stages"].pop(i)
            s["stages"][-1].pop("until", None)
            return s, f"removed stage {i}"
        if len(s["stages"]) < 3:
            new = grammar.stage(rng, s["population"], s.get("offspring", s["population"]), f"stage{len(s['stages'])}", None)
            s["stages"][-1]["until"] = {"pred": "evals_fraction", "ge": float(np.round(rng.uniform(0.3, 0.85), 2))}
            s["stages"].append(new)
            _fix_until_order(s)
            return s, "appended a new final stage (multi-stage composition)"
        return s, "no-op (stage limit)"
    if what == "repair":
        s["repair"] = {"op": str(rng.choice(["project", "reflect", "resample"]))}
        return s, f"repair -> {s['repair']['op']}"
    s["init"] = {"op": str(rng.choice(["lhs", "uniform", "opposition"]))}
    return s, f"init -> {s['init']['op']}"


def _fix_until_order(s: dict) -> None:
    prev = 0.0
    for st in s["stages"][:-1]:
        u = st.get("until")
        if u and u.get("pred") == "evals_fraction":
            u["ge"] = float(max(u["ge"], prev + 0.05))
            u["ge"] = float(min(u["ge"], 0.95))
            prev = u["ge"]


def crossover_algorithms(a: dict, b: dict, rng) -> tuple[dict, str]:
    """Child takes a's skeleton and one stage's variation (or a whole stage) from b."""
    c = copy.deepcopy(a)
    i = int(rng.integers(len(c["stages"])))
    j = int(rng.integers(len(b["stages"])))
    if rng.random() < 0.6:
        c["stages"][i]["variation"] = copy.deepcopy(b["stages"][j]["variation"])
        return c, f"crossover: stage {i} variation taken from second parent stage {j}"
    st = copy.deepcopy(b["stages"][j])
    if i < len(c["stages"]) - 1:
        st["until"] = c["stages"][i].get("until", {"pred": "evals_fraction", "ge": 0.5})
    else:
        st.pop("until", None)
    if st.get("survival", {}).get("op") in ("pairwise", "metropolis", "threshold") and c.get("offspring", c["population"]) != c["population"]:
        st["survival"] = {"op": "plus"}
    if st.get("survival", {}).get("op") == "comma" and c.get("offspring", c["population"]) < c["population"]:
        st["survival"] = {"op": "plus"}
    c["stages"][i] = st
    return c, f"crossover: stage {i} replaced by second parent's stage {j}"


def valid_algorithm(spec: dict, types: set) -> bool:
    return check_algorithm(spec, types).ok


def valid_operator(expr: dict, types: set) -> bool:
    r = check_operator(expr, types)
    return r.ok and r.info.get("node_type") in ("producer", "transformer")
