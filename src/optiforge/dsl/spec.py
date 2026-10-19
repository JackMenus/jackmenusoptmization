"""The restricted operator / algorithm language (declarative, JSON-serialisable).

An **operator expression** produces one offspring.  Every node has a static
type:

* ``producer``    - builds a child from selected parents / the population
                    (recombination, differential moves, ``copy``),
* ``transformer`` - maps a child to a child (mutations, conditional/random
                    wrappers, type-restricted application).

Combinators: ``seq`` (a producer followed by transformers, or transformers
only), ``mix`` (weighted random choice), ``if`` (conditional on a search-state
predicate), ``per_type`` (apply a transformer only to one variable type),
``prob`` (apply with probability p), ``repeat`` (apply n times), and ``code``
(a reference to a sandboxed, separately validated Python operator).

An **algorithm** is a population loop with an initialiser, one or more
*stages* (each with parent selection, a variation expression, survival /
acceptance, step-size adaptation and an optional switching predicate), a
restart policy, a boundary/repair policy, and an optional local-refinement
stage.  Nothing in this language can perform I/O or evaluate the objective
other than through the interpreter, which is why declarative candidates can
run without operating-system isolation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

DSL_VERSION = 1

ALL_TYPES = ("real", "integer", "binary", "categorical", "permutation")

# name -> (node type, variable types it can change, {param: (kind, lo, hi, default)})
OPS: dict[str, tuple[str, frozenset, dict]] = {
    # producers
    "copy": ("producer", frozenset(), {}),
    "uniform_cx": ("producer", frozenset(ALL_TYPES), {"p": ("float", 0.05, 0.95, 0.5)}),
    "blend_cx": ("producer", frozenset(ALL_TYPES), {"alpha": ("float", 0.0, 1.0, 0.3)}),
    "sbx": ("producer", frozenset(ALL_TYPES), {"eta": ("float", 1.0, 50.0, 15.0)}),
    "arithmetic_cx": ("producer", frozenset(ALL_TYPES), {}),
    "de_rand1": ("producer", frozenset({"real", "integer", "binary", "categorical"}),
                 {"F": ("float", 0.1, 1.5, 0.6), "CR": ("float", 0.0, 1.0, 0.9)}),
    "de_best1": ("producer", frozenset({"real", "integer", "binary", "categorical"}),
                 {"F": ("float", 0.1, 1.5, 0.6), "CR": ("float", 0.0, 1.0, 0.9)}),
    "de_current_to_pbest": ("producer", frozenset({"real", "integer", "binary", "categorical"}),
                            {"F": ("float", 0.1, 1.5, 0.6), "CR": ("float", 0.0, 1.0, 0.9), "p": ("float", 0.02, 0.5, 0.1)}),
    # transformers (mutations)
    "gaussian": ("transformer", frozenset({"real", "integer"}),
                 {"sigma": ("float", 1e-4, 1.0, 0.1), "rate": ("rate", 0.0, 1.0, None), "adaptive": ("bool", 0, 1, True)}),
    "cauchy": ("transformer", frozenset({"real", "integer"}),
               {"sigma": ("float", 1e-4, 1.0, 0.05), "rate": ("rate", 0.0, 1.0, None), "adaptive": ("bool", 0, 1, True)}),
    "uniform_reset": ("transformer", frozenset({"real", "integer", "binary", "categorical"}), {"rate": ("rate", 0.0, 1.0, None)}),
    "bit_flip": ("transformer", frozenset({"binary"}), {"rate": ("rate", 0.0, 1.0, None)}),
    "categorical_reset": ("transformer", frozenset({"categorical"}), {"rate": ("rate", 0.0, 1.0, None)}),
    "integer_step": ("transformer", frozenset({"integer"}), {"rate": ("rate", 0.0, 1.0, None), "max_step": ("int", 1, 20, 1)}),
    "perm_swap": ("transformer", frozenset({"permutation"}), {}),
    "perm_insert": ("transformer", frozenset({"permutation"}), {}),
    "perm_reverse": ("transformer", frozenset({"permutation"}), {}),
    "perm_scramble": ("transformer", frozenset({"permutation"}), {"max_len": ("int", 2, 10, 4)}),
    "type_aware": ("transformer", frozenset(ALL_TYPES), {"sigma": ("float", 1e-4, 1.0, 0.1), "adaptive": ("bool", 0, 1, True)}),
    "toward_best": ("transformer", frozenset({"real", "integer"}), {"w": ("float", 0.0, 1.0, 0.3)}),
}
COMBINATORS = ("seq", "mix", "if", "per_type", "prob", "repeat", "code")

PREDICATES = {
    "stagnation": ("int", 1, 10_000),  # generations without improvement >= value
    "evals_fraction": ("float", 0.0, 1.0),
    "success_rate": ("float", 0.0, 1.0),
    "feasible_fraction": ("float", 0.0, 1.0),
    "diversity": ("float", 0.0, 1.0),
    "random": ("float", 0.0, 1.0),
}
COMPARATORS = ("ge", "lt")

INIT_OPS = {"uniform": {}, "lhs": {}, "opposition": {}}
SELECTION_OPS = {"tournament": {"k": ("int", 2, 10, 2)}, "random": {}, "best": {}}
SURVIVAL_OPS = {
    "plus": {},
    "comma": {},
    "pairwise": {},  # child i replaces member i if not worse (DE-style)
    "metropolis": {"t0": ("float", 1e-6, 1e6, 1.0), "cooling": ("float", 0.5, 0.99999, 0.995)},
    "threshold": {"tau": ("float", 0.0, 1.0, 0.01)},
}
STEP_OPS = {
    "constant": {},
    "one_fifth": {"factor": ("float", 1.01, 3.0, 1.2), "target": ("float", 0.05, 0.5, 0.2)},
    "decay": {"rate": ("float", 0.9, 0.99999, 0.995), "floor": ("float", 1e-6, 1.0, 1e-3)},
}
RESTART_OPS = {
    "stagnation": {"patience": ("int", 2, 10_000, 30), "keep_best": ("int", 0, 10, 1),
                   "reinit": ("choice", ("lhs", "uniform", "around_best"), None, "lhs"),
                   "scale": ("float", 0.01, 1.0, 0.2)},
}
REFINE_OPS = {"pattern": {"evals_fraction": ("float", 0.0, 0.5, 0.1), "every": ("int", 0, 10_000, 0),
                          "step0": ("float", 1e-4, 0.5, 0.05)}}
REPAIR_OPS = {"project": {}, "reflect": {}, "resample": {}, "problem_repair": {}}

MAX_DEPTH = 8
MAX_NODES = 40
MAX_STAGES = 4


@dataclass
class CheckReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def canonical(spec: Any) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def spec_hash(spec: Any) -> str:
    return hashlib.sha256(canonical(spec).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# static type checking
# ---------------------------------------------------------------------------

def _check_params(where: str, given: dict, specs: dict, errors: list) -> None:
    for k in given:
        if k == "op":
            continue
        if k not in specs:
            errors.append(f"{where}: unknown parameter '{k}'")
    for k, sp in specs.items():
        if k not in given:
            continue
        v = given[k]
        kind, lo, hi = sp[0], sp[1], sp[2]
        if kind == "float":
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
                errors.append(f"{where}: '{k}'={v!r} must be a number in [{lo}, {hi}]")
        elif kind == "int":
            if not isinstance(v, int) or isinstance(v, bool) or not (lo <= v <= hi):
                errors.append(f"{where}: '{k}'={v!r} must be an integer in [{lo}, {hi}]")
        elif kind == "rate":
            if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo < v <= hi)):
                errors.append(f"{where}: '{k}'={v!r} must be null or a number in ({lo}, {hi}]")
        elif kind == "bool":
            if not isinstance(v, bool):
                errors.append(f"{where}: '{k}' must be true/false")
        elif kind == "choice":
            if v not in lo:
                errors.append(f"{where}: '{k}'={v!r} must be one of {lo}")


def check_predicate(p: Any, where: str, errors: list) -> None:
    if not isinstance(p, dict) or "pred" not in p:
        errors.append(f"{where}: predicate must be an object with 'pred'")
        return
    name = p["pred"]
    if name in ("and", "or"):
        args = p.get("args")
        if not isinstance(args, list) or len(args) < 2:
            errors.append(f"{where}: '{name}' needs a list of >= 2 predicates in 'args'")
            return
        for i, a in enumerate(args):
            check_predicate(a, f"{where}.{name}[{i}]", errors)
        return
    if name == "not":
        check_predicate(p.get("arg"), f"{where}.not", errors)
        return
    if name not in PREDICATES:
        errors.append(f"{where}: unknown predicate '{name}'")
        return
    kind, lo, hi = PREDICATES[name]
    if name == "random":
        v = p.get("p")
        if not isinstance(v, (int, float)) or not (lo <= v <= hi):
            errors.append(f"{where}: random predicate needs 'p' in [0, 1]")
        return
    cmp = [c for c in COMPARATORS if c in p]
    if len(cmp) != 1:
        errors.append(f"{where}: predicate '{name}' needs exactly one of {COMPARATORS}")
        return
    v = p[cmp[0]]
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi
    if kind == "int":
        ok = ok and float(v).is_integer()
    if not ok:
        errors.append(f"{where}: '{cmp[0]}' value {v!r} out of range [{lo}, {hi}]")
    extra = set(p) - {"pred", cmp[0]}
    if extra:
        errors.append(f"{where}: unexpected keys {sorted(extra)}")


_COMBINATOR_KEYS = {
    "seq": {"op", "children"}, "mix": {"op", "children", "weights"}, "if": {"op", "cond", "then", "else"},
    "prob": {"op", "p", "child"}, "repeat": {"op", "n", "child"}, "code": {"op", "candidate"},
}


def check_expr(e: Any, where: str, errors: list, depth: int = 0, counter: list | None = None,
               code_ops: dict | None = None) -> tuple[str | None, set]:
    """Return (node type, variable types the expression can change)."""
    counter = counter if counter is not None else [0]
    counter[0] += 1
    if depth > MAX_DEPTH:
        errors.append(f"{where}: expression deeper than {MAX_DEPTH}")
        return None, set()
    if not isinstance(e, dict) or "op" not in e:
        errors.append(f"{where}: expression must be an object with 'op'")
        return None, set()
    op = e["op"]
    allowed_keys = _COMBINATOR_KEYS.get(op)
    if allowed_keys is not None:
        extra = set(e) - allowed_keys
        if extra:
            errors.append(f"{where}<{op}>: unknown keys {sorted(extra)}")
    if op in OPS:
        ntype, types, pspec = OPS[op]
        _check_params(f"{where}<{op}>", e, pspec, errors)
        return ntype, set(types)
    if op == "seq":
        ch = e.get("children")
        if not isinstance(ch, list) or not ch:
            errors.append(f"{where}<seq>: needs non-empty 'children'")
            return None, set()
        res = [check_expr(c, f"{where}.seq[{i}]", errors, depth + 1, counter, code_ops) for i, c in enumerate(ch)]
        for i, (t, _) in enumerate(res[1:], start=1):
            if t == "producer":
                errors.append(f"{where}.seq[{i}]: a producer may only appear first in a sequence")
        types = set().union(*[r[1] for r in res])
        return ("producer" if res[0][0] == "producer" else "transformer"), types
    if op == "mix":
        ch = e.get("children")
        w = e.get("weights")
        if not isinstance(ch, list) or len(ch) < 2:
            errors.append(f"{where}<mix>: needs >= 2 'children'")
            return None, set()
        if w is not None and (not isinstance(w, list) or len(w) != len(ch) or any(
                not isinstance(x, (int, float)) or x < 0 for x in w) or sum(w) <= 0):
            errors.append(f"{where}<mix>: 'weights' must be non-negative numbers, one per child, positive sum")
        res = [check_expr(c, f"{where}.mix[{i}]", errors, depth + 1, counter, code_ops) for i, c in enumerate(ch)]
        kinds = {r[0] for r in res}
        if len(kinds) > 1:
            errors.append(f"{where}<mix>: children must all be producers or all transformers")
        return res[0][0], set().union(*[r[1] for r in res])
    if op == "if":
        check_predicate(e.get("cond"), f"{where}<if>.cond", errors)
        if "then" not in e or "else" not in e:
            errors.append(f"{where}<if>: needs 'then' and 'else'")
            return None, set()
        t1 = check_expr(e["then"], f"{where}.then", errors, depth + 1, counter, code_ops)
        t2 = check_expr(e["else"], f"{where}.else", errors, depth + 1, counter, code_ops)
        if t1[0] != t2[0]:
            errors.append(f"{where}<if>: 'then' and 'else' must have the same node type")
        return t1[0], t1[1] | t2[1]
    if op == "per_type":
        types = set()
        branches = {k: v for k, v in e.items() if k != "op"}
        if not branches:
            errors.append(f"{where}<per_type>: needs at least one type branch")
        for k, v in branches.items():
            if k not in ALL_TYPES:
                errors.append(f"{where}<per_type>: unknown variable type '{k}'")
                continue
            t, ty = check_expr(v, f"{where}.{k}", errors, depth + 1, counter, code_ops)
            if t != "transformer":
                errors.append(f"{where}<per_type>.{k}: branch must be a transformer")
            if k not in ty:
                errors.append(f"{where}<per_type>.{k}: branch cannot change '{k}' variables")
            types.add(k)
        return "transformer", types
    if op == "prob":
        p = e.get("p")
        if not isinstance(p, (int, float)) or not (0 < p <= 1):
            errors.append(f"{where}<prob>: 'p' must be in (0, 1]")
        t, ty = check_expr(e.get("child"), f"{where}.prob", errors, depth + 1, counter, code_ops)
        if t != "transformer":
            errors.append(f"{where}<prob>: child must be a transformer")
        return "transformer", ty
    if op == "repeat":
        n = e.get("n")
        if not isinstance(n, int) or not (1 <= n <= 10):
            errors.append(f"{where}<repeat>: 'n' must be an integer in [1, 10]")
        t, ty = check_expr(e.get("child"), f"{where}.repeat", errors, depth + 1, counter, code_ops)
        if t != "transformer":
            errors.append(f"{where}<repeat>: child must be a transformer")
        return "transformer", ty
    if op == "code":
        cid = e.get("candidate")
        if not isinstance(cid, str):
            errors.append(f"{where}<code>: needs a 'candidate' id string")
            return None, set()
        info = (code_ops or {}).get(cid)
        if info is None:
            errors.append(f"{where}<code>: candidate '{cid}' is not a validated code operator available here")
            return "transformer", set()
        return "transformer", set(info.get("types", []))
    errors.append(f"{where}: unknown op '{op}'")
    return None, set()


def check_algorithm(spec: Any, types_present: set | None = None, code_ops: dict | None = None) -> CheckReport:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(spec, dict):
        return CheckReport(False, ["algorithm spec must be an object"])
    if spec.get("dsl_version") != DSL_VERSION:
        errors.append(f"dsl_version must be {DSL_VERSION}")
    if spec.get("kind") != "algorithm":
        errors.append("kind must be 'algorithm'")
    allowed = {"dsl_version", "kind", "name", "population", "offspring", "init", "stages", "restart", "refine", "repair"}
    extra = set(spec) - allowed
    if extra:
        errors.append(f"unknown top-level keys {sorted(extra)}")
    pop = spec.get("population")
    off = spec.get("offspring", pop)
    if not isinstance(pop, int) or not (1 <= pop <= 500):
        errors.append("population must be an integer in [1, 500]")
    if not isinstance(off, int) or not (1 <= off <= 1000):
        errors.append("offspring must be an integer in [1, 1000]")
    init = spec.get("init", {"op": "lhs"})
    if not isinstance(init, dict) or init.get("op") not in INIT_OPS:
        errors.append(f"init.op must be one of {sorted(INIT_OPS)}")
    rep = spec.get("repair", {"op": "project"})
    if not isinstance(rep, dict) or rep.get("op") not in REPAIR_OPS:
        errors.append(f"repair.op must be one of {sorted(REPAIR_OPS)}")
    stages = spec.get("stages")
    covered: set = set()
    n_nodes = [0]
    if not isinstance(stages, list) or not (1 <= len(stages) <= MAX_STAGES):
        errors.append(f"stages must be a list of 1..{MAX_STAGES} stages")
        stages = []
    for i, st in enumerate(stages):
        w = f"stages[{i}]"
        if not isinstance(st, dict):
            errors.append(f"{w}: must be an object")
            continue
        extra = set(st) - {"name", "until", "parent_selection", "variation", "survival", "step"}
        if extra:
            errors.append(f"{w}: unknown keys {sorted(extra)}")
        if "until" in st:
            check_predicate(st["until"], f"{w}.until", errors)
        elif i < len(stages) - 1:
            errors.append(f"{w}: every stage but the last needs an 'until' predicate")
        sel = st.get("parent_selection", {"op": "tournament"})
        if not isinstance(sel, dict) or sel.get("op") not in SELECTION_OPS:
            errors.append(f"{w}.parent_selection.op must be one of {sorted(SELECTION_OPS)}")
        else:
            _check_params(f"{w}.parent_selection", sel, SELECTION_OPS[sel["op"]], errors)
        sv = st.get("survival", {"op": "plus"})
        if not isinstance(sv, dict) or sv.get("op") not in SURVIVAL_OPS:
            errors.append(f"{w}.survival.op must be one of {sorted(SURVIVAL_OPS)}")
        else:
            _check_params(f"{w}.survival", sv, SURVIVAL_OPS[sv["op"]], errors)
            if sv["op"] == "pairwise" and isinstance(pop, int) and off != pop:
                errors.append(f"{w}: pairwise survival requires offspring == population")
            if sv["op"] == "comma" and isinstance(pop, int) and isinstance(off, int) and off < pop:
                errors.append(f"{w}: comma survival requires offspring >= population")
        stp = st.get("step", {"op": "constant"})
        if not isinstance(stp, dict) or stp.get("op") not in STEP_OPS:
            errors.append(f"{w}.step.op must be one of {sorted(STEP_OPS)}")
        else:
            _check_params(f"{w}.step", stp, STEP_OPS[stp["op"]], errors)
        if "variation" not in st:
            errors.append(f"{w}: missing 'variation'")
            continue
        t, ty = check_expr(st["variation"], f"{w}.variation", errors, 0, n_nodes, code_ops)
        if t is None:
            continue
        covered |= ty
        if types_present is not None:
            missing = set(types_present) - ty
            if missing:
                errors.append(f"{w}.variation can never change variables of type {sorted(missing)}")
    if n_nodes[0] > MAX_NODES:
        errors.append(f"expression too large: {n_nodes[0]} nodes > {MAX_NODES}")
    rs = spec.get("restart")
    if rs is not None:
        if not isinstance(rs, dict) or rs.get("op") not in RESTART_OPS:
            errors.append(f"restart.op must be one of {sorted(RESTART_OPS)} (or null)")
        else:
            _check_params("restart", rs, RESTART_OPS[rs["op"]], errors)
    rf = spec.get("refine")
    if rf is not None:
        if not isinstance(rf, dict) or rf.get("op") not in REFINE_OPS:
            errors.append(f"refine.op must be one of {sorted(REFINE_OPS)} (or null)")
        else:
            _check_params("refine", rf, REFINE_OPS[rf["op"]], errors)
    info = {"nodes": n_nodes[0], "stages": len(stages), "types_changed": sorted(covered)}
    return CheckReport(not errors, errors, warnings, info)


def check_operator(expr: Any, types_present: set | None = None, code_ops: dict | None = None) -> CheckReport:
    errors: list[str] = []
    t, ty = check_expr(expr, "operator", errors, 0, [0], code_ops)
    if t is not None and types_present is not None:
        missing = set(types_present) - ty
        if missing:
            errors.append(f"operator can never change variables of type {sorted(missing)}")
    return CheckReport(not errors, errors, [], {"node_type": t, "types_changed": sorted(ty)})


def compatible_kinds(types_changed: set) -> list[str]:
    """Problem kinds whose variable types this expression can cover (coarse summary for the registry)."""
    out = []
    if "real" in types_changed:
        out.append("continuous")
    if types_changed & {"integer", "binary", "categorical", "permutation"}:
        out.append("discrete")
    if "real" in types_changed and types_changed & {"integer", "binary", "categorical"}:
        out.append("mixed")
    return out


# ---------------------------------------------------------------------------
# host template for evaluating stand-alone operators
# ---------------------------------------------------------------------------

def host_algorithm(variation: dict, population: int = 20, name: str = "host") -> dict:
    """Fixed (mu+lambda) host used to benchmark a single operator against the default operator."""
    return {
        "dsl_version": DSL_VERSION, "kind": "algorithm", "name": name, "population": population,
        "offspring": population, "init": {"op": "lhs"}, "repair": {"op": "project"},
        "stages": [{"name": "main", "parent_selection": {"op": "tournament", "k": 2}, "variation": copy.deepcopy(variation),
                    "survival": {"op": "plus"}, "step": {"op": "one_fifth", "factor": 1.2, "target": 0.2}}],
        "restart": {"op": "stagnation", "patience": 60, "keep_best": 1, "reinit": "lhs", "scale": 0.2},
        "refine": None,
    }


DEFAULT_OPERATOR = {"op": "seq", "children": [{"op": "uniform_cx"}, {"op": "type_aware", "sigma": 0.1}]}
