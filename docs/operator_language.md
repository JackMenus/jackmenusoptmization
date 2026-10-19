# Operator and algorithm language (DSL v1)

Candidates in the language are JSON objects. `optiforge.dsl.spec` holds the complete grammar: `OPS`, `PREDICATES`,
`*_OPS` and the limits. `check_operator` and `check_algorithm` perform a static check before anything runs. The
check covers:

* unknown ops, keys and parameters;
* parameter ranges;
* node types;
* type coverage: the variation must be able to change every variable type the target problems contain;
* stage ordering;
* survival compatibility, for example `pairwise` requires offspring equal to population;
* size limits: depth 8, 40 nodes, 4 stages.

## Expressions

Every expression has a static node type.

* **Producers** create a child from parents: `copy`, `uniform_cx`, `blend_cx`, `sbx`, `arithmetic_cx`, `de_rand1`,
  `de_best1` and `de_current_to_pbest`. A producer may appear only first in a `seq`.
* **Transformers** modify a child: `gaussian` and `cauchy` (optionally with adaptive steps), `uniform_reset`,
  `bit_flip`, `categorical_reset`, `integer_step`, `perm_swap`, `perm_insert`, `perm_reverse`, `perm_scramble`,
  `type_aware` and `toward_best`.

The combinators are:

| combinator | meaning |
|---|---|
| `{"op":"seq","children":[...]}` | a producer followed by transformers, or transformers only |
| `{"op":"mix","children":[...],"weights":[...]}` | weighted random choice; all children must have the same node type |
| `{"op":"if","cond":<pred>,"then":e1,"else":e2}` | conditional on search state; both branches must have the same node type |
| `{"op":"per_type","real":t1,"integer":t2,...}` | apply a transformer only to the positions of one variable type |
| `{"op":"prob","p":0.3,"child":t}` / `{"op":"repeat","n":2,"child":t}` | probabilistic or repeated application |
| `{"op":"code","candidate":"ope-pyt-..."}` | a validated sandboxed Python operator |

Predicates take the form `{"pred": name, "ge"|"lt": value}`. The names are `stagnation` (generations without
improvement), `evals_fraction`, `success_rate`, `feasible_fraction` and `diversity`. There is also
`{"pred":"random","p":...}`, and predicates combine with `{"pred":"and"|"or","args":[...]}` and
`{"pred":"not","arg":...}`.

## Algorithms

```json
{"dsl_version": 1, "kind": "algorithm", "name": "two-stage", "population": 16, "offspring": 16,
 "init": {"op": "lhs"}, "repair": {"op": "reflect"},
 "stages": [
   {"name": "explore", "until": {"pred": "evals_fraction", "ge": 0.5}, "parent_selection": {"op": "random"},
    "variation": {"op": "de_rand1", "F": 0.7, "CR": 0.9}, "survival": {"op": "pairwise"}},
   {"name": "exploit", "parent_selection": {"op": "tournament", "k": 3},
    "variation": {"op": "if", "cond": {"pred": "stagnation", "ge": 5},
                  "then": {"op": "seq", "children": [{"op": "copy"}, {"op": "cauchy", "sigma": 0.2}]},
                  "else": {"op": "seq", "children": [{"op": "sbx"}, {"op": "gaussian", "sigma": 0.05}]}},
    "survival": {"op": "plus"}, "step": {"op": "one_fifth"}}],
 "restart": {"op": "stagnation", "patience": 40, "keep_best": 1, "reinit": "around_best", "scale": 0.1},
 "refine": {"op": "pattern", "evals_fraction": 0.1, "step0": 0.05}}
```

The available components are:

* init: `uniform`, `lhs` or `opposition`;
* repair: `project`, `reflect`, `resample` or `problem_repair`;
* survival: `plus`, `comma`, `pairwise`, `metropolis` (with `t0` and `cooling`) or `threshold`;
* step adaptation: `constant`, `one_fifth` or `decay`;
* restarts on stagnation, which keep the best points and reinitialise by LHS, uniform sampling or around the best
  point;
* local refinement by pattern search, on a share of the budget, at the end or every k generations.

A stand-alone operator is benchmarked inside `host_algorithm(op)`, a fixed (mu+lambda) loop. Its baseline in that
host is `DEFAULT_OPERATOR`, which is uniform crossover followed by a type-aware mutation.

Because the language contains no I/O, no imports and no host access, a declarative candidate is executed only by
the trusted interpreter. This is why the language remains available for autonomous generation when OS isolation
is not available.
