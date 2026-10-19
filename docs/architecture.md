# Architecture

```
                         +---------------------- development layer (B) ----------------------+
                         |  features.analyze -> providers (grammar | evolution | code | LLM)   |
                         |      -> registry.add -> validation.gates -> benchmarks.run_matrix    |
                         |      -> retain/discard -> ... -> _select (validation split)          |
                         |      -> freeze -> _final_test (held-out split) -> report             |
                         +--------------------------------|-------------------------------------+
                                                          | DSLAlgorithm / SandboxedCodeOperator
 Problem --(selection.rules / evidence)--> Solver.solve --+--> SearchContext --> Evaluator --> Problem
                                                |                                   |
                                                +--> finalize (re-evaluation, guarantees) --> OptimizationResult
```

## Solution layer (A)

* **`core.space`.** Each variable has a type (real, integer, binary, categorical or permutation) and bounds. A
  `Space` maps the variables onto one flat float vector with a per-position type code. Categorical values are
  encoded as indices and a permutation of length n occupies n positions. The space provides sampling, Latin
  hypercube sampling, projection (rounding, clipping, and rank repair for permutations), a strict domain check,
  enumeration of finite spaces, and decoding to and from named values.
* **`core.problem`.** `Problem` holds the space, the objective, the sense (`min` or `max`), the constraints, and
  optional gradient, repair, verifier, noise flag, evaluation kind, convexity declaration, `LinearStructure` and
  `KnownOptimum`. `validate()` raises `ProblemDefinitionError` or `UnsupportedProblemError` for configurations the
  platform does not support. Examples are gradients on non-continuous spaces, convexity declared for discrete
  spaces, linear structure combined with permutations, more than one objective, and duplicate constraint names.
* **`core.evaluator`.** The evaluator is the only path to the objective, and it applies the following rules:
  * Points outside the domain are rejected without calling user code.
  * Exceptions and NaN or Inf values are recorded as invalid evaluations rather than crashes.
  * Evaluation, time and target budgets are enforced, along with a stage cap that composite solvers use.
  * Each evaluation can be charged to a development `BudgetLedger`.
  * Candidates are ranked feasibility-first (Deb's rules): valid before invalid, feasible before infeasible, then
    by lower violation, then by objective.
* **`core.context`.** `SearchContext` is what a solver sees. It provides evaluation, repair (projection followed by
  the problem's own repair), a stagnation tracker, guarantee hints and diagnostics.
* **`core.finalize`.** This step turns a search into an `OptimizationResult`. The best point is re-evaluated against
  the original objective and constraints, and repeatedly when the objective is noisy. Violations are recomputed
  per named constraint. The independent verifier runs when one exists. Solver hints are reconciled against the
  verified point, and any hint that refers to another point or fails verification is dropped. Status and
  guarantees are then assigned (see `guarantees.md`).
* **`solvers`.** Every solver declares its capabilities, and `incompatibilities()` explains a refusal. `solve()`
  handles budget exhaustion, cancellation and solver exceptions, always runs `close()`, and always finalises.
  `IterativeSolver` adds JSON checkpoints between iterations, covering the RNG state, the evaluator snapshot and
  the solver state. Resuming reproduces an uninterrupted run exactly for deterministic problems, which
  `tests/test_solvers.py` checks for every checkpointing solver.

  | family | solvers |
  |---|---|
  | continuous local (gradient) | `lbfgsb` (analytic or finite-difference gradient), `slsqp` (constraints) |
  | continuous local (derivative-free) | `nelder_mead`, `pattern_search` (also used on the integer coordinates of mixed problems) |
  | population global | `cmaes` (IPOP restarts), `de` (rand/1 and current-to-pbest), `genetic` (type-aware) |
  | multistart and global+local | `multistart`, `hybrid` (global stage on a share of the budget, then local refinement) |
  | discrete | `random_search`, `greedy_ls` (greedy construction plus neighbourhood search), `simulated_annealing`, `tabu`, `genetic` |
  | exact | `exhaustive` (enumeration certificate), `milp` (SciPy HiGHS on a declared `LinearStructure`, spot-checked first) |
* **`selection.rules`.** Rules R1 to R13 are applied in order, and each decision records which rule fired and why. A
  `mode` argument (`local`, `global` or `hybrid`) restricts the strategy. `selection.evidence` can override the
  rules with a champion candidate only when validated benchmark evidence matches the problem profile: same space
  kind, covered variable types, dimension within a factor of 1.5, same constraint class, budget within a factor of
  2, at least 12 paired runs, and p < alpha.
* **`dynamics`.** `DiscreteTimeSystem` is exact for its own model. `ContinuousTimeSystem` integrates with SciPy
  `solve_ivp` and enforces path constraints on a sample grid. `simulation_problem` attaches a verifier that
  re-integrates on a 10x denser grid with tolerances 100x tighter. Results are labelled
  `feasibility_verified_at_sample_points_only`, and when the denser check disagrees the status becomes
  `feasible_discretized_only`.
* **`multiobjective`.** This is an extension point. It provides weighted-sum and Chebyshev scalarisation into an
  ordinary `Problem`, weight sweeps, and non-dominated filtering. No native multi-objective solver is included.

## Development layer (B)

* **Candidates.** A candidate is either an *operator* (one variation expression, benchmarked inside a fixed
  (mu+lambda) host algorithm) or a whole *algorithm*. An algorithm has an initialiser, one or more stages each
  with its own switching predicate, parent selection, variation, survival and acceptance, step adaptation,
  restarts, repair, and optional pattern-search refinement. Candidates are written in the declarative language
  (`docs/operator_language.md`) or as sandboxed Python source.
* **Providers.** Each provider returns `Proposal`s:
  * `GrammarProvider` samples from a typed grammar whose op weights are biased by the features of the training
    problems (variable types, constraints, ruggedness probes).
  * `EvolutionProvider` mutates and crosses over retained candidates and records their parents.
  * `CodeTemplateProvider` composes Python mutation operators from code fragments. Some combinations are invalid on
    purpose, so the gates are exercised. It is enabled only when the isolation probe passes.
  * `LLMProvider` is optional. It wraps an `LLMClient`: `AnthropicClient` or the deterministic `ScriptedLLMClient`.
    Each call is charged to `model_calls`, and every proposal still goes through the static checks and the gates.
* **Registry (`registry.store`).** Each candidate is stored as append-only `vNNNN.json` versions. A record holds the
  id (a hash of the canonical source), source, parents and history, provider and rationale, compatible kinds,
  hyperparameters, validation report, benchmark entries per split, limitations and status
  (`proposed`, `rejected`, `validated`, `retained`, `discarded`, `champion`, `baseline`). Only validated,
  retained, champion and baseline records can be materialised for ordinary runs.
* **Gates (`validation.gates`).** The seven gates are:
  1. Interface and types: the static type check, or a sandboxed contract test for code.
  2. Bounds and types: no out-of-domain evaluations and no raw code outputs outside the domain.
  3. Numerical validity: no non-finite offspring, and survival of a region where the objective returns NaN.
  4. Constraint handling: the candidate reaches a feasible point.
  5. Reproducibility: the same seed gives an identical history.
  6. Termination and resource limits: the evaluation budget is respected and a 0.3 s time limit is honoured.
  7. Failure handling: an objective that raises is absorbed.
  The gates run on purpose-built validation problems, never on benchmark instances, and their evaluations are
  charged to the `validation` ledger category.
* **Loop (`generation.loop`).** The loop runs in this order:
  1. Analyse the training problems.
  2. Run the baselines on train.
  3. For each generation, propose, register, validate, benchmark on train, then retain the best k and discard the
     rest.
  4. Select on the validation split. Candidates are compared with every baseline using a paired one-sided Wilcoxon
     test against the best baseline on validation. A champion must have p < alpha, a mean score that is no worse,
     and a feasibility rate that is no worse.
  5. Freeze the selection.
  6. Demonstrate the best generated operator and composition, if any were validated.
  7. Run the held-out test split once.
  The test split is technically unreachable until `_selection_frozen` is set. `tests/test_generation.py` checks
  that no non-test benchmark call sees a test instance and that test calls come last.
* **Budgets.** `Budget` bounds one solve. `BudgetLedger` bounds development as a whole, including model calls. It
  has categories `analysis`, `validation`, `train_baselines`, `train`, `selection`, `test` and `model_calls`, a
  reserved test share, a wall-time cap, and reserve/settle accounting for work done in worker processes.

## Trust boundaries

The trusted code base is everything under `src/optiforge`: the evaluator, the benchmark definitions and scoring
(`benchmarks/runner.py`), the gates, the registry writer and the loop. Declarative candidates are data that the
trusted interpreter executes, so they cannot do I/O. Python candidates run only in the sandbox (`docs/isolation.md`).
They receive coordinates and bounds and return coordinates. They never see the objective, and they cannot write to
the platform code, benchmark definitions, registry or results.
