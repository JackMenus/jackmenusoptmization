# Delivery report: optiforge 0.1.0

Date: 2026-10-19. Host: Linux x86_64, 4 CPUs, Python 3.13.16, numpy 2.5.3, scipy 1.18.1, PyYAML 6.0.1, click 8.5.0,
pytest 9.1.1.

This report contains only results that were actually observed. Every number below comes from a file in `results/` or a
run record in `results/runs/`. You can reproduce the experiments with `results/run_all.sh`; it took about 22 minutes
on this host.

## 1. What was implemented

The work followed the three phases in the order requested.

**Phase 1: core**

* **Problem interface (`core/`).** Supports real, integer, binary, categorical, permutation and mixed variables with
  bounds and a documented flat encoding, and either min or max. Further features:
  * inequality and equality constraints, each with a name and a tolerance;
  * optional analytic gradients, a repair hook, an independent verifier, and noisy, black-box or simulation
    evaluation;
  * an optional declared linear structure and an optional known optimum, the latter used only for benchmark gaps.
  * Unsupported configurations raise explicit errors.
* **Evaluator.** Rejects out-of-domain points without calling user code. Captures exceptions and NaN or Inf values
  as data rather than crashes. Enforces the evaluation, time and target budgets, charges a development ledger, and
  ranks results feasibility-first.
* **Result finalisation.** The reported point is re-evaluated against the original objective and constraints. A
  noisy objective is re-evaluated 5 times, and the mean, standard deviation and repeat count are reported. Violations
  are reported per constraint, solver claims are reconciled with the verified point, and the guarantee types in
  `docs/guarantees.md` are applied.
* **Solver portfolio, 15 solvers.**
  * Gradient local: `lbfgsb`, `slsqp`.
  * Derivative-free local: `nelder_mead`, `pattern_search`.
  * Population global: `cmaes` (IPOP), `de`, `genetic`.
  * Multistart and global+local: `multistart`, `hybrid`.
  * Discrete: `random_search`, `greedy_ls`, `simulated_annealing`, `tabu`, `genetic`.
  * Mixed: type-aware `genetic`, `de`, `hybrid` (GA followed by compass search).
  * Exact: `exhaustive` (enumeration certificate) and `milp` (HiGHS on a declared linear model, spot-checked
    against the real functions first).
* **Run controls.** Every solve supports warm starts, cooperative cancellation and per-solver stagnation restarts.
  `cmaes`, `de`, `genetic`, `greedy_ls`, `pattern_search`, `random_search`, `simulated_annealing` and `tabu` also
  support checkpoint and resume.

**Phase 2: operator generation and validation**

* **Typed declarative operator and algorithm language (`dsl/`).** Covers init, neighbourhood and mutation moves,
  recombination, selection and acceptance (plus, comma, pairwise, Metropolis, threshold), repair and projection,
  adaptive steps (1/5 rule, decay), stagnation restarts, pattern-search refinement, multi-stage algorithms with
  switching predicates, and `if`, `mix`, `per_type`, `prob` and `repeat` combinators. A static checker covers node
  types, parameter ranges, type coverage and size limits. A trusted interpreter executes the result.
* **Generators (`generation/`).**
  * A feature-biased grammar.
  * Evolution: mutation and crossover of retained candidates, with lineage recorded.
  * A code-template grammar that writes Python mutation operators. Some combinations are deliberately unsafe, and the
    gates must catch them.
  * An optional language-model adapter (`AnthropicClient`, model `claude-opus-5-5`), with a deterministic
    `ScriptedLLMClient` for offline use and tests. Model calls are charged to the budget, and every model proposal
    goes through the same checks and gates as any other.
* **OS sandbox for generated Python (`isolation/`).** Runs as an unprivileged uid with user, mount, PID, net, IPC and
  UTS namespaces, `pivot_root` onto a tmpfs that exposes only a read-only `/usr`, an empty environment, no
  capabilities, `no_new_privs`, CPU, memory, file-size, file-count and process rlimits, a per-request timeout and an
  output cap. A probe checks these properties from inside the sandbox. If the probe fails, code generation is
  disabled. See `docs/isolation.md`.
* **Seven validation gates (`validation/`).** Interface and types, bounds and types, numerical validity, constraint
  handling, reproducibility, termination and resource limits, and failure handling.
* **Versioned, append-only registry (`registry/`).** Records id, source, parents and history, provider and rationale,
  compatible kinds, hyperparameters, validation report, per-split benchmark entries, limitations and status. A
  candidate can be used for ordinary solves only when its status is validated, retained, champion or baseline.

**Phase 3: benchmark-driven selection and the remaining layers**

* **Benchmark library (`benchmarks/`).** 18 families: convex, nonconvex and multimodal continuous; constrained
  continuous, including an equality constraint and CEC g06; knapsack, max-cut, integer quadratic and TSP with exact
  optima; two mixed families; and two simulations (a continuous-time PD-control ODE and a discrete-time inventory
  model).
* **Suites and scoring.** Six suites, each with disjoint train, validation and test splits. Scoring uses
  matched budgets and seeds and is normalised to a reference value and to random search. Reports include feasibility
  rate, gaps, runtime per evaluation, variability, failure rate and paired Wilcoxon tests. See
  `docs/benchmarking.md`.
* **Development loop.** Train-split generation, then validation-split selection, then a single held-out test run.
  Selection is frozen before the test split runs, and a test asserts that order. The total budget is ledgered and
  covers analysis, validation, benchmarking, test and model calls. Demonstrations of a generated operator and a
  generated composition are included.
* **Solver selection.** Transparent selection rules R1 to R13, each with its rationale, and local, global and hybrid
  modes. An evidence-based override applies only when a validated champion matches the problem profile and has at
  least 12 significant paired validation runs.
* **Simulation adapters (`dynamics/`).** For continuous-time systems, feasibility on the sample grid is reported
  separately from a denser re-integration. Discrete-time systems are verified exactly.
* **Multi-objective extension point.** Weighted-sum and Chebyshev scalarisation, weight sweeps and Pareto filtering.
* **Interfaces.** A Python API (`optiforge.solve`, `optiforge.plan`) and a CLI with `solve`, `plan`, `develop`,
  `benchmark`, `inspect`, `registry list|show`, `sandbox-check`, `solvers` and `problems`.
* **Experiments.** YAML-driven, with run records containing the config, environment, JSON-lines events and results.
* **Supporting material.** Ten configs, four example user problems, and the docs in `docs/`.

## 2. Install and run

See `README.md`. In short:

```sh
pip install -r requirements-lock.txt && pip install -e .
python3 -m pytest -q
optiforge sandbox-check
optiforge solve configs/solve_milp.yaml
sh results/run_all.sh
```

The pinned versions were checked as follows. A fresh virtual environment was built from `requirements-lock.txt` and
the package was installed into it non-editable. `pip freeze` showed exactly the pinned versions (plus pytest's own
dependencies), and the full test suite passed against that install (`results/pytest_clean_venv.txt`).

## 3. Tests that ran

There are 111 tests and all passed, both in place (`results/pytest_output.txt`, 25.8 s) and against the clean
virtual environment. The sandbox tests ran rather than being skipped, because isolation is available on this host.

| file | covers |
|---|---|
| `test_core.py` (18) | encoding, projection, enumeration; rejection of unsupported configs; evaluator capture of errors, NaN and domain violations; budget and cancellation; maximisation; ledger reserve and settle; re-evaluated results; "no feasible found" not counted as infeasibility; penalty and repair not redefining the problem; noisy repeats |
| `test_solvers.py` (37) | every continuous solver on a sphere; discrete solvers on OneMax; permutation solvers on TSP within 10% of Held-Karp; exhaustive optimality and infeasibility certificates; refusal when the budget is too small; MILP certificate and detection of a false linear declaration; checkpoint and resume equal to an uninterrupted run for all 8 checkpointing solvers; warm start; reproducibility; budget never exceeded; time budget |
| `test_dsl.py` (13) | the static checker accepts valid specs and rejects unknown ops and keys, out-of-range parameters, producer misuse and missing type coverage; the interpreter is correct and reproducible; a multi-stage conditional algorithm with restarts and refinement; mixed `per_type`; grammar, mutation and crossover validity |
| `test_isolation.py` (12) | all probe checks pass; hostile code fails to read host files or `/etc/shadow`, write `/usr` or the runner, reach the network, see an injected `ANTHROPIC_API_KEY`, or import the platform; an infinite loop times out; 3 GiB allocation is blocked; oversize output is capped; errors are reported |
| `test_generation.py` (11) | registry versioning, history and duplicate handling; unvalidated candidates are refused; the gates pass good DSL and code and reject an unsafe code operator (no boundary handling) and a raising one; scripted-LLM proposals are charged, statically checked and stop at the model-call cap; code generation is disabled without isolation; splits are disjoint; a full development-loop integration test asserts that no non-test phase ever sees a test instance and that test runs come last |
| `test_selection_dynamics_mo.py` (13) | the selection rules; mode restriction; evidence override only with at least 12 significant runs and a matching budget; continuous-time feasibility labelled sampled-only versus discrete-time exact; the PD-control simulation; scalarisation, sweep and Pareto front; scoring excludes not-applicable runs |
| `test_cli.py` (7) | solve and inspect with run records; MILP certificate via the CLI; plan rationale; checkpoint and `--resume` through the CLI; a factory problem; a small benchmark; listing commands |

## 4. Experiments that ran and what they show

### 4.1 Baseline portfolio benchmark (`results/benchmark_portfolio.md`)

The benchmark ran 17 test-split instances (one per family) × 5 seeds × 14 solvers, with 3000 evaluations per run,
for 1190 runs in 265 s.

| solver | mean score | feasible | solved (gap ≤ 1e-6) |
|---|---|---|---|
| `auto` (rule-based selection) | 0.043 | 1.00 | 0.79 |
| `hybrid` | 0.075 | 1.00 | 0.73 |
| `de` | 0.113 | 1.00 | 0.27 |
| `genetic` | 0.321 | 0.91 | 0.25 |
| `simulated_annealing` | 0.498 | 0.93 | 0.22 |
| `random_search` | 1.098 | 0.88 | 0.02 |

These are the solvers that apply to every family. Solvers restricted to one problem class are listed in the full
table.

* Every solver beat random search in paired comparisons. For `auto`: 82 wins, 1 loss, p = 3e-15.
* `auto` beat `de` (p = 2e-8) and `genetic` (p = 5e-11).
* `auto` was **not** significantly better than `hybrid` (p = 0.97) or `cmaes` (p = 0.91). `cmaes` won more pairs
  against `auto` on the continuous families (18 against 9).
* The rules are therefore a reasonable default, not an optimum. The weakest choice was on `inventory`, where `auto`
  scored 0.40 against 0.00 for `de`, `greedy_ls` and `hybrid`.
* Population methods struggle with equality constraints. On `hyperplane_sphere`, `cmaes` scored 0.60 and `genetic`
  and `simulated_annealing` found no feasible point in any run. SLSQP-based `multistart`, which is what `auto` picks
  there, solved it.
* Two reporting fixes were made after this run and before writing it up:
  * not-applicable runs (for example `tabu` on continuous problems) are now excluded from per-family means and paired
    tests, where they had been counted as score 2;
  * reference values now name their source (analytic, DP or enumeration, or the published g06 value).
  The saved rows were rescored. The runs themselves were not repeated.

### 4.2 Example solves (`results/solve_*.log`)

| config | result |
|---|---|
| `solve_milp` (production planning) | `optimal_certified`, objective 1136, 21 evaluations, HiGHS gap 0 |
| `solve_knapsack_exact` (n = 16) | `optimal_certified` by enumerating all 65,536 points |
| `solve_rastrigin` (d = 8) | objective 4.97, a local minimum, labelled heuristic plus first-order stationary; the global value is 0 |
| `solve_pressure_vessel` (mixed) | 6343.45, feasible. This is 4.7% above the commonly cited literature value of 6059.71, which was not reproduced here and is not verified |
| `solve_facility_mixed` | 230.03, feasible, heuristic |
| `solve_pd_control` (ODE) | `feasible_discretized_only`. The point is feasible on the 120-sample grid, but the denser re-integration (1200 samples, rtol 1e-7) found an overshoot violation of 2.0e-4. The result says this rather than claiming feasibility |

### 4.3 Autonomous development runs (`results/develop_*.log`)

Each run used 6 generations × 10 proposals, retained 4, gave every solver the same budget, and ran the held-out test
split once after selection. All three ran with sandboxed code generation enabled. No language model was used. The
registry is shared across the three runs, so the candidate counts are cumulative.

**Continuous multimodal** (Rastrigin, Ackley, Styblinski-Tang, Rosenbrock; 340 s; 7.9M of 12M units)

* 55 candidates were validated and 5 were rejected by the gates.
* On validation, the best generated algorithm scored 0.0005 against 0.103 for CMA-ES. It was still **not selected**
  at the pre-specified alpha = 0.05 (p = 0.08). It wins large on Rastrigin and Styblinski-Tang and loses by about
  1e-11 on Ackley, where both reach the optimum. No champion was declared, and the selection rule was not changed
  after seeing the outcome.
* On test, the demonstrated composition `alg-dsl-ae8b1f11f2` (from evolution; parent `alg-dsl-9b5973b42a`) scored
  0.006 against 0.106 for CMA-ES (p = 0.04). It combines arithmetic crossover with a success-rate-conditional
  Gaussian, threshold acceptance, step decay and pattern-search refinement.
* **Negative finding (post-hoc diagnostic, `results/posthoc_rotation_check.json`).** The advantage comes from
  exploiting separability. On randomly rotated Rastrigin (5-D, 10 seeds, 4000 evaluations) the same algorithm reached
  a median of 15.4, against 2.0 for CMA-ES and 24.4 for random search. On the unrotated version it solved all 10
  runs. The suite contains only separable multimodal functions, which is a gap in the benchmark design that this
  exposed.

**Discrete combinatorial** (knapsack, max-cut, integer quadratic; 231 s)

* Three generated algorithms solved every validation run (score 0) but were not significantly better than tabu
  search (0.011), because most pairs were ties.
* On test, the top validation candidate `alg-dsl-6ff0002790` scored 0.022, **worse** than GA at 0.0075. The
  demonstrated operator `ope-dsl-05dea32d73` scored 0.0045 (p = 0.31). This counts as no improvement.

**Mixed** (mixed Rastrigin, mixed constrained; 223 s)

* The sandboxed generated Python operator `ope-pyt-c4d678ee14` became champion on validation, scoring 0.029 against
  0.144 for GA (p = 0.0012).
* On held-out test it scored 0.051 against 0.236 for GA, winning 30 of 30 paired runs (p = 9e-10).
* The generated DSL operator `ope-dsl-0a55ccc850` (evolution lineage) also beat GA on test (p = 2e-6).
* The generated multi-stage composition `alg-dsl-26bfd13429` did not reach significance (p = 0.06).
* The evidence was used end to end (`results/evidence_selection_*`): `optiforge plan` on an unseen instance
  (mixed_constrained, seed 500) overrode the rules with the champion. The champion found 1.70 against 2.77 for the
  rule-based hybrid. That is a single instance and should be read as anecdotal.
* Caveat: this suite has two families. The champion's evidence just meets the 12-paired-run minimum.

**Demonstrated end to end**

* Generated operators: DSL `ope-dsl-0a55ccc850` and code `ope-pyt-c4d678ee14`.
* Generated compositions: `alg-dsl-ae8b1f11f2` and `alg-dsl-26bfd13429`.
* Each was proposed, gated, benchmarked on train, retained, compared on validation and evaluated once on test.
  `optiforge registry show <id> --path registry` prints its full record.

## 5. Guarantees supported

* **Global optimality.** Certified only by complete enumeration of a finite deterministic domain, or by a MILP solve
  on a declared linear model that passed spot checks and re-verification.
* **Infeasibility.** Certified only by complete enumeration.
* **Conditional global optimality.** Stated only for a user-declared convex problem with a gradient at a stationary
  point, and labelled as conditional on that declaration.
* **Local properties.** First-order stationarity is reported only when the projected-gradient norm is measured below
  tolerance. Otherwise only "local method converged" or "heuristic" is reported. Proven local optimality is never
  claimed.
* **Feasibility.** Exact for the problem as defined. Sampled-only for continuous-time simulations.
  `feasible_discretized_only` is used when a denser check disagrees.
* **No feasible point found** is explicitly not evidence of infeasibility.
* **Noisy objectives** never receive optimality claims.

## 6. Limitations that remain

* **Sandbox.** There is no seccomp filter, so the namespaces are the boundary and the kernel syscall surface is
  reachable. It is Linux only and needs unprivileged user namespaces and an interpreter under `/usr`. Without these,
  code generation turns itself off.
* **LLM provider.** The live Anthropic path was not exercised: no API call was made, and there are no credentials in
  this environment. Only the scripted client was tested. The adapter code follows the current SDK, but its behaviour
  against the real API is unverified.
* **Benchmark design.** The multimodal continuous families are shifted but not rotated, which favours coordinate-wise
  operators (see 4.3). Each family has only 2 to 3 instances per split, so statistical power is low. In the discrete
  suite, many runs reach the exact optimum, which produces ties that make significance hard to reach.
* **Ties in paired tests.** The Wilcoxon test treats differences of order 1e-11 as wins or losses. A tie tolerance
  would change the continuous selection outcome. It was deliberately not added after seeing the results.
* **Results.** The pressure vessel example is 4.7% above the literature value. Population methods are weak on
  equality constraints. Greedy construction for permutations has no problem-specific heuristic. The rule-based
  `auto` choice is poor on the inventory simulation.
* **Multi-objective.** Only scalarisation utilities exist. There are no native Pareto solvers such as NSGA-II.
* **Learned selection.** Selection is an evidence-gated override of a single champion. No learned model of algorithm
  performance exists.
* **Checkpointing.** Supported by 8 of the 15 solvers. The SciPy-wrapped local methods (`lbfgsb`, `slsqp`,
  `nelder_mead`) and the composites cannot resume mid-run.
* **One development-loop bug, found and fixed during this delivery.** When the same candidate was both the best
  operator and the best code operator, the held-out test ran it twice, which is why `ope-pyt-8a2643a783` shows
  120 runs in the continuous report. The runs are deterministic duplicates, so the means are unaffected. The fix took
  effect for the mixed run.
* **Delivery.** There is no remote repository. The code is a local git repository.
