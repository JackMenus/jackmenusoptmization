# Changelog

## Unreleased

### Security
* Generated Python operators now run under a seccomp-BPF system-call allowlist (x86_64) loaded inside the sandbox
  before the candidate is compiled. Process creation, exec, sockets, ptrace, signals to other processes, mount and
  namespace calls, bpf, keyctl, io_uring and prctl are denied. `optiforge sandbox-check` requires the new checks
  `seccomp_filter_active`, `fork_blocked`, `exec_blocked` and `socket_blocked`; if any fails, generated code is
  disabled. On non-x86_64 hosts generated code is now disabled (previously it ran without a filter).
* Sandbox replies must echo the request id and be JSON objects; otherwise the worker is killed (previously a
  non-object reply leaked the worker process, and extra lines desynchronised later replies).
* A code operator restarts its sandbox worker at most 3 times, so the per-worker CPU rlimit cannot be renewed
  indefinitely. Failed workers are now closed (their temporary directories were leaked before).
* `OPTIFORGE_REQUIRE_ISOLATION=1` makes the isolation tests fail rather than skip when the probe fails.

### Correctness (certificates and statuses)
* **Behaviour change.** `milp` no longer issues `global_optimum_certified` or `infeasibility_certified` for a
  `linear=` declaration attached to user-written functions. Such problems now get
  `global_optimum_conditional_on_assumptions` (status `feasible`). Build the problem with the new
  `Problem.from_linear(space, model, sense=..., tol=...)` to obtain certificates; `examples.production_planning`
  was migrated and still certifies 1136. Previously a declaration that was wrong only near the optimum, or declared
  constraints on an unconstrained problem, were certified.
* `milp` checks the declaration at the box corners and at the solution point as well as at random points (default
  50, minimum 20), solves the tolerance-relaxed model, honours `Budget.max_time` and reports a HiGHS time-out as
  termination `budget_time` without any statement.
* `finalize` drops certificates when the problem proves non-deterministic between search and verification, and
  drops global certificates when the independent verifier rejects the point or raises.
* Status `error` is now used when every evaluation failed, when nothing was evaluated, and when the verification of
  the reported point failed (previously `no_feasible_found`, whose text said the point "minimises constraint
  violation").
* A constraint that raises while `finalize` builds the per-constraint report no longer crashes `solve()`.
* `lbfgsb` no longer makes the convexity-conditional statement for maximisation problems.
* `Space.encode` distinguishes categorical choices that compare equal but differ in type (`1` and `True`).

### Accounting
* `slsqp` (and `multistart`/`hybrid`, which use it) obtains constraint values only through the evaluator, so every
  constraint call is counted, budget- and time-gated and cancellable. On the portfolio benchmark this raises the
  reported evaluation counts of SLSQP runs on constrained problems about fivefold (finite-difference constraint
  Jacobians were previously free); objectives and feasibility are unchanged. A `gradient=` on a constraint is now
  passed to SLSQP as its Jacobian.
* `finalize` takes per-constraint violations from the verification evaluation instead of calling the constraints
  again.

### Experiment methodology
* `paired_comparison` reports a bootstrap 95% CI of the mean paired difference, the median difference, the
  matched-pairs rank-biserial effect size and practical wins/losses/ties at `min_effect` (default 0.01 score units).
* **Behaviour change.** Development-loop selection now requires a Holm-adjusted p-value below alpha, a mean
  improvement of at least `min_effect`, and a CI entirely below zero (`multiple_testing: none`, `min_effect: 0`
  restore the 0.1.0 rule). The evidence-based selector uses the adjusted p-value when it is recorded.
* Suites carry a `version` and a `test_status`; version-1 suites whose test splits were reported in 0.1.0 are marked
  historical. Suite `continuous_general` (version 2) adds held-out test-only families.

* Round 6 ran the pre-registered development loop on `continuous_general` v2 (`lab/r6-develop-v2`): no champion,
  no improvement shown on test. That suite's test split is now marked historical.

### Behavioural de-duplication (round 7)
* The development loop fingerprints each validated candidate (evaluation sequence on one train instance per family,
  seed 0, 200 evaluations each) and does not benchmark candidates that behave like an earlier candidate or a
  baseline (status `duplicate`). Re-fingerprinting round 6: 45 benchmarked candidates had 38 distinct behaviours; one
  group of 7 code-template operators (4 of them filled the archive) and one pair of DSL operators. Equal fingerprints
  coincided exactly with equal train results (`lab/r7-dedup`).

### Discrete benchmark suite v2 (round 8)
* New families with exact optima: `knapsack_correlated` (dynamic programming), `multi_knapsack`, `independent_set`
  and `set_cover` (MILP with HiGHS at relative gap 0; the rounded solution is re-checked), and suite
  `discrete_general` v2 (fresh instance seeds 2010-2032, held-out test family `set_cover`). Sizes were calibrated on
  pilot seeds that no split uses. Brute-force tests confirm the optima on small instances.
* Expensive exact optima are cached per process, and the benchmark runner builds every instance once before forking.
* `integer_quadratic` enumerates in chunks (same optimum and tie-breaking, bounded memory).

### Rule selection uses host_ga (rounds 15-16)
* **Behaviour change.** `select_by_rules(..., ga="host_ga")` is the new default for rules R10 (constrained
  discrete), R12 (mixed with permutations) and R13 (mixed hybrid's global stage); `ga="genetic"` restores 0.1.0.
  Pre-registered confirmation on 55 fresh instances (`lab/r16-host-confirm`): host_ga beat genetic (mean -0.042,
  CI [-0.065, -0.020]) and the new rules beat the old ones (mean -0.022, CI [-0.042, -0.003]).
* Known regression: on `mixed_constrained` the new rules are worse (score 0.079 vs 0.017); also slightly worse on
  set_cover and independent_set. The benchmark runner passes `auto` parameters to the rules.

### Host control baseline (round 14)
* Ablation on development instances (`lab/r14-ablation`): the multi-knapsack advantage of the round-10 candidates
  comes from the operator language's (mu+lambda) host, not from the generated operators (the default operator in the
  same host scores 0.225 vs the champion's 0.212 and genetic's 0.318; comma survival 0.796).
* New solver `host_ga` (that host with the default operator). **Behaviour change:** it is now one of the development
  loop's default baselines for every problem kind, so a generated operator must beat the default operator in the same
  host. Rounds 6 and 10 ran without this control.

### Multi-knapsack hypothesis (round 13)
* Pre-registered on 20 new multidimensional knapsack instances (n 40/60, m 5/10): both round-10 GA variants beat
  genetic (Holm p 7e-6, CIs below zero), reducing the mean relative gap to the optimum by about 0.6-1.0 percentage
  points at 5000 evaluations. The advantage is specific to this family (round 12 found none elsewhere).

### Confirmation on fresh discrete instances (round 12)
* Suite `discrete_general_v3` (fresh seeds, 5 test instances per family). Pre-registered confirmation of the two
  round-10 candidates against genetic: neither confirmed (Holm p 0.071 for both; CIs include zero). Both are better
  only on `multi_knapsack`, which is a post-hoc hypothesis. Registry records carry the result as a limitation.

### Development loop on discrete_general v2 (round 10) and tabu 1.1.0 (round 11)
* Pre-registered run (`lab/r10-develop-discrete`): a champion was selected on validation (beat genetic, CI below
  zero) but was **not confirmed** on test (CI [-0.050, +0.004]); the registry record carries that limitation. The
  best generated algorithm (not the selected champion) did better than genetic on test (CI [-0.080, -0.016]); that is
  exploratory and needs fresh instances. The `discrete_general` v2 test split is now historical.
* **Behaviour change.** The pre-registered secondary analysis confirmed exchange moves for tabu on the test split
  (mean -0.19 vs tabu 1.0.0, CI [-0.256, -0.134]), so `tabu` 1.1.0 uses them by default (`exchange=False` restores
  1.0.0). Portfolio benchmark: only the five tabu runs on knapsack changed (mean 421.2 -> 435.8, maximise).

### Exchange moves for binary problems (round 9)
* `operators.primitives.binary_exchanges` / `random_exchange` (one 1 -> 0 plus one 0 -> 1); `neighborhood(...,
  exchange=True)`. Opt-in parameters `tabu(exchange=True)` (half of the sampled neighbourhood from exchanges, both
  coordinates become tabu) and `simulated_annealing(exchange_prob=...)`. Defaults unchanged (portfolio 1190/1190
  identical). Validation evidence (`lab/r9-exchange`): tabu improves (mean -0.18, CI below zero), SA does not; both
  remain behind `genetic`. Not yet confirmed on test.

### Fixes from the audit list (round 10, parallel branch)
* Noisy problems: the reported point counts as feasible only if all 5 verification repeats are valid and feasible
  (it was the last repeat only); the violation report uses the worst repeat; `noise.feasible_repeats` is reported.
* Resuming from a checkpoint written at the terminating iteration returns that termination immediately (it used to
  iterate again: pattern_search 136 vs 132 evaluations); guarantee hints and diagnostics are checkpointed.
* Sandbox objects inherited by a `fork` child (benchmark worker pools) are detached in the child: it can no longer
  send requests to, or kill on garbage collection, the parent's sandbox worker (before the fix a child's request
  succeeded). `close()` also closes the worker's pipes.
* **Behaviour change.** The evidence selector ignores champions whose validation record has no multiplicity-adjusted
  p-value or CI (selections made before 2026-10-20, including the 0.1.0 mixed champion `ope-pyt-c4d678ee14`); they
  stay in the registry and can be re-selected under the current rule.

### Fixes from the audit list (round 8)
* The evaluator passes user code the exact in-domain point: discrete coordinates within 1e-9 of an integer are
  rounded and reals within 1e-9 of a bound are clipped (recorded points too). Portfolio benchmark: 1190/1190 runs
  unchanged.
* `Space.decode` raises on a categorical code outside the choice range instead of wrapping to the last choice.

### Operator language
* New transformer `correlated_gaussian`: a rotation-invariant mutation from the elite population covariance; the
  grammar can now propose it for real variables. Existing candidates and their hashes are unchanged.

### Fixes from the second independent review (2026-10-20)
* `milp` on `Problem.from_linear` problems with real variables usually failed verification (the tolerance-relaxed
  optimum sat just outside the tolerance) and lost the certificate. It now takes the relaxed model's bound, finds a
  point that verifies (relaxed optimum, else strict model, else strict model tightened by 1e-7 / 1e-6), and
  certifies with an explicit objective tolerance. 200/200 random mixed-integer models certify at tol 1e-9.
* Model problems skip spot checks (they are redundant), which also stops tied optima from losing the certificate
  and lets the adapter work with budgets of a few evaluations.
* An infeasibility certificate is dropped when the verification evaluation fails (status `error`).
* A verifier that raises no longer produces the "denser re-simulation found violations" statement; it drops global
  certificates and keeps status `feasible`.
* SLSQP skips a start point whose constraints cannot be evaluated (vector constraints previously crashed the run);
  declared constraint Jacobians are counted as gradient evaluations and are time- and cancel-gated.
* `correlated_gaussian` caches its factor by population content, not `id()`.
* Selection compares the paired mean difference (not unpaired means) with `min_effect`; the rank-biserial effect
  uses the same 1e-9 zero threshold as the win/tie counts; the evidence selector also requires the CI below zero and
  a mean improvement of at least `min_effect` (0.01 for older records).
* The development loop records each test-split run in `<registry>/test_access.jsonl` and warns in the report when
  the suite version's test split was run before.
* Suite `continuous_general`: the held-out family `styblinski_tang` (used in 0.1.0 training) is replaced by the new
  `griewank_rotated`; instance generation of the new families now depends on the dimension. The round-3
  characterization used the earlier instance definitions (commit b72d9c1).
* Adding `correlated_gaussian` to the grammar changes its random proposal stream: development runs of earlier
  versions do not reproduce from their seeds on this version (existing candidates and their hashes are unchanged).
