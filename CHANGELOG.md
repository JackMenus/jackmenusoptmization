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

### Operator language
* New transformer `correlated_gaussian`: a rotation-invariant mutation from the elite population covariance; the
  grammar can now propose it for real variables. Existing candidates and their hashes are unchanged.
