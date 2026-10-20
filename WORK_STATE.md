# WORK_STATE

Recovery file for continued development. Check it against `git log` and `git status` before resuming.

Last updated: 2026-10-20, after round 14 (session of the project thread "optiforge 自主研发迭代").

## Goal and priorities

Jack's standing brief (2026-10-20): develop optiforge autonomously in rounds
(inspect → choose → implement → test → experiment → review → save → next).

* P0: correctness of statuses and certificates, isolation of generated code, reproducibility.
* P1: algorithm generalisation (rotated, non-separable, ill-conditioned, constrained families), mixed and discrete
  improvements, selection against strong baselines with the selection cost counted.
* P2: live Claude proposal path (only with credentials and a budget), native multi-objective, CLI/CI polish.

Authorisations that are **not** given: pushing to any remote, publishing, paid API calls (no credentials or budget
exist), tool/connector/device setup. The repository lives only in the project folder
`/mnt/project-files/optimization-platform/` (branch `master`).

## How the work is done

* Development happens in a local clone (`/home/claude/dev/optiforge`, venv `.venv` built from
  `requirements-lock.txt`, `pip install -e .`). Commits are then brought into the project-folder repository with
  `cd /tmp && git -C /mnt/attach/project-files/optimization-platform fetch <clone> master && git -C
  /mnt/attach/project-files/optimization-platform merge --ff-only FETCH_HEAD` (running git with cwd `/home/claude`
  or using `pull` failed with "Unable to read current working directory"). A new session can clone from the project
  folder instead: `git clone /mnt/project-files/optimization-platform/ <dir>` (needs `git config --global
  safe.directory '*'`).
* Tests: `OPTIFORGE_REQUIRE_ISOLATION=1 python -m pytest -q -p no:cacheprovider` (the variable makes the isolation
  tests fail instead of skipping when the sandbox probe fails). 170 tests as of round 14.
* Experiments: every run is listed in `lab/EXPERIMENTS.jsonl` (code version, command, config, seeds, raw output
  location, result). Raw outputs are under `lab/<round>/`. Pre-registrations (`PREREG.md`) are written before a run
  that touches an untouched test split.

## Rounds completed

| round | commit | change | verification |
|---|---|---|---|
| 0 | `7046745` (delivered 0.1.0) | baseline | 111/111 tests; sandbox-check passes; portfolio benchmark re-run reproduces all 1190 runs exactly (`lab/r0-baseline`) |
| 1 | `32e71af` | seccomp-BPF allowlist in the sandbox runner; probe requires it; reply ids; capped worker restarts; `OPTIFORGE_REQUIRE_ISOLATION` | 7 hostile tests; 36 registry code operators give identical outputs before/after (`lab/r1-seccomp`) |
| 2 | `42e5aab` | 12 audit defects in certificates and statuses (MILP certified black-box functions, verifier/non-determinism, error status, ...) | 20 regression tests; six example solves unchanged; portfolio benchmark identical (`lab/r2-certificates`) |
| 3 | `b72d9c1`, `e63c972` | rotated / non-separable families, suite `continuous_general` v2; characterization on train+validation | `lab/r3-continuous`: the 0.1.0 candidate fails on rotated Rastrigin (0.80 vs CMA-ES 0.10-0.20) |
| 4 | `eceaa64` | SLSQP constraint calls through the evaluator; declared constraint Jacobians | portfolio: 48/1190 runs differ, only in (now honest) evaluation counts (~5x) and objective changes <= 3e-8 (`lab/r4-slsqp`) |
| 5 | `3f1d338` | effect sizes, practical threshold, Holm-corrected selection; DSL `correlated_gaussian` | unit tests; docs/benchmarking.md |
| 5b | `47be67a` | fixes from the second independent review (MILP mixed-integer certificates, ties, verifier errors, SLSQP vector constraints, cache key, test-access ledger, held-out family) | review scripts in `lab/r5-review2`; 200/200 random mixed-integer models certify |
| 6 | (this commit) | development loop on `continuous_general` v2, pre-registered in `lab/r6-develop-v2/PREREG.md` | no champion; test: hybrid 0.047, cmaes 0.051, best candidate 0.056; **no improvement shown**; test split now historical (`lab/r6-develop-v2`) |
| 7 | (this commit) | behavioural fingerprints: validated candidates that behave like an earlier candidate or a baseline are not benchmarked (status `duplicate`) | 2 new tests; round-6 re-fingerprint: 45 candidates, 38 behaviours, fingerprint groups match identical train means exactly (`lab/r7-dedup`) |
| 8 | (this commit) | discrete suite `discrete_general` v2 (4 new exactly-solved families); evaluator snaps near-domain points; categorical decode range check | brute-force optimum tests; pilot + train/validation characterisation (`lab/r8-discrete-v2`); portfolio 1190/1190 identical |
| 9 | (this commit) | opt-in binary exchange moves for tabu / SA | validation: tabu+exchange beats tabu (CI below 0), SA no effect, both behind genetic; defaults unchanged (`lab/r9-exchange`) |
| 10 | `df8c9f1` | pre-registered development loop on `discrete_general` v2 + exchange confirmation (one test pass) | champion not confirmed on test (CI [-0.050, +0.004]); exchange confirmed (CI [-0.256, -0.134]); test split historical (`lab/r10-develop-discrete`) |
| 10b | `29b56a3` | audit fixes: noisy feasibility over all repeats, converged checkpoint resume, fork-safe sandbox objects, evidence selector requires adjusted p + CI | each regression test fails on the old code |
| 11 | `ae540ad` | tabu 1.1.0: exchange moves by default | portfolio: 5/1190 runs changed (tabu on knapsack, 4 better, 1 worse) |
| 12 | (this commit) | suite `discrete_general_v3`; pre-registered confirmation of the round-10 candidates | neither confirmed vs genetic (Holm p 0.071; CIs include 0); both better only on multi_knapsack (`lab/r12-confirm-discrete`) |
| 13 | (this commit) | pre-registered multi-knapsack hypothesis on 20 new instances (4 sizes) | supported for both candidates (CI below 0; ~0.6-1.0 pp smaller relative gap); family-specific (`lab/r13-mkp`) |
| 14 | (this commit) | ablation of the multi-knapsack gain; new baseline `host_ga`, added to the loop's default baselines | the gain is the host's plus-survival, not the generated operator (`lab/r14-ablation`) |

## Audit findings still open

* Status `error` is kept for "every evaluation non-finite" (documented in docs/guarantees.md; the message gives counts).
* Sandbox: `RLIMIT_NPROC` is only set as root (seccomp blocks fork and clone, so this is defence in depth). Fork-pool
  children now detach inherited sandbox objects. Empirical escape probing beyond the tests was not done.

## Known limitations (carried from DELIVERY_REPORT, re-checked)

* Live Anthropic proposal path: **unverified** (no credentials, no budget). Only the scripted client is tested.
* Multi-objective: scalarisation only.
* Benchmarks: each family has 2-3 instances per split; one generator seed per development run.
* Every suite's test split is now historical: v1 suites (viewed in 0.1.0 and round 0), `continuous_general` v2
  (round 6), `discrete_general` v2 (round 10) and `discrete_general_v3` (round 12). New confirmatory claims need new suite versions.
* Normalised scores on correlated knapsacks are sensitive: the random-search median is within about 2% of the
  optimum, so small gaps give large score differences; raw gaps are in the run records.

## Next tasks (directly executable)

1. Round 14 showed the round-13 gain is the host (plus-survival), now available as `host_ga`. Next: compare
   `host_ga` with `genetic` across discrete and mixed families on development instances; if it is broadly better,
   pre-register a test on fresh instances before changing any default (selection rules use `genetic`).
2. Mixed suite v2: the 0.1.0 mixed champion does not transfer to discrete problems (round 8) and its record
   predates the current selection rule (now ignored by the evidence selector). Re-evaluate it on new mixed instances.
3. Selection: count the feature-probe cost of `auto`; compare it with `hybrid` / `genetic` with practical thresholds.
4. Continuous: v2 test historical; a v3 suite is needed before any new continuous claim.

## Blockers

* None for offline work. Online LLM validation is blocked on credentials and a budget from Jack.
