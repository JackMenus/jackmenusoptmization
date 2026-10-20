# WORK_STATE

Recovery file for continued development. Check it against `git log` and `git status` before resuming.

Last updated: 2026-10-20, after round 8 (session of the project thread "optiforge 自主研发迭代").

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
  tests fail instead of skipping when the sandbox probe fails). 161 tests as of round 8.
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

## Audit findings still open

* Noisy verification takes feasibility from the last repeat only.
* Resuming a checkpoint taken at a converged iteration iterates again (pattern_search 136 vs 132 evaluations, same
  x); `guarantee_hints` are not checkpointed.
* Status `error` is also used when every evaluation returns non-finite values (a reviewer questioned whether that is
  an execution failure or an unsuccessful search).
* Sandbox: `RLIMIT_NPROC` only set as root (fork is blocked by seccomp, so this is defence in depth only);
  benchmark runner uses `fork` pools, which could inherit live sandbox pipes. Empirical escape probing beyond the
  tests was not completed.
* Evidence selector: champions recorded before 2026-10-20 have no adjusted p-value and pass on the raw one.

## Known limitations (carried from DELIVERY_REPORT, re-checked)

* Live Anthropic proposal path: **unverified** (no credentials, no budget). Only the scripted client is tested.
* Multi-objective: scalarisation only.
* Benchmarks: multimodal continuous families are separable; the generated continuous algorithm fails on rotated
  Rastrigin (post-hoc diagnostic). Each family has 2-3 instances per split.
* The test-split instances of the three development suites (instance seeds 200-202) and the portfolio test
  instance (seed 300) were viewed in the 0.1.0 delivery and in round 0. They are now **historical diagnostic /
  regression data**; new confirmatory claims need fresh, versioned test instances.

## Next tasks (directly executable)

1. (Optional) rerun the development loop with de-duplication on a fresh suite version (`continuous_general` v3, new
   instance seeds); v2's test split is historical.
2. Discrete neighbourhoods: on `discrete_general` v2 (train+validation) tabu and greedy_ls are worse than the
   random-search median on correlated knapsacks (single-flip moves stall at tight capacity). Develop exchange (add-one /
   drop-one) moves as a new solver version, compare on train/validation, pre-register, then run the v2 test split once.
3. Mixed suite: reproduce the 0.1.0 code-operator champion on new independent instances and constraint scenarios.
4. Selection: count the feature-probe cost of `auto`; compare it with `hybrid` and `cmaes` with practical thresholds.
5. Remaining audit items above.

## Blockers

* None for offline work. Online LLM validation is blocked on credentials and a budget from Jack.
