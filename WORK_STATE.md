# WORK_STATE

Recovery file for continued development. Check it against `git log` and `git status` before resuming.

Last updated: 2026-10-20 (session of the project thread "optiforge 自主研发").

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
  `requirements-lock.txt`, `pip install -e .`). Commits are then pulled into the project-folder repository with
  `git -C /mnt/project-files/optimization-platform pull --ff-only <clone> master`.
* Tests: `OPTIFORGE_REQUIRE_ISOLATION=1 python -m pytest -q -p no:cacheprovider` (the variable makes the isolation
  tests fail instead of skipping when the sandbox probe fails).
* Experiments: every run is listed in `lab/EXPERIMENTS.jsonl` (code version, command, config, seeds, raw output
  location, result). Raw outputs are under `lab/<round>/`.

## Rounds completed

| round | commit | change | verification |
|---|---|---|---|
| 0 | `7046745` (delivered 0.1.0) | baseline | 111/111 tests pass in the locked env; `optiforge sandbox-check` passes; portfolio benchmark re-run reproduces all 1190 runs exactly (`lab/r0-baseline`) |
| 1 | see `git log` ("sandbox: seccomp ...") | seccomp-BPF allowlist in the sandbox runner; probe requires it; reply ids; capped worker restarts; `OPTIFORGE_REQUIRE_ISOLATION` | 118 tests; 7 new hostile tests (fork, exec, subprocess, os.system, sockets incl. unix/netlink, threads, kill, ptrace, unshare, prctl, forged replies, restart cap); the 36 Python candidates in `registry/` give identical outputs before and after (`lab/r1-seccomp`) |
| 2 | see `git log` ("certificates: ...") | 12 audit defects in certificates and statuses fixed (see CHANGELOG) | 138 tests (20 new in `tests/test_certificates.py`); audit reproduction scripts in `lab/r2-certificates/d*.py`; six example solves unchanged; portfolio benchmark regression in `lab/EXPERIMENTS.jsonl` |

## Audit findings still open

From the 2026-10-20 independent reviews (scripts in `lab/r2-certificates/`):

* SLSQP (inside `slsqp`, `multistart`) calls constraint functions outside the Evaluator: not counted, not
  time-gated, not cancellable (`solvers/continuous.py` around the SLSQP constraint wrappers). Matters for
  simulation constraints.
* `Space.domain_violations` accepts integer-like values within 1e-9 and passes the unrounded vector to user code;
  categorical decode of values below -0.5 wraps to the last choice.
* Noisy verification takes feasibility from the last repeat only.
* Resuming a checkpoint taken at a converged iteration iterates again (pattern_search 136 vs 132 evaluations, same
  x); `guarantee_hints` are not checkpointed.
* Sandbox: `RLIMIT_NPROC` only set as root (fork is now blocked by seccomp, so this is defence in depth only);
  benchmark runner uses `fork` pools, which could inherit live sandbox pipes.
* Empirical escape probing of the sandbox (hostile code beyond the tests) was not completed by the reviewer.

## Known limitations (carried from DELIVERY_REPORT, re-checked)

* Live Anthropic proposal path: **unverified** (no credentials, no budget). Only the scripted client is tested.
* Multi-objective: scalarisation only.
* Benchmarks: multimodal continuous families are separable; the generated continuous algorithm fails on rotated
  Rastrigin (post-hoc diagnostic). Each family has 2-3 instances per split.
* The test-split instances of the three development suites (instance seeds 200-202) and the portfolio test
  instance (seed 300) were viewed in the 0.1.0 delivery and in round 0. They are now **historical diagnostic /
  regression data**; new confirmatory claims need fresh, versioned test instances.

## Next tasks (directly executable)

1. Round 3 (P1 + experiment rules): add rotated / non-separable / ill-conditioned continuous families (rotated
   Rastrigin, rotated Ackley, Schwefel 1.2 / rotated ellipsoid, Rosenbrock already non-separable) and versioned
   split definitions (`splits v2`) with fresh test seeds; keep v1 for regression. Then re-run the continuous
   development loop on train/validation only and freeze before any test.
2. Route SLSQP constraint calls through the Evaluator (P0, simulation constraints).
3. Discrete suite: analyse ties (most runs hit the optimum) — larger instances or a fixed-budget gap metric.
4. Selection: count feature-probe cost; compare `auto` with `hybrid` and `cmaes` on the new families.

## Blockers

* None for offline work. Online LLM validation is blocked on credentials and a budget from Jack.
