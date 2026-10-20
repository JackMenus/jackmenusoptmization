# optiforge

A modular optimisation platform with two layers:

* **Solution optimisation.** One `Problem` interface covers real, integer, binary, categorical, permutation and mixed
  variables, minimisation or maximisation, inequality and equality constraints, optional gradients, noisy, black-box
  and simulator objectives. A baseline portfolio of local, global, hybrid and exact solvers is included, and a
  transparent rule-based selector picks among them.
* **Algorithm development.** An autonomous loop generates new search operators and multi-stage algorithms, validates
  them, benchmarks them and retains or discards them, all within a ledgered budget. Candidates come from a typed
  declarative operator language, from evolution of earlier candidates, from a grammar of Python code fragments that
  runs only inside an OS sandbox, and optionally from a language model. Each candidate is recorded in a versioned
  registry. Candidates are selected on a validation split and reported once on a held-out test split that never
  guided any decision.

Results are reported conservatively. Every reported point is re-evaluated against the original problem.
Global optimality is claimed only from complete enumeration or a verified MILP solve, and "no feasible point found" is
never presented as infeasibility. See [docs/guarantees.md](docs/guarantees.md).

## Install

Requires Linux or macOS with Python 3.11 or newer. The test suite and experiments were run with Python 3.13.16 on
Linux x86_64 using the exact versions in `requirements-lock.txt`. A clean virtual environment was built from that
file and the package installed into it during delivery.

```sh
git clone https://github.com/JackMenus/jackmenusoptmization.git optiforge && cd optiforge
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt      # numpy 2.5.3, scipy 1.18.1, PyYAML 6.0.1, click 8.5.0, pytest 9.1.1
pip install -e .                          # installs the `optiforge` command
pip install anthropic                     # optional: only for the language-model proposal adapter
```

Sandboxed execution of generated Python code needs Linux with `unshare` and `setpriv` (util-linux), unprivileged
user namespaces, and a Python interpreter under `/usr`. Run `optiforge sandbox-check` to see whether isolation is
available. If it is not, code generation is switched off automatically and development uses only the declarative
language.

## Quick start

```python
import numpy as np
from optiforge import Problem, Space, Real, Integer, Budget, solve
from optiforge.core.constraints import ineq

space = Space([Real("x", -5, 5), Real("y", -5, 5), Integer("n", 0, 10)])
problem = Problem(space, lambda v: (v[0] - 1) ** 2 + (v[1] + 2) ** 2 + (v[2] - 3) ** 2,
                  constraints=[ineq(lambda v: v[0] + v[1] - 1.0, name="sum<=1")])
result = solve(problem, "auto", Budget(max_evaluations=5000), seed=0)
print(result.summary())
for g in result.guarantees:
    print(g.kind, "-", g.statement)
print(result.diagnostics["plan"]["rationale"])   # why this solver was chosen
```

## Command line

```sh
optiforge solvers                                    # the portfolio
optiforge problems                                   # benchmark families and suites (with splits)
optiforge plan configs/solve_rastrigin.yaml          # selection plan and rationale
optiforge solve configs/solve_milp.yaml              # -> runs/<timestamp>-solve-*/ (config, environment, events, result)
optiforge solve configs/solve_pd_control.yaml --checkpoint-every 5
optiforge solve configs/solve_pd_control.yaml --resume runs/<that run>   # continue from the checkpoint
optiforge benchmark configs/benchmark_portfolio.yaml # matched-budget, multi-seed comparison
optiforge develop configs/develop_continuous.yaml    # autonomous operator/algorithm development
optiforge inspect runs/<run> --events                # summarise any run record
optiforge registry list --path registry              # candidates, versions and statuses
optiforge registry show <candidate-id> --path registry
optiforge sandbox-check                              # verify the isolation properties from inside a sandbox
```

`--solver candidate:<id>` (with `registry:` set in the config) runs a candidate from the registry, but only if its
status is validated, retained, champion or baseline.

Every run writes a record directory containing the resolved config, an environment snapshot (Python and package
versions, platform, git revision if any), a JSON-lines event log and the result or report.

## Tests

```sh
python3 -m pytest -q            # 176 tests: unit, numerical, sandbox (skipped without isolation), CLI, integration
OPTIFORGE_REQUIRE_ISOLATION=1 python3 -m pytest -q   # fail instead of skipping when the sandbox is unavailable
```

## Layout

| path | contents |
|---|---|
| `src/optiforge/core` | variable spaces, problems, constraints, budgets and ledger, evaluator, checkpointing, result finalisation |
| `src/optiforge/solvers` | continuous, discrete, exact and composite (multistart, hybrid, pattern search) solvers |
| `src/optiforge/operators` | primitive variation operators shared by solvers and the DSL interpreter |
| `src/optiforge/dsl` | the typed operator/algorithm language (checker) and its interpreter |
| `src/optiforge/generation` | grammar and evolution generators, code-template generator, LLM adapter, development loop |
| `src/optiforge/isolation` | the Linux namespace sandbox for generated code |
| `src/optiforge/validation` | validation gates |
| `src/optiforge/registry` | versioned, append-only candidate registry |
| `src/optiforge/benchmarks` | benchmark families, suites with splits, matched-budget runner and scoring |
| `src/optiforge/selection` | rule-based and evidence-based solver selection |
| `src/optiforge/dynamics` | discrete-time and continuous-time simulation adapters |
| `src/optiforge/multiobjective` | scalarisation and Pareto utilities (an extension point, not native MO solvers) |
| `configs/` | solve, benchmark and development experiment configurations |
| `examples/` | user-defined problems (pressure vessel, production planning MILP, noisy quadratic, mixed facility) |
| `results/` | the experiment outputs and test log produced for this delivery (`results/run_all.sh` reproduces them) |
| `docs/` | architecture, operator language, isolation, guarantees, benchmarking methodology |

| `lab/` | experiments after 0.1.0: one directory per round with scripts, pre-registrations and raw outputs; `lab/EXPERIMENTS.jsonl` lists every run |

Read `DELIVERY_REPORT.md` for the 0.1.0 delivery, `CHANGELOG.md` for every change since, and `WORK_STATE.md` for the
current state of development (rounds, open issues, next tasks).

## License

MIT, see [LICENSE](LICENSE).
