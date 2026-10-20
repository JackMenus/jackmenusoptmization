# Development run report: suite `discrete_general`

Wall time 727.7 s. Stop reason: generations_completed. Ledger: 12207531 of 34000000 units used ({'analysis': 800, 'train_baselines': 750000, 'fingerprint': 54000, 'validation': 152731, 'train': 6300000, 'selection': 1350000, 'test': 3600000}).

Isolation available: True (linux namespaces (user, mount, pid, net, ipc, uts) + pivot_root + capability drop + rlimits + seccomp syscall allowlist (x86_64)).
Candidates by final status: {'baseline': 9, 'retained': 18, 'discarded': 223, 'duplicate': 7, 'rejected': 35, 'champion': 2}

## Generations

| gen | proposed | duplicates (source) | rejected | validated | duplicates (behaviour) | benchmarked | best candidate train score | best baseline train score |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 0 | 2 | 8 | 0 | 8 | 0.2650 | 0.2624 |
| 1 | 10 | 0 | 2 | 8 | 2 | 6 | 0.2342 | 0.2624 |
| 2 | 10 | 0 | 2 | 8 | 0 | 8 | 0.2342 | 0.2624 |
| 3 | 10 | 0 | 1 | 9 | 2 | 7 | 0.2256 | 0.2624 |
| 4 | 10 | 0 | 2 | 8 | 0 | 8 | 0.2133 | 0.2624 |
| 5 | 10 | 0 | 2 | 8 | 3 | 5 | 0.2107 | 0.2624 |

Scores: 0 = known optimum (or best baseline value), 1 = random-search median at the same budget, 2 = infeasible. Lower is better.

## Selection (validation split)

ope-dsl-d23ef2a66f beat the best baseline 'genetic' on the validation split (mean score 0.1736 vs 0.2600, 95% CI of paired difference [-0.1433, -0.0289], one-sided Wilcoxon p=0.00339, holm-adjusted 0.0126 < 0.05, required improvement 0.01)

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `ope-dsl-d23ef2a66f` | 30 | 0.1736 | 0.0885 | 0.1931 | 1.00 | 0.00 | 0.37 | 0.246 |
| `ope-dsl-713f0c7749` | 30 | 0.1828 | 0.0885 | 0.2170 | 1.00 | 0.00 | 0.37 | 0.354 |
| `ope-dsl-58a7ed97aa` | 30 | 0.1905 | 0.0922 | 0.2278 | 1.00 | 0.00 | 0.37 | 0.241 |
| `ope-dsl-3d1699e693` | 30 | 0.1905 | 0.0922 | 0.2278 | 1.00 | 0.00 | 0.37 | 0.229 |
| `genetic` | 30 | 0.2600 | 0.2995 | 0.1744 | 1.00 | 0.00 | 0.20 | 0.211 |
| `greedy_ls` | 30 | 0.6020 | 0.5782 | 0.5334 | 1.00 | 0.00 | 0.20 | 0.075 |
| `tabu` | 30 | 0.7004 | 0.4908 | 0.6524 | 1.00 | 0.00 | 0.10 | 0.076 |
| `simulated_annealing` | 30 | 0.7535 | 0.9057 | 0.3500 | 1.00 | 0.00 | 0.13 | 0.149 |
| `random_search` | 30 | 1.2089 | 1.0000 | 0.4413 | 0.80 | 0.00 | 0.00 | 0.070 |

## Held-out test split (evaluated once, after selection)

Suite version 2; test-split status before this run: untouched.

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `alg-dsl-f18700ed7f` | 90 | 0.1945 | 0.1802 | 0.1823 | 1.00 | 0.00 | 0.30 | 0.298 |
| `ope-dsl-d23ef2a66f` | 90 | 0.2204 | 0.2111 | 0.1950 | 1.00 | 0.00 | 0.26 | 0.245 |
| `genetic` | 90 | 0.2426 | 0.2303 | 0.1992 | 1.00 | 0.00 | 0.23 | 0.207 |
| `ope-pyt-ace837be3b` | 90 | 0.2755 | 0.2157 | 0.2414 | 1.00 | 0.00 | 0.18 | 0.482 |
| `greedy_ls` | 90 | 0.5543 | 0.3167 | 0.5545 | 1.00 | 0.00 | 0.23 | 0.083 |
| `tabu` | 90 | 0.5669 | 0.2750 | 0.6349 | 1.00 | 0.00 | 0.20 | 0.084 |
| `simulated_annealing` | 90 | 0.7691 | 0.8513 | 0.3517 | 1.00 | 0.00 | 0.02 | 0.174 |
| `random_search` | 90 | 1.1619 | 1.0000 | 0.4063 | 0.83 | 0.00 | 0.00 | 0.080 |

Best baseline on test: `genetic`.

- `ope-dsl-d23ef2a66f` vs `genetic`: mean score difference -0.0222 over 90 paired runs (95% CI [-0.0500, +0.0039]; wins 39, losses 29, ties 22; beyond 0.01: 39 better, 29 worse; one-sided Wilcoxon p=0.05)
- `alg-dsl-f18700ed7f` vs `genetic`: mean score difference -0.0481 over 90 paired runs (95% CI [-0.0798, -0.0163]; wins 46, losses 22, ties 22; beyond 0.01: 45 better, 21 worse; one-sided Wilcoxon p=0.00195)
- `ope-pyt-ace837be3b` vs `genetic`: mean score difference +0.0329 over 90 paired runs (95% CI [-0.0012, +0.0665]; wins 32, losses 47, ties 11; beyond 0.01: 32 better, 45 worse; one-sided Wilcoxon p=0.97)

Demonstration candidates: {'operator': 'ope-dsl-d23ef2a66f', 'algorithm': 'alg-dsl-f18700ed7f', 'code_operator': 'ope-pyt-ace837be3b'}
