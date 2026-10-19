# Development run report: suite `discrete_combinatorial`

Wall time 230.9 s. Stop reason: generations_completed. Ledger: 4620670 of 12000000 units used ({'analysis': 480, 'train_baselines': 216000, 'validation': 165190, 'train': 2862000, 'selection': 432000, 'test': 945000}).

Isolation available: True (linux namespaces (user, mount, pid, net, ipc, uts) + pivot_root + capability drop + rlimits).
Candidates by final status: {'baseline': 7, 'retained': 8, 'discarded': 100, 'rejected': 12}

## Generations

| gen | proposed | duplicates | rejected | validated | benchmarked | best candidate train score | best baseline train score |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 0 | 2 | 8 | 8 | 0.0076 | 0.0128 |
| 1 | 10 | 0 | 1 | 9 | 9 | 0.0000 | 0.0128 |
| 2 | 10 | 0 | 1 | 9 | 9 | 0.0000 | 0.0128 |
| 3 | 10 | 0 | 1 | 9 | 9 | 0.0000 | 0.0128 |
| 4 | 10 | 0 | 2 | 8 | 8 | 0.0000 | 0.0128 |
| 5 | 10 | 0 | 0 | 10 | 10 | 0.0000 | 0.0128 |

Scores: 0 = verified optimum (or best baseline value), 1 = random-search median at the same budget, 2 = infeasible. Lower is better.

## Selection (validation split)

no retained candidate beat the best baseline 'tabu' on the validation split at alpha=0.05; the baselines are retained for ordinary use

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `alg-dsl-6ff0002790` | 18 | 0.0000 | 0.0000 | 0.0000 | 1.00 | 0.00 | 1.00 | 0.165 |
| `alg-dsl-8eb0fbeb7a` | 18 | 0.0000 | 0.0000 | 0.0000 | 1.00 | 0.00 | 1.00 | 0.154 |
| `alg-dsl-00bef57051` | 18 | 0.0000 | 0.0000 | 0.0000 | 1.00 | 0.00 | 1.00 | 0.156 |
| `tabu` | 18 | 0.0113 | 0.0000 | 0.0479 | 1.00 | 0.00 | 0.94 | 0.033 |
| `ope-dsl-05dea32d73` | 18 | 0.0130 | 0.0000 | 0.0480 | 1.00 | 0.00 | 0.89 | 0.128 |
| `genetic` | 18 | 0.0138 | 0.0000 | 0.0512 | 1.00 | 0.00 | 0.89 | 0.103 |
| `simulated_annealing` | 18 | 0.3383 | 0.0000 | 0.5418 | 1.00 | 0.00 | 0.67 | 0.093 |
| `random_search` | 18 | 0.6642 | 0.9844 | 0.5591 | 1.00 | 0.00 | 0.39 | 0.032 |

## Held-out test split (evaluated once, after selection)

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `ope-dsl-05dea32d73` | 45 | 0.0045 | 0.0000 | 0.0218 | 1.00 | 0.00 | 0.96 | 0.149 |
| `genetic` | 45 | 0.0075 | 0.0000 | 0.0271 | 1.00 | 0.00 | 0.91 | 0.110 |
| `ope-pyt-2ce37d2df8` | 45 | 0.0138 | 0.0000 | 0.0724 | 1.00 | 0.00 | 0.93 | 0.285 |
| `alg-dsl-6ff0002790` | 45 | 0.0222 | 0.0000 | 0.0998 | 1.00 | 0.00 | 0.93 | 0.155 |
| `tabu` | 45 | 0.0439 | 0.0000 | 0.1305 | 1.00 | 0.00 | 0.87 | 0.035 |
| `simulated_annealing` | 45 | 0.3485 | 0.0000 | 0.4711 | 1.00 | 0.00 | 0.60 | 0.092 |
| `random_search` | 45 | 0.9363 | 1.0000 | 0.5322 | 1.00 | 0.00 | 0.13 | 0.032 |

Best baseline on test: `genetic`.

- `ope-dsl-05dea32d73` vs `genetic`: mean score difference -0.0030 over 45 paired runs (wins 4, losses 2, ties 39; one-sided Wilcoxon p=0.307)
- `alg-dsl-6ff0002790` vs `genetic`: mean score difference +0.0148 over 45 paired runs (wins 4, losses 3, ties 38; one-sided Wilcoxon p=0.411)
- `ope-pyt-2ce37d2df8` vs `genetic`: mean score difference +0.0063 over 45 paired runs (wins 4, losses 3, ties 38; one-sided Wilcoxon p=0.408)

Demonstration candidates: {'operator': 'ope-dsl-05dea32d73', 'algorithm': 'alg-dsl-6ff0002790', 'code_operator': 'ope-pyt-2ce37d2df8'}
