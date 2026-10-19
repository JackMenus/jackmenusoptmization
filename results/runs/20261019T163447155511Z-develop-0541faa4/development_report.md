# Development run report: suite `continuous_multimodal`

Wall time 340.3 s. Stop reason: generations_completed. Ledger: 7934918 of 12000000 units used ({'analysis': 640, 'train_baselines': 307257, 'validation': 171580, 'train': 5280000, 'selection': 690471, 'test': 1484970}).

Isolation available: True (linux namespaces (user, mount, pid, net, ipc, uts) + pivot_root + capability drop + rlimits).
Candidates by final status: {'baseline': 4, 'discarded': 51, 'retained': 4, 'rejected': 5}

## Generations

| gen | proposed | duplicates | rejected | validated | benchmarked | best candidate train score | best baseline train score |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 0 | 1 | 9 | 9 | 0.0012 | 0.1150 |
| 1 | 10 | 0 | 0 | 10 | 10 | 0.0012 | 0.1150 |
| 2 | 10 | 0 | 1 | 9 | 9 | 0.0012 | 0.1150 |
| 3 | 10 | 0 | 1 | 9 | 9 | 0.0007 | 0.1150 |
| 4 | 10 | 0 | 1 | 9 | 9 | 0.0007 | 0.1150 |
| 5 | 10 | 0 | 1 | 9 | 9 | 0.0007 | 0.1150 |

Scores: 0 = verified optimum (or best baseline value), 1 = random-search median at the same budget, 2 = infeasible. Lower is better.

## Selection (validation split)

no retained candidate beat the best baseline 'cmaes' on the validation split at alpha=0.05; the baselines are retained for ordinary use

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `alg-dsl-a688ed739e` | 24 | 0.0005 | 0.0000 | 0.0013 | 1.00 | 0.00 | 0.50 | 0.119 |
| `alg-dsl-ae8b1f11f2` | 24 | 0.0170 | 0.0000 | 0.0805 | 1.00 | 0.00 | 0.46 | 0.106 |
| `alg-dsl-b3d8eb36a5` | 24 | 0.0170 | 0.0000 | 0.0805 | 1.00 | 0.00 | 0.46 | 0.108 |
| `alg-dsl-af7e12e16c` | 24 | 0.0170 | 0.0000 | 0.0805 | 1.00 | 0.00 | 0.46 | 0.108 |
| `cmaes` | 24 | 0.1034 | 0.0000 | 0.1565 | 1.00 | 0.00 | 0.58 | 0.066 |
| `de` | 24 | 0.1366 | 0.0039 | 0.2421 | 1.00 | 0.00 | 0.00 | 0.101 |
| `nelder_mead` | 24 | 0.1936 | 0.1235 | 0.2061 | 1.00 | 0.00 | 0.38 | 0.078 |
| `random_search` | 24 | 1.0048 | 1.0000 | 0.2465 | 1.00 | 0.00 | 0.00 | 0.043 |

## Held-out test split (evaluated once, after selection)

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `ope-pyt-8a2643a783` | 120 | 0.0028 | 0.0001 | 0.0072 | 1.00 | 0.00 | 0.25 | 0.281 |
| `alg-dsl-ae8b1f11f2` | 60 | 0.0063 | 0.0000 | 0.0408 | 1.00 | 0.00 | 0.48 | 0.112 |
| `cmaes` | 60 | 0.1060 | 0.0000 | 0.1736 | 1.00 | 0.00 | 0.60 | 0.075 |
| `de` | 60 | 0.1196 | 0.0038 | 0.2137 | 1.00 | 0.00 | 0.00 | 0.103 |
| `nelder_mead` | 60 | 0.2232 | 0.2030 | 0.2156 | 1.00 | 0.00 | 0.33 | 0.080 |
| `random_search` | 60 | 1.0071 | 1.0000 | 0.2959 | 1.00 | 0.00 | 0.00 | 0.034 |

Best baseline on test: `cmaes`.

- `ope-pyt-8a2643a783` vs `cmaes`: mean score difference -0.1032 over 60 paired runs (wins 22, losses 38, ties 0; one-sided Wilcoxon p=0.1)
- `alg-dsl-ae8b1f11f2` vs `cmaes`: mean score difference -0.0997 over 60 paired runs (wins 28, losses 29, ties 3; one-sided Wilcoxon p=0.0425)

Demonstration candidates: {'operator': 'ope-pyt-8a2643a783', 'algorithm': 'alg-dsl-ae8b1f11f2', 'code_operator': 'ope-pyt-8a2643a783'}
