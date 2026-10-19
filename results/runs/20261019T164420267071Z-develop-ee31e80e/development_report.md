# Development run report: suite `mixed`

Wall time 222.5 s. Stop reason: generations_completed. Ledger: 2931559 of 12000000 units used ({'analysis': 320, 'train_baselines': 140778, 'validation': 155615, 'train': 1728000, 'selection': 284949, 'test': 621897}).

Isolation available: True (linux namespaces (user, mount, pid, net, ipc, uts) + pivot_root + capability drop + rlimits).
Candidates by final status: {'baseline': 8, 'retained': 11, 'discarded': 144, 'rejected': 24, 'champion': 1}

## Generations

| gen | proposed | duplicates | rejected | validated | benchmarked | best candidate train score | best baseline train score |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 0 | 2 | 8 | 8 | 0.1172 | 0.1867 |
| 1 | 10 | 0 | 1 | 9 | 9 | 0.0880 | 0.1867 |
| 2 | 10 | 0 | 2 | 8 | 8 | 0.0821 | 0.1867 |
| 3 | 10 | 0 | 1 | 9 | 9 | 0.0654 | 0.1867 |
| 4 | 10 | 0 | 3 | 7 | 7 | 0.0654 | 0.1867 |
| 5 | 10 | 0 | 3 | 7 | 7 | 0.0654 | 0.1867 |

Scores: 0 = known optimum (or best baseline value), 1 = random-search median at the same budget, 2 = infeasible. Lower is better.

## Selection (validation split)

ope-pyt-c4d678ee14 beat the best baseline 'genetic' on the validation split (mean score 0.0292 vs 0.1444, one-sided Wilcoxon p=0.00122 < 0.05)

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `ope-pyt-c4d678ee14` | 12 | 0.0292 | 0.0166 | 0.0317 | 1.00 | 0.00 | 0.00 | 0.355 |
| `ope-dsl-9186b83ad3` | 12 | 0.1173 | 0.0501 | 0.2577 | 1.00 | 0.00 | 0.00 | 0.203 |
| `ope-dsl-0a55ccc850` | 12 | 0.1328 | 0.0545 | 0.2576 | 1.00 | 0.00 | 0.00 | 0.224 |
| `genetic` | 12 | 0.1444 | 0.0891 | 0.1471 | 1.00 | 0.00 | 0.00 | 0.182 |
| `ope-dsl-39e44e4cc2` | 12 | 0.1609 | 0.0931 | 0.2571 | 1.00 | 0.00 | 0.00 | 0.232 |
| `hybrid` | 12 | 0.3330 | 0.3072 | 0.3129 | 1.00 | 0.00 | 0.00 | 0.106 |
| `de` | 12 | 0.4412 | 0.4249 | 0.1932 | 1.00 | 0.00 | 0.00 | 0.112 |
| `random_search` | 12 | 1.0576 | 1.0000 | 0.1896 | 1.00 | 0.00 | 0.00 | 0.062 |

## Held-out test split (evaluated once, after selection)

| solver | runs | mean score | median | std | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `ope-pyt-c4d678ee14` | 30 | 0.0505 | 0.0261 | 0.0738 | 1.00 | 0.00 | 0.00 | 0.303 |
| `ope-dsl-0a55ccc850` | 30 | 0.1061 | 0.0835 | 0.1139 | 1.00 | 0.00 | 0.00 | 0.234 |
| `alg-dsl-26bfd13429` | 30 | 0.2055 | 0.1443 | 0.2489 | 1.00 | 0.00 | 0.00 | 0.217 |
| `genetic` | 30 | 0.2363 | 0.2733 | 0.1807 | 1.00 | 0.00 | 0.00 | 0.188 |
| `hybrid` | 30 | 0.2947 | 0.2373 | 0.2707 | 1.00 | 0.00 | 0.07 | 0.106 |
| `de` | 30 | 0.4737 | 0.4615 | 0.1261 | 1.00 | 0.00 | 0.00 | 0.131 |
| `random_search` | 30 | 1.0542 | 1.0000 | 0.2887 | 1.00 | 0.00 | 0.00 | 0.055 |

Best baseline on test: `genetic`.

- `ope-pyt-c4d678ee14` vs `genetic`: mean score difference -0.1857 over 30 paired runs (wins 30, losses 0, ties 0; one-sided Wilcoxon p=9.31e-10)
- `ope-dsl-0a55ccc850` vs `genetic`: mean score difference -0.1302 over 30 paired runs (wins 27, losses 3, ties 0; one-sided Wilcoxon p=2.21e-06)
- `alg-dsl-26bfd13429` vs `genetic`: mean score difference -0.0307 over 30 paired runs (wins 18, losses 12, ties 0; one-sided Wilcoxon p=0.0621)

Demonstration candidates: {'operator': 'ope-dsl-0a55ccc850', 'algorithm': 'alg-dsl-26bfd13429', 'code_operator': 'ope-pyt-c4d678ee14'}
