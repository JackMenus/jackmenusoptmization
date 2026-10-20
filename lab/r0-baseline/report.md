# Benchmark: suite `baseline_portfolio`, split `test`

17 instances x 5 seeds, budget 3000 evaluations per run, wall time 291.78 s.

Score: 0 = verified optimum (or best baseline value when no optimum is known), 1 = random-search median at the same budget, 2 = no feasible result. Lower is better.

| solver | runs | n/a | mean score | median | std | IQR | feasible | failures | solved (gap<=1e-6) | ms/eval |
|---|---|---|---|---|---|---|---|---|---|---|
| `auto` | 85 | 0 | 0.0428 | 0.0000 | 0.2224 | 0.0000 | 1.00 | 0.00 | 0.79 | 0.391 |
| `lbfgsb` | 25 | 60 | 0.0520 | 0.0000 | 0.1060 | 0.0000 | 1.00 | 0.00 | 0.80 | 0.612 |
| `slsqp` | 50 | 35 | 0.0653 | 0.0000 | 0.1354 | 0.0000 | 1.00 | 0.00 | 0.78 | 1.674 |
| `hybrid` | 85 | 0 | 0.0750 | 0.0000 | 0.1861 | 0.0000 | 1.00 | 0.00 | 0.73 | 0.599 |
| `cmaes` | 50 | 35 | 0.0858 | 0.0000 | 0.2215 | 0.0000 | 1.00 | 0.00 | 0.78 | 0.774 |
| `de` | 80 | 5 | 0.1132 | 0.0002 | 0.1907 | 0.2261 | 1.00 | 0.00 | 0.27 | 0.599 |
| `nelder_mead` | 50 | 35 | 0.2378 | 0.0011 | 0.5992 | 0.0742 | 0.90 | 0.00 | 0.50 | 0.293 |
| `greedy_ls` | 85 | 0 | 0.2559 | 0.0463 | 0.4921 | 0.2629 | 0.94 | 0.00 | 0.27 | 0.623 |
| `genetic` | 85 | 0 | 0.3212 | 0.0394 | 0.6070 | 0.2538 | 0.91 | 0.00 | 0.25 | 0.619 |
| `tabu` | 25 | 60 | 0.3391 | 0.0000 | 0.6665 | 0.3182 | 1.00 | 0.00 | 0.75 | 0.257 |
| `multistart` | 85 | 0 | 0.4791 | 0.0000 | 0.7849 | 0.6308 | 1.00 | 0.00 | 0.53 | 1.614 |
| `simulated_annealing` | 85 | 0 | 0.4976 | 0.3071 | 0.5697 | 0.6538 | 0.93 | 0.00 | 0.22 | 0.489 |
| `pattern_search` | 85 | 0 | 0.7801 | 0.1612 | 0.9016 | 1.9963 | 0.94 | 0.00 | 0.14 | 0.117 |
| `random_search` | 85 | 0 | 1.0978 | 1.0000 | 0.5625 | 0.5592 | 0.88 | 0.00 | 0.02 | 0.537 |

## Mean score per family

| family | `auto` | `cmaes` | `de` | `genetic` | `greedy_ls` | `hybrid` | `lbfgsb` | `multistart` | `nelder_mead` | `pattern_search` | `random_search` | `simulated_annealing` | `slsqp` | `tabu` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ackley | 0.000 | 0.000 | 0.001 | 0.176 | 0.221 | 0.000 | 0.000 | 1.123 | 0.280 | 0.000 | 0.957 | 0.020 | 0.393 | — |
| disk_rosenbrock | 0.000 | 0.000 | 0.000 | 0.361 | 0.505 | 0.000 | — | 0.000 | 0.000 | 0.306 | 1.204 | 0.983 | 0.000 | — |
| ellipsoid | 0.000 | 0.000 | 0.000 | 0.077 | 0.110 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.850 | 0.427 | 0.000 | — |
| g06 | -0.000 | -0.000 | 0.297 | 1.346 | 0.550 | -0.000 | — | -0.000 | 2.000 | 2.000 | 2.000 | 1.011 | -0.000 | — |
| halfspace_sphere | -0.000 | -0.000 | 0.001 | 0.037 | 0.048 | -0.000 | — | -0.000 | -0.000 | 0.384 | 0.951 | 0.634 | -0.000 | — |
| hyperplane_sphere | -0.000 | 0.599 | 0.206 | 2.000 | 2.000 | -0.000 | — | -0.000 | 0.002 | 0.143 | 2.000 | 2.000 | -0.000 | — |
| integer_quadratic | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.000 | — | 0.108 | 0.883 | 0.000 | — | 0.000 |
| inventory | 0.400 | — | 0.000 | 0.400 | 0.000 | 0.000 | — | 0.000 | — | 1.700 | 1.188 | 0.100 | — | 1.200 |
| knapsack | 0.055 | — | 0.045 | 0.055 | 0.055 | 0.141 | — | 2.000 | — | 2.000 | 0.932 | 0.800 | — | 0.495 |
| maxcut | 0.000 | — | 0.000 | 0.196 | 0.000 | 0.196 | — | 2.000 | — | 2.000 | 1.039 | 0.000 | — | 0.000 |
| mixed_constrained | 0.047 | — | 0.589 | 0.023 | 0.248 | 0.591 | — | 0.316 | — | 2.000 | 0.936 | 0.449 | — | — |
| mixed_rastrigin | 0.040 | — | 0.316 | 0.273 | 0.172 | 0.034 | — | 0.031 | — | 0.047 | 0.927 | 0.363 | — | — |
| pd_control | 0.000 | 0.000 | 0.000 | 0.228 | 0.136 | 0.000 | — | 0.000 | 0.000 | 0.566 | 0.933 | 0.841 | 0.000 | — |
| rastrigin | 0.186 | 0.259 | 0.353 | 0.165 | 0.269 | 0.215 | 0.260 | 0.675 | 0.074 | 0.000 | 0.836 | 0.219 | 0.260 | — |
| rosenbrock | 0.000 | 0.000 | 0.004 | 0.021 | 0.026 | 0.000 | 0.000 | 0.000 | 0.022 | 0.001 | 0.859 | 0.457 | 0.000 | — |
| sphere | 0.000 | 0.000 | 0.000 | 0.006 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.973 | 0.153 | 0.000 | — |
| tsp | 0.000 | — | — | 0.097 | 0.000 | 0.097 | — | 2.000 | — | 2.000 | 1.194 | 0.000 | — | 0.000 |

## Reference values

| instance | reference | kind | random-search median |
|---|---|---|---|
| ackley[dim=5]#300 | 0 | known optimum (analytic) | 5.88916 |
| disk_rosenbrock#0 | 0 | known optimum (analytic (unconstrained minimiser is feasible)) | 0.00956475 |
| ellipsoid[dim=5]#300 | 0 | known optimum (analytic) | 295.199 |
| g06#0 | -6961.81 | known optimum (published reference value (Liang et al. 2006, CEC2006 suite)) | -2577.65 |
| halfspace_sphere[dim=5]#300 | 2.07773 | known optimum (analytic (Euclidean projection)) | 5.26225 |
| hyperplane_sphere[dim=5]#300 | 0.737655 | known optimum (analytic (projection onto hyperplane)) | 37.5021 |
| integer_quadratic[n=4]#300 | 1.12824 | known optimum (complete enumeration) | 3.79549 |
| inventory#300 | 27.3917 | best baseline value (not verified optimal) | 27.525 |
| knapsack[n=20]#300 | -443 | known optimum (dynamic programming) | -399 |
| maxcut[n=14]#300 | -11.8961 | known optimum (complete enumeration (2^(n-1) cuts)) | -11.6962 |
| mixed_constrained[n=4]#300 | -4.42989 | known optimum (enumeration of binaries + SLSQP on each convex QP subproblem (numerical, tol 1e-6)) | 0.690343 |
| mixed_rastrigin#300 | 0 | known optimum (analytic (separable: Rastrigin min 0 at the offset, integer part 0 at b)) | 47.132 |
| pd_control#300 | 0.295698 | best baseline value (not verified optimal) | 0.304341 |
| rastrigin[dim=5]#300 | 0 | known optimum (analytic) | 26.8123 |
| rosenbrock[dim=5]#300 | 0 | known optimum (analytic) | 176.477 |
| sphere[dim=5]#300 | 0 | known optimum (analytic) | 2.30931 |
| tsp[n=9]#300 | 2.30238 | known optimum (Held-Karp dynamic programming) | 2.33971 |
