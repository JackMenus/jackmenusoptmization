# Round 16 pre-registration: host_ga vs genetic on fresh discrete and mixed instances

Written and committed before the run.

## Origin

Round 14 (ablation) and round 15 (development data: train/validation of discrete_general_v3, discrete_combinatorial,
mixed, permutation) suggest the (mu+lambda) host `host_ga` is better than the generational `genetic` on most
discrete and mixed families (round 15: mean -0.055, CI [-0.077, -0.034]; no family clearly worse).

## Frozen design

* Instances: 11 families x instance seeds 5000-5004 (55 instances, never run): knapsack_correlated n50 and n100,
  multi_knapsack (40, 5), independent_set n40, set_cover (40, 30), integer_quadratic (6, 7), knapsack n25,
  maxcut n14, tsp n10, mixed_rastrigin, mixed_constrained n4. All families are included, also those where ties are
  expected, to avoid choosing families by their result.
* Solvers: random_search, tabu, simulated_annealing, genetic (these four define the score reference), host_ga,
  auto (rules with ga=genetic, the current default) and auto[ga=host_ga]. Solver seeds 900-902; 5000 evaluations.
* Script: `lab/r16-host-confirm/confirm.py` (no options).
* H1: host_ga better than genetic. H2: auto[ga=host_ga] better than auto. Holm over the two. Each is confirmed only if
  the adjusted one-sided Wilcoxon p < 0.05, the mean score difference <= -0.01 and the bootstrap 95% CI is below zero.
* Pre-committed action: if H2 is confirmed, the rules default becomes ga=host_ga (version bump of the rule set,
  documented); if not, the default stays genetic and host_ga remains available.
