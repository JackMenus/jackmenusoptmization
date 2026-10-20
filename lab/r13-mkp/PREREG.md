# Round 13 pre-registration: do the round-10 GA variants beat genetic on multidimensional knapsacks?

Written and committed before the run.

## Origin of the hypothesis (post hoc)

In rounds 10 (v2 test split) and 12 (v3 test split) the two frozen round-10 candidates `alg-dsl-f18700ed7f` and
`ope-dsl-d23ef2a66f` were better than `genetic` on `multi_knapsack` (n=40, m=5) and not on the other families. That
was noticed after looking at per-family results, so it is a hypothesis, not a finding.

## Frozen design

* Instances: `multi_knapsack` with (n, m) in {(40, 5), (40, 10), (60, 5), (60, 10)}, instance seeds 4000-4004 (20
  instances, none run before; seed 4100 was used only to time the exact MILP solve).
* Solvers: the two candidates (registry records unchanged since round 10), `genetic`, plus random_search, tabu and
  simulated_annealing for score normalisation. Solver seeds 800-804; 5000 evaluations per run.
* Script: `lab/r13-mkp/mkp.py` (no options).
* Decision rule per candidate vs genetic, Holm over the two candidates: adjusted one-sided Wilcoxon p < 0.05, mean
  score difference <= -0.01 and bootstrap 95% CI below zero -> "supported"; otherwise "not supported". Per-size
  results are reported but not tested separately.
