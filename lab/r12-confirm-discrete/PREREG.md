# Round 12 pre-registration: confirming round-10 candidates on fresh discrete instances

Written and committed before the run.

## Question

Round 10 produced two frozen candidates. On the (now historical) `discrete_general` v2 test split, the best
generated algorithm `alg-dsl-f18700ed7f` beat `genetic` (mean -0.048, CI [-0.080, -0.016]) although it was not the
selected champion; the selected champion `ope-dsl-d23ef2a66f` did not (CI [-0.050, +0.004]). Do these hold on
instances no decision has used?

## Frozen design

* Candidates: registry records `alg-dsl-f18700ed7f` (primary) and `ope-dsl-d23ef2a66f` (secondary), unchanged
  since round 10. Comparator: `genetic` (the best baseline in rounds 8 and 10).
* Data: suite `discrete_general_v3`, test split only: instance seeds 3030-3034 for the five v2 families plus
  `set_cover` (30 instances). These instances were generated for this round and have not been run.
* Solver seeds 700-704; 5000 evaluations per run; baselines random_search, simulated_annealing, genetic, tabu
  (now 1.1.0), greedy_ls (only for the score normalisation and context).
* Script: `lab/r12-confirm-discrete/confirm.py` (no options).
* Decision rule per candidate: Holm-adjusted (over the two candidates) one-sided Wilcoxon p < 0.05, mean score
  difference <= -0.01, bootstrap 95% CI of the paired difference below zero. Otherwise "not confirmed".
* Reported regardless of outcome: per-family means, practical wins/losses, rank-biserial effect, raw and adjusted
  p-values.

## Caveats stated before the run

* Same families and sizes as the data on which the primary candidate was noticed, so this confirms transfer to new
  instances, not to new problem classes (set_cover is not new to the candidates' evaluation history either).
* The normalised score is sensitive on correlated knapsacks; gaps are reported too.
* The reference values for scores come from these baselines on the same runs, as in all earlier rounds.
