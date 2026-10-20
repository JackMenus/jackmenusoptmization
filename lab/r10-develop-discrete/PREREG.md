# Round 10 pre-registration: development loop on suite `discrete_general` (v2)

Written and committed before the run. Nothing below may be changed after the run starts.

## Questions

1. (Primary) With behavioural de-duplication, can the development loop produce a candidate that beats the
   strongest baseline on hard binary/integer problems with exact optima, including a held-out family (`set_cover`)?
2. (Secondary) Does the opt-in binary exchange neighbourhood for tabu (round 9, validation evidence only) hold up on
   the untouched test split?

## Frozen design

* Config: `develop_discrete_general.yaml` (5000 evaluations per run for every solver; fingerprint budget 200).
* Code: the commit that adds this file (recorded in the run's `environment.json`).
* Generation/retention: train split (instance seeds 2010, 2011) only. Selection: validation split (2020, 2021) only.
  Baselines: random_search, simulated_annealing, genetic, tabu, greedy_ls.
* Selection rule: Holm-adjusted one-sided Wilcoxon p < 0.05 against the best validation baseline, mean score
  improvement >= 0.01, bootstrap 95% CI of the paired mean difference below zero, feasibility not lower.
* Test: instance seeds 2030-2032 for the five training families plus test-only `set_cover` (n_sets 40,
  n_elements 30); solver seeds 500-504; run once, after selection is frozen. This split has never been run (its
  instances were only constructed to check that exact optima exist).
* Primary report on test, whatever the outcome: champion (if any), best generated operator / algorithm / code
  operator by training score, against the best test baseline: mean/median scores, feasibility and failure rates,
  evaluations, runtime, CI of the paired difference, rank-biserial effect, practical wins/losses/ties, raw p-value;
  `set_cover` reported separately.
* Secondary analysis, run after the development loop on the same test instances with solver seeds 500-504:
  `python lab/r9-exchange/compare.py <out> --split test --budget 5000 --seeds 500 501 502 503 504`. Hypothesis:
  `tabu+exchange` better than `tabu`; family of two comparisons (also `sa+exchange` vs `simulated_annealing`),
  Holm-adjusted. Confirmed only if Holm p < 0.05, mean difference <= -0.01 and CI below zero.

## Known caveats, stated before the run

* Validation is not pristine: round 9 used it to compare the exchange variants, and round 8 characterised the
  baselines and the 0.1.0 mixed champion on train + validation.
* One run of the loop (one generator seed).
* The normalised score divides by (random-search median - optimum); on correlated knapsacks that span is about 2% of
  the optimum, so small absolute gaps give large score differences. Gaps are reported alongside scores.

## What counts as positive

Primary: a champion selected on validation whose test comparison against the best test baseline has a CI entirely
below zero and mean improvement >= 0.01; otherwise "no improvement shown". Secondary: as stated above; otherwise
"not confirmed".
