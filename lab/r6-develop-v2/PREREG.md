# Round 6 pre-registration: development loop on suite `continuous_general` (v2)

Written before the run. Nothing below may be changed after the run starts.

## Question

After adding a rotation-invariant mutation (`correlated_gaussian`) to the operator language, can the development
loop produce a candidate that beats the strongest baseline on rotated / non-separable / ill-conditioned continuous
problems, including two problem families it never sees during generation or selection?

## Frozen design

* Config: `develop_continuous_general.yaml` in this directory (budget 5000 evaluations per run for every solver).
* Code: the commit recorded in the run's `environment.json` / `development_report.json`.
* Generation and retention use only the train split (instance seeds 1010, 1011); selection uses only the validation
  split (1020, 1021). Baselines: random_search, cmaes, de, nelder_mead, hybrid.
* Selection rule (fixed in code and config): Holm-adjusted one-sided Wilcoxon p < 0.05 against the best validation
  baseline, mean score improvement >= 0.01, bootstrap 95% CI of the paired mean difference below zero, feasibility
  rate not lower.
* Test: instance seeds 1030-1032 for the eight training families plus the test-only families `bent_cigar_rotated`
  (d=10) and `griewank_rotated` (d=8), which no suite has used before; solver seeds 500-504; run once after
  selection is frozen. This split has never been run.
* Reported on test, whatever the outcome: the champion (if any), the best generated operator, the best generated
  algorithm and the best code operator by training score, against the best test baseline, with mean/median scores,
  feasibility and failure rates, evaluations, runtime, CI of the paired difference, rank-biserial effect, practical
  wins/losses/ties and the raw p-value. Per-family results are reported separately for the held-out families.

## Known caveats, stated before the run

* The validation split is not pristine: round 3 ran baselines and the 0.1.0 candidate on train + validation
  (with the earlier instance definitions of the new families), and that diagnosis motivated `correlated_gaussian`.
  Only the test split is untouched.
* One run of the development loop (one generator seed). It says whether this configuration found something, not how
  often the loop succeeds.

## What would count as a positive result

A champion selected on validation whose test comparison against the best test baseline has a CI entirely below zero
and a mean improvement of at least 0.01. Anything else is reported as "no improvement shown". A win only on the
training families and not on the held-out families is reported as such.
