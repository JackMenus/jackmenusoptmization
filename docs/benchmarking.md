# Benchmarking methodology

## Problems

The families live in `benchmarks/problems.py`. Each is a deterministic function of (family, seed), with seeded
rotations and shifts.

| class | families | reference value |
|---|---|---|
| convex | `sphere`, `ellipsoid` | analytic optimum 0 |
| nonconvex or multimodal | `rosenbrock`, `rastrigin`, `ackley`, `styblinski_tang` | analytic |
| constrained continuous | `halfspace_sphere`, `hyperplane_sphere` (equality), `disk_rosenbrock`, `g06` | analytic or literature (`g06`: -6961.81388, marked as literature) |
| discrete combinatorial | `knapsack` (dynamic programming), `maxcut` (enumeration), `integer_quadratic` (enumeration), `tsp` (Held-Karp) | exact, computed per instance |
| mixed | `mixed_rastrigin` (real, integer, categorical), `mixed_constrained` | analytic or enumerated |
| simulation | `pd_control` (continuous-time ODE, sampled path constraint), `inventory` (discrete-time lost-sales (s,S) policy) | none known; the best baseline value is used and marked unverified |

## Suites and splits

A suite lists families and instance seeds per split: train [0, 1], validation [100, 101] and test [200, 201, 202].
`disk_rosenbrock` and `g06` appear only in the test split of `constrained_continuous`, so selection is also tested
on unseen families there. `check_disjoint()` asserts that no instance appears in two splits.

The development loop uses train for generation and retention and validation for selection. It runs test exactly
once, after the selection is frozen. Test results are written to the registry for the record, but nothing reads
them back to make a decision.

## Scoring

Each instance has a reference value `f_ref`: the known optimum when there is one (analytic, exact enumeration or dynamic
programming, or for `g06` a published reference value; the report names the method), otherwise the best value any
baseline reached, with that fact recorded. `f_rand` is the median that random search reaches at the same budget.
A run that ends at internal (minimisation) value f scores

    s = clip((f - f_ref) / (f_rand - f_ref), -1, 2)

so 0 means the reference was reached and 1 means random-search level. A run with no feasible result scores 2.
Lower is better. Every solver receives the same evaluation budget and the same seeds. When a solver is not
applicable (for example `lbfgsb` on a permutation problem), the run is counted as "n/a" rather than as a failure.

Reports include, per solver:

* mean, median, standard deviation and IQR of the score;
* feasibility rate and failure rate;
* the solved rate (gap at most 1e-6 relative to a verified optimum);
* median gap;
* mean runtime and runtime per evaluation, which shows overhead;
* per-family means.

Comparisons are paired by (instance, seed) and use a one-sided Wilcoxon signed-rank test with
`zero_method="zsplit"`, reporting wins, losses and ties.

## Caveats

* A score that uses an unverified reference can be negative if a later solver beats every baseline.
* The scale differs between instances. A score of 0.1 on Rastrigin and 0.1 on knapsack mean different absolute gaps.
* With 2 to 3 instances per family, 3 to 5 seeds and Wilcoxon tests, the statistical power is limited. A p-value
  above alpha is "no evidence of a difference", not evidence of equality.
