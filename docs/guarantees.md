# What a result does and does not claim

Every `OptimizationResult` carries:

* the best point (both encoded and decoded);
* the objective in the original sense, taken from a *verification* evaluation rather than from search bookkeeping;
* feasibility, per-constraint violations and the maximum violation;
* evaluation, gradient and verification counts;
* runtime and seed;
* solver and operator versions;
* the incumbent history;
* the termination reason;
* the problem assumptions;
* a list of `Guarantee` objects, each with a kind, a plain-language statement, a method, a tolerance and its
  assumptions.

| kind | issued when | meaning |
|---|---|---|
| `global_optimum_certified` | `exhaustive` evaluated every point of a finite deterministic domain; `milp` solved a declared linear model to optimality *and* the declaration matched the real functions on spot checks *and* the returned point re-verifies | the reported point is globally optimal within the stated tolerances |
| `infeasibility_certified` | `exhaustive` evaluated every point and none was feasible | the finite problem has no feasible point within tolerances |
| `global_optimum_conditional_on_assumptions` | `lbfgsb` reached a projected-gradient stationary point on a problem that the *user declared* convex and that has an analytic gradient | global optimality holds only if the declaration is true; it is not verified |
| `first_order_stationary_point` | projected-gradient norm below tolerance at the reported point | local stationarity only; not a proof of a local minimum |
| `local_method_converged` | a local method's own stopping test fired (simplex size, compass mesh, neighbourhood exhausted) | no proof of stationarity or local optimality |
| `feasibility_verified` | the original constraints hold at the reported point on re-evaluation | exact feasibility for the problem as defined |
| `feasibility_verified_at_sample_points_only` | constraints come from a sampled continuous-time simulation | feasibility between samples, or of the continuous system, is not proven |
| `heuristic_best_found` | every feasible result without a certificate | best found within the budget; nothing more |
| `no_feasible_solution_found` | no feasible evaluation | explicitly **not** evidence that the problem is infeasible |

The following rules are enforced in `core/finalize.py` and are tested:

* Penalties, repairs and surrogates never redefine the problem. The reported objective and violations are
  recomputed from the original functions, and a penalty method's internal value is never reported.
* A hint is dropped, with a message, if it refers to a point other than the reported one, if the reported point
  does not verify as feasible (for optimality claims), or if the objective is noisy (for optimality claims).
* A noisy objective is re-evaluated five times. The mean, standard deviation and number of repeats are reported,
  and no optimality claim is made.
* When the denser re-simulation of a continuous-time problem rejects a point that was feasible on the grid, the
  status becomes `feasible_discretized_only` and the message says so.
* Statuses are `optimal_certified`, `feasible`, `feasible_discretized_only`, `infeasible_certified`,
  `no_feasible_found` and `error`. Only the first and fourth are certificates.
