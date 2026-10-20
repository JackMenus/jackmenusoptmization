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
| `global_optimum_certified` | `exhaustive` evaluated every point of a finite deterministic domain without any failed evaluation; or `milp` proved optimality of a linear model **that is the problem**, because the problem was built with `Problem.from_linear` (objective and constraints generated from the model), and the returned point re-verifies | the reported point is globally optimal within the stated tolerances |
| `infeasibility_certified` | `exhaustive` evaluated every point and none was feasible; or `milp` found the tolerance-relaxed model of a `Problem.from_linear` problem infeasible | the problem as defined has no feasible point within tolerances |
| `global_optimum_conditional_on_assumptions` | `lbfgsb` reached a projected-gradient stationary point on a minimisation problem that the *user declared* convex and that has an analytic gradient; or `milp` proved optimality of a `linear=` declaration attached to user-written functions, which agreed with the declaration at the box corners, at random spot checks and at the solution | global optimality holds only if the declaration is true; it is not verified |
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
* Every certificate (global or infeasibility) is dropped when the reported point's objective or feasibility changes
  between search and verification, because that shows the problem is not deterministic.
* A global certificate is dropped when the problem's independent verifier rejects the point or raises.
* A `linear=` declaration on user-written functions is never trusted for a certificate, however many spot checks
  pass: a declaration that is wrong only near the optimum passes random checks. Only `Problem.from_linear`, where
  the functions are generated from the model, gives the MILP adapter a certificate. For such a problem the adapter
  solves the model with every constraint relaxed by its tolerance (the problem as defined), which gives a lower
  bound; it then needs a point that verifies as feasible on re-evaluation (the relaxed optimum, or the strict model's
  optimum, tightened inward if HiGHS's own 1e-7 tolerance leaves it just outside). The certificate states an absolute
  objective tolerance (`certificate_tol` x (1 + |f|), default 1e-6) and is withheld when the verified point is further
  than that from the bound. Spot checks are only run for `linear=` declarations on user-written functions (at least
  20). `Budget.max_time` is honoured; a HiGHS time-out ends with termination `budget_time` and no certificate.
* Statuses are `optimal_certified`, `feasible`, `feasible_discretized_only`, `infeasible_certified`,
  `no_feasible_found` and `error`. Only the first and fourth are certificates.

## Status meanings

| status | meaning |
|---|---|
| `optimal_certified` | proven global optimum (see the certificate's method and assumptions) |
| `feasible` | a feasible point was found and verified; its guarantees say whether it is merely heuristic, locally stationary or conditionally optimal |
| `feasible_discretized_only` | feasible on the simulation grid but rejected by the denser independent check |
| `infeasible_certified` | proven that no feasible point exists |
| `no_feasible_found` | the search ran but found no feasible point; **not** evidence of infeasibility |
| `error` | an execution failure: every evaluation failed, nothing was evaluated, the solver raised before any valid evaluation, or the verification evaluation of the reported point failed |

The termination reason is reported separately (`budget_evaluations`, `budget_time`, `cancelled`, `converged`,
`error`, ...). A time-out with a verified feasible incumbent has status `feasible` and termination `budget_time`.
