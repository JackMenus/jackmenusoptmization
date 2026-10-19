"""Adapters for simulation-based system optimisation.

Two kinds of dynamical systems are kept distinct:

* :class:`DiscreteTimeSystem` - ``x[k+1] = f(x[k], k, theta)``.  The model *is*
  discrete, so a path constraint checked at every step is checked exactly for
  that model (feasibility scope ``"exact"``).
* :class:`ContinuousTimeSystem` - ``dx/dt = f(t, x, theta)`` integrated with
  SciPy's ``solve_ivp``.  Path constraints can only be checked at sample
  times of a numerical solution, so feasibility is reported with scope
  ``"sampled"``; the attached verifier re-integrates with tighter tolerances on
  a denser grid, but satisfaction between samples is still not proven.

:func:`simulation_problem` turns either system into an ordinary
:class:`~optiforge.core.problem.Problem` whose decision variables are the
system parameters ``theta`` (controller gains, policy levels, schedules...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..core.constraints import ineq
from ..core.problem import Problem
from ..core.space import Space


class SimulationError(RuntimeError):
    pass


@dataclass
class DiscreteTimeSystem:
    step: Callable[[np.ndarray, int, np.ndarray], np.ndarray]  # (x_k, k, theta) -> x_{k+1}
    x0: np.ndarray
    horizon: int
    kind: str = field(default="discrete_time", init=False)

    def simulate(self, theta: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(self.x0, dtype=float)
        X = [x]
        for k in range(self.horizon):
            x = np.asarray(self.step(x, k, theta), dtype=float)
            if not np.all(np.isfinite(x)):
                raise SimulationError(f"state became non-finite at step {k}")
            X.append(x)
        return np.arange(self.horizon + 1, dtype=float), np.array(X)


@dataclass
class ContinuousTimeSystem:
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray]  # (t, x, theta) -> dx/dt
    x0: np.ndarray
    t_final: float
    method: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-9
    kind: str = field(default="continuous_time", init=False)

    def simulate(self, theta: np.ndarray, n_samples: int = 200, rtol: float | None = None,
                 atol: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        t_eval = np.linspace(0.0, self.t_final, n_samples)
        sol = solve_ivp(lambda t, x: self.rhs(t, x, theta), (0.0, self.t_final), np.asarray(self.x0, dtype=float),
                        method=self.method, t_eval=t_eval, rtol=rtol or self.rtol, atol=atol or self.atol)
        if not sol.success:
            raise SimulationError(f"integration failed: {sol.message}")
        return sol.t, sol.y.T


PathConstraint = tuple  # (name, fn(t_array, X_array, theta) -> array of g values, g <= 0 required)


def simulation_problem(
    system,
    space: Space,
    cost: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    path_constraints: Sequence[PathConstraint] = (),
    name: str = "simulation",
    n_samples: int = 200,
    verify_factor: int = 10,
    tol: float = 1e-6,
) -> Problem:
    """Build a Problem from a simulated system.

    ``cost(t, X, theta)`` returns the scalar objective; each path constraint
    returns ``g`` values over the trajectory and is enforced as ``max g <= 0``.
    A one-entry cache avoids simulating twice when objective and constraints
    are evaluated at the same point.
    """
    cache: dict = {}

    def run(theta):
        key = np.asarray(theta, dtype=float).tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = system.simulate(np.asarray(theta, dtype=float), n_samples=n_samples)
        return cache[key]

    def objective(theta):
        t, X = run(theta)
        return float(cost(t, X, np.asarray(theta)))

    cons = []
    for cname, g in path_constraints:
        cons.append(ineq((lambda theta, g=g: float(np.max(g(*run(theta), np.asarray(theta))))), name=cname, tol=tol))

    continuous = getattr(system, "kind", "") == "continuous_time"

    def verifier(theta):
        theta = np.asarray(theta, dtype=float)
        if continuous:
            t, X = system.simulate(theta, n_samples=n_samples * verify_factor,
                                   rtol=system.rtol / 100, atol=system.atol / 100)
        else:
            t, X = system.simulate(theta)
        worst = {cname: float(np.max(g(t, X, theta))) for cname, g in path_constraints}
        ok = all(v <= tol for v in worst.values())
        return {
            "feasible": ok,
            "scope": (f"sampled: {len(t)} points of a re-integration with rtol={system.rtol / 100:g}"
                      if continuous else "exact for the discrete-time model"),
            "max_constraint_values": worst,
            "objective_dense": float(cost(t, X, theta)),
        }

    meta = {"system_kind": system.kind, "n_samples": n_samples,
            "feasibility_scope": "sampled" if continuous else "exact"}
    return Problem(space, objective, constraints=cons, name=name, evaluation="simulation", verifier=verifier,
                   metadata=meta)
