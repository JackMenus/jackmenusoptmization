"""Baseline solver portfolio. Importing this package registers every solver."""
from .base import SOLVERS, IterativeSolver, Solver, get_solver, register_solver
from . import continuous, discrete, exact, composite  # noqa: F401  (registration side effects)
from . import host  # noqa: F401,E402  (after composite: the operator-language interpreter imports it)

__all__ = ["SOLVERS", "Solver", "IterativeSolver", "get_solver", "register_solver"]
