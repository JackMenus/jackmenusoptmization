"""Baseline solver portfolio. Importing this package registers every solver."""
from .base import SOLVERS, IterativeSolver, Solver, get_solver, register_solver
from . import continuous, discrete, exact, composite  # noqa: F401  (registration side effects)

__all__ = ["SOLVERS", "Solver", "IterativeSolver", "get_solver", "register_solver"]
