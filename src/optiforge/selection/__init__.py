"""Solver selection: transparent rules plus evidence-based refinement."""
from .rules import SolverPlan, characteristics, select_by_rules

__all__ = ["SolverPlan", "characteristics", "select_by_rules"]
