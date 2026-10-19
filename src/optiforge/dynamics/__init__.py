"""Discrete-time and continuous-time system adapters for simulation-based optimisation."""
from .systems import ContinuousTimeSystem, DiscreteTimeSystem, SimulationError, simulation_problem

__all__ = ["ContinuousTimeSystem", "DiscreteTimeSystem", "SimulationError", "simulation_problem"]
