"""``host_ga``: the fixed (mu+lambda) host algorithm of the operator language with its default operator.

Generated operators are benchmarked inside this host (``dsl.spec.host_algorithm``). Round 14 found that the host
itself, with the default operator, accounts for the multi-knapsack advantage of the round-10 candidates over the
generational ``genetic`` baseline. ``host_ga`` makes the host available as a baseline, so a generated operator has to
beat the default operator in the same host, not only a different GA.
"""
from __future__ import annotations

import copy

from ..dsl.interpreter import DSLAlgorithm
from ..dsl.spec import DEFAULT_OPERATOR, host_algorithm
from .base import register_solver


@register_solver
class HostGA(DSLAlgorithm):
    name = "host_ga"
    version = "1.0.0"
    family = "global"
    supports_noise = False
    description = ("(mu+lambda) evolutionary algorithm: tournament(2) selection, uniform crossover + type-aware "
                   "mutation, plus-survival, one-fifth step rule, stagnation restarts (the operator-language host).")

    def __init__(self, population: int = 20, sigma: float = 0.1) -> None:
        op = copy.deepcopy(DEFAULT_OPERATOR)
        op["children"][1]["sigma"] = float(sigma)  # type_aware mutation scale (0.1 = the default operator)
        super().__init__(host_algorithm(op, population=int(population), name="host_ga"), "host_ga")
        self.name, self.version = "host_ga", "1.0.0"
        self.params = {"population": int(population), "sigma": float(sigma)}

    @classmethod
    def default_params(cls) -> dict:
        return {"population": 20, "sigma": 0.1}
