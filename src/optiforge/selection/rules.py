"""Transparent rule-based solver selection.

Rules are evaluated in order; each decision records the rule that fired and
why, so a plan can be audited.  Learned (evidence-based) selection is layered
on top in :mod:`optiforge.selection.evidence` and only overrides these rules
when enough validated benchmark evidence exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.budget import Budget
from ..core.problem import Problem


@dataclass
class SolverPlan:
    solver: str
    params: dict = field(default_factory=dict)
    strategy: str = "global"  # exact | local | global | hybrid
    rationale: list[str] = field(default_factory=list)
    source: str = "rules"
    evidence: dict | None = None

    def to_dict(self) -> dict:
        return {"solver": self.solver, "params": self.params, "strategy": self.strategy,
                "rationale": self.rationale, "source": self.source, "evidence": self.evidence}


def characteristics(problem: Problem, budget: Budget, eval_cost_s: float | None = None) -> dict:
    d = problem.describe()
    d["budget_evaluations"] = budget.max_evaluations
    d["eval_cost_s"] = eval_cost_s
    d["has_permutation"] = bool(problem.space.perm_blocks)
    d["has_categorical"] = bool(problem.space.cat_mask.any())
    d["evals_per_dim"] = (budget.max_evaluations or 0) / max(problem.space.dim, 1)
    return d


def select_by_rules(problem: Problem, budget: Budget, eval_cost_s: float | None = None,
                    mode: str = "auto", exhaustive_limit: int = 200_000, ga: str = "host_ga") -> SolverPlan:
    """``mode``: ``auto`` | ``local`` | ``global`` | ``hybrid`` (restricts the strategy).

    ``ga`` is the evolutionary solver used by the discrete-constrained and mixed rules (R10, R12, R13):
    ``host_ga`` ((mu+lambda), the default since round 16, lab/r16-host-confirm) or ``genetic`` (generational, the
    0.1.0 choice; still better on mixed_constrained in round 16).
    """
    if ga not in ("genetic", "host_ga"):
        raise ValueError(f"ga must be 'genetic' or 'host_ga', not {ga!r}")
    c = characteristics(problem, budget, eval_cost_s)
    why: list[str] = [f"characteristics: kind={c['space_kind']} dim={c['dim']} constraints={c['n_constraints']} "
                      f"(eq={c['n_equality']}) gradient={c['gradient']} noisy={c['noisy']} budget={c['budget_evaluations']}"]
    card = c["cardinality"]
    expensive = eval_cost_s is not None and eval_cost_s > 0.05
    if expensive:
        why.append(f"expensive evaluations (~{eval_cost_s:.3g}s each): prefer sample-efficient methods")

    if mode == "auto" and card is not None and not c["noisy"] and card <= min(exhaustive_limit, budget.max_evaluations or 0):
        why.append(f"R1: finite domain with {card} points fits the budget -> complete enumeration (certifiable)")
        return SolverPlan("exhaustive", {}, "exact", why)
    if mode == "auto" and c["linear_declared"] and not c["noisy"] and not c["has_permutation"] and not c["has_categorical"]:
        why.append("R2: linear structure declared -> MILP/LP adapter (HiGHS), certifiable after spot checks")
        return SolverPlan("milp", {}, "exact", why)

    kind = c["space_kind"]
    if kind == "continuous":
        if mode == "local":
            if c["n_constraints"] and not c["noisy"]:
                why.append("R3a: local mode with constraints -> SLSQP")
                return SolverPlan("slsqp", {}, "local", why)
            if c["gradient"] and not c["noisy"]:
                why.append("R3b: local mode with gradient -> L-BFGS-B")
                return SolverPlan("lbfgsb", {}, "local", why)
            why.append("R3c: local mode, derivative-free -> Nelder-Mead")
            return SolverPlan("nelder_mead", {}, "local", why)
        if c["noisy"]:
            why.append("R4: noisy objective -> CMA-ES (population ranking is robust to noise; no finite differences)")
            return SolverPlan("cmaes", {}, "global", why)
        if c["n_constraints"]:
            if mode == "global":
                why.append("R5g: constrained, global mode -> DE with feasibility rules")
                return SolverPlan("de", {"strategy": "current_to_pbest"}, "global", why)
            n_starts = 4 if expensive else 8
            why.append(f"R5: smooth constrained continuous problem -> multi-start SLSQP ({n_starts} starts); "
                       "SLSQP handles equalities that population methods struggle to satisfy")
            return SolverPlan("multistart", {"local": "slsqp", "n_starts": n_starts}, "hybrid", why)
        if c["convex_declared"] and c["gradient"] and mode == "auto":
            why.append("R6: declared convex with gradient -> L-BFGS-B (stationary point is global under the declaration)")
            return SolverPlan("lbfgsb", {}, "local", why)
        if expensive or c["evals_per_dim"] < 60:
            why.append("R7: small budget per dimension -> multi-start local search")
            local = "lbfgsb" if c["gradient"] else "nelder_mead"
            return SolverPlan("multistart", {"local": local, "n_starts": 4}, "hybrid", why)
        if mode == "global":
            why.append("R8g: global mode -> CMA-ES with IPOP restarts")
            return SolverPlan("cmaes", {}, "global", why)
        local = "lbfgsb" if c["gradient"] else "nelder_mead"
        why.append(f"R8: unconstrained continuous, unknown modality -> hybrid CMA-ES exploration then {local} refinement")
        return SolverPlan("hybrid", {"global_solver": "cmaes", "local": local, "global_fraction": 0.8}, "hybrid", why)

    if kind == "discrete":
        if mode == "local":
            why.append("R9a: local mode -> greedy construction + neighbourhood improvement")
            return SolverPlan("greedy_ls", {}, "local", why)
        if c["has_permutation"]:
            why.append("R9: permutation variables -> simulated annealing with swap/insert/reversal moves")
            return SolverPlan("simulated_annealing", {}, "global", why)
        if c["n_constraints"]:
            why.append(f"R10: constrained discrete -> evolutionary algorithm ({ga}) with feasibility-first selection")
            return SolverPlan(ga, {}, "global", why)
        why.append("R11: unconstrained discrete -> tabu search")
        return SolverPlan("tabu", {}, "global", why)

    # mixed
    if c["has_permutation"]:
        why.append(f"R12: mixed with permutations -> type-aware evolutionary algorithm ({ga})")
        return SolverPlan(ga, {}, "global", why)
    if mode == "local":
        why.append("R13a: local mode -> compass search on real/integer coordinates")
        return SolverPlan("pattern_search", {}, "local", why)
    why.append(f"R13: mixed variables -> hybrid: type-aware evolutionary exploration ({ga}), then compass-search "
               "refinement")
    return SolverPlan("hybrid", {"global_solver": ga, "local": "pattern_search", "global_fraction": 0.8},
                      "hybrid", why)
