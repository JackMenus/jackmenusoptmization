"""Evidence-based refinement of rule-based solver plans.

A generated candidate is recommended instead of the rule-based choice only if
the registry holds *validated* evidence for a matching problem profile:

* status ``champion`` (passed all validation gates and beat the best baseline
  on a validation split with a one-sided paired Wilcoxon test),
* the evidence profile matches the problem: same space kind, the problem's
  variable types are covered, dimension within the evidence range (x1.5),
  same constrained/unconstrained class,
* the evaluation budget is within a factor 2 of the benchmarked budget,
* at least ``min_runs`` paired runs.

Otherwise the rule-based plan stands and the reason is recorded.  This is a
nearest-profile lookup, not a trained model: there is not enough evidence in
a single development run to justify a learned selector.
"""
from __future__ import annotations

from ..core.budget import Budget
from ..core.problem import Problem
from ..registry.store import Registry
from .rules import SolverPlan


class EvidenceSelector:
    def __init__(self, registry: Registry, min_runs: int = 12, alpha: float = 0.05) -> None:
        self.registry = registry
        self.min_runs = min_runs
        self.alpha = alpha

    def matches(self, problem: Problem, budget: Budget, entry: dict) -> tuple[bool, str]:
        prof = entry.get("problem_profile")
        if not prof:
            return False, "evidence has no problem profile"
        sp = problem.space
        if sp.kind != prof["kind"]:
            return False, f"space kind {sp.kind} != evidence kind {prof['kind']}"
        if not sp.types_present <= set(prof["types"]):
            return False, f"types {sorted(sp.types_present)} not covered by evidence types {prof['types']}"
        lo, hi = min(prof["dims"]), max(prof["dims"])
        if not (lo / 1.5 <= sp.dim <= hi * 1.5):
            return False, f"dimension {sp.dim} outside evidence range [{lo}, {hi}] x1.5"
        if bool(problem.has_constraints) != bool(prof["constrained"]):
            return False, "constraint class differs from the evidence"
        b = entry.get("budget")
        if b and budget.max_evaluations and not (b / 2 <= budget.max_evaluations <= 2 * b):
            return False, f"budget {budget.max_evaluations} not within x2 of benchmarked budget {b}"
        comp = entry.get("vs_best_baseline", {})
        if comp.get("pairs", 0) < self.min_runs:
            return False, f"only {comp.get('pairs', 0)} paired runs (< {self.min_runs})"
        p = comp.get("p_value_a_better")
        if p is None or p >= self.alpha:
            return False, f"validation p-value {p} not below {self.alpha}"
        return True, (f"champion beat baseline '{comp.get('baseline')}' on a validation split "
                      f"({comp['pairs']} paired runs, p={p:.3g}, mean score diff {comp.get('mean_diff'):+.3f})")

    def refine(self, problem: Problem, budget: Budget, plan: SolverPlan) -> SolverPlan:
        reasons = []
        best = None
        for meta in self.registry.list(status="champion"):
            rec = self.registry.get(meta["id"])
            for entry in rec.get("benchmarks", []):
                if entry.get("split") != "validation":
                    continue
                ok, why = self.matches(problem, budget, entry)
                if ok:
                    score = entry["summary"]["mean_score"]
                    if best is None or score < best[0]:
                        best = (score, rec, why, entry)
                else:
                    reasons.append(f"{rec['id']}: {why}")
        if best is None:
            plan.rationale.append("evidence: no validated champion matches this problem profile; rule-based plan kept"
                                  + (f" ({'; '.join(reasons[:3])})" if reasons else ""))
            return plan
        score, rec, why, entry = best
        return SolverPlan(f"candidate:{rec['id']}", {}, "generated", plan.rationale + [f"evidence override: {why}"],
                          source="evidence", evidence={"candidate": rec["id"], "validation_summary": entry["summary"],
                                                       "comparison": entry.get("vs_best_baseline")})
