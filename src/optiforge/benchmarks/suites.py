"""Benchmark suites with disjoint train / validation / test instance splits."""
from __future__ import annotations

from dataclasses import dataclass, field

from .problems import FAMILIES, make_problem

# Families whose instance does not depend on the seed: they may only appear in
# one split (we put them in "test" or in baseline-only suites) to avoid leakage.
FIXED_FAMILIES = {"disk_rosenbrock", "g06"}


@dataclass(frozen=True)
class InstanceSpec:
    family: str
    seed: int
    params: tuple = ()

    @property
    def id(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.family}[{p}]#{self.seed}" if p else f"{self.family}#{self.seed}"

    def make(self):
        return make_problem(self.family, self.seed, **dict(self.params))

    def to_dict(self) -> dict:
        return {"family": self.family, "seed": self.seed, "params": dict(self.params)}


@dataclass
class Suite:
    name: str
    families: list[tuple[str, dict]]
    splits: dict[str, list[int]] = field(default_factory=dict)
    description: str = ""

    def instances(self, split: str) -> list[InstanceSpec]:
        seeds = self.splits[split]
        out = []
        for fam, params in self.families:
            if fam not in FAMILIES:
                raise KeyError(fam)
            if fam in FIXED_FAMILIES:
                if split == "test":
                    out.append(InstanceSpec(fam, 0, tuple(sorted(params.items()))))
                continue
            out += [InstanceSpec(fam, s, tuple(sorted(params.items()))) for s in seeds]
        return out

    def check_disjoint(self) -> None:
        tr, va, te = (set(self.splits.get(k, [])) for k in ("train", "validation", "test"))
        if tr & va or tr & te or va & te:
            raise ValueError(f"suite {self.name}: instance seeds overlap between splits")

    def to_dict(self) -> dict:
        return {"name": self.name, "families": [[f, p] for f, p in self.families], "splits": self.splits,
                "description": self.description}


STD_SPLITS = {"train": [0, 1], "validation": [100, 101], "test": [200, 201, 202]}

SUITES: dict[str, Suite] = {
    "continuous_multimodal": Suite(
        "continuous_multimodal",
        [("rastrigin", {"dim": 6}), ("ackley", {"dim": 6}), ("styblinski_tang", {"dim": 6}), ("rosenbrock", {"dim": 6})],
        dict(STD_SPLITS), "Nonconvex continuous functions (5 of 6 multimodal), dimension 6."),
    "discrete_combinatorial": Suite(
        "discrete_combinatorial",
        [("knapsack", {"n": 25}), ("maxcut", {"n": 14}), ("integer_quadratic", {"n": 4})],
        dict(STD_SPLITS), "Binary/integer combinatorial problems with verified optima."),
    "permutation": Suite("permutation", [("tsp", {"n": 10})], dict(STD_SPLITS), "Euclidean TSP, Held-Karp optima."),
    "mixed": Suite("mixed", [("mixed_rastrigin", {}), ("mixed_constrained", {"n": 4})], dict(STD_SPLITS),
                   "Mixed real/integer/binary/categorical problems with verified optima."),
    "constrained_continuous": Suite(
        "constrained_continuous",
        [("halfspace_sphere", {"dim": 5}), ("hyperplane_sphere", {"dim": 5}), ("disk_rosenbrock", {}), ("g06", {})],
        dict(STD_SPLITS), "Convex and nonconvex constrained continuous problems."),
    "baseline_portfolio": Suite(
        "baseline_portfolio",
        [("sphere", {"dim": 5}), ("ellipsoid", {"dim": 5}), ("rosenbrock", {"dim": 5}), ("rastrigin", {"dim": 5}),
         ("ackley", {"dim": 5}), ("halfspace_sphere", {"dim": 5}), ("hyperplane_sphere", {"dim": 5}),
         ("disk_rosenbrock", {}), ("g06", {}), ("knapsack", {"n": 20}), ("tsp", {"n": 9}), ("maxcut", {"n": 14}),
         ("integer_quadratic", {"n": 4}), ("mixed_rastrigin", {}), ("mixed_constrained", {"n": 4}),
         ("pd_control", {}), ("inventory", {})],
        {"train": [], "validation": [], "test": [300]},
        "One test instance of every family, including both simulation-based systems."),
}


def get_suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"unknown suite '{name}'. Available: {sorted(SUITES)}")
    s = SUITES[name]
    s.check_disjoint()
    return s
