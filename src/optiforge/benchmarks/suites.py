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
    version: str = "1"
    # (family, params) entries used ONLY in the test split: held-out problem families, not just held-out instances
    test_only: list[tuple[str, dict]] = field(default_factory=list)
    # data that has already been looked at and used to steer development (historical diagnostic, not confirmatory)
    test_status: str = "untouched"

    def instances(self, split: str) -> list[InstanceSpec]:
        seeds = self.splits[split]
        out = []
        fams = list(self.families) + (list(self.test_only) if split == "test" else [])
        for fam, params in fams:
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
        shared = {f for f, _ in self.test_only} & {f for f, _ in self.families}
        if shared:
            raise ValueError(f"suite {self.name}: test-only families also appear in train/validation: {sorted(shared)}")

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "families": [[f, p] for f, p in self.families],
                "test_only": [[f, p] for f, p in self.test_only], "splits": self.splits,
                "description": self.description, "test_status": self.test_status}


STD_SPLITS = {"train": [0, 1], "validation": [100, 101], "test": [200, 201, 202]}
# Version-1 suites: their test splits were run and reported in the 0.1.0 delivery (2026-10-19), so they are
# historical diagnostic / regression data and may not support new confirmatory claims.
V1_TEST_STATUS = "historical: viewed in the 0.1.0 delivery (2026-10-19); regression/diagnostic use only"

SUITES: dict[str, Suite] = {
    "continuous_multimodal": Suite(
        "continuous_multimodal",
        [("rastrigin", {"dim": 6}), ("ackley", {"dim": 6}), ("styblinski_tang", {"dim": 6}), ("rosenbrock", {"dim": 6})],
        dict(STD_SPLITS), "Nonconvex continuous functions (5 of 6 multimodal), dimension 6.", test_status=V1_TEST_STATUS),
    "discrete_combinatorial": Suite(
        "discrete_combinatorial",
        [("knapsack", {"n": 25}), ("maxcut", {"n": 14}), ("integer_quadratic", {"n": 4})],
        dict(STD_SPLITS), "Binary/integer combinatorial problems with verified optima.", test_status=V1_TEST_STATUS),
    "permutation": Suite("permutation", [("tsp", {"n": 10})], dict(STD_SPLITS), "Euclidean TSP, Held-Karp optima."),
    "mixed": Suite("mixed", [("mixed_rastrigin", {}), ("mixed_constrained", {"n": 4})], dict(STD_SPLITS),
                   "Mixed real/integer/binary/categorical problems with verified optima.", test_status=V1_TEST_STATUS),
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
        "One test instance of every family, including both simulation-based systems.", test_status=V1_TEST_STATUS),
    # Version 2 (2026-10-20). Rotated / non-separable / ill-conditioned families in two dimensions, fresh instance
    # seeds, and two families that appear only in the test split. Its test split was run once in round 6
    # (lab/r6-develop-v2) and is now historical.
    "continuous_general": Suite(
        "continuous_general",
        [("rastrigin", {"dim": 5}), ("rastrigin_rotated", {"dim": 5}), ("rastrigin_rotated", {"dim": 10}),
         ("ackley_rotated", {"dim": 5}), ("ackley_rotated", {"dim": 10}), ("ellipsoid", {"dim": 10, "log10_cond": 6}),
         ("schwefel_1_2", {"dim": 10}), ("rosenbrock", {"dim": 10})],
        {"train": [1010, 1011], "validation": [1020, 1021], "test": [1030, 1031, 1032]},
        "Separable and rotated multimodal, ill-conditioned and non-separable unimodal continuous functions, "
        "dimensions 5 and 10; test adds two held-out families that no suite uses elsewhere (rotated bent cigar at dimension 10, rotated Griewank at dimension 8).",
        version="2",
        test_only=[("bent_cigar_rotated", {"dim": 10}), ("griewank_rotated", {"dim": 8})],
        test_status="historical: test split run once in round 6 (2026-10-20, lab/r6-develop-v2); "
                    "regression/diagnostic use only"),
    # Version 2 of the discrete suite (2026-10-20, round 8). Sizes were chosen from a pilot on instance seeds 1100-1101
    # (lab/r8-discrete-v2/pilot.json), which no split uses: families where the baselines mostly reach the optimum at
    # 3000-10000 evaluations (maxcut n=20) were left out so that comparisons are not dominated by ties.
    "discrete_general": Suite(
        "discrete_general",
        [("knapsack_correlated", {"n": 50}), ("knapsack_correlated", {"n": 100}), ("multi_knapsack", {"n": 40, "m": 5}),
         ("independent_set", {"n": 40}), ("integer_quadratic", {"n": 6, "radius": 7})],
        {"train": [2010, 2011], "validation": [2020, 2021], "test": [2030, 2031, 2032]},
        "Correlated and multidimensional knapsacks, weighted independent set and an integer quadratic, with exact "
        "optima (dynamic programming, MILP with HiGHS at gap 0, enumeration); test adds a held-out family "
        "(weighted set cover) that no other suite uses.",
        version="2",
        test_only=[("set_cover", {"n_sets": 40, "n_elements": 30})],
        test_status="historical: test split run once in round 10 (2026-10-20, lab/r10-develop-discrete); "
                    "regression/diagnostic use only"),
    # Version 3 of the discrete suite (2026-10-20, round 12): the v2 families with fresh instance seeds and five test
    # instances per family, for confirming candidates frozen in round 10 on data no decision has used.
    "discrete_general_v3": Suite(
        "discrete_general_v3",
        [("knapsack_correlated", {"n": 50}), ("knapsack_correlated", {"n": 100}), ("multi_knapsack", {"n": 40, "m": 5}),
         ("independent_set", {"n": 40}), ("integer_quadratic", {"n": 6, "radius": 7})],
        {"train": [3010, 3011], "validation": [3020, 3021], "test": [3030, 3031, 3032, 3033, 3034]},
        "discrete_general v2 families with fresh instance seeds; five test instances per family; test also includes "
        "set_cover (seen as a held-out family in the v2 test split, new instances here).",
        version="3",
        test_only=[("set_cover", {"n_sets": 40, "n_elements": 30})],
        test_status="historical: test split run once in round 12 (2026-10-20, lab/r12-confirm-discrete); "
                    "regression/diagnostic use only"),
}


def get_suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"unknown suite '{name}'. Available: {sorted(SUITES)}")
    s = SUITES[name]
    s.check_disjoint()
    return s
