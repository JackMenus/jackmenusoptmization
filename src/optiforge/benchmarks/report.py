"""Configuration-driven benchmark experiments (baselines and registered candidates)."""
from __future__ import annotations

import time

from ..core.budget import Budget
from ..registry.store import Registry
from .runner import SolverSpec, builtin, paired_comparison, per_family, reference_table, run_matrix, summarize
from .suites import get_suite


def run_benchmark(cfg: dict) -> dict:
    suite = get_suite(cfg.get("suite", "baseline_portfolio"))
    split = cfg.get("split", "test")
    inst = suite.instances(split)
    families = cfg.get("families")
    if families:
        inst = [i for i in inst if i.family in families]
    budget = Budget(max_evaluations=int(cfg.get("budget", 3000)), max_time=cfg.get("max_time"))
    seeds = list(cfg.get("seeds", [0, 1, 2]))
    names = list(cfg.get("solvers", ["random_search", "auto"]))
    if "random_search" not in names:
        names.insert(0, "random_search")
    specs = [builtin(n) for n in names]
    if cfg.get("candidates"):
        reg = Registry(cfg.get("registry", "registry"))
        specs += [SolverSpec(c, record=reg.get(c)) for c in cfg["candidates"]]
    t = time.time()
    rows = run_matrix(specs, inst, seeds, budget, None, "benchmark", cfg.get("workers"))
    base_names = [n for n in names]
    ref = reference_table([r for r in rows if r["solver"] in base_names])
    summ = summarize(rows, ref)
    comps = {}
    ref_solver = cfg.get("compare_against")
    if ref_solver:
        comps = {s.label: paired_comparison(rows, s.label, ref_solver) for s in specs if s.label != ref_solver}
    return {"suite": suite.to_dict(), "split": split, "instances": [i.id for i in inst], "budget": budget.to_dict(),
            "seeds": seeds, "solvers": [s.label for s in specs], "summary": summ, "per_family": per_family(rows),
            "comparisons": comps, "reference": ref, "rows": rows, "wall_time_s": round(time.time() - t, 2)}


def render_benchmark_markdown(rep: dict) -> str:
    L = [f"# Benchmark: suite `{rep['suite']['name']}`, split `{rep['split']}`", "",
         f"{len(rep['instances'])} instances x {len(rep['seeds'])} seeds, budget {rep['budget']['max_evaluations']} evaluations "
         f"per run, wall time {rep['wall_time_s']} s.", "",
         "Score: 0 = verified optimum (or best baseline value when no optimum is known), 1 = random-search median at the "
         "same budget, 2 = no feasible result. Lower is better.", "",
         "| solver | runs | n/a | mean score | median | std | IQR | feasible | failures | solved (gap<=1e-6) | ms/eval |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in sorted(rep["summary"].items(), key=lambda kv: kv[1]["mean_score"] if kv[1]["mean_score"] is not None else 9):
        sr = v["solved_rate"]
        L.append(f"| `{s}` | {v['runs']} | {v['not_applicable_runs']} | {v['mean_score']:.4f} | {v['median_score']:.4f} | "
                 f"{v['std_score']:.4f} | {v['iqr_score']:.4f} | {v['feasibility_rate']:.2f} | {v['failure_rate']:.2f} | "
                 f"{'n/a' if sr is None else f'{sr:.2f}'} | {v['runtime_per_eval_ms']:.3f} |")
    fams = rep["per_family"]
    solvers = sorted({s for d in fams.values() for s in d})
    L += ["", "## Mean score per family", "", "| family | " + " | ".join(f"`{s}`" for s in solvers) + " |",
          "|---|" + "---|" * len(solvers)]
    for f, d in sorted(fams.items()):
        L.append(f"| {f} | " + " | ".join("—" if s not in d else f"{d[s]:.3f}" for s in solvers) + " |")
    ref = rep["reference"]
    L += ["", "## Reference values", "", "| instance | reference | kind | random-search median |", "|---|---|---|---|"]
    for i, r in sorted(ref.items()):
        fr = "n/a" if r["f_ref"] is None else f"{r['f_ref']:.6g}"
        fq = "n/a" if r["f_rand"] is None else f"{r['f_rand']:.6g}"
        L.append(f"| {i} | {fr} | {r['ref_kind']} | {fq} |")
    if rep.get("comparisons"):
        L += ["", "## Paired comparisons", ""]
        for s, c in rep["comparisons"].items():
            p = c.get("p_value_a_better")
            L.append(f"- `{s}`: mean diff {c['mean_diff'] if c['mean_diff'] is None else round(c['mean_diff'], 4)}, wins {c['wins']}, "
                     f"losses {c['losses']}, ties {c['ties']}, p={'n/a' if p is None else f'{p:.3g}'}")
    return "\n".join(L) + "\n"
