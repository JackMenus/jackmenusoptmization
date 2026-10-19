"""Command-line interface: ``optiforge <command>``."""
from __future__ import annotations

import json
import os
import sys

import click

from ..core.result import _jsonable


def _dump(obj) -> str:
    return json.dumps(_jsonable(obj), indent=2)


@click.group()
@click.version_option(package_name="optiforge")
def cli():
    """optiforge: optimisation platform with an autonomous operator/algorithm development layer."""


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--solver", default=None, help="Override the solver (name, 'auto', or 'candidate:<id>').")
@click.option("--seed", type=int, default=None)
@click.option("--budget", type=int, default=None, help="Override max objective evaluations.")
@click.option("--runs-dir", default=None, help="Where run records go (default: config runs_dir or ./runs).")
@click.option("--checkpoint-every", type=int, default=None, help="Write a checkpoint every K iterations.")
@click.option("--resume", "resume_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Resume from the checkpoint of an earlier solve run directory.")
def solve(config, solver, seed, budget, runs_dir, checkpoint_every, resume_dir):
    """Solve the problem described by CONFIG."""
    from ..api import solve as api_solve
    from ..experiments.config import load_config, resolve_budget, resolve_problem
    from ..experiments.store import create_run
    from ..registry.store import Registry
    cfg = load_config(config)
    if solver:
        cfg["solver"] = solver
    if seed is not None:
        cfg["seed"] = seed
    if budget is not None:
        cfg.setdefault("budget", {})
        cfg["budget"] = {**(cfg["budget"] if isinstance(cfg["budget"], dict) else {}), "max_evaluations": budget}
    problem = resolve_problem(cfg["problem"])
    b = resolve_budget(cfg.get("budget"))
    rec = create_run(runs_dir or cfg.get("runs_dir", "runs"), "solve", cfg)
    rec.attach_logging()
    kw = {}
    ckpt = None
    if resume_dir:
        ckpt = os.path.join(resume_dir, "checkpoint.json")
        kw.update(checkpoint_path=ckpt, resume=True, checkpoint_every=checkpoint_every or cfg.get("checkpoint_every", 10))
        rec.event("resume", from_checkpoint=ckpt)
    elif checkpoint_every or cfg.get("checkpoint_every"):
        ckpt = rec.artifact("checkpoint.json")
        kw.update(checkpoint_path=ckpt, checkpoint_every=checkpoint_every or cfg["checkpoint_every"])
    reg = Registry(cfg["registry"]) if cfg.get("registry") else None
    try:
        res = api_solve(problem, cfg.get("solver", "auto"), b, seed=cfg.get("seed", 0), mode=cfg.get("mode", "auto"),
                        solver_params=cfg.get("solver_params"), registry=reg, **kw)
    finally:
        rec.detach_logging()
    rec.write_json("result.json", res.to_dict())
    rec.event("run_finished", status=res.status, objective=res.objective, evaluations=res.n_evaluations)
    click.echo(res.summary())
    for g in res.guarantees:
        click.echo(f"  - {g.kind}: {g.statement}")
    for m in res.messages:
        click.echo(f"  ! {m}")
    click.echo(f"run record: {rec.path}")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--registry", default=None, help="Registry to consult for validated evidence.")
def plan(config, registry):
    """Show the solver-selection plan (and its rationale) for CONFIG's problem."""
    from ..api import plan as api_plan
    from ..experiments.config import load_config, resolve_budget, resolve_problem
    from ..registry.store import Registry
    cfg = load_config(config)
    p = resolve_problem(cfg["problem"])
    reg = Registry(registry or cfg["registry"]) if (registry or cfg.get("registry")) else None
    click.echo(_dump(api_plan(p, resolve_budget(cfg.get("budget")), mode=cfg.get("mode", "auto"), registry=reg).to_dict()))


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--runs-dir", default=None)
def develop(config, runs_dir):
    """Run the autonomous operator/algorithm development loop described by CONFIG."""
    from ..experiments.config import load_config
    from ..experiments.store import create_run
    from ..generation.loop import DevelopmentConfig, DevelopmentLoop
    cfg = load_config(config)
    dc = DevelopmentConfig.from_dict(cfg.get("development", cfg))
    rec = create_run(runs_dir or cfg.get("runs_dir", "runs"), "develop", cfg)
    rec.attach_logging()
    try:
        loop = DevelopmentLoop(dc, rec.path, event=lambda name, **kw: rec.event(name, **kw))
        rep = loop.run()
    finally:
        rec.detach_logging()
    rec.write_json("report.json", {k: v for k, v in rep.items() if k != "test"} | {"test": {k: v for k, v in rep["test"].items() if k != "rows"}})
    click.echo(open(os.path.join(rec.path, "development_report.md")).read())
    click.echo(f"run record: {rec.path}")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--runs-dir", default=None)
def benchmark(config, runs_dir):
    """Run a benchmark experiment (baselines and optional registered candidates)."""
    from ..benchmarks.report import render_benchmark_markdown, run_benchmark
    from ..experiments.config import load_config
    from ..experiments.store import create_run
    cfg = load_config(config)
    rec = create_run(runs_dir or cfg.get("runs_dir", "runs"), "benchmark", cfg)
    rep = run_benchmark(cfg.get("benchmark", cfg))
    rec.write_json("report.json", rep)
    md = render_benchmark_markdown(rep)
    with open(rec.artifact("report.md"), "w") as fh:
        fh.write(md)
    rec.event("benchmark_finished", wall_time_s=rep["wall_time_s"])
    click.echo(md)
    click.echo(f"run record: {rec.path}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--events", is_flag=True, help="Also print the structured event log.")
def inspect(run_dir, events):
    """Summarise a run record directory."""
    from ..experiments.store import load_run
    r = load_run(run_dir)
    click.echo(f"run: {r['path']}")
    click.echo(f"kind: {r.get('config', {}).get('kind', '?')}  python: {r.get('environment', {}).get('python', '?').split()[0]}  "
               f"packages: {r.get('environment', {}).get('packages')}")
    if "result" in r:
        res = r["result"]
        click.echo(f"solver: {res['solver']} {res['solver_version']}  status: {res['status']}  objective: {res['objective']}  "
                   f"feasible: {res['feasible']}  evaluations: {res['n_evaluations']}  runtime: {res['runtime_s']:.3f}s  "
                   f"termination: {res['termination_reason']}  seed: {res['seed']}")
        for g in res["guarantees"]:
            click.echo(f"  - {g['kind']}: {g['statement']}")
        click.echo(f"convergence history: {len(res['history'])} incumbent updates; last: {res['history'][-1] if res['history'] else None}")
    if "report" in r:
        rep = r["report"]
        if "selection" in rep:
            click.echo("selection: " + rep["selection"].get("decision", ""))
        if "summary" in rep:
            for s, v in rep["summary"].items():
                click.echo(f"  {s}: mean score {v['mean_score']}, feasible {v['feasibility_rate']}")
    if events:
        for e in r.get("events", []):
            click.echo(json.dumps(e))


@cli.group()
def registry():
    """Inspect the candidate registry."""


@registry.command("list")
@click.option("--path", default="registry", show_default=True)
@click.option("--status", default=None)
def registry_list(path, status):
    from ..registry.store import Registry
    for r in Registry(path).list(status=status):
        click.echo(f"{r['id']}  v{r['version']}  {r['status']:<10} {r['kind']:<9} {r['representation']:<7} {r.get('provider', '')}")


@registry.command("show")
@click.argument("candidate_id")
@click.option("--path", default="registry", show_default=True)
@click.option("--version", type=int, default=None)
def registry_show(candidate_id, path, version):
    from ..registry.store import Registry
    click.echo(_dump(Registry(path).get(candidate_id, version)))


@cli.command("sandbox-check")
def sandbox_check():
    """Probe operating-system isolation for generated code."""
    from ..isolation.sandbox import probe
    rep = probe(force=True)
    click.echo(_dump(rep.to_dict()))
    sys.exit(0 if rep.available else 1)


@cli.command()
def solvers():
    """List the solver portfolio."""
    from ..solvers import SOLVERS
    for name, cls in sorted(SOLVERS.items()):
        s = cls()
        i = s.info()
        click.echo(f"{name:<20} {i['family']:<8} kinds={','.join(i['space_kinds']):<28} {i['description']}")


@cli.command()
def problems():
    """List benchmark problem families and suites."""
    from ..benchmarks.problems import FAMILIES
    from ..benchmarks.suites import SUITES
    for n, f in sorted(FAMILIES.items()):
        click.echo(f"{n:<20} [{', '.join(sorted(f.tags))}] {f.description}")
    click.echo("")
    for n, s in sorted(SUITES.items()):
        click.echo(f"suite {n}: {[f for f, _ in s.families]} splits={s.splits}")


def main():
    cli()


if __name__ == "__main__":
    main()
