"""CLI: solve / plan / inspect / benchmark / registry / sandbox-check, including checkpoint resume."""
import json
import os

import yaml
from click.testing import CliRunner

from optiforge.cli.main import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(name):
    return os.path.join(ROOT, "configs", name)


def invoke(*args):
    r = CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return r.output


def run_dir(out):
    return out.strip().splitlines()[-1].split("run record: ")[1]


def test_solve_and_inspect(tmp_path):
    out = invoke("solve", cfg("solve_rastrigin.yaml"), "--budget", 2000, "--runs-dir", tmp_path)
    assert "status=feasible" in out and "heuristic_best_found" in out
    d = run_dir(out)
    res = json.load(open(os.path.join(d, "result.json")))
    assert res["n_evaluations"] <= 2000 and res["seed"] == 0
    assert os.path.exists(os.path.join(d, "events.jsonl")) and os.path.exists(os.path.join(d, "environment.json"))
    ins = invoke("inspect", d, "--events")
    assert "termination" in ins and "run_finished" in ins


def test_exact_certificate_via_cli(tmp_path):
    out = invoke("solve", cfg("solve_milp.yaml"), "--runs-dir", tmp_path)
    assert "optimal_certified" in out


def test_plan_shows_rationale():
    out = json.loads(invoke("plan", cfg("solve_knapsack_exact.yaml")))
    assert out["solver"] == "exhaustive" and out["rationale"]


def test_checkpoint_and_resume(tmp_path):
    out = invoke("solve", cfg("solve_rastrigin.yaml"), "--solver", "cmaes", "--budget", 1500,
                 "--checkpoint-every", 2, "--runs-dir", tmp_path)
    d = run_dir(out)
    assert os.path.exists(os.path.join(d, "checkpoint.json"))
    first = json.load(open(os.path.join(d, "result.json")))
    out2 = invoke("solve", cfg("solve_rastrigin.yaml"), "--solver", "cmaes", "--budget", 1500, "--resume", d,
                  "--runs-dir", tmp_path)
    second = json.load(open(os.path.join(run_dir(out2), "result.json")))
    assert second["objective"] == first["objective"] and second["n_evaluations"] == first["n_evaluations"]


def test_factory_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = invoke("solve", cfg("solve_facility_mixed.yaml"), "--budget", 1500, "--runs-dir", tmp_path)
    assert "facility" in out


def test_benchmark_small(tmp_path):
    c = {"kind": "benchmark", "benchmark": {"suite": "continuous_multimodal", "split": "validation", "budget": 300,
                                            "seeds": [0, 1], "solvers": ["random_search", "de", "auto"],
                                            "compare_against": "random_search", "families": ["sphere", "rastrigin"]}}
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(c))
    out = invoke("benchmark", p, "--runs-dir", tmp_path)
    assert "| `de` |" in out and "Paired comparisons" in out
    rep = json.load(open(os.path.join(run_dir(out), "report.json")))
    assert rep["summary"]["random_search"]["runs"] == len(rep["instances"]) * 2 > 0


def test_listing_commands():
    assert "cmaes" in invoke("solvers")
    assert "knapsack" in invoke("problems")
    r = CliRunner().invoke(cli, ["sandbox-check"])
    assert r.exit_code in (0, 1) and "available" in r.output
