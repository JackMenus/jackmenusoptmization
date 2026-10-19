"""Turning registry records / proposals into runnable solvers."""
from __future__ import annotations

from ..core.errors import CandidateRejected
from ..dsl.interpreter import DSLAlgorithm
from ..dsl.spec import host_algorithm
from ..registry.store import USABLE, Registry
from .code_operators import SandboxedCodeOperator


def code_host_variation(cid: str) -> dict:
    return {"op": "seq", "children": [{"op": "uniform_cx"}, {"op": "code", "candidate": cid}]}


def build_solver(rec: dict, allow_unvalidated: bool = False) -> DSLAlgorithm:
    """Build a solver for a record.  Ordinary use requires a validated status; the
    validation and benchmarking pipeline passes ``allow_unvalidated=True``."""
    if not allow_unvalidated and rec.get("status") not in USABLE:
        raise CandidateRejected(f"candidate {rec['id']} has status '{rec.get('status')}' and may not be used for "
                                "ordinary optimisation runs (it must pass validation first)")
    cid = rec["id"]
    if rec["representation"] == "dsl":
        if rec["kind"] == "algorithm":
            return DSLAlgorithm(rec["source"], cid)
        return DSLAlgorithm(host_algorithm(rec["source"], name=f"host[{cid}]"), cid)
    if rec["representation"] == "python":
        meta = rec.get("meta", {})
        op = SandboxedCodeOperator(cid, rec["source"], meta.get("types", ["real", "integer"]), version=f"v{rec.get('version', 1)}")
        return DSLAlgorithm(host_algorithm(code_host_variation(cid), name=f"host[{cid}]"), cid, code_ops={cid: op})
    raise CandidateRejected(f"unknown representation {rec['representation']}")


def solver_from_registry(registry: Registry | None, cid: str) -> DSLAlgorithm:
    if registry is None:
        raise ValueError("a registry is required to run a candidate")
    return build_solver(registry.get(cid))
