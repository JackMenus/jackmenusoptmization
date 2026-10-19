"""Versioned, append-only registry of generated (and baseline) candidates.

Layout::

    <root>/candidates/<candidate_id>/v0001.json, v0002.json, ...
    <root>/index.json            (latest version and status per candidate; rebuildable)

A candidate id is derived from the content hash of its representation, so the
same operator proposed twice maps to the same id.  Every change (validation
outcome, benchmark result, status change, limitation) is written as a *new
version*; earlier versions are never modified.  Each record carries:

``id, version, created, kind (operator|algorithm), representation
(dsl|python), source, parents, history, compatible_kinds, hyperparameters,
validation, benchmarks, limitations, status, provider, rationale``.

Status values: ``proposed`` -> ``rejected`` | ``validated`` -> ``retained`` |
``discarded`` -> ``champion`` (passed validation-split comparison against the
baselines) | ``baseline``.  Only ``validated``/``retained``/``champion``
candidates may be materialised for ordinary optimisation runs.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import tempfile
from typing import Any

from ..core.result import _jsonable
from ..dsl.spec import canonical
import hashlib

USABLE = ("validated", "retained", "champion", "baseline")


def candidate_id(kind: str, representation: str, source: Any) -> str:
    body = canonical(source) if not isinstance(source, str) else source
    return f"{kind[:3]}-{representation[:3]}-" + hashlib.sha256(body.encode()).hexdigest()[:10]


class Registry:
    def __init__(self, root: str) -> None:
        self.root = root
        self.cdir = os.path.join(root, "candidates")
        os.makedirs(self.cdir, exist_ok=True)

    # ------------------------------------------------------------------
    def _write(self, path: str, obj: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
        with os.fdopen(fd, "w") as fh:
            json.dump(_jsonable(obj), fh, indent=2)
        os.replace(tmp, path)

    def versions(self, cid: str) -> list[int]:
        d = os.path.join(self.cdir, cid)
        if not os.path.isdir(d):
            return []
        return sorted(int(f[1:5]) for f in os.listdir(d) if f.startswith("v") and f.endswith(".json"))

    def get(self, cid: str, version: int | None = None) -> dict:
        vs = self.versions(cid)
        if not vs:
            raise KeyError(f"no candidate {cid}")
        v = vs[-1] if version is None else version
        with open(os.path.join(self.cdir, cid, f"v{v:04d}.json")) as fh:
            return json.load(fh)

    def exists(self, cid: str) -> bool:
        return bool(self.versions(cid))

    def add(self, *, kind: str, representation: str, source: Any, parents: list[str] | None = None,
            provider: str = "", rationale: str = "", operation: str = "created", hyperparameters: dict | None = None,
            compatible_kinds: list[str] | None = None, status: str = "proposed", extra: dict | None = None) -> dict:
        cid = candidate_id(kind, representation, source)
        if self.exists(cid):
            rec = self.get(cid)
            return self.update(cid, history_entry={"operation": "re-proposed", "provider": provider,
                                                   "parents": parents or []})
        rec = {
            "id": cid, "version": 1, "created": _now(), "kind": kind, "representation": representation,
            "source": source, "parents": parents or [], "provider": provider, "rationale": rationale,
            "history": [{"version": 1, "time": _now(), "operation": operation, "parents": parents or []}],
            "compatible_kinds": compatible_kinds or [], "hyperparameters": hyperparameters or {},
            "validation": None, "benchmarks": [], "limitations": [], "status": status,
        }
        if extra:
            rec.update(extra)
        d = os.path.join(self.cdir, cid)
        os.makedirs(d, exist_ok=True)
        self._write(os.path.join(d, "v0001.json"), rec)
        self._index_put(rec)
        return rec

    def update(self, cid: str, *, history_entry: dict | None = None, **fields) -> dict:
        rec = copy.deepcopy(self.get(cid))
        v = rec["version"] + 1
        for k, val in fields.items():
            if k == "benchmarks_append":
                rec["benchmarks"] = rec.get("benchmarks", []) + [val]
            elif k == "limitations_append":
                rec["limitations"] = rec.get("limitations", []) + [val]
            else:
                rec[k] = val
        entry = {"version": v, "time": _now(), **(history_entry or {"operation": "updated", "fields": sorted(fields)})}
        rec["history"] = rec.get("history", []) + [entry]
        rec["version"] = v
        self._write(os.path.join(self.cdir, cid, f"v{v:04d}.json"), rec)
        self._index_put(rec)
        return rec

    # ------------------------------------------------------------------
    def _index_path(self) -> str:
        return os.path.join(self.root, "index.json")

    def _index(self) -> dict:
        p = self._index_path()
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
        return {}

    def _index_put(self, rec: dict) -> None:
        idx = self._index()
        idx[rec["id"]] = {"version": rec["version"], "status": rec["status"], "kind": rec["kind"],
                          "representation": rec["representation"], "provider": rec.get("provider", ""),
                          "name": (rec["source"].get("name") if isinstance(rec["source"], dict) else None)}
        self._write(self._index_path(), idx)

    def rebuild_index(self) -> dict:
        idx = {}
        for cid in sorted(os.listdir(self.cdir)):
            if self.versions(cid):
                r = self.get(cid)
                idx[cid] = {"version": r["version"], "status": r["status"], "kind": r["kind"],
                            "representation": r["representation"], "provider": r.get("provider", "")}
        self._write(self._index_path(), idx)
        return idx

    def list(self, status: str | None = None, kind: str | None = None) -> list[dict]:
        out = []
        for cid, meta in sorted(self._index().items()):
            if status is not None and meta["status"] != status:
                continue
            if kind is not None and meta["kind"] != kind:
                continue
            out.append({"id": cid, **meta})
        return out

    def usable(self, cid: str) -> bool:
        return self.exists(cid) and self.get(cid)["status"] in USABLE


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
