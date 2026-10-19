"""Reproducible experiment records.

Each run gets a directory ``<runs_dir>/<UTC timestamp>-<kind>-<short hash>/`` with

* ``config.json``    - the fully resolved configuration that was executed,
* ``environment.json`` - Python, platform, and pinned package versions,
* ``events.jsonl``   - structured log events (one JSON object per line),
* ``result.json`` / ``report.json`` - outputs written by the command,
* any artefacts (checkpoints, per-run tables).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import sys
import time
from importlib import metadata
from typing import Any

from ..core.result import _jsonable

PACKAGES = ("numpy", "scipy", "PyYAML", "click", "pytest", "anthropic")


def environment_info() -> dict:
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    from .. import __version__
    return {"optiforge": __version__, "python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "packages": vers, "cpu_count": os.cpu_count()}


class JsonLinesHandler(logging.Handler):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        evt = {"t": round(record.created, 4), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        extra = getattr(record, "event", None)
        if isinstance(extra, dict):
            evt.update(extra)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(_jsonable(evt)) + "\n")


class RunRecord:
    def __init__(self, path: str, kind: str, config: dict) -> None:
        self.path = path
        self.kind = kind
        self.config = config
        self.events_path = os.path.join(path, "events.jsonl")
        self._handler: JsonLinesHandler | None = None

    def attach_logging(self, level: int = logging.INFO) -> None:
        self._handler = JsonLinesHandler(self.events_path)
        self._handler.setLevel(level)
        root = logging.getLogger("optiforge")
        root.addHandler(self._handler)
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)

    def detach_logging(self) -> None:
        if self._handler is not None:
            logging.getLogger("optiforge").removeHandler(self._handler)
            self._handler = None

    def event(self, name: str, **fields: Any) -> None:
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps(_jsonable({"t": round(time.time(), 4), "event": name, **fields})) + "\n")

    def write_json(self, name: str, obj: Any) -> str:
        p = os.path.join(self.path, name)
        with open(p, "w") as fh:
            json.dump(_jsonable(obj), fh, indent=2)
        return p

    def artifact(self, name: str) -> str:
        return os.path.join(self.path, name)


def create_run(runs_dir: str, kind: str, config: dict) -> RunRecord:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    h = hashlib.sha256(json.dumps(_jsonable(config), sort_keys=True).encode()).hexdigest()[:8]
    path = os.path.join(runs_dir, f"{stamp}-{kind}-{h}")
    os.makedirs(path, exist_ok=False)
    rec = RunRecord(path, kind, config)
    rec.write_json("config.json", config)
    rec.write_json("environment.json", environment_info())
    rec.event("run_started", kind=kind, config_hash=h)
    return rec


def load_run(path: str) -> dict:
    out: dict[str, Any] = {"path": path}
    for name in ("config.json", "environment.json", "result.json", "report.json"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            with open(p) as fh:
                out[name[:-5]] = json.load(fh)
    ev = os.path.join(path, "events.jsonl")
    if os.path.exists(ev):
        with open(ev) as fh:
            out["events"] = [json.loads(line) for line in fh if line.strip()]
    return out
