"""Operating-system isolation for untrusted generated code (Linux).

Generated Python operators run in a persistent worker process that is

* started as an unprivileged user (``nobody``) when the platform runs as root,
* placed in fresh **user, mount, PID, network, IPC and UTS namespaces**
  (``unshare``) - the network namespace has no interfaces, so all network
  access fails,
* given a new root filesystem (``pivot_root`` onto a private tmpfs) that
  contains only a read-only bind of ``/usr`` (the Python runtime), a
  read-only runner directory and a small private tmpfs ``/tmp``; the host
  root is detached, so ``/home``, ``/root``, ``/etc``, project files and the
  platform's evaluator, benchmark definitions, registry and validation code
  are not reachable,
* stripped of all capabilities (empty bounding set, ``no_new_privs``),
* started with an empty environment (no inherited credentials/tokens),
* bounded by rlimits on CPU time, address space, file size, open files and
  processes, plus a per-request wall-clock timeout and a cap on the bytes it
  may return.

The parent process only exchanges JSON lines with the worker and validates
every value it returns; the worker never sees the objective function.

Limitations (stated, not hidden): there is no seccomp syscall filter, so the
kernel's syscall surface is still reachable; isolation relies on Linux
namespaces being correctly implemented.  This is substantially stronger than
an AST filter or a plain subprocess but weaker than a virtual machine.  If the
probe below fails, arbitrary code execution is disabled and only the
declarative DSL is used for autonomous generation.
"""
from __future__ import annotations

import json
import os
import resource
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

from ..core.errors import IsolationUnavailable

NOBODY = 65534

_INNER = r"""
set -eu
ROOT="$1"; RUNNER="$2"; PY="$3"
mount -t tmpfs -o size=8m,mode=755 tmpfs "$ROOT"
cd "$ROOT"
mkdir usr tmp proc dev runner oldroot
mount --rbind /usr usr
mount -o remount,bind,ro,nosuid,nodev usr
for d in bin sbin lib lib64 lib32; do
  if [ -L "/$d" ]; then ln -s "$(readlink /$d)" "$d";
  elif [ -d "/$d" ]; then mkdir "$d"; mount --rbind "/$d" "$d"; mount -o remount,bind,ro,nosuid,nodev "$d"; fi
done
mount --bind "$RUNNER" runner
mount -o remount,bind,ro,nosuid,nodev runner
mount -t tmpfs -o size=4m,mode=1777,nosuid,nodev tmpfs tmp
touch dev/null && mount --bind /dev/null dev/null
mount -t proc -o nosuid,nodev,noexec proc proc
pivot_root . oldroot
cd /
umount -l /oldroot
rmdir /oldroot
exec /usr/bin/env -i /usr/bin/setpriv --bounding-set=-all --inh-caps=-all --no-new-privs -- "$PY" -I /runner/runner.py
"""

_RUNNER = r'''
import json, math, random, sys
def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
def main():
    first = sys.stdin.readline()
    if not first:
        return
    msg = json.loads(first)
    ns = {"__name__": "candidate", "math": math, "random": random}
    try:
        exec(compile(msg["source"], "<candidate>", "exec"), ns)
        fn = ns[msg.get("entry", "mutate")]
        if not callable(fn):
            raise TypeError("entry point is not callable")
    except BaseException as exc:
        reply({"ok": False, "error": "load: %s: %s" % (type(exc).__name__, exc)}); return
    reply({"ok": True})
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "exit":
            return
        try:
            rng = random.Random(req["seed"])
            y = fn(list(req["x"]), req["info"], rng)
            reply({"ok": True, "y": list(map(float, y))})
        except BaseException as exc:
            reply({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
main()
'''

_PROBE = r'''
def mutate(x, info, rng):
    import os, socket
    out = {}
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1); out["network_blocked"] = False
    except OSError:
        out["network_blocked"] = True
    try:
        open("/usr/.optiforge_probe", "w"); out["usr_readonly"] = False
    except OSError:
        out["usr_readonly"] = True
    out["root_listing"] = sorted(os.listdir("/"))
    out["host_paths_hidden"] = not any(os.path.exists(p) for p in ("/home", "/root", "/etc", "/mnt", "/var", "/oldroot"))
    out["env_empty"] = len([k for k in os.environ if k not in ("PWD", "SHLVL", "_", "LC_CTYPE")]) == 0
    status = open("/proc/self/status").read()
    cap = [l.split()[1] for l in status.splitlines() if l.startswith("CapEff:")][0]
    out["no_capabilities"] = int(cap, 16) == 0
    nnp = [l.split()[1] for l in status.splitlines() if l.startswith("NoNewPrivs:")]
    out["no_new_privs"] = bool(nnp) and nnp[0] == "1"
    uid_map = open("/proc/self/uid_map").read().split()
    out["outside_uid"] = int(uid_map[1]) if len(uid_map) >= 2 else -1
    out["pid_namespace"] = os.getpid() < 10
    import resource
    out["rlimit_as"] = resource.getrlimit(resource.RLIMIT_AS)[0]
    out["rlimit_cpu"] = resource.getrlimit(resource.RLIMIT_CPU)[0]
    import json
    return [float(len(json.dumps(out)))] + [float(ord(c)) for c in json.dumps(out)]
'''


@dataclass
class SandboxConfig:
    cpu_seconds: int = 120  # total CPU seconds for the worker's lifetime
    memory_bytes: int = 1 << 30
    file_bytes: int = 1 << 20
    max_open_files: int = 64
    max_processes: int = 64
    request_timeout: float = 5.0
    startup_timeout: float = 10.0
    max_output_bytes: int = 1 << 20


@dataclass
class IsolationReport:
    available: bool
    checks: dict = field(default_factory=dict)
    reason: str = ""
    mechanism: str = "linux namespaces (user, mount, pid, net, ipc, uts) + pivot_root + capability drop + rlimits"

    def to_dict(self) -> dict:
        return {"available": self.available, "checks": self.checks, "reason": self.reason, "mechanism": self.mechanism}


_probe_cache: IsolationReport | None = None
_lock = threading.Lock()


def _python_path() -> str:
    return os.path.realpath(sys.executable)


def _prereqs() -> str | None:
    if not sys.platform.startswith("linux"):
        return "not Linux"
    for tool in ("unshare", "setpriv"):
        if shutil.which(tool) is None:
            return f"'{tool}' not found"
    if not _python_path().startswith("/usr/"):
        return f"python interpreter {_python_path()} is not under /usr (cannot be exposed read-only)"
    return None


class SandboxProcess:
    """A persistent isolated worker running one candidate's source code."""

    def __init__(self, source: str, entry: str = "mutate", config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        why = _prereqs()
        if why:
            raise IsolationUnavailable(why)
        self.dir = tempfile.mkdtemp(prefix="optiforge-sbx-")
        os.chmod(self.dir, 0o755)
        self.root = os.path.join(self.dir, "root")
        self.runner = os.path.join(self.dir, "runner")
        os.makedirs(self.root)
        os.makedirs(self.runner)
        for d in (self.root, self.runner):
            os.chmod(d, 0o755)
        with open(os.path.join(self.runner, "runner.py"), "w") as fh:
            fh.write(_RUNNER)
        with open(os.path.join(self.dir, "inner.sh"), "w") as fh:
            fh.write(_INNER)
        for f in ("runner/runner.py", "inner.sh"):
            os.chmod(os.path.join(self.dir, f), 0o644)
        self.stderr_path = os.path.join(self.dir, "stderr.txt")
        cmd = []
        as_root = os.geteuid() == 0
        if as_root:
            cmd += ["setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--inh-caps=-all"]
        cmd += ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--net", "--ipc", "--uts",
                "sh", os.path.join(self.dir, "inner.sh"), self.root, self.runner, _python_path()]
        cfg = self.config

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (cfg.memory_bytes, cfg.memory_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (cfg.file_bytes, cfg.file_bytes))
            resource.setrlimit(resource.RLIMIT_NOFILE, (cfg.max_open_files, cfg.max_open_files))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if as_root:
                resource.setrlimit(resource.RLIMIT_NPROC, (cfg.max_processes, cfg.max_processes))
            os.setsid()

        self._err = open(self.stderr_path, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, env={},
                                     cwd="/", preexec_fn=limits, close_fds=True)
        self._buf = b""
        self.calls = 0
        resp = self._roundtrip({"source": source, "entry": entry}, cfg.startup_timeout)
        if not resp.get("ok"):
            self.close()
            raise RuntimeError(f"candidate failed to load in sandbox: {resp.get('error')}")

    # ------------------------------------------------------------------
    def _readline(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("sandbox request timed out")
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError("sandbox worker exited: " + self.stderr_tail())
            self._buf += chunk
            if len(self._buf) > self.config.max_output_bytes:
                raise RuntimeError("sandbox output limit exceeded")
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _roundtrip(self, msg: dict, timeout: float) -> dict:
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
            return json.loads(self._readline(timeout))
        except (BrokenPipeError, TimeoutError, RuntimeError, ValueError) as exc:
            self.kill()
            raise RuntimeError(f"sandbox failure: {exc}") from exc

    def call(self, x: list, info: dict, seed: int) -> dict:
        self.calls += 1
        return self._roundtrip({"x": x, "info": info, "seed": int(seed)}, self.config.request_timeout)

    def stderr_tail(self, n: int = 600) -> str:
        try:
            self._err.flush()
            with open(self.stderr_path, "rb") as fh:
                return fh.read()[-n:].decode(errors="replace")
        except OSError:
            return ""

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"cmd": "exit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=1)
            except Exception:
                pass
        self.kill()
        try:
            self._err.close()
        except Exception:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def __del__(self):  # best effort
        try:
            if getattr(self, "proc", None) is not None and self.proc.poll() is None:
                self.kill()
        except Exception:
            pass


def probe(force: bool = False) -> IsolationReport:
    """Start a sandbox and verify each isolation property from the inside."""
    global _probe_cache
    with _lock:
        if _probe_cache is not None and not force:
            return _probe_cache
        why = _prereqs()
        if why:
            _probe_cache = IsolationReport(False, {}, why)
            return _probe_cache
        try:
            sb = SandboxProcess(_PROBE, config=SandboxConfig(request_timeout=10))
            try:
                resp = sb.call([0.0], {}, 0)
            finally:
                sb.close()
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error"))
            y = resp["y"]
            checks = json.loads("".join(chr(int(c)) for c in y[1:]))
        except Exception as exc:
            _probe_cache = IsolationReport(False, {}, f"sandbox probe failed: {exc}")
            return _probe_cache
        required = ["network_blocked", "usr_readonly", "host_paths_hidden", "env_empty", "no_capabilities",
                    "no_new_privs", "pid_namespace"]
        failed = [k for k in required if not checks.get(k)]
        if checks.get("outside_uid") == 0:
            failed.append("outside_uid_is_root")
        if checks.get("rlimit_as", -1) in (-1, resource.RLIM_INFINITY):
            failed.append("no_memory_limit")
        ok = not failed
        _probe_cache = IsolationReport(ok, checks, "" if ok else f"isolation checks failed: {failed}")
        return _probe_cache
