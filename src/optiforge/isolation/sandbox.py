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
* restricted, once the runner has started and before the candidate's source is
  compiled, by a seccomp-BPF system-call allowlist (x86_64): no process
  creation or exec, no sockets, no ptrace, mount, namespace, bpf or keyring
  calls, no signals to other processes.

The parent process only exchanges JSON lines with the worker and validates
every value it returns; the worker never sees the objective function.

Limitations (stated, not hidden): the syscall allowlist still admits file,
memory and futex calls, so isolation relies on the kernel's implementation of
those calls, of namespaces and of seccomp.  The filter is defined for x86_64
only; on other architectures the probe fails and generated code is disabled.
This is substantially stronger than an AST filter or a plain subprocess but
weaker than a virtual machine.  If the probe below fails, arbitrary code
execution is disabled and only the declarative DSL is used for autonomous
generation.
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

# x86_64 system calls the worker may make once the candidate is loaded.  Everything else fails with
# EPERM (ioctl with ENOTTY, so isatty() is simply False); a foreign syscall ABI kills the process.
# Process creation (clone/fork/vfork/execve), sockets, ptrace, mount/unshare/setns, bpf, keyctl, kill,
# prctl, io_uring and file creation/removal outside open() are all outside the list.
SECCOMP_ALLOW_X86_64 = {
    "read": 0, "write": 1, "open": 2, "close": 3, "stat": 4, "fstat": 5, "lstat": 6, "poll": 7, "lseek": 8,
    "mmap": 9, "mprotect": 10, "munmap": 11, "brk": 12, "rt_sigaction": 13, "rt_sigprocmask": 14,
    "rt_sigreturn": 15, "pread64": 17, "readv": 19, "writev": 20, "access": 21, "sched_yield": 24,
    "mremap": 25, "madvise": 28, "dup": 32, "dup2": 33, "nanosleep": 35, "getpid": 39, "exit": 60,
    "uname": 63, "fcntl": 72, "getcwd": 79, "readlink": 89, "getrusage": 98, "sysinfo": 99, "times": 100,
    "getuid": 102, "getgid": 104, "geteuid": 107, "getegid": 108, "getppid": 110, "sigaltstack": 131,
    "fstatfs": 138, "arch_prctl": 158, "gettid": 186, "futex": 202, "sched_getaffinity": 204,
    "getdents64": 217, "set_tid_address": 218, "clock_gettime": 228, "clock_getres": 229,
    "clock_nanosleep": 230, "exit_group": 231, "openat": 257, "newfstatat": 262, "readlinkat": 267,
    "set_robust_list": 273, "dup3": 292, "prlimit64": 302, "getrandom": 318, "statx": 332, "rseq": 334,
    "faccessat2": 439,
}
SECCOMP_ERRNO_X86_64 = {16: 25}  # ioctl -> ENOTTY

_RUNNER = r'''
import ctypes, json, math, random, sys

ALLOW = sorted(set(%(allow)r))
ERRNO = %(errno)r


def install_seccomp():
    # Classic BPF filter loaded with prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER).  no_new_privs is already set by
    # setpriv; it is set again here so the filter can never be loaded without it.
    if not sys.platform.startswith("linux") or __import__("platform").machine() != "x86_64":
        raise OSError("seccomp filter is only defined for linux x86_64")
    LD_W_ABS, JEQ_K, RET_K = 0x20, 0x15, 0x06
    ARCH_X86_64, KILL_PROCESS, ALLOW_RET, ERRNO_RET = 0xC000003E, 0x80000000, 0x7FFF0000, 0x00050000
    prog = [(LD_W_ABS, 0, 0, 4), (JEQ_K, 1, 0, ARCH_X86_64), (RET_K, 0, 0, KILL_PROCESS), (LD_W_ABS, 0, 0, 0)]
    special = sorted(ERRNO.items())
    n_tests = len(ALLOW) + len(special)
    # layout after the tests: [RET EPERM (default)] [RET ALLOW] [RET errno_i ...]
    allow_at = len(prog) + n_tests + 1
    for i, nr in enumerate(ALLOW):
        here = len(prog)
        prog.append((JEQ_K, allow_at - here - 1, 0, nr))
    for j, (nr, err) in enumerate(special):
        here = len(prog)
        prog.append((JEQ_K, allow_at + 1 + j - here - 1, 0, nr))
    prog.append((RET_K, 0, 0, ERRNO_RET | 1))  # EPERM
    prog.append((RET_K, 0, 0, ALLOW_RET))
    for nr, err in special:
        prog.append((RET_K, 0, 0, ERRNO_RET | err))
    if any(not 0 <= jt <= 255 for _, jt, _, _ in prog):
        raise OSError("seccomp program jump out of range")

    class SockFilter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    arr = (SockFilter * len(prog))(*[SockFilter(*ins) for ins in prog])
    fprog = SockFprog(len(prog), ctypes.cast(arr, ctypes.POINTER(SockFilter)))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(38, 1, None, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    if libc.prctl(22, 2, ctypes.cast(ctypes.pointer(fprog), ctypes.c_void_p), 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()


def main():
    first = sys.stdin.readline()
    if not first:
        return
    msg = json.loads(first)
    try:
        install_seccomp()
    except BaseException as exc:
        reply({"ok": False, "error": "seccomp: %%s: %%s" %% (type(exc).__name__, exc)}); return
    ns = {"__name__": "candidate", "math": math, "random": random}
    try:
        exec(compile(msg["source"], "<candidate>", "exec"), ns)
        fn = ns[msg.get("entry", "mutate")]
        if not callable(fn):
            raise TypeError("entry point is not callable")
    except BaseException as exc:
        reply({"ok": False, "error": "load: %%s: %%s" %% (type(exc).__name__, exc)}); return
    reply({"ok": True})
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "exit":
            return
        try:
            rng = random.Random(req["seed"])
            y = fn(list(req["x"]), req["info"], rng)
            reply({"ok": True, "id": req.get("id"), "y": list(map(float, y))})
        except BaseException as exc:
            reply({"ok": False, "id": req.get("id"), "error": "%%s: %%s" %% (type(exc).__name__, exc)})
main()
''' % {"allow": sorted(SECCOMP_ALLOW_X86_64.values()), "errno": SECCOMP_ERRNO_X86_64}

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
    sec = [l.split()[1] for l in status.splitlines() if l.startswith("Seccomp:")]
    out["seccomp_filter_active"] = bool(sec) and sec[0] == "2"
    try:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0); out["fork_blocked"] = False
    except OSError:
        out["fork_blocked"] = True
    try:
        os.execv("/usr/bin/true", ["true"]); out["exec_blocked"] = False
    except OSError:
        out["exec_blocked"] = True
    try:
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close(); out["socket_blocked"] = False
    except OSError:
        out["socket_blocked"] = True
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
    mechanism: str = ("linux namespaces (user, mount, pid, net, ipc, uts) + pivot_root + capability drop + rlimits "
                      "+ seccomp syscall allowlist (x86_64)")

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
            resp = json.loads(self._readline(timeout))
            if not isinstance(resp, dict):
                raise ValueError(f"malformed reply of type {type(resp).__name__}")
            if "id" in msg and resp.get("id") != msg["id"]:
                raise ValueError("reply does not answer the current request (protocol desynchronised)")
            resp.pop("id", None)
            return resp
        except (BrokenPipeError, TimeoutError, RuntimeError, ValueError) as exc:
            self.kill()
            raise RuntimeError(f"sandbox failure: {exc}") from exc

    def call(self, x: list, info: dict, seed: int) -> dict:
        self.calls += 1
        return self._roundtrip({"id": self.calls, "x": x, "info": info, "seed": int(seed)}, self.config.request_timeout)

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
                    "no_new_privs", "pid_namespace", "seccomp_filter_active", "fork_blocked", "exec_blocked",
                    "socket_blocked"]
        failed = [k for k in required if not checks.get(k)]
        if checks.get("outside_uid") == 0:
            failed.append("outside_uid_is_root")
        if checks.get("rlimit_as", -1) in (-1, resource.RLIM_INFINITY):
            failed.append("no_memory_limit")
        ok = not failed
        _probe_cache = IsolationReport(ok, checks, "" if ok else f"isolation checks failed: {failed}")
        return _probe_cache
