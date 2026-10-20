"""Operating-system isolation of generated code: properties are checked from inside the sandbox."""
import os

import pytest

from optiforge.isolation.sandbox import SandboxConfig, SandboxProcess, probe

pytestmark = pytest.mark.usefixtures("needs_isolation")

HERE = os.path.abspath(__file__)


def run(src, x=(0.0,), cfg=None):
    sb = SandboxProcess(src, config=cfg or SandboxConfig(request_timeout=3))
    try:
        return sb.call(list(x), {}, 0)
    finally:
        sb.close()


def attempt(body):
    return f'''
def mutate(x, info, rng):
    try:
{chr(10).join("        " + l for l in body.strip().splitlines())}
        return [1.0]
    except BaseException:
        return [0.0]
'''


def test_probe_reports_all_required_properties():
    rep = probe(force=True)
    assert rep.available, rep.reason
    for k in ("network_blocked", "usr_readonly", "host_paths_hidden", "env_empty", "no_capabilities", "no_new_privs",
              "pid_namespace"):
        assert rep.checks[k], k


def test_benign_operator_roundtrip():
    r = run("def mutate(x, info, rng):\n    return [v + 1 for v in x]\n", x=[1.0, 2.0])
    assert r == {"ok": True, "y": [2.0, 3.0]}


def test_cannot_read_host_files():
    r = run(attempt(f"open({HERE!r}).read()"))
    assert r["y"] == [0.0]
    r = run(attempt("open('/etc/shadow').read()"))
    assert r["y"] == [0.0]


def test_cannot_write_usr_or_runner():
    assert run(attempt("open('/usr/evil', 'w').write('x')"))["y"] == [0.0]
    assert run(attempt("open('/runner/runner.py', 'w').write('x')"))["y"] == [0.0]


def test_no_network():
    body = "import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=1)"
    assert run(attempt(body))["y"] == [0.0]


def test_no_inherited_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    # Python's own locale coercion (PEP 538) may set LC_CTYPE inside the sandbox; nothing is inherited.
    src = ("import os\ndef mutate(x, info, rng):\n"
           "    return [float(len([k for k in os.environ if k != 'LC_CTYPE'])), float('ANTHROPIC_API_KEY' in os.environ)]\n")
    assert run(src)["y"] == [0.0, 0.0]


def test_cannot_import_platform_code():
    assert run(attempt("import optiforge"))["y"] == [0.0]


def test_infinite_loop_times_out():
    with pytest.raises(RuntimeError, match="timed out"):
        run("def mutate(x, info, rng):\n    while True:\n        pass\n", cfg=SandboxConfig(request_timeout=1))


def test_memory_limit():
    src = attempt("b = bytearray(3 << 30)")
    assert run(src, cfg=SandboxConfig(request_timeout=5, memory_bytes=1 << 30))["y"] == [0.0]


def test_output_cap():
    src = "def mutate(x, info, rng):\n    return [1.0] * 400000\n"
    with pytest.raises(RuntimeError, match="output limit"):
        run(src, cfg=SandboxConfig(request_timeout=5, max_output_bytes=1 << 16))


def test_exception_is_reported_not_raised():
    r = run("def mutate(x, info, rng):\n    raise ValueError('nope')\n")
    assert not r["ok"] and "ValueError" in r["error"]


def test_load_failure_rejected():
    with pytest.raises(RuntimeError, match="failed to load"):
        SandboxProcess("this is not python")


# --- seccomp system-call filter -------------------------------------------------------------

def test_probe_reports_seccomp_properties():
    rep = probe(force=True)
    for k in ("seccomp_filter_active", "fork_blocked", "exec_blocked", "socket_blocked"):
        assert rep.checks[k], k


def test_cannot_fork_exec_or_spawn():
    assert run(attempt("import os\npid = os.fork()\nif pid == 0:\n    os._exit(0)"))["y"] == [0.0]
    assert run(attempt("import os\nos.execv('/usr/bin/true', ['true'])"))["y"] == [0.0]
    assert run(attempt("import subprocess\nsubprocess.run(['/usr/bin/true'], check=True)"))["y"] == [0.0]
    # os.system cannot raise; it must report failure instead of running the shell
    r = run("import os\ndef mutate(x, info, rng):\n    return [float(os.system('/usr/bin/true') != 0)]\n")
    assert r["y"] == [1.0]


def test_cannot_open_any_socket_or_thread():
    assert run(attempt("import socket\nsocket.socket(socket.AF_UNIX, socket.SOCK_STREAM)"))["y"] == [0.0]
    assert run(attempt("import socket\nsocket.socket(socket.AF_NETLINK, socket.SOCK_RAW)"))["y"] == [0.0]
    assert run(attempt("import threading\nt = threading.Thread(target=print)\nt.start()"))["y"] == [0.0]


def test_cannot_signal_ptrace_or_unshare():
    assert run(attempt("import os, signal\nos.kill(1, signal.SIGKILL)"))["y"] == [0.0]
    body = ("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
            "if libc.ptrace(0, 0, 0, 0) != 0:\n    raise OSError('ptrace refused')")
    assert run(attempt(body))["y"] == [0.0]
    assert run(attempt("import os\nos.unshare(os.CLONE_NEWUSER)"))["y"] == [0.0]


def test_seccomp_cannot_be_lifted_by_candidate():
    # a candidate may add filters (only ever stricter) but prctl itself is outside the allowlist
    body = ("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
            "if libc.prctl(22, 0, 0, 0, 0) != 0:\n    raise OSError('prctl refused')")
    assert run(attempt(body))["y"] == [0.0]


def test_malformed_or_desynchronised_replies_kill_the_worker():
    # extra line written straight to the protocol channel: the next reply no longer matches its request
    src = ("import sys\ndef mutate(x, info, rng):\n"
           "    sys.stdout.write('{\"ok\": true, \"id\": 999, \"y\": [5.0]}\\n')\n    return [1.0]\n")
    sb = SandboxProcess(src, config=SandboxConfig(request_timeout=3))
    try:
        with pytest.raises(RuntimeError, match="desynchronised"):
            sb.call([0.0], {}, 0)
        assert sb.proc.poll() is not None
    finally:
        sb.close()
    src = "import sys\ndef mutate(x, info, rng):\n    sys.stdout.write('[1, 2]\\n')\n    return [1.0]\n"
    sb = SandboxProcess(src, config=SandboxConfig(request_timeout=3))
    try:
        with pytest.raises(RuntimeError, match="malformed reply"):
            sb.call([0.0], {}, 0)
    finally:
        sb.close()


def test_code_operator_restarts_are_capped():
    import numpy as np
    from optiforge.core.errors import CandidateRejected
    from optiforge.core.space import Real, Space
    from optiforge.generation.code_operators import SandboxedCodeOperator
    op = SandboxedCodeOperator("crash", "def mutate(x, info, rng):\n    while True:\n        pass\n", ["real"],
                               config=SandboxConfig(request_timeout=0.5))
    sp = Space([Real("x", -1, 1)])
    rng = np.random.default_rng(0)
    msgs = []
    for _ in range(SandboxedCodeOperator.MAX_RESTARTS + 3):
        with pytest.raises(CandidateRejected) as ei:
            op.transform(np.zeros(1), sp, rng, 1.0)
        msgs.append(str(ei.value))
    op.close()
    assert op.stats["starts"] == SandboxedCodeOperator.MAX_RESTARTS + 1
    assert "candidate disabled" in msgs[-1]


BENIGN = "def mutate(x, info, rng):\n    return [v + 1 for v in x]\n"


def test_forked_child_cannot_use_or_kill_parent_sandbox():
    import multiprocessing as mp
    sb = SandboxProcess(BENIGN)
    try:
        def child(q):
            try:
                sb.call([0.5], {}, 0)
                q.put("call succeeded")
            except RuntimeError as exc:
                q.put("refused" if "inherited" in str(exc) else f"other error: {exc}")
            sb.close()
            sb.kill()
        q = mp.get_context("fork").Queue()
        pr = mp.get_context("fork").Process(target=child, args=(q,))
        pr.start()
        pr.join(20)
        assert q.get(timeout=5) == "refused"
        assert sb.proc.poll() is None  # the parent's worker survived the child's close/kill
        assert sb.call([0.5], {}, 0) == {"ok": True, "y": [1.5]}
    finally:
        sb.close()
