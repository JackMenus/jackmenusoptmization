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
