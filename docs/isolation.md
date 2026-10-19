# Isolation of generated code

Generated Python is treated as untrusted. It runs only inside `optiforge.isolation.sandbox.SandboxProcess`, and only
after `probe()` has confirmed the isolation properties *from inside* a sandbox. If the probe fails, the platform
disables code generation (`default_providers` adds a note) and also refuses to build code operators
(`SandboxedCodeOperator` raises `IsolationUnavailable`). Autonomous generation then falls back to the declarative
language.

## Mechanism (Linux)

1. When the parent runs as root, the worker first drops to uid and gid 65534 (`nobody`) with no supplementary groups
   and no inheritable capabilities (`setpriv`).
2. `unshare --user --map-root-user --mount --pid --fork --net --ipc --uts` gives the worker new user, mount, PID,
   network, IPC and UTS namespaces. The new network namespace has only a downed loopback interface.
3. The new root is a private 8 MiB tmpfs. It contains `/usr` (a read-only, nosuid, nodev bind, which supplies the
   Python runtime), the runner directory (read-only), a 4 MiB private `/tmp`, `/dev/null` and a fresh `/proc`. A
   `pivot_root` follows and the old root is unmounted lazily. `/home`, `/root`, `/etc`, the project folder, the
   platform source, the registry, the benchmark definitions and the results are therefore not present in the
   worker's filesystem at all.
4. The worker is started with `exec env -i setpriv --bounding-set=-all --inh-caps=-all --no-new-privs python -I`.
   The environment is empty, so no tokens or credentials are inherited, the capability bounding set is empty,
   `no_new_privs` is set, and Python runs in isolated mode.
5. rlimits apply: CPU 120 s for the lifetime of the worker, address space 1 GiB, file size 1 MiB, 64 open files,
   64 processes, and no core dumps. On top of these there is a 5 s wall-clock timeout per request and a 1 MiB cap
   on returned output. The worker is killed as a process group on timeout or protocol violation.
6. The parent exchanges JSON lines with the worker and validates every returned vector (shape, finiteness, domain).
   The worker never receives the objective or the problem.

## Verification

`optiforge sandbox-check` runs a probe program inside the sandbox and requires all of the following:
`network_blocked`, `usr_readonly`, `host_paths_hidden`, `env_empty`, `no_capabilities`, `no_new_privs`,
`pid_namespace`, a non-root uid outside the namespace, and a finite address-space limit. All of them passed on the
delivery host.

`tests/test_isolation.py` runs hostile candidates and checks that each attempt fails: reading this test file and
`/etc/shadow`, writing to `/usr` or the runner, opening a network connection, reading an injected
`ANTHROPIC_API_KEY`, importing the platform, an infinite loop (timeout), allocating 3 GiB (memory limit), returning
a 400k-element vector (output cap), and raising or failing to load (reported as an error).

## Limitations

* There is no seccomp filter, so the kernel's full syscall surface is reachable from inside the namespaces. The
  isolation is only as strong as the kernel's namespace implementation. It is much stronger than an AST filter or a
  plain subprocess, which this project does not describe as sandboxes, but weaker than a virtual machine or gVisor.
* It is Linux only. Unprivileged user namespaces must be enabled, and the interpreter must live under `/usr`.
* The CPU rlimit is a lifetime budget for each persistent worker. A worker that dies is restarted by the next call,
  and that failure is counted against the candidate.
