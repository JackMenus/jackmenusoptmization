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
6. Once the runner has read its first message, and before the candidate source is compiled, the runner loads a
   seccomp-BPF filter with `prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER)` (pure `ctypes`, no extra dependency). The
   filter is an allowlist of about 60 x86_64 system calls: file reads and writes, `openat`/`stat`, memory
   management, futex, clocks, `getrandom`, signal handling for the worker itself and exit. All other calls fail with
   `EPERM` (`ioctl` with `ENOTTY`). That includes `clone`/`fork`/`vfork`, `execve`, every socket call, `ptrace`,
   `kill`/`tgkill`, `mount`, `unshare`/`setns`, `bpf`, `keyctl`, `io_uring`, `perf_event_open` and `prctl`. A system
   call made through a foreign ABI (i386 or x32) kills the worker. The filter cannot be removed: seccomp filters only
   accumulate, and `prctl` is itself denied.
7. The parent exchanges JSON lines with the worker and validates every returned vector (shape, finiteness, domain).
   Each request carries a sequence id that the reply must echo. A reply that is not a JSON object, or that answers
   a different request (for example because the candidate wrote to the protocol channel itself), kills the worker.
   The worker never receives the objective or the problem.
8. A code operator restarts a dead or killed worker at most 3 times (each worker has its own CPU rlimit); after that
   the candidate is disabled for the rest of the run.

## Verification

`optiforge sandbox-check` runs a probe program inside the sandbox and requires all of the following:
`network_blocked`, `usr_readonly`, `host_paths_hidden`, `env_empty`, `no_capabilities`, `no_new_privs`,
`pid_namespace`, `seccomp_filter_active` (`Seccomp: 2` in `/proc/self/status`), `fork_blocked`, `exec_blocked`,
`socket_blocked`, a non-root uid outside the namespace, and a finite address-space limit. If any check fails,
generated Python is not executed.

Set `OPTIFORGE_REQUIRE_ISOLATION=1` when running the tests on a host where isolation is expected: the isolation
tests then fail instead of being skipped when the probe fails.

`tests/test_isolation.py` runs hostile candidates and checks that each attempt fails: reading this test file and
`/etc/shadow`, writing to `/usr` or the runner, opening a network connection, reading an injected
`ANTHROPIC_API_KEY`, importing the platform, an infinite loop (timeout), allocating 3 GiB (memory limit), returning
a 400k-element vector (output cap), and raising or failing to load (reported as an error). Since 0.1.1 it also
checks that forking, exec, `subprocess`, `os.system`, Unix and netlink sockets, threads, `kill`, `ptrace`,
`unshare` and `prctl` fail, that a forged or malformed reply kills the worker, and that worker restarts are
capped.

## Limitations

* The seccomp allowlist still admits file, memory, futex and clock system calls, so isolation depends on the
  kernel's implementation of those calls, of namespaces and of seccomp itself. It is much stronger than an AST
  filter or a plain subprocess, which this project does not describe as sandboxes, but weaker than a virtual
  machine or gVisor.
* The filter is defined for x86_64 only. On other architectures (for example aarch64) the probe fails and code
  generation is switched off.
* Inside the sandbox the candidate can still write up to 4 MiB to its private `/tmp`, and it has no numpy (the
  sandbox interpreter is the system Python with the standard library only).
* It is Linux only. Unprivileged user namespaces must be enabled, and the interpreter must live under `/usr`.
* The CPU rlimit is a lifetime budget for each persistent worker. A worker that dies is restarted by the next call
  (at most 3 times), and each failure is counted against the candidate.
* `RLIMIT_NPROC` is only set when the platform runs as root, because without root it would count every process of
  the invoking user. Process creation is blocked by the seccomp filter in either case.
