# Changelog

## Unreleased

### Security
* Generated Python operators now run under a seccomp-BPF system-call allowlist (x86_64) loaded inside the sandbox
  before the candidate is compiled. Process creation, exec, sockets, ptrace, signals to other processes, mount and
  namespace calls, bpf, keyctl, io_uring and prctl are denied. `optiforge sandbox-check` requires the new checks
  `seccomp_filter_active`, `fork_blocked`, `exec_blocked` and `socket_blocked`; if any fails, generated code is
  disabled. On non-x86_64 hosts generated code is now disabled (previously it ran without a filter).
* Sandbox replies must echo the request id and be JSON objects; otherwise the worker is killed (previously a
  non-object reply leaked the worker process, and extra lines desynchronised later replies).
* A code operator restarts its sandbox worker at most 3 times, so the per-worker CPU rlimit cannot be renewed
  indefinitely. Failed workers are now closed (their temporary directories were leaked before).
* `OPTIFORGE_REQUIRE_ISOLATION=1` makes the isolation tests fail rather than skip when the probe fails.
