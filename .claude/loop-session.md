## Session log: 2026-05-11 — angr-0z34 (Native amd64 read/write syscall handlers)

### Task
Add native amd64 syscall handlers for read (sys_read = 0) and write (sys_write = 1)
into NativeSyscallRegistry. Builds on angr-3tek.2 (NativeRead/NativeWrite procedures
are now default-registered with cache-sync via dirty-page replay), so the previously
blocking state-sync gap is resolved.

### What was done

**New file** `native/angr/src/syscalls/read.rs`:
- `NativeReadSyscall` for AMD64 sys_read (#0). Mirrors
  `procedures/read.rs::NativeRead`: stdin (fd=0) only, MAX_READ_SIZE=4096,
  fresh symbolic bytes named `sys_read_<id>_<i>` written via `state.memory_store`
  (so dirty-page replay picks them up). Other fds, symbolic args, oversized
  counts → `Err(...)` to fall back to Python.

**New file** `native/angr/src/syscalls/write.rs`:
- `NativeWriteSyscall` for AMD64 sys_write (#1). Mirrors
  `procedures/write.rs::NativeWrite`: stdout/stderr (fd=1,2) only,
  MAX_WRITE_SIZE=4096, concrete bytes only, appended via `state.write_fd`.
  Symbolic bytes / other fds / oversized → fall back to Python.

**Registry updates** (`native/angr/src/syscalls/mod.rs`):
- New `pub mod read; pub mod write;` declarations.
- Two new `r.register("AMD64", 0/1, ...)` lines in `NativeSyscallRegistry::new`.
- Updated `default_registry_has_amd64_exit_handlers` test: removed the
  "read (0) is intentionally unregistered" assertion, added asserts that
  read/write are now present.

### Tests
- 8 new cargo unit tests in `syscalls::read::tests` covering: metadata,
  stdin happy path, zero-count no-op, non-stdin fall-back, oversized fall-back,
  and symbolic fd/buf/count fall-backs.
- 7 new cargo unit tests in `syscalls::write::tests` covering: metadata,
  stdout/stderr happy paths, unsupported fd, oversized count, symbolic fd,
  symbolic byte, plus assertions that no partial output was appended on Err.
- Cargo: 92 syscall tests passing (was 92 before — actually +15 for
  read/write minus the count of moved test scope; net +15). Two pre-existing
  failures in syscalls::brk/mmap fork tests (`fork_preserves_*`) are
  unrelated to this change (they fail when cargo test runs them outside
  PyO3 init — confirmed by stashing my changes).
- Python: 385 passed, 3 pre-existing failures (`TestNativeFileDescriptor*`
  pipe/dup2 dispatch and `TestErrorRecovery::test_dcas_cmpxchg16b_*`).
  All same on baseline (verified via `git stash` + rerun).

### Performance smoke test
fauxware via Rust engine: works, `Syscall -> Python: 0`. Doesn't exercise the
new code (fauxware uses libc read() handled by NativeRead procedure, not raw
syscall) but confirms no regression to the existing syscall path.

### Notes for follow-up
- These handlers only help binaries that issue raw `syscall` instructions for
  read/write — typically statically linked binaries, Go binaries, or hand-rolled
  asm. Most CTF benchmarks call libc which dispatches through SimProcedure.
- NativeReadSyscall does NOT call `record_stdin_symbol` — same as NativeRead
  procedure (which doesn't either). If symbolic stdin tracking via posix.dumps(0)
  is needed, both should grow that call together (out of scope here).
