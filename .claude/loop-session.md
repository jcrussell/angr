# Loop session notes (2026-05-08, 143rd loop session)

## Task: angr-xg0o — Native file-descriptor SimProcedures: dup, dup2, pipe (closed)

### Status: complete; closed

### Summary
Added native dup, dup2, pipe SimProcedures alongside the pre-existing
open/close/lseek in `native/angr/src/procedures/fileops.rs`, with new
`FileSystem::dup`, `dup2`, `pipe` methods in `state.rs`. Registered all
three in `NativeProcedureRegistry::new`. Coverage: 9 new cargo unit
tests on the SimProcedures + 1 on FileSystem methods + 2 Python-level
integration tests (pipe, dup2) verifying native dispatch through the
Rust manager.

### FileSystem semantics
- `dup(oldfd)`: clones FileDescriptor (name/position/flags/content) to
  next available fd; returns None if oldfd not open.
- `dup2(oldfd, newfd)`: clones to newfd, bumping `next_fd` past it.
  Same-fd no-op when open.
- `pipe()`: allocates two consecutive fds — read end (`<pipe:r>`,
  ReadOnly) and write end (`<pipe:w>`, WriteOnly). Does NOT model
  write→read data flow (each end has its own content buffer).

### Pipe SimProcedure detail
NativePipe writes the two 32-bit fds out byte-by-byte to pipefd[0..4]
and pipefd[4..8] with arch endianness (`is_little_endian()` on
RustSimState's arch). Returns 0 on success, like POSIX `pipe(2)`.

### Integration test design constraints (key learning)
Two architectural constraints make chained pipe→dup2 in a single
mgr.run hard to test from Python:
 1. Native dispatch only fires when hook addr is NOT in
    `binary_regions` (i.e. not in an executable section of a real
    binary). Python `_load_binary_regions` skips externs, so an
    in-fauxware non-executable section like `.ctors` (0x600e30)
    qualifies — the addr is recognized as a real-binary address (so
    `_run_python_init_if_needed` short-circuits) but native dispatch
    still fires.
 2. There is no public Python API to update an active state's
    registers between mgr.run calls. So we cannot set RDI/RSI for the
    second native procedure call after the first one returns.
 3. After native pipe returns at PIPE_ADDR, the dispatch reads
    return_addr from rust_memory at SP. The Python state's
    `state.memory.store(rsp, RET_ADDR, ...)` does NOT reliably make
    those bytes visible in rust_memory on first run, so the post-call
    state lift errors at PC=0. `active_states_map_memory` after the
    manager is created DOES populate Rust memory correctly.

We split the integration coverage:
 - Cargo: `test_pipe_then_dup2_to_stdin` exercises the full chained
   pipe+dup2 sequence at the FileSystem level (no register state).
 - Python: two separate tests for pipe (creates fds 3,4) and dup2(0, 7)
   (clones stdin to fd 7) — each verifies native dispatch fires
   (`native_procedure_stats`) and the resulting FD layout.

### Verification
- Cargo tests: 16/16 in `procedures::fileops`, 1/1 new
  `state::tests::test_filesystem_dup_dup2_pipe`.
- Python tests: 300/300 passing (was 298; +2 new integration tests).

### Memories saved
None (the Rust dispatch constraints documented above are mostly
captured in the test docstring; the behavior of native procs on
in-binary non-exec hook addrs is reusable knowledge for future libc
procedure tests).

### Files changed
- native/angr/src/state.rs (+92 lines: FileSystem::dup/dup2/pipe + tests)
- native/angr/src/procedures/fileops.rs (+296 lines: 3 procs + 9 tests)
- native/angr/src/procedures/mod.rs (+3 lines: register the 3 new procs)
- tests/engines/test_rust_exploration.py (+133 lines: 2 integration tests)
