## Session log: 2026-05-17 — angr-gffd closed (symbolic syscall fix)

### Status: CLOSED

### Task

**angr-gffd (P3, bug)** — Fix silent symbolic-syscall-num divergence + honor
NO_SYMBOLIC_SYSCALL_RESOLUTION.

### Root cause

`CallbackInterpreter::get_syscall_num` in `interpreter_cb/exits.rs` was
returning `.as_u64().unwrap_or(0)` — collapsing a symbolic syscall register
to a concrete `0`. On amd64, syscall 0 is `read` (NativeReadSyscall), so
any symbolic `rax` at a `syscall` instruction silently dispatched to
NativeReadSyscall with whatever happened to be in `rdi/rsi/rdx`.

Python's `engines/successors.py::_resolve_syscall` (line 343-367) handles
symbolic syscall numbers by enumeration (default) or short-circuit to the
unknown-syscall stub when `NO_SYMBOLIC_SYSCALL_RESOLUTION` is set. Rust
never reached that logic — the silent unwrap_or(0) preempted it.

### Resolution

- `get_syscall_num` returns `Option<u64>` (None for symbolic register OR
  arch without syscall_num_offset).
- Propagated `Option<u64>` through `BlockResult::Syscall`,
  `RunResult::Syscall`, `CallbackReason::Syscall`, and
  `ExplorationEvent::need_syscall` / `LoopExecutionEvent::syscall_num`
  (which were already `Option<u64>`).
- `stepping.rs::RunResult::Syscall` skips the native syscall registry
  when `num` is None and forces the Python callback. Python's
  `_handle_syscall_callback` calls `engine.process(state)`, which reads
  the still-symbolic register through `_resolve_syscall` and either
  enumerates (default) or routes to the unknown-syscall stub
  (NO_SYMBOLIC_SYSCALL_RESOLUTION).

### Files modified

- `native/angr/src/interpreter_cb/exits.rs` — `get_syscall_num` returns
  `Option<u64>`; tests updated (3 changes).
- `native/angr/src/interpreter_cb/mod.rs` — `BlockResult::Syscall::num` ->
  `Option<u64>`.
- `native/angr/src/callbacks.rs` — `RunResult::Syscall::num` ->
  `Option<u64>`; `LoopExecutionEvent` conversion passes through.
- `native/angr/src/exploration/mod.rs` — `CallbackReason::Syscall::num` ->
  `Option<u64>`; `need_syscall` accepts `Option<u64>`.
- `native/angr/src/exploration/stepping.rs` — `and_then` to gate native
  dispatch on concrete num.
- `docs/advanced-topics/rust_engine.rst` — bumped Overview + matrix.
- `tests/engines/test_rust_exploration.py` — 2 new tests:
  - `test_symbolic_syscall_num_forces_python_fallback`
  - `test_concrete_syscall_num_still_uses_native_dispatch`

### Verification

- `cargo check --release`: clean.
- Rust unit tests: 11 in `interpreter_cb::exits::tests` all pass (1 renamed:
  `get_syscall_num_returns_zero_for_symbolic_rax` →
  `get_syscall_num_returns_none_for_symbolic_rax`).
- `tests/engines/test_rust_exploration.py`: 448/448 pass (was 446, +2 new).
- `tests/engines/test_rust_integration.py`: 22 pass, 2 xfailed (unchanged).

### Memories planned

- `invariant-rust-honored-simoptions` — append `NO_SYMBOLIC_SYSCALL_RESOLUTION`.
- `gffd-root-cause` — record `get_syscall_num`'s `unwrap_or(0)` silent
  divergence (symbolic-rax-→-read on amd64).
- `invariant-syscall-num-symbolic-routes-python` — invariant.

### Followup ideas

Same candidates as previous session's followup list still viable
(ZERO_FILL_UNCONSTRAINED_REGISTERS is the next paired SimOption — note
the current Rust default is zero-fill via `RegisterFile::new`'s
`vec![0; size]`, so an audit is needed to decide what "honor" should
really mean here).
