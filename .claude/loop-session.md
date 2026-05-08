# Loop session notes (2026-05-08, 142nd loop session)

## Task: angr-js62 — Python-level integration tests for symbolic libc procedures (closed)

### Status: complete; closed

### Summary
Added 10 Python-level integration tests under
`TestSymbolicLibcProcedures` in `tests/engines/test_rust_exploration.py`.
Native Rust unit tests in `native/angr/src/procedures/*.rs` exercise each
libc procedure directly on a `RustSimState`; these new tests cover the
surrounding integration path that the Rust tests do not — claripy↔Z3
bridging, ITE-chain construction across the FFI boundary, callback
dispatch + name-matching from `proj._sim_procedures`, and final rax
export back to the Python `SimState`.

Coverage: strlen, strcmp, strchr, memchr, memcmp, atoi, strtol, strtoul,
isdigit, isalpha.

### Approach
Each test hooks fauxware's `.fini` area (0x4008c0, mapped but not
normally executed) with a Python angr SimProcedure, sets up registers +
memory with symbolic input, runs `mgr.run(max_steps=1)`, and asserts
solver min/max on rax.

### Key pitfall during prototyping
`proj.factory.blank_state(addr=X)` where X is not inside
`find_object_containing(X) → real binary` triggers
`_run_python_init_if_needed` which then runs Python init to main,
ignoring the hook entirely. Must hook an addr inside the actual fauxware
mapped segments (0x400000-0x400a74). Discovered after seeing concrete
strlen("hello") return 8 instead of 5 — the Rust manager was running
fauxware's main, never our hook.

### Verification
- `pytest tests/engines/test_rust_exploration.py` — **298/298 passing**
  (was 288 before; +10 new integration tests).

### Memories saved
- `invariant-libc-test-hook-addr` — hook must be inside a real loaded
  binary segment (fauxware 0x400000-0x400a74), not in cle## loader
  objects (TLS, kernel) or unmapped gaps.
- `pattern-libc-symbolic-integration-test` — full setup recipe for the
  test pattern (hook addr, blank_state with ZERO_FILL, BUF_ADDR=0x601100,
  RET_ADDR=0x4008b0, register layout, max_steps=1, assertion style).

### Files changed
- tests/engines/test_rust_exploration.py (+247 lines, +1 new test class)

### Unblocks
None directly; this is documentation/coverage hardening for future
work on libc procedures (e.g., angr-xg0o file-descriptor procedures).
