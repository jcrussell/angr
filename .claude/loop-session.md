## Session log: 2026-05-09 — angr-kwwd SMC Python-lift bytes sync (204th loop session, COMPLETE)

### Task
angr-kwwd (P4): SMC: sync Rust-side stores to Python state.memory for lift_block.
Followup to angr-k67f. Picked option (a): pass dirty bytes from rust_memory
through call_lift_block so _cb_lift_block can lift via byte_string=, avoiding
the stale-cle-binary read.

### What landed (commit 3b0e7690d)
1. `CallbackInterpreter::get_or_lift_block` (execution.rs): before falling
   back to the Python lift callback, if `is_code_range_dirtied(addr, 4096)`
   and rust_memory is present, read up to 4096 bytes via
   `read_concrete_bytes_for_lift` (with hook_addrs clamping) and pass them
   to `call_lift_block` as `dirty_bytes`.
2. `PythonCallbacks::call_lift_block` (callbacks.rs): now takes
   `Option<&[u8]>`. When Some, calls Python with `(addr, opt_level_or_None,
   PyBytes)` so the Python signature stays positional-compatible.
3. `_cb_lift_block` (rust_manager.py): accepts `dirty_bytes: bytes = None`
   kwarg; when set, threads into `factory.block(addr, byte_string=...)`.
4. Three new tests in `TestErrorRecovery`:
   - test_cb_lift_block_uses_dirty_bytes_when_provided — Python contract
   - test_cb_lift_block_static_binary_when_no_dirty_bytes — back-compat
   - test_smc_rust_passes_dirty_bytes_to_python_lift — full Rust→Python e2e
     using the wrap-and-reinstall callback pattern (set_callbacks clones)

### Tests
- 369/369 Python tests pass (was 366; added 3).
- 611/611 cargo unit tests pass.

### Findings saved (bd remember)
- smc-python-lift-bytes-channel — both halves of SMC support (k67f + kwwd)
- invariant-shellcode-binary-regions-test — load_shellcode + is_in_binary
- invariant-set-callbacks-clones — PythonCallbacks Clone semantics

### Files modified
- native/angr/src/callbacks.rs — call_lift_block signature
- native/angr/src/interpreter_cb/execution.rs — dirty-bytes resolve before Python lift
- angr/exploration/rust_manager.py — _cb_lift_block dirty_bytes kwarg
- tests/engines/test_rust_exploration.py — 3 new tests

### Status
COMPLETE. SMC binaries on default builds now execute post-store
instructions correctly. Native-lift builds were already covered by
angr-k67f; this closes the parity gap.
