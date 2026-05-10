## Session log: 2026-05-10 — angr-3uye.2 (212th loop session, COMPLETE)

### Task
Implement the unconstrained-PC routing fix in the Rust engine after the
trace bead (angr-3uye.1) localized the divergence to
interpreter_cb/exits.rs:189 (Ijk_Ret to address 0 routed as UnmodeledCall
→ generic skip → deadended, instead of unconstrained).

### Implementation
`native/angr/src/interpreter_cb/exits.rs:182-198` — inserted a guard
before the `!is_in_binary` UnmodeledCall fallback:

```rust
if jumpkind.is_ret() && self.call_stack.is_empty() && !self.is_in_binary(target) {
    return BlockResult::UnconstrainedJump {
        min_target: target, max_target: target,
        limit: self.config.max_symbolic_ip_targets,
        jumpkind,
    };
}
```

The `call_stack.is_empty()` guard is the safety net: normal in-binary
function rets always have a frame pushed by an earlier `Ijk_Call`, so
they keep the existing UnmodeledCall path. Only "ret from a state that
never called anything" — the angr-3uye repro — hits this new branch.

### Verification
- `cargo check --release`: clean (7.18s).
- Rebuild via `tools/rebuild-rust.sh --cargo-only` (venv pip is broken:
  `pip._vendor.resolvelib.structs` import failure; out of scope here).
- Repro: `load_shellcode(b"\xc3"); blank_state(addr=0x1000);
  rsp=0x7fff_0000; mgr.run(max_steps=2)` now lands in
  `{'unconstrained': 1}`, matching Python (was: `{'deadended': 1, addr=0}`).
- Full suite: 370 passed, 3 pre-existing failures (dcas, pipe, dup2 —
  verified pre-existing via `git stash` before/after).
- Fauxware --both: both engines pass (rust 0.28s vs python 0.40s).

### Regression test added
`tests/engines/test_rust_exploration.py::TestUnconstrainedRet::test_ret_with_empty_call_stack_routes_to_unconstrained` —
asserts `len(mgr.unconstrained) == 1` and `len(mgr.deadended) == 0` for
the angr-3uye repro. **Critical**: the test constructs the manager with
`save_unconstrained=True`, because the default `False` drops the
unconstrained stash after every `run()` iteration (rust_manager.py:2867-2875).

### Files changed
- `native/angr/src/interpreter_cb/exits.rs` (+18 lines)
- `tests/engines/test_rust_exploration.py` (+37 lines, new test class)

### Memories saved
- `3uye-2-call-stack-empty-as-unconstrained-trigger` — fix location and
  the call_stack.is_empty() invariant.
- `invariant-save-unconstrained-default-false` — RustExplorationManager
  defaults `save_unconstrained=False` and drops the stash after each
  run — tests/repros must pass `save_unconstrained=True` or the stash
  appears empty even when the underlying routing is correct.

### Closed beads
- `angr-3uye.2` (this fix bead).

### Caveats / followups
- This is a **defensive route** at the exit handler, not a root fix for
  the upstream sync divergence. `rust_state_sync.py:_sync_stack_page`
  still eagerly concretizes the lazy-symbolic stack to zeros (filtered
  out by `_has_user_symbolic_var`). If a future test relies on the
  popped IP carrying the symbolic identity *across* the Python→Rust
  boundary (e.g., constraints added in Python on `mem_7fff*` must be
  visible in Rust), the sync path would still need work. The narrow fix
  here matches Python's observable behaviour for the canonical repro;
  deeper sync preservation is a separate epic.
- Venv pip is broken — used `tools/rebuild-rust.sh --cargo-only`. If
  this recurs, consider filing a bead.

### Status
COMPLETE.
