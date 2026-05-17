## Session log: 2026-05-17 — angr-yl5n Implement NO_IP_CONCRETIZATION in Rust engine

### Status: CLOSED — commit 332b5fcba. 442/442 pass (+2 net tests).

### Task

**angr-yl5n (P2, feature)** — Wire NO_IP_CONCRETIZATION SimOption to Rust engine.
Previously silently ignored — symbolic jump targets at block boundaries were
always enumerated by the AddressConcretizer. Python's behavior
(engines/successors.py:292-296): when the option is set, max_targets is forced
to 0 with skip_max_targets_warning=True, routing the state to
unconstrained_successors without warning.

### What landed

Rust side:
- `RustSimState::no_ip_concretization` field + `set_no_ip_concretization` /
  `no_ip_concretization` accessors (`native/angr/src/state.rs`). Propagated
  through all five fork sites (fork, fork_true, fork_false, fork_from_snapshot,
  merge) and all three constructors.
- `CallbackInterpreter::no_ip_concretization` field
  (`native/angr/src/interpreter_cb/mod.rs`), copied on interpreter fork.
- `eval_next_addr_concretized` (`interpreter_cb/exits.rs`) short-circuits:
  after the concrete fast-path, if `next_val.is_symbolic()` AND the flag is
  set, returns `ConcretizedJump::TooMany{0,0,0}` → `BlockResult::UnconstrainedJump`
  → `StepError::Unconstrained` → unconstrained stash. No solver enumeration.
- `run_interpreter_step` (`stepping.rs`) propagates state's flag to the
  interpreter alongside lazy_solves.
- `RustExplorationManager::state_no_ip_concretization(state_id)` accessor
  (`exploration/mod.rs` + `state_api.rs`) for Python visibility.

Python side:
- `rust_manager.py::_add_rust_state` calls
  `rust_state.set_no_ip_concretization(True)` when
  `o.NO_IP_CONCRETIZATION in state.options`.
- `_apply_state_metadata`'s allow-list extended to include
  `o.NO_IP_CONCRETIZATION` (add+discard mirror so cache reuse cannot leak
  or drop the option).

Tests (+2 net):
- `test_no_ip_concretization_propagates_to_rust` — mirror of
  test_enable_nx_propagates_to_rust; verifies the option-to-flag wiring.
- `test_no_ip_concretization_routes_symbolic_jump_to_unconstrained` —
  end-to-end on a shellcode `jmp rax` (rax = unconstrained-symbolic) under
  `save_unconstrained=True`. With the option, the state lands in
  unconstrained instead of enumerating.

Docs (`docs/advanced-topics/rust_engine.rst`):
- Overview "Honored options" list now includes NO_IP_CONCRETIZATION.
- Honored matrix entry describing the routing and pointing at
  `eval_next_addr_concretized`.
- Implement-table entry flipped to "Honored as of 2026-05-17 (angr-yl5n)".

### Memories saved

- `invariant-rust-honored-simoptions` — updated to 7 honored options
  (added NO_IP_CONCRETIZATION). Documents wiring path and access pattern.
- `invariant-no-ip-concretization-routing` — Python semantics
  (max_targets=0 + skip_max_targets_warning) and Rust mirror; the
  short-circuit fires for any symbolic IP BEFORE ITE fast-path and
  AddressConcretizer, matching Python's upstream `if next_val.is_symbolic()`.
  Default exit only; mid-block Exit falls back to Python on symbolic
  addresses before the flag would matter.

### Files modified

- native/angr/src/state.rs (+41)
- native/angr/src/interpreter_cb/mod.rs (+7)
- native/angr/src/interpreter_cb/exits.rs (+12)
- native/angr/src/exploration/stepping.rs (+3)
- native/angr/src/exploration/mod.rs (+7)
- native/angr/src/exploration/state_api.rs (+4)
- angr/exploration/rust_manager.py (+12/-4)
- tests/engines/test_rust_exploration.py (+54)
- docs/advanced-topics/rust_engine.rst (+15/-3)

### Followup work for next session

Sister task still open (P2):
- angr-ph9z — KEEP_IP_SYMBOLIC. **Out of scope for this session.** Real gap
  per docs: Rust always concretizes IP at block boundaries because the engine
  drives PC from a `u64`. Implementing this requires teaching `set_pc` /
  `run_interpreter_step` to optionally keep the IP register set to a symbolic
  expression while still advancing the step driver — more invasive than the
  NO_IP_CONCRETIZATION change. Worth a dedicated session.
