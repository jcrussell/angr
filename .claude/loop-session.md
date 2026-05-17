## Session log: 2026-05-17 — angr-ph9z Implement KEEP_IP_SYMBOLIC in Rust engine

### Status: CLOSED — commit 786ca0b26. 444/444 pass (+2 net tests).

### Task

**angr-ph9z (P2, feature)** — Wire KEEP_IP_SYMBOLIC SimOption to Rust engine.
Sister task to angr-yl5n (NO_IP_CONCRETIZATION) which landed previously.

### Why this was tractable (despite prior session saying "more invasive")

The prior session's followup note worried KEEP_IP_SYMBOLIC was too invasive
because the engine "drives PC from a u64." That turns out to be *exactly*
why it works — `RustSimState.pc` is a separate `u64` field from the IP
register. `state.set_pc(addr)` writes both, but `state.set_ip(symbolic_bv)`
writes only the register and leaves `pc` alone for Expression-variant BVs
(as_u64 returns None). So the chain `set_pc(concrete_addr)` followed by
`set_ip(symbolic_expr)` cleanly separates "what drives the next block lift"
from "what the IP register reads as."

### What landed

Rust side (mirrors NO_IP_CONCRETIZATION wiring):
- `RustSimState::keep_ip_symbolic` field + `set_keep_ip_symbolic` /
  `keep_ip_symbolic` accessors (`native/angr/src/state.rs`). Propagated
  through all five fork sites (fork, fork_true, fork_false,
  fork_from_snapshot, merge) and all three constructors.
- `CallbackInterpreter::keep_ip_symbolic` field +
  `symbolic_ip_at_exit: Option<RustBV>` slot + `take_symbolic_ip_at_exit()`
  helper (`native/angr/src/interpreter_cb/mod.rs`).
- `eval_next_addr_concretized` (`interpreter_cb/exits.rs`):
  - AddressConcretizer Single branch: when `keep_ip_symbolic`, skip
    `assume_true(next_val == addr)` and stash `next_val` in
    `symbolic_ip_at_exit`.
  - ITE fast-path Single branch: same stash (preserves the original ITE
    expression).
- `step_state_with_skip` (`stepping.rs`): after `state.set_pc(new_pc)`, if
  `Some(sym_ip) = step.symbolic_ip_at_exit`, call `state.set_ip(sym_ip)` to
  overwrite the IP register with the symbolic AST.
- `handle_symbolic_jump_target` (multi-target): branch on
  `state.keep_ip_symbolic()` — skip per-fork `add_constraint(target == addr)`
  and call `forked.set_ip(target_expr.clone())` after `forked.set_pc(addr)`.
- `run_interpreter_step` propagates `state.keep_ip_symbolic()` to the
  interpreter alongside `no_ip_concretization`.
- `RustExplorationManager::state_keep_ip_symbolic(state_id)` accessor
  (`exploration/mod.rs` + `state_api.rs`) for Python visibility.

Python side:
- `rust_manager.py::_add_rust_state`: when `o.KEEP_IP_SYMBOLIC ∈
  state.options`, call `rust_state.set_keep_ip_symbolic(True)`.
- `_apply_state_metadata` allow-list extended to include
  `o.KEEP_IP_SYMBOLIC` (add+discard mirror).

Tests (+2 net, 442 → 444):
- `test_keep_ip_symbolic_propagates_to_rust` — mirror of the
  NO_IP_CONCRETIZATION propagation test.
- `test_keep_ip_symbolic_leaves_ip_register_symbolic_after_fork` — uses
  `mgr.step(1)` (not `run(max_steps=1)` — the latter would step into the
  zero-padded region past the targets and the forks would wash out of the
  active stash). Baseline returns concrete rip for each fork via
  `get_state_register("rip")`; with KEEP_IP_SYMBOLIC the same call returns
  None (symbolic).

Docs (`docs/advanced-topics/rust_engine.rst`):
- Overview "Honored options" list now includes KEEP_IP_SYMBOLIC.
- Honored matrix entry describing the routing through `symbolic_ip_at_exit`
  and `handle_symbolic_jump_target`.
- Suggested-fix table entry flipped to "Honored as of 2026-05-17 (angr-ph9z)".

### Memories saved

- `invariant-rust-honored-simoptions` — updated to 8 honored options (added
  KEEP_IP_SYMBOLIC). Wiring path notes for both new options.
- `invariant-keep-ip-symbolic-routing` — semantics (skip narrowing
  constraint, leave IP register symbolic), why Rust's u64 `pc` field makes
  this clean (as_u64() returns None for Expression BVs so set_ip doesn't
  re-concretize the pc field), where each case is handled (interpreter Single
  vs manager multi-target).

### Files modified

- native/angr/src/state.rs (+38)
- native/angr/src/interpreter_cb/mod.rs (+25)
- native/angr/src/interpreter_cb/exits.rs (+24)
- native/angr/src/exploration/stepping.rs (+45/-6)
- native/angr/src/exploration/mod.rs (+9)
- native/angr/src/exploration/state_api.rs (+4)
- angr/exploration/rust_manager.py (+13/-4)
- tests/engines/test_rust_exploration.py (+94)
- docs/advanced-topics/rust_engine.rst (+19/-3)

### Followup work for next session

`bd ready` after this close:
- angr-kcf.1 (P2) — codegate angrybird BFS triage. Research-shaped task.
- angr-fk0m (P2) — refactor sync/cache/export mixins. Notes already say
  "purely cosmetic", deferred multiple times. Probably skip.
- angr-prem (P2) / angr-prem.1 (P3) — MemoryLayer trait. Architecture work.
- angr-34w.12 (P2) — grub OOM. Big.

No immediate followup needed for the KEEP_IP_SYMBOLIC implementation itself.
