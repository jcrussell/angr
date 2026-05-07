# Loop session notes (2026-05-07, 112th loop session)

## Task: angr-ygjh — State plugin coverage: callstack, libc, trace (DONE)

### What landed (commit 7f769212d)

- New `_sync_rust_callstack_to_state(state, state_id)` in
  `angr/exploration/rust_state_export.py` (next to the existing
  register/memory sync helpers).
- Wired into all four paths inside `_get_stash_states`:
  cached fast-path (L207), parent-state copy (L239), stepping-state
  copy (L262), snapshot fallback (L284).
- Reads `self._rust_mgr.get_state_call_stack(state_id)` (push-order:
  outermost first), builds an angr CallStack linked list bottom-up so
  the head is the most-recent Rust call, then
  `state.register_plugin("callstack", chain)`.
- New test `TestCallStackProxy::test_simstate_callstack_synced_after_explore`
  asserts `mgr.found[0].callstack.func_addr == raw[-1].callee_addr`
  (top frame matches the most recent Rust call).

### Why this was the right scope

The callstack proxy work from session 111 (angr-nnov) gave callers
`state.callstack` on RustStateProxy. But `mgr.found` returns full angr
SimStates, and those still had only the entry-state's empty CallStack.
This session closes the gap on the angr SimState path. libc / trace
plugin coverage was deprioritized in the bead body ("can wait"), so the
callstack-only acceptance criterion is what was needed.

### Test results

259/259 passing in test_rust_exploration.py (was 258 before the new test).

### Memories saved

- `invariant-callstack-sync-export-pipeline` — there are four export
  paths in `_get_stash_states`; any per-state sync helper must hit all
  four.
- `callstack-rebuild-pattern` — exact algorithm to rebuild an angr
  CallStack chain from a push-order Rust frames list, including the
  `next_frame` ordering rule and `register_plugin` install.
- `venv-binaries-can-disappear` — recovery procedure when `.venv/bin`
  vanishes between sessions while site-packages survives.

### Setup hiccup

`.venv/` was missing `bin/`, `include/`, and `pyvenv.cfg` at session
start (only `lib/` survived). Recovered by copying a fresh
`python3 -m venv` reference's bin/include and writing `pyvenv.cfg`
manually. Site-packages (angr editable, claripy, z3-solver) were intact.

### Next session candidates

P1 / P2 ready:
- angr-pufm (P1) symbolic concretization fallback — multi-session,
  needs splitting per audit notes.
- angr-prem (P2) MemoryLayer trait refactor.
- angr-fk0m (P2) unify state mixin classes.
- angr-4j5u (P2) decompose 95-field god struct.
- angr-m2hf (P2) unified error trait.
- angr-wqao (P2) split rust_manager.py 2800 lines.

P3 well-bounded:
- angr-cmy1 — register_procedure() PyO3 API (Python callable shim).
- angr-3vrj — StateMetadata dataclass (56 call sites, mostly
  mechanical).
- angr-x3xu — SolverBridge protocol (Python-only).
