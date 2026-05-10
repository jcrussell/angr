## Session log: 2026-05-10 — angr-mq8l (216th loop session, COMPLETE)

### Task
Decide on state.inspect support and execute. Chose Option B (document as
hard limitation) over Option A (multi-session breakpoint dispatch in the
Rust step loop).

### Background
- `angr-osuu` (closed) already replaced silent no-op with NotImplementedError
  on registration through `_NoOpInspectProxy` (rust_state_proxy.py:776-797).
- Existing error message pointed at `angr-osuu` — a closed bead, not durable.
- The Rust `InspectionManager` scaffold (state.rs, commit a07a999f4) records
  events into a ring buffer but is NOT auto-fired from the interpreter loop.

### Decision: Option B

Rationale:
- Option A would need: per-event call sites scattered through interpreter /
  memory / solver / exit handlers; a marshalling path from Rust events to
  Python callables; reentrancy handling for action callbacks that mutate
  state. Multi-session and design-first.
- Option B was partially in place since angr-osuu. This session completes
  it by writing the formal limitation doc and pointing the error message
  there instead of at the closed bead.

### Files changed

- `docs/RUST_STATE_INSPECT.md` (new, 56 lines) — affected API, workaround,
  why dispatch wasn't implemented, decision history.
- `CLAUDE.md` (+9 lines) — new "state.inspect Unsupported" section,
  positioned right after "SimOption Coverage" to mirror that pattern.
- `angr/exploration/rust_state_proxy.py` (1-line message change) —
  `_INSPECT_NOT_IMPLEMENTED_MSG` now references docs/RUST_STATE_INSPECT.md
  instead of bead `angr-osuu`.
- `tests/engines/test_rust_exploration.py` (5-line regex change) —
  `TestInspectProxy.test_inspect_breakpoint_calls_raise` regex updated
  from `"angr-osuu"` to `"RUST_STATE_INSPECT"`.

### Verification

- `pytest tests/engines/test_rust_exploration.py::TestInspectProxy`: 1 passed.
- `pytest tests/engines/test_rust_exploration.py`: 376 passed, 3 pre-existing
  failed (dcas/pipe/dup2 — same set as session 215). Zero regressions.
- No Rust rebuild needed (Python + docs only).

### Memories saved/updated

- `invariant-rust-inspect-unsupported` (new) — durable record of the
  decision and the rationale for non-dispatch.
- `inspection-system` (updated) — cross-references the new invariant
  memory so future readers see the policy decision alongside the scaffold.

### Closed beads

- `angr-mq8l`.

### Commit

dc7d42866  docs(rust-symex): formalize state.inspect limitation (Option B) — angr-mq8l

### Status

COMPLETE.
