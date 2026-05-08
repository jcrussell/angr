# Loop session notes (2026-05-08, 136th loop session)

## Task: angr-osuu — state.inspect.b silently no-ops — break API loudly

### Status: complete; closed after commit ee97a1810

### Change
Replaced _NoOpInspectProxy silent no-ops at angr/exploration/rust_state_proxy.py:555
with NotImplementedError carrying a clear "angr-osuu" pointer message.
b / make_breakpoint / add_breakpoint / remove_breakpoint / action all raise.
Removed the catch-all __getattr__ that returned no-op callables for arbitrary
attrs — unsupported paths now surface as AttributeError or NotImplementedError.

Updated TestInspectProxy regression (test_inspect_breakpoint_calls_succeed →
test_inspect_breakpoint_calls_raise) to assert the loud behaviour.

### Verification
262/262 tests passing (pure Python change; no Rust rebuild needed).

### Memory saved
- invariant-rust-engine-loud-failures: prefer NotImplementedError over silent
  no-ops when a Python feature is unsupported by the Rust symex engine,
  citing the breakpoint dispatch bug as motivation.

### Files changed
- angr/exploration/rust_state_proxy.py (+15 -19 logical)
- tests/engines/test_rust_exploration.py (+18 -10 logical)

### Caveat for future work
Inspection events ARE supported in Rust (see InspectionManager in state.rs,
memory `inspection-system`) — bitmask + ring buffer of records — but events
are NOT auto-fired from the interpreter loop, and the dispatcher is not
wired through to Python BP callbacks. Implementing real breakpoint dispatch
(option (a) in angr-osuu) is the next step if real demand emerges; track
in a separate bead when needed.
