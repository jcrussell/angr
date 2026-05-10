# Rust Engine `state.inspect` Limitation

The Rust symbolic-execution engine (`RustExplorationManager`) does **not**
dispatch `state.inspect` breakpoints. Any code that registers an inspect
hook on a `RustStateProxy` will raise `NotImplementedError` at registration
time so that the failure is loud.

## Affected API

All five registration methods on `state.inspect` raise:

- `state.inspect.b(event, when=..., action=...)`
- `state.inspect.make_breakpoint(event, ...)`
- `state.inspect.add_breakpoint(event, bp)`
- `state.inspect.remove_breakpoint(event, idx_or_bp)`
- `state.inspect.action(event, action)`

Inspect events covered (and therefore unsupported) include `mem_read`,
`mem_write`, `reg_read`, `reg_write`, `tmp_read`, `tmp_write`,
`address_concretization`, `expr`, `statement`, `instruction`, `irsb`,
`constraints`, `exit`, `fork`, `symbolic_variable`, `simprocedure`,
`engine_process`, `path_step`, `dirty`, and `syscall`.

## Why it raises instead of warning

`_NoOpInspectProxy` previously silently accepted breakpoint registration
but never fired any callback (see angr-osuu). That caused taint-tracking
and analyzer techniques to appear to work but produce wrong results.
Raising `NotImplementedError` at registration prevents users from
unknowingly depending on a feature the engine cannot fulfill.

## Workaround

Drop back to the Python engine for analyses that rely on `state.inspect`:

```python
import angr

proj = angr.Project("/path/to/binary", auto_load_libs=False)
state = proj.factory.entry_state()

# Python engine (default) — state.inspect works normally
mgr = proj.factory.simulation_manager(state)
state.inspect.b("mem_read", when=angr.BP_BEFORE, action=my_callback)
mgr.explore(find=0x401234)
```

If only part of an exploration needs inspect, run the Rust engine first
to reach an interesting region and then continue with the Python engine
from the resulting state(s) — `mgr.found[i]` returns full `SimState`
objects that can seed a Python `SimulationManager`.

## Why not implement breakpoint plumbing in Rust?

The Rust `InspectionManager` (in `native/angr/src/state.rs`) already has a
ring buffer and per-event bitmask scaffold (commit a07a999f4), but events
are not auto-fired from the interpreter loop. Adding dispatch would
require:

1. Per-event call sites scattered throughout the VEX interpreter, memory
   model, solver, and exit handlers.
2. A marshalling path that turns Rust events into Python callable
   invocations on the right `SimState`-equivalent object (today the
   callback API receives a `SimState`, not a `RustStateProxy`).
3. Care about reentrancy: an `action` callback that mutates state must
   round-trip through the Rust state without breaking interpreter
   invariants.

That is multi-session work that needs design first. Until and unless
that demand materializes, the documented limitation is the contract.

## Decision history

- `angr-osuu` (2026-05-08): replaced silent no-op with
  `NotImplementedError` on registration.
- `angr-mq8l` (2026-05-10): formalized the limitation (Option B), pointed
  the error message at this document, and updated CLAUDE.md.
