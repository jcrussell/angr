# Loop session notes (2026-05-07, 101st loop session)

## Task: angr-3zs6 — Define FallbackStrategy enum (CLOSED)

Bead closed (commit 136eaec91).

### What changed

Added `FallbackStrategy` enum and `CbExecutionError::strategy()` method to
`native/angr/src/interpreter_cb/mod.rs`. Refactored
`execution.rs::run_until_event` to dispatch via `.strategy()` instead of
matching on variant names.

- `FallbackStrategy::PythonCallback` — block hands off to Python VEX engine
  (RunResult::NeedPythonVEX). Used for `Unsupported`, `NeedPythonFallback`.
- `FallbackStrategy::Panic` — surfaces as `RunResult::Error`, state errored
  stash. Used for `Memory`, `Op`, `InvalidIR`, `TypeMismatch`, `UnknownTemp`,
  `Callback`, `LiftError`.
- `FallbackStrategy::Silent` — reserved (`#[allow(dead_code)]`). No current
  variant uses it. Documents that the interpreter's silent-substitution
  paths (or_else fallbacks in expressions.rs unop/binop/triop/qop) return
  `Ok(...)` and don't reach the dispatcher.

The match in `strategy()` is exhaustive so adding a CbExecutionError variant
forces an explicit policy decision. Each variant carries a doc comment
referencing its strategy.

### Behavior contract

Unchanged. `Unsupported` and `NeedPythonFallback` still map to NeedPythonVEX;
everything else to Error.

### Verification

- `cargo check`: clean.
- `cargo build --release`: clean (had to copy
  `target/release/librustylib.so` → `angr/rustylib.cpython-312-x86_64-linux-gnu.so`
  manually because the venv pip install fails — see `z3-header-fallback` memory).
- pytest tests/engines/test_rust_exploration.py → 254/254 passed.

### Memories saved

- `invariant-fallback-strategy-enum` — captures the new enum, dispatcher
  contract, and where silent fallbacks live.
- `z3-header-fallback` — Z3_SYS_Z3_HEADER=/usr/include/z3.h workaround for
  the .venv z3 package missing include/, plus the cargo→manual-copy bypass
  for the broken pip install.

## Bead state

`angr-3zs6` CLOSED. `angr-pufm` (P1) still open — its remaining piece is the
lazy guarded-entries optimization.

## Suggested next slices

- `angr-pufm` lazy guarded-entries piece.
- `angr-fk0m` — Unify rust_state_sync/rust_state_cache/rust_state_export mixins.
- `angr-b6og` — Wire flamegraph/pprof into criterion benches.
- `angr-m2hf` — Unified error trait + single PyO3 conversion site
  (FallbackStrategy will likely become part of that taxonomy).
