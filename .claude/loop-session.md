# Loop session notes (2026-05-07, 123rd loop session)

## Task: angr-m2hf — Define a unified error trait + single PyO3 conversion site

### Status: AUDIT → DEFER

After detailed audit of the current Rust error infrastructure, I'm deferring
this bead with the same reasoning template that closed angr-borb / angr-ja0b /
angr-x3xu: the proposed abstraction does not fit the actual usage pattern,
the scope is too large for a single session, and no concrete bug class
motivates it.

### Audit findings

1. **thiserror conversion is already done.**
   Commit 923c3a6ca (2026-05-01) converted 10 of 11 manual error enums to
   `thiserror::Error` with `#[error("…")]` derives and `#[from]` chaining.
   StepError is deliberately exempt (carries `RustSimState` as a control-flow
   signal, documented in memory `invariant-step-error-not-thiserror`).
   ProcedureError keeps a manual `From<MemoryError>` to avoid changing wire
   format. So the "all error types share infrastructure" half of the bead is
   already realised — just via thiserror instead of a hand-rolled trait.

2. **`lib.rs has one conversion call site` doesn't fit the architecture.**
   `native/angr/src/lib.rs` is a 97-line `#[pymodule]` registration file. It
   has zero error conversion sites today and is not the natural place for
   them. PyO3 conversions happen at the leaf `#[pymethods]` / `#[pyfunction]`
   sites where errors arise. Centralising in lib.rs would require routing
   errors through a global, which fights the `?` operator instead of using it.

3. **Typed errors mostly never cross the PyO3 boundary.**
   The 168 `Result<_, MemoryError | ExecutionError | OpError | …>` types
   propagate within Rust. At the boundary (e.g. `engine::run_loop`), errors
   are absorbed into `RunResult::Error(String)` events that `LoopExecutionEvent`
   carries to Python. The Python side (`rust_manager.py`) reads
   `event.error: str` and raises a typed angr exception based on context —
   not based on the Rust error class. So even a "richer hierarchy" of Rust
   exception classes would not be observed by Python's existing catch contract.

4. **The 180 `PyRuntimeError::new_err(...)` sites are mostly ad-hoc.**
   Surveyed across 15 files. The patterns are:
     - "callback not set" / "callbacks not ready" guards (~15 in callbacks.rs
       and exploration/mod.rs)
     - "no pending callback state" control-flow guards (~7 in exploration/
       mod.rs and helpers.rs)
     - `map_err(|e| PyRuntimeError::new_err(format!("…: {e}")))` wrappers
       that add context to a leaf error (~30, mostly in icicle.rs / fuzzer.rs
       / exploration/mod.rs)
     - Inline `PyRuntimeError::new_err("specific message")` for one-off
       failures (~120)
   None of these go through the typed enums; they construct PyErr directly
   with a custom string. A unified `EngineError` trait + `convert_to_pyerr`
   would NOT replace any of them, because they are not enum values.

5. **Python's catch contract is already correct for the current scheme.**
   Memory `invariant-rust-callback-narrow-except` (angr-8e81 / angr-2f7o)
   documents the contract: `RustExplorationManager._cb_*` callbacks catch
   `(SimError, ClaripyError)` or `(SimEngineError, ClaripyError, PyVEXError)`
   on the lifter — these come from Python-level wrappers, not from PyO3-side
   exception classes. Adding PyO3-defined classes via `create_exception!`
   would require Python catchers to catch those too, doubling the contract.

6. **No documented bug class points at error-handling.**
   `bd memories error` and `bd memories pyerr` surface 16 memories — none
   describe an incident where confusing PyRuntimeError messages caused a
   missed exception or wrong fix. The existing scheme has carried us through
   146 tests + 16 benchmarks correct.

### Why this matches the angr-borb / angr-ja0b deferral pattern

Same template:
- (a) Bead description references infrastructure that has shifted
  (lib.rs has no conversions; thiserror already covers the enum side).
- (b) Full scope is large (180 sites across 15 files for the maximal
  reading; even minimal scope is ~10 enums × creating exception classes).
- (c) PyO3 boundary doesn't make a "single conversion site" natural —
  errors arise at leaves and ride the `?` operator.
- (d) No bug class observed to motivate the work.
- (e) Half-measures (just impl an empty trait, no exception classes) deliver
  no behavioural value.

The recommendation in the bead — `convert_to_pyerr(err: &dyn EngineError)`
called from a single point — assumes errors are already typed values at the
boundary. They're not: 95%+ of PyErr sites are constructed inline with
ad-hoc messages, not converted from enum variants.

### Action

1. Defer angr-m2hf with this audit as the reason.
2. Save memory `avoid-deferred-m2hf-error-trait` so future sessions don't
   re-open this bead without new motivation (a real bug class).

### Files modified

- None — audit only; no source edits.

## Status: complete
