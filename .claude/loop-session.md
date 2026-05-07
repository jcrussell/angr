# Loop session notes (2026-05-07, 124th loop session)

## Task: angr-prem — Introduce a MemoryLayer trait so the interpreter is generic over backend

### Status: AUDIT → DEFER

After audit, deferring with the same reasoning template as angr-borb /
angr-ja0b / angr-x3xu / angr-m2hf. The proposed trait does not fit the
actual usage pattern.

### Audit findings

1. **`memory.rs is pure data` is wrong.**
   `native/angr/src/memory/mod.rs` (and its sibling load.rs / store.rs) is
   not pure data — its `MemoryError` enum drives control flow at the
   call site:
     - `MemoryError::UnmappedPageInRegion { page_addr }` → fetch the
       Python page and retry the operation.
     - `MemoryError::Unmapped { addr, size }` → fall through to Python.
     - `MemoryError::SymbolicAddress { description }` → fall through to
       Python (Python's wide concretization handles this case natively).
     - Other variants → propagate as `CbExecutionError`.

   These signals are interpreter-layer concerns, not backend-layer.

2. **try_rust_memory_load is not a peer to load_from_callback.**
   - `try_rust_memory_load` (interpreter_cb/expressions.rs:619) returns
     `Result<Option<RustBV>>`: `Some` = Rust handled, `None` = fall
     through, `Err` = hard error. It also takes `&self.concretizer` and
     timing stats — interpreter-level inputs.
   - `load_from_callback` (interpreter_cb/mod.rs:1015) returns plain
     `Result<RustBV>`. Python is the page authority and concretizer; it
     doesn't have UnmappedPageInRegion semantics at all.

3. **The Load expression walks ~8 sources, not 2.**
   `expressions.rs:45` consults: try_rust_memory_load → pending_symbolic_stores
   → pending_stores buffer → all_flushed_symbolic_stores → all_flushed_stores
   → prefetch cache → try_read_concrete_memory → load_from_callback. Most
   of these are caches, not "memory layers." A trait with two impls would
   collapse only 2 of those cases.

4. **`trait MemoryLayer { fn load(addr, size); fn store(addr, value, size); }`
   discards the rich semantics.** A faithful unification would need:
     - Two-phase result (`Some` / `None` / `Err`) for fall-through.
     - Page-fetch hook so the layer can request more data before retrying.
     - Concretization context.
     - Timing stats path.
   At that point the "trait" is the entire interpreter Load/Store routine,
   not a memory backend.

5. **PythonCallbackMemory is not a peer impl.**
   The Python callback layer doesn't have UnmappedPageInRegion or
   SymbolicAddress concepts because Python owns the full memory model.
   A `MemoryLayer for PythonCallbacks` impl would either always return
   "handled" (Some) or would have to fabricate signals that aren't
   meaningful in the Python model.

6. **No documented bug class motivates the work.**
   `bd memories memory` and `bd memories rust-memory` produce no
   incident pointing at the dual-path design. The current scheme has
   carried us through 146 tests + 16 benchmarks correct.

### Why this matches the prior deferral pattern

Same template as angr-borb / angr-ja0b / angr-x3xu / angr-m2hf:
- (a) Bead description references infrastructure that has shifted
  ("memory.rs is pure data" was once true; now it carries control-flow
  signals).
- (b) Full scope is large (~8 sources to unify; lazy-page protocol to
  thread; concretizer to plumb through).
- (c) The "two parallel code paths" pitch doesn't match reality — the
  Python path is a fall-through target, not a peer backend.
- (d) No bug class observed to motivate the work.
- (e) Half-measures (trait wrapping only SymbolicMemory, with Python kept
  separate) deliver no behavioural value.

### Action

1. Defer angr-prem with this audit as the reason. (done)
2. Save memory `avoid-deferred-prem-memory-layer` so future sessions
   don't re-open without (a) a concrete bug class showing the dual
   path causes wrong behavior, OR (b) a third backend that needs the
   trait. (done)

### Files modified

- None — audit only; no source edits.

## Status: complete
