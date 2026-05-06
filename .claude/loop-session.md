# Loop session notes (2026-05-06, eighty-ninth loop session — DONE)

## Status: COMPLETE — angr-hnd4 closed

Task: Audit/remove unused memory_store_ast callback hook.

## Outcome

Removed `memory_load_ast` and `memory_store_ast` from PythonCallbacks.
Both hooks were dead — setters never called from any Python file, so
the fields were always `None`. Their callsites silently no-op'd
(store) or returned zeros (load) when unset — strictly worse than
failing loud.

## Files changed

- `native/angr/src/callbacks.rs`: removed both fields, both setters,
  both call methods (-101 lines), updated traverse_fields/clear_fields
  field lists.
- `native/angr/src/interpreter_cb/statements.rs:1179-1188`: replaced
  legacy fallback with `Err(CbExecutionError::Unsupported(...))`.
- `native/angr/src/interpreter_cb/expressions.rs:194-243`: removed
  legacy fallback (50 lines), replaced with explicit Unsupported error
  when `memory_load_symbolic_full` is not set.

Net diff: -158 +10. Tests: 243/243 passing.

## Bonus finding (saved as memories)

While auditing, found that BOTH `memory_store_symbolic_full` and
`memory_load_symbolic_full` are ALSO never set from Python code —
only `memory_store_symbolic_value` is wired (rust_manager.py:711).
This means any TooLarge ConcretizationResult on store/load now hits
the new Unsupported error. Saved as
`invariant-symbolic-full-callbacks-unset` for angr-pufm future work.

Also saved `avoid-silent-no-op-callback-fallbacks` as an anti-pattern
memory.

## Build env note

`pip install -e .` is broken (pip 24.0 ImportError on pip._vendor.resolvelib).
Used the documented direct-cargo-build workaround:
`Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --release` then
`cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`.

Commit: f4c931eff
