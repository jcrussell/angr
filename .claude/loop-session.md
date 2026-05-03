# Loop session notes (2026-05-03, twenty-ninth session)

## Task: angr-8em4 — CLOSED
"Audit and replace panic-inducing patterns in Rust engine"

## Outcome
The bead claimed 543 panic-inducing sites needing per-module subtask
splitting. Re-audit found that after the prior 2026-04-20 audit + 72
commits, the picture had become much smaller:

- 509 `.unwrap()` calls remain in native/angr/src
- Only **4** are in production code; the other 505 are inside
  `#[cfg(test)]` test modules.
- The 4 production sites have all been replaced.

## Changes (commit 9192bc506)

  state.rs:2143  `fd_info(fd).unwrap()`           -> `filter_map`
  state.rs:2149  `InspectEvent::from_u8.unwrap()` -> `filter_map`
  fuzzer/monitor.rs:113  `call1.unwrap()`         -> map PyErr to
                                                    `libafl::Error::Unknown`
                                                    (same pattern as executor.rs)
  automaton/subset_construction.rs:42  `HashMap.get.unwrap()`
                                                  -> `expect(...)`
                                                     with invariant comment

The two state.rs filter_map fixes follow the `invariant-ffi-no-unwrap`
memory: snapshot iterators where the lookup is "guaranteed" should
filter_map instead so future code changes degrade gracefully rather
than panicking the Python interpreter.

## Verification
- `cargo check --release` clean.
- 208/208 tests pass.
- Re-running the awk audit on production code shows zero remaining
  `.unwrap()` outside `#[cfg(test)]` blocks.

## Files
- native/angr/src/state.rs
- native/angr/src/fuzzer/monitor.rs
- native/angr/src/automaton/subset_construction.rs

## Memories saved/updated
- `panic-audit-2026-05-03`: post-fix snapshot + the awk verification command
- `invariant-ffi-no-unwrap`: appended note that this pattern was applied
  again 2026-05-03 in state.rs::export_full

## Next ready (P3, no blockers)
- angr-cbko: native exit/abort SimProcs (warning: avoid memory says this
  caused infinite zero-length hook loops in v2; needs find/avoid sync)
- angr-bgv0: Z3 floating point theory support (30+ ops in vex/ops.rs)
- angr-awm3: CAS/LLSC statement handling (statements.rs:836-838)

## Remaining big methods (still untracked, for a future refactor session)
From last session's note — top unsplit methods:

  rust_state_export.py:
    _get_stash_states               139 lines
    _sync_exported_constraints       98 lines
    _sync_rust_memory_to_state       79 lines
  rust_state_sync.py:
    _extract_wide_symbolic_regions  111 lines
    _extract_register_changes        71 lines
    _sync_registers_to_rust          66 lines
  rust_manager.py:
    _cb_resolve_function            100 lines
    _cb_sync_constraints             87 lines
    merge                            76 lines
