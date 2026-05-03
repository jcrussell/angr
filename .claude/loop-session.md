# Loop session notes (2026-05-03, twenty-fifth session)

## Task: angr-2fs0
Decompose _handle_simprocedure_callback (~478 lines) and deduplicate
register/memory extraction.

## Status
- DONE. Commit 07a74c928. Closed angr-2fs0.

## What landed
Decomposed the SimProcedure callback handler in rust_callback_dispatch.py
into a 193-line dispatcher (down from 485) that calls 8 focused helpers:

| Helper                                | Purpose                                          |
|---------------------------------------|--------------------------------------------------|
| `_find_simprocedure`                  | Lookup by addr, name fallback, type check        |
| `_try_simproc_deadend_fast_path`      | Skip state creation for NO_RET / cached exit     |
| `_snapshot_orig_state`                | Full copy / register snapshot / no-copy logic    |
| `_capture_stdin_from_successors`      | Track stdin packets added by fgets/read/etc.     |
| `_capture_continuation_data`          | Stash procedure_data + register cont hooks       |
| `_handle_callback_with_successors`    | Success path with successors (~120 lines)        |
| `_push_continuation_address`          | Push self.call() cont addr to stack              |
| `_handle_callback_no_successors`      | Terminal/zero-length/normal no-successor cases   |

Plus deduplication helper in rust_state_sync.py:
- `_snapshot_registers_from_bundle(bundle_regs, arch)` — produces the
  same dict format as `_snapshot_registers` but skips reading from
  state. Replaces inline 14-line bundle->snapshot construction.

## Key invariant preserved
Per-procedure timer increments and perf_stats counters: every early-
return path increments _perf_stats counters EXCEPT the NO_RET-with-
successors deadend (line ~614 in original) — that path was treated as
"aborted before completion." Saved to memory
`invariant-simproc-callback-perfstats`.

## Tests
- 208/208 passing in 6.76s.
- fauxware --engine rust: finds SOSNEAKY.
- ais3_crackme --engine rust: 0.85s, finds flag.

## Files modified
- angr/exploration/rust_callback_dispatch.py (dispatcher + 8 helpers)
- angr/exploration/rust_state_sync.py (1 deduplication helper)

## Memories saved
- `invariant-simproc-callback-perfstats`: which early-return paths
  must finalize perf_stats counters.
- `callback-bundle-snapshot-helper`: documents the new
  _snapshot_registers_from_bundle and when it's used.

## Other ready P3 tasks
- angr-w4os: Python bridge cleanup (split sync/export/cache/init methods)
- angr-cbko: native exit/abort SimProcs
- angr-8em4: replace panic patterns
- angr-3ijo: bincode for VEX IRSB serialization
- angr-bgv0: Z3 floating point theory
- angr-awm3: CAS/LLSC statement handling
- angr-v4db: extract god-methods in Python bridge layer
