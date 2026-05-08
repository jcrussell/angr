# Loop session notes (2026-05-08, 148th loop session)

## Task: angr-3zhl — load_concrete partial-overlap byte-merge (closed)

### Status: complete; closed.

### Summary
Fixed the long-standing bug captured in
`invariant-partial-overlap-byte-merge`: when a wider symbolic value was
followed by a later partially-overlapping store, the exact-address fast
path in `load_concrete` returned the stale wider value entire,
dropping the later store's bytes.

Strategy:
- Detect overlap on load: scan for any `symbolic_objects` entry
  whose key lies in `(addr, addr+size)`.
- If any: call new `try_byte_merge_load` which walks `[addr, addr+size)`
  byte-by-byte. Each byte resolves via:
  1. `symbolic_objects[byte_addr]` (start of value) → byte 0 lane.
  2. `symbolic_spans[byte_addr]` (interior of wider value) → offset lane.
  Endianness-aware extraction; LE/BE folds chosen so the accumulator
  is always the high half of `concat`.
- store-side metadata is unchanged; the fix is entirely on the load
  side.

### Verification
- New regression test (`test_load_concrete_partial_overlap_later_store_wins`)
  failed before fix, passed after. Asserts the LE byte-merge result for
  sym1 (64b) @ 0x1000 then sym2 (64b) @ 0x1004; load 8B @ 0x1000 must
  equal `concat(sym2[31:0], sym1[31:0])`.
- 597/597 Rust unit tests pass.
- 301/301 Python tests pass (test_rust_exploration.py).

### Memories saved
- `invariant-partial-overlap-byte-merge` (updated): fix recorded; load
  side handles overlap, store side untouched.
- `invariant-symbolic-spans-staleness` (new): document why
  `symbolic_spans[addr]` (the start byte itself) is not refreshed by
  store and why that is OK given the lookup order.

### Files
- native/angr/src/memory/load.rs (load_concrete + new helpers)
- native/angr/src/memory/tests.rs (1 new test)

### Commit
813dbcce3 fix(memory): byte-merge load when later store partially overwrites — angr-3zhl
