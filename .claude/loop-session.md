# Loop session notes (2026-05-05, forty-seventh loop session)

## Task: angr-v1q2 — DONE
Fix wide-symbolic partial-load extract for LE memory; add LE counterpart
to `test_big_endian_128bit_wide_symbolic_store`.

## What I changed
`native/angr/src/memory.rs`:
- `load_concrete` exact-address partial path (~line 542): branch on
  `self.endness`. BE: `(total_bits-1, total_bits-size*8)`. LE:
  `(size*8-1, 0)`.
- `load_concrete` symbolic_spans partial path (~line 560): branch on
  `self.endness`. BE: `(total_bits-off-1, total_bits-off-size*8)`.
  LE: `(off+size*8-1, off)` where `off = base_offset*8`.
- Added `test_little_endian_128bit_wide_symbolic_store` mirroring the
  BE counterpart. Pins a 128-bit BVS to a known u128, exercises
  per-byte loads (LSB at addr+0), 4-byte load at offset 4 (bits
  [63:32]), and 8-byte halves (low at offset 0, high at offset 8).

## Result
- 391/391 Rust unit tests pass (`cargo test --release --lib`).
- 214/214 pytest passing in `tests/engines/test_rust_exploration.py`.

## Beads / memory
- angr-v1q2 claimed → closed.
- `invariant-le-wide-symbolic-byte-layout` saved: LE byte layout
  convention for wide symbolic objects in symbolic_objects map.
- Created follow-up bead **angr-76mo** (P3): the multi-page
  has_symbolic slow path in `load_concrete` (~lines 648-666) has
  TWO additional partial-extract sub-paths that are LE-hardcoded:
  per-byte concat and the wider-symbolic linear scan. Symmetric to
  this fix; rarely hit; left for a separate task.

## Key insight (for future ports)
Wide symbolic values are stored in `symbolic_objects` as-is at the
base address. The endianness only affects which bits correspond to
which byte offset. Don't forget that `store_symbolic` does NOT
byte-reverse the BV — convention is encoded purely in the load-time
extract.

## Files modified
- `native/angr/src/memory.rs` — fix in `load_concrete`, new LE test in
  `mod tests {}`.
