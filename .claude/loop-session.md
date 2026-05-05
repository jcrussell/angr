# Loop session notes (2026-05-05, forty-third loop session)

## Task: angr-wyxb — DONE
Add `test_symbolic_store_partial_overlap_constraint_propagation`
exercising two symbolic stores whose ranges partially overlap and
verifying that constraints from each store survive in the solver.

## Test added (memory.rs:2685-2766)
- Symbolic addresses `addr1`, `addr2` constrained to 0x1000 and 0x1004.
- Symbolic values `sym1`, `sym2`; sym1 carries an explicit value
  constraint (0xDEAD_BEEF_F00D_BABE).
- Two `store_symbolic` calls: covers [0x1000:0x1008) and
  [0x1004:0x100C) — overlap at [0x1004:0x1008).
- Asserts:
  - context still SAT after the stores
  - addr1's solution (== 0x1000) preserved (probing != 0x1000 is UNSAT
    in a forked context)
  - sym1's value constraint preserved (probing sym1 == 0 is UNSAT)
  - load[0x1000:4] succeeds and is evaluable

## Verification
- `cargo test --release test_symbolic_store_partial_overlap_constraint_propagation`
  passes (1/1).
- 213/213 RustExplorationManager tests still pass.
- `cargo check --release` clean.

## Files modified
- `native/angr/src/memory.rs` — new unit test in `mod tests`.

## Beads / memory
- angr-wyxb claimed → closed.
