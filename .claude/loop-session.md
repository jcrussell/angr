# Loop session notes (2026-05-05, fifty-ninth loop session — DONE)

## Status: COMPLETE — angr-xok8 closed

## Task: angr-xok8 (P2)
Tests: unaligned wide access crossing 3 pages with mixed permissions.

## What was done
Added three Rust unit tests under `memory::tests` in `native/angr/src/memory.rs`:

1. `test_permission_enforcement_unaligned_store_two_pages_middle_readonly`
   — bead's literal scenario: 32-byte store at 0x1FF0 across pages 1,2
   with middle R-only. Exercises the multi-page slow path in store_concrete
   and verifies Permission error points at addr=0x2000.

2. `test_permission_enforcement_wide_store_three_pages_middle_readonly`
   — true 3-page store: 8208-byte BV at 0x1FF0 spans pages 1,2,3 with
   middle R-only. Locks down that check_perms_range iterates
   start_page..=end_page and flags the middle page.

3. `test_permission_enforcement_wide_load_three_pages_middle_writeonly`
   — dual of (2): 8208-byte load with middle W-only surfaces
   Permission { required=R, actual=W, addr=0x2000 }.

Memory tests: 24 → 27 passing. Python tests: 218/218 passing. Commit a0178d65e.

## Key empirical findings
- The bead's literal claim "32 bytes...3 pages" is impossible. With
  PAGE_SIZE=4096, a 32-byte access can cover at most 2 pages. To get a
  true 3-page span you need >4096-byte access width.
- RustBV::concrete accepts any `width: u32` even though its concrete
  value field is u128. For permission tests this is fine: check_perms_range
  fires before any bytes are written, so the actual data doesn't matter.
- Test pattern for 3-page span: `RustBV::concrete(value, 8208 * 8)` at
  addr=0x1FF0 covers pages 0x1, 0x2, 0x3 (last byte at 0x3FFF).

## Memories saved
- `invariant-perm-check-3page-test-pattern` — pitfall (32-byte limit) +
  the wide-BV pattern future tests should reuse for true >2-page spans.

## Build env reminder
- Cargo build/test still required `Z3_SYS_Z3_HEADER=/usr/include/z3.h`.
- Pure-test change: NO `pip install -e .` rebuild needed (Python suite
  passed against existing .so).

## Next-up (still ready)
- angr-eygl (P1) Differential test harness — BIG, multi-session
- angr-pufm (P1) Symbolic address concretization fallback — large feature
- angr-2i4n (P2) Tests: error path coverage (unmapped, OOM, Z3 timeout)
- angr-io1t (P2) Tests: VEX FP edge cases (NaN, infinity, conversion)
- angr-prem (P2) Introduce a MemoryLayer trait
- angr-fk0m (P2) Unify rust_state_sync / rust_state_cache / rust_state_export mixins
