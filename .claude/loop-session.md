# Loop session notes (2026-05-03, thirty-eighth loop session)

## Task: angr-7l12 — CLOSED
"Enforce NX (execute) permission on basic block fetch in Rust interpreter"

Plus: deferred angr-borb (StateId/Address newtype refactor) — bead
description references deleted pyapi.rs and full scope is 283 sites
across 25 files; too large/stale for a single session.

## Outcome
With STRICT_PAGE_ACCESS / `enforce_permissions=true`, basic-block
fetch now rejects mapped non-executable pages. Combined with the
existing R/W enforcement on load/store, the Rust engine now mirrors
angr's STRICT_PAGE_ACCESS semantics for all three access types.

Test count: 210 pass (was 209, added 1 Python integration test +
4 Rust unit tests).

## What changed
- **native/angr/src/memory.rs**
  - Added `pub fn check_executable(&self, addr: u64) -> Result<(), MemoryError>`.
    No-op when `enforce_permissions` is false or page is unmapped.
    Errors only on mapped-without-X.
  - 4 new Rust unit tests covering: rejection, success, unmapped
    pass-through, no-op when disabled.
- **native/angr/src/interpreter_cb/execution.rs**
  - In `get_or_lift_block`, call `mem.check_executable(addr)` before
    the cache lookup so even cached IRSBs get rejected after a page's
    X bit is stripped. Failure mapped to `CbExecutionError::Memory`.
- **tests/engines/test_rust_exploration.py**
  - New `test_strict_page_access_blocks_nx_block_fetch`: builds a
    `_RustExplorationManager` with no-op callbacks, maps a page perm=6
    (RW, no X), points PC at it, runs, asserts the state lands in
    `errored` and `lift_block` was never invoked at the NX address.

## Verification
- `cargo check --release` clean.
- `cargo test --release --lib check_executable` 4/4 pass.
- 210/210 RustExplorationManager tests pass.
- fauxware single-engine benchmark still finds SOSNEAKY in ~0.37s.

## Memories saved
- `invariant-nx-block-fetch` — where the X check fires + why unmapped pages pass through
- `invariant-rust-perm-mapping-coarse` — current Python-side perm mapping is hardcoded RWX/RW; what the NX check actually catches today
- `strict-page-access-wiring` — updated to include the new NX check
- `avoid-deferred-borb-newtypes` — rationale for deferring angr-borb

## Potential follow-up
- Improve `angr/exploration/rust_state_sync.py` to use loader's
  per-page permissions instead of hardcoded `7`/`6`. Would let the
  NX check (and the load/store R/W check) catch real per-section
  violations (e.g., `.text` write attempts), not just stack-X /
  unmapped-X. Bigger change — needs to touch 4-5 map_memory_data
  call sites and verify no regressions across all benchmarks.
- Consider auto-enabling STRICT_PAGE_ACCESS by default for real
  Project flows once the perm mapping is granular enough that it
  doesn't break legit code. Currently still opt-in.
