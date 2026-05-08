# Loop session notes (2026-05-08, 138th loop session)

## Task: angr-0cnm — mmap_base diverges between Rust and Python state

### Status: complete; closed

### Change
- Rust (`exploration/mod.rs`):
  - `get_state_mmap_base(state_id)` — returns Rust's per-state mmap_base
    (uses `find_state`, so works for stashes + pending callback state).
  - `set_state_mmap_base(state_id, addr)` — useful for tests and for any
    future Python→Rust sync (uses `find_state_mut` — stashes only).
- Python (`rust_state_export.py`):
  - `_sync_rust_mmap_base_to_state(state, state_id)` — pushes Rust value
    into `state.heap.mmap_base` only if Rust > Python (max-merge avoids
    clobbering a Python-side advance).
  - Wired into all three export paths in `_get_stash_states`:
    cached state, parent-state copy, stepping-state copy, and snapshot
    fallback (4 sites total).
- Tests (`test_rust_exploration.py`, new `TestMmapBaseSync` class):
  - `test_get_state_mmap_base_default` — getter returns 0xC100_0000.
  - `test_set_state_mmap_base_round_trips` — setter/getter round-trip.
  - `test_get_state_mmap_base_unknown_state_raises` — error path.
  - `test_export_path_syncs_rust_mmap_base_into_state_heap` — bumps Rust
    mmap_base, calls `_get_stash_states('active')`, asserts the synced
    state's `heap.mmap_base` matches. **Verified to fail pre-fix.**
  - `test_export_path_does_not_clobber_higher_python_mmap_base` —
    locks down the max-merge semantics.

### Verification
- `cargo build --release` clean.
- 269/269 tests passing in `tests/engines/test_rust_exploration.py` (was 264).
- Pre-fix verification: stashed Python changes, ran the end-to-end test, it
  failed with `state.heap.mmap_base = 0xc1000000` vs Rust's `0xc1001000`.

### Memories saved
- `invariant-rust-python-state-sync-direction` — use max(rust, python)
  when syncing fields both sides can mutate.
- `invariant-posix-brk-drift-mirrors-mmap-base` — posix_brk has the same
  drift risk; same fix pattern applies.

### Follow-up bead created
- angr-as3c (P2 bug) — apply the same sync fix to posix_brk.

### Files changed
- native/angr/src/exploration/mod.rs
- angr/exploration/rust_state_export.py
- tests/engines/test_rust_exploration.py
