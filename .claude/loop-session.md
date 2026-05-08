## Session log: 2026-05-08, 159th loop session

### Task: angr-5yxf — Tighten _init_cache user-symbolic guard (parity with disk cache)

Mirrored the disk-cache user-symbolic guard in the in-memory init cache
path. Changes in `angr/exploration/rust_manager.py`:

1. Added `_compute_mem_init_key(state, cache_key)` next to
   `_compute_disk_init_key`. Returns `''` when cache_key is empty OR when
   `_state_has_user_symbolic(state)` is True.
2. In `_run_python_init_if_needed`, compute `mem_key` and pass it to
   `_try_in_memory_init_cache` and `_step_python_to_main` (the latter
   forwards to `_save_init_state_to_caches`). This gates BOTH read and
   write paths.

Did NOT rename downstream `cache_key` parameter to `mem_key` — semantics
unchanged, just the source value is now gated. Less diff = less risk.
Did NOT remove the `_isolate_class_caches` test fixture — it remains as
defense-in-depth.

### Verification
- All 8 TestEdgeCases pass (production fix is in; test fixture is now
  redundant but kept).
- Smoke: state with BVS .data store → in-mem cache stays at 0 (was: 1, polluted)
- Smoke: clean state, fresh disk cache → in-mem cache populates to 1
  (regression check — no behavior change for the common path)
- Full suite: 332/332 in tests/engines/test_rust_exploration.py

### Status
done — to be committed and bead closed next.

### Followups
None. The fix is self-contained.
