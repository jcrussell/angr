## Session log: 2026-05-09, 176th loop session

### Dirty-state resolution

Inherited a 1-line dirty change in `angr/exploration/rust_manager.py`:
`_max_state_cache_size = 500` (reverting the angr-qm7w optimization
from 8 back to 500). Likely the previous (timed-out, iter 3) session
trying to test whether qm7w caused csgames2018 regression.

Verified the dirty change does NOT fix the regression:
- With cache=500 (dirty): csgames2018 timed out >90s
- With cache=8 (clean):   csgames2018 timed out >90s
Both fail identically. Cache size is NOT causal. Reverted (git stash drop).

### Task: angr-g7zs — csgames2018 regression — CLOSED

Skipped the bisect; instead used the runtime log "No cached state for ID
6, using blank state fallback" + fwrite filling thousands of 4096-byte
pages as the smoking gun, then traced into `_cleanup_state_cache`.

**Root cause:** the live filter at rust_manager.py:2603-2605 deleted
state 0 from `_state_cache` once state 0 left `active_set` (after it
forked into descendants). The qm7w pinning logic only applied later
at the overflow step, so state 0 was already gone by the time pinning
ran. When forked state 6 fired its first Python callback, the lookup
walked self → root_id (from Rust's `sm.roots()`) → ancestry; none of
those IDs were in the Python cache, so the callback fell through to
`_create_blank_state_fallback` where fwrite triggered
default_filler_mixin to lazily fill memory pages — runaway runtime.

**Fix (commit 7fe7baf79):** one-liner in `_cleanup_state_cache` —
`live |= set(self._state_roots.values())` before the eviction loop, so
roots survive the live filter (the same protection they already had
during the overflow step).

Why this only surfaced after qm7w (67d46f940): pre-qm7w, cleanup wasn't
called after every callback and the cap was 500 (rarely triggered), so
state 0 had a much narrower window to be evicted. Post-qm7w, cleanup
runs on every callback exit and the cap is tight (8), so any spurious
eviction got hit immediately.

### Verification

- csgames2018: timed out >90s → 0.96s (back to 0.96 baseline)
- 342/342 rust exploration tests passing
- Full regression suite: 12/12 benchmarks pass

### Memories saved

- `state-cache-root-eviction-bug` — invariant: any state-cache cleanup
  must merge roots into the 'live' set OR enforce pinning at every
  delete site, not just the overflow step.
- `csgames2018-cache-ruled-out` — recorded mid-investigation that
  cache size alone (8 vs 500) is not the cause.
- `benchmark-csgames2018-fix-2026-05-09` — before/after numbers and
  fix commit pointer.

### Next session

Pick up `bd ready`. Several P1/P2 items waiting.
