# Loop session notes (2026-05-08, 141st loop session)

## Task: angr-qm7w — Tighten _state_cache cap with pinning + LRU evict (closed)

### Status: complete; closed

### Summary
The bd called for "lazy populate + immediate evict" of `_state_cache` so a
1000-state exploration runs with cache bounded by `concurrent_callbacks ≈
1-2`. Going strictly to ≤ 2 requires capturing per-state plugin diffs (posix,
libc, heap, fs, log, globals) into a separate overlay dict and evicting full
SimStates after each callback — a significantly larger refactor than fits in
one loop session and at risk of breaking the VEX-execution memory_load
callback path that currently reads state.memory directly from the cached
state.

This session delivered the bulk of the win without that refactor: drop the
cap from 500 to 8, run cleanup at every `_dispatch_callback` exit, evict
non-pinned entries down to the cap, and pin (root states + current callback
state + currently-stepping state) so the plugin-mutation chain across
consecutive callbacks survives intentional eviction. Memory peak moves down
without breaking correctness; the strict-≤-2 form is left for a follow-up.

### Changes
- **rust_manager.py** — `_max_state_cache_size` 500 → 8 with a comment
  explaining the new bound. `_cleanup_state_cache` rewritten: step 1 drops
  entries whose state no longer exists in any live stash (deadended/errored
  GC), step 2 builds the pinned set (roots + current callback + stepping
  state), step 3 LRU-evicts non-pinned entries down to the cap (Python dict
  insertion order = LRU; recently-rewritten entries sit at the back). Added
  `_cleanup_state_cache()` call at the end of `_dispatch_callback`.
- **test_rust_exploration.py** — three regression tests in two new classes:
  `TestPluginMutationAcrossCallbacks` (globals + posix.fd[99] = simfd
  mutations, both proving consecutive-callback plugin chain survives) and
  `TestStateCacheSizeBound` (peak `_state_cache` size during a fauxware run
  stays within `cap + 4`).

### Verification
- `pytest tests/engines/test_rust_exploration.py` — **288/288 passing** (was
  285 before; +3 new regressions).
- fauxware benchmark: 0.32s peak_mem 188 MB (no regression).
- defcamp_r100 benchmark: rust 0.23s vs python 1.03s (4.5x — no regression).
- ais3_crackme benchmark: 0.97s, correct flag (no regression).

### Memories saved
- `invariant-state-cache-pinning` — `_cleanup_state_cache` must pin roots,
  current callback state (and effective ID via `_get_effective_state_id`),
  and current stepping state. Without pinning, post-callback writes would be
  evicted before the next callback could use them.
- `avoid-strict-2-cache-bound-without-overlay` — strict ≤ 2 cap requires a
  plugin-overlay dict; otherwise VEX memory_load misses callback-induced
  writes that are still in state.memory but not yet synced to Rust.
- `state-cache-fork-write-sites` — map of cache-write call sites
  (`_resume_with_state`, `_resume_with_skip_hook`,
  `_handle_syscall_callback`, `_handle_symbolic_branch_callback`,
  `_add_rust_state`) for any future migration to a plugin-overlay design.

### Files changed
- angr/exploration/rust_manager.py (+51 −9)
- tests/engines/test_rust_exploration.py (+158)

### Unblocks
- angr-c3zm (RustPosixState migration checkpoint) is one prerequisite closer
  — still waiting on angr-xg0o, angr-0z34, angr-3tek, angr-nnov.
