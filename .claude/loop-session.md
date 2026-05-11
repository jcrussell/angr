## Session log: 2026-05-11 — angr-hv22 (profile _overlay_relocated_sections, RATIONALE)

### Task
Profile `_overlay_relocated_sections` (claimed 6.1 ms / 36 % of warm
`_add_rust_state` per cProfile) and decide between optimization or
written rationale.

### Profile (wall-clock, fauxware, 20 iterations)
Per-call (ms): min=1.19  max=7.74 (cold)  mean=1.79  median=1.475

Section breakdown (one full sweep, fauxware):
  Sections in range:    29
  Concrete (overlaid):  25
  Symbolic (skipped):   4

Size histogram:  <256 = 24,  <4K = 5  (all sections are tiny)

Per-section operation cost (one full sweep, 29 sections):
  state.memory.load:    1.250 ms (97 %)
  solver.eval+to_bytes: 0.026 ms  (2 %)
  rust map FFI:         0.016 ms (~1 %)

The 6.1 ms cProfile number in the bead is inflated relative to wall-clock,
same artefact we hit in z8xa (cProfile penalises code with lots of nested
attribute access through angr's 14-layer memory-mixin chain).

### Decision: rationale, no code change

1. **Wall-clock is 1.5 ms, not 6.1 ms.** The slow path's true cost is
   ~97 % `state.memory.load` walking the angr mixin stack (same root cause
   as z8xa register sync). solver.eval + FFI together are negligible.

2. **Naive manager-scope caching is unsafe.** Current code gates each
   section overlay on `val.symbolic` per-state. Caching section bytes
   keyed by binary path would lose this gate, so a forked successor that
   has symbolic bytes in a .data section would have those bytes
   overwritten by stale concrete patches from the first state.

3. **Disk cache already handles the safe case.** When state has no
   user-symbolic, `_compute_disk_init_key` succeeds, the disk pickle is
   saved, and `_try_fast_memory_sync` short-circuits the entire memory
   sync (including section patches) on later runs. The slow path only
   runs in the unsafe case where caching is risky.

4. **In-process gain is small.** `_overlay_relocated_sections` runs once
   per `_add_rust_state` — initial seed (1-2x), merged states (rare),
   forked successors (handful). Even at 5-10 calls per fauxware run,
   total cost is ~10-15 ms in a ~1 s exploration; <2 %.

5. **One niche optimization considered and rejected**: when
   `_try_in_memory_init_cache` hits without populating `_mem_cache`, the
   slow path runs unnecessarily even though disk cache exists. Fixing
   it would help only when multiple managers are constructed for the
   same binary in the same process — rare outside benchmark loops, and
   `_isolate_class_caches` clears the in-memory cache between tests.

### Tests
146/146 pass (no code changes).

### Memory saved
`hv22-overlay-relocated-bottleneck` — full profile breakdown + rationale.

### Status
COMPLETE — bead angr-hv22 closed as no-action.
