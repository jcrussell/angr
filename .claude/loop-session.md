## Session log: 2026-05-17 — angr-9maq mma_howtouse leak fixed

### Status: CLOSED — surgical fix in rust_state_sync.py landed
            (commit fced54a07). 437/437 tests pass.

### Task

**angr-9maq (P1, bug)** — "mma_howtouse regressed 6.5s/286MB ->
55s/1888MB; leak fix appears bypassed". Filed by prior session as
a blocker for angr-518z acceptance.

### Root cause

Commit 293aa8163 (angr-7vcx, 2026-05-14, "sync user-mapped
concrete pages to Rust") REMOVED the `if any(concrete)` zero-page
filter in `angr/exploration/rust_state_sync.py::_sync_extra_python_pages`
to fix a separate bug (user `map_region`'d pages with no content
getting silently de-mapped in Rust).

But Python state.memory._pages contains thousands of ZERO_FILL /
SYMBOL_FILL_UNCONSTRAINED_MEMORY filler pages that look identical
to user-mapped zero pages via `concrete_load(0, page_size)`.
Mapping all of them via `rust_state.map_memory_data(page_addr,
[0]*4096, perms)` allocated a 4KB buffer in Rust per page per
state. Callable-heavy workloads (mma_howtouse: 45 short-lived
managers) hit this every state-sync, blowing peak memory and
per-call time 10x.

### Fix

`angr/exploration/rust_state_sync.py::_sync_extra_python_pages`:
- Still call `add_lazy_region(page_addr, page_size)` for the zero
  case so the page is recognized as user-mapped. The first store
  goes through `store_concrete_automap_internal` in
  `native/angr/src/memory/store.rs:640` which auto-allocates a
  zero page when the address is in a lazy region.
- Skip the eager `map_memory_data(page_addr, [0]*4096, perms)`
  call for all-zero pages. Non-zero pages keep the existing
  eager-map behaviour (so any data the Python side wrote during
  stage 1 still gets seen by Rust).

Net diff: +17 lines, -6 lines in one file.

### Bisection

- Wrote `/tmp/leak_repro.py` — subprocess wrapper with RLIMIT_AS=4GB
  that runs N callable invocations and reports per-call time +
  ru_maxrss. 10 calls was enough to expose 1.93s vs 0.17s.
- Narrowed 151 commits in 7 steps via
  `git checkout <commit> -- native/angr/src angr/exploration;
  cargo build --release; cp ... .so; run repro`.
  Each step ~30s incremental cargo + 20s reproducer = 50s.
- Pinned to commit 293aa8163 (angr-7vcx) — diff between the
  known-good parent cf71a2ad2 and 293aa8163 was a single file
  (rust_state_sync.py) so the smoking gun was visible immediately.

### Test result

437 / 437 test_rust_exploration.py pass on HEAD.
TestRustConcreteMemoryStoreRoundTrip (the angr-7vcx regression
guard) still passes — store-to-mapped-zero-page round-trip works
via lazy auto-mapping.

PR-time benchmark gate (`run_regression.py --rust-only --skip-bimodal`):
13/13 pass. One soft warning on unmapped_analysis 18% slower
(0.79s -> 0.96s, baseline value pre-7vcx) — borderline noise
threshold, no test failure.

### Benchmark before/after

| benchmark        | before fix      | after fix       | baseline        |
|------------------|-----------------|-----------------|-----------------|
| mma_howtouse     | 55.04s 1888MB   | 7.23s 277MB     | 6.51s 286MB     |
| ais3_crackme     | 2.00s 929MB     | 0.91s 369MB     | 0.84s 367MB     |
| fauxware         | 0.21s 176MB     | 0.21s 176MB     | 0.39s 251MB     |
| unmapped_analysis| ~0.96s 274MB    | 0.96s 274MB     | 0.79s 270MB     |

Per-call mma_howtouse timing in Callable loop: 1.93s -> 0.18s.

### Files modified

- `angr/exploration/rust_state_sync.py` — guard
  `map_memory_data` call with `if any(concrete):`; updated
  docstring/comment to cite both angr-7vcx and angr-9maq.

### Memories saved

- `9maq-root-cause` — full root-cause writeup (filter removal in
  293aa8163 mapped thousands of filler pages per state).
- `invariant-lazy-region-auto-map` — invariant: lazy_region +
  store_concrete_automap_internal is sufficient for stores; eager
  map_memory_data is an allocation optimisation, not a correctness
  requirement.
- `benchmark-9maq-fix-2026-05-17` — before/after numbers.
- `9maq-bisect-method` — reusable bisect pattern with
  `/tmp/leak_repro.py` + cargo-only rebuild loop.

### Followup work for next session

- Now that angr-9maq is closed, angr-518z's acceptance can be
  re-measured: confirm the cleanup() flag actually does help
  mma_howtouse drop from 7.23s -> closer to the 5.07s gain
  predicted by `mma-howtouse-cache-clear-speedup` memory.
- Consider whether angr-7vcx's other change in `_scan_symbolic_pages`
  (scanning `symbolic_data` instead of `symbolic_bitmap`) has a
  similar untested cost path. Not observed in benchmarks but worth
  a quick check on Callable workloads.
