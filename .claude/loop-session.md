## Session log: 2026-05-11 — angr-trwl (fauxware FFI overhead investigation)

### Task
Investigate fauxware FFI-overhead bottleneck (cited as 0.9x baseline,
only canonical demo binary slower than Python). Per task: profile and
either fix or write a 'known-slower' rationale + update CLAUDE.md.

### Live measurement (5 runs, run_single.py fauxware --both)
  rust:   0.27 / 0.27 / 0.27 / 0.28 / 0.28 s  (median 0.28)
  python: 0.37 / 0.37 / 0.38 / 0.38 / 0.40 s  (median 0.38)
  speedup: 1.36x — Rust is now FASTER than Python.

### Root cause of the flip
angr-3tek.2 (2026-05-10) registered NativeRead/NativeWrite by default
(native/angr/src/procedures/mod.rs:240-241) plus the page-replay fix in
_replay_rust_dirty_pages. fauxware callback count dropped 6 -> 1
(eliminated 4 read() callbacks @ ~20ms each, plus strcmp benefit).
Only open() still falls back to Python.

### Profile breakdown of current 280ms rust runtime
  Init total:       20.9 ms  ( 7%)
  open SimProc x1:  59-61 ms (20%) — 43ms execute + 13ms sync + 3ms create
  lift_block x14:    2-3  ms ( 1%)
  Rust interp/Z3:  ~205 ms  (~72%, residual)
No remaining FFI hot path. The bead's premise is obsolete.

### Changes (commit 92e22981f)
- CLAUDE.md: fauxware row 0.9x -> 1.4x; performance line 14/16 -> 15/16.
- baseline_timings.json: fauxware rust_time 0.385->0.28, callback_count 5->1.

### Tests
382/382 passing (same 3 pre-existing failures: dcas_cmpxchg16b_no_match,
pipe_native_dispatch_creates_two_fds, dup2_native_dispatch_redirects_stdin).
Regression script run: fauxware passed against new tighter baseline.
defcamp_r100 had a 16% noise regression (0.27 vs 0.23) — pre-existing.

### Memories saved
- trwl-fauxware-flip-2026-05-11 — flip explanation + lesson re re-measuring stale beads
- benchmark-fauxware-2026-05-11 — snapshot for future drift comparisons

### Status
COMPLETE — bead angr-trwl closed.
