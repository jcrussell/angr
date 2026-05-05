# Loop session notes (2026-05-05, fifty-fifth loop session — DONE)

## Status: COMPLETE — angr-nwbx closed

## What was done
**angr-nwbx** (P1): Cache claripy→Z3 AST pointer for register sync.

Added `_z3_ptr_cache: dict[(hash, length) -> (z3_obj, ast_ptr)]` on
`RustExplorationManager`, consulted by a new
`RustStateSyncMixin._cached_z3_ast_ptr()` helper. The helper replaces the
direct `z3_backend.convert(reg_val).as_ast().value` walk in
`_sync_registers_to_rust`. Holds a strong ref to the z3 wrapper so the AST
pointer stays valid; bounded eviction at 1024 entries. Surfaces
`z3_ptr_cache_hits/_misses` in `stats()` and run_single.py output.

Tests: 214/214 pass. Fast-tier regression: 12/12 in 20.2s. fauxware:
0.37–0.40s (within noise of 0.38s baseline). Commit: a589c0708.

## Key empirical finding
The symbolic-register import path is NOT a hot path on the current
benchmark suite. Cache hit/miss across 7 benchmarks:
- fauxware: 0/0, ais3_crackme: 0/2, csaw_wyvern: 0/0,
- flareon2015_5: 0/8, securityfest_fairlight: 0/2,
- sym-write: 0/2, strcpy_find: 0/2.

Reason: SimProcedure callbacks operate entirely inside Rust state.
Register sync from Python only happens during initial state seeding (or
disk-cache restoration), so the bead's premise (50µs × 16 regs ×
N callbacks) overstated by ~100x. The cache lands as observability +
safety net for repeated imports.

## Memories saved
- `register-sync-bottleneck` — empirical proof that register sync is not
  a hotspot; bead premise was wrong.
- `fauxware-cost-breakdown` — fauxware time accounting: 124ms in
  SimProcedures (open 59ms, strcmp 21ms, 4×read 16.6ms each), 17.6ms sync
  back, 9.2ms state create. Win-paths: native read SimProcedure, reduce
  sync_back, reduce state copy.

## Build env reminder
Pure-Python change — no `pip install -e .` rebuild needed. Use
`Z3_SYS_Z3_HEADER=/usr/include/z3.h` for any cargo check.

## Next-up (still ready, P1)
- angr-eygl Differential test harness — test infra (highest-ROI per audit)
- angr-pufm Symbolic address concretization fallback — large feature
- angr-2i4n Tests: error path coverage — test infra
- angr-prem MemoryLayer trait — refactor

If re-attempting perf wins around fauxware, target the SimProcedure
execution itself (read SimProc is 16.6ms × 4 = 66ms of fauxware's 380ms)
per the fauxware-cost-breakdown memory. The convert/register-sync path
is firmly NOT the bottleneck.
