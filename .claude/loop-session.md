## Session log: 2026-05-14 (afternoon)

### Completed this session

**angr-g9hy (MIPS32 $t0 symbolic accumulation bug) — CLOSED**
in commit ceba7816b.

Despite the bead title pointing at MIPS32 symbolic register collapse,
the root cause was NOT in the Rust engine at all. It was in the disk
init cache invalidation gate `_state_has_user_symbolic` (rust_manager.py),
which only scanned MEMORY for user symbolic data and ignored REGISTERS.

Repro flow:
1. User: `state = proj.factory.blank_state(addr=entry)`
2. User: `state.regs.a0 = claripy.BVS("a0", 32)`
3. User: `RustExplorationManager(proj, [state])`

`_state_has_user_symbolic(state)` returned False (memory clean), so
`_compute_disk_init_key` returned a valid cache key. The cached state
(with concrete a0=0 — `_extract_register_snapshot` skips symbolics)
silently replaced the user's state. `_sync_registers_to_rust` then ran
the fast precomputed_regs path, pushing a0=0 to Rust. The user's
symbolic a0 BVS was lost. Rust ran the entire MIPS32 chain with
concrete a0=0, producing concrete t0=0, BEQ concrete-false, no FOUND.

Fix: extended `_state_has_user_symbolic` to iterate
`state.registers._pages.items()` for `symbolic_data` entries whose
variable names don't start with `(mem_, reg_, unconstrained)` — the
default symbol-fill prefixes. User-named BVSes are detected and the
cache key returns '' (caching disabled), so the slow
`_sync_registers_to_rust` path runs and pushes symbolic a0 properly.

False-start: first attempt also iterated `getattr(state.regs, X)` over
all arch registers in the precomputed_regs fast path of
`_sync_registers_to_rust`. That triggered the default fill (BVS alloc +
warning log per uninitialized register) for ~80 x86_64 regs, causing
20-130% regression on ais3/csgames/defcamp/etc. Reverted. The cache
invalidation in `_state_has_user_symbolic` already suffices because it
forces the slow path which iterates `state.regs.*` ONCE (the original
behavior); the fast path doesn't need to handle symbolic regs.

Regression test added: `test_mips32_symbolic_register_survives_disk_init_cache`
in `TestMultiArchSupport`. N=30 multi-block accumulator, fails
deterministically pre-fix, passes post-fix.

Memories saved:
- `disk-init-cache-symbolic-reg-invariant` — rule for cache designers
- `g9hy-root-cause-not-mips` — root cause is not arch-specific
- `avoid-state-regs-iter-in-init` — anti-pattern for hot init paths
- `avoid-rust-mips-symbolic-accumulation-bench` — updated to FIXED

### Tests / Build state at session end
- Rust: cargo check clean.
- Python: 396/396 passing (test_rust_exploration.py, includes the new
  regression test). One test (test_model_stability_constraint_order)
  fails intermittently when the full suite runs — Z3 model-picker order
  flake, predates this session, not introduced by my changes (passes
  in isolation pre and post fix).
- Benchmarks: post-fix timings flat vs pre-fix on fauxware/defcamp_r100/
  ais3_crackme/csgames2018 (within noise). Baseline JSON values in
  baseline_timings.json don't match this machine, so run_regression.py
  reports false-positive regressions for those benchmarks — unrelated.

### Next picks
- angr-myty (perf dashboard) — substantial CI/Pages work.
- angr-pogf (lazy memory design) — research/design.
- The "non-determinism" description in the original bead was misleading
  (it was actually deterministic disk-cache replacement); future bead
  reports describing non-deterministic correctness on the Rust engine
  may also turn out to be disk-cache or fast-sync issues.
