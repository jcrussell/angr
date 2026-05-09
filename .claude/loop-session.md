## Session log: 2026-05-09 — angr-lvem CLOSED (184th loop session)

### Task: ARM and MIPS real-binary integration tests

Reduced scope to ARM-only because no AArch64 or MIPS binaries are
available locally and the loop env has no network for cloning
github.com/angr/binaries. Spawned angr-800o for the AArch64/MIPS
follow-up.

### Files modified

- tests/engines/test_rust_exploration.py (+41) — added
  test_arm32_explore_real_binary inside TestMultiArchSupport.
- CLAUDE.md (+3 / -3) — promoted ARM (32-bit) from Skeleton to
  Experimental in the architecture matrix; trimmed Skeleton-language
  to ARM64/MIPS.

### Test design

- Loads ~/repos/angr-examples/examples/android_arm_license_validation/validate
  (ARMEL).
- blank_state(addr=0x401760), 80-bit BVS at 0xffe00000, r0 set to point
  at the BVS.
- explore(find=0x401840, avoid=0x401854, num_find=1, max_steps=2000).
- Asserts found ≥ 1 AND found[0].solver.satisfiable() — locks both the
  exploration completion and the symbolic-input round-trip through
  the Rust solver.
- Skips when the binary is missing.

### Commits

- 32328959f — test(arch): ARM32 real-binary integration test for Rust engine
- c0c0ddf75 — docs(arch): promote ARM (32-bit) from Skeleton to Experimental

### Memories saved

- invariant-arm-integration-test-binary-path — explore-end-to-end
  proof that ARMEL works in the Rust engine; gates on os.path.exists.
- arch-matrix-arm-supported-2026-05-09 — ARM promoted to Experimental;
  benchmark still missing for full Supported status.

### Other actions

- Released angr-wqao.1 (mechanical refactor; parent angr-wqao
  deferred). See avoid-deferred-wqao1-disk-cache-save-extract memory.
- Created angr-800o (AArch64/MIPS integration tests blocked on
  binaries).
