## Session log: 2026-05-09 — angr-800o (AArch64 + MIPS32 integration tests)

### Task

Sibling/follow-up of angr-lvem (ARM landed in 32328959f). Add real-binary
end-to-end integration tests for aarch64 and mips32 in
tests/engines/test_rust_exploration.py. No AArch64/MIPS32 binaries ship
with angr-examples and no cross-compiler is available locally, so the
tests embed hand-assembled instructions and load them via cle's Blob
backend.

### Resolved dirty state from prior session

Previous session left three uncommitted files. Diagnosed and fixed an
inadvertent regression in TestCallableStepFunc::test_callable_with_rust_engine:

- The original `_load_binary_regions` skipped externs/tls/kernel via
  `if obj.binary is None`, but those pseudo-objects actually have a
  synthetic `obj.binary='cle##externs'` (str, not None) — they just
  happened to have no executable *sections*, so nothing leaked through.
- Adding the Blob-loader segments fallback exposed the gap: cle##externs
  has an executable *segment* at 0x700000-0x700030, which got loaded as
  binary code, and the Rust interpreter started lifting through extern
  trampolines, splitting the Callable test on a symbolic condition.
- Fix: explicitly skip pseudo-objects with `obj.binary.startswith("cle##")`
  before applying the segments fallback.

### Files modified

- angr/exploration/rust_manager.py: `_load_binary_regions` skips cle##
  pseudo-objects and falls back to segments only when sections expose
  none — needed for Blob-loaded aarch64/mips32 blobs.
- angr/exploration/rust_state_sync.py: register name lists for AARCH64
  (x0..x30, sp, pc) and MIPS32 (full GPR set + pc/hi/lo) so register
  sync to/from Rust covers both new arches.
- tests/engines/test_rust_exploration.py: two new tests
  (test_aarch64_explore_blob, test_mips32_explore_blob) loading 7
  hand-assembled instructions per arch via Blob; both assert the
  symbolic input is constrained to 42 in the found state.
- CLAUDE.md: promoted ARM64 and MIPS32 from Skeleton to Experimental
  in the support matrix.

### Test status

354/354 passing (was 352 before the new tests).
