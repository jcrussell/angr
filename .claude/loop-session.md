## Session log: 2026-05-08, 161st loop session

### Task: angr-byev — Architecture support status matrix in CLAUDE.md (P1)

**Outcome: closed.** Commit edeb2fb74 adds an "Architecture Support Matrix"
section to CLAUDE.md with per-arch counts and Supported / Experimental /
Skeleton labels.

### Findings

Counted via grep on `tests/engines/test_rust_exploration.py` and file
inspection of `native/angr/src/arch/`:

| Arch        | Unit | Integration   | Benchmarks | Calling conv     | Status       |
|-------------|------|---------------|------------|------------------|--------------|
| AMD64       | ~110 | ~268 fauxware | 15/16      | SystemV + MS x64 | Supported    |
| x86 32-bit  | 2    | 1 (Cdecl ret) | 1 (flareon2015_2) | Cdecl     | Experimental |
| ARM         | 2    | 0             | 0          | ARMEABI          | Skeleton     |
| ARM64       | 1    | 0             | 0          | AArch64          | Skeleton     |
| MIPS32      | 4    | 0             | 0          | NONE (silent SystemV fallback) | Skeleton |
| MIPS64      | 0    | 0             | 0          | NONE (silent SystemV fallback) | Skeleton |

### Surprising finding (saved to bd memory)

`default_cc_for_arch` in `native/angr/src/arch/calling_conventions.rs:361`
only matches amd64/x86/arm/arm64 — MIPS falls through the wildcard and
gets `SystemVAMD64`, which uses x86_64 register offsets (RDI=72 etc.).
Any MIPS SimProcedure arg extraction will read garbage. This is latent
because no MIPS integration test runs through that path. Saved as
`bd memory invariant-mips-no-calling-convention`.

### Files changed
- `CLAUDE.md` — added Architecture Support Matrix section
- `.claude/projects/-home-ubuntu-repos-angr/memory/project_core_goal.md` —
  qualified the multi-arch claim, refreshed test-suite section

### bd memories saved
- `invariant-mips-no-calling-convention` — MIPS CC fallback bug
- `arch-support-matrix-2026-05-08` — snapshot of per-arch coverage

### Followups
None opened. Possible future beads if anyone wants to promote an arch:
- Add MIPS32 calling convention (MIPSO32) — required before any MIPS
  binary integration test will work correctly
- Add an x86 pytest integration test using flareon2015_2 or similar
  (currently the only end-to-end x86 coverage is the benchmark, not pytest)
- Add ARM/ARM64 integration tests (would require small Linux binaries)
