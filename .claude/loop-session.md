## Session log: 2026-05-11 — angr-gxhf.3 (MIPS64 LE real-binary integration test)

### Task
Add a MIPS64 binary-driven integration test, completing the gxhf epic
that promotes ARM64 / MIPS32 / MIPS64 from Skeleton to Experimental.

### Approach: inline MIPS64 LE ELF64
Same inline-ELF pattern as test_aarch64_explore_real_elf (commit 5b6f034ba)
and test_mips32_explore_le_real_elf (commit ce8ed0aff). No MIPS64 binaries
ship locally and no cross-compiler available, so the ELF is constructed
inline with struct.pack:
  - ELF64 header (64 bytes) — EI_CLASS=ELF64, EI_DATA=LSB, e_machine=EM_MIPS=0x08,
    e_flags=0x60000000 (EF_MIPS_ARCH_64; N64 ABI implied by EI_CLASS=ELF64).
  - ELF64 PT_LOAD phdr (56 bytes) — field order: p_type, p_flags first.
  - 7 MIPS64 instructions: same encoding as MIPS32 since registers are
    still 5 bits and ADDIU sign-extends to 64-bit on MIPS64.
    ADDIU t0,zero,42 → BEQ a0,t0,+3 → NOP delay → B +2 → NOP delay →
    NOP (found) → NOP (avoid)

Program asserts the engine drives a0 to 42 to hit the found address.

### Verification
- New test (single): PASS (1.68s)
- TestMultiArchSupport (19 tests): all PASS (was 18, +1 new)
- Full suite: 385 passed, 3 failed (same 3 pre-existing failures:
  dcas_cmpxchg16b_no_match, pipe_native_dispatch_creates_two_fds,
  dup2_native_dispatch_redirects_stdin)

### Files modified
- tests/engines/test_rust_exploration.py (+129 lines)
- CLAUDE.md (MIPS64 row Skeleton → Experimental: 2 integration tests
  now; Skeleton-means paragraph reframed since no arch remains in that
  state; "What's wired up but unverified" section updated)

### Commit / bead
- 18850a432 test(arch): MIPS64 little-endian real-ELF integration test
- angr-gxhf.3 closed; angr-gxhf epic auto-closed (all subtasks done).

### Memories saved
- invariant-mips64-inline-elf — MIPS64 ELF64 layout for inline-test ELFs:
  EM_MIPS=0x08, EI_CLASS=2, e_flags=0x60000000 (no separate N64 ABI flag —
  implied by EI_CLASS=ELF64). Base instruction encodings match MIPS32.

### Status
COMPLETE — angr-gxhf.3 closed; angr-gxhf epic complete. MIPS64 promoted
from Skeleton to Experimental. All three non-amd64 64-bit arches
(ARM64, MIPS32, MIPS64) now have real-binary integration test coverage.
