## Session log: 2026-05-11 — angr-gxhf.2 (MIPS32 LE real-binary integration test)

### Task
Add a MIPS32 binary-driven integration test that exercises LE end-to-end
through the interpreter (BE was already covered by the existing blob test).
Also exercise cle's ELF loader on MIPS (previously only Blob-tested).

### Approach: inline MIPS32 LE ELF32
Followed the AArch64 inline-ELF pattern from the previous session
(test_aarch64_explore_real_elf, commit 5b6f034ba). No MIPS LE binaries
ship locally and no cross-compiler available, so the ELF is constructed
inline with struct.pack:
  - ELF32 header (52 bytes) — EI_DATA=LSB, e_machine=EM_MIPS=0x08,
    e_flags=EF_MIPS_ARCH_32 | EF_MIPS_ABI_O32
  - ELF32 PT_LOAD phdr (32 bytes) — field order differs from ELF64!
  - 7 MIPS32 instrs: ADDIU t0,zero,42 → BEQ a0,t0,+3 → NOP delay →
    B +2 → NOP delay → NOP (found) → NOP (avoid)

Program asserts the engine drives a0 to 42 to hit the found address.

### Initial failure: wrong target addresses
First test run got `found=0 / deadended=2 / Lift error at 0x0`. Cause:
used `find=BASE+0x14` but ELF code lives at vaddr = BASE + EHDR_SIZE +
PHDR_SIZE = BASE + 0x54 (NOT BASE — Blob places code at base_addr but
ELF starts code at file offset after ehdr+phdr). Fixed to ENTRY+0x14.
Saved as memory `avoid-inline-elf-base-addr-confusion`.

### Verification
- Single-test pytest: PASS (1.50s)
- TestMultiArchSupport class: 18/18 PASS (was 17, +1 new)
- Full suite: 384 passed, 3 failed (same 3 pre-existing failures:
  dcas_cmpxchg16b_no_match, pipe_native_dispatch_creates_two_fds,
  dup2_native_dispatch_redirects_stdin)

### Files modified
- tests/engines/test_rust_exploration.py (+121 lines)
- CLAUDE.md (MIPS32 integration column 2 → 3; endianness note updated
  to reflect BE+LE end-to-end coverage)

### Commit / bead
- ce8ed0aff test(arch): MIPS32 little-endian real-ELF integration test
- angr-gxhf.2 closed.

### Memories saved
- invariant-mips32-inline-elf-opcodes — MIPS32 instruction encodings
  (ADDIU, BEQ, B, NOP) + ELF32 header/phdr layout + EM_MIPS / e_flags
  values for inline-ELF tests
- avoid-inline-elf-base-addr-confusion — branch targets are relative to
  ENTRY (BASE + EHDR_SIZE + PHDR_SIZE), not BASE — Blob is direct but
  ELF has a header in front

### Status
COMPLETE — angr-gxhf.2 closed; MIPS32 promoted from "BE-only blob coverage"
to "BE+LE end-to-end with ELF loader path exercised". Remaining gxhf
subtask: gxhf.3 (MIPS64 binary-driven test).
