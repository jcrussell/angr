# Loop session notes (2026-05-01, fourth session)

## Closed this session

### angr-742d — Register accessor macros (offset/size lookup) — commit a25cdc3ef

Replaced two hand-rolled `match VexArch { ... => (offset, size) }`
tables with Arch trait method calls.

- Added `syscall_num_offset() -> Option<u32>` to Arch trait
  (default impl returns None).
- Implemented for AMD64 (RAX), X86 (EAX), ARM (R7/EABI),
  ARM64 (X8), MIPS32 (v0), MIPS64 (v0).
- exits.rs `get_syscall_num()` now uses
  `self.registers.arch().syscall_num_offset()` + `arch.bytes()`.
- prefetch.rs `get_stack_pointer()` now uses `arch.sp_offset()`
  + `arch.bytes()`.

Latent bug fixed: prefetch.rs's hand-coded SP table had wrong
offsets for ARM (52 vs actual 60) and ARM64 (52 vs actual 264).
This meant `is_stack_region()` never matched real ARM/ARM64
stacks — stack prefetching was effectively disabled on those
archs. Routing through the trait fixes it silently.

Tests: 207/207 passing. cargo test arch:: 21/21 passing.

Saved memories:
- `prefetch-sp-offset-bug` — concrete details of the latent bug.
- `invariant-arch-register-offsets` — guidance to always go
  through Arch trait methods, never duplicate the VexArch match.

## Closed previously

(see git log / earlier loop-session.md backups)

## Ready P3 tasks remaining

- angr-vt0t (categorize remaining ~280 except blocks)
- angr-8em4 (panic audit — 543 sites)
- angr-3ijo (bincode for IRSB serialization spike)
- angr-bgv0 (Z3 FP theory)
- angr-awm3 (CAS/LLSC statement handling)
- angr-7c9j (feature flag correctness in CI)
- angr-dja4 (expand benchmark baseline)
- angr-sc8h (replace solver fallback monkey-patching)
- angr-wpi7 (consolidate P1-P19/GAP fix workarounds)
