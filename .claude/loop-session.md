## Session log: 2026-05-10 — angr-gzk8 (208th loop session, COMPLETE)

### Task
Fix or guard MIPS64 calling-convention silent fallback to SystemV_AMD64.
`default_cc_for_arch` was returning `SystemVAMD64` for unrecognized
arches (including MIPS64), meaning a MIPS64 SimProcedure would read
its first arg from AMD64 RDI (offset 72) instead of MIPS $a0
(offset 48). Same risk class as the latent x86 Cdecl return-register
bug (commit 5329d8222) that hid for months.

### Resolution (preferred path: implement MipsN64)
Added `MipsN64` calling convention to `native/angr/src/arch/calling_conventions.rs`:
- N64 ABI: 8 integer args in $a0-$a7 (R4-R11, VEX MIPS64 offsets
  48, 56, 64, 72, 80, 88, 96, 104)
- Return register: $v0 (R2, offset 32)
- Return addr: $ra (R31, offset 264) — register-based, no SP pop
- `stack_arg_offset = 0` (N64 does NOT reserve a save area for register
  args, unlike O32's 16-byte window)
- Aliases: `mips64`, `mips64el`, `mips64le`, `mips64be`

Defense-in-depth: changed `default_cc_for_arch` to **panic** on
unknown arches instead of silently falling back to SystemV_AMD64.
The exploration-manager construction path already gates on
`arch_from_name` which only accepts the 6 supported arches, so the
panic is a backstop, not user-facing.

### Tests added
- `test_mips_n64_arg_registers` (Rust) — locks the N64 offsets/values
- `test_default_cc_for_arch_unknown_panics` (Rust) — guards the loud-failure invariant
- Updated `test_default_cc_for_arch_registry` to cover all 4 mips64 aliases
- Added `MIPS_N64` to `test_arch_aliases_disjoint`
- `test_mips64_native_procedure_round_trip` (Python) — end-to-end native
  strlen with N64 args/return registers

### Results
- Rust unit tests: 11/11 CC tests pass (added 2)
- Python integration: 370/370 pass (added 1, was 369)
- Smoke test confirmed: `RustExplorationManager('ppc')` raises
  `ValueError` (caught earlier at `arch_from_name`);
  `RustExplorationManager('mips64')` succeeds.

### Files modified
- `native/angr/src/arch/calling_conventions.rs` (added MipsN64 + panic;
  updated 3 tests + added 2)
- `tests/engines/test_rust_exploration.py` (added MIPS64 native-proc
  round-trip test; refreshed stale MIPS32 comment)

### Status
COMPLETE. Unblocks angr-gxhf.3 (MIPS64 binary-driven integration test).
