## Session log: 2026-05-09 — angr-zjr7 (189th loop session, in progress)

### Task
Per-arch register table macro: const REGISTERS array + impl_arch_registers.
Each arch in native/angr/src/arch/{amd64,x86,arm,arm64,mips}.rs has 3 match
statements (register_offset/register_size/register_name) that duplicate
register names. Replace with a per-arch CANONICAL/ALIASES const table +
shared lookup helpers, so each register is listed once with its (offset,
size).

### Approach
Plan:
1. Add lookup helpers in arch/mod.rs:
   - lookup_register_offset(name, canonical, aliases) -> Option<u32>
   - lookup_register_size(name, canonical, aliases) -> Option<u32>
   - lookup_register_name(offset, canonical) -> Option<&'static str>
2. For each arch: define CANONICAL and ALIASES const slices of (name,
   offset, size) tuples. Replace the 3 match-statement methods with
   one-liner calls to the lookup helpers.
3. Preserve register_names() behavior exactly (separate REGISTER_NAMES
   const slice — small redundancy with CANONICAL).
4. Preserve current bug-for-bug behavior of MIPS64 (incomplete
   register_name) by putting only the originally-mapped registers in
   CANONICAL, the rest go in ALIASES.

### Files
- native/angr/src/arch/mod.rs  (add helpers)
- native/angr/src/arch/amd64.rs (rewrite using table)
- native/angr/src/arch/x86.rs (rewrite using table)
- native/angr/src/arch/arm.rs (rewrite using table)
- native/angr/src/arch/arm64.rs (rewrite using table)
- native/angr/src/arch/mips.rs (rewrite using table)

### Status: implementing
