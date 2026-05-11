## Session log: 2026-05-11 — angr-bkcs.1 (NEON Q-register scaffolding)

### Task
Land scaffolding-only step of angr-bkcs (parent ARM/AArch64 NEON SIMD
epic, auto-deferred after 3 dirty iterations). Need: register defs in
arch/arm.rs + arm64.rs, IROp variants for NEON ops, opcode_map entries
routing through dispatch, dispatch sites panic explicitly instead of
silently falling back. No actual op implementations.

### Surprise: arch defs already in place
arm.rs already had D0..D31 + Q0..Q15. arm64.rs already had Q0..Q31 +
V0..V31 aliases + D0..D31 (lower half). Past dirty iterations must have
failed downstream (op implementations); register tables landed already.

### Approach
Instead of N new IROp variants (one per NEON family), one variant:

  IROp::NeonUnimplemented(&'static str)

The &'static str captures the original Iop name (e.g. "Iop_Dup8x8") so
the panic message points at the exact opcode. Single arm in
VEXOps::{unop, binop, ternop, qop} panics with "NEON op X not yet
implemented". opcode_map.rs::parse_neon_unimplemented() enumerates
~150 NEON-only Iop names across Dup, Narrow, QNarrow, Widen,
Get/SetElem, RecipEst/Step, RSqrtEst/Step, QAdd/QSub, Avg, Reverse,
Pw{Add,Min,Max,AddL}, Polynomial(l)Mul, Cnt/Clz/Cls, Sh{l,r,a,al}x*,
QSh{l,al}x*. Hooked into parse_opcode() BEFORE the Raw(0) fallback.

### Tests added
- arch/arm.rs::test_neon_q_and_d_registers — locks D/Q overlap
- arch/arm64.rs::test_neon_q_registers — locks V/Q alias + D lower half
- vex/opcode_map.rs::test_neon_unimplemented_routing — opcode -> variant
- vex/opcode_map.rs::test_neon_does_not_shadow_existing_mappings —
  confirms Iop_Add8x8 / Iop_ShlN32x4 / Iop_CmpEQ32Fx4 still hit their
  existing handlers (sanity that the scaffold did not regress coverage).

### Build / test
- cargo check --release: clean
- cargo clippy: 1 pre-existing error in exploration/pending_api.rs:293
  (unrelated, verified by git stash)
- cargo test --release --lib: 617/617 passing
- Python suite: 382/385 (same 3 pre-existing failures —
  dcas_cmpxchg16b_no_match, pipe_native_dispatch_creates_two_fds,
  dup2_native_dispatch_redirects_stdin)
- Rebuild via tools/rebuild-rust.sh --cargo-only because venv pip is
  broken (pip._vendor.resolvelib import error)

### Commit / bead
- 48f949fde feat(rust-symex): scaffold ARM/AArch64 NEON SIMD opcodes
- angr-bkcs.1 closed. Sibling bkcs.2 implements ops one-by-one by
  removing the opcode_map entry and adding a real IROp variant.

### Memories saved
- invariant-neon-scaffolding-panic-not-fallback — design rationale,
  how bkcs.2 should peel ops off this scaffold
- bkcs1-arch-defs-already-present — surprise that arch defs were
  pre-landed; grep before duplicating
- avoid-broken-venv-pip-rebuild — workaround for the current broken
  pip; use rebuild-rust.sh --cargo-only

### Status
COMPLETE — angr-bkcs.1 closed.
