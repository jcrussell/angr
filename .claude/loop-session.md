# Loop session notes (2026-05-07, 132nd loop session)

## Task: angr-qrhl — Trait-based VEX op dispatcher

### Status: closed (deferred after audit)

### Outcome
Deferred per the same template as angr-prem / angr-csd1 / angr-m2hf /
angr-wqao / angr-ja0b — large abstraction, weak fit, no motivating bug,
hot-path performance risk.

Memory saved: avoid-deferred-qrhl-vex-trait-dispatcher

### Audit summary
- vex/ops.rs has 5 #[inline] dispatch entry points; 4 hot-path call sites
  in interpreter_cb/expressions.rs.
- IROp enum has 224 variants, MOST carrying an IRType parameter that
  current arms forward to width_binop!/width_unop! macros (compile to
  direct RustBV intrinsics). A trait keyed on op kind cannot access
  the IRType without re-matching the variant, defeating the abstraction.
- Acceptance "dispatch overhead within 5%" is structurally hard — match
  jump-tables beat Box<dyn>+HashMap.
- "Adding a new op" workflow has the same touch-point count either way
  (still need IROp variant + opcode_map entry); trait rehomes work
  rather than reducing it.
- No concrete external-crate plugin consumer; no bug class motivates.
- Recent VEX additions (angr-n28w transcendentals, angr-3ekz packed FP
  cmp) used match arms successfully — the exact workflow the trait
  would complicate.

### What follows
Open beads (5 remaining):
- angr-bkcs P3 NEON SIMD
- angr-czph P3 lazy LOAD
- angr-qh5u P3 lazy STORE
- angr-0z34 P4 read/write syscalls (blocked on state-cache sync)

NEON or lazy-load are the substantive next options, both multi-session
design efforts.
