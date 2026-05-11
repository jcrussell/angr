## Session log: 2026-05-11 — angr-34w.30 (init overhead, CLOSED as misdiagnosis)

### Task
Reduce add_state init overhead via direct Rust state population. Bead claimed
RustSimState + add_state + get_state_ids takes ~13ms on first call and a
`create_and_add_state()` API would save ~5ms per init.

### Investigation
Measured the actual cost of each phase in `_add_rust_state` on fauxware:

WARM path (entry_state with disk-cached init): TOTAL=1.2ms
  symbolic_pages (extract+import):  0.95ms  (79% — the real bottleneck)
  concretize_stack:                 0.10ms
  sync_memory:                      0.03ms
  sync_regs:                        0.02ms
  new_rustsim + add + 2 get_ids:    0.03ms  (bead's target = 2.2%)

COLD path (blank_state, no cache): TOTAL=25-37ms
  _concretize_stack_registers:     11.9ms  (solver.eval on rsp/rbp)
  _sync_registers_to_rust:          6.9ms
  _sync_memory_to_rust:             5.6ms
  symbolic_pages:                   0.6ms

Even "first call" Z3 ctx init inside RustSimState construction is 9us total,
not 13ms. The bead was off by ~100x.

### Decision
Closed the bead with detailed rationale. No code change.

### Memories saved
- `add-state-init-breakdown-2026-05-11`: full warm/cold breakdown

### Follow-up beads filed
- `angr-l9h7` (P3): Profile _extract_symbolic_pages / import_symbolic_to_state
  (the 0.95ms warm-path bottleneck).
- `angr-sgbn` (P3): Profile _concretize_stack_registers (12ms cold-path,
  largest single phase for non-cached states).

### Closed beads
- `angr-34w.30` (with reason explaining the measurement evidence).

### Files changed
None (source). Only .claude/loop-session.md.

### Status
COMPLETE — bead closed as misdiagnosed, follow-ups filed with real targets.
