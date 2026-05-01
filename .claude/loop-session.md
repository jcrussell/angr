# Loop session notes (2026-05-01, tenth session)

## Closed this session

### angr-l85c — Fix vex-engine-only (no z3) build

Final feature-flag build fix. Pre-fix: 43 cargo-check errors when
building `--no-default-features --features "vex-engine"`. All E0599
"no method named X found" — the interpreter and exploration code call
`assume_true`, `assume_false`, `check_branch_feasibility`,
`set_timeout` unconditionally, but those methods are
`#[cfg(feature = "vex-engine-z3")]`-gated.

Fix: added 4 no-z3 stubs in
`native/angr/src/symbolic/context.rs` near the existing
"Mock implementations when Z3 is not available" block (around lines
1611-1637). Followed the established pattern (`push`/`pop`/`is_sat`/
`can_be_true`/`eval` etc. all already had no-z3 mocks).

Stub semantics:

- `assume_true` / `assume_false`: still push to
  `assumed_constraints_local` so `get_assumed_constraints()` (the
  Python export path) sees the constraint. That field is *not*
  z3-gated by design — it backs both feature combos.
- `check_branch_feasibility`: concrete fast-path
  `(v != 0, v == 0)`; symbolic returns `(true, true)` to match the
  existing `can_be_true`/`can_be_false` stubs.
- `set_timeout`: no-op.

**Verification:**
- `cargo check --no-default-features --features ""` clean
- `cargo check --no-default-features --features "automaton"` clean
- `cargo check --no-default-features --features "vex-engine"` clean
  (was 43 errors)
- `cargo check --no-default-features --features "vex-engine,vex-engine-z3"` clean
- `cargo check --release` (default) clean
- `cargo test --no-default-features --features "vex-engine"` 346/346
- `cargo test --no-default-features --features ""` 44/44
- `cargo test --no-default-features --features "automaton"` 59/59
- `python -m pytest tests/engines/test_rust_exploration.py` 208/208
- `run_single.py fauxware --engine rust` recovers SOSNEAKY

**Commit:** 97cf549ea (1 file, +29 −0)

**Memory saved:** `invariant-no-z3-stub-export` — captures the
non-obvious requirement that no-z3 stubs of assume_true/assume_false
still push to the un-gated `assumed_constraints_local` so the Python
export path doesn't silently lose constraints.

## Unblocked

- **angr-7c9j** (CI feature-flag correctness tests) — was waiting on
  both angr-asth and angr-l85c. Both now closed; ready to pick up.

## Ready P-tasks remaining

- angr-vt0t (P3 categorize remaining ~280 except blocks)
- angr-8em4 (P3 panic audit — 543 sites, must split)
- angr-3ijo (P3 bincode for IRSB serialization spike)
- angr-bgv0 (P3 Z3 FP theory)
- angr-awm3 (P3 CAS/LLSC statement handling)
- angr-dja4 (P3 expand benchmark baseline)
- angr-v4db (P3 extract god-methods)
- angr-7c9j (P3 feature-flag correctness in CI — now unblocked!)
