# Loop session notes (2026-05-01, eleventh session)

## Closed this session

### angr-7c9j — Add feature flag correctness tests to CI

Final feature-flag CI follow-up. Pre-fix: `rust_feature_flags` matrix
in `.github/workflows/nightly-ci.yml` only ran `cargo check` +
`cargo test` per combo. Compile/unit-test coverage existed, but no
named test exercised each combo's primary public API end-to-end —
so a feature-gating regression that compiled and passed unit tests
but broke the public surface (cf. angr-asth, angr-l85c) could land
silently.

Fix: added `native/angr/tests/feature_flag_smoke.rs` with
feature-gated tests:

| Combo                          | Smoke tests                                            |
|--------------------------------|--------------------------------------------------------|
| `""`                           | concrete arithmetic, symbolic handle construction (2)  |
| `automaton`                    | + DFA build/transition (3)                             |
| `vex-engine`                   | + VEX IR public surface (Endness::Little/Big)  (3)     |
| `vex-engine,vex-engine-z3`     | + Z3 solve `x == 0xDEADBEEF` (4)                       |
| default (release)              | all of the above (5)                                   |

`cargo test` already picks these up; `nightly-ci.yml` gains an
explicit `cargo test --test feature_flag_smoke` step per matrix
entry so the named "Behavioral smoke test" is searchable in CI logs.

**Verification:**
- `cargo test --no-default-features --features ""` → 44 unit + 2 smoke
- `cargo test --no-default-features --features automaton` → 59 + 3
- `cargo test --no-default-features --features vex-engine` → 346 + 3
- `cargo test --no-default-features --features vex-engine,vex-engine-z3` → 349 + 4
- `cargo test --release` (default) → 364 + 5
- `python -m pytest tests/engines/test_rust_exploration.py` → 208/208
- `run_single.py fauxware --engine rust` → SOSNEAKY recovered, 0.35s

**Commit:** f3f3c19e3 (2 files, +94 −0)

**Memories saved:**
- `invariant-feature-flag-smoke` — what each combo's smoke test
  exercises, how to extend when adding combos.
- `gotcha-endness-variants` — `rustylib::vex::ir::Endness` variants
  are `Little`/`Big`, not `LE`/`BE` (cost one rebuild).
- `benchmark-feature-flag-test-counts` — per-combo counts as
  baseline for detecting missing feature-gated tests.

## Ready P-tasks remaining

After this session the nightly feature-flag matrix has end-to-end
behavioral coverage in addition to unit tests. Remaining ready
items (`bd ready`):

- angr-vt0t (P3 categorize remaining ~280 except blocks)
- angr-8em4 (P3 panic audit — 543 sites, must split)
- angr-3ijo (P3 bincode for IRSB serialization spike)
- angr-bgv0 (P3 Z3 FP theory)
- angr-awm3 (P3 CAS/LLSC statement handling)
- angr-dja4 (P3 expand benchmark baseline)
- angr-v4db (P3 extract god-methods)
- angr-4dxi (P4 memory permission enforcement)
- angr-borb (P4 StateId / Address newtypes)
