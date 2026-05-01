# Loop session notes (2026-05-01, fifth session)

## Closed this session

### angr-gste — Remove dead code — commit 1992563dd

Removed 8 dead items, all `#[allow(dead_code)]` and `#[warn(unused)]`
warnings now resolved:

Original 5 of 6 from the task description:
- exploration/stepping.rs: `step_state()` wrapper
- vex/ops.rs: `round_f32_to_int()`, `round_f64_to_int()`
- memory.rs: `PendingWrite::compute_page_hint()`
- automaton/python_bindings.rs: `ObjectMapper::get_state_by_id()`
- exploration/mod.rs: `NativeProcStats::constraint_sync_failures`

Cascading dead code also removed:
- symbolic/value.rs: `concrete_base_and_sym_width()` (only caller was compute_page_hint)
- vex/pyvex_bridge.rs: unused module-level imports `Endness/IROp/JumpKind`
  (IROp/JumpKind kept in test scope where needed)
- exploration/helpers.rs: `push_to_active_or_drop()` (see below)

Skipped: vex/dirty.rs `CpuidValues` ebx/ecx fields — design intent
preservation. Documents CPUID feature flags for future complete emulation.

### Major latent issue surfaced — angr-jmiz (created P2 bug)

`exploration/resume.rs` and `exploration/run_loop.rs` claim in their
header comments to be "included into mod.rs" but mod.rs only declares
`mod stepping` and `mod helpers`. **Those files are never compiled.**

Consequence: `push_to_active_or_drop` (the only enforcement of
`max_active_states`) was called only from those dead files. The live
mod.rs pushes raw `push_back(STASH_ACTIVE)` everywhere — so
`set_max_active_states()` silently does nothing.

Same pattern as `pyapi.rs` (see `invariant-pyapi-dead-code` memory).

Saved memories:
- `invariant-dead-source-files` — list of dead files + how to detect.
- `bug-max-active-states-unenforced` — concrete latent bug for follow-up.

Tests: 207/207 Python passing; 364/364 cargo lib tests; cargo release
build warning-free.

## Closed previously

### angr-742d — Register accessor macros — commit a25cdc3ef
Replaced hand-rolled VexArch match with Arch trait methods. Latent
SP-offset bug for ARM/ARM64 fixed silently.

## Ready P-tasks remaining

- angr-jmiz (P2 — NEW: dead source-files audit + max_active_states fix)
- angr-vt0t (P3 categorize remaining ~280 except blocks)
- angr-8em4 (P3 panic audit — 543 sites)
- angr-3ijo (P3 bincode for IRSB serialization spike)
- angr-bgv0 (P3 Z3 FP theory)
- angr-awm3 (P3 CAS/LLSC statement handling)
- angr-7c9j (P3 feature flag correctness in CI)
- angr-dja4 (P3 expand benchmark baseline)
- angr-sc8h (P3 replace solver fallback monkey-patching)
- angr-wpi7 (P3 consolidate P1-P19/GAP fix workarounds)
- angr-v4db (P3 extract god-methods)
