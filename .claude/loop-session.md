## Session log: 2026-05-10 — angr-25za (214th loop session, COMPLETE)

### Task
SSOT: make Rust the source of truth for register widths and names; delete
Python duplicates.

Targets:
- `RustRegisterProxy._get_register_width` (rust_state_proxy.py:291-323) —
  ~30 lines of hardcoded prefix matching for AMD64/X86/ARM/AARCH/MIPS.
- `_supported_register_names` (rust_state_sync.py:97-129) — hardcoded
  per-arch register lists.

### Implementation

**Rust (`native/angr/src/engine.rs`):** added two `#[pyfunction]`s
exposing the existing `Arch::register_size` / `Arch::register_names`
(arch/mod.rs:64,70):

- `register_size_for_arch(arch_name, reg_name) -> Option<u32>` — bytes
  (caller multiplies for bits).
- `register_names_for_arch(arch_name) -> Vec<String>` — canonical names.

Both use `arch_from_name` (case-insensitive; accepts angr-style
"AMD64"/"AARCH64"/"ARMEL"/"MIPS32" etc.).

**Python:**

- `RustRegisterProxy._get_register_width` (rust_state_proxy.py): 30 lines
  of prefix-matching → 4-line FFI call with `arch.bits` fallback for
  registers Rust doesn't model.
- `_supported_register_names` (rust_state_sync.py): per-arch hardcoded
  lists → one-line `register_names_for_arch(arch.name)` delegation.

### Behavior change worth noting

`_supported_register_names` used to return a deliberate GP-only subset
(17 names for AMD64: rax..r15+rip). The Rust canonical list is wider
(43 names for AMD64: adds cc_op/cc_dep1/cc_dep2/cc_ndep, dflag, acflag,
idflag, fs_const, gs_const, sseround, xmm0..15). The slow sync path
now does `getattr()` for each, which on `blank_state` triggers angr's
lazy-fill warnings for those previously-skipped registers. `entry_state`
callers unaffected (those regs zero-init at state creation). 373/373
tests at parity (3 pre-existing dcas/pipe/dup2 failures unchanged).

### Verification

- `cargo check --release` — clean.
- Rebuild via `tools/rebuild-rust.sh --cargo-only` (venv pip broken;
  see avoid-pip-install-broken-venv memory).
- Smoke test: `register_size_for_arch` correctly returns 8 for rax,
  4 for eax, 1 for al, 16 for xmm0, 8 for x0 (aarch64), 4 for r0
  (armel), 4/8 for a0 (mips32/mips64). Bogus names → None.
  All angr archinfo names ('AMD64','X86','AARCH64','ARMEL','ARMHF',
  'ARM','MIPS32','MIPS64') resolve to the correct register count.
- `pytest tests/engines/test_rust_exploration.py`: 370 passed,
  3 pre-existing failed (dcas/pipe/dup2). Zero regressions.
- `tests/benchmarks/run_regression.py`: 12/12 passed in 18.6s.
- fauxware single bench: rust 0.28s, python 0.38s — no regression.

### Files changed

- `native/angr/src/engine.rs` (+27 lines, two PyO3 functions).
- `angr/exploration/rust_state_proxy.py` (+import, -25 LOC body).
- `angr/exploration/rust_state_sync.py` (+import, -23 LOC body).

Commit: `e71ad806d refactor(rust-symex): SSOT for register widths/names via PyO3 — angr-25za`

### Memories saved

- `invariant-supported-register-names-now-rust-canonical` —
  documents the wider canonical list now returned (43 names AMD64),
  blank_state-vs-entry_state side effect on filler warnings, and
  where to add new registers (per-arch CANONICAL slice).
- `avoid-pip-install-broken-venv` — venv's pip is broken (vendored
  resolvelib ImportError); use `tools/rebuild-rust.sh --cargo-only`
  to rebuild the .so directly via cargo+copy.

### Closed beads

- `angr-25za`.

### Status

COMPLETE.
