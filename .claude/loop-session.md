# Loop session notes (2026-05-03, thirty-sixth loop session)

## Task: angr-4dxi — CLOSED
"Add memory permission enforcement in Rust memory model"

## Outcome
Per-page R/W permission enforcement now wired into Rust SymbolicMemory
load/store paths. Off by default to mirror angr's STRICT_PAGE_ACCESS
option semantics (opt-in, no behavior change for existing callers).
Commit 3894f7060.

## What changed
- **native/angr/src/memory.rs**
  - New `enforce_permissions: bool` field on SymbolicMemory (default false).
  - New `set_enforce_permissions` / `enforce_permissions` accessors.
  - New `check_perms_range(start, end, required) -> Result<(), MemoryError>`
    helper. No-op when flag is off. Returns
    `MemoryError::Permission { addr, required, actual }`.
  - Permission checks injected into the 3 inner helpers:
    `load_concrete`, `load_concrete_lazy_inner`, `store_concrete`.
    All other load/store APIs route through one of these, so checks
    pick up transitively across symbolic/lazy/automap variants.
  - New `Permission::W` and `Permission::X` constants for single-bit
    checks (store needs W only, not RW; matches Python angr behavior).
  - Field copied through `fork()`.
  - 6 new unit tests: default-off, R-write blocked, W-read blocked,
    RWX allowed, cross-page violation, fork propagation.
- **native/angr/src/state.rs**
  - `set_enforce_permissions` / `enforce_permissions` on inner `State`.
  - `py_set_enforce_permissions` / `py_enforce_permissions` exposed on
    pyclass (`#[pyo3(name = "set_enforce_permissions")]`).

## Verification
- `cargo test --release --lib memory::` — 12/12 pass (6 new + 6 existing).
- 208/208 RustExplorationManager tests pass.
- fauxware single benchmark sanity-checked: still runs cleanly with the
  default (off) flag.

## Memories saved
- `invariant-mem-perm-default-off`
- `invariant-mem-perm-check-points`
- `memory-perm-rust-vs-python`

## Potential follow-up (not done in this session)
- Plumb a Python-side hook that flips the flag when STRICT_PAGE_ACCESS
  is in `state.options`, so the Rust engine matches Python's behavior
  automatically. Currently the flag must be flipped manually via the
  exposed API.
- Add execute-permission check on basic-block fetch (Rust interpreter
  step path). Currently the X bit is stored but only checked if the
  caller wraps `check_perms_range(.., Permission::X)` themselves. This
  task explicitly listed NX violations as not detected; the fetch path
  is in interpreter_cb/* and would need its own injection point.
- Decide whether to default the flag ON for "real" angr Project flows
  (per-state via the manager) once Python-side STRICT_PAGE_ACCESS
  detection is wired up.
