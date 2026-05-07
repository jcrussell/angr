# Loop session notes (2026-05-07, 114th loop session)

## Task: angr-4t2u — NAMING_CONVENTIONS.md (DONE)

### What landed (commit 572fe4585)

- New `native/angr/NAMING_CONVENTIONS.md` (93 lines, well under 200)
  codifying parameter naming for `native/angr/src`:
  - Canonical names: `addr` (u64), `ctx` (&SymContext), `py` (Python),
    `state` (&mut RustSimState), `data` (&[u8]), `bv` (&RustBV
    generic), `value` (&RustBV stored).
  - Method prefixes (`call_*`, `has_*`, `is_*`, `try_*`, `add_*`,
    `set_*`).
  - Local bindings in VEX op handlers (`val`, `vec`, `lo`, `hi`,
    etc.) explicitly allowed.
  - RustBV ownership guidance.
  - Intentional exceptions:
    - `segmentlist.rs` uses `address: u64` because it mirrors the
      angr Python public API in `angr/rustylib/__init__.pyi`.
    - `value: &RustBV` over `bv: &RustBV` in memory store paths
      (semantic role is "value being stored").
    - `bytes: &[u8]` (not `data:`) in lift/decode paths
      (specifically machine-code bytes).

### No rename pass

Codebase audit showed the crate is already consistent:
- `addr: u64` 240 sites vs `address: u64` 8 sites (all in
  segmentlist, intentional).
- `ctx: &SymContext` 166 sites, no `context` or `cx` outliers.
- `py: Python` 102 sites, no `python` outliers.
- `state: &mut RustSimState` 85 sites + 13 owned + 5 borrowed, no
  `st`/`sim_state` outliers.

Outliers like `val: RustBV` (3) live in VEX op handlers
(`vex/ops.rs`, `vex/ccall.rs`) where the doc explicitly endorses
short local names. Not renamed.

### Test results

- 261/261 Python tests pass (no code changed; doc-only commit).
- No Rust changes, so cargo check skipped.

### Memories saved

- `invariant-naming-conventions` — full convention map, including
  intentional exceptions, with pointers to the doc and the public
  Python API rationale.

### Next session candidates

P1 / P2 ready (still unblocked):
- angr-pufm (P1) symbolic concretization fallback — multi-session.
- angr-prem (P2) MemoryLayer trait refactor.
- angr-fk0m (P2) unify state mixin classes (rust_state_sync /
  rust_state_cache / rust_state_export).
- angr-4j5u (P2) decompose 95-field god struct.
- angr-m2hf (P2) unified error trait.
- angr-wqao (P2) split rust_manager.py 2800 lines.

P3 ready:
- angr-3vrj — StateMetadata dataclass (~85 metadata refs across 5
  files; bigger than the bead's "56 sites" estimate).
- angr-ja0b — StepOutcome trait (rejected this session as the
  proposed trait design doesn't fit the actual variant shapes;
  see analysis below).
- angr-khth — init pipeline phase split (init pipeline already
  fairly modular: `_run_python_init_if_needed`,
  `_try_in_memory_init_cache`, `_try_disk_init_cache`,
  `_load_init_from_disk_cache`, `_save_init_to_disk_cache`,
  `_apply_state_metadata`).
- angr-csd1 — split PythonCallbacks into focused traits (touches
  48 callsites across 12 Rust files).
- angr-x3xu — SolverBridge protocol (note: bead description
  overstates RustSolverContext coupling; manager mostly calls
  state.solver.eval, not RustSolverContext directly).
- angr-qrhl — VEX op trait dispatcher (needs perf measurement).
- angr-czph, angr-bkcs, angr-n28w — feature work.

### Notes on angr-ja0b (skipped)

The current StepError has 4 variants with very different shapes:
- NeedCallback(PendingCallback) — rich, drives the run loop's
  callback-dispatch flow with find/avoid handling and event
  construction (~180 lines in mod.rs:2752–2929).
- Deadended(state) / Error(state, msg) / Unconstrained(state) — each
  one-liner: push to terminal stash.

A trait with `action() -> Action` + `state() -> &RustSimState` would
push the rich NeedCallback handling either into the outcome's
`action()` (bloats the trait impl with run-loop knowledge) or back
into the run loop's match arm (defeats the trait's purpose). The
current pattern matching is the natural Rust expression.
