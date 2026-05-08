## Session log: 2026-05-08, 169th loop session

### Task: angr-4j5u.2 (closed) — Extract ConstraintSolver + ConstraintTracker

Second child of the angr-4j5u decomposition epic. Group six constraint-
related fields off `RustExplorationManager` into two sub-structs in a new
`constraints.rs` module.

### What changed

**New: native/angr/src/exploration/constraints.rs (~50 lines)**

- `ConstraintSolver`: solver knobs propagated to per-state solvers.
    - `lazy_solves: bool`
    - `solver_timeout_ms: u32` (default 30000 via `new()`)
- `ConstraintTracker`: per-run bookkeeping.
    - `uniqueness_registers: Vec<String>`
    - `uniqueness_set: HashSet<u64>`
    - `skip_find_predicate_states: HashSet<u64>`
    - `skip_avoid_predicate_states: HashSet<u64>`
    - `#[derive(Default)]` only.

Both use `pub(crate)` direct fields. Same load-bearing-simplicity pattern
as ProfilingCollector — see memory `invariant-constraint-substructs-direct-fields`.

**native/angr/src/exploration/mod.rs**
- New `mod constraints;` + `use self::constraints::{ConstraintSolver, ConstraintTracker};`.
- Removed six standalone field declarations on `RustExplorationManager`,
  replaced with two sub-struct fields.
- Constructor: six fields collapse to
  `constraint_solver: ConstraintSolver::new()` +
  `constraint_tracker: ConstraintTracker::default()`.
- ~28 callsites updated mechanically.

**native/angr/src/exploration/stepping.rs**
- 5 `self.lazy_solves` → `self.constraint_solver.lazy_solves` replacements.
- Includes the `interp.lazy_solves = self.lazy_solves` propagation site.

**native/angr/src/exploration/helpers.rs**
- 3 `self.uniqueness_*` → `self.constraint_tracker.uniqueness_*` replacements.

### Field count on the manager: 40 → 36 (net -4)

Six fields collapsed to two sub-struct fields.

### Build hiccup (recurring, same as 4j5u.1)

`pip install -e . --no-build-isolation --no-deps` still doesn't work because
`.venv/bin/pip` is missing. Used the cargo + cp fallback per
`avoid-broken-venv-pip-fallback-cargo-build`:

```
cargo build --manifest-path native/angr/Cargo.toml --release
cp -f target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so
```

Also, `run_single.py` subprocess can't import angr without
`PYTHONPATH=/home/ubuntu/repos/angr` prefix. Multiprocessing.spawn inherits
sys.executable but not cwd, and the editable install's `.pth` file is
missing (only `__editable_*.pyc` survives). Pre-existing venv issue, not
caused by the refactor.

### Final state

- 342/342 tests passing on tests/engines/test_rust_exploration.py (19.4s)
- fauxware: OK rust 0.35s peak_mem=188MB
  (matches 4j5u.1 baseline of 0.34s/188MB within noise)
- Commit: `4e66c8d13` on rust-symex

### Memories saved

- `invariant-constraint-substructs-direct-fields` — same direct-field
  pattern as ProfilingCollector; do not add methods as cleanup.

### Next children in the angr-4j5u sequence

- 4j5u.3 — Extract MemoryConfiguration struct
- 4j5u.4 — Extract ExecutionEnvironment struct
- 4j5u.5 — Consolidate orchestrator (final pass)
