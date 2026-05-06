# Loop session notes (2026-05-06, eighty-second loop session — DONE)

## Status: COMPLETE — angr-eygl closed

## Task: angr-eygl (P1) — Differential test harness: side-by-side Python vs Rust step diffing

### What landed
- `tests/benchmarks/diff_state.py` — new module:
  - `_capture_state(state)` — cheap signature: addr, regs (concretized), constraint count,
    satisfiability, history depth.
  - `_capture_manager(manager, step_idx)` — snapshots all stashes
    (active/found/deadended/avoid/errored/unconstrained).
  - `install_snapshotter(manager, snapshots, interval, max_snapshots)` — wraps
    `manager.step` (via `types.MethodType` so HookSet sees a bound method) and
    appends a snapshot after every Nth call.
  - `_patch_rust_explore_to_single_step` — replaces RustExplorationManager's
    `.explore` and `.run` with a step(1) loop, since Rust's normal explore
    batches in C and never goes through `.step()`.
  - `_align_snapshots` — skips past leading snapshots until both engines'
    first active state is at the same address. Needed because Rust auto-steps
    _start->main during construction (`_run_python_init_if_needed`).
  - `compare_snapshots` — produces a divergence report.
- `tests/benchmarks/run_single.py`:
  - `--diff-state`, `--diff-interval`, `--diff-max-snapshots` CLI flags.
  - `_run_in_child` accepts/threads diff-state args; for python engine it
    monkey-patches `simulation_manager` to install the snapshotter on the
    SimulationManager that solve.py constructs.
  - Returns snapshots in the result dict; `_run_diff_state` orchestrates
    py + rust runs and prints the diff.

### Verified on 6 benchmarks (AC: ≥5)
- fauxware            DIFF FAIL  (real stack/reg divergence at main due to Rust auto-init)
- defcamp_r100        DIFF FAIL  (full_init_state, similar auto-init divergence)
- ais3_crackme        DIFF FAIL
- google2016_unbreakable_0  DIFF OK
- google2016_unbreakable_1  DIFF FAIL
- flareon2015_2       DIFF FAIL

### Known limitation
- Examples that use callable find/avoid predicates (e.g. csgames2018, sym-write)
  fail on the Rust side because `_patch_rust_explore_to_single_step` doesn't
  invoke the predicates during the step loop. Documented as next-up bead.

### Why divergences are real, not harness bugs
- `RustExplorationManager._run_python_init_if_needed` runs Python through
  _start->__libc_start_main->main during construction (180ms savings,
  cached on disk). Python engine starts at _start. After `_align_snapshots`
  Python catches up to main, but the auto-init path produces different
  stack/reg layout than Python's per-block stepping — so even at "same
  address" the regs differ by ~0x30 bytes of stack.

### Verification
- pytest tests/engines/test_rust_exploration.py: 243/243 passing.

### Commit
- (next) feat(benchmarks): differential test harness for Python vs Rust step diffing
