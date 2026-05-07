# Loop session notes (2026-05-07, 116th loop session)

Two related P3 architectural-refactor beads closed wontfix in this
session: angr-ja0b (StepOutcome trait) and angr-x3xu (SolverBridge
protocol). Both proposed adding abstraction layers that, on real code
review, decouple coupling that doesn't exist or that the existing
enum/duck-typing already expresses cleanly.

## Task 1: angr-ja0b — Replace StepError variants with a richer StepOutcome trait (REJECTED)

### What the bead proposed

Replace pattern-matching on `StepError` variants in
`native/angr/src/exploration/{mod.rs,stepping.rs}` with a trait:

```
trait StepOutcome {
    fn action(&self) -> Action;  // Callback / Deadend / Continue / Error
    fn state(&self) -> &RustSimState;
    fn reason(&self) -> &str;
}
```

Stepping returns trait objects; the run loop dispatches via `outcome.action()`.

### Why rejected (real code review)

Looked at the only call site — `mod.rs:2735–2947`. The four StepError
arms are not just "different actions" — each variant CARRIES different
data that the call site uses, and the data shapes are fundamentally
incompatible with a trait that exposes only `state()` + `reason()`:

- `NeedCallback(PendingCallback)` — uses 7 fields of `pending`:
  `reason`, `state`, `deferred_forks`, `fork_snapshots`,
  `stored_conditions`, `pre_callback_snapshot`. The handler is ~180
  lines and its core work — replaying deferred branch forks against
  stored conditions/snapshots — is unique to this variant.
- `Deadended(state)` — pushes state to STASH_DEADENDED. 1 line.
- `Error(state, message)` — pushes onto `self.errors` then to
  STASH_ERRORED. 5 lines. `reason()` could return `message` here.
- `Unconstrained(state)` — pushes to STASH_UNCONSTRAINED. 3 lines.

A `StepOutcome` trait with `action()`/`state()`/`reason()` cannot
expose `pending.deferred_forks`, `pending.fork_snapshots`, or
`pending.stored_conditions`. To use them the run loop would need to
downcast or call a NeedCallback-only method — at which point the trait
adds nothing over the existing enum match.

### Existing memory already records the design intent

`invariant-step-error-not-thiserror`: StepError was deliberately left
as a hand-rolled enum (rather than `thiserror`) because it carries
`RustSimState` as a CONTROL-FLOW SIGNAL, not a true error. Promoting
it to a trait would obscure that signal further, not clarify it.

### Verdict

Pattern-matching is the natural Rust expression for this dispatch.
The bead's trait would either lose type information (if untyped) or
require downcasting (if typed), neither of which is an improvement.
Closing as wontfix.

### Memory saved

- `avoid-step-outcome-trait-refactor` — rationale for keeping the
  StepError enum match instead of trait dispatch, with the data-shape
  argument and pointer to mod.rs:2735–2947.

### No code changes; no build/test required.

---

## Task 2: angr-x3xu — Define a SolverBridge protocol (REJECTED)

### What the bead proposed

Replace direct manager calls to `RustSolverContext` with a Python
`SolverBridge` Protocol; provide `RustSolverBridge` and
`ClaripySolverBridge` implementations.

### Why rejected (real code review)

- `angr/exploration/rust_manager.py` has **zero** references to
  `RustSolverContext` / `RustSolverFallback` / `RustSolverProxy`. The
  manager already has no direct concrete-type coupling to decouple.
  The bead's premise (`manager calls into RustSolverContext directly`)
  is incorrect.
- The Rust solver is reached either via
  `self._rust_mgr.fork_state_solver(state_id)` (PyO3 type used as a
  duck-typed object) or via `RustSolverFallback`
  (rust_state_export.py:20), which already wraps the solver and
  patches `state.solver` post-exploration. `RustSolverProxy`
  (rust_state_proxy.py:19) is a second duck-typed wrapper for the
  SimProcedure path. Both wrappers expose the same eval / satisfiable
  / min / max contract.
- Tests do NOT mock/substitute the solver (grep
  `mock.*solver|FakeSolver|StubSolver` in `tests/` returns nothing).
  The bead's "swap the backend in tests" value-prop is hypothetical.
- A Protocol class would document the duck-typed contract but provide
  no decoupling gain — there is no caller to redirect away from a
  concrete import.

### Memory saved

- `avoid-solver-bridge-protocol` — code-evidence rationale, file
  pointers, and the hypothetical-test-harness caveat.

### No code changes; no build/test required.

---

## Carried over from previous session (115th, angr-khth init-pipeline split)

`_load_init_from_disk_cache` (rust_manager.py:1398) was split into
three independently testable phases: `_load_init_pickle` (pure I/O),
`_deserialize_init_state` (pure SimState construction), and
`_apply_init_side_effects` (manager-owned metadata). Memory saved:
`invariant-init-pipeline-phases`. See git log f32505930 / aed1ec175.

### Next session candidates (P3 ready)

- angr-3vrj (P3) StateMetadata dataclass — 86 sites across 6 files;
  refactor scope larger than the bead lists. Could attempt focused
  first slice.
- angr-csd1 (P3) split PythonCallbacks into focused traits — 48
  callsites across 12 Rust files. Multi-session.
- angr-qrhl (P3) trait-based VEX op dispatcher — needs perf
  measurement to justify.

### Other ready beads (P1/P2 multi-session)

- angr-pufm (P1), angr-prem (P2), angr-fk0m (P2), angr-4j5u (P2),
  angr-m2hf (P2), angr-wqao (P2), angr-bkcs (P3), angr-czph (P3),
  angr-n28w (P3).
