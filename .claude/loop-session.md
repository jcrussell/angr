# Loop session notes (2026-05-07, 125th loop session)

## Task: angr-fk0m — Unify rust_state_sync / rust_state_cache / rust_state_export mixins

### Status: AUDIT → DEFER

After audit, deferring with the same reasoning template as the prior
architecture refactor deferrals (angr-borb / angr-ja0b / angr-x3xu /
angr-m2hf / angr-prem). The proposed unification doesn't fit the actual
mixin structure.

### Audit findings

1. **Mixins are already phase-organized by direction.**
   - `rust_state_sync.py` — 37 methods, Python→Rust direction (register
     sync, memory sync, page sync, lazy regions).
   - `rust_state_cache.py` — 15 methods, state metadata management
     (handle lookup, predicate eval skip flags).
   - `rust_state_export.py` — 29 methods, Rust→Python direction (state
     export, plugin restoration, RustSolverFallback wrapper class).

   These ARE the three phases the bead proposes (Synchronization /
   Caching / Export). Renaming them into a `RustStateCoordinator` with
   `phase_sync()`, `phase_cache()`, `phase_export()` methods is purely
   cosmetic — same boundaries, same methods, different file.

2. **Total LoC is large.** 1587 + 400 + 955 = 2942 lines across the
   three mixins. A real merge would be a substantial refactor with no
   behavioral payoff.

3. **Multiple documented invariants run across these files.**
   - `invariant-callstack-sync-export-pipeline`: export has 4 paths
     (cached fast path L207, parent-state ...).
   - `invariant-rust-solver-fallback-class`: RustSolverFallback owns
     per-state Rust solver fallback wiring; lives in export module.
   - `disk-cache-register-filter`: disk init cache stores ALL archinfo
     registers — invariant in sync module.
   - `callstack-rebuild-pattern`: detailed pattern for CallStack rebuild
     — invariant in export module.
   - `flareon5-post-exploration-constraints`: root cause was in solver
     fallback (export module).

   Unifying risks breaking these invariants without test signal.

4. **No documented bug class motivates the work.**
   `bd memories drift` and `bd memories cache invalidation` produce no
   incident pointing at mixin fragmentation. The current scheme has
   carried us through 146 tests + 16 benchmarks correct.

5. **The "drift risk" claim is unsupported.** The bead asserts "Each
   has its own cache invalidation rules; no shared invariant." But the
   mixins don't share state — they share `self._rust_mgr`,
   `self._project`, `self._state_metadata`, and the StateMetadata
   dataclass (per `invariant-state-metadata-dataclass`). Cache
   invalidation is centralized in StateMetadata, not duplicated across
   mixins.

### Why this matches the prior deferral pattern

Same template as angr-borb / angr-ja0b / angr-x3xu / angr-m2hf / angr-prem:
- (a) Bead description references infrastructure that has shifted
  (mixins are already phase-organized; "fragmentation" pitch overstates
  drift).
- (b) Full scope is large (~3000 lines + 81 methods to consolidate).
- (c) Proposed RustStateCoordinator phases = current mixin boundaries
  with renamed methods. No architectural improvement.
- (d) No bug class observed to motivate the work.
- (e) Half-measures (e.g., merging cache into sync) would break
  documented invariants without behavioural value.

### Action

1. Defer angr-fk0m with this audit as the reason.
2. Save memory `avoid-deferred-fk0m-state-mixin-unification` so future
   sessions don't re-open without (a) a concrete bug showing mixin
   drift, OR (b) a refactor that genuinely changes responsibilities,
   not just renames classes.

### Files modified

- None — audit only; no source edits.

## Status: complete (deferred with audit + memory saved)
