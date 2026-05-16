## Session log: 2026-05-16 — angr-fk0m.1: audit state-mixin overlaps

### Status: closing

### Task

Audit cross-mixin coupling between `angr/exploration/rust_state_sync.py`,
`rust_state_cache.py`, `rust_state_export.py`. No code change.

### Findings

Method counts (current):
- sync:   46 methods (was 37 at prior audit, 2026-05-07)
- cache:  14 methods (was 15)
- export: 36 methods (was 29)
Total: 96 methods (~18% growth since prior audit).

Cross-mixin call inventory:
- sync → cache: 8 call sites, all `self._register_handle(...)` (AST handle
  registration). Locations: rust_state_sync.py:1282, 1430, 1453, 1543,
  1808, 1873, 1888, 1949.
- cache → export: 1 call site,
  `self._attach_rust_solver_fallback(state, state_id)` at
  rust_state_cache.py:404, when materializing a Python SimState for
  predicate evaluation.
- export → sync/cache: 0 direct calls (export reads cache via
  `self._state_cache`/`self._lookup_handle` through the manager surface
  but does not call sync methods directly).

Composition: `RustExplorationManager(RustCallbackDispatchMixin,
RustStateSyncMixin, RustStateCacheMixin, RustStateExportMixin, ...)` —
mixins layered in dependency order (sync → cache → export uses what
sync produces, cache binds export-side fallback during materialization).

### Conclusion

Prior audit (avoid-deferred-fk0m-state-mixin-unification memory)
remains valid: mixins are phase-organized; nothing meaningfully moves
to a shared base. The 9 cross-mixin call sites are dependency edges,
not duplicated logic — extracting a base class would add layering
without removing code.

### Action

- Refresh deferral memory with updated method counts and cross-mixin
  call inventory (9 edges total).
- Close angr-fk0m.1 with reference to memory.
- Leave children .2/.3/.4 untouched — the user decides whether to mark
  them deferred.
