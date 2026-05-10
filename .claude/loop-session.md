## Session log: 2026-05-10 — angr-3tek.1 (209th loop session, COMPLETE)

### Task
Root-cause analysis for the symbolic_objects stale-cache divergence
that blocks NativeRead/NativeWrite. Pure analysis bead — no production
code change. Acceptance: a one-paragraph patch description that
angr-3tek.2 can implement directly.

### Findings (full trace in bd notes on angr-3tek.1)

**The stale cache:** `RustExplorationManager._state_cache: dict[int, SimState]`
at `angr/exploration/rust_manager.py:756`. Populated on initial state
register (`rust_manager.py:2414, 2498`) and after every callback successor
(`rust_callback_dispatch.py:1064, 1217, 1402, 1594`). Never invalidated by
Rust-side memory mutations because Rust has no callback into Python on
`memory_store`.

**Python entry point:** `_handle_simprocedure_callback`
(`rust_callback_dispatch.py:371`) → `_create_state_for_callback`
(`rust_callback_dispatch.py:1733`). The latter does only two sync
steps: `_install_rust_memory_proxy` (eager-copies non-zero pointer
slots from the SP page only) and `_restore_symbolic_pages` (replays
from a Python-pushed snapshot that native procs never touch). So
concrete bytes outside SP and any symbolic byte from a native proc
are invisible to the next Python SimProc fallback.

**Already-built tooling for the fix:** Rust's `SymbolicMemory.dirty_pages`
(`native/angr/src/memory/mod.rs:88`) tracks per-state mutations
(updated at `memory/store.rs:92, 103, 132`; cleared on fork at
`mod.rs:345`). PyO3 already exposes pending-state versions:
`_get_pending_dirty_pages` (`exploration/pending_api.rs:227`),
`_clear_pending_dirty_tracking` (:231), and `_pending_memory_load_page` (:431).

**Missing piece:** No PyO3 method to fetch SYMBOLIC bytes from a
Rust page. Needs a wrapper around `symbolic_objects: FxHashMap<u64, RustBV>`
(`memory/mod.rs:79`) using the existing `rustbv_to_claripy` helper
(already used at `pending_api.rs:66, 84, 249`).

### Fix proposal for angr-3tek.2
Don't kill the cache — invalidate-and-replay just the dirty pages.
1. Add `_pending_memory_load_symbolic_page(page_addr) -> dict[u64, AST]`
   in `pending_api.rs`.
2. In `_create_state_for_callback`, AFTER `_install_rust_memory_proxy`
   and BEFORE `_restore_symbolic_pages`, walk
   `_get_pending_dirty_pages()` and replay both concrete and symbolic
   bytes (symbolic LAST per-page so they overwrite concrete defaults
   at the same addresses), then `_clear_pending_dirty_tracking()`.
3. Re-enable the registry lines at `procedures/mod.rs:240-241`.
4. Update `avoid-enabling-native-read` memory.

### Memories saved
- `3tek-root-cause` — full trace summary
- `invariant-rust-dirty-pages-pending` — the dirty-page mechanism
  (so future agents don't reinvent it)
- `invariant-3tek2-replay-ordering` — order-of-operations gotcha
  for the fix (symbolic stores must come AFTER the SP-page concrete
  copy, not before)
- Updated `avoid-enabling-native-read` to point to `3tek-root-cause`

### Files modified
None. Pure analysis bead.

### Status
COMPLETE. Unblocks angr-3tek.2 (re-enable native read/write with the
cache-sync fix). The next session can pick that up directly with the
patch description in hand.
