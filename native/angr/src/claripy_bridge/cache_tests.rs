//! In-module unit tests for `claripy_bridge/cache.rs` (angr-c4xcs.7).
//!
//! Sibling-extracted from the former inline `#[cfg(test)] mod tests`
//! (rust-module-root-split-recipe). The worker-local-clear test exercises
//! the cross-thread global registry; the added expression-pointer-cache tests
//! stay purely thread-local (no `global_registry()` mutation) so they cannot
//! race the global-registry test when cargo runs the suite in parallel.

use super::*;

// angr-1ilq.2: the worker-local clear must leave the cross-thread global
// SymbolicIdentityRegistry intact. Under the Option-A parallel model a
// worker clears only its own thread-local caches at a task boundary; if
// that path also wiped the global registry it would invalidate symbol
// identity every sibling worker still holds. This pins the split:
// register a symbol (populating the global registry), run the worker-local
// clear, then assert the global identity still resolves.
#[test]
fn test_worker_local_clear_preserves_global_registry() {
    // Distinctive id unlikely to collide with `SymContext::next_id`-minted ids
    // in sibling tests sharing the process-global registry.
    const SYMBOL_ID: u64 = 0x1A2B_3C4D_5E6F;

    Python::initialize();
    Python::attach(|py| {
        global_registry().register(
            0x7777,
            SYMBOL_ID,
            "ilq2_sym",
            64,
            SymbolKind::BitVector,
            py.None(),
        );

        // Precondition: identity is live in the global registry.
        assert!(
            global_registry().has_original(SYMBOL_ID),
            "setup: registry must hold the symbol after store",
        );

        // Worker-local clear: drops this thread's caches only.
        clear_worker_local_caches();

        // The global registry — and thus cross-thread symbol identity —
        // survives the worker-local clear.
        assert!(
            global_registry().has_original(SYMBOL_ID),
            "worker-local clear must NOT wipe the global registry",
        );

        // Contrast: the exploration-start reset DOES drop the global
        // identity (and is main-thread / start-of-run only).
        reset_for_new_exploration();
        assert!(
            !global_registry().has_original(SYMBOL_ID),
            "reset_for_new_exploration must clear the global registry",
        );
    });
}

// The worker-teardown clear must EMPTY the thread's `AST_CACHE`, so no `RustBV`
// (whose z3 ASTs are bound to the worker's soon-to-drop `z3ctx`) survives into
// the `thread_local!` destructor phase. `worker_thread` in scheduler_pool.rs calls
// `clear_worker_local_caches()` before returning for exactly this reason; this
// pins the invariant it relies on — that the clear leaves the cache empty.
//
// What that buys is *eager release* of the worker's ASTs at a known point, not
// UAF prevention as angr-bjk8 / angr-1yge9.9 originally claimed: a cached AST
// owns an `Rc`-cloned `Context`, so `Z3_del_context` cannot run while it lives,
// and `panic = "abort"` means no unwind can skip the clear. See angr-9ke6b.39
// and the corrected comment at the scheduler_pool.rs call site.
#[test]
fn test_worker_local_clear_empties_ast_cache() {
    // Start from a clean slate on this (possibly test-runner-reused) thread.
    clear_worker_local_caches();
    tl_cache!(AST_CACHE, put(0x5151_i64, RustBV::concrete(0x1234, 64)));
    assert_eq!(
        ast_cache_len(),
        1,
        "setup: AST_CACHE holds the one entry just put",
    );

    clear_worker_local_caches();

    assert_eq!(
        ast_cache_len(),
        0,
        "worker-teardown clear must empty AST_CACHE so no RustBV outlives z3ctx",
    );
}

