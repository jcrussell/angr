//! In-module unit tests for `claripy_bridge/cache.rs` (angr-c4xcs.7).
//!
//! Sibling-extracted from the former inline `#[cfg(test)] mod tests`
//! (rust-mod-tests-sibling-extraction). The worker-local-clear test exercises
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
    // Distinctive id unlikely to collide with allocate_id()-minted ids in
    // sibling tests sharing the process-global registry.
    const SYMBOL_ID: u64 = 0x1A2B_3C4D_5E6F;

    Python::initialize();
    Python::attach(|py| {
        store_claripy_ast_with_info(
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
        assert!(
            get_claripy_ast(SYMBOL_ID).is_some(),
            "symbol identity must still resolve via the global registry \
             after a worker-local clear",
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
// the `thread_local!` destructor phase. `worker_thread` in scheduler.rs calls
// `clear_worker_local_caches()` before returning for exactly this reason; this
// pins the invariant it relies on — that the clear leaves the cache empty.
//
// What that buys is *eager release* of the worker's ASTs at a known point, not
// UAF prevention as angr-bjk8 / angr-1yge9.9 originally claimed: a cached AST
// owns an `Rc`-cloned `Context`, so `Z3_del_context` cannot run while it lives,
// and `panic = "abort"` means no unwind can skip the clear. See angr-9ke6b.39
// and the corrected comment at the scheduler.rs call site.
#[test]
fn test_worker_local_clear_empties_ast_cache() {
    // Start from a clean slate on this (possibly test-runner-reused) thread.
    clear_worker_local_caches();
    tl_cache!(AST_CACHE, put(0x5151_i64, RustBV::concrete(0x1234, 64)));
    assert_eq!(
        cache_stats().0,
        1,
        "setup: AST_CACHE holds the one entry just put",
    );

    clear_worker_local_caches();

    assert_eq!(
        cache_stats().0,
        0,
        "worker-teardown clear must empty AST_CACHE so no RustBV outlives z3ctx",
    );
}

// The Rust→claripy compound-expression cache (EXPRESSION_BY_OPERANDS_PTR) is
// thread-local and keyed by the operands Arc pointer. A store then get with the
// same key must return the pinned AST; the RustBV held alongside pins the
// operands Arc alive. Purely thread-local — no global registry involved.
#[test]
fn test_expression_by_operands_roundtrip() {
    Python::initialize();
    Python::attach(|py| {
        let bv = RustBV::concrete(0xdead_beef, 64);
        let key = 0xABCD_1234_usize;
        store_expression_ast_by_operands(key, bv, py.None());

        let hit = get_expression_ast_by_operands(py, key);
        assert!(
            hit.is_some(),
            "stored operands-ptr entry must be retrievable"
        );
    });
}

// A miss on the expression-pointer cache must return None rather than panicking
// or returning a stale entry.
#[test]
fn test_expression_by_operands_miss_returns_none() {
    Python::initialize();
    Python::attach(|py| {
        // A key we never stored — reserved high value to avoid colliding with a
        // real Arc pointer a sibling test might have inserted.
        let miss = get_expression_ast_by_operands(py, 0xFFFF_FFFF_FFF0);
        assert!(miss.is_none(), "unstored operands ptr must miss");
    });
}

// angr-ph300.54: `evict_claripy_ast` drops a symbol-id registration from the
// global registry (the sole rust_id → AST store). This backs the export-side
// C5 width-mismatch guard, which evicts a stale (aliased) AST before re-minting
// under the correct width.
#[test]
fn test_evict_claripy_ast_clears_both_stores() {
    const SYMBOL_ID: u64 = 0x54AB_CDEF_0123;

    Python::initialize();
    Python::attach(|py| {
        store_claripy_ast_with_info(
            0x54AA,
            SYMBOL_ID,
            "evict_sym",
            32,
            SymbolKind::BitVector,
            py.None(),
        );
        assert!(
            global_registry().has_original(SYMBOL_ID),
            "setup: registry must hold the symbol after store",
        );
        assert!(
            get_claripy_ast(SYMBOL_ID).is_some(),
            "setup: symbol must resolve before eviction",
        );

        evict_claripy_ast(SYMBOL_ID);

        assert!(
            !global_registry().has_original(SYMBOL_ID),
            "evict must remove the global-registry entry",
        );
        assert!(
            get_claripy_ast(SYMBOL_ID).is_none(),
            "evict must make the symbol unresolvable via the global registry",
        );
    });
}
