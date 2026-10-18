//! In-module unit tests for `claripy_bridge/cache.rs` (angr-c4xcs.7).
//!
//! Sibling-extracted from the former inline `#[cfg(test)] mod tests`
//! (rust-module-root-split-recipe).

use super::*;

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

