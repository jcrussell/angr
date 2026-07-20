//! Thread-local AST identity / conversion caches for the claripy bridge.
//!
//! Holds the three thread-local caches (`AST_CACHE`, `CLARIPY_AST_CACHE`,
//! `EXPRESSION_BY_OPERANDS_PTR`) and the `tl_cache!` shorthand macro, plus the
//! public store/get/lookup/clear helpers that wrap them. The cross-cache
//! invariants C1-C6 are documented in the parent module rustdoc (see `super`).
//! Both `super::import` and `super::export` depend on these helpers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;

use crate::symbolic::{RustBV, global_registry};

/// Maximum number of AST nodes to cache.
const AST_CACHE_SIZE: usize = 10000;

// Thread-local LRU mapping `claripy_ast.__hash__() → RustBV`.
//
// Direction: claripy→Rust. Caches the result of `claripy_to_rustbv`
// for hot reconversion (e.g. the same constraint walked many times
// during exploration).
//
// Key: `ast.hash()` (Python `Py_hash_t`, fits in `i64`). Hashes are
// stable across CPython garbage collection unlike `id(ast)`, which
// can be reused after collection.
//
// Value: the converted `RustBV`. `RustBV::clone` is a refcount bump
// on the operand `Arc<[RustBV]>`, so cache hits are cheap.
//
// Width invariant: hashes are content-addressed and include length,
// but a stale entry from a recycled hash slot could in principle
// produce a width mismatch. Defended at the use site (~L478) — see
// cross-cache invariant C5.
//
// Invalidation: cleared by `clear_ast_cache` (block boundary /
// significant constraint changes) and `reset_for_new_exploration`
// (exploration restart). Capacity-bounded LRU eviction beyond that. See cross-cache
// invariant C3 — partial clears are never correct.
//
// Coherence with CLARIPY_AST_CACHE: a `Symbolic`/`Constrained` hit
// in this cache implies the underlying `symbol_id` was previously
// registered via `store_claripy_ast{,_with_info}`, so the inverse
// mapping lives in `CLARIPY_AST_CACHE` and the global registry (C2).
thread_local! {
    pub(super) static AST_CACHE: RefCell<LruCache<i64, RustBV>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(AST_CACHE_SIZE).expect("AST_CACHE_SIZE is a non-zero constant")));
}

/// Short-hand for `CACHE.with(|c| c.borrow_mut().…)` against the thread-local
/// AST caches in this module. Accepts an arbitrary method/chain so both
/// mutating operations and `.get(&k).cloned()` reads collapse to one line.
macro_rules! tl_cache {
    ($cache:ident, $($call:tt)*) => {
        $cache.with(|c| c.borrow_mut().$($call)*)
    };
}

// Thread-local mapping `rust_symbol_id → original claripy AST`.
//
// Direction: Rust→claripy, for leaf symbolic variables. Caches the
// original `BVS("x", 32)` Python object so that `RustBV::Symbolic { id }`
// can be returned to Python as the *same* AST identity, preserving
// any annotations and ensuring constraints on `x` continue to apply
// to the exported value.
//
// Key: `rust_id: u64`, the symbol's Rust-side identifier allocated by
// `SymbolicIdentityRegistry::allocate_id`. MUST NOT be
// `RustBV::EXPRESSION_ID` — see cross-cache invariant C1.
//
// Value: owned `Py<PyAny>` (`pyo3` GIL-independent reference).
//
// Why `std::HashMap` not `FxHashMap`: see cross-cache invariant C6.
//
// Why unbounded `HashMap`, not `LruCache`: symbol identities must
// survive for the entire exploration — evicting them would break
// identity preservation and force constraint duplication. The set of
// leaf symbols is bounded by the binary's symbolic input surface and
// grows slowly (tens to low thousands), so unbounded growth is
// acceptable.
//
// Invalidation: cleared only by `clear_ast_cache` (block boundary,
// but normally called when starting fresh) and `reset_for_new_exploration`
// (exploration restart). See cross-cache invariant C3.
//
// Coherence with SymbolicIdentityRegistry: forward subset relation
// `CLARIPY_AST_CACHE ⊆ global_registry` per cross-cache invariant
// C2. Asserted at insert time in `store_claripy_ast{,_with_info}`.
// `get_claripy_ast` checks global first to honor C2's directionality
// and a coherence-violation `debug_assert` fires if the thread-local
// holds an entry the registry does not.
thread_local! {
    static CLARIPY_AST_CACHE: RefCell<HashMap<u64, Py<PyAny>>> =
        RefCell::new(HashMap::new());
}

// Thread-local LRU mapping `Arc::as_ptr(operands) → (RustBV pin, claripy AST)`.
//
// Direction: Rust→claripy, for compound `Expression` variants.
// Maps the operands `Arc<[RustBV]>` raw pointer of a freshly-imported
// claripy `Expression` to its original Python AST. Used by
// `rustbv_to_claripy_memo` to return the imported AST verbatim
// instead of rebuilding from `BVOp + operands`, preserving claripy
// annotations (and any other AST metadata) attached to the
// `Expression` node itself — without this, annotations on
// intermediate expression nodes are dropped on the Rust→Python
// return trip even when leaves carry annotations that would
// otherwise propagate. See bd `angr-ykdq`.
//
// Key: `Arc::as_ptr(operands) as usize`. The operands `Arc` is
// created fresh by each builder call (`add_into`, `mul_into`, ...)
// and is stable across `RustBV::clone()` (refcount bump), so a
// cloned `Expression` shares its parent's cache entry. Two distinct
// claripy `Expression`s get distinct `Arc`s.
//
// Value: `(RustBV, Py<PyAny>)`. The `RustBV` clone is held alongside
// the Python AST specifically to pin the operands `Arc` alive: the
// cache key is a raw pointer, and without holding a strong reference
// the allocator could reuse the same address for an unrelated
// `Expression`'s operands while the entry is still in cache —
// surfacing as a wrong-AST return. The `Py<PyAny>` is the payload
// returned on hit.
//
// Invalidation: cleared by `clear_ast_cache` and `reset_for_new_exploration`
// (C3). Capacity-bounded LRU eviction (10000) beyond that. On
// eviction, the held `RustBV` drops its refcount, allowing the
// operands `Arc` to be freed — safe because the evicted entry is no
// longer reachable via this cache. This is the sole Rust→claripy
// cache for compound Expression nodes (see C4).
thread_local! {
    static EXPRESSION_BY_OPERANDS_PTR: RefCell<LruCache<usize, (RustBV, Py<PyAny>)>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(10000).expect("expression-by-ptr cache capacity is a non-zero constant")));
}

/// Store a claripy AST for later retrieval.
/// Called when converting claripy→RustBV for symbolic values.
///
/// This stores in both the global registry (for cross-thread access)
/// and the thread-local cache (for fast repeated access).
///
/// `symbol_id` MUST be a real allocated leaf-symbol id, not
/// `RustBV::EXPRESSION_ID` — compound expressions use the
/// expression caches, not this one.
pub fn store_claripy_ast(symbol_id: u64, ast: Py<PyAny>) {
    debug_assert_ne!(
        symbol_id,
        RustBV::EXPRESSION_ID,
        "CLARIPY_AST_CACHE must not be keyed by the EXPRESSION_ID sentinel; \
         use store_expression_ast_by_operands for compound nodes",
    );

    tl_cache!(CLARIPY_AST_CACHE, insert(symbol_id, ast.clone()));

    // Also store in global registry via public method
    // Note: We use a dummy hash (0) since we only have the symbol_id here
    global_registry().register_by_id(symbol_id, ast);

    // Coherence: the thread-local CLARIPY_AST_CACHE entry must also be
    // visible in the global registry. `get_claripy_ast` checks global
    // first; if the post-condition breaks, cross-thread lookups would
    // miss while same-thread lookups hit, masking the bug.
    debug_assert!(
        global_registry().has_original(symbol_id),
        "store_claripy_ast: global registry missing rust_id={symbol_id} after insert",
    );
}

/// Store a claripy AST with full symbol information.
///
/// This is the preferred method when symbol name and width are available,
/// as it enables name-based lookup for better identity preservation.
///
/// `symbol_id` MUST be a real allocated leaf-symbol id, not
/// `RustBV::EXPRESSION_ID`.
pub fn store_claripy_ast_with_info(
    py_hash: i64,
    symbol_id: u64,
    name: &str,
    width: u32,
    ast: Py<PyAny>,
) {
    debug_assert_ne!(
        symbol_id,
        RustBV::EXPRESSION_ID,
        "CLARIPY_AST_CACHE must not be keyed by the EXPRESSION_ID sentinel; \
         use store_expression_ast_by_operands for compound nodes",
    );

    tl_cache!(CLARIPY_AST_CACHE, insert(symbol_id, ast.clone()));

    // Register in global registry with full information
    global_registry().register(py_hash, symbol_id, name, width, ast);

    debug_assert!(
        global_registry().has_original(symbol_id),
        "store_claripy_ast_with_info: global registry missing rust_id={symbol_id} after insert",
    );
}

/// Retrieve a previously stored claripy AST by symbol ID.
/// Called when converting RustBV→claripy to return the original AST.
///
/// Checks global registry first (for cross-thread access),
/// then falls back to thread-local cache.
pub fn get_claripy_ast(symbol_id: u64) -> Option<Py<PyAny>> {
    // Check global registry first (survives across threads/callbacks)
    if let Some(ast) = global_registry().get_original_ast(symbol_id) {
        return Some(ast);
    }

    // Fall back to thread-local cache. Per cross-cache invariant C2,
    // every CLARIPY_AST_CACHE entry should also live in the global
    // registry, so a hit on this branch means the registry lost the
    // entry without `clear_ast_cache` also clearing the thread-local
    // (i.e. someone called `clear_global_registry` in isolation, which
    // violates C3). Surface that in debug builds.
    let local = tl_cache!(CLARIPY_AST_CACHE, get(&symbol_id).cloned());
    debug_assert!(
        local.is_none(),
        "C2 violation: CLARIPY_AST_CACHE hit for rust_id={symbol_id} but \
         global registry missing the entry (clear_global_registry called \
         without clear_ast_cache?)",
    );
    local
}

/// Evict a stale claripy AST registration for a symbol id from both the
/// global registry and the thread-local `CLARIPY_AST_CACHE`.
///
/// Mirrors the import-side C5 width-mismatch guard on the export path: if a
/// cache hit hands back an AST whose width disagrees with the requested
/// symbol width (an id-level aliasing bug — see angr-owr37), the caller drops
/// the stale entry and re-mints. Clearing both stores together preserves the
/// `CLARIPY_AST_CACHE ⊆ global_registry` invariant (C2/C3).
pub fn evict_claripy_ast(symbol_id: u64) {
    global_registry().remove(symbol_id);
    tl_cache!(CLARIPY_AST_CACHE, remove(&symbol_id));
}

/// Look up a symbol by its Python hash.
///
/// This is used during import to check if we've already imported this symbol.
pub fn lookup_symbol_by_hash(py_hash: i64) -> Option<u64> {
    global_registry().lookup_by_hash(py_hash)
}

/// Look up the canonical Rust-side name of a registered symbol id.
///
/// A symbol's Z3 constant is named after its Rust name, so an importer that
/// resolves a symbol by id must rebuild it with this name (angr-izov2).
pub fn lookup_symbol_name_by_id(rust_id: u64) -> Option<String> {
    global_registry().lookup_name_by_id(rust_id)
}

/// Look up symbol info by name (deprecated - use lookup_symbol_by_name_and_width).
///
/// This is used when we receive a symbol by name and need to find its Rust ID.
pub fn lookup_symbol_by_name(name: &str) -> Option<crate::symbolic::SymbolInfo> {
    global_registry().lookup_by_name(name)
}

/// Look up symbol info by name and width.
///
/// This is the preferred method after D2 fix which uses width-qualified names.
pub fn lookup_symbol_by_name_and_width(
    name: &str,
    width: u32,
) -> Option<crate::symbolic::SymbolInfo> {
    global_registry().lookup_by_name_and_width(name, width)
}

/// Store the original claripy AST keyed by an imported Expression's
/// operands Arc pointer. The BV clone is held alongside to pin the
/// operands Arc alive (preventing pointer reuse on free).
pub fn store_expression_ast_by_operands(operands_ptr: usize, bv: RustBV, ast: Py<PyAny>) {
    tl_cache!(EXPRESSION_BY_OPERANDS_PTR, put(operands_ptr, (bv, ast)));
}

/// Retrieve a previously stored claripy AST by an Expression's operands
/// Arc pointer. Returns None on miss.
pub fn get_expression_ast_by_operands(py: Python<'_>, operands_ptr: usize) -> Option<Py<PyAny>> {
    tl_cache!(
        EXPRESSION_BY_OPERANDS_PTR,
        get(&operands_ptr).map(|(_, ast)| ast.clone_ref(py))
    )
}

/// Clear all AST conversion caches.
/// Call this at block boundaries or when the constraint set changes significantly.
///
/// Note: This clears thread-local caches but NOT the global registry. It is the
/// **worker-local** clear — safe to call from any exploration worker thread
/// because it only touches this thread's caches (see
/// [`clear_worker_local_caches`] for the threading-intent alias). Use
/// [`reset_for_new_exploration`] to also wipe the cross-thread global registry.
pub fn clear_ast_cache() {
    tl_cache!(AST_CACHE, clear());
    tl_cache!(CLARIPY_AST_CACHE, clear());
    tl_cache!(EXPRESSION_BY_OPERANDS_PTR, clear());
}

/// Worker-local cache clear: drops only this thread's three thread-local AST
/// caches, never the process-wide `SymbolicIdentityRegistry`.
///
/// angr-1ilq.2: a self-documenting entry point for the Option-A parallel
/// scheduler (1ilq.3) to call on per-worker task-boundary / teardown. It is a
/// thin alias for [`clear_ast_cache`] kept distinct so a worker-side call reads
/// as worker-scoped and a future contributor does not reach for
/// [`reset_for_new_exploration`] (which wipes global symbol identity that
/// sibling workers depend on). See cross-cache invariant C3 in the parent
/// module rustdoc.
pub fn clear_worker_local_caches() {
    clear_ast_cache();
}

/// Reset all caches for a fresh exploration: the thread-local AST caches **and**
/// the process-wide global registry.
///
/// **Main-thread / exploration-start only.** This wipes the cross-thread
/// `SymbolicIdentityRegistry`, which holds canonical symbol identity shared by
/// every worker. Under the Option-A parallel model (1ilq) it MUST NOT be called
/// from a worker mid-run — doing so would invalidate live symbol IDs other
/// workers still hold. Workers clear only their own caches via
/// [`clear_worker_local_caches`].
///
/// Clears the thread-local caches first, then the global registry, so the
/// ordering still honors cross-cache invariant C3 (no partial clear is correct).
pub fn reset_for_new_exploration() {
    clear_ast_cache();
    crate::symbolic::clear_global_registry();
}

/// Get the current cache hit/miss statistics.
#[cfg(test)]
pub fn cache_stats() -> (usize, usize) {
    // Returns (len, cap) for debugging
    AST_CACHE.with(|cache| {
        let c = cache.borrow();
        (c.len(), c.cap().get())
    })
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod cache_tests;
