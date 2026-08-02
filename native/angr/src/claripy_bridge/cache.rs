//! Thread-local AST identity / conversion caches for the claripy bridge.
//!
//! Holds the two thread-local caches (`AST_CACHE`,
//! `EXPRESSION_BY_OPERANDS_PTR`) and the `tl_cache!` shorthand macro, plus the
//! public store/get/lookup/clear helpers that wrap them. The cross-cache
//! invariants C3-C5 are documented in the parent module rustdoc (see `super`).
//! Both `super::import` and `super::export` depend on these helpers.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this is a
//! Python-boundary module, so `unwrap`/`expect` are denied here — a claripy AST
//! handed in from Python must never be able to reach a panic. The two
//! cache-capacity conversions that used to `expect` are now `const`s
//! ([`AST_CACHE_SIZE_NZ`], [`EXPRESSION_CACHE_SIZE_NZ`]), so their non-zero
//! proof happens at compile time instead.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;

use crate::symbolic::{RustBV, SymbolKind, global_registry};

/// Maximum number of AST nodes to cache.
const AST_CACHE_SIZE: usize = 10000;

/// [`AST_CACHE_SIZE`] pre-validated for `LruCache::new`. Doing the conversion
/// in a `const` moves the non-zero proof to compile time — a zero capacity
/// fails the build rather than panicking at first use.
const AST_CACHE_SIZE_NZ: NonZeroUsize = match NonZeroUsize::new(AST_CACHE_SIZE) {
    Some(n) => n,
    None => panic!("AST_CACHE_SIZE must be non-zero"),
};

/// Capacity of `EXPRESSION_BY_OPERANDS_PTR`, pre-validated like
/// [`AST_CACHE_SIZE_NZ`].
const EXPRESSION_CACHE_SIZE_NZ: NonZeroUsize = match NonZeroUsize::new(10000) {
    Some(n) => n,
    None => panic!("expression-by-ptr cache capacity must be non-zero"),
};

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
// produce a width mismatch. Defended at the use site — the `AST_CACHE`
// hit arm of `import::claripy_to_rustbv_depth` — see cross-cache
// invariant C5.
//
// Invalidation: cleared by `clear_ast_cache` (block boundary /
// significant constraint changes) and `reset_for_new_exploration`
// (exploration restart). Capacity-bounded LRU eviction beyond that. See cross-cache
// invariant C3 — partial clears are never correct.
//
// Coherence with the global registry: a `Symbolic`/`Constrained` hit
// in this cache implies the underlying `symbol_id` was previously
// registered via `store_claripy_ast_with_info`, so the inverse
// mapping (rust_id → original AST) lives in the `SymbolicIdentityRegistry`.
thread_local! {
    pub(super) static AST_CACHE: RefCell<LruCache<i64, RustBV>> =
        RefCell::new(LruCache::new(AST_CACHE_SIZE_NZ));
}

/// Short-hand for `CACHE.with(|c| c.borrow_mut().…)` against the thread-local
/// AST caches in this module. Accepts an arbitrary method/chain so both
/// mutating operations and `.get(&k).cloned()` reads collapse to one line.
macro_rules! tl_cache {
    ($cache:ident, $($call:tt)*) => {
        $cache.with(|c| c.borrow_mut().$($call)*)
    };
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
        RefCell::new(LruCache::new(EXPRESSION_CACHE_SIZE_NZ));
}

/// Store a claripy AST with full symbol information.
///
/// This is the preferred method when symbol name and width are available,
/// as it enables name-based lookup for better identity preservation.
///
/// `symbol_id` MUST be a real allocated leaf-symbol id, not
/// `RustBV::EXPRESSION_ID`.
pub(crate) fn store_claripy_ast_with_info(
    py_hash: i64,
    symbol_id: u64,
    name: &str,
    width: u32,
    kind: SymbolKind,
    ast: Py<PyAny>,
) {
    // Always-on, NOT `debug_assert_ne!` (angr-9ke6b.220): a sentinel-keyed
    // registry entry silently returns an unrelated AST on the next lookup,
    // and this runs once per imported/exported leaf, not per step.
    assert_ne!(
        symbol_id,
        RustBV::EXPRESSION_ID,
        "symbol identity registry must not be keyed by the EXPRESSION_ID sentinel; \
         use store_expression_ast_by_operands for compound nodes",
    );

    // Register in global registry with full information.
    global_registry().register(py_hash, symbol_id, name, width, kind, ast);
}

/// Retrieve a previously stored claripy AST by symbol ID.
/// Called when converting RustBV→claripy to return the original AST.
///
/// Reads from the process-global `SymbolicIdentityRegistry` (cross-thread,
/// survives across callbacks).
pub(crate) fn get_claripy_ast(symbol_id: u64) -> Option<Py<PyAny>> {
    global_registry().get_original_ast(symbol_id)
}

/// Evict a stale claripy AST registration for a symbol id from the global
/// registry.
///
/// Mirrors the import-side C5 width-mismatch guard on the export path: if a
/// cache hit hands back an AST whose width disagrees with the requested
/// symbol width (an id-level aliasing bug — see angr-owr37), the caller drops
/// the stale entry and re-mints.
pub(crate) fn evict_claripy_ast(symbol_id: u64) {
    global_registry().remove(symbol_id);
}

/// Look up a symbol by its Python hash.
///
/// This is used during import to check if we've already imported this symbol.
pub(crate) fn lookup_symbol_by_hash(py_hash: i64) -> Option<u64> {
    global_registry().lookup_by_hash(py_hash)
}

/// Look up the canonical Rust-side name of a registered symbol id.
///
/// A symbol's Z3 constant is named after its Rust name, so an importer that
/// resolves a symbol by id must rebuild it with this name (angr-izov2).
pub(crate) fn lookup_symbol_name_by_id(rust_id: u64) -> Option<String> {
    global_registry().lookup_name_by_id(rust_id)
}

/// Look up symbol info by name, width and claripy sort.
///
/// The key is name+width+sort: width separates same-name symbols of different
/// widths (D2 fix), and the sort keeps `BVS(name, 1)` and `BoolS(name)` from
/// aliasing to one id (angr-9ke6b.38).
pub(crate) fn lookup_symbol_by_name_and_width(
    name: &str,
    width: u32,
    kind: SymbolKind,
) -> Option<crate::symbolic::SymbolInfo> {
    global_registry().lookup_by_name_and_width(name, width, kind)
}

/// Store the original claripy AST keyed by an imported Expression's
/// operands Arc pointer. The BV clone is held alongside to pin the
/// operands Arc alive (preventing pointer reuse on free).
pub(crate) fn store_expression_ast_by_operands(operands_ptr: usize, bv: RustBV, ast: Py<PyAny>) {
    tl_cache!(EXPRESSION_BY_OPERANDS_PTR, put(operands_ptr, (bv, ast)));
}

/// Retrieve a previously stored claripy AST by an Expression's operands
/// Arc pointer. Returns None on miss.
pub(crate) fn get_expression_ast_by_operands(
    py: Python<'_>,
    operands_ptr: usize,
) -> Option<Py<PyAny>> {
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
pub(crate) fn clear_ast_cache() {
    tl_cache!(AST_CACHE, clear());
    tl_cache!(EXPRESSION_BY_OPERANDS_PTR, clear());
}

/// Worker-local cache clear: drops only this thread's two thread-local AST
/// caches, never the process-wide `SymbolicIdentityRegistry`.
///
/// angr-1ilq.2: a self-documenting entry point for the Option-A parallel
/// scheduler (1ilq.3) to call on per-worker task-boundary / teardown. It is a
/// thin alias for [`clear_ast_cache`] kept distinct so a worker-side call reads
/// as worker-scoped and a future contributor does not reach for
/// [`reset_for_new_exploration`] (which wipes global symbol identity that
/// sibling workers depend on). See cross-cache invariant C3 in the parent
/// module rustdoc.
///
/// The `worker_thread` teardown call in `exploration/scheduler.rs` is eager
/// resource release, **not** a use-after-free guard: cached `RustBV` ASTs each
/// own an `Rc`-cloned `Context`, so the underlying `Z3_context` outlives them
/// regardless of drop order, and the shipped build is `panic = "abort"` so no
/// unwind can skip this call. See angr-9ke6b.39 (which corrects the earlier
/// angr-bjk8 / angr-1yge9.9 framing) and the comment at that call site.
#[cfg_attr(not(feature = "vex-engine-z3"), allow(dead_code))]
pub(crate) fn clear_worker_local_caches() {
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
///
/// **Test-only by design** (angr-9ke6b.218 item 4, resolving the .214 flag).
/// Despite the name, no exploration start calls this and none should: the
/// exploration-start path clears only the thread-local caches via
/// `clear_ast_cache`, so the global registry deliberately survives across
/// managers in one process. Wiping it when a second manager starts — managers
/// coexist; a bench builds one per `simulation_manager()` call — would strand
/// the rust ids that the first manager's live states and `RustBV`s still hold,
/// and `rustbv_to_claripy` would mint a renamed BVS that no existing constraint
/// binds (angr-izov2). Bounding registry growth is angr-9ke6b.222's GC problem,
/// which angr-9ke6b.40 already ruled cannot be solved with an approximate
/// active set. This exists so `cache_tests` can start from a clean registry.
#[cfg(test)]
pub(crate) fn reset_for_new_exploration() {
    clear_ast_cache();
    crate::symbolic::clear_global_registry();
}

/// Get the current cache hit/miss statistics.
#[cfg(test)]
pub(crate) fn cache_stats() -> (usize, usize) {
    // Returns (len, cap) for debugging
    AST_CACHE.with(|cache| {
        let c = cache.borrow();
        (c.len(), c.cap().get())
    })
}

#[cfg(test)]
#[path = "cache_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod cache_tests;
