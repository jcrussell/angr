//! Thread-local AST identity / conversion caches for the claripy bridge.
//!
//! Holds the thread-local `AST_CACHE` and the `tl_cache!` shorthand macro, plus the
//! public store/get/lookup/clear helpers that wrap them. The cross-cache
//! invariants C3-C5 are documented in the parent module rustdoc (see `super`).
//! Both `super::import` and `super::export` depend on these helpers.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this is a
//! Python-boundary module, so `unwrap`/`expect` are denied here — a claripy AST
//! handed in from Python must never be able to reach a panic. The
//! cache-capacity conversion is a `const` ([`AST_CACHE_SIZE_NZ`]), so its
//! non-zero proof happens at compile time.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;

use crate::symbolic::RustBV;

/// Maximum number of AST nodes to cache.
const AST_CACHE_SIZE: usize = 10000;

/// [`AST_CACHE_SIZE`] pre-validated for `LruCache::new`. Doing the conversion
/// in a `const` moves the non-zero proof to compile time — a zero capacity
/// fails the build rather than panicking at first use.
const AST_CACHE_SIZE_NZ: NonZeroUsize = match NonZeroUsize::new(AST_CACHE_SIZE) {
    Some(n) => n,
    None => panic!("AST_CACHE_SIZE must be non-zero"),
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
// Value: the converted `RustBV`; cloning one is a refcount bump.
//
// Width invariant: hashes are content-addressed and include length,
// but a stale entry from a recycled hash slot could in principle
// produce a width mismatch. Defended at the use site — the `AST_CACHE`
// hit arm of `import::claripy_to_rustbv_depth` — see cross-cache
// invariant C5.
//
// Invalidation: cleared by `clear_ast_cache` (block boundary /
// significant constraint changes, worker teardown). Capacity-bounded LRU
// eviction beyond that. See cross-cache invariant C3.
//
thread_local! {
    static AST_CACHE: RefCell<LruCache<i64, RustBV>> =
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

/// Retrieve the `RustBV` previously converted from the claripy AST with this
/// hash. Returns `None` on miss.
///
/// The caller MUST re-check the returned BV's width against the AST's — see
/// cross-cache invariant C5 — and drop a mismatching entry with
/// [`evict_bv_by_ast_hash`].
pub(super) fn get_bv_by_ast_hash(ast_hash: i64) -> Option<RustBV> {
    tl_cache!(AST_CACHE, get(&ast_hash).cloned())
}

/// Width of a claripy AST as the cross-cache invariant C5 width guards read it.
///
/// Reads `ast.length`; a `Bool` AST has `length is None` and is reported as
/// width 1, matching the RustBV bool representation. Used by the C5 guard,
/// the `AST_CACHE`-hit arm of `import::claripy_to_rustbv_depth`
/// (angr-5mnx3.12).
///
/// SILENT(cat-b): a `.length` read that *raises* is conflated with the
/// legitimate `length is None` (Bool) case and also yields 1. The loss is
/// bounded: a caller whose true width is not 1 then evicts and reconverts,
/// paying the cached conversion but never returning a wrong-width value
/// (angr-sqfj8.23).
pub(super) fn claripy_ast_guard_width(ast: &Bound<'_, PyAny>) -> u32 {
    ast.getattr("length")
        .ok()
        .and_then(|l| l.extract::<u32>().ok())
        .unwrap_or(1)
}

/// Drop a stale claripy-hash → `RustBV` entry, so the caller can reconvert and
/// re-store. Used by the C5 width-mismatch guard on the import path.
pub(super) fn evict_bv_by_ast_hash(ast_hash: i64) {
    tl_cache!(AST_CACHE, pop(&ast_hash));
}

/// Store the `RustBV` converted from a claripy AST, keyed by claripy's stable
/// hash.
pub(super) fn store_bv_by_ast_hash(ast_hash: i64, bv: RustBV) {
    tl_cache!(AST_CACHE, put(ast_hash, bv));
}

/// Clear all AST conversion caches.
/// Call this at block boundaries or when the constraint set changes significantly.
///
/// Only this thread's cache, so it is safe to call from any exploration
/// worker thread (see [`clear_worker_local_caches`] for the threading-intent
/// alias).
pub(crate) fn clear_ast_cache() {
    tl_cache!(AST_CACHE, clear());
}

/// Worker-local cache clear: drops only this thread's thread-local AST cache.
///
/// angr-1ilq.2: a self-documenting entry point for the Option-A parallel
/// scheduler (1ilq.3) to call on per-worker task-boundary / teardown, a thin
/// alias for [`clear_ast_cache`] kept distinct so a worker-side call reads as
/// worker-scoped. See cross-cache invariant C3 in the parent module rustdoc.
///
/// The `worker_thread` teardown call in `exploration/scheduler_pool.rs` is eager
/// resource release, **not** a use-after-free guard: cached `RustBV` ASTs each
/// own an `Rc`-cloned `Context`, so the underlying `Z3_context` outlives them
/// regardless of drop order, and the shipped build is `panic = "abort"` so no
/// unwind can skip this call. See angr-9ke6b.39 (which corrects the earlier
/// angr-bjk8 / angr-1yge9.9 framing) and the comment at that call site.
#[cfg_attr(not(feature = "vex-engine-z3"), allow(dead_code))]
pub(crate) fn clear_worker_local_caches() {
    clear_ast_cache();
}

/// Number of entries currently held in this thread's [`AST_CACHE`].
///
/// Capacity is a compile-time constant ([`AST_CACHE_SIZE_NZ`]), so it was
/// dropped from the return type in angr-c7xno.15 — every caller only ever
/// wanted the occupancy, and a returned constant is not a statistic.
#[cfg(test)]
pub(crate) fn ast_cache_len() -> usize {
    AST_CACHE.with(|cache| cache.borrow().len())
}

test_submod!("cache_tests.rs" => cache_tests);
