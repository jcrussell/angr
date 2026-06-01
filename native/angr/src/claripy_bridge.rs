//! Bridge between claripy ASTs and RustBV.
//!
//! This module provides bidirectional conversion between Python claripy ASTs
//! and Rust RustBV values, enabling native symbolic execution in Rust.
//!
//! ## Identity Preservation
//!
//! A critical requirement is preserving symbolic identity across the FFI boundary:
//! - When `BVS("x", 32)` is imported from Python, it should keep its identity
//! - When exported back to Python, the original AST should be returned
//! - This ensures constraints on the original `x` apply to the exported value
//!
//! Identity preservation uses two mechanisms:
//! 1. **Global registry** (`SymbolicIdentityRegistry`): Cross-thread, persistent
//! 2. **Thread-local caches**: Fast access for repeated conversions

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyTuple};

use crate::symbolic::{RustBV, RustBVHandle, RustSymbolTable, SymContext, global_registry};

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
// produce a width mismatch. `claripy_to_rustbv` defends against this
// by checking `cached_bv.width() == ast.length` on hit and evicting
// + reconverting on mismatch (~L351).
//
// Invalidation: cleared by `clear_ast_cache` (block boundary /
// significant constraint changes) and `clear_all_caches` (exploration
// restart). Capacity-bounded LRU eviction beyond that.
//
// Coherence with CLARIPY_AST_CACHE: a `Symbolic`/`Constrained` hit
// in this cache implies the underlying `symbol_id` was previously
// registered via `store_claripy_ast{,_with_info}`, so the inverse
// mapping lives in `CLARIPY_AST_CACHE` and the global registry.
thread_local! {
    static AST_CACHE: RefCell<LruCache<i64, RustBV>> =
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
// `RustBV::EXPRESSION_ID` (`u64::MAX`) — that sentinel is reserved
// for compound `Expression` variants which use `EXPRESSION_CACHE` /
// `EXPRESSION_BY_OPERANDS_PTR` instead. Enforced by `debug_assert!`
// in `store_claripy_ast{,_with_info}`.
//
// Value: owned `Py<PyAny>` (`pyo3` GIL-independent reference).
//
// Why unbounded `HashMap`, not `LruCache`: symbol identities must
// survive for the entire exploration — evicting them would break
// identity preservation and force constraint duplication. The set of
// leaf symbols is bounded by the binary's symbolic input surface and
// grows slowly (tens to low thousands), so unbounded growth is
// acceptable.
//
// Invalidation: cleared only by `clear_ast_cache` (block boundary,
// but normally called when starting fresh) and `clear_all_caches`
// (exploration restart).
//
// Coherence with SymbolicIdentityRegistry: every entry inserted here
// is ALSO registered in the global registry (via `register_by_id` /
// `register` in `store_claripy_ast*`), so `get_claripy_ast` can fall
// back to the global registry on thread-local miss. The global
// registry is the source of truth for cross-thread/callback access;
// this thread-local is a fast hot path. Enforced by `debug_assert!`
// in `store_claripy_ast{,_with_info}` that
// `global_registry().has_original(symbol_id)` holds after insert.
thread_local! {
    static CLARIPY_AST_CACHE: RefCell<HashMap<u64, Py<PyAny>>> =
        RefCell::new(HashMap::new());
}

// Thread-local LRU mapping `expression_hash → original claripy AST`.
//
// Direction: Rust→claripy, for compound `Expression` variants.
// Counterpart to `CLARIPY_AST_CACHE` for non-leaf nodes. Lets
// `rustbv_to_claripy_memo` return the imported AST verbatim instead
// of reconstructing it from `BVOp` + operands, which would drop any
// claripy annotations attached to the intermediate node and burn
// allocations.
//
// Key: a content-derived `u64` expression hash (computed by the
// caller from the `RustBV::Expression` structure). Hash collisions
// are vanishingly rare given the 64-bit space; the downstream
// consequence of a collision is a wrong-AST return, which would
// surface as a Python-side constraint mismatch.
//
// Value: owned `Py<PyAny>` for the original Python expression.
//
// Invalidation: cleared by `clear_ast_cache` and `clear_all_caches`.
// Capacity-bounded LRU eviction (10000) beyond that — eviction is
// safe here because Expression nodes can always be reconstructed
// from `op + operands` (just with a fresh AST identity).
//
// Coherence with EXPRESSION_BY_OPERANDS_PTR: both caches serve the
// same goal (return the original Python AST for an `Expression`) but
// key on different surfaces — this one on the hash, the other on the
// operands `Arc` pointer. Either hit is sufficient; they may diverge
// under LRU eviction.
thread_local! {
    static EXPRESSION_CACHE: RefCell<LruCache<u64, Py<PyAny>>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(10000).expect("expression cache capacity is a non-zero constant")));
}

// Thread-local LRU mapping `Arc::as_ptr(operands) → (RustBV pin, claripy AST)`.
//
// Direction: Rust→claripy, alternative key for `EXPRESSION_CACHE`.
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
// Invalidation: cleared by `clear_ast_cache` and `clear_all_caches`.
// Capacity-bounded LRU eviction (10000) beyond that. On eviction,
// the held `RustBV` drops its refcount, allowing the operands `Arc`
// to be freed — safe because the evicted entry is no longer
// reachable via this cache.
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
         use store_expression_ast / store_expression_ast_by_operands for compound nodes",
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
         use store_expression_ast / store_expression_ast_by_operands for compound nodes",
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

    // Fall back to thread-local cache
    tl_cache!(CLARIPY_AST_CACHE, get(&symbol_id).cloned())
}

/// Look up a symbol by its Python hash.
///
/// This is used during import to check if we've already imported this symbol.
pub fn lookup_symbol_by_hash(py_hash: i64) -> Option<u64> {
    global_registry().lookup_by_hash(py_hash)
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

/// Store a claripy AST in the expression cache by expression hash.
/// Called when converting claripy→RustBV for compound expressions.
pub fn store_expression_ast(expr_hash: u64, ast: Py<PyAny>) {
    tl_cache!(EXPRESSION_CACHE, put(expr_hash, ast));
}

/// Retrieve a claripy AST from the expression cache by expression hash.
/// Called when converting RustBV→claripy to return the original AST.
pub fn get_expression_ast(expr_hash: u64) -> Option<Py<PyAny>> {
    tl_cache!(EXPRESSION_CACHE, get(&expr_hash).cloned())
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
/// Note: This clears thread-local caches but NOT the global registry.
/// Use `clear_all_caches()` to clear everything including global state.
pub fn clear_ast_cache() {
    tl_cache!(AST_CACHE, clear());
    tl_cache!(CLARIPY_AST_CACHE, clear());
    tl_cache!(EXPRESSION_CACHE, clear());
    tl_cache!(EXPRESSION_BY_OPERANDS_PTR, clear());
}

/// Clear all caches including the global registry.
/// Call this at the start of a new exploration to prevent stale mappings.
pub fn clear_all_caches() {
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

/// Error type for claripy bridge operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BridgeError {
    /// Unsupported claripy operation.
    #[error("unsupported claripy op: {0}")]
    UnsupportedOp(String),
    /// Type mismatch.
    #[error("type mismatch: {0}")]
    TypeMismatch(String),
    /// Python error.
    #[error("python error: {0}")]
    PythonError(String),
    /// Invalid arguments.
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
}

impl From<PyErr> for BridgeError {
    fn from(err: PyErr) -> Self {
        BridgeError::PythonError(err.to_string())
    }
}

/// Check if a Python object is a RustBVHandle.
///
/// This is a fast check that allows bypassing claripy conversion entirely
/// when the Python side returns a handle instead of a claripy AST.
pub fn is_rust_handle(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<RustBVHandle>()
}

/// Try to extract a RustBV from a RustBVHandle via the symbol table.
///
/// This is the fast path for handle-based operations. If the object is a
/// RustBVHandle, we look up the RustBV directly from the symbol table,
/// completely bypassing claripy AST conversion.
///
/// Returns None if the object is not a handle or the handle ID is not found.
pub fn try_handle_to_rustbv(
    obj: &Bound<'_, PyAny>,
    symbol_table: &RustSymbolTable,
) -> Option<RustBV> {
    if let Ok(handle) = obj.extract::<RustBVHandle>() {
        symbol_table.get(handle.id())
    } else {
        None
    }
}

/// Convert a Python object to RustBV, trying handle first, then claripy.
///
/// This is the primary entry point for Python -> Rust conversion on the hot path.
/// It first checks if the object is a RustBVHandle (fast path), and only falls
/// back to claripy conversion if necessary.
///
/// Returns the RustBV, or an error if conversion fails.
pub fn python_to_rustbv(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    symbol_table: &RustSymbolTable,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    // Fast path: check for RustBVHandle first
    if let Some(bv) = try_handle_to_rustbv(obj, symbol_table) {
        return Ok(bv);
    }

    // Slow path: claripy AST conversion
    if is_claripy_ast(obj) {
        claripy_to_rustbv(py, obj, ctx)
    } else {
        let type_name = obj
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        Err(BridgeError::TypeMismatch(format!(
            "expected RustBVHandle or claripy AST, got {}",
            type_name
        )))
    }
}

/// Try to extract a concrete BVV value directly from a claripy AST.
/// Returns Some((value, width)) if the AST is a BVV, None otherwise.
/// This is much cheaper than full claripy_to_rustbv conversion.
#[inline]
pub fn try_extract_bvv(ast: &Bound<'_, PyAny>) -> Option<(u128, u32)> {
    let op: String = ast.getattr("op").ok()?.extract().ok()?;
    if op != "BVV" {
        return None;
    }
    let args = ast.getattr("args").ok()?;
    let args_tuple = args.cast::<PyTuple>().ok()?;
    let value: u128 = extract_int_value(args_tuple.get_item(0).ok()?).ok()?;
    let width: u32 = args_tuple.get_item(1).ok()?.extract().ok()?;
    Some((value, width))
}

/// Convert a claripy AST to a RustBV.
///
/// This recursively converts the claripy expression tree to RustBV operations.
/// Supports: BVV, BVS, arithmetic, bitwise, comparison, and extension ops.
/// Uses thread-local LRU caching with claripy's stable `__hash__` to avoid
/// redundant conversions across constraint additions.
pub fn claripy_to_rustbv(
    py: Python<'_>,
    ast: &Bound<'_, PyAny>,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    // Get the operation name first to determine caching strategy
    let op: String = ast.getattr("op")?.extract()?;
    let op_str = op.as_str();

    // Cache all immutable AST nodes using claripy's stable __hash__.
    // BVV (concrete) nodes are cheap to create and don't need caching.
    // All symbolic/compound nodes benefit from caching.
    let use_cache = op_str != "BVV";

    // Get claripy's stable hash for cache lookup.
    // Claripy ASTs use their internal _hash attribute which is a consistent identifier.
    // We use ast.hash() which returns Python's Py_hash_t (guaranteed to fit in i64).
    let ast_hash: i64 = if use_cache {
        // Use ast.hash() method from PyAny which properly handles Py_hash_t
        ast.hash()? as i64
    } else {
        0 // Not used
    };

    // Check LRU cache for previously converted AST
    if use_cache {
        let cached = tl_cache!(AST_CACHE, get(&ast_hash).cloned());
        if let Some(cached_bv) = cached {
            // Defensive width check — claripy hashes are content-addressed and
            // already include length, so collisions are exceedingly rare, but
            // returning a wrong-width BV would silently corrupt downstream ops.
            // Bool ASTs have length=None and we represent them as width-1 BVs.
            let expected_width: u32 = ast
                .getattr("length")
                .ok()
                .and_then(|l| l.extract::<u32>().ok())
                .unwrap_or(1);
            if cached_bv.width() == expected_width {
                return Ok(cached_bv);
            }
            // Width mismatch: evict the stale entry and fall through to
            // reconvert. The recomputed BV will be re-cached below.
            tl_cache!(AST_CACHE, pop(&ast_hash));
        }
    }

    let args = ast.getattr("args")?;

    let result = match op_str {
        // Concrete bitvector value
        "BVV" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let value: u128 = extract_int_value(args_tuple.get_item(0)?)?;
            let width: u32 = args_tuple.get_item(1)?.extract()?;
            Ok(RustBV::concrete(value, width))
        }

        // Symbolic bitvector value
        "BVS" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let name: String = args_tuple.get_item(0)?.extract()?;
            // Width might be in args[1] or in .length attribute
            let width: u32 = if args_tuple.len() > 1 {
                args_tuple.get_item(1)?.extract().unwrap_or_else(|_| {
                    ast.getattr("length")
                        .and_then(|l| l.extract())
                        .unwrap_or(64)
                })
            } else {
                ast.getattr("length")?.extract()?
            };

            // CRITICAL: Check global registry first for identity preservation
            // If this symbol was already imported, return the existing RustBV
            // to maintain identity across Python<->Rust boundary
            if let Some(existing_id) = lookup_symbol_by_hash(ast_hash) {
                // Symbol already registered, return a reference to it
                return Ok(RustBV::symbolic_with_id(existing_id, &name, width));
            }

            // Also check by name+width for cases where the hash changed but name is stable
            // D2 Fix: Use width-qualified lookup to avoid collisions
            if let Some(info) = lookup_symbol_by_name_and_width(&name, width) {
                // Symbol with same name/width exists, return reference
                return Ok(RustBV::symbolic_with_id(info.rust_id, &name, width));
            }

            // Create new symbol and register with full info
            let bv = RustBV::symbolic(ctx, &name, width);
            // Store the original claripy AST so we can return it when converting back
            // This preserves symbol identity for Python's memory model
            if let RustBV::Symbolic { id, .. } = &bv {
                store_claripy_ast_with_info(ast_hash, *id, &name, width, ast.clone().unbind());
            }
            Ok(bv)
        }

        // Arithmetic operations
        "__add__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__add__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.add(&right, ctx))
        }

        "__sub__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__sub__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sub(&right, ctx))
        }

        "__mul__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__mul__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.mul(&right, ctx))
        }

        "__floordiv__" | "SDiv" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("div requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sdiv(&right, ctx))
        }

        "UDiv" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UDiv requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.udiv(&right, ctx))
        }

        "__mod__" | "SMod" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("mod requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.srem(&right, ctx))
        }

        "URem" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("URem requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.urem(&right, ctx))
        }

        "__neg__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("__neg__ requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            Ok(val.neg(ctx))
        }

        // Bitwise operations
        "__and__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__and__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.and(&right, ctx))
        }

        "__or__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__or__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.or(&right, ctx))
        }

        "__xor__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__xor__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.xor(&right, ctx))
        }

        "__invert__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("__invert__ requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            Ok(val.not(ctx))
        }

        // Shift operations
        "__lshift__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs(
                    "__lshift__ requires 2 args".into(),
                ));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.shl(&amt, ctx))
        }

        "LShR" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("LShR requires 2 args".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.lshr(&amt, ctx))
        }

        "__rshift__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs(
                    "__rshift__ requires 2 args".into(),
                ));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.ashr(&amt, ctx))
        }

        "RotateLeft" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs(
                    "RotateLeft requires 2 args".into(),
                ));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.rotl(&amt, ctx))
        }

        "RotateRight" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs(
                    "RotateRight requires 2 args".into(),
                ));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.rotr(&amt, ctx))
        }

        // Extension operations
        "ZeroExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ZeroExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv(py, &args_list[1], ctx)?;
            let new_width = val.width() + extend_bits;
            Ok(val.zero_extend(new_width, ctx))
        }

        "SignExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SignExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv(py, &args_list[1], ctx)?;
            let new_width = val.width() + extend_bits;
            Ok(val.sign_extend(new_width, ctx))
        }

        "Extract" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 3 {
                return Err(BridgeError::InvalidArgs("Extract requires 3 args".into()));
            }
            let high: u32 = args_list[0].extract()?;
            let low: u32 = args_list[1].extract()?;
            let val = claripy_to_rustbv(py, &args_list[2], ctx)?;
            let val_width = val.width();

            // P5 fix: Validate Extract bounds to prevent runtime errors
            if high >= val_width {
                return Err(BridgeError::InvalidArgs(format!(
                    "Extract high={} >= width={}",
                    high, val_width
                )));
            }
            if low > high {
                return Err(BridgeError::InvalidArgs(format!(
                    "Extract low={} > high={}",
                    low, high
                )));
            }

            Ok(val.extract(high, low, ctx))
        }

        "Concat" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs(
                    "Concat requires at least 1 arg".into(),
                ));
            }
            let mut result = claripy_to_rustbv(py, &args_list[0], ctx)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv(py, arg, ctx)?;
                result = result.concat(&next, ctx);
            }
            Ok(result)
        }

        // Comparison operations (return 1-bit result)
        "__eq__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__eq__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.eq(&right, ctx))
        }

        "__ne__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__ne__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ne(&right, ctx))
        }

        "ULT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ULT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ult(&right, ctx))
        }

        "ULE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ULE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ule(&right, ctx))
        }

        "UGT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UGT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ugt(&right, ctx))
        }

        "UGE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UGE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.uge(&right, ctx))
        }

        "SLT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SLT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.slt(&right, ctx))
        }

        "SLE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SLE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sle(&right, ctx))
        }

        "SGT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SGT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sgt(&right, ctx))
        }

        "SGE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SGE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sge(&right, ctx))
        }

        // Boolean constant
        "BoolV" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let value: bool = args_tuple.get_item(0)?.extract()?;
            // Return 1-bit BV (1 for true, 0 for false)
            Ok(RustBV::concrete(if value { 1 } else { 0 }, 1))
        }

        // If-then-else
        "If" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 3 {
                return Err(BridgeError::InvalidArgs("If requires 3 args".into()));
            }
            let cond = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let then_val = claripy_to_rustbv(py, &args_list[1], ctx)?;
            let else_val = claripy_to_rustbv(py, &args_list[2], ctx)?;
            Ok(cond.ite(&then_val, &else_val, ctx))
        }

        // Boolean operations (for constraints)
        "And" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                // Empty And is True
                return Ok(RustBV::concrete(1, 1));
            }
            // Boolean And: all 1-bit values must be 1
            let mut result = claripy_to_rustbv(py, &args_list[0], ctx)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv(py, arg, ctx)?;
                result = result.and(&next, ctx);
            }
            Ok(result)
        }

        "Or" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                // Empty Or is False
                return Ok(RustBV::concrete(0, 1));
            }
            // Boolean Or: at least one 1-bit value must be 1
            let mut result = claripy_to_rustbv(py, &args_list[0], ctx)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv(py, arg, ctx)?;
                result = result.or(&next, ctx);
            }
            Ok(result)
        }

        "Not" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Not requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            // Boolean Not: invert 1-bit value
            Ok(val.not(ctx))
        }

        // Reverse bytes
        "Reverse" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Reverse requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            // Byte-reverse the value
            reverse_bytes(&val, ctx)
        }

        _ => Err(BridgeError::UnsupportedOp(op_str.to_string())),
    };

    // Cache all symbolic/compound AST nodes using claripy's stable hash.
    // This dramatically reduces conversion overhead when the same expressions
    // appear in multiple constraints.
    if use_cache {
        if let Ok(ref bv) = result {
            // Forward cache: claripy hash → RustBV
            tl_cache!(AST_CACHE, put(ast_hash, bv.clone()));
            // Reverse cache: store original claripy AST for later retrieval
            // This preserves AST identity when converting back to Python
            // Use the ast_hash as a positive u64 key
            let expr_key = ast_hash as u64;
            store_expression_ast(expr_key, ast.clone().unbind());

            // For Expression results, also key by the operands Arc pointer so
            // `rustbv_to_claripy_memo` can return the original AST verbatim
            // (preserving annotations attached at the Expression level).
            if let RustBV::Expression { operands, .. } = bv {
                let operands_ptr = Arc::as_ptr(operands) as *const () as usize;
                store_expression_ast_by_operands(operands_ptr, bv.clone(), ast.clone().unbind());
            }
        }
    }

    result
}

/// Ensure a Py<PyAny> is a claripy AST, wrapping ints/bools if needed.
///
/// This is a defensive function to handle cases where a Python int or bool
/// might be returned from cache or operations instead of a proper claripy AST.
/// Operations like Extract require claripy ASTs and will fail with
/// "'int' object has no attribute 'length'" if passed an int.
fn ensure_claripy_ast(
    py: Python<'_>,
    obj: &Py<PyAny>,
    claripy_mod: &Bound<'_, PyAny>,
    width_hint: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let bound = obj.bind(py);

    // Check if it's already a claripy AST by checking for 'op' attribute
    match bound.hasattr("op") {
        Ok(true) => {
            return Ok(obj.clone());
        }
        Ok(false) => {
            let type_name = bound
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            log::debug!(
                "ensure_claripy_ast: object {} missing 'op' attr, wrapping",
                type_name
            );
        }
        Err(e) => {
            log::warn!("ensure_claripy_ast: hasattr('op') failed: {}", e);
        }
    }

    // Check the actual Python type to distinguish bool from int
    // IMPORTANT: In Python, bool is a subclass of int, so we must check bool FIRST
    // but use is_instance_of, not extract, because extract::<bool>() succeeds for ints too
    let type_name = bound
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Check if it's exactly a Python bool (not an int that happens to be 0 or 1)
    if type_name == "bool" {
        if let Ok(bool_val) = bound.extract::<bool>() {
            // If width hint is provided, wrap as BVV (for use in BV operations)
            // Otherwise wrap as BoolV (for use in Bool operations)
            if let Some(w) = width_hint {
                let val: i64 = if bool_val { 1 } else { 0 };
                return claripy_mod.call_method1("BVV", (val, w)).map(|o| o.into());
            }
            return claripy_mod
                .call_method1("BoolV", (bool_val,))
                .map(|o| o.into());
        }
    }

    // If it's an int, wrap in BVV with the provided width hint
    // Try i128 first for larger values, then fall back to i64
    if type_name == "int" {
        let width = width_hint.unwrap_or(64);
        // Try to extract as i128 for larger values
        if let Ok(int_val) = bound.extract::<i128>() {
            log::debug!(
                "ensure_claripy_ast: wrapping int {} in BVV with width {}",
                int_val,
                width
            );
            // For values that fit in i64, use that (more compatible)
            if int_val >= i64::MIN as i128 && int_val <= i64::MAX as i128 {
                return claripy_mod
                    .call_method1("BVV", (int_val as i64, width))
                    .map(|o| o.into());
            } else {
                // For larger values, pass as Python int directly
                return claripy_mod
                    .call_method1("BVV", (&bound, width))
                    .map(|o| o.into());
            }
        }
        // Fallback: pass the Python object directly and let claripy handle it
        log::debug!(
            "ensure_claripy_ast: wrapping large int in BVV with width {}",
            width
        );
        return claripy_mod
            .call_method1("BVV", (&bound, width))
            .map(|o| o.into());
    }

    // Otherwise return as-is and hope for the best
    log::warn!(
        "ensure_claripy_ast: unknown type {}, returning as-is",
        type_name
    );
    Ok(obj.clone())
}

/// Convert a RustBV back to a claripy AST.
///
/// This is used when returning symbolic results to Python.
/// For Expression variants, this recursively reconstructs the claripy AST
/// from the operation tree, preserving the original expression structure.
pub fn rustbv_to_claripy(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Memoize by RustBV pointer identity to dedupe shared subtrees in DAGs.
    // sym-write's symbolic-store ITE chains have ~25 unique Arc-shared
    // subtrees expanded into a 142k-node tree without dedup; converting that
    // takes ~2.7s vs ~tens of ms with memoization.
    let mut memo: HashMap<usize, Py<PyAny>> = HashMap::new();
    rustbv_to_claripy_memo(py, bv, claripy_mod, &mut memo)
}

fn rustbv_to_claripy_memo(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
    memo: &mut HashMap<usize, Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    use crate::symbolic::BVOp;

    // Check cache first for Symbolic variants
    // This preserves AST identity across FFI boundary
    if let RustBV::Symbolic { id, width, .. } = bv {
        if let Some(cached) = get_claripy_ast(*id) {
            // Validate cached value is a claripy AST, not an int
            let cached_valid = ensure_claripy_ast(py, &cached, claripy_mod, Some(*width))?;
            return Ok(cached_valid);
        }
    }

    // For Expression variants imported via claripy_to_rustbv, look up the
    // original AST keyed by the operands Arc pointer. This returns the
    // Python-side AST verbatim, preserving any annotations attached at the
    // Expression level (which are otherwise dropped when we rebuild from
    // BVOp+operands). See angr-ykdq.
    if let RustBV::Expression { operands, .. } = bv {
        let operands_ptr = Arc::as_ptr(operands) as *const () as usize;
        if let Some(cached) = get_expression_ast_by_operands(py, operands_ptr) {
            return Ok(cached);
        }
    }

    // Memoization: only Expression variants are worth caching (the recursive case
    // with potential DAG sharing). For Expression, key by `bv` pointer so two
    // sibling references to the same operand inside a shared `Arc<[RustBV]>` only
    // pay the conversion cost once.
    let memo_key = if matches!(bv, RustBV::Expression { .. }) {
        let k = bv as *const RustBV as usize;
        if let Some(cached) = memo.get(&k) {
            return Ok(cached.clone_ref(py));
        }
        Some(k)
    } else {
        None
    };

    let result: PyResult<Py<PyAny>> = match bv {
        RustBV::Concrete { value, width } => {
            // Create claripy.BVV(value, width)
            if *width <= 64 {
                claripy_mod
                    .call_method1("BVV", (*value as i64, *width))
                    .map(|obj| obj.into())
            } else if *width % 8 == 0 {
                // Byte-aligned: use bytes for exact representation
                let byte_count = *width as usize / 8;
                let bytes = value.to_be_bytes();
                let start = bytes.len().saturating_sub(byte_count);
                let py_bytes = PyBytes::new(py, &bytes[start..]);
                claripy_mod
                    .call_method1("BVV", (py_bytes, *width))
                    .map(|obj| obj.into())
            } else {
                // Non-byte-aligned: use Python int to avoid string/size mismatch
                // claripy.BVV(int_value, width) works for any width
                let py_int = PyInt::new(py, *value);
                claripy_mod
                    .call_method1("BVV", (py_int, *width))
                    .map(|obj| obj.into())
            }
        }
        RustBV::Symbolic {
            id: _, name, width, ..
        } => {
            // Cache was already checked above, so this is a symbol created purely in Rust
            // Create new claripy.BVS(name, width)
            claripy_mod
                .call_method1("BVS", (&**name, *width))
                .map(|obj| obj.into())
        }
        RustBV::Constrained { value, width, .. } => {
            // For constrained values, return the concrete value
            if *width <= 64 {
                claripy_mod
                    .call_method1("BVV", (*value as i64, *width))
                    .map(|obj| obj.into())
            } else if *width % 8 == 0 {
                let byte_count = *width as usize / 8;
                let bytes = value.to_be_bytes();
                let start = bytes.len().saturating_sub(byte_count);
                let py_bytes = PyBytes::new(py, &bytes[start..]);
                claripy_mod
                    .call_method1("BVV", (py_bytes, *width))
                    .map(|obj| obj.into())
            } else {
                // Non-byte-aligned: use Python int
                let py_int = PyInt::new(py, *value);
                claripy_mod
                    .call_method1("BVV", (py_int, *width))
                    .map(|obj| obj.into())
            }
        }
        RustBV::Expression { op, operands, .. } => {
            // Recursively convert operands to claripy ASTs
            let raw_args: Vec<Py<PyAny>> = operands
                .iter()
                .map(|operand| rustbv_to_claripy_memo(py, operand, claripy_mod, memo))
                .collect::<Result<_, _>>()?;

            // Validate all args to ensure they're claripy ASTs with correct widths
            let args: Vec<Py<PyAny>> = raw_args
                .iter()
                .enumerate()
                .map(|(i, arg)| {
                    let width = operands.get(i).map(|o| o.width());
                    ensure_claripy_ast(py, arg, claripy_mod, width)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // For binary ops, ensure operand widths match (resize if needed)
            let args = if args.len() == 2 && !matches!(op, BVOp::Extract(_, _) | BVOp::Concat) {
                let a0 = args[0].bind(py);
                let a1 = args[1].bind(py);
                let w0: Option<u32> = a0.getattr("length").ok().and_then(|l| l.extract().ok());
                let w1: Option<u32> = a1.getattr("length").ok().and_then(|l| l.extract().ok());
                // Handle Bool/BV mismatch: convert Bool to BV(1)
                let (w0, w1) = match (w0, w1) {
                    (None, Some(w)) => {
                        // arg0 is Bool, arg1 is BV — convert Bool to BV(1)
                        let bv = claripy_mod.call_method1(
                            "If",
                            (
                                &args[0],
                                claripy_mod.call_method1("BVV", (1i32, w))?,
                                claripy_mod.call_method1("BVV", (0i32, w))?,
                            ),
                        )?;
                        return {
                            // Redo the operation with the converted operand
                            let args_fixed = vec![bv.unbind(), args[1].clone()];
                            // Fall through to the match op block below
                            // by replacing args
                            match op {
                                BVOp::And => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__and__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                                BVOp::Or => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__or__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                                BVOp::Xor => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__xor__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                                _ => {
                                    // For other ops, just use the converted args
                                    let a = args_fixed[0].bind(py);
                                    a.call_method1("__add__", (&args_fixed[1],))
                                        .map(|o| o.into())
                                }
                            }
                        };
                    }
                    (Some(w), None) => {
                        // arg0 is BV, arg1 is Bool
                        let bv = claripy_mod.call_method1(
                            "If",
                            (
                                &args[1],
                                claripy_mod.call_method1("BVV", (1i32, w))?,
                                claripy_mod.call_method1("BVV", (0i32, w))?,
                            ),
                        )?;
                        return {
                            let args_fixed = vec![args[0].clone(), bv.unbind()];
                            match op {
                                BVOp::And => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__and__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                                BVOp::Or => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__or__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                                BVOp::Xor => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__xor__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                                _ => args_fixed[0]
                                    .bind(py)
                                    .call_method1("__add__", (&args_fixed[1],))
                                    .map(|o| o.into()),
                            }
                        };
                    }
                    (Some(a), Some(b)) => (a, b),
                    (None, None) => {
                        return Ok(args[0].clone());
                    } // Both Bool — just return first
                };
                if w0 != w1 {
                    if w0 < w1 {
                        let extended = claripy_mod.call_method1("ZeroExt", (w1 - w0, &args[0]))?;
                        vec![extended.unbind(), args[1].clone()]
                    } else {
                        let extended = claripy_mod.call_method1("ZeroExt", (w0 - w1, &args[1]))?;
                        vec![args[0].clone(), extended.unbind()]
                    }
                } else {
                    args
                }
            } else {
                args
            };

            // Build the claripy expression based on the operation
            match op {
                // Arithmetic operations (binary, use method on first arg)
                BVOp::Add => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__add__", (&args[1],)).map(|o| o.into())
                }
                BVOp::Sub => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__sub__", (&args[1],)).map(|o| o.into())
                }
                BVOp::Mul => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__mul__", (&args[1],)).map(|o| o.into())
                }
                BVOp::UDiv => claripy_mod
                    .call_method1("UDiv", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::SDiv => claripy_mod
                    .call_method1("SDiv", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::URem => claripy_mod
                    .call_method1("URem", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::SRem => claripy_mod
                    .call_method1("SMod", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Neg => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method0("__neg__").map(|o| o.into())
                }

                // Bitwise operations
                BVOp::And => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__and__", (&args[1],))?;
                    // Check for NotImplemented (width mismatch etc)
                    if result.is_none()
                        || result
                            .get_type()
                            .name()
                            .map_or(false, |n| n == "NotImplementedType")
                    {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "__and__ returned NotImplemented",
                        ));
                    }
                    Ok(result.into())
                }
                BVOp::Or => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__or__", (&args[1],))?;
                    if result.is_none()
                        || result
                            .get_type()
                            .name()
                            .map_or(false, |n| n == "NotImplementedType")
                    {
                        let t0 = args[0]
                            .bind(py)
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or("?".into());
                        let t1 = args[1]
                            .bind(py)
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or("?".into());
                        let w0: String = args[0]
                            .bind(py)
                            .getattr("length")
                            .map(|l| format!("{}", l))
                            .unwrap_or("?".into());
                        let w1: String = args[1]
                            .bind(py)
                            .getattr("length")
                            .map(|l| format!("{}", l))
                            .unwrap_or("?".into());
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "__or__ NotImpl: {}(w={}) | {}(w={})",
                            t0, w0, t1, w1
                        )));
                    }
                    Ok(result.into())
                }
                BVOp::Xor => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__xor__", (&args[1],))?;
                    if result.is_none()
                        || result
                            .get_type()
                            .name()
                            .map_or(false, |n| n == "NotImplementedType")
                    {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "__xor__ returned NotImplemented",
                        ));
                    }
                    Ok(result.into())
                }
                BVOp::Not => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method0("__invert__").map(|o| o.into())
                }

                // Shift operations
                BVOp::Shl => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__lshift__", (&args[1],))
                        .map(|o| o.into())
                }
                BVOp::Lshr => claripy_mod
                    .call_method1("LShR", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Ashr => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__rshift__", (&args[1],))
                        .map(|o| o.into())
                }
                BVOp::RotL => claripy_mod
                    .call_method1("RotateLeft", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::RotR => claripy_mod
                    .call_method1("RotateRight", (&args[0], &args[1]))
                    .map(|o| o.into()),

                // Extension operations (args already validated)
                BVOp::ZeroExt(extend_bits) => {
                    let arg0_type = args[0]
                        .bind(py)
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());

                    // If arg0 is a Bool, convert it to a 1-bit BV first
                    // claripy.ZeroExt requires a BV, not a Bool
                    if arg0_type == "Bool" {
                        // Use claripy.If(cond, BVV(1, 1), BVV(0, 1)) to convert Bool to 1-bit BV
                        let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
                        let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
                        let bv1 = claripy_mod.call_method1("If", (&args[0], one, zero))?;
                        claripy_mod
                            .call_method1("ZeroExt", (*extend_bits, bv1))
                            .map(|o| o.into())
                    } else {
                        claripy_mod
                            .call_method1("ZeroExt", (*extend_bits, &args[0]))
                            .map(|o| o.into())
                    }
                }
                BVOp::SignExt(extend_bits) => {
                    let arg0_type = args[0]
                        .bind(py)
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());

                    // If arg0 is a Bool, convert it to a 1-bit BV first
                    // claripy.SignExt requires a BV, not a Bool
                    if arg0_type == "Bool" {
                        let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
                        let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
                        let bv1 = claripy_mod.call_method1("If", (&args[0], one, zero))?;
                        claripy_mod
                            .call_method1("SignExt", (*extend_bits, bv1))
                            .map(|o| o.into())
                    } else {
                        claripy_mod
                            .call_method1("SignExt", (*extend_bits, &args[0]))
                            .map(|o| o.into())
                    }
                }
                BVOp::Extract(high, low) => {
                    let arg0_type = args[0]
                        .bind(py)
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());

                    // If arg0 is a Bool, convert it to a 1-bit BV first
                    // claripy.Extract requires a BV, not a Bool
                    if arg0_type == "Bool" {
                        let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
                        let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
                        let bv1 = claripy_mod.call_method1("If", (&args[0], one, zero))?;
                        claripy_mod
                            .call_method1("Extract", (*high, *low, bv1))
                            .map(|o| o.into())
                    } else {
                        claripy_mod
                            .call_method1("Extract", (*high, *low, &args[0]))
                            .map(|o| o.into())
                    }
                }
                BVOp::Concat => {
                    // Concat takes multiple args (already validated)
                    if args.len() == 2 {
                        claripy_mod
                            .call_method1("Concat", (&args[0], &args[1]))
                            .map(|o| o.into())
                    } else {
                        // For multi-arg concat, build a tuple
                        let args_tuple = pyo3::types::PyTuple::new(py, &args)?;
                        claripy_mod
                            .call_method1("Concat", args_tuple)
                            .map(|o| o.into())
                    }
                }

                // Comparison operations
                // Note: __eq__ and __ne__ on claripy BVV objects may return Python bool,
                // not claripy Bool. We must wrap Python bools to ensure claripy AST output.
                BVOp::Eq => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__eq__", (&args[1],))?;
                    // If result is Python bool/int (concrete comparison result),
                    // wrap it in claripy.BoolV. Use extract::<bool> which works for
                    // both PyBool and PyInt (True/False are ints in Python).
                    if let Ok(bool_val) = result.extract::<bool>() {
                        claripy_mod
                            .call_method1("BoolV", (bool_val,))
                            .map(|o| o.into())
                    } else {
                        Ok(result.into())
                    }
                }
                BVOp::Ne => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__ne__", (&args[1],))?;
                    // If result is Python bool/int (concrete comparison result),
                    // wrap it in claripy.BoolV. Use extract::<bool> which works for
                    // both PyBool and PyInt (True/False are ints in Python).
                    if let Ok(bool_val) = result.extract::<bool>() {
                        claripy_mod
                            .call_method1("BoolV", (bool_val,))
                            .map(|o| o.into())
                    } else {
                        Ok(result.into())
                    }
                }
                BVOp::Ult => claripy_mod
                    .call_method1("ULT", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Ule => claripy_mod
                    .call_method1("ULE", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Ugt => claripy_mod
                    .call_method1("UGT", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Uge => claripy_mod
                    .call_method1("UGE", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Slt => claripy_mod
                    .call_method1("SLT", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Sle => claripy_mod
                    .call_method1("SLE", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Sgt => claripy_mod
                    .call_method1("SGT", (&args[0], &args[1]))
                    .map(|o| o.into()),
                BVOp::Sge => claripy_mod
                    .call_method1("SGE", (&args[0], &args[1]))
                    .map(|o| o.into()),

                // Conditional
                BVOp::Ite => {
                    // If(cond, then_val, else_val)
                    claripy_mod
                        .call_method1("If", (&args[0], &args[1], &args[2]))
                        .map(|o| o.into())
                }

                // Utility operations
                BVOp::Reverse => claripy_mod
                    .call_method1("Reverse", (&args[0],))
                    .map(|o| o.into()),
                BVOp::Clz | BVOp::Ctz | BVOp::Popcount => {
                    let op_name = match op {
                        BVOp::Clz => "clz",
                        BVOp::Ctz => "ctz",
                        BVOp::Popcount => "popcount",
                        _ => unreachable!(),
                    };
                    let width = bv.width();

                    // If operand is concrete, compute actual result
                    if let Some(operand) = operands.get(0) {
                        if let Some(concrete_val) = operand.as_u128() {
                            let result = match op {
                                BVOp::Clz => {
                                    // Count leading zeros, adjusting for width
                                    if concrete_val == 0 {
                                        width as u128
                                    } else {
                                        let leading = (concrete_val as u128).leading_zeros();
                                        // Adjust for actual bit width (128 - width)
                                        (leading - (128 - width)) as u128
                                    }
                                }
                                BVOp::Ctz => {
                                    // Count trailing zeros
                                    if concrete_val == 0 {
                                        width as u128
                                    } else {
                                        concrete_val.trailing_zeros().min(width) as u128
                                    }
                                }
                                BVOp::Popcount => {
                                    // Count ones
                                    concrete_val.count_ones() as u128
                                }
                                _ => unreachable!(),
                            };
                            return claripy_mod
                                .call_method1("BVV", (result as i64, width))
                                .map(|o| o.into());
                        }
                    }

                    // For symbolic input, create fresh variable (limitation - no constraint relationship)
                    log::debug!(
                        "Creating unconstrained {} result for symbolic input (constraint relationship lost)",
                        op_name
                    );
                    claripy_mod
                        .call_method1("BVS", (format!("{}_result", op_name), width))
                        .map(|o| o.into())
                }
                // Float ops: claripy's fpAdd/fpSub etc. need an FSort argument
                // and rounding mode; round-tripping a Z3 FP expression through
                // claripy is fragile. The Rust engine keeps the Z3 FP
                // constraint internally (via build_fp_z3_ast_cached) — for the
                // Python side we expose a fresh symbolic BV at the result
                // width (varies for FtoI/CmpXxx). Constraint info is lost
                // when the value crosses back to claripy, but the in-engine
                // solver still sees the FP terms.
                BVOp::Float { kind, prec } => {
                    let width = kind.result_bits(*prec);
                    let name = format!("fp_{:?}_{:?}_result", kind, prec);
                    claripy_mod
                        .call_method1("BVS", (name, width))
                        .map(|o| o.into())
                }
            }
        }
    };

    match result {
        Ok(ast) => {
            if let Some(k) = memo_key {
                memo.insert(k, ast.clone_ref(py));
            }
            Ok(ast)
        }
        Err(e) => Err(e),
    }
}

/// Extract an integer value from a Python object.
/// Handles both regular ints and large ints.
fn extract_int_value(obj: Bound<'_, PyAny>) -> Result<u128, BridgeError> {
    // Try extracting as i64 first (fast path)
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(v as u128);
    }

    // Try extracting as u64
    if let Ok(v) = obj.extract::<u64>() {
        return Ok(v as u128);
    }

    // Try extracting as u128
    if let Ok(v) = obj.extract::<u128>() {
        return Ok(v);
    }

    // For larger values, use Python's int.to_bytes
    let bit_length: usize = obj.call_method0("bit_length")?.extract().unwrap_or(128);
    let byte_length = (bit_length + 7) / 8;
    let byte_length = byte_length.max(1).min(16); // Clamp to 1-16 bytes

    let bytes_obj = obj.call_method1("to_bytes", (byte_length, "little"))?;
    let bytes: Vec<u8> = bytes_obj.extract()?;

    let mut value: u128 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if i < 16 {
            value |= (b as u128) << (i * 8);
        }
    }

    Ok(value)
}

/// Byte-reverse a RustBV value.
fn reverse_bytes(bv: &RustBV, ctx: &SymContext) -> Result<RustBV, BridgeError> {
    let width = bv.width();
    if width % 8 != 0 {
        return Err(BridgeError::InvalidArgs(
            "Reverse requires byte-aligned width".into(),
        ));
    }

    if let Some(value) = bv.as_u128() {
        // Concrete case: reverse bytes
        let num_bytes = width / 8;
        let mut reversed: u128 = 0;
        for i in 0..num_bytes {
            let byte = (value >> (i * 8)) & 0xFF;
            reversed |= byte << ((num_bytes - 1 - i) * 8);
        }
        Ok(RustBV::concrete(reversed, width))
    } else {
        // Symbolic case: build concatenation of reversed byte extracts
        let num_bytes = width / 8;
        let mut result = bv.extract(7, 0, ctx);
        for i in 1..num_bytes {
            let byte = bv.extract((i + 1) * 8 - 1, i * 8, ctx);
            result = result.concat(&byte, ctx);
        }
        Ok(result)
    }
}

/// Check if a Python object is a claripy AST.
pub fn is_claripy_ast(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("op").unwrap_or(false) && obj.hasattr("args").unwrap_or(false)
}

/// Phase 4 Fix: Get a stable identifier for a claripy AST.
///
/// This computes a hash that is more stable than Python's `__hash__` by:
/// 1. Using the internal `_hash` attribute if available (most stable)
/// 2. Computing a deterministic hash from op + args structure
///
/// This prevents duplicate symbols when Python's hash changes due to
/// garbage collection or object reallocation.
pub fn get_stable_ast_id(ast: &Bound<'_, PyAny>) -> Result<i64, BridgeError> {
    // Try internal _hash first (most stable across claripy versions)
    if let Ok(internal_hash) = ast.getattr("_hash") {
        if let Ok(hash_val) = internal_hash.extract::<i64>() {
            return Ok(hash_val);
        }
    }

    // Try __hash__ attribute directly (for cached hash)
    if let Ok(hash_method) = ast.getattr("__hash__") {
        if let Ok(hash_val) = hash_method.call0() {
            if let Ok(h) = hash_val.extract::<i64>() {
                return Ok(h);
            }
        }
    }

    // Fall back to PyAny.hash() which calls Python's hash()
    ast.hash()
        .map(|h| h as i64)
        .map_err(|e| BridgeError::PythonError(format!("hash failed: {}", e)))
}

/// Get the width (in bits) of a claripy AST.
pub fn get_ast_width(ast: &Bound<'_, PyAny>) -> Option<u32> {
    ast.getattr("length").ok()?.extract().ok()
}

/// Check if a claripy AST is concrete (BVV).
pub fn is_concrete_ast(ast: &Bound<'_, PyAny>) -> bool {
    if let Ok(op) = ast.getattr("op") {
        if let Ok(op_str) = op.extract::<String>() {
            return op_str == "BVV";
        }
    }
    false
}

/// Extract the concrete value from a BVV AST.
pub fn extract_concrete_value(ast: &Bound<'_, PyAny>) -> Option<u128> {
    if !is_concrete_ast(ast) {
        return None;
    }

    let args = ast.getattr("args").ok()?;
    let args_tuple = args.cast::<PyTuple>().ok()?;
    extract_int_value(args_tuple.get_item(0).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_int_value_small() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let val = 42i64.into_pyobject(py).unwrap();
            assert_eq!(extract_int_value(val.into_any().clone()).unwrap(), 42);
        });
    }
}
