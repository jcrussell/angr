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
//! 1. **Global registry** (`SymbolicIdentityRegistry`): Cross-thread, persistent.
//!    Sole store for the `rust_id → original leaf AST` mapping.
//! 2. **Thread-local caches**: Fast access for repeated conversions
//!    (`AST_CACHE`, `EXPRESSION_BY_OPERANDS_PTR`).
//!
//! # Cross-cache invariants
//!
//! Two thread-local caches sit in this module (`AST_CACHE`,
//! `EXPRESSION_BY_OPERANDS_PTR`),
//! plus the shared `SymbolicIdentityRegistry`. Per-cache ownership /
//! invalidation / coherence is documented at each `thread_local!` block
//! (see angr-a2br.3, commit 9e4108df0). The invariants below cut ACROSS
//! caches and pin how the caches interact as a group. Each is referenced
//! from the per-cache docs by number so enforcement-site comments do not
//! duplicate the rationale.
//!
//! - **C1. EXPRESSION_ID sentinel boundary.** The `SymbolicIdentityRegistry`
//!   leaf-AST keys are real leaf-symbol ids allocated by
//!   `SymbolicIdentityRegistry::allocate_id`; compound `RustBV::Expression`
//!   nodes carry `id == RustBV::EXPRESSION_ID` (the `u64::MAX` sentinel) and
//!   route through `EXPRESSION_BY_OPERANDS_PTR`
//!   instead. Crossing this boundary corrupts `RustBV::Expression { id: u64 }`
//!   semantics — a leaf id stored in the Expression cache would collide
//!   with an unrelated operands pointer, and vice versa. Enforced by
//!   `debug_assert_ne!(symbol_id, RustBV::EXPRESSION_ID)` in
//!   `store_claripy_ast{,_with_info}`. See `value.rs` `RustBV::Expression`
//!   contract.
//!
//! - **C3. Unified `clear_ast_cache` invalidation.** Both
//!   thread-locals are cleared together in `clear_ast_cache`; a
//!   partial clear is never correct. Clearing only `AST_CACHE` would leave
//!   `EXPRESSION_BY_OPERANDS_PTR` still pinning operands `Arc`s for imports
//!   the next conversion will rebuild. `reset_for_new_exploration`
//!   extends the clear to the global registry; do NOT call
//!   `clear_global_registry()` in isolation — that drops the canonical
//!   `rust_id → AST` mapping while the thread-locals still reference those
//!   ids. Under the Option-A parallel model (angr-1ilq) the global clear is
//!   exploration-start / main-thread only; a worker clears only its own
//!   thread-locals via `clear_worker_local_caches` so it cannot wipe
//!   symbol identity that sibling workers still hold.
//!
//! - **C4. Expression nodes are cached by operands pointer only.**
//!   `EXPRESSION_BY_OPERANDS_PTR` (keyed by `Arc::as_ptr(operands) as
//!   usize`) returns the original Python `Expression` verbatim to
//!   preserve annotations (see angr-ykdq). It is the SOLE Rust→claripy
//!   expression cache: the export path (`rustbv_to_claripy_memo`) has
//!   only a `RustBV` in hand and cannot recompute the import-time
//!   claripy `ast_hash`, so a hash-keyed cache was structurally
//!   unreadable on export and was removed (angr-fawo). A miss falls
//!   back to rebuilding from `BVOp + operands` (losing annotations,
//!   never correctness). The cache additionally pins the operands
//!   `Arc` alive by storing a `RustBV` clone in the value slot —
//!   without that, allocator reuse of the raw pointer would produce a
//!   wrong-AST return.
//!
//! - **C5. Width check is enforced ONLY on `AST_CACHE` hits.** Other
//!   caches do not need it. `AST_CACHE` is keyed by claripy's
//!   content-addressed `__hash__()`, which already encodes length, so
//!   width-mismatched hits are extremely rare — but in the case of a
//!   recycled hash slot a wrong-width return would silently corrupt
//!   downstream VEX ops. The check at the use site (line ~478) evicts
//!   and reconverts on mismatch; Bool ASTs have `ast.length == None`
//!   and are treated as width 1 (matches RustBV bool representation).
//!   See `invariant-ast-cache-width-check`.
//!
//! ## Python-side counterparts
//!
//! The Python `RustExplorationManager` maintains an orthogonal set of
//! caches (in-memory `_init_cache`, disk init cache, `_mem_cache`,
//! `_state_cache`, `_state_metadata`). Those are NOT documented here;
//! see `state.rs` module rustdoc invariants **I3**, **I4**, **I5**,
//! **I6** for the cross-FFI rules and the bd memories
//! `invariant-init-cache-options-allowlist`,
//! `invariant-init-cache-options-mirror`,
//! `invariant-init-cache-user-symbolic`,
//! `invariant-init-cache-lazy-regions-order`,
//! `invariant-memcache-only-on-entry`,
//! `disk-cache-register-filter`,
//! `disk-init-cache-symbolic-reg-invariant`,
//! `invariant-state-metadata-dataclass` for the full rationale.

use pyo3::prelude::*;

#[macro_use]
mod cache;
mod export;
mod import;

pub(crate) use cache::*;
pub(crate) use export::*;
pub(crate) use import::*;

/// Error type for claripy bridge operations.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants; intra-crate matches must include a wildcard arm.
#[non_exhaustive]
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
    /// Recursive AST/Expression-tree descent exceeded the depth guard
    /// (angr-2a3i9). Raised instead of recursing further so a
    /// pathologically deep tree fails with a catchable error rather than
    /// overflowing the native Rust stack (SIGSEGV on the guard page, not
    /// visible to CPython's own recursion-limit counter since neither
    /// `claripy_to_rustbv` nor `rustbv_to_claripy_memo` is
    /// `#[pyfunction]`-wrapped).
    #[error("recursion depth limit exceeded: {0}")]
    RecursionLimit(String),
}

impl From<PyErr> for BridgeError {
    fn from(err: PyErr) -> Self {
        BridgeError::PythonError(err.to_string())
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

    // RustBV::Concrete stores its value in a u128, so any integer that needs
    // more than 128 significant bits cannot be represented exactly. Silently
    // masking to the low 128 bits (the old `clamp(1, 16)` behavior) produced a
    // wrong concrete value, e.g. `claripy.BVV(1 << 200, 256)` imported as 0.
    // Error loudly instead so callers don't operate on truncated data
    // (angr-cxw7). Wide *symbolic* BVs still work via the Concat path; this
    // only rejects wide concrete literals.
    if bit_length > 128 {
        return Err(BridgeError::InvalidArgs(format!(
            "concrete integer needs {bit_length} bits but RustBV::Concrete is capped at 128; \
             wide concrete BVV import is unsupported"
        )));
    }
    let byte_length = bit_length.div_ceil(8);
    let byte_length = byte_length.clamp(1, 16);

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

/// Check if a Python object is a claripy AST.
pub(crate) fn is_claripy_ast(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("op").unwrap_or(false) && obj.hasattr("args").unwrap_or(false)
}

#[cfg(test)]
#[path = "../claripy_bridge_tests.rs"]
mod tests;
