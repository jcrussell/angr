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
//! The numbering is deliberately non-contiguous: **C2** ("`CLARIPY_AST_CACHE`
//! ⊆ global_registry") and **C6** ("`CLARIPY_AST_CACHE` uses `std::HashMap`")
//! both described the `CLARIPY_AST_CACHE` thread-local, which was dropped as
//! dead in angr-4xaga.2. The survivors kept their original numbers so that the
//! existing `C3`/`C4`/`C5` citations — in `cache.rs`, `export.rs`,
//! `cache_tests.rs` and `docs/advanced-topics/rust_engine.rst` — stayed valid;
//! renumber only if every one of those is swept in the same commit.
//!
//! - **C1. EXPRESSION_ID sentinel boundary.** The `SymbolicIdentityRegistry`
//!   leaf-AST keys are real leaf-symbol ids allocated by `SymContext::next_id`
//!   (the registry itself mints nothing); compound `RustBV::Expression`
//!   nodes carry `id == RustBV::EXPRESSION_ID` (the `u64::MAX` sentinel) and
//!   route through `EXPRESSION_BY_OPERANDS_PTR`
//!   instead. Crossing this boundary corrupts `RustBV::Expression { id: u64 }`
//!   semantics — a leaf id stored in the Expression cache would collide
//!   with an unrelated operands pointer, and vice versa. Enforced by an
//!   always-on `assert_ne!(symbol_id, RustBV::EXPRESSION_ID)` in
//!   `store_claripy_ast_with_info` (promoted from `debug_assert_ne!` in
//!   angr-9ke6b.220). See `value.rs` `RustBV::Expression` contract.
//!
//! - **C3. Unified `clear_ast_cache` invalidation.** Both
//!   thread-locals are cleared together in `clear_ast_cache`; a
//!   partial clear is never correct. Clearing only `AST_CACHE` would leave
//!   `EXPRESSION_BY_OPERANDS_PTR` still pinning operands `Arc`s for imports
//!   the next conversion will rebuild. `reset_for_new_exploration`
//!   extends the clear to the global registry; do NOT call
//!   `clear_global_registry()` in isolation — that drops the canonical
//!   `rust_id → AST` mapping while the thread-locals still reference those
//!   ids. The extended clear is `#[cfg(test)]` (angr-9ke6b.218 item 4):
//!   production never wipes the global registry, because symbol identity is
//!   shared across coexisting managers and across sibling workers under the
//!   Option-A parallel model (angr-1ilq). A worker clears only its own
//!   thread-locals via `clear_worker_local_caches`.
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
//! - **C5. An `AST_CACHE` hit is width-checked before it is returned.**
//!   `AST_CACHE` is keyed by claripy's content-addressed `__hash__()`,
//!   which already encodes length, so width-mismatched hits are
//!   extremely rare — but in the case of a recycled hash slot a
//!   wrong-width return would silently corrupt downstream VEX ops. The
//!   check at the use site (`import::claripy_to_rustbv_depth`, the
//!   `AST_CACHE` hit arm) evicts and reconverts on mismatch; Bool ASTs
//!   have `ast.length == None` and are treated as width 1 (matches
//!   RustBV bool representation). See `invariant-ast-cache-width-check`.
//!
//!   `EXPRESSION_BY_OPERANDS_PTR` needs no such check: its key is an
//!   operands `Arc` pointer whose `RustBV` value slot pins the `Arc`
//!   alive (C4), so the key cannot be recycled under a live entry. The
//!   `SymbolicIdentityRegistry` DOES need an analogous guard, for a
//!   different reason — its `rust_id → AST` key can alias across symbols
//!   (angr-owr37), not merely collide — so `export::rustbv_to_claripy_memo`
//!   width-checks a registry hit against the requested symbol width and
//!   evicts via `evict_claripy_ast` on mismatch. The two guards share the
//!   Bool/unreadable-`length` fallback in `cache::claripy_ast_guard_width`;
//!   they are deliberately separate call sites because their recovery
//!   differs (reconvert vs re-mint).
//!
//! ## Python-side counterparts
//!
//! The Python `RustExplorationManager` maintains an orthogonal set of
//! caches (in-memory `_init_cache`, disk init cache, `_mem_cache`,
//! `_state_cache`). Per-state metadata is NOT among them — it lives on
//! the Rust `RustSimState` and is freed when Rust drops the state (see
//! `RustStateCacheMixin::_cleanup_symbolic_pages_cache`). None of these
//! are documented here; see the `state` module rustdoc invariants
//! **I3**, **I4**, **I5**, **I6** for the cross-FFI rules and the bd
//! memories
//! `invariant-init-cache-plugin-coverage` (what the init cache does and
//! does not carry; the option allow-list is a *mirror*, not additive),
//! `invariant-apply-state-metadata-option-allowlist` (value-changing
//! SimOptions must be on that list),
//! `invariant-init-cache-concrete-input-digest` (argv/env in the key),
//! `disk-init-cache-symbolic-reg-invariant` (the user-symbolic
//! cache-disable gate), `disk-cache-register-filter`,
//! `invariant-state-cache-mirror-id`, and
//! `invariant-lazy-region-auto-map` for the full rationale.
//!
//! `_mem_cache` has no memory of its own (the former citation here named
//! a key that never existed, angr-sqfj8.16): it is **consume-once**, set
//! by `RustExplorationManager._load_init_from_disk_cache`'s caller and
//! taken — then reset to `None` — by `_try_fast_memory_sync` on the
//! first `_sync_memory_to_rust`, so it is a one-shot hand-off, not a
//! cache that can go stale. `invariant-state-id-cache-epoch-audit`
//! records it under that classification alongside the other Python-side
//! caches that are *not* the stale-serve-after-Rust-steps shape.

use pyo3::prelude::*;

// `tl_cache!` is scoped to `cache` (and its `cache_tests` child) now that
// `AST_CACHE` is private behind wrappers (angr-9ke6b.44), so no `#[macro_use]`.
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

/// Wording hub for a failed `claripy_to_rustbv` at the PyO3 boundary
/// (angr-12jjk.4).
///
/// `what` names the thing being imported — `"register rax"`,
/// `"memory 0x400080"`, `"content byte 3"` — and is rendered as
/// `AST import failed (<what>): <err>`. Call it from inside the `map_err`
/// closure so the `format!` for a dynamic `what` stays off the success path.
///
/// The `ValueError` class is the one every import site already raised: a
/// bridge failure means the caller handed us an AST this engine cannot
/// represent, which is a bad-argument condition, not an engine fault.
/// Exports use [`ast_export_err`] and `RuntimeError` for the mirror-image
/// reason.
///
/// Both helpers take the error as `impl Display` rather than a concrete type
/// because the two directions disagree: `claripy_to_rustbv` fails with
/// [`BridgeError`], `rustbv_to_claripy` with `PyErr`.
pub(crate) fn ast_import_err(what: &str, e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("AST import failed ({what}): {e}"))
}

/// Wording hub for a failed `rustbv_to_claripy` at the PyO3 boundary
/// (angr-12jjk.4). See [`ast_import_err`] for the `what` convention.
///
/// Raises `RuntimeError`: the caller supplied no AST here, so a failure to
/// build one out of a value the engine itself is holding is an engine fault.
pub(crate) fn ast_export_err(what: &str, e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("AST export failed ({what}): {e}"))
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

    // For larger values, use Python's int.to_bytes. Propagate an extract
    // failure rather than assuming some width: guessing here would either
    // truncate a wide value or spuriously trip the >128 rejection below, and
    // both are the silently-wrong-answer outcome the angr-cxw7 comment exists
    // to prevent (angr-sqfj8.24).
    let bit_length: usize = obj.call_method0("bit_length")?.extract()?;

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

    // `int.to_bytes(byte_length, ..)` returns exactly `byte_length` bytes and
    // `byte_length <= 16` follows from the `bit_length > 128` rejection above,
    // so this is unreachable for a well-behaved `int`. Check it anyway rather
    // than skipping the excess bytes inside the loop: `obj` is caller-supplied
    // Python, so a `to_bytes` override can hand back a wider buffer, and
    // masking that to the low 128 bits is the silently-wrong-answer outcome
    // the `bit_length > 128` rejection exists to prevent (angr-cxw7).
    if bytes.len() > 16 {
        return Err(BridgeError::InvalidArgs(format!(
            "to_bytes returned {} bytes for a {bit_length}-bit integer; \
             RustBV::Concrete holds at most 16",
            bytes.len()
        )));
    }

    let mut value: u128 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        value |= (b as u128) << (i * 8);
    }

    Ok(value)
}

/// Check if a Python object is a claripy AST.
///
/// SILENT(cat-a): a `hasattr` that itself errors (a `__getattr__` that raises)
/// is treated as "attribute absent", i.e. "not an AST". This is a dispatch
/// gate only — both call sites (`load_from_callback` in `interpreter/mod.rs`
/// and `try_convert_symbolic_value` in `interpreter/expressions.rs`) answer
/// `false` by falling through to a fresh symbolic BV, which is exactly what a
/// failed
/// `claripy_to_rustbv` on such an object would have produced anyway. Nothing
/// downstream can observe the difference, so there is no error to propagate.
pub(crate) fn is_claripy_ast(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("op").unwrap_or(false) && obj.hasattr("args").unwrap_or(false)
}

test_submod!("../claripy_bridge_tests.rs" => tests);
