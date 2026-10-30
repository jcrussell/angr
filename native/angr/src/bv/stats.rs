//! The value layer's own counters: AST-construction shapes and the
//! construction-time rewrites `value_ops.rs` counts. Process-global, so the
//! ops record without threading state through; `symbolic::stats` reads them
//! into `get_solver_stats()` and resets them with the rest, so the Python
//! stats keys are unchanged (angr-mx00p.116.7).

use std::sync::atomic::{AtomicU64, Ordering};

// AST construction (specific) — Reverse/Concat/Extract emissions. These
// are the AST shapes most often blamed when claripy<->Rust mismatch surfaces
// (angr-tlvl-style residuals). Counted at the public `RustBV::{reverse,
// concat, extract}` entry points; internal recursion within `*_into` is
// not double-counted.
pub static BVOP_REVERSE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static BVOP_CONCAT_COUNT: AtomicU64 = AtomicU64::new(0);
pub static BVOP_EXTRACT_COUNT: AtomicU64 = AtomicU64::new(0);

// angr-g7nq: trivial-constraint fast path counters. Bumped from
// `value.rs::{eq_into,ne_into,ult_into,ule_into,ugt_into,uge_into}` when
// the `Cmp(ZeroExt(k, x), BVV)` (or commuted form) pattern is recognized
// and either collapsed to a smaller AST (the "high bits zero" case) or
// short-circuited to a concrete answer (the "high bits nonzero" case, an
// unsatisfiable/tautological subexpression). Separate counts for the two
// outcomes make it possible to attribute hits without inspecting the
// emitted ASTs.
pub static ZEXT_CMP_COLLAPSE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ZEXT_CMP_TRIVIAL_DECIDE_COUNT: AtomicU64 = AtomicU64::new(0);

// angr-1joc (a) commutative-arg-order: how often the RustBV-level
// `canonicalize_commutative` actually swaps operands. Swap rate ≈ how often
// the un-canonicalized RustBV order WOULD have produced a distinct Z3 AST.
// Bumped from `value.rs::canonicalize_commutative`.
pub static RUSTBV_COMMUTATIVE_CANONICALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static RUSTBV_COMMUTATIVE_SWAP_COUNT: AtomicU64 = AtomicU64::new(0);

/// Record a public `RustBV::reverse` call.
#[inline]
pub fn record_bvop_reverse() {
    BVOP_REVERSE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Record a public `RustBV::concat` call.
#[inline]
pub fn record_bvop_concat() {
    BVOP_CONCAT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Record a public `RustBV::extract` call.
#[inline]
pub fn record_bvop_extract() {
    BVOP_EXTRACT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-g7nq: record a `Cmp(ZeroExt(k, x), BVV)` collapse — high k bits of
/// const are zero, so the comparison is rewritten on the W-k-bit operands.
#[inline]
pub fn record_zext_cmp_collapse() {
    ZEXT_CMP_COLLAPSE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-g7nq: record a `Cmp(ZeroExt(k, x), BVV)` trivial decision — the
/// constant has a nonzero high k-bit prefix, so the subexpression is
/// structurally unsatisfiable (Eq) or always satisfiable (Ne) or has a
/// constant unsigned ordering, and the result is a concrete 0/1 bool.
#[inline]
pub fn record_zext_cmp_trivial_decide() {
    ZEXT_CMP_TRIVIAL_DECIDE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-1joc: record a `canonicalize_commutative` invocation. `swapped` is
/// true when the canonical order required swapping the operand pair.
#[inline]
pub fn record_commutative_canonicalize(swapped: bool) {
    RUSTBV_COMMUTATIVE_CANONICALIZE_COUNT.fetch_add(1, Ordering::Relaxed);
    if swapped {
        RUSTBV_COMMUTATIVE_SWAP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}
