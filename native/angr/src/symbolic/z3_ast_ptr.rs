//! Typed, refcounted wrapper around a raw `Z3_ast` pointer.
//!
//! Replaces the `usize` parameter-passing pattern at the claripy ↔ Rust FFI
//! boundary (`extract_z3_ast_ptr` / `add_constraint_raw`). Carries a Z3
//! [`Context`] clone so the underlying `Z3_context` outlives the AST — the
//! same ownership shape z3-rs uses for [`z3::ast::Bool`] / [`z3::ast::BV`].
//!
//! # Refcount discipline
//!
//! Z3 ASTs are reference-counted (see `Z3_inc_ref` / `Z3_dec_ref` in the
//! Z3 C API). [`Z3AstPtr::from_borrowed_raw`] takes its own ref so the
//! handle's lifetime is independent of whatever holder originally produced
//! the pointer (typically claripy's Z3 backend). [`Drop`] releases that ref.
//!
//! # Non-`Copy` / non-`Clone`
//!
//! The wrapper is intentionally non-`Copy`. To duplicate the reference,
//! use [`Z3AstPtr::clone_ref`], which performs a fresh `Z3_inc_ref`.
//! Bitwise duplication would call `Drop` twice and underflow the refcount.

#![cfg(feature = "vex-engine-z3")]

use std::ptr::NonNull;

use z3::Context;
use z3_sys::{_Z3_ast, Z3_ast, Z3_dec_ref, Z3_inc_ref};

/// Owned reference to a `Z3_ast` belonging to some [`Context`].
///
/// The handle keeps a `Context` clone so the underlying `Z3_context` (and
/// therefore the AST itself) outlives the wrapper. Construction takes a
/// fresh `Z3_inc_ref`; `Drop` matches it with `Z3_dec_ref`.
pub struct Z3AstPtr {
    ptr: NonNull<_Z3_ast>,
    ctx: Context,
}

impl Z3AstPtr {
    /// Wrap a borrowed `Z3_ast` pointer, taking a fresh reference.
    ///
    /// Returns `None` if `ptr_value == 0`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that, at the moment of this call:
    /// - `ptr_value` is either zero or a valid `Z3_ast` belonging to `ctx`.
    /// - When non-zero, the underlying AST has refcount ≥ 1 (i.e., some
    ///   other holder is keeping it alive long enough for `Z3_inc_ref` to
    ///   succeed).
    ///
    /// Both preconditions are satisfied by pointers extracted from
    /// claripy's z3 backend — claripy retains the AST in its cache for the
    /// duration of the Python expression's lifetime.
    pub unsafe fn from_borrowed_raw(ctx: &Context, ptr_value: usize) -> Option<Self> {
        let nn = NonNull::new(ptr_value as *mut _Z3_ast)?;
        // SAFETY: caller guarantees `nn` is a live `Z3_ast` in `ctx`.
        unsafe {
            Z3_inc_ref(ctx.get_z3_context(), nn);
        }
        Some(Self {
            ptr: nn,
            ctx: ctx.clone(),
        })
    }

    /// Get the underlying `Z3_ast` pointer without releasing the ref.
    pub fn as_z3_ast(&self) -> Z3_ast {
        self.ptr
    }

    /// Get the raw pointer as `usize` (interop with diagnostic counters /
    /// HashSet keys that still operate on raw addresses).
    pub fn as_usize(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    /// Get the bound [`Context`] for this AST.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Query the Z3 sort kind of the wrapped AST.
    ///
    /// A pure metadata read. Consumers use it to verify the sort before
    /// wrapping the raw AST as a [`z3::ast::BV`] / [`z3::ast::Bool`]:
    /// wrapping a Bool node as a BV (or vice-versa) and then operating on
    /// it trips Z3's error handler, which aborts the process rather than
    /// returning a recoverable error.
    ///
    /// Returns `SortKind::Unknown` if Z3 cannot resolve the AST's sort
    /// (should not happen for a live AST, but keeps the call total).
    pub fn sort_kind(&self) -> z3_sys::SortKind {
        let raw_ctx = self.ctx.get_z3_context();
        // SAFETY: `self.ptr` is a live `Z3_ast` in `self.ctx` (we hold a
        // ref taken in `from_borrowed_raw`/`clone_ref`). `Z3_get_sort` on a
        // live AST yields a live `Z3_sort` in the same context, and
        // `Z3_get_sort_kind` is a pure metadata read on it.
        unsafe {
            match z3_sys::Z3_get_sort(raw_ctx, self.ptr) {
                Some(sort) => z3_sys::Z3_get_sort_kind(raw_ctx, sort),
                None => z3_sys::SortKind::Unknown,
            }
        }
    }

    /// Whether the wrapped AST is Bool-sorted.
    ///
    /// The BV counterpart is [`Z3AstPtr::bv_width`] — every caller that
    /// wants "is this a BV?" also needs the width to wrap it, so there is
    /// deliberately no separate `is_bv()`.
    pub fn is_bool(&self) -> bool {
        self.sort_kind() == z3_sys::SortKind::Bool
    }

    /// Bit width of the wrapped AST, or `None` if it is not BV-sorted.
    ///
    /// This is the authoritative width for a raw Z3 AST reaching us from
    /// claripy's backend: it comes from the Z3 sort itself, so it cannot
    /// disagree with the node we are about to `BV::wrap`. Callers used to
    /// read claripy's `.length` attribute instead (angr-9ke6b.202), which
    /// both duplicated the wrap's precondition and had no correct answer
    /// when the attribute was missing.
    pub fn bv_width(&self) -> Option<u32> {
        let raw_ctx = self.ctx.get_z3_context();
        // SAFETY: `self.ptr` is a live `Z3_ast` in `self.ctx` (we hold a ref
        // taken in `from_borrowed_raw`/`clone_ref`). `Z3_get_sort` on a live
        // AST yields a live `Z3_sort` in the same context, and
        // `Z3_get_bv_sort_size` is a pure metadata read that is only valid
        // on a BV sort — which the `SortKind::BV` match guarantees.
        unsafe {
            let sort = z3_sys::Z3_get_sort(raw_ctx, self.ptr)?;
            if z3_sys::Z3_get_sort_kind(raw_ctx, sort) != z3_sys::SortKind::BV {
                return None;
            }
            Some(z3_sys::Z3_get_bv_sort_size(raw_ctx, sort))
        }
    }

    /// Duplicate the handle by taking another reference. Cheaper than
    /// rebuilding the AST but still costs one FFI call.
    pub fn clone_ref(&self) -> Self {
        // SAFETY: self.ptr is live (we hold a ref) and `self.ctx` is the
        // context the ref was taken under.
        unsafe {
            Z3_inc_ref(self.ctx.get_z3_context(), self.ptr);
        }
        Self {
            ptr: self.ptr,
            ctx: self.ctx.clone(),
        }
    }
}

impl Drop for Z3AstPtr {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is live because we hold the ref taken in
        // `from_borrowed_raw` / `clone_ref`. `self.ctx` keeps the
        // `Z3_context` alive for this dec_ref call.
        unsafe {
            Z3_dec_ref(self.ctx.get_z3_context(), self.ptr);
        }
    }
}

// Z3 ASTs are bound to a thread-local context in z3-rs 0.19+; passing the
// handle across threads is unsafe regardless of `Sync`. We deliberately do
// not implement `Send` or `Sync`.

#[cfg(test)]
#[path = "z3_ast_ptr_tests.rs"]
mod tests;
