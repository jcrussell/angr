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
    /// Returns [`SortKind::Unknown`] if Z3 cannot resolve the AST's sort
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
    pub fn is_bool(&self) -> bool {
        self.sort_kind() == z3_sys::SortKind::Bool
    }

    /// Whether the wrapped AST is bit-vector-sorted.
    pub fn is_bv(&self) -> bool {
        self.sort_kind() == z3_sys::SortKind::BV
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
mod tests {
    use super::*;
    use z3::ast::{Ast, Bool};

    /// Constructing and dropping a `Z3AstPtr` should not crash and should
    /// not affect the underlying AST's liveness while the original holder
    /// still keeps a ref.
    #[test]
    fn test_from_borrowed_raw_inc_dec_balanced() {
        let ctx = Context::thread_local();
        let original = Bool::from_bool(true);
        let raw_ptr = original.get_z3_ast().as_ptr() as usize;

        // SAFETY: `original` keeps the AST alive; raw_ptr is its Z3_ast.
        let wrapped = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, raw_ptr) }.expect("non-null ptr");

        assert_eq!(wrapped.as_usize(), raw_ptr);
        drop(wrapped);

        // `original` still in scope — AST must still be valid. Reading its
        // boolean value would crash if Drop had taken the last ref.
        assert_eq!(original.as_bool(), Some(true));
    }

    #[test]
    fn test_from_borrowed_raw_null_returns_none() {
        let ctx = Context::thread_local();
        // SAFETY: passing zero is the documented null case.
        let result = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, 0) };
        assert!(result.is_none(), "null ptr must yield None");
    }

    #[test]
    fn test_clone_ref_independently_droppable() {
        let ctx = Context::thread_local();
        let original = Bool::from_bool(false);
        let raw_ptr = original.get_z3_ast().as_ptr() as usize;

        // SAFETY: `original` keeps the AST alive; raw_ptr is its Z3_ast.
        let a = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, raw_ptr) }.unwrap();
        let b = a.clone_ref();
        assert_eq!(a.as_usize(), b.as_usize());

        // Both handles should drop independently without underflow.
        drop(a);
        drop(b);

        // `original` still keeps the AST alive.
        assert_eq!(original.as_bool(), Some(false));
    }
}
