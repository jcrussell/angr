//! Unit tests for the raw-`Z3_ast`-pointer helpers.
//!
//! Split from the parent module per `rust-mod-tests-sibling-extraction`.
//!
//! angr-sqfj8.125: `z3_ast_to_eval_bv` is the single site that decides how a
//! borrowed Z3 node is lowered for evaluation — the Bool→1-bit rule from
//! angr-58ks and the decline-any-other-sort rule from angr-9ke6b.202 both
//! live there — yet it had no in-crate coverage. It does not need claripy:
//! the `Z3AstPtr` can be borrowed from a Z3 node built directly in Rust,
//! which is exactly the shape `extract_z3_ast_ptr` hands it. (The sibling
//! `extract_z3_ast_ptr` / `ast_to_bv_for_eval` do need claripy and stay
//! covered by `tests/engines/rust/test_solver_ops.py`.)

#![cfg(feature = "vex-engine-z3")]

use super::*;

use z3::ast::Ast;

/// Borrow a `Z3AstPtr` from a live Z3 node, mirroring what
/// `extract_z3_ast_ptr` produces from claripy's backend.
fn borrow<T: Ast>(node: &T) -> Z3AstPtr {
    let ctx = z3::Context::thread_local();
    let ptr = node.get_z3_ast().as_ptr() as usize;
    // SAFETY: `node` is still in scope, so the `Z3_ast` is live and its
    // refcount is >= 1 — the same claripy-AST-alive precondition
    // `extract_z3_ast_ptr` documents.
    unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) }.expect("non-null ptr")
}

/// angr-58ks: a Bool-sorted node is *lowered* to a 1-bit BV rather than
/// BV-wrapped. Wrapping it would trip Z3's error handler, which aborts the
/// process under `panic = "abort"` instead of returning an error.
#[test]
fn test_z3_ast_to_eval_bv_lowers_bool_to_1bit() {
    let ctx = RustSolverContext::new();
    for (value, expected) in [(true, 1u128), (false, 0u128)] {
        let node = z3::ast::Bool::from_bool(value);
        let bv = z3_ast_to_eval_bv(&borrow(&node)).expect("Bool must lower");
        assert_eq!(bv.width(), 1, "Bool lowers to exactly one bit");
        assert_eq!(ctx.i().ctx().eval(&bv), Some(expected));
    }
}

/// angr-9ke6b.202: the width comes from the Z3 sort, not from a separately
/// read claripy `.length` that could disagree with it.
#[test]
fn test_z3_ast_to_eval_bv_takes_width_from_the_z3_sort() {
    let ctx = RustSolverContext::new();
    for width in [1u32, 8, 32, 64, 128] {
        let node = z3::ast::BV::from_u64(1, width);
        let bv = z3_ast_to_eval_bv(&borrow(&node)).expect("BV must wrap");
        assert_eq!(bv.width(), width);
        assert_eq!(ctx.i().ctx().eval(&bv), Some(1));
    }
}

/// A symbolic (non-constant) BV survives the wrap and stays tied to the
/// solver's own view of that constant — the property that makes this path
/// worth having at all (it preserves identity with asserted constraints).
#[test]
fn test_z3_ast_to_eval_bv_preserves_constraint_identity() {
    let ctx = RustSolverContext::new();
    let node = z3::ast::BV::new_const("z3ptr_ident_x", 8);
    ctx.i().ctx().assume_true(&{
        let wrapped = z3_ast_to_eval_bv(&borrow(&node)).expect("BV must wrap");
        let seven = crate::symbolic::RustBV::concrete(7, 8);
        wrapped.eq(&seven, &ctx.i().ctx())
    });

    let bv = z3_ast_to_eval_bv(&borrow(&node)).expect("BV must wrap");
    assert_eq!(ctx.i().ctx().eval(&bv), Some(7));
}

/// Neither BV nor Bool: declined, so the caller can report the sort itself
/// (`eval_z3_ast_ptr` turns this into a `PyRuntimeError`) instead of
/// mis-wrapping the node.
#[test]
fn test_z3_ast_to_eval_bv_declines_other_sorts() {
    let int_node = z3::ast::Int::from_i64(5);
    let ptr = borrow(&int_node);
    assert!(!ptr.is_bool());
    assert_eq!(ptr.bv_width(), None);
    assert!(z3_ast_to_eval_bv(&ptr).is_none());
}
