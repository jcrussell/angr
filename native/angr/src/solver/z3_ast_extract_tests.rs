//! Unit tests for the claripy-node evaluation helper.
//!
//! Split from the parent module per `rust-module-root-split-recipe`.
//!
//! angr-sqfj8.125: `node_to_eval_bv` is the single site that decides how a
//! claripy node is turned into a value to evaluate — the Bool→1-bit rule from
//! angr-58ks and the decline-any-other-sort rule from angr-9ke6b.202 both
//! live there. It needs no Python: the nodes are built directly in clarirs,
//! which is what `claripy_node` reads out of a claripy AST. (The sibling
//! `claripy_node` / `ast_to_bv_for_eval` do need claripy and stay covered by
//! `tests/engines/rust/test_solver_ops.py`.)

#![allow(clippy::unwrap_used, reason = "test fixtures")]

use super::*;

use clarirs_core::prelude::{AstFactory, BitVec, FSort};

use crate::claripy::ast::GLOBAL_CONTEXT;

/// angr-58ks: a Bool-sorted node becomes a 1-bit value, never a mis-sorted BV.
#[test]
fn test_node_to_eval_bv_lowers_bool_to_1bit() {
    let ctx = RustSolverContext::new();
    for (value, expected) in [(true, 1u128), (false, 0u128)] {
        let node = GLOBAL_CONTEXT.boolv(value).unwrap();
        let bv = node_to_eval_bv(&node).expect("Bool must lower");
        assert_eq!(bv.width(), 1, "Bool lowers to exactly one bit");
        assert_eq!(ctx.i().ctx().eval(&bv), Some(expected));
    }
}

/// angr-9ke6b.202: the width is the node's own.
#[test]
fn test_node_to_eval_bv_takes_width_from_the_node() {
    let ctx = RustSolverContext::new();
    for width in [1u32, 8, 32, 64, 128] {
        let node = GLOBAL_CONTEXT.bvv(BitVec::from((1u64, width))).unwrap();
        let bv = node_to_eval_bv(&node).expect("BV must convert");
        assert_eq!(bv.width(), width);
        assert_eq!(ctx.i().ctx().eval(&bv), Some(1));
    }
}

/// A symbolic node is the same variable the solver constrains, which is what
/// makes this path worth having: a constraint built from the node binds the
/// evaluated value.
#[test]
fn test_node_to_eval_bv_preserves_constraint_identity() {
    let ctx = RustSolverContext::new();
    let node = GLOBAL_CONTEXT.bvs("evalnode_ident_x", 8).unwrap();
    let seven = crate::symbolic::RustBV::concrete(7, 8);
    let x = node_to_eval_bv(&node).unwrap();
    ctx.i().ctx().assume_true(&x.eq(&seven, &ctx.i().ctx()));

    let again = node_to_eval_bv(&node).unwrap();
    assert_eq!(ctx.i().ctx().eval(&again), Some(7));
}

/// Neither BV nor Bool: declined with the sort, so the caller can report it
/// (`eval_claripy_node` turns this into a `PyRuntimeError`).
#[test]
fn test_node_to_eval_bv_declines_other_sorts() {
    let node = GLOBAL_CONTEXT.fps("evalnode_float", FSort::f32()).unwrap();
    let err = node_to_eval_bv(&node).unwrap_err();
    assert!(err.contains("Float"), "{err}");
}
