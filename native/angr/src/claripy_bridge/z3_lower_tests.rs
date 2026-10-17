//! `z3_lower` must agree with the engine's own Z3 terms: a lowered constraint
//! binds the same constants a `RustBV` builds, and every op keeps clarirs'
//! semantics. Each test asserts a lowered constraint next to a
//! hand-built z3-rs term and checks satisfiability, so a renamed constant or
//! a wrong op shows up as a flipped verdict.

#![allow(clippy::unwrap_used, reason = "test fixtures")]

use clarirs_core::prelude::*;
use z3::ast::{Ast, BV, Bool};

use super::{BridgeError, lower};
use crate::claripy::ast::GLOBAL_CONTEXT;

fn bvv(value: u64, width: u32) -> AstRef<'static> {
    GLOBAL_CONTEXT.bvv(BitVec::from((value, width))).unwrap()
}

/// Wrap a lowered Bool-sorted term as a z3-rs `Bool` on the same context.
fn as_bool(ast: &AstRef<'static>) -> Bool {
    let lowered = lower(ast).unwrap();
    assert!(lowered.is_bool(), "expected a Bool-sorted lowering");
    // SAFETY: `lowered` holds a reference to a live Bool term of the
    // thread-local context; `wrap` takes its own.
    unsafe { Bool::wrap(lowered.context(), lowered.as_z3_ast()) }
}

fn check(constraints: &[Bool]) -> z3::SatResult {
    let solver = z3::Solver::new();
    for c in constraints {
        solver.assert(c);
    }
    solver.check()
}

#[test]
fn bv_leaf_names_the_same_constant_as_rustbv() {
    // `x + 1 == 5` lowered, against the engine's own `x` (a plain
    // `BV::new_const`, which is what `RustBV::Symbolic` builds for a BV leaf).
    let x = GLOBAL_CONTEXT.bvs("lower_x", 32).unwrap();
    let sum = GLOBAL_CONTEXT.add(x, bvv(1, 32)).unwrap();
    let constraint = as_bool(&GLOBAL_CONTEXT.eq_(sum, bvv(5, 32)).unwrap());

    let engine_x = BV::new_const("lower_x", 32);
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(&[constraint.clone(), engine_x.eq(BV::from_u64(4, 32))]), z3::SatResult::Sat);
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(&[constraint, engine_x.eq(BV::from_u64(3, 32))]), z3::SatResult::Unsat);
}

#[test]
fn concat_extract_and_reverse_keep_bit_order() {
    let y = GLOBAL_CONTEXT.bvs("lower_y", 16).unwrap();
    // Reverse(y) == 0x3412  <=>  y == 0x1234
    let reversed = GLOBAL_CONTEXT.byte_reverse(y.clone()).unwrap();
    let rev = as_bool(&GLOBAL_CONTEXT.eq_(reversed, bvv(0x3412, 16)).unwrap());
    // Concat(Extract(7, 0, y), Extract(15, 8, y)) is the same swap.
    let lo = GLOBAL_CONTEXT.extract(y.clone(), 7, 0).unwrap();
    let hi = GLOBAL_CONTEXT.extract(y, 15, 8).unwrap();
    let swapped = GLOBAL_CONTEXT.concat([lo, hi]).unwrap();
    let cat = as_bool(&GLOBAL_CONTEXT.eq_(swapped, bvv(0x3412, 16)).unwrap());

    let engine_y = BV::new_const("lower_y", 16);
    let not_1234 = engine_y.eq(BV::from_u64(0x1234, 16)).not();
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(&[rev, not_1234.clone()]), z3::SatResult::Unsat);
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(&[cat, not_1234]), z3::SatResult::Unsat);
}

#[test]
fn float_ops_lower_with_ieee_semantics() {
    // f * 2.0 == 6.0 forces f == 3.0: doubling is exact, so no rounding can
    // admit a neighbour (an `f + 1.0 == 3.0` premise would not — 2 - ulp
    // ties to 3.0 under round-nearest-even).
    let f = GLOBAL_CONTEXT.fps("lower_f", FSort::f64()).unwrap();
    let two = GLOBAL_CONTEXT.fpv(2.0f64).unwrap();
    let six = GLOBAL_CONTEXT.fpv(6.0f64).unwrap();
    let product = GLOBAL_CONTEXT.fp_mul(f.clone(), two, FPRM::NearestTiesToEven).unwrap();
    let constraint = as_bool(&GLOBAL_CONTEXT.eq_(product, six).unwrap());

    let three = GLOBAL_CONTEXT.fpv(3.0f64).unwrap();
    let not_three = as_bool(&GLOBAL_CONTEXT.neq(f, three).unwrap());
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(std::slice::from_ref(&constraint)), z3::SatResult::Sat);
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(&[constraint, not_three]), z3::SatResult::Unsat);
}

#[test]
fn bool_xor_lowers() {
    // `claripy_to_rustbv` has no arm for a Bool `Xor`; the lowering does.
    let a = GLOBAL_CONTEXT.bools("lower_a").unwrap();
    let b = GLOBAL_CONTEXT.bools("lower_b").unwrap();
    let xor = as_bool(&GLOBAL_CONTEXT.xor([a, b]).unwrap());
    let engine_a = Bool::new_const("lower_a");
    let engine_b = Bool::new_const("lower_b");
    // satresult-exempt: test oracle comparing two decided verdicts
    assert_eq!(check(&[xor, engine_a.eq(&engine_b)]), z3::SatResult::Unsat);
}

#[test]
fn shared_subtrees_lower_once() {
    // A balanced DAG whose tree expansion is 2^24 nodes; without the memo
    // this would not finish.
    let mut node = GLOBAL_CONTEXT.bvs("lower_dag", 8).unwrap();
    for _ in 0..24 {
        node = GLOBAL_CONTEXT.add(node.clone(), node).unwrap();
    }
    assert!(lower(&node).unwrap().bv_width() == Some(8));
}

#[test]
fn strings_are_refused_not_mislowered() {
    let s = GLOBAL_CONTEXT.strings("lower_s").unwrap();
    let len = GLOBAL_CONTEXT.str_len(s).unwrap();
    assert!(matches!(lower(&len), Err(BridgeError::UnsupportedOp(_))));
}

#[test]
fn extending_a_lowered_term_builds_only_the_new_node() {
    // The engine wraps each op's result as it is built, so lowering a chain
    // one prefix at a time must cost one node per step, not the whole prefix.
    let one = bvv(1, 32);
    let mut acc = GLOBAL_CONTEXT.bvs("lower_chain", 32).unwrap();
    lower(&acc).unwrap();
    lower(&one).unwrap();
    for _ in 0..64 {
        acc = GLOBAL_CONTEXT.add(acc, one.clone()).unwrap();
        let before = super::NODES_BUILT.get();
        lower(&acc).unwrap();
        assert_eq!(super::NODES_BUILT.get() - before, 1);
    }
    // Lowering it again builds nothing.
    let before = super::NODES_BUILT.get();
    lower(&acc).unwrap();
    assert_eq!(super::NODES_BUILT.get(), before);
}

#[test]
fn a_context_swap_does_not_reuse_terms_from_the_old_context() {
    let x = GLOBAL_CONTEXT.bvs("lower_swap", 32).unwrap();
    let original = z3::Context::thread_local();
    let first = lower(&x).unwrap();
    assert_eq!(first.context().get_z3_context(), original.get_z3_context());

    let fresh = z3::Context::new(&z3::Config::new());
    z3::Context::set_thread_local(&fresh);
    let second = lower(&x).unwrap();
    z3::Context::set_thread_local(&original);
    assert_eq!(second.context().get_z3_context(), fresh.get_z3_context());
}
