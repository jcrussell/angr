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

/// The conversions that take a rounding mode and a sort lower to working
/// terms. Each used to build its sort before its rounding mode, and Z3 frees
/// an unreferenced sort at the next call, so every one came back null.
#[test]
fn rounded_conversions_lower() {
    let gc = &*GLOBAL_CONTEXT;
    let rne = FPRM::NearestTiesToEven;
    let minus_one = bvv(0xFFFF_FFFF, 32);
    let cases = [
        (gc.bv_to_fp_signed(minus_one.clone(), FSort::f64(), rne).unwrap(), -1.0),
        (gc.bv_to_fp_unsigned(minus_one, FSort::f64(), rne).unwrap(), 4_294_967_295.0),
        (gc.fp_to_fp(gc.fpv(1.5f32).unwrap(), FSort::f64(), rne).unwrap(), 1.5),
    ];
    for (got, want) in cases {
        let differs = as_bool(&gc.neq(got.clone(), gc.fpv(want).unwrap()).unwrap());
        // satresult-exempt: test oracle comparing two decided verdicts
        assert_eq!(check(&[differs]), z3::SatResult::Unsat, "{got:?} != {want}");
    }
}

/// `fp.min`/`fp.max` take the number over a NaN, `fp.fma` rounds once and
/// `fp.roundToIntegral` follows its rounding mode: each `got != want` is unsat.
#[test]
fn min_max_fma_and_round_to_integral_lower_with_ieee_semantics() {
    let gc = &*GLOBAL_CONTEXT;
    let v = |x: f64| gc.fpv(x).unwrap();
    let rne = FPRM::NearestTiesToEven;
    let cases = [
        (gc.fp_min(v(f64::NAN), v(1.0)).unwrap(), 1.0),
        (gc.fp_max(v(2.0), v(f64::NAN)).unwrap(), 2.0),
        // (1 + eps) * (1 - eps) - 1 is -2^-104 only when the product is not
        // rounded first.
        (
            gc.fp_fma(v(1.0 + f64::EPSILON), v(1.0 - f64::EPSILON), v(-1.0), rne)
                .unwrap(),
            -(2f64.powi(-104)),
        ),
        (gc.fp_round_to_integral(v(2.5), rne).unwrap(), 2.0),
        (gc.fp_round_to_integral(v(2.5), FPRM::TowardPositive).unwrap(), 3.0),
    ];
    for (got, want) in cases {
        let differs = as_bool(&gc.neq(got.clone(), v(want)).unwrap());
        // satresult-exempt: test oracle comparing two decided verdicts
        assert_eq!(check(&[differs]), z3::SatResult::Unsat, "{got:?} != {want}");
    }
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

    // `with_z3_config` installs a fresh context as this thread's thread-local
    // for the closure; only raw pointers (as `usize`) leave it.
    let (lowered_in, fresh) = z3::with_z3_config(&z3::Config::new(), || {
        let second = lower(&x).unwrap();
        (
            second.context().get_z3_context().as_ptr() as usize,
            z3::Context::thread_local().get_z3_context().as_ptr() as usize,
        )
    });
    assert_ne!(fresh, original.get_z3_context().as_ptr() as usize);
    assert_eq!(lowered_in, fresh);
}
