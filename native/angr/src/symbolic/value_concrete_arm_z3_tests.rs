//! The solver-facing half of `bv/value_concrete_arm_tests.rs`: these tests lower or solve, so
//! they stay in `symbolic` (angr-mx00p.116.7).

use crate::bv::*;
// Tests pinning the *concrete* arms of shift/rotate/divide against their Z3
// (symbolic) counterparts — the arms most prone to silently disagreeing on
// amounts >= 2^32, full-width operands, and division by zero.

use crate::symbolic::ToZ3;
use crate::symbolic::SymContext;
// angr-g2je6: SMT-LIB bvsdiv is total but asymmetric — x / 0 is -1 for x >= 0
// and +1 for x < 0. The concrete fold in `sdiv_into` used to return all-ones
// unconditionally, so sdiv(-5, 0) differed depending on whether the operands
// arrived concrete or symbolic-then-pinned.
#[test]
fn test_sdiv_by_zero_concrete_matches_z3() {
    for dividend in [5u128, 0, (-5i64) as u64 as u128, 1u128 << 63] {
        let ctx = SymContext::new_mock();
        let concrete = RustBV::concrete(dividend, 64).sdiv(&RustBV::concrete(0, 64));
        let folded = concrete.as_u128().expect("concrete sdiv must fold");

        let d = RustBV::symbol("sdiv_dvd", 64);
        let pinned = d.eq(&RustBV::concrete(dividend, 64));
        ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
        let symbolic = d.sdiv(&RustBV::concrete(0, 64));
        assert_eq!(
            Some(folded),
            ctx.eval(&symbolic),
            "sdiv({dividend:#x}, 0): concrete fold disagrees with Z3 bvsdiv"
        );
    }
}
// Sibling coverage for the ops that were already Z3-correct, so a future
// "simplification" of the zero-divisor arms cannot silently break them.
#[test]
fn test_udiv_srem_urem_by_zero_concrete_matches_z3() {
    for dividend in [7u128, 0, (-7i64) as u64 as u128] {
        let ctx = SymContext::new_mock();
        let d = RustBV::symbol("dvd0", 64);
        let pinned = d.eq(&RustBV::concrete(dividend, 64));
        ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
        let zero = RustBV::concrete(0, 64);
        let c = RustBV::concrete(dividend, 64);

        assert_eq!(
            c.udiv(&zero).as_u128(),
            ctx.eval(&d.udiv(&zero))
        );
        assert_eq!(
            c.urem(&zero).as_u128(),
            ctx.eval(&d.urem(&zero))
        );
        assert_eq!(
            c.srem(&zero).as_u128(),
            ctx.eval(&d.srem(&zero))
        );
    }
}
// angr-n0irt.15: the wrapping_div/wrapping_rem in sdiv_into/srem_into exist
// solely to survive the i128::MIN / -1 overflow, which plain `/`,`%` would
// panic on even in release (panic=abort -> SIGABRT). That path is only
// reachable at width 128: sign_extend's i128 result can equal i128::MIN only
// when the operand is 1u128<<127. The by-zero tests above are width-64 and
// never a non-zero divisor at width 128, so a revert to plain `/`,`%` during a
// future dedup pass would go uncaught. This pins the exact overflow case.
#[test]
fn test_sdiv_srem_min_by_neg_one_width_128() {
    let ctx = SymContext::new_mock();
    let min = 1u128 << 127; // i128::MIN reinterpreted as u128
    let neg_one = u128::MAX; // -1 at width 128
    let a = RustBV::concrete(min, 128);
    let b = RustBV::concrete(neg_one, 128);

    // Concrete folds must not panic and must match Z3 bvsdiv/bvsrem, which
    // wrap: MIN / -1 == MIN, MIN % -1 == 0.
    let sdiv = a.sdiv(&b);
    let srem = a.srem(&b);
    assert_eq!(
        sdiv.as_u128(),
        Some(min),
        "sdiv(i128::MIN, -1) must wrap to i128::MIN, not panic"
    );
    assert_eq!(
        srem.as_u128(),
        Some(0),
        "srem(i128::MIN, -1) must wrap to 0, not panic"
    );

    // Symbolic-pinned parity: same expression with a symbolic dividend pinned
    // to MIN, evaluated through Z3, must agree with the concrete fold.
    let d = RustBV::symbol("sdiv128_min", 128);
    let pinned = d.eq(&RustBV::concrete(min, 128));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(
        sdiv.as_u128(),
        ctx.eval(&d.sdiv(&b)),
        "concrete sdiv fold disagrees with Z3 bvsdiv at width 128"
    );
    assert_eq!(
        srem.as_u128(),
        ctx.eval(&d.srem(&b)),
        "concrete srem fold disagrees with Z3 bvsrem at width 128"
    );
}
