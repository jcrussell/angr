// angr-9hleg: integer widening-mul / divmod / sign-extend tests (mirror of ops/int_arith.rs).

use super::*;
use crate::vex::ir::IRType;

#[test]
fn test_mul_widening() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(0xFFFFFFFF, 32);
    let b = RustBV::concrete(0xFFFFFFFF, 32);

    let result = VEXOps::binop(IROp::MullU(IRType::I32), a, b, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128(), Some(0xFFFFFFFE00000001));
}

#[test]
fn test_divmod_u64_to_32_concrete() {
    let ctx = SymContext::new_mock();
    // 100 / 7 = 14 rem 2
    let dvd = RustBV::concrete(100, 64);
    let dvs = RustBV::concrete(7, 32);
    let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs, &ctx).unwrap();
    let v = result.as_u128().unwrap() as u64;
    assert_eq!(v & 0xFFFF_FFFF, 14, "quotient");
    assert_eq!((v >> 32) & 0xFFFF_FFFF, 2, "remainder");
}

#[test]
fn test_divmod_u64_to_32_symbolic_dividend() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::symbolic(&ctx, "dvd", 64);
    let dvs = RustBV::concrete(7, 32);
    let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.is_symbolic());
}

#[test]
fn test_divmod_s64_to_32_symbolic_divisor() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::concrete(0xFFFF_FFFF_FFFF_FF9C, 64); // -100 as i64
    let dvs = RustBV::symbolic(&ctx, "dvs", 32);
    let result = VEXOps::binop(IROp::DivModS64to32, dvd, dvs, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.is_symbolic());
}

#[test]
fn test_divmod_u128_to_64_concrete() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::concrete(1000, 128);
    let dvs = RustBV::concrete(13, 64);
    let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs, &ctx).unwrap();
    let v = result.as_u128().unwrap();
    assert_eq!(v & 0xFFFF_FFFF_FFFF_FFFF, 76, "quotient = 1000/13");
    assert_eq!((v >> 64) & 0xFFFF_FFFF_FFFF_FFFF, 12, "remainder = 1000%13");
}

#[test]
fn test_divmod_s128_to_64_int_min_by_neg_one() {
    // i128::MIN / -1 overflows; the concrete arm must wrap (not panic) to match
    // Z3 bvsdiv/bvsrem. dividend = 1<<127 (= i128::MIN as signed 128), divisor =
    // 0xFFFF..FF (= -1 as signed 64). Reference (Python engine): rax/quotient = 0,
    // rdx/remainder = 0. Regression for angr-n0xru (was SIGABRT under panic=abort).
    let ctx = SymContext::new_mock();
    let dvd = RustBV::concrete(1u128 << 127, 128);
    let dvs = RustBV::concrete(u64::MAX as u128, 64);
    let result = VEXOps::binop(IROp::DivModS128to64, dvd, dvs, &ctx).unwrap();
    let v = result.as_u128().unwrap();
    assert_eq!(
        v & 0xFFFF_FFFF_FFFF_FFFF,
        0,
        "quotient = i128::MIN / -1 wrapped"
    );
    assert_eq!((v >> 64) & 0xFFFF_FFFF_FFFF_FFFF, 0, "remainder = 0");
}

#[test]
fn test_divmod_u128_to_64_symbolic() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::symbolic(&ctx, "dvd128", 128);
    let dvs = RustBV::concrete(7, 64);
    let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert!(result.is_symbolic());
}

#[test]
fn test_sign_extend() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits

    let result = VEXOps::unop(
        IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I32,
        },
        a,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
}

// angr-g2je6: the concrete zero-divisor arm of `divmod_double_to_single` must
// produce the same packed value as the symbolic arm (Z3's total div/mod), so a
// guest DivMod's result does not depend on whether its operands arrived
// concrete. Each case builds both and compares; the symbolic side pins the
// divisor to 0 with a constraint so the symbolic path is actually taken.
fn divmod_zero_divisor_matches_symbolic(op: IROp, dividend: u128, dividend_w: u32, divisor_w: u32) {
    let ctx = SymContext::new_mock();

    let concrete = VEXOps::binop(
        op,
        RustBV::concrete(dividend, dividend_w),
        RustBV::concrete(0, divisor_w),
        &ctx,
    )
    .unwrap();
    assert_eq!(concrete.width(), dividend_w);
    let concrete_v = concrete.as_u128().expect("concrete arm must fold");

    let dvs = RustBV::symbolic(&ctx, "dvs_zero", divisor_w);
    let pinned = dvs.eq(&RustBV::concrete(0, divisor_w), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    let symbolic = VEXOps::binop(op, RustBV::concrete(dividend, dividend_w), dvs, &ctx).unwrap();
    assert!(symbolic.is_symbolic(), "symbolic arm must not fold");
    let symbolic_v = ctx.eval(&symbolic).expect("symbolic result must be sat");

    assert_eq!(
        concrete_v, symbolic_v,
        "{op:?}: concrete zero-divisor {concrete_v:#x} != symbolic {symbolic_v:#x}"
    );
}

#[test]
fn test_divmod_u64_to_32_zero_divisor_matches_symbolic() {
    divmod_zero_divisor_matches_symbolic(IROp::DivModU64to32, 100, 64, 32);
}

#[test]
fn test_divmod_s64_to_32_zero_divisor_positive_dividend() {
    divmod_zero_divisor_matches_symbolic(IROp::DivModS64to32, 100, 64, 32);
}

#[test]
fn test_divmod_s64_to_32_zero_divisor_negative_dividend() {
    // -100 as i64: bvsdiv(x, 0) = +1 here, not all-ones.
    divmod_zero_divisor_matches_symbolic(IROp::DivModS64to32, 0xFFFF_FFFF_FFFF_FF9C, 64, 32);
}

#[test]
fn test_divmod_u128_to_64_zero_divisor_matches_symbolic() {
    divmod_zero_divisor_matches_symbolic(IROp::DivModU128to64, 1000, 128, 64);
}

#[test]
fn test_divmod_s128_to_64_zero_divisor_negative_dividend() {
    divmod_zero_divisor_matches_symbolic(IROp::DivModS128to64, (-1000i128) as u128, 128, 64);
}
