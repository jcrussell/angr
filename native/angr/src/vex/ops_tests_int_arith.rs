// angr-9hleg: integer widening-mul / divmod / sign-extend tests (mirror of ops_int_arith.rs).

use super::*;

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
