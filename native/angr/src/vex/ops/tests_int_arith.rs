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

/// Evaluate `op` with one operand symbolic but pinned by a constraint to the
/// given value, and with both concrete; the two must agree (angr-vmi2l.58:
/// the concrete arm once leaked an overflowing quotient into the remainder).
/// Returns the agreed packed `remainder:quotient`.
fn divmod_concrete_matches_pinned_symbolic(
    op: IROp,
    (dvd, dividend_w): (u128, u32),
    (dvs, divisor_w): (u128, u32),
    symbolic_dividend: bool,
) -> u128 {
    let ctx = SymContext::new_mock();
    let concrete = VEXOps::binop(
        op,
        RustBV::concrete(dvd, dividend_w),
        RustBV::concrete(dvs, divisor_w),
        &ctx,
    )
    .unwrap()
    .as_u128()
    .expect("concrete arm must fold");

    let pin = |name: &str, v: u128, w: u32| {
        let sym = RustBV::symbolic(&ctx, name, w);
        ctx.add_constraint(sym.to_z3_ast().unwrap().eq(RustBV::concrete(v, w).to_z3_ast().unwrap()));
        sym
    };
    let (left, right) = if symbolic_dividend {
        (pin("dvd", dvd, dividend_w), RustBV::concrete(dvs, divisor_w))
    } else {
        (RustBV::concrete(dvd, dividend_w), pin("dvs", dvs, divisor_w))
    };
    let symbolic = VEXOps::binop(op, left, right, &ctx).unwrap();
    assert_eq!(symbolic.width(), dividend_w);
    assert!(symbolic.is_symbolic(), "{op:?}: symbolic arm must not fold");
    let symbolic = ctx.eval(&symbolic).expect("pinned operands must be sat");
    assert_eq!(
        concrete, symbolic,
        "{op:?}: concrete {concrete:#x} != pinned symbolic {symbolic:#x}"
    );
    concrete
}

#[test]
fn test_divmod_u64_to_32_symbolic_dividend() {
    // 100 / 7 = 14 rem 2.
    let v = divmod_concrete_matches_pinned_symbolic(
        IROp::DivModU64to32,
        (100, 64),
        (7, 32),
        true,
    );
    assert_eq!(v, 14 | (2 << 32));
}

#[test]
fn test_divmod_s64_to_32_symbolic_divisor() {
    // -100 / 7 = -14 rem -2 (truncating).
    let v = divmod_concrete_matches_pinned_symbolic(
        IROp::DivModS64to32,
        (0xFFFF_FFFF_FFFF_FF9C, 64),
        (7, 32),
        false,
    );
    assert_eq!(v, ((-14i32) as u32 as u128) | (((-2i32) as u32 as u128) << 32));
}

/// An overflowing unsigned quotient: libVEX guards only a zero divisor, so
/// amd64 `div ecx` with rdx=1, rax=0, rcx=1 reaches the op with quotient
/// 2^32. Both arms truncate it to 0 and the remainder half stays 0, as in the
/// Python engine (angr-vmi2l.58).
#[test]
fn test_divmod_overflowing_quotient_matches_pinned_symbolic() {
    for symbolic_dividend in [true, false] {
        let v = divmod_concrete_matches_pinned_symbolic(
            IROp::DivModU64to32,
            (1 << 32, 64),
            (1, 32),
            symbolic_dividend,
        );
        assert_eq!(v, 0, "DivModU64to32(2^32, 1)");
        let v = divmod_concrete_matches_pinned_symbolic(
            IROp::DivModU128to64,
            ((0x77 << 64) | 0x1234, 128),
            (0x10, 64),
            symbolic_dividend,
        );
        let q = (((0x77u128 << 64) | 0x1234) / 0x10) & u128::from(u64::MAX);
        assert_eq!(v, q | (4 << 64), "DivModU128to64 overflow");
        let v = divmod_concrete_matches_pinned_symbolic(
            IROp::DivModS64to32,
            (5 << 32, 64),
            (2, 32),
            symbolic_dividend,
        );
        // (5 << 32) / 2 = 0x2_8000_0000: the low half 0x8000_0000, rem 0.
        assert_eq!(v, 0x8000_0000, "DivModS64to32((5 << 32) / 2)");
    }
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
    // rdx/remainder = 0 (regression, angr-n0xru).
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
    // 1000 / 7 = 142 rem 6.
    let v = divmod_concrete_matches_pinned_symbolic(
        IROp::DivModU128to64,
        (1000, 128),
        (7, 64),
        true,
    );
    assert_eq!(v, 142 | (6 << 64));
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
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
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

/// `Iop_DivMod{U,S}{32to32,64to64}` (angr-mx00p.6): same-width operands,
/// `Concat(rem, quot)` at double width, as angr's Python `_op_divmod`.
#[test]
fn test_divmod_same_width_concrete() {
    let ctx = SymContext::new_mock();
    let same = |a: u128, b: u128, ty: IRType, signed: bool| {
        let bits = ty.bits();
        VEXOps::binop(
            IROp::DivModSameWidth { ty, signed },
            RustBV::concrete(a, bits),
            RustBV::concrete(b, bits),
            &ctx,
        )
        .unwrap()
    };
    let r = same(100, 7, IRType::I64, false);
    assert_eq!(r.width(), 128);
    assert_eq!(r.as_u128(), Some((2u128 << 64) | 14), "rem:quot");
    // -100 / 7 = -14 rem -2 (signed remainder takes the dividend's sign).
    let r = same(0xFFFF_FF9C, 7, IRType::I32, true);
    assert_eq!(r.width(), 64);
    assert_eq!(r.as_u128(), Some((0xFFFF_FFFEu128 << 32) | 0xFFFF_FFF2));
    // Division by zero follows Z3: unsigned quotient all-ones, remainder the
    // dividend; signed quotient +1 for a negative dividend.
    let r = same(9, 0, IRType::I32, false);
    assert_eq!(r.as_u128(), Some((9u128 << 32) | 0xFFFF_FFFF));
    let r = same(0xFFFF_FFF7, 0, IRType::I32, true);
    assert_eq!(r.as_u128(), Some((0xFFFF_FFF7u128 << 32) | 1));
    // MIN / -1 wraps to MIN with remainder 0.
    let r = same(0x8000_0000, 0xFFFF_FFFF, IRType::I32, true);
    assert_eq!(r.as_u128(), Some(0x8000_0000));
}

#[test]
fn test_divmod_same_width_symbolic_and_width_checked() {
    let ctx = SymContext::new_mock();
    let op = IROp::DivModSameWidth {
        ty: IRType::I64,
        signed: true,
    };
    let r = VEXOps::binop(
        op,
        RustBV::symbolic(&ctx, "a", 64),
        RustBV::concrete(3, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(r.width(), 128);
    assert!(r.is_symbolic());
    // A 32-bit divisor is an operand-width error, not a silent extension.
    assert!(
        VEXOps::binop(
            op,
            RustBV::concrete(1, 64),
            RustBV::concrete(1, 32),
            &ctx
        )
        .is_err()
    );
}

/// pyvex 10's `Iop_{DivU,DivS,ModU,ModS}128` reach the width-generic arms
/// at 128 bits (angr-mx00p.6).
#[test]
fn test_div_mod_128_concrete() {
    let ctx = SymContext::new_mock();
    let big = (1u128 << 100) + 5;
    let op = |op: IROp, a: u128, b: u128| {
        VEXOps::binop(op, RustBV::concrete(a, 128), RustBV::concrete(b, 128), &ctx)
            .unwrap()
            .as_u128()
    };
    assert_eq!(op(IROp::DivU(IRType::I128), big, 1 << 50), Some(1 << 50));
    assert_eq!(op(IROp::ModU(IRType::I128), big, 1 << 50), Some(5));
    // -7 / 2 = -3 rem -1.
    let neg7 = 7u128.wrapping_neg();
    assert_eq!(op(IROp::DivS(IRType::I128), neg7, 2), Some(3u128.wrapping_neg()));
    assert_eq!(op(IROp::ModS(IRType::I128), neg7, 2), Some(1u128.wrapping_neg()));
}
