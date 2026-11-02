// angr-9hleg: integer widening-mul / divmod / sign-extend tests (mirror of ops/int_arith.rs).

use super::test_helpers::{pinned_symbol, z3_eval};
use super::*;
use crate::vex::ir::IRType;

#[test]
fn test_mul_widening() {

    let a = RustBV::concrete(0xFFFFFFFF, 32);
    let b = RustBV::concrete(0xFFFFFFFF, 32);

    let result = VEXOps::binop(IROp::MullU(IRType::I32), a, b).unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128(), Some(0xFFFFFFFE00000001));
}

#[test]
fn test_divmod_u64_to_32_concrete() {
    // 100 / 7 = 14 rem 2
    let dvd = RustBV::concrete(100, 64);
    let dvs = RustBV::concrete(7, 32);
    let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs).unwrap();
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
    let concrete = VEXOps::binop(
        op,
        RustBV::concrete(dvd, dividend_w),
        RustBV::concrete(dvs, divisor_w),
    )
    .unwrap()
    .as_u128()
    .expect("concrete arm must fold");

    let (left, right, pin) = if symbolic_dividend {
        let (sym, pin) = pinned_symbol("dvd", dvd, dividend_w);
        (sym, RustBV::concrete(dvs, divisor_w), pin)
    } else {
        let (sym, pin) = pinned_symbol("dvs", dvs, divisor_w);
        (RustBV::concrete(dvd, dividend_w), sym, pin)
    };
    let symbolic = VEXOps::binop(op, left, right).unwrap();
    assert_eq!(symbolic.width(), dividend_w);
    assert!(symbolic.is_symbolic(), "{op:?}: symbolic arm must not fold");
    let symbolic = z3_eval(&symbolic, &[pin]).expect("pinned operands must be sat");
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
    let dvd = RustBV::concrete(1000, 128);
    let dvs = RustBV::concrete(13, 64);
    let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs).unwrap();
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
    let dvd = RustBV::concrete(1u128 << 127, 128);
    let dvs = RustBV::concrete(u64::MAX as u128, 64);
    let result = VEXOps::binop(IROp::DivModS128to64, dvd, dvs).unwrap();
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

    let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits

    let result = VEXOps::unop(
        IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I32,
        },
        a,
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
    let concrete = VEXOps::binop(
        op,
        RustBV::concrete(dividend, dividend_w),
        RustBV::concrete(0, divisor_w),
    )
    .unwrap();
    assert_eq!(concrete.width(), dividend_w);
    let concrete_v = concrete.as_u128().expect("concrete arm must fold");

    let (dvs, pinned) = pinned_symbol("dvs_zero", 0, divisor_w);
    let symbolic = VEXOps::binop(op, RustBV::concrete(dividend, dividend_w), dvs).unwrap();
    assert!(symbolic.is_symbolic(), "symbolic arm must not fold");
    let symbolic_v = z3_eval(&symbolic, &[pinned]).expect("symbolic result must be sat");

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
    let same = |a: u128, b: u128, ty: IRType, signed: bool| {
        let bits = ty.bits();
        VEXOps::binop(
            IROp::DivModSameWidth { ty, signed },
            RustBV::concrete(a, bits),
            RustBV::concrete(b, bits),
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
    let op = IROp::DivModSameWidth {
        ty: IRType::I64,
        signed: true,
    };
    let r = VEXOps::binop(
        op,
        RustBV::symbol("a", 64),
        RustBV::concrete(3, 64),
    )
    .unwrap();
    assert_eq!(r.width(), 128);
    assert!(r.is_symbolic());
    // A 32-bit divisor is an operand-width error, not a silent extension.
    assert!(
        VEXOps::binop(
            op,
            RustBV::concrete(1, 64),
            RustBV::concrete(1, 32)
        )
        .is_err()
    );
}

/// A symbolic same-width DivMod divides at the operand width, the term
/// irop.py builds, not a double-width division Z3 proves far slower
/// (angr-mx00p.100).
#[test]
fn test_divmod_same_width_symbolic_divides_at_operand_width() {
    for signed in [false, true] {
        let (a, b) = (RustBV::symbol("a", 32), RustBV::symbol("b", 32));
        let r = VEXOps::binop(
            IROp::DivModSameWidth {
                ty: IRType::I32,
                signed,
            },
            a.clone(),
            b.clone(),
        )
        .unwrap();
        let expected = if signed {
            a.srem(&b).concat_into(a.sdiv(&b))
        } else {
            a.urem(&b).concat_into(a.udiv(&b))
        };
        assert_eq!(r, expected, "signed={signed}");
    }
}

/// pyvex 10's `Iop_{DivU,DivS,ModU,ModS}128` reach the width-generic arms
/// at 128 bits (angr-mx00p.6).
#[test]
fn test_div_mod_128_concrete() {
    let big = (1u128 << 100) + 5;
    let op = |op: IROp, a: u128, b: u128| {
        VEXOps::binop(op, RustBV::concrete(a, 128), RustBV::concrete(b, 128))
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

/// angr-mx00p.100.13: `CmpORD`/`CmpORDU` return libVEX's same-width
/// 0x8 (less) / 0x4 (greater) / 0x2 (equal) code; the signed and unsigned
/// forms split on a set top bit. A pinned symbol must agree with the literal.
#[test]
fn test_cmp_ord() {
    for (ty, w) in [(IRType::I32, 32u32), (IRType::I64, 64)] {
        let top = 1u128 << (w - 1);
        for (a, b, signed, unsigned) in [
            (3, 7, 0x8, 0x8),
            (7, 3, 0x4, 0x4),
            (5, 5, 0x2, 0x2),
            (top, 1, 0x8, 0x4),
            (1, top, 0x4, 0x8),
        ] {
            for (op, want) in [(IROp::CmpORD(ty), signed), (IROp::CmpORDU(ty), unsigned)] {
                let got = VEXOps::binop(op, RustBV::concrete(a, w), RustBV::concrete(b, w)).unwrap();
                assert_eq!(got.width(), w, "{op:?}");
                assert_eq!(got.as_u128(), Some(want), "{op:?}({a:#x}, {b:#x})");
                let (sym, pin) = pinned_symbol("cmp_ord_a", a, w);
                let sym_got = VEXOps::binop(op, sym, RustBV::concrete(b, w)).unwrap();
                assert!(sym_got.is_symbolic(), "{op:?}");
                assert_eq!(z3_eval(&sym_got, &[pin]), Some(want), "{op:?}({a:#x}, {b:#x}) symbolic");
            }
        }
    }
}
