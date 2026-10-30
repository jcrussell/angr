//! Expression-simplification tests: algebraic identities, the shift/mul →
//! Concat rewrites, and the pre-Z3 `Extract` rewrite pass (angr-p8cz).

use super::super::Node;
use super::*;

// =========================================================================
// Expression simplification tests
// =========================================================================

#[test]
fn test_add_identity() {
    let x = RustBV::symbol("x", 32);
    let zero = RustBV::zero(32);
    // x + 0 → x (should return symbolic, not expression)
    let r1 = x.add(&zero);
    assert!(r1.leaf_var().is_some());
    // 0 + x → x
    let r2 = zero.add(&x);
    assert!(r2.leaf_var().is_some());
}

#[test]
fn test_sub_identity() {
    let x = RustBV::symbol("x", 32);
    let zero = RustBV::zero(32);
    // x - 0 → x
    let r = x.sub(&zero);
    assert!(r.leaf_var().is_some());
}

#[test]
fn test_mul_identity_and_annihilator() {
    let x = RustBV::symbol("x", 32);
    let zero = RustBV::zero(32);
    let one = RustBV::concrete(1, 32);
    // x * 0 → 0
    assert_eq!(x.mul(&zero).as_u128(), Some(0));
    // 0 * x → 0
    assert_eq!(zero.mul(&x).as_u128(), Some(0));
    // x * 1 → x
    assert!(x.mul(&one).leaf_var().is_some());
    // 1 * x → x
    assert!(one.mul(&x).leaf_var().is_some());
}

#[test]
fn test_and_identity_and_annihilator() {
    let x = RustBV::symbol("x", 8);
    let zero = RustBV::zero(8);
    let ones = RustBV::ones(8);
    // x & 0 → 0
    assert_eq!(x.and(&zero).as_u128(), Some(0));
    // x & 0xFF → x
    assert!(x.and(&ones).leaf_var().is_some());
    // 0xFF & x → x
    assert!(ones.and(&x).leaf_var().is_some());
}

#[test]
fn test_or_identity_and_annihilator() {
    let x = RustBV::symbol("x", 8);
    let zero = RustBV::zero(8);
    let ones = RustBV::ones(8);
    // x | 0 → x
    assert!(x.or(&zero).leaf_var().is_some());
    // 0 | x → x
    assert!(zero.or(&x).leaf_var().is_some());
    // x | 0xFF → 0xFF
    assert_eq!(x.or(&ones).as_u128(), Some(0xFF));
}

#[test]
fn test_xor_identity() {
    let x = RustBV::symbol("x", 32);
    let zero = RustBV::zero(32);
    // x ^ 0 → x
    assert!(x.xor(&zero).leaf_var().is_some());
    // 0 ^ x → x
    assert!(zero.xor(&x).leaf_var().is_some());
}

#[test]
fn test_not_double_negation() {
    let x = RustBV::symbol("x", 32);
    // not(not(x)) → x
    let r = x.not().not();
    assert_eq!(r.leaf_var(), x.leaf_var(), "expected x back, got {r:?}");
}

#[test]
fn test_neg_double_negation() {
    let x = RustBV::symbol("x", 32);
    // neg(neg(x)) → x
    let r = x.neg().neg();
    assert_eq!(r.leaf_var(), x.leaf_var(), "expected x back, got {r:?}");
}

#[test]
fn test_reverse_double() {
    let x = RustBV::symbol("x", 32);
    // reverse(reverse(x)) → x
    let r = x.reverse().reverse();
    assert_eq!(r.leaf_var(), x.leaf_var(), "expected x back, got {r:?}");
}

#[test]
fn test_reverse_non_byte_aligned_width_is_identity_not_truncation() {
    // angr-sqfj8.92: a byte reverse is undefined over a partial byte. The old
    // code guarded the precondition with a `debug_assert!` and, in release,
    // swapped only the `w / 8` whole bytes — silently zeroing the high bits of
    // a concrete operand (12-bit 0xABC came back as 0x0BC-swapped garbage).
    // Every profile now returns the operand unchanged instead, so no bit is
    // dropped and the concrete answer matches what the Z3 lowering already
    // produces for a misaligned width.
    for (width, value) in [(12u32, 0xABCu128), (9, 0x1FF), (33, 0x1_2345_6789)] {
        let rev = RustBV::concrete(value, width).reverse();
        assert_eq!(rev.width(), width, "width {width} must be preserved");
        assert_eq!(
            rev.as_u128(),
            Some(value),
            "width {width}: misaligned reverse must keep every bit"
        );
    }

    // Symbolic operands take the same no-op path, so no `Reverse` node is
    // built for a width whose claripy export would raise.
    let sym = RustBV::symbol("misaligned", 12);
    let rev = sym.reverse();
    assert!(rev.leaf_var().is_some(), "{rev:?}");
    assert_eq!(rev.width(), 12);

    // Byte-aligned widths still reverse normally.
    let aligned = RustBV::concrete(0x1122, 16).reverse();
    assert_eq!(aligned.as_u128(), Some(0x2211));
}

#[test]
fn test_reverse_wide_concrete_declines_fold_instead_of_overshifting() {
    // angr-03vl4.60: a `Concrete` keeps only its low 128 bits, so a byte
    // reverse at width > 128 has no readable source for the bytes it must move
    // down — and the fold loop's `v >> (i * 8)` / `byte << ((n - 1 - i) * 8)`
    // shifts a u128 by up to `w - 8`, which aborts under `debug_assertions`
    // and silently wraps mod 128 in release. Same resolution as a non-trivial
    // wide rotate (see `define_rotate_pair`): decline the fold, stay symbolic.
    for width in [136u32, 192, 256] {
        let rev = RustBV::concrete(0x1122_3344, width).reverse();
        assert_eq!(rev.width(), width, "width {width} must be preserved");
        assert!(
            rev.node().is_some_and(|n| n.op == BVOp::Reverse),
            "width {width}: wide reverse must stay symbolic, got {rev:?}"
        );
    }

    // The boundary width still folds concretely.
    let at_ceiling = RustBV::concrete(0xAA, 128).reverse();
    assert_eq!(at_ceiling.as_u128(), Some(0xAA << 120));
}

#[test]
fn test_shift_by_zero() {
    let x = RustBV::symbol("x", 32);
    let zero = RustBV::zero(32);
    // x << 0 → x
    assert!(x.shl(&zero).leaf_var().is_some());
    // x >> 0 → x
    assert!(x.lshr(&zero).leaf_var().is_some());
    // x >>> 0 → x
    assert!(x.ashr(&zero).leaf_var().is_some());
}

#[test]
fn test_shift_zero_value() {
    let x = RustBV::symbol("x", 32);
    let zero = RustBV::zero(32);
    // 0 << x → 0
    assert_eq!(zero.shl(&x).as_u128(), Some(0));
    // 0 >> x → 0
    assert_eq!(zero.lshr(&x).as_u128(), Some(0));
}

#[test]
fn test_shl_concrete_amount_rewrites_to_concat() {
    let x = RustBV::symbol("x", 32);
    let four = RustBV::concrete(4, 32);
    // sym << 4 → Concat(Extract(27, 0, sym), 0^4)
    let r = x.shl(&four);
    assert_eq!(r.width(), 32);
    match r.node() {
        Some(Node {
            op: BVOp::Concat,
            operands,
        }) => {
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[1].as_u128(), Some(0)); // low bits are zero
            assert_eq!(operands[1].width(), 4);
            assert_eq!(operands[0].width(), 28); // top bits extracted from x
        }
        _ => panic!("expected Concat, got {r:?}"),
    }
    // Behavior preserved when LHS happens to be concrete (still constant-folds).
    let v = RustBV::concrete(0x1234, 32);
    assert_eq!(v.shl(&four).as_u128(), Some(0x12340));
}

#[test]
fn test_shl_concrete_amount_at_or_above_width_yields_zero() {
    let x = RustBV::symbol("x", 32);
    // sym << 32 → 0
    let r = x.shl(&RustBV::concrete(32, 32));
    assert_eq!(r.as_u128(), Some(0));
    // sym << 999 (oversized amount) → 0
    let r = x.shl(&RustBV::concrete(999, 32));
    assert_eq!(r.as_u128(), Some(0));
}

#[test]
fn test_lshr_concrete_amount_rewrites_to_concat() {
    let x = RustBV::symbol("x", 32);
    let eight = RustBV::concrete(8, 32);
    // sym >> 8 → Concat(0^8, Extract(31, 8, sym))
    let r = x.lshr(&eight);
    assert_eq!(r.width(), 32);
    match r.node() {
        Some(Node {
            op: BVOp::Concat,
            operands,
        }) => {
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[0].as_u128(), Some(0));
            assert_eq!(operands[0].width(), 8);
            assert_eq!(operands[1].width(), 24);
        }
        _ => panic!("expected Concat, got {r:?}"),
    }
    let r = x.lshr(&RustBV::concrete(32, 32));
    assert_eq!(r.as_u128(), Some(0));
}

#[test]
fn test_ashr_concrete_amount_rewrites_to_sign_extend() {
    let x = RustBV::symbol("x", 32);
    let four = RustBV::concrete(4, 32);
    // sym >>> 4 → SignExt(4, Extract(31, 4, sym))  [extends 28-bit slice by 4 bits]
    let r = x.ashr(&four);
    assert_eq!(r.width(), 32);
    match r.node() {
        Some(Node {
            op: BVOp::SignExt(4),
            operands,
        }) => {
            assert_eq!(operands[0].width(), 28);
        }
        _ => panic!("expected SignExt(4), got {r:?}"),
    }
    // Beyond width: SignExt of MSB (1-bit slice extended by 31 bits).
    let r = x.ashr(&RustBV::concrete(64, 32));
    assert_eq!(r.width(), 32);
    match r.node() {
        Some(Node {
            op: BVOp::SignExt(31),
            operands,
        }) => {
            assert_eq!(operands[0].width(), 1);
        }
        _ => panic!("expected SignExt(31), got {r:?}"),
    }
}

#[test]
fn test_mul_by_power_of_two_rewrites_to_shl_then_concat() {
    let x = RustBV::symbol("x", 32);
    let eight = RustBV::concrete(8, 32);
    // sym * 8 → sym << 3 → Concat(Extract(28, 0, sym), 0^3)
    let r = x.mul(&eight);
    assert_eq!(r.width(), 32);
    match r.node() {
        Some(Node {
            op: BVOp::Concat,
            operands,
        }) => {
            assert_eq!(operands[1].as_u128(), Some(0));
            assert_eq!(operands[1].width(), 3);
            assert_eq!(operands[0].width(), 29);
        }
        _ => panic!("expected Concat (via Shl), got {r:?}"),
    }
    // Commutative case: 8 * sym → same shape.
    let r = eight.mul(&x);
    match r.node() {
        Some(Node {
            op: BVOp::Concat,
            operands,
        }) => {
            assert_eq!(operands[1].as_u128(), Some(0));
            assert_eq!(operands[1].width(), 3);
        }
        _ => panic!("expected Concat (commutative), got {r:?}"),
    }
}

#[test]
fn test_mul_by_non_power_of_two_stays_as_mul() {
    let x = RustBV::symbol("x", 32);
    // sym * 3 → still Mul (no rewrite for non-pow2 constants)
    let r = x.mul(&RustBV::concrete(3, 32));
    assert_eq!(
        r.node().map(|n| n.op),
        Some(BVOp::Mul),
        "expected Mul, got {r:?}"
    );
}

#[test]
fn test_sign_extend_identity() {
    let x = RustBV::symbol("x", 32);
    // sign_extend to same width → x
    let r = x.sign_extend(32);
    assert!(r.leaf_var().is_some());
    assert_eq!(r.width(), 32);
}

#[test]
fn test_extract_zero_ext() {
    let x = RustBV::symbol("x", 8);
    let ext = x.zero_extend(32); // 8-bit → 32-bit
    // Extract low 8 bits → original x
    let lo = ext.extract(7, 0);
    assert_eq!(lo.leaf_var(), x.leaf_var(), "expected x back, got {lo:?}");
    assert_eq!(lo.width(), 8);
    // Extract high 8 bits → 0
    let hi = ext.extract(31, 24);
    assert_eq!(hi.as_u128(), Some(0));
}

#[test]
fn test_extract_sign_ext_low_bits() {
    let x = RustBV::symbol("x", 8);
    let ext = x.sign_extend(32);
    // Extract low 8 bits → original x
    let lo = ext.extract(7, 0);
    assert_eq!(lo.leaf_var(), x.leaf_var(), "expected x back, got {lo:?}");
    assert_eq!(lo.width(), 8);
}
