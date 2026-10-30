//! `Cmp(ZeroExt(k, x), BVV)` trivial-constraint fast-path tests (angr-g7nq),
//! plus the two operand-canonicalization / Z3-hash-cons regression guards that
//! back the same fold's design assumptions.

use super::*;

// =========================================================================
// angr-g7nq: Cmp(ZeroExt(k, x), BVV) trivial-constraint fast path
// =========================================================================

#[test]
fn test_zext_eq_high_bits_nonzero_is_false() {
    // ZeroExt(8, x:8) == 0x100 — high byte nonzero → folds to concrete 0.
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let c = SymBV::concrete(0x100, 16);
    let r = zx.eq(&c);
    assert_eq!(r.as_u64(), Some(0));
    // Commuted form folds the same way.
    let r_rev = c.eq(&zx);
    assert_eq!(r_rev.as_u64(), Some(0));
}

#[test]
fn test_zext_ne_high_bits_nonzero_is_true() {
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let c = SymBV::concrete(0x100, 16);
    let r = zx.ne(&c);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_eq_high_bits_zero_collapses() {
    // ZeroExt(8, x:8) == 0x42 — high byte zero → narrows to Eq(x, 0x42),
    // which is still symbolic but should be a width-8 Expression not the
    // width-16 form, evidenced by the operand width.
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = zx.eq(&SymBV::concrete(0x42, 16));
    // Result is a width-1 Eq expression over width-8 operands.
    let node = r
        .node()
        .unwrap_or_else(|| panic!("expected narrowed Eq expression, got {r:?}"));
    let (op, operands, width) = (&node.op, &node.operands, &r.width());
    assert_eq!(*op, BVOp::Eq);
    assert_eq!(*width, 1);
    assert_eq!(operands.len(), 2);
    assert_eq!(operands[0].width(), 8);
    assert_eq!(operands[1].width(), 8);
    assert_eq!(operands[1].as_u64(), Some(0x42));
}

#[test]
fn test_zext_ult_high_bits_nonzero_is_true() {
    // ZeroExt(8, x:8) < 0x200 — all zext values < 256, so always < 0x200.
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = zx.ult(&SymBV::concrete(0x200, 16));
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ult_const_zero_is_false() {
    // ZeroExt(8, x:8) < 0 — never true.
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = zx.ult(&SymBV::concrete(0, 16));
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_ult_swapped_high_bits_nonzero_is_false() {
    // 0x200 < ZeroExt(8, x:8) — never true (RHS < 256).
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = SymBV::concrete(0x200, 16).ult(&zx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_ule_high_bits_nonzero_is_true() {
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = zx.ule(&SymBV::concrete(0x200, 16));
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ule_swapped_const_zero_is_true() {
    // 0 <= ZeroExt(x) — always true.
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = SymBV::concrete(0, 16).ule(&zx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ugt_high_bits_nonzero_is_false() {
    // ZeroExt(8, x:8) > 0x200 — never true (LHS < 256 <= 0x200).
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = zx.ugt(&SymBV::concrete(0x200, 16));
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_uge_high_bits_nonzero_is_false() {
    let x = SymBV::symbol("x", 8);
    let zx = x.zero_extend(16);
    let r = zx.uge(&SymBV::concrete(0x200, 16));
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_cmp_no_fold_when_both_symbolic() {
    // Cmp(ZeroExt(x), ZeroExt(y)) — no const side, no fold.
    let x = SymBV::symbol("x", 8);
    let y = SymBV::symbol("y", 8);
    let zx = x.zero_extend(16);
    let zy = y.zero_extend(16);
    let r = zx.eq(&zy);
    // Should be a regular Eq expression at width 16 — no fold.
    let node = r
        .node()
        .unwrap_or_else(|| panic!("expected Eq expression, got {r:?}"));
    let (op, operands) = (&node.op, &node.operands);
    assert_eq!(*op, BVOp::Eq);
    assert_eq!(operands[0].width(), 16);
    assert_eq!(operands[1].width(), 16);
}

#[test]
fn test_zext_cmp_no_fold_for_signed_ops() {
    // SignExt(k, x) is not handled — the comparison should pass through.
    let x = SymBV::symbol("x", 8);
    let sx = x.sign_extend(16); // SignExt, not ZeroExt
    let r = sx.eq(&SymBV::concrete(0x100, 16));
    // Should be Eq expression (no fold).
    let node = r
        .node()
        .unwrap_or_else(|| panic!("expected Eq expression, got {r:?}"));
    let (op, operands) = (&node.op, &node.operands);
    assert_eq!(*op, BVOp::Eq);
    assert_eq!(operands[0].width(), 16);
}

#[test]
fn test_zext_cmp_chain_collapse_to_narrowest() {
    // ZeroExt(16, ZeroExt(8, x:8)) == 0xFF — both extends collapse cleanly.
    // The inner zero_extend collapses via constant fold; the outer is what
    // we're testing. After the first call, the inner zx is a ZeroExt(8, x);
    // wrapping in zero_extend(32, ...) produces ZeroExt(16, ZeroExt(8, x))
    // which by the operand structure is treated as ZeroExt(16, <inner>),
    // where <inner> has width 16. The const 0xFF has zero high 16 bits, so
    // we narrow to Eq(<inner-as-zext>, 0xFF:16); that then narrows again
    // recursively. Final: Eq(x, 0xFF:8).
    let x = SymBV::symbol("x", 8);
    let zx16 = x.zero_extend(16);
    let zx32 = zx16.zero_extend(32);
    let r = zx32.eq(&SymBV::concrete(0xFF, 32));
    let node = r
        .node()
        .unwrap_or_else(|| panic!("expected narrowed Eq, got {r:?}"));
    let (op, operands, width) = (&node.op, &node.operands, &r.width());
    assert_eq!(*op, BVOp::Eq);
    assert_eq!(*width, 1);
    // Should have narrowed to width-8 operands.
    assert_eq!(operands[0].width(), 8);
    assert_eq!(operands[1].as_u64(), Some(0xFF));
}
