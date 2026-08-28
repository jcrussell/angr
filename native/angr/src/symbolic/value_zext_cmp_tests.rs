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
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let c = RustBV::concrete(0x100, 16);
    let r = zx.eq(&c, &ctx);
    assert_eq!(r.as_u64(), Some(0));
    // Commuted form folds the same way.
    let r_rev = c.eq(&zx, &ctx);
    assert_eq!(r_rev.as_u64(), Some(0));
}

#[test]
fn test_zext_ne_high_bits_nonzero_is_true() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let c = RustBV::concrete(0x100, 16);
    let r = zx.ne(&c, &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_eq_high_bits_zero_collapses() {
    // ZeroExt(8, x:8) == 0x42 — high byte zero → narrows to Eq(x, 0x42),
    // which is still symbolic but should be a width-8 Expression not the
    // width-16 form, evidenced by the operand width.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.eq(&RustBV::concrete(0x42, 16), &ctx);
    // Result is a width-1 Eq expression over width-8 operands.
    match &r {
        RustBV::Expression {
            op,
            operands,
            width,
            ..
        } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(*width, 1);
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[0].width(), 8);
            assert_eq!(operands[1].width(), 8);
            assert_eq!(operands[1].as_u64(), Some(0x42));
        }
        _ => panic!("expected narrowed Eq expression, got {r:?}"),
    }
}

#[test]
fn test_zext_eq_collapse_solver_consistency() {
    // After narrowing, asserting Eq(zext(x), 0x42) must still pin x to 0x42.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let eq = zx.eq(&RustBV::concrete(0x42, 16), &ctx);
    ctx.add_constraint(eq.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&x), Some(0x42));
    // Also verify the wider zext expression evaluates to the constant.
    assert_eq!(ctx.eval(&zx), Some(0x42));
}

#[test]
fn test_zext_ult_high_bits_nonzero_is_true() {
    // ZeroExt(8, x:8) < 0x200 — all zext values < 256, so always < 0x200.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ult(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ult_const_zero_is_false() {
    // ZeroExt(8, x:8) < 0 — never true.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ult(&RustBV::concrete(0, 16), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_ult_swapped_high_bits_nonzero_is_false() {
    // 0x200 < ZeroExt(8, x:8) — never true (RHS < 256).
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = RustBV::concrete(0x200, 16).ult(&zx, &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_ule_high_bits_nonzero_is_true() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ule(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ule_swapped_const_zero_is_true() {
    // 0 <= ZeroExt(x) — always true.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = RustBV::concrete(0, 16).ule(&zx, &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ugt_high_bits_nonzero_is_false() {
    // ZeroExt(8, x:8) > 0x200 — never true (LHS < 256 <= 0x200).
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ugt(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_uge_high_bits_nonzero_is_false() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.uge(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_cmp_no_fold_when_both_symbolic() {
    // Cmp(ZeroExt(x), ZeroExt(y)) — no const side, no fold.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let y = RustBV::symbolic(&ctx, "y", 8);
    let zx = x.zero_extend(16, &ctx);
    let zy = y.zero_extend(16, &ctx);
    let r = zx.eq(&zy, &ctx);
    // Should be a regular Eq expression at width 16 — no fold.
    match &r {
        RustBV::Expression { op, operands, .. } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(operands[0].width(), 16);
            assert_eq!(operands[1].width(), 16);
        }
        _ => panic!("expected Eq expression, got {r:?}"),
    }
}

#[test]
fn test_zext_cmp_no_fold_for_signed_ops() {
    // SignExt(k, x) is not handled — the comparison should pass through.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let sx = x.sign_extend(16, &ctx); // SignExt, not ZeroExt
    let r = sx.eq(&RustBV::concrete(0x100, 16), &ctx);
    // Should be Eq expression (no fold).
    match &r {
        RustBV::Expression { op, operands, .. } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(operands[0].width(), 16);
        }
        _ => panic!("expected Eq expression, got {r:?}"),
    }
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
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx16 = x.zero_extend(16, &ctx);
    let zx32 = zx16.zero_extend(32, &ctx);
    let r = zx32.eq(&RustBV::concrete(0xFF, 32), &ctx);
    match &r {
        RustBV::Expression {
            op,
            operands,
            width,
            ..
        } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(*width, 1);
            // Should have narrowed to width-8 operands.
            assert_eq!(operands[0].width(), 8);
            assert_eq!(operands[1].as_u64(), Some(0xFF));
        }
        _ => panic!("expected narrowed Eq, got {r:?}"),
    }
}

/// Regression guard for the angr-behq finding (2026-05-21):
/// Z3's AST hash-cons already de-dupes structurally-equal RustBV trees
/// at to_z3_ast time. Two structurally-equal Expression trees produce
/// the SAME Z3_ast pointer — so RustBV-level construction hash-cons
/// would NOT reduce Z3 AST node count for the to_z3_ast() output.
///
/// If this test ever fails, the assumption that drove closing
/// angr-behq is broken and that bead should be re-opened.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn z3_already_dedupes_structurally_equal_rustbv_trees() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic_with_id(2001, "x_behq", 32);
    let y = RustBV::symbolic_with_id(2002, "y_behq", 32);

    // Two structurally-identical add(x, 5) trees built independently.
    let p1 = x.clone().add_into(RustBV::concrete(5, 32), &ctx);
    let p2 = x.add_into(RustBV::concrete(5, 32), &ctx);
    let p1_ptr = p1.to_z3_ast().get_z3_ast().as_ptr();
    let p2_ptr = p2.to_z3_ast().get_z3_ast().as_ptr();
    assert_eq!(p1_ptr, p2_ptr, "Z3 should canonicalize add(x,5)");

    // Two structurally-identical mul(add(x,5), y) trees, depth 2.
    let q1 = p1.mul(&y, &ctx);
    let q2 = p2.mul(&y, &ctx);
    let q1_ptr = q1.to_z3_ast().get_z3_ast().as_ptr();
    let q2_ptr = q2.to_z3_ast().get_z3_ast().as_ptr();
    assert_eq!(q1_ptr, q2_ptr, "Z3 should canonicalize mul(add(x,5),y)");
}

/// Regression guard for angr-kkpr (2026-05-21):
/// RustBV constructors for commutative ops (add/mul/and/or/xor/eq/ne)
/// canonicalize operand order via `canonical_sort_key`. After this fix,
/// `add(x, y)` and `add(y, x)` produce the SAME RustBV (and therefore
/// the same Z3 AST). Z3 itself does NOT normalize commutative arg order
/// at mk_bv* time — preprocessing tactics do, but only on assertions —
/// so the canonicalization happens on the Rust side at construction.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn commutative_ops_canonicalize_operand_order() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic_with_id(3001, "x_kkpr", 32);
    let y = RustBV::symbolic_with_id(3002, "y_kkpr", 32);
    let c = RustBV::concrete(7, 32);

    // add(x, y) and add(y, x) → same RustBV → same Z3 AST.
    let r1 = x.clone().add_into(y.clone(), &ctx);
    let r2 = y.clone().add_into(x.clone(), &ctx);
    assert_eq!(
        r1.to_z3_ast().get_z3_ast().as_ptr(),
        r2.to_z3_ast().get_z3_ast().as_ptr(),
        "add(x,y) and add(y,x) should canonicalize to the same Z3 AST"
    );

    // add(x, c) and add(c, x) → same RustBV → same Z3 AST.
    // (concrete should sort to the right, so both become add(x, c).)
    let r3 = x.clone().add_into(c.clone(), &ctx);
    let r4 = c.add_into(x.clone(), &ctx);
    assert_eq!(
        r3.to_z3_ast().get_z3_ast().as_ptr(),
        r4.to_z3_ast().get_z3_ast().as_ptr(),
        "add(x,c) and add(c,x) should canonicalize to the same Z3 AST"
    );
    // Verify concrete is in operand[1] after canonicalization.
    match &r3 {
        RustBV::Expression { operands, .. } => {
            assert!(
                matches!(operands[1], RustBV::Concrete { .. }),
                "concrete should sort to the right of symbolic"
            );
        }
        _ => panic!("expected Expression for add(x, c)"),
    }

    // Same property for mul / and / or / xor / eq / ne.
    type BinOp = fn(RustBV, RustBV, &SymContext) -> RustBV;
    let cases: &[(&str, BinOp)] = &[
        ("mul", |a, b, c| a.mul_into(b, c)),
        ("and", |a, b, c| a.and_into(b, c)),
        ("or", |a, b, c| a.or_into(b, c)),
        ("xor", |a, b, c| a.xor_into(b, c)),
        ("eq", |a, b, c| a.eq_into(b, c)),
        ("ne", |a, b, c| a.ne_into(b, c)),
    ];
    for (name, build) in cases {
        let lhs = build(x.clone(), y.clone(), &ctx);
        let rhs = build(y.clone(), x.clone(), &ctx);
        assert_eq!(
            lhs.to_z3_ast().get_z3_ast().as_ptr(),
            rhs.to_z3_ast().get_z3_ast().as_ptr(),
            "{name}(x,y) and {name}(y,x) should canonicalize to the same Z3 AST"
        );
    }
}
