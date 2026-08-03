// angr-9hleg: core VEXOps dispatch / general binop tests (mirror of `ops/mod.rs`).

use super::*;

#[test]
fn test_add_op() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(5, 32);
    let b = RustBV::concrete(3, 32);

    let result = VEXOps::binop(IROp::Add(IRType::I32), a, b, &ctx).unwrap();
    assert_eq!(result.as_u64(), Some(8));
}

#[test]
fn test_comparison_ops() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(5, 32);
    let b = RustBV::concrete(10, 32);

    let lt = VEXOps::binop(IROp::CmpLTU(IRType::I32), a.clone(), b.clone(), &ctx).unwrap();
    assert_eq!(lt.as_u64(), Some(1));

    let eq = VEXOps::binop(IROp::CmpEQ(IRType::I32), a, b, &ctx).unwrap();
    assert_eq!(eq.as_u64(), Some(0));
}

#[test]
fn test_vector_add() {
    let ctx = SymContext::new_mock();

    // Two vectors of 4 x i32
    // [1, 2, 3, 4] + [10, 20, 30, 40] = [11, 22, 33, 44]
    let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
    let b: u128 = 10 | (20 << 32) | (30 << 64) | (40 << 96);

    let av = RustBV::concrete(a, 128);
    let bv = RustBV::concrete(b, 128);

    let result = VEXOps::binop(
        IROp::VAdd {
            elem: IRType::I32,
            count: 4,
        },
        av,
        bv,
        &ctx,
    )
    .unwrap();
    let rv = result.as_u128().unwrap();

    assert_eq!(rv & 0xFFFFFFFF, 11);
    assert_eq!((rv >> 32) & 0xFFFFFFFF, 22);
    assert_eq!((rv >> 64) & 0xFFFFFFFF, 33);
    assert_eq!((rv >> 96) & 0xFFFFFFFF, 44);
}

#[test]
#[allow(clippy::identity_op)] // explicit 4-lane layout reads better than the minimized form
fn test_vector_cmp_eq() {
    let ctx = SymContext::new_mock();

    // Two vectors of 4 x i32
    // [1, 2, 3, 4] == [1, 0, 3, 0] -> [0xFFFFFFFF, 0, 0xFFFFFFFF, 0]
    let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
    let b: u128 = 1 | (0 << 32) | (3 << 64) | (0 << 96);

    let av = RustBV::concrete(a, 128);
    let bv = RustBV::concrete(b, 128);

    let result = VEXOps::binop(
        IROp::VCmpEQ {
            elem: IRType::I32,
            count: 4,
        },
        av,
        bv,
        &ctx,
    )
    .unwrap();
    let rv = result.as_u128().unwrap();

    assert_eq!(rv & 0xFFFFFFFF, 0xFFFFFFFF); // 1 == 1
    assert_eq!((rv >> 32) & 0xFFFFFFFF, 0); // 2 != 0
    assert_eq!((rv >> 64) & 0xFFFFFFFF, 0xFFFFFFFF); // 3 == 3
    assert_eq!((rv >> 96) & 0xFFFFFFFF, 0); // 4 != 0
}

#[test]
fn test_concat() {
    let ctx = SymContext::new_mock();

    let hi = RustBV::concrete(0xDEAD, 16);
    let lo = RustBV::concrete(0xBEEF, 16);

    let result = VEXOps::binop(IROp::Concat { ty: IRType::I32 }, hi, lo, &ctx).unwrap();
    assert_eq!(result.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_bit_manipulation() {
    let ctx = SymContext::new_mock();

    // CLZ (count leading zeros)
    let a = RustBV::concrete(0x0F00, 16); // binary: 0000111100000000
    let clz = VEXOps::unop(IROp::Clz(IRType::I16), a.clone(), &ctx).unwrap();
    assert_eq!(clz.as_u64(), Some(4)); // 4 leading zeros

    // CTZ (count trailing zeros)
    let ctz = VEXOps::unop(IROp::Ctz(IRType::I16), a.clone(), &ctx).unwrap();
    assert_eq!(ctz.as_u64(), Some(8)); // 8 trailing zeros

    // PopCount
    let pop = VEXOps::unop(IROp::PopCount(IRType::I16), a, &ctx).unwrap();
    assert_eq!(pop.as_u64(), Some(4)); // 4 bits set
}

// =========================================================================
// angr-cudgw.15: binop family-dispatch misroute degrades to OpError, not
// panic. The top-level VEXOps::binop() routing guard and each family fn's
// accepted op-set are two hand-maintained lists; IROp is a plain (non
// #[non_exhaustive]) derive enum, so a guard/family drift compiles clean
// and would otherwise hit an `unreachable!()` that aborts the whole
// process. Each family fn now returns OpError::NotBinary on a misroute so
// the caller falls through to the existing Python-fallback path. These
// tests call the family fns directly with an op that belongs to a
// DIFFERENT family, simulating that drift.
// =========================================================================

#[test]
fn binop_arith_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // And is bitwise, not arith — a deliberate misroute.
    let err = VEXOps::binop_arith(IROp::And(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_bitwise_shift_cmp_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is arith, not bitwise/shift/cmp — a deliberate misroute.
    let err = VEXOps::binop_bitwise_shift_cmp(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_float_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is scalar-int, not float — a deliberate misroute.
    let err = VEXOps::binop_float(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_vec_int_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is scalar-int, not vector-int — a deliberate misroute.
    let err = VEXOps::binop_vec_int(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_vec_float_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is scalar-int, not vector-float — a deliberate misroute.
    let err = VEXOps::binop_vec_float(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}
