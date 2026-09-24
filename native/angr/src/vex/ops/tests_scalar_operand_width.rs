// angr-fs8kb.95: operand-width rejection across the SCALAR op families.
//
// The sibling `tests_vec_operand_width.rs` pins the same contract for the
// packed-vector family (swept by angr-3fb7p). The scalar families shared both
// properties that made `debug_assert_eq!` the wrong tool there:
// `VEXOps::binop`/`unop` match on the opcode alone and never check an operand
// against the `IRType` the opcode carries, and `RustBV`'s own guard
// (`define_binop_pair!` in `symbolic/value_ops.rs`) is itself a
// `debug_assert_eq!` — so in `[profile.release]` a width-mismatched
// `Iop_Add64` folded a wrong constant on the `as_u128` path instead of
// failing. They now route through `VEXOps::require_operand_width` too.
//
// These tests run in every profile, so they pin the release behaviour the old
// `debug_assert!`s could not: an `Err`, not a plausible-looking constant.

use super::*;
use crate::vex::FCmpKind;
use crate::vex::ir::IRType;

/// Assert that `result` is the typed width rejection naming `site`, rather
/// than a folded value.
#[track_caller]
fn assert_width_rejected(result: Result<RustBV, OpError>, site: &str) {
    match result {
        Err(e @ OpError::OperandWidthMismatch { .. }) => {
            let msg = e.to_string();
            assert!(
                msg.contains(site) && msg.contains("operand width"),
                "expected an operand-width rejection naming {site}, got: {msg}"
            );
        }
        other => panic!("expected an operand-width rejection for {site}, got {other:?}"),
    }
}

/// `width_binop!` checks both operands. A 32-bit operand under `Iop_Add64`
/// used to fold to a 32-bit `Concrete` in release.
#[test]
fn test_width_binop_rejects_either_narrow_operand() {
    let ctx = SymContext::new_mock();
    let op = IROp::Add(IRType::I64);
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(1, 32), RustBV::concrete(2, 64), &ctx),
        "binop add_into left",
    );
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(1, 64), RustBV::concrete(2, 32), &ctx),
        "binop add_into right",
    );
}

/// The comparison arms go through the same macro, and their result width
/// (1 bit) hides the mismatch from any downstream width check.
#[test]
fn test_width_binop_rejects_narrow_compare_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::binop(
            IROp::CmpEQ(IRType::I32),
            RustBV::concrete(0, 32),
            RustBV::concrete(0, 8),
            &ctx,
        ),
        "binop eq_into right",
    );
}

/// `width_unop!` — same macro, unary path.
#[test]
fn test_width_unop_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(IROp::Not(IRType::I32), RustBV::concrete(0, 16), &ctx),
        "unop not_into arg",
    );
}

/// Hand-rolled arms that do NOT use `width_unop!`: the extend/truncate family
/// reads `from` out of the opcode and applies it to a guest-data operand.
#[test]
fn test_extend_and_truncate_reject_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(
            IROp::SignExtend {
                from: IRType::I32,
                to: IRType::I64,
            },
            RustBV::concrete(0, 16),
            &ctx,
        ),
        "unop SignExtend arg",
    );
    assert_width_rejected(
        VEXOps::unop(
            IROp::Truncate {
                from: IRType::I64,
                to: IRType::I32,
            },
            RustBV::concrete(0, 32),
            &ctx,
        ),
        "unop Truncate arg",
    );
}

/// `Iop_Extract`'s `low_bit`/`to` come from the opcode; a short operand makes
/// the derived `hi` read past its end.
#[test]
fn test_extract_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(
            IROp::Extract {
                from: IRType::I64,
                to: IRType::I8,
                low_bit: 56,
            },
            RustBV::concrete(0, 32),
            &ctx,
        ),
        "unop Extract arg",
    );
}

/// Shifts normalize the *amount* to the operand width, so only `left` is
/// checked — but it is checked against the opcode's type, not its own width.
#[test]
fn test_shift_rejects_narrow_left_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::binop(
            IROp::Shl(IRType::I64),
            RustBV::concrete(1, 32),
            RustBV::concrete(4, 8),
            &ctx,
        ),
        "binop Shl left",
    );
}

/// `Iop_Concat`'s operand widths are not individually fixed by the opcode,
/// only their sum is — so the guard is a postcondition on the joined result.
#[test]
fn test_concat_rejects_wrong_total_width() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::binop(
            IROp::Concat { ty: IRType::I64 },
            RustBV::concrete(0, 32),
            RustBV::concrete(0, 16),
            &ctx,
        ),
        "binop Concat result",
    );
}

/// `int_arith.rs`: the widening multiply extends both operands to
/// `2 * ty.bits()`, so a narrow operand silently changed the product's width.
#[test]
fn test_widening_mul_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    let op = IROp::MullS(IRType::I32);
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(3, 16), RustBV::concrete(5, 32), &ctx),
        "widening_mul left",
    );
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(3, 32), RustBV::concrete(5, 16), &ctx),
        "widening_mul right",
    );
}

/// `int_arith.rs`: the two divmod entry points pin absolute widths, and the
/// shared `divmod_double_to_single` pins the 2:1 relation between them.
#[test]
fn test_divmod_rejects_mismatched_operands() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::binop(
            IROp::DivModU64to32,
            RustBV::concrete(9, 32),
            RustBV::concrete(2, 32),
            &ctx,
        ),
        "divmod_64_to_32 dividend",
    );
    assert_width_rejected(
        VEXOps::binop(
            IROp::DivModS64to32,
            RustBV::concrete(9, 64),
            RustBV::concrete(2, 16),
            &ctx,
        ),
        "divmod_64_to_32 divisor",
    );
    assert_width_rejected(
        VEXOps::binop(
            IROp::DivModU128to64,
            RustBV::concrete(9, 64),
            RustBV::concrete(2, 64),
            &ctx,
        ),
        "divmod_128_to_64 dividend",
    );
}

/// `conversions.rs`: an int-to-float conversion reads the source width from
/// the opcode and the payload from the operand.
#[test]
fn test_int_to_float_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(IROp::I32StoF64, RustBV::concrete(0, 16), &ctx),
        "int_to_float arg",
    );
}

/// `conversions.rs`: and the float-to-int direction reads it from the source
/// `FloatPrec`.
#[test]
fn test_float_to_int_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(IROp::F64toI32S, RustBV::concrete(0, 32), &ctx),
        "float_to_int arg",
    );
}

/// `float_cmp.rs`'s two FP-compare helpers live outside the `vec_*` modules
/// the angr-3fb7p sweep walked, so they kept their `debug_assert!`s a round
/// longer despite deriving lane offsets exactly the same way.
#[test]
fn test_float_packed_cmp_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::binop(
            IROp::FCmpVecPacked {
                kind: FCmpKind::Eq,
                elem: IRType::F32,
                count: 4,
            },
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 128),
            &ctx,
        ),
        "vec_float_packed_cmp left",
    );
}
