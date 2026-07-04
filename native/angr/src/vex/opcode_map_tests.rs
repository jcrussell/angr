//! Tests for [`super::super::opcode_map`] — VEX opcode string parsing (`parse_opcode`).
//!
//! Extracted from the `mod tests` block in `opcode_map.rs` (see
//! `rust-mod-tests-sibling-extraction` bd memory).

use super::*;

#[test]
fn test_parse_arithmetic() {
    assert_eq!(parse_opcode("Iop_Add32"), IROp::Add(IRType::I32));
    assert_eq!(parse_opcode("Iop_Add64"), IROp::Add(IRType::I64));
    assert_eq!(parse_opcode("Iop_Sub32"), IROp::Sub(IRType::I32));
    assert_eq!(parse_opcode("Iop_Mul64"), IROp::Mul(IRType::I64));
}

#[test]
fn test_parse_bitwise() {
    assert_eq!(parse_opcode("Iop_And32"), IROp::And(IRType::I32));
    assert_eq!(parse_opcode("Iop_Or64"), IROp::Or(IRType::I64));
    assert_eq!(parse_opcode("Iop_Xor32"), IROp::Xor(IRType::I32));
    assert_eq!(parse_opcode("Iop_Not64"), IROp::Not(IRType::I64));
}

#[test]
fn test_parse_shift() {
    assert_eq!(parse_opcode("Iop_Shl32"), IROp::Shl(IRType::I32));
    assert_eq!(parse_opcode("Iop_Shr64"), IROp::Shr(IRType::I64));
    assert_eq!(parse_opcode("Iop_Sar32"), IROp::Sar(IRType::I32));
}

#[test]
fn test_parse_comparison() {
    assert_eq!(parse_opcode("Iop_CmpEQ32"), IROp::CmpEQ(IRType::I32));
    assert_eq!(parse_opcode("Iop_CmpNE64"), IROp::CmpNE(IRType::I64));
    assert_eq!(parse_opcode("Iop_CmpLT32S"), IROp::CmpLT(IRType::I32));
    assert_eq!(parse_opcode("Iop_CmpLT64U"), IROp::CmpLTU(IRType::I64));
}

#[test]
fn test_parse_conversion() {
    assert_eq!(
        parse_opcode("Iop_32Sto64"),
        IROp::SignExtend {
            from: IRType::I32,
            to: IRType::I64
        }
    );
    assert_eq!(
        parse_opcode("Iop_32Uto64"),
        IROp::ZeroExtend {
            from: IRType::I32,
            to: IRType::I64
        }
    );
    assert_eq!(
        parse_opcode("Iop_64to32"),
        IROp::Truncate {
            from: IRType::I64,
            to: IRType::I32
        }
    );
}

#[test]
fn test_parse_type() {
    assert_eq!(parse_type("Ity_I32"), Some(IRType::I32));
    assert_eq!(parse_type("Ity_I64"), Some(IRType::I64));
    assert_eq!(parse_type("Ity_V128"), Some(IRType::V128));
    assert_eq!(parse_type("Unknown"), None);
}

#[test]
fn test_parse_jumpkind() {
    assert_eq!(
        parse_jumpkind("Ijk_Boring"),
        super::super::ir::JumpKind::Boring
    );
    assert_eq!(parse_jumpkind("Ijk_Call"), super::super::ir::JumpKind::Call);
    assert_eq!(parse_jumpkind("Ijk_Ret"), super::super::ir::JumpKind::Ret);
}

#[test]
fn test_unmapped_opcode() {
    // Unknown opcodes now route through IROp::Unmapped (angr-tkbr.2).
    // Dispatch in VEXOps::unop/binop/triop/qop surfaces this as
    // OpError::UnsupportedVexOp, which the engine maps to
    // RustUnsupportedVexOpError.
    match parse_opcode("Iop_UnknownOp") {
        IROp::Unmapped(name) => assert_eq!(name, "Iop_UnknownOp"),
        other => panic!("expected Unmapped, got {other:?}"),
    }
    // The interner must dedupe — same name returns the same pointer.
    let a = match parse_opcode("Iop_UnknownOp") {
        IROp::Unmapped(n) => n,
        _ => unreachable!(),
    };
    let b = match parse_opcode("Iop_UnknownOp") {
        IROp::Unmapped(n) => n,
        _ => unreachable!(),
    };
    assert!(std::ptr::eq(a, b), "interner must dedupe by string");
}

#[test]
fn test_neon_unimplemented_scaffold_is_empty() {
    // The NeonUnimplemented scaffold (parse_neon_unimplemented) routes
    // claimed-but-unimplemented NEON opcodes through
    // IROp::NeonUnimplemented(name) so dispatch panics with the original
    // opcode name instead of silently producing a fresh-symbolic value.
    // As of angr-cudgw.6 no NEON op routes there anymore — the last
    // placeholder, Iop_PwAdd32Fx2 (FP pairwise add), graduated to
    // IROp::VFPwAdd. This test guards that graduation: if Iop_PwAdd32Fx2
    // ever regresses back to the sentinel, this fails. When a future NEON
    // op is parked via the scaffold, add a positive routing assertion here.
    match parse_opcode("Iop_PwAdd32Fx2") {
        IROp::VFPwAdd { elem, count } => {
            assert_eq!(elem, IRType::F32);
            assert_eq!(count, 2);
        }
        other => panic!("Iop_PwAdd32Fx2 expected VFPwAdd, got {other:?}"),
    }
}

#[test]
fn test_neon_does_not_shadow_existing_mappings() {
    // Sanity: opcodes already mapped to real IROps (VAdd/VShlN/etc.)
    // must not be intercepted by the NEON-unimplemented scaffold.
    assert!(matches!(parse_opcode("Iop_Add8x8"), IROp::VAdd { .. }));
    assert!(matches!(parse_opcode("Iop_ShlN32x4"), IROp::VShlN { .. }));
    assert!(matches!(
        parse_opcode("Iop_CmpEQ32Fx4"),
        IROp::FCmpVecPacked { .. }
    ));
    // angr-bkcs.2: Mul8x{8,16} + GetElem/SetElem are real ops, not
    // NeonUnimplemented placeholders.
    assert!(matches!(parse_opcode("Iop_Mul8x8"), IROp::VMul { .. }));
    assert!(matches!(parse_opcode("Iop_Mul8x16"), IROp::VMul { .. }));
    assert!(matches!(
        parse_opcode("Iop_GetElem8x8"),
        IROp::VGetElem { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_GetElem32x4"),
        IROp::VGetElem { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_GetElem64x2"),
        IROp::VGetElem { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_SetElem8x8"),
        IROp::VSetElem { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_SetElem32x4"),
        IROp::VSetElem { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_SetElem64x2"),
        IROp::VSetElem { .. }
    ));
    // angr-hzs0: Dup / Widen / NarrowUn / NarrowBin / QNarrow{Un,Bin}
    // are real ops, no longer routed through NeonUnimplemented.
    assert!(matches!(parse_opcode("Iop_Dup8x8"), IROp::VDup { .. }));
    assert!(matches!(parse_opcode("Iop_Dup32x4"), IROp::VDup { .. }));
    assert!(matches!(
        parse_opcode("Iop_Widen8Sto16x8"),
        IROp::VWiden { signed: true, .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_Widen32Uto64x2"),
        IROp::VWiden { signed: false, .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_NarrowUn16to8x8"),
        IROp::VNarrowUn { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_NarrowBin16to8x8"),
        IROp::VNarrowBin { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_QNarrowUn16Sto8Sx8"),
        IROp::VQNarrowUn { .. }
    ));
    assert!(matches!(
        parse_opcode("Iop_QNarrowBin16Sto8Sx8"),
        IROp::VQNarrowBin { .. }
    ));
    // angr-tukg.7: Shl/Shr/Sar/Sal{N}x{M} are real shift-by-vector ops,
    // not the by-immediate ShlN/ShrN/SarN variants and not unimplemented
    // placeholders. Sal routes to VShl (same bit semantics on LE shift).
    assert!(matches!(parse_opcode("Iop_Shl8x8"), IROp::VShl { .. }));
    assert!(matches!(parse_opcode("Iop_Shl64x2"), IROp::VShl { .. }));
    assert!(matches!(parse_opcode("Iop_Shr16x4"), IROp::VShr { .. }));
    assert!(matches!(parse_opcode("Iop_Shr32x4"), IROp::VShr { .. }));
    assert!(matches!(parse_opcode("Iop_Sar8x16"), IROp::VSar { .. }));
    assert!(matches!(parse_opcode("Iop_Sar64x2"), IROp::VSar { .. }));
    assert!(matches!(parse_opcode("Iop_Sal8x8"), IROp::VShl { .. }));
    assert!(matches!(parse_opcode("Iop_Sal64x2"), IROp::VShl { .. }));
}

#[test]
fn test_parse_vreverse_routing() {
    // angr-tukg.4: Iop_Reverse{N}sIn{M}_x{K} variants route to VReverse
    // with the expected (sub_width, elem, count) decomposition.
    let cases: &[(&str, u8, IRType, u8)] = &[
        ("Iop_Reverse8sIn16_x4", 8, IRType::I16, 4),
        ("Iop_Reverse8sIn16_x8", 8, IRType::I16, 8),
        ("Iop_Reverse8sIn32_x2", 8, IRType::I32, 2),
        ("Iop_Reverse8sIn32_x4", 8, IRType::I32, 4),
        ("Iop_Reverse8sIn64_x1", 8, IRType::I64, 1),
        ("Iop_Reverse8sIn64_x2", 8, IRType::I64, 2),
        ("Iop_Reverse16sIn32_x2", 16, IRType::I32, 2),
        ("Iop_Reverse16sIn32_x4", 16, IRType::I32, 4),
        ("Iop_Reverse16sIn64_x1", 16, IRType::I64, 1),
        ("Iop_Reverse16sIn64_x2", 16, IRType::I64, 2),
        ("Iop_Reverse32sIn64_x1", 32, IRType::I64, 1),
        ("Iop_Reverse32sIn64_x2", 32, IRType::I64, 2),
        ("Iop_Reverse1sIn8_x8", 1, IRType::I8, 8),
        ("Iop_Reverse1sIn8_x16", 1, IRType::I8, 16),
    ];
    for (op_str, sw, e, c) in cases {
        match parse_opcode(op_str) {
            IROp::VReverse {
                sub_width,
                elem,
                count,
            } => {
                assert_eq!(sub_width, *sw, "{op_str}: sub_width");
                assert_eq!(elem, *e, "{op_str}: elem");
                assert_eq!(count, *c, "{op_str}: count");
            }
            other => panic!("{op_str}: expected VReverse, got {other:?}"),
        }
    }
}
