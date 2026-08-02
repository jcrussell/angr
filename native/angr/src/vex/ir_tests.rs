use super::*;
use crate::vex::parse_opcode;

#[test]
fn test_ir_type_sizes() {
    assert_eq!(IRType::I1.bits(), 1);
    assert_eq!(IRType::I8.bits(), 8);
    assert_eq!(IRType::I32.bits(), 32);
    assert_eq!(IRType::I64.bits(), 64);
    assert_eq!(IRType::V128.bits(), 128);
}

#[test]
fn test_ir_const_types() {
    assert_eq!(IRConst::U8(42).get_type(), IRType::I8);
    assert_eq!(IRConst::U32(42).get_type(), IRType::I32);
    assert_eq!(IRConst::U64(42).get_type(), IRType::I64);
}

/// `IROp::result_type()` for the width-preserving packed lane ops must derive
/// the total width from elem*count, not hardcode V128 (angr-9ke6b.163). The
/// expectations are cross-checked against `parse_opcode`, so the two sources
/// of truth (opcode_map's mapped shapes and result_type's width rule) cannot
/// drift apart silently.
#[test]
fn test_packed_lane_result_type_tracks_mapped_width() {
    let cases: &[(&str, IRType)] = &[
        // D-reg NEON: 64-bit totals — these were also wrong under the old
        // hardcoded-V128 arm.
        ("Iop_Add8x8", IRType::I64),
        ("Iop_Sub16x4", IRType::I64),
        ("Iop_Mul32x2", IRType::I64),
        ("Iop_ShlN8x8", IRType::I64),
        ("Iop_ShrN16x4", IRType::I64),
        ("Iop_SarN32x2", IRType::I64),
        ("Iop_CmpEQ8x8", IRType::I64),
        ("Iop_CmpGT16Sx4", IRType::I64),
        ("Iop_CmpGT32Ux2", IRType::I64),
        // Q-reg / SSE: 128-bit totals.
        ("Iop_Add8x16", IRType::V128),
        ("Iop_Sub64x2", IRType::V128),
        ("Iop_Mul16x8", IRType::V128),
        ("Iop_ShlN64x2", IRType::V128),
        ("Iop_CmpEQ32x4", IRType::V128),
        ("Iop_CmpGT8Sx16", IRType::V128),
        // AVX2: 256-bit totals — the bug this test pins down.
        ("Iop_Add8x32", IRType::V256),
        ("Iop_Add16x16", IRType::V256),
        ("Iop_Add32x8", IRType::V256),
        ("Iop_Add64x4", IRType::V256),
        ("Iop_Sub8x32", IRType::V256),
        ("Iop_Sub16x16", IRType::V256),
        ("Iop_Sub32x8", IRType::V256),
        ("Iop_Sub64x4", IRType::V256),
        // Widening multiplies: full-lane doubles the lane width, MullEven
        // halves the lane count — both land on V128 for every mapped shape.
        ("Iop_Mull8Ux8", IRType::V128),
        ("Iop_Mull32Sx2", IRType::V128),
        ("Iop_MullEven8Ux16", IRType::V128),
        ("Iop_MullEven32Sx4", IRType::V128),
        ("Iop_QDMull16Sx4", IRType::V128),
        ("Iop_QDMull32Sx2", IRType::V128),
    ];

    for (op_str, expected) in cases {
        let op = parse_opcode(op_str);
        assert!(
            !matches!(op, IROp::Unmapped(_)),
            "{op_str} is not mapped by parse_opcode"
        );
        assert_eq!(
            op.result_type(),
            Some(*expected),
            "{op_str}: result_type disagrees with the mapped lane width"
        );
    }
}

#[test]
fn test_irsb_creation() {
    let irsb = IRSB::new(0x1000, VexArch::AMD64);
    assert_eq!(irsb.addr, 0x1000);
    assert_eq!(irsb.num_instructions(), 0);
}
