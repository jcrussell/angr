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
fn test_setv128lo64_resolves_to_binop_not_reinterpret() {
    // parse_float runs before parse_vector in parse_opcode, so
    // "Iop_SetV128lo64" must map to the binop variant (upper 64 bits
    // preserved) — NOT a Reinterpret(I64->V128). Locks in angr-ph300.62.
    assert_eq!(parse_opcode("Iop_SetV128lo64"), IROp::SetV128lo64);
    assert_eq!(parse_opcode("Iop_SetV128lo32"), IROp::SetV128lo32);
}

#[test]
fn test_parse_type() {
    assert_eq!(parse_type("Ity_I32"), Some(IRType::I32));
    assert_eq!(parse_type("Ity_I64"), Some(IRType::I64));
    assert_eq!(parse_type("Ity_V128"), Some(IRType::V128));
    assert_eq!(parse_type("Unknown"), None);
}

/// Decimal/quad float types have no dedicated `IRType` variant; the fallback
/// arms in `parse_type` must at least preserve the declared bit width. See the
/// comment on those arms for why an approximation beats `None` here.
#[test]
fn test_parse_type_decimal_and_quad_floats_preserve_width() {
    for (ty_str, bits) in [
        ("Ity_D32", 32),
        ("Ity_D64", 64),
        ("Ity_D128", 128),
        ("Ity_F128", 128),
    ] {
        let ty = parse_type(ty_str).unwrap_or_else(|| panic!("{ty_str} must map to some IRType"));
        assert_eq!(ty.bits(), bits, "{ty_str} narrowed to {} bits", ty.bits());
    }
}

#[test]
fn test_parse_jumpkind() {
    assert_eq!(
        parse_jumpkind("Ijk_Boring"),
        super::super::ir::JumpKind::Boring
    );
    assert_eq!(parse_jumpkind("Ijk_Call"), super::super::ir::JumpKind::Call);
    assert_eq!(parse_jumpkind("Ijk_Ret"), super::super::ir::JumpKind::Ret);

    // angr-36vvn.8: these 3 real VEX jumpkinds previously fell through to
    // Boring; each must now round-trip through ijk_name.
    for jk in [
        super::super::ir::JumpKind::FlushDCacheLine,
        super::super::ir::JumpKind::ExtV128,
        super::super::ir::JumpKind::Extension,
    ] {
        assert_eq!(parse_jumpkind(jk.ijk_name()), jk);
    }

    // angr-sqfj8.111: the trap/signal/privileged/int kinds were the next
    // batch to fall through to Boring, masking guest traps as ordinary
    // fallthrough exits.
    for jk in [
        super::super::ir::JumpKind::NoRedir,
        super::super::ir::JumpKind::SigILL,
        super::super::ir::JumpKind::SigTRAP,
        super::super::ir::JumpKind::SigSEGV,
        super::super::ir::JumpKind::SigBUS,
        super::super::ir::JumpKind::SigFPE,
        super::super::ir::JumpKind::SigFPE_IntDiv,
        super::super::ir::JumpKind::SigFPE_IntOvf,
        super::super::ir::JumpKind::Privileged,
        super::super::ir::JumpKind::Sys_int,
        super::super::ir::JumpKind::Sys_int32,
    ] {
        assert_eq!(parse_jumpkind(jk.ijk_name()), jk);
    }
}

/// Every `IRJumpKind` tag in `vendor/pyvex_ffi.h` must parse to a
/// non-`Boring` variant (angr-sqfj8.111). Guards the whole class rather
/// than the specific kinds this bead added: a future VEX bump that grows
/// the C enum should fail here, not silently execute a trap as
/// fallthrough.
#[test]
fn test_parse_jumpkind_covers_every_vex_tag() {
    for tag in [
        "Ijk_Boring",
        "Ijk_Call",
        "Ijk_Ret",
        "Ijk_ClientReq",
        "Ijk_Yield",
        "Ijk_EmWarn",
        "Ijk_EmFail",
        "Ijk_NoDecode",
        "Ijk_MapFail",
        "Ijk_InvalICache",
        "Ijk_FlushDCache",
        "Ijk_NoRedir",
        "Ijk_SigILL",
        "Ijk_SigTRAP",
        "Ijk_SigSEGV",
        "Ijk_SigBUS",
        "Ijk_SigFPE",
        "Ijk_SigFPE_IntDiv",
        "Ijk_SigFPE_IntOvf",
        "Ijk_Privileged",
        "Ijk_Sys_syscall",
        "Ijk_Sys_int",
        "Ijk_Sys_int32",
        "Ijk_Sys_int128",
        "Ijk_Sys_int129",
        "Ijk_Sys_int130",
        "Ijk_Sys_int145",
        "Ijk_Sys_int210",
        "Ijk_Sys_sysenter",
    ] {
        let jk = parse_jumpkind(tag);
        assert_eq!(jk.ijk_name(), tag, "{tag} did not round-trip");
        if tag != "Ijk_Boring" {
            assert_ne!(
                jk,
                super::super::ir::JumpKind::Boring,
                "{tag} fell through to Boring"
            );
        }
    }
}

/// `Ijk_Sys_int`/`Ijk_Sys_int32` are syscall exits, not ordinary jumps:
/// angr classifies on the `Ijk_Sys` prefix (`engines/successors.py`), so
/// `is_syscall` must too. `Ijk_Sig*`/`Ijk_Privileged` are traps and must
/// not be mistaken for either.
#[test]
fn test_jumpkind_syscall_and_trap_classification() {
    use super::super::ir::JumpKind;

    for jk in [JumpKind::Sys_int, JumpKind::Sys_int32] {
        assert!(jk.is_syscall(), "{} must be a syscall", jk.ijk_name());
        assert!(!jk.is_trap(), "{} must not be a trap", jk.ijk_name());
    }
    for jk in [
        JumpKind::SigILL,
        JumpKind::SigTRAP,
        JumpKind::SigSEGV,
        JumpKind::SigBUS,
        JumpKind::SigFPE,
        JumpKind::SigFPE_IntDiv,
        JumpKind::SigFPE_IntOvf,
        JumpKind::Privileged,
    ] {
        assert!(jk.is_trap(), "{} must be a trap", jk.ijk_name());
        assert!(!jk.is_syscall(), "{} must not be a syscall", jk.ijk_name());
        assert!(!jk.is_call() && !jk.is_ret());
    }
    // NoRedir is an ordinary jump with a translation hint, not a trap.
    assert!(!JumpKind::NoRedir.is_trap());
    assert!(!JumpKind::NoRedir.is_syscall());
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
fn test_widening_vector_multiply_routing() {
    // angr-ph300.78: the widening vector-multiply families now map to
    // IROp::VMull (implemented in ops_vec_permute_mul::vec_mull). Real libVEX
    // names put S/U AFTER the size (e.g. Iop_Mull32Sx2); the former phantom
    // "Iop_MullS32x4" arm libVEX never emits still routes to Unmapped.
    //
    // Full-lane family (Iop_Mull{N}{S,U}x{M}, (I64,I64)->V128, NEON VMULL).
    for (op, elem, count, signed) in [
        ("Iop_Mull8Ux8", IRType::I8, 8u8, false),
        ("Iop_Mull8Sx8", IRType::I8, 8, true),
        ("Iop_Mull16Ux4", IRType::I16, 4, false),
        ("Iop_Mull16Sx4", IRType::I16, 4, true),
        ("Iop_Mull32Ux2", IRType::I32, 2, false),
        ("Iop_Mull32Sx2", IRType::I32, 2, true),
    ] {
        assert_eq!(
            parse_opcode(op),
            IROp::VMull {
                elem,
                count,
                signed,
                even: false
            },
            "{op}"
        );
    }
    // Even-lane family (Iop_MullEven{N}{S,U}x{M}, (V128,V128)->V128, PMUL[U]DQ).
    for (op, elem, count, signed) in [
        ("Iop_MullEven8Ux16", IRType::I8, 16u8, false),
        ("Iop_MullEven8Sx16", IRType::I8, 16, true),
        ("Iop_MullEven16Ux8", IRType::I16, 8, false),
        ("Iop_MullEven16Sx8", IRType::I16, 8, true),
        ("Iop_MullEven32Ux4", IRType::I32, 4, false),
        ("Iop_MullEven32Sx4", IRType::I32, 4, true),
    ] {
        assert_eq!(
            parse_opcode(op),
            IROp::VMull {
                elem,
                count,
                signed,
                even: true
            },
            "{op}"
        );
    }
    // The phantom "Iop_MullS32x4" name libVEX never emits stays Unmapped.
    match parse_opcode("Iop_MullS32x4") {
        IROp::Unmapped(name) => assert_eq!(name, "Iop_MullS32x4"),
        other => panic!("Iop_MullS32x4 (phantom) expected Unmapped, got {other:?}"),
    }
    // Guard the regression: the scalar widening multiplies with the SAME
    // "Iop_MullS"/"Iop_MullU" prefix must still map (they share the prefix the
    // vector arms key on, so a careless refactor could shadow them).
    assert_eq!(parse_opcode("Iop_MullS32"), IROp::MullS(IRType::I32));
    assert_eq!(parse_opcode("Iop_MullU16"), IROp::MullU(IRType::I16));
}

#[test]
fn test_qdmull_opcode_mapping() {
    // Signed doubling saturating widening multiply (angr-1yge9.5). Only these
    // two names exist in libVEX; both (I64,I64)->V128, always signed/full-lane.
    assert_eq!(
        parse_opcode("Iop_QDMull16Sx4"),
        IROp::VQDMull {
            elem: IRType::I16,
            count: 4
        }
    );
    assert_eq!(
        parse_opcode("Iop_QDMull32Sx2"),
        IROp::VQDMull {
            elem: IRType::I32,
            count: 2
        }
    );
    // The "Iop_QDMull" prefix must not shadow the plain "Iop_Mull" arm.
    assert_eq!(
        parse_opcode("Iop_Mull16Sx4"),
        IROp::VMull {
            elem: IRType::I16,
            count: 4,
            signed: true,
            even: false
        }
    );
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

/// angr-9ke6b.160: the unsigned vector greater-than family
/// (`Iop_CmpGT{N}Ux{M}`) used to be entirely unmapped, so ARM NEON `VCGT.U*`
/// fell back to Python. Pin both polarities of the whole table, including the
/// absence of `Iop_CmpGT64Ux1` (libVEX defines no D-reg 64-bit lane).
#[test]
fn test_parse_vec_cmp_gt_signed_and_unsigned() {
    let cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_CmpGT8Sx8", IRType::I8, 8, true),
        ("Iop_CmpGT16Sx4", IRType::I16, 4, true),
        ("Iop_CmpGT32Sx2", IRType::I32, 2, true),
        ("Iop_CmpGT8Sx16", IRType::I8, 16, true),
        ("Iop_CmpGT16Sx8", IRType::I16, 8, true),
        ("Iop_CmpGT32Sx4", IRType::I32, 4, true),
        ("Iop_CmpGT64Sx2", IRType::I64, 2, true),
        ("Iop_CmpGT8Ux8", IRType::I8, 8, false),
        ("Iop_CmpGT16Ux4", IRType::I16, 4, false),
        ("Iop_CmpGT32Ux2", IRType::I32, 2, false),
        ("Iop_CmpGT8Ux16", IRType::I8, 16, false),
        ("Iop_CmpGT16Ux8", IRType::I16, 8, false),
        ("Iop_CmpGT32Ux4", IRType::I32, 4, false),
        ("Iop_CmpGT64Ux2", IRType::I64, 2, false),
    ];
    for &(op_str, elem, count, signed) in cases {
        assert_eq!(
            parse_opcode(op_str),
            IROp::VCmpGT {
                elem,
                count,
                signed
            },
            "{op_str}"
        );
    }
    assert!(matches!(parse_opcode("Iop_CmpGT64Ux1"), IROp::Unmapped(_)));
}

/// angr-9ke6b.233: `IROp::Raw` had no producer, so every x87 / FRECPX
/// transcendental parsed to `Unmapped` and the libm fast paths in
/// `vex::transcendentals` were dead outside their own unit tests.
#[test]
fn test_parse_transcendental() {
    use crate::vex::transcendentals as tr;
    let cases: &[(&str, u32)] = &[
        ("Iop_SinF64", tr::IOP_SIN_F64),
        ("Iop_CosF64", tr::IOP_COS_F64),
        ("Iop_TanF64", tr::IOP_TAN_F64),
        ("Iop_2xm1F64", tr::IOP_2XM1_F64),
        ("Iop_RecpExpF64", tr::IOP_RECPEXP_F64),
        ("Iop_RecpExpF32", tr::IOP_RECPEXP_F32),
        ("Iop_AtanF64", tr::IOP_ATAN_F64),
        ("Iop_Yl2xF64", tr::IOP_YL2X_F64),
        ("Iop_Yl2xp1F64", tr::IOP_YL2XP1_F64),
        ("Iop_ScaleF64", tr::IOP_SCALE_F64),
    ];
    for &(op_str, tag) in cases {
        assert_eq!(parse_opcode(op_str), IROp::Raw(tag), "{op_str}");
    }
    // Iop_PRem*F64 is deliberately out of scope — still Unmapped.
    assert!(matches!(parse_opcode("Iop_PRemF64"), IROp::Unmapped(_)));
}
