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

/// angr-sqfj8.120: `parse_comparison` documents the `Iop_CmpORD*` family as an
/// intentional gap — ordered comparison yields a full-width -1/0 result rather
/// than the 1-bit result every mapped `Iop_Cmp*` arm produces, so routing it
/// through the neighboring `CmpLT`/`CmpLE` arms would be silently wrong (a
/// widening of those arms is the plausible way this breaks, since the tuple
/// suffixes `32S`/`32U`/`64S`/`64U` are shared). Pin the gap so it stays a
/// deliberate `Unmapped` Python fallback instead of a wrong answer.
#[test]
fn test_parse_cmp_ord_stays_unmapped() {
    for absent in [
        "Iop_CmpORD32S",
        "Iop_CmpORD32U",
        "Iop_CmpORD64S",
        "Iop_CmpORD64U",
    ] {
        assert!(
            matches!(parse_opcode(absent), IROp::Unmapped(_)),
            "{absent} must stay unmapped (full-width -1/0 result, not a 1-bit compare)"
        );
    }
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

/// The logged wrapper both marshalling paths use must pass known types through
/// unchanged and land on the documented `I64` default for an unknown one.
#[test]
fn test_parse_type_or_log_defaults_to_i64() {
    assert_eq!(parse_type_or_log("Ity_I32", "test"), IRType::I32);
    assert_eq!(parse_type_or_log("Ity_V256", "test"), IRType::V256);
    assert_eq!(parse_type_or_log("Ity_D64", "test"), IRType::I64);
    assert_eq!(parse_type_or_log("Ity_Nonsense", "test"), IRType::I64);
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

/// Both `Iend_*` strings must round-trip, and an unrecognized one must land on
/// the documented (logged) little-endian default rather than panicking — the
/// `SILENT(cat-c)` arm on `parse_endness`, previously untested (angr-03vl4.69).
#[test]
fn test_parse_endness_defaults_to_little() {
    use super::super::ir::Endness;

    assert_eq!(parse_endness("Iend_LE"), Endness::Little);
    assert_eq!(parse_endness("Iend_BE"), Endness::Big);
    for unknown in ["Iend_ME", "Iend", "", "iend_be"] {
        assert_eq!(
            parse_endness(unknown),
            Endness::Little,
            "unknown endness {unknown:?} must fall back to little-endian"
        );
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
    // Dispatch in VEXOps::unop/binop/qop/binop_with_rm surfaces this as
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
    // IROp::VMull (implemented in ops::vec_permute_mul::vec_mull). Real libVEX
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
    // IROp::NeonUnimplemented(name) so dispatch returns
    // OpError::UnsupportedNeon carrying the original opcode name instead of
    // silently producing a fresh-symbolic value.
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

/// angr-sqfj8.113 / .118: the three shift-by-immediate families cover the same
/// widths, up through the AVX2 256-bit shapes. `Iop_SarN64x2` used to fall
/// through to `Unmapped` on the false premise that pyvex declares no such op,
/// and the whole 16x16/32x8/64x4 tier was missing.
#[test]
fn test_shift_by_immediate_families_cover_the_same_widths() {
    let widths: [(&str, IRType, u8); 10] = [
        ("8x8", IRType::I8, 8),
        ("8x16", IRType::I8, 16),
        ("16x4", IRType::I16, 4),
        ("16x8", IRType::I16, 8),
        ("16x16", IRType::I16, 16),
        ("32x2", IRType::I32, 2),
        ("32x4", IRType::I32, 4),
        ("32x8", IRType::I32, 8),
        ("64x2", IRType::I64, 2),
        ("64x4", IRType::I64, 4),
    ];
    for (suffix, want_elem, want_count) in widths {
        for family in ["ShlN", "ShrN", "SarN"] {
            // The one legitimate hole in the grid: arithmetic qword shift
            // arrived with AVX-512 (VPSRAQ), so VEX declares no Iop_SarN64x4.
            if family == "SarN" && suffix == "64x4" {
                assert!(
                    matches!(parse_opcode("Iop_SarN64x4"), IROp::Unmapped(_)),
                    "Iop_SarN64x4 is not a real VEX op; mapping it would be a dead arm"
                );
                continue;
            }
            let op_str = format!("Iop_{family}{suffix}");
            let (elem, count) = match parse_opcode(&op_str) {
                IROp::VShlN { elem, count }
                | IROp::VShrN { elem, count }
                | IROp::VSarN { elem, count } => (elem, count),
                other => panic!("{op_str} expected a V{family} mapping, got {other:?}"),
            };
            assert_eq!(elem, want_elem, "{op_str} element type");
            assert_eq!(count, want_count, "{op_str} lane count");
        }
    }

    // AVX2 has no byte shift-by-immediate, so no family declares an 8x32 form.
    for family in ["ShlN", "ShrN", "SarN"] {
        let op_str = format!("Iop_{family}8x32");
        assert!(
            matches!(parse_opcode(&op_str), IROp::Unmapped(_)),
            "{op_str} is not a real VEX op; mapping it would be a dead arm"
        );
    }
}

/// angr-sqfj8.114: the NEON D-reg 2-lane float shape. VEX declares `32Fx2`
/// for Add/Sub/Mul/Min/Max only — Div/Sqrt have no such opcode, so those
/// strings staying unmapped is the correct outcome, not a second gap.
#[test]
fn test_packed_float_32fx2_maps() {
    for op_str in [
        "Iop_Add32Fx2",
        "Iop_Sub32Fx2",
        "Iop_Mul32Fx2",
        "Iop_Min32Fx2",
        "Iop_Max32Fx2",
    ] {
        let (elem, count) = match parse_opcode(op_str) {
            IROp::VFAdd { elem, count }
            | IROp::VFSub { elem, count }
            | IROp::VFMul { elem, count }
            | IROp::VFMin { elem, count }
            | IROp::VFMax { elem, count } => (elem, count),
            other => panic!("{op_str} expected a packed-float mapping, got {other:?}"),
        };
        assert_eq!(elem, IRType::F32, "{op_str} element type");
        assert_eq!(count, 2, "{op_str} lane count");
    }
    for absent in ["Iop_Div32Fx2", "Iop_Sqrt32Fx2"] {
        assert!(
            matches!(parse_opcode(absent), IROp::Unmapped(_)),
            "{absent} is not a VEX op"
        );
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
        // AVX2 256-bit (angr-sqfj8.112) — signed only.
        ("Iop_CmpGT8Sx32", IRType::I8, 32, true),
        ("Iop_CmpGT16Sx16", IRType::I16, 16, true),
        ("Iop_CmpGT32Sx8", IRType::I32, 8, true),
        ("Iop_CmpGT64Sx4", IRType::I64, 4, true),
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
    // libVEX declares no unsigned 256-bit greater-than (VPCMPGT* is signed),
    // so these must stay unmapped rather than silently parsing (angr-sqfj8.112).
    for absent in [
        "Iop_CmpGT8Ux32",
        "Iop_CmpGT16Ux16",
        "Iop_CmpGT32Ux8",
        "Iop_CmpGT64Ux4",
    ] {
        assert!(
            matches!(parse_opcode(absent), IROp::Unmapped(_)),
            "{absent} is not a libVEX op and must not parse"
        );
    }
}

/// angr-sqfj8.112: the AVX2 256-bit vector equality compares
/// (`Iop_CmpEQ{N}x{M}` with a 256-bit total) were unmapped even though the
/// sibling `VAdd`/`VSub` families already carried the same shapes.
#[test]
fn test_parse_vec_cmp_eq_shapes() {
    let cases: &[(&str, IRType, u8)] = &[
        ("Iop_CmpEQ8x8", IRType::I8, 8),
        ("Iop_CmpEQ16x4", IRType::I16, 4),
        ("Iop_CmpEQ32x2", IRType::I32, 2),
        ("Iop_CmpEQ8x16", IRType::I8, 16),
        ("Iop_CmpEQ16x8", IRType::I16, 8),
        ("Iop_CmpEQ32x4", IRType::I32, 4),
        ("Iop_CmpEQ64x2", IRType::I64, 2),
        ("Iop_CmpEQ8x32", IRType::I8, 32),
        ("Iop_CmpEQ16x16", IRType::I16, 16),
        ("Iop_CmpEQ32x8", IRType::I32, 8),
        ("Iop_CmpEQ64x4", IRType::I64, 4),
    ];
    for &(op_str, elem, count) in cases {
        assert_eq!(
            parse_opcode(op_str),
            IROp::VCmpEQ { elem, count },
            "{op_str}"
        );
    }
}

/// angr-0bh1z: the four IEEE-754-2008 max-number/min-number opcodes (AArch32
/// VMAXNM/VMINNM) were unmapped on ARM32, a Supported architecture. They must
/// stay distinct from the compare-and-select `Iop_Max32Fx4`-style vector ops,
/// which parse to `VFMax`/`VFMin` and have different NaN behaviour.
#[test]
fn test_parse_max_num_min_num() {
    assert_eq!(parse_opcode("Iop_MaxNumF32"), IROp::FMaxNum(IRType::F32));
    assert_eq!(parse_opcode("Iop_MaxNumF64"), IROp::FMaxNum(IRType::F64));
    assert_eq!(parse_opcode("Iop_MinNumF32"), IROp::FMinNum(IRType::F32));
    assert_eq!(parse_opcode("Iop_MinNumF64"), IROp::FMinNum(IRType::F64));
    // The vector Max/Min family is untouched by the new arms.
    assert_eq!(
        parse_opcode("Iop_Max32Fx4"),
        IROp::VFMax {
            elem: IRType::F32,
            count: 4
        }
    );
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

/// angr-sqfj8.117: the `VPerm` arm carried a dead `Iop_Perm8x32` string (no
/// such opcode in `vendor/pyvex_ffi.h`) while the real `Iop_Perm32x4` /
/// `Iop_Perm32x8` were unmapped. Also pins `result_type`, which used to be a
/// hardcoded `V128` and was therefore wrong for the 64- and 256-bit members.
#[test]
fn test_parse_perm() {
    let cases: &[(&str, IRType, u8, IRType)] = &[
        ("Iop_Perm8x8", IRType::I8, 8, IRType::I64),
        ("Iop_Perm8x16", IRType::I8, 16, IRType::V128),
        ("Iop_Perm32x4", IRType::I32, 4, IRType::V128),
        ("Iop_Perm32x8", IRType::I32, 8, IRType::V256),
    ];
    for &(op_str, elem, count, result) in cases {
        let op = parse_opcode(op_str);
        assert_eq!(op, IROp::VPerm { elem, count }, "{op_str}");
        assert_eq!(op.result_type(), Some(result), "{op_str} result_type");
    }
    // Never existed in VEX — the dead string that motivated this bead.
    assert!(matches!(parse_opcode("Iop_Perm8x32"), IROp::Unmapped(_)));
    // Real, but deliberately unmapped: a triop whose two-table form Python's
    // two-argument `_op_generic_Perm` cannot model either.
    assert!(matches!(parse_opcode("Iop_Perm8x16x2"), IROp::Unmapped(_)));
}

/// Harness 5 (opcode-map completeness check, `plans/we-keep-finding-bugs-
/// optimized-acorn.md`): every real `Iop_*` opcode name the pinned VEX
/// version declares must be either mapped by `parse_opcode` or named in
/// `KNOWN_UNMAPPED_GROUPS` below with a reason. A new pyvex opcode silently
/// falling through to `IROp::Unmapped` with no allowlist entry is the
/// op-coverage-gap shape from angr-9ke6b.160 / angr-sqfj8.113 /
/// angr-sqfj8.117 — this test exists so the next instance is a hard failure
/// instead of a silent perf cliff nobody notices until an audit finds it.
///
/// Ground truth is `vendor/pyvex_ffi.h` — the vendored cffi cdef
/// `tools/regen-pyvex-ffi-header.py` writes from the installed pyvex's own
/// `IROp` C enum (see that script's docstring), so it is exactly the set of
/// opcodes this pyvex pin can emit — not a hand-maintained approximation.
/// `libvex_lifter_tests.rs::test_vendored_header_ilgop_variant_set_is_unchanged`
/// established this "diff against the vendored header" pattern for the
/// sibling `ILGop_*` tag family; this reuses the same
/// find-the-next-`Prefix_`-token extraction rather than a second one.
///
/// Same "diff against a foreign ground truth + self-cleaning allowlist"
/// shape `tests/engines/rust/test_arch_offset_parity.py` already proved out
/// for arch/register coverage against `archinfo` (its `KNOWN_MISSING`
/// table is the direct template for `KNOWN_UNMAPPED_GROUPS` below).
fn vendored_iop_names() -> Vec<&'static str> {
    let header = include_str!("../../vendor/pyvex_ffi.h");
    let mut found: Vec<&str> = Vec::new();
    let mut rest = header;
    while let Some(pos) = rest.find("Iop_") {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(tail.len());
        found.push(&tail[..end]);
        rest = &tail[end..];
    }
    // Iop_INVALID / Iop_LAST are enum sentinels, not real opcodes pyvex can
    // ever hand to `parse_opcode`.
    found.retain(|s| *s != "Iop_INVALID" && *s != "Iop_LAST");
    found.sort_unstable();
    found.dedup();
    found
}

/// pyvex opcodes with no entry in `opcode_map.rs`'s `parse_*` dispatch tables —
/// each falls through `parse_opcode` to `IROp::Unmapped`, which dispatch turns
/// into `OpError::UnsupportedVexOp` and which the live-exploration path routes
/// to the (slower, but still correct) Python engine. Not a wrong-answer bug —
/// a silently-widening performance cliff, the shape behind angr-9ke6b.160,
/// angr-sqfj8.113 and angr-sqfj8.117 ("op-coverage-gap"): a real gap with no
/// error to notice it by. Grouped the same way `KNOWN_MISSING` groups register
/// names in `tests/engines/rust/test_arch_offset_parity.py` — one reason per
/// mnemonic family rather than per opcode, since libVEX enumerates every
/// width/signedness/lane-count combination of a family as its own tag.
///
/// Self-cleaning: `test_known_unmapped_groups_are_still_entirely_unmapped`
/// asserts every name here is still both a real vendored opcode and still
/// unmapped, so implementing one reddens that test until the name is removed —
/// it cannot rot into a permanent allowlist.
const KNOWN_UNMAPPED_GROUPS: &[(&[&str], &str)] = &[
    // DECIMAL_FLOAT
    (
        &[
            "Iop_AddD128",
            "Iop_AddD64",
            "Iop_CmpD128",
            "Iop_CmpD64",
            "Iop_CmpExpD128",
            "Iop_CmpExpD64",
            "Iop_D128HItoD64",
            "Iop_D128LOtoD64",
            "Iop_D128toD64",
            "Iop_D128toF128",
            "Iop_D128toF32",
            "Iop_D128toF64",
            "Iop_D128toI32S",
            "Iop_D128toI32U",
            "Iop_D128toI64S",
            "Iop_D128toI64U",
            "Iop_D32toD64",
            "Iop_D32toF128",
            "Iop_D32toF32",
            "Iop_D32toF64",
            "Iop_D64HLtoD128",
            "Iop_D64toD128",
            "Iop_D64toD32",
            "Iop_D64toF128",
            "Iop_D64toF32",
            "Iop_D64toF64",
            "Iop_D64toI32S",
            "Iop_D64toI32U",
            "Iop_D64toI64S",
            "Iop_D64toI64U",
            "Iop_DivD128",
            "Iop_DivD64",
            "Iop_ExtractExpD128",
            "Iop_ExtractExpD64",
            "Iop_ExtractSigD128",
            "Iop_ExtractSigD64",
            "Iop_F128toD128",
            "Iop_F128toD32",
            "Iop_F128toD64",
            "Iop_F32toD128",
            "Iop_F32toD32",
            "Iop_F32toD64",
            "Iop_F64toD128",
            "Iop_F64toD32",
            "Iop_F64toD64",
            "Iop_I32StoD128",
            "Iop_I32StoD64",
            "Iop_I32UtoD128",
            "Iop_I32UtoD64",
            "Iop_I64StoD128",
            "Iop_I64StoD64",
            "Iop_I64UtoD128",
            "Iop_I64UtoD64",
            "Iop_InsertExpD128",
            "Iop_InsertExpD64",
            "Iop_MulD128",
            "Iop_MulD64",
            "Iop_QuantizeD128",
            "Iop_QuantizeD64",
            "Iop_ReinterpD64asI64",
            "Iop_ReinterpI64asD64",
            "Iop_RoundD128toInt",
            "Iop_RoundD64toInt",
            "Iop_ShlD128",
            "Iop_ShlD64",
            "Iop_ShrD128",
            "Iop_ShrD64",
            "Iop_SignificanceRoundD128",
            "Iop_SignificanceRoundD64",
            "Iop_SubD128",
            "Iop_SubD64",
        ],
        "IEEE 754-2008 decimal floating point (D32/D64/D128) — only PowerPC and s390x lift this, both unsupported architectures (docs/advanced-topics/rust_engine.rst); no IRType/evaluator support exists for the DFP encoding.",
    ),
    // BCD
    (
        &[
            "Iop_BCD128toI128S",
            "Iop_BCDAdd",
            "Iop_BCDSub",
            "Iop_BCDtoDPB",
            "Iop_DPBtoBCD",
            "Iop_I128StoBCD128",
        ],
        "Binary-coded-decimal <-> densely-packed-decimal conversion and BCD arithmetic — PowerPC decimal128 support, unsupported architecture.",
    ),
    // QUAD_FLOAT
    (
        &[
            "Iop_AbsF128",
            "Iop_AddF128",
            "Iop_CmpF128",
            "Iop_DivF128",
            "Iop_F128HItoF64",
            "Iop_F128LOtoF64",
            "Iop_F128toF32",
            "Iop_F128toF64",
            "Iop_F128toI128S",
            "Iop_F128toI32S",
            "Iop_F128toI32U",
            "Iop_F128toI64S",
            "Iop_F128toI64U",
            "Iop_F32toF128",
            "Iop_F64HLtoF128",
            "Iop_F64toF128",
            "Iop_I32StoF128",
            "Iop_I32UtoF128",
            "Iop_I64StoF128",
            "Iop_I64UtoF128",
            "Iop_MAddF128",
            "Iop_MSubF128",
            "Iop_MulF128",
            "Iop_NegF128",
            "Iop_NegMAddF128",
            "Iop_NegMSubF128",
            "Iop_RndF128",
            "Iop_RoundF128toInt",
            "Iop_SqrtF128",
            "Iop_SubF128",
            "Iop_TruncF128toI32S",
            "Iop_TruncF128toI32U",
            "Iop_TruncF128toI64S",
            "Iop_TruncF128toI64U",
        ],
        "128-bit (\"quad\"/F128) IEEE binary float arithmetic and conversions — PowerPC/s390x extended precision, unsupported architectures; distinct from the V128 128-bit *vector* type, which is mapped elsewhere.",
    ),
    // HALF_FLOAT
    (
        &[
            "Iop_F16toF32",
            "Iop_F16toF32x4",
            "Iop_F16toF64",
            "Iop_F16toF64x2",
            "Iop_F32toF16",
            "Iop_F32toF16x4",
            "Iop_F64toF16",
            "Iop_F64toF16x2",
        ],
        "F16 (half-precision) <-> F32/F64 conversion — ARMv8.2 FP16 extension; IRType::F16 exists for width-tagging but no arithmetic/conversion evaluator is wired up yet.",
    ),
    // PPC_R32_AND_MISC_FP
    (
        &[
            "Iop_AddF64r32",
            "Iop_DivF64r32",
            "Iop_MAddF64r32",
            "Iop_MSubF64r32",
            "Iop_MulF64r32",
            "Iop_SubF64r32",
            "Iop_F64toI16S",
            "Iop_RSqrtEst5GoodF64",
            "Iop_TruncF64asF32",
            "Iop_RoundF64toF32",
            "Iop_RoundF64toF64_NEAREST",
            "Iop_RoundF64toF64_NegINF",
            "Iop_RoundF64toF64_PosINF",
            "Iop_RoundF64toF64_ZERO",
        ],
        "PowerPC \"round to 32-bit precision after the op\" (*F64r32) arithmetic plus adjacent scalar FP helpers (F64toI16S, RSqrtEst5GoodF64, TruncF64asF32, RoundF64toF32, RoundF64toF64_*) — PowerPC-specific FP helper ops, not yet implemented.",
    ),
    // (The ARM32 VMAXNM/VMINNM group that used to sit here — Iop_MaxNumF32/F64,
    // Iop_MinNumF32/F64 — was dropped when angr-0bh1z implemented them as
    // IROp::FMaxNum/FMinNum; test_parse_max_num_min_num pins the mapping.)
    // X87_PREM
    (
        &[
            "Iop_PRem1C3210F64",
            "Iop_PRem1F64",
            "Iop_PRemC3210F64",
            "Iop_PRemF64",
        ],
        "x87 FPREM/FPREM1 partial-remainder ops — deliberately out of scope (see the parse_transcendental doc comment and test_parse_transcendental, which pins Iop_PRemF64 unmapped); no other op in this family is implemented either.",
    ),
    // CMP_ORD
    (
        &[
            "Iop_CmpORD32S",
            "Iop_CmpORD32U",
            "Iop_CmpORD64S",
            "Iop_CmpORD64U",
        ],
        "PowerPC \"ordered compare\" (full-width -1/0 result, not a 1-bit compare) — deliberately unmapped, pinned by test_parse_cmp_ord_stays_unmapped (angr-sqfj8.120).",
    ),
    // CMP_NEZ
    (
        &[
            "Iop_CmpNEZ128x1",
            "Iop_CmpNEZ16",
            "Iop_CmpNEZ16x16",
            "Iop_CmpNEZ16x2",
            "Iop_CmpNEZ16x4",
            "Iop_CmpNEZ16x8",
            "Iop_CmpNEZ32",
            "Iop_CmpNEZ32x2",
            "Iop_CmpNEZ32x4",
            "Iop_CmpNEZ32x8",
            "Iop_CmpNEZ64",
            "Iop_CmpNEZ64x2",
            "Iop_CmpNEZ64x4",
            "Iop_CmpNEZ8",
            "Iop_CmpNEZ8x16",
            "Iop_CmpNEZ8x32",
            "Iop_CmpNEZ8x4",
            "Iop_CmpNEZ8x8",
            "Iop_CmpwNEZ32",
            "Iop_CmpwNEZ64",
        ],
        "\"compare not-equal-zero\" scalar and vector family (the VEX idiom for boolean-from-integer / truthiness checks) — no IROp variant exists yet; not yet implemented.",
    ),
    // PPC_MISC_INT
    (
        &[
            "Iop_Left16",
            "Iop_Left32",
            "Iop_Left64",
            "Iop_Left8",
            "Iop_Max32U",
        ],
        "PowerPC-specific integer helpers: Left{8,16,32,64} (isolate/replicate leftmost set bit, backs POWER's cntlz-adjacent idioms) and Max32U (unsigned max, used in POWER carry generation) — not yet implemented.",
    ),
    // EXTENDED_DIVIDE
    (
        &[
            "Iop_DivModS64to64",
            "Iop_DivS32E",
            "Iop_DivS64E",
            "Iop_DivU32E",
            "Iop_DivU64E",
        ],
        "PowerPC \"extended\" divide (DivS32E/DivS64E/DivU32E/DivU64E, different rounding/overflow semantics than the plain DivS/DivU already mapped) and DivModS64to64 (64/64 combined divmod) — not yet implemented.",
    ),
    // CRYPTO
    (
        &[
            "Iop_CipherLV128",
            "Iop_CipherSV128",
            "Iop_CipherV128",
            "Iop_NCipherLV128",
            "Iop_NCipherV128",
            "Iop_SHA256",
            "Iop_SHA512",
        ],
        "PowerPC/POWER8 crypto extension AES (Cipher*V128/NCipher*V128 — vcipher/vncipher) and SHA (SHA256/SHA512 — vshasigma) block ops, emitted by guest_ppc_toIR.c — NOT ARMv8; ARM64's own crypto extension (AESE/AESD/SHA1H/SHA256H) lowers through dirty-helper calls, never through parse_opcode. PowerPC is an unsupported architecture (docs/advanced-topics/rust_engine.rst). Not yet implemented.",
    ),
    // MULI128BY10
    (
        &[
            "Iop_MulI128by10",
            "Iop_MulI128by10Carry",
            "Iop_MulI128by10E",
            "Iop_MulI128by10ECarry",
        ],
        "128-bit multiply-by-10 with carry-out — PowerPC decimal128 helper family (pairs with the BCD group above) — not yet implemented.",
    ),
    // ARM_HALFSIMD_GPR
    (
        &[
            "Iop_Add16x2",
            "Iop_Add8x4",
            "Iop_Sub16x2",
            "Iop_Sub8x4",
            "Iop_HAdd16Sx2",
            "Iop_HAdd16Ux2",
            "Iop_HAdd8Sx4",
            "Iop_HAdd8Ux4",
            "Iop_HSub16Sx2",
            "Iop_HSub16Ux2",
            "Iop_HSub8Sx4",
            "Iop_HSub8Ux4",
            "Iop_QAdd16Sx2",
            "Iop_QAdd16Ux2",
            "Iop_QAdd8Sx4",
            "Iop_QAdd8Ux4",
            "Iop_QSub16Sx2",
            "Iop_QSub16Ux2",
            "Iop_QSub8Sx4",
            "Iop_QSub8Ux4",
            "Iop_Sad8Ux4",
            "Iop_QAdd32S",
            "Iop_QSub32S",
        ],
        "ARMv6 \"half SIMD\" packed-integer arithmetic (2x16 or 4x8 lanes packed into a single 32-bit GPR, e.g. UADD16/SADD16/UHADD16/USAD8) — a pre-NEON packed-integer extension distinct from the D/Q-register vector families already mapped; not yet implemented.",
    ),
    // FIXED_POINT_CONVERT
    (
        &[
            "Iop_F32ToFixed32Sx2_RZ",
            "Iop_F32ToFixed32Sx4_RZ",
            "Iop_F32ToFixed32Ux2_RZ",
            "Iop_F32ToFixed32Ux4_RZ",
            "Iop_Fixed32SToF32x2_RN",
            "Iop_Fixed32SToF32x4_RN",
            "Iop_Fixed32UToF32x2_RN",
            "Iop_Fixed32UToF32x4_RN",
            "Iop_FtoI32Sx2_RZ",
            "Iop_FtoI32Sx4_RZ",
            "Iop_FtoI32Ux2_RZ",
            "Iop_FtoI32Ux4_RZ",
            "Iop_I32StoFx2",
            "Iop_I32StoFx4",
            "Iop_I32UtoFx2",
            "Iop_I32UtoFx4",
            "Iop_QFtoI32Sx4_RZ",
            "Iop_QFtoI32Ux4_RZ",
        ],
        "ARM NEON fixed-point <-> float conversion family (VCVT with an embedded fractional-bits immediate, plus the plain float<->int Fx2/Fx4 D/Q-reg forms) — not yet implemented.",
    ),
    // NEON_MULHI
    (
        &[
            "Iop_MulHi16Sx16",
            "Iop_MulHi16Sx4",
            "Iop_MulHi16Sx8",
            "Iop_MulHi16Ux16",
            "Iop_MulHi16Ux4",
            "Iop_MulHi16Ux8",
            "Iop_MulHi32Sx4",
            "Iop_MulHi32Ux4",
            "Iop_MulHi8Sx16",
            "Iop_MulHi8Ux16",
            "Iop_QDMulHi16Sx4",
            "Iop_QDMulHi16Sx8",
            "Iop_QDMulHi32Sx2",
            "Iop_QDMulHi32Sx4",
            "Iop_QRDMulHi16Sx4",
            "Iop_QRDMulHi16Sx8",
            "Iop_QRDMulHi32Sx2",
            "Iop_QRDMulHi32Sx4",
        ],
        "NEON/AVX2 \"high half of widening multiply\" (MulHi) and the ARM doubling variants (QDMulHi/QRDMulHi) — not yet implemented.",
    ),
    // AVX2_MUL
    (
        &["Iop_Mul16x16", "Iop_Mul32x8"],
        "AVX2 256-bit packed integer multiply (16x16/32x8) — the 128-bit VMul family is mapped, this width tier is not yet implemented.",
    ),
    // LANE_SHUFFLE_ODD_EVEN
    (
        &[
            "Iop_CatEvenLanes16x4",
            "Iop_CatEvenLanes16x8",
            "Iop_CatEvenLanes32x4",
            "Iop_CatEvenLanes8x16",
            "Iop_CatEvenLanes8x8",
            "Iop_CatOddLanes16x4",
            "Iop_CatOddLanes16x8",
            "Iop_CatOddLanes32x4",
            "Iop_CatOddLanes8x16",
            "Iop_CatOddLanes8x8",
            "Iop_InterleaveEvenLanes16x4",
            "Iop_InterleaveEvenLanes16x8",
            "Iop_InterleaveEvenLanes32x4",
            "Iop_InterleaveEvenLanes8x16",
            "Iop_InterleaveEvenLanes8x8",
            "Iop_InterleaveOddLanes16x4",
            "Iop_InterleaveOddLanes16x8",
            "Iop_InterleaveOddLanes32x4",
            "Iop_InterleaveOddLanes8x16",
            "Iop_InterleaveOddLanes8x8",
        ],
        "Odd/even-lane deinterleave (CatOddLanes/CatEvenLanes) and interleave (InterleaveOddLanes/InterleaveEvenLanes) vector shuffles — siblings of the already-mapped InterleaveHI/InterleaveLO family; not yet implemented.",
    ),
    // CLZ_CTZ_WIDE
    (
        &[
            "Iop_Clz64x2",
            "Iop_Ctz16x8",
            "Iop_Ctz32x4",
            "Iop_Ctz64x2",
            "Iop_Ctz8x16",
        ],
        "NEON Q-reg count-leading/trailing-zeros at widths beyond what parse_vector's VClz table covers (64x2) plus the whole Ctz{8,16,32,64} vector family — not yet implemented.",
    ),
    // AVG_WIDE
    (
        &[
            "Iop_Avg16Ux16",
            "Iop_Avg64Sx2",
            "Iop_Avg64Ux2",
            "Iop_Avg8Ux32",
        ],
        "NEON/AVX2 rounding halving-add (VAvg family) at widths beyond what parse_vector covers (64-bit lanes, AVX2 256-bit 16-lane) — not yet implemented.",
    ),
    // PWADDL_WIDE
    (
        &["Iop_PwAddL64Ux2"],
        "NEON pairwise widening add (VPwAddL family) — the 64-bit-lane Q-reg width is the one shape parse_vector's table does not cover; not yet implemented.",
    ),
    // QADD_QSUB_AVX2
    (
        &[
            "Iop_QAdd16Sx16",
            "Iop_QAdd16Ux16",
            "Iop_QAdd8Sx32",
            "Iop_QAdd8Ux32",
            "Iop_QSub16Sx16",
            "Iop_QSub16Ux16",
            "Iop_QSub8Sx32",
            "Iop_QSub8Ux32",
        ],
        "AVX2 256-bit saturating add/sub (VQAdd/VQSub family) — the 128-bit and D-reg forms are mapped, this width tier is not yet implemented.",
    ),
    // ROL
    (
        &["Iop_Rol16x8", "Iop_Rol32x4", "Iop_Rol64x2", "Iop_Rol8x16"],
        "NEON Q-reg vector rotate-left (distinct from the Shl/Shr/Sar-by-vector family already mapped) — not yet implemented.",
    ),
    // ROUND_F32X4
    (
        &[
            "Iop_RoundF32x4_RM",
            "Iop_RoundF32x4_RN",
            "Iop_RoundF32x4_RP",
            "Iop_RoundF32x4_RZ",
        ],
        "SSE4.1 ROUNDPS-style packed-float round-to-integer with an explicit rounding-mode suffix (RM/RN/RP/RZ) — not yet implemented.",
    ),
    // SH_RSH_BIDIRECTIONAL
    (
        &[
            "Iop_Rsh16Sx8",
            "Iop_Rsh16Ux8",
            "Iop_Rsh32Sx4",
            "Iop_Rsh32Ux4",
            "Iop_Rsh64Sx2",
            "Iop_Rsh64Ux2",
            "Iop_Rsh8Sx16",
            "Iop_Rsh8Ux16",
            "Iop_Sh16Sx8",
            "Iop_Sh16Ux8",
            "Iop_Sh32Sx4",
            "Iop_Sh32Ux4",
            "Iop_Sh64Sx2",
            "Iop_Sh64Ux2",
            "Iop_Sh8Sx16",
            "Iop_Sh8Ux16",
        ],
        "AArch64 NEON bidirectional shift-by-vector (SSHL/USHL-style, sign of the shift-amount lane selects direction) and its rounding variant (SRSHL/URSHL) — distinct from the ShlN/ShrN-by-immediate and Shl/Shr/Sar-by-vector families already mapped; not yet implemented.",
    ),
    // QSHLNSAT
    (
        &[
            "Iop_QShlNsatSS16x4",
            "Iop_QShlNsatSS16x8",
            "Iop_QShlNsatSS32x2",
            "Iop_QShlNsatSS32x4",
            "Iop_QShlNsatSS64x1",
            "Iop_QShlNsatSS64x2",
            "Iop_QShlNsatSS8x16",
            "Iop_QShlNsatSS8x8",
            "Iop_QShlNsatSU16x4",
            "Iop_QShlNsatSU16x8",
            "Iop_QShlNsatSU32x2",
            "Iop_QShlNsatSU32x4",
            "Iop_QShlNsatSU64x1",
            "Iop_QShlNsatSU64x2",
            "Iop_QShlNsatSU8x16",
            "Iop_QShlNsatSU8x8",
            "Iop_QShlNsatUU16x4",
            "Iop_QShlNsatUU16x8",
            "Iop_QShlNsatUU32x2",
            "Iop_QShlNsatUU32x4",
            "Iop_QShlNsatUU64x1",
            "Iop_QShlNsatUU64x2",
            "Iop_QShlNsatUU8x16",
            "Iop_QShlNsatUU8x8",
        ],
        "NEON saturating shift-left-by-immediate with mixed source/dest signedness (QShlNsat{SS,SU,UU}) — parse_vector maps the same-signedness QShl/QSal-by-vector family; the by-immediate mixed-signedness forms are not yet implemented.",
    ),
    // QAND_SHIFT_CARRY
    (
        &[
            "Iop_QandQRSarNnarrow16Sto8Sx8",
            "Iop_QandQRSarNnarrow16Sto8Ux8",
            "Iop_QandQRSarNnarrow32Sto16Sx4",
            "Iop_QandQRSarNnarrow32Sto16Ux4",
            "Iop_QandQRSarNnarrow64Sto32Sx2",
            "Iop_QandQRSarNnarrow64Sto32Ux2",
            "Iop_QandQRShrNnarrow16Uto8Ux8",
            "Iop_QandQRShrNnarrow32Uto16Ux4",
            "Iop_QandQRShrNnarrow64Uto32Ux2",
            "Iop_QandQSarNnarrow16Sto8Sx8",
            "Iop_QandQSarNnarrow16Sto8Ux8",
            "Iop_QandQSarNnarrow32Sto16Sx4",
            "Iop_QandQSarNnarrow32Sto16Ux4",
            "Iop_QandQSarNnarrow64Sto32Sx2",
            "Iop_QandQSarNnarrow64Sto32Ux2",
            "Iop_QandQShrNnarrow16Uto8Ux8",
            "Iop_QandQShrNnarrow32Uto16Ux4",
            "Iop_QandQShrNnarrow64Uto32Ux2",
            "Iop_QandSQRsh16x8",
            "Iop_QandSQRsh32x4",
            "Iop_QandSQRsh64x2",
            "Iop_QandSQRsh8x16",
            "Iop_QandSQsh16x8",
            "Iop_QandSQsh32x4",
            "Iop_QandSQsh64x2",
            "Iop_QandSQsh8x16",
            "Iop_QandUQRsh16x8",
            "Iop_QandUQRsh32x4",
            "Iop_QandUQRsh64x2",
            "Iop_QandUQRsh8x16",
            "Iop_QandUQsh16x8",
            "Iop_QandUQsh32x4",
            "Iop_QandUQsh64x2",
            "Iop_QandUQsh8x16",
        ],
        "ARMv8 saturating-shift-and-narrow families that also report a carry/overflow flag (QandQ{R}{Sar,Shr}Nnarrow*, QandS/UQ{R}sh*) — not yet implemented.",
    ),
    // QADDEXT
    (
        &[
            "Iop_QAddExtSUsatUU16x8",
            "Iop_QAddExtSUsatUU32x4",
            "Iop_QAddExtSUsatUU64x2",
            "Iop_QAddExtSUsatUU8x16",
            "Iop_QAddExtUSsatSS16x8",
            "Iop_QAddExtUSsatSS32x4",
            "Iop_QAddExtUSsatSS64x2",
            "Iop_QAddExtUSsatSS8x16",
        ],
        "ARMv8.2 saturating \"extended\" add with mixed signed/unsigned inputs (SUQADD/USQADD-style) — not yet implemented.",
    ),
    // POLYNOMIAL_MULADD
    (
        &[
            "Iop_PolynomialMulAdd16x8",
            "Iop_PolynomialMulAdd32x4",
            "Iop_PolynomialMulAdd64x2",
            "Iop_PolynomialMulAdd8x16",
        ],
        "ARMv8 GF(2) polynomial multiply-accumulate (PMULL-adjacent, used by CRC/AES-GCM software paths) — the plain PolynomialMul family is mapped, the fused multiply-add form is not yet implemented.",
    ),
    // PERM_TRIOP
    (
        &["Iop_Perm8x16x2"],
        "the two-table AArch64 TBL form of Iop_Perm — deliberately unmapped because it is a triop and the whole VPerm family exists only to route to Python's two-argument _op_generic_Perm (see the parse_special doc comment, angr-sqfj8.117).",
    ),
    // V256_LANE_ACCESS
    (
        &[
            "Iop_64x4toV256",
            "Iop_V128HLtoV256",
            "Iop_V256to64_0",
            "Iop_V256to64_1",
            "Iop_V256to64_2",
            "Iop_V256to64_3",
            "Iop_V256toV128_0",
            "Iop_V256toV128_1",
        ],
        "AVX 256-bit lane extraction/construction (V256<->V128/I64 slicing, 64x4toV256) — not yet implemented.",
    ),
    // V128_MISC
    (
        &[
            "Iop_Add128x1",
            "Iop_Sub128x1",
            "Iop_SarV128",
            "Iop_ShlV128",
            "Iop_ShrV128",
            "Iop_Slice64",
            "Iop_SliceV128",
            "Iop_V128to32",
            "Iop_ZeroHI112ofV128",
            "Iop_ZeroHI120ofV128",
            "Iop_ZeroHI64ofV128",
            "Iop_ZeroHI96ofV128",
        ],
        "assorted V128 lane/bit-width helpers (128-bit-wide single-lane Add/Sub, ShlV128/ShrV128/SarV128 whole-register shift, Slice64/SliceV128, V128to32 narrowing, ZeroHI*ofV128 upper-bits clear) — not yet implemented.",
    ),
];

/// The completeness gate: every vendored `Iop_*` name must be mapped by
/// `parse_opcode`, or explicitly named (with a reason) in
/// `KNOWN_UNMAPPED_GROUPS`. Fails loudly, listing every offending name, so a
/// future VEX-pin bump that adds opcodes cannot silently grow the gap.
#[test]
fn test_opcode_map_completeness_against_vendored_header() {
    let known: std::collections::HashSet<&str> = KNOWN_UNMAPPED_GROUPS
        .iter()
        .flat_map(|(names, _)| names.iter().copied())
        .collect();

    let mut newly_unmapped: Vec<&str> = vendored_iop_names()
        .into_iter()
        .filter(|name| matches!(parse_opcode(name), IROp::Unmapped(_)) && !known.contains(name))
        .collect();
    newly_unmapped.sort_unstable();

    assert!(
        newly_unmapped.is_empty(),
        "{} pyvex opcode(s) have no parse_opcode mapping and no \
         KNOWN_UNMAPPED_GROUPS entry: {newly_unmapped:?}. This is the \
         op-coverage-gap shape from angr-9ke6b.160 / angr-sqfj8.113 / \
         angr-sqfj8.117 — either add a parse_* arm mapping the opcode, or \
         add a KNOWN_UNMAPPED_GROUPS entry stating why it stays unmapped.",
        newly_unmapped.len()
    );
}

/// The self-cleaning half of `KNOWN_UNMAPPED_GROUPS`, mirroring
/// `test_known_missing_groups_are_still_entirely_missing` in
/// `test_arch_offset_parity.py`: every named opcode must still be a real
/// vendored name (catches a group rotting after a VEX-pin bump drops an
/// opcode) and must still be unmapped (catches a group that was never
/// cleaned up after someone implemented the opcode it names).
#[test]
fn test_known_unmapped_groups_are_still_entirely_unmapped() {
    let vendored: std::collections::HashSet<&str> = vendored_iop_names().into_iter().collect();
    for (names, reason) in KNOWN_UNMAPPED_GROUPS {
        for &name in *names {
            assert!(
                vendored.contains(name),
                "KNOWN_UNMAPPED_GROUPS entry ({reason}) names {name:?}, which \
                 vendor/pyvex_ffi.h no longer declares — the group has rotted; drop it."
            );
            assert!(
                matches!(parse_opcode(name), IROp::Unmapped(_)),
                "KNOWN_UNMAPPED_GROUPS entry ({reason}) claims {name:?} is \
                 unmapped, but parse_opcode now maps it — drop it from the group \
                 so the completeness check covers it directly."
            );
        }
    }
}

/// angr-sqfj8.143: `Iop_PwBitMtxXpose64x2` (PPC vgbbd) routes to
/// `IROp::VPwBitMtxXpose` and reports a V128 result. It is the one `Pw*`
/// opcode that is not NEON, so it is parsed by an explicit string match
/// rather than a lane-shape table.
#[test]
fn test_parse_pw_bit_mtx_xpose() {
    assert_eq!(parse_opcode("Iop_PwBitMtxXpose64x2"), IROp::VPwBitMtxXpose);
    assert_eq!(
        IROp::VPwBitMtxXpose.result_type(),
        Some(IRType::V128),
        "vgbbd is V128 -> V128"
    );

    // libVEX declares exactly one shape (vendor/pyvex_ffi.h). Pin the absence
    // of the plausible-looking siblings so a future audit does not "restore
    // symmetry" by inventing opcodes that do not exist.
    for op_str in [
        "Iop_PwBitMtxXpose32x4",
        "Iop_PwBitMtxXpose8x16",
        "Iop_PwBitMtxXpose64x4",
        "Iop_PwBitMtxXpose64x1",
    ] {
        assert!(
            matches!(parse_opcode(op_str), IROp::Unmapped(_)),
            "{op_str} is not declared by libVEX and must stay unmapped"
        );
    }
}
