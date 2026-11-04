//! Tests for [`super`] and for `vex::opcode_map`'s coverage, cross-checked
//! against upstream's libVEX op table `ailment::vexop_names::IROP_NAMES`.
//! They live here, not under `vex/`, because `vex/` names nothing outside
//! `vex` and `angr_bv` (angr-mx00p.51.6).

use super::*;

/// Opcode-map completeness check: every real `Iop_*` opcode name the pinned VEX
/// version declares must be either mapped by `parse_opcode` or named in
/// `KNOWN_UNMAPPED_GROUPS` below with a reason. A new pyvex opcode silently
/// falling through to `IROp::Unmapped` with no allowlist entry is the
/// op-coverage-gap shape (angr-9ke6b.160, angr-sqfj8.113, angr-sqfj8.117);
/// this test exists so the next instance is a hard failure instead of a
/// coverage gap nobody notices until a state errors on it.
///
/// Ground truth is upstream's `ailment::vexop_names::IROP_NAMES`, the
/// libVEX `IROp` enum in order, which `ailment::vexop::op_name` (and so the
/// native lifter) decodes op tags through — exactly the set of opcodes the
/// pinned pyvex can emit, not a hand-maintained approximation.
/// `test_native_lift_parity.py` keeps the native lifter's own enum tables in
/// step with pyvex the same way.
///
/// Same "diff against a foreign ground truth + self-cleaning allowlist"
/// shape `tests/engines/rust/test_arch_offset_parity.py` already proved out
/// for arch/register coverage against `archinfo` (its `KNOWN_MISSING`
/// table is the direct template for `KNOWN_UNMAPPED_GROUPS` below).
fn libvex_iop_names() -> Vec<&'static str> {
    // Iop_INVALID / Iop_LAST are enum sentinels, not real opcodes pyvex can
    // ever hand to `parse_opcode`.
    let mut found: Vec<&str> = IROP_NAMES
        .iter()
        .copied()
        .filter(|s| *s != "Iop_INVALID" && *s != "Iop_LAST")
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// pyvex opcodes with no entry in `opcode_map.rs`'s `parse_*` dispatch tables —
/// each falls through `parse_opcode` to `IROp::Unmapped`, which dispatch turns
/// into `OpError::UnsupportedVexOp`; the interpreter surfaces that as a fatal
/// `CbExecutionError::Op`, so a state that reaches the op lands in the errored
/// stash (`RustUnsupportedVexOpError` on the `execute_irsb_for_test` path).
/// Not a wrong-answer bug, but a coverage gap that the tests below keep from
/// growing unnoticed ("op-coverage-gap"). Grouped the same way `KNOWN_MISSING` groups register
/// names in `tests/engines/rust/test_arch_offset_parity.py` — one reason per
/// mnemonic family rather than per opcode, since libVEX enumerates every
/// width/signedness/lane-count combination of a family as its own tag.
///
/// Self-cleaning: `test_known_unmapped_groups_are_still_entirely_unmapped`
/// asserts every name here is still both a real libVEX opcode and still
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
            "Iop_F64toF16x2_DEP",
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
            "Iop_RSqrtEst5GoodF64",
            "Iop_TruncF64asF32",
            "Iop_RoundF64toF32",
            "Iop_RoundF64toF64_NEAREST",
            "Iop_RoundF64toF64_NegINF",
            "Iop_RoundF64toF64_PosINF",
            "Iop_RoundF64toF64_ZERO",
        ],
        "PowerPC \"round to 32-bit precision after the op\" (*F64r32) arithmetic plus adjacent scalar FP helpers (RSqrtEst5GoodF64, TruncF64asF32, RoundF64toF32, RoundF64toF64_*) — PowerPC-specific FP helper ops, not yet implemented.",
    ),
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
            "Iop_DivS32E",
            "Iop_DivS64E",
            "Iop_DivU32E",
            "Iop_DivU64E",
        ],
        "PowerPC \"extended\" divide (DivS32E/DivS64E/DivU32E/DivU64E, different rounding/overflow semantics than the plain DivS/DivU already mapped) — not yet implemented.",
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
            "Iop_QF32toI32Sx4_RZ",
            "Iop_QF32toI32Ux4_RZ",
        ],
        "ARM NEON fixed-point <-> float conversion family (VCVT with an embedded fractional-bits immediate, plus the plain float<->int Fx2/Fx4 D/Q-reg forms) — not yet implemented.",
    ),
    // NEON_MULHI — the plain Iop_MulHi{N}{U,S}x{M} family is mapped to
    // IROp::VMulHi (angr-0jh0j.61); only the ARM doubling variants remain.
    (
        &[
            "Iop_QDMulHi16Sx4",
            "Iop_QDMulHi16Sx8",
            "Iop_QDMulHi32Sx2",
            "Iop_QDMulHi32Sx4",
            "Iop_QRDMulHi16Sx4",
            "Iop_QRDMulHi16Sx8",
            "Iop_QRDMulHi32Sx2",
            "Iop_QRDMulHi32Sx4",
        ],
        "ARM doubling variants of the high-half widening multiply (VQDMULH/VQRDMULH: product doubled, then saturated, with the rounding form adding a half-ULP first) — the plain MulHi family is mapped, these are not yet implemented.",
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
    // CTZ
    (
        &[
            "Iop_Ctz16x8",
            "Iop_Ctz32x4",
            "Iop_Ctz64x2",
            "Iop_Ctz8x16",
        ],
        "NEON Q-reg per-lane count-trailing-zeros — the whole Ctz{8,16,32,64} vector family has no IROp variant; not yet implemented.",
    ),
    // ROL
    (
        &["Iop_Rol16x8", "Iop_Rol32x4", "Iop_Rol64x2", "Iop_Rol8x16"],
        "NEON Q-reg vector rotate-left (distinct from the Shl/Shr/Sar-by-vector family already mapped) — not yet implemented.",
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
        "the two-table AArch64 TBL form of Iop_Perm — deliberately unmapped because it is a triop, while VPerm and Python's _op_generic_Perm are both two-argument (see the parse_special doc comment).",
    ),
    // V128_MISC
    (
        &[
            "Iop_Add128x1",
            "Iop_Sub128x1",
            "Iop_Slice64",
            "Iop_ZeroHI112ofV128",
            "Iop_ZeroHI120ofV128",
            "Iop_ZeroHI64ofV128",
            "Iop_ZeroHI96ofV128",
        ],
        "assorted V128 lane/bit-width helpers (128-bit-wide single-lane Add/Sub, Slice64, ZeroHI*ofV128 upper-bits clear) — not yet implemented.",
    ),
    // PYVEX10_NEW — ops the libVEX bundled with pyvex 10 added that the
    // engine has not mapped yet (128-bit integer divide/modulo, the DivMod
    // forms, F16 arithmetic, I128/V128/F128 reinterprets, MIPS Rotx, ...);
    // a state that reaches one errors until it is mapped (angr-mx00p.6).
    (
        &[
            "Iop_2xMultU64Add128CarryOut",
            "Iop_Abs16Fx8",
            "Iop_AbsF16",
            "Iop_Add16Fx8",
            "Iop_AddF16",
            "Iop_CmpEQ16Fx8",
            "Iop_CmpF16",
            "Iop_CmpLE16Fx8",
            "Iop_CmpLT16Fx8",
            "Iop_D128toI128S",
            "Iop_DivS128E",
            "Iop_DivU128E",
            "Iop_Exp2_32Fx4",
            "Iop_F16toF32x8",
            "Iop_F32toF16x4_DEP",
            "Iop_F32toF16x8",
            "Iop_F32x4_2toQ16x8",
            "Iop_F64x2_2toQ32x4",
            "Iop_I128StoD128",
            "Iop_I128StoF128",
            "Iop_I128UtoF128",
            "Iop_Log2_32Fx4",
            "Iop_Log2_64Fx2",
            "Iop_Neg16Fx8",
            "Iop_NegF16",
            "Iop_PackEvenLanes16x8",
            "Iop_PackEvenLanes32x4",
            "Iop_PackEvenLanes8x16",
            "Iop_PackOddLanes16x8",
            "Iop_PackOddLanes32x4",
            "Iop_PackOddLanes8x16",
            "Iop_PwExtUSMulQAdd8x16",
            "Iop_ReinterpF128asI128",
            "Iop_ReinterpI128asF128",
            "Iop_Rotx32",
            "Iop_Rotx64",
            "Iop_RoundF32toIntA0",
            "Iop_RoundF32toIntE",
            "Iop_RoundF64toIntA0",
            "Iop_RoundF64toIntE",
            "Iop_Scale2_32Fx4",
            "Iop_Scale2_64Fx2",
            "Iop_Sqrt16Fx8",
            "Iop_SqrtF16",
            "Iop_Sub16Fx8",
            "Iop_SubF16",
            "Iop_TruncF128toI128S",
            "Iop_TruncF128toI128U",
        ],
        "added by the libVEX bundled with pyvex 10 and not yet mapped.",
    ),
    // AVX512_EVEX — every op past libVEX's Iop_FIRST_EVEX marker. Only
    // EVEX-encoded code lifts to them. The V512 whole-register ops are
    // mapped (angr-mx00p.125.7); these lane ops are not (angr-mx00p.5).
    (
        &[
            "Iop_16Sto8x8",
            "Iop_CfD32x16",
            "Iop_CfD32x4",
            "Iop_CfD32x8",
            "Iop_CfD64x2",
            "Iop_CfD64x4",
            "Iop_CfD64x8",
            "Iop_Classify32F0x4",
            "Iop_Classify32x16",
            "Iop_Classify32x4",
            "Iop_Classify32x8",
            "Iop_Classify64F0x2",
            "Iop_Classify64x2",
            "Iop_Classify64x4",
            "Iop_Classify64x8",
            "Iop_Cmp16Sx16",
            "Iop_Cmp16Sx32",
            "Iop_Cmp16Sx8",
            "Iop_Cmp16Ux16",
            "Iop_Cmp16Ux32",
            "Iop_Cmp16Ux8",
            "Iop_Cmp32Fx16",
            "Iop_Cmp32Fx4",
            "Iop_Cmp32Fx8",
            "Iop_Cmp32Sx16",
            "Iop_Cmp32Sx4",
            "Iop_Cmp32Sx8",
            "Iop_Cmp32Ux16",
            "Iop_Cmp32Ux4",
            "Iop_Cmp32Ux8",
            "Iop_Cmp64Fx2",
            "Iop_Cmp64Fx4",
            "Iop_Cmp64Fx8",
            "Iop_Cmp64Sx2",
            "Iop_Cmp64Sx4",
            "Iop_Cmp64Sx8",
            "Iop_Cmp64Ux2",
            "Iop_Cmp64Ux4",
            "Iop_Cmp64Ux8",
            "Iop_Cmp8Sx16",
            "Iop_Cmp8Sx32",
            "Iop_Cmp8Sx64",
            "Iop_Cmp8Ux16",
            "Iop_Cmp8Ux32",
            "Iop_Cmp8Ux64",
            "Iop_CmpNEZ32x16",
            "Iop_CmpNEZ64x8",
            "Iop_Compress16x8",
            "Iop_Compress32x16",
            "Iop_Compress32x4",
            "Iop_Compress32x8",
            "Iop_Compress64x2",
            "Iop_Compress64x4",
            "Iop_Compress64x8",
            "Iop_Compress8x16",
            "Iop_DPBUSD",
            "Iop_DPBUSDS",
            "Iop_DPWSSD",
            "Iop_DPWSSDS",
            "Iop_Exp32x16",
            "Iop_Exp64x8",
            "Iop_Expand32x16",
            "Iop_Expand32x4",
            "Iop_Expand32x8",
            "Iop_Expand64x2",
            "Iop_Expand64x4",
            "Iop_Expand64x8",
            "Iop_ExtractExp32F0x4",
            "Iop_ExtractExp32x16",
            "Iop_ExtractExp64F0x2",
            "Iop_ExtractExp64x8",
            "Iop_F32toI32U_SKX",
            "Iop_F64toI32U_SKX",
            "Iop_F64toI64U_SKX",
            "Iop_FixupImm32F0x4",
            "Iop_FixupImm32x16",
            "Iop_FixupImm64F0x2",
            "Iop_FixupImm64x8",
            "Iop_GetMant32F0x4",
            "Iop_GetMant32x16",
            "Iop_GetMant64F0x2",
            "Iop_GetMant64x8",
            "Iop_Gfni",
            "Iop_GfniInv",
            "Iop_GfniMult",
            "Iop_I32UtoF32_SKX",
            "Iop_I64UtoF64_SKX",
            "Iop_MULTISHIFTQB",
            "Iop_Max64Sx8",
            "Iop_Max64Ux8",
            "Iop_Min64Sx8",
            "Iop_Min64Ux8",
            "Iop_PMADD52HUQ",
            "Iop_PMADD52LUQ",
            "Iop_Perm16x16",
            "Iop_Perm16x32",
            "Iop_Perm16x8",
            "Iop_Perm32x16",
            "Iop_Perm64x4",
            "Iop_Perm64x8",
            "Iop_Perm8x32",
            "Iop_Perm8x64",
            "Iop_PermI16x16",
            "Iop_PermI16x32",
            "Iop_PermI16x8",
            "Iop_PermI32x16",
            "Iop_PermI32x4",
            "Iop_PermI32x8",
            "Iop_PermI64x2",
            "Iop_PermI64x4",
            "Iop_PermI64x8",
            "Iop_PermI8x16",
            "Iop_PermI8x32",
            "Iop_PermI8x64",
            "Iop_PopCount8x16",
            "Iop_PrintI64",
            "Iop_RSqrt14_32F0x4",
            "Iop_RSqrt14_32x16",
            "Iop_RSqrt14_64F0x2",
            "Iop_RSqrt14_64x8",
            "Iop_RSqrt28_32F0x4",
            "Iop_RSqrt28_32x16",
            "Iop_RSqrt28_64F0x2",
            "Iop_RSqrt28_64x8",
            "Iop_Range32F0x4",
            "Iop_Range32x4",
            "Iop_Range64F0x2",
            "Iop_Range64x2",
            "Iop_Recip14_32F0x4",
            "Iop_Recip14_32x16",
            "Iop_Recip14_64F0x2",
            "Iop_Recip14_64x8",
            "Iop_Recip28_32F0x4",
            "Iop_Recip28_32x16",
            "Iop_Recip28_64F0x2",
            "Iop_Recip28_64x8",
            "Iop_Reduce32F0x4",
            "Iop_Reduce32x4",
            "Iop_Reduce64F0x2",
            "Iop_Reduce64x2",
            "Iop_RoundScale32F0x4",
            "Iop_RoundScale32x16",
            "Iop_RoundScale64F0x2",
            "Iop_RoundScale64x8",
            "Iop_Scale32F0x4",
            "Iop_Scale32x16",
            "Iop_Scale64F0x2",
            "Iop_Scale64x8",
            "Iop_Ternlog32x16",
            "Iop_Ternlog64x8",
            "Iop_Test16x16",
            "Iop_Test16x32",
            "Iop_Test16x8",
            "Iop_Test32x16",
            "Iop_Test32x4",
            "Iop_Test32x8",
            "Iop_Test64x2",
            "Iop_Test64x4",
            "Iop_Test64x8",
            "Iop_Test8x16",
            "Iop_Test8x32",
            "Iop_Test8x64",
            "Iop_TestN16x16",
            "Iop_TestN16x32",
            "Iop_TestN16x8",
            "Iop_TestN32x16",
            "Iop_TestN32x4",
            "Iop_TestN32x8",
            "Iop_TestN64x2",
            "Iop_TestN64x4",
            "Iop_TestN64x8",
            "Iop_TestN8x16",
            "Iop_TestN8x32",
            "Iop_TestN8x64",
            "Iop_VDBPSADBW",
            "Iop_VPSHLDVD_128",
            "Iop_VPSHLDVQ_128",
            "Iop_VPSHLDW_128",
            "Iop_VPSHRDD_128",
            "Iop_VPSHRDQ_128",
            "Iop_VPSHRDVD_128",
            "Iop_VPSHRDVQ_128",
            "Iop_VPSHRDVW_128",
            "Iop_VPSHRDW_128",
            "Iop_VPSHUFBITQMB_128",
            "Iop_VPSHUFBITQMB_256",
            "Iop_VPSHUFBITQMB_512",
        ],
        "AVX-512 (EVEX) lane ops: only EVEX-encoded code reaches them, and none is mapped.",
    ),
    // Range markers inside the IROp enum, not operations.
    (
        &[
            "Iop_FIRST_EVEX",
        ],
        "libVEX range marker (Iop_FIRST_EVEX), not an operation.",
    ),
];

/// The completeness gate: every libVEX `Iop_*` name must be mapped by
/// `parse_opcode`, or explicitly named (with a reason) in
/// `KNOWN_UNMAPPED_GROUPS`. Fails loudly, listing every offending name, so a
/// future VEX-pin bump that adds opcodes cannot silently grow the gap.
#[test]
fn test_opcode_map_completeness_against_libvex_opcodes() {
    let known: std::collections::HashSet<&str> = KNOWN_UNMAPPED_GROUPS
        .iter()
        .flat_map(|(names, _)| names.iter().copied())
        .collect();

    let mut newly_unmapped: Vec<&str> = libvex_iop_names()
        .into_iter()
        .filter(|name| matches!(parse_opcode(name), IROp::Unmapped(_)) && !known.contains(name))
        .collect();
    newly_unmapped.sort_unstable();

    assert!(
        newly_unmapped.is_empty(),
        "{} pyvex opcode(s) have no parse_opcode mapping and no \
         KNOWN_UNMAPPED_GROUPS entry: {newly_unmapped:?}. This is the \
         op-coverage-gap bug shape (an opcode silently lifting to Unmapped) \
         — either add a parse_* arm mapping the opcode, or \
         add a KNOWN_UNMAPPED_GROUPS entry stating why it stays unmapped.",
        newly_unmapped.len()
    );
}

/// The self-cleaning half of `KNOWN_UNMAPPED_GROUPS`, mirroring
/// `test_known_missing_groups_are_still_entirely_missing` in
/// `test_arch_offset_parity.py`: every named opcode must still be a real
/// libVEX name (catches a group rotting after a VEX-pin bump drops an
/// opcode) and must still be unmapped (catches a group that was never
/// cleaned up after someone implemented the opcode it names).
#[test]
fn test_known_unmapped_groups_are_still_entirely_unmapped() {
    let libvex: std::collections::HashSet<&str> = libvex_iop_names().into_iter().collect();
    for (names, reason) in KNOWN_UNMAPPED_GROUPS {
        for &name in *names {
            assert!(
                libvex.contains(name),
                "KNOWN_UNMAPPED_GROUPS entry ({reason}) names {name:?}, which \
                 IROP_NAMES no longer lists — the group has rotted; drop it."
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

/// `is_int_lane_vec` against the libVEX names: a `Vec` op is integer exactly
/// when its name has no float lane (`32Fx4`, `64F0x2`, `16Fx8`, or the
/// conversion spelling `F32x4` of `RoundF32x4_RM`) and is not a
/// reciprocal or square-root estimate or step, which `irop.py` builds from
/// fresh symbols. `vex_irop`'s `SERVED` admits a `Vec` op on this predicate.
#[test]
fn test_is_int_lane_vec_matches_the_libvex_names() {
    let float_lane = |name: &str| {
        let b = name.as_bytes();
        b.windows(3)
            .any(|w| w[0].is_ascii_digit() && w[1] == b'F' && (w[2] == b'x' || w[2] == b'0'))
            || name.contains("F32x")
            || name.contains("F64x")
    };
    let estimate = |name: &str| name.contains("Est") || name.contains("Step");
    let mut integer = 0;
    for &name in IROP_NAMES {
        let op = parse_opcode(name);
        if iropclass(&op) != VexOpFamily::Vec {
            assert!(!is_int_lane_vec(&op), "{name} ({op:?}) is not a Vec op");
            continue;
        }
        let expected = !float_lane(name) && !estimate(name);
        assert_eq!(is_int_lane_vec(&op), expected, "{name} ({op:?})");
        integer += usize::from(expected);
    }
    assert!(integer > 0);
}
