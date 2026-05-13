//! Mapping from pyvex opcode strings to Rust IROp enum.
//!
//! pyvex uses string opcodes like "Iop_Add32" while Rust uses parameterized
//! operations like `IROp::Add(IRType::I32)`. This module provides the translation.

use super::ir::{FCmpKind, IROp, IRType};

/// Convert a pyvex operation string to Rust IROp.
///
/// Returns `IROp::Raw(0)` with a log warning for unmapped operations.
pub fn parse_opcode(op_str: &str) -> IROp {
    // Fast path for common operations
    if let Some(op) = parse_arithmetic(op_str) {
        return op;
    }
    if let Some(op) = parse_bitwise(op_str) {
        return op;
    }
    if let Some(op) = parse_shift(op_str) {
        return op;
    }
    if let Some(op) = parse_comparison(op_str) {
        return op;
    }
    if let Some(op) = parse_conversion(op_str) {
        return op;
    }
    if let Some(op) = parse_float(op_str) {
        return op;
    }
    if let Some(op) = parse_vector(op_str) {
        return op;
    }
    if let Some(op) = parse_special(op_str) {
        return op;
    }
    if let Some(op) = parse_neon_unimplemented(op_str) {
        return op;
    }

    // Unmapped operation
    log::warn!("Unmapped VEX operation: {}", op_str);
    IROp::Raw(0)
}

/// Parse arithmetic operations: Add, Sub, Mul, Div, Mod, Neg, Mull
fn parse_arithmetic(op_str: &str) -> Option<IROp> {
    // Match patterns like Iop_Add32, Iop_Sub64, etc.
    match op_str {
        // Add
        "Iop_Add8" => Some(IROp::Add(IRType::I8)),
        "Iop_Add16" => Some(IROp::Add(IRType::I16)),
        "Iop_Add32" => Some(IROp::Add(IRType::I32)),
        "Iop_Add64" => Some(IROp::Add(IRType::I64)),

        // Sub
        "Iop_Sub8" => Some(IROp::Sub(IRType::I8)),
        "Iop_Sub16" => Some(IROp::Sub(IRType::I16)),
        "Iop_Sub32" => Some(IROp::Sub(IRType::I32)),
        "Iop_Sub64" => Some(IROp::Sub(IRType::I64)),

        // Mul (low half)
        "Iop_Mul8" => Some(IROp::Mul(IRType::I8)),
        "Iop_Mul16" => Some(IROp::Mul(IRType::I16)),
        "Iop_Mul32" => Some(IROp::Mul(IRType::I32)),
        "Iop_Mul64" => Some(IROp::Mul(IRType::I64)),

        // MullS (signed widening multiply)
        "Iop_MullS8" => Some(IROp::MullS(IRType::I8)),
        "Iop_MullS16" => Some(IROp::MullS(IRType::I16)),
        "Iop_MullS32" => Some(IROp::MullS(IRType::I32)),
        "Iop_MullS64" => Some(IROp::MullS(IRType::I64)),

        // MullU (unsigned widening multiply)
        "Iop_MullU8" => Some(IROp::MullU(IRType::I8)),
        "Iop_MullU16" => Some(IROp::MullU(IRType::I16)),
        "Iop_MullU32" => Some(IROp::MullU(IRType::I32)),
        "Iop_MullU64" => Some(IROp::MullU(IRType::I64)),

        // DivS (signed division)
        "Iop_DivS32" => Some(IROp::DivS(IRType::I32)),
        "Iop_DivS64" => Some(IROp::DivS(IRType::I64)),

        // DivU (unsigned division)
        "Iop_DivU32" => Some(IROp::DivU(IRType::I32)),
        "Iop_DivU64" => Some(IROp::DivU(IRType::I64)),

        // DivMod - combined division and modulo
        "Iop_DivModU64to32" => Some(IROp::DivModU64to32),
        "Iop_DivModS64to32" => Some(IROp::DivModS64to32),
        "Iop_DivModU128to64" => Some(IROp::DivModU128to64),
        "Iop_DivModS128to64" => Some(IROp::DivModS128to64),

        // Neg
        "Iop_Neg8" => Some(IROp::Neg(IRType::I8)),
        "Iop_Neg16" => Some(IROp::Neg(IRType::I16)),
        "Iop_Neg32" => Some(IROp::Neg(IRType::I32)),
        "Iop_Neg64" => Some(IROp::Neg(IRType::I64)),

        // High half multiply
        "Iop_MulHi32S" => Some(IROp::MulHi {
            ty: IRType::I32,
            signed: true,
        }),
        "Iop_MulHi32U" => Some(IROp::MulHi {
            ty: IRType::I32,
            signed: false,
        }),
        "Iop_MulHi64S" => Some(IROp::MulHi {
            ty: IRType::I64,
            signed: true,
        }),
        "Iop_MulHi64U" => Some(IROp::MulHi {
            ty: IRType::I64,
            signed: false,
        }),

        _ => None,
    }
}

/// Parse bitwise operations: And, Or, Xor, Not
fn parse_bitwise(op_str: &str) -> Option<IROp> {
    match op_str {
        // And
        "Iop_And1" => Some(IROp::And(IRType::I1)),
        "Iop_And8" => Some(IROp::And(IRType::I8)),
        "Iop_And16" => Some(IROp::And(IRType::I16)),
        "Iop_And32" => Some(IROp::And(IRType::I32)),
        "Iop_And64" => Some(IROp::And(IRType::I64)),
        "Iop_AndV128" => Some(IROp::VAnd(IRType::V128)),
        "Iop_AndV256" => Some(IROp::VAnd(IRType::V256)),

        // Or
        "Iop_Or1" => Some(IROp::Or(IRType::I1)),
        "Iop_Or8" => Some(IROp::Or(IRType::I8)),
        "Iop_Or16" => Some(IROp::Or(IRType::I16)),
        "Iop_Or32" => Some(IROp::Or(IRType::I32)),
        "Iop_Or64" => Some(IROp::Or(IRType::I64)),
        "Iop_OrV128" => Some(IROp::VOr(IRType::V128)),
        "Iop_OrV256" => Some(IROp::VOr(IRType::V256)),

        // Xor
        "Iop_Xor1" => Some(IROp::Xor(IRType::I1)),
        "Iop_Xor8" => Some(IROp::Xor(IRType::I8)),
        "Iop_Xor16" => Some(IROp::Xor(IRType::I16)),
        "Iop_Xor32" => Some(IROp::Xor(IRType::I32)),
        "Iop_Xor64" => Some(IROp::Xor(IRType::I64)),
        "Iop_XorV128" => Some(IROp::VXor(IRType::V128)),
        "Iop_XorV256" => Some(IROp::VXor(IRType::V256)),

        // Not
        "Iop_Not1" => Some(IROp::Not(IRType::I1)),
        "Iop_Not8" => Some(IROp::Not(IRType::I8)),
        "Iop_Not16" => Some(IROp::Not(IRType::I16)),
        "Iop_Not32" => Some(IROp::Not(IRType::I32)),
        "Iop_Not64" => Some(IROp::Not(IRType::I64)),
        "Iop_NotV128" => Some(IROp::VNot(IRType::V128)),
        "Iop_NotV256" => Some(IROp::VNot(IRType::V256)),

        _ => None,
    }
}

/// Parse shift operations: Shl, Shr, Sar (scalar and vector)
fn parse_shift(op_str: &str) -> Option<IROp> {
    match op_str {
        // Shl (logical left shift)
        "Iop_Shl8" => Some(IROp::Shl(IRType::I8)),
        "Iop_Shl16" => Some(IROp::Shl(IRType::I16)),
        "Iop_Shl32" => Some(IROp::Shl(IRType::I32)),
        "Iop_Shl64" => Some(IROp::Shl(IRType::I64)),

        // Shr (logical right shift)
        "Iop_Shr8" => Some(IROp::Shr(IRType::I8)),
        "Iop_Shr16" => Some(IROp::Shr(IRType::I16)),
        "Iop_Shr32" => Some(IROp::Shr(IRType::I32)),
        "Iop_Shr64" => Some(IROp::Shr(IRType::I64)),

        // Sar (arithmetic right shift)
        "Iop_Sar8" => Some(IROp::Sar(IRType::I8)),
        "Iop_Sar16" => Some(IROp::Sar(IRType::I16)),
        "Iop_Sar32" => Some(IROp::Sar(IRType::I32)),
        "Iop_Sar64" => Some(IROp::Sar(IRType::I64)),

        // Vector shift left by immediate (ShlN) - count is parsed at runtime
        "Iop_ShlN8x8" => Some(IROp::VShlN {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_ShlN8x16" => Some(IROp::VShlN {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_ShlN16x4" => Some(IROp::VShlN {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_ShlN16x8" => Some(IROp::VShlN {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_ShlN32x2" => Some(IROp::VShlN {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_ShlN32x4" => Some(IROp::VShlN {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_ShlN64x2" => Some(IROp::VShlN {
            elem: IRType::I64,
            count: 2,
        }),

        // Vector shift right logical by immediate (ShrN)
        "Iop_ShrN8x8" => Some(IROp::VShrN {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_ShrN8x16" => Some(IROp::VShrN {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_ShrN16x4" => Some(IROp::VShrN {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_ShrN16x8" => Some(IROp::VShrN {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_ShrN32x2" => Some(IROp::VShrN {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_ShrN32x4" => Some(IROp::VShrN {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_ShrN64x2" => Some(IROp::VShrN {
            elem: IRType::I64,
            count: 2,
        }),

        // Vector shift right arithmetic by immediate (SarN)
        "Iop_SarN8x8" => Some(IROp::VSarN {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_SarN8x16" => Some(IROp::VSarN {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_SarN16x4" => Some(IROp::VSarN {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_SarN16x8" => Some(IROp::VSarN {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_SarN32x2" => Some(IROp::VSarN {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_SarN32x4" => Some(IROp::VSarN {
            elem: IRType::I32,
            count: 4,
        }),

        _ => None,
    }
}

/// Parse comparison operations
fn parse_comparison(op_str: &str) -> Option<IROp> {
    match op_str {
        // CmpEQ
        "Iop_CmpEQ8" => Some(IROp::CmpEQ(IRType::I8)),
        "Iop_CmpEQ16" => Some(IROp::CmpEQ(IRType::I16)),
        "Iop_CmpEQ32" => Some(IROp::CmpEQ(IRType::I32)),
        "Iop_CmpEQ64" => Some(IROp::CmpEQ(IRType::I64)),

        // CmpNE
        "Iop_CmpNE8" => Some(IROp::CmpNE(IRType::I8)),
        "Iop_CmpNE16" => Some(IROp::CmpNE(IRType::I16)),
        "Iop_CmpNE32" => Some(IROp::CmpNE(IRType::I32)),
        "Iop_CmpNE64" => Some(IROp::CmpNE(IRType::I64)),

        // ExpCmp ("expensive" comparisons - same semantics as regular comparisons)
        "Iop_ExpCmpNE8" => Some(IROp::CmpNE(IRType::I8)),
        "Iop_ExpCmpNE16" => Some(IROp::CmpNE(IRType::I16)),
        "Iop_ExpCmpNE32" => Some(IROp::CmpNE(IRType::I32)),
        "Iop_ExpCmpNE64" => Some(IROp::CmpNE(IRType::I64)),

        // CmpLT (signed)
        "Iop_CmpLT32S" => Some(IROp::CmpLT(IRType::I32)),
        "Iop_CmpLT64S" => Some(IROp::CmpLT(IRType::I64)),

        // CmpLE (signed)
        "Iop_CmpLE32S" => Some(IROp::CmpLE(IRType::I32)),
        "Iop_CmpLE64S" => Some(IROp::CmpLE(IRType::I64)),

        // CmpLTU (unsigned)
        "Iop_CmpLT32U" => Some(IROp::CmpLTU(IRType::I32)),
        "Iop_CmpLT64U" => Some(IROp::CmpLTU(IRType::I64)),

        // CmpLEU (unsigned)
        "Iop_CmpLE32U" => Some(IROp::CmpLEU(IRType::I32)),
        "Iop_CmpLE64U" => Some(IROp::CmpLEU(IRType::I64)),

        // Ordered comparison (returns full width, not just 1 bit)
        // These VEX ops compare and return -1 or 0 in full width
        "Iop_CmpORD32S" | "Iop_CmpORD32U" | "Iop_CmpORD64S" | "Iop_CmpORD64U" => {
            // Map to basic comparison for now
            None
        }

        _ => None,
    }
}

/// Parse type conversion operations
fn parse_conversion(op_str: &str) -> Option<IROp> {
    match op_str {
        // Sign extensions (NtoM format: extend N-bit to M-bit signed)
        "Iop_1Sto8" => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I8,
        }),
        "Iop_1Sto16" => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I16,
        }),
        "Iop_1Sto32" => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I32,
        }),
        "Iop_1Sto64" => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I64,
        }),
        "Iop_8Sto16" => Some(IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I16,
        }),
        "Iop_8Sto32" => Some(IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I32,
        }),
        "Iop_8Sto64" => Some(IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I64,
        }),
        "Iop_16Sto32" => Some(IROp::SignExtend {
            from: IRType::I16,
            to: IRType::I32,
        }),
        "Iop_16Sto64" => Some(IROp::SignExtend {
            from: IRType::I16,
            to: IRType::I64,
        }),
        "Iop_32Sto64" => Some(IROp::SignExtend {
            from: IRType::I32,
            to: IRType::I64,
        }),

        // Zero extensions (NtoM format: extend N-bit to M-bit unsigned)
        "Iop_1Uto8" => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I8,
        }),
        "Iop_1Uto16" => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I16,
        }),
        "Iop_1Uto32" => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I32,
        }),
        "Iop_1Uto64" => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I64,
        }),
        "Iop_8Uto16" => Some(IROp::ZeroExtend {
            from: IRType::I8,
            to: IRType::I16,
        }),
        "Iop_8Uto32" => Some(IROp::ZeroExtend {
            from: IRType::I8,
            to: IRType::I32,
        }),
        "Iop_8Uto64" => Some(IROp::ZeroExtend {
            from: IRType::I8,
            to: IRType::I64,
        }),
        "Iop_16Uto32" => Some(IROp::ZeroExtend {
            from: IRType::I16,
            to: IRType::I32,
        }),
        "Iop_16Uto64" => Some(IROp::ZeroExtend {
            from: IRType::I16,
            to: IRType::I64,
        }),
        "Iop_32Uto64" => Some(IROp::ZeroExtend {
            from: IRType::I32,
            to: IRType::I64,
        }),

        // Truncations (NtoM format: truncate N-bit to M-bit)
        "Iop_64to32" => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I32,
        }),
        "Iop_64to16" => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I16,
        }),
        "Iop_64to8" => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I8,
        }),
        "Iop_64to1" => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I1,
        }),
        "Iop_32to16" => Some(IROp::Truncate {
            from: IRType::I32,
            to: IRType::I16,
        }),
        "Iop_32to8" => Some(IROp::Truncate {
            from: IRType::I32,
            to: IRType::I8,
        }),
        "Iop_32to1" => Some(IROp::Truncate {
            from: IRType::I32,
            to: IRType::I1,
        }),
        "Iop_16to8" => Some(IROp::Truncate {
            from: IRType::I16,
            to: IRType::I8,
        }),
        "Iop_128to64" => Some(IROp::Truncate {
            from: IRType::I128,
            to: IRType::I64,
        }),

        // High-half extraction
        "Iop_64HIto32" => Some(IROp::Extract {
            from: IRType::I64,
            to: IRType::I32,
            low_bit: 32,
        }),
        "Iop_32HIto16" => Some(IROp::Extract {
            from: IRType::I32,
            to: IRType::I16,
            low_bit: 16,
        }),
        "Iop_16HIto8" => Some(IROp::Extract {
            from: IRType::I16,
            to: IRType::I8,
            low_bit: 8,
        }),
        "Iop_128HIto64" => Some(IROp::Extract {
            from: IRType::I128,
            to: IRType::I64,
            low_bit: 64,
        }),

        // Concatenation
        "Iop_32HLto64" => Some(IROp::Concat { ty: IRType::I64 }),
        "Iop_64HLto128" => Some(IROp::Concat { ty: IRType::I128 }),
        "Iop_64HLtoV128" => Some(IROp::Concat { ty: IRType::V128 }),
        "Iop_16HLto32" => Some(IROp::Concat { ty: IRType::I32 }),
        "Iop_8HLto16" => Some(IROp::Concat { ty: IRType::I16 }),

        // Bit manipulation
        "Iop_Clz32" => Some(IROp::Clz(IRType::I32)),
        "Iop_Clz64" => Some(IROp::Clz(IRType::I64)),
        "Iop_Ctz32" => Some(IROp::Ctz(IRType::I32)),
        "Iop_Ctz64" => Some(IROp::Ctz(IRType::I64)),
        "Iop_PopCount8" => Some(IROp::PopCount(IRType::I8)),
        "Iop_PopCount16" => Some(IROp::PopCount(IRType::I16)),
        "Iop_PopCount32" => Some(IROp::PopCount(IRType::I32)),
        "Iop_PopCount64" => Some(IROp::PopCount(IRType::I64)),

        _ => None,
    }
}

/// Parse floating point operations
fn parse_float(op_str: &str) -> Option<IROp> {
    match op_str {
        // Basic FP arithmetic
        "Iop_AddF32" => Some(IROp::FAdd(IRType::F32)),
        "Iop_AddF64" => Some(IROp::FAdd(IRType::F64)),
        "Iop_SubF32" => Some(IROp::FSub(IRType::F32)),
        "Iop_SubF64" => Some(IROp::FSub(IRType::F64)),
        "Iop_MulF32" => Some(IROp::FMul(IRType::F32)),
        "Iop_MulF64" => Some(IROp::FMul(IRType::F64)),
        "Iop_DivF32" => Some(IROp::FDiv(IRType::F32)),
        "Iop_DivF64" => Some(IROp::FDiv(IRType::F64)),

        // Unary FP ops
        "Iop_NegF32" => Some(IROp::FNeg(IRType::F32)),
        "Iop_NegF64" => Some(IROp::FNeg(IRType::F64)),
        "Iop_AbsF32" => Some(IROp::FAbs(IRType::F32)),
        "Iop_AbsF64" => Some(IROp::FAbs(IRType::F64)),
        "Iop_SqrtF32" => Some(IROp::FSqrt(IRType::F32)),
        "Iop_SqrtF64" => Some(IROp::FSqrt(IRType::F64)),

        // Fused multiply-add/sub (Qops with rounding mode)
        "Iop_MAddF32" => Some(IROp::FMAdd(IRType::F32)),
        "Iop_MAddF64" => Some(IROp::FMAdd(IRType::F64)),
        "Iop_MSubF32" => Some(IROp::FMSub(IRType::F32)),
        "Iop_MSubF64" => Some(IROp::FMSub(IRType::F64)),

        // Scalar-in-vector float ops (SSE scalar: ADDSS, SUBSS, MULSS, DIVSS, etc.)
        "Iop_Add32F0x4" => Some(IROp::VFAddS { elem: IRType::F32 }),
        "Iop_Add64F0x2" => Some(IROp::VFAddS { elem: IRType::F64 }),
        "Iop_Sub32F0x4" => Some(IROp::VFSubS { elem: IRType::F32 }),
        "Iop_Sub64F0x2" => Some(IROp::VFSubS { elem: IRType::F64 }),
        "Iop_Mul32F0x4" => Some(IROp::VFMulS { elem: IRType::F32 }),
        "Iop_Mul64F0x2" => Some(IROp::VFMulS { elem: IRType::F64 }),
        "Iop_Div32F0x4" => Some(IROp::VFDivS { elem: IRType::F32 }),
        "Iop_Div64F0x2" => Some(IROp::VFDivS { elem: IRType::F64 }),
        "Iop_Sqrt32F0x4" => Some(IROp::VFSqrtS { elem: IRType::F32 }),
        "Iop_Sqrt64F0x2" => Some(IROp::VFSqrtS { elem: IRType::F64 }),
        "Iop_Max32F0x4" => Some(IROp::VFMaxS { elem: IRType::F32 }),
        "Iop_Max64F0x2" => Some(IROp::VFMaxS { elem: IRType::F64 }),
        "Iop_Min32F0x4" => Some(IROp::VFMinS { elem: IRType::F32 }),
        "Iop_Min64F0x2" => Some(IROp::VFMinS { elem: IRType::F64 }),

        // Packed (whole-vector) float ops — SSE / AVX
        "Iop_Add32Fx4" => Some(IROp::VFAdd {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Add64Fx2" => Some(IROp::VFAdd {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Add32Fx8" => Some(IROp::VFAdd {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Add64Fx4" => Some(IROp::VFAdd {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Sub32Fx4" => Some(IROp::VFSub {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Sub64Fx2" => Some(IROp::VFSub {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Sub32Fx8" => Some(IROp::VFSub {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Sub64Fx4" => Some(IROp::VFSub {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Mul32Fx4" => Some(IROp::VFMul {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Mul64Fx2" => Some(IROp::VFMul {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Mul32Fx8" => Some(IROp::VFMul {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Mul64Fx4" => Some(IROp::VFMul {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Div32Fx4" => Some(IROp::VFDiv {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Div64Fx2" => Some(IROp::VFDiv {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Div32Fx8" => Some(IROp::VFDiv {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Div64Fx4" => Some(IROp::VFDiv {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Sqrt32Fx4" => Some(IROp::VFSqrt {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Sqrt64Fx2" => Some(IROp::VFSqrt {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Sqrt32Fx8" => Some(IROp::VFSqrt {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Sqrt64Fx4" => Some(IROp::VFSqrt {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Abs32Fx4" => Some(IROp::VFAbs {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Abs64Fx2" => Some(IROp::VFAbs {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Abs32Fx8" => Some(IROp::VFAbs {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Abs64Fx4" => Some(IROp::VFAbs {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Min32Fx4" => Some(IROp::VFMin {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Min64Fx2" => Some(IROp::VFMin {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Min32Fx8" => Some(IROp::VFMin {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Min64Fx4" => Some(IROp::VFMin {
            elem: IRType::F64,
            count: 4,
        }),
        "Iop_Max32Fx4" => Some(IROp::VFMax {
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_Max64Fx2" => Some(IROp::VFMax {
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_Max32Fx8" => Some(IROp::VFMax {
            elem: IRType::F32,
            count: 8,
        }),
        "Iop_Max64Fx4" => Some(IROp::VFMax {
            elem: IRType::F64,
            count: 4,
        }),

        // SetV128lo operations
        "Iop_SetV128lo32" => Some(IROp::SetV128lo32),
        "Iop_SetV128lo64" => Some(IROp::SetV128lo64),

        // FP comparisons (scalar I1 result — used internally and for ccall lifts)
        "Iop_CmpF32" => Some(IROp::FComCC(IRType::F32)),
        "Iop_CmpF64" => Some(IROp::FComCC(IRType::F64)),

        // SSE scalar-lane compares: lane 0 → all-1s/0 mask, upper lanes pass-through.
        // Result type is V128 — distinct from the I1-returning FCmpEQ/LT/LE.
        "Iop_CmpEQ32F0x4" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Eq,
            ty: IRType::F32,
        }),
        "Iop_CmpEQ64F0x2" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Eq,
            ty: IRType::F64,
        }),
        "Iop_CmpLT32F0x4" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Lt,
            ty: IRType::F32,
        }),
        "Iop_CmpLT64F0x2" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Lt,
            ty: IRType::F64,
        }),
        "Iop_CmpLE32F0x4" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Le,
            ty: IRType::F32,
        }),
        "Iop_CmpLE64F0x2" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Le,
            ty: IRType::F64,
        }),
        "Iop_CmpUN32F0x4" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Un,
            ty: IRType::F32,
        }),
        "Iop_CmpUN64F0x2" => Some(IROp::FCmpScalarLane {
            kind: FCmpKind::Un,
            ty: IRType::F64,
        }),

        // Packed FP compares (SSE cmpps/cmppd, ARM NEON 32Fx2). Per-lane mask:
        // each lane independently produces all-1s (true) or 0 (false).
        // 32Fx2 returns I64 (ARM NEON), 32Fx4 / 64Fx2 return V128 (SSE).
        "Iop_CmpEQ32Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Eq,
            elem: IRType::F32,
            count: 2,
        }),
        "Iop_CmpGT32Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Gt,
            elem: IRType::F32,
            count: 2,
        }),
        "Iop_CmpGE32Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Ge,
            elem: IRType::F32,
            count: 2,
        }),
        "Iop_CmpEQ32Fx4" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Eq,
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_CmpLT32Fx4" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Lt,
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_CmpLE32Fx4" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Le,
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_CmpGT32Fx4" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Gt,
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_CmpGE32Fx4" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Ge,
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_CmpUN32Fx4" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Un,
            elem: IRType::F32,
            count: 4,
        }),
        "Iop_CmpEQ64Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Eq,
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_CmpLT64Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Lt,
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_CmpLE64Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Le,
            elem: IRType::F64,
            count: 2,
        }),
        "Iop_CmpUN64Fx2" => Some(IROp::FCmpVecPacked {
            kind: FCmpKind::Un,
            elem: IRType::F64,
            count: 2,
        }),

        // FP conversions
        "Iop_F32toF64" => Some(IROp::F32toF64),
        "Iop_F64toF32" => Some(IROp::F64toF32),
        "Iop_I32StoF32" => Some(IROp::I32StoF32),
        "Iop_I32StoF64" => Some(IROp::I32StoF64),
        "Iop_I64StoF32" => Some(IROp::I64StoF32),
        "Iop_I64StoF64" => Some(IROp::I64StoF64),
        "Iop_I32UtoF32" => Some(IROp::I32UtoF32),
        "Iop_I32UtoF64" => Some(IROp::I32UtoF64),
        "Iop_I64UtoF32" => Some(IROp::I64UtoF32),
        "Iop_I64UtoF64" => Some(IROp::I64UtoF64),
        "Iop_F32toI32S" => Some(IROp::F32toI32S),
        "Iop_F64toI32S" => Some(IROp::F64toI32S),
        "Iop_F32toI64S" => Some(IROp::F32toI64S),
        "Iop_F64toI64S" => Some(IROp::F64toI64S),
        "Iop_F32toI32U" => Some(IROp::F32toI32U),
        "Iop_F64toI32U" => Some(IROp::F64toI32U),
        "Iop_F32toI64U" => Some(IROp::F32toI64U),
        "Iop_F64toI64U" => Some(IROp::F64toI64U),

        // Rounding
        "Iop_RoundF32toInt" => Some(IROp::RoundF32toInt),
        "Iop_RoundF64toInt" => Some(IROp::RoundF64toInt),

        // Reinterpret as different type
        "Iop_ReinterpF32asI32" => Some(IROp::Reinterpret {
            from: IRType::F32,
            to: IRType::I32,
        }),
        "Iop_ReinterpI32asF32" => Some(IROp::Reinterpret {
            from: IRType::I32,
            to: IRType::F32,
        }),
        "Iop_ReinterpF64asI64" => Some(IROp::Reinterpret {
            from: IRType::F64,
            to: IRType::I64,
        }),
        "Iop_ReinterpI64asF64" => Some(IROp::Reinterpret {
            from: IRType::I64,
            to: IRType::F64,
        }),

        _ => None,
    }
}

/// Parse vector/SIMD operations
fn parse_vector(op_str: &str) -> Option<IROp> {
    match op_str {
        // Vector add (8-bit elements)
        "Iop_Add8x8" => Some(IROp::VAdd {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_Add8x16" => Some(IROp::VAdd {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_Add8x32" => Some(IROp::VAdd {
            elem: IRType::I8,
            count: 32,
        }),

        // Vector add (16-bit elements)
        "Iop_Add16x4" => Some(IROp::VAdd {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_Add16x8" => Some(IROp::VAdd {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_Add16x16" => Some(IROp::VAdd {
            elem: IRType::I16,
            count: 16,
        }),

        // Vector add (32-bit elements)
        "Iop_Add32x2" => Some(IROp::VAdd {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_Add32x4" => Some(IROp::VAdd {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_Add32x8" => Some(IROp::VAdd {
            elem: IRType::I32,
            count: 8,
        }),

        // Vector add (64-bit elements)
        "Iop_Add64x2" => Some(IROp::VAdd {
            elem: IRType::I64,
            count: 2,
        }),
        "Iop_Add64x4" => Some(IROp::VAdd {
            elem: IRType::I64,
            count: 4,
        }),

        // Vector sub (8-bit elements)
        "Iop_Sub8x8" => Some(IROp::VSub {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_Sub8x16" => Some(IROp::VSub {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_Sub8x32" => Some(IROp::VSub {
            elem: IRType::I8,
            count: 32,
        }),

        // Vector sub (16-bit elements)
        "Iop_Sub16x4" => Some(IROp::VSub {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_Sub16x8" => Some(IROp::VSub {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_Sub16x16" => Some(IROp::VSub {
            elem: IRType::I16,
            count: 16,
        }),

        // Vector sub (32-bit elements)
        "Iop_Sub32x2" => Some(IROp::VSub {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_Sub32x4" => Some(IROp::VSub {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_Sub32x8" => Some(IROp::VSub {
            elem: IRType::I32,
            count: 8,
        }),

        // Vector sub (64-bit elements)
        "Iop_Sub64x2" => Some(IROp::VSub {
            elem: IRType::I64,
            count: 2,
        }),
        "Iop_Sub64x4" => Some(IROp::VSub {
            elem: IRType::I64,
            count: 4,
        }),

        // Vector multiply (8-bit elements — NEON only, VMUL.I8)
        "Iop_Mul8x8" => Some(IROp::VMul {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_Mul8x16" => Some(IROp::VMul {
            elem: IRType::I8,
            count: 16,
        }),

        // Vector multiply (16-bit elements)
        "Iop_Mul16x4" => Some(IROp::VMul {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_Mul16x8" => Some(IROp::VMul {
            elem: IRType::I16,
            count: 8,
        }),

        // Vector multiply (32-bit elements)
        "Iop_Mul32x2" => Some(IROp::VMul {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_Mul32x4" => Some(IROp::VMul {
            elem: IRType::I32,
            count: 4,
        }),

        // NEON lane extract — Iop_GetElem{N}x{M}: (vec, idx) -> scalar lane
        "Iop_GetElem8x8" => Some(IROp::VGetElem {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_GetElem16x4" => Some(IROp::VGetElem {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_GetElem32x2" => Some(IROp::VGetElem {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_GetElem8x16" => Some(IROp::VGetElem {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_GetElem16x8" => Some(IROp::VGetElem {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_GetElem32x4" => Some(IROp::VGetElem {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_GetElem64x2" => Some(IROp::VGetElem {
            elem: IRType::I64,
            count: 2,
        }),

        // NEON lane insert — Iop_SetElem{N}x{M}: (vec, idx, val) -> vec
        "Iop_SetElem8x8" => Some(IROp::VSetElem {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_SetElem16x4" => Some(IROp::VSetElem {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_SetElem32x2" => Some(IROp::VSetElem {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_SetElem8x16" => Some(IROp::VSetElem {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_SetElem16x8" => Some(IROp::VSetElem {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_SetElem32x4" => Some(IROp::VSetElem {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_SetElem64x2" => Some(IROp::VSetElem {
            elem: IRType::I64,
            count: 2,
        }),

        // Vector multiply keeping low half (PMULLD - SSE4.1)
        "Iop_MullS32x4" => Some(IROp::VMulLo {
            elem: IRType::I32,
            count: 4,
        }),

        // Vector compare equal
        "Iop_CmpEQ8x8" => Some(IROp::VCmpEQ {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_CmpEQ8x16" => Some(IROp::VCmpEQ {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_CmpEQ16x4" => Some(IROp::VCmpEQ {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_CmpEQ16x8" => Some(IROp::VCmpEQ {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_CmpEQ32x2" => Some(IROp::VCmpEQ {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_CmpEQ32x4" => Some(IROp::VCmpEQ {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_CmpEQ64x2" => Some(IROp::VCmpEQ {
            elem: IRType::I64,
            count: 2,
        }),

        // Vector compare greater than
        "Iop_CmpGT8Sx8" => Some(IROp::VCmpGT {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_CmpGT8Sx16" => Some(IROp::VCmpGT {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_CmpGT16Sx4" => Some(IROp::VCmpGT {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_CmpGT16Sx8" => Some(IROp::VCmpGT {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_CmpGT32Sx2" => Some(IROp::VCmpGT {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_CmpGT32Sx4" => Some(IROp::VCmpGT {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_CmpGT64Sx2" => Some(IROp::VCmpGT {
            elem: IRType::I64,
            count: 2,
        }),

        // Vector interleave
        "Iop_InterleaveLO8x8" | "Iop_InterleaveLO8x16" => {
            Some(IROp::VInterleaveLO { elem: IRType::I8 })
        }
        "Iop_InterleaveLO16x4" | "Iop_InterleaveLO16x8" => {
            Some(IROp::VInterleaveLO { elem: IRType::I16 })
        }
        "Iop_InterleaveLO32x2" | "Iop_InterleaveLO32x4" => {
            Some(IROp::VInterleaveLO { elem: IRType::I32 })
        }
        "Iop_InterleaveLO64x2" => Some(IROp::VInterleaveLO { elem: IRType::I64 }),

        "Iop_InterleaveHI8x8" | "Iop_InterleaveHI8x16" => {
            Some(IROp::VInterleaveHI { elem: IRType::I8 })
        }
        "Iop_InterleaveHI16x4" | "Iop_InterleaveHI16x8" => {
            Some(IROp::VInterleaveHI { elem: IRType::I16 })
        }
        "Iop_InterleaveHI32x2" | "Iop_InterleaveHI32x4" => {
            Some(IROp::VInterleaveHI { elem: IRType::I32 })
        }
        "Iop_InterleaveHI64x2" => Some(IROp::VInterleaveHI { elem: IRType::I64 }),

        // Packed integer min — signed
        "Iop_Min8Sx8" => Some(IROp::VMin {
            elem: IRType::I8,
            count: 8,
            signed: true,
        }),
        "Iop_Min8Sx16" => Some(IROp::VMin {
            elem: IRType::I8,
            count: 16,
            signed: true,
        }),
        "Iop_Min8Sx32" => Some(IROp::VMin {
            elem: IRType::I8,
            count: 32,
            signed: true,
        }),
        "Iop_Min16Sx4" => Some(IROp::VMin {
            elem: IRType::I16,
            count: 4,
            signed: true,
        }),
        "Iop_Min16Sx8" => Some(IROp::VMin {
            elem: IRType::I16,
            count: 8,
            signed: true,
        }),
        "Iop_Min16Sx16" => Some(IROp::VMin {
            elem: IRType::I16,
            count: 16,
            signed: true,
        }),
        "Iop_Min32Sx2" => Some(IROp::VMin {
            elem: IRType::I32,
            count: 2,
            signed: true,
        }),
        "Iop_Min32Sx4" => Some(IROp::VMin {
            elem: IRType::I32,
            count: 4,
            signed: true,
        }),
        "Iop_Min32Sx8" => Some(IROp::VMin {
            elem: IRType::I32,
            count: 8,
            signed: true,
        }),
        "Iop_Min64Sx2" => Some(IROp::VMin {
            elem: IRType::I64,
            count: 2,
            signed: true,
        }),
        "Iop_Min64Sx4" => Some(IROp::VMin {
            elem: IRType::I64,
            count: 4,
            signed: true,
        }),

        // Packed integer min — unsigned
        "Iop_Min8Ux8" => Some(IROp::VMin {
            elem: IRType::I8,
            count: 8,
            signed: false,
        }),
        "Iop_Min8Ux16" => Some(IROp::VMin {
            elem: IRType::I8,
            count: 16,
            signed: false,
        }),
        "Iop_Min8Ux32" => Some(IROp::VMin {
            elem: IRType::I8,
            count: 32,
            signed: false,
        }),
        "Iop_Min16Ux4" => Some(IROp::VMin {
            elem: IRType::I16,
            count: 4,
            signed: false,
        }),
        "Iop_Min16Ux8" => Some(IROp::VMin {
            elem: IRType::I16,
            count: 8,
            signed: false,
        }),
        "Iop_Min16Ux16" => Some(IROp::VMin {
            elem: IRType::I16,
            count: 16,
            signed: false,
        }),
        "Iop_Min32Ux2" => Some(IROp::VMin {
            elem: IRType::I32,
            count: 2,
            signed: false,
        }),
        "Iop_Min32Ux4" => Some(IROp::VMin {
            elem: IRType::I32,
            count: 4,
            signed: false,
        }),
        "Iop_Min32Ux8" => Some(IROp::VMin {
            elem: IRType::I32,
            count: 8,
            signed: false,
        }),
        "Iop_Min64Ux2" => Some(IROp::VMin {
            elem: IRType::I64,
            count: 2,
            signed: false,
        }),
        "Iop_Min64Ux4" => Some(IROp::VMin {
            elem: IRType::I64,
            count: 4,
            signed: false,
        }),

        // Packed integer max — signed
        "Iop_Max8Sx8" => Some(IROp::VMax {
            elem: IRType::I8,
            count: 8,
            signed: true,
        }),
        "Iop_Max8Sx16" => Some(IROp::VMax {
            elem: IRType::I8,
            count: 16,
            signed: true,
        }),
        "Iop_Max8Sx32" => Some(IROp::VMax {
            elem: IRType::I8,
            count: 32,
            signed: true,
        }),
        "Iop_Max16Sx4" => Some(IROp::VMax {
            elem: IRType::I16,
            count: 4,
            signed: true,
        }),
        "Iop_Max16Sx8" => Some(IROp::VMax {
            elem: IRType::I16,
            count: 8,
            signed: true,
        }),
        "Iop_Max16Sx16" => Some(IROp::VMax {
            elem: IRType::I16,
            count: 16,
            signed: true,
        }),
        "Iop_Max32Sx2" => Some(IROp::VMax {
            elem: IRType::I32,
            count: 2,
            signed: true,
        }),
        "Iop_Max32Sx4" => Some(IROp::VMax {
            elem: IRType::I32,
            count: 4,
            signed: true,
        }),
        "Iop_Max32Sx8" => Some(IROp::VMax {
            elem: IRType::I32,
            count: 8,
            signed: true,
        }),
        "Iop_Max64Sx2" => Some(IROp::VMax {
            elem: IRType::I64,
            count: 2,
            signed: true,
        }),
        "Iop_Max64Sx4" => Some(IROp::VMax {
            elem: IRType::I64,
            count: 4,
            signed: true,
        }),

        // Packed integer max — unsigned
        "Iop_Max8Ux8" => Some(IROp::VMax {
            elem: IRType::I8,
            count: 8,
            signed: false,
        }),
        "Iop_Max8Ux16" => Some(IROp::VMax {
            elem: IRType::I8,
            count: 16,
            signed: false,
        }),
        "Iop_Max8Ux32" => Some(IROp::VMax {
            elem: IRType::I8,
            count: 32,
            signed: false,
        }),
        "Iop_Max16Ux4" => Some(IROp::VMax {
            elem: IRType::I16,
            count: 4,
            signed: false,
        }),
        "Iop_Max16Ux8" => Some(IROp::VMax {
            elem: IRType::I16,
            count: 8,
            signed: false,
        }),
        "Iop_Max16Ux16" => Some(IROp::VMax {
            elem: IRType::I16,
            count: 16,
            signed: false,
        }),
        "Iop_Max32Ux2" => Some(IROp::VMax {
            elem: IRType::I32,
            count: 2,
            signed: false,
        }),
        "Iop_Max32Ux4" => Some(IROp::VMax {
            elem: IRType::I32,
            count: 4,
            signed: false,
        }),
        "Iop_Max32Ux8" => Some(IROp::VMax {
            elem: IRType::I32,
            count: 8,
            signed: false,
        }),
        "Iop_Max64Ux2" => Some(IROp::VMax {
            elem: IRType::I64,
            count: 2,
            signed: false,
        }),
        "Iop_Max64Ux4" => Some(IROp::VMax {
            elem: IRType::I64,
            count: 4,
            signed: false,
        }),

        // Packed integer absolute value
        "Iop_Abs8x8" => Some(IROp::VAbs {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_Abs8x16" => Some(IROp::VAbs {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_Abs8x32" => Some(IROp::VAbs {
            elem: IRType::I8,
            count: 32,
        }),
        "Iop_Abs16x4" => Some(IROp::VAbs {
            elem: IRType::I16,
            count: 4,
        }),
        "Iop_Abs16x8" => Some(IROp::VAbs {
            elem: IRType::I16,
            count: 8,
        }),
        "Iop_Abs16x16" => Some(IROp::VAbs {
            elem: IRType::I16,
            count: 16,
        }),
        "Iop_Abs32x2" => Some(IROp::VAbs {
            elem: IRType::I32,
            count: 2,
        }),
        "Iop_Abs32x4" => Some(IROp::VAbs {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_Abs32x8" => Some(IROp::VAbs {
            elem: IRType::I32,
            count: 8,
        }),
        "Iop_Abs64x2" => Some(IROp::VAbs {
            elem: IRType::I64,
            count: 2,
        }),
        "Iop_Abs64x4" => Some(IROp::VAbs {
            elem: IRType::I64,
            count: 4,
        }),

        // V128/V256 to/from conversions
        "Iop_V128to64" => Some(IROp::Truncate {
            from: IRType::V128,
            to: IRType::I64,
        }),
        "Iop_V128HIto64" => Some(IROp::Extract {
            from: IRType::V128,
            to: IRType::I64,
            low_bit: 64,
        }),
        "Iop_64UtoV128" => Some(IROp::ZeroExtend {
            from: IRType::I64,
            to: IRType::V128,
        }),
        "Iop_32UtoV128" => Some(IROp::ZeroExtend {
            from: IRType::I32,
            to: IRType::V128,
        }),
        "Iop_SetV128lo64" => Some(IROp::Reinterpret {
            from: IRType::I64,
            to: IRType::V128,
        }),

        _ => None,
    }
}

/// Parse ARM/AArch64 NEON SIMD opcodes that have been claimed but not yet
/// implemented. Hits here route through `IROp::NeonUnimplemented(name)` so
/// dispatch in `VEXOps::unop` / `binop` / etc. panics with the original
/// opcode name instead of silently returning a fresh-symbolic value.
///
/// Implementations are added one-at-a-time in angr-bkcs.2 by:
///   1. Removing the opcode's entry from this function.
///   2. Adding it to `parse_vector` (or `parse_float`) with a real IROp variant.
///   3. Wiring that variant into `VEXOps::unop` / `binop`.
///
/// Scope: D-register (Ity_I64) and Q-register (Ity_V128) opcodes that pyvex
/// emits for ARM/AArch64 NEON and that this engine currently has no handler
/// for. Opcodes already handled by `parse_vector` (e.g. `Iop_Add8x8` ->
/// `VAdd`) are deliberately excluded so we do not regress existing coverage.
fn parse_neon_unimplemented(op_str: &str) -> Option<IROp> {
    let op = match op_str {
        // Dup (broadcast scalar across lanes)
        "Iop_Dup8x8" => "Iop_Dup8x8",
        "Iop_Dup16x4" => "Iop_Dup16x4",
        "Iop_Dup32x2" => "Iop_Dup32x2",
        "Iop_Dup8x16" => "Iop_Dup8x16",
        "Iop_Dup16x8" => "Iop_Dup16x8",
        "Iop_Dup32x4" => "Iop_Dup32x4",

        // Narrow (truncating, non-saturating)
        "Iop_NarrowBin16to8x8" => "Iop_NarrowBin16to8x8",
        "Iop_NarrowBin32to16x4" => "Iop_NarrowBin32to16x4",
        "Iop_NarrowBin16to8x16" => "Iop_NarrowBin16to8x16",
        "Iop_NarrowBin32to16x8" => "Iop_NarrowBin32to16x8",
        "Iop_NarrowBin64to32x4" => "Iop_NarrowBin64to32x4",
        "Iop_NarrowUn16to8x8" => "Iop_NarrowUn16to8x8",
        "Iop_NarrowUn32to16x4" => "Iop_NarrowUn32to16x4",
        "Iop_NarrowUn64to32x2" => "Iop_NarrowUn64to32x2",

        // QNarrow (saturating narrow)
        "Iop_QNarrowBin16Sto8Sx8" => "Iop_QNarrowBin16Sto8Sx8",
        "Iop_QNarrowBin16Sto8Ux8" => "Iop_QNarrowBin16Sto8Ux8",
        "Iop_QNarrowBin32Sto16Sx4" => "Iop_QNarrowBin32Sto16Sx4",
        "Iop_QNarrowBin32Sto16Ux4" => "Iop_QNarrowBin32Sto16Ux4",
        "Iop_QNarrowBin16Sto8Sx16" => "Iop_QNarrowBin16Sto8Sx16",
        "Iop_QNarrowBin16Sto8Ux16" => "Iop_QNarrowBin16Sto8Ux16",
        "Iop_QNarrowBin16Uto8Ux16" => "Iop_QNarrowBin16Uto8Ux16",
        "Iop_QNarrowBin32Sto16Sx8" => "Iop_QNarrowBin32Sto16Sx8",
        "Iop_QNarrowBin32Sto16Ux8" => "Iop_QNarrowBin32Sto16Ux8",
        "Iop_QNarrowBin32Uto16Ux8" => "Iop_QNarrowBin32Uto16Ux8",
        "Iop_QNarrowBin64Sto32Sx4" => "Iop_QNarrowBin64Sto32Sx4",
        "Iop_QNarrowBin64Uto32Ux4" => "Iop_QNarrowBin64Uto32Ux4",
        "Iop_QNarrowUn16Sto8Sx8" => "Iop_QNarrowUn16Sto8Sx8",
        "Iop_QNarrowUn16Sto8Ux8" => "Iop_QNarrowUn16Sto8Ux8",
        "Iop_QNarrowUn16Uto8Ux8" => "Iop_QNarrowUn16Uto8Ux8",
        "Iop_QNarrowUn32Sto16Sx4" => "Iop_QNarrowUn32Sto16Sx4",
        "Iop_QNarrowUn32Sto16Ux4" => "Iop_QNarrowUn32Sto16Ux4",
        "Iop_QNarrowUn32Uto16Ux4" => "Iop_QNarrowUn32Uto16Ux4",
        "Iop_QNarrowUn64Sto32Sx2" => "Iop_QNarrowUn64Sto32Sx2",
        "Iop_QNarrowUn64Sto32Ux2" => "Iop_QNarrowUn64Sto32Ux2",
        "Iop_QNarrowUn64Uto32Ux2" => "Iop_QNarrowUn64Uto32Ux2",

        // Widen (sign- or zero-extend element width, halving lane count)
        "Iop_Widen8Sto16x8" => "Iop_Widen8Sto16x8",
        "Iop_Widen8Uto16x8" => "Iop_Widen8Uto16x8",
        "Iop_Widen16Sto32x4" => "Iop_Widen16Sto32x4",
        "Iop_Widen16Uto32x4" => "Iop_Widen16Uto32x4",
        "Iop_Widen32Sto64x2" => "Iop_Widen32Sto64x2",
        "Iop_Widen32Uto64x2" => "Iop_Widen32Uto64x2",

        // NOTE: Iop_GetElem* / Iop_SetElem* (lane extract/insert) implemented
        // in angr-bkcs.2 — routed through parse_vector to IROp::VGetElem /
        // IROp::VSetElem above.

        // Reciprocal estimate / Newton-Raphson step (FP)
        "Iop_RecipEst32Fx2" => "Iop_RecipEst32Fx2",
        "Iop_RecipEst32Fx4" => "Iop_RecipEst32Fx4",
        "Iop_RecipEst64Fx2" => "Iop_RecipEst64Fx2",
        "Iop_RecipStep32Fx2" => "Iop_RecipStep32Fx2",
        "Iop_RecipStep32Fx4" => "Iop_RecipStep32Fx4",
        "Iop_RecipStep64Fx2" => "Iop_RecipStep64Fx2",
        "Iop_RSqrtEst32Fx2" => "Iop_RSqrtEst32Fx2",
        "Iop_RSqrtEst32Fx4" => "Iop_RSqrtEst32Fx4",
        "Iop_RSqrtEst64Fx2" => "Iop_RSqrtEst64Fx2",
        "Iop_RSqrtStep32Fx2" => "Iop_RSqrtStep32Fx2",
        "Iop_RSqrtStep32Fx4" => "Iop_RSqrtStep32Fx4",
        "Iop_RSqrtStep64Fx2" => "Iop_RSqrtStep64Fx2",

        // Reciprocal estimate (integer, NEON-only)
        "Iop_RecipEst32Ux2" => "Iop_RecipEst32Ux2",
        "Iop_RecipEst32Ux4" => "Iop_RecipEst32Ux4",
        "Iop_RSqrtEst32Ux2" => "Iop_RSqrtEst32Ux2",
        "Iop_RSqrtEst32Ux4" => "Iop_RSqrtEst32Ux4",

        // Saturating integer add/sub (NEON Q-prefixed)
        "Iop_QAdd8Sx8" => "Iop_QAdd8Sx8",
        "Iop_QAdd16Sx4" => "Iop_QAdd16Sx4",
        "Iop_QAdd32Sx2" => "Iop_QAdd32Sx2",
        "Iop_QAdd64Sx1" => "Iop_QAdd64Sx1",
        "Iop_QAdd8Ux8" => "Iop_QAdd8Ux8",
        "Iop_QAdd16Ux4" => "Iop_QAdd16Ux4",
        "Iop_QAdd32Ux2" => "Iop_QAdd32Ux2",
        "Iop_QAdd64Ux1" => "Iop_QAdd64Ux1",
        "Iop_QAdd8Sx16" => "Iop_QAdd8Sx16",
        "Iop_QAdd16Sx8" => "Iop_QAdd16Sx8",
        "Iop_QAdd32Sx4" => "Iop_QAdd32Sx4",
        "Iop_QAdd64Sx2" => "Iop_QAdd64Sx2",
        "Iop_QAdd8Ux16" => "Iop_QAdd8Ux16",
        "Iop_QAdd16Ux8" => "Iop_QAdd16Ux8",
        "Iop_QAdd32Ux4" => "Iop_QAdd32Ux4",
        "Iop_QAdd64Ux2" => "Iop_QAdd64Ux2",
        "Iop_QSub8Sx8" => "Iop_QSub8Sx8",
        "Iop_QSub16Sx4" => "Iop_QSub16Sx4",
        "Iop_QSub32Sx2" => "Iop_QSub32Sx2",
        "Iop_QSub64Sx1" => "Iop_QSub64Sx1",
        "Iop_QSub8Ux8" => "Iop_QSub8Ux8",
        "Iop_QSub16Ux4" => "Iop_QSub16Ux4",
        "Iop_QSub32Ux2" => "Iop_QSub32Ux2",
        "Iop_QSub64Ux1" => "Iop_QSub64Ux1",
        "Iop_QSub8Sx16" => "Iop_QSub8Sx16",
        "Iop_QSub16Sx8" => "Iop_QSub16Sx8",
        "Iop_QSub32Sx4" => "Iop_QSub32Sx4",
        "Iop_QSub64Sx2" => "Iop_QSub64Sx2",
        "Iop_QSub8Ux16" => "Iop_QSub8Ux16",
        "Iop_QSub16Ux8" => "Iop_QSub16Ux8",
        "Iop_QSub32Ux4" => "Iop_QSub32Ux4",
        "Iop_QSub64Ux2" => "Iop_QSub64Ux2",

        // Averaging
        "Iop_Avg8Ux8" => "Iop_Avg8Ux8",
        "Iop_Avg16Ux4" => "Iop_Avg16Ux4",
        "Iop_Avg32Ux2" => "Iop_Avg32Ux2",
        "Iop_Avg8Sx8" => "Iop_Avg8Sx8",
        "Iop_Avg16Sx4" => "Iop_Avg16Sx4",
        "Iop_Avg32Sx2" => "Iop_Avg32Sx2",
        "Iop_Avg8Ux16" => "Iop_Avg8Ux16",
        "Iop_Avg16Ux8" => "Iop_Avg16Ux8",
        "Iop_Avg32Ux4" => "Iop_Avg32Ux4",
        "Iop_Avg8Sx16" => "Iop_Avg8Sx16",
        "Iop_Avg16Sx8" => "Iop_Avg16Sx8",
        "Iop_Avg32Sx4" => "Iop_Avg32Sx4",

        // Reverse bytes within lanes
        "Iop_Reverse8sIn16_x4" => "Iop_Reverse8sIn16_x4",
        "Iop_Reverse8sIn32_x2" => "Iop_Reverse8sIn32_x2",
        "Iop_Reverse8sIn64_x1" => "Iop_Reverse8sIn64_x1",
        "Iop_Reverse16sIn32_x2" => "Iop_Reverse16sIn32_x2",
        "Iop_Reverse16sIn64_x1" => "Iop_Reverse16sIn64_x1",
        "Iop_Reverse32sIn64_x1" => "Iop_Reverse32sIn64_x1",
        "Iop_Reverse8sIn16_x8" => "Iop_Reverse8sIn16_x8",
        "Iop_Reverse8sIn32_x4" => "Iop_Reverse8sIn32_x4",
        "Iop_Reverse8sIn64_x2" => "Iop_Reverse8sIn64_x2",
        "Iop_Reverse16sIn32_x4" => "Iop_Reverse16sIn32_x4",
        "Iop_Reverse16sIn64_x2" => "Iop_Reverse16sIn64_x2",
        "Iop_Reverse32sIn64_x2" => "Iop_Reverse32sIn64_x2",
        "Iop_Reverse1sIn8_x8" => "Iop_Reverse1sIn8_x8",
        "Iop_Reverse1sIn8_x16" => "Iop_Reverse1sIn8_x16",

        // Pairwise add/min/max (NEON)
        "Iop_PwAdd8x8" => "Iop_PwAdd8x8",
        "Iop_PwAdd16x4" => "Iop_PwAdd16x4",
        "Iop_PwAdd32x2" => "Iop_PwAdd32x2",
        "Iop_PwAdd32Fx2" => "Iop_PwAdd32Fx2",
        "Iop_PwAdd8x16" => "Iop_PwAdd8x16",
        "Iop_PwAdd16x8" => "Iop_PwAdd16x8",
        "Iop_PwAdd32x4" => "Iop_PwAdd32x4",
        "Iop_PwAddL8Sx8" => "Iop_PwAddL8Sx8",
        "Iop_PwAddL8Ux8" => "Iop_PwAddL8Ux8",
        "Iop_PwAddL16Sx4" => "Iop_PwAddL16Sx4",
        "Iop_PwAddL16Ux4" => "Iop_PwAddL16Ux4",
        "Iop_PwAddL32Sx2" => "Iop_PwAddL32Sx2",
        "Iop_PwAddL32Ux2" => "Iop_PwAddL32Ux2",
        "Iop_PwAddL8Sx16" => "Iop_PwAddL8Sx16",
        "Iop_PwAddL8Ux16" => "Iop_PwAddL8Ux16",
        "Iop_PwAddL16Sx8" => "Iop_PwAddL16Sx8",
        "Iop_PwAddL16Ux8" => "Iop_PwAddL16Ux8",
        "Iop_PwAddL32Sx4" => "Iop_PwAddL32Sx4",
        "Iop_PwAddL32Ux4" => "Iop_PwAddL32Ux4",
        "Iop_PwMin8Sx8" => "Iop_PwMin8Sx8",
        "Iop_PwMin16Sx4" => "Iop_PwMin16Sx4",
        "Iop_PwMin32Sx2" => "Iop_PwMin32Sx2",
        "Iop_PwMin8Ux8" => "Iop_PwMin8Ux8",
        "Iop_PwMin16Ux4" => "Iop_PwMin16Ux4",
        "Iop_PwMin32Ux2" => "Iop_PwMin32Ux2",
        "Iop_PwMax8Sx8" => "Iop_PwMax8Sx8",
        "Iop_PwMax16Sx4" => "Iop_PwMax16Sx4",
        "Iop_PwMax32Sx2" => "Iop_PwMax32Sx2",
        "Iop_PwMax8Ux8" => "Iop_PwMax8Ux8",
        "Iop_PwMax16Ux4" => "Iop_PwMax16Ux4",
        "Iop_PwMax32Ux2" => "Iop_PwMax32Ux2",

        // Polynomial multiply (carry-less, NEON crypto-adjacent)
        "Iop_PolynomialMul8x8" => "Iop_PolynomialMul8x8",
        "Iop_PolynomialMul8x16" => "Iop_PolynomialMul8x16",
        "Iop_PolynomialMull8x8" => "Iop_PolynomialMull8x8",

        // Per-lane count operations
        "Iop_Cnt8x8" => "Iop_Cnt8x8",
        "Iop_Cnt8x16" => "Iop_Cnt8x16",
        "Iop_Clz8x8" => "Iop_Clz8x8",
        "Iop_Clz16x4" => "Iop_Clz16x4",
        "Iop_Clz32x2" => "Iop_Clz32x2",
        "Iop_Clz8x16" => "Iop_Clz8x16",
        "Iop_Clz16x8" => "Iop_Clz16x8",
        "Iop_Clz32x4" => "Iop_Clz32x4",
        "Iop_Cls8x8" => "Iop_Cls8x8",
        "Iop_Cls16x4" => "Iop_Cls16x4",
        "Iop_Cls32x2" => "Iop_Cls32x2",
        "Iop_Cls8x16" => "Iop_Cls8x16",
        "Iop_Cls16x8" => "Iop_Cls16x8",
        "Iop_Cls32x4" => "Iop_Cls32x4",

        // Vector shift by *vector* (NEON-only; ShlN/ShrN/SarN are by immediate)
        "Iop_Shl8x8" => "Iop_Shl8x8",
        "Iop_Shl16x4" => "Iop_Shl16x4",
        "Iop_Shl32x2" => "Iop_Shl32x2",
        "Iop_Shl64x1" => "Iop_Shl64x1",
        "Iop_Shl8x16" => "Iop_Shl8x16",
        "Iop_Shl16x8" => "Iop_Shl16x8",
        "Iop_Shl32x4" => "Iop_Shl32x4",
        "Iop_Shl64x2" => "Iop_Shl64x2",
        "Iop_Shr8x8" => "Iop_Shr8x8",
        "Iop_Shr16x4" => "Iop_Shr16x4",
        "Iop_Shr32x2" => "Iop_Shr32x2",
        "Iop_Shr64x1" => "Iop_Shr64x1",
        "Iop_Shr8x16" => "Iop_Shr8x16",
        "Iop_Shr16x8" => "Iop_Shr16x8",
        "Iop_Shr32x4" => "Iop_Shr32x4",
        "Iop_Shr64x2" => "Iop_Shr64x2",
        "Iop_Sar8x8" => "Iop_Sar8x8",
        "Iop_Sar16x4" => "Iop_Sar16x4",
        "Iop_Sar32x2" => "Iop_Sar32x2",
        "Iop_Sar64x1" => "Iop_Sar64x1",
        "Iop_Sar8x16" => "Iop_Sar8x16",
        "Iop_Sar16x8" => "Iop_Sar16x8",
        "Iop_Sar32x4" => "Iop_Sar32x4",
        "Iop_Sar64x2" => "Iop_Sar64x2",
        "Iop_Sal8x8" => "Iop_Sal8x8",
        "Iop_Sal16x4" => "Iop_Sal16x4",
        "Iop_Sal32x2" => "Iop_Sal32x2",
        "Iop_Sal64x1" => "Iop_Sal64x1",
        "Iop_Sal8x16" => "Iop_Sal8x16",
        "Iop_Sal16x8" => "Iop_Sal16x8",
        "Iop_Sal32x4" => "Iop_Sal32x4",
        "Iop_Sal64x2" => "Iop_Sal64x2",

        // Saturating shifts (NEON QShl/QSal/QShlN)
        "Iop_QShl8x8" => "Iop_QShl8x8",
        "Iop_QShl16x4" => "Iop_QShl16x4",
        "Iop_QShl32x2" => "Iop_QShl32x2",
        "Iop_QShl64x1" => "Iop_QShl64x1",
        "Iop_QShl8x16" => "Iop_QShl8x16",
        "Iop_QShl16x8" => "Iop_QShl16x8",
        "Iop_QShl32x4" => "Iop_QShl32x4",
        "Iop_QShl64x2" => "Iop_QShl64x2",
        "Iop_QSal8x8" => "Iop_QSal8x8",
        "Iop_QSal16x4" => "Iop_QSal16x4",
        "Iop_QSal32x2" => "Iop_QSal32x2",
        "Iop_QSal64x1" => "Iop_QSal64x1",
        "Iop_QSal8x16" => "Iop_QSal8x16",
        "Iop_QSal16x8" => "Iop_QSal16x8",
        "Iop_QSal32x4" => "Iop_QSal32x4",
        "Iop_QSal64x2" => "Iop_QSal64x2",

        _ => return None,
    };
    Some(IROp::NeonUnimplemented(op))
}

/// Parse special and x86-specific operations
fn parse_special(op_str: &str) -> Option<IROp> {
    match op_str {
        // x86 PCLMUL
        "Iop_Perm8x8" | "Iop_Perm8x16" | "Iop_Perm8x32" => Some(IROp::VPerm { elem: IRType::I8 }),

        // x86 CRC
        "Iop_Crc32C" => Some(IROp::Crc32C),

        // PCLMUL variants
        "Iop_PclmulLQLQ" => Some(IROp::PclmulLQLQ),
        "Iop_PclmulHQHQ" => Some(IROp::PclmulHQHQ),
        "Iop_PclmulLQHQ" => Some(IROp::PclmulLQHQ),
        "Iop_PclmulHQLQ" => Some(IROp::PclmulHQLQ),

        _ => None,
    }
}

/// Parse an IR type string from pyvex.
pub fn parse_type(ty_str: &str) -> Option<IRType> {
    match ty_str {
        "Ity_I1" => Some(IRType::I1),
        "Ity_I8" => Some(IRType::I8),
        "Ity_I16" => Some(IRType::I16),
        "Ity_I32" => Some(IRType::I32),
        "Ity_I64" => Some(IRType::I64),
        "Ity_I128" => Some(IRType::I128),
        "Ity_F16" => Some(IRType::F16),
        "Ity_F32" => Some(IRType::F32),
        "Ity_F64" => Some(IRType::F64),
        "Ity_F128" | "Ity_D128" => Some(IRType::F80), // Map to closest
        "Ity_V128" => Some(IRType::V128),
        "Ity_V256" => Some(IRType::V256),
        _ => None,
    }
}

/// Parse an endianness string from pyvex.
pub fn parse_endness(end_str: &str) -> super::ir::Endness {
    match end_str {
        "Iend_LE" => super::ir::Endness::Little,
        "Iend_BE" => super::ir::Endness::Big,
        _ => super::ir::Endness::Little, // Default to little endian
    }
}

/// Parse a jump kind string from pyvex.
pub fn parse_jumpkind(jk_str: &str) -> super::ir::JumpKind {
    use super::ir::JumpKind;

    match jk_str {
        "Ijk_Boring" => JumpKind::Boring,
        "Ijk_Call" => JumpKind::Call,
        "Ijk_Ret" => JumpKind::Ret,
        "Ijk_Sys_syscall" => JumpKind::Sys_syscall,
        "Ijk_Sys_int128" => JumpKind::Sys_int128,
        "Ijk_Sys_int129" => JumpKind::Sys_int129,
        "Ijk_Sys_int130" => JumpKind::Sys_int130,
        "Ijk_Sys_int145" => JumpKind::Sys_int145,
        "Ijk_Sys_int210" => JumpKind::Sys_int210,
        "Ijk_Sys_sysenter" => JumpKind::Sys_sysenter,
        "Ijk_ClientReq" => JumpKind::ClientReq,
        "Ijk_Yield" => JumpKind::Yield,
        "Ijk_EmWarn" => JumpKind::EmWarn,
        "Ijk_EmFail" => JumpKind::EmFail,
        "Ijk_NoDecode" => JumpKind::NoDecode,
        "Ijk_MapFail" => JumpKind::MapFail,
        "Ijk_InvalICache" => JumpKind::InvalICache,
        "Ijk_FlushDCache" => JumpKind::FlushDCache,
        _ => JumpKind::Boring,
    }
}

// =============================================================================
// Numeric opcode parsing (for native FFI)
// =============================================================================

/// IROp base value from libvex (Iop_INVALID = 0x1400)
const IOP_BASE: u32 = 0x1400;

/// Convert a numeric IROp code to Rust IROp.
///
/// This is used by the native FFI when converting C IRSB structures.
/// The codes are sequential from 0x1400 (Iop_INVALID).
pub fn parse_opcode_from_u32(code: u32) -> Option<IROp> {
    // Subtract base to get offset
    if code <= IOP_BASE {
        return None; // INVALID or below
    }

    // Common operations are sequential from 0x1401
    // This mapping matches libvex_ir.h enum order exactly
    match code {
        // Add8..Add64: 0x1401-0x1404
        0x1401 => Some(IROp::Add(IRType::I8)),
        0x1402 => Some(IROp::Add(IRType::I16)),
        0x1403 => Some(IROp::Add(IRType::I32)),
        0x1404 => Some(IROp::Add(IRType::I64)),

        // Sub8..Sub64: 0x1405-0x1408
        0x1405 => Some(IROp::Sub(IRType::I8)),
        0x1406 => Some(IROp::Sub(IRType::I16)),
        0x1407 => Some(IROp::Sub(IRType::I32)),
        0x1408 => Some(IROp::Sub(IRType::I64)),

        // Mul8..Mul64: 0x1409-0x140C
        0x1409 => Some(IROp::Mul(IRType::I8)),
        0x140A => Some(IROp::Mul(IRType::I16)),
        0x140B => Some(IROp::Mul(IRType::I32)),
        0x140C => Some(IROp::Mul(IRType::I64)),

        // Or8..Or64: 0x140D-0x1410
        0x140D => Some(IROp::Or(IRType::I8)),
        0x140E => Some(IROp::Or(IRType::I16)),
        0x140F => Some(IROp::Or(IRType::I32)),
        0x1410 => Some(IROp::Or(IRType::I64)),

        // And8..And64: 0x1411-0x1414
        0x1411 => Some(IROp::And(IRType::I8)),
        0x1412 => Some(IROp::And(IRType::I16)),
        0x1413 => Some(IROp::And(IRType::I32)),
        0x1414 => Some(IROp::And(IRType::I64)),

        // Xor8..Xor64: 0x1415-0x1418
        0x1415 => Some(IROp::Xor(IRType::I8)),
        0x1416 => Some(IROp::Xor(IRType::I16)),
        0x1417 => Some(IROp::Xor(IRType::I32)),
        0x1418 => Some(IROp::Xor(IRType::I64)),

        // Shl8..Shl64: 0x1419-0x141C
        0x1419 => Some(IROp::Shl(IRType::I8)),
        0x141A => Some(IROp::Shl(IRType::I16)),
        0x141B => Some(IROp::Shl(IRType::I32)),
        0x141C => Some(IROp::Shl(IRType::I64)),

        // Shr8..Shr64: 0x141D-0x1420
        0x141D => Some(IROp::Shr(IRType::I8)),
        0x141E => Some(IROp::Shr(IRType::I16)),
        0x141F => Some(IROp::Shr(IRType::I32)),
        0x1420 => Some(IROp::Shr(IRType::I64)),

        // Sar8..Sar64: 0x1421-0x1424
        0x1421 => Some(IROp::Sar(IRType::I8)),
        0x1422 => Some(IROp::Sar(IRType::I16)),
        0x1423 => Some(IROp::Sar(IRType::I32)),
        0x1424 => Some(IROp::Sar(IRType::I64)),

        // CmpEQ8..CmpEQ64: 0x1425-0x1428
        0x1425 => Some(IROp::CmpEQ(IRType::I8)),
        0x1426 => Some(IROp::CmpEQ(IRType::I16)),
        0x1427 => Some(IROp::CmpEQ(IRType::I32)),
        0x1428 => Some(IROp::CmpEQ(IRType::I64)),

        // CmpNE8..CmpNE64: 0x1429-0x142C
        0x1429 => Some(IROp::CmpNE(IRType::I8)),
        0x142A => Some(IROp::CmpNE(IRType::I16)),
        0x142B => Some(IROp::CmpNE(IRType::I32)),
        0x142C => Some(IROp::CmpNE(IRType::I64)),

        // Not8..Not64: 0x142D-0x1430
        0x142D => Some(IROp::Not(IRType::I8)),
        0x142E => Some(IROp::Not(IRType::I16)),
        0x142F => Some(IROp::Not(IRType::I32)),
        0x1430 => Some(IROp::Not(IRType::I64)),

        // CasCmpEQ and CasCmpNE: 0x1431-0x143C (same as CmpEQ/CmpNE)
        0x1431 => Some(IROp::CmpEQ(IRType::I8)),
        0x1432 => Some(IROp::CmpEQ(IRType::I16)),
        0x1433 => Some(IROp::CmpEQ(IRType::I32)),
        0x1434 => Some(IROp::CmpEQ(IRType::I64)),
        0x1435 => Some(IROp::CmpNE(IRType::I8)),
        0x1436 => Some(IROp::CmpNE(IRType::I16)),
        0x1437 => Some(IROp::CmpNE(IRType::I32)),
        0x1438 => Some(IROp::CmpNE(IRType::I64)),

        // ExpCmpNE: 0x1439-0x143C (same as CmpNE)
        0x1439 => Some(IROp::CmpNE(IRType::I8)),
        0x143A => Some(IROp::CmpNE(IRType::I16)),
        0x143B => Some(IROp::CmpNE(IRType::I32)),
        0x143C => Some(IROp::CmpNE(IRType::I64)),

        // MullS8..MullS64: 0x143D-0x1440
        0x143D => Some(IROp::MullS(IRType::I8)),
        0x143E => Some(IROp::MullS(IRType::I16)),
        0x143F => Some(IROp::MullS(IRType::I32)),
        0x1440 => Some(IROp::MullS(IRType::I64)),

        // MullU8..MullU64: 0x1441-0x1444
        0x1441 => Some(IROp::MullU(IRType::I8)),
        0x1442 => Some(IROp::MullU(IRType::I16)),
        0x1443 => Some(IROp::MullU(IRType::I32)),
        0x1444 => Some(IROp::MullU(IRType::I64)),

        // Clz64, Clz32: 0x1445-0x1446
        0x1445 => Some(IROp::Clz(IRType::I64)),
        0x1446 => Some(IROp::Clz(IRType::I32)),

        // Ctz64, Ctz32: 0x1447-0x1448
        0x1447 => Some(IROp::Ctz(IRType::I64)),
        0x1448 => Some(IROp::Ctz(IRType::I32)),

        // Skip: Clz32x4, Ctz32x4, Clz64x2, Ctz64x2
        // Skip: PopCount64, PopCount32
        // CmpLT32S..CmpLE64U: 0x1453-0x145A
        0x1453 => Some(IROp::CmpLT(IRType::I32)),
        0x1454 => Some(IROp::CmpLT(IRType::I64)),
        0x1455 => Some(IROp::CmpLE(IRType::I32)),
        0x1456 => Some(IROp::CmpLE(IRType::I64)),
        0x1457 => Some(IROp::CmpLTU(IRType::I32)),
        0x1458 => Some(IROp::CmpLTU(IRType::I64)),
        0x1459 => Some(IROp::CmpLEU(IRType::I32)),
        0x145A => Some(IROp::CmpLEU(IRType::I64)),

        // DivU32..DivS64: 0x1463-0x1466
        0x1463 => Some(IROp::DivU(IRType::I32)),
        0x1464 => Some(IROp::DivS(IRType::I32)),
        0x1465 => Some(IROp::DivU(IRType::I64)),
        0x1466 => Some(IROp::DivS(IRType::I64)),

        // DivModU128to64, DivModS128to64: 0x146A-0x146B
        0x146A => Some(IROp::DivModU128to64),
        0x146B => Some(IROp::DivModS128to64),
        // DivModU64to32, DivModS64to32: 0x146C-0x146D
        0x146C => Some(IROp::DivModU64to32),
        0x146D => Some(IROp::DivModS64to32),

        // Conversions (approximate locations, exact values from libvex_ir.h)
        // 8Uto16: 0x1479, 8Uto32: 0x147A, 8Uto64: 0x147B
        0x1479 => Some(IROp::ZeroExtend {
            from: IRType::I8,
            to: IRType::I16,
        }),
        0x147A => Some(IROp::ZeroExtend {
            from: IRType::I8,
            to: IRType::I32,
        }),
        0x147B => Some(IROp::ZeroExtend {
            from: IRType::I8,
            to: IRType::I64,
        }),

        // 16Uto32: 0x147C, 16Uto64: 0x147D
        0x147C => Some(IROp::ZeroExtend {
            from: IRType::I16,
            to: IRType::I32,
        }),
        0x147D => Some(IROp::ZeroExtend {
            from: IRType::I16,
            to: IRType::I64,
        }),

        // 32Uto64: 0x147E
        0x147E => Some(IROp::ZeroExtend {
            from: IRType::I32,
            to: IRType::I64,
        }),

        // 8Sto16: 0x147F, 8Sto32: 0x1480, 8Sto64: 0x1481
        0x147F => Some(IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I16,
        }),
        0x1480 => Some(IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I32,
        }),
        0x1481 => Some(IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I64,
        }),

        // 16Sto32: 0x1482, 16Sto64: 0x1483
        0x1482 => Some(IROp::SignExtend {
            from: IRType::I16,
            to: IRType::I32,
        }),
        0x1483 => Some(IROp::SignExtend {
            from: IRType::I16,
            to: IRType::I64,
        }),

        // 32Sto64: 0x1484
        0x1484 => Some(IROp::SignExtend {
            from: IRType::I32,
            to: IRType::I64,
        }),

        // Truncations
        // 64to8: 0x1485, 64to16: 0x1486
        0x1485 => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I8,
        }),
        0x1486 => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I16,
        }),

        // 64to32: 0x1487
        0x1487 => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I32,
        }),

        // 32to8: 0x1488, 32to16: 0x1489 (mapped as truncate from I32)
        0x1488 => Some(IROp::Truncate {
            from: IRType::I32,
            to: IRType::I8,
        }),
        0x1489 => Some(IROp::Truncate {
            from: IRType::I32,
            to: IRType::I16,
        }),

        // 64HIto32: 0x148A - extract high 32 bits
        0x148A => Some(IROp::Extract {
            from: IRType::I64,
            to: IRType::I32,
            low_bit: 32,
        }),

        // 32HIto16: 0x148B - extract high 16 bits
        0x148B => Some(IROp::Extract {
            from: IRType::I32,
            to: IRType::I16,
            low_bit: 16,
        }),

        // 16HIto8: 0x148C - extract high 8 bits
        0x148C => Some(IROp::Extract {
            from: IRType::I16,
            to: IRType::I8,
            low_bit: 8,
        }),

        // 1Uto8, 1Uto32, 1Uto64
        0x148D => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I8,
        }),
        0x148E => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I32,
        }),
        0x148F => Some(IROp::ZeroExtend {
            from: IRType::I1,
            to: IRType::I64,
        }),

        // 1Sto8, 1Sto16, 1Sto32, 1Sto64
        0x1490 => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I8,
        }),
        0x1491 => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I16,
        }),
        0x1492 => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I32,
        }),
        0x1493 => Some(IROp::SignExtend {
            from: IRType::I1,
            to: IRType::I64,
        }),

        // 32to1, 64to1
        0x1494 => Some(IROp::Truncate {
            from: IRType::I32,
            to: IRType::I1,
        }),
        0x1495 => Some(IROp::Truncate {
            from: IRType::I64,
            to: IRType::I1,
        }),

        // 8to32, 8to64 (narrowing, but actually same as truncate conceptually)

        // 32HLto64: 0x1498 - concatenate two 32-bit values into 64-bit
        0x1498 => Some(IROp::Concat { ty: IRType::I64 }),

        // 64HLto128: 0x1499 - concatenate two 64-bit values into 128-bit
        0x1499 => Some(IROp::Concat { ty: IRType::I128 }),

        // 128to64, 128HIto64
        0x149A => Some(IROp::Truncate {
            from: IRType::I128,
            to: IRType::I64,
        }),
        0x149B => Some(IROp::Extract {
            from: IRType::I128,
            to: IRType::I64,
            low_bit: 64,
        }),

        // MulHi32U, MulHi64U, MulHi32S, MulHi64S
        0x149C => Some(IROp::MulHi {
            ty: IRType::I32,
            signed: false,
        }),
        0x149D => Some(IROp::MulHi {
            ty: IRType::I64,
            signed: false,
        }),
        0x149E => Some(IROp::MulHi {
            ty: IRType::I32,
            signed: true,
        }),
        0x149F => Some(IROp::MulHi {
            ty: IRType::I64,
            signed: true,
        }),

        // Any other code - return Raw
        _ => None,
    }
}

#[cfg(test)]
mod tests {
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
        // Unknown opcodes should return Raw(0)
        assert_eq!(parse_opcode("Iop_UnknownOp"), IROp::Raw(0));
    }

    #[test]
    fn test_neon_unimplemented_routing() {
        // NEON-only opcodes route through IROp::NeonUnimplemented with the
        // original opcode string captured. Dispatch in VEXOps::unop/binop
        // panics on this variant — the scaffolding makes missing NEON
        // coverage visible immediately instead of silently producing a
        // fresh-symbolic value.
        for op in [
            "Iop_Dup8x8",
            "Iop_NarrowBin16to8x8",
            "Iop_QNarrowBin16Sto8Sx8",
            "Iop_Widen8Sto16x8",
            "Iop_RecipEst32Fx4",
            "Iop_QAdd8Sx8",
            "Iop_Avg8Ux8",
            "Iop_Reverse8sIn32_x2",
            "Iop_PwAdd16x4",
            "Iop_PolynomialMull8x8",
            "Iop_Cnt8x8",
            "Iop_Shl8x16",
        ] {
            match parse_opcode(op) {
                IROp::NeonUnimplemented(name) => assert_eq!(name, op),
                other => panic!("{} expected NeonUnimplemented, got {:?}", op, other),
            }
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
    }
}
