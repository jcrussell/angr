//! Mapping from pyvex opcode strings to Rust IROp enum.
//!
//! pyvex uses string opcodes like "Iop_Add32" while Rust uses parameterized
//! operations like `IROp::Add(IRType::I32)`. This module provides the translation.

use super::ir::{IROp, IRType};

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

/// Parse shift operations: Shl, Shr, Sar
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
        "Iop_16HLto32" => Some(IROp::Concat { ty: IRType::I32 }),
        "Iop_8HLto16" => Some(IROp::Concat { ty: IRType::I16 }),

        // Bit manipulation
        "Iop_Clz32" => Some(IROp::Clz(IRType::I32)),
        "Iop_Clz64" => Some(IROp::Clz(IRType::I64)),
        "Iop_Ctz32" => Some(IROp::Ctz(IRType::I32)),
        "Iop_Ctz64" => Some(IROp::Ctz(IRType::I64)),
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

        // FP comparisons
        "Iop_CmpEQ32F0x4" | "Iop_CmpF32" => Some(IROp::FCmpEQ(IRType::F32)),
        "Iop_CmpEQ64F0x2" | "Iop_CmpF64" => Some(IROp::FCmpEQ(IRType::F64)),
        "Iop_CmpLT32F0x4" => Some(IROp::FCmpLT(IRType::F32)),
        "Iop_CmpLT64F0x2" => Some(IROp::FCmpLT(IRType::F64)),
        "Iop_CmpLE32F0x4" => Some(IROp::FCmpLE(IRType::F32)),
        "Iop_CmpLE64F0x2" => Some(IROp::FCmpLE(IRType::F64)),

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
        "Iop_InterleaveLO8x8" | "Iop_InterleaveLO8x16" => Some(IROp::VInterleaveLO {
            elem: IRType::I8,
        }),
        "Iop_InterleaveLO16x4" | "Iop_InterleaveLO16x8" => Some(IROp::VInterleaveLO {
            elem: IRType::I16,
        }),
        "Iop_InterleaveLO32x2" | "Iop_InterleaveLO32x4" => Some(IROp::VInterleaveLO {
            elem: IRType::I32,
        }),
        "Iop_InterleaveLO64x2" => Some(IROp::VInterleaveLO { elem: IRType::I64 }),

        "Iop_InterleaveHI8x8" | "Iop_InterleaveHI8x16" => Some(IROp::VInterleaveHI {
            elem: IRType::I8,
        }),
        "Iop_InterleaveHI16x4" | "Iop_InterleaveHI16x8" => Some(IROp::VInterleaveHI {
            elem: IRType::I16,
        }),
        "Iop_InterleaveHI32x2" | "Iop_InterleaveHI32x4" => Some(IROp::VInterleaveHI {
            elem: IRType::I32,
        }),
        "Iop_InterleaveHI64x2" => Some(IROp::VInterleaveHI { elem: IRType::I64 }),

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
        "Iop_SetV128lo64" => Some(IROp::Reinterpret {
            from: IRType::I64,
            to: IRType::V128,
        }),

        _ => None,
    }
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
        assert_eq!(parse_jumpkind("Ijk_Boring"), super::super::ir::JumpKind::Boring);
        assert_eq!(parse_jumpkind("Ijk_Call"), super::super::ir::JumpKind::Call);
        assert_eq!(parse_jumpkind("Ijk_Ret"), super::super::ir::JumpKind::Ret);
    }

    #[test]
    fn test_unmapped_opcode() {
        // Unknown opcodes should return Raw(0)
        assert_eq!(parse_opcode("Iop_UnknownOp"), IROp::Raw(0));
    }
}
