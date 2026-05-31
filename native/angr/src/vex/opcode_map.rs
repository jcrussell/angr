//! Mapping from pyvex opcode strings to Rust IROp enum.
//!
//! pyvex uses string opcodes like "Iop_Add32" while Rust uses parameterized
//! operations like `IROp::Add(IRType::I32)`. This module provides the translation.

use super::ir::{FCmpKind, IROp, IRType};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Intern an unmapped opcode name into a `&'static str` so it can ride in
/// `IROp::Unmapped(&'static str)` without breaking the enum's `Copy`
/// derive. Same string contents always return the same interned pointer —
/// repeated lookups for the same opcode name do not leak duplicates.
///
/// Used only on the unmapped-opcode error path (angr-tkbr.2). The set of
/// unmapped names is small and bounded in practice (a few dozen per arch
/// at most), so the persistent leak is acceptable.
fn intern_unmapped_op(op_str: &str) -> &'static str {
    static INTERN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let intern = INTERN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = intern.lock().expect("unmapped-op intern mutex poisoned");
    if let Some(s) = guard.get(op_str) {
        return *s;
    }
    let leaked: &'static str = Box::leak(op_str.to_string().into_boxed_str());
    guard.insert(leaked);
    leaked
}

// =============================================================================
// Width-family macros (angr-hfr0)
// =============================================================================
//
// The parse_* functions below dispatch hundreds of pyvex opcode strings that
// share a common width-family shape (`"Iop_<Prefix><Suffix>" =>
// IROp::<Variant>(IRType::I<N>)` and friends). The macros below compress each
// width-family into a single block. Each macro:
//
//   1. tries `strip_prefix($prefix)` on the opcode string,
//   2. inner-matches the remaining suffix against the per-arm literal,
//   3. on a hit `return`s `Some(IROp::...)` from the enclosing fn,
//   4. on a miss falls through so the next macro / arm can try.
//
// The `return` is intentional — it short-circuits once a family hits. The
// enclosing parse_* fn must return `Option<IROp>` (which all of them do).
// Macros can safely share a prefix (e.g. "Iop_And" maps both numeric-width
// arms and V128/V256 vector arms); each block only fires when its inner arm
// matches.

/// IROp::Variant(IRType::Ty)
macro_rules! tuple_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => $ty:ident ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant(IRType::$ty)), )*
                _ => {}
            }
        }
    }};
}

/// IROp::Variant { elem: IRType::Ty, count: N }
macro_rules! vec_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant { elem: IRType::$elem, count: $count }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::Variant { elem: IRType::Ty }
macro_rules! scalar_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => $elem:ident ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant { elem: IRType::$elem }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::Variant { from: IRType::From, to: IRType::To }
macro_rules! cast_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $to:ident) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant { from: IRType::$from, to: IRType::$to }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::Variant { elem: IRType::Ty, count: N, signed: bool }
macro_rules! vec_signed_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal, $signed:literal) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant { elem: IRType::$elem, count: $count, signed: $signed }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::VWiden { from: IRType::From, count: N, signed: bool }
macro_rules! vec_widen_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal, $signed:literal) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant { from: IRType::$from, count: $count, signed: $signed }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::VNarrowUn / VNarrowBin { from: IRType::From, count: N }
macro_rules! vec_narrow_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant { from: IRType::$from, count: $count }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::VQNarrowUn / VQNarrowBin { from, count, src_signed, dst_signed }
macro_rules! vec_qnarrow_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal, $src:literal, $dst:literal) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::$variant {
                    from: IRType::$from, count: $count,
                    src_signed: $src, dst_signed: $dst,
                }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::FCmpVecPacked { kind, elem, count }
macro_rules! fcmp_vec_arms {
    ($s:expr; $prefix:literal => $kind:ident { $( $sfx:literal => ($elem:ident, $count:literal) ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::FCmpVecPacked {
                    kind: FCmpKind::$kind, elem: IRType::$elem, count: $count,
                }), )*
                _ => {}
            }
        }
    }};
}

/// IROp::FCmpScalarLane { kind, ty }
macro_rules! fcmp_scalar_arms {
    ($s:expr; $prefix:literal => $kind:ident { $( $sfx:literal => $ty:ident ),* $(,)? }) => {{
        if let Some(rest) = $s.strip_prefix($prefix) {
            match rest {
                $( $sfx => return Some(IROp::FCmpScalarLane {
                    kind: FCmpKind::$kind, ty: IRType::$ty,
                }), )*
                _ => {}
            }
        }
    }};
}

/// Convert a pyvex operation string to Rust IROp.
///
/// Returns `IROp::Unmapped(name)` (interned `&'static str`) for opcodes
/// with no entry in the parse_* dispatch. Dispatch in
/// `VEXOps::unop`/`binop`/`ternop`/`qop` surfaces this as
/// `OpError::UnsupportedVexOp { op_name }`, which the engine maps to
/// `RustUnsupportedVexOpError(op_name, arch)`. Before angr-tkbr.2 this
/// path silently rewrote to `IROp::Raw(0)` and `log::warn!`-ed the
/// name, which masked missing coverage and produced fresh-symbolic
/// results that were hard to attribute.
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
    if let Some(op) = parse_vreverse(op_str) {
        return op;
    }
    if let Some(op) = parse_special(op_str) {
        return op;
    }
    if let Some(op) = parse_neon_unimplemented(op_str) {
        return op;
    }

    // Unmapped operation — capture the name so dispatch can surface
    // RustUnsupportedVexOpError instead of silently producing fresh
    // symbolic results. See angr-tkbr.2.
    log::warn!("Unmapped VEX operation: {}", op_str);
    IROp::Unmapped(intern_unmapped_op(op_str))
}

/// Parse arithmetic operations: Add, Sub, Mul, Div, Mod, Neg, Mull
fn parse_arithmetic(op_str: &str) -> Option<IROp> {
    tuple_arms!(op_str; "Iop_Add"   => Add   { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Sub"   => Sub   { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Mul"   => Mul   { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_MullS" => MullS { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_MullU" => MullU { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_DivS"  => DivS  { "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_DivU"  => DivU  { "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Neg"   => Neg   { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });

    match op_str {
        // DivMod - combined division and modulo
        "Iop_DivModU64to32" => Some(IROp::DivModU64to32),
        "Iop_DivModS64to32" => Some(IROp::DivModS64to32),
        "Iop_DivModU128to64" => Some(IROp::DivModU128to64),
        "Iop_DivModS128to64" => Some(IROp::DivModS128to64),

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
    tuple_arms!(op_str; "Iop_And" => And  { "1" => I1, "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_And" => VAnd { "V128" => V128, "V256" => V256 });
    tuple_arms!(op_str; "Iop_Or"  => Or   { "1" => I1, "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Or"  => VOr  { "V128" => V128, "V256" => V256 });
    tuple_arms!(op_str; "Iop_Xor" => Xor  { "1" => I1, "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Xor" => VXor { "V128" => V128, "V256" => V256 });
    tuple_arms!(op_str; "Iop_Not" => Not  { "1" => I1, "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Not" => VNot { "V128" => V128, "V256" => V256 });
    None
}

/// Parse shift operations: Shl, Shr, Sar (scalar and vector)
fn parse_shift(op_str: &str) -> Option<IROp> {
    tuple_arms!(op_str; "Iop_Shl" => Shl { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Shr" => Shr { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_Sar" => Sar { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });

    // Vector shift {left, right-logical, right-arithmetic} by immediate.
    // NOTE: SarN deliberately omits 64x2 (no NEON SARN.I64 in pyvex).
    vec_arms!(op_str; "Iop_ShlN" => VShlN {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_ShrN" => VShrN {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_SarN" => VSarN {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
    });
    None
}

/// Parse comparison operations
fn parse_comparison(op_str: &str) -> Option<IROp> {
    tuple_arms!(op_str; "Iop_CmpEQ"    => CmpEQ { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_CmpNE"    => CmpNE { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    // ExpCmpNE ("expensive" compare) has identical semantics to CmpNE.
    tuple_arms!(op_str; "Iop_ExpCmpNE" => CmpNE { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    // CasCmpEQ / CasCmpNE: identical semantics to CmpEQ / CmpNE — the "Cas"
    // prefix only signals to optimizers that the operand pattern came from
    // a CAS lowering (see parse_opcode_from_u32's 0x1431-0x1438 arms for
    // the integer-code path). Before angr-tkbr.2 these silently rewrote
    // to IROp::Raw(0) and produced a fresh-symbolic zero, which masked
    // the missing mapping for cmpxchg/cmpxchg16b lifts.
    tuple_arms!(op_str; "Iop_CasCmpEQ" => CmpEQ { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
    tuple_arms!(op_str; "Iop_CasCmpNE" => CmpNE { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });

    tuple_arms!(op_str; "Iop_CmpLT" => CmpLT  { "32S" => I32, "64S" => I64 });
    tuple_arms!(op_str; "Iop_CmpLE" => CmpLE  { "32S" => I32, "64S" => I64 });
    tuple_arms!(op_str; "Iop_CmpLT" => CmpLTU { "32U" => I32, "64U" => I64 });
    tuple_arms!(op_str; "Iop_CmpLE" => CmpLEU { "32U" => I32, "64U" => I64 });

    // Ordered comparison (CmpORD): returns full-width -1 or 0; unmapped for now.
    None
}

/// Parse type conversion operations
fn parse_conversion(op_str: &str) -> Option<IROp> {
    // Sign extensions (NtoM format: extend N-bit to M-bit signed).
    cast_arms!(op_str; "Iop_" => SignExtend {
        "1Sto8"   => (I1, I8),   "1Sto16"  => (I1, I16),  "1Sto32"  => (I1, I32),  "1Sto64"  => (I1, I64),
        "8Sto16"  => (I8, I16),  "8Sto32"  => (I8, I32),  "8Sto64"  => (I8, I64),
        "16Sto32" => (I16, I32), "16Sto64" => (I16, I64),
        "32Sto64" => (I32, I64),
    });
    // Zero extensions (NtoM format: extend N-bit to M-bit unsigned).
    cast_arms!(op_str; "Iop_" => ZeroExtend {
        "1Uto8"   => (I1, I8),   "1Uto16"  => (I1, I16),  "1Uto32"  => (I1, I32),  "1Uto64"  => (I1, I64),
        "8Uto16"  => (I8, I16),  "8Uto32"  => (I8, I32),  "8Uto64"  => (I8, I64),
        "16Uto32" => (I16, I32), "16Uto64" => (I16, I64),
        "32Uto64" => (I32, I64),
    });
    // Truncations (NtoM format: truncate N-bit to M-bit).
    cast_arms!(op_str; "Iop_" => Truncate {
        "64to32" => (I64, I32), "64to16" => (I64, I16), "64to8" => (I64, I8), "64to1" => (I64, I1),
        "32to16" => (I32, I16), "32to8"  => (I32, I8),  "32to1" => (I32, I1),
        "16to8"  => (I16, I8),
        "128to64" => (I128, I64),
    });

    match op_str {
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

        _ => {
            // Bit manipulation
            tuple_arms!(op_str; "Iop_Clz"      => Clz      { "32" => I32, "64" => I64 });
            tuple_arms!(op_str; "Iop_Ctz"      => Ctz      { "32" => I32, "64" => I64 });
            tuple_arms!(op_str; "Iop_PopCount" => PopCount { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
            None
        }
    }
}

/// Parse floating point operations
fn parse_float(op_str: &str) -> Option<IROp> {
    // Basic FP arithmetic + unary + fused multiply-add/sub.
    tuple_arms!(op_str; "Iop_Add"  => FAdd  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Sub"  => FSub  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Mul"  => FMul  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Div"  => FDiv  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Neg"  => FNeg  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Abs"  => FAbs  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Sqrt" => FSqrt { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_MAdd" => FMAdd { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_MSub" => FMSub { "F32" => F32, "F64" => F64 });

    // Scalar-in-vector float ops (SSE scalar: ADDSS, SUBSS, MULSS, DIVSS, etc.).
    scalar_arms!(op_str; "Iop_Add"  => VFAddS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Sub"  => VFSubS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Mul"  => VFMulS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Div"  => VFDivS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Sqrt" => VFSqrtS { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Max"  => VFMaxS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Min"  => VFMinS  { "32F0x4" => F32, "64F0x2" => F64 });

    // Packed (whole-vector) float ops — SSE / AVX.
    vec_arms!(op_str; "Iop_Add"  => VFAdd  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Sub"  => VFSub  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Mul"  => VFMul  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Div"  => VFDiv  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Sqrt" => VFSqrt { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Abs"  => VFAbs  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Min"  => VFMin  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Max"  => VFMax  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });

    // FP reciprocal estimate (1/x) — RCPPS / NEON FRECPE. F0x4 = SSE scalar.
    scalar_arms!(op_str; "Iop_RecipEst" => VFRecipEstS { "32F0x4" => F32 });
    vec_arms!(op_str; "Iop_RecipEst" => VFRecipEst {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "32Fx8" => (F32, 8), "64Fx2" => (F64, 2),
    });
    // FP Newton-Raphson reciprocal step — NEON FRECPS.
    vec_arms!(op_str; "Iop_RecipStep" => VFRecipStep {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2),
    });
    // FP reciprocal-sqrt estimate (1/sqrt(x)) — RSQRTPS / NEON FRSQRTE.
    scalar_arms!(op_str; "Iop_RSqrtEst" => VFRSqrtEstS { "32F0x4" => F32 });
    vec_arms!(op_str; "Iop_RSqrtEst" => VFRSqrtEst {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "32Fx8" => (F32, 8), "64Fx2" => (F64, 2),
    });
    // FP Newton-Raphson reciprocal-sqrt step — NEON FRSQRTS.
    vec_arms!(op_str; "Iop_RSqrtStep" => VFRSqrtStep {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2),
    });

    // FP comparisons (scalar I1 result — used internally and for ccall lifts).
    tuple_arms!(op_str; "Iop_CmpF" => FComCC { "32" => F32, "64" => F64 });

    // SSE scalar-lane compares: lane 0 → all-1s/0 mask, upper lanes pass-through.
    fcmp_scalar_arms!(op_str; "Iop_CmpEQ" => Eq { "32F0x4" => F32, "64F0x2" => F64 });
    fcmp_scalar_arms!(op_str; "Iop_CmpLT" => Lt { "32F0x4" => F32, "64F0x2" => F64 });
    fcmp_scalar_arms!(op_str; "Iop_CmpLE" => Le { "32F0x4" => F32, "64F0x2" => F64 });
    fcmp_scalar_arms!(op_str; "Iop_CmpUN" => Un { "32F0x4" => F32, "64F0x2" => F64 });

    // Packed FP compares (SSE cmpps/cmppd, ARM NEON 32Fx2). Per-lane mask:
    // each lane independently produces all-1s (true) or 0 (false).
    // 32Fx2 returns I64 (ARM NEON), 32Fx4 / 64Fx2 return V128 (SSE).
    fcmp_vec_arms!(op_str; "Iop_CmpEQ" => Eq { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    fcmp_vec_arms!(op_str; "Iop_CmpLT" => Lt { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    fcmp_vec_arms!(op_str; "Iop_CmpLE" => Le { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    fcmp_vec_arms!(op_str; "Iop_CmpGT" => Gt { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4) });
    fcmp_vec_arms!(op_str; "Iop_CmpGE" => Ge { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4) });
    fcmp_vec_arms!(op_str; "Iop_CmpUN" => Un { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });

    // Reinterpret as different type.
    cast_arms!(op_str; "Iop_Reinterp" => Reinterpret {
        "F32asI32" => (F32, I32), "I32asF32" => (I32, F32),
        "F64asI64" => (F64, I64), "I64asF64" => (I64, F64),
    });

    match op_str {
        // SetV128lo operations
        "Iop_SetV128lo32" => Some(IROp::SetV128lo32),
        "Iop_SetV128lo64" => Some(IROp::SetV128lo64),

        // FP conversions (these are nullary tuple variants — no IRType inside).
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

        _ => None,
    }
}

/// Parse vector/SIMD operations
fn parse_vector(op_str: &str) -> Option<IROp> {
    // Vector add/sub: 8/16/32/64-bit elements across NEON-D / NEON-Q / AVX widths.
    vec_arms!(op_str; "Iop_Add" => VAdd {
        "8x8" => (I8, 8), "8x16" => (I8, 16), "8x32" => (I8, 32),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });
    vec_arms!(op_str; "Iop_Sub" => VSub {
        "8x8" => (I8, 8), "8x16" => (I8, 16), "8x32" => (I8, 32),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });
    // Vector multiply: 8-bit is NEON-only (VMUL.I8); 16/32-bit are SSE+NEON.
    vec_arms!(op_str; "Iop_Mul" => VMul {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
    });
    // Vector multiply keeping low half (PMULLD - SSE4.1)
    vec_arms!(op_str; "Iop_MullS" => VMulLo { "32x4" => (I32, 4) });

    // NEON lane extract / insert — Iop_{Get,Set}Elem{N}x{M}: (vec, idx[, val]).
    vec_arms!(op_str; "Iop_GetElem" => VGetElem {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_SetElem" => VSetElem {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    // NEON broadcast scalar to vector — Iop_Dup{N}x{M}.
    vec_arms!(op_str; "Iop_Dup" => VDup {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON widen lane element width — Iop_Widen{N}{S/U}to{2N}x{M}.
    vec_widen_arms!(op_str; "Iop_Widen" => VWiden {
        "8Sto16x8"  => (I8, 8, true),  "8Uto16x8"  => (I8, 8, false),
        "16Sto32x4" => (I16, 4, true), "16Uto32x4" => (I16, 4, false),
        "32Sto64x2" => (I32, 2, true), "32Uto64x2" => (I32, 2, false),
    });

    // NEON narrow (truncating) — Iop_Narrow{Un,Bin}{N}to{N/2}x{M}.
    vec_narrow_arms!(op_str; "Iop_NarrowUn" => VNarrowUn {
        "16to8x8" => (I16, 8), "32to16x4" => (I32, 4), "64to32x2" => (I64, 2),
    });
    vec_narrow_arms!(op_str; "Iop_NarrowBin" => VNarrowBin {
        "16to8x8" => (I16, 8), "32to16x4" => (I32, 4),
        "16to8x16" => (I16, 16), "32to16x8" => (I32, 8), "64to32x4" => (I64, 4),
    });

    // NEON saturating narrow — Iop_QNarrow{Un,Bin}{N}{S/U}to{N/2}{S/U}x{M}.
    vec_qnarrow_arms!(op_str; "Iop_QNarrowUn" => VQNarrowUn {
        "16Sto8Sx8"  => (I16, 8, true,  true),
        "16Sto8Ux8"  => (I16, 8, true,  false),
        "16Uto8Ux8"  => (I16, 8, false, false),
        "32Sto16Sx4" => (I32, 4, true,  true),
        "32Sto16Ux4" => (I32, 4, true,  false),
        "32Uto16Ux4" => (I32, 4, false, false),
        "64Sto32Sx2" => (I64, 2, true,  true),
        "64Sto32Ux2" => (I64, 2, true,  false),
        "64Uto32Ux2" => (I64, 2, false, false),
    });
    vec_qnarrow_arms!(op_str; "Iop_QNarrowBin" => VQNarrowBin {
        "16Sto8Sx8"  => (I16, 8,  true,  true),
        "16Sto8Ux8"  => (I16, 8,  true,  false),
        "32Sto16Sx4" => (I32, 4,  true,  true),
        "32Sto16Ux4" => (I32, 4,  true,  false),
        "16Sto8Sx16" => (I16, 16, true,  true),
        "16Sto8Ux16" => (I16, 16, true,  false),
        "16Uto8Ux16" => (I16, 16, false, false),
        "32Sto16Sx8" => (I32, 8,  true,  true),
        "32Sto16Ux8" => (I32, 8,  true,  false),
        "32Uto16Ux8" => (I32, 8,  false, false),
        "64Sto32Sx4" => (I64, 4,  true,  true),
        "64Uto32Ux4" => (I64, 4,  false, false),
    });

    // Vector compare equal / greater-than (signed).
    vec_arms!(op_str; "Iop_CmpEQ" => VCmpEQ {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
        "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_CmpGT" => VCmpGT {
        "8Sx8" => (I8, 8), "8Sx16" => (I8, 16),
        "16Sx4" => (I16, 4), "16Sx8" => (I16, 8),
        "32Sx2" => (I32, 2), "32Sx4" => (I32, 4),
        "64Sx2" => (I64, 2),
    });

    // Vector interleave — count is implicit from elem.
    scalar_arms!(op_str; "Iop_InterleaveLO" => VInterleaveLO {
        "8x8" => I8, "8x16" => I8,
        "16x4" => I16, "16x8" => I16,
        "32x2" => I32, "32x4" => I32,
        "64x2" => I64,
    });
    scalar_arms!(op_str; "Iop_InterleaveHI" => VInterleaveHI {
        "8x8" => I8, "8x16" => I8,
        "16x4" => I16, "16x8" => I16,
        "32x2" => I32, "32x4" => I32,
        "64x2" => I64,
    });

    // Packed integer min/max — signed (S suffix) and unsigned (U suffix).
    vec_signed_arms!(op_str; "Iop_Min" => VMin {
        "8Sx8" => (I8, 8, true), "8Sx16" => (I8, 16, true), "8Sx32" => (I8, 32, true),
        "16Sx4" => (I16, 4, true), "16Sx8" => (I16, 8, true), "16Sx16" => (I16, 16, true),
        "32Sx2" => (I32, 2, true), "32Sx4" => (I32, 4, true), "32Sx8" => (I32, 8, true),
        "64Sx2" => (I64, 2, true), "64Sx4" => (I64, 4, true),
        "8Ux8" => (I8, 8, false), "8Ux16" => (I8, 16, false), "8Ux32" => (I8, 32, false),
        "16Ux4" => (I16, 4, false), "16Ux8" => (I16, 8, false), "16Ux16" => (I16, 16, false),
        "32Ux2" => (I32, 2, false), "32Ux4" => (I32, 4, false), "32Ux8" => (I32, 8, false),
        "64Ux2" => (I64, 2, false), "64Ux4" => (I64, 4, false),
    });
    vec_signed_arms!(op_str; "Iop_Max" => VMax {
        "8Sx8" => (I8, 8, true), "8Sx16" => (I8, 16, true), "8Sx32" => (I8, 32, true),
        "16Sx4" => (I16, 4, true), "16Sx8" => (I16, 8, true), "16Sx16" => (I16, 16, true),
        "32Sx2" => (I32, 2, true), "32Sx4" => (I32, 4, true), "32Sx8" => (I32, 8, true),
        "64Sx2" => (I64, 2, true), "64Sx4" => (I64, 4, true),
        "8Ux8" => (I8, 8, false), "8Ux16" => (I8, 16, false), "8Ux32" => (I8, 32, false),
        "16Ux4" => (I16, 4, false), "16Ux8" => (I16, 8, false), "16Ux16" => (I16, 16, false),
        "32Ux2" => (I32, 2, false), "32Ux4" => (I32, 4, false), "32Ux8" => (I32, 8, false),
        "64Ux2" => (I64, 2, false), "64Ux4" => (I64, 4, false),
    });

    // Packed integer absolute value — Iop_Abs{N}x{M}.
    vec_arms!(op_str; "Iop_Abs" => VAbs {
        "8x8" => (I8, 8), "8x16" => (I8, 16), "8x32" => (I8, 32),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });

    // V128/V256 to/from conversions.
    cast_arms!(op_str; "Iop_" => Truncate    { "V128to64"   => (V128, I64) });
    cast_arms!(op_str; "Iop_" => ZeroExtend  { "64UtoV128"  => (I64, V128), "32UtoV128" => (I32, V128) });
    cast_arms!(op_str; "Iop_" => Reinterpret { "SetV128lo64" => (I64, V128) });

    match op_str {
        "Iop_V128HIto64" => Some(IROp::Extract {
            from: IRType::V128,
            to: IRType::I64,
            low_bit: 64,
        }),
        _ => None,
    }
}

/// Parse NEON byte/halfword/word/bit reversal opcodes —
/// `Iop_Reverse{sub_width}sIn{elem_width}_x{count}`. Returns
/// `IROp::VReverse { sub_width, elem, count }` which dispatches through
/// `VEXOps::unop` to `vec_reverse`. Implemented in angr-tukg.4.
///
/// ARM ISA mapping (DDI 0487 C7.2.288 / C7.2.297-300):
///   - `Reverse1sIn8_*`   → RBIT (bit reverse within each byte)
///   - `Reverse8sIn16_*`  → REV16 (byte swap within halfwords)
///   - `Reverse8sIn32_*`  → REV32 (byte swap within words)
///   - `Reverse8sIn64_*`  → REV64 (byte swap within doublewords)
///   - `Reverse16sIn32_*` → REV32 (halfword swap within words)
///   - `Reverse16sIn64_*` → REV64 (halfword swap within doublewords)
///   - `Reverse32sIn64_*` → REV64 (word swap within doublewords)
fn parse_vreverse(op_str: &str) -> Option<IROp> {
    let (sub_width, elem, count) = match op_str {
        // Byte reversal within 16-bit halfwords (REV16).
        "Iop_Reverse8sIn16_x4" => (8u8, IRType::I16, 4u8),
        "Iop_Reverse8sIn16_x8" => (8, IRType::I16, 8),
        // Byte reversal within 32-bit words (REV32 / x86 BSWAP-like).
        "Iop_Reverse8sIn32_x2" => (8, IRType::I32, 2),
        "Iop_Reverse8sIn32_x4" => (8, IRType::I32, 4),
        // Byte reversal within 64-bit doublewords (REV64).
        "Iop_Reverse8sIn64_x1" => (8, IRType::I64, 1),
        "Iop_Reverse8sIn64_x2" => (8, IRType::I64, 2),
        // Halfword reversal within 32-bit words.
        "Iop_Reverse16sIn32_x2" => (16, IRType::I32, 2),
        "Iop_Reverse16sIn32_x4" => (16, IRType::I32, 4),
        // Halfword reversal within 64-bit doublewords.
        "Iop_Reverse16sIn64_x1" => (16, IRType::I64, 1),
        "Iop_Reverse16sIn64_x2" => (16, IRType::I64, 2),
        // Word swap within 64-bit doublewords.
        "Iop_Reverse32sIn64_x1" => (32, IRType::I64, 1),
        "Iop_Reverse32sIn64_x2" => (32, IRType::I64, 2),
        // Bit reversal within each byte (RBIT).
        "Iop_Reverse1sIn8_x8" => (1, IRType::I8, 8),
        "Iop_Reverse1sIn8_x16" => (1, IRType::I8, 16),
        _ => return None,
    };
    Some(IROp::VReverse {
        sub_width,
        elem,
        count,
    })
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
        // NOTE: Iop_GetElem* / Iop_SetElem* (lane extract/insert) implemented
        // in angr-bkcs.2 — routed through parse_vector to IROp::VGetElem /
        // IROp::VSetElem above. Iop_Dup* / Iop_Widen* / Iop_Narrow{Bin,Un}* /
        // Iop_QNarrow{Bin,Un}* implemented in angr-hzs0 — routed through
        // parse_vector to IROp::VDup / IROp::VWiden / IROp::VNarrow{Un,Bin} /
        // IROp::VQNarrow{Un,Bin}.

        // NOTE: FP RecipEst / RecipStep / RSqrtEst / RSqrtStep ({32,64}{F0,Fx}*)
        // implemented in angr-iyon — routed through parse_float to
        // IROp::VFRecipEst{,S} / VFRecipStep / VFRSqrtEst{,S} / VFRSqrtStep.

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

        // NOTE: Iop_Reverse{N}sIn{M}_x{K} (byte/halfword/word/bit reversal
        // within lane) implemented in angr-tukg.4 — routed through
        // parse_vreverse to IROp::VReverse.

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
        // Unknown opcodes now route through IROp::Unmapped (angr-tkbr.2).
        // Dispatch in VEXOps::unop/binop/triop/qop surfaces this as
        // OpError::UnsupportedVexOp, which the engine maps to
        // RustUnsupportedVexOpError.
        match parse_opcode("Iop_UnknownOp") {
            IROp::Unmapped(name) => assert_eq!(name, "Iop_UnknownOp"),
            other => panic!("expected Unmapped, got {:?}", other),
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
    fn test_neon_unimplemented_routing() {
        // NEON-only opcodes route through IROp::NeonUnimplemented with the
        // original opcode string captured. Dispatch in VEXOps::unop/binop
        // panics on this variant — the scaffolding makes missing NEON
        // coverage visible immediately instead of silently producing a
        // fresh-symbolic value.
        for op in [
            "Iop_RecipEst32Ux4",
            "Iop_QAdd8Sx8",
            "Iop_Avg8Ux8",
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
                    assert_eq!(sub_width, *sw, "{}: sub_width", op_str);
                    assert_eq!(elem, *e, "{}: elem", op_str);
                    assert_eq!(count, *c, "{}: count", op_str);
                }
                other => panic!("{}: expected VReverse, got {:?}", op_str, other),
            }
        }
    }
}
