//! Mapping from pyvex opcode strings to Rust IROp enum.
//!
//! pyvex uses string opcodes like "Iop_Add32" while Rust uses parameterized
//! operations like `IROp::Add(IRType::I32)`. This module provides the translation.
//!
//! **Panic policy (angr-9ke6b.212):** opcode strings arrive from pyvex, so an
//! unrecognized name must never panic — it interns into
//! `IROp::Unmapped(&'static str)` and surfaces as a typed error downstream. The
//! one `expect` is the intern-set mutex poison guard, unreachable under the
//! crate's `panic = "abort"` profile (workspace `Cargo.toml`, angr-1cue).
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

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
#[allow(
    clippy::expect_used,
    reason = "`INTERN` poison guard: poison requires an unwind out of a live `MutexGuard`, which `panic = \"abort\"` forecloses — see the module Panic policy header"
)]
fn intern_unmapped_op(op_str: &str) -> &'static str {
    static INTERN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let intern = INTERN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = intern.lock().expect("unmapped-op intern mutex poisoned");
    if let Some(s) = guard.get(op_str) {
        return s;
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
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant(IRType::$ty)), )*
            _ => {}
        }
    }};
}

/// IROp::Variant { elem: IRType::Ty, count: N }
macro_rules! vec_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { elem: IRType::$elem, count: $count }), )*
            _ => {}
        }
    }};
}

/// IROp::Variant { elem: IRType::Ty }
macro_rules! scalar_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => $elem:ident ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { elem: IRType::$elem }), )*
            _ => {}
        }
    }};
}

/// IROp::Variant { from: IRType::From, to: IRType::To }
macro_rules! cast_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $to:ident) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { from: IRType::$from, to: IRType::$to }), )*
            _ => {}
        }
    }};
}

/// IROp::Variant { elem: IRType::Ty, count: N, signed: bool }
macro_rules! vec_signed_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal, $signed:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { elem: IRType::$elem, count: $count, signed: $signed }), )*
            _ => {}
        }
    }};
}

/// IROp::VMull { elem: IRType::Ty, count: N, signed: bool, even: bool }
///
/// vec_signed_arms! plus one `even` bool — the same one-bool generalization
/// vec_qnarrow_arms! applies to vec_narrow_arms!. Both Mull families share the
/// {elem,count,signed} shape; `even` is baked per-invocation via the arm table
/// so the full-lane ("Iop_Mull") and even-lane ("Iop_MullEven") prefixes reuse
/// the same macro.
macro_rules! vec_mull_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal, $signed:literal, $even:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant {
                elem: IRType::$elem, count: $count, signed: $signed, even: $even,
            }), )*
            _ => {}
        }
    }};
}

/// IROp::VWiden { from: IRType::From, count: N, signed: bool }
macro_rules! vec_widen_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal, $signed:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { from: IRType::$from, count: $count, signed: $signed }), )*
            _ => {}
        }
    }};
}

/// IROp::VNarrowUn / VNarrowBin { from: IRType::From, count: N }
macro_rules! vec_narrow_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { from: IRType::$from, count: $count }), )*
            _ => {}
        }
    }};
}

/// IROp::VQNarrowUn / VQNarrowBin { from, count, src_signed, dst_signed }
macro_rules! vec_qnarrow_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal, $src:literal, $dst:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant {
                from: IRType::$from, count: $count,
                src_signed: $src, dst_signed: $dst,
            }), )*
            _ => {}
        }
    }};
}

/// IROp::FCmpVecPacked { kind, elem, count }
macro_rules! fcmp_vec_arms {
    ($s:expr; $prefix:literal => $kind:ident { $( $sfx:literal => ($elem:ident, $count:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::FCmpVecPacked {
                kind: FCmpKind::$kind, elem: IRType::$elem, count: $count,
            }), )*
            _ => {}
        }
    }};
}

/// IROp::FCmpScalarLane { kind, ty }
macro_rules! fcmp_scalar_arms {
    ($s:expr; $prefix:literal => $kind:ident { $( $sfx:literal => $ty:ident ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::FCmpScalarLane {
                kind: FCmpKind::$kind, ty: IRType::$ty,
            }), )*
            _ => {}
        }
    }};
}

/// Convert a pyvex operation string to Rust IROp.
///
/// Returns `IROp::Unmapped(name)` (interned `&'static str`) for opcodes
/// with no entry in the parse_* dispatch. Dispatch in
/// `VEXOps::unop`/`binop`/`ternop`/`qop` surfaces this as
/// `OpError::UnsupportedVexOp { op_name }`. On the test-only path that
/// maps to a typed `RustUnsupportedVexOpError(op_name, arch)`; in live
/// exploration it is stringified into the errored stash (see the
/// "Test-only taxonomy vs. the live exploration path" note in
/// `errors.rs`). Before angr-tkbr.2 this
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

    // Unmapped operation — capture the name so dispatch can surface a
    // typed UnsupportedVexOp error instead of silently producing fresh
    // symbolic results. See angr-tkbr.2.
    log::warn!("Unmapped VEX operation: {op_str}");
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
    // a CAS lowering. Before angr-tkbr.2 these silently rewrote
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

    // NEON pairwise FP add — `Iop_PwAdd32Fx2` (ARM VPADD.F32, D-reg). The only
    // FP variant of the `Pw*` family; the integer `Iop_PwAdd{N}x{M}` are routed
    // in parse_vector to VPwAdd. VEX emits only the 32Fx2 shape. Matched here
    // (parse_float runs before parse_neon_unimplemented in parse_opcode).
    vec_arms!(op_str; "Iop_PwAdd" => VFPwAdd { "32Fx2" => (F32, 2) });

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

    // NEON saturating add/sub — Iop_QAdd{N}{S/U}x{M} / Iop_QSub{N}{S/U}x{M}.
    // Per-lane saturating arithmetic; S/U selects clamp range. D-reg
    // (total=64) and Q-reg (total=128) variants.
    vec_signed_arms!(op_str; "Iop_QAdd" => VQAdd {
        "8Sx8" => (I8, 8, true), "16Sx4" => (I16, 4, true),
        "32Sx2" => (I32, 2, true), "64Sx1" => (I64, 1, true),
        "8Ux8" => (I8, 8, false), "16Ux4" => (I16, 4, false),
        "32Ux2" => (I32, 2, false), "64Ux1" => (I64, 1, false),
        "8Sx16" => (I8, 16, true), "16Sx8" => (I16, 8, true),
        "32Sx4" => (I32, 4, true), "64Sx2" => (I64, 2, true),
        "8Ux16" => (I8, 16, false), "16Ux8" => (I16, 8, false),
        "32Ux4" => (I32, 4, false), "64Ux2" => (I64, 2, false),
    });
    vec_signed_arms!(op_str; "Iop_QSub" => VQSub {
        "8Sx8" => (I8, 8, true), "16Sx4" => (I16, 4, true),
        "32Sx2" => (I32, 2, true), "64Sx1" => (I64, 1, true),
        "8Ux8" => (I8, 8, false), "16Ux4" => (I16, 4, false),
        "32Ux2" => (I32, 2, false), "64Ux1" => (I64, 1, false),
        "8Sx16" => (I8, 16, true), "16Sx8" => (I16, 8, true),
        "32Sx4" => (I32, 4, true), "64Sx2" => (I64, 2, true),
        "8Ux16" => (I8, 16, false), "16Ux8" => (I16, 8, false),
        "32Ux4" => (I32, 4, false), "64Ux2" => (I64, 2, false),
    });

    // NEON vector shift by vector — `Iop_Shl{N}x{M}` / `Iop_Shr{N}x{M}` /
    // `Iop_Sar{N}x{M}` / `Iop_Sal{N}x{M}`. Both operands are full-vector;
    // each lane shifts by the corresponding count lane (Z3 bvshl/bvlshr/bvashr
    // semantics — counts ≥ lane width produce zero or sign-fill). `Sal` shares
    // semantics with `Shl` on two's complement; libVEX emits both names from
    // ARM SSHL/USHL decomposition (positive-count branches).
    vec_arms!(op_str; "Iop_Shl" => VShl {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2), "64x1" => (I64, 1),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_Sal" => VShl {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2), "64x1" => (I64, 1),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_Shr" => VShr {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2), "64x1" => (I64, 1),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });
    vec_arms!(op_str; "Iop_Sar" => VSar {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2), "64x1" => (I64, 1),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4), "64x2" => (I64, 2),
    });

    // NEON saturating shift-left by vector — `Iop_QShl{N}x{M}` (unsigned) /
    // `Iop_QSal{N}x{M}` (signed). Signedness is encoded in the prefix, not an
    // S/U infix, so it can't vary within one arm table — but `vec_signed_arms!`
    // still fits: route each prefix through its own invocation with the sign
    // baked as a constant into every arm. Maps to ARM UQSHL / SQSHL.
    vec_signed_arms!(op_str; "Iop_QShl" => VQShlSat {
        "8x8" => (I8, 8, false), "16x4" => (I16, 4, false),
        "32x2" => (I32, 2, false), "64x1" => (I64, 1, false),
        "8x16" => (I8, 16, false), "16x8" => (I16, 8, false),
        "32x4" => (I32, 4, false), "64x2" => (I64, 2, false),
    });
    vec_signed_arms!(op_str; "Iop_QSal" => VQShlSat {
        "8x8" => (I8, 8, true), "16x4" => (I16, 4, true),
        "32x2" => (I32, 2, true), "64x1" => (I64, 1, true),
        "8x16" => (I8, 16, true), "16x8" => (I16, 8, true),
        "32x4" => (I32, 4, true), "64x2" => (I64, 2, true),
    });
    // NEON pairwise add — `Iop_PwAdd{N}x{M}` (no signedness, binary). Output
    // has the same lane shape as the inputs; first half from a, second half
    // from b. Iop_PwAdd32Fx2 is the float variant routed via parse_float to
    // IROp::VFPwAdd, NOT here.
    vec_arms!(op_str; "Iop_PwAdd" => VPwAdd {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON pairwise widening add — `Iop_PwAddL{N}{S/U}x{M}` (unary). Lane
    // width doubles and count halves; total preserved.
    vec_signed_arms!(op_str; "Iop_PwAddL" => VPwAddL {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
        "8Sx16" => (I8, 16, true), "8Ux16" => (I8, 16, false),
        "16Sx8" => (I16, 8, true), "16Ux8" => (I16, 8, false),
        "32Sx4" => (I32, 4, true), "32Ux4" => (I32, 4, false),
    });

    // NEON pairwise integer min/max — `Iop_PwMin{N}{S/U}x{M}` /
    // `Iop_PwMax{N}{S/U}x{M}`. Same shape as VPwAdd; D-reg-only (no x16/x8/x4
    // emitted by libVEX for these).
    vec_signed_arms!(op_str; "Iop_PwMin" => VPwMin {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
    });
    vec_signed_arms!(op_str; "Iop_PwMax" => VPwMax {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
    });

    // NEON rounding halving add (a.k.a. rounding-average) —
    // `Iop_Avg{N}{S/U}x{M}`. Binary; output shape matches inputs. Both D-reg
    // (total=64) and Q-reg (total=128) variants exist for 8/16/32-bit lanes.
    // Maps to ARM URHADD/SRHADD; SSE PAVGB/PAVGW (unsigned-only) lifts here too.
    vec_signed_arms!(op_str; "Iop_Avg" => VAvg {
        "8Sx8"  => (I8, 8, true),  "8Ux8"  => (I8, 8, false),
        "16Sx4" => (I16, 4, true), "16Ux4" => (I16, 4, false),
        "32Sx2" => (I32, 2, true), "32Ux2" => (I32, 2, false),
        "8Sx16" => (I8, 16, true), "8Ux16" => (I8, 16, false),
        "16Sx8" => (I16, 8, true), "16Ux8" => (I16, 8, false),
        "32Sx4" => (I32, 4, true), "32Ux4" => (I32, 4, false),
    });

    // NEON per-byte popcount — `Iop_Cnt8x{8,16}` (unary, 8-bit lanes only).
    // ARM CNT (DDI 0487 C7.2.62).
    match op_str {
        "Iop_Cnt8x8" => return Some(IROp::VCnt { count: 8 }),
        "Iop_Cnt8x16" => return Some(IROp::VCnt { count: 16 }),
        _ => {}
    }

    // SSE byte-mask extract — `Iop_GetMSBs8x{8,16}` (x86 PMOVMSKB, unary).
    // Reduces a vector of N bytes to an N-bit integer of their MSBs.
    match op_str {
        "Iop_GetMSBs8x8" => return Some(IROp::VGetMSBs { count: 8 }),
        "Iop_GetMSBs8x16" => return Some(IROp::VGetMSBs { count: 16 }),
        _ => {}
    }

    // NEON per-lane count leading zeros — `Iop_Clz{N}x{M}` (unary). ARM CLZ
    // (DDI 0487 C7.2.57). D-reg (total=64) and Q-reg (total=128) shapes for
    // 8/16/32-bit lanes.
    vec_arms!(op_str; "Iop_Clz" => VClz {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON per-lane count leading sign bits — `Iop_Cls{N}x{M}` (unary). ARM
    // CLS (DDI 0487 C7.2.56). Same shapes as Clz.
    vec_arms!(op_str; "Iop_Cls" => VCls {
        "8x8" => (I8, 8), "16x4" => (I16, 4), "32x2" => (I32, 2),
        "8x16" => (I8, 16), "16x8" => (I16, 8), "32x4" => (I32, 4),
    });

    // NEON GF(2) polynomial multiply — `Iop_PolynomialMul8x{8,16}` (non-
    // widening) and `Iop_PolynomialMull8x8` (widening). ARM PMUL / PMULL
    // (DDI 0487 C7.2.281). Only 8-bit lanes are emitted by libVEX.
    match op_str {
        "Iop_PolynomialMul8x8" => {
            return Some(IROp::VPolynomialMul {
                count: 8,
                widen: false,
            });
        }
        "Iop_PolynomialMul8x16" => {
            return Some(IROp::VPolynomialMul {
                count: 16,
                widen: false,
            });
        }
        "Iop_PolynomialMull8x8" => {
            return Some(IROp::VPolynomialMul {
                count: 8,
                widen: true,
            });
        }
        _ => {}
    }

    // Vector multiply: 8-bit is NEON-only (VMUL.I8); 16/32-bit are SSE+NEON.
    vec_arms!(op_str; "Iop_Mul" => VMul {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8),
        "32x2" => (I32, 2), "32x4" => (I32, 4),
    });
    // Widening vector multiply (angr-ph300.78). Two families, both -> V128:
    //   Iop_Mull{N}{S,U}x{M}      full-lane, (I64,I64)->V128, NEON VMULL
    //   Iop_MullEven{N}{S,U}x{M}  even-lane, (V128,V128)->V128, SSE PMULDQ/PMULUDQ
    // `even` selects which input lanes contribute; `signed` picks sign vs zero
    // extension. libVEX puts S/U AFTER the lane size (e.g. Iop_Mull32Sx2), so
    // these do not collide with the "Iop_Mul" VMul arm above. The "Iop_Mull"
    // prefix cannot false-match "Iop_MullEven*" — the leftover "Even8Ux16" hits
    // no suffix — so the two invocations are order-independent.
    vec_mull_arms!(op_str; "Iop_Mull" => VMull {
        "8Ux8" => (I8, 8, false, false), "8Sx8" => (I8, 8, true, false),
        "16Ux4" => (I16, 4, false, false), "16Sx4" => (I16, 4, true, false),
        "32Ux2" => (I32, 2, false, false), "32Sx2" => (I32, 2, true, false),
    });
    vec_mull_arms!(op_str; "Iop_MullEven" => VMull {
        "8Ux16" => (I8, 16, false, true), "8Sx16" => (I8, 16, true, true),
        "16Ux8" => (I16, 8, false, true), "16Sx8" => (I16, 8, true, true),
        "32Ux4" => (I32, 4, false, true), "32Sx4" => (I32, 4, true, true),
    });
    // Signed doubling saturating widening multiply — Iop_QDMull{N}Sx{M}
    // ((I64,I64)->V128, NEON VQDMULL). Only 16Sx4 / 32Sx2 exist in libVEX;
    // always signed, always full-lane. A dedicated match (not a macro) since
    // there are just two opcodes and no U/even axes to enumerate.
    match op_str {
        "Iop_QDMull16Sx4" => {
            return Some(IROp::VQDMull {
                elem: IRType::I16,
                count: 4,
            });
        }
        "Iop_QDMull32Sx2" => {
            return Some(IROp::VQDMull {
                elem: IRType::I32,
                count: 2,
            });
        }
        _ => {}
    }

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
    // Signed (S) and unsigned (U) suffixes both exist in libVEX; the U family
    // backs ARM NEON VCGT.U8/U16/U32 and the SSE/AVX unsigned compares
    // (angr-9ke6b.160). libVEX defines no Iop_CmpGT64Ux1 (D-reg), so that
    // suffix is absent by design rather than omission.
    vec_signed_arms!(op_str; "Iop_CmpGT" => VCmpGT {
        "8Sx8" => (I8, 8, true), "8Sx16" => (I8, 16, true),
        "16Sx4" => (I16, 4, true), "16Sx8" => (I16, 8, true),
        "32Sx2" => (I32, 2, true), "32Sx4" => (I32, 4, true),
        "64Sx2" => (I64, 2, true),
        "8Ux8" => (I8, 8, false), "8Ux16" => (I8, 16, false),
        "16Ux4" => (I16, 4, false), "16Ux8" => (I16, 8, false),
        "32Ux2" => (I32, 2, false), "32Ux4" => (I32, 4, false),
        "64Ux2" => (I64, 2, false),
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

    // NEON integer reciprocal estimate — Iop_RecipEst32Ux{2,4} (URECPE) and
    // Iop_RSqrtEst32Ux{2,4} (URSQRTE). Fresh-symbolic per lane via VIRecipEst /
    // VIRSqrtEst. Implemented in angr-tukg.5.
    if let Some(rest) = op_str.strip_prefix("Iop_RecipEst") {
        match rest {
            "32Ux2" => return Some(IROp::VIRecipEst { count: 2 }),
            "32Ux4" => return Some(IROp::VIRecipEst { count: 4 }),
            _ => {}
        }
    }
    if let Some(rest) = op_str.strip_prefix("Iop_RSqrtEst") {
        match rest {
            "32Ux2" => return Some(IROp::VIRSqrtEst { count: 2 }),
            "32Ux4" => return Some(IROp::VIRSqrtEst { count: 4 }),
            _ => {}
        }
    }

    // V128/V256 to/from conversions.
    cast_arms!(op_str; "Iop_" => Truncate    { "V128to64"   => (V128, I64) });
    cast_arms!(op_str; "Iop_" => ZeroExtend  { "64UtoV128"  => (I64, V128), "32UtoV128" => (I32, V128) });
    // NOTE: no "SetV128lo64" arm here — parse_float claims it first (returns
    // IROp::SetV128lo64, a binop preserving the upper 64 bits), so a Reinterpret
    // arm would be both dead and semantically wrong. See angr-ph300.62.

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
    // No NEON op currently routes here: every op listed in the NOTEs below is
    // fully implemented (the last scaffold, Iop_PwAdd32Fx2, became IROp::VFPwAdd
    // in angr-cudgw.6). The `NeonUnimplemented` sentinel and this fn are kept as
    // the scaffold point for the next NEON op — add
    // `"Iop_Foo" => return Some(IROp::NeonUnimplemented("Iop_Foo")),` to a match
    // on `op_str` here, alongside a NOTE once it graduates to a real handler.
    let _ = op_str;
    None
    // Historical coverage notes (op family -> implementing bead/route):
    /* match op_str {
        // NOTE: Iop_GetElem* / Iop_SetElem* (lane extract/insert) implemented
        // in angr-bkcs.2 — routed through parse_vector to IROp::VGetElem /
        // IROp::VSetElem above. Iop_Dup* / Iop_Widen* / Iop_Narrow{Bin,Un}* /
        // Iop_QNarrow{Bin,Un}* implemented in angr-hzs0 — routed through
        // parse_vector to IROp::VDup / IROp::VWiden / IROp::VNarrow{Un,Bin} /
        // IROp::VQNarrow{Un,Bin}.

        // NOTE: FP RecipEst / RecipStep / RSqrtEst / RSqrtStep ({32,64}{F0,Fx}*)
        // implemented in angr-iyon — routed through parse_float to
        // IROp::VFRecipEst{,S} / VFRecipStep / VFRSqrtEst{,S} / VFRSqrtStep.

        // NOTE: Iop_RecipEst32Ux{2,4} (URECPE) and Iop_RSqrtEst32Ux{2,4}
        // (URSQRTE) implemented in angr-tukg.5 — routed through parse_vector
        // to IROp::VIRecipEst / IROp::VIRSqrtEst (fresh-symbolic per lane).

        // NOTE: Iop_QAdd{N}{S/U}x{M} / Iop_QSub{N}{S/U}x{M} (NEON saturating
        // integer add/sub) implemented in angr-tukg.1 — routed through
        // parse_vector to IROp::VQAdd / IROp::VQSub.

        // NOTE: Iop_Avg{N}{S/U}x{M} (rounding halving add) implemented in
        // angr-tukg.3 — routed through parse_vector to IROp::VAvg.

        // NOTE: Iop_Reverse{N}sIn{M}_x{K} (byte/halfword/word/bit reversal
        // within lane) implemented in angr-tukg.4 — routed through
        // parse_vreverse to IROp::VReverse.

        // NOTE: integer Iop_PwAdd{N}x{M}, Iop_PwAddL{N}{S/U}x{M},
        // Iop_PwMin{N}{S/U}x{M}, Iop_PwMax{N}{S/U}x{M} implemented in
        // angr-tukg.2 — routed through parse_vector to IROp::VPwAdd /
        // VPwAddL / VPwMin / VPwMax. The FP pairwise Iop_PwAdd32Fx2 is
        // implemented in angr-cudgw.6 — routed through parse_float to
        // IROp::VFPwAdd. No Pw* op remains unimplemented.

        // NOTE: Iop_PolynomialMul8x{8,16} / Iop_PolynomialMull8x8 (NEON GF(2)
        // carry-less multiply) implemented in angr-tukg.6 — routed through
        // parse_vector to IROp::VPolynomialMul.

        // NOTE: Iop_Cnt8x{8,16} (per-byte popcount), Iop_Clz{N}x{M} and
        // Iop_Cls{N}x{M} (per-lane count-leading-zeros / count-leading-sign-
        // bits) implemented in angr-tukg.6 — routed through parse_vector to
        // IROp::VCnt / VClz / VCls.

        // Vector shift by *vector* (Shl/Shr/Sar/Sal{N}x{M}) routed to
        // parse_vector → IROp::VShl / VShr / VSar (Sal → VShl) in angr-tukg.7.

        // NOTE: Iop_QShl{N}x{M} / Iop_QSal{N}x{M} (NEON saturating shift-left
        // by vector) implemented in angr-tukg.8 — routed through parse_vector
        // to IROp::VQShlSat. QShlN (shift-by-immediate) is still unimplemented.
        _ => return None,
    }; */
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
        "Ijk_FlushDCacheLine" => JumpKind::FlushDCacheLine,
        "Ijk_ExtV128" => JumpKind::ExtV128,
        "Ijk_Extension" => JumpKind::Extension,
        _ => JumpKind::Boring,
    }
}

#[cfg(test)]
#[path = "opcode_map_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod opcode_map_tests;
