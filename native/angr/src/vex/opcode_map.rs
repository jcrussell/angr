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

use super::ir::{IROp, IRType};
use super::opcode_map_float::{parse_float, parse_transcendental};
use super::opcode_map_vector::{parse_neon_unimplemented, parse_vector, parse_vreverse};
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
//
// Each macro carries a `pub(super) use` re-export: `macro_rules!` textual scope
// covers only the rest of this module and its children, but the FP and SIMD
// families live in the sibling modules `opcode_map_float` and
// `opcode_map_vector`, which import them by path.

/// IROp::Variant(IRType::Ty)
macro_rules! tuple_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => $ty:ident ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant(IRType::$ty)), )*
            _ => {}
        }
    }};
}
pub(super) use tuple_arms;

/// IROp::Variant { elem: IRType::Ty, count: N }
macro_rules! vec_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { elem: IRType::$elem, count: $count }), )*
            _ => {}
        }
    }};
}
pub(super) use vec_arms;

/// IROp::Variant { elem: IRType::Ty }
macro_rules! scalar_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => $elem:ident ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { elem: IRType::$elem }), )*
            _ => {}
        }
    }};
}
pub(super) use scalar_arms;

/// IROp::Variant { from: IRType::From, to: IRType::To }
macro_rules! cast_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $to:ident) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { from: IRType::$from, to: IRType::$to }), )*
            _ => {}
        }
    }};
}
pub(super) use cast_arms;

/// IROp::Variant { elem: IRType::Ty, count: N, signed: bool }
macro_rules! vec_signed_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($elem:ident, $count:literal, $signed:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { elem: IRType::$elem, count: $count, signed: $signed }), )*
            _ => {}
        }
    }};
}
pub(super) use vec_signed_arms;

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
pub(super) use vec_mull_arms;

/// IROp::VWiden { from: IRType::From, count: N, signed: bool }
macro_rules! vec_widen_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal, $signed:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { from: IRType::$from, count: $count, signed: $signed }), )*
            _ => {}
        }
    }};
}
pub(super) use vec_widen_arms;

/// IROp::VNarrowUn / VNarrowBin { from: IRType::From, count: N }
macro_rules! vec_narrow_arms {
    ($s:expr; $prefix:literal => $variant:ident { $( $sfx:literal => ($from:ident, $count:literal) ),* $(,)? }) => {{
        match $s.strip_prefix($prefix) {
            $( Some($sfx) => return Some(IROp::$variant { from: IRType::$from, count: $count }), )*
            _ => {}
        }
    }};
}
pub(super) use vec_narrow_arms;

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
pub(super) use vec_qnarrow_arms;

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
pub(super) use fcmp_vec_arms;

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
pub(super) use fcmp_scalar_arms;

/// Convert a pyvex operation string to Rust IROp.
///
/// Returns `IROp::Unmapped(name)` (interned `&'static str`) for opcodes
/// with no entry in the parse_* dispatch. Dispatch in
/// `VEXOps::unop`/`binop`/`qop`/`binop_with_rm` surfaces this as
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
    if let Some(op) = parse_transcendental(op_str) {
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
    // No scalar-integer Iop_Neg arm: real VEX only defines float negates.
    // `vendor/pyvex_ffi.h` declares exactly Iop_NegF32, Iop_NegF64,
    // Iop_NegF128 and the vector Iop_Neg32Fx2 / Iop_Neg32Fx4 / Iop_Neg64Fx2 —
    // all mapped in `parse_float` except Iop_NegF128, which `IRType` has no
    // F128 variant to map to. There is no Iop_NegF16 (the header's only F16
    // entries are Ity_F16 and the F16<->F32/F64 conversions), so an
    // `Iop_Neg{F16,...}` brace-enumeration would name an opcode that never
    // existed. Integer negation lifts as `0 - x` via Iop_Sub.

    match op_str {
        // DivMod - combined division and modulo
        "Iop_DivModU64to32" => Some(IROp::DivModU64to32),
        "Iop_DivModS64to32" => Some(IROp::DivModS64to32),
        "Iop_DivModU128to64" => Some(IROp::DivModU128to64),
        "Iop_DivModS128to64" => Some(IROp::DivModS128to64),

        // No scalar high-half-multiply arm: real VEX has no Iop_MulHi{32,64}
        // (the header only defines the vector Iop_MulHi<w>{U,S}x<n> family,
        // which parse_vector maps to IROp::VMulHi).
        // x86 IMUL/MUL lift to Iop_MullS32/Iop_MullU32 followed by
        // Iop_64HIto32, both of which are mapped above / in parse_conversion.
        _ => None,
    }
}

/// Parse bitwise operations: And, Or, Xor, Not
///
/// The `"1" => I1` arms on And/Or/Xor are unreachable on the VEX version this
/// crate is pinned to: the `IROp` enum in `vendor/pyvex_ffi.h` defines `Iop_Not1`
/// but no `Iop_And1` / `Iop_Or1` / `Iop_Xor1`, and `libpyvex.so` carries no such
/// name strings either, so no lift can emit them. They are kept as
/// forward-compatible placeholders: `tuple_arms!` matches the suffix exactly, so
/// an arm for an opcode that does not exist costs one failed compare and cannot
/// misfire onto a real opcode. Drop or confirm them when the pyvex/VEX pin is
/// next bumped.
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
    // The 8x8/16x4/32x2 shapes are NEON D-reg, the 8x16/16x8/32x4/64x2 shapes
    // SSE/NEON Q-reg, and the 16x16/32x8/64x4 shapes AVX2 (angr-sqfj8.118).
    // (Before angr-sqfj8.113 SarN omitted 64x2 on the false premise that pyvex
    // has no such op; `Iop_SarN64x2` is declared right alongside its ShlN/ShrN
    // siblings and was silently falling to IROp::Unmapped.)
    //
    // Two asymmetries below are real, not gaps — both mirror the x86 ISA and
    // are declared exactly this way in `vendor/pyvex_ffi.h`:
    //   * no `8x32`: AVX2 has no byte shift-by-immediate (VPSLLB et al. do not
    //     exist), so VEX declares no 8x32 form for any of the three families;
    //   * no `SarN64x4`: arithmetic right shift of qwords arrived only with
    //     AVX-512 (VPSRAQ), so VEX declares ShlN64x4/ShrN64x4 but no SarN64x4.
    // Adding either would be a dead arm matching a string no lift can emit.
    vec_arms!(op_str; "Iop_ShlN" => VShlN {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });
    vec_arms!(op_str; "Iop_ShrN" => VShrN {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2), "64x4" => (I64, 4),
    });
    vec_arms!(op_str; "Iop_SarN" => VSarN {
        "8x8" => (I8, 8), "8x16" => (I8, 16),
        "16x4" => (I16, 4), "16x8" => (I16, 8), "16x16" => (I16, 16),
        "32x2" => (I32, 2), "32x4" => (I32, 4), "32x8" => (I32, 8),
        "64x2" => (I64, 2),
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
    // Pinned by `test_parse_cmp_ord_stays_unmapped` (angr-sqfj8.120) — do not let
    // the CmpLT/CmpLE arms above widen onto the shared 32S/32U/64S/64U suffixes.
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
    // The I1 source has no 16-bit destination here: libVEX declares
    // `Iop_1Uto8/32/64` but no `Iop_1Uto16`, even though the signed table's
    // `Iop_1Sto16` above is real (angr-0jh0j.62).
    cast_arms!(op_str; "Iop_" => ZeroExtend {
        "1Uto8"   => (I1, I8),   "1Uto32"  => (I1, I32),  "1Uto64"  => (I1, I64),
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
            // Bit manipulation.
            //
            // Only the plain `Iop_Clz{32,64}` / `Iop_Ctz{32,64}` (undefined
            // result on a zero input) are mapped. libVEX now prefers the
            // zero-input-safe `Iop_ClzNat{32,64}` / `Iop_CtzNat{32,64}`
            // variants, but those are not defined in the VEX version this
            // crate is pinned to — `vendor/pyvex_ffi.h` has zero `ClzNat` /
            // `CtzNat` hits, so no lift can currently emit them and the gap is
            // unreachable. Map them when the pyvex/VEX pin is next bumped; the
            // `tuple_arms!` suffix match is exact, so `Iop_ClzNat32` falls
            // through to `None` (deferred to Python) rather than being
            // silently mistaken for `Iop_Clz32`. Same "unmapped for now"
            // situation as the `Iop_CmpORD*` gap in `parse_comparison`.
            //
            // The `Iop_PopCount*` arms are the mirror image: *no* scalar
            // popcount opcode exists at any width on this pin — `IROp` in
            // `vendor/pyvex_ffi.h` has zero `PopCount` entries and `libpyvex.so`
            // no `PopCount` name string (x86 POPCNT lifts to the `gen_POPCOUNT`
            // dirty helper, not an IROp). They are kept as forward-compatible
            // placeholders because `IROp::PopCount`'s evaluator (`popcount_into`,
            // dispatched in `VEXOps::unop`) and its unit tests are already
            // written and correct; deleting only the parse arms would strand a
            // working implementation with no route in. Same exact-suffix
            // reasoning as the And1/Or1/Xor1 placeholders in `parse_bitwise`.
            tuple_arms!(op_str; "Iop_Clz"      => Clz      { "32" => I32, "64" => I64 });
            tuple_arms!(op_str; "Iop_Ctz"      => Ctz      { "32" => I32, "64" => I64 });
            tuple_arms!(op_str; "Iop_PopCount" => PopCount { "8" => I8, "16" => I16, "32" => I32, "64" => I64 });
            None
        }
    }
}

/// Parse special and x86-specific operations
fn parse_special(op_str: &str) -> Option<IROp> {
    match op_str {
        // Byte/word lane permute (PSHUFB on x86, VPERMD on AVX2, TBL/TBX on
        // NEON). Not PCLMUL — that is the correctly-labeled block further
        // down. angr-sqfj8.117: the arm previously listed a nonexistent
        // `Iop_Perm8x32` (dead string) while the real `Iop_Perm32x4` /
        // `Iop_Perm32x8` went unmapped; verified against the full
        // `Iop_Perm*` set in `vendor/pyvex_ffi.h`.
        //
        // `Iop_Perm8x16x2` (the two-table AArch64 TBL form) is deliberately
        // NOT mapped: it is a *triop*, and the whole `VPerm` family exists
        // only to route to Python's VEX engine, whose `_op_generic_Perm`
        // (`angr/engines/vex/claripy/irop.py`) is two-argument. Mapping it
        // would move the failure, not fix it.
        "Iop_Perm8x8" => Some(IROp::VPerm {
            elem: IRType::I8,
            count: 8,
        }),
        "Iop_Perm8x16" => Some(IROp::VPerm {
            elem: IRType::I8,
            count: 16,
        }),
        "Iop_Perm32x4" => Some(IROp::VPerm {
            elem: IRType::I32,
            count: 4,
        }),
        "Iop_Perm32x8" => Some(IROp::VPerm {
            elem: IRType::I32,
            count: 8,
        }),

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
        // Decimal floats (`Ity_D32`/`Ity_D64`/`Ity_D128`) and quad floats
        // (`Ity_F128`) have no `IRType` variant, so they are mapped to the
        // same-width variant instead. `IRType` is a *width* tag in this engine
        // — nothing outside `IRType::bits()`/`bytes()` matches on which variant
        // it is (`IRType::F80` is not even constructible from any other path),
        // so preserving the declared width is the only property that matters
        // for temp/register sizing and the `debug_assert_eq!(width, ty.bits())`
        // checks in the `ops/` macros. Mapping the 128-bit types to
        // `IRType::F80` (80 bits) instead, as this arm used to, made
        // `bytes()` report 10 rather than 16.
        //
        // Returning `None` here would be worse, not safer: every `parse_type`
        // caller in `pyvex_bridge.rs` and `libvex_lifter.rs` goes through
        // `parse_type_or_log`, which defaults to `IRType::I64`, so an unmapped
        // type becomes 64 bits (loudly, but still wrongly). A real fix needs `IRType::{D32, D64, D128, F128}` variants plus
        // evaluator support, which is only worth doing alongside PowerPC /
        // S390X — the only architectures that emit these types, and both listed
        // as unsupported in docs/advanced-topics/rust_engine.rst. Until then no
        // lifted architecture can reach this arm. Same "pin the placeholder,
        // don't pretend it's right" treatment as the Clz/Ctz and `Iop_And1`
        // notes above.
        "Ity_D32" => Some(IRType::I32),
        "Ity_D64" => Some(IRType::I64),
        "Ity_F128" | "Ity_D128" => Some(IRType::V128),
        "Ity_V128" => Some(IRType::V128),
        "Ity_V256" => Some(IRType::V256),
        _ => None,
    }
}

/// [`parse_type`], substituting `Ity_I64` and logging when the type string is
/// one this engine has no [`IRType`] for.
///
/// `context` should identify the call site (the IR field being typed) so the
/// warning is actionable.
// SILENT(cat-c): `IRType` is a width tag, so defaulting to `I64` mis-sizes the
// value at every use site — a 32-bit temp read back as 64 bits, or a load that
// fetches twice the bytes it should. Every caller on both marshalling paths
// (`pyvex_bridge`'s JSON deserializer and `libvex_lifter`'s FFI
// `c_type_parse`) must go through this single logged fallback rather than a
// bare `.unwrap_or(IRType::I64)`, so table drift is traceable on whichever
// path a block happens to take (angr-c7xno.90).
pub fn parse_type_or_log(ty_str: &str, context: &str) -> IRType {
    silent_default!(
        cat_c,
        parse_type(ty_str),
        IRType::I64,
        "Unknown pyvex IR type string {ty_str:?} ({context}); assuming Ity_I64 \
         — the value is mis-sized wherever it is used"
    )
}

/// Parse an endianness string from pyvex, substituting `Iend_LE` and logging
/// when the string is neither `Iend_LE` nor `Iend_BE`.
// SILENT(cat-c): endianness is not a hint — defaulting a big-endian load or
// store to little-endian byte-swaps the value silently, and the result is
// indistinguishable from a correct LE access downstream. Both marshalling
// paths funnel here (`pyvex_bridge`'s JSON deserializer at every Load/Store/
// StoreG/LoadG/CAS site, and `libvex_lifter::c_endness` after its own
// discriminant lookup), so this is the single place table drift on either
// path can be traced from (angr-03vl4.69).
pub fn parse_endness(end_str: &str) -> super::ir::Endness {
    use super::ir::Endness;

    let parsed = match end_str {
        "Iend_LE" => Some(Endness::Little),
        "Iend_BE" => Some(Endness::Big),
        _ => None,
    };
    silent_default!(
        cat_c,
        parsed,
        Endness::Little,
        "Unknown pyvex endianness string {end_str:?}; assuming Iend_LE — a \
         big-endian access is byte-swapped wherever it is used"
    )
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
        "Ijk_Sys_int" => JumpKind::Sys_int,
        "Ijk_Sys_int32" => JumpKind::Sys_int32,
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
        "Ijk_NoRedir" => JumpKind::NoRedir,
        "Ijk_SigILL" => JumpKind::SigILL,
        "Ijk_SigTRAP" => JumpKind::SigTRAP,
        "Ijk_SigSEGV" => JumpKind::SigSEGV,
        "Ijk_SigBUS" => JumpKind::SigBUS,
        "Ijk_SigFPE" => JumpKind::SigFPE,
        "Ijk_SigFPE_IntDiv" => JumpKind::SigFPE_IntDiv,
        "Ijk_SigFPE_IntOvf" => JumpKind::SigFPE_IntOvf,
        "Ijk_Privileged" => JumpKind::Privileged,
        // SILENT(cat-c): every `IRJumpKind` in vendor/pyvex_ffi.h now has an
        // arm above, so reaching this one means the lifter emitted a tag this
        // build does not know. Boring is the only non-terminal answer we can
        // give, but it is a wrong-answer risk (angr-sqfj8.111 was exactly that
        // for the trap kinds), hence the warn.
        _ => {
            log::warn!(
                "parse_jumpkind: unrecognized VEX jumpkind {jk_str:?}, treating as Ijk_Boring"
            );
            JumpKind::Boring
        }
    }
}

test_submod!("opcode_map_tests.rs" => opcode_map_tests);
