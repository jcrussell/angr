//! Float/int conversion VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared free fn they call (`build_float_expr`,
//! which lives in `ops/lane_traits.rs` and is re-exported by `ops`) stays
//! visible by the descendant-module rule.
//!
//! Covers: int↔float and float↔float conversions (RNE-implicit unops and the
//! rounding-mode-aware binop variants), plus the round-to-int ops. The
//! `define_*` codegen macros are private to this module since they are only
//! invoked here; they emit `pub(super)` stubs so the generated conversion
//! methods stay reachable from the dispatch in `ops`.

use super::float_arith::{apply_rounding, round_ties_to_even};
use super::{OpError, VEXOps, build_float_expr};
use crate::symbolic::{FloatOpKind, FloatPrec, RustBV, SymContext};

/// Generate concrete float-to-float conversion stubs.
macro_rules! define_float_to_float {
    ($($name:ident: $src_ty:ty, $src_uty:ty, $dst_ty:ty, $src_prec:expr, $dst_prec:expr;)*) => {
        $(
            pub(super) fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::float_to_float(arg, $src_prec, $dst_prec, |v| {
                    (<$src_ty>::from_bits(v as $src_uty) as $dst_ty).to_bits() as u128
                })
            }
        )*
    };
}

/// Generate concrete int-to-float conversion stubs.
/// Entry shape: `name: src_int_ty, src_width_bits, signed_flag, dst_float_ty, dst_prec;`
macro_rules! define_int_to_float {
    ($($name:ident: $src_int:ty, $src_width:expr, $signed:expr, $dst_ty:ty, $dst_prec:expr;)*) => {
        $(
            pub(super) fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::int_to_float(arg, $src_width, $signed, $dst_prec, |v| {
                    ((v as $src_int) as $dst_ty).to_bits() as u128
                })
            }
        )*
    };
}

/// Generate concrete float-to-signed-int conversion stubs (RNE round-ties-even).
/// Entry shape: `name: src_float_ty, src_uint_ty, src_prec, dst_signed_int_ty, dst_unsigned_int_ty, dst_width_bits;`
macro_rules! define_float_to_int_signed {
    ($($name:ident: $src_ty:ty, $src_uty:ty, $src_prec:expr, $dst_int:ty, $dst_uint:ty, $dst_width:expr;)*) => {
        $(
            pub(super) fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::float_to_int(arg, $src_prec, $dst_width, true, |v| {
                    (round_ties_to_even(<$src_ty>::from_bits(v as $src_uty)) as $dst_int as $dst_uint) as u128
                })
            }
        )*
    };
}

/// Generate concrete float-to-unsigned-int conversion stubs (RNE round-ties-even).
/// Entry shape: `name: src_float_ty, src_uint_ty, src_prec, dst_unsigned_int_ty, dst_width_bits;`
macro_rules! define_float_to_int_unsigned {
    ($($name:ident: $src_ty:ty, $src_uty:ty, $src_prec:expr, $dst_uint:ty, $dst_width:expr;)*) => {
        $(
            pub(super) fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::float_to_int(arg, $src_prec, $dst_width, false, |v| {
                    round_ties_to_even(<$src_ty>::from_bits(v as $src_uty)) as $dst_uint as u128
                })
            }
        )*
    };
}

impl VEXOps {
    /// Int-to-float conversion (unary, RNE implicit). Symbolic operands
    /// route through Z3 FP via `FloatOpKind::ConvertItoF`.
    fn int_to_float(
        arg: RustBV,
        src_bits: u32,
        signed: bool,
        dst_prec: FloatPrec,
        concrete: fn(u128) -> u128,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), src_bits);
        if let Some(v) = arg.as_u128() {
            return Ok(RustBV::concrete(concrete(v), dst_prec.bits()));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertItoF {
                src_bits: src_bits as u8,
                signed,
            },
            dst_prec,
            vec![arg],
        ))
    }

    /// Float-to-int conversion (unary, RNE implicit). Symbolic operands
    /// route through Z3 FP via `FloatOpKind::ConvertFtoI`.
    fn float_to_int(
        arg: RustBV,
        src_prec: FloatPrec,
        dst_bits: u32,
        signed: bool,
        concrete: fn(u128) -> u128,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), src_prec.bits());
        if let Some(v) = arg.as_u128() {
            return Ok(RustBV::concrete(concrete(v), dst_bits));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertFtoI {
                dst_bits: dst_bits as u8,
                signed,
            },
            src_prec,
            vec![arg],
        ))
    }

    /// Float-to-float conversion (unary, RNE implicit). Symbolic operands
    /// route through Z3 FP via `FloatOpKind::ConvertFtoF`.
    fn float_to_float(
        arg: RustBV,
        src_prec: FloatPrec,
        dst_prec: FloatPrec,
        concrete: fn(u128) -> u128,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), src_prec.bits());
        if let Some(v) = arg.as_u128() {
            return Ok(RustBV::concrete(concrete(v), dst_prec.bits()));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertFtoF { src_prec },
            dst_prec,
            vec![arg],
        ))
    }

    // --- Float-to-float conversions ---
    define_float_to_float! {
        f32_to_f64: f32, u32, f64, FloatPrec::F32, FloatPrec::F64;
        f64_to_f32: f64, u64, f32, FloatPrec::F64, FloatPrec::F32;
    }

    // --- Int-to-float conversions ---
    define_int_to_float! {
        i32s_to_f32: i32, 32, true,  f32, FloatPrec::F32;
        i32s_to_f64: i32, 32, true,  f64, FloatPrec::F64;
        i64s_to_f32: i64, 64, true,  f32, FloatPrec::F32;
        i64s_to_f64: i64, 64, true,  f64, FloatPrec::F64;
        i32u_to_f32: u32, 32, false, f32, FloatPrec::F32;
        i32u_to_f64: u32, 32, false, f64, FloatPrec::F64;
        i64u_to_f32: u64, 64, false, f32, FloatPrec::F32;
        i64u_to_f64: u64, 64, false, f64, FloatPrec::F64;
    }

    // --- Float-to-int conversions (round ties to even) ---
    define_float_to_int_signed! {
        f32_to_i32s: f32, u32, FloatPrec::F32, i32, u32, 32;
        f64_to_i32s: f64, u64, FloatPrec::F64, i32, u32, 32;
        f32_to_i64s: f32, u32, FloatPrec::F32, i64, u64, 64;
        f64_to_i64s: f64, u64, FloatPrec::F64, i64, u64, 64;
    }
    define_float_to_int_unsigned! {
        f32_to_i32u: f32, u32, FloatPrec::F32, u32, 32;
        f64_to_i32u: f64, u64, FloatPrec::F64, u32, 32;
        f32_to_i64u: f32, u32, FloatPrec::F32, u64, 64;
        f64_to_i64u: f64, u64, FloatPrec::F64, u64, 64;
    }

    /// Round F32 to integer using specified rounding mode (binop version).
    /// left = rounding mode (U32), right = value (F32)
    /// VEX rounding modes: 0=nearest, 1=down(-inf), 2=up(+inf), 3=zero(truncate)
    pub(super) fn round_f32_to_int_with_mode(
        mode: RustBV,
        value: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(m), Some(v)) = (mode.as_u128(), value.as_u128()) {
            let f = f32::from_bits(v as u32);
            let rounded = apply_rounding(f, m as u32);
            // Normalize -0.0 to +0.0 to match Python VEX behavior
            let normalized = if rounded == 0.0 { 0.0f32 } else { rounded };
            let result = normalized.to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Ok(build_float_expr(
            FloatOpKind::RoundToInt,
            FloatPrec::F32,
            vec![mode, value],
        ))
    }

    /// Round F64 to integer using specified rounding mode (binop version).
    /// left = rounding mode (U32), right = value (F64)
    pub(super) fn round_f64_to_int_with_mode(
        mode: RustBV,
        value: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(m), Some(v)) = (mode.as_u128(), value.as_u128()) {
            let f = f64::from_bits(v as u64);
            let rounded = apply_rounding(f, m as u32);
            // Normalize -0.0 to +0.0 to match Python VEX behavior
            let normalized = if rounded == 0.0 { 0.0f64 } else { rounded };
            let result = normalized.to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Ok(build_float_expr(
            FloatOpKind::RoundToInt,
            FloatPrec::F64,
            vec![mode, value],
        ))
    }

    // =========================================================================
    // Rounding-mode aware float-to-int conversions
    // VEX rounding modes: 0=nearest, 1=down(-inf), 2=up(+inf), 3=zero(truncate)
    // =========================================================================

    /// Float-to-int conversion with explicit rounding mode (binop). Routes
    /// through Z3 FP via `FloatOpKind::ConvertFtoIRm` for symbolic.
    fn float_to_int_rm(
        rm: RustBV,
        arg: RustBV,
        src_prec: FloatPrec,
        dst_bits: u32,
        signed: bool,
        concrete: fn(u128, u32) -> u128,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), src_prec.bits());
        if let (Some(rm_val), Some(v)) = (rm.as_u128(), arg.as_u128()) {
            return Ok(RustBV::concrete(concrete(v, rm_val as u32), dst_bits));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertFtoIRm {
                dst_bits: dst_bits as u8,
                signed,
            },
            src_prec,
            vec![rm, arg],
        ))
    }

    /// Float-to-float conversion with explicit rounding mode (binop). Routes
    /// through Z3 FP via `FloatOpKind::ConvertFtoFRm` for symbolic.
    fn float_to_float_rm(
        rm: RustBV,
        arg: RustBV,
        src_prec: FloatPrec,
        dst_prec: FloatPrec,
        concrete: fn(u128, u32) -> u128,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), src_prec.bits());
        if let (Some(rm_val), Some(v)) = (rm.as_u128(), arg.as_u128()) {
            return Ok(RustBV::concrete(
                concrete(v, rm_val as u32),
                dst_prec.bits(),
            ));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertFtoFRm { src_prec },
            dst_prec,
            vec![rm, arg],
        ))
    }

    /// Narrow an f64 to f32 under an explicit VEX rounding mode.
    ///
    /// Rust's `as f32` cast is hard-wired to round-nearest-ties-to-even, so it
    /// only implements `rm & 0x3 == 0`. For the directed modes we start from
    /// that nearest result and, when it landed on the wrong side of the exact
    /// value, step one ULP toward the requested direction. `next_up`/
    /// `next_down` also give the IEEE-754 overflow behaviour for free
    /// (`+inf.next_down() == f32::MAX`, so RZ/RD saturate rather than
    /// overflowing to infinity) and the underflow behaviour (`0.0.next_up()`
    /// is the smallest subnormal, so RU on a tiny positive value does not
    /// flush to zero).
    fn narrow_f64_to_f32_rm(v: f64, rm: u32) -> f32 {
        let nearest = v as f32;
        // NaN and exactly-representable values are rounding-mode independent.
        // `f64::from(nearest) == v` also covers infinities.
        if rm & 0x3 == 0 || v.is_nan() || f64::from(nearest) == v {
            return nearest;
        }
        let toward_neg_inf = match rm & 0x3 {
            1 => true,    // toward -infinity
            2 => false,   // toward +infinity
            _ => v > 0.0, // toward zero: down when positive, up when negative
        };
        if toward_neg_inf {
            if f64::from(nearest) > v {
                nearest.next_down()
            } else {
                nearest
            }
        } else if f64::from(nearest) < v {
            nearest.next_up()
        } else {
            nearest
        }
    }

    // --- Rounding-mode float conversions (binop variants) ---
    pub(super) fn f64_to_f32_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_float_rm(rm, arg, FloatPrec::F64, FloatPrec::F32, |v, rm| {
            Self::narrow_f64_to_f32_rm(f64::from_bits(v as u64), rm).to_bits() as u128
        })
    }
    pub(super) fn f32_to_i32s_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 32, true, |v, rm| {
            (apply_rounding(f32::from_bits(v as u32), rm) as i32 as u32) as u128
        })
    }
    pub(super) fn f64_to_i32s_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 32, true, |v, rm| {
            (apply_rounding(f64::from_bits(v as u64), rm) as i32 as u32) as u128
        })
    }
    pub(super) fn f32_to_i64s_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 64, true, |v, rm| {
            (apply_rounding(f32::from_bits(v as u32), rm) as i64 as u64) as u128
        })
    }
    pub(super) fn f64_to_i64s_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 64, true, |v, rm| {
            (apply_rounding(f64::from_bits(v as u64), rm) as i64 as u64) as u128
        })
    }
    pub(super) fn f32_to_i32u_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 32, false, |v, rm| {
            apply_rounding(f32::from_bits(v as u32), rm) as u32 as u128
        })
    }
    pub(super) fn f64_to_i32u_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 32, false, |v, rm| {
            apply_rounding(f64::from_bits(v as u64), rm) as u32 as u128
        })
    }
    pub(super) fn f32_to_i64u_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 64, false, |v, rm| {
            apply_rounding(f32::from_bits(v as u32), rm) as u64 as u128
        })
    }
    pub(super) fn f64_to_i64u_rm(
        rm: RustBV,
        arg: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 64, false, |v, rm| {
            apply_rounding(f64::from_bits(v as u64), rm) as u64 as u128
        })
    }
}
