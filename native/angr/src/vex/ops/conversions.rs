//! Float/int conversion VEX op helpers.
//!
//! Covers: int↔float and float↔float conversions (RNE-implicit unops and the
//! rounding-mode-aware binop variants), plus the round-to-int ops. The
//! `define_*` codegen macros are private to this module since they are only
//! invoked here; they emit `pub(super)` stubs so the generated conversion
//! methods stay reachable from the dispatch in `ops`.

use super::float_arith::apply_rounding;
use super::{OpError, VEXOps, build_float_expr};
use crate::symbolic::{FloatOpKind, FloatPrec, RustBV, SymContext, VexRm};

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
                    (<$src_ty>::from_bits(v as $src_uty).round_ties_even() as $dst_int as $dst_uint) as u128
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
                    <$src_ty>::from_bits(v as $src_uty).round_ties_even() as $dst_uint as u128
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
        Self::require_operand_width("int_to_float arg", arg.width(), src_bits)?;
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

    /// Int-to-float conversion under an explicit VEX rounding mode (the
    /// binop form, `rm` first): `Iop_I{32,64}{S,U}toF{32,64}` except the
    /// exact `I32{S,U}toF64` unops. Symbolic operands route through
    /// `FloatOpKind::ConvertItoFRm`.
    pub(super) fn int_to_float_rm(
        rm: RustBV,
        arg: RustBV,
        src_bits: u32,
        signed: bool,
        dst_prec: FloatPrec,
    ) -> Result<RustBV, OpError> {
        Self::require_operand_width("int_to_float_rm arg", arg.width(), src_bits)?;
        let mode = rm.as_u128().map(VexRm::decode).transpose()?;
        if let (Some(mode), Some(v)) = (mode, arg.as_u128()) {
            let value = if signed {
                Self::sign_extend_low_to_i128(v, src_bits)
            } else {
                (v & Self::low_bit_mask_u128(src_bits)) as i128
            };
            return Ok(RustBV::concrete(
                Self::int_to_float_bits_rm(value, dst_prec, mode),
                dst_prec.bits(),
            ));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertItoFRm {
                src_bits: src_bits as u8,
                signed,
            },
            dst_prec,
            vec![rm, arg],
        ))
    }

    /// `value` as IEEE bits at `prec`, rounded by VEX mode `mode`. Rust's
    /// int-to-float cast rounds to nearest, ties to even; when that is
    /// inexact and the mode points the other way, the answer is the adjacent
    /// float. Ties away from zero differs from it only on an exact tie, where
    /// it takes the neighbour of larger magnitude.
    fn int_to_float_bits_rm(value: i128, prec: FloatPrec, mode: VexRm) -> u128 {
        macro_rules! directed {
            ($f:ty) => {{
                let near = value as $f;
                // Exact: |value| < 2^64, so `near` is an integer of at most
                // 2^64 in magnitude and fits i128.
                let back = near as i128;
                let rounded = match mode {
                    VexRm::TowardNegative if back > value => near.next_down(),
                    VexRm::TowardPositive if back < value => near.next_up(),
                    VexRm::TowardZero if value >= 0 && back > value => near.next_down(),
                    VexRm::TowardZero if value < 0 && back < value => near.next_up(),
                    VexRm::NearestAway if back != value => {
                        let other = if back > value {
                            near.next_down()
                        } else {
                            near.next_up()
                        };
                        // Both neighbours are integers within 2^64 + 1 ulp of
                        // `value`, so the distances are exact in i128.
                        let tie = (back - value).abs() == (other as i128 - value).abs();
                        if tie && other.abs() > near.abs() {
                            other
                        } else {
                            near
                        }
                    }
                    VexRm::NearestEven
                    | VexRm::TowardNegative
                    | VexRm::TowardPositive
                    | VexRm::TowardZero
                    | VexRm::NearestAway => near,
                };
                rounded.to_bits() as u128
            }};
        }
        match prec {
            FloatPrec::F32 => directed!(f32),
            FloatPrec::F64 => directed!(f64),
        }
    }

    /// Run `f` on each 32-bit lane of `v` (`count` lanes, lane 0 lowest) and
    /// concatenate the 32-bit results in the same order.
    fn map_lanes32(
        v: RustBV,
        count: u8,
        f: impl Fn(RustBV) -> Result<RustBV, OpError>,
    ) -> Result<RustBV, OpError> {
        let lanes = u32::from(count);
        Self::require_operand_width("map_lanes32 arg", v.width(), 32 * lanes)?;
        let mut out: Option<RustBV> = None;
        for i in (0..lanes).rev() {
            let lane = f(v.extract(32 * i + 31, 32 * i))?;
            out = Some(match out {
                None => lane,
                Some(high) => high.concat_into(lane),
            });
        }
        out.ok_or_else(|| OpError::UnsupportedVectorOp("zero-lane conversion".to_string()))
    }

    /// `Iop_I32{S,U}toF32x{count}`: each 32-bit integer lane to F32 under
    /// VEX rounding mode `rm` (the `_DEP` unop forms pass nearest).
    pub(super) fn vec_i32_to_f32(
        rm: RustBV,
        v: RustBV,
        count: u8,
        signed: bool,
    ) -> Result<RustBV, OpError> {
        Self::map_lanes32(v, count, |lane| {
            Self::int_to_float_rm(rm.clone(), lane, 32, signed, FloatPrec::F32)
        })
    }

    /// `Iop_F32toI32{S,U}x{count}`: each F32 lane to a 32-bit integer under
    /// VEX rounding mode `rm` (the `_RZ` unop forms pass toward-zero).
    pub(super) fn vec_f32_to_i32(
        rm: RustBV,
        v: RustBV,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::map_lanes32(v, count, |lane| {
            if signed {
                Self::f32_to_i32s_rm(rm.clone(), lane, ctx)
            } else {
                Self::f32_to_i32u_rm(rm.clone(), lane, ctx)
            }
        })
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
        Self::require_operand_width("float_to_int arg", arg.width(), src_prec.bits())?;
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
        Self::require_operand_width("float_to_float arg", arg.width(), src_prec.bits())?;
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
    /// The mode is decoded by `VexRm::decode`.
    pub(super) fn round_f32_to_int_with_mode(
        mode: RustBV,
        value: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let m = mode.as_u128().map(VexRm::decode).transpose()?;
        if let (Some(m), Some(v)) = (m, value.as_u128()) {
            let f = f32::from_bits(v as u32);
            // IEEE roundToIntegral keeps the sign of zero, as do the
            // symbolic branch (`FloatOpKind::RoundToInt`) and the Python
            // engine's `fpRoundToIntegral` (angr-vmi2l.57).
            let result = apply_rounding(f, m).to_bits();
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
        let m = mode.as_u128().map(VexRm::decode).transpose()?;
        if let (Some(m), Some(v)) = (m, value.as_u128()) {
            let f = f64::from_bits(v as u64);
            // IEEE roundToIntegral keeps the sign of zero, as do the
            // symbolic branch (`FloatOpKind::RoundToInt`) and the Python
            // engine's `fpRoundToIntegral` (angr-vmi2l.57).
            let result = apply_rounding(f, m).to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Ok(build_float_expr(
            FloatOpKind::RoundToInt,
            FloatPrec::F64,
            vec![mode, value],
        ))
    }

    // =========================================================================
    // Rounding-mode aware float-to-int conversions (modes per `VexRm`)
    // =========================================================================

    /// Float-to-int conversion with explicit rounding mode (binop). Routes
    /// through Z3 FP via `FloatOpKind::ConvertFtoIRm` for symbolic.
    fn float_to_int_rm(
        rm: RustBV,
        arg: RustBV,
        src_prec: FloatPrec,
        dst_bits: u32,
        signed: bool,
        concrete: fn(u128, VexRm) -> u128,
    ) -> Result<RustBV, OpError> {
        Self::require_operand_width("float_to_int_rm arg", arg.width(), src_prec.bits())?;
        let mode = rm.as_u128().map(VexRm::decode).transpose()?;
        if let (Some(mode), Some(v)) = (mode, arg.as_u128()) {
            return Ok(RustBV::concrete(concrete(v, mode), dst_bits));
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
        concrete: fn(u128, VexRm) -> u128,
    ) -> Result<RustBV, OpError> {
        Self::require_operand_width("float_to_float_rm arg", arg.width(), src_prec.bits())?;
        let mode = rm.as_u128().map(VexRm::decode).transpose()?;
        if let (Some(mode), Some(v)) = (mode, arg.as_u128()) {
            return Ok(RustBV::concrete(concrete(v, mode), dst_prec.bits()));
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
    /// only implements `VexRm::NearestEven`. For the directed modes we start
    /// from that nearest result and, when it landed on the wrong side of the
    /// exact value, step one ULP toward the requested direction. `next_up`/
    /// `next_down` also give the IEEE-754 overflow behaviour for free
    /// (`+inf.next_down() == f32::MAX`, so RZ/RD saturate rather than
    /// overflowing to infinity) and the underflow behaviour (`0.0.next_up()`
    /// is the smallest subnormal, so RU on a tiny positive value does not
    /// flush to zero). Ties away from zero differs from the cast only on an
    /// exact tie, which is the midpoint of two adjacent f32s and so exact in
    /// f64; a tie past `f32::MAX` already rounds to infinity in both modes.
    fn narrow_f64_to_f32_rm(v: f64, rm: VexRm) -> f32 {
        let nearest = v as f32;
        // NaN and exactly-representable values are rounding-mode independent.
        // `f64::from(nearest) == v` also covers infinities.
        if rm == VexRm::NearestEven || v.is_nan() || f64::from(nearest) == v {
            return nearest;
        }
        let toward_neg_inf = match rm {
            VexRm::TowardNegative => true,
            VexRm::TowardPositive => false,
            // Toward zero: down when positive, up when negative.
            VexRm::TowardZero => v > 0.0,
            VexRm::NearestEven => return nearest,
            VexRm::NearestAway => {
                let other = if f64::from(nearest) > v {
                    nearest.next_down()
                } else {
                    nearest.next_up()
                };
                let midpoint = (f64::from(nearest) + f64::from(other)) / 2.0;
                return if midpoint == v && other.abs() > nearest.abs() {
                    other
                } else {
                    nearest
                };
            }
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
