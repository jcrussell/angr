//! VEX operation implementations.
//!
//! This module implements VEX operations using a parameterized approach.
//! Instead of ~200 separate implementations (e.g., add8, add16, add32, add64),
//! we have a single implementation per operation type that handles all widths.

use std::sync::Arc;

use crate::symbolic::{BVOp, FloatOpKind, FloatPrec, RustBV, SymContext};

use super::ir::{IROp, IRType};

/// Map a VEX float `IRType` to a Z3 FP precision.
#[inline]
fn float_prec_of(ty: IRType) -> Option<FloatPrec> {
    match ty {
        IRType::F32 => Some(FloatPrec::F32),
        IRType::F64 => Some(FloatPrec::F64),
        _ => None,
    }
}

/// Build a symbolic float-op expression. Used by float_* helpers below
/// when operands are not fully concrete; routes through Z3 FP theory in
/// `build_fp_z3_ast_cached`.
fn build_float_expr(
    kind: FloatOpKind,
    prec: FloatPrec,
    operands: Vec<RustBV>,
) -> RustBV {
    let width = kind.result_bits(prec);
    RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width,
        op: BVOp::Float { kind, prec },
        operands: operands.into_iter().map(Arc::new).collect::<Vec<_>>().into(),
    }
}

/// Compress same-width unary arms `assert width(arg) == ty.bits(); arg.$method(ctx)`.
macro_rules! width_unop {
    ($arg:ident, $ty:expr, $method:ident, $ctx:expr) => {{
        debug_assert_eq!($arg.width(), $ty.bits());
        Ok($arg.$method($ctx))
    }};
}

/// Compress same-width binary arms `assert width(left)==width(right)==ty.bits(); left.$method(right, ctx)`.
macro_rules! width_binop {
    ($left:ident, $right:ident, $ty:expr, $method:ident, $ctx:expr) => {{
        debug_assert_eq!($left.width(), $ty.bits());
        debug_assert_eq!($right.width(), $ty.bits());
        Ok($left.$method($right, $ctx))
    }};
}

/// VEX operation executor.
///
/// This struct provides methods to execute VEX operations on `RustBV` values.
pub struct VEXOps;

impl VEXOps {
    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// Execute a unary operation.
    #[inline]
    pub fn unop(
        op: IROp,
        arg: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            IROp::Not(ty) => width_unop!(arg, ty, not_into, ctx),
            IROp::Neg(ty) => width_unop!(arg, ty, neg_into, ctx),
            IROp::Clz(ty) => width_unop!(arg, ty, clz_into, ctx),
            IROp::Ctz(ty) => width_unop!(arg, ty, ctz_into, ctx),
            IROp::PopCount(ty) => width_unop!(arg, ty, popcount_into, ctx),

            // Sign/Zero extension
            IROp::SignExtend { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                Ok(arg.sign_extend_into(to.bits(), ctx))
            }

            IROp::ZeroExtend { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                Ok(arg.zero_extend_into(to.bits(), ctx))
            }

            // Truncation
            IROp::Truncate { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                Ok(arg.truncate_into(to.bits(), ctx))
            }

            // Extraction (unary form - low_bit is encoded in opcode)
            IROp::Extract { from, to, low_bit } => {
                debug_assert_eq!(arg.width(), from.bits());
                let hi = low_bit as u32 + to.bits() - 1;
                let lo = low_bit as u32;
                Ok(arg.extract_into(hi, lo, ctx))
            }

            // Float operations
            IROp::FNeg(ty) => Self::float_neg(arg, ty, ctx),
            IROp::FAbs(ty) => Self::float_abs(arg, ty, ctx),
            IROp::FSqrt(ty) => Self::float_sqrt(arg, ty, ctx),

            // Float conversions
            IROp::F32toF64 => Self::f32_to_f64(arg, ctx),
            IROp::F64toF32 => Self::f64_to_f32(arg, ctx),
            IROp::I32StoF32 => Self::i32s_to_f32(arg, ctx),
            IROp::I32StoF64 => Self::i32s_to_f64(arg, ctx),
            IROp::I64StoF32 => Self::i64s_to_f32(arg, ctx),
            IROp::I64StoF64 => Self::i64s_to_f64(arg, ctx),
            IROp::I32UtoF32 => Self::i32u_to_f32(arg, ctx),
            IROp::I32UtoF64 => Self::i32u_to_f64(arg, ctx),
            IROp::I64UtoF32 => Self::i64u_to_f32(arg, ctx),
            IROp::I64UtoF64 => Self::i64u_to_f64(arg, ctx),
            IROp::F32toI32S => Self::f32_to_i32s(arg, ctx),
            IROp::F64toI32S => Self::f64_to_i32s(arg, ctx),
            IROp::F32toI64S => Self::f32_to_i64s(arg, ctx),
            IROp::F64toI64S => Self::f64_to_i64s(arg, ctx),
            IROp::F32toI32U => Self::f32_to_i32u(arg, ctx),
            IROp::F64toI32U => Self::f64_to_i32u(arg, ctx),
            IROp::F32toI64U => Self::f32_to_i64u(arg, ctx),
            IROp::F64toI64U => Self::f64_to_i64u(arg, ctx),

            // Vector not
            IROp::VNot(ty) => width_unop!(arg, ty, not_into, ctx),

            // Reinterpret (just changes type, not bits)
            IROp::Reinterpret { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                if from.bits() == to.bits() {
                    Ok(arg)
                } else if to.bits() > from.bits() {
                    Ok(arg.zero_extend_into(to.bits(), ctx))
                } else {
                    Ok(arg.truncate_into(to.bits(), ctx))
                }
            }

            // Scalar-in-vector sqrt (SQRTSS/SQRTSD)
            IROp::VFSqrtS { elem } => Self::vec_float_scalar_sqrt(arg, elem, ctx),

            _ => Err(OpError::NotUnary(op)),
        }
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// Execute a binary operation.
    #[inline]
    pub fn binop(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // Arithmetic
            IROp::Add(ty) => width_binop!(left, right, ty, add_into, ctx),
            IROp::Sub(ty) => width_binop!(left, right, ty, sub_into, ctx),
            IROp::Mul(ty) => width_binop!(left, right, ty, mul_into, ctx),
            IROp::DivU(ty) => width_binop!(left, right, ty, udiv_into, ctx),
            IROp::DivS(ty) => width_binop!(left, right, ty, sdiv_into, ctx),
            IROp::ModU(ty) => width_binop!(left, right, ty, urem_into, ctx),
            IROp::ModS(ty) => width_binop!(left, right, ty, srem_into, ctx),

            // Widening multiply
            IROp::MullU(ty) => Self::widening_mul(left, right, ty, false, ctx),
            IROp::MullS(ty) => Self::widening_mul(left, right, ty, true, ctx),

            // High half of multiplication
            IROp::MulHi { ty, signed } => Self::mul_hi(left, right, ty, signed, ctx),

            // DivMod: 64-bit / 32-bit -> 64-bit (low=quotient, high=remainder)
            IROp::DivModU64to32 => Self::divmod_64_to_32(left, right, false, ctx),
            IROp::DivModS64to32 => Self::divmod_64_to_32(left, right, true, ctx),

            // DivMod: 128-bit / 64-bit -> 128-bit (low=quotient, high=remainder)
            IROp::DivModU128to64 => Self::divmod_128_to_64(left, right, false, ctx),
            IROp::DivModS128to64 => Self::divmod_128_to_64(left, right, true, ctx),

            // Bitwise
            IROp::And(ty) => width_binop!(left, right, ty, and_into, ctx),
            IROp::Or(ty) => width_binop!(left, right, ty, or_into, ctx),
            IROp::Xor(ty) => width_binop!(left, right, ty, xor_into, ctx),

            // Shifts — normalize shift amount width to match operand
            IROp::Shl(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                let amt = Self::normalize_shift_amount(right, left.width(), ctx);
                Ok(left.shl_into(amt, ctx))
            }

            IROp::Shr(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                let amt = Self::normalize_shift_amount(right, left.width(), ctx);
                Ok(left.lshr_into(amt, ctx))
            }

            IROp::Sar(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                let amt = Self::normalize_shift_amount(right, left.width(), ctx);
                Ok(left.ashr_into(amt, ctx))
            }

            // Comparisons
            IROp::CmpEQ(ty) => width_binop!(left, right, ty, eq_into, ctx),
            IROp::CmpNE(ty) => width_binop!(left, right, ty, ne_into, ctx),
            IROp::CmpLT(ty) => width_binop!(left, right, ty, slt_into, ctx),
            IROp::CmpLE(ty) => width_binop!(left, right, ty, sle_into, ctx),
            IROp::CmpLTU(ty) => width_binop!(left, right, ty, ult_into, ctx),
            IROp::CmpLEU(ty) => width_binop!(left, right, ty, ule_into, ctx),

            // Float arithmetic
            IROp::FAdd(ty) => Self::float_add(left, right, ty, ctx),
            IROp::FSub(ty) => Self::float_sub(left, right, ty, ctx),
            IROp::FMul(ty) => Self::float_mul(left, right, ty, ctx),
            IROp::FDiv(ty) => Self::float_div(left, right, ty, ctx),

            // Float comparisons
            IROp::FCmpEQ(ty) => Self::float_cmp_eq(left, right, ty, ctx),
            IROp::FCmpLT(ty) => Self::float_cmp_lt(left, right, ty, ctx),
            IROp::FCmpLE(ty) => Self::float_cmp_le(left, right, ty, ctx),

            // Float rounding with mode (left = rounding mode, right = value)
            IROp::RoundF32toInt => Self::round_f32_to_int_with_mode(left, right, ctx),
            IROp::RoundF64toInt => Self::round_f64_to_int_with_mode(left, right, ctx),

            // Scalar-in-vector float operations (SSE scalar ops)
            IROp::VFAddS { elem } => Self::vec_float_scalar_op(left, right, elem, "add", ctx),
            IROp::VFSubS { elem } => Self::vec_float_scalar_op(left, right, elem, "sub", ctx),
            IROp::VFMulS { elem } => Self::vec_float_scalar_op(left, right, elem, "mul", ctx),
            IROp::VFDivS { elem } => Self::vec_float_scalar_op(left, right, elem, "div", ctx),

            // Vector bitwise
            IROp::VAnd(ty) => width_binop!(left, right, ty, and_into, ctx),
            IROp::VOr(ty) => width_binop!(left, right, ty, or_into, ctx),
            IROp::VXor(ty) => width_binop!(left, right, ty, xor_into, ctx),

            // Concatenate
            IROp::Concat { ty } => {
                let result = left.concat_into(right, ctx);
                debug_assert_eq!(result.width(), ty.bits());
                Ok(result)
            }

            // Vector operations (simplified - just doing element-wise)
            IROp::VAdd { elem, count } => Self::vec_binop(left, right, elem, count, "add", ctx),
            IROp::VSub { elem, count } => Self::vec_binop(left, right, elem, count, "sub", ctx),
            IROp::VMul { elem, count } => Self::vec_binop(left, right, elem, count, "mul", ctx),
            IROp::VMulLo { elem, count } => Self::vec_mul_lo(left, right, elem, count, ctx),

            // Vector compare operations
            IROp::VCmpEQ { elem, count } => Self::vec_cmp(left, right, elem, count, "eq", ctx),
            IROp::VCmpGT { elem, count } => Self::vec_cmp(left, right, elem, count, "gt", ctx),

            // Vector interleave
            IROp::VInterleaveLO { elem } => Self::vec_interleave_lo(left, right, elem, ctx),
            IROp::VInterleaveHI { elem } => Self::vec_interleave_hi(left, right, elem, ctx),

            // Vector shifts by immediate
            IROp::VShlN { elem, count } => Self::vec_shl_n(left, right, elem, count, ctx),
            IROp::VShrN { elem, count } => Self::vec_shr_n(left, right, elem, count, ctx),
            IROp::VSarN { elem, count } => Self::vec_sar_n(left, right, elem, count, ctx),

            // Raw opcode
            IROp::Raw(code) => Err(OpError::RawOpcode(code)),

            // Float conversions that take a rounding mode as the first argument
            // VEX rounding modes: 0=nearest, 1=down, 2=up, 3=zero (truncate)
            IROp::F64toF32 => {
                // left = rounding mode, right = F64 value
                Self::f64_to_f32_rm(left, right, ctx)
            }
            IROp::F32toI32S => {
                // left = rounding mode, right = F32 value
                Self::f32_to_i32s_rm(left, right, ctx)
            }
            IROp::F64toI32S => {
                // left = rounding mode, right = F64 value
                Self::f64_to_i32s_rm(left, right, ctx)
            }
            IROp::F32toI64S => {
                Self::f32_to_i64s_rm(left, right, ctx)
            }
            IROp::F64toI64S => {
                Self::f64_to_i64s_rm(left, right, ctx)
            }
            IROp::F32toI32U => {
                Self::f32_to_i32u_rm(left, right, ctx)
            }
            IROp::F64toI32U => {
                Self::f64_to_i32u_rm(left, right, ctx)
            }
            IROp::F32toI64U => {
                Self::f32_to_i64u_rm(left, right, ctx)
            }
            IROp::F64toI64U => {
                Self::f64_to_i64u_rm(left, right, ctx)
            }

            // Scalar-in-vector max/min
            IROp::VFMaxS { elem } => Self::vec_float_scalar_max(left, right, elem, ctx),
            IROp::VFMinS { elem } => Self::vec_float_scalar_min(left, right, elem, ctx),

            // SetV128lo: set low bits of V128
            IROp::SetV128lo32 => {
                // left = V128, right = I32 value to put in low 32 bits
                Self::set_v128_lo32(left, right, ctx)
            }
            IROp::SetV128lo64 => {
                // left = V128, right = I64 value to put in low 64 bits
                Self::set_v128_lo64(left, right, ctx)
            }

            _ => Err(OpError::NotBinary(op)),
        }
    }

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// Execute a ternary operation (for ITE, etc.).
    #[inline]
    #[allow(unused_variables)]
    pub fn ternop(
        op: IROp,
        arg1: RustBV,
        arg2: RustBV,
        arg3: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // Extraction takes (value, start_bit_as_u8, length_as_u8)
            // Note: This is for cases where extract is done as a ternary op
            IROp::Extract { from, to, low_bit } => {
                debug_assert_eq!(arg1.width(), from.bits());
                let hi = low_bit as u32 + to.bits() - 1;
                let lo = low_bit as u32;
                Ok(arg1.extract_into(hi, lo, ctx))
            }
            _ => Err(OpError::NotTernary(op)),
        }
    }

    // =========================================================================
    // Quaternary Operations
    // =========================================================================

    /// Execute a quaternary operation. Used for fused multiply-add/sub
    /// where VEX delivers (rounding_mode, a, b, c) but the rm has already
    /// been stripped by the caller (so this takes a, b, c).
    #[inline]
    pub fn qop(
        op: IROp,
        a: RustBV,
        b: RustBV,
        c: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            IROp::FMAdd(ty) => Self::float_madd(a, b, c, ty, ctx),
            IROp::FMSub(ty) => Self::float_msub(a, b, c, ty, ctx),
            _ => Err(OpError::NotQuaternary(op)),
        }
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Widening multiply.
    #[inline]
    fn widening_mul(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_width = ty.bits();
        let out_width = in_width * 2;

        // Extend both operands
        let (left_ext, right_ext) = if signed {
            (
                left.sign_extend_into(out_width, ctx),
                right.sign_extend_into(out_width, ctx),
            )
        } else {
            (
                left.zero_extend_into(out_width, ctx),
                right.zero_extend_into(out_width, ctx),
            )
        };

        Ok(left_ext.mul_into(right_ext, ctx))
    }

    /// High half of multiplication.
    #[inline]
    fn mul_hi(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let width = ty.bits();
        let double_width = width * 2;

        // Extend and multiply
        let (left_ext, right_ext) = if signed {
            (
                left.sign_extend_into(double_width, ctx),
                right.sign_extend_into(double_width, ctx),
            )
        } else {
            (
                left.zero_extend_into(double_width, ctx),
                right.zero_extend_into(double_width, ctx),
            )
        };

        let product = left_ext.mul_into(right_ext, ctx);

        // Extract high half
        Ok(product.extract_into(double_width - 1, width, ctx))
    }

    /// Concatenate symbolic vector elements into a single BV, where
    /// `elements[0]` is the low-order element and `elements[len-1]` is the
    /// high-order element. Used by all vector ops with a symbolic fallback.
    #[inline]
    fn concat_le_elements(mut elements: Vec<RustBV>, ctx: &SymContext) -> RustBV {
        let mut result = elements
            .pop()
            .expect("vec elements guaranteed non-empty by counted loop");
        while let Some(elem) = elements.pop() {
            result = result.concat_into(elem, ctx);
        }
        result
    }

    /// DivMod: 64-bit dividend / 32-bit divisor -> 64-bit result.
    /// Low 32 bits = quotient, High 32 bits = remainder.
    fn divmod_64_to_32(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(dividend.width(), 64);
        debug_assert_eq!(divisor.width(), 32);

        // For concrete values, compute directly
        if let (Some(dvd), Some(dvs)) = (dividend.as_u128(), divisor.as_u128()) {
            let dvd = dvd as u64;
            let dvs = dvs as u32;

            if dvs == 0 {
                // Division by zero - return 0 (caller should have checked)
                return Ok(RustBV::concrete(0, 64));
            }

            let (quotient, remainder) = if signed {
                // Signed division
                let dvd_signed = dvd as i64;
                let dvs_signed = dvs as i32 as i64;
                let q = (dvd_signed / dvs_signed) as u32;
                let r = (dvd_signed % dvs_signed) as u32;
                (q, r)
            } else {
                // Unsigned division
                let q = (dvd / dvs as u64) as u32;
                let r = (dvd % dvs as u64) as u32;
                (q, r)
            };

            // Pack: low 32 bits = quotient, high 32 bits = remainder
            let result = (quotient as u64) | ((remainder as u64) << 32);
            return Ok(RustBV::concrete(result as u128, 64));
        }

        // Symbolic case: extend divisor to 64 bits, do full-width div/mod,
        // then pack the low 32 bits of each into the result. Z3 defines
        // div/mod by zero (udiv→all-ones, urem→dividend, sdiv→±1, srem→dividend),
        // matching claripy's behavior.
        let divisor_64 = if signed {
            divisor.sign_extend_into(64, ctx)
        } else {
            divisor.zero_extend_into(64, ctx)
        };
        let quotient_64 = if signed {
            dividend.sdiv(&divisor_64, ctx)
        } else {
            dividend.udiv(&divisor_64, ctx)
        };
        let remainder_64 = if signed {
            dividend.srem(&divisor_64, ctx)
        } else {
            dividend.urem(&divisor_64, ctx)
        };
        let quotient_32 = quotient_64.extract_into(31, 0, ctx);
        let remainder_32 = remainder_64.extract_into(31, 0, ctx);
        Ok(remainder_32.concat_into(quotient_32, ctx))
    }

    /// DivMod: 128-bit dividend / 64-bit divisor -> 128-bit result.
    /// Low 64 bits = quotient, High 64 bits = remainder.
    fn divmod_128_to_64(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(dividend.width(), 128);
        debug_assert_eq!(divisor.width(), 64);

        // For concrete values, compute directly
        if let (Some(dvd), Some(dvs)) = (dividend.as_u128(), divisor.as_u128()) {
            let dvs = dvs as u64;

            if dvs == 0 {
                // Division by zero - return 0 (caller should have checked)
                return Ok(RustBV::concrete(0, 128));
            }

            let (quotient, remainder) = if signed {
                // Signed division: treat as i128 / i64
                let dvd_signed = dvd as i128;
                let dvs_signed = dvs as i64 as i128;
                let q = (dvd_signed / dvs_signed) as u64;
                let r = (dvd_signed % dvs_signed) as u64;
                (q, r)
            } else {
                // Unsigned division: u128 / u64
                let q = (dvd / dvs as u128) as u64;
                let r = (dvd % dvs as u128) as u64;
                (q, r)
            };

            // Pack: low 64 bits = quotient, high 64 bits = remainder
            let result = (quotient as u128) | ((remainder as u128) << 64);
            return Ok(RustBV::concrete(result, 128));
        }

        // Symbolic case: same recipe as 64→32, scaled to 128/64.
        let divisor_128 = if signed {
            divisor.sign_extend_into(128, ctx)
        } else {
            divisor.zero_extend_into(128, ctx)
        };
        let quotient_128 = if signed {
            dividend.sdiv(&divisor_128, ctx)
        } else {
            dividend.udiv(&divisor_128, ctx)
        };
        let remainder_128 = if signed {
            dividend.srem(&divisor_128, ctx)
        } else {
            dividend.urem(&divisor_128, ctx)
        };
        let quotient_64 = quotient_128.extract_into(63, 0, ctx);
        let remainder_64 = remainder_128.extract_into(63, 0, ctx);
        Ok(remainder_64.concat_into(quotient_64, ctx))
    }

    /// Vector element-wise binary operation.
    fn vec_binop(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        op: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values, we can do this efficiently
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let _hi = lo + elem_width - 1;
                let mask = (1u128 << elem_width) - 1;

                let l_elem = (l >> lo) & mask;
                let r_elem = (r >> lo) & mask;

                let res_elem = match op {
                    "add" => l_elem.wrapping_add(r_elem) & mask,
                    "sub" => l_elem.wrapping_sub(r_elem) & mask,
                    "mul" => l_elem.wrapping_mul(r_elem) & mask,
                    _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
                };

                result |= res_elem << lo;
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // For symbolic, we need to extract, operate, and concatenate
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            let res_elem = match op {
                "add" => l_elem.add_into(r_elem, ctx),
                "sub" => l_elem.sub_into(r_elem, ctx),
                "mul" => l_elem.mul_into(r_elem, ctx),
                _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
            };

            elements.push(res_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Vector multiply keeping low half (PMULLD).
    /// Performs signed widening multiply on each element pair, keeping only the low bits.
    fn vec_mul_lo(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values, compute directly
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let mask = (1u128 << elem_width) - 1;

            for i in 0..count {
                let lo = (i as u32) * elem_width;

                let l_elem = (l >> lo) & mask;
                let r_elem = (r >> lo) & mask;

                // Sign-extend to perform signed multiply
                let l_signed = if elem_width == 32 {
                    (l_elem as u32 as i32 as i64) as u64
                } else if elem_width == 16 {
                    (l_elem as u16 as i16 as i32) as u32 as u64
                } else if elem_width == 8 {
                    (l_elem as u8 as i8 as i16) as u16 as u64
                } else {
                    l_elem as u64
                };

                let r_signed = if elem_width == 32 {
                    (r_elem as u32 as i32 as i64) as u64
                } else if elem_width == 16 {
                    (r_elem as u16 as i16 as i32) as u32 as u64
                } else if elem_width == 8 {
                    (r_elem as u8 as i8 as i16) as u16 as u64
                } else {
                    r_elem as u64
                };

                // Multiply and keep low bits
                let product = l_signed.wrapping_mul(r_signed);
                let res_elem = (product as u128) & mask;

                result |= res_elem << lo;
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // For symbolic values, fall back to element-wise
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            // For symbolic, just do regular multiply (low bits are the same for signed/unsigned)
            let res_elem = l_elem.mul_into(r_elem, ctx);
            elements.push(res_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }

    // =========================================================================
    // Vector Comparison Operations
    // =========================================================================

    /// Vector element-wise comparison.
    fn vec_cmp(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        op: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let elem_mask = (1u128 << elem_width) - 1;
            let all_ones = elem_mask;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let l_elem = (l >> lo) & elem_mask;
                let r_elem = (r >> lo) & elem_mask;

                let cmp_result = match op {
                    "eq" => l_elem == r_elem,
                    "gt" => {
                        // Signed comparison
                        let sign_bit = 1u128 << (elem_width - 1);
                        let l_signed = if l_elem & sign_bit != 0 {
                            (l_elem | !elem_mask) as i128
                        } else {
                            l_elem as i128
                        };
                        let r_signed = if r_elem & sign_bit != 0 {
                            (r_elem | !elem_mask) as i128
                        } else {
                            r_elem as i128
                        };
                        l_signed > r_signed
                    }
                    _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
                };

                // Result is all 1s if true, all 0s if false
                if cmp_result {
                    result |= all_ones << lo;
                }
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // For symbolic, fall back to element-wise
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            let cmp_result = match op {
                "eq" => l_elem.eq_into(r_elem, ctx),
                "gt" => l_elem.sgt_into(r_elem, ctx),
                _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
            };

            // Extend the 1-bit result to full element width (all 1s or all 0s)
            let extended = cmp_result.sign_extend_into(elem_width, ctx);
            elements.push(extended);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Vector interleave low halves.
    fn vec_interleave_lo(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = left.width();
        let count = total_width / elem_width;
        let half_count = count / 2;

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let elem_mask = (1u128 << elem_width) - 1;

            for i in 0..half_count {
                let src_lo = (i as u32) * elem_width;
                let dst_lo = (i as u32) * 2 * elem_width;

                let l_elem = (l >> src_lo) & elem_mask;
                let r_elem = (r >> src_lo) & elem_mask;

                // VEX InterleaveLO: right goes to even positions, left to odd
                result |= r_elem << dst_lo;
                result |= l_elem << (dst_lo + elem_width);
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic case
        let mut elements: Vec<RustBV> = Vec::new();

        for i in 0..half_count {
            let src_lo = (i as u32) * elem_width;
            let src_hi = src_lo + elem_width - 1;

            let l_elem = left.extract(src_hi, src_lo, ctx);
            let r_elem = right.extract(src_hi, src_lo, ctx);

            // VEX InterleaveLO: right goes to even positions, left to odd
            elements.push(r_elem);
            elements.push(l_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Vector interleave high halves.
    fn vec_interleave_hi(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = left.width();
        let count = total_width / elem_width;
        let half_count = count / 2;

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let elem_mask = (1u128 << elem_width) - 1;

            for i in 0..half_count {
                let src_lo = ((half_count + i) as u32) * elem_width;
                let dst_lo = (i as u32) * 2 * elem_width;

                let l_elem = (l >> src_lo) & elem_mask;
                let r_elem = (r >> src_lo) & elem_mask;

                // VEX InterleaveHI: right goes to even positions, left to odd
                result |= r_elem << dst_lo;
                result |= l_elem << (dst_lo + elem_width);
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic case
        let mut elements: Vec<RustBV> = Vec::new();

        for i in 0..half_count {
            let src_lo = ((half_count + i) as u32) * elem_width;
            let src_hi = src_lo + elem_width - 1;

            let l_elem = left.extract(src_hi, src_lo, ctx);
            let r_elem = right.extract(src_hi, src_lo, ctx);

            // VEX InterleaveHI: right goes to even positions, left to odd
            elements.push(r_elem);
            elements.push(l_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }

    // =========================================================================
    // Vector Shift Operations (by immediate)
    // =========================================================================

    /// Vector shift left by immediate.
    fn vec_shl_n(
        vec: RustBV,
        shift_amt: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(vec.width(), total_width);

        // Concrete shift amount: keep the existing fast paths.
        if let Some(s) = shift_amt.as_u128() {
            let shift = s as u32;

            if shift >= elem_width {
                return Ok(RustBV::concrete(0, total_width));
            }

            if let Some(v) = vec.as_u128() {
                let mut result: u128 = 0;
                let elem_mask = (1u128 << elem_width) - 1;

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;
                    let shifted = (elem_val << shift) & elem_mask;
                    result |= shifted << lo;
                }

                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic shift amount (or symbolic vector with concrete shift):
        // resize the count to the lane width and apply per-lane.  Z3 bvshl
        // returns 0 when the shift count is >= the operand width, matching
        // the concrete semantics above.
        let resized_shift = Self::resize_vec_shift_amount(shift_amt, elem_width, ctx)?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.shl_into(resized_shift.clone(), ctx);
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Resize a vector shift amount to the lane width.
    ///
    /// VEX `ShlN` / `ShrN` / `SarN` take an I8 count. When the lane is wider,
    /// zero-extend so Z3's shift ops see matching widths. A wider count would
    /// require an ITE on the high bits — not seen in real VEX, so reject.
    fn resize_vec_shift_amount(
        shift_amt: RustBV,
        elem_width: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match shift_amt.width().cmp(&elem_width) {
            std::cmp::Ordering::Equal => Ok(shift_amt),
            std::cmp::Ordering::Less => Ok(shift_amt.zero_extend_into(elem_width, ctx)),
            std::cmp::Ordering::Greater => Err(OpError::UnsupportedVectorOp(
                "vector shift amount wider than lane".to_string(),
            )),
        }
    }

    /// Vector shift right logical by immediate.
    fn vec_shr_n(
        vec: RustBV,
        shift_amt: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(vec.width(), total_width);

        if let Some(s) = shift_amt.as_u128() {
            let shift = s as u32;

            if shift >= elem_width {
                return Ok(RustBV::concrete(0, total_width));
            }

            if let Some(v) = vec.as_u128() {
                let mut result: u128 = 0;
                let elem_mask = (1u128 << elem_width) - 1;

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;
                    let shifted = elem_val >> shift;
                    result |= shifted << lo;
                }

                return Ok(RustBV::concrete(result, total_width));
            }
        }

        let resized_shift = Self::resize_vec_shift_amount(shift_amt, elem_width, ctx)?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.lshr_into(resized_shift.clone(), ctx);
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Vector shift right arithmetic by immediate.
    fn vec_sar_n(
        vec: RustBV,
        shift_amt: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(vec.width(), total_width);

        if let Some(s) = shift_amt.as_u128() {
            let shift = s as u32;

            if let Some(v) = vec.as_u128() {
                let mut result: u128 = 0;
                let elem_mask = (1u128 << elem_width) - 1;
                let sign_bit = 1u128 << (elem_width - 1);

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;

                    // Arithmetic shift - preserve sign
                    let shifted = if shift >= elem_width {
                        // Shift >= width: result is all sign bits
                        if elem_val & sign_bit != 0 {
                            elem_mask  // All 1s
                        } else {
                            0  // All 0s
                        }
                    } else {
                        // Check if negative (sign bit set)
                        if elem_val & sign_bit != 0 {
                            // Negative: shift and fill with 1s
                            let shifted_val = elem_val >> shift;
                            let fill_mask = (elem_mask << (elem_width - shift)) & elem_mask;
                            (shifted_val | fill_mask) & elem_mask
                        } else {
                            // Positive: simple logical shift
                            elem_val >> shift
                        }
                    };

                    result |= shifted << lo;
                }

                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic shift amount (or symbolic vector with concrete shift):
        // Z3 bvashr replicates the sign bit when the shift count is >= the
        // operand width, matching the concrete sign-fill semantics above.
        let resized_shift = Self::resize_vec_shift_amount(shift_amt, elem_width, ctx)?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.ashr_into(resized_shift.clone(), ctx);
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    // =========================================================================
    // Float Operations (using bit manipulation for now)
    // =========================================================================

    fn float_neg(
        arg: RustBV,
        ty: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // Flip the sign bit
        let sign_bit = match ty {
            IRType::F32 => 31,
            IRType::F64 => 63,
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        let mask = RustBV::concrete(1u128 << sign_bit, arg.width());
        Ok(arg.xor_into(mask, ctx))
    }

    fn float_abs(
        arg: RustBV,
        ty: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // Clear the sign bit
        let mask = match ty {
            IRType::F32 => RustBV::concrete(0x7FFFFFFF, 32),
            IRType::F64 => RustBV::concrete(0x7FFFFFFFFFFFFFFF, 64),
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        Ok(arg.and_into(mask, ctx))
    }

    fn float_sqrt(
        arg: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // For concrete values, compute directly
        if let Some(v) = arg.as_u128() {
            let result = match ty {
                IRType::F32 => {
                    let f = f32::from_bits(v as u32);
                    f.sqrt().to_bits() as u128
                }
                IRType::F64 => {
                    let f = f64::from_bits(v as u64);
                    f.sqrt().to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }

        // Symbolic: build a Z3 FP expression so constraints propagate.
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Sqrt, prec, vec![arg]))
    }

    fn float_add(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    (lf + rf).to_bits() as u128
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    (lf + rf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Add, prec, vec![left, right]))
    }

    fn float_sub(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    (lf - rf).to_bits() as u128
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    (lf - rf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Sub, prec, vec![left, right]))
    }

    fn float_mul(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    (lf * rf).to_bits() as u128
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    (lf * rf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Mul, prec, vec![left, right]))
    }

    fn float_div(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    (lf / rf).to_bits() as u128
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    (lf / rf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Div, prec, vec![left, right]))
    }

    /// Fused multiply-add: a*b + c
    fn float_madd(
        a: RustBV,
        b: RustBV,
        c: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(av), Some(bv), Some(cv)) = (a.as_u128(), b.as_u128(), c.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let af = f32::from_bits(av as u32);
                    let bf = f32::from_bits(bv as u32);
                    let cf = f32::from_bits(cv as u32);
                    af.mul_add(bf, cf).to_bits() as u128
                }
                IRType::F64 => {
                    let af = f64::from_bits(av as u64);
                    let bf = f64::from_bits(bv as u64);
                    let cf = f64::from_bits(cv as u64);
                    af.mul_add(bf, cf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Fma, prec, vec![a, b, c]))
    }

    /// Fused multiply-sub: a*b - c
    fn float_msub(
        a: RustBV,
        b: RustBV,
        c: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(av), Some(bv), Some(cv)) = (a.as_u128(), b.as_u128(), c.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let af = f32::from_bits(av as u32);
                    let bf = f32::from_bits(bv as u32);
                    let cf = f32::from_bits(cv as u32);
                    af.mul_add(bf, -cf).to_bits() as u128
                }
                IRType::F64 => {
                    let af = f64::from_bits(av as u64);
                    let bf = f64::from_bits(bv as u64);
                    let cf = f64::from_bits(cv as u64);
                    af.mul_add(bf, -cf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Fms, prec, vec![a, b, c]))
    }

    /// Scalar float operation in vector (SSE scalar ops like ADDSS, DIVSS).
    /// Operates on element 0 only, passes through other elements from left operand.
    #[inline]
    fn vec_float_scalar_op(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        op: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    // Extract element 0 (lowest 32 bits)
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);

                    // Perform operation on element 0
                    let res0 = match op {
                        "add" => l0 + r0,
                        "sub" => l0 - r0,
                        "mul" => l0 * r0,
                        "div" => l0 / r0,
                        _ => return Err(OpError::UnsupportedVectorOp(format!("vec_float_scalar_{}", op))),
                    };

                    // Keep upper 96 bits from left, replace lower 32 bits with result
                    let upper = l & !0xFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                IRType::F64 => {
                    // Extract element 0 (lowest 64 bits)
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);

                    // Perform operation on element 0
                    let res0 = match op {
                        "add" => l0 + r0,
                        "sub" => l0 - r0,
                        "mul" => l0 * r0,
                        "div" => l0 / r0,
                        _ => return Err(OpError::UnsupportedVectorOp(format!("vec_float_scalar_{}", op))),
                    };

                    // Keep upper 64 bits from left, replace lower 64 bits with result
                    let upper = l & !0xFFFFFFFFFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        let kind = match op {
            "add" => FloatOpKind::Add,
            "sub" => FloatOpKind::Sub,
            "mul" => FloatOpKind::Mul,
            "div" => FloatOpKind::Div,
            _ => return Err(OpError::UnsupportedVectorOp(format!("vec_float_scalar_{}", op))),
        };
        Self::vec_float_scalar_lane_binop(left, right, elem, kind, ctx)
    }

    /// Scalar sqrt in vector (SQRTSS/SQRTSD).
    fn vec_float_scalar_sqrt(
        arg: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), 128);

        if let Some(v) = arg.as_u128() {
            let result = match elem {
                IRType::F32 => {
                    let val = f32::from_bits(v as u32);
                    let res = val.sqrt();
                    // Keep upper 96 bits, replace lower 32 bits with result
                    let upper = v & !0xFFFFFFFFu128;
                    upper | (res.to_bits() as u128)
                }
                IRType::F64 => {
                    let val = f64::from_bits(v as u64);
                    let res = val.sqrt();
                    // Keep upper 64 bits, replace lower 64 bits with result
                    let upper = v & !0xFFFFFFFFFFFFFFFFu128;
                    upper | (res.to_bits() as u128)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let lo = arg.extract(lane_bits - 1, 0, ctx);
        let upper = arg.extract(127, lane_bits, ctx);
        let res_lane = build_float_expr(FloatOpKind::Sqrt, prec, vec![lo]);
        Ok(upper.concat_into(res_lane, ctx))
    }

    /// Scalar max in vector (MAXSS/MAXSD).
    fn vec_float_scalar_max(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);
                    let res0 = if l0 > r0 { l0 } else { r0 };
                    let upper = l & !0xFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                IRType::F64 => {
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);
                    let res0 = if l0 > r0 { l0 } else { r0 };
                    let upper = l & !0xFFFFFFFFFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        Self::vec_float_scalar_lane_minmax(left, right, elem, /*is_max=*/ true, ctx)
    }

    /// Scalar min in vector (MINSS/MINSD).
    fn vec_float_scalar_min(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);
                    let res0 = if l0 < r0 { l0 } else { r0 };
                    let upper = l & !0xFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                IRType::F64 => {
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);
                    let res0 = if l0 < r0 { l0 } else { r0 };
                    let upper = l & !0xFFFFFFFFFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        Self::vec_float_scalar_lane_minmax(left, right, elem, /*is_max=*/ false, ctx)
    }

    /// Symbolic fallback for SSE scalar binary float ops (Add/Sub/Mul/Div).
    /// Extracts lane 0, runs the op via Z3 FP, concats back with upper bits.
    fn vec_float_scalar_lane_binop(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        kind: FloatOpKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let l_lo = left.extract(lane_bits - 1, 0, ctx);
        let r_lo = right.extract(lane_bits - 1, 0, ctx);
        let upper = left.extract(127, lane_bits, ctx);
        let res_lane = build_float_expr(kind, prec, vec![l_lo, r_lo]);
        Ok(upper.concat_into(res_lane, ctx))
    }

    /// Symbolic fallback for SSE scalar MAXSS/MINSS/MAXSD/MINSD on lane 0.
    /// Encodes Rust's `>`/`<` semantics (NaN-returns-right) as
    /// `ITE(cond, l, r)` with `cond = FCmpLt(r, l)` for max or
    /// `cond = FCmpLt(l, r)` for min — matches the concrete branch above.
    fn vec_float_scalar_lane_minmax(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        is_max: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let l_lo = left.extract(lane_bits - 1, 0, ctx);
        let r_lo = right.extract(lane_bits - 1, 0, ctx);
        let upper = left.extract(127, lane_bits, ctx);
        // max(l, r) = if l > r then l else r ⟺ ITE(r < l, l, r)
        // min(l, r) = if l < r then l else r ⟺ ITE(l < r, l, r)
        let (cmp_left, cmp_right) = if is_max {
            (r_lo.clone(), l_lo.clone())
        } else {
            (l_lo.clone(), r_lo.clone())
        };
        let cond = build_float_expr(FloatOpKind::CmpLt, prec, vec![cmp_left, cmp_right]);
        let res_lane = cond.ite_into(l_lo, r_lo, ctx);
        Ok(upper.concat_into(res_lane, ctx))
    }

    /// Set low 32 bits of V128.
    fn set_v128_lo32(
        vec: RustBV,
        val: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(vec.width(), 128);
        debug_assert_eq!(val.width(), 32);

        if let (Some(v), Some(lo)) = (vec.as_u128(), val.as_u128()) {
            let upper = v & !0xFFFFFFFFu128;
            let result = upper | (lo & 0xFFFFFFFF);
            return Ok(RustBV::concrete(result, 128));
        }
        // For symbolic, concatenate upper 96 bits with the value
        let upper = vec.extract(127, 32, ctx);
        let result = upper.concat(&val, ctx);
        Ok(result)
    }

    /// Set low 64 bits of V128.
    fn set_v128_lo64(
        vec: RustBV,
        val: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(vec.width(), 128);
        debug_assert_eq!(val.width(), 64);

        if let (Some(v), Some(lo)) = (vec.as_u128(), val.as_u128()) {
            let upper = v & !0xFFFFFFFFFFFFFFFFu128;
            let result = upper | (lo & 0xFFFFFFFFFFFFFFFF);
            return Ok(RustBV::concrete(result, 128));
        }
        // For symbolic, concatenate upper 64 bits with the value
        let upper = vec.extract(127, 64, ctx);
        let result = upper.concat(&val, ctx);
        Ok(result)
    }

    fn float_cmp_eq(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf == rf { 1u128 } else { 0u128 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf == rf { 1u128 } else { 0u128 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 1));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::CmpEq, prec, vec![left, right]))
    }

    fn float_cmp_lt(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf < rf { 1u128 } else { 0u128 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf < rf { 1u128 } else { 0u128 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 1));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::CmpLt, prec, vec![left, right]))
    }

    fn float_cmp_le(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf <= rf { 1u128 } else { 0u128 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf <= rf { 1u128 } else { 0u128 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 1));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::CmpLe, prec, vec![left, right]))
    }

    // Float conversions for concrete values.
    // Generated via macros to eliminate boilerplate across 18 conversion functions.

    /// Normalize shift amount width to match the operand width.
    #[inline]
    fn normalize_shift_amount(amt: RustBV, target_width: u32, ctx: &SymContext) -> RustBV {
        if amt.width() != target_width {
            if amt.width() > target_width {
                amt.truncate(target_width, ctx)
            } else {
                amt.zero_extend(target_width, ctx)
            }
        } else {
            amt
        }
    }

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
            FloatOpKind::ConvertItoF { src_bits: src_bits as u8, signed },
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
            FloatOpKind::ConvertFtoI { dst_bits: dst_bits as u8, signed },
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
    fn f32_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_float(arg, FloatPrec::F32, FloatPrec::F64, |v| (f32::from_bits(v as u32) as f64).to_bits() as u128)
    }
    fn f64_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_float(arg, FloatPrec::F64, FloatPrec::F32, |v| (f64::from_bits(v as u64) as f32).to_bits() as u128)
    }

    // --- Int-to-float conversions ---
    fn i32s_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 32, true, FloatPrec::F32, |v| ((v as i32) as f32).to_bits() as u128)
    }
    fn i32s_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 32, true, FloatPrec::F64, |v| ((v as i32) as f64).to_bits() as u128)
    }
    fn i64s_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 64, true, FloatPrec::F32, |v| ((v as i64) as f32).to_bits() as u128)
    }
    fn i64s_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 64, true, FloatPrec::F64, |v| ((v as i64) as f64).to_bits() as u128)
    }
    fn i32u_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 32, false, FloatPrec::F32, |v| ((v as u32) as f32).to_bits() as u128)
    }
    fn i32u_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 32, false, FloatPrec::F64, |v| ((v as u32) as f64).to_bits() as u128)
    }
    fn i64u_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 64, false, FloatPrec::F32, |v| ((v as u64) as f32).to_bits() as u128)
    }
    fn i64u_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::int_to_float(arg, 64, false, FloatPrec::F64, |v| ((v as u64) as f64).to_bits() as u128)
    }

    // --- Float-to-int conversions (round ties to even) ---
    fn f32_to_i32s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F32, 32, true, |v| (Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as i32 as u32) as u128)
    }
    fn f64_to_i32s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F64, 32, true, |v| (Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as i32 as u32) as u128)
    }
    fn f32_to_i64s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F32, 64, true, |v| (Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as i64 as u64) as u128)
    }
    fn f64_to_i64s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F64, 64, true, |v| (Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as i64 as u64) as u128)
    }
    fn f32_to_i32u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F32, 32, false, |v| Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as u32 as u128)
    }
    fn f64_to_i32u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F64, 32, false, |v| Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as u32 as u128)
    }
    fn f32_to_i64u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F32, 64, false, |v| Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as u64 as u128)
    }
    fn f64_to_i64u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int(arg, FloatPrec::F64, 64, false, |v| Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as u64 as u128)
    }

    /// Round F32 to integer using specified rounding mode (binop version).
    /// left = rounding mode (U32), right = value (F32)
    /// VEX rounding modes: 0=nearest, 1=down(-inf), 2=up(+inf), 3=zero(truncate)
    fn round_f32_to_int_with_mode(
        mode: RustBV,
        value: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(m), Some(v)) = (mode.as_u128(), value.as_u128()) {
            let f = f32::from_bits(v as u32);
            let rounded = match m & 0x3 {
                0 => Self::round_ties_to_even_f32(f),  // nearest, ties to even
                1 => f.floor(),                        // toward -infinity
                2 => f.ceil(),                         // toward +infinity
                3 => f.trunc(),                        // toward zero
                _ => Self::round_ties_to_even_f32(f),  // default to nearest
            };
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
    fn round_f64_to_int_with_mode(
        mode: RustBV,
        value: RustBV,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(m), Some(v)) = (mode.as_u128(), value.as_u128()) {
            let f = f64::from_bits(v as u64);
            let rounded = match m & 0x3 {
                0 => Self::round_ties_to_even_f64(f),  // nearest, ties to even
                1 => f.floor(),                        // toward -infinity
                2 => f.ceil(),                         // toward +infinity
                3 => f.trunc(),                        // toward zero
                _ => Self::round_ties_to_even_f64(f),  // default to nearest
            };
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

    /// Round f32 to nearest integer, ties to even (banker's rounding)
    fn round_ties_to_even_f32(f: f32) -> f32 {
        let rounded = f.round();
        // Check if we're exactly at a .5 case
        let frac = f - f.trunc();
        if frac.abs() == 0.5 {
            // Ties to even: round to the nearest even number
            let truncated = f.trunc();
            if (truncated as i32) % 2 == 0 {
                truncated
            } else {
                rounded
            }
        } else {
            rounded
        }
    }

    /// Round f64 to nearest integer, ties to even (banker's rounding)
    fn round_ties_to_even_f64(f: f64) -> f64 {
        let rounded = f.round();
        // Check if we're exactly at a .5 case
        let frac = f - f.trunc();
        if frac.abs() == 0.5 {
            // Ties to even: round to the nearest even number
            let truncated = f.trunc();
            if (truncated as i64) % 2 == 0 {
                truncated
            } else {
                rounded
            }
        } else {
            rounded
        }
    }

    /// Apply rounding mode to f32 value
    fn apply_rounding_f32(f: f32, rm: u32) -> f32 {
        match rm & 0x3 {
            0 => Self::round_ties_to_even_f32(f),  // nearest, ties to even (banker's rounding)
            1 => f.floor(),    // toward negative infinity
            2 => f.ceil(),     // toward positive infinity
            3 => f.trunc(),    // toward zero (truncate)
            _ => Self::round_ties_to_even_f32(f),  // default to nearest
        }
    }

    /// Apply rounding mode to f64 value
    fn apply_rounding_f64(f: f64, rm: u32) -> f64 {
        match rm & 0x3 {
            0 => Self::round_ties_to_even_f64(f),  // nearest, ties to even (banker's rounding)
            1 => f.floor(),    // toward negative infinity
            2 => f.ceil(),     // toward positive infinity
            3 => f.trunc(),    // toward zero (truncate)
            _ => Self::round_ties_to_even_f64(f),  // default to nearest
        }
    }

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
            FloatOpKind::ConvertFtoIRm { dst_bits: dst_bits as u8, signed },
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
            return Ok(RustBV::concrete(concrete(v, rm_val as u32), dst_prec.bits()));
        }
        Ok(build_float_expr(
            FloatOpKind::ConvertFtoFRm { src_prec },
            dst_prec,
            vec![rm, arg],
        ))
    }

    // --- Rounding-mode float conversions (binop variants) ---
    fn f64_to_f32_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        // Note: concrete path ignores rounding mode (uses direct cast); the
        // symbolic path correctly threads rm through Z3 FP.
        Self::float_to_float_rm(rm, arg, FloatPrec::F64, FloatPrec::F32,
            |v, _rm| (f64::from_bits(v as u64) as f32).to_bits() as u128)
    }
    fn f32_to_i32s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 32, true,
            |v, rm| (Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as i32 as u32) as u128)
    }
    fn f64_to_i32s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 32, true,
            |v, rm| (Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as i32 as u32) as u128)
    }
    fn f32_to_i64s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 64, true,
            |v, rm| (Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as i64 as u64) as u128)
    }
    fn f64_to_i64s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 64, true,
            |v, rm| (Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as i64 as u64) as u128)
    }
    fn f32_to_i32u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 32, false,
            |v, rm| Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as u32 as u128)
    }
    fn f64_to_i32u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 32, false,
            |v, rm| Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as u32 as u128)
    }
    fn f32_to_i64u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 64, false,
            |v, rm| Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as u64 as u128)
    }
    fn f64_to_i64u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 64, false,
            |v, rm| Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as u64 as u128)
    }
}

/// Errors from VEX operation execution.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OpError {
    /// Operation is not a unary operation.
    #[error("operation {0:?} is not unary")]
    NotUnary(IROp),
    /// Operation is not a binary operation.
    #[error("operation {0:?} is not binary")]
    NotBinary(IROp),
    /// Operation is not a ternary operation.
    #[error("operation {0:?} is not ternary")]
    NotTernary(IROp),
    /// Operation is not a quaternary operation.
    #[error("operation {0:?} is not quaternary")]
    NotQuaternary(IROp),
    /// Type mismatch.
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: IRType, got: IRType },
    /// Invalid float type.
    #[error("invalid float type: {0:?}")]
    InvalidFloatType(IRType),
    /// Symbolic float operations not supported.
    #[error("symbolic float operations not supported")]
    SymbolicFloatUnsupported,
    /// Unsupported vector operation.
    #[error("unsupported vector operation: {0}")]
    UnsupportedVectorOp(String),
    /// Raw/unimplemented opcode.
    #[error("raw/unimplemented opcode: {0}")]
    RawOpcode(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_op() {
        let ctx = SymContext::new_mock();

        let a = RustBV::concrete(5, 32);
        let b = RustBV::concrete(3, 32);

        let result = VEXOps::binop(IROp::Add(IRType::I32), a, b, &ctx).unwrap();
        assert_eq!(result.as_u64(), Some(8));
    }

    #[test]
    fn test_mul_widening() {
        let ctx = SymContext::new_mock();

        let a = RustBV::concrete(0xFFFFFFFF, 32);
        let b = RustBV::concrete(0xFFFFFFFF, 32);

        let result = VEXOps::binop(IROp::MullU(IRType::I32), a, b, &ctx).unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128(), Some(0xFFFFFFFE00000001));
    }

    #[test]
    fn test_comparison_ops() {
        let ctx = SymContext::new_mock();

        let a = RustBV::concrete(5, 32);
        let b = RustBV::concrete(10, 32);

        let lt = VEXOps::binop(IROp::CmpLTU(IRType::I32), a.clone(), b.clone(), &ctx).unwrap();
        assert_eq!(lt.as_u64(), Some(1));

        let eq = VEXOps::binop(IROp::CmpEQ(IRType::I32), a.clone(), b.clone(), &ctx).unwrap();
        assert_eq!(eq.as_u64(), Some(0));
    }

    #[test]
    fn test_divmod_u64_to_32_concrete() {
        let ctx = SymContext::new_mock();
        // 100 / 7 = 14 rem 2
        let dvd = RustBV::concrete(100, 64);
        let dvs = RustBV::concrete(7, 32);
        let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs, &ctx).unwrap();
        let v = result.as_u128().unwrap() as u64;
        assert_eq!(v & 0xFFFF_FFFF, 14, "quotient");
        assert_eq!((v >> 32) & 0xFFFF_FFFF, 2, "remainder");
    }

    #[test]
    fn test_divmod_u64_to_32_symbolic_dividend() {
        let ctx = SymContext::new_mock();
        let dvd = RustBV::symbolic(&ctx, "dvd", 64);
        let dvs = RustBV::concrete(7, 32);
        let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs, &ctx).unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.is_symbolic());
    }

    #[test]
    fn test_divmod_s64_to_32_symbolic_divisor() {
        let ctx = SymContext::new_mock();
        let dvd = RustBV::concrete(0xFFFF_FFFF_FFFF_FF9C, 64); // -100 as i64
        let dvs = RustBV::symbolic(&ctx, "dvs", 32);
        let result = VEXOps::binop(IROp::DivModS64to32, dvd, dvs, &ctx).unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.is_symbolic());
    }

    #[test]
    fn test_divmod_u128_to_64_concrete() {
        let ctx = SymContext::new_mock();
        let dvd = RustBV::concrete(1000, 128);
        let dvs = RustBV::concrete(13, 64);
        let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs, &ctx).unwrap();
        let v = result.as_u128().unwrap();
        assert_eq!(v & 0xFFFF_FFFF_FFFF_FFFF, 76, "quotient = 1000/13");
        assert_eq!((v >> 64) & 0xFFFF_FFFF_FFFF_FFFF, 12, "remainder = 1000%13");
    }

    #[test]
    fn test_divmod_u128_to_64_symbolic() {
        let ctx = SymContext::new_mock();
        let dvd = RustBV::symbolic(&ctx, "dvd128", 128);
        let dvs = RustBV::concrete(7, 64);
        let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs, &ctx).unwrap();
        assert_eq!(result.width(), 128);
        assert!(result.is_symbolic());
    }

    #[test]
    fn test_sign_extend() {
        let ctx = SymContext::new_mock();

        let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits

        let result = VEXOps::unop(
            IROp::SignExtend {
                from: IRType::I8,
                to: IRType::I32,
            },
            a,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
    }

    #[test]
    fn test_float_add_concrete() {
        let ctx = SymContext::new_mock();

        let a = RustBV::concrete(1.5f32.to_bits() as u128, 32);
        let b = RustBV::concrete(2.5f32.to_bits() as u128, 32);

        let result = VEXOps::binop(IROp::FAdd(IRType::F32), a, b, &ctx).unwrap();
        let result_f = f32::from_bits(result.as_u64().unwrap() as u32);
        assert!((result_f - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_qop_fmadd_concrete_f64() {
        let ctx = SymContext::new_mock();

        // 2.0 * 3.0 + 4.0 = 10.0
        let a = RustBV::concrete(2.0f64.to_bits() as u128, 64);
        let b = RustBV::concrete(3.0f64.to_bits() as u128, 64);
        let c = RustBV::concrete(4.0f64.to_bits() as u128, 64);

        let result = VEXOps::qop(IROp::FMAdd(IRType::F64), a, b, c, &ctx).unwrap();
        let result_f = f64::from_bits(result.as_u64().unwrap());
        assert!((result_f - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_qop_fmsub_concrete_f32() {
        let ctx = SymContext::new_mock();

        // 5.0 * 2.0 - 3.0 = 7.0
        let a = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let b = RustBV::concrete(2.0f32.to_bits() as u128, 32);
        let c = RustBV::concrete(3.0f32.to_bits() as u128, 32);

        let result = VEXOps::qop(IROp::FMSub(IRType::F32), a, b, c, &ctx).unwrap();
        let result_f = f32::from_bits(result.as_u64().unwrap() as u32);
        assert!((result_f - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_qop_rejects_non_quaternary() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0, 64);
        let b = RustBV::concrete(0, 64);
        let c = RustBV::concrete(0, 64);

        let err = VEXOps::qop(IROp::Add(IRType::I64), a, b, c, &ctx).unwrap_err();
        match err {
            OpError::NotQuaternary(_) => (),
            other => panic!("expected NotQuaternary, got {:?}", other),
        }
    }

    #[test]
    fn test_vector_add() {
        let ctx = SymContext::new_mock();

        // Two vectors of 4 x i32
        // [1, 2, 3, 4] + [10, 20, 30, 40] = [11, 22, 33, 44]
        let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
        let b: u128 = 10 | (20 << 32) | (30 << 64) | (40 << 96);

        let av = RustBV::concrete(a, 128);
        let bv = RustBV::concrete(b, 128);

        let result = VEXOps::binop(IROp::VAdd { elem: IRType::I32, count: 4 }, av, bv, &ctx).unwrap();
        let rv = result.as_u128().unwrap();

        assert_eq!(rv & 0xFFFFFFFF, 11);
        assert_eq!((rv >> 32) & 0xFFFFFFFF, 22);
        assert_eq!((rv >> 64) & 0xFFFFFFFF, 33);
        assert_eq!((rv >> 96) & 0xFFFFFFFF, 44);
    }

    #[test]
    fn test_vector_cmp_eq() {
        let ctx = SymContext::new_mock();

        // Two vectors of 4 x i32
        // [1, 2, 3, 4] == [1, 0, 3, 0] -> [0xFFFFFFFF, 0, 0xFFFFFFFF, 0]
        let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
        let b: u128 = 1 | (0 << 32) | (3 << 64) | (0 << 96);

        let av = RustBV::concrete(a, 128);
        let bv = RustBV::concrete(b, 128);

        let result = VEXOps::binop(IROp::VCmpEQ { elem: IRType::I32, count: 4 }, av, bv, &ctx).unwrap();
        let rv = result.as_u128().unwrap();

        assert_eq!(rv & 0xFFFFFFFF, 0xFFFFFFFF); // 1 == 1
        assert_eq!((rv >> 32) & 0xFFFFFFFF, 0);  // 2 != 0
        assert_eq!((rv >> 64) & 0xFFFFFFFF, 0xFFFFFFFF); // 3 == 3
        assert_eq!((rv >> 96) & 0xFFFFFFFF, 0);  // 4 != 0
    }

    #[test]
    fn test_concat() {
        let ctx = SymContext::new_mock();

        let hi = RustBV::concrete(0xDEAD, 16);
        let lo = RustBV::concrete(0xBEEF, 16);

        let result = VEXOps::binop(IROp::Concat { ty: IRType::I32 }, hi, lo, &ctx).unwrap();
        assert_eq!(result.as_u64(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_bit_manipulation() {
        let ctx = SymContext::new_mock();

        // CLZ (count leading zeros)
        let a = RustBV::concrete(0x0F00, 16); // binary: 0000111100000000
        let clz = VEXOps::unop(IROp::Clz(IRType::I16), a.clone(), &ctx).unwrap();
        assert_eq!(clz.as_u64(), Some(4)); // 4 leading zeros

        // CTZ (count trailing zeros)
        let ctz = VEXOps::unop(IROp::Ctz(IRType::I16), a.clone(), &ctx).unwrap();
        assert_eq!(ctz.as_u64(), Some(8)); // 8 trailing zeros

        // PopCount
        let pop = VEXOps::unop(IROp::PopCount(IRType::I16), a, &ctx).unwrap();
        assert_eq!(pop.as_u64(), Some(4)); // 4 bits set
    }

    #[test]
    fn test_vec_float_scalar_add() {
        let ctx = SymContext::new_mock();

        // SSE scalar add: ADDSS xmm0, xmm1
        // xmm0 = [4.0f, 0, 0, 0], xmm1 = [2.0f, 0, 0, 0]
        // Result: xmm0 = [6.0f, 0, 0, 0]
        let f4_bits = 4.0f32.to_bits() as u128;
        let f2_bits = 2.0f32.to_bits() as u128;

        let xmm0 = RustBV::concrete(f4_bits, 128);
        let xmm1 = RustBV::concrete(f2_bits, 128);

        let result = VEXOps::binop(IROp::VFAddS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let result_val = result.as_u128().unwrap();
        let result_f32 = f32::from_bits((result_val & 0xFFFFFFFF) as u32);

        assert!((result_f32 - 6.0).abs() < 0.0001, "Expected 6.0, got {}", result_f32);
    }

    #[test]
    fn test_vec_float_scalar_div() {
        let ctx = SymContext::new_mock();

        // SSE scalar div: DIVSS xmm0, xmm1
        // xmm0 = [6.0f, 0, 0, 0], xmm1 = [2.0f, 0, 0, 0]
        // Result: xmm0 = [3.0f, 0, 0, 0]
        let f6_bits = 6.0f32.to_bits() as u128;
        let f2_bits = 2.0f32.to_bits() as u128;

        let xmm0 = RustBV::concrete(f6_bits, 128);
        let xmm1 = RustBV::concrete(f2_bits, 128);

        let result = VEXOps::binop(IROp::VFDivS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let result_val = result.as_u128().unwrap();
        let result_f32 = f32::from_bits((result_val & 0xFFFFFFFF) as u32);

        assert!((result_f32 - 3.0).abs() < 0.0001, "Expected 3.0, got {}", result_f32);
    }

    /// Symbolic FAdd: solving `x + 2.0 == 5.0` should yield x == 3.0.
    ///
    /// This is the canonical "constraint propagation" check for the new Z3
    /// FP-theory wiring: previously the symbolic branch returned a fresh
    /// unconstrained symbol and the solver would accept any value of x.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_add_symbolic_constraint() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        let sum = VEXOps::binop(IROp::FAdd(IRType::F32), x.clone(), two, &ctx).unwrap();

        // Constrain: sum's IEEE bits == bits(5.0).
        let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let eq = sum.to_z3_ast()._eq(&five.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after FAdd symbolic constraint");

        let model_x = ctx.eval(&x).expect("eval(x) returned None");
        let result_f = f32::from_bits(model_x as u32);
        assert!(
            (result_f - 3.0).abs() < 1e-6,
            "Expected x == 3.0, got {}",
            result_f
        );
    }

    /// Symbolic FSqrt: `sqrt(x) == 4.0` should give x == 16.0.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_sqrt_symbolic_constraint() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "sqrt_x", 64);

        let sqrt_x = VEXOps::unop(IROp::FSqrt(IRType::F64), x.clone(), &ctx).unwrap();

        let four = RustBV::concrete(4.0f64.to_bits() as u128, 64);
        let eq = sqrt_x.to_z3_ast()._eq(&four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after FSqrt symbolic constraint");

        let model_x = ctx.eval(&x).expect("eval(x) returned None");
        let result_f = f64::from_bits(model_x as u64);
        assert!(
            (result_f - 16.0).abs() < 1e-9,
            "Expected x == 16.0, got {}",
            result_f
        );
    }

    /// Symbolic FCmpLT: bracket x with `1.0 < x < 2.0` via two symbolic
    /// FCmpLT comparisons. Solver should accept and produce x in (1.0, 2.0).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_cmp_lt_symbolic_constraint() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "fcmp_x", 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        // x < 2.0
        let lt_x_two =
            VEXOps::binop(IROp::FCmpLT(IRType::F32), x.clone(), two, &ctx).unwrap();
        // 1.0 < x
        let lt_one_x =
            VEXOps::binop(IROp::FCmpLT(IRType::F32), one, x.clone(), &ctx).unwrap();

        ctx.assume_true(&lt_x_two);
        ctx.assume_true(&lt_one_x);
        assert!(ctx.is_sat(), "expected SAT for 1.0 < x < 2.0");

        let model_x = ctx.eval(&x).expect("eval(x) returned None");
        let result_f = f32::from_bits(model_x as u32);
        assert!(
            result_f > 1.0 && result_f < 2.0,
            "Expected 1.0 < x < 2.0, got {}",
            result_f
        );
    }

    /// Symbolic VFAddS (ADDSS-style): `x_low + 2.0 == 5.0` should yield x_low == 3.0,
    /// and the upper 96 bits of `xmm0` must pass through unchanged.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_float_scalar_add_symbolic() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let xmm0 = RustBV::symbolic(&ctx, "xmm0", 128);
        // xmm1 = [2.0f, 0, 0, 0]
        let xmm1 = RustBV::concrete(2.0f32.to_bits() as u128, 128);

        let result = VEXOps::binop(
            IROp::VFAddS { elem: IRType::F32 },
            xmm0.clone(),
            xmm1,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        // Constrain low 32 bits of result to bits(5.0).
        let res_lo = result.extract(31, 0, &ctx);
        let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast()._eq(&five.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after VFAddS symbolic constraint");

        let model_x = ctx.eval(&xmm0).expect("eval(xmm0) returned None");
        let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 3.0).abs() < 1e-6,
            "Expected lane0 == 3.0, got {}",
            lane0
        );

        // Verify upper 96 bits of result equal upper 96 bits of xmm0 (passthrough).
        let upper_in = xmm0.extract(127, 32, &ctx);
        let upper_out = result.extract(127, 32, &ctx);
        let eq_upper = upper_in.to_z3_ast()._eq(&upper_out.to_z3_ast());
        ctx.add_constraint(eq_upper);
        assert!(ctx.is_sat(), "expected upper-bits passthrough to hold");
    }

    /// Symbolic VFSqrtS (SQRTSS-style): sqrt(low32(xmm)) == 4.0 → low32 == 16.0.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_float_scalar_sqrt_symbolic() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let xmm = RustBV::symbolic(&ctx, "xmm_sqrt", 128);

        let result = VEXOps::unop(IROp::VFSqrtS { elem: IRType::F32 }, xmm.clone(), &ctx).unwrap();
        assert_eq!(result.width(), 128);

        let res_lo = result.extract(31, 0, &ctx);
        let four = RustBV::concrete(4.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast()._eq(&four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after VFSqrtS symbolic constraint");

        let model_x = ctx.eval(&xmm).expect("eval(xmm) returned None");
        let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 16.0).abs() < 1e-4,
            "Expected lane0 == 16.0, got {}",
            lane0
        );
    }

    /// Symbolic VFMaxS (MAXSS-style): with `xmm0` symbolic and `xmm1 = [3.0, ...]`,
    /// constrain low32(result) == 5.0 — solver must pick xmm0.lane0 == 5.0 (since
    /// max(5.0, 3.0) == 5.0). Also a model where xmm0.lane0 == 1.0 must NOT satisfy
    /// the constraint (we don't test that here, but the ITE encoding guarantees it).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_float_scalar_max_symbolic() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let xmm0 = RustBV::symbolic(&ctx, "xmm_max", 128);
        let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

        let result = VEXOps::binop(
            IROp::VFMaxS { elem: IRType::F32 },
            xmm0.clone(),
            xmm1,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        let res_lo = result.extract(31, 0, &ctx);
        let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast()._eq(&five.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after VFMaxS == 5.0");

        let model_x = ctx.eval(&xmm0).expect("eval(xmm0) returned None");
        let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 5.0).abs() < 1e-6,
            "Expected lane0 == 5.0 (since max(lane0, 3.0) == 5.0), got {}",
            lane0
        );
    }

    /// Symbolic VFMinS (MINSS-style): with `xmm1 = [3.0, ...]` and target == 1.0,
    /// solver must pick xmm0.lane0 == 1.0 (since min(1.0, 3.0) == 1.0).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_float_scalar_min_symbolic() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let xmm0 = RustBV::symbolic(&ctx, "xmm_min", 128);
        let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

        let result = VEXOps::binop(
            IROp::VFMinS { elem: IRType::F32 },
            xmm0.clone(),
            xmm1,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        let res_lo = result.extract(31, 0, &ctx);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast()._eq(&one.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after VFMinS == 1.0");

        let model_x = ctx.eval(&xmm0).expect("eval(xmm0) returned None");
        let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 1.0).abs() < 1e-6,
            "Expected lane0 == 1.0 (since min(lane0, 3.0) == 1.0), got {}",
            lane0
        );
    }

    // =========================================================================
    // FP edge cases (angr-io1t)
    // =========================================================================

    /// RoundF32toInt with symbolic rm: value=-2.5f32, target=-3.0f32 forces
    /// rm low-2-bits == 1 (round toward -inf). Exercises the 4-way ITE built
    /// by build_fp_round_to_int_cached.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_round_f32_to_int_symbolic_rm() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let rm = RustBV::symbolic(&ctx, "rm_f32", 32);
        let value = RustBV::concrete((-2.5f32).to_bits() as u128, 32);

        let result =
            VEXOps::binop(IROp::RoundF32toInt, rm.clone(), value, &ctx).unwrap();
        let target = RustBV::concrete((-3.0f32).to_bits() as u128, 32);
        let eq = result.to_z3_ast()._eq(&target.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT for round(-2.5)==-3.0");

        let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
        assert_eq!(
            (model_rm as u32) & 0x3,
            1,
            "expected rm low2 bits == 1 (round toward -inf), got {}",
            model_rm & 0x3
        );
    }

    /// RoundF64toInt with symbolic rm: value=2.5f64, target=3.0f64 forces
    /// rm low-2-bits == 2 (round toward +inf). RNE on 2.5 ties to even (2).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_round_f64_to_int_symbolic_rm() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let rm = RustBV::symbolic(&ctx, "rm_f64", 32);
        let value = RustBV::concrete(2.5f64.to_bits() as u128, 64);

        let result =
            VEXOps::binop(IROp::RoundF64toInt, rm.clone(), value, &ctx).unwrap();
        let target = RustBV::concrete(3.0f64.to_bits() as u128, 64);
        let eq = result.to_z3_ast()._eq(&target.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT for round(2.5)==3.0");

        let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
        assert_eq!(
            (model_rm as u32) & 0x3,
            2,
            "expected rm low2 bits == 2 (round toward +inf), got {}",
            model_rm & 0x3
        );
    }

    /// F32→I32S with NaN: Rust `as` cast collapses NaN to 0.
    #[test]
    fn test_f32_to_i32s_nan() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(f32::NAN.to_bits() as u128, 32);
        let result = VEXOps::unop(IROp::F32toI32S, arg, &ctx).unwrap();
        assert_eq!(result.width(), 32);
        assert_eq!(result.as_u64(), Some(0), "NaN as i32 must be 0");
    }

    /// F32→I32S with +inf: saturates to i32::MAX.
    #[test]
    fn test_f32_to_i32s_pos_infinity() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(f32::INFINITY.to_bits() as u128, 32);
        let result = VEXOps::unop(IROp::F32toI32S, arg, &ctx).unwrap();
        assert_eq!(
            result.as_u64(),
            Some(i32::MAX as u32 as u64),
            "+inf as i32 must saturate to i32::MAX"
        );
    }

    /// F32→I32S with -inf: saturates to i32::MIN.
    #[test]
    fn test_f32_to_i32s_neg_infinity() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(f32::NEG_INFINITY.to_bits() as u128, 32);
        let result = VEXOps::unop(IROp::F32toI32S, arg, &ctx).unwrap();
        assert_eq!(
            result.as_u64(),
            Some(i32::MIN as u32 as u64),
            "-inf as i32 must saturate to i32::MIN"
        );
    }

    /// F32→I32S with very large magnitude: also saturates.
    #[test]
    fn test_f32_to_i32s_overflow() {
        let ctx = SymContext::new_mock();
        let big = RustBV::concrete(1e30f32.to_bits() as u128, 32);
        let result = VEXOps::unop(IROp::F32toI32S, big, &ctx).unwrap();
        assert_eq!(
            result.as_u64(),
            Some(i32::MAX as u32 as u64),
            "1e30 must saturate to i32::MAX"
        );

        let neg_big = RustBV::concrete((-1e30f32).to_bits() as u128, 32);
        let neg_result = VEXOps::unop(IROp::F32toI32S, neg_big, &ctx).unwrap();
        assert_eq!(
            neg_result.as_u64(),
            Some(i32::MIN as u32 as u64),
            "-1e30 must saturate to i32::MIN"
        );
    }

    /// I32S→F32 of INT32_MIN: -2^31 is exactly representable in F32.
    #[test]
    fn test_i32s_to_f32_int_min() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(i32::MIN as u32 as u128, 32);
        let result = VEXOps::unop(IROp::I32StoF32, arg, &ctx).unwrap();
        let f = f32::from_bits(result.as_u64().unwrap() as u32);
        assert_eq!(f, -2147483648.0f32, "I32_MIN must round-trip to -2^31 as f32");
    }

    /// F64→F32 (no-rm unop) precision overflow: 1e300 exceeds f32::MAX.
    /// Rust `as f32` saturates toward +inf, matching IEEE-754 behavior under RNE.
    #[test]
    fn test_f64_to_f32_overflow() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(1e300f64.to_bits() as u128, 64);
        let result = VEXOps::unop(IROp::F64toF32, arg, &ctx).unwrap();
        assert_eq!(result.width(), 32);
        let f = f32::from_bits(result.as_u64().unwrap() as u32);
        assert!(f.is_infinite() && f.is_sign_positive(), "1e300 → +inf, got {}", f);
    }

    /// F64→F32 of NaN: result is still NaN. Just check is_nan; the exact
    /// payload bits aren't part of the contract.
    #[test]
    fn test_f64_to_f32_nan() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(f64::NAN.to_bits() as u128, 64);
        let result = VEXOps::unop(IROp::F64toF32, arg, &ctx).unwrap();
        assert_eq!(result.width(), 32);
        let f = f32::from_bits(result.as_u64().unwrap() as u32);
        assert!(f.is_nan(), "NaN must remain NaN after F64→F32");
    }

    /// F64→F32 of -infinity: preserved as -infinity.
    #[test]
    fn test_f64_to_f32_neg_infinity() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(f64::NEG_INFINITY.to_bits() as u128, 64);
        let result = VEXOps::unop(IROp::F64toF32, arg, &ctx).unwrap();
        let f = f32::from_bits(result.as_u64().unwrap() as u32);
        assert!(f.is_infinite() && f.is_sign_negative(), "-inf preserved");
    }

    /// VFSubS concrete lane isolation: SUBSS xmm0, xmm1 — only lane 0 changes,
    /// upper 96 bits of xmm0 pass through unchanged.
    #[test]
    fn test_vec_float_scalar_sub_concrete_lane_isolation() {
        let ctx = SymContext::new_mock();

        // xmm0: lane0=10.0, upper bits = 0xDEAD_BEEF_CAFE_BABE_1234_5678 (96 bits)
        let upper_pattern: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678u128 << 32;
        let xmm0_bits = upper_pattern | (10.0f32.to_bits() as u128);
        let xmm0 = RustBV::concrete(xmm0_bits, 128);
        let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

        let result =
            VEXOps::binop(IROp::VFSubS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let rv = result.as_u128().unwrap();
        let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
        assert!((lane0 - 7.0).abs() < 1e-6, "10.0 - 3.0 == 7.0, got {}", lane0);
        assert_eq!(rv & !0xFFFF_FFFFu128, upper_pattern, "upper 96 bits must pass through");
    }

    /// VFMulS concrete lane isolation: MULSS xmm0, xmm1.
    #[test]
    fn test_vec_float_scalar_mul_concrete_lane_isolation() {
        let ctx = SymContext::new_mock();

        let upper_pattern: u128 = 0xFEED_FACE_BAAD_F00D_8BAD_F00Du128 << 32;
        let xmm0_bits = upper_pattern | (4.0f32.to_bits() as u128);
        let xmm0 = RustBV::concrete(xmm0_bits, 128);
        let xmm1 = RustBV::concrete(2.5f32.to_bits() as u128, 128);

        let result =
            VEXOps::binop(IROp::VFMulS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let rv = result.as_u128().unwrap();
        let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
        assert!((lane0 - 10.0).abs() < 1e-6, "4.0 * 2.5 == 10.0, got {}", lane0);
        assert_eq!(rv & !0xFFFF_FFFFu128, upper_pattern, "upper 96 bits must pass through");
    }

    /// VFMaxS concrete with lane0=NaN: Rust `>` returns false for NaN, so
    /// max picks the right operand. Documents the SSE max-is-not-IEEE-max
    /// semantics encoded by the ITE in vec_float_scalar_lane_minmax.
    #[test]
    fn test_vec_float_scalar_max_nan_concrete() {
        let ctx = SymContext::new_mock();

        let xmm0 = RustBV::concrete(f32::NAN.to_bits() as u128, 128);
        let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

        let result =
            VEXOps::binop(IROp::VFMaxS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let rv = result.as_u128().unwrap();
        let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
        // NaN > 3.0 is false, so max picks 3.0 (the right operand).
        assert!(
            (lane0 - 3.0).abs() < 1e-6,
            "MAXSS(NaN, 3.0) returns the right operand, got {}",
            lane0
        );
    }

    /// Symbolic ShlN16x8: shift count is symbolic; constrain to 4 and verify
    /// each lane is `lane << 4`. Exercises the symbolic-shift fallback that
    /// builds Z3 `bvshl` expressions per lane.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_shl_n_symbolic_shift() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();

        // Vector: [0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008]
        let mut v: u128 = 0;
        for i in 0..8u32 {
            v |= ((i + 1) as u128) << (i * 16);
        }
        let vec = RustBV::concrete(v, 128);
        let shift = RustBV::symbolic(&ctx, "shl_amt", 8);

        let result = VEXOps::binop(
            IROp::VShlN { elem: IRType::I16, count: 8 },
            vec,
            shift.clone(),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        // Constrain shift == 4.
        let four = RustBV::concrete(4, 8);
        let eq = shift.to_z3_ast()._eq(&four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        for i in 0..8u32 {
            let lane = (model >> (i * 16)) & 0xFFFF;
            let expected = ((i as u128 + 1) << 4) & 0xFFFF;
            assert_eq!(lane, expected, "lane {} expected {:#x}, got {:#x}", i, expected, lane);
        }
    }

    /// Symbolic ShrN32x4: shift count is symbolic; constrain to 8 and verify
    /// each lane is `lane >> 8`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_shr_n_symbolic_shift() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();

        // Vector: [0xAABBCCDD, 0x11223344, 0xDEADBEEF, 0xCAFEBABE]
        let lanes: [u32; 4] = [0xAABBCCDD, 0x11223344, 0xDEADBEEF, 0xCAFEBABE];
        let mut v: u128 = 0;
        for (i, lane) in lanes.iter().enumerate() {
            v |= (*lane as u128) << (i * 32);
        }
        let vec = RustBV::concrete(v, 128);
        let shift = RustBV::symbolic(&ctx, "shr_amt", 8);

        let result = VEXOps::binop(
            IROp::VShrN { elem: IRType::I32, count: 4 },
            vec,
            shift.clone(),
            &ctx,
        )
        .unwrap();

        let eight = RustBV::concrete(8, 8);
        let eq = shift.to_z3_ast()._eq(&eight.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after constraining shift == 8");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        for (i, lane) in lanes.iter().enumerate() {
            let got = ((model >> (i * 32)) & 0xFFFF_FFFF) as u32;
            let expected = lane >> 8;
            assert_eq!(got, expected, "lane {} expected {:#x}, got {:#x}", i, expected, got);
        }
    }

    /// Symbolic SarN16x8 with negative lanes: shift count is symbolic; constrain
    /// to 4 and verify sign-extending shift (negative values stay negative).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_sar_n_symbolic_shift() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();

        // Lanes: mix of positive and negative i16 values.
        let lanes: [i16; 8] = [-1, -32768, -16, 0, 1, 0x4000, -2, 256];
        let mut v: u128 = 0;
        for (i, lane) in lanes.iter().enumerate() {
            v |= ((*lane as u16) as u128) << (i * 16);
        }
        let vec = RustBV::concrete(v, 128);
        let shift = RustBV::symbolic(&ctx, "sar_amt", 8);

        let result = VEXOps::binop(
            IROp::VSarN { elem: IRType::I16, count: 8 },
            vec,
            shift.clone(),
            &ctx,
        )
        .unwrap();

        let four = RustBV::concrete(4, 8);
        let eq = shift.to_z3_ast()._eq(&four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        for (i, lane) in lanes.iter().enumerate() {
            let got = ((model >> (i * 16)) & 0xFFFF) as u16 as i16;
            let expected = lane >> 4;  // arithmetic shift in Rust on i16
            assert_eq!(got, expected, "lane {} expected {}, got {}", i, expected, got);
        }
    }

    /// Symbolic ShlN with unbounded shift: just verify a Z3 expression is
    /// produced rather than an UnsupportedVectorOp error. Documents the
    /// "fully unconstrained" path stays inside the solver.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_shl_n_unbounded_shift() {
        let ctx = SymContext::new_mock();

        let vec = RustBV::concrete(0x1111_2222_3333_4444u128, 64);
        let shift = RustBV::symbolic(&ctx, "shl_amt_free", 8);

        let result = VEXOps::binop(
            IROp::VShlN { elem: IRType::I16, count: 4 },
            vec,
            shift,
            &ctx,
        )
        .expect("unbounded symbolic shift must not error");
        assert_eq!(result.width(), 64);
    }
}
