//! VEX operation implementations.
//!
//! This module implements VEX operations using a parameterized approach.
//! Instead of ~200 separate implementations (e.g., add8, add16, add32, add64),
//! we have a single implementation per operation type that handles all widths.

use crate::symbolic::{RustBV, SymContext};

use super::ir::{IROp, IRType};

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

        // Get the shift amount (usually 8-bit immediate)
        let shift = match shift_amt.as_u128() {
            Some(s) => s as u32,
            None => return Err(OpError::UnsupportedVectorOp("symbolic shift amount".to_string())),
        };

        // If shift >= element width, result is all zeros
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

        // Symbolic case - do element-wise
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let shift_bv = RustBV::concrete(shift as u128, elem_width);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.shl_into(shift_bv.clone(), ctx);
            elements.push(shifted);
        }

        Ok(Self::concat_le_elements(elements, ctx))
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

        let shift = match shift_amt.as_u128() {
            Some(s) => s as u32,
            None => return Err(OpError::UnsupportedVectorOp("symbolic shift amount".to_string())),
        };

        // If shift >= element width, result is all zeros
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

        // Symbolic case
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let shift_bv = RustBV::concrete(shift as u128, elem_width);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.lshr_into(shift_bv.clone(), ctx);
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

        let shift = match shift_amt.as_u128() {
            Some(s) => s as u32,
            None => return Err(OpError::UnsupportedVectorOp("symbolic shift amount".to_string())),
        };

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

        // Symbolic case
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let shift_bv = RustBV::concrete(shift as u128, elem_width);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.ashr_into(shift_bv.clone(), ctx);
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

        // For symbolic, we'd need Z3 float theory or leave unconstrained
        // For now, create a fresh symbolic value (imprecise but sound)
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
    }

    /// Scalar float operation in vector (SSE scalar ops like ADDSS, DIVSS).
    /// Operates on element 0 only, passes through other elements from left operand.
    #[inline]
    fn vec_float_scalar_op(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        op: &str,
        _ctx: &SymContext,
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
        Err(OpError::SymbolicFloatUnsupported)
    }

    /// Scalar sqrt in vector (SQRTSS/SQRTSD).
    fn vec_float_scalar_sqrt(
        arg: RustBV,
        elem: IRType,
        _ctx: &SymContext,
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
        Err(OpError::SymbolicFloatUnsupported)
    }

    /// Scalar max in vector (MAXSS/MAXSD).
    fn vec_float_scalar_max(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        _ctx: &SymContext,
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
        Err(OpError::SymbolicFloatUnsupported)
    }

    /// Scalar min in vector (MINSS/MINSD).
    fn vec_float_scalar_min(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        _ctx: &SymContext,
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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

    /// Float-to-float or int-to-float conversion (no rounding needed).
    fn float_convert_simple(arg: RustBV, convert: fn(u128) -> u128, out_bits: u32) -> Result<RustBV, OpError> {
        if let Some(v) = arg.as_u128() {
            return Ok(RustBV::concrete(convert(v), out_bits));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    /// Float-to-int conversion with round-ties-to-even (unary, no rounding mode arg).
    fn float_to_int_rte(arg: RustBV, convert: fn(u128) -> u128, out_bits: u32) -> Result<RustBV, OpError> {
        if let Some(v) = arg.as_u128() {
            return Ok(RustBV::concrete(convert(v), out_bits));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    // --- Float-to-float conversions ---
    fn f32_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| (f32::from_bits(v as u32) as f64).to_bits() as u128, 64)
    }
    fn f64_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| (f64::from_bits(v as u64) as f32).to_bits() as u128, 32)
    }

    // --- Int-to-float conversions ---
    fn i32s_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as i32) as f32).to_bits() as u128, 32)
    }
    fn i32s_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as i32) as f64).to_bits() as u128, 64)
    }
    fn i64s_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as i64) as f32).to_bits() as u128, 32)
    }
    fn i64s_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as i64) as f64).to_bits() as u128, 64)
    }
    fn i32u_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as u32) as f32).to_bits() as u128, 32)
    }
    fn i32u_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as u32) as f64).to_bits() as u128, 64)
    }
    fn i64u_to_f32(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as u64) as f32).to_bits() as u128, 32)
    }
    fn i64u_to_f64(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_simple(arg, |v| ((v as u64) as f64).to_bits() as u128, 64)
    }

    // --- Float-to-int conversions (round ties to even) ---
    fn f32_to_i32s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| (Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as i32 as u32) as u128, 32)
    }
    fn f64_to_i32s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| (Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as i32 as u32) as u128, 32)
    }
    fn f32_to_i64s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| (Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as i64 as u64) as u128, 64)
    }
    fn f64_to_i64s(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| (Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as i64 as u64) as u128, 64)
    }
    fn f32_to_i32u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as u32 as u128, 32)
    }
    fn f64_to_i32u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as u32 as u128, 32)
    }
    fn f32_to_i64u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| Self::round_ties_to_even_f32(f32::from_bits(v as u32)) as u64 as u128, 64)
    }
    fn f64_to_i64u(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rte(arg, |v| Self::round_ties_to_even_f64(f64::from_bits(v as u64)) as u64 as u128, 64)
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
        Err(OpError::SymbolicFloatUnsupported)
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
        Err(OpError::SymbolicFloatUnsupported)
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

    /// Float conversion with rounding mode (binop: rm, value -> result).
    fn float_convert_rm(rm: RustBV, arg: RustBV, convert: fn(u128, u32) -> u128, out_bits: u32) -> Result<RustBV, OpError> {
        if let (Some(rm_val), Some(v)) = (rm.as_u128(), arg.as_u128()) {
            return Ok(RustBV::concrete(convert(v, rm_val as u32), out_bits));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    // --- Rounding-mode float conversions (binop variants) ---
    fn f64_to_f32_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        // Note: ignores rounding mode, uses direct cast
        Self::float_convert_rm(rm, arg, |v, _rm| (f64::from_bits(v as u64) as f32).to_bits() as u128, 32)
    }
    fn f32_to_i32s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| (Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as i32 as u32) as u128, 32)
    }
    fn f64_to_i32s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| (Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as i32 as u32) as u128, 32)
    }
    fn f32_to_i64s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| (Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as i64 as u64) as u128, 64)
    }
    fn f64_to_i64s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| (Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as i64 as u64) as u128, 64)
    }
    fn f32_to_i32u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as u32 as u128, 32)
    }
    fn f64_to_i32u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as u32 as u128, 32)
    }
    fn f32_to_i64u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as u64 as u128, 64)
    }
    fn f64_to_i64u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_convert_rm(rm, arg, |v, rm| Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as u64 as u128, 64)
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
}
