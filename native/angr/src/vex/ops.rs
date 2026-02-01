//! VEX operation implementations.
//!
//! This module implements VEX operations using a parameterized approach.
//! Instead of ~200 separate implementations (e.g., add8, add16, add32, add64),
//! we have a single implementation per operation type that handles all widths.

use crate::symbolic::{RustBV, SymContext};

use super::ir::{IROp, IRType};

/// VEX operation executor.
///
/// This struct provides methods to execute VEX operations on `RustBV` values.
pub struct VEXOps;

impl VEXOps {
    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// Execute a unary operation.
    pub fn unop<'ctx>(
        op: IROp,
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        match op {
            IROp::Not(ty) => {
                debug_assert_eq!(arg.width(), ty.bits());
                Ok(arg.not(ctx))
            }

            IROp::Neg(ty) => {
                debug_assert_eq!(arg.width(), ty.bits());
                Ok(arg.neg(ctx))
            }

            IROp::Clz(ty) => {
                debug_assert_eq!(arg.width(), ty.bits());
                Ok(arg.clz(ctx))
            }

            IROp::Ctz(ty) => {
                debug_assert_eq!(arg.width(), ty.bits());
                Ok(arg.ctz(ctx))
            }

            IROp::PopCount(ty) => {
                debug_assert_eq!(arg.width(), ty.bits());
                Ok(arg.popcount(ctx))
            }

            // Sign/Zero extension
            IROp::SignExtend { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                Ok(arg.sign_extend(to.bits(), ctx))
            }

            IROp::ZeroExtend { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                Ok(arg.zero_extend(to.bits(), ctx))
            }

            // Truncation
            IROp::Truncate { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                Ok(arg.truncate(to.bits(), ctx))
            }

            // Extraction (unary form - low_bit is encoded in opcode)
            IROp::Extract { from, to, low_bit } => {
                debug_assert_eq!(arg.width(), from.bits());
                let hi = low_bit as u32 + to.bits() - 1;
                let lo = low_bit as u32;
                Ok(arg.extract(hi, lo, ctx))
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
            IROp::RoundF32toInt => Self::round_f32_to_int(arg, ctx),
            IROp::RoundF64toInt => Self::round_f64_to_int(arg, ctx),

            // Vector not
            IROp::VNot(ty) => {
                debug_assert_eq!(arg.width(), ty.bits());
                Ok(arg.not(ctx))
            }

            // Reinterpret (just changes type, not bits)
            IROp::Reinterpret { from, to } => {
                debug_assert_eq!(arg.width(), from.bits());
                if from.bits() == to.bits() {
                    Ok(arg)
                } else if to.bits() > from.bits() {
                    Ok(arg.zero_extend(to.bits(), ctx))
                } else {
                    Ok(arg.truncate(to.bits(), ctx))
                }
            }

            _ => Err(OpError::NotUnary(op)),
        }
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// Execute a binary operation.
    pub fn binop<'ctx>(
        op: IROp,
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        match op {
            // Arithmetic
            IROp::Add(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.add(&right, ctx))
            }

            IROp::Sub(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.sub(&right, ctx))
            }

            IROp::Mul(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.mul(&right, ctx))
            }

            IROp::DivU(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.udiv(&right, ctx))
            }

            IROp::DivS(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.sdiv(&right, ctx))
            }

            IROp::ModU(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.urem(&right, ctx))
            }

            IROp::ModS(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.srem(&right, ctx))
            }

            // Widening multiply
            IROp::MullU(ty) => Self::widening_mul(left, right, ty, false, ctx),
            IROp::MullS(ty) => Self::widening_mul(left, right, ty, true, ctx),

            // High half of multiplication
            IROp::MulHi { ty, signed } => Self::mul_hi(left, right, ty, signed, ctx),

            // DivMod: 64-bit / 32-bit -> 64-bit (low=quotient, high=remainder)
            IROp::DivModU64to32 => Self::divmod_64_to_32(left, right, false, ctx),
            IROp::DivModS64to32 => Self::divmod_64_to_32(left, right, true, ctx),

            // Bitwise
            IROp::And(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.and(&right, ctx))
            }

            IROp::Or(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.or(&right, ctx))
            }

            IROp::Xor(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.xor(&right, ctx))
            }

            // Shifts
            IROp::Shl(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                // Shift amount might be different width, adjust
                let amt = if right.width() != left.width() {
                    if right.width() > left.width() {
                        right.truncate(left.width(), ctx)
                    } else {
                        right.zero_extend(left.width(), ctx)
                    }
                } else {
                    right
                };
                Ok(left.shl(&amt, ctx))
            }

            IROp::Shr(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                let amt = if right.width() != left.width() {
                    if right.width() > left.width() {
                        right.truncate(left.width(), ctx)
                    } else {
                        right.zero_extend(left.width(), ctx)
                    }
                } else {
                    right
                };
                Ok(left.lshr(&amt, ctx))
            }

            IROp::Sar(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                let amt = if right.width() != left.width() {
                    if right.width() > left.width() {
                        right.truncate(left.width(), ctx)
                    } else {
                        right.zero_extend(left.width(), ctx)
                    }
                } else {
                    right
                };
                Ok(left.ashr(&amt, ctx))
            }

            // Comparisons
            IROp::CmpEQ(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.eq(&right, ctx))
            }

            IROp::CmpNE(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.ne(&right, ctx))
            }

            IROp::CmpLT(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.slt(&right, ctx))
            }

            IROp::CmpLE(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.sle(&right, ctx))
            }

            IROp::CmpLTU(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.ult(&right, ctx))
            }

            IROp::CmpLEU(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.ule(&right, ctx))
            }

            // Float arithmetic
            IROp::FAdd(ty) => Self::float_add(left, right, ty, ctx),
            IROp::FSub(ty) => Self::float_sub(left, right, ty, ctx),
            IROp::FMul(ty) => Self::float_mul(left, right, ty, ctx),
            IROp::FDiv(ty) => Self::float_div(left, right, ty, ctx),

            // Float comparisons
            IROp::FCmpEQ(ty) => Self::float_cmp_eq(left, right, ty, ctx),
            IROp::FCmpLT(ty) => Self::float_cmp_lt(left, right, ty, ctx),
            IROp::FCmpLE(ty) => Self::float_cmp_le(left, right, ty, ctx),

            // Scalar-in-vector float operations (SSE scalar ops)
            IROp::VFAddS { elem } => Self::vec_float_scalar_op(left, right, elem, "add", ctx),
            IROp::VFSubS { elem } => Self::vec_float_scalar_op(left, right, elem, "sub", ctx),
            IROp::VFMulS { elem } => Self::vec_float_scalar_op(left, right, elem, "mul", ctx),
            IROp::VFDivS { elem } => Self::vec_float_scalar_op(left, right, elem, "div", ctx),

            // Vector bitwise
            IROp::VAnd(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.and(&right, ctx))
            }

            IROp::VOr(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.or(&right, ctx))
            }

            IROp::VXor(ty) => {
                debug_assert_eq!(left.width(), ty.bits());
                debug_assert_eq!(right.width(), ty.bits());
                Ok(left.xor(&right, ctx))
            }

            // Concatenate
            IROp::Concat { ty } => {
                let result = left.concat(&right, ctx);
                debug_assert_eq!(result.width(), ty.bits());
                Ok(result)
            }

            // Vector operations (simplified - just doing element-wise)
            IROp::VAdd { elem, count } => Self::vec_binop(left, right, elem, count, "add", ctx),
            IROp::VSub { elem, count } => Self::vec_binop(left, right, elem, count, "sub", ctx),
            IROp::VMul { elem, count } => Self::vec_binop(left, right, elem, count, "mul", ctx),

            // Vector compare operations
            IROp::VCmpEQ { elem, count } => Self::vec_cmp(left, right, elem, count, "eq", ctx),
            IROp::VCmpGT { elem, count } => Self::vec_cmp(left, right, elem, count, "gt", ctx),

            // Vector interleave
            IROp::VInterleaveLO { elem } => Self::vec_interleave_lo(left, right, elem, ctx),
            IROp::VInterleaveHI { elem } => Self::vec_interleave_hi(left, right, elem, ctx),

            // Raw opcode
            IROp::Raw(code) => Err(OpError::RawOpcode(code)),

            _ => Err(OpError::NotBinary(op)),
        }
    }

    // =========================================================================
    // Ternary Operations
    // =========================================================================

    /// Execute a ternary operation (for ITE, etc.).
    pub fn ternop<'ctx>(
        op: IROp,
        arg1: RustBV<'ctx>,
        arg2: RustBV<'ctx>,
        arg3: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        match op {
            // Extraction takes (value, start_bit_as_u8, length_as_u8)
            // Note: This is for cases where extract is done as a ternary op
            IROp::Extract { from, to, low_bit } => {
                debug_assert_eq!(arg1.width(), from.bits());
                let hi = low_bit as u32 + to.bits() - 1;
                let lo = low_bit as u32;
                Ok(arg1.extract(hi, lo, ctx))
            }
            _ => Err(OpError::NotTernary(op)),
        }
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Widening multiply.
    fn widening_mul<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        signed: bool,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        let in_width = ty.bits();
        let out_width = in_width * 2;

        // Extend both operands
        let (left_ext, right_ext) = if signed {
            (
                left.sign_extend(out_width, ctx),
                right.sign_extend(out_width, ctx),
            )
        } else {
            (
                left.zero_extend(out_width, ctx),
                right.zero_extend(out_width, ctx),
            )
        };

        Ok(left_ext.mul(&right_ext, ctx))
    }

    /// High half of multiplication.
    fn mul_hi<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        signed: bool,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        let width = ty.bits();
        let double_width = width * 2;

        // Extend and multiply
        let (left_ext, right_ext) = if signed {
            (
                left.sign_extend(double_width, ctx),
                right.sign_extend(double_width, ctx),
            )
        } else {
            (
                left.zero_extend(double_width, ctx),
                right.zero_extend(double_width, ctx),
            )
        };

        let product = left_ext.mul(&right_ext, ctx);

        // Extract high half
        Ok(product.extract(double_width - 1, width, ctx))
    }

    /// DivMod: 64-bit dividend / 32-bit divisor -> 64-bit result.
    /// Low 32 bits = quotient, High 32 bits = remainder.
    fn divmod_64_to_32<'ctx>(
        dividend: RustBV<'ctx>,
        divisor: RustBV<'ctx>,
        signed: bool,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

        // For symbolic values, we'd need to implement symbolic division
        // For now, fall back to concrete evaluation if possible
        Err(OpError::UnsupportedVectorOp("symbolic DivMod".to_string()))
    }

    /// Vector element-wise binary operation.
    fn vec_binop<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        elem: IRType,
        count: u8,
        op: &str,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values, we can do this efficiently
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let hi = lo + elem_width - 1;
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
        let mut elements: Vec<RustBV<'ctx>> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            let res_elem = match op {
                "add" => l_elem.add(&r_elem, ctx),
                "sub" => l_elem.sub(&r_elem, ctx),
                "mul" => l_elem.mul(&r_elem, ctx),
                _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
            };

            elements.push(res_elem);
        }

        // Concatenate from high to low
        let mut result = elements.pop().unwrap();
        while let Some(elem) = elements.pop() {
            result = result.concat(&elem, ctx);
        }

        Ok(result)
    }

    // =========================================================================
    // Vector Comparison Operations
    // =========================================================================

    /// Vector element-wise comparison.
    fn vec_cmp<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        elem: IRType,
        count: u8,
        op: &str,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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
        let mut elements: Vec<RustBV<'ctx>> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            let cmp_result = match op {
                "eq" => l_elem.eq(&r_elem, ctx),
                "gt" => l_elem.sgt(&r_elem, ctx),
                _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
            };

            // Extend the 1-bit result to full element width (all 1s or all 0s)
            let extended = cmp_result.sign_extend(elem_width, ctx);
            elements.push(extended);
        }

        // Concatenate
        let mut result = elements.pop().unwrap();
        while let Some(elem) = elements.pop() {
            result = result.concat(&elem, ctx);
        }

        Ok(result)
    }

    /// Vector interleave low halves.
    fn vec_interleave_lo<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        elem: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

                result |= l_elem << dst_lo;
                result |= r_elem << (dst_lo + elem_width);
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic case
        let mut elements: Vec<RustBV<'ctx>> = Vec::new();

        for i in 0..half_count {
            let src_lo = (i as u32) * elem_width;
            let src_hi = src_lo + elem_width - 1;

            let l_elem = left.extract(src_hi, src_lo, ctx);
            let r_elem = right.extract(src_hi, src_lo, ctx);

            elements.push(l_elem);
            elements.push(r_elem);
        }

        // Concatenate from high to low
        elements.reverse();
        let mut result = elements.pop().unwrap();
        while let Some(elem) = elements.pop() {
            result = elem.concat(&result, ctx);
        }

        Ok(result)
    }

    /// Vector interleave high halves.
    fn vec_interleave_hi<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        elem: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

                result |= l_elem << dst_lo;
                result |= r_elem << (dst_lo + elem_width);
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic case
        let mut elements: Vec<RustBV<'ctx>> = Vec::new();

        for i in 0..half_count {
            let src_lo = ((half_count + i) as u32) * elem_width;
            let src_hi = src_lo + elem_width - 1;

            let l_elem = left.extract(src_hi, src_lo, ctx);
            let r_elem = right.extract(src_hi, src_lo, ctx);

            elements.push(l_elem);
            elements.push(r_elem);
        }

        // Concatenate from high to low
        elements.reverse();
        let mut result = elements.pop().unwrap();
        while let Some(elem) = elements.pop() {
            result = elem.concat(&result, ctx);
        }

        Ok(result)
    }

    // =========================================================================
    // Float Operations (using bit manipulation for now)
    // =========================================================================

    fn float_neg<'ctx>(
        arg: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        // Flip the sign bit
        let sign_bit = match ty {
            IRType::F32 => 31,
            IRType::F64 => 63,
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        let mask = RustBV::concrete(1u128 << sign_bit, arg.width());
        Ok(arg.xor(&mask, ctx))
    }

    fn float_abs<'ctx>(
        arg: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        // Clear the sign bit
        let mask = match ty {
            IRType::F32 => RustBV::concrete(0x7FFFFFFF, 32),
            IRType::F64 => RustBV::concrete(0x7FFFFFFFFFFFFFFF, 64),
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        Ok(arg.and(&mask, ctx))
    }

    fn float_sqrt<'ctx>(
        arg: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_add<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_sub<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_mul<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_div<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    /// Scalar float operation in vector (SSE scalar ops like ADDSS, DIVSS).
    /// Operates on element 0 only, passes through other elements from left operand.
    fn vec_float_scalar_op<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        elem: IRType,
        op: &str,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_cmp_eq<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_cmp_lt<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    fn float_cmp_le<'ctx>(
        left: RustBV<'ctx>,
        right: RustBV<'ctx>,
        ty: IRType,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
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

    // Float conversions for concrete values
    fn f32_to_f64<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f32::from_bits(v as u32);
            let result = (f as f64).to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f64_to_f32<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f64::from_bits(v as u64);
            let result = (f as f32).to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i32s_to_f32<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as i32;
            let result = (i as f32).to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i32s_to_f64<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as i32;
            let result = (i as f64).to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i64s_to_f64<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as i64;
            let result = (i as f64).to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f32_to_i32s<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f32::from_bits(v as u32);
            let result = (f as i32) as u32;
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f64_to_i32s<'ctx>(
        arg: RustBV<'ctx>,
        ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f64::from_bits(v as u64);
            let result = (f as i32) as u32;
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f64_to_i64s<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f64::from_bits(v as u64);
            let result = f as i64 as u64;
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i64s_to_f32<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as i64;
            let result = (i as f32).to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i32u_to_f32<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as u32;
            let result = (i as f32).to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i32u_to_f64<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as u32;
            let result = (i as f64).to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i64u_to_f32<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as u64;
            let result = (i as f32).to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn i64u_to_f64<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let i = v as u64;
            let result = (i as f64).to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f32_to_i64s<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f32::from_bits(v as u32);
            let result = f as i64 as u64;
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f32_to_i32u<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f32::from_bits(v as u32);
            let result = f as u32;
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f64_to_i32u<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f64::from_bits(v as u64);
            let result = f as u32;
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f32_to_i64u<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f32::from_bits(v as u32);
            let result = f as u64;
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn f64_to_i64u<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f64::from_bits(v as u64);
            let result = f as u64;
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn round_f32_to_int<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f32::from_bits(v as u32);
            let result = f.round().to_bits();
            return Ok(RustBV::concrete(result as u128, 32));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }

    fn round_f64_to_int<'ctx>(
        arg: RustBV<'ctx>,
        _ctx: &'ctx SymContext<'ctx>,
    ) -> Result<RustBV<'ctx>, OpError> {
        if let Some(v) = arg.as_u128() {
            let f = f64::from_bits(v as u64);
            let result = f.round().to_bits();
            return Ok(RustBV::concrete(result as u128, 64));
        }
        Err(OpError::SymbolicFloatUnsupported)
    }
}

/// Errors from VEX operation execution.
#[derive(Debug, Clone)]
pub enum OpError {
    /// Operation is not a unary operation.
    NotUnary(IROp),
    /// Operation is not a binary operation.
    NotBinary(IROp),
    /// Operation is not a ternary operation.
    NotTernary(IROp),
    /// Type mismatch.
    TypeMismatch { expected: IRType, got: IRType },
    /// Invalid float type.
    InvalidFloatType(IRType),
    /// Symbolic float operations not supported.
    SymbolicFloatUnsupported,
    /// Unsupported vector operation.
    UnsupportedVectorOp(String),
    /// Raw/unimplemented opcode.
    RawOpcode(u32),
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpError::NotUnary(op) => write!(f, "operation {:?} is not unary", op),
            OpError::NotBinary(op) => write!(f, "operation {:?} is not binary", op),
            OpError::NotTernary(op) => write!(f, "operation {:?} is not ternary", op),
            OpError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {:?}, got {:?}", expected, got)
            }
            OpError::InvalidFloatType(ty) => write!(f, "invalid float type: {:?}", ty),
            OpError::SymbolicFloatUnsupported => write!(f, "symbolic float operations not supported"),
            OpError::UnsupportedVectorOp(op) => write!(f, "unsupported vector operation: {}", op),
            OpError::RawOpcode(code) => write!(f, "raw/unimplemented opcode: {}", code),
        }
    }
}

impl std::error::Error for OpError {}

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
