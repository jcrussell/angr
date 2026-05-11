//! VEX operation implementations.
//!
//! This module implements VEX operations using a parameterized approach.
//! Instead of ~200 separate implementations (e.g., add8, add16, add32, add64),
//! we have a single implementation per operation type that handles all widths.

use std::sync::Arc;

use crate::symbolic::{BVOp, FloatOpKind, FloatPrec, RustBV, SymContext};

use super::ir::{FCmpKind, IROp, IRType};
use super::transcendentals;

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
        operands: Arc::<[RustBV]>::from(operands),
    }
}

/// Maximum arity supported by `FloatLaneOp`. Sized for the current set of
/// per-lane FP ops (unary Sqrt/Abs and binary Add/Sub/Mul/Div/Min/Max). Sized
/// to 2 today; bump if a ternary lane op is added (e.g. fused multiply-add).
const FLOAT_LANE_OP_MAX_ARITY: usize = 2;

/// Per-lane FP op contract used by `VEXOps::vec_float_lane_op`.
///
/// Each impl must provide BOTH a concrete fast path (for f32/f64 lanes) and
/// a symbolic Z3 expression builder, so adding a new op cannot accidentally
/// drop one of the two branches — the previous duplicated functions
/// (`vec_float_op`, `vec_float_unop`, `vec_float_minmax`) made the symbolic
/// fallback easy to forget when extending.
trait FloatLaneOp {
    /// Number of operand lanes consumed (1 for unary, 2 for binary).
    fn arity(&self) -> usize;
    /// Apply to a single concrete f32 lane. `args.len() == self.arity()`.
    fn concrete_f32(&self, args: &[f32]) -> f32;
    /// Apply to a single concrete f64 lane. `args.len() == self.arity()`.
    fn concrete_f64(&self, args: &[f64]) -> f64;
    /// Build the symbolic per-lane expression. `args.len() == self.arity()`.
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, ctx: &SymContext) -> RustBV;
}

struct FAdd;
struct FSub;
struct FMul;
struct FDiv;
struct FSqrt;
struct FAbs;
/// FP min: matches Rust `<` semantics (NaN passes through right).
struct FMin;
/// FP max: matches Rust `>` semantics (NaN passes through right).
struct FMax;

impl FloatLaneOp for FAdd {
    fn arity(&self) -> usize { 2 }
    fn concrete_f32(&self, a: &[f32]) -> f32 { a[0] + a[1] }
    fn concrete_f64(&self, a: &[f64]) -> f64 { a[0] + a[1] }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
        build_float_expr(FloatOpKind::Add, prec, args)
    }
}
impl FloatLaneOp for FSub {
    fn arity(&self) -> usize { 2 }
    fn concrete_f32(&self, a: &[f32]) -> f32 { a[0] - a[1] }
    fn concrete_f64(&self, a: &[f64]) -> f64 { a[0] - a[1] }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
        build_float_expr(FloatOpKind::Sub, prec, args)
    }
}
impl FloatLaneOp for FMul {
    fn arity(&self) -> usize { 2 }
    fn concrete_f32(&self, a: &[f32]) -> f32 { a[0] * a[1] }
    fn concrete_f64(&self, a: &[f64]) -> f64 { a[0] * a[1] }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
        build_float_expr(FloatOpKind::Mul, prec, args)
    }
}
impl FloatLaneOp for FDiv {
    fn arity(&self) -> usize { 2 }
    fn concrete_f32(&self, a: &[f32]) -> f32 { a[0] / a[1] }
    fn concrete_f64(&self, a: &[f64]) -> f64 { a[0] / a[1] }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
        build_float_expr(FloatOpKind::Div, prec, args)
    }
}
impl FloatLaneOp for FSqrt {
    fn arity(&self) -> usize { 1 }
    fn concrete_f32(&self, a: &[f32]) -> f32 { a[0].sqrt() }
    fn concrete_f64(&self, a: &[f64]) -> f64 { a[0].sqrt() }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
        build_float_expr(FloatOpKind::Sqrt, prec, args)
    }
}
impl FloatLaneOp for FAbs {
    fn arity(&self) -> usize { 1 }
    fn concrete_f32(&self, a: &[f32]) -> f32 { a[0].abs() }
    fn concrete_f64(&self, a: &[f64]) -> f64 { a[0].abs() }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
        build_float_expr(FloatOpKind::Abs, prec, args)
    }
}
impl FloatLaneOp for FMin {
    fn arity(&self) -> usize { 2 }
    fn concrete_f32(&self, a: &[f32]) -> f32 {
        if a[0] < a[1] { a[0] } else { a[1] }
    }
    fn concrete_f64(&self, a: &[f64]) -> f64 {
        if a[0] < a[1] { a[0] } else { a[1] }
    }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, ctx: &SymContext) -> RustBV {
        // min(l, r) = if l < r then l else r ⟺ ITE(l < r, l, r)
        let mut iter = args.into_iter();
        let l_lane = iter.next().expect("FMin arity 2");
        let r_lane = iter.next().expect("FMin arity 2");
        let cond = build_float_expr(
            FloatOpKind::CmpLt, prec, vec![l_lane.clone(), r_lane.clone()],
        );
        cond.ite_into(l_lane, r_lane, ctx)
    }
}
impl FloatLaneOp for FMax {
    fn arity(&self) -> usize { 2 }
    fn concrete_f32(&self, a: &[f32]) -> f32 {
        if a[0] > a[1] { a[0] } else { a[1] }
    }
    fn concrete_f64(&self, a: &[f64]) -> f64 {
        if a[0] > a[1] { a[0] } else { a[1] }
    }
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, ctx: &SymContext) -> RustBV {
        // max(l, r) = if l > r then l else r ⟺ ITE(r < l, l, r)
        let mut iter = args.into_iter();
        let l_lane = iter.next().expect("FMax arity 2");
        let r_lane = iter.next().expect("FMax arity 2");
        let cond = build_float_expr(
            FloatOpKind::CmpLt, prec, vec![r_lane.clone(), l_lane.clone()],
        );
        cond.ite_into(l_lane, r_lane, ctx)
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

/// Generate concrete float-to-float conversion stubs.
/// Entry shape: `name: src_float_ty, src_uint_ty, dst_float_ty, src_prec, dst_prec;`
macro_rules! define_float_to_float {
    ($($name:ident: $src_ty:ty, $src_uty:ty, $dst_ty:ty, $src_prec:expr, $dst_prec:expr;)*) => {
        $(
            fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
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
            fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::int_to_float(arg, $src_width, $signed, $dst_prec, |v| {
                    ((v as $src_int) as $dst_ty).to_bits() as u128
                })
            }
        )*
    };
}

/// Generate concrete float-to-signed-int conversion stubs (RNE round-ties-even).
/// Entry shape: `name: src_float_ty, src_uint_ty, src_prec, round_fn, dst_signed_int_ty, dst_unsigned_int_ty, dst_width_bits;`
macro_rules! define_float_to_int_signed {
    ($($name:ident: $src_ty:ty, $src_uty:ty, $src_prec:expr, $round:ident, $dst_int:ty, $dst_uint:ty, $dst_width:expr;)*) => {
        $(
            fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::float_to_int(arg, $src_prec, $dst_width, true, |v| {
                    (Self::$round(<$src_ty>::from_bits(v as $src_uty)) as $dst_int as $dst_uint) as u128
                })
            }
        )*
    };
}

/// Generate concrete float-to-unsigned-int conversion stubs (RNE round-ties-even).
/// Entry shape: `name: src_float_ty, src_uint_ty, src_prec, round_fn, dst_unsigned_int_ty, dst_width_bits;`
macro_rules! define_float_to_int_unsigned {
    ($($name:ident: $src_ty:ty, $src_uty:ty, $src_prec:expr, $round:ident, $dst_uint:ty, $dst_width:expr;)*) => {
        $(
            fn $name(arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
                Self::float_to_int(arg, $src_prec, $dst_width, false, |v| {
                    Self::$round(<$src_ty>::from_bits(v as $src_uty)) as $dst_uint as u128
                })
            }
        )*
    };
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

            // Packed integer absolute value
            IROp::VAbs { elem, count } => Self::vec_int_abs(arg, elem, count, ctx),

            // Packed float sqrt / abs (whole vector)
            IROp::VFSqrt { elem, count } => Self::vec_float_lane_op(&[arg], elem, count, &FSqrt, ctx),
            IROp::VFAbs { elem, count } => Self::vec_float_lane_op(&[arg], elem, count, &FAbs, ctx),

            // NEON scaffolding: fail loudly rather than silently fall back to
            // a fresh-symbolic result. Implementations land in angr-bkcs.2.
            IROp::NeonUnimplemented(name) => panic!("NEON op {} not yet implemented", name),

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

            IROp::FCmpScalarLane { kind, ty } => {
                Self::vec_float_scalar_lane_cmp(left, right, kind, ty, ctx)
            }
            IROp::FCmpVecPacked { kind, elem, count } => {
                Self::vec_float_packed_cmp(left, right, kind, elem, count, ctx)
            }
            IROp::FComCC(ty) => Self::float_com_cc(left, right, ty, ctx),

            // Float rounding with mode (left = rounding mode, right = value)
            IROp::RoundF32toInt => Self::round_f32_to_int_with_mode(left, right, ctx),
            IROp::RoundF64toInt => Self::round_f64_to_int_with_mode(left, right, ctx),

            // Scalar-in-vector float operations (SSE scalar ops)
            IROp::VFAddS { elem } => Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Add, ctx),
            IROp::VFSubS { elem } => Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Sub, ctx),
            IROp::VFMulS { elem } => Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Mul, ctx),
            IROp::VFDivS { elem } => Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Div, ctx),

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

            // NEON lane extract (Iop_GetElem{N}x{M}): (vec, idx) -> lane.
            IROp::VGetElem { elem, count } => Self::vec_get_elem(left, right, elem, count, ctx),

            // Vector interleave
            IROp::VInterleaveLO { elem } => Self::vec_interleave_lo(left, right, elem, ctx),
            IROp::VInterleaveHI { elem } => Self::vec_interleave_hi(left, right, elem, ctx),

            // Vector shifts by immediate
            IROp::VShlN { elem, count } => Self::vec_shl_n(left, right, elem, count, ctx),
            IROp::VShrN { elem, count } => Self::vec_shr_n(left, right, elem, count, ctx),
            IROp::VSarN { elem, count } => Self::vec_sar_n(left, right, elem, count, ctx),

            // Packed integer min/max
            IROp::VMin { elem, count, signed } => {
                Self::vec_int_minmax(left, right, elem, count, signed, /*is_max=*/ false, ctx)
            }
            IROp::VMax { elem, count, signed } => {
                Self::vec_int_minmax(left, right, elem, count, signed, /*is_max=*/ true, ctx)
            }

            // Packed FP arithmetic
            IROp::VFAdd { elem, count } => Self::vec_float_lane_op(&[left, right], elem, count, &FAdd, ctx),
            IROp::VFSub { elem, count } => Self::vec_float_lane_op(&[left, right], elem, count, &FSub, ctx),
            IROp::VFMul { elem, count } => Self::vec_float_lane_op(&[left, right], elem, count, &FMul, ctx),
            IROp::VFDiv { elem, count } => Self::vec_float_lane_op(&[left, right], elem, count, &FDiv, ctx),

            // Packed FP min/max
            IROp::VFMin { elem, count } => Self::vec_float_lane_op(&[left, right], elem, count, &FMin, ctx),
            IROp::VFMax { elem, count } => Self::vec_float_lane_op(&[left, right], elem, count, &FMax, ctx),

            // Raw opcode — try concrete x87 transcendental fast path first
            // (Iop_SinF64, Iop_CosF64, Iop_TanF64, Iop_2xm1F64, Iop_RecpExp*).
            // These arrive as Binop(rm, x); `left` carries rm, `right` the value.
            // Symbolic falls through to the existing fresh-symbolic fallback.
            IROp::Raw(code) => {
                if let Some(result) = transcendentals::try_concrete_binop_rm(code, &left, &right) {
                    Ok(result)
                } else {
                    Err(OpError::RawOpcode(code))
                }
            }

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

            // Iop_SqrtF{32,64} is a VEX Binop (arg1=rm, arg2=value) that we
            // model as IROp::FSqrt — keep the translation here so the
            // interpreter's Binop dispatch routes the rm through
            // unop_with_rm instead of falling back to a fresh symbolic.
            IROp::FSqrt(_) => Self::unop_with_rm(op, left, right, ctx),

            // NEON scaffolding: fail loudly rather than silently fall back.
            IROp::NeonUnimplemented(name) => panic!("NEON op {} not yet implemented", name),

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
            // NEON scaffolding: fail loudly rather than silently fall back.
            IROp::NeonUnimplemented(name) => panic!("NEON op {} not yet implemented", name),
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
            // NEON scaffolding: fail loudly rather than silently fall back.
            IROp::NeonUnimplemented(name) => panic!("NEON op {} not yet implemented", name),
            _ => Err(OpError::NotQuaternary(op)),
        }
    }

    // =========================================================================
    // Rounding-mode aware float arithmetic
    // =========================================================================

    /// Binary FP arithmetic with explicit VEX rounding mode. Used by the
    /// Triop dispatch to honor the rm operand on `Iop_AddF{32,64}`,
    /// `Iop_SubF*`, `Iop_MulF*`, `Iop_DivF*`. Concrete RNE (rm low-2-bits == 0)
    /// keeps the existing native-f32/f64 fast path; non-RNE concrete or
    /// symbolic rm builds a Z3 FP expression with the explicit rm so the
    /// rm-aware fold in `build_fp_arith_rm_cached` produces the right value.
    #[inline]
    pub fn binop_with_rm(
        op: IROp,
        rm: RustBV,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // Iop_SetElem* is a VEX Triop but does NOT carry a rounding mode.
        // The Triop dispatch in expressions.rs passes (rm, left, right) as
        // the raw three operands (vec, idx, val); reinterpret accordingly.
        if let IROp::VSetElem { elem, count } = op {
            return Self::vec_set_elem(rm, left, right, elem, count, ctx);
        }

        let (kind_rm, ty) = match op {
            IROp::FAdd(t) => (FloatOpKind::AddRm, t),
            IROp::FSub(t) => (FloatOpKind::SubRm, t),
            IROp::FMul(t) => (FloatOpKind::MulRm, t),
            IROp::FDiv(t) => (FloatOpKind::DivRm, t),
            IROp::Raw(code) => {
                // x87 Triop transcendentals: Iop_AtanF64, Iop_Yl2xF64,
                // Iop_Yl2xp1F64, Iop_ScaleF64. Concrete-only fast path;
                // symbolic falls through to the existing fresh-symbolic
                // fallback in expressions.rs::IRExpr::Triop.
                if let Some(result) = transcendentals::try_concrete_triop_rm(code, &rm, &left, &right) {
                    return Ok(result);
                }
                return Self::binop(op, left, right, ctx);
            }
            _ => return Self::binop(op, left, right, ctx),
        };

        // RNE concrete: keep the native-f{32,64} fast path. Most code uses
        // RNE; routing through Z3 here would be a measurable regression on
        // FP-heavy benchmarks.
        if let Some(m) = rm.as_u128() {
            if m & 0x3 == 0 {
                return Self::binop(op, left, right, ctx);
            }
        }

        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(kind_rm, prec, vec![rm, left, right]))
    }

    /// Unary FP op (Sqrt) with explicit VEX rounding mode. RNE concrete keeps
    /// the existing native-f{32,64} fast path; non-RNE/symbolic rm routes
    /// through `FloatOpKind::SqrtRm`.
    #[inline]
    pub fn unop_with_rm(
        op: IROp,
        rm: RustBV,
        arg: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let (kind_rm, ty) = match op {
            IROp::FSqrt(t) => (FloatOpKind::SqrtRm, t),
            _ => return Self::unop(op, arg, ctx),
        };

        if let Some(m) = rm.as_u128() {
            if m & 0x3 == 0 {
                return Self::unop(op, arg, ctx);
            }
        }

        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(kind_rm, prec, vec![rm, arg]))
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
        Self::divmod_double_to_single(dividend, divisor, signed, ctx)
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
        Self::divmod_double_to_single(dividend, divisor, signed, ctx)
    }

    /// Generic DivMod: dividend (width = `2 * divisor.width()`) divided by
    /// divisor; returns a value of the dividend's width packed as
    /// `low half = quotient, high half = remainder`. Both halves are the
    /// divisor's width.
    ///
    /// Z3 defines div/mod by zero totally (udiv→all-ones, urem→dividend,
    /// sdiv→±1, srem→dividend), matching claripy, so the symbolic path
    /// needs no explicit zero guard. The concrete path returns 0 on a
    /// zero divisor — callers are expected to have checked.
    fn divmod_double_to_single(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let dividend_w = dividend.width();
        let divisor_w = divisor.width();
        debug_assert_eq!(dividend_w, divisor_w * 2);

        if let (Some(dvd), Some(dvs)) = (dividend.as_u128(), divisor.as_u128()) {
            let dvd = dvd & Self::low_bit_mask_u128(dividend_w);
            let dvs = dvs & Self::low_bit_mask_u128(divisor_w);

            if dvs == 0 {
                return Ok(RustBV::concrete(0, dividend_w));
            }

            let half_mask = Self::low_bit_mask_u128(divisor_w);
            let (quotient, remainder) = if signed {
                let dvd_i = Self::sign_extend_low_to_i128(dvd, dividend_w);
                let dvs_i = Self::sign_extend_low_to_i128(dvs, divisor_w);
                let q = (dvd_i / dvs_i) as u128 & half_mask;
                let r = (dvd_i % dvs_i) as u128 & half_mask;
                (q, r)
            } else {
                (dvd / dvs, dvd % dvs)
            };

            let result = quotient | (remainder << divisor_w);
            return Ok(RustBV::concrete(result, dividend_w));
        }

        let divisor_full = if signed {
            divisor.sign_extend_into(dividend_w, ctx)
        } else {
            divisor.zero_extend_into(dividend_w, ctx)
        };
        let quotient_full = if signed {
            dividend.sdiv(&divisor_full, ctx)
        } else {
            dividend.udiv(&divisor_full, ctx)
        };
        let remainder_full = if signed {
            dividend.srem(&divisor_full, ctx)
        } else {
            dividend.urem(&divisor_full, ctx)
        };
        let quotient_half = quotient_full.extract_into(divisor_w - 1, 0, ctx);
        let remainder_half = remainder_full.extract_into(divisor_w - 1, 0, ctx);
        Ok(remainder_half.concat_into(quotient_half, ctx))
    }

    /// Mask covering the low `width` bits of a u128.
    #[inline]
    fn low_bit_mask_u128(width: u32) -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
    }

    /// Sign-extend the low `width` bits of `value` to a 128-bit signed value.
    #[inline]
    fn sign_extend_low_to_i128(value: u128, width: u32) -> i128 {
        if width == 0 || width >= 128 {
            return value as i128;
        }
        let shift = 128 - width;
        ((value as i128) << shift) >> shift
    }

    /// Sign-extend the low `width` bits of `value` to 64 bits, returning the
    /// resulting bit pattern as `u64`. Used by concrete-fast-path vector ops
    /// that perform signed multiplication via `wrapping_mul` at u64 — only the
    /// low `2 * width` bits of the product are kept by the caller, so any
    /// higher-bit representation works.
    #[inline]
    fn sign_extend_low_to_u64(value: u128, width: u32) -> u64 {
        if width == 0 || width >= 64 {
            return value as u64;
        }
        let mask = (1u64 << width) - 1;
        let masked = (value as u64) & mask;
        let sign_bit = 1u64 << (width - 1);
        if masked & sign_bit != 0 {
            masked | !mask
        } else {
            masked
        }
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

    /// NEON lane extract (Iop_GetElem{N}x{M}).
    ///
    /// `vec` is the source vector (width = elem.bits() * count), `idx` is
    /// Ity_I8 (the lane index — 0 = lowest lane). Result is one lane.
    ///
    /// For concrete `idx < count`, this is a simple bit-slice. Concrete
    /// out-of-range indices saturate at the highest valid lane (matches
    /// pyvex's behavior of treating `idx % count` as the effective lane).
    /// Symbolic `idx` builds an ITE chain over all `count` lanes.
    fn vec_get_elem(
        vec: RustBV,
        idx: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(idx.width(), 8);

        // Concrete fast path
        if let (Some(v), Some(i)) = (vec.as_u128(), idx.as_u128()) {
            let lane = (i as u8) % count;
            let lo = lane as u32 * elem_width;
            let mask = if elem_width == 128 { u128::MAX } else { (1u128 << elem_width) - 1 };
            let result = (v >> lo) & mask;
            return Ok(RustBV::concrete(result, elem_width));
        }

        // Concrete idx, symbolic vec: bit-slice
        if let Some(i) = idx.as_u128() {
            let lane = (i as u8) % count;
            let lo = lane as u32 * elem_width;
            let hi = lo + elem_width - 1;
            return Ok(vec.extract(hi, lo, ctx));
        }

        // Symbolic idx: ITE chain over all lanes. count is small (<= 16).
        let mut result = vec.extract(elem_width - 1, 0, ctx);
        for lane in 1..count {
            let lo = lane as u32 * elem_width;
            let hi = lo + elem_width - 1;
            let lane_val = vec.extract(hi, lo, ctx);
            let lane_idx = RustBV::concrete(lane as u128, 8);
            let cond = idx.eq(&lane_idx, ctx);
            result = cond.ite(&lane_val, &result, ctx);
        }
        Ok(result)
    }

    /// NEON lane insert (Iop_SetElem{N}x{M}).
    ///
    /// `vec` is the source vector, `idx` (Ity_I8) selects the lane, `val`
    /// (width = elem.bits()) is the replacement. Returns the modified
    /// vector. Concrete out-of-range indices wrap (matches pyvex).
    /// Symbolic `idx` builds an ITE chain.
    fn vec_set_elem(
        vec: RustBV,
        idx: RustBV,
        val: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(idx.width(), 8);
        debug_assert_eq!(val.width(), elem_width);

        // Concrete vec + val + idx: bit-twiddle.
        if let (Some(v), Some(i), Some(x)) = (vec.as_u128(), idx.as_u128(), val.as_u128()) {
            let lane = (i as u8) % count;
            let lo = lane as u32 * elem_width;
            let mask_elem = if elem_width == 128 { u128::MAX } else { (1u128 << elem_width) - 1 };
            let shifted_mask = mask_elem << lo;
            let cleared = v & !shifted_mask;
            let new_val = cleared | ((x & mask_elem) << lo);
            return Ok(RustBV::concrete(new_val, total_width));
        }

        // Concrete idx, symbolic vec/val: rebuild from element slices.
        if let Some(i) = idx.as_u128() {
            let lane = (i as u8) % count;
            let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
            for slot in 0..count {
                if slot == lane {
                    elements.push(val.clone());
                } else {
                    let lo = slot as u32 * elem_width;
                    let hi = lo + elem_width - 1;
                    elements.push(vec.extract(hi, lo, ctx));
                }
            }
            return Ok(Self::concat_le_elements(elements, ctx));
        }

        // Symbolic idx: ITE chain — for each lane, select val if idx==lane
        // else the original lane bits, then rebuild the vector.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for slot in 0..count {
            let lo = slot as u32 * elem_width;
            let hi = lo + elem_width - 1;
            let orig_lane = vec.extract(hi, lo, ctx);
            let slot_idx = RustBV::concrete(slot as u128, 8);
            let cond = idx.eq(&slot_idx, ctx);
            let chosen = cond.ite(&val, &orig_lane, ctx);
            elements.push(chosen);
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

                // Sign-extend each lane to 64 bits, then multiply at u64 and
                // mask to the lane width — high bits beyond `2 * elem_width`
                // are discarded by the mask, so any 64-bit representation
                // matching the lane's signed value in the low bits suffices.
                let l_signed = Self::sign_extend_low_to_u64(l_elem, elem_width);
                let r_signed = Self::sign_extend_low_to_u64(r_elem, elem_width);

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
    /// Only Add/Sub/Mul/Div are accepted; other kinds are a programmer error.
    #[inline]
    fn vec_float_scalar_op(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        kind: FloatOpKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);
        debug_assert!(
            matches!(kind, FloatOpKind::Add | FloatOpKind::Sub | FloatOpKind::Mul | FloatOpKind::Div),
            "vec_float_scalar_op only supports Add/Sub/Mul/Div, got {:?}",
            kind,
        );

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);
                    let res0 = match kind {
                        FloatOpKind::Add => l0 + r0,
                        FloatOpKind::Sub => l0 - r0,
                        FloatOpKind::Mul => l0 * r0,
                        FloatOpKind::Div => l0 / r0,
                        _ => unreachable!(),
                    };
                    let upper = l & !0xFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                IRType::F64 => {
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);
                    let res0 = match kind {
                        FloatOpKind::Add => l0 + r0,
                        FloatOpKind::Sub => l0 - r0,
                        FloatOpKind::Mul => l0 * r0,
                        FloatOpKind::Div => l0 / r0,
                        _ => unreachable!(),
                    };
                    let upper = l & !0xFFFFFFFFFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
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

    // =========================================================================
    // Packed integer min/max/abs
    // =========================================================================

    /// Packed integer per-lane min or max. Handles signed/unsigned via the
    /// `signed` flag and min/max via `is_max`.
    fn vec_int_minmax(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        is_max: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // Concrete fast path (only when total fits in u128).
        if total_width <= 128 {
            if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
                let mut result: u128 = 0;
                let elem_mask: u128 = if elem_width == 128 { u128::MAX } else { (1u128 << elem_width) - 1 };
                let sign_bit: u128 = 1u128 << (elem_width - 1);

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let l_elem = (l >> lo) & elem_mask;
                    let r_elem = (r >> lo) & elem_mask;

                    let pick_left = if signed {
                        // Sign-extend each lane to i128 for comparison.
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
                        if is_max { l_signed >= r_signed } else { l_signed <= r_signed }
                    } else {
                        if is_max { l_elem >= r_elem } else { l_elem <= r_elem }
                    };

                    let chosen = if pick_left { l_elem } else { r_elem };
                    result |= (chosen & elem_mask) << lo;
                }
                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic per-lane fallback.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            // For max: ITE(l >= r, l, r); for min: ITE(l <= r, l, r).
            let cond = match (signed, is_max) {
                (true, true) => l_elem.clone().sge_into(r_elem.clone(), ctx),
                (true, false) => l_elem.clone().sle_into(r_elem.clone(), ctx),
                (false, true) => l_elem.clone().uge_into(r_elem.clone(), ctx),
                (false, false) => l_elem.clone().ule_into(r_elem.clone(), ctx),
            };
            elements.push(cond.ite_into(l_elem, r_elem, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Packed integer per-lane absolute value. Returns the unsigned
    /// representation of `|signed_lane|`. INT_MIN stays INT_MIN (matches the
    /// PABS* hardware behavior).
    fn vec_int_abs(
        arg: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total_width);

        // Concrete fast path.
        if total_width <= 128 {
            if let Some(v) = arg.as_u128() {
                let mut result: u128 = 0;
                let elem_mask: u128 = if elem_width == 128 { u128::MAX } else { (1u128 << elem_width) - 1 };
                let sign_bit: u128 = 1u128 << (elem_width - 1);

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;
                    // |x| = (x ^ -1) + 1 when x is negative (two's complement),
                    // otherwise x. Simulated within elem_width bits.
                    let abs_val = if elem_val & sign_bit != 0 {
                        // -x in elem_width bits = (~x + 1) & mask
                        ((!elem_val).wrapping_add(1)) & elem_mask
                    } else {
                        elem_val
                    };
                    result |= abs_val << lo;
                }
                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic per-lane fallback: ITE(elem < 0, -elem, elem).
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let zero = RustBV::concrete(0, elem_width);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = arg.extract(hi, lo, ctx);
            let neg = elem_val.clone().neg_into(ctx);
            let is_neg = elem_val.clone().slt_into(zero.clone(), ctx);
            elements.push(is_neg.ite_into(neg, elem_val, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    // =========================================================================
    // Packed FP arithmetic / unary / min-max
    // =========================================================================

    /// Generic per-lane FP dispatcher. Handles the lane loop for both the
    /// concrete (extract via shift+mask, run f32/f64 op, repack) and the
    /// symbolic (extract via .extract(hi,lo), build per-lane Z3 expression,
    /// concat) paths. The trait `FloatLaneOp` provides the per-op specifics,
    /// forcing each impl to define both branches in lockstep.
    fn vec_float_lane_op(
        args: &[RustBV],
        elem: IRType,
        count: u8,
        op: &dyn FloatLaneOp,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(args.len(), op.arity());
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        for a in args {
            debug_assert_eq!(a.width(), total_width);
        }

        // Concrete fast path: every operand must fit in u128.
        if total_width <= 128 {
            let concrete: Option<Vec<u128>> = args.iter().map(|a| a.as_u128()).collect();
            if let Some(concrete) = concrete {
                if !matches!(elem, IRType::F32 | IRType::F64) {
                    return Err(OpError::InvalidFloatType(elem));
                }
                let arity = concrete.len();
                let mut result: u128 = 0;
                let elem_mask: u128 = (1u128 << elem_width) - 1;
                let mut buf32 = [0f32; FLOAT_LANE_OP_MAX_ARITY];
                let mut buf64 = [0f64; FLOAT_LANE_OP_MAX_ARITY];
                debug_assert!(arity <= FLOAT_LANE_OP_MAX_ARITY);
                for i in 0..count {
                    let shift = (i as u32) * elem_width;
                    let lane_bits = match elem {
                        IRType::F32 => {
                            for (idx, raw) in concrete.iter().enumerate() {
                                buf32[idx] = f32::from_bits(((raw >> shift) & elem_mask) as u32);
                            }
                            op.concrete_f32(&buf32[..arity]).to_bits() as u128
                        }
                        IRType::F64 => {
                            for (idx, raw) in concrete.iter().enumerate() {
                                buf64[idx] = f64::from_bits(((raw >> shift) & elem_mask) as u64);
                            }
                            op.concrete_f64(&buf64[..arity]).to_bits() as u128
                        }
                        _ => unreachable!(),
                    };
                    result |= lane_bits << shift;
                }
                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic per-lane fallback.
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let lane_args: Vec<RustBV> = args.iter().map(|a| a.extract(hi, lo, ctx)).collect();
            elements.push(op.symbolic(lane_args, prec, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
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

    /// SSE scalar-lane FP compare (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}).
    /// Operates on lane 0 only; result is V128 with lane 0 set to all-1s on
    /// true and 0 on false. Upper lanes pass through from `left`.
    fn vec_float_scalar_lane_cmp(
        left: RustBV,
        right: RustBV,
        kind: FCmpKind,
        ty: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        let lane_bits = prec.bits();
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);
        let l_lo = left.extract(lane_bits - 1, 0, ctx);
        let r_lo = right.extract(lane_bits - 1, 0, ctx);
        let upper = left.extract(127, lane_bits, ctx);

        let lane_mask: u128 = if lane_bits == 32 {
            0xFFFF_FFFF
        } else {
            0xFFFF_FFFF_FFFF_FFFF
        };

        // Concrete fast path
        if let (Some(l), Some(r)) = (l_lo.as_u128(), r_lo.as_u128()) {
            let truth = match (ty, kind) {
                (IRType::F32, FCmpKind::Eq) => f32::from_bits(l as u32) == f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Lt) => f32::from_bits(l as u32) <  f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Le) => f32::from_bits(l as u32) <= f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Gt) => f32::from_bits(l as u32) >  f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Ge) => f32::from_bits(l as u32) >= f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Un) => {
                    f32::from_bits(l as u32).is_nan() || f32::from_bits(r as u32).is_nan()
                }
                (IRType::F64, FCmpKind::Eq) => f64::from_bits(l as u64) == f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Lt) => f64::from_bits(l as u64) <  f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Le) => f64::from_bits(l as u64) <= f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Gt) => f64::from_bits(l as u64) >  f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Ge) => f64::from_bits(l as u64) >= f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Un) => {
                    f64::from_bits(l as u64).is_nan() || f64::from_bits(r as u64).is_nan()
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            let lane_val = if truth { lane_mask } else { 0 };
            let lane = RustBV::concrete(lane_val, lane_bits);
            return Ok(upper.concat_into(lane, ctx));
        }

        // Symbolic path: build 1-bit compare, then sign-extend to lane_bits
        // (sign-extend turns 1 → all-1s, 0 → all-0s).
        let cmp_1bit = match kind {
            FCmpKind::Eq => build_float_expr(FloatOpKind::CmpEq, prec, vec![l_lo, r_lo]),
            FCmpKind::Lt => build_float_expr(FloatOpKind::CmpLt, prec, vec![l_lo, r_lo]),
            FCmpKind::Le => build_float_expr(FloatOpKind::CmpLe, prec, vec![l_lo, r_lo]),
            // Gt(a,b) = Lt(b,a); Ge(a,b) = Le(b,a). Not emitted by SSE
            // scalar-lane opcodes today, but exhaustive for the shared FCmpKind enum.
            FCmpKind::Gt => build_float_expr(FloatOpKind::CmpLt, prec, vec![r_lo, l_lo]),
            FCmpKind::Ge => build_float_expr(FloatOpKind::CmpLe, prec, vec![r_lo, l_lo]),
            FCmpKind::Un => {
                // IEEE 754: NaN != NaN. So `(x == x)` is false iff x is NaN.
                // un = NOT(l_eq_l) OR NOT(r_eq_r)
                let l_eq_l = build_float_expr(
                    FloatOpKind::CmpEq, prec, vec![l_lo.clone(), l_lo],
                );
                let r_eq_r = build_float_expr(
                    FloatOpKind::CmpEq, prec, vec![r_lo.clone(), r_lo],
                );
                l_eq_l.not_into(ctx).or_into(r_eq_r.not_into(ctx), ctx)
            }
        };
        let lane = cmp_1bit.sign_extend_into(lane_bits, ctx);
        Ok(upper.concat_into(lane, ctx))
    }

    /// Packed FP compare (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}).
    /// Each lane independently produces all-1s (true) or 0 (false) of width
    /// `elem.bits()`. Total result width = elem.bits() * count.
    fn vec_float_packed_cmp(
        left: RustBV,
        right: RustBV,
        kind: FCmpKind,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        let lane_mask: u128 = if elem_width == 32 {
            0xFFFF_FFFF
        } else if elem_width == 64 {
            0xFFFF_FFFF_FFFF_FFFF
        } else {
            return Err(OpError::InvalidFloatType(elem));
        };

        // Concrete fast path: extract each lane, compare, repack. The concrete
        // truth-table for Gt/Ge follows the IEEE 754 ordered semantics — Rust's
        // `>` and `>=` on f32/f64 already return false when either operand is
        // NaN, matching VEX. The Un case independently checks NaN.
        if total_width <= 128 {
            if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
                let mut result: u128 = 0;
                let elem_mask: u128 = (1u128 << elem_width) - 1;
                for i in 0..count {
                    let shift = (i as u32) * elem_width;
                    let l_bits = (l >> shift) & elem_mask;
                    let r_bits = (r >> shift) & elem_mask;
                    let truth = match (elem, kind) {
                        (IRType::F32, FCmpKind::Eq) => f32::from_bits(l_bits as u32) == f32::from_bits(r_bits as u32),
                        (IRType::F32, FCmpKind::Lt) => f32::from_bits(l_bits as u32) <  f32::from_bits(r_bits as u32),
                        (IRType::F32, FCmpKind::Le) => f32::from_bits(l_bits as u32) <= f32::from_bits(r_bits as u32),
                        (IRType::F32, FCmpKind::Gt) => f32::from_bits(l_bits as u32) >  f32::from_bits(r_bits as u32),
                        (IRType::F32, FCmpKind::Ge) => f32::from_bits(l_bits as u32) >= f32::from_bits(r_bits as u32),
                        (IRType::F32, FCmpKind::Un) => {
                            f32::from_bits(l_bits as u32).is_nan() || f32::from_bits(r_bits as u32).is_nan()
                        }
                        (IRType::F64, FCmpKind::Eq) => f64::from_bits(l_bits as u64) == f64::from_bits(r_bits as u64),
                        (IRType::F64, FCmpKind::Lt) => f64::from_bits(l_bits as u64) <  f64::from_bits(r_bits as u64),
                        (IRType::F64, FCmpKind::Le) => f64::from_bits(l_bits as u64) <= f64::from_bits(r_bits as u64),
                        (IRType::F64, FCmpKind::Gt) => f64::from_bits(l_bits as u64) >  f64::from_bits(r_bits as u64),
                        (IRType::F64, FCmpKind::Ge) => f64::from_bits(l_bits as u64) >= f64::from_bits(r_bits as u64),
                        (IRType::F64, FCmpKind::Un) => {
                            f64::from_bits(l_bits as u64).is_nan() || f64::from_bits(r_bits as u64).is_nan()
                        }
                        _ => return Err(OpError::InvalidFloatType(elem)),
                    };
                    let lane_val = if truth { lane_mask } else { 0 };
                    result |= lane_val << shift;
                }
                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic per-lane fallback: build a 1-bit predicate per lane,
        // sign-extend to elem width, concatenate (low-order lane first).
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let l_lane = left.extract(hi, lo, ctx);
            let r_lane = right.extract(hi, lo, ctx);
            let cmp_1bit = match kind {
                FCmpKind::Eq => build_float_expr(FloatOpKind::CmpEq, prec, vec![l_lane, r_lane]),
                FCmpKind::Lt => build_float_expr(FloatOpKind::CmpLt, prec, vec![l_lane, r_lane]),
                FCmpKind::Le => build_float_expr(FloatOpKind::CmpLe, prec, vec![l_lane, r_lane]),
                // Gt(a,b) ≡ Lt(b,a); Ge(a,b) ≡ Le(b,a). Z3 has CmpLt/CmpLe;
                // swapping operands is cheaper than introducing new variants.
                FCmpKind::Gt => build_float_expr(FloatOpKind::CmpLt, prec, vec![r_lane, l_lane]),
                FCmpKind::Ge => build_float_expr(FloatOpKind::CmpLe, prec, vec![r_lane, l_lane]),
                FCmpKind::Un => {
                    // IEEE 754: NaN != NaN. So `(x == x)` is false iff x is NaN.
                    let l_eq_l = build_float_expr(
                        FloatOpKind::CmpEq, prec, vec![l_lane.clone(), l_lane],
                    );
                    let r_eq_r = build_float_expr(
                        FloatOpKind::CmpEq, prec, vec![r_lane.clone(), r_lane],
                    );
                    l_eq_l.not_into(ctx).or_into(r_eq_r.not_into(ctx), ctx)
                }
            };
            elements.push(cmp_1bit.sign_extend_into(elem_width, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// x87 FCOM-style compare (Iop_CmpF32, Iop_CmpF64). Returns I32 with the
    /// VEX-defined encoding:
    ///   0x40 = EQ, 0x01 = LT, 0x00 = GT, 0x45 = UN (either operand is NaN).
    fn float_com_cc(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // Concrete fast path
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result: u128 = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf.is_nan() || rf.is_nan() { 0x45 }
                    else if lf <  rf { 0x01 }
                    else if lf == rf { 0x40 }
                    else { 0x00 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf.is_nan() || rf.is_nan() { 0x45 }
                    else if lf <  rf { 0x01 }
                    else if lf == rf { 0x40 }
                    else { 0x00 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 32));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;

        // Symbolic: compose un/lt/eq predicates, then nest ITEs.
        // un  = NOT(l == l) OR NOT(r == r)   [IEEE 754 NaN check]
        // lt  = l < r                         [false if either is NaN]
        // eq  = l == r                        [false if either is NaN]
        let l_eq_l = build_float_expr(
            FloatOpKind::CmpEq, prec, vec![left.clone(), left.clone()],
        );
        let r_eq_r = build_float_expr(
            FloatOpKind::CmpEq, prec, vec![right.clone(), right.clone()],
        );
        let un = l_eq_l.not_into(ctx).or_into(r_eq_r.not_into(ctx), ctx);
        let lt = build_float_expr(
            FloatOpKind::CmpLt, prec, vec![left.clone(), right.clone()],
        );
        let eq = build_float_expr(FloatOpKind::CmpEq, prec, vec![left, right]);

        let v_un = RustBV::concrete(0x45, 32);
        let v_lt = RustBV::concrete(0x01, 32);
        let v_eq = RustBV::concrete(0x40, 32);
        let v_gt = RustBV::concrete(0x00, 32);
        // un ? 0x45 : (lt ? 0x01 : (eq ? 0x40 : 0x00))
        let inner = eq.ite_into(v_eq, v_gt, ctx);
        let middle = lt.ite_into(v_lt, inner, ctx);
        Ok(un.ite_into(v_un, middle, ctx))
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
        f32_to_i32s: f32, u32, FloatPrec::F32, round_ties_to_even_f32, i32, u32, 32;
        f64_to_i32s: f64, u64, FloatPrec::F64, round_ties_to_even_f64, i32, u32, 32;
        f32_to_i64s: f32, u32, FloatPrec::F32, round_ties_to_even_f32, i64, u64, 64;
        f64_to_i64s: f64, u64, FloatPrec::F64, round_ties_to_even_f64, i64, u64, 64;
    }
    define_float_to_int_unsigned! {
        f32_to_i32u: f32, u32, FloatPrec::F32, round_ties_to_even_f32, u32, 32;
        f64_to_i32u: f64, u64, FloatPrec::F64, round_ties_to_even_f64, u32, 32;
        f32_to_i64u: f32, u32, FloatPrec::F32, round_ties_to_even_f32, u64, 64;
        f64_to_i64u: f64, u64, FloatPrec::F64, round_ties_to_even_f64, u64, 64;
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

    /// FDiv with concrete RNE rm goes through the native-f32 fast path.
    /// 1.0/10.0 under RNE is 0x3DCCCCCD (correctly rounded up).
    #[test]
    fn test_float_div_with_rm_rne_fastpath_f32() {
        let ctx = SymContext::new_mock();
        let rm_rne = RustBV::concrete(0, 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rne, one, ten, &ctx).unwrap();
        // RNE keeps the native-f32 fast path; result is fully concrete (not
        // a Z3 expression).
        assert!(!result.is_symbolic());
        assert_eq!(result.as_u64(), Some(0x3DCCCCCD));
    }

    /// FDiv with concrete RZ (toward zero) rm: 1.0/10.0 truncates the last
    /// mantissa bit → 0x3DCCCCCC (one ULP below RNE).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_div_with_rm_rz_f32() {
        let ctx = SymContext::new_mock();
        let rm_rz = RustBV::concrete(3, 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rz, one, ten, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3DCCCCCC, "1/10 with RZ rounds toward zero");
    }

    /// FDiv with concrete RU (toward +inf) rm: 1.0/10.0 = 0x3DCCCCCD (same
    /// as RNE because the discarded bits push up).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_div_with_rm_ru_f32() {
        let ctx = SymContext::new_mock();
        let rm_ru = RustBV::concrete(2, 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_ru, one, ten, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3DCCCCCD);
    }

    /// FDiv with concrete RD (toward -inf) rm on a positive result equals RZ:
    /// 1.0/10.0 → 0x3DCCCCCC.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_div_with_rm_rd_f32() {
        let ctx = SymContext::new_mock();
        let rm_rd = RustBV::concrete(1, 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rd, one, ten, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3DCCCCCC);
    }

    /// FDiv with symbolic rm: constraining result == 0x3DCCCCCC forces rm
    /// low-2-bits ∈ {1, 3} (RD or RZ); 0x3DCCCCCD forces rm low-2-bits ∈
    /// {0, 2}. Exercises the 4-way ITE built by `build_fp_arith_rm_cached`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_div_with_symbolic_rm_f32() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();
        let rm = RustBV::symbolic(&ctx, "rm_div_f32", 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

        let result = VEXOps::binop_with_rm(
            IROp::FDiv(IRType::F32),
            rm.clone(),
            one,
            ten,
            &ctx,
        )
        .unwrap();
        let target = RustBV::concrete(0x3DCCCCCC, 32);
        let eq = result.to_z3_ast()._eq(&target.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT for div(1,10) == 0x3DCCCCCC");

        let model_rm = ctx.eval(&rm).expect("eval(rm) returned None") as u32;
        let low2 = model_rm & 0x3;
        assert!(
            low2 == 1 || low2 == 3,
            "expected rm low2 ∈ {{1, 3}} (RD/RZ), got {}",
            low2
        );
    }

    /// FAdd with concrete RZ on inexact-sum operands. 0x3F800001 + 0x3F800002
    /// = 2.0 + 1.5ulp (exact). RNE rounds-to-even → 2.0 + 2ulp = 0x40000002;
    /// RZ truncates → 2.0 + 1ulp = 0x40000001.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_add_with_rm_rz_f32() {
        let ctx = SymContext::new_mock();
        let rm_rz = RustBV::concrete(3, 32);
        let a = RustBV::concrete(0x3F800001, 32);
        let b = RustBV::concrete(0x3F800002, 32);

        let result =
            VEXOps::binop_with_rm(IROp::FAdd(IRType::F32), rm_rz, a, b, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x40000001, "RZ truncates 1.5ulp tie down");
    }

    /// SqrtRm: sqrt(2.0f32) under RU (toward +inf). True value is between
    /// 0x3FB504F3 (RNE) and 0x3FB504F4; RU pushes up by one ulp.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_sqrt_with_rm_ru_f32() {
        let ctx = SymContext::new_mock();
        let rm_ru = RustBV::concrete(2, 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::unop_with_rm(IROp::FSqrt(IRType::F32), rm_ru, two, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3FB504F4, "sqrt(2) under RU rounds up one ulp");
    }

    /// SqrtRm: sqrt(2.0f32) RNE keeps the native fast path (no Z3).
    #[test]
    fn test_float_sqrt_with_rm_rne_fastpath_f32() {
        let ctx = SymContext::new_mock();
        let rm_rne = RustBV::concrete(0, 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::unop_with_rm(IROp::FSqrt(IRType::F32), rm_rne, two, &ctx).unwrap();
        assert!(!result.is_symbolic());
        assert_eq!(result.as_u64(), Some(0x3FB504F3));
    }

    /// Iop_SqrtF32 reaches us as a VEX Binop (arg1=rm, arg2=value). Verify
    /// that VEXOps::binop routes it through unop_with_rm so the rounding
    /// mode is honored — RU on sqrt(2.0f32) must round up by one ulp.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_sqrt_via_binop_with_rm_ru_f32() {
        let ctx = SymContext::new_mock();
        let rm_ru = RustBV::concrete(2, 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::binop(IROp::FSqrt(IRType::F32), rm_ru, two, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3FB504F4, "binop FSqrt+RU rounds up one ulp");
    }

    /// Iop_SqrtF64 via binop with RNE: native fast path returns sqrt(16.0)=4.0.
    #[test]
    fn test_float_sqrt_via_binop_rne_fastpath_f64() {
        let ctx = SymContext::new_mock();
        let rm_rne = RustBV::concrete(0, 32);
        let sixteen = RustBV::concrete(16.0f64.to_bits() as u128, 64);

        let result =
            VEXOps::binop(IROp::FSqrt(IRType::F64), rm_rne, sixteen, &ctx).unwrap();
        assert!(!result.is_symbolic(), "RNE concrete must stay native");
        assert_eq!(result.as_u64(), Some(4.0f64.to_bits()));
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

    /// Concrete coverage for every scalar-in-vector FP IROp at F64 precision.
    /// The {add,sub,mul,div,sqrt,max,min} S-suffixed variants all write to
    /// lane 0 (low 64 bits) and pass the upper 64 bits of `xmm0` through.
    #[test]
    fn test_vec_float_scalar_all_variants_f64() {
        let ctx = SymContext::new_mock();
        let upper_pattern: u128 = 0xCAFE_BABE_DEAD_BEEFu128 << 64;

        let xmm0 = |lane0: f64| {
            RustBV::concrete(upper_pattern | (lane0.to_bits() as u128), 128)
        };
        let xmm1 = |lane0: f64| RustBV::concrete(lane0.to_bits() as u128, 128);

        let cases: Vec<(IROp, f64, f64, f64)> = vec![
            (IROp::VFAddS { elem: IRType::F64 }, 4.0, 1.5, 5.5),
            (IROp::VFSubS { elem: IRType::F64 }, 5.0, 1.25, 3.75),
            (IROp::VFMulS { elem: IRType::F64 }, 3.0, 2.5, 7.5),
            (IROp::VFDivS { elem: IRType::F64 }, 9.0, 4.0, 2.25),
            (IROp::VFMaxS { elem: IRType::F64 }, 1.5, 2.5, 2.5),
            (IROp::VFMinS { elem: IRType::F64 }, 1.5, 2.5, 1.5),
        ];

        for (op, l, r, expected) in cases {
            let result = VEXOps::binop(op.clone(), xmm0(l), xmm1(r), &ctx).unwrap();
            let rv = result.as_u128().unwrap();
            let lane0 = f64::from_bits((rv & 0xFFFF_FFFF_FFFF_FFFFu128) as u64);
            assert!(
                (lane0 - expected).abs() < 1e-9,
                "{:?}: lane0={} expected={}",
                op,
                lane0,
                expected,
            );
            assert_eq!(
                rv & !0xFFFF_FFFF_FFFF_FFFFu128,
                upper_pattern,
                "{:?}: upper 64 bits must pass through",
                op,
            );
        }

        // Unary VFSqrtS{F64}: sqrt(16.0) = 4.0, upper 64 bits pass through.
        let arg = xmm0(16.0);
        let sqrt_res = VEXOps::unop(IROp::VFSqrtS { elem: IRType::F64 }, arg, &ctx).unwrap();
        let sv = sqrt_res.as_u128().unwrap();
        let sqrt_lane0 = f64::from_bits((sv & 0xFFFF_FFFF_FFFF_FFFFu128) as u64);
        assert!(
            (sqrt_lane0 - 4.0).abs() < 1e-9,
            "VFSqrtS{{F64}}: lane0={} expected=4.0",
            sqrt_lane0,
        );
        assert_eq!(
            sv & !0xFFFF_FFFF_FFFF_FFFFu128,
            upper_pattern,
            "VFSqrtS{{F64}}: upper 64 bits must pass through",
        );
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

    // =========================================================================
    // Packed integer min/max/abs tests
    // =========================================================================

    /// PMINSW-style: signed min over 8x i16 lanes, mix of positive and negative.
    #[test]
    fn test_vec_int_min_signed_concrete() {
        let ctx = SymContext::new_mock();

        let l: [i16; 8] = [-5, 100,    0, -32768,  1,    -1, 32767, -2];
        let r: [i16; 8] = [-3, 200, -100, -32767, -1,     0, 32766,  3];
        let exp: [i16; 8] = [
            -5, 100, -100, -32768, -1, -1, 32766, -2,
        ];

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..8 {
            lv |= ((l[i] as u16) as u128) << (i as u32 * 16);
            rv |= ((r[i] as u16) as u128) << (i as u32 * 16);
        }
        let result = VEXOps::binop(
            IROp::VMin { elem: IRType::I16, count: 8, signed: true },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..8 {
            let lane = ((got >> (i as u32 * 16)) & 0xFFFF) as u16 as i16;
            assert_eq!(lane, exp[i], "lane {} expected {}, got {}", i, exp[i], lane);
        }
    }

    /// PMAXUB-style: unsigned max over 16x u8 lanes.
    #[test]
    fn test_vec_int_max_unsigned_concrete() {
        let ctx = SymContext::new_mock();

        let l: [u8; 16] = [0xFF, 0x00, 0x80, 0x7F,  1,  2,  3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let r: [u8; 16] = [0x00, 0xFF, 0x7F, 0x80,  9,  8,  7, 6, 5, 4, 3, 2, 1,  0,  0,  0];
        let mut exp = [0u8; 16];
        for i in 0..16 {
            exp[i] = if l[i] > r[i] { l[i] } else { r[i] };
        }

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..16 {
            lv |= (l[i] as u128) << (i as u32 * 8);
            rv |= (r[i] as u128) << (i as u32 * 8);
        }
        let result = VEXOps::binop(
            IROp::VMax { elem: IRType::I8, count: 16, signed: false },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..16 {
            let lane = ((got >> (i as u32 * 8)) & 0xFF) as u8;
            assert_eq!(lane, exp[i], "lane {} expected {:#x}, got {:#x}", i, exp[i], lane);
        }
    }

    /// PABSW-style: per-lane absolute value over 8x i16 lanes (incl. INT_MIN
    /// which stays INT_MIN under two's-complement |x|).
    #[test]
    fn test_vec_int_abs_concrete() {
        let ctx = SymContext::new_mock();

        let v: [i16; 8] = [-5, 100, 0, -32768, 1, -1, 32767, -200];
        let exp: [u16; 8] = [5, 100, 0, 0x8000 /* INT_MIN stays */, 1, 1, 32767, 200];

        let mut bits: u128 = 0;
        for i in 0..8 {
            bits |= ((v[i] as u16) as u128) << (i as u32 * 16);
        }
        let result = VEXOps::unop(
            IROp::VAbs { elem: IRType::I16, count: 8 },
            RustBV::concrete(bits, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..8 {
            let lane = ((got >> (i as u32 * 16)) & 0xFFFF) as u16;
            assert_eq!(lane, exp[i], "lane {} expected {:#x}, got {:#x}", i, exp[i], lane);
        }
    }

    /// Symbolic VMax (signed): constrain right == 7, derive left from a free
    /// 4x i32 vector, and verify that asserting result == [7, 7, 7, 7] forces
    /// every lane of left to be <= 7.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_int_max_symbolic_signed() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();

        // r = [7, 7, 7, 7] as i32x4
        let mut rv: u128 = 0;
        for i in 0..4u32 {
            rv |= (7u128) << (i * 32);
        }
        let r = RustBV::concrete(rv, 128);
        let l = RustBV::symbolic(&ctx, "vmax_l", 128);

        let result = VEXOps::binop(
            IROp::VMax { elem: IRType::I32, count: 4, signed: true },
            l.clone(),
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        // Constrain result == [7, 7, 7, 7]; this only requires l <= 7 per lane,
        // so the constraint must remain SAT.
        let exp = RustBV::concrete(rv, 128);
        ctx.add_constraint(result.to_z3_ast()._eq(&exp.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT after constraining max == 7");
    }

    // =========================================================================
    // Packed FP add/sub/mul/div/sqrt/abs/min/max tests
    // =========================================================================

    /// ADDPS-style: 4x f32 add, concrete.
    #[test]
    fn test_vec_float_add_concrete_f32x4() {
        let ctx = SymContext::new_mock();

        let l = [1.0f32, 2.5, -3.0, 0.5];
        let r = [10.0f32, -2.5, 3.0, 8.0];
        let exp: [f32; 4] = [11.0, 0.0, 0.0, 8.5];

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..4u32 {
            lv |= (l[i as usize].to_bits() as u128) << (i * 32);
            rv |= (r[i as usize].to_bits() as u128) << (i * 32);
        }
        let result = VEXOps::binop(
            IROp::VFAdd { elem: IRType::F32, count: 4 },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..4 {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - exp[i]).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    /// DIVPD-style: 2x f64 div, concrete.
    #[test]
    fn test_vec_float_div_concrete_f64x2() {
        let ctx = SymContext::new_mock();

        let l = [10.0f64, -8.0];
        let r = [4.0f64, 2.0];
        let exp = [2.5f64, -4.0];

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..2u32 {
            lv |= (l[i as usize].to_bits() as u128) << (i * 64);
            rv |= (r[i as usize].to_bits() as u128) << (i * 64);
        }
        let result = VEXOps::binop(
            IROp::VFDiv { elem: IRType::F64, count: 2 },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..2 {
            let lane = f64::from_bits(((got >> (i as u32 * 64)) & 0xFFFFFFFFFFFFFFFFu128) as u64);
            assert!(
                (lane - exp[i]).abs() < 1e-12,
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    /// SQRTPS-style: 4x f32 sqrt, concrete.
    #[test]
    fn test_vec_float_sqrt_concrete_f32x4() {
        let ctx = SymContext::new_mock();

        let v = [4.0f32, 9.0, 16.0, 25.0];
        let exp = [2.0f32, 3.0, 4.0, 5.0];

        let mut bits: u128 = 0;
        for i in 0..4u32 {
            bits |= (v[i as usize].to_bits() as u128) << (i * 32);
        }
        let result = VEXOps::unop(
            IROp::VFSqrt { elem: IRType::F32, count: 4 },
            RustBV::concrete(bits, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..4 {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - exp[i]).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    /// Iop_Abs32Fx4-style: per-lane fabs (clears sign bit).
    #[test]
    fn test_vec_float_abs_concrete_f32x4() {
        let ctx = SymContext::new_mock();

        let v = [-1.5f32, 2.5, -0.0, f32::NEG_INFINITY];
        let exp = [1.5f32, 2.5, 0.0, f32::INFINITY];

        let mut bits: u128 = 0;
        for i in 0..4u32 {
            bits |= (v[i as usize].to_bits() as u128) << (i * 32);
        }
        let result = VEXOps::unop(
            IROp::VFAbs { elem: IRType::F32, count: 4 },
            RustBV::concrete(bits, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..4 {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert_eq!(
                lane.to_bits(),
                exp[i].to_bits(),
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    /// MAXPS-style: per-lane max of two f32x4 vectors.
    #[test]
    fn test_vec_float_max_concrete_f32x4() {
        let ctx = SymContext::new_mock();

        let l = [1.0f32, -2.0, 3.0, 0.5];
        let r = [4.0f32, -3.0, 2.0, 0.6];
        let exp = [4.0f32, -2.0, 3.0, 0.6];

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..4u32 {
            lv |= (l[i as usize].to_bits() as u128) << (i * 32);
            rv |= (r[i as usize].to_bits() as u128) << (i * 32);
        }
        let result = VEXOps::binop(
            IROp::VFMax { elem: IRType::F32, count: 4 },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..4 {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - exp[i]).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    /// MINPD-style: per-lane min of two f64x2 vectors.
    #[test]
    fn test_vec_float_min_concrete_f64x2() {
        let ctx = SymContext::new_mock();

        let l = [1.5f64, -2.5];
        let r = [-1.5f64, 0.5];
        let exp = [-1.5f64, -2.5];

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..2u32 {
            lv |= (l[i as usize].to_bits() as u128) << (i * 64);
            rv |= (r[i as usize].to_bits() as u128) << (i * 64);
        }
        let result = VEXOps::binop(
            IROp::VFMin { elem: IRType::F64, count: 2 },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for i in 0..2 {
            let lane = f64::from_bits(((got >> (i as u32 * 64)) & 0xFFFFFFFFFFFFFFFFu128) as u64);
            assert!(
                (lane - exp[i]).abs() < 1e-12,
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    /// Symbolic VFAdd: build a free f32x4 left vector, constrain it so each
    /// lane equals 1.0, add a concrete [2.0, 3.0, 4.0, 5.0], and verify the
    /// model agrees with [3.0, 4.0, 5.0, 6.0] per lane.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_float_add_symbolic_f32x4() {
        use z3::ast::Ast;

        let ctx = SymContext::new_mock();

        let mut rv: u128 = 0;
        let consts = [2.0f32, 3.0, 4.0, 5.0];
        for i in 0..4u32 {
            rv |= (consts[i as usize].to_bits() as u128) << (i * 32);
        }
        let r = RustBV::concrete(rv, 128);

        let mut lv_target: u128 = 0;
        for i in 0..4u32 {
            lv_target |= (1.0f32.to_bits() as u128) << (i * 32);
        }
        let l = RustBV::symbolic(&ctx, "vfadd_l", 128);
        ctx.add_constraint(l.to_z3_ast()._eq(&RustBV::concrete(lv_target, 128).to_z3_ast()));

        let result = VEXOps::binop(
            IROp::VFAdd { elem: IRType::F32, count: 4 },
            l,
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert!(ctx.is_sat(), "expected SAT");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        let exp = [3.0f32, 4.0, 5.0, 6.0];
        for i in 0..4 {
            let lane = f32::from_bits(((model >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - exp[i]).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                exp[i],
                lane
            );
        }
    }

    // ---- FCmpScalarLane (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}) ----

    fn make_v128_lane0(lane0: u128, upper96: u128) -> u128 {
        debug_assert!(lane0 <= 0xFFFF_FFFF);
        (upper96 << 32) | lane0
    }
    fn make_v128_lane0_64(lane0: u128, upper64: u128) -> u128 {
        debug_assert!(lane0 <= 0xFFFF_FFFF_FFFF_FFFF);
        (upper64 << 64) | lane0
    }

    #[test]
    fn test_fcmp_scalar_lane_eq_f32_concrete_true() {
        // CMPEQSS: lane0(left)==lane0(right) → 0xFFFFFFFF in lane0; upper from left.
        let ctx = SymContext::new_mock();
        let upper = 0xDEAD_BEEF_DEAD_BEEF_DEAD_BEEFu128;
        let l = RustBV::concrete(
            make_v128_lane0(2.0f32.to_bits() as u128, upper), 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Eq, ty: IRType::F32 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.width(), 128);
        let v = res.as_u128().expect("concrete result");
        assert_eq!(v & 0xFFFF_FFFF, 0xFFFF_FFFF, "lane0 should be all-1s");
        assert_eq!(v >> 32, upper, "upper96 must passthrough from left");
    }

    #[test]
    fn test_fcmp_scalar_lane_eq_f32_concrete_false() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(1.0f32.to_bits() as u128, 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Eq, ty: IRType::F32 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        assert_eq!(v & 0xFFFF_FFFF, 0, "lane0 should be 0 on false");
    }

    #[test]
    fn test_fcmp_scalar_lane_lt_f32_concrete() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(1.0f32.to_bits() as u128, 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Lt, ty: IRType::F32 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF);
    }

    #[test]
    fn test_fcmp_scalar_lane_le_f64_concrete_eq() {
        let ctx = SymContext::new_mock();
        let upper = 0x123456789ABCDEF0u128;
        let l = RustBV::concrete(
            make_v128_lane0_64(2.5f64.to_bits() as u128, upper), 128);
        let r = RustBV::concrete(2.5f64.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Le, ty: IRType::F64 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        assert_eq!(v & 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(v >> 64, upper, "upper64 passthrough from left");
    }

    #[test]
    fn test_fcmp_scalar_lane_un_f32_concrete_nan() {
        // CMPUNORD: returns true if either operand is NaN.
        let ctx = SymContext::new_mock();
        let nan = f32::NAN.to_bits() as u128;
        let l = RustBV::concrete(nan, 128);
        let r = RustBV::concrete(1.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Un, ty: IRType::F32 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF);
    }

    #[test]
    fn test_fcmp_scalar_lane_un_f64_concrete_ordered() {
        // Both ordered → CMPUNORD returns 0.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(1.0f64.to_bits() as u128, 128);
        let r = RustBV::concrete(2.0f64.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Un, ty: IRType::F64 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF_FFFF_FFFF, 0);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fcmp_scalar_lane_eq_f32_symbolic() {
        // Symbolic: constrain low32(result)==0xFFFFFFFF given right=2.0 and
        // some symbolic left → solver must pick left.lane0 == 2.0.
        use z3::ast::Ast;
        let ctx = SymContext::new_mock();
        let l = RustBV::symbolic(&ctx, "fcmp_lane_l", 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane { kind: FCmpKind::Eq, ty: IRType::F32 },
            l.clone(), r, &ctx,
        ).unwrap();
        assert_eq!(res.width(), 128);
        let lo32 = res.extract(31, 0, &ctx);
        let true_mask = RustBV::concrete(0xFFFF_FFFF, 32);
        ctx.add_constraint(lo32.to_z3_ast()._eq(&true_mask.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT after FCmpScalarLane Eq mask=all1s");

        let model = ctx.eval(&l).expect("eval(l) returned None");
        let lane0 = f32::from_bits((model & 0xFFFF_FFFF) as u32);
        assert!((lane0 - 2.0).abs() < 1e-6, "expected lane0==2.0, got {}", lane0);
    }

    // ---- FComCC (Iop_CmpF32, Iop_CmpF64, x87 FCOM) ----

    #[test]
    fn test_fcom_cc_f32_concrete_eq() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(2.5f32.to_bits() as u128, 32);
        let r = RustBV::concrete(2.5f32.to_bits() as u128, 32);
        let res = VEXOps::binop(IROp::FComCC(IRType::F32), l, r, &ctx).unwrap();
        assert_eq!(res.width(), 32);
        assert_eq!(res.as_u128().unwrap(), 0x40);
    }

    #[test]
    fn test_fcom_cc_f32_concrete_lt() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 32);
        let res = VEXOps::binop(IROp::FComCC(IRType::F32), l, r, &ctx).unwrap();
        assert_eq!(res.as_u128().unwrap(), 0x01);
    }

    #[test]
    fn test_fcom_cc_f64_concrete_gt() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(3.0f64.to_bits() as u128, 64);
        let r = RustBV::concrete(2.0f64.to_bits() as u128, 64);
        let res = VEXOps::binop(IROp::FComCC(IRType::F64), l, r, &ctx).unwrap();
        assert_eq!(res.as_u128().unwrap(), 0x00);
    }

    #[test]
    fn test_fcom_cc_f64_concrete_unordered() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(f64::NAN.to_bits() as u128, 64);
        let r = RustBV::concrete(1.0f64.to_bits() as u128, 64);
        let res = VEXOps::binop(IROp::FComCC(IRType::F64), l, r, &ctx).unwrap();
        assert_eq!(res.as_u128().unwrap(), 0x45);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fcom_cc_symbolic_lt() {
        // Symbolic: constrain FComCC(x, 5.0) == 0x01 → x must be < 5.0 (and not NaN).
        use z3::ast::Ast;
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "fcom_x", 64);
        let five = RustBV::concrete(5.0f64.to_bits() as u128, 64);
        let res = VEXOps::binop(IROp::FComCC(IRType::F64), x.clone(), five, &ctx).unwrap();
        assert_eq!(res.width(), 32);
        let want = RustBV::concrete(0x01, 32);
        ctx.add_constraint(res.to_z3_ast()._eq(&want.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for FComCC(x, 5.0) == LT");
        let model_x = ctx.eval(&x).expect("eval(x) None");
        let xf = f64::from_bits(model_x as u64);
        assert!(!xf.is_nan() && xf < 5.0, "expected x < 5.0 and not NaN, got {}", xf);
    }

    // ---- FCmpVecPacked (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}) ----

    /// Pack four f32 values into a single 128-bit vector (lane 0 first).
    fn pack_4xf32(a: f32, b: f32, c: f32, d: f32) -> u128 {
        let mut r: u128 = 0;
        r |= (a.to_bits() as u128) << 0;
        r |= (b.to_bits() as u128) << 32;
        r |= (c.to_bits() as u128) << 64;
        r |= (d.to_bits() as u128) << 96;
        r
    }
    fn pack_2xf64(a: f64, b: f64) -> u128 {
        ((b.to_bits() as u128) << 64) | (a.to_bits() as u128)
    }
    fn pack_2xf32_64(a: f32, b: f32) -> u128 {
        ((b.to_bits() as u128) << 32) | (a.to_bits() as u128)
    }

    #[test]
    fn test_fcmp_packed_eq_32fx4_concrete() {
        // CMPEQPS lane-by-lane: lanes 0 and 2 equal, lanes 1 and 3 differ.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(1.0, 2.0, 3.0, 4.0), 128);
        let r = RustBV::concrete(pack_4xf32(1.0, 5.0, 3.0, 7.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Eq, elem: IRType::F32, count: 4 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.width(), 128);
        let v = res.as_u128().unwrap();
        // Expect lane 0 = 0xFFFFFFFF, lane 1 = 0, lane 2 = 0xFFFFFFFF, lane 3 = 0.
        let expected: u128 =
            (0xFFFF_FFFFu128 << 0) | (0u128 << 32) | (0xFFFF_FFFFu128 << 64) | (0u128 << 96);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_lt_32fx4_concrete() {
        // CMPLTPS: 1.0 < 2.0 (T), 5.0 < 3.0 (F), -1.0 < 0.0 (T), 4.0 < 4.0 (F).
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(1.0, 5.0, -1.0, 4.0), 128);
        let r = RustBV::concrete(pack_4xf32(2.0, 3.0,  0.0, 4.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Lt, elem: IRType::F32, count: 4 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 = 0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 64);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_gt_32fx4_concrete() {
        // CMPGTPS: 5.0 > 2.0 (T), 1.0 > 3.0 (F), 4.0 > 4.0 (F), 9.0 > 0.0 (T).
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(5.0, 1.0, 4.0, 9.0), 128);
        let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 4.0, 0.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Gt, elem: IRType::F32, count: 4 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 = 0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 96);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_ge_32fx4_concrete() {
        // CMPGEPS: 5.0>=2.0 T, 3.0>=3.0 T, 1.0>=2.0 F, 4.0>=4.0 T.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(5.0, 3.0, 1.0, 4.0), 128);
        let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 2.0, 4.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Ge, elem: IRType::F32, count: 4 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 =
            0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 32) | (0u128 << 64) | (0xFFFF_FFFFu128 << 96);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_le_64fx2_concrete() {
        // CMPLEPD: 1.0<=2.0 T, 3.0<=3.0 T.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_2xf64(1.0, 3.0), 128);
        let r = RustBV::concrete(pack_2xf64(2.0, 3.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Le, elem: IRType::F64, count: 2 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 = 0xFFFF_FFFF_FFFF_FFFFu128 | (0xFFFF_FFFF_FFFF_FFFFu128 << 64);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_un_32fx4_nan_in_one_lane() {
        // CMPUNPS: NaN in lane 1 only → only lane 1 set.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(1.0, f32::NAN, 3.0, 4.0), 128);
        let r = RustBV::concrete(pack_4xf32(2.0, 5.0, 3.0, 4.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Un, elem: IRType::F32, count: 4 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 = 0xFFFF_FFFFu128 << 32;
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_un_64fx2_nan_in_either() {
        // CMPUNPD: lane 0 has NaN on right, lane 1 ordered.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_2xf64(1.0, 3.0), 128);
        let r = RustBV::concrete(pack_2xf64(f64::NAN, 4.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Un, elem: IRType::F64, count: 2 },
            l, r, &ctx,
        ).unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fcmp_packed_eq_32fx2_concrete_i64() {
        // ARM NEON Iop_CmpEQ32Fx2 returns I64 (2 lanes of 32-bit float in 64 bits).
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_2xf32_64(1.0, 2.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
        let r = RustBV::concrete(pack_2xf32_64(1.0, 5.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Eq, elem: IRType::F32, count: 2 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.width(), 64);
        let v = res.as_u128().unwrap();
        // Lane 0 equal → 0xFFFFFFFF; lane 1 unequal → 0.
        assert_eq!(v as u64, 0xFFFF_FFFFu64);
    }

    #[test]
    fn test_fcmp_packed_gt_32fx2_concrete_i64() {
        // ARM NEON Iop_CmpGT32Fx2.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_2xf32_64(5.0, 1.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
        let r = RustBV::concrete(pack_2xf32_64(2.0, 3.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Gt, elem: IRType::F32, count: 2 },
            l, r, &ctx,
        ).unwrap();
        assert_eq!(res.width(), 64);
        let v = res.as_u128().unwrap();
        assert_eq!(v as u64, 0xFFFF_FFFFu64);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fcmp_packed_lt_32fx4_symbolic() {
        // Symbolic vector `l` against concrete `r`. Constrain low lane mask to all-1s
        // → solver must satisfy lane 0 of l < lane 0 of r (= 5.0). Other lanes free.
        use z3::ast::Ast;
        let ctx = SymContext::new_mock();
        let l = RustBV::symbolic(&ctx, "fpkd_lt_l", 128);
        let r = RustBV::concrete(pack_4xf32(5.0, 1.0, 1.0, 1.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked { kind: FCmpKind::Lt, elem: IRType::F32, count: 4 },
            l.clone(), r, &ctx,
        ).unwrap();
        assert_eq!(res.width(), 128);
        let lane0_mask = res.extract(31, 0, &ctx);
        let true_mask = RustBV::concrete(0xFFFF_FFFF, 32);
        ctx.add_constraint(lane0_mask.to_z3_ast()._eq(&true_mask.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for lane0 LT");
        let model = ctx.eval(&l).expect("eval(l) None");
        let lane0 = f32::from_bits((model & 0xFFFF_FFFF) as u32);
        assert!(lane0 < 5.0 && !lane0.is_nan(), "expected lane0 < 5.0 and not NaN, got {}", lane0);
    }

    // -------------------------------------------------------------------------
    // NEON SIMD (angr-bkcs.2)
    // -------------------------------------------------------------------------

    #[test]
    fn test_vmul_8x8_concrete() {
        let ctx = SymContext::new_mock();
        // 8 lanes of 8-bit, lane i = i for both → product = i*i mod 256.
        // l = 0x0706050403020100, r = same. result lane i = i*i.
        let l = RustBV::concrete(0x0706_0504_0302_0100u128, 64);
        let r = RustBV::concrete(0x0706_0504_0302_0100u128, 64);
        let res = VEXOps::binop(IROp::VMul { elem: IRType::I8, count: 8 }, l, r, &ctx).unwrap();
        assert_eq!(res.width(), 64);
        let v = res.as_u128().unwrap();
        // lane i (bits [8i+7:8i]) should equal i*i.
        for i in 0u128..8 {
            let lane = (v >> (i * 8)) & 0xFF;
            assert_eq!(lane, (i * i) & 0xFF, "lane {} of Mul8x8", i);
        }
    }

    #[test]
    fn test_vmul_8x16_concrete() {
        let ctx = SymContext::new_mock();
        // All lanes = 3, multiplied by all lanes = 5 → all lanes = 15.
        let lo64 = 0x0303_0303_0303_0303u128;
        let l = RustBV::concrete(lo64 | (lo64 << 64), 128);
        let lo5 = 0x0505_0505_0505_0505u128;
        let r = RustBV::concrete(lo5 | (lo5 << 64), 128);
        let res = VEXOps::binop(IROp::VMul { elem: IRType::I8, count: 16 }, l, r, &ctx).unwrap();
        assert_eq!(res.width(), 128);
        let v = res.as_u128().unwrap();
        for i in 0..16 {
            let lane = (v >> (i * 8)) & 0xFF;
            assert_eq!(lane, 15, "lane {} of Mul8x16", i);
        }
    }

    #[test]
    fn test_vget_elem_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
        // Lane 0 = 0x11, lane 7 = 0x88.
        for (lane, expected) in
            [(0u128, 0x11), (1, 0x22), (2, 0x33), (3, 0x44), (4, 0x55), (5, 0x66), (6, 0x77), (7, 0x88)]
        {
            let idx = RustBV::concrete(lane, 8);
            let res = VEXOps::binop(
                IROp::VGetElem { elem: IRType::I8, count: 8 },
                vec.clone(),
                idx,
                &ctx,
            )
            .unwrap();
            assert_eq!(res.width(), 8);
            assert_eq!(res.as_u128().unwrap(), expected, "lane {}", lane);
        }
    }

    #[test]
    fn test_vget_elem_16x8_concrete() {
        let ctx = SymContext::new_mock();
        // V128 with 8 lanes of 16 bits. Lane 3 = 0xDEAD.
        let mut payload: u128 = 0;
        payload |= (0xDEAD as u128) << (3 * 16);
        payload |= (0xBEEF as u128) << (7 * 16);
        let vec = RustBV::concrete(payload, 128);
        let res = VEXOps::binop(
            IROp::VGetElem { elem: IRType::I16, count: 8 },
            vec.clone(),
            RustBV::concrete(3, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 16);
        assert_eq!(res.as_u128().unwrap(), 0xDEAD);

        let res = VEXOps::binop(
            IROp::VGetElem { elem: IRType::I16, count: 8 },
            vec,
            RustBV::concrete(7, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.as_u128().unwrap(), 0xBEEF);
    }

    #[test]
    fn test_vget_elem_64x2_concrete() {
        let ctx = SymContext::new_mock();
        let lo = 0xAAAA_BBBB_CCCC_DDDDu128;
        let hi = 0x1111_2222_3333_4444u128;
        let vec = RustBV::concrete(lo | (hi << 64), 128);
        let r0 = VEXOps::binop(
            IROp::VGetElem { elem: IRType::I64, count: 2 },
            vec.clone(),
            RustBV::concrete(0, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(r0.as_u128().unwrap(), lo);
        let r1 = VEXOps::binop(
            IROp::VGetElem { elem: IRType::I64, count: 2 },
            vec,
            RustBV::concrete(1, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(r1.as_u128().unwrap(), hi);
    }

    #[test]
    fn test_vset_elem_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0x0u128, 64);
        // Set lane 3 to 0xFF.
        let res = VEXOps::binop_with_rm(
            IROp::VSetElem { elem: IRType::I8, count: 8 },
            vec,
            RustBV::concrete(3, 8),
            RustBV::concrete(0xFF, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 64);
        assert_eq!(res.as_u128().unwrap(), 0xFF00_0000u128);
    }

    #[test]
    fn test_vset_elem_16x8_concrete() {
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0u128, 128);
        // Set lane 5 to 0xCAFE in a 16x8 vector.
        let res = VEXOps::binop_with_rm(
            IROp::VSetElem { elem: IRType::I16, count: 8 },
            vec,
            RustBV::concrete(5, 8),
            RustBV::concrete(0xCAFE, 16),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        assert_eq!(res.as_u128().unwrap(), (0xCAFE as u128) << (5 * 16));
    }

    #[test]
    fn test_vset_elem_preserves_other_lanes() {
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0xDEAD_BEEF_CAFE_F00Du128, 64);
        // Overwrite lane 2 (byte 2) with 0x77.
        let res = VEXOps::binop_with_rm(
            IROp::VSetElem { elem: IRType::I8, count: 8 },
            vec,
            RustBV::concrete(2, 8),
            RustBV::concrete(0x77, 8),
            &ctx,
        )
        .unwrap();
        let v = res.as_u128().unwrap();
        // Original byte 2 was 0xFE; expect 0x77 in its place, rest unchanged.
        let expected: u128 = (0xDEAD_BEEF_CAFE_F00Du128 & !(0xFFu128 << 16)) | (0x77u128 << 16);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_vset_elem_round_trip_via_get() {
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0u128, 128);
        let inserted = VEXOps::binop_with_rm(
            IROp::VSetElem { elem: IRType::I32, count: 4 },
            vec,
            RustBV::concrete(2, 8),
            RustBV::concrete(0x1234_5678, 32),
            &ctx,
        )
        .unwrap();
        let lane = VEXOps::binop(
            IROp::VGetElem { elem: IRType::I32, count: 4 },
            inserted,
            RustBV::concrete(2, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(lane.as_u128().unwrap(), 0x1234_5678);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vget_elem_symbolic_idx() {
        // Build a concrete vector with distinct lane values, then read
        // through a symbolic idx and constrain it to return a specific lane.
        use z3::ast::Ast;
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
        let sym_idx = RustBV::symbolic(&ctx, "get_idx", 8);
        let res = VEXOps::binop(
            IROp::VGetElem { elem: IRType::I8, count: 8 },
            vec,
            sym_idx.clone(),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 8);
        // Constrain result to 0x66 → solver must pick idx == 5.
        let target = RustBV::concrete(0x66, 8);
        ctx.add_constraint(res.to_z3_ast()._eq(&target.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for lane==0x66");
        let model_idx = ctx.eval(&sym_idx).expect("eval(idx) None");
        // idx must be 5 mod 8 (modulo because ITE chain ignores high bits).
        assert_eq!(model_idx & 0x7, 5, "expected idx&7 == 5, got {}", model_idx);
    }
}
