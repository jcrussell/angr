//! VEX operation implementations.
//!
//! This module implements VEX operations using a parameterized approach.
//! Instead of ~200 separate implementations (e.g., add8, add16, add32, add64),
//! we have a single implementation per operation type that handles all widths.

use std::sync::Arc;

use crate::symbolic::{BVOp, FloatOpKind, FloatPrec, RustBV, SymContext, VexOpFamily};

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
fn build_float_expr(kind: FloatOpKind, prec: FloatPrec, operands: Vec<RustBV>) -> RustBV {
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

/// Generate a `FloatLaneOp` impl for a binary op whose concrete path is an
/// infix operator and whose symbolic path is a single `FloatOpKind`.
macro_rules! impl_float_lane_binop {
    ($name:ident, $op:tt, $kind:expr) => {
        impl FloatLaneOp for $name {
            fn arity(&self) -> usize {
                2
            }
            fn concrete_f32(&self, a: &[f32]) -> f32 {
                a[0] $op a[1]
            }
            fn concrete_f64(&self, a: &[f64]) -> f64 {
                a[0] $op a[1]
            }
            fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
                build_float_expr($kind, prec, args)
            }
        }
    };
}

/// Generate a `FloatLaneOp` impl for a unary op whose concrete path is a
/// method call on the lane and whose symbolic path is a single `FloatOpKind`.
macro_rules! impl_float_lane_unop {
    ($name:ident, $method:ident, $kind:expr) => {
        impl FloatLaneOp for $name {
            fn arity(&self) -> usize {
                1
            }
            fn concrete_f32(&self, a: &[f32]) -> f32 {
                a[0].$method()
            }
            fn concrete_f64(&self, a: &[f64]) -> f64 {
                a[0].$method()
            }
            fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, _ctx: &SymContext) -> RustBV {
                build_float_expr($kind, prec, args)
            }
        }
    };
}

impl_float_lane_binop!(FAdd, +, FloatOpKind::Add);
impl_float_lane_binop!(FSub, -, FloatOpKind::Sub);
impl_float_lane_binop!(FMul, *, FloatOpKind::Mul);
impl_float_lane_binop!(FDiv, /, FloatOpKind::Div);
impl_float_lane_unop!(FSqrt, sqrt, FloatOpKind::Sqrt);
impl_float_lane_unop!(FAbs, abs, FloatOpKind::Abs);

/// Build a symbolic min/max ITE over two operand lanes. `swap_cmp_args=false`
/// gives `ITE(l < r, l, r)` (min); `true` gives `ITE(r < l, l, r)` (max).
/// The symbolic path always uses `CmpLt`, swapping operand order rather than
/// minting a separate `CmpGt` kind.
fn float_minmax_symbolic(
    args: Vec<RustBV>,
    prec: FloatPrec,
    ctx: &SymContext,
    swap_cmp_args: bool,
) -> RustBV {
    let mut iter = args.into_iter();
    let l_lane = iter.next().expect("FMin/FMax arity 2");
    let r_lane = iter.next().expect("FMin/FMax arity 2");
    let cmp_args = if swap_cmp_args {
        vec![r_lane.clone(), l_lane.clone()]
    } else {
        vec![l_lane.clone(), r_lane.clone()]
    };
    let cond = build_float_expr(FloatOpKind::CmpLt, prec, cmp_args);
    cond.ite_into(l_lane, r_lane, ctx)
}

/// Generate a `FloatLaneOp` impl for FP min/max. `$cmp` is the operator that
/// decides "left wins" on the concrete path (e.g. `<` for FMin, `>` for FMax);
/// `$swap_cmp_args` adapts that to the always-CmpLt symbolic path.
macro_rules! impl_float_lane_minmax {
    ($name:ident, $cmp:tt, $swap_cmp_args:expr) => {
        impl FloatLaneOp for $name {
            fn arity(&self) -> usize {
                2
            }
            fn concrete_f32(&self, a: &[f32]) -> f32 {
                if a[0] $cmp a[1] { a[0] } else { a[1] }
            }
            fn concrete_f64(&self, a: &[f64]) -> f64 {
                if a[0] $cmp a[1] { a[0] } else { a[1] }
            }
            fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, ctx: &SymContext) -> RustBV {
                float_minmax_symbolic(args, prec, ctx, $swap_cmp_args)
            }
        }
    };
}

impl_float_lane_minmax!(FMin, <, false);
impl_float_lane_minmax!(FMax, >, true);

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

/// Per-lane shift kind for `vec_shift_vec` (the shift-by-vector dispatch
/// shared by `VShl`, `VShr`, and `VSar`).
#[derive(Copy, Clone, Debug)]
enum VecShiftKind {
    /// Logical left shift (Z3 `bvshl`); `Iop_Shl` / `Iop_Sal`.
    Shl,
    /// Logical right shift (Z3 `bvlshr`); `Iop_Shr`.
    Shr,
    /// Arithmetic right shift (Z3 `bvashr`); `Iop_Sar`.
    Sar,
}

/// Per-pair combiner kind for `vec_pairwise_binop` (the binary pairwise
/// dispatch shared by `VPwAdd`, `VPwMin`, `VPwMax`).
#[derive(Copy, Clone, Debug)]
enum PwOp {
    /// Pairwise add (Iop_PwAdd{N}x{M}).
    Add,
    /// Pairwise signed min (Iop_PwMin{N}Sx{M}).
    MinS,
    /// Pairwise unsigned min (Iop_PwMin{N}Ux{M}).
    MinU,
    /// Pairwise signed max (Iop_PwMax{N}Sx{M}).
    MaxS,
    /// Pairwise unsigned max (Iop_PwMax{N}Ux{M}).
    MaxU,
}

/// Per-lane counting kind for `vec_lane_count` (the unary dispatch shared by
/// `VClz` and `VCls`).
#[derive(Copy, Clone, Debug)]
enum LaneCountKind {
    /// Count leading zeros — Iop_Clz{N}x{M}.
    Clz,
    /// Count leading sign bits (excluding the MSB) — Iop_Cls{N}x{M}.
    Cls,
}

/// Classify an `IROp` into a coarse-grained family for instrumentation
/// (angr-2j5v). Counts roll up into `vex_op_<family>` counters via the
/// `record_vex_*` recorder fns at the entry of the IRExpr dispatch in
/// `interpreter/expressions.rs`.
///
/// `Vec` captures everything prefixed with `V*` (SIMD/NEON). `Fp` captures
/// scalar FP plus the unprefixed FP conversions. `Other` is the catch-all
/// for `Raw` opcode escapes, the `NeonUnimplemented` typed-error sentinel,
/// and the `Unmapped` typed-error sentinel (angr-tkbr.2).
#[inline]
pub fn iropclass(op: &IROp) -> VexOpFamily {
    match op {
        // Integer arithmetic (incl. widening / divmod / mul-hi / neg)
        IROp::Add(_)
        | IROp::Sub(_)
        | IROp::Mul(_)
        | IROp::MullS(_)
        | IROp::MullU(_)
        | IROp::DivS(_)
        | IROp::DivU(_)
        | IROp::ModS(_)
        | IROp::ModU(_)
        | IROp::Neg(_)
        | IROp::DivModU64to32
        | IROp::DivModS64to32
        | IROp::DivModU128to64
        | IROp::DivModS128to64
        | IROp::MulHi { .. } => VexOpFamily::Arith,

        // Bitwise logic
        IROp::And(_) | IROp::Or(_) | IROp::Xor(_) | IROp::Not(_) => VexOpFamily::Logic,

        // Shifts
        IROp::Shl(_) | IROp::Shr(_) | IROp::Sar(_) => VexOpFamily::Shift,

        // Integer comparison
        IROp::CmpEQ(_)
        | IROp::CmpNE(_)
        | IROp::CmpLT(_)
        | IROp::CmpLE(_)
        | IROp::CmpLTU(_)
        | IROp::CmpLEU(_) => VexOpFamily::Cmp,

        // Width adjustment, bit-count, reinterpret, concat/extract
        IROp::SignExtend { .. }
        | IROp::ZeroExtend { .. }
        | IROp::Truncate { .. }
        | IROp::Clz(_)
        | IROp::Ctz(_)
        | IROp::PopCount(_)
        | IROp::Reinterpret { .. }
        | IROp::Concat { .. }
        | IROp::Extract { .. } => VexOpFamily::Ext,

        // Scalar FP (arith + cmp + conversions + rounding)
        IROp::FAdd(_)
        | IROp::FSub(_)
        | IROp::FMul(_)
        | IROp::FDiv(_)
        | IROp::FNeg(_)
        | IROp::FAbs(_)
        | IROp::FSqrt(_)
        | IROp::FMAdd(_)
        | IROp::FMSub(_)
        | IROp::FCmpEQ(_)
        | IROp::FCmpLT(_)
        | IROp::FCmpLE(_)
        | IROp::FCmpScalarLane { .. }
        | IROp::FCmpVecPacked { .. }
        | IROp::FComCC(_)
        | IROp::F32toF64
        | IROp::F64toF32
        | IROp::I32StoF32
        | IROp::I32StoF64
        | IROp::I64StoF32
        | IROp::I64StoF64
        | IROp::I32UtoF32
        | IROp::I32UtoF64
        | IROp::I64UtoF32
        | IROp::I64UtoF64
        | IROp::F32toI32S
        | IROp::F64toI32S
        | IROp::F32toI64S
        | IROp::F64toI64S
        | IROp::F32toI32U
        | IROp::F64toI32U
        | IROp::F32toI64U
        | IROp::F64toI64U
        | IROp::RoundF32toInt
        | IROp::RoundF64toInt => VexOpFamily::Fp,

        // SIMD/NEON — every V*-prefixed variant plus the V128 setters.
        IROp::VFAddS { .. }
        | IROp::VFSubS { .. }
        | IROp::VFMulS { .. }
        | IROp::VFDivS { .. }
        | IROp::VFSqrtS { .. }
        | IROp::VFMaxS { .. }
        | IROp::VFMinS { .. }
        | IROp::SetV128lo32
        | IROp::SetV128lo64
        | IROp::VAdd { .. }
        | IROp::VSub { .. }
        | IROp::VMul { .. }
        | IROp::VMulLo { .. }
        | IROp::VAnd(_)
        | IROp::VOr(_)
        | IROp::VXor(_)
        | IROp::VNot(_)
        | IROp::VShlN { .. }
        | IROp::VShrN { .. }
        | IROp::VSarN { .. }
        | IROp::VShl { .. }
        | IROp::VShr { .. }
        | IROp::VSar { .. }
        | IROp::VCmpEQ { .. }
        | IROp::VCmpGT { .. }
        | IROp::VInterleaveLO { .. }
        | IROp::VInterleaveHI { .. }
        | IROp::VPerm { .. }
        | IROp::VGetElem { .. }
        | IROp::VSetElem { .. }
        | IROp::VDup { .. }
        | IROp::VWiden { .. }
        | IROp::VNarrowUn { .. }
        | IROp::VNarrowBin { .. }
        | IROp::VQNarrowUn { .. }
        | IROp::VQNarrowBin { .. }
        | IROp::VReverse { .. }
        | IROp::VQAdd { .. }
        | IROp::VQSub { .. }
        | IROp::VQShlSat { .. }
        | IROp::VPwAdd { .. }
        | IROp::VPwAddL { .. }
        | IROp::VPwMin { .. }
        | IROp::VPwMax { .. }
        | IROp::VAvg { .. }
        | IROp::VCnt { .. }
        | IROp::VClz { .. }
        | IROp::VCls { .. }
        | IROp::VPolynomialMul { .. }
        | IROp::VMin { .. }
        | IROp::VMax { .. }
        | IROp::VAbs { .. }
        | IROp::VFAdd { .. }
        | IROp::VFSub { .. }
        | IROp::VFMul { .. }
        | IROp::VFDiv { .. }
        | IROp::VFSqrt { .. }
        | IROp::VFAbs { .. }
        | IROp::VFMin { .. }
        | IROp::VFMax { .. }
        | IROp::VFRecipEst { .. }
        | IROp::VFRecipStep { .. }
        | IROp::VFRSqrtEst { .. }
        | IROp::VFRSqrtStep { .. }
        | IROp::VFRecipEstS { .. }
        | IROp::VFRSqrtEstS { .. }
        | IROp::VIRecipEst { .. }
        | IROp::VIRSqrtEst { .. } => VexOpFamily::Vec,

        // x86-specific carry-less multiply / CRC32 — classified as Arith
        // (they're integer ops in the polynomial / checksum sense).
        IROp::PclmulLQLQ
        | IROp::PclmulHQHQ
        | IROp::PclmulLQHQ
        | IROp::PclmulHQLQ
        | IROp::Crc32C => VexOpFamily::Arith,

        // Raw opcode escape + NEON panic sentinel + unmapped-opcode
        // typed-error sentinel (angr-tkbr.2): not pre-classified.
        IROp::NeonUnimplemented(_) | IROp::Unmapped(_) | IROp::Raw(_) => VexOpFamily::Other,
    }
}

impl VEXOps {
    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// Execute a unary operation.
    #[inline]
    pub fn unop(op: IROp, arg: RustBV, ctx: &SymContext) -> Result<RustBV, OpError> {
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

            // SSE scalar-in-vector reciprocal/rsqrt estimate (RCPSS / RSQRTSS).
            // Lane 0 fresh-symbolic, upper lanes pass through.
            IROp::VFRecipEstS { elem } => Self::vec_float_scalar_fresh(arg, elem, "RecipEst", ctx),
            IROp::VFRSqrtEstS { elem } => Self::vec_float_scalar_fresh(arg, elem, "RSqrtEst", ctx),

            // Packed integer absolute value
            IROp::VAbs { elem, count } => Self::vec_int_abs(arg, elem, count, ctx),

            // Packed float sqrt / abs (whole vector)
            IROp::VFSqrt { elem, count } => {
                Self::vec_float_lane_op(&[arg], elem, count, &FSqrt, ctx)
            }
            IROp::VFAbs { elem, count } => Self::vec_float_lane_op(&[arg], elem, count, &FAbs, ctx),

            // Packed FP reciprocal / reciprocal-sqrt estimate. Returns a fresh
            // symbolic per lane: VEX leaves precision implementation-defined and
            // angr Python uses the same conservative pattern (_op_fgeneric_RSqrtEst
            // returns BVS). Mirrors that policy uniformly across RecipEst/RSqrtEst.
            IROp::VFRecipEst { elem, count } => {
                Self::vec_float_fresh_per_lane(elem, count, "RecipEst", ctx)
            }
            IROp::VFRSqrtEst { elem, count } => {
                Self::vec_float_fresh_per_lane(elem, count, "RSqrtEst", ctx)
            }

            // Packed integer reciprocal / reciprocal-sqrt estimate (ARM URECPE
            // / URSQRTE, 32-bit lanes). Fresh-symbolic per lane: claripy has no
            // generic handler for these integer ops, so any precision answer
            // would be more faithful than angr Python and could diverge. See
            // VFRecipEst for the same policy on FP variants.
            IROp::VIRecipEst { count } => Self::vec_int_fresh_per_lane(32, count, "RecipEst", ctx),
            IROp::VIRSqrtEst { count } => Self::vec_int_fresh_per_lane(32, count, "RSqrtEst", ctx),

            // NEON broadcast scalar to vector
            IROp::VDup { elem, count } => Self::vec_dup(arg, elem, count, ctx),

            // NEON widen each lane (sign- or zero-extend)
            IROp::VWiden {
                from,
                count,
                signed,
            } => Self::vec_widen(arg, from, count, signed, ctx),

            // NEON unary narrow (truncating)
            IROp::VNarrowUn { from, count } => Self::vec_narrow_un(arg, from, count, ctx),

            // NEON unary saturating narrow
            IROp::VQNarrowUn {
                from,
                count,
                src_signed,
                dst_signed,
            } => Self::vec_qnarrow_un(arg, from, count, src_signed, dst_signed, ctx),

            // NEON byte/halfword/word/bit reversal within each lane
            IROp::VReverse {
                sub_width,
                elem,
                count,
            } => Self::vec_reverse(arg, sub_width, elem, count, ctx),

            // NEON pairwise widening add (unary). Iop_PwAddL{N}{S/U}x{M}.
            IROp::VPwAddL {
                elem,
                count,
                signed,
            } => Self::vec_pairwise_add_long(arg, elem, count, signed, ctx),

            // NEON per-byte popcount — Iop_Cnt8x{8,16}.
            IROp::VCnt { count } => Self::vec_cnt(arg, count, ctx),

            // NEON per-lane count leading zeros — Iop_Clz{N}x{M}.
            IROp::VClz { elem, count } => {
                Self::vec_lane_count(arg, elem, count, LaneCountKind::Clz, ctx)
            }

            // NEON per-lane count leading sign bits — Iop_Cls{N}x{M}.
            IROp::VCls { elem, count } => {
                Self::vec_lane_count(arg, elem, count, LaneCountKind::Cls, ctx)
            }

            // NEON scaffolding: surface as a typed error rather than silently
            // falling back. interpreter::expressions special-cases
            // `UnsupportedNeon` to skip the fresh-symbolic synthesizer.
            // Implementations land in angr-bkcs.2.
            IROp::NeonUnimplemented(name) => Err(OpError::UnsupportedNeon { name }),

            // Unmapped opcode — captured at parse time, surfaces here so the
            // engine maps it to RustUnsupportedVexOpError (angr-tkbr.2).
            IROp::Unmapped(name) => Err(OpError::UnsupportedVexOp {
                op_name: name.to_string(),
            }),

            _ => Err(OpError::NotUnary(op)),
        }
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// Execute a binary operation.
    ///
    /// Top-level dispatch routes each `IROp` variant to a per-family helper
    /// (`binop_arith`, `binop_bitwise_shift_cmp`, `binop_float`,
    /// `binop_vec_int`, `binop_vec_float`, `binop_misc`). Each helper is
    /// exhaustive over its assigned arms; unguarded variants fall through
    /// to `binop_misc` which returns `OpError::NotBinary` for the default
    /// case along with the typed-error sentinels (`NeonUnimplemented`,
    /// `Unmapped`, `Raw`, `SetV128lo*`, `FSqrt`).
    #[inline]
    pub fn binop(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match &op {
            // -- Scalar arithmetic (incl widening, mul-hi, divmod) --
            IROp::Add(_)
            | IROp::Sub(_)
            | IROp::Mul(_)
            | IROp::DivU(_)
            | IROp::DivS(_)
            | IROp::ModU(_)
            | IROp::ModS(_)
            | IROp::MullU(_)
            | IROp::MullS(_)
            | IROp::MulHi { .. }
            | IROp::DivModU64to32
            | IROp::DivModS64to32
            | IROp::DivModU128to64
            | IROp::DivModS128to64 => Self::binop_arith(op, left, right, ctx),

            // -- Scalar bitwise / shift / compare --
            IROp::And(_)
            | IROp::Or(_)
            | IROp::Xor(_)
            | IROp::Shl(_)
            | IROp::Shr(_)
            | IROp::Sar(_)
            | IROp::CmpEQ(_)
            | IROp::CmpNE(_)
            | IROp::CmpLT(_)
            | IROp::CmpLE(_)
            | IROp::CmpLTU(_)
            | IROp::CmpLEU(_) => Self::binop_bitwise_shift_cmp(op, left, right, ctx),

            // -- Scalar floating point (arith, cmp, com, rounding, conversions w/ rm) --
            IROp::FAdd(_)
            | IROp::FSub(_)
            | IROp::FMul(_)
            | IROp::FDiv(_)
            | IROp::FCmpEQ(_)
            | IROp::FCmpLT(_)
            | IROp::FCmpLE(_)
            | IROp::FComCC(_)
            | IROp::RoundF32toInt
            | IROp::RoundF64toInt
            | IROp::F64toF32
            | IROp::F32toI32S
            | IROp::F64toI32S
            | IROp::F32toI64S
            | IROp::F64toI64S
            | IROp::F32toI32U
            | IROp::F64toI32U
            | IROp::F32toI64U
            | IROp::F64toI64U => Self::binop_float(op, left, right, ctx),

            // -- Vector integer (packed integer ops, narrows, interleaves, shifts, bitwise) --
            IROp::VAnd(_)
            | IROp::VOr(_)
            | IROp::VXor(_)
            | IROp::Concat { .. }
            | IROp::VAdd { .. }
            | IROp::VSub { .. }
            | IROp::VMul { .. }
            | IROp::VMulLo { .. }
            | IROp::VQAdd { .. }
            | IROp::VQSub { .. }
            | IROp::VQShlSat { .. }
            | IROp::VPwAdd { .. }
            | IROp::VPwMin { .. }
            | IROp::VPwMax { .. }
            | IROp::VAvg { .. }
            | IROp::VPolynomialMul { .. }
            | IROp::VCmpEQ { .. }
            | IROp::VCmpGT { .. }
            | IROp::VGetElem { .. }
            | IROp::VNarrowBin { .. }
            | IROp::VQNarrowBin { .. }
            | IROp::VInterleaveLO { .. }
            | IROp::VInterleaveHI { .. }
            | IROp::VShlN { .. }
            | IROp::VShrN { .. }
            | IROp::VSarN { .. }
            | IROp::VShl { .. }
            | IROp::VShr { .. }
            | IROp::VSar { .. }
            | IROp::VMin { .. }
            | IROp::VMax { .. } => Self::binop_vec_int(op, left, right, ctx),

            // -- Vector floating point (packed FP + scalar-in-vector FP) --
            IROp::FCmpScalarLane { .. }
            | IROp::FCmpVecPacked { .. }
            | IROp::VFAddS { .. }
            | IROp::VFSubS { .. }
            | IROp::VFMulS { .. }
            | IROp::VFDivS { .. }
            | IROp::VFAdd { .. }
            | IROp::VFSub { .. }
            | IROp::VFMul { .. }
            | IROp::VFDiv { .. }
            | IROp::VFMin { .. }
            | IROp::VFMax { .. }
            | IROp::VFRecipStep { .. }
            | IROp::VFRSqrtStep { .. }
            | IROp::VFMaxS { .. }
            | IROp::VFMinS { .. } => Self::binop_vec_float(op, left, right, ctx),

            // -- Misc: Raw transcendentals, SetV128lo*, FSqrt(rm), NEON
            // sentinels, unmapped, default NotBinary --
            _ => Self::binop_misc(op, left, right, ctx),
        }
    }

    /// Scalar integer arithmetic (same-width add/sub/mul/div/mod, widening
    /// multiply, high-half multiply, packed divmod). All arms are guarded
    /// by the top-level `binop()` dispatch.
    fn binop_arith(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            IROp::Add(ty) => width_binop!(left, right, ty, add_into, ctx),
            IROp::Sub(ty) => width_binop!(left, right, ty, sub_into, ctx),
            IROp::Mul(ty) => width_binop!(left, right, ty, mul_into, ctx),
            IROp::DivU(ty) => width_binop!(left, right, ty, udiv_into, ctx),
            IROp::DivS(ty) => width_binop!(left, right, ty, sdiv_into, ctx),
            IROp::ModU(ty) => width_binop!(left, right, ty, urem_into, ctx),
            IROp::ModS(ty) => width_binop!(left, right, ty, srem_into, ctx),

            // Widening multiply (result width is 2 * ty.bits()).
            IROp::MullU(ty) => Self::widening_mul(left, right, ty, false, ctx),
            IROp::MullS(ty) => Self::widening_mul(left, right, ty, true, ctx),

            // High half of multiplication.
            IROp::MulHi { ty, signed } => Self::mul_hi(left, right, ty, signed, ctx),

            // DivMod: 64-bit / 32-bit -> 64-bit (low=quotient, high=remainder).
            IROp::DivModU64to32 => Self::divmod_64_to_32(left, right, false, ctx),
            IROp::DivModS64to32 => Self::divmod_64_to_32(left, right, true, ctx),

            // DivMod: 128-bit / 64-bit -> 128-bit (low=quotient, high=remainder).
            IROp::DivModU128to64 => Self::divmod_128_to_64(left, right, false, ctx),
            IROp::DivModS128to64 => Self::divmod_128_to_64(left, right, true, ctx),

            _ => unreachable!("binop_arith called with non-arith op: {op:?}"),
        }
    }

    /// Scalar bitwise (and/or/xor), shifts (shl/shr/sar with shift-amount
    /// width normalization), and integer comparisons. All arms are guarded
    /// by the top-level `binop()` dispatch.
    fn binop_bitwise_shift_cmp(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // Bitwise.
            IROp::And(ty) => width_binop!(left, right, ty, and_into, ctx),
            IROp::Or(ty) => width_binop!(left, right, ty, or_into, ctx),
            IROp::Xor(ty) => width_binop!(left, right, ty, xor_into, ctx),

            // Shifts — normalize shift amount width to match operand.
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

            // Integer comparisons (signed and unsigned).
            IROp::CmpEQ(ty) => width_binop!(left, right, ty, eq_into, ctx),
            IROp::CmpNE(ty) => width_binop!(left, right, ty, ne_into, ctx),
            IROp::CmpLT(ty) => width_binop!(left, right, ty, slt_into, ctx),
            IROp::CmpLE(ty) => width_binop!(left, right, ty, sle_into, ctx),
            IROp::CmpLTU(ty) => width_binop!(left, right, ty, ult_into, ctx),
            IROp::CmpLEU(ty) => width_binop!(left, right, ty, ule_into, ctx),

            _ => {
                unreachable!("binop_bitwise_shift_cmp called with non-bitwise/shift/cmp op: {op:?}")
            }
        }
    }

    /// Scalar floating point: arithmetic, comparisons, comparison-with-CC,
    /// rounding, and the float-conversion-with-rm family. All arms are
    /// guarded by the top-level `binop()` dispatch.
    fn binop_float(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // FP arithmetic.
            IROp::FAdd(ty) => Self::float_add(left, right, ty, ctx),
            IROp::FSub(ty) => Self::float_sub(left, right, ty, ctx),
            IROp::FMul(ty) => Self::float_mul(left, right, ty, ctx),
            IROp::FDiv(ty) => Self::float_div(left, right, ty, ctx),

            // FP comparisons.
            IROp::FCmpEQ(ty) => Self::float_cmp_eq(left, right, ty, ctx),
            IROp::FCmpLT(ty) => Self::float_cmp_lt(left, right, ty, ctx),
            IROp::FCmpLE(ty) => Self::float_cmp_le(left, right, ty, ctx),
            IROp::FComCC(ty) => Self::float_com_cc(left, right, ty, ctx),

            // Float rounding with mode (left = rounding mode, right = value).
            IROp::RoundF32toInt => Self::round_f32_to_int_with_mode(left, right, ctx),
            IROp::RoundF64toInt => Self::round_f64_to_int_with_mode(left, right, ctx),

            // Float conversions that take a rounding mode as the first arg.
            // VEX rounding modes: 0=nearest, 1=down, 2=up, 3=zero (truncate).
            IROp::F64toF32 => Self::f64_to_f32_rm(left, right, ctx),
            IROp::F32toI32S => Self::f32_to_i32s_rm(left, right, ctx),
            IROp::F64toI32S => Self::f64_to_i32s_rm(left, right, ctx),
            IROp::F32toI64S => Self::f32_to_i64s_rm(left, right, ctx),
            IROp::F64toI64S => Self::f64_to_i64s_rm(left, right, ctx),
            IROp::F32toI32U => Self::f32_to_i32u_rm(left, right, ctx),
            IROp::F64toI32U => Self::f64_to_i32u_rm(left, right, ctx),
            IROp::F32toI64U => Self::f32_to_i64u_rm(left, right, ctx),
            IROp::F64toI64U => Self::f64_to_i64u_rm(left, right, ctx),

            _ => unreachable!("binop_float called with non-float op: {op:?}"),
        }
    }

    /// Packed integer ops: arithmetic, saturating, pairwise, shifts (by
    /// immediate and by vector), min/max, narrowing, interleave, GF(2)
    /// polynomial multiply, lane extract, plus vector bitwise (VAnd/Or/Xor)
    /// and the scalar Concat (lumped here because it shares the structural
    /// "produce wider vector from two narrower" shape). All arms are guarded
    /// by the top-level `binop()` dispatch.
    fn binop_vec_int(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // Vector bitwise (whole-vector and/or/xor — same-width).
            IROp::VAnd(ty) => width_binop!(left, right, ty, and_into, ctx),
            IROp::VOr(ty) => width_binop!(left, right, ty, or_into, ctx),
            IROp::VXor(ty) => width_binop!(left, right, ty, xor_into, ctx),

            // Concatenate (scalar; ty is the result width).
            IROp::Concat { ty } => {
                let result = left.concat_into(right, ctx);
                debug_assert_eq!(result.width(), ty.bits());
                Ok(result)
            }

            // Vector arithmetic (element-wise).
            IROp::VAdd { elem, count } => Self::vec_binop(left, right, elem, count, "add", ctx),
            IROp::VSub { elem, count } => Self::vec_binop(left, right, elem, count, "sub", ctx),
            IROp::VMul { elem, count } => Self::vec_binop(left, right, elem, count, "mul", ctx),
            IROp::VMulLo { elem, count } => Self::vec_mul_lo(left, right, elem, count, ctx),

            // NEON saturating integer add/sub.
            IROp::VQAdd {
                elem,
                count,
                signed,
            } => Self::vec_int_saturating(
                left, right, elem, count, signed, /*is_sub=*/ false, ctx,
            ),
            IROp::VQSub {
                elem,
                count,
                signed,
            } => Self::vec_int_saturating(
                left, right, elem, count, signed, /*is_sub=*/ true, ctx,
            ),

            // NEON saturating shift-left by vector (Iop_QShl* / Iop_QSal*).
            IROp::VQShlSat {
                elem,
                count,
                signed,
            } => Self::vec_qshl_sat(left, right, elem, count, signed, ctx),

            // NEON pairwise integer add/min/max (binary).
            IROp::VPwAdd { elem, count } => {
                Self::vec_pairwise_binop(left, right, elem, count, PwOp::Add, ctx)
            }
            IROp::VPwMin {
                elem,
                count,
                signed,
            } => Self::vec_pairwise_binop(
                left,
                right,
                elem,
                count,
                if signed { PwOp::MinS } else { PwOp::MinU },
                ctx,
            ),
            IROp::VPwMax {
                elem,
                count,
                signed,
            } => Self::vec_pairwise_binop(
                left,
                right,
                elem,
                count,
                if signed { PwOp::MaxS } else { PwOp::MaxU },
                ctx,
            ),

            // NEON rounding halving add (a.k.a. rounding-average).
            IROp::VAvg {
                elem,
                count,
                signed,
            } => Self::vec_rounding_avg(left, right, elem, count, signed, ctx),

            // NEON GF(2) polynomial multiply (PMUL / PMULL).
            IROp::VPolynomialMul { count, widen } => {
                Self::vec_polynomial_mul(left, right, count, widen, ctx)
            }

            // Vector compare operations.
            IROp::VCmpEQ { elem, count } => Self::vec_cmp(left, right, elem, count, "eq", ctx),
            IROp::VCmpGT { elem, count } => Self::vec_cmp(left, right, elem, count, "gt", ctx),

            // NEON lane extract (Iop_GetElem{N}x{M}): (vec, idx) -> lane.
            IROp::VGetElem { elem, count } => Self::vec_get_elem(left, right, elem, count, ctx),

            // NEON binary narrow (truncating).
            IROp::VNarrowBin { from, count } => Self::vec_narrow_bin(left, right, from, count, ctx),

            // NEON binary saturating narrow.
            IROp::VQNarrowBin {
                from,
                count,
                src_signed,
                dst_signed,
            } => Self::vec_qnarrow_bin(left, right, from, count, src_signed, dst_signed, ctx),

            // Vector interleave.
            IROp::VInterleaveLO { elem } => Self::vec_interleave_lo(left, right, elem, ctx),
            IROp::VInterleaveHI { elem } => Self::vec_interleave_hi(left, right, elem, ctx),

            // Vector shifts by immediate.
            IROp::VShlN { elem, count } => Self::vec_shl_n(left, right, elem, count, ctx),
            IROp::VShrN { elem, count } => Self::vec_shr_n(left, right, elem, count, ctx),
            IROp::VSarN { elem, count } => Self::vec_sar_n(left, right, elem, count, ctx),

            // Vector shifts by vector (per-lane shift count).
            IROp::VShl { elem, count } => {
                Self::vec_shift_vec(left, right, elem, count, VecShiftKind::Shl, ctx)
            }
            IROp::VShr { elem, count } => {
                Self::vec_shift_vec(left, right, elem, count, VecShiftKind::Shr, ctx)
            }
            IROp::VSar { elem, count } => {
                Self::vec_shift_vec(left, right, elem, count, VecShiftKind::Sar, ctx)
            }

            // Packed integer min/max.
            IROp::VMin {
                elem,
                count,
                signed,
            } => Self::vec_int_minmax(
                left, right, elem, count, signed, /*is_max=*/ false, ctx,
            ),
            IROp::VMax {
                elem,
                count,
                signed,
            } => Self::vec_int_minmax(left, right, elem, count, signed, /*is_max=*/ true, ctx),

            _ => unreachable!("binop_vec_int called with non-vec-int op: {op:?}"),
        }
    }

    /// Packed floating point: arithmetic, min/max, comparison helpers
    /// (scalar-lane and packed), scalar-in-vector arith and min/max, and
    /// Newton-Raphson reciprocal/rsqrt-step (fresh symbolic per lane).
    /// All arms are guarded by the top-level `binop()` dispatch.
    fn binop_vec_float(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // Scalar-lane / packed FP comparison helpers.
            IROp::FCmpScalarLane { kind, ty } => {
                Self::vec_float_scalar_lane_cmp(left, right, kind, ty, ctx)
            }
            IROp::FCmpVecPacked { kind, elem, count } => {
                Self::vec_float_packed_cmp(left, right, kind, elem, count, ctx)
            }

            // Scalar-in-vector float arithmetic (SSE scalar ops).
            IROp::VFAddS { elem } => {
                Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Add, ctx)
            }
            IROp::VFSubS { elem } => {
                Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Sub, ctx)
            }
            IROp::VFMulS { elem } => {
                Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Mul, ctx)
            }
            IROp::VFDivS { elem } => {
                Self::vec_float_scalar_op(left, right, elem, FloatOpKind::Div, ctx)
            }

            // Packed FP arithmetic.
            IROp::VFAdd { elem, count } => {
                Self::vec_float_lane_op(&[left, right], elem, count, &FAdd, ctx)
            }
            IROp::VFSub { elem, count } => {
                Self::vec_float_lane_op(&[left, right], elem, count, &FSub, ctx)
            }
            IROp::VFMul { elem, count } => {
                Self::vec_float_lane_op(&[left, right], elem, count, &FMul, ctx)
            }
            IROp::VFDiv { elem, count } => {
                Self::vec_float_lane_op(&[left, right], elem, count, &FDiv, ctx)
            }

            // Packed FP min/max.
            IROp::VFMin { elem, count } => {
                Self::vec_float_lane_op(&[left, right], elem, count, &FMin, ctx)
            }
            IROp::VFMax { elem, count } => {
                Self::vec_float_lane_op(&[left, right], elem, count, &FMax, ctx)
            }

            // NEON Newton-Raphson reciprocal / rsqrt step. Operands consumed
            // but the result is a fresh symbolic per lane (matches angr Python's
            // conservative handling — no `_op_fgeneric_RecipStep` /
            // `_op_fgeneric_RSqrtStep`). Refinement loops typically follow
            // with additional steps that converge regardless of the seed.
            IROp::VFRecipStep { elem, count } => {
                let _ = (left, right);
                Self::vec_float_fresh_per_lane(elem, count, "RecipStep", ctx)
            }
            IROp::VFRSqrtStep { elem, count } => {
                let _ = (left, right);
                Self::vec_float_fresh_per_lane(elem, count, "RSqrtStep", ctx)
            }

            // Scalar-in-vector max/min.
            IROp::VFMaxS { elem } => {
                Self::vec_float_scalar_minmax(left, right, elem, /*is_max=*/ true, ctx)
            }
            IROp::VFMinS { elem } => {
                Self::vec_float_scalar_minmax(left, right, elem, /*is_max=*/ false, ctx)
            }

            _ => unreachable!("binop_vec_float called with non-vec-float op: {op:?}"),
        }
    }

    /// Miscellaneous binops that don't fit a single family: x87
    /// transcendental fast path via `IROp::Raw`, SetV128lo32/64, the
    /// `IROp::FSqrt(rm, value)` arrival pattern, and the typed-error
    /// sentinels (NeonUnimplemented, Unmapped). Also serves as the
    /// dispatch fallback returning `OpError::NotBinary` for unmatched ops.
    fn binop_misc(
        op: IROp,
        left: RustBV,
        right: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match op {
            // Raw opcode — try concrete x87 transcendental fast path first
            // (Iop_SinF64, Iop_CosF64, Iop_TanF64, Iop_2xm1F64, Iop_RecpExp*).
            // These arrive as Binop(rm, x); `left` carries rm, `right` the value.
            // For symbolic inputs to Iop_{Sin,Cos,Tan,2xm1}F64, try the
            // concretize-and-pin fallback (angr-i5lj.1, angr-i5lj.2).
            // Iop_RecpExp* has a closed-form path and never needs concretize.
            IROp::Raw(code) => {
                if let Some(result) = transcendentals::try_concrete_binop_rm(code, &left, &right) {
                    Ok(result)
                } else if let Some(result) =
                    transcendentals::try_concretize_binop_rm(code, &left, &right, ctx)
                {
                    Ok(result)
                } else {
                    Err(OpError::RawOpcode(code))
                }
            }

            // SetV128lo: set low bits of V128.
            IROp::SetV128lo32 => Self::set_v128_lo32(left, right, ctx),
            IROp::SetV128lo64 => Self::set_v128_lo64(left, right, ctx),

            // Iop_SqrtF{32,64} is a VEX Binop (arg1=rm, arg2=value) that we
            // model as IROp::FSqrt — keep the translation here so the
            // interpreter's Binop dispatch routes the rm through
            // unop_with_rm instead of falling back to a fresh symbolic.
            IROp::FSqrt(_) => Self::unop_with_rm(op, left, right, ctx),

            // NEON scaffolding: surface as a typed error rather than silently
            // falling back. interpreter::expressions special-cases
            // `UnsupportedNeon` to skip the fresh-symbolic synthesizer.
            IROp::NeonUnimplemented(name) => Err(OpError::UnsupportedNeon { name }),

            // Unmapped opcode (angr-tkbr.2).
            IROp::Unmapped(name) => Err(OpError::UnsupportedVexOp {
                op_name: name.to_string(),
            }),

            other => Err(OpError::NotBinary(other)),
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
            // NEON scaffolding: surface as a typed error rather than silently
            // falling back. interpreter::expressions special-cases
            // `UnsupportedNeon` to skip the fresh-symbolic synthesizer.
            IROp::NeonUnimplemented(name) => Err(OpError::UnsupportedNeon { name }),

            // Unmapped opcode (angr-tkbr.2).
            IROp::Unmapped(name) => Err(OpError::UnsupportedVexOp {
                op_name: name.to_string(),
            }),
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
            // NEON scaffolding: surface as a typed error rather than silently
            // falling back. interpreter::expressions special-cases
            // `UnsupportedNeon` to skip the fresh-symbolic synthesizer.
            IROp::NeonUnimplemented(name) => Err(OpError::UnsupportedNeon { name }),

            // Unmapped opcode (angr-tkbr.2).
            IROp::Unmapped(name) => Err(OpError::UnsupportedVexOp {
                op_name: name.to_string(),
            }),
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
                // Iop_Yl2xp1F64, Iop_ScaleF64. Try concrete libm path first;
                // for symbolic inputs, fall back to the concretize-and-pin
                // path (angr-i5lj.1, angr-i5lj.2). Out-of-scope triops fall
                // through to the fresh-symbolic fallback in
                // expressions.rs::IRExpr::Triop.
                if let Some(result) =
                    transcendentals::try_concrete_triop_rm(code, &rm, &left, &right)
                {
                    return Ok(result);
                }
                if let Some(result) =
                    transcendentals::try_concretize_triop_rm(code, &rm, &left, &right, ctx)
                {
                    return Ok(result);
                }
                return Self::binop(op, left, right, ctx);
            }
            _ => return Self::binop(op, left, right, ctx),
        };

        // RNE concrete: keep the native-f{32,64} fast path. Most code uses
        // RNE; routing through Z3 here would be a measurable regression on
        // FP-heavy benchmarks.
        if let Some(m) = rm.as_u128()
            && m & 0x3 == 0
        {
            return Self::binop(op, left, right, ctx);
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

        if let Some(m) = rm.as_u128()
            && m & 0x3 == 0
        {
            return Self::unop(op, arg, ctx);
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
            let mask = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
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
            let mask_elem = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
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

    /// NEON broadcast scalar to vector (Iop_Dup{N}x{M}). Replicates `arg` into
    /// `count` lanes of width `elem.bits()`.
    fn vec_dup(arg: RustBV, elem: IRType, count: u8, ctx: &SymContext) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(arg.width(), elem_width);

        // Concrete fast path.
        if total_width <= 128
            && let Some(v) = arg.as_u128()
        {
            let mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let lane = v & mask;
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * elem_width;
                result |= lane << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic: concat the same value `count` times.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            elements.push(arg.clone());
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON widen each lane (Iop_Widen{N}{S/U}to{2N}x{M}). Sign- or
    /// zero-extends each lane from `from.bits()` to `from.bits()*2`.
    fn vec_widen(
        arg: RustBV,
        from: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width * 2;
        let in_total = from_width * count as u32;
        debug_assert_eq!(arg.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && (to_width * count as u32) <= 128
            && let Some(v) = arg.as_u128()
        {
            let in_mask: u128 = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let out_mask: u128 = if to_width == 128 {
                u128::MAX
            } else {
                (1u128 << to_width) - 1
            };
            let sign_bit: u128 = 1u128 << (from_width - 1);
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * from_width;
                let lane = (v >> lo) & in_mask;
                let widened = if signed && (lane & sign_bit != 0) {
                    // Sign-extend: fill upper (to_width - from_width) bits with 1s.
                    (lane | !in_mask) & out_mask
                } else {
                    lane
                };
                let out_lo = (i as u32) * to_width;
                result |= widened << out_lo;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: extract each lane, extend, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * from_width;
            let hi = lo + from_width - 1;
            let lane = arg.extract(hi, lo, ctx);
            let widened = if signed {
                lane.sign_extend_into(to_width, ctx)
            } else {
                lane.zero_extend_into(to_width, ctx)
            };
            elements.push(widened);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON unary narrow (Iop_NarrowUn{N}to{N/2}x{M}). Truncates each lane
    /// from `from.bits()` to `from.bits()/2`.
    fn vec_narrow_un(
        arg: RustBV,
        from: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width / 2;
        let in_total = from_width * count as u32;
        debug_assert_eq!(arg.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && let Some(v) = arg.as_u128()
        {
            let to_mask: u128 = (1u128 << to_width) - 1;
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * from_width;
                let lane = (v >> lo) & to_mask;
                let out_lo = (i as u32) * to_width;
                result |= lane << out_lo;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: extract each low half-lane, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * from_width;
            let hi = lo + to_width - 1;
            elements.push(arg.extract(hi, lo, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON binary narrow (Iop_NarrowBin{N}to{N/2}x{M}). Each input has
    /// `count/2` lanes of width `from`; result has `count` lanes of width
    /// `from/2` with `left` providing the low half and `right` the high half.
    fn vec_narrow_bin(
        left: RustBV,
        right: RustBV,
        from: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width / 2;
        let per_input = (count / 2) as u32;
        let in_total = from_width * per_input;
        debug_assert_eq!(left.width(), in_total);
        debug_assert_eq!(right.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && (to_width * count as u32) <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let to_mask: u128 = (1u128 << to_width) - 1;
            let mut result: u128 = 0;
            for i in 0..per_input {
                let lo = i * from_width;
                let lane_l = (l >> lo) & to_mask;
                let lane_r = (r >> lo) & to_mask;
                let out_lo_l = i * to_width;
                let out_lo_r = (per_input + i) * to_width;
                result |= lane_l << out_lo_l;
                result |= lane_r << out_lo_r;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: extract each low half-lane from both operands, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + to_width - 1;
            elements.push(left.extract(hi, lo, ctx));
        }
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + to_width - 1;
            elements.push(right.extract(hi, lo, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Saturate one concrete `from_width`-bit lane into a `to_width`-bit
    /// result. `src_signed` interprets the input; `dst_signed` selects the
    /// output range. Returns the truncated unsigned bit pattern.
    #[inline]
    fn saturate_lane(
        lane: u128,
        from_width: u32,
        to_width: u32,
        src_signed: bool,
        dst_signed: bool,
    ) -> u128 {
        let to_mask: u128 = (1u128 << to_width) - 1;
        // Reinterpret the source lane as i128.
        let val_i: i128 = if src_signed {
            Self::sign_extend_low_to_i128(lane, from_width)
        } else {
            // unsigned source — masking width-bits into i128 keeps it
            // non-negative because from_width <= 64 in all NEON QNarrow ops.
            (lane
                & if from_width == 128 {
                    u128::MAX
                } else {
                    (1u128 << from_width) - 1
                }) as i128
        };
        let (min_i, max_i): (i128, i128) = if dst_signed {
            let half = 1i128 << (to_width - 1);
            (-half, half - 1)
        } else {
            (0i128, ((1u128 << to_width) - 1) as i128)
        };
        let clamped = val_i.clamp(min_i, max_i);
        (clamped as u128) & to_mask
    }

    /// NEON byte/halfword/word/bit reversal within each lane —
    /// `Iop_Reverse{sub_width}sIn{elem.bits()}_x{count}`. Reverses the
    /// `elem.bits() / sub_width` sub-units of width `sub_width` inside each
    /// `elem`-wide lane (`count` lanes total). Total width is preserved at
    /// `elem.bits() * count`.
    ///
    /// Encodes the ARM AArch64 REV*/RBIT semantics that pyvex lifts from
    /// VRBIT (sub_width=1), VREV16 (8-in-16), VREV32 (8/16-in-32), and
    /// VREV64 (8/16/32-in-64) — see ARM DDI 0487 C7.2.297-300 (REV*) and
    /// C7.2.288 (RBIT). Matches angr Python's only explicit reference,
    /// `_op_Iop_Reverse32sIn64_x2` in
    /// `angr/engines/vex/claripy/irop.py:599`, generalised to the full
    /// family of `Iop_Reverse{n}sIn{m}_x{k}` opcodes that the Python engine
    /// otherwise marks unsupported.
    fn vec_reverse(
        arg: RustBV,
        sub_width: u8,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let sub_width_u32 = sub_width as u32;
        let total = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total);
        debug_assert!(sub_width_u32 > 0 && sub_width_u32 <= elem_width);
        debug_assert_eq!(elem_width % sub_width_u32, 0);
        let sub_per_elem = elem_width / sub_width_u32;

        // Concrete fast path: shuffle bits within each lane using integer ops.
        if total <= 128
            && let Some(v) = arg.as_u128()
        {
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sub_mask: u128 = (1u128 << sub_width_u32) - 1;
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let lane_lo = i * elem_width;
                let lane = (v >> lane_lo) & elem_mask;
                let mut reversed_lane: u128 = 0;
                for j in 0..sub_per_elem {
                    let src_lo = j * sub_width_u32;
                    let dst_lo = (sub_per_elem - 1 - j) * sub_width_u32;
                    let sub = (lane >> src_lo) & sub_mask;
                    reversed_lane |= sub << dst_lo;
                }
                result |= reversed_lane << lane_lo;
            }
            return Ok(RustBV::concrete(result, total));
        }

        // Symbolic: extract each sub-unit, place at the mirrored position
        // inside its lane, concat back together.
        let n_subs = (count as u32) * sub_per_elem;
        let mut elements: Vec<RustBV> = Vec::with_capacity(n_subs as usize);
        // concat_le_elements puts elements[0] at LSB, elements[n-1] at MSB.
        // Walk output sub-units from LSB to MSB; within each lane, output
        // sub-unit `j` pulls from input sub-unit `sub_per_elem - 1 - j`.
        for lane in 0..count as u32 {
            let lane_lo = lane * elem_width;
            for j in 0..sub_per_elem {
                let src_lo = lane_lo + (sub_per_elem - 1 - j) * sub_width_u32;
                let src_hi = src_lo + sub_width_u32 - 1;
                elements.push(arg.extract(src_hi, src_lo, ctx));
            }
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON unary saturating narrow (Iop_QNarrowUn{N}{S/U}to{N/2}{S/U}x{M}).
    fn vec_qnarrow_un(
        arg: RustBV,
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width / 2;
        let in_total = from_width * count as u32;
        debug_assert_eq!(arg.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && let Some(v) = arg.as_u128()
        {
            let in_mask: u128 = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * from_width;
                let lane = (v >> lo) & in_mask;
                let sat = Self::saturate_lane(lane, from_width, to_width, src_signed, dst_signed);
                let out_lo = (i as u32) * to_width;
                result |= sat << out_lo;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: per-lane ITE clamp.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * from_width;
            let hi = lo + from_width - 1;
            let lane = arg.extract(hi, lo, ctx);
            elements.push(Self::saturate_lane_symbolic(
                lane, from_width, to_width, src_signed, dst_signed, ctx,
            ));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON binary saturating narrow (Iop_QNarrowBin{N}{S/U}to{N/2}{S/U}x{M}).
    fn vec_qnarrow_bin(
        left: RustBV,
        right: RustBV,
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width / 2;
        let per_input = (count / 2) as u32;
        let in_total = from_width * per_input;
        debug_assert_eq!(left.width(), in_total);
        debug_assert_eq!(right.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && (to_width * count as u32) <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let in_mask: u128 = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let mut result: u128 = 0;
            for i in 0..per_input {
                let lo = i * from_width;
                let lane_l = (l >> lo) & in_mask;
                let lane_r = (r >> lo) & in_mask;
                let sat_l =
                    Self::saturate_lane(lane_l, from_width, to_width, src_signed, dst_signed);
                let sat_r =
                    Self::saturate_lane(lane_r, from_width, to_width, src_signed, dst_signed);
                let out_lo_l = i * to_width;
                let out_lo_r = (per_input + i) * to_width;
                result |= sat_l << out_lo_l;
                result |= sat_r << out_lo_r;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: per-lane clamp from each operand, concatenate.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + from_width - 1;
            let lane = left.extract(hi, lo, ctx);
            elements.push(Self::saturate_lane_symbolic(
                lane, from_width, to_width, src_signed, dst_signed, ctx,
            ));
        }
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + from_width - 1;
            let lane = right.extract(hi, lo, ctx);
            elements.push(Self::saturate_lane_symbolic(
                lane, from_width, to_width, src_signed, dst_signed, ctx,
            ));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Symbolic clamp of one `from_width`-bit lane into `to_width` bits.
    /// Builds an ITE: `if lane > max -> max; else if lane < min -> min; else lane[low to_width]`.
    fn saturate_lane_symbolic(
        lane: RustBV,
        from_width: u32,
        to_width: u32,
        src_signed: bool,
        dst_signed: bool,
        ctx: &SymContext,
    ) -> RustBV {
        debug_assert_eq!(lane.width(), from_width);

        // Build the saturation range constants in `from_width` bits so the
        // comparison ops have matching widths.
        let (max_val, min_val): (u128, u128) = if dst_signed {
            let half = 1u128 << (to_width - 1);
            // max = 2^(to_width-1) - 1, min = -2^(to_width-1)
            // In `from_width` bits (two's complement): min = (-half) & mask
            let from_mask = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let max = half - 1;
            let min = (!(half - 1) + 1) & from_mask; // -half in from_width bits
            (max, min)
        } else {
            // Unsigned dst: [0, 2^to_width - 1]
            let max = (1u128 << to_width) - 1;
            (max, 0)
        };

        let max_bv = RustBV::concrete(max_val, from_width);
        let min_bv = RustBV::concrete(min_val, from_width);

        // gt_max: compare with src_signed semantics
        let gt_max = if src_signed {
            lane.sgt(&max_bv, ctx)
        } else {
            lane.ugt(&max_bv, ctx)
        };

        // lt_min: only meaningful when min could be < lane. If src is unsigned
        // and dst is unsigned, min == 0 so lt_min is always false; skip to
        // truncate the upper-clamped value.
        let truncated = lane.extract(to_width - 1, 0, ctx);
        let max_truncated = max_bv.extract(to_width - 1, 0, ctx);

        let upper_clamped = gt_max.ite(&max_truncated, &truncated, ctx);

        if !src_signed && !dst_signed {
            return upper_clamped;
        }

        let lt_min = if src_signed {
            lane.slt(&min_bv, ctx)
        } else {
            lane.ult(&min_bv, ctx)
        };
        let min_truncated = min_bv.extract(to_width - 1, 0, ctx);

        lt_min.ite(&min_truncated, &upper_clamped, ctx)
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
                let src_lo = i * elem_width;
                let dst_lo = i * 2 * elem_width;

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
            let src_lo = i * elem_width;
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
                let src_lo = (half_count + i) * elem_width;
                let dst_lo = i * 2 * elem_width;

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
            let src_lo = (half_count + i) * elem_width;
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
                            elem_mask // All 1s
                        } else {
                            0 // All 0s
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

    /// NEON vector shift by *vector*: lane `i` of result = `lane_a[i] OP lane_b[i]`
    /// where `OP` is logical-left / logical-right / arithmetic-right per `kind`.
    /// Maps to `Iop_Shl{N}x{M}` / `Iop_Sal{N}x{M}` (left), `Iop_Shr{N}x{M}` (lshr),
    /// and `Iop_Sar{N}x{M}` (ashr); see ARM USHL/SSHL (DDI 0487 C7.2.310, C7.2.291).
    /// Z3 `bvshl`/`bvlshr`/`bvashr` semantics handle out-of-range counts the
    /// same way the concrete fast path does (≥ lane width → 0 for shl/lshr,
    /// sign-fill for ashr).
    fn vec_shift_vec(
        vec: RustBV,
        amts: RustBV,
        elem: IRType,
        count: u8,
        kind: VecShiftKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(amts.width(), total_width);

        // Concrete fast path: both operands fit in u128 (covers every NEON
        // shape we route here — 64- and 128-bit vectors).
        if total_width <= 128
            && let (Some(v), Some(s)) = (vec.as_u128(), amts.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sign_bit: u128 = 1u128 << (elem_width - 1);

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let a = (v >> lo) & elem_mask;
                // Use the full elem_width-wide count as an unsigned int
                // — matches Z3 bvshl/bvlshr/bvashr semantics (count ≥
                // operand width collapses to 0 or sign-fill).
                let amt = (s >> lo) & elem_mask;

                let shifted: u128 = match kind {
                    VecShiftKind::Shl => {
                        if amt >= elem_width as u128 {
                            0
                        } else {
                            (a << amt) & elem_mask
                        }
                    }
                    VecShiftKind::Shr => {
                        if amt >= elem_width as u128 {
                            0
                        } else {
                            a >> amt
                        }
                    }
                    VecShiftKind::Sar => {
                        let neg = a & sign_bit != 0;
                        if amt >= elem_width as u128 {
                            if neg { elem_mask } else { 0 }
                        } else if neg {
                            let shifted_val = a >> amt;
                            let fill_mask = (elem_mask << (elem_width as u128 - amt)) & elem_mask;
                            (shifted_val | fill_mask) & elem_mask
                        } else {
                            a >> amt
                        }
                    }
                };

                result |= (shifted & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback. Both operands are split into lane-width
        // slices; Z3 bvshl/bvlshr/bvashr produce the matching out-of-range
        // behaviour, so no extra width guards are needed.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let a = vec.extract(hi, lo, ctx);
            let b = amts.extract(hi, lo, ctx);
            let shifted = match kind {
                VecShiftKind::Shl => a.shl_into(b, ctx),
                VecShiftKind::Shr => a.lshr_into(b, ctx),
                VecShiftKind::Sar => a.ashr_into(b, ctx),
            };
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    // =========================================================================
    // Float Operations (using bit manipulation for now)
    // =========================================================================

    fn float_neg(arg: RustBV, ty: IRType, ctx: &SymContext) -> Result<RustBV, OpError> {
        // Flip the sign bit
        let sign_bit = match ty {
            IRType::F32 => 31,
            IRType::F64 => 63,
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        let mask = RustBV::concrete(1u128 << sign_bit, arg.width());
        Ok(arg.xor_into(mask, ctx))
    }

    fn float_abs(arg: RustBV, ty: IRType, ctx: &SymContext) -> Result<RustBV, OpError> {
        // Clear the sign bit
        let mask = match ty {
            IRType::F32 => RustBV::concrete(0x7FFFFFFF, 32),
            IRType::F64 => RustBV::concrete(0x7FFFFFFFFFFFFFFF, 64),
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        Ok(arg.and_into(mask, ctx))
    }

    fn float_sqrt(arg: RustBV, ty: IRType, _ctx: &SymContext) -> Result<RustBV, OpError> {
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
            matches!(
                kind,
                FloatOpKind::Add | FloatOpKind::Sub | FloatOpKind::Mul | FloatOpKind::Div
            ),
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

    /// SSE scalar-in-vector reciprocal/rsqrt estimate (RCPSS / RSQRTSS).
    /// VEX leaves the lane-0 result implementation-defined, so we hand back a
    /// fresh symbolic of the lane width; upper 96 bits pass through from arg.
    /// Used for `Iop_RecipEst32F0x4` and `Iop_RSqrtEst32F0x4`. The arg is still
    /// consumed (the upper-lane passthrough preserves it) so dataflow is sane.
    fn vec_float_scalar_fresh(
        arg: RustBV,
        elem: IRType,
        name: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), 128);
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let upper = arg.extract(127, lane_bits, ctx);
        let lane = RustBV::symbolic(ctx, name, lane_bits);
        Ok(upper.concat_into(lane, ctx))
    }

    /// Packed FP fresh-symbolic per lane. Used for `VFRecipEst`/`VFRSqrtEst`
    /// (precision implementation-defined) and `VFRecipStep`/`VFRSqrtStep`
    /// (angr Python has no generic handler, so a fresh symbolic per lane is
    /// the conservative match — Newton-Raphson refinement converges anyway).
    fn vec_float_fresh_per_lane(
        elem: IRType,
        count: u8,
        name: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let lane_bits = match elem {
            IRType::F32 => 32,
            IRType::F64 => 64,
            _ => return Err(OpError::InvalidFloatType(elem)),
        };
        let mut lanes: Vec<RustBV> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            lanes.push(RustBV::symbolic(ctx, name, lane_bits));
        }
        Ok(Self::concat_le_elements(lanes, ctx))
    }

    /// Packed integer fresh-symbolic per lane. Used for `VIRecipEst` /
    /// `VIRSqrtEst` (ARM URECPE / URSQRTE) where claripy has no generic
    /// handler — emitting a fresh symbolic per lane keeps the Rust engine
    /// from diverging from angr Python (which raises) while still letting
    /// the binary's Newton-Raphson refinement loop converge.
    fn vec_int_fresh_per_lane(
        lane_bits: u32,
        count: u8,
        name: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let mut lanes: Vec<RustBV> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            lanes.push(RustBV::symbolic(ctx, name, lane_bits));
        }
        Ok(Self::concat_le_elements(lanes, ctx))
    }

    /// Scalar max/min in vector (MAXSS/MAXSD/MINSS/MINSD). `is_max` picks the
    /// `>`/`<` comparison; both encode SSE's "NaN returns right" semantics
    /// (the concrete branch via Rust's `>`/`<`, the symbolic fallback via
    /// `vec_float_scalar_lane_minmax`'s ITE).
    fn vec_float_scalar_minmax(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        is_max: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);
                    let res0 = if is_max {
                        if l0 > r0 { l0 } else { r0 }
                    } else if l0 < r0 {
                        l0
                    } else {
                        r0
                    };
                    let upper = l & !0xFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                IRType::F64 => {
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);
                    let res0 = if is_max {
                        if l0 > r0 { l0 } else { r0 }
                    } else if l0 < r0 {
                        l0
                    } else {
                        r0
                    };
                    let upper = l & !0xFFFFFFFFFFFFFFFFu128;
                    upper | (res0.to_bits() as u128)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        Self::vec_float_scalar_lane_minmax(left, right, elem, is_max, ctx)
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
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
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
                    if is_max {
                        l_signed >= r_signed
                    } else {
                        l_signed <= r_signed
                    }
                } else {
                    if is_max {
                        l_elem >= r_elem
                    } else {
                        l_elem <= r_elem
                    }
                };

                let chosen = if pick_left { l_elem } else { r_elem };
                result |= (chosen & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
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

    /// NEON pairwise widening add — `Iop_PwAddL{N}{S/U}x{M}`. Unary.
    /// Output element i (width `2*elem`) = sext_or_zext(a[2i]) +
    /// sext_or_zext(a[2i+1]). Output lane count = `count / 2`; output total
    /// width = input total width.
    fn vec_pairwise_add_long(
        arg: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total_width);
        debug_assert!(count >= 2 && count.is_multiple_of(2));
        let out_pairs = count / 2;
        let out_elem_width = elem_width * 2;

        // Per-lane symbolic build — RustBV ops short-circuit when concrete.
        let mut elements: Vec<RustBV> = Vec::with_capacity(out_pairs as usize);
        for i in 0..out_pairs {
            let lo_a = (2 * i as u32) * elem_width;
            let hi_a = lo_a + elem_width - 1;
            let lo_b = (2 * i as u32 + 1) * elem_width;
            let hi_b = lo_b + elem_width - 1;
            let a_lane = arg.extract(hi_a, lo_a, ctx);
            let b_lane = arg.extract(hi_b, lo_b, ctx);
            let a_wide = if signed {
                a_lane.sign_extend_into(out_elem_width, ctx)
            } else {
                a_lane.zero_extend_into(out_elem_width, ctx)
            };
            let b_wide = if signed {
                b_lane.sign_extend_into(out_elem_width, ctx)
            } else {
                b_lane.zero_extend_into(out_elem_width, ctx)
            };
            elements.push(a_wide.add_into(b_wide, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON binary pairwise op — `Iop_PwAdd{N}x{M}` / `Iop_PwMin{N}{S/U}x{M}` /
    /// `Iop_PwMax{N}{S/U}x{M}`. Output lane shape matches the inputs. Per-lane:
    ///   * result[i]           = op(a[2i],   a[2i+1])              for i < count/2
    ///   * result[count/2 + i] = op(b[2i],   b[2i+1])              for i < count/2
    fn vec_pairwise_binop(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        op: PwOp,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);
        debug_assert!(count >= 2 && count.is_multiple_of(2));
        let half = count / 2;

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        // First half: pairs from `left`.
        for i in 0..half {
            let lo_a = (2 * i as u32) * elem_width;
            let hi_a = lo_a + elem_width - 1;
            let lo_b = (2 * i as u32 + 1) * elem_width;
            let hi_b = lo_b + elem_width - 1;
            let a = left.extract(hi_a, lo_a, ctx);
            let b = left.extract(hi_b, lo_b, ctx);
            elements.push(Self::pw_combine(a, b, op, ctx));
        }
        // Second half: pairs from `right`.
        for i in 0..half {
            let lo_a = (2 * i as u32) * elem_width;
            let hi_a = lo_a + elem_width - 1;
            let lo_b = (2 * i as u32 + 1) * elem_width;
            let hi_b = lo_b + elem_width - 1;
            let a = right.extract(hi_a, lo_a, ctx);
            let b = right.extract(hi_b, lo_b, ctx);
            elements.push(Self::pw_combine(a, b, op, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    #[inline]
    fn pw_combine(a: RustBV, b: RustBV, op: PwOp, ctx: &SymContext) -> RustBV {
        match op {
            PwOp::Add => a.add_into(b, ctx),
            PwOp::MinS => {
                let cond = a.clone().sle_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
            PwOp::MinU => {
                let cond = a.clone().ule_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
            PwOp::MaxS => {
                let cond = a.clone().sge_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
            PwOp::MaxU => {
                let cond = a.clone().uge_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
        }
    }

    /// NEON rounding halving add — `Iop_Avg{N}{S/U}x{M}`. Per-lane:
    ///   `((a[i] + b[i] + 1) >> 1)` truncated to `elem` bits.
    /// Each lane is sign- or zero-extended to `elem+1` bits to absorb the carry
    /// from `+1`, summed, shifted right by 1 (logical — the high bit of the
    /// widened sum carries the rounding bit for both signedness conventions),
    /// then truncated back to `elem` bits.
    fn vec_rounding_avg(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);
        let wide = elem_width + 1;
        let one_wide = RustBV::concrete(1, wide);

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let lo = i * elem_width;
            let hi = lo + elem_width - 1;
            let a_lane = left.extract(hi, lo, ctx);
            let b_lane = right.extract(hi, lo, ctx);
            let a_wide = if signed {
                a_lane.sign_extend_into(wide, ctx)
            } else {
                a_lane.zero_extend_into(wide, ctx)
            };
            let b_wide = if signed {
                b_lane.sign_extend_into(wide, ctx)
            } else {
                b_lane.zero_extend_into(wide, ctx)
            };
            let sum = a_wide.add_into(b_wide, ctx).add_into(one_wide.clone(), ctx);
            let shifted = sum.lshr_into(RustBV::concrete(1, wide), ctx);
            // Truncate to elem bits — for signed, the (elem)th bit of `shifted`
            // is the original sign bit by construction, so the low `elem` bits
            // are the correct two's-complement representation.
            elements.push(shifted.extract(elem_width - 1, 0, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON per-byte popcount — `Iop_Cnt8x{8,16}` (ARM CNT). Each 8-bit lane
    /// is replaced by the count of set bits in that lane (0..=8). Result
    /// width equals input width (8 * count bits). Symbolic fast-path: each
    /// bit of the lane is zero-extended to 8 bits and summed.
    fn vec_cnt(arg: RustBV, count: u8, ctx: &SymContext) -> Result<RustBV, OpError> {
        let total = 8u32 * count as u32;
        debug_assert_eq!(arg.width(), total);

        // Concrete fast path: iterate bytes, count_ones each.
        if let Some(v) = arg.as_u128() {
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let lane = ((v >> (i * 8)) as u8) as u32;
                let popcnt = lane.count_ones() as u128;
                result |= popcnt << (i * 8);
            }
            return Ok(RustBV::concrete(result, total));
        }

        // Symbolic: per byte, sum the 8 bits as 8-bit zero-extended adds.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let lane_lo = i * 8;
            let mut acc = RustBV::concrete(0, 8);
            for b in 0..8 {
                let bit = arg.extract(lane_lo + b, lane_lo + b, ctx);
                let bit_ext = bit.zero_extend_into(8, ctx);
                acc = acc.add_into(bit_ext, ctx);
            }
            elements.push(acc);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON per-lane Clz/Cls — `Iop_Clz{N}x{M}` / `Iop_Cls{N}x{M}`. Each
    /// `elem`-wide lane is replaced by:
    ///   * `Clz`: number of leading-zero bits, in `[0, N]` (all-zero → N).
    ///   * `Cls`: number of consecutive bits below the MSB that equal the
    ///     MSB, in `[0, N-1]` (all-same → N-1).
    ///
    /// Implementation mirrors claripy's `_op_generic_Clz` ITE-chain pattern
    /// (irop.py L700) but applied per-lane. For Cls, the chain runs over the
    /// non-MSB bits and the comparison is against the lane's MSB.
    fn vec_lane_count(
        arg: RustBV,
        elem: IRType,
        count: u8,
        kind: LaneCountKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total);

        // Concrete fast path (lanes ≤ 32 bits — width fits in u32).
        if let Some(v) = arg.as_u128() {
            let mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let lane = (v >> (i * elem_width)) & mask;
                let res_lane = match kind {
                    LaneCountKind::Clz => {
                        if lane == 0 {
                            elem_width as u128
                        } else {
                            // u128 leading zeros minus the padding above elem_width.
                            (lane.leading_zeros() - (128 - elem_width)) as u128
                        }
                    }
                    LaneCountKind::Cls => {
                        let sign = (lane >> (elem_width - 1)) & 1;
                        // Flip if sign is 1 so we count leading zeros of the
                        // result; then strip the MSB and clz it.
                        let flipped = if sign == 1 { (!lane) & mask } else { lane };
                        // Clear the MSB and look at the bits below.
                        let body = flipped & (mask >> 1);
                        if body == 0 {
                            // All bits below MSB matched sign → result = N-1.
                            (elem_width - 1) as u128
                        } else {
                            // leading_zeros of body within (elem_width-1) bits.
                            // body has its MSB-1 bit at position elem_width-2.
                            (body.leading_zeros() - (128 - (elem_width - 1))) as u128
                        }
                    }
                };
                result |= res_lane << (i * elem_width);
            }
            return Ok(RustBV::concrete(result, total));
        }

        // Symbolic ITE-chain per lane (mirrors claripy's _op_generic_Clz shape).
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let lane_lo = i * elem_width;
            let lane_hi = lane_lo + elem_width - 1;
            let lane = arg.extract(lane_hi, lane_lo, ctx);
            let res = match kind {
                LaneCountKind::Clz => Self::clz_chain(lane, elem_width, ctx),
                LaneCountKind::Cls => Self::cls_chain(lane, elem_width, ctx),
            };
            elements.push(res);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Build an ITE chain that returns clz of an `n`-bit lane in `n` bits.
    /// Mirrors `_op_generic_Clz` in `angr/engines/vex/claripy/irop.py`.
    fn clz_chain(lane: RustBV, n: u32, ctx: &SymContext) -> RustBV {
        let mut expr = RustBV::concrete(n as u128, n);
        let one = RustBV::concrete(1, 1);
        for a in 0..n {
            let bit = lane.extract(a, a, ctx);
            let cond = bit.eq_into(one.clone(), ctx);
            let then_v = RustBV::concrete((n - a - 1) as u128, n);
            expr = cond.ite_into(then_v, expr, ctx);
        }
        expr
    }

    /// Build an ITE chain that returns cls (count leading sign bits, excluding
    /// the MSB) of an `n`-bit lane in `n` bits. Default value is `n-1` (all
    /// bits below the MSB match the MSB); the chain returns `n - 2 - a` for
    /// the highest non-MSB position `a` whose bit differs from the MSB.
    fn cls_chain(lane: RustBV, n: u32, ctx: &SymContext) -> RustBV {
        let sign = lane.extract(n - 1, n - 1, ctx);
        let mut expr = RustBV::concrete((n - 1) as u128, n);
        for a in 0..(n - 1) {
            let bit = lane.extract(a, a, ctx);
            let cond = bit.ne_into(sign.clone(), ctx);
            let then_v = RustBV::concrete((n - 2 - a) as u128, n);
            expr = cond.ite_into(then_v, expr, ctx);
        }
        expr
    }

    /// NEON GF(2) polynomial multiply — `Iop_PolynomialMul8x{8,16}` (non-
    /// widening) and `Iop_PolynomialMull8x8` (widening). Per-lane carry-less
    /// multiply over GF(2): the product is the XOR of shifted copies of `b`
    /// selected by the bits of `a`. The non-widening result keeps the low
    /// 8 bits; the widening result keeps all 16.
    fn vec_polynomial_mul(
        left: RustBV,
        right: RustBV,
        count: u8,
        widen: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_total = 8u32 * count as u32;
        debug_assert_eq!(left.width(), in_total);
        debug_assert_eq!(right.width(), in_total);
        let out_elem: u32 = if widen { 16 } else { 8 };

        // Concrete fast path.
        if let (Some(a_all), Some(b_all)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let a = ((a_all >> (i * 8)) as u8) as u16;
                let b = ((b_all >> (i * 8)) as u8) as u16;
                // Carry-less mul into 16-bit accumulator.
                let mut prod: u16 = 0;
                for bit in 0..8 {
                    if (a >> bit) & 1 != 0 {
                        prod ^= b << bit;
                    }
                }
                let lane_val = if widen {
                    prod as u128
                } else {
                    (prod & 0xFF) as u128
                };
                result |= lane_val << (i * out_elem);
            }
            let out_total = out_elem * count as u32;
            return Ok(RustBV::concrete(result, out_total));
        }

        // Symbolic: build per-lane polynomial mul as XOR of conditional shifts
        // of `b`. Work in 16 bits to capture the full product; truncate to 8
        // for the non-widening case.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let one = RustBV::concrete(1, 1);
        let zero16 = RustBV::concrete(0, 16);
        for i in 0..count as u32 {
            let lo = i * 8;
            let hi = lo + 7;
            let a_lane = left.extract(hi, lo, ctx);
            let b_lane = right.extract(hi, lo, ctx);
            // Widen b to 16 bits for shifting (max shift is 7).
            let b_wide = b_lane.zero_extend_into(16, ctx);
            let mut acc = zero16.clone();
            for bit in 0..8u32 {
                let bit_a = a_lane.extract(bit, bit, ctx);
                let cond = bit_a.eq_into(one.clone(), ctx);
                let shifted = b_wide
                    .clone()
                    .shl_into(RustBV::concrete(bit as u128, 16), ctx);
                let addend = cond.ite_into(shifted, zero16.clone(), ctx);
                acc = acc.xor_into(addend, ctx);
            }
            let lane_out = if widen { acc } else { acc.extract(7, 0, ctx) };
            elements.push(lane_out);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON per-lane saturating integer add/sub —
    /// Iop_QAdd{N}{S/U}x{M} / Iop_QSub{N}{S/U}x{M}.
    ///
    /// Mirrors the claripy reference at
    /// `angr/engines/vex/claripy/irop.py::_op_generic_QAdd` (signed):
    ///   * Detect overflow with sign-bit algebra:
    ///     QAdd: `(~(top_a ^ top_b)) & (top_a ^ top_r)` — both inputs same
    ///     sign, result flips → overflow.
    ///     QSub: `( (top_a ^ top_b)) & (top_a ^ top_r)` — inputs differ,
    ///     result's sign differs from minuend → overflow.
    ///   * Saturated cap: `INT_MAX + ~top_r` — yields INT_MAX when the result
    ///     would be "too positive" (top_r=0 → +1 wraps to INT_MIN) and INT_MIN
    ///     when "too negative" (top_r=1 → +0 keeps INT_MAX). Actually:
    ///     top_r=1 (negative result, meaning positive overflow) → ~top_r=0
    ///     → cap = INT_MAX.
    ///     top_r=0 (positive result, meaning negative overflow) → ~top_r=1
    ///     → cap = INT_MAX + 1 = INT_MIN (two's complement wrap).
    ///
    /// Unsigned QAdd: overflow iff `res < a` (carry); cap = UINT_MAX.
    /// Unsigned QSub: overflow iff `res > a` (borrow); cap = 0.
    fn vec_int_saturating(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        is_sub: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // Concrete fast path (fits in u128 — total_width <= 128 covers all
        // currently-mapped NEON shapes).
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sign_bit: u128 = 1u128 << (elem_width - 1);
            let smax: u128 = sign_bit - 1; // 0x7F... in elem_width bits
            let smin: u128 = sign_bit; // 0x80...
            let umax: u128 = elem_mask;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let a = (l >> lo) & elem_mask;
                let b = (r >> lo) & elem_mask;

                let sat = if signed {
                    // Sign-extend each lane to i128 to compute the true
                    // arithmetic result, then clamp into [-smax-1, smax].
                    let a_signed = if a & sign_bit != 0 {
                        (a | !elem_mask) as i128
                    } else {
                        a as i128
                    };
                    let b_signed = if b & sign_bit != 0 {
                        (b | !elem_mask) as i128
                    } else {
                        b as i128
                    };
                    let raw: i128 = if is_sub {
                        a_signed - b_signed
                    } else {
                        a_signed + b_signed
                    };
                    let max_signed = smax as i128;
                    let min_signed = -(sign_bit as i128);
                    if raw > max_signed {
                        smax
                    } else if raw < min_signed {
                        smin
                    } else {
                        (raw as u128) & elem_mask
                    }
                } else if is_sub {
                    // Unsigned subtract: clamp underflow to 0.
                    if a >= b { (a - b) & elem_mask } else { 0 }
                } else {
                    // Unsigned add: clamp overflow to UINT_MAX.
                    let raw = a + b; // both < 2^N, sum < 2^(N+1) ≤ 2^128
                    if raw > umax { umax } else { raw }
                };

                result |= (sat & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback. Translates the claripy algorithm:
        //   top_x = lane[N-1]; res = a ± b; top_r = res[N-1].
        //   signed:   cap_cond = (xor_sign_match ^ (top_a XOR top_r)) == 1.
        //             cap      = (-1)/2 + ~top_r   (signed semantics).
        //   unsigned add: cap_cond = ULT(res, a); cap = -1.
        //   unsigned sub: cap_cond = UGT(res, a); cap =  0.
        let smax_bv = RustBV::concrete(
            ((1u128 << (elem_width - 1)) - 1) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let smin_bv = RustBV::concrete(
            (1u128 << (elem_width - 1)) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let umax_bv = RustBV::concrete((!0u128) >> (128 - elem_width), elem_width);
        let zero_bv = RustBV::concrete(0, elem_width);

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let a = left.extract(hi, lo, ctx);
            let b = right.extract(hi, lo, ctx);

            let res = if is_sub {
                a.clone().sub_into(b.clone(), ctx)
            } else {
                a.clone().add_into(b.clone(), ctx)
            };

            let lane = if signed {
                // Sign-bit relationships dictate the cap and the overflow flag.
                let top_a = a.extract(elem_width - 1, elem_width - 1, ctx);
                let top_b = b.extract(elem_width - 1, elem_width - 1, ctx);
                let top_r = res.extract(elem_width - 1, elem_width - 1, ctx);

                // QAdd: ~(a ^ b) & (a ^ r); QSub: (a ^ b) & (a ^ r).
                let a_xor_b = top_a.clone().xor_into(top_b, ctx);
                let a_xor_r = top_a.xor_into(top_r.clone(), ctx);
                let lhs = if is_sub {
                    a_xor_b
                } else {
                    a_xor_b.not_into(ctx)
                };
                let overflow_flag = lhs.and_into(a_xor_r, ctx);
                let cap_cond = overflow_flag.eq(&RustBV::concrete(1, 1), ctx);

                // cap = INT_MAX when top_r = 1 (positive overflow → clamp high),
                // cap = INT_MIN when top_r = 0 (negative overflow → clamp low).
                let top_r_one = top_r.eq(&RustBV::concrete(1, 1), ctx);
                let cap = top_r_one.ite(&smax_bv, &smin_bv, ctx);

                cap_cond.ite(&cap, &res, ctx)
            } else if is_sub {
                let cap_cond = res.ugt(&a, ctx);
                cap_cond.ite(&zero_bv, &res, ctx)
            } else {
                let cap_cond = res.ult(&a, ctx);
                cap_cond.ite(&umax_bv, &res, ctx)
            };

            elements.push(lane);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON saturating shift-left by vector — `Iop_QShl{N}x{M}` (unsigned,
    /// `signed=false`) / `Iop_QSal{N}x{M}` (signed, `signed=true`). Maps to
    /// ARM UQSHL / SQSHL (DDI 0487 C7.2.327 / C7.2.298). Per-lane semantics
    /// (`amt` = shift-amount lane, sign-extended; `a` = data lane):
    ///   * `amt >= 0`: left shift by `amt`; saturate to UMAX (unsigned) or
    ///     SMAX/SMIN (signed, based on sign of `a`). Counts ≥ lane width
    ///     saturate unless `a == 0`.
    ///   * `amt < 0`: right shift by `-amt`; logical (unsigned) or arithmetic
    ///     (signed). Counts ≥ lane width collapse to 0 / sign-fill.
    ///
    /// Derived from libVEX `host_generic_simd*` h_generic_calc_QShl* helpers;
    /// claripy has no `_op_generic_QShl` / `_op_generic_QSal` reference.
    fn vec_qshl_sat(
        vec: RustBV,
        amts: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(amts.width(), total_width);

        // Concrete fast path: both operands fit in u128 (covers every NEON
        // shape we route here — 64- and 128-bit vectors).
        if total_width <= 128
            && let (Some(v), Some(s)) = (vec.as_u128(), amts.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sign_bit: u128 = 1u128 << (elem_width - 1);
            let smax: u128 = sign_bit - 1; // 0x7F...
            let smin: u128 = sign_bit; // 0x80...
            let umax: u128 = elem_mask;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let a = (v >> lo) & elem_mask;
                let amt_raw = (s >> lo) & elem_mask;
                // Sign-extend the amt lane: ARM SQSHL/UQSHL treat the
                // shift-amount lane as signed (negative → right shift).
                let amt_signed: i128 = if amt_raw & sign_bit != 0 {
                    (amt_raw | !elem_mask) as i128
                } else {
                    amt_raw as i128
                };

                let sat = if amt_signed >= 0 {
                    let shift_amt = amt_signed as u32;
                    if signed {
                        // Sal: signed left shift with overflow → SMAX/SMIN.
                        let a_signed = if a & sign_bit != 0 {
                            (a | !elem_mask) as i128
                        } else {
                            a as i128
                        };
                        let smax_i = smax as i128;
                        let smin_i = -(sign_bit as i128);
                        if shift_amt >= elem_width {
                            // Out-of-range left shift: any nonzero a → saturate.
                            if a_signed > 0 {
                                smax
                            } else if a_signed < 0 {
                                smin
                            } else {
                                0
                            }
                        } else {
                            // i128 shift never overflows for our widths
                            // (max elem_width = 64, shift_amt < 64, so
                            // |a_signed| < 2^63 → |raw| < 2^127).
                            let raw = a_signed << shift_amt;
                            if raw > smax_i {
                                smax
                            } else if raw < smin_i {
                                smin
                            } else {
                                (raw as u128) & elem_mask
                            }
                        }
                    } else {
                        // Shl: unsigned left shift with overflow → UMAX.
                        if shift_amt >= elem_width {
                            if a != 0 { umax } else { 0 }
                        } else {
                            // a < 2^elem_width and shift_amt < elem_width,
                            // so raw fits in u128 (elem_width ≤ 64).
                            let raw = a << shift_amt;
                            if (raw & !elem_mask) != 0 {
                                umax
                            } else {
                                raw & elem_mask
                            }
                        }
                    }
                } else {
                    // amt < 0 → right shift by -amt.
                    let r_amt = (-amt_signed) as u32;
                    if signed {
                        // Ashr: out-of-range → sign-fill.
                        let neg = a & sign_bit != 0;
                        if r_amt >= elem_width {
                            if neg { elem_mask } else { 0 }
                        } else if neg {
                            let shifted_val = a >> r_amt;
                            let fill_mask =
                                (elem_mask << (elem_width as u128 - r_amt as u128)) & elem_mask;
                            (shifted_val | fill_mask) & elem_mask
                        } else {
                            a >> r_amt
                        }
                    } else {
                        // Lshr: out-of-range → 0.
                        if r_amt >= elem_width { 0 } else { a >> r_amt }
                    }
                };

                result |= (sat & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback. The Z3 bvshl/bvlshr/bvashr semantics
        // (count ≥ width → 0 or sign-fill) match the "out of range" branches
        // of QShl/QSal, so no explicit width guards are needed; overflow is
        // detected by the `(shl >> amt) != a` round-trip check.
        let smax_bv = RustBV::concrete(
            ((1u128 << (elem_width - 1)) - 1) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let smin_bv = RustBV::concrete(
            (1u128 << (elem_width - 1)) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let umax_bv = RustBV::concrete((!0u128) >> (128 - elem_width), elem_width);
        let zero_bv = RustBV::concrete(0, elem_width);
        let bit_one = RustBV::concrete(1, 1);

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let a = vec.extract(hi, lo, ctx);
            let b = amts.extract(hi, lo, ctx);

            // Left-shift branch (amt ≥ 0). Both signed and unsigned saturate
            // when the round-trip `(a << amt) >> amt` differs from `a`.
            let shl_res = a.clone().shl_into(b.clone(), ctx);

            let left_result = if signed {
                // Ashr round-trip detects signed overflow (and OOR shifts).
                let recovered = shl_res.clone().ashr_into(b.clone(), ctx);
                let no_overflow = recovered.eq(&a, ctx);
                let a_top = a.extract(elem_width - 1, elem_width - 1, ctx);
                let a_is_neg = a_top.eq(&bit_one, ctx);
                let cap = a_is_neg.ite(&smin_bv, &smax_bv, ctx);
                no_overflow.ite(&shl_res, &cap, ctx)
            } else {
                // Lshr round-trip detects unsigned overflow.
                let recovered = shl_res.clone().lshr_into(b.clone(), ctx);
                let no_overflow = recovered.eq(&a, ctx);
                no_overflow.ite(&shl_res, &umax_bv, ctx)
            };

            // Right-shift branch (amt < 0). Use -b as the count; Z3 handles
            // OOR (count ≥ width → 0 or sign-fill) natively.
            let neg_amt = zero_bv.clone().sub_into(b.clone(), ctx);
            let right_result = if signed {
                a.clone().ashr_into(neg_amt, ctx)
            } else {
                a.clone().lshr_into(neg_amt, ctx)
            };

            // Dispatch on the sign of amt (top bit).
            let amt_top = b.extract(elem_width - 1, elem_width - 1, ctx);
            let amt_is_neg = amt_top.eq(&bit_one, ctx);
            let lane = amt_is_neg.ite(&right_result, &left_result, ctx);

            elements.push(lane);
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
        if total_width <= 128
            && let Some(v) = arg.as_u128()
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
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
    fn set_v128_lo32(vec: RustBV, val: RustBV, ctx: &SymContext) -> Result<RustBV, OpError> {
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
    fn set_v128_lo64(vec: RustBV, val: RustBV, ctx: &SymContext) -> Result<RustBV, OpError> {
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
        Ok(build_float_expr(
            FloatOpKind::CmpEq,
            prec,
            vec![left, right],
        ))
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
        Ok(build_float_expr(
            FloatOpKind::CmpLt,
            prec,
            vec![left, right],
        ))
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
        Ok(build_float_expr(
            FloatOpKind::CmpLe,
            prec,
            vec![left, right],
        ))
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
                (IRType::F32, FCmpKind::Lt) => f32::from_bits(l as u32) < f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Le) => f32::from_bits(l as u32) <= f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Gt) => f32::from_bits(l as u32) > f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Ge) => f32::from_bits(l as u32) >= f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Un) => {
                    f32::from_bits(l as u32).is_nan() || f32::from_bits(r as u32).is_nan()
                }
                (IRType::F64, FCmpKind::Eq) => f64::from_bits(l as u64) == f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Lt) => f64::from_bits(l as u64) < f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Le) => f64::from_bits(l as u64) <= f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Gt) => f64::from_bits(l as u64) > f64::from_bits(r as u64),
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
                // un = isNaN(l) OR isNaN(r). IsNaN is a unary primitive so no
                // operand clone is needed (vs. CmpEq(x, x) which doubles x).
                let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![l_lo]);
                let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![r_lo]);
                l_nan.or_into(r_nan, ctx)
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
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = (1u128 << elem_width) - 1;
            for i in 0..count {
                let shift = (i as u32) * elem_width;
                let l_bits = (l >> shift) & elem_mask;
                let r_bits = (r >> shift) & elem_mask;
                let truth = match (elem, kind) {
                    (IRType::F32, FCmpKind::Eq) => {
                        f32::from_bits(l_bits as u32) == f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Lt) => {
                        f32::from_bits(l_bits as u32) < f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Le) => {
                        f32::from_bits(l_bits as u32) <= f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Gt) => {
                        f32::from_bits(l_bits as u32) > f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Ge) => {
                        f32::from_bits(l_bits as u32) >= f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Un) => {
                        f32::from_bits(l_bits as u32).is_nan()
                            || f32::from_bits(r_bits as u32).is_nan()
                    }
                    (IRType::F64, FCmpKind::Eq) => {
                        f64::from_bits(l_bits as u64) == f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Lt) => {
                        f64::from_bits(l_bits as u64) < f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Le) => {
                        f64::from_bits(l_bits as u64) <= f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Gt) => {
                        f64::from_bits(l_bits as u64) > f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Ge) => {
                        f64::from_bits(l_bits as u64) >= f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Un) => {
                        f64::from_bits(l_bits as u64).is_nan()
                            || f64::from_bits(r_bits as u64).is_nan()
                    }
                    _ => return Err(OpError::InvalidFloatType(elem)),
                };
                let lane_val = if truth { lane_mask } else { 0 };
                result |= lane_val << shift;
            }
            return Ok(RustBV::concrete(result, total_width));
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
                    // un = isNaN(l) OR isNaN(r); IsNaN is a unary primitive.
                    let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![l_lane]);
                    let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![r_lane]);
                    l_nan.or_into(r_nan, ctx)
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
                    if lf.is_nan() || rf.is_nan() {
                        0x45
                    } else if lf < rf {
                        0x01
                    } else if lf == rf {
                        0x40
                    } else {
                        0x00
                    }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf.is_nan() || rf.is_nan() {
                        0x45
                    } else if lf < rf {
                        0x01
                    } else if lf == rf {
                        0x40
                    } else {
                        0x00
                    }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 32));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;

        // Symbolic: compose un/lt/eq predicates, then nest ITEs.
        // un  = isNaN(l) OR isNaN(r)   [unary primitive, no operand clone]
        // lt  = l < r                  [false if either is NaN]
        // eq  = l == r                 [false if either is NaN]
        let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![left.clone()]);
        let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![right.clone()]);
        let un = l_nan.or_into(r_nan, ctx);
        let lt = build_float_expr(FloatOpKind::CmpLt, prec, vec![left.clone(), right.clone()]);
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
                0 => Self::round_ties_to_even_f32(f), // nearest, ties to even
                1 => f.floor(),                       // toward -infinity
                2 => f.ceil(),                        // toward +infinity
                3 => f.trunc(),                       // toward zero
                _ => Self::round_ties_to_even_f32(f), // default to nearest
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
                0 => Self::round_ties_to_even_f64(f), // nearest, ties to even
                1 => f.floor(),                       // toward -infinity
                2 => f.ceil(),                        // toward +infinity
                3 => f.trunc(),                       // toward zero
                _ => Self::round_ties_to_even_f64(f), // default to nearest
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
            0 => Self::round_ties_to_even_f32(f), // nearest, ties to even (banker's rounding)
            1 => f.floor(),                       // toward negative infinity
            2 => f.ceil(),                        // toward positive infinity
            3 => f.trunc(),                       // toward zero (truncate)
            _ => Self::round_ties_to_even_f32(f), // default to nearest
        }
    }

    /// Apply rounding mode to f64 value
    fn apply_rounding_f64(f: f64, rm: u32) -> f64 {
        match rm & 0x3 {
            0 => Self::round_ties_to_even_f64(f), // nearest, ties to even (banker's rounding)
            1 => f.floor(),                       // toward negative infinity
            2 => f.ceil(),                        // toward positive infinity
            3 => f.trunc(),                       // toward zero (truncate)
            _ => Self::round_ties_to_even_f64(f), // default to nearest
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

    // --- Rounding-mode float conversions (binop variants) ---
    fn f64_to_f32_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        // Note: concrete path ignores rounding mode (uses direct cast); the
        // symbolic path correctly threads rm through Z3 FP.
        Self::float_to_float_rm(rm, arg, FloatPrec::F64, FloatPrec::F32, |v, _rm| {
            (f64::from_bits(v as u64) as f32).to_bits() as u128
        })
    }
    fn f32_to_i32s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 32, true, |v, rm| {
            (Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as i32 as u32) as u128
        })
    }
    fn f64_to_i32s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 32, true, |v, rm| {
            (Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as i32 as u32) as u128
        })
    }
    fn f32_to_i64s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 64, true, |v, rm| {
            (Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as i64 as u64) as u128
        })
    }
    fn f64_to_i64s_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 64, true, |v, rm| {
            (Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as i64 as u64) as u128
        })
    }
    fn f32_to_i32u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 32, false, |v, rm| {
            Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as u32 as u128
        })
    }
    fn f64_to_i32u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 32, false, |v, rm| {
            Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as u32 as u128
        })
    }
    fn f32_to_i64u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F32, 64, false, |v, rm| {
            Self::apply_rounding_f32(f32::from_bits(v as u32), rm) as u64 as u128
        })
    }
    fn f64_to_i64u_rm(rm: RustBV, arg: RustBV, _ctx: &SymContext) -> Result<RustBV, OpError> {
        Self::float_to_int_rm(rm, arg, FloatPrec::F64, 64, false, |v, rm| {
            Self::apply_rounding_f64(f64::from_bits(v as u64), rm) as u64 as u128
        })
    }
}

/// Errors from VEX operation execution.
///
/// `#[non_exhaustive]` per angr-irwe: new variants land in minor
/// versions as more ops gain explicit failure modes (e.g. additional
/// NEON scaffold buckets). Match sites must include a wildcard arm.
#[non_exhaustive]
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
    /// NEON op that hasn't been implemented yet (angr-bkcs scaffold).
    ///
    /// Distinct from [`Self::UnsupportedVectorOp`] because the silent
    /// fresh-symbolic fallback in `interpreter::expressions` swallows
    /// generic `OpError`s — this variant is propagated explicitly so
    /// missing NEON coverage surfaces as `RustUnsupportedVexOpError`
    /// instead of producing wrong results that are hard to attribute.
    /// See `invariant-neon-scaffolding-panic-not-fallback` (bd memories).
    #[error("NEON op {name} not yet implemented")]
    UnsupportedNeon { name: &'static str },
    /// Opcode string with no entry in `parse_opcode` (angr-tkbr.2).
    ///
    /// Routed from `IROp::Unmapped(name)`. Like `UnsupportedNeon`, this
    /// is propagated explicitly past the silent fresh-symbolic fallback
    /// in `interpreter::expressions` so callers see the real op name
    /// in a typed `RustUnsupportedVexOpError` instead of getting a
    /// fresh-symbolic value of the wrong width.
    #[error("unmapped VEX opcode: {op_name}")]
    UnsupportedVexOp { op_name: String },
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

        let result = VEXOps::binop(
            IROp::VAdd {
                elem: IRType::I32,
                count: 4,
            },
            av,
            bv,
            &ctx,
        )
        .unwrap();
        let rv = result.as_u128().unwrap();

        assert_eq!(rv & 0xFFFFFFFF, 11);
        assert_eq!((rv >> 32) & 0xFFFFFFFF, 22);
        assert_eq!((rv >> 64) & 0xFFFFFFFF, 33);
        assert_eq!((rv >> 96) & 0xFFFFFFFF, 44);
    }

    #[test]
    #[allow(clippy::identity_op)] // explicit 4-lane layout reads better than the minimized form
    fn test_vector_cmp_eq() {
        let ctx = SymContext::new_mock();

        // Two vectors of 4 x i32
        // [1, 2, 3, 4] == [1, 0, 3, 0] -> [0xFFFFFFFF, 0, 0xFFFFFFFF, 0]
        let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
        let b: u128 = 1 | (0 << 32) | (3 << 64) | (0 << 96);

        let av = RustBV::concrete(a, 128);
        let bv = RustBV::concrete(b, 128);

        let result = VEXOps::binop(
            IROp::VCmpEQ {
                elem: IRType::I32,
                count: 4,
            },
            av,
            bv,
            &ctx,
        )
        .unwrap();
        let rv = result.as_u128().unwrap();

        assert_eq!(rv & 0xFFFFFFFF, 0xFFFFFFFF); // 1 == 1
        assert_eq!((rv >> 32) & 0xFFFFFFFF, 0); // 2 != 0
        assert_eq!((rv >> 64) & 0xFFFFFFFF, 0xFFFFFFFF); // 3 == 3
        assert_eq!((rv >> 96) & 0xFFFFFFFF, 0); // 4 != 0
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

        assert!(
            (result_f32 - 6.0).abs() < 0.0001,
            "Expected 6.0, got {}",
            result_f32
        );
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

        assert!(
            (result_f32 - 3.0).abs() < 0.0001,
            "Expected 3.0, got {}",
            result_f32
        );
    }

    /// Symbolic FAdd: solving `x + 2.0 == 5.0` should yield x == 3.0.
    ///
    /// This is the canonical "constraint propagation" check for the new Z3
    /// FP-theory wiring: previously the symbolic branch returned a fresh
    /// unconstrained symbol and the solver would accept any value of x.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_add_symbolic_constraint() {
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        let sum = VEXOps::binop(IROp::FAdd(IRType::F32), x.clone(), two, &ctx).unwrap();

        // Constrain: sum's IEEE bits == bits(5.0).
        let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let eq = sum.to_z3_ast().eq(five.to_z3_ast());
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
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "sqrt_x", 64);

        let sqrt_x = VEXOps::unop(IROp::FSqrt(IRType::F64), x.clone(), &ctx).unwrap();

        let four = RustBV::concrete(4.0f64.to_bits() as u128, 64);
        let eq = sqrt_x.to_z3_ast().eq(four.to_z3_ast());
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
        let lt_x_two = VEXOps::binop(IROp::FCmpLT(IRType::F32), x.clone(), two, &ctx).unwrap();
        // 1.0 < x
        let lt_one_x = VEXOps::binop(IROp::FCmpLT(IRType::F32), one, x.clone(), &ctx).unwrap();

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
        let ctx = SymContext::new_mock();
        let xmm0 = RustBV::symbolic(&ctx, "xmm0", 128);
        // xmm1 = [2.0f, 0, 0, 0]
        let xmm1 = RustBV::concrete(2.0f32.to_bits() as u128, 128);

        let result =
            VEXOps::binop(IROp::VFAddS { elem: IRType::F32 }, xmm0.clone(), xmm1, &ctx).unwrap();
        assert_eq!(result.width(), 128);

        // Constrain low 32 bits of result to bits(5.0).
        let res_lo = result.extract(31, 0, &ctx);
        let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast().eq(five.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(
            ctx.is_sat(),
            "expected SAT after VFAddS symbolic constraint"
        );

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
        let eq_upper = upper_in.to_z3_ast().eq(upper_out.to_z3_ast());
        ctx.add_constraint(eq_upper);
        assert!(ctx.is_sat(), "expected upper-bits passthrough to hold");
    }

    /// Symbolic VFSqrtS (SQRTSS-style): sqrt(low32(xmm)) == 4.0 → low32 == 16.0.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_float_scalar_sqrt_symbolic() {
        let ctx = SymContext::new_mock();
        let xmm = RustBV::symbolic(&ctx, "xmm_sqrt", 128);

        let result = VEXOps::unop(IROp::VFSqrtS { elem: IRType::F32 }, xmm.clone(), &ctx).unwrap();
        assert_eq!(result.width(), 128);

        let res_lo = result.extract(31, 0, &ctx);
        let four = RustBV::concrete(4.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast().eq(four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(
            ctx.is_sat(),
            "expected SAT after VFSqrtS symbolic constraint"
        );

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
        let ctx = SymContext::new_mock();
        let xmm0 = RustBV::symbolic(&ctx, "xmm_max", 128);
        let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

        let result =
            VEXOps::binop(IROp::VFMaxS { elem: IRType::F32 }, xmm0.clone(), xmm1, &ctx).unwrap();
        assert_eq!(result.width(), 128);

        let res_lo = result.extract(31, 0, &ctx);
        let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast().eq(five.to_z3_ast());
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
        let ctx = SymContext::new_mock();
        let xmm0 = RustBV::symbolic(&ctx, "xmm_min", 128);
        let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

        let result =
            VEXOps::binop(IROp::VFMinS { elem: IRType::F32 }, xmm0.clone(), xmm1, &ctx).unwrap();
        assert_eq!(result.width(), 128);

        let res_lo = result.extract(31, 0, &ctx);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let eq = res_lo.to_z3_ast().eq(one.to_z3_ast());
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
        let ctx = SymContext::new_mock();
        let rm = RustBV::symbolic(&ctx, "rm_f32", 32);
        let value = RustBV::concrete((-2.5f32).to_bits() as u128, 32);

        let result = VEXOps::binop(IROp::RoundF32toInt, rm.clone(), value, &ctx).unwrap();
        let target = RustBV::concrete((-3.0f32).to_bits() as u128, 32);
        let eq = result.to_z3_ast().eq(target.to_z3_ast());
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
        let ctx = SymContext::new_mock();
        let rm = RustBV::symbolic(&ctx, "rm_f64", 32);
        let value = RustBV::concrete(2.5f64.to_bits() as u128, 64);

        let result = VEXOps::binop(IROp::RoundF64toInt, rm.clone(), value, &ctx).unwrap();
        let target = RustBV::concrete(3.0f64.to_bits() as u128, 64);
        let eq = result.to_z3_ast().eq(target.to_z3_ast());
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

        let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rz, one, ten, &ctx).unwrap();
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

        let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_ru, one, ten, &ctx).unwrap();
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

        let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rd, one, ten, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3DCCCCCC);
    }

    /// FDiv with symbolic rm: constraining result == 0x3DCCCCCC forces rm
    /// low-2-bits ∈ {1, 3} (RD or RZ); 0x3DCCCCCD forces rm low-2-bits ∈
    /// {0, 2}. Exercises the 4-way ITE built by `build_fp_arith_rm_cached`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_float_div_with_symbolic_rm_f32() {
        let ctx = SymContext::new_mock();
        let rm = RustBV::symbolic(&ctx, "rm_div_f32", 32);
        let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
        let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

        let result =
            VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm.clone(), one, ten, &ctx).unwrap();
        let target = RustBV::concrete(0x3DCCCCCC, 32);
        let eq = result.to_z3_ast().eq(target.to_z3_ast());
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

        let result = VEXOps::binop_with_rm(IROp::FAdd(IRType::F32), rm_rz, a, b, &ctx).unwrap();
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

        let result = VEXOps::unop_with_rm(IROp::FSqrt(IRType::F32), rm_ru, two, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3FB504F4, "sqrt(2) under RU rounds up one ulp");
    }

    /// SqrtRm: sqrt(2.0f32) RNE keeps the native fast path (no Z3).
    #[test]
    fn test_float_sqrt_with_rm_rne_fastpath_f32() {
        let ctx = SymContext::new_mock();
        let rm_rne = RustBV::concrete(0, 32);
        let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

        let result = VEXOps::unop_with_rm(IROp::FSqrt(IRType::F32), rm_rne, two, &ctx).unwrap();
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

        let result = VEXOps::binop(IROp::FSqrt(IRType::F32), rm_ru, two, &ctx).unwrap();
        let bits = ctx.eval(&result).expect("eval failed") as u32;
        assert_eq!(bits, 0x3FB504F4, "binop FSqrt+RU rounds up one ulp");
    }

    /// Iop_SqrtF64 via binop with RNE: native fast path returns sqrt(16.0)=4.0.
    #[test]
    fn test_float_sqrt_via_binop_rne_fastpath_f64() {
        let ctx = SymContext::new_mock();
        let rm_rne = RustBV::concrete(0, 32);
        let sixteen = RustBV::concrete(16.0f64.to_bits() as u128, 64);

        let result = VEXOps::binop(IROp::FSqrt(IRType::F64), rm_rne, sixteen, &ctx).unwrap();
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
        assert_eq!(
            f, -2147483648.0f32,
            "I32_MIN must round-trip to -2^31 as f32"
        );
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
        assert!(
            f.is_infinite() && f.is_sign_positive(),
            "1e300 → +inf, got {}",
            f
        );
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

        let result = VEXOps::binop(IROp::VFSubS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let rv = result.as_u128().unwrap();
        let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 7.0).abs() < 1e-6,
            "10.0 - 3.0 == 7.0, got {}",
            lane0
        );
        assert_eq!(
            rv & !0xFFFF_FFFFu128,
            upper_pattern,
            "upper 96 bits must pass through"
        );
    }

    /// VFMulS concrete lane isolation: MULSS xmm0, xmm1.
    #[test]
    fn test_vec_float_scalar_mul_concrete_lane_isolation() {
        let ctx = SymContext::new_mock();

        let upper_pattern: u128 = 0xFEED_FACE_BAAD_F00D_8BAD_F00Du128 << 32;
        let xmm0_bits = upper_pattern | (4.0f32.to_bits() as u128);
        let xmm0 = RustBV::concrete(xmm0_bits, 128);
        let xmm1 = RustBV::concrete(2.5f32.to_bits() as u128, 128);

        let result = VEXOps::binop(IROp::VFMulS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
        let rv = result.as_u128().unwrap();
        let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 10.0).abs() < 1e-6,
            "4.0 * 2.5 == 10.0, got {}",
            lane0
        );
        assert_eq!(
            rv & !0xFFFF_FFFFu128,
            upper_pattern,
            "upper 96 bits must pass through"
        );
    }

    /// Concrete coverage for every scalar-in-vector FP IROp at F64 precision.
    /// The {add,sub,mul,div,sqrt,max,min} S-suffixed variants all write to
    /// lane 0 (low 64 bits) and pass the upper 64 bits of `xmm0` through.
    #[test]
    fn test_vec_float_scalar_all_variants_f64() {
        let ctx = SymContext::new_mock();
        let upper_pattern: u128 = 0xCAFE_BABE_DEAD_BEEFu128 << 64;

        let xmm0 = |lane0: f64| RustBV::concrete(upper_pattern | (lane0.to_bits() as u128), 128);
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
            let result = VEXOps::binop(op, xmm0(l), xmm1(r), &ctx).unwrap();
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

        let result = VEXOps::binop(IROp::VFMaxS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
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
        let ctx = SymContext::new_mock();

        // Vector: [0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008]
        let mut v: u128 = 0;
        for i in 0..8u32 {
            v |= ((i + 1) as u128) << (i * 16);
        }
        let vec = RustBV::concrete(v, 128);
        let shift = RustBV::symbolic(&ctx, "shl_amt", 8);

        let result = VEXOps::binop(
            IROp::VShlN {
                elem: IRType::I16,
                count: 8,
            },
            vec,
            shift.clone(),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        // Constrain shift == 4.
        let four = RustBV::concrete(4, 8);
        let eq = shift.to_z3_ast().eq(four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        for i in 0..8u32 {
            let lane = (model >> (i * 16)) & 0xFFFF;
            let expected = ((i as u128 + 1) << 4) & 0xFFFF;
            assert_eq!(
                lane, expected,
                "lane {} expected {:#x}, got {:#x}",
                i, expected, lane
            );
        }
    }

    /// Symbolic ShrN32x4: shift count is symbolic; constrain to 8 and verify
    /// each lane is `lane >> 8`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_shr_n_symbolic_shift() {
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
            IROp::VShrN {
                elem: IRType::I32,
                count: 4,
            },
            vec,
            shift.clone(),
            &ctx,
        )
        .unwrap();

        let eight = RustBV::concrete(8, 8);
        let eq = shift.to_z3_ast().eq(eight.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after constraining shift == 8");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        for (i, lane) in lanes.iter().enumerate() {
            let got = ((model >> (i * 32)) & 0xFFFF_FFFF) as u32;
            let expected = lane >> 8;
            assert_eq!(
                got, expected,
                "lane {} expected {:#x}, got {:#x}",
                i, expected, got
            );
        }
    }

    /// Symbolic SarN16x8 with negative lanes: shift count is symbolic; constrain
    /// to 4 and verify sign-extending shift (negative values stay negative).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_sar_n_symbolic_shift() {
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
            IROp::VSarN {
                elem: IRType::I16,
                count: 8,
            },
            vec,
            shift.clone(),
            &ctx,
        )
        .unwrap();

        let four = RustBV::concrete(4, 8);
        let eq = shift.to_z3_ast().eq(four.to_z3_ast());
        ctx.add_constraint(eq);
        assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        for (i, lane) in lanes.iter().enumerate() {
            let got = ((model >> (i * 16)) & 0xFFFF) as u16 as i16;
            let expected = lane >> 4; // arithmetic shift in Rust on i16
            assert_eq!(
                got, expected,
                "lane {} expected {}, got {}",
                i, expected, got
            );
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
            IROp::VShlN {
                elem: IRType::I16,
                count: 4,
            },
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

        let l: [i16; 8] = [-5, 100, 0, -32768, 1, -1, 32767, -2];
        let r: [i16; 8] = [-3, 200, -100, -32767, -1, 0, 32766, 3];
        let exp: [i16; 8] = [-5, 100, -100, -32768, -1, -1, 32766, -2];

        let mut lv: u128 = 0;
        let mut rv: u128 = 0;
        for i in 0..8 {
            lv |= ((l[i] as u16) as u128) << (i as u32 * 16);
            rv |= ((r[i] as u16) as u128) << (i as u32 * 16);
        }
        let result = VEXOps::binop(
            IROp::VMin {
                elem: IRType::I16,
                count: 8,
                signed: true,
            },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = ((got >> (i as u32 * 16)) & 0xFFFF) as u16 as i16;
            assert_eq!(
                lane, expected,
                "lane {} expected {}, got {}",
                i, expected, lane
            );
        }
    }

    /// PMAXUB-style: unsigned max over 16x u8 lanes.
    #[test]
    fn test_vec_int_max_unsigned_concrete() {
        let ctx = SymContext::new_mock();

        let l: [u8; 16] = [
            0xFF, 0x00, 0x80, 0x7F, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        ];
        let r: [u8; 16] = [0x00, 0xFF, 0x7F, 0x80, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0];
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
            IROp::VMax {
                elem: IRType::I8,
                count: 16,
                signed: false,
            },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = ((got >> (i as u32 * 8)) & 0xFF) as u8;
            assert_eq!(
                lane, expected,
                "lane {} expected {:#x}, got {:#x}",
                i, expected, lane
            );
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
        for (i, &lane) in v.iter().enumerate() {
            bits |= ((lane as u16) as u128) << (i as u32 * 16);
        }
        let result = VEXOps::unop(
            IROp::VAbs {
                elem: IRType::I16,
                count: 8,
            },
            RustBV::concrete(bits, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = ((got >> (i as u32 * 16)) & 0xFFFF) as u16;
            assert_eq!(
                lane, expected,
                "lane {} expected {:#x}, got {:#x}",
                i, expected, lane
            );
        }
    }

    // =========================================================================
    // VReverse — byte/halfword/word/bit reversal within lane (angr-tukg.4).
    // =========================================================================

    /// Iop_Reverse8sIn32_x2 — byte-swap within each 32-bit word (REV32
    /// applied to a NEON D-register). 64-bit total.
    #[test]
    fn test_vec_reverse_8in32_x2_concrete() {
        let ctx = SymContext::new_mock();
        // Two 32-bit lanes: low = 0x11223344, high = 0xAABBCCDD.
        let v: u128 = 0xAABBCCDD_11223344u128;
        let result = VEXOps::unop(
            IROp::VReverse {
                sub_width: 8,
                elem: IRType::I32,
                count: 2,
            },
            RustBV::concrete(v, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        // Low lane bytes reversed: 0x11223344 → 0x44332211.
        // High lane: 0xAABBCCDD → 0xDDCCBBAA.
        let expected: u128 = 0xDDCCBBAA_44332211u128;
        assert_eq!(result.as_u128().unwrap(), expected);
    }

    /// Iop_Reverse32sIn64_x2 — swap the two 32-bit halves of each 64-bit
    /// lane. Directly mirrors the only explicit Python reference at
    /// `angr/engines/vex/claripy/irop.py:_op_Iop_Reverse32sIn64_x2`.
    #[test]
    fn test_vec_reverse_32in64_x2_concrete_matches_python_ref() {
        let ctx = SymContext::new_mock();
        // Python ref: Concat(arg[95:64], arg[127:96], arg[31:0], arg[63:32]).
        // Pick a 128-bit value with distinct 32-bit slices to exercise every
        // permutation slot.
        let v: u128 = 0xAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDDu128;
        let result = VEXOps::unop(
            IROp::VReverse {
                sub_width: 32,
                elem: IRType::I64,
                count: 2,
            },
            RustBV::concrete(v, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        // Slices of v (LSB → MSB indexing): [31:0]=DDDDDDDD, [63:32]=CCCCCCCC,
        //                                  [95:64]=BBBBBBBB, [127:96]=AAAAAAAA.
        // Python Concat(MSB→LSB): [95:64], [127:96], [31:0], [63:32]
        //   = BBBBBBBB AAAAAAAA DDDDDDDD CCCCCCCC (MSB→LSB)
        let expected: u128 = 0xBBBBBBBB_AAAAAAAA_DDDDDDDD_CCCCCCCCu128;
        assert_eq!(result.as_u128().unwrap(), expected);
    }

    /// Iop_Reverse1sIn8_x8 — RBIT: reverse the bit order inside each byte.
    #[test]
    fn test_vec_reverse_1in8_x8_concrete() {
        let ctx = SymContext::new_mock();
        // Byte 0 = 0b10110010 = 0xB2; reversed = 0b01001101 = 0x4D.
        // Byte 1 = 0xFF (palindrome); reversed = 0xFF.
        // Byte 2 = 0x01; reversed = 0x80.
        // Byte 3 = 0x80; reversed = 0x01.
        // Byte 4 = 0xA5; reversed = 0xA5 (10100101 → 10100101).
        // Byte 5 = 0x00; reversed = 0x00.
        // Byte 6 = 0x0F; reversed = 0xF0.
        // Byte 7 = 0xF0; reversed = 0x0F.
        let in_bytes: [u8; 8] = [0xB2, 0xFF, 0x01, 0x80, 0xA5, 0x00, 0x0F, 0xF0];
        let exp_bytes: [u8; 8] = [0x4D, 0xFF, 0x80, 0x01, 0xA5, 0x00, 0xF0, 0x0F];
        let mut v: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            v |= (in_bytes[i] as u128) << (i * 8);
            e |= (exp_bytes[i] as u128) << (i * 8);
        }
        let result = VEXOps::unop(
            IROp::VReverse {
                sub_width: 1,
                elem: IRType::I8,
                count: 8,
            },
            RustBV::concrete(v, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Reverse16sIn64_x2 — halfword swap inside each 64-bit lane,
    /// applied across two lanes (128-bit Q-register form).
    #[test]
    fn test_vec_reverse_16in64_x2_concrete() {
        let ctx = SymContext::new_mock();
        // Lane 0 (low 64): halfwords [0x1111, 0x2222, 0x3333, 0x4444] (LSB→MSB).
        // After reversal: [0x4444, 0x3333, 0x2222, 0x1111].
        // Lane 1 (high 64): halfwords [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD].
        // After reversal: [0xDDDD, 0xCCCC, 0xBBBB, 0xAAAA].
        let v: u128 = 0xDDDD_CCCC_BBBB_AAAA_4444_3333_2222_1111u128;
        let expected: u128 = 0xAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444u128;
        let result = VEXOps::unop(
            IROp::VReverse {
                sub_width: 16,
                elem: IRType::I64,
                count: 2,
            },
            RustBV::concrete(v, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), expected);
    }

    /// Involution: applying VReverse twice is the identity (any permutation
    /// that swaps positions i ↔ n-1-i is its own inverse). Exercise on a
    /// symbolic 128-bit input via a Z3 equivalence check.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_reverse_double_apply_is_identity() {
        let ctx = SymContext::new_mock();
        for (sub_width, elem, count) in [
            (8u8, IRType::I32, 4u8), // Reverse8sIn32_x4
            (16, IRType::I64, 2),    // Reverse16sIn64_x2
            (32, IRType::I64, 2),    // Reverse32sIn64_x2
            (1, IRType::I8, 16),     // Reverse1sIn8_x16
        ] {
            let width = elem.bits() * count as u32;
            let arg = RustBV::symbolic(&ctx, "vrev_arg", width);
            let op = IROp::VReverse {
                sub_width,
                elem,
                count,
            };
            let once = VEXOps::unop(op, arg.clone(), &ctx).unwrap();
            let twice = VEXOps::unop(op, once, &ctx).unwrap();
            // Assert there is no satisfying assignment where twice != arg.
            ctx.push();
            ctx.add_constraint(twice.to_z3_ast().eq(arg.to_z3_ast()).not());
            assert!(
                !ctx.is_sat(),
                "double-apply must equal identity for sub_width={} elem={:?} count={}",
                sub_width,
                elem,
                count
            );
            ctx.pop();
        }
    }

    /// Symbolic parity vs claripy reference: `Iop_Reverse32sIn64_x2` must
    /// produce the same bits as the explicit Python implementation
    /// `Concat(arg[95:64], arg[127:96], arg[31:0], arg[63:32])` for any
    /// 128-bit input. Verified through a Z3 universality check.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_reverse_32in64_x2_symbolic_matches_python_ref() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::symbolic(&ctx, "vrev_arg", 128);
        let got = VEXOps::unop(
            IROp::VReverse {
                sub_width: 32,
                elem: IRType::I64,
                count: 2,
            },
            arg.clone(),
            &ctx,
        )
        .unwrap();

        // Build the Python reference: Concat(arg[95:64], arg[127:96],
        //                                    arg[31:0],  arg[63:32]).
        // `concat_le_elements` indexes 0 → LSB, so push in LSB→MSB order:
        //   bits [31:0]  output  ← arg[63:32]
        //   bits [63:32] output  ← arg[31:0]
        //   bits [95:64] output  ← arg[127:96]
        //   bits [127:96] output ← arg[95:64]
        let py = VEXOps::concat_le_elements(
            vec![
                arg.extract(63, 32, &ctx),
                arg.extract(31, 0, &ctx),
                arg.extract(127, 96, &ctx),
                arg.extract(95, 64, &ctx),
            ],
            &ctx,
        );

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VReverse 32sIn64_x2 must match the Python reference Concat pattern"
        );
        ctx.pop();
    }

    /// Symbolic VMax (signed): constrain right == 7, derive left from a free
    /// 4x i32 vector, and verify that asserting result == [7, 7, 7, 7] forces
    /// every lane of left to be <= 7.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vec_int_max_symbolic_signed() {
        let ctx = SymContext::new_mock();

        // r = [7, 7, 7, 7] as i32x4
        let mut rv: u128 = 0;
        for i in 0..4u32 {
            rv |= (7u128) << (i * 32);
        }
        let r = RustBV::concrete(rv, 128);
        let l = RustBV::symbolic(&ctx, "vmax_l", 128);

        let result = VEXOps::binop(
            IROp::VMax {
                elem: IRType::I32,
                count: 4,
                signed: true,
            },
            l.clone(),
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);

        // Constrain result == [7, 7, 7, 7]; this only requires l <= 7 per lane,
        // so the constraint must remain SAT.
        let exp = RustBV::concrete(rv, 128);
        ctx.add_constraint(result.to_z3_ast().eq(exp.to_z3_ast()));
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
            IROp::VFAdd {
                elem: IRType::F32,
                count: 4,
            },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - expected).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                expected,
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
            IROp::VFDiv {
                elem: IRType::F64,
                count: 2,
            },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f64::from_bits(((got >> (i as u32 * 64)) & 0xFFFFFFFFFFFFFFFFu128) as u64);
            assert!(
                (lane - expected).abs() < 1e-12,
                "lane {} expected {}, got {}",
                i,
                expected,
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
            IROp::VFSqrt {
                elem: IRType::F32,
                count: 4,
            },
            RustBV::concrete(bits, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - expected).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                expected,
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
            IROp::VFAbs {
                elem: IRType::F32,
                count: 4,
            },
            RustBV::concrete(bits, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert_eq!(
                lane.to_bits(),
                expected.to_bits(),
                "lane {} expected {}, got {}",
                i,
                expected,
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
            IROp::VFMax {
                elem: IRType::F32,
                count: 4,
            },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f32::from_bits(((got >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - expected).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                expected,
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
            IROp::VFMin {
                elem: IRType::F64,
                count: 2,
            },
            RustBV::concrete(lv, 128),
            RustBV::concrete(rv, 128),
            &ctx,
        )
        .unwrap();
        let got = result.as_u128().unwrap();
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f64::from_bits(((got >> (i as u32 * 64)) & 0xFFFFFFFFFFFFFFFFu128) as u64);
            assert!(
                (lane - expected).abs() < 1e-12,
                "lane {} expected {}, got {}",
                i,
                expected,
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
        ctx.add_constraint(
            l.to_z3_ast()
                .eq(RustBV::concrete(lv_target, 128).to_z3_ast()),
        );

        let result = VEXOps::binop(
            IROp::VFAdd {
                elem: IRType::F32,
                count: 4,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert!(ctx.is_sat(), "expected SAT");

        let model = ctx.eval(&result).expect("eval(result) returned None");
        let exp = [3.0f32, 4.0, 5.0, 6.0];
        for (i, &expected) in exp.iter().enumerate() {
            let lane = f32::from_bits(((model >> (i as u32 * 32)) & 0xFFFFFFFF) as u32);
            assert!(
                (lane - expected).abs() < 1e-6,
                "lane {} expected {}, got {}",
                i,
                expected,
                lane
            );
        }
    }

    // ---- Newton-Raphson FP estimate / step (angr-iyon) ----
    //
    // VEX leaves Recip/RSqrt Est precision implementation-defined, and the Step
    // ops in angr Python have no dedicated handler — both branches collapse to
    // a fresh symbolic per lane. These tests pin the *shape* (lane count, width,
    // upper-lane passthrough for the SSE F0x4 variants) rather than the value.
    //
    // Convenience: turn a width-N RustBV into its u128 representation via the
    // solver so the result of a fresh-symbolic-per-lane op is observable.
    fn eval_v128(ctx: &SymContext, bv: &RustBV) -> u128 {
        assert!(ctx.is_sat(), "expected SAT for eval");
        ctx.eval(bv).expect("eval returned None")
    }
    fn eval_i64(ctx: &SymContext, bv: &RustBV) -> u64 {
        eval_v128(ctx, bv) as u64
    }
    fn eval_i32(ctx: &SymContext, bv: &RustBV) -> u32 {
        eval_v128(ctx, bv) as u32
    }

    /// SSE RCPSS shape: 128-bit result, lane 0 fresh symbolic (32-bit width),
    /// upper 96 bits passed through unchanged from the arg.
    #[test]
    fn test_vfrecip_est_s_f32_upper_passthrough() {
        let ctx = SymContext::new_mock();
        let upper96 = 0xDEAD_BEEF_CAFE_BABE_1234_5678u128;
        let arg = RustBV::concrete((upper96 << 32) | 0x4080_0000u128, 128); // lane0 = 4.0f32
        let result = VEXOps::unop(IROp::VFRecipEstS { elem: IRType::F32 }, arg, &ctx).unwrap();
        assert_eq!(result.width(), 128);
        let v = eval_v128(&ctx, &result);
        assert_eq!(v >> 32, upper96, "upper 96 bits must pass through");
    }

    /// SSE RSQRTSS shape: same as RCPSS — upper 96 bits passthrough, lane 0 fresh.
    #[test]
    fn test_vfrsqrt_est_s_f32_upper_passthrough() {
        let ctx = SymContext::new_mock();
        let upper96 = 0xAAAA_BBBB_CCCC_DDDD_EEEE_FFFFu128;
        let arg = RustBV::concrete((upper96 << 32) | 0x4400_0000u128, 128); // lane0 = 512.0f32
        let result = VEXOps::unop(IROp::VFRSqrtEstS { elem: IRType::F32 }, arg, &ctx).unwrap();
        assert_eq!(result.width(), 128);
        let v = eval_v128(&ctx, &result);
        assert_eq!(v >> 32, upper96, "upper 96 bits must pass through");
    }

    /// NEON Iop_RecipEst32Fx2 (D-reg, 2x f32 = 64-bit result).
    #[test]
    fn test_vfrecip_est_packed_f32x2_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 64);
        let result = VEXOps::unop(
            IROp::VFRecipEst {
                elem: IRType::F32,
                count: 2,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        let _ = eval_i64(&ctx, &result);
    }

    /// SSE RCPPS / NEON Q-reg Iop_RecipEst32Fx4 (4x f32 = 128-bit result).
    #[test]
    fn test_vfrecip_est_packed_f32x4_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 128);
        let result = VEXOps::unop(
            IROp::VFRecipEst {
                elem: IRType::F32,
                count: 4,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        let _ = eval_v128(&ctx, &result);
    }

    /// AVX Iop_RecipEst32Fx8 (8x f32 = 256-bit result).
    #[test]
    fn test_vfrecip_est_packed_f32x8_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 256);
        let result = VEXOps::unop(
            IROp::VFRecipEst {
                elem: IRType::F32,
                count: 8,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 256);
    }

    /// NEON Iop_RecipEst64Fx2 (2x f64 = 128-bit result).
    #[test]
    fn test_vfrecip_est_packed_f64x2_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 128);
        let result = VEXOps::unop(
            IROp::VFRecipEst {
                elem: IRType::F64,
                count: 2,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
    }

    /// Iop_RSqrtEst*: same widths as RecipEst, separate dispatch arm.
    #[test]
    fn test_vfrsqrt_est_packed_f64x2_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 128);
        let result = VEXOps::unop(
            IROp::VFRSqrtEst {
                elem: IRType::F64,
                count: 2,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
    }

    /// NEON Iop_RecipStep32Fx2: D-reg binary, 64-bit result.
    #[test]
    fn test_vfrecip_step_packed_f32x2_shape() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0u128, 64);
        let b = RustBV::concrete(0u128, 64);
        let result = VEXOps::binop(
            IROp::VFRecipStep {
                elem: IRType::F32,
                count: 2,
            },
            a,
            b,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        let _ = eval_i64(&ctx, &result);
    }

    /// NEON Iop_RecipStep64Fx2: Q-reg binary, 128-bit result.
    #[test]
    fn test_vfrecip_step_packed_f64x2_shape() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0u128, 128);
        let b = RustBV::concrete(0u128, 128);
        let result = VEXOps::binop(
            IROp::VFRecipStep {
                elem: IRType::F64,
                count: 2,
            },
            a,
            b,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
    }

    /// NEON Iop_RSqrtStep32Fx4: Q-reg binary, 128-bit result.
    #[test]
    fn test_vfrsqrt_step_packed_f32x4_shape() {
        let ctx = SymContext::new_mock();
        let a = RustBV::concrete(0u128, 128);
        let b = RustBV::concrete(0u128, 128);
        let result = VEXOps::binop(
            IROp::VFRSqrtStep {
                elem: IRType::F32,
                count: 4,
            },
            a,
            b,
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
    }

    /// NEON Iop_RecipEst32Ux2 (D-reg URECPE, 2x u32 = 64-bit result). Fresh
    /// symbolic per lane; shape is what matters here.
    #[test]
    fn test_virecip_est_packed_u32x2_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 64);
        let result = VEXOps::unop(IROp::VIRecipEst { count: 2 }, arg, &ctx).unwrap();
        assert_eq!(result.width(), 64);
        // Symbolic: should not be concrete because each lane was minted fresh.
        assert!(result.as_u128().is_none());
    }

    /// NEON Iop_RecipEst32Ux4 (Q-reg URECPE, 4x u32 = 128-bit result).
    #[test]
    fn test_virecip_est_packed_u32x4_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 128);
        let result = VEXOps::unop(IROp::VIRecipEst { count: 4 }, arg, &ctx).unwrap();
        assert_eq!(result.width(), 128);
        assert!(result.as_u128().is_none());
    }

    /// NEON Iop_RSqrtEst32Ux2 (D-reg URSQRTE, 2x u32 = 64-bit result).
    #[test]
    fn test_virsqrt_est_packed_u32x2_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 64);
        let result = VEXOps::unop(IROp::VIRSqrtEst { count: 2 }, arg, &ctx).unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.as_u128().is_none());
    }

    /// NEON Iop_RSqrtEst32Ux4 (Q-reg URSQRTE, 4x u32 = 128-bit result).
    #[test]
    fn test_virsqrt_est_packed_u32x4_shape() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0u128, 128);
        let result = VEXOps::unop(IROp::VIRSqrtEst { count: 4 }, arg, &ctx).unwrap();
        assert_eq!(result.width(), 128);
        assert!(result.as_u128().is_none());
    }

    /// End-to-end opcode-string routing for the integer NEON RecipEst /
    /// RSqrtEst ops — these resolve to the new VIRecipEst / VIRSqrtEst
    /// variants instead of NeonUnimplemented. Catches regressions where
    /// parse_vector and parse_neon_unimplemented fall out of sync.
    #[test]
    fn test_int_recip_rsqrt_opcode_routing() {
        use crate::vex::opcode_map::parse_opcode;
        for (name, count) in [("Iop_RecipEst32Ux2", 2u8), ("Iop_RecipEst32Ux4", 4)] {
            match parse_opcode(name) {
                IROp::VIRecipEst { count: c } => assert_eq!(c, count),
                other => panic!("{}: expected VIRecipEst, got {:?}", name, other),
            }
        }
        for (name, count) in [("Iop_RSqrtEst32Ux2", 2u8), ("Iop_RSqrtEst32Ux4", 4)] {
            match parse_opcode(name) {
                IROp::VIRSqrtEst { count: c } => assert_eq!(c, count),
                other => panic!("{}: expected VIRSqrtEst, got {:?}", name, other),
            }
        }
    }

    /// End-to-end opcode-string routing: all 16 FP Recip/RSqrt opcodes
    /// resolve to the new IROp variants (and not to NeonUnimplemented).
    /// Catches regressions where parse_float and parse_neon_unimplemented
    /// fall out of sync.
    #[test]
    fn test_recip_rsqrt_opcode_routing() {
        use crate::vex::opcode_map::parse_opcode;
        // Est family: F0x4 → VFRecipEstS/VFRSqrtEstS, others → packed.
        let recip_est_packed = [
            ("Iop_RecipEst32Fx2", IRType::F32, 2),
            ("Iop_RecipEst32Fx4", IRType::F32, 4),
            ("Iop_RecipEst32Fx8", IRType::F32, 8),
            ("Iop_RecipEst64Fx2", IRType::F64, 2),
        ];
        for (name, elem, count) in recip_est_packed {
            match parse_opcode(name) {
                IROp::VFRecipEst { elem: e, count: c } => {
                    assert_eq!(e, elem);
                    assert_eq!(c, count);
                }
                other => panic!("{}: expected VFRecipEst, got {:?}", name, other),
            }
        }
        assert!(matches!(
            parse_opcode("Iop_RecipEst32F0x4"),
            IROp::VFRecipEstS { elem: IRType::F32 }
        ));
        let rsqrt_est_packed = [
            ("Iop_RSqrtEst32Fx2", IRType::F32, 2),
            ("Iop_RSqrtEst32Fx4", IRType::F32, 4),
            ("Iop_RSqrtEst32Fx8", IRType::F32, 8),
            ("Iop_RSqrtEst64Fx2", IRType::F64, 2),
        ];
        for (name, elem, count) in rsqrt_est_packed {
            match parse_opcode(name) {
                IROp::VFRSqrtEst { elem: e, count: c } => {
                    assert_eq!(e, elem);
                    assert_eq!(c, count);
                }
                other => panic!("{}: expected VFRSqrtEst, got {:?}", name, other),
            }
        }
        assert!(matches!(
            parse_opcode("Iop_RSqrtEst32F0x4"),
            IROp::VFRSqrtEstS { elem: IRType::F32 }
        ));
        // Step family: NEON-only, no F0x4 form.
        let step_pairs = [
            ("Iop_RecipStep32Fx2", IRType::F32, 2),
            ("Iop_RecipStep32Fx4", IRType::F32, 4),
            ("Iop_RecipStep64Fx2", IRType::F64, 2),
        ];
        for (name, elem, count) in step_pairs {
            match parse_opcode(name) {
                IROp::VFRecipStep { elem: e, count: c } => {
                    assert_eq!(e, elem);
                    assert_eq!(c, count);
                }
                other => panic!("{}: expected VFRecipStep, got {:?}", name, other),
            }
        }
        let rsqrt_step_pairs = [
            ("Iop_RSqrtStep32Fx2", IRType::F32, 2),
            ("Iop_RSqrtStep32Fx4", IRType::F32, 4),
            ("Iop_RSqrtStep64Fx2", IRType::F64, 2),
        ];
        for (name, elem, count) in rsqrt_step_pairs {
            match parse_opcode(name) {
                IROp::VFRSqrtStep { elem: e, count: c } => {
                    assert_eq!(e, elem);
                    assert_eq!(c, count);
                }
                other => panic!("{}: expected VFRSqrtStep, got {:?}", name, other),
            }
        }
        // Suppress the unused-helper lint when running this test alone:
        let _ = eval_i32;
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
        let l = RustBV::concrete(make_v128_lane0(2.0f32.to_bits() as u128, upper), 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane {
                kind: FCmpKind::Eq,
                ty: IRType::F32,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpScalarLane {
                kind: FCmpKind::Eq,
                ty: IRType::F32,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        let v = res.as_u128().unwrap();
        assert_eq!(v & 0xFFFF_FFFF, 0, "lane0 should be 0 on false");
    }

    #[test]
    fn test_fcmp_scalar_lane_lt_f32_concrete() {
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(1.0f32.to_bits() as u128, 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane {
                kind: FCmpKind::Lt,
                ty: IRType::F32,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF);
    }

    #[test]
    fn test_fcmp_scalar_lane_le_f64_concrete_eq() {
        let ctx = SymContext::new_mock();
        let upper = 0x123456789ABCDEF0u128;
        let l = RustBV::concrete(make_v128_lane0_64(2.5f64.to_bits() as u128, upper), 128);
        let r = RustBV::concrete(2.5f64.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane {
                kind: FCmpKind::Le,
                ty: IRType::F64,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpScalarLane {
                kind: FCmpKind::Un,
                ty: IRType::F32,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF);
    }

    #[test]
    fn test_fcmp_scalar_lane_un_f64_concrete_ordered() {
        // Both ordered → CMPUNORD returns 0.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(1.0f64.to_bits() as u128, 128);
        let r = RustBV::concrete(2.0f64.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane {
                kind: FCmpKind::Un,
                ty: IRType::F64,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF_FFFF_FFFF, 0);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fcmp_scalar_lane_eq_f32_symbolic() {
        // Symbolic: constrain low32(result)==0xFFFFFFFF given right=2.0 and
        // some symbolic left → solver must pick left.lane0 == 2.0.
        let ctx = SymContext::new_mock();
        let l = RustBV::symbolic(&ctx, "fcmp_lane_l", 128);
        let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
        let res = VEXOps::binop(
            IROp::FCmpScalarLane {
                kind: FCmpKind::Eq,
                ty: IRType::F32,
            },
            l.clone(),
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let lo32 = res.extract(31, 0, &ctx);
        let true_mask = RustBV::concrete(0xFFFF_FFFF, 32);
        ctx.add_constraint(lo32.to_z3_ast().eq(true_mask.to_z3_ast()));
        assert!(
            ctx.is_sat(),
            "expected SAT after FCmpScalarLane Eq mask=all1s"
        );

        let model = ctx.eval(&l).expect("eval(l) returned None");
        let lane0 = f32::from_bits((model & 0xFFFF_FFFF) as u32);
        assert!(
            (lane0 - 2.0).abs() < 1e-6,
            "expected lane0==2.0, got {}",
            lane0
        );
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
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "fcom_x", 64);
        let five = RustBV::concrete(5.0f64.to_bits() as u128, 64);
        let res = VEXOps::binop(IROp::FComCC(IRType::F64), x.clone(), five, &ctx).unwrap();
        assert_eq!(res.width(), 32);
        let want = RustBV::concrete(0x01, 32);
        ctx.add_constraint(res.to_z3_ast().eq(want.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for FComCC(x, 5.0) == LT");
        let model_x = ctx.eval(&x).expect("eval(x) None");
        let xf = f64::from_bits(model_x as u64);
        assert!(
            !xf.is_nan() && xf < 5.0,
            "expected x < 5.0 and not NaN, got {}",
            xf
        );
    }

    // ---- FCmpVecPacked (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}) ----

    /// Pack four f32 values into a single 128-bit vector (lane 0 first).
    #[allow(clippy::identity_op)] // explicit lane shifts (incl. `<< 0`) keep the helper symmetric
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
    #[allow(clippy::identity_op)] // explicit 4-lane mask layout reads better than the minimized form
    fn test_fcmp_packed_eq_32fx4_concrete() {
        // CMPEQPS lane-by-lane: lanes 0 and 2 equal, lanes 1 and 3 differ.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(1.0, 2.0, 3.0, 4.0), 128);
        let r = RustBV::concrete(pack_4xf32(1.0, 5.0, 3.0, 7.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked {
                kind: FCmpKind::Eq,
                elem: IRType::F32,
                count: 4,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
        let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 0.0, 4.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked {
                kind: FCmpKind::Lt,
                elem: IRType::F32,
                count: 4,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpVecPacked {
                kind: FCmpKind::Gt,
                elem: IRType::F32,
                count: 4,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        let v = res.as_u128().unwrap();
        let expected: u128 = 0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 96);
        assert_eq!(v, expected);
    }

    #[test]
    #[allow(clippy::identity_op)] // explicit 4-lane mask layout reads better than the minimized form
    fn test_fcmp_packed_ge_32fx4_concrete() {
        // CMPGEPS: 5.0>=2.0 T, 3.0>=3.0 T, 1.0>=2.0 F, 4.0>=4.0 T.
        let ctx = SymContext::new_mock();
        let l = RustBV::concrete(pack_4xf32(5.0, 3.0, 1.0, 4.0), 128);
        let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 2.0, 4.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked {
                kind: FCmpKind::Ge,
                elem: IRType::F32,
                count: 4,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpVecPacked {
                kind: FCmpKind::Le,
                elem: IRType::F64,
                count: 2,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpVecPacked {
                kind: FCmpKind::Un,
                elem: IRType::F32,
                count: 4,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpVecPacked {
                kind: FCmpKind::Un,
                elem: IRType::F64,
                count: 2,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpVecPacked {
                kind: FCmpKind::Eq,
                elem: IRType::F32,
                count: 2,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
            IROp::FCmpVecPacked {
                kind: FCmpKind::Gt,
                elem: IRType::F32,
                count: 2,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 64);
        let v = res.as_u128().unwrap();
        assert_eq!(v as u64, 0xFFFF_FFFFu64);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fcmp_packed_lt_32fx4_symbolic() {
        // Symbolic vector `l` against concrete `r`. Constrain low lane mask to all-1s
        // → solver must satisfy lane 0 of l < lane 0 of r (= 5.0). Other lanes free.
        let ctx = SymContext::new_mock();
        let l = RustBV::symbolic(&ctx, "fpkd_lt_l", 128);
        let r = RustBV::concrete(pack_4xf32(5.0, 1.0, 1.0, 1.0), 128);
        let res = VEXOps::binop(
            IROp::FCmpVecPacked {
                kind: FCmpKind::Lt,
                elem: IRType::F32,
                count: 4,
            },
            l.clone(),
            r,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let lane0_mask = res.extract(31, 0, &ctx);
        let true_mask = RustBV::concrete(0xFFFF_FFFF, 32);
        ctx.add_constraint(lane0_mask.to_z3_ast().eq(true_mask.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for lane0 LT");
        let model = ctx.eval(&l).expect("eval(l) None");
        let lane0 = f32::from_bits((model & 0xFFFF_FFFF) as u32);
        assert!(
            lane0 < 5.0 && !lane0.is_nan(),
            "expected lane0 < 5.0 and not NaN, got {}",
            lane0
        );
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
        let res = VEXOps::binop(
            IROp::VMul {
                elem: IRType::I8,
                count: 8,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
        let res = VEXOps::binop(
            IROp::VMul {
                elem: IRType::I8,
                count: 16,
            },
            l,
            r,
            &ctx,
        )
        .unwrap();
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
        for (lane, expected) in [
            (0u128, 0x11),
            (1, 0x22),
            (2, 0x33),
            (3, 0x44),
            (4, 0x55),
            (5, 0x66),
            (6, 0x77),
            (7, 0x88),
        ] {
            let idx = RustBV::concrete(lane, 8);
            let res = VEXOps::binop(
                IROp::VGetElem {
                    elem: IRType::I8,
                    count: 8,
                },
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
        payload |= 0xDEAD_u128 << (3 * 16);
        payload |= 0xBEEF_u128 << (7 * 16);
        let vec = RustBV::concrete(payload, 128);
        let res = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I16,
                count: 8,
            },
            vec.clone(),
            RustBV::concrete(3, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 16);
        assert_eq!(res.as_u128().unwrap(), 0xDEAD);

        let res = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I16,
                count: 8,
            },
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
            IROp::VGetElem {
                elem: IRType::I64,
                count: 2,
            },
            vec.clone(),
            RustBV::concrete(0, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(r0.as_u128().unwrap(), lo);
        let r1 = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I64,
                count: 2,
            },
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
            IROp::VSetElem {
                elem: IRType::I8,
                count: 8,
            },
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
            IROp::VSetElem {
                elem: IRType::I16,
                count: 8,
            },
            vec,
            RustBV::concrete(5, 8),
            RustBV::concrete(0xCAFE, 16),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        assert_eq!(res.as_u128().unwrap(), 0xCAFE_u128 << (5 * 16));
    }

    #[test]
    fn test_vset_elem_preserves_other_lanes() {
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0xDEAD_BEEF_CAFE_F00Du128, 64);
        // Overwrite lane 2 (byte 2) with 0x77.
        let res = VEXOps::binop_with_rm(
            IROp::VSetElem {
                elem: IRType::I8,
                count: 8,
            },
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
            IROp::VSetElem {
                elem: IRType::I32,
                count: 4,
            },
            vec,
            RustBV::concrete(2, 8),
            RustBV::concrete(0x1234_5678, 32),
            &ctx,
        )
        .unwrap();
        let lane = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I32,
                count: 4,
            },
            inserted,
            RustBV::concrete(2, 8),
            &ctx,
        )
        .unwrap();
        assert_eq!(lane.as_u128().unwrap(), 0x1234_5678);
    }

    // -------------------------------------------------------------------
    // NEON Dup / Widen / Narrow / QNarrow tests (angr-hzs0)
    // -------------------------------------------------------------------

    #[test]
    fn test_vdup_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0xABu128, 8);
        let res = VEXOps::unop(
            IROp::VDup {
                elem: IRType::I8,
                count: 8,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 64);
        assert_eq!(res.as_u128().unwrap(), 0xABAB_ABAB_ABAB_ABABu128);
    }

    #[test]
    fn test_vdup_16x8_concrete() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0xCAFEu128, 16);
        let res = VEXOps::unop(
            IROp::VDup {
                elem: IRType::I16,
                count: 8,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let expected: u128 = (0..8).fold(0u128, |acc, i| acc | (0xCAFE_u128 << (i * 16)));
        assert_eq!(res.as_u128().unwrap(), expected);
    }

    #[test]
    fn test_vdup_32x4_concrete() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::concrete(0xDEAD_BEEFu128, 32);
        let res = VEXOps::unop(
            IROp::VDup {
                elem: IRType::I32,
                count: 4,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let expected: u128 = (0..4).fold(0u128, |acc, i| acc | (0xDEAD_BEEF_u128 << (i * 32)));
        assert_eq!(res.as_u128().unwrap(), expected);
    }

    #[test]
    fn test_vwiden_8sto16x8_signed() {
        let ctx = SymContext::new_mock();
        // 8 lanes of I8: lane 0 = 0xFF (= -1 signed), lane 1 = 0x7F (= 127),
        // lane 2 = 0x80 (= -128), rest = 0.
        let arg_val: u128 = 0xFFu128 | (0x7Fu128 << 8) | (0x80u128 << 16);
        let arg = RustBV::concrete(arg_val, 64);
        let res = VEXOps::unop(
            IROp::VWiden {
                from: IRType::I8,
                count: 8,
                signed: true,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let v = res.as_u128().unwrap();
        // Lane 0: 0xFFFF (sign-ext of 0xFF), Lane 1: 0x007F, Lane 2: 0xFF80.
        assert_eq!(v & 0xFFFF, 0xFFFF, "lane 0");
        assert_eq!((v >> 16) & 0xFFFF, 0x007F, "lane 1");
        assert_eq!((v >> 32) & 0xFFFF, 0xFF80, "lane 2");
        assert_eq!((v >> 48) & 0xFFFF, 0x0000, "lane 3");
    }

    #[test]
    fn test_vwiden_8uto16x8_unsigned() {
        let ctx = SymContext::new_mock();
        let arg_val: u128 = 0xFFu128 | (0x80u128 << 8);
        let arg = RustBV::concrete(arg_val, 64);
        let res = VEXOps::unop(
            IROp::VWiden {
                from: IRType::I8,
                count: 8,
                signed: false,
            },
            arg,
            &ctx,
        )
        .unwrap();
        let v = res.as_u128().unwrap();
        assert_eq!(v & 0xFFFF, 0x00FF, "lane 0 zero-extended");
        assert_eq!((v >> 16) & 0xFFFF, 0x0080, "lane 1 zero-extended");
    }

    #[test]
    fn test_vwiden_32sto64x2_signed() {
        let ctx = SymContext::new_mock();
        // Lane 0 = 0x80000000 (= INT_MIN signed), Lane 1 = 0x12345678.
        let arg_val: u128 = 0x8000_0000u128 | (0x1234_5678u128 << 32);
        let arg = RustBV::concrete(arg_val, 64);
        let res = VEXOps::unop(
            IROp::VWiden {
                from: IRType::I32,
                count: 2,
                signed: true,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let v = res.as_u128().unwrap();
        let lo = v as u64;
        let hi = (v >> 64) as u64;
        assert_eq!(lo, 0xFFFF_FFFF_8000_0000u64, "lane 0 sign-extended");
        assert_eq!(hi, 0x0000_0000_1234_5678u64, "lane 1 zero-positive");
    }

    #[test]
    fn test_vnarrow_un_16to8x8_concrete() {
        let ctx = SymContext::new_mock();
        // Input V128 = 8 lanes of I16: 0x1122, 0x3344, 0x5566, 0x7788, 0x99AA,
        // 0xBBCC, 0xDDEE, 0xFF00.
        let lanes_in: [u16; 8] = [
            0x1122, 0x3344, 0x5566, 0x7788, 0x99AA, 0xBBCC, 0xDDEE, 0xFF00,
        ];
        let mut v: u128 = 0;
        for (i, &lane) in lanes_in.iter().enumerate() {
            v |= (lane as u128) << (i * 16);
        }
        let arg = RustBV::concrete(v, 128);
        let res = VEXOps::unop(
            IROp::VNarrowUn {
                from: IRType::I16,
                count: 8,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 64);
        let out = res.as_u128().unwrap();
        // Each output lane = low byte of input lane.
        for (i, &lane) in lanes_in.iter().enumerate() {
            let got = (out >> (i * 8)) & 0xFF;
            assert_eq!(got, (lane & 0xFF) as u128, "out lane {}", i);
        }
    }

    #[test]
    fn test_vnarrow_bin_16to8x16_concrete() {
        let ctx = SymContext::new_mock();
        // Each input is V128 with 8 lanes of I16. left lanes -> output lanes 0..8;
        // right lanes -> output lanes 8..16.
        let lanes_l: [u16; 8] = [
            0x0011, 0x0022, 0x0033, 0x0044, 0x0055, 0x0066, 0x0077, 0x0088,
        ];
        let lanes_r: [u16; 8] = [
            0x0099, 0x00AA, 0x00BB, 0x00CC, 0x00DD, 0x00EE, 0x00FF, 0x0001,
        ];
        let mut l: u128 = 0;
        let mut r: u128 = 0;
        for (i, &lane) in lanes_l.iter().enumerate() {
            l |= (lane as u128) << (i * 16);
        }
        for (i, &lane) in lanes_r.iter().enumerate() {
            r |= (lane as u128) << (i * 16);
        }
        let res = VEXOps::binop(
            IROp::VNarrowBin {
                from: IRType::I16,
                count: 16,
            },
            RustBV::concrete(l, 128),
            RustBV::concrete(r, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let out = res.as_u128().unwrap();
        for (i, &lane) in lanes_l.iter().enumerate() {
            let got = (out >> (i * 8)) & 0xFF;
            assert_eq!(got, (lane & 0xFF) as u128, "left lane {}", i);
        }
        for (i, &lane) in lanes_r.iter().enumerate() {
            let got = (out >> ((8 + i) * 8)) & 0xFF;
            assert_eq!(got, (lane & 0xFF) as u128, "right lane {}", i);
        }
    }

    #[test]
    fn test_vqnarrow_un_16sto8sx8_saturates() {
        let ctx = SymContext::new_mock();
        // signed I16 source, signed I8 target. Range [-128, 127].
        // Lane 0 = 1000 (clamps to 127), lane 1 = -200 (clamps to -128 = 0x80),
        // lane 2 = 50 (passes through), lane 3 = -50 (passes through).
        let v: u128 = (1000i16 as u16 as u128)
            | ((-200i16 as u16 as u128) << 16)
            | ((50i16 as u16 as u128) << 32)
            | ((-50i16 as u16 as u128) << 48);
        let arg = RustBV::concrete(v, 128);
        let res = VEXOps::unop(
            IROp::VQNarrowUn {
                from: IRType::I16,
                count: 8,
                src_signed: true,
                dst_signed: true,
            },
            arg,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 64);
        let out = res.as_u128().unwrap();
        assert_eq!((out & 0xFF) as u8, 127, "lane 0 saturated to 127");
        assert_eq!(((out >> 8) & 0xFF) as u8, 0x80, "lane 1 saturated to -128");
        assert_eq!(((out >> 16) & 0xFF) as u8, 50, "lane 2 unchanged");
        assert_eq!(
            ((out >> 24) & 0xFF) as u8,
            (-50i8) as u8,
            "lane 3 unchanged"
        );
    }

    #[test]
    fn test_vqnarrow_un_16sto8ux8_signed_to_unsigned() {
        let ctx = SymContext::new_mock();
        // Signed source -> unsigned dst. Range [0, 255].
        // -1 (0xFFFF) -> 0; 300 -> 255; 100 -> 100.
        let v: u128 = (0xFFFFu128) | ((300u128) << 16) | ((100u128) << 32);
        let arg = RustBV::concrete(v, 128);
        let res = VEXOps::unop(
            IROp::VQNarrowUn {
                from: IRType::I16,
                count: 8,
                src_signed: true,
                dst_signed: false,
            },
            arg,
            &ctx,
        )
        .unwrap();
        let out = res.as_u128().unwrap();
        assert_eq!((out & 0xFF) as u8, 0, "negative -> 0");
        assert_eq!(((out >> 8) & 0xFF) as u8, 255, "300 -> 255");
        assert_eq!(((out >> 16) & 0xFF) as u8, 100, "passes through");
    }

    #[test]
    #[allow(clippy::identity_op)] // explicit 8-lane layout reads better than the minimized form
    fn test_vqnarrow_un_16uto8ux8_unsigned() {
        let ctx = SymContext::new_mock();
        // Unsigned source -> unsigned dst. Range [0, 255]. 256 saturates to 255.
        let v: u128 = (256u128) | ((100u128) << 16) | ((0u128) << 32) | ((0xFFFFu128) << 48);
        let arg = RustBV::concrete(v, 128);
        let res = VEXOps::unop(
            IROp::VQNarrowUn {
                from: IRType::I16,
                count: 8,
                src_signed: false,
                dst_signed: false,
            },
            arg,
            &ctx,
        )
        .unwrap();
        let out = res.as_u128().unwrap();
        assert_eq!((out & 0xFF) as u8, 255, "256 -> 255");
        assert_eq!(((out >> 8) & 0xFF) as u8, 100);
        assert_eq!(((out >> 16) & 0xFF) as u8, 0);
        assert_eq!(((out >> 24) & 0xFF) as u8, 255, "0xFFFF -> 255");
    }

    #[test]
    fn test_vqnarrow_bin_16sto8sx16_two_inputs() {
        let ctx = SymContext::new_mock();
        // 8 lanes per input of signed I16. Left lane 0 = 200 (>127 -> 127),
        // right lane 7 = -300 (< -128 -> -128).
        let mut l: u128 = 0;
        let mut r: u128 = 0;
        l |= 200u128;
        l |= (50i16 as u16 as u128) << 16;
        r |= 5i16 as u16 as u128;
        r |= ((-300i16) as u16 as u128) << (7 * 16);

        let res = VEXOps::binop(
            IROp::VQNarrowBin {
                from: IRType::I16,
                count: 16,
                src_signed: true,
                dst_signed: true,
            },
            RustBV::concrete(l, 128),
            RustBV::concrete(r, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 128);
        let out = res.as_u128().unwrap();
        // Left lane 0 -> output byte 0: 127.
        assert_eq!((out & 0xFF) as u8, 127, "left lane 0 sat to 127");
        // Left lane 1 -> output byte 1: 50.
        assert_eq!(((out >> 8) & 0xFF) as u8, 50, "left lane 1 unchanged");
        // Right lane 0 -> output byte 8: 5.
        assert_eq!(((out >> 64) & 0xFF) as u8, 5, "right lane 0 unchanged");
        // Right lane 7 -> output byte 15: -128.
        assert_eq!(
            ((out >> (15 * 8)) & 0xFF) as u8,
            0x80,
            "right lane 7 sat to -128"
        );
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vdup_symbolic_arg() {
        // Symbolic 8-bit value, dup to 8x8. Constrain the output to a known
        // pattern and check the solver picks the right scalar.
        let ctx = SymContext::new_mock();
        let arg = RustBV::symbolic(&ctx, "dup_arg", 8);
        let res = VEXOps::unop(
            IROp::VDup {
                elem: IRType::I8,
                count: 8,
            },
            arg.clone(),
            &ctx,
        )
        .unwrap();
        let target = RustBV::concrete(0x4242_4242_4242_4242u128, 64);
        ctx.add_constraint(res.to_z3_ast().eq(target.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for dup to 0x42 broadcast");
        let model_arg = ctx.eval(&arg).expect("eval(arg) None");
        assert_eq!(model_arg, 0x42);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vqnarrow_un_symbolic_saturates() {
        // Symbolic I16 saturating to signed I8. Constrain output lane to 127
        // and require source > 127 to confirm saturation kicked in.
        let ctx = SymContext::new_mock();
        let arg = RustBV::symbolic(&ctx, "qn_arg", 128); // 8 lanes I16
        let res = VEXOps::unop(
            IROp::VQNarrowUn {
                from: IRType::I16,
                count: 8,
                src_signed: true,
                dst_signed: true,
            },
            arg.clone(),
            &ctx,
        )
        .unwrap();
        // Constrain output lane 0 = 127 AND source lane 0 = 200.
        let out_lane0 = res.extract(7, 0, &ctx);
        ctx.add_constraint(
            out_lane0
                .to_z3_ast()
                .eq(RustBV::concrete(127, 8).to_z3_ast()),
        );
        let src_lane0 = arg.extract(15, 0, &ctx);
        ctx.add_constraint(
            src_lane0
                .to_z3_ast()
                .eq(RustBV::concrete(200, 16).to_z3_ast()),
        );
        assert!(
            ctx.is_sat(),
            "expected SAT: src lane 0 = 200 saturates to 127"
        );
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vget_elem_symbolic_idx() {
        // Build a concrete vector with distinct lane values, then read
        // through a symbolic idx and constrain it to return a specific lane.
        let ctx = SymContext::new_mock();
        let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
        let sym_idx = RustBV::symbolic(&ctx, "get_idx", 8);
        let res = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I8,
                count: 8,
            },
            vec,
            sym_idx.clone(),
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 8);
        // Constrain result to 0x66 → solver must pick idx == 5.
        let target = RustBV::concrete(0x66, 8);
        ctx.add_constraint(res.to_z3_ast().eq(target.to_z3_ast()));
        assert!(ctx.is_sat(), "expected SAT for lane==0x66");
        let model_idx = ctx.eval(&sym_idx).expect("eval(idx) None");
        // idx must be 5 mod 8 (modulo because ITE chain ignores high bits).
        assert_eq!(model_idx & 0x7, 5, "expected idx&7 == 5, got {}", model_idx);
    }

    // =========================================================================
    // angr-tukg.1 — NEON saturating add/sub (VQAdd / VQSub).
    // =========================================================================

    /// Iop_QAdd8Sx8 — signed 8-bit saturating add over 8 lanes. Exercises:
    /// non-overflowing add, positive overflow → INT8_MAX (0x7F), negative
    /// overflow → INT8_MIN (0x80), exact-boundary cases.
    #[test]
    fn test_vqadd_8sx8_concrete() {
        let ctx = SymContext::new_mock();
        // Lane layout (LSB→MSB):
        //   0:  100 + 100 = 200, signed overflow → clamp to +127 (0x7F).
        //   1:  -100 + -100 = -200, signed underflow → clamp to -128 (0x80).
        //   2:   50 + 60 = 110, no overflow → 110 (0x6E).
        //   3:  -50 + -60 = -110, no overflow → -110 (0x92).
        //   4:  127 + 1  = INT_MAX+1 → clamp to +127.
        //   5: -128 + -1 = INT_MIN-1 → clamp to -128.
        //   6:  127 + -1 = 126 (no overflow).
        //   7: -128 + 1  = -127 (no overflow).
        let lanes_a: [i8; 8] = [100, -100, 50, -50, 127, -128, 127, -128];
        let lanes_b: [i8; 8] = [100, -100, 60, -60, 1, -1, -1, 1];
        let lanes_e: [i8; 8] = [127, -128, 110, -110, 127, -128, 126, -127];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= ((lanes_a[i] as u8) as u128) << (i * 8);
            b |= ((lanes_b[i] as u8) as u128) << (i * 8);
            e |= ((lanes_e[i] as u8) as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VQAdd {
                elem: IRType::I8,
                count: 8,
                signed: true,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_QAdd8Ux8 — unsigned 8-bit saturating add. Overflow clamps to 0xFF.
    #[test]
    fn test_vqadd_8ux8_concrete() {
        let ctx = SymContext::new_mock();
        // 0: 200+100 = 300 → clamp to 255 (0xFF).
        // 1: 255+1   = 256 → clamp to 255.
        // 2: 0+0 → 0.
        // 3: 200+55 = 255 (boundary, no clamp).
        // 4: 200+56 = 256 → clamp.
        // 5: 50+50  = 100.
        // 6,7: 0 fillers.
        let lanes_a: [u8; 8] = [200, 255, 0, 200, 200, 50, 0, 0];
        let lanes_b: [u8; 8] = [100, 1, 0, 55, 56, 50, 0, 0];
        let lanes_e: [u8; 8] = [255, 255, 0, 255, 255, 100, 0, 0];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (lanes_a[i] as u128) << (i * 8);
            b |= (lanes_b[i] as u128) << (i * 8);
            e |= (lanes_e[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VQAdd {
                elem: IRType::I8,
                count: 8,
                signed: false,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_QSub16Sx4 — signed 16-bit saturating sub. Verify both overflow
    /// directions and the no-overflow path.
    #[test]
    fn test_vqsub_16sx4_concrete() {
        let ctx = SymContext::new_mock();
        // 0:  30000 - (-10000) = 40000 → clamp to 32767.
        // 1: -30000 - 10000    = -40000 → clamp to -32768.
        // 2: 100 - 50 = 50 (no overflow).
        // 3: -100 - (-50) = -50 (no overflow).
        let lanes_a: [i16; 4] = [30000, -30000, 100, -100];
        let lanes_b: [i16; 4] = [-10000, 10000, 50, -50];
        let lanes_e: [i16; 4] = [32767, -32768, 50, -50];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..4 {
            a |= ((lanes_a[i] as u16) as u128) << (i * 16);
            b |= ((lanes_b[i] as u16) as u128) << (i * 16);
            e |= ((lanes_e[i] as u16) as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VQSub {
                elem: IRType::I16,
                count: 4,
                signed: true,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_QSub32Ux4 — unsigned 32-bit saturating sub on a 128-bit Q-reg.
    /// Underflow clamps to 0.
    #[test]
    fn test_vqsub_32ux4_concrete() {
        let ctx = SymContext::new_mock();
        let lanes_a: [u32; 4] = [100, 0xFFFF_FFFF, 1, 0];
        let lanes_b: [u32; 4] = [50, 1, 5, 5]; // underflow on lane 2 and 3
        let lanes_e: [u32; 4] = [50, 0xFFFF_FFFE, 0, 0];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..4 {
            a |= (lanes_a[i] as u128) << (i * 32);
            b |= (lanes_b[i] as u128) << (i * 32);
            e |= (lanes_e[i] as u128) << (i * 32);
        }
        let result = VEXOps::binop(
            IROp::VQSub {
                elem: IRType::I32,
                count: 4,
                signed: false,
            },
            RustBV::concrete(a, 128),
            RustBV::concrete(b, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Symbolic universality: Iop_QAdd8Sx8 must produce the same bits as the
    /// claripy reference at `_op_generic_QAdd` for any 64-bit input. Built per
    /// lane using the explicit sign-bit overflow formula (cap_cond + cap).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vqadd_8sx8_symbolic_matches_python_ref() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vqadd_a", 64);
        let b = RustBV::symbolic(&ctx, "vqadd_b", 64);
        let got = VEXOps::binop(
            IROp::VQAdd {
                elem: IRType::I8,
                count: 8,
                signed: true,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();

        // Reference: per-lane signed saturating add as in irop.py.
        let smax = RustBV::concrete(0x7F, 8);
        let smin = RustBV::concrete(0x80, 8);
        let mut lanes = Vec::with_capacity(8);
        for i in 0..8u32 {
            let lo = i * 8;
            let hi = lo + 7;
            let a_lane = a.extract(hi, lo, &ctx);
            let b_lane = b.extract(hi, lo, &ctx);
            let res = a_lane.clone().add_into(b_lane.clone(), &ctx);
            let top_a = a_lane.extract(7, 7, &ctx);
            let top_b = b_lane.extract(7, 7, &ctx);
            let top_r = res.extract(7, 7, &ctx);
            // ~(top_a ^ top_b) & (top_a ^ top_r) == 1
            let signs_match = top_a.clone().xor_into(top_b, &ctx).not_into(&ctx);
            let r_flipped = top_a.xor_into(top_r.clone(), &ctx);
            let overflow = signs_match
                .and_into(r_flipped, &ctx)
                .eq(&RustBV::concrete(1, 1), &ctx);
            let cap = top_r
                .eq(&RustBV::concrete(1, 1), &ctx)
                .ite(&smax, &smin, &ctx);
            lanes.push(overflow.ite(&cap, &res, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VQAdd 8Sx8 must match the claripy QAdd reference for all 64-bit inputs"
        );
        ctx.pop();
    }

    /// Parse routing: Iop_QAdd / Iop_QSub variants land on VQAdd / VQSub with
    /// the expected (elem, count, signed) decomposition. Covers a sampling
    /// across D-reg (total=64) and Q-reg (total=128) shapes plus both
    /// signedness conventions.
    #[test]
    fn test_parse_vqaddsub_routing() {
        use crate::vex::opcode_map::parse_opcode;

        let qadd_cases: &[(&str, IRType, u8, bool)] = &[
            ("Iop_QAdd8Sx8", IRType::I8, 8, true),
            ("Iop_QAdd16Ux4", IRType::I16, 4, false),
            ("Iop_QAdd32Sx2", IRType::I32, 2, true),
            ("Iop_QAdd64Ux1", IRType::I64, 1, false),
            ("Iop_QAdd8Ux16", IRType::I8, 16, false),
            ("Iop_QAdd16Sx8", IRType::I16, 8, true),
            ("Iop_QAdd64Sx2", IRType::I64, 2, true),
        ];
        for (op, e, c, s) in qadd_cases {
            match parse_opcode(op) {
                IROp::VQAdd {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(signed, *s, "{}: signed", op);
                }
                other => panic!("{}: expected VQAdd, got {:?}", op, other),
            }
        }

        let qsub_cases: &[(&str, IRType, u8, bool)] = &[
            ("Iop_QSub8Sx8", IRType::I8, 8, true),
            ("Iop_QSub32Ux4", IRType::I32, 4, false),
            ("Iop_QSub64Sx2", IRType::I64, 2, true),
        ];
        for (op, e, c, s) in qsub_cases {
            match parse_opcode(op) {
                IROp::VQSub {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(signed, *s, "{}: signed", op);
                }
                other => panic!("{}: expected VQSub, got {:?}", op, other),
            }
        }
    }

    // =========================================================================
    // angr-tukg.2 — NEON pairwise add/min/max (VPwAdd / VPwAddL / VPwMin / VPwMax).
    // =========================================================================

    /// Iop_PwAdd16x4 — pairwise add over 4 lanes of 16 bits, two 64-bit
    /// sources. Output[0..2] from `a`, output[2..4] from `b`. Tests both
    /// halves and a sample of values.
    #[test]
    fn test_vpwadd_16x4_concrete() {
        let ctx = SymContext::new_mock();
        // a lanes (LSB→MSB): 1, 2, 3, 4 → pairs (1+2, 3+4) = 3, 7.
        // b lanes:           10, 20, 100, 200 → pairs (30, 300).
        // Output (LSB→MSB): 3, 7, 30, 300.
        let a_lanes: [u16; 4] = [1, 2, 3, 4];
        let b_lanes: [u16; 4] = [10, 20, 100, 200];
        let e_lanes: [u16; 4] = [3, 7, 30, 300];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..4 {
            a |= (a_lanes[i] as u128) << (i * 16);
            b |= (b_lanes[i] as u128) << (i * 16);
            e |= (e_lanes[i] as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VPwAdd {
                elem: IRType::I16,
                count: 4,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PwAdd8x16 — Q-reg variant: 16 lanes of 8 bits. Output[0..8] from
    /// `a`, output[8..16] from `b`. Wrap-around per lane (e.g. 0xFF+0x01=0x00).
    #[test]
    fn test_vpwadd_8x16_concrete() {
        let ctx = SymContext::new_mock();
        // a: pairs (10+20, 30+40, ..., 70+80) → 30, 70, 110, ..., 0xFF+0x01=0x00.
        // Use explicit lanes for clarity.
        let a_lanes: [u8; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 0xFF, 0x01, 0, 0, 0, 0, 0, 0];
        let b_lanes: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0x80, 0x80, 0, 0, 0, 0];
        // a-half output (8 lanes): (10+20, 30+40, 50+60, 70+80, 0xFF+0x01=0x00, 0, 0, 0)
        //                       = (30, 70, 110, 150, 0, 0, 0, 0)
        // b-half output (8 lanes): (1+2, 3+4, 5+6, 7+8, 9+10, 0x80+0x80=0x00, 0, 0)
        //                       = (3, 7, 11, 15, 19, 0, 0, 0)
        let e_lanes: [u8; 16] = [30, 70, 110, 150, 0, 0, 0, 0, 3, 7, 11, 15, 19, 0, 0, 0];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..16 {
            a |= (a_lanes[i] as u128) << (i * 8);
            b |= (b_lanes[i] as u128) << (i * 8);
            e |= (e_lanes[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VPwAdd {
                elem: IRType::I8,
                count: 16,
            },
            RustBV::concrete(a, 128),
            RustBV::concrete(b, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PwAddL8Sx8 — signed widening pairwise add. Input is 8 lanes × 8 bits;
    /// output is 4 lanes × 16 bits. Negative sources must sign-extend before
    /// adding so the sum doesn't lose its sign.
    #[test]
    fn test_vpwaddl_8sx8_concrete() {
        let ctx = SymContext::new_mock();
        // a lanes (signed i8): -100, -100, 100, 100, -1, -1, 1, 1
        // Pairs: -200, 200, -2, 2 (as i16).
        let a_lanes: [i8; 8] = [-100, -100, 100, 100, -1, -1, 1, 1];
        let e_lanes: [i16; 4] = [-200, 200, -2, 2];
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for (i, &lane) in a_lanes.iter().enumerate() {
            a |= ((lane as u8) as u128) << (i * 8);
        }
        for (i, &lane) in e_lanes.iter().enumerate() {
            e |= ((lane as u16) as u128) << (i * 16);
        }
        let result = VEXOps::unop(
            IROp::VPwAddL {
                elem: IRType::I8,
                count: 8,
                signed: true,
            },
            RustBV::concrete(a, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PwAddL8Ux16 — unsigned widening pairwise add, Q-reg. Verifies that
    /// 0xFF + 0xFF widens to 0x01FE rather than overflowing in 8 bits.
    #[test]
    fn test_vpwaddl_8ux16_concrete() {
        let ctx = SymContext::new_mock();
        let a_lanes: [u8; 16] = [
            0xFF, 0xFF, 0x80, 0x80, 0x01, 0x02, 0, 0, 100, 50, 200, 100, 0, 0, 0, 0,
        ];
        let e_lanes: [u16; 8] = [0x01FE, 0x0100, 0x0003, 0, 150, 300, 0, 0];
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for (i, &lane) in a_lanes.iter().enumerate() {
            a |= (lane as u128) << (i * 8);
        }
        for (i, &lane) in e_lanes.iter().enumerate() {
            e |= (lane as u128) << (i * 16);
        }
        let result = VEXOps::unop(
            IROp::VPwAddL {
                elem: IRType::I8,
                count: 16,
                signed: false,
            },
            RustBV::concrete(a, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PwMin16Sx4 — pairwise signed min. Exercises mixed-sign pairs and
    /// confirms output[2..4] comes from `b`.
    #[test]
    fn test_vpwmin_16sx4_concrete() {
        let ctx = SymContext::new_mock();
        // a: -100, 100, 200, -200 → pairs min(-100,100)=-100, min(200,-200)=-200
        // b: 30000, -1, 0, 0       → pairs min=-1, min=0
        let a_lanes: [i16; 4] = [-100, 100, 200, -200];
        let b_lanes: [i16; 4] = [30000, -1, 0, 0];
        let e_lanes: [i16; 4] = [-100, -200, -1, 0];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..4 {
            a |= ((a_lanes[i] as u16) as u128) << (i * 16);
            b |= ((b_lanes[i] as u16) as u128) << (i * 16);
            e |= ((e_lanes[i] as u16) as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VPwMin {
                elem: IRType::I16,
                count: 4,
                signed: true,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PwMax8Ux8 — pairwise unsigned max. Distinguishes 0x80 (signed -128)
    /// from 0x01 to confirm unsigned compare.
    #[test]
    fn test_vpwmax_8ux8_concrete() {
        let ctx = SymContext::new_mock();
        // a: 0x80, 0x01, 0x10, 0x10, 0xFF, 0xFF, 0, 0
        //   → unsigned max pairs: 0x80, 0x10, 0xFF, 0
        // b: 0, 0, 0, 0, 50, 60, 70, 80
        //   → max: 0, 0, 60, 80
        let a_lanes: [u8; 8] = [0x80, 0x01, 0x10, 0x10, 0xFF, 0xFF, 0, 0];
        let b_lanes: [u8; 8] = [0, 0, 0, 0, 50, 60, 70, 80];
        let e_lanes: [u8; 8] = [0x80, 0x10, 0xFF, 0, 0, 0, 60, 80];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (a_lanes[i] as u128) << (i * 8);
            b |= (b_lanes[i] as u128) << (i * 8);
            e |= (e_lanes[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VPwMax {
                elem: IRType::I8,
                count: 8,
                signed: false,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Symbolic universality (spec-replay): Iop_PwAdd16x4 must produce the same
    /// bits as a hand-built reference for any 64-bit input pair. Claripy has no
    /// `_op_generic_PwAdd`, so we reference-build inline per `z3-spec-replay-test-template`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vpwadd_16x4_matches_spec_replay() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vpwadd_a", 64);
        let b = RustBV::symbolic(&ctx, "vpwadd_b", 64);
        let got = VEXOps::binop(
            IROp::VPwAdd {
                elem: IRType::I16,
                count: 4,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();

        // Reference: 2 pairs from a, then 2 pairs from b.
        let mut lanes = Vec::with_capacity(4);
        for src in [&a, &b] {
            for i in 0..2u32 {
                let lo0 = 2 * i * 16;
                let lo1 = (2 * i + 1) * 16;
                let l0 = src.extract(lo0 + 15, lo0, &ctx);
                let l1 = src.extract(lo1 + 15, lo1, &ctx);
                lanes.push(l0.add_into(l1, &ctx));
            }
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VPwAdd 16x4 must match the spec-replay reference for all 64-bit inputs"
        );
        ctx.pop();
    }

    /// Symbolic universality (spec-replay): Iop_PwAddL16Sx4 widens each lane
    /// before adding. Reference uses explicit sign-extend.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vpwaddl_16sx4_matches_spec_replay() {
        let ctx = SymContext::new_mock();
        let arg = RustBV::symbolic(&ctx, "vpwaddl_a", 64);
        let got = VEXOps::unop(
            IROp::VPwAddL {
                elem: IRType::I16,
                count: 4,
                signed: true,
            },
            arg.clone(),
            &ctx,
        )
        .unwrap();

        // Reference: 2 pairs, each pair sign-extended to 32 bits then added.
        let mut lanes = Vec::with_capacity(2);
        for i in 0..2u32 {
            let lo0 = 2 * i * 16;
            let lo1 = (2 * i + 1) * 16;
            let l0 = arg.extract(lo0 + 15, lo0, &ctx).sign_extend_into(32, &ctx);
            let l1 = arg.extract(lo1 + 15, lo1, &ctx).sign_extend_into(32, &ctx);
            lanes.push(l0.add_into(l1, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VPwAddL 16Sx4 must match the spec-replay sign-extend reference"
        );
        ctx.pop();
    }

    /// Symbolic universality (spec-replay): Iop_PwMin16Sx4 — signed pairwise
    /// min via SLE/ITE for each pair.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vpwmin_16sx4_matches_spec_replay() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vpwmin_a", 64);
        let b = RustBV::symbolic(&ctx, "vpwmin_b", 64);
        let got = VEXOps::binop(
            IROp::VPwMin {
                elem: IRType::I16,
                count: 4,
                signed: true,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();

        let mut lanes = Vec::with_capacity(4);
        for src in [&a, &b] {
            for i in 0..2u32 {
                let lo0 = 2 * i * 16;
                let lo1 = (2 * i + 1) * 16;
                let l0 = src.extract(lo0 + 15, lo0, &ctx);
                let l1 = src.extract(lo1 + 15, lo1, &ctx);
                let cond = l0.clone().sle_into(l1.clone(), &ctx);
                lanes.push(cond.ite_into(l0, l1, &ctx));
            }
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VPwMin 16Sx4 must match the spec-replay signed-min reference"
        );
        ctx.pop();
    }

    /// Parse routing: Iop_PwAdd / PwAddL / PwMin / PwMax variants land on
    /// VPwAdd / VPwAddL / VPwMin / VPwMax with the expected decomposition.
    /// Iop_PwAdd32Fx2 remains in NeonUnimplemented.
    #[test]
    fn test_parse_pairwise_routing() {
        use crate::vex::opcode_map::parse_opcode;

        let pwadd_cases: &[(&str, IRType, u8)] = &[
            ("Iop_PwAdd8x8", IRType::I8, 8),
            ("Iop_PwAdd16x4", IRType::I16, 4),
            ("Iop_PwAdd32x2", IRType::I32, 2),
            ("Iop_PwAdd8x16", IRType::I8, 16),
            ("Iop_PwAdd16x8", IRType::I16, 8),
            ("Iop_PwAdd32x4", IRType::I32, 4),
        ];
        for (op, e, c) in pwadd_cases {
            match parse_opcode(op) {
                IROp::VPwAdd { elem, count } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                }
                other => panic!("{}: expected VPwAdd, got {:?}", op, other),
            }
        }

        let pwaddl_cases: &[(&str, IRType, u8, bool)] = &[
            ("Iop_PwAddL8Sx8", IRType::I8, 8, true),
            ("Iop_PwAddL8Ux8", IRType::I8, 8, false),
            ("Iop_PwAddL16Sx4", IRType::I16, 4, true),
            ("Iop_PwAddL16Ux4", IRType::I16, 4, false),
            ("Iop_PwAddL32Sx2", IRType::I32, 2, true),
            ("Iop_PwAddL32Ux2", IRType::I32, 2, false),
            ("Iop_PwAddL8Sx16", IRType::I8, 16, true),
            ("Iop_PwAddL8Ux16", IRType::I8, 16, false),
            ("Iop_PwAddL16Sx8", IRType::I16, 8, true),
            ("Iop_PwAddL16Ux8", IRType::I16, 8, false),
            ("Iop_PwAddL32Sx4", IRType::I32, 4, true),
            ("Iop_PwAddL32Ux4", IRType::I32, 4, false),
        ];
        for (op, e, c, s) in pwaddl_cases {
            match parse_opcode(op) {
                IROp::VPwAddL {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(signed, *s, "{}: signed", op);
                }
                other => panic!("{}: expected VPwAddL, got {:?}", op, other),
            }
        }

        let pwmin_cases: &[(&str, IRType, u8, bool)] = &[
            ("Iop_PwMin8Sx8", IRType::I8, 8, true),
            ("Iop_PwMin8Ux8", IRType::I8, 8, false),
            ("Iop_PwMin16Sx4", IRType::I16, 4, true),
            ("Iop_PwMin16Ux4", IRType::I16, 4, false),
            ("Iop_PwMin32Sx2", IRType::I32, 2, true),
            ("Iop_PwMin32Ux2", IRType::I32, 2, false),
        ];
        for (op, e, c, s) in pwmin_cases {
            match parse_opcode(op) {
                IROp::VPwMin {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(signed, *s, "{}: signed", op);
                }
                other => panic!("{}: expected VPwMin, got {:?}", op, other),
            }
        }

        let pwmax_cases: &[(&str, IRType, u8, bool)] = &[
            ("Iop_PwMax8Sx8", IRType::I8, 8, true),
            ("Iop_PwMax8Ux8", IRType::I8, 8, false),
            ("Iop_PwMax16Sx4", IRType::I16, 4, true),
            ("Iop_PwMax16Ux4", IRType::I16, 4, false),
            ("Iop_PwMax32Sx2", IRType::I32, 2, true),
            ("Iop_PwMax32Ux2", IRType::I32, 2, false),
        ];
        for (op, e, c, s) in pwmax_cases {
            match parse_opcode(op) {
                IROp::VPwMax {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(signed, *s, "{}: signed", op);
                }
                other => panic!("{}: expected VPwMax, got {:?}", op, other),
            }
        }

        // Float pairwise add stays unimplemented.
        match parse_opcode("Iop_PwAdd32Fx2") {
            IROp::NeonUnimplemented(name) => assert_eq!(name, "Iop_PwAdd32Fx2"),
            other => panic!("Iop_PwAdd32Fx2 expected NeonUnimplemented, got {:?}", other),
        }
    }

    // =========================================================================
    // angr-tukg.3 — NEON rounding halving add (VAvg).
    // =========================================================================

    /// Iop_Avg8Ux8 — unsigned rounding-average over 8 lanes of 8 bits.
    /// Exercises the round-up at the half-way point (`(a+b+1) >> 1`) and the
    /// no-overflow guarantee for 0xFF+0xFF.
    #[test]
    fn test_vavg_8ux8_concrete() {
        let ctx = SymContext::new_mock();
        // Per-lane: rounded average of u8 values.
        //   lane 0: avg(0, 0) = 0.
        //   lane 1: avg(1, 1) = 1.
        //   lane 2: avg(1, 2) = 2  (round up; truncating would give 1).
        //   lane 3: avg(0xFF, 0xFF) = 0xFF (no overflow — widening absorbs +1).
        //   lane 4: avg(0xFE, 0xFF) = 0xFF (round up; truncating would give 0xFE).
        //   lane 5: avg(0x10, 0x20) = 0x18.
        //   lane 6: avg(0x80, 0x80) = 0x80.
        //   lane 7: avg(0x7F, 0x01) = 0x40.
        let a_lanes: [u8; 8] = [0, 1, 1, 0xFF, 0xFE, 0x10, 0x80, 0x7F];
        let b_lanes: [u8; 8] = [0, 1, 2, 0xFF, 0xFF, 0x20, 0x80, 0x01];
        let e_lanes: [u8; 8] = [0, 1, 2, 0xFF, 0xFF, 0x18, 0x80, 0x40];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (a_lanes[i] as u128) << (i * 8);
            b |= (b_lanes[i] as u128) << (i * 8);
            e |= (e_lanes[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VAvg {
                elem: IRType::I8,
                count: 8,
                signed: false,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Avg16Ux8 — Q-reg variant: 8 lanes of 16 bits. 0xFFFF+0xFFFF
    /// rounding-average must stay 0xFFFF (no truncation loss).
    #[test]
    fn test_vavg_16ux8_concrete() {
        let ctx = SymContext::new_mock();
        let a_lanes: [u16; 8] = [0, 1, 0xFFFF, 0xFFFE, 0x1000, 0x8000, 0x7FFF, 0x0123];
        let b_lanes: [u16; 8] = [0, 2, 0xFFFF, 0xFFFF, 0x2000, 0x8000, 0x0001, 0x0456];
        // Avg = (a+b+1) >> 1
        let e_lanes: [u16; 8] = [0, 2, 0xFFFF, 0xFFFF, 0x1800, 0x8000, 0x4000, 0x02BD];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (a_lanes[i] as u128) << (i * 16);
            b |= (b_lanes[i] as u128) << (i * 16);
            e |= (e_lanes[i] as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VAvg {
                elem: IRType::I16,
                count: 8,
                signed: false,
            },
            RustBV::concrete(a, 128),
            RustBV::concrete(b, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Avg8Sx8 — signed rounding-average. Validates that two -128 lanes
    /// give -128 (sign-extension fence) and mixed-sign lanes round correctly.
    #[test]
    fn test_vavg_8sx8_concrete() {
        let ctx = SymContext::new_mock();
        // Per-lane signed avg with round-half-up.
        //   lane 0: avg(-128, -128) = -128.
        //   lane 1: avg(127, 127)   = 127.
        //   lane 2: avg(-1, 0)      = 0 (round up: (-1+0+1)/2=0).
        //   lane 3: avg(-2, -1)     = -1.
        //   lane 4: avg(-100, 100)  = 0.
        //   lane 5: avg(-100, 101)  = 1.
        //   lane 6: avg(127, -128)  = 0 (the +1 makes the sum -1+1=0; >>1=0).
        //   lane 7: avg(50, 51)     = 51.
        let a_lanes: [i8; 8] = [-128, 127, -1, -2, -100, -100, 127, 50];
        let b_lanes: [i8; 8] = [-128, 127, 0, -1, 100, 101, -128, 51];
        let e_lanes: [i8; 8] = [-128, 127, 0, -1, 0, 1, 0, 51];
        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= ((a_lanes[i] as u8) as u128) << (i * 8);
            b |= ((b_lanes[i] as u8) as u128) << (i * 8);
            e |= ((e_lanes[i] as u8) as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VAvg {
                elem: IRType::I8,
                count: 8,
                signed: true,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Symbolic universality (spec-replay): Iop_Avg16Ux4 must equal the
    /// reference `((zext(a)+zext(b)+1) >> 1)[15:0]` per lane for all 64-bit
    /// inputs. Claripy has no `_op_generic_Avg`; the test uses the
    /// `z3-spec-replay-test-template` pattern.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vavg_16ux4_symbolic_universal_unsigned() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vavg_a", 64);
        let b = RustBV::symbolic(&ctx, "vavg_b", 64);
        let got = VEXOps::binop(
            IROp::VAvg {
                elem: IRType::I16,
                count: 4,
                signed: false,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();

        let mut lanes = Vec::with_capacity(4);
        for i in 0..4u32 {
            let lo = i * 16;
            let hi = lo + 15;
            let al = a.extract(hi, lo, &ctx).zero_extend_into(17, &ctx);
            let bl = b.extract(hi, lo, &ctx).zero_extend_into(17, &ctx);
            let sum = al
                .add_into(bl, &ctx)
                .add_into(RustBV::concrete(1, 17), &ctx);
            let shifted = sum.lshr_into(RustBV::concrete(1, 17), &ctx);
            lanes.push(shifted.extract(15, 0, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VAvg 16Ux4 must match the unsigned spec-replay reference"
        );
        ctx.pop();
    }

    /// Symbolic universality (spec-replay): Iop_Avg8Sx8 signed rounding-avg.
    /// Reference sign-extends each lane to 9 bits before summing.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vavg_8sx8_symbolic_universal_signed() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vavg_sa", 64);
        let b = RustBV::symbolic(&ctx, "vavg_sb", 64);
        let got = VEXOps::binop(
            IROp::VAvg {
                elem: IRType::I8,
                count: 8,
                signed: true,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();

        let mut lanes = Vec::with_capacity(8);
        for i in 0..8u32 {
            let lo = i * 8;
            let hi = lo + 7;
            let al = a.extract(hi, lo, &ctx).sign_extend_into(9, &ctx);
            let bl = b.extract(hi, lo, &ctx).sign_extend_into(9, &ctx);
            let sum = al.add_into(bl, &ctx).add_into(RustBV::concrete(1, 9), &ctx);
            let shifted = sum.lshr_into(RustBV::concrete(1, 9), &ctx);
            lanes.push(shifted.extract(7, 0, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VAvg 8Sx8 must match the signed spec-replay reference"
        );
        ctx.pop();
    }

    /// Parse routing: all 12 Iop_Avg variants land on IROp::VAvg with the
    /// expected `(elem, count, signed)` decomposition. No Avg op should remain
    /// in NeonUnimplemented.
    #[test]
    fn test_parse_avg_routing() {
        use crate::vex::opcode_map::parse_opcode;

        let cases: &[(&str, IRType, u8, bool)] = &[
            ("Iop_Avg8Ux8", IRType::I8, 8, false),
            ("Iop_Avg16Ux4", IRType::I16, 4, false),
            ("Iop_Avg32Ux2", IRType::I32, 2, false),
            ("Iop_Avg8Sx8", IRType::I8, 8, true),
            ("Iop_Avg16Sx4", IRType::I16, 4, true),
            ("Iop_Avg32Sx2", IRType::I32, 2, true),
            ("Iop_Avg8Ux16", IRType::I8, 16, false),
            ("Iop_Avg16Ux8", IRType::I16, 8, false),
            ("Iop_Avg32Ux4", IRType::I32, 4, false),
            ("Iop_Avg8Sx16", IRType::I8, 16, true),
            ("Iop_Avg16Sx8", IRType::I16, 8, true),
            ("Iop_Avg32Sx4", IRType::I32, 4, true),
        ];
        for (op, e, c, s) in cases {
            match parse_opcode(op) {
                IROp::VAvg {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(signed, *s, "{}: signed", op);
                }
                other => panic!("{}: expected VAvg, got {:?}", op, other),
            }
        }
    }

    // =========================================================================
    // angr-tukg.6 — NEON per-lane Cnt / Clz / Cls + GF(2) PolynomialMul.
    // =========================================================================

    /// Iop_Cnt8x8 — per-byte popcount over 8 lanes. Covers all bits-set,
    /// no-bits-set, single-bit, and mid-density patterns.
    #[test]
    fn test_vcnt_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let lanes: [u8; 8] = [0x00, 0xFF, 0x01, 0x80, 0x0F, 0xF0, 0x55, 0xAA];
        // 0,8,1,1,4,4,4,4 — popcount per lane.
        let expected: [u8; 8] = [0, 8, 1, 1, 4, 4, 4, 4];
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (lanes[i] as u128) << (i * 8);
            e |= (expected[i] as u128) << (i * 8);
        }
        let result = VEXOps::unop(IROp::VCnt { count: 8 }, RustBV::concrete(a, 64), &ctx).unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Cnt8x16 — Q-reg popcount: 16 bytes.
    #[test]
    fn test_vcnt_8x16_concrete() {
        let ctx = SymContext::new_mock();
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..16u32 {
            // Lane i has popcount i % 9.
            let v: u8 = (1u16.wrapping_shl(i % 9).wrapping_sub(1)) as u8;
            a |= (v as u128) << (i * 8);
            e |= ((i % 9) as u128) << (i * 8);
        }
        let result =
            VEXOps::unop(IROp::VCnt { count: 16 }, RustBV::concrete(a, 128), &ctx).unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Clz8x8 — per-byte count leading zeros.
    #[test]
    fn test_vclz_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let lanes: [u8; 8] = [0x00, 0x80, 0x40, 0x01, 0xFF, 0x10, 0x08, 0x7F];
        let expected: [u8; 8] = [8, 0, 1, 7, 0, 3, 4, 1];
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (lanes[i] as u128) << (i * 8);
            e |= (expected[i] as u128) << (i * 8);
        }
        let result = VEXOps::unop(
            IROp::VClz {
                elem: IRType::I8,
                count: 8,
            },
            RustBV::concrete(a, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Clz32x4 — Q-reg 32-bit lane clz. Covers all-zero (→ 32), MSB set
    /// (→ 0), and a mid-range value.
    #[test]
    fn test_vclz_32x4_concrete() {
        let ctx = SymContext::new_mock();
        let lanes: [u32; 4] = [0x00000000, 0x80000000, 0x00010000, 0xFFFFFFFF];
        let expected: [u32; 4] = [32, 0, 15, 0];
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..4 {
            a |= (lanes[i] as u128) << (i * 32);
            e |= (expected[i] as u128) << (i * 32);
        }
        let result = VEXOps::unop(
            IROp::VClz {
                elem: IRType::I32,
                count: 4,
            },
            RustBV::concrete(a, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Cls8x8 — per-byte count leading sign bits (excluding MSB).
    /// All-same → N-1; first mismatch determines result.
    #[test]
    fn test_vcls_8x8_concrete() {
        let ctx = SymContext::new_mock();
        // Inputs covering both sign polarities.
        // 0x00 (0b00000000) → all 7 non-MSB bits match MSB(0) → 7.
        // 0xFF (0b11111111) → all 7 non-MSB bits match MSB(1) → 7.
        // 0x01 (0b00000001) → bit6=0,bit5=0,...,bit1=0 match MSB, bit0=1 differs → 6.
        // 0x02 (0b00000010) → bit1=1 differs at pos 1 → first mismatch at pos 1 → 5.
        // 0x40 (0b01000000) → bit6=1 differs from MSB(0) → 0.
        // 0xC0 (0b11000000) → bit6=1 matches MSB(1); bit5=0 differs at pos 5 → 1.
        // 0x80 (0b10000000) → MSB=1; bits 6..0 all 0, all differ → 0.
        // 0x7F (0b01111111) → MSB=0; bit6=1 differs at pos 6 → 0.
        let lanes: [u8; 8] = [0x00, 0xFF, 0x01, 0x02, 0x40, 0xC0, 0x80, 0x7F];
        let expected: [u8; 8] = [7, 7, 6, 5, 0, 1, 0, 0];
        let mut a: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (lanes[i] as u128) << (i * 8);
            e |= (expected[i] as u128) << (i * 8);
        }
        let result = VEXOps::unop(
            IROp::VCls {
                elem: IRType::I8,
                count: 8,
            },
            RustBV::concrete(a, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PolynomialMul8x8 — GF(2) multiply per byte, low 8 bits.
    /// Reference (closed-form per byte):
    ///   a=0x01, b=0x57 → 0x01*0x57 = 0x57.
    ///   a=0x02, b=0x57 → 0x57<<1 = 0xAE; low8 = 0xAE.
    ///   a=0x03, b=0x57 → 0x57 ^ 0xAE = 0xF9; low8 = 0xF9.
    ///   a=0x80, b=0x80 → 0x80<<7 = 0x4000; low8 = 0x00.
    ///   a=0xFF, b=0x01 → XOR of 0x01<<0..7 = 0xFF.
    ///   a=0xC0, b=0x55 → (0x55<<6)^(0x55<<7) = 0x1540 ^ 0x2A80 = 0x3FC0; low8=0xC0.
    ///   a=0x00, b=0xFF → 0.
    ///   a=0xFF, b=0xFF → low8 of GF(2) 0xFF*0xFF; computed below.
    #[test]
    fn test_vpolynomial_mul_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let a_lanes: [u8; 8] = [0x01, 0x02, 0x03, 0x80, 0xFF, 0xC0, 0x00, 0xFF];
        let b_lanes: [u8; 8] = [0x57, 0x57, 0x57, 0x80, 0x01, 0x55, 0xFF, 0xFF];

        // Compute expected via the same algorithm — keeps the test honest.
        let mut expected: [u8; 8] = [0; 8];
        for i in 0..8 {
            let (a, b) = (a_lanes[i] as u16, b_lanes[i] as u16);
            let mut prod: u16 = 0;
            for bit in 0..8 {
                if (a >> bit) & 1 != 0 {
                    prod ^= b << bit;
                }
            }
            expected[i] = (prod & 0xFF) as u8;
        }
        // Sanity-check a couple of the closed-form values to catch a bad
        // expected-table generator.
        assert_eq!(expected[0], 0x57);
        assert_eq!(expected[1], 0xAE);
        assert_eq!(expected[2], 0xF9);
        assert_eq!(expected[3], 0x00);
        assert_eq!(expected[4], 0xFF);
        assert_eq!(expected[6], 0x00);

        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (a_lanes[i] as u128) << (i * 8);
            b |= (b_lanes[i] as u128) << (i * 8);
            e |= (expected[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VPolynomialMul {
                count: 8,
                widen: false,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_PolynomialMull8x8 — widening GF(2) multiply: 8x8 → 8x16. Per lane
    /// keeps the full 16-bit product. Output total = 128.
    #[test]
    fn test_vpolynomial_mull_8x8_concrete() {
        let ctx = SymContext::new_mock();
        let a_lanes: [u8; 8] = [0x01, 0x02, 0x80, 0x40, 0xFF, 0x10, 0x55, 0x00];
        let b_lanes: [u8; 8] = [0x57, 0x57, 0x80, 0x02, 0xFF, 0x10, 0xAA, 0xFF];
        let mut expected: [u16; 8] = [0; 8];
        for i in 0..8 {
            let (a, b) = (a_lanes[i] as u16, b_lanes[i] as u16);
            let mut prod: u16 = 0;
            for bit in 0..8 {
                if (a >> bit) & 1 != 0 {
                    prod ^= b << bit;
                }
            }
            expected[i] = prod;
        }
        // Widening keeps the full product. 0x80*0x80 over GF(2) = 0x4000.
        assert_eq!(expected[2], 0x4000);
        // 0x01*0x57 = 0x57; widening must preserve this (low byte only).
        assert_eq!(expected[0], 0x0057);

        let mut a: u128 = 0;
        let mut b: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            a |= (a_lanes[i] as u128) << (i * 8);
            b |= (b_lanes[i] as u128) << (i * 8);
            e |= (expected[i] as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VPolynomialMul {
                count: 8,
                widen: true,
            },
            RustBV::concrete(a, 64),
            RustBV::concrete(b, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Symbolic universality (spec-replay): Iop_Cnt8x8 must equal the
    /// per-byte bit-sum reference for all 64-bit inputs.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vcnt_8x8_symbolic_universal() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vcnt_a", 64);
        let got = VEXOps::unop(IROp::VCnt { count: 8 }, a.clone(), &ctx).unwrap();

        let mut lanes = Vec::with_capacity(8);
        for i in 0..8u32 {
            let lo = i * 8;
            let mut acc = RustBV::concrete(0, 8);
            for b in 0..8 {
                let bit = a.extract(lo + b, lo + b, &ctx);
                acc = acc.add_into(bit.zero_extend_into(8, &ctx), &ctx);
            }
            lanes.push(acc);
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VCnt 8x8 must match the spec-replay reference"
        );
        ctx.pop();
    }

    /// Symbolic universality (spec-replay): Iop_Clz8x8 must match the
    /// claripy-style ITE chain reference per lane.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vclz_8x8_symbolic_universal() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vclz_a", 64);
        let got = VEXOps::unop(
            IROp::VClz {
                elem: IRType::I8,
                count: 8,
            },
            a.clone(),
            &ctx,
        )
        .unwrap();

        let mut lanes = Vec::with_capacity(8);
        let one = RustBV::concrete(1, 1);
        for i in 0..8u32 {
            let lo = i * 8;
            let hi = lo + 7;
            let lane = a.extract(hi, lo, &ctx);
            // Build claripy-style ITE chain inline.
            let mut expr = RustBV::concrete(8, 8);
            for b in 0..8u32 {
                let bit = lane.extract(b, b, &ctx);
                let cond = bit.eq_into(one.clone(), &ctx);
                let then_v = RustBV::concrete((8 - b - 1) as u128, 8);
                expr = cond.ite_into(then_v, expr, &ctx);
            }
            lanes.push(expr);
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VClz 8x8 must match the spec-replay reference"
        );
        ctx.pop();
    }

    /// Symbolic universality (spec-replay): Iop_PolynomialMul8x8 must match
    /// the XOR-of-shifts reference per lane (low 8 bits).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vpolynomial_mul_8x8_symbolic_universal() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "vpmul_a", 64);
        let b = RustBV::symbolic(&ctx, "vpmul_b", 64);
        let got = VEXOps::binop(
            IROp::VPolynomialMul {
                count: 8,
                widen: false,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();

        let one = RustBV::concrete(1, 1);
        let mut lanes = Vec::with_capacity(8);
        for i in 0..8u32 {
            let lo = i * 8;
            let hi = lo + 7;
            let a_lane = a.extract(hi, lo, &ctx);
            let b_lane = b.extract(hi, lo, &ctx);
            let b_wide = b_lane.zero_extend_into(16, &ctx);
            let mut acc = RustBV::concrete(0, 16);
            for bit in 0..8u32 {
                let bit_a = a_lane.extract(bit, bit, &ctx);
                let cond = bit_a.eq_into(one.clone(), &ctx);
                let shifted = b_wide
                    .clone()
                    .shl_into(RustBV::concrete(bit as u128, 16), &ctx);
                let addend = cond.ite_into(shifted, RustBV::concrete(0, 16), &ctx);
                acc = acc.xor_into(addend, &ctx);
            }
            lanes.push(acc.extract(7, 0, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);

        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "VPolynomialMul 8x8 must match the spec-replay reference"
        );
        ctx.pop();
    }

    /// Parse routing: all Cnt/Clz/Cls/PolynomialMul opcodes land on their real
    /// IROp variants (not NeonUnimplemented).
    #[test]
    fn test_parse_cnt_clz_cls_pmul_routing() {
        use crate::vex::opcode_map::parse_opcode;

        // VCnt
        for (op, c) in &[("Iop_Cnt8x8", 8u8), ("Iop_Cnt8x16", 16)] {
            match parse_opcode(op) {
                IROp::VCnt { count } => assert_eq!(count, *c, "{}: count", op),
                other => panic!("{}: expected VCnt, got {:?}", op, other),
            }
        }

        // VClz
        let clz_cases: &[(&str, IRType, u8)] = &[
            ("Iop_Clz8x8", IRType::I8, 8),
            ("Iop_Clz16x4", IRType::I16, 4),
            ("Iop_Clz32x2", IRType::I32, 2),
            ("Iop_Clz8x16", IRType::I8, 16),
            ("Iop_Clz16x8", IRType::I16, 8),
            ("Iop_Clz32x4", IRType::I32, 4),
        ];
        for (op, e, c) in clz_cases {
            match parse_opcode(op) {
                IROp::VClz { elem, count } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                }
                other => panic!("{}: expected VClz, got {:?}", op, other),
            }
        }

        // VCls
        let cls_cases: &[(&str, IRType, u8)] = &[
            ("Iop_Cls8x8", IRType::I8, 8),
            ("Iop_Cls16x4", IRType::I16, 4),
            ("Iop_Cls32x2", IRType::I32, 2),
            ("Iop_Cls8x16", IRType::I8, 16),
            ("Iop_Cls16x8", IRType::I16, 8),
            ("Iop_Cls32x4", IRType::I32, 4),
        ];
        for (op, e, c) in cls_cases {
            match parse_opcode(op) {
                IROp::VCls { elem, count } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                }
                other => panic!("{}: expected VCls, got {:?}", op, other),
            }
        }

        // VPolynomialMul (non-widening and widening)
        let pmul_cases: &[(&str, u8, bool)] = &[
            ("Iop_PolynomialMul8x8", 8, false),
            ("Iop_PolynomialMul8x16", 16, false),
            ("Iop_PolynomialMull8x8", 8, true),
        ];
        for (op, c, w) in pmul_cases {
            match parse_opcode(op) {
                IROp::VPolynomialMul { count, widen } => {
                    assert_eq!(count, *c, "{}: count", op);
                    assert_eq!(widen, *w, "{}: widen", op);
                }
                other => panic!("{}: expected VPolynomialMul, got {:?}", op, other),
            }
        }
    }

    // =========================================================================
    // angr-tukg.7 — NEON vector-shift-by-vector (VShl / VShr / VSar).
    // =========================================================================

    /// Iop_Shl8x8 — left shift each of 8 lanes by the corresponding count lane.
    /// Covers: zero shift (identity), in-range shifts, and out-of-range counts
    /// (≥ lane width → zero, matching Z3 bvshl).
    #[test]
    fn test_vshl_8x8_concrete() {
        let ctx = SymContext::new_mock();
        // Lane layout (LSB→MSB): vec lanes, then shift lanes.
        //   0:  0x01 << 0  = 0x01.
        //   1:  0x01 << 1  = 0x02.
        //   2:  0x01 << 7  = 0x80.
        //   3:  0x01 << 8  = 0 (count == lane width).
        //   4:  0x01 << 255 = 0 (count > lane width).
        //   5:  0xFF << 4  = 0xF0 (high bits shifted out).
        //   6:  0x55 << 1  = 0xAA.
        //   7:  0x80 << 1  = 0 (high bit shifted out).
        let lanes_v: [u8; 8] = [0x01, 0x01, 0x01, 0x01, 0x01, 0xFF, 0x55, 0x80];
        let lanes_s: [u8; 8] = [0, 1, 7, 8, 255, 4, 1, 1];
        let lanes_e: [u8; 8] = [0x01, 0x02, 0x80, 0x00, 0x00, 0xF0, 0xAA, 0x00];
        let mut v: u128 = 0;
        let mut s: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            v |= (lanes_v[i] as u128) << (i * 8);
            s |= (lanes_s[i] as u128) << (i * 8);
            e |= (lanes_e[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VShl {
                elem: IRType::I8,
                count: 8,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Shr16x8 — logical right shift over 8x i16 lanes on a 128-bit vector.
    /// Verifies zero-fill (Shr discards the sign bit) and out-of-range counts.
    #[test]
    fn test_vshr_16x8_concrete() {
        let ctx = SymContext::new_mock();
        let lanes_v: [u16; 8] = [
            0x8000, 0xFFFF, 0xABCD, 0x0001, 0xFFFF, 0x4000, 0x0F0F, 0x1234,
        ];
        let lanes_s: [u16; 8] = [15, 8, 4, 0, 16, 1, 4, 100];
        // 0x8000 >> 15 = 1 (no sign extend).
        // 0xFFFF >> 8  = 0x00FF.
        // 0xABCD >> 4  = 0x0ABC.
        // 0x0001 >> 0  = 0x0001.
        // 0xFFFF >> 16 = 0 (count == width).
        // 0x4000 >> 1  = 0x2000.
        // 0x0F0F >> 4  = 0x00F0.
        // 0x1234 >> 100 = 0 (count > width).
        let lanes_e: [u16; 8] = [1, 0x00FF, 0x0ABC, 0x0001, 0, 0x2000, 0x00F0, 0];
        let mut v: u128 = 0;
        let mut s: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..8 {
            v |= (lanes_v[i] as u128) << (i * 16);
            s |= (lanes_s[i] as u128) << (i * 16);
            e |= (lanes_e[i] as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VShr {
                elem: IRType::I16,
                count: 8,
            },
            RustBV::concrete(v, 128),
            RustBV::concrete(s, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Sar32x2 — arithmetic right shift over 2x i32 lanes on a 64-bit
    /// vector. Verifies sign-fill on negative inputs and out-of-range counts.
    #[test]
    fn test_vsar_32x2_concrete() {
        let ctx = SymContext::new_mock();
        // 0: -1i32 (0xFFFF_FFFF) >> 4   = -1 (sign-fill keeps all bits set).
        // 1: 0x4000_0000   >> 1   = 0x2000_0000 (positive → logical shift).
        // Note: 0xFFFF_FFFF >> 32 would also be all-1 in arithmetic shift,
        // but Z3 bvashr semantics for count >= width are sign-fill which is
        // matched by our concrete fast path.
        let lanes_v: [u32; 2] = [0xFFFF_FFFF, 0x4000_0000];
        let lanes_s: [u32; 2] = [4, 1];
        let lanes_e: [u32; 2] = [0xFFFF_FFFF, 0x2000_0000];
        let mut v: u128 = 0;
        let mut s: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..2 {
            v |= (lanes_v[i] as u128) << (i * 32);
            s |= (lanes_s[i] as u128) << (i * 32);
            e |= (lanes_e[i] as u128) << (i * 32);
        }
        let result = VEXOps::binop(
            IROp::VSar {
                elem: IRType::I32,
                count: 2,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Sar32x2 — out-of-range count produces sign-fill (-1 for negative
    /// lanes, 0 for positive). Concrete fast path mirrors Z3 bvashr.
    #[test]
    fn test_vsar_32x2_oor_sign_fill() {
        let ctx = SymContext::new_mock();
        // 0: -1i32 >> 64 = sign-fill = 0xFFFF_FFFF.
        // 1:  1i32 >> 32 = sign-fill = 0 (positive).
        let lanes_v: [u32; 2] = [0xFFFF_FFFF, 0x0000_0001];
        let lanes_s: [u32; 2] = [64, 32];
        let lanes_e: [u32; 2] = [0xFFFF_FFFF, 0];
        let mut v: u128 = 0;
        let mut s: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..2 {
            v |= (lanes_v[i] as u128) << (i * 32);
            s |= (lanes_s[i] as u128) << (i * 32);
            e |= (lanes_e[i] as u128) << (i * 32);
        }
        let result = VEXOps::binop(
            IROp::VSar {
                elem: IRType::I32,
                count: 2,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_Sal8x8 (== Iop_Shl8x8 bit-for-bit): a single concrete check that
    /// both opcodes parse to VShl and produce identical results.
    #[test]
    fn test_vsal_routes_to_vshl_and_matches() {
        use crate::vex::opcode_map::parse_opcode;
        let ctx = SymContext::new_mock();
        let v = 0x1234_5678_9ABC_DEF0u128;
        let s = 0x0102_0304_0506_0708u128; // per-lane counts 8,7,6,5,4,3,2,1
        let shl_res = VEXOps::binop(
            IROp::VShl {
                elem: IRType::I8,
                count: 8,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        // Sal{N}x{M} must parse to the same IROp variant.
        assert!(matches!(parse_opcode("Iop_Sal8x8"), IROp::VShl { .. }));
        let sal_op = parse_opcode("Iop_Sal8x8");
        let sal_res = VEXOps::binop(
            sal_op,
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(sal_res.as_u128().unwrap(), shl_res.as_u128().unwrap());
    }

    /// Z3 universality parity vs the explicit claripy reference for
    /// `Iop_Shl16x4` (operation_map["Shl"] = "__lshift__"; vector dispatch
    /// per `_op_vector_mapped` extracts each lane and applies `bvshl`).
    /// Symbolic inputs + add_constraint(got ≠ py).not() → assert UNSAT.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vshl_16x4_symbolic_matches_python_ref() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "shl_a_16x4", 64);
        let b = RustBV::symbolic(&ctx, "shl_b_16x4", 64);
        let got = VEXOps::binop(
            IROp::VShl {
                elem: IRType::I16,
                count: 4,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();
        // Build the claripy reference: per-lane bvshl. _op_vector_mapped
        // concats lanes high→low; we use concat_le_elements (low→high), so
        // the resulting BV is structurally equivalent.
        let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
        for i in 0..4 {
            let lo = i * 16;
            let hi = lo + 15;
            let al = a.extract(hi, lo, &ctx);
            let bl = b.extract(hi, lo, &ctx);
            lanes.push(al.shl_into(bl, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);
        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        let universal = !ctx.is_sat();
        ctx.pop();
        assert!(
            universal,
            "VShl 16x4 must match the claripy __lshift__ reference for all 64-bit inputs"
        );
    }

    /// Z3 universality parity vs the claripy reference for `Iop_Sar8x16`
    /// (operation_map["Sar"] = "__rshift__" → bvashr; 16 lanes of 8 bits
    /// across a 128-bit vector).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vsar_8x16_symbolic_matches_python_ref() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "sar_a_8x16", 128);
        let b = RustBV::symbolic(&ctx, "sar_b_8x16", 128);
        let got = VEXOps::binop(
            IROp::VSar {
                elem: IRType::I8,
                count: 16,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();
        let mut lanes: Vec<RustBV> = Vec::with_capacity(16);
        for i in 0..16 {
            let lo = i * 8;
            let hi = lo + 7;
            let al = a.extract(hi, lo, &ctx);
            let bl = b.extract(hi, lo, &ctx);
            lanes.push(al.ashr_into(bl, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);
        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        let universal = !ctx.is_sat();
        ctx.pop();
        assert!(
            universal,
            "VSar 8x16 must match the claripy __rshift__ reference for all 128-bit inputs"
        );
    }

    /// Parse routing: all 8 VShl shapes (8x8 .. 64x2) and the matching Sal
    /// aliases land on VShl with the expected (elem, count); same for VShr/VSar.
    #[test]
    fn test_parse_vshift_routing() {
        use crate::vex::opcode_map::parse_opcode;
        let shl_cases: &[(&str, IRType, u8)] = &[
            ("Iop_Shl8x8", IRType::I8, 8),
            ("Iop_Shl16x4", IRType::I16, 4),
            ("Iop_Shl32x2", IRType::I32, 2),
            ("Iop_Shl64x1", IRType::I64, 1),
            ("Iop_Shl8x16", IRType::I8, 16),
            ("Iop_Shl16x8", IRType::I16, 8),
            ("Iop_Shl32x4", IRType::I32, 4),
            ("Iop_Shl64x2", IRType::I64, 2),
            // Sal aliases route to the same variant.
            ("Iop_Sal8x8", IRType::I8, 8),
            ("Iop_Sal16x4", IRType::I16, 4),
            ("Iop_Sal32x2", IRType::I32, 2),
            ("Iop_Sal64x1", IRType::I64, 1),
            ("Iop_Sal8x16", IRType::I8, 16),
            ("Iop_Sal16x8", IRType::I16, 8),
            ("Iop_Sal32x4", IRType::I32, 4),
            ("Iop_Sal64x2", IRType::I64, 2),
        ];
        for (op, e, c) in shl_cases {
            match parse_opcode(op) {
                IROp::VShl { elem, count } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                }
                other => panic!("{}: expected VShl, got {:?}", op, other),
            }
        }

        let shr_cases: &[(&str, IRType, u8)] = &[
            ("Iop_Shr8x8", IRType::I8, 8),
            ("Iop_Shr16x4", IRType::I16, 4),
            ("Iop_Shr32x2", IRType::I32, 2),
            ("Iop_Shr64x1", IRType::I64, 1),
            ("Iop_Shr8x16", IRType::I8, 16),
            ("Iop_Shr16x8", IRType::I16, 8),
            ("Iop_Shr32x4", IRType::I32, 4),
            ("Iop_Shr64x2", IRType::I64, 2),
        ];
        for (op, e, c) in shr_cases {
            match parse_opcode(op) {
                IROp::VShr { elem, count } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                }
                other => panic!("{}: expected VShr, got {:?}", op, other),
            }
        }

        let sar_cases: &[(&str, IRType, u8)] = &[
            ("Iop_Sar8x8", IRType::I8, 8),
            ("Iop_Sar16x4", IRType::I16, 4),
            ("Iop_Sar32x2", IRType::I32, 2),
            ("Iop_Sar64x1", IRType::I64, 1),
            ("Iop_Sar8x16", IRType::I8, 16),
            ("Iop_Sar16x8", IRType::I16, 8),
            ("Iop_Sar32x4", IRType::I32, 4),
            ("Iop_Sar64x2", IRType::I64, 2),
        ];
        for (op, e, c) in sar_cases {
            match parse_opcode(op) {
                IROp::VSar { elem, count } => {
                    assert_eq!(elem, *e, "{}: elem", op);
                    assert_eq!(count, *c, "{}: count", op);
                }
                other => panic!("{}: expected VSar, got {:?}", op, other),
            }
        }
    }

    // =========================================================================
    // angr-tukg.8 — NEON saturating vector shifts (VQShlSat).
    // =========================================================================

    /// Iop_QShl8x8 — unsigned saturating left shift by vector (D-reg).
    /// Covers in-range left shift, OOR left shift (amt ≥ width → UMAX if
    /// `a != 0` else 0), overflow saturation to UMAX, and the negative-amt
    /// branch (right shift via logical shift, with OOR → 0).
    #[test]
    fn test_vqshl_8x8_concrete_unsigned() {
        let ctx = SymContext::new_mock();
        // amt is sign-extended as i8: 0xFF = -1, 0xFC = -4, 0xF8 = -8.
        let lanes_v: [u8; 8] = [0x01, 0x01, 0x01, 0x80, 0x40, 0xFF, 0x10, 0x80];
        let lanes_s: [u8; 8] = [0, 7, 8, 1, 1, 0xFF, 0xFC, 0xF8];
        // 0x01<<0  = 0x01.    0x01<<7  = 0x80.   0x01<<8 OOR, a!=0 → 0xFF.
        // 0x80<<1  overflow → 0xFF.              0x40<<1  = 0x80 (no overflow).
        // 0xFF >> 1 (amt=-1) = 0x7F (lshr).      0x10>>4 (amt=-4) = 0x01.
        // 0x80>>8 OOR → 0 (lshr).
        let lanes_e: [u8; 8] = [0x01, 0x80, 0xFF, 0xFF, 0x80, 0x7F, 0x01, 0x00];
        let (mut v, mut s, mut e) = (0u128, 0u128, 0u128);
        for i in 0..8 {
            v |= (lanes_v[i] as u128) << (i * 8);
            s |= (lanes_s[i] as u128) << (i * 8);
            e |= (lanes_e[i] as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I8,
                count: 8,
                signed: false,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_QSal16x4 — signed saturating left shift by vector (D-reg).
    /// Covers positive overflow → SMAX, negative overflow → SMIN, in-range
    /// shift, and the negative-amt branch (right shift via ashr, sign-fill).
    #[test]
    fn test_vqsal_16x4_concrete_signed() {
        let ctx = SymContext::new_mock();
        // i16: SMAX=0x7FFF, SMIN=0x8000, -1=0xFFFF.
        // lane 0: 0x1000 (positive) << 3 = 0x8000 (negative when truncated) →
        //   overflow → SMAX = 0x7FFF.
        // lane 1: 0xF000 (= -4096) << 1 = 0xE000 (= -8192); ashr(0xE000,1)
        //   = 0xF000 == a → no overflow → 0xE000.
        // lane 2: 0xFFFF (= -1) with amt = -1 (0xFFFF sign-extended): ashr
        //   by 1 → 0xFFFF (sign-fill).
        // lane 3: 0x0040 (positive) with amt = 16 (OOR): a > 0 → SMAX.
        let lanes_v: [u16; 4] = [0x1000, 0xF000, 0xFFFF, 0x0040];
        let lanes_s: [u16; 4] = [3, 1, 0xFFFF, 16];
        let lanes_e: [u16; 4] = [0x7FFF, 0xE000, 0xFFFF, 0x7FFF];
        let (mut v, mut s, mut e) = (0u128, 0u128, 0u128);
        for i in 0..4 {
            v |= (lanes_v[i] as u128) << (i * 16);
            s |= (lanes_s[i] as u128) << (i * 16);
            e |= (lanes_e[i] as u128) << (i * 16);
        }
        let result = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I16,
                count: 4,
                signed: true,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 64);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_QSal8x16 — signed saturating left shift over 16x i8 lanes
    /// (Q-reg/128-bit). Covers OOR-with-negative-input → SMIN saturation
    /// (a < 0 with amt > width).
    #[test]
    fn test_vqsal_8x16_concrete_smin_saturation() {
        let ctx = SymContext::new_mock();
        // Build a 16-lane vector: alternating positive overflow (a=1, amt=8 OOR)
        // and negative overflow (a=0xFF=-1, amt=8 OOR).
        // amt=8 for all lanes. Positive a=1 (>0) → SMAX=0x7F.
        //                     Negative a=0xFF (<0) → SMIN=0x80.
        let mut v: u128 = 0;
        let mut s: u128 = 0;
        let mut e: u128 = 0;
        for i in 0..16 {
            let (a, expected) = if i % 2 == 0 {
                (1u8, 0x7Fu8)
            } else {
                (0xFFu8, 0x80u8)
            };
            v |= (a as u128) << (i * 8);
            s |= 8u128 << (i * 8);
            e |= (expected as u128) << (i * 8);
        }
        let result = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I8,
                count: 16,
                signed: true,
            },
            RustBV::concrete(v, 128),
            RustBV::concrete(s, 128),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.width(), 128);
        assert_eq!(result.as_u128().unwrap(), e);
    }

    /// Iop_QShl64x1 — width=64 edge case (single-lane D-reg). Exercises the
    /// elem_width = 64 boundary of the concrete fast path (elem_mask uses
    /// the full u64 range; sign-extension via the |!elem_mask| branch).
    #[test]
    fn test_vqshl_64x1_concrete_width_boundary() {
        let ctx = SymContext::new_mock();
        // Unsigned: 0x0000_0000_0000_0001 << 63 = 0x8000_0000_0000_0000.
        // Round-trip: (0x8000... >> 63) = 1 == a. No overflow. Result OK.
        let v = 1u128;
        let s = 63u128;
        let e = 0x8000_0000_0000_0000u128;
        let result = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I64,
                count: 1,
                signed: false,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.as_u128().unwrap(), e);

        // Same input, signed: a=1 (positive), shift left by 63 → 0x8000...
        // which is SMIN as signed. Overflow → SMAX = 0x7FFF_FFFF_FFFF_FFFF.
        let result_s = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I64,
                count: 1,
                signed: true,
            },
            RustBV::concrete(v, 64),
            RustBV::concrete(s, 64),
            &ctx,
        )
        .unwrap();
        assert_eq!(result_s.as_u128().unwrap(), 0x7FFF_FFFF_FFFF_FFFFu128);
    }

    /// Parse routing for all 16 saturating-shift opcodes: Iop_QShl{N}x{M}
    /// (signed=false) and Iop_QSal{N}x{M} (signed=true), 8 shapes each.
    #[test]
    fn test_parse_vqshlsat_routing() {
        use crate::vex::opcode_map::parse_opcode;
        let shapes: &[(&str, IRType, u8)] = &[
            ("8x8", IRType::I8, 8),
            ("16x4", IRType::I16, 4),
            ("32x2", IRType::I32, 2),
            ("64x1", IRType::I64, 1),
            ("8x16", IRType::I8, 16),
            ("16x8", IRType::I16, 8),
            ("32x4", IRType::I32, 4),
            ("64x2", IRType::I64, 2),
        ];
        for (sfx, elem_e, count_e) in shapes {
            for (prefix, want_signed) in [("Iop_QShl", false), ("Iop_QSal", true)] {
                let name = format!("{}{}", prefix, sfx);
                match parse_opcode(&name) {
                    IROp::VQShlSat {
                        elem,
                        count,
                        signed,
                    } => {
                        assert_eq!(elem, *elem_e, "{}: elem", name);
                        assert_eq!(count, *count_e, "{}: count", name);
                        assert_eq!(signed, want_signed, "{}: signed", name);
                    }
                    other => panic!("{}: expected VQShlSat, got {:?}", name, other),
                }
            }
        }
    }

    /// Symbolic parity: the saturating shift behaves identically to a hand-
    /// rolled per-lane ITE chain over Z3 bvshl/bvlshr/bvashr + round-trip
    /// overflow detection. There is no `_op_generic_QShl` in claripy, so the
    /// reference is the same algorithm encoded straight from the spec. This
    /// catches encoding mistakes (wrong cap, swapped then/else, sign-bit
    /// extraction errors) without depending on a Python reference.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vqshl_16x4_symbolic_universal_unsigned() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "qshl_a_16x4", 64);
        let b = RustBV::symbolic(&ctx, "qshl_b_16x4", 64);
        let got = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I16,
                count: 4,
                signed: false,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();
        // Reference: per-lane spec replay with the same primitives as the
        // helper. Any encoding drift will surface as a SAT counter-example.
        let umax = RustBV::concrete(0xFFFFu128, 16);
        let zero16 = RustBV::concrete(0, 16);
        let bit_one = RustBV::concrete(1, 1);
        let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
        for i in 0..4 {
            let lo = i * 16;
            let hi = lo + 15;
            let al = a.extract(hi, lo, &ctx);
            let bl = b.extract(hi, lo, &ctx);
            let shl_v = al.clone().shl_into(bl.clone(), &ctx);
            let recovered = shl_v.clone().lshr_into(bl.clone(), &ctx);
            let no_overflow = recovered.eq(&al, &ctx);
            let left_branch = no_overflow.ite(&shl_v, &umax, &ctx);
            let neg_amt = zero16.clone().sub_into(bl.clone(), &ctx);
            let right_branch = al.clone().lshr_into(neg_amt, &ctx);
            let amt_top = bl.extract(15, 15, &ctx);
            let amt_is_neg = amt_top.eq(&bit_one, &ctx);
            lanes.push(amt_is_neg.ite(&right_branch, &left_branch, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);
        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        let universal = !ctx.is_sat();
        ctx.pop();
        assert!(
            universal,
            "VQShlSat 16x4 (unsigned) must match the per-lane spec for all 64-bit inputs"
        );
    }

    /// Same parity check for the signed (QSal) branch — verifies the SMAX/
    /// SMIN cap selection from the data-sign bit.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_vqsal_16x4_symbolic_universal_signed() {
        let ctx = SymContext::new_mock();
        let a = RustBV::symbolic(&ctx, "qsal_a_16x4", 64);
        let b = RustBV::symbolic(&ctx, "qsal_b_16x4", 64);
        let got = VEXOps::binop(
            IROp::VQShlSat {
                elem: IRType::I16,
                count: 4,
                signed: true,
            },
            a.clone(),
            b.clone(),
            &ctx,
        )
        .unwrap();
        let smax = RustBV::concrete(0x7FFFu128, 16);
        let smin = RustBV::concrete(0x8000u128, 16);
        let zero16 = RustBV::concrete(0, 16);
        let bit_one = RustBV::concrete(1, 1);
        let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
        for i in 0..4 {
            let lo = i * 16;
            let hi = lo + 15;
            let al = a.extract(hi, lo, &ctx);
            let bl = b.extract(hi, lo, &ctx);
            let shl_v = al.clone().shl_into(bl.clone(), &ctx);
            let recovered = shl_v.clone().ashr_into(bl.clone(), &ctx);
            let no_overflow = recovered.eq(&al, &ctx);
            let a_top = al.extract(15, 15, &ctx);
            let a_is_neg = a_top.eq(&bit_one, &ctx);
            let cap = a_is_neg.ite(&smin, &smax, &ctx);
            let left_branch = no_overflow.ite(&shl_v, &cap, &ctx);
            let neg_amt = zero16.clone().sub_into(bl.clone(), &ctx);
            let right_branch = al.clone().ashr_into(neg_amt, &ctx);
            let amt_top = bl.extract(15, 15, &ctx);
            let amt_is_neg = amt_top.eq(&bit_one, &ctx);
            lanes.push(amt_is_neg.ite(&right_branch, &left_branch, &ctx));
        }
        let py = VEXOps::concat_le_elements(lanes, &ctx);
        ctx.push();
        ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
        let universal = !ctx.is_sat();
        ctx.pop();
        assert!(
            universal,
            "VQShlSat 16x4 (signed) must match the per-lane spec for all 64-bit inputs"
        );
    }
}
