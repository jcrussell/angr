//! VEX operation implementations.
//!
//! This module implements VEX operations using a parameterized approach.
//! Instead of ~200 separate implementations (e.g., add8, add16, add32, add64),
//! we have a single implementation per operation type that handles all widths.

use std::sync::Arc;

use crate::symbolic::{BVOp, FloatOpKind, FloatPrec, RustBV, SymContext, VexOpFamily};

use super::ir::{IROp, IRType};
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
        | IROp::VFPwAdd { .. }
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
            | IROp::VFPwAdd { .. }
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

            // Misroute (a maintainer added this op to the top-level routing
            // guard but not here, or vice-versa). Degrade to the Python
            // fallback via the existing UnsupportedVexOp path rather than
            // aborting the process. See angr-cudgw.15.
            _ => Err(OpError::NotBinary(op)),
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

            // Misroute (guard/family drift): degrade to Python fallback
            // instead of panicking. See angr-cudgw.15.
            _ => Err(OpError::NotBinary(op)),
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

            // Misroute (guard/family drift): degrade to Python fallback
            // instead of panicking. See angr-cudgw.15.
            _ => Err(OpError::NotBinary(op)),
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

            // Misroute (guard/family drift): degrade to Python fallback
            // instead of panicking. See angr-cudgw.15.
            _ => Err(OpError::NotBinary(op)),
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

            // NEON pairwise FP add (Iop_PwAdd32Fx2).
            IROp::VFPwAdd { elem, count } => {
                Self::vec_float_pairwise_add(left, right, elem, count, ctx)
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

            // Misroute (guard/family drift): degrade to Python fallback
            // instead of panicking. See angr-cudgw.15.
            _ => Err(OpError::NotBinary(op)),
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

    // Vector element-wise binary op (vec_binop, the add/sub/mul packed-integer
    // family) lives in the `vec_binop` child module
    // (#[path = "ops_vec_binop.rs"] at the bottom of this file).

    // Vector sub-unit reversal (vec_reverse, Iop_Reverse*) and low-half
    // packed multiply (vec_mul_lo, PMULLD) live in the `vec_permute_mul`
    // child module (#[path = "ops_vec_permute_mul.rs"] at the bottom of this
    // file).

    // Vector element-wise compare (vec_cmp) and low/high interleave
    // (vec_interleave_lo/hi) live in the `vec_compare` child module
    // (#[path = "ops_vec_compare.rs"] at the bottom of this file).

    // =========================================================================
    // Float Operations (using bit manipulation for now)
    // =========================================================================
    //
    // Scalar float arith (float_neg/abs/sqrt/add/sub/mul/div/madd/msub) lives
    // in the `float_arith` child module (#[path = "ops_float_arith.rs"] at the
    // bottom of this file). The shared FloatLaneOp trait/structs/macros and the
    // build_float_expr/float_prec_of free fns stay here — they are reused by
    // the packed/vector float paths and the rounding-mode variants.
    //
    // SSE scalar-in-vector FP ops (vec_float_scalar_op/sqrt/minmax/fresh and
    // the fresh-per-lane reciprocal/rsqrt fallbacks) live in the
    // `vec_float_scalar` child module (#[path = "ops_vec_float_scalar.rs"] at
    // the bottom of this file).

    // =========================================================================
    // Packed FP arithmetic / unary / min-max
    // =========================================================================

    // The generic per-lane FP dispatcher `vec_float_lane_op` lives in the
    // `vec_float_lane` child module (#[path = "ops_vec_float_lane.rs"] at the
    // bottom of this file).

    // SetV128lo32 / SetV128lo64 (low-lane insertion) live in the
    // `vec_set_lo` child module (#[path = "ops_vec_set_lo.rs"] at the bottom
    // of this file).

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

/// Integer widening-multiply / divmod helpers, split out of this file
/// (angr-cudgw.18). Declared as a child module so its `pub(super)` methods
/// remain callable from the binop dispatch above and the shared helpers it
/// references (e.g. `Self::sign_extend_low_to_i128`) stay visible.
#[path = "ops_int_arith.rs"]
mod int_arith;

#[path = "ops_float_arith.rs"]
mod float_arith;

/// Float/int conversion ops (int↔float, float↔float, round-to-int, and the
/// rounding-mode binop variants), split out of this file (angr-cudgw.18).
/// Declared as a child module so its `pub(super)` methods remain callable from
/// the unop/binop dispatch above, and `build_float_expr` (shared free fn that
/// stays in this file) stays visible by the descendant rule.
#[path = "ops_conversions.rs"]
mod conversions;

/// Floating-point comparison ops (scalar FCmp/CmpF, SSE scalar-lane compare,
/// packed FP compare), split out of this file (angr-cudgw.18). Declared as a
/// child module so its `pub(super)` methods remain callable from the binop
/// dispatch above, and the shared free fns / `Self::concat_le_elements` they
/// reference (which stay in this file) stay visible by the descendant rule.
#[path = "ops_float_cmp.rs"]
mod float_cmp;

#[path = "ops_vec_lane.rs"]
mod vec_lane;

#[path = "ops_vec_shift.rs"]
mod vec_shift;

/// NEON/SIMD pairwise vector ops (Iop_PwAddL, Iop_PwAdd/PwMin/PwMax,
/// Iop_PwAdd32Fx2, Iop_Avg), split out of this file (angr-cudgw.18). Declared
/// as a child module so its `pub(super)` methods remain callable from the
/// unop/binop dispatch above, and the shared siblings they reference
/// (`Self::concat_le_elements`, `Self::vec_float_lane_op`, the `PwOp` enum, the
/// `FAdd` marker — all of which stay in this file) stay visible via super/the
/// descendant rule.
#[path = "ops_vec_pairwise.rs"]
mod vec_pairwise;

/// NEON/SIMD integer saturation ops (Iop_QNarrowUn/QNarrowBin, Iop_QAdd/QSub,
/// Iop_QShl/QSal), split out of this file (angr-cudgw.18). Declared as a child
/// module so its `pub(super)` methods remain callable from the unop/binop
/// dispatch above, and the shared siblings they reference
/// (`Self::sign_extend_low_to_i128`, `Self::concat_le_elements` — both stay in
/// this file) stay visible via the descendant rule.
#[path = "ops_vec_saturate.rs"]
mod vec_saturate;

/// NEON/SIMD per-lane bit-counting ops (Iop_Cnt8x, Iop_Clz/Cls{N}x{M}), split
/// out of this file (angr-cudgw.18). Declared as a child module so its
/// `pub(super)` methods remain callable from the unop dispatch above, and the
/// shared siblings they reference (`Self::concat_le_elements`, the
/// `LaneCountKind` enum — both stay in this file) stay visible via the
/// descendant rule.
#[path = "ops_vec_count.rs"]
mod vec_count;

/// NEON/SSE packed-integer arithmetic ops (Iop_Min/Max{U,S}, Iop_PolynomialMul,
/// Iop_Abs/PABS), split out of this file (angr-cudgw.18). Declared as a child
/// module so its `pub(super)` methods remain callable from the unop/binop
/// dispatch above, and the shared sibling they reference
/// (`Self::concat_le_elements`, which stays in this file) stays visible via the
/// descendant rule.
#[path = "ops_vec_int_arith.rs"]
mod vec_int_arith;

/// SSE scalar-in-vector FP ops (ADDSS/SUBSS/MULSS/DIVSS and F64 variants,
/// SQRTSS/SQRTSD, MAXSS/MINSS, RCPSS/RSQRTSS estimates) plus the
/// fresh-symbolic-per-lane reciprocal/rsqrt fallbacks, split out of this file
/// (angr-cudgw.18). Declared as a child module so its `pub(super)` methods
/// remain callable from the binop/unop dispatch above, and the shared siblings
/// they reference (`Self::concat_le_elements`, `build_float_expr`,
/// `float_prec_of`, all of which stay in this file) stay visible via the
/// descendant rule.
#[path = "ops_vec_float_scalar.rs"]
mod vec_float_scalar;

/// Vector element-wise comparison and low/high interleave ops, split out of
/// this file (angr-cudgw.18). Declared as a child module so its `pub(super)`
/// methods remain callable from the binop dispatch above, and the shared
/// sibling they reference (`Self::concat_le_elements`, which stays in this
/// file) stays visible via the descendant rule.
#[path = "ops_vec_compare.rs"]
mod vec_compare;

/// Vector sub-unit reversal (Iop_Reverse*) and low-half packed multiply
/// (PMULLD) ops, split out of this file (angr-cudgw.18). Declared as a child
/// module so its `pub(super)` methods remain callable from the unop/binop
/// dispatch above, and the shared siblings they reference
/// (`Self::concat_le_elements`, `Self::sign_extend_low_to_u64`, which stay in
/// this file) stay visible via the descendant rule.
#[path = "ops_vec_permute_mul.rs"]
mod vec_permute_mul;

/// V128 low-lane insertion ops (SetV128lo32 / SetV128lo64), extracted from
/// this file (angr-cudgw.18). Declared as a child module so its `pub(super)`
/// methods remain callable from the binop dispatch above.
#[path = "ops_vec_set_lo.rs"]
mod vec_set_lo;

/// Generic per-lane packed FP dispatcher (`vec_float_lane_op`), extracted from
/// this file (angr-cudgw.18). Declared as a child module so its `pub(super)`
/// method remains callable from the binop/unop dispatch above, and the shared
/// siblings it references (`Self::concat_le_elements`, the `float_prec_of` free
/// fn, the `FloatLaneOp` trait, `FLOAT_LANE_OP_MAX_ARITY`, all of which stay in
/// this file) stay visible via the descendant rule.
#[path = "ops_vec_float_lane.rs"]
mod vec_float_lane;

/// Generic packed-integer element-wise binop dispatcher (`vec_binop`, the
/// add/sub/mul family), extracted from this file (angr-cudgw.18). Declared as
/// a child module so its `pub(super)` method remains callable from the binop
/// dispatch above, and the shared sibling it references
/// (`Self::concat_le_elements`, which stays in this file) stays visible via
/// the descendant rule.
#[path = "ops_vec_binop.rs"]
mod vec_binop;

// angr-9hleg: the former monolithic ops_tests.rs (5288 lines) was split by
// family to mirror the ops_*.rs source modules. Shared SIMD lane-test helpers
// live in ops_test_helpers.rs (pub(super)); each ops_tests_<fam>.rs sibling
// imports them via `use super::ops_test_helpers::*`. All are children of `ops`
// so `use super::*` reaches ops's private items.
#[cfg(test)]
#[path = "ops_test_helpers.rs"]
mod ops_test_helpers;

#[cfg(test)]
#[path = "ops_tests_core.rs"]
mod tests_core;

#[cfg(test)]
#[path = "ops_tests_int_arith.rs"]
mod tests_int_arith;

#[cfg(test)]
#[path = "ops_tests_float_arith.rs"]
mod tests_float_arith;

#[cfg(test)]
#[path = "ops_tests_float_cmp.rs"]
mod tests_float_cmp;

#[cfg(test)]
#[path = "ops_tests_conversions.rs"]
mod tests_conversions;

#[cfg(test)]
#[path = "ops_tests_vec_lane.rs"]
mod tests_vec_lane;

#[cfg(test)]
#[path = "ops_tests_vec_shift.rs"]
mod tests_vec_shift;

#[cfg(test)]
#[path = "ops_tests_vec_saturate.rs"]
mod tests_vec_saturate;

#[cfg(test)]
#[path = "ops_tests_vec_count.rs"]
mod tests_vec_count;

#[cfg(test)]
#[path = "ops_tests_vec_int_arith.rs"]
mod tests_vec_int_arith;

#[cfg(test)]
#[path = "ops_tests_vec_pairwise.rs"]
mod tests_vec_pairwise;

#[cfg(test)]
#[path = "ops_tests_vec_permute_mul.rs"]
mod tests_vec_permute_mul;

#[cfg(test)]
#[path = "ops_tests_vec_float_lane.rs"]
mod tests_vec_float_lane;

#[cfg(test)]
#[path = "ops_tests_vec_float_scalar.rs"]
mod tests_vec_float_scalar;
