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
    //
    // Scalar float arith (float_neg/abs/sqrt/add/sub/mul/div/madd/msub) lives
    // in the `float_arith` child module (#[path = "ops_float_arith.rs"] at the
    // bottom of this file). The shared FloatLaneOp trait/structs/macros and the
    // build_float_expr/float_prec_of free fns stay here — they are reused by
    // the packed/vector float paths and the rounding-mode variants.

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

    /// NEON pairwise FP add — `Iop_PwAdd32Fx2` (ARM VPADD.F32). Binary; the FP
    /// analogue of `vec_pairwise_binop` with `PwOp::Add`, but the per-pair
    /// combine is an FP add (via the `FAdd` `FloatLaneOp` so both the concrete
    /// and symbolic branches stay in lockstep). Output lane shape matches the
    /// inputs: first half from `left`, second half from `right`. For the only
    /// VEX-emitted shape (`32Fx2`) this yields `[a0+a1, b0+b1]`.
    fn vec_float_pairwise_add(
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
        debug_assert!(count >= 2 && count.is_multiple_of(2));
        let half = count / 2;

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        // First half from `left`, then second half from `right` (same
        // interleave order as the integer `vec_pairwise_binop`).
        for src in [&left, &right] {
            for i in 0..half {
                let lo_a = (2 * i as u32) * elem_width;
                let hi_a = lo_a + elem_width - 1;
                let lo_b = (2 * i as u32 + 1) * elem_width;
                let hi_b = lo_b + elem_width - 1;
                let a = src.extract(hi_a, lo_a, ctx);
                let b = src.extract(hi_b, lo_b, ctx);
                // Reuse the single-lane FP-add path (count=1) so the concrete
                // and symbolic branches match the packed `VFAdd`.
                elements.push(Self::vec_float_lane_op(&[a, b], elem, 1, &FAdd, ctx)?);
            }
        }
        Ok(Self::concat_le_elements(elements, ctx))
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

#[cfg(test)]
#[path = "ops_tests.rs"]
mod ops_tests;
