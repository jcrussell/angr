//! Per-lane vector op traits and the shared float-expression builders.
//!
//! Extracted from `ops/mod.rs` (angr-9ke6b.170) to shrink the dispatch file
//! down to the five `VEXOps` entry points. Holds the `FloatLaneOp` /
//! `IntLaneOp` contracts, their marker types and impl-generating macros, and
//! the `float_prec_of` / `build_float_expr` free fns the scalar-FP siblings
//! share. Declared as a child module of `ops` (plain `mod` decl in
//! `ops/mod.rs`), which re-exports every item below so the existing
//! `use super::{FloatLaneOp, build_float_expr, ...}` imports in the sibling
//! modules keep resolving unchanged.

use std::sync::Arc;

use crate::symbolic::{BVOp, FloatOpKind, FloatPrec, RustBV, SymContext};
use crate::vex::ir::IRType;

use super::VEXOps;

/// Map a VEX float `IRType` to a Z3 FP precision.
#[inline]
pub(super) fn float_prec_of(ty: IRType) -> Option<FloatPrec> {
    match ty {
        IRType::F32 => Some(FloatPrec::F32),
        IRType::F64 => Some(FloatPrec::F64),
        _ => None,
    }
}

/// Build a symbolic float-op expression. Used by float_* helpers below
/// when operands are not fully concrete; routes through Z3 FP theory in
/// `build_fp_z3_ast_cached`.
pub(super) fn build_float_expr(
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
        memo: Default::default(),
    }
}

/// Maximum arity supported by `FloatLaneOp`. Sized for the current set of
/// per-lane FP ops (unary Sqrt/Abs and binary Add/Sub/Mul/Div/Min/Max). Sized
/// to 2 today; bump if a ternary lane op is added (e.g. fused multiply-add).
pub(super) const FLOAT_LANE_OP_MAX_ARITY: usize = 2;

/// Per-lane FP op contract used by `VEXOps::vec_float_lane_op`.
///
/// Each impl must provide BOTH a concrete fast path (for f32/f64 lanes) and
/// a symbolic Z3 expression builder, so adding a new op cannot accidentally
/// drop one of the two branches — the previous duplicated functions
/// (`vec_float_op`, `vec_float_unop`, `vec_float_minmax`) made the symbolic
/// fallback easy to forget when extending.
pub(super) trait FloatLaneOp {
    /// Number of operand lanes consumed (1 for unary, 2 for binary).
    fn arity(&self) -> usize;
    /// Apply to a single concrete f32 lane. `args.len() == self.arity()`.
    fn concrete_f32(&self, args: &[f32]) -> f32;
    /// Apply to a single concrete f64 lane. `args.len() == self.arity()`.
    fn concrete_f64(&self, args: &[f64]) -> f64;
    /// Build the symbolic per-lane expression. `args.len() == self.arity()`.
    fn symbolic(&self, args: Vec<RustBV>, prec: FloatPrec, ctx: &SymContext) -> RustBV;
}

pub(super) struct FAdd;
pub(super) struct FSub;
pub(super) struct FMul;
pub(super) struct FDiv;
pub(super) struct FSqrt;
pub(super) struct FAbs;
/// FP min: matches Rust `<` semantics (NaN passes through right).
pub(super) struct FMin;
/// FP max: matches Rust `>` semantics (NaN passes through right).
pub(super) struct FMax;

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
#[allow(
    clippy::expect_used,
    reason = "arity-2 dispatch invariant: the only route here is `FloatLaneOp::symbolic` from `vec_float_lane_op`, which builds `lane_args` one-for-one from its `args`, and every `FMin`/`FMax` call site passes the array literal `&[left, right]` — see the module Panic policy header"
)]
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

/// Maximum arity supported by `IntLaneOp`. Sized for the current set of
/// per-lane integer ops (unary Abs and binary Add/Sub/Mul/CmpEQ/CmpGT/Min/Max).
pub(super) const INT_LANE_OP_MAX_ARITY: usize = 2;

/// Per-lane integer op contract used by `VEXOps::vec_int_lane_op`.
///
/// Mirrors [`FloatLaneOp`] for the packed-integer family: each impl provides
/// BOTH a concrete fast path (operating on `u128` lanes already masked to
/// `elem_width`) and a symbolic Z3 expression builder, so adding a new op
/// cannot accidentally drop one of the two branches. Replaces the four
/// near-identical per-lane loops that lived in `vec_binop`, `vec_cmp`,
/// `vec_int_minmax`, and `vec_int_abs`.
pub(super) trait IntLaneOp {
    /// Number of operand lanes consumed (1 for unary, 2 for binary).
    fn arity(&self) -> usize;
    /// Apply to concrete lanes already masked to `elem_width`. The driver
    /// re-masks the return value, so impls need not mask the low bits again.
    fn concrete_lane(&self, lanes: &[u128], elem_width: u32) -> u128;
    /// Build the symbolic per-lane expression. Must be exactly `elem_width`
    /// bits wide. `lanes.len() == self.arity()`.
    fn symbolic_lane(&self, lanes: &[RustBV], elem_width: u32, ctx: &SymContext) -> RustBV;
}

pub(super) struct IAdd;
pub(super) struct ISub;
pub(super) struct IMul;
/// Per-lane equality compare; lane is all-ones on equal, all-zeros otherwise.
pub(super) struct ICmpEq;
/// Per-lane greater-than; lane is all-ones when `l > r`. `signed` picks the
/// signed (`Iop_CmpGT{N}Sx{M}`) vs unsigned (`Iop_CmpGT{N}Ux{M}`) comparison.
pub(super) struct ICmpGt {
    pub(super) signed: bool,
}
/// Per-lane signed/unsigned integer min or max.
pub(super) struct IMinMax {
    pub(super) signed: bool,
    pub(super) is_max: bool,
}
/// Per-lane absolute value (PABS*); INT_MIN stays INT_MIN.
pub(super) struct IAbs;

/// Generate an `IntLaneOp` impl for a wrapping arithmetic binop whose concrete
/// path is a `u128::wrapping_*` and whose symbolic path is a single RustBV
/// method.
macro_rules! impl_int_lane_arith {
    ($name:ident, $wrapping:ident, $method:ident) => {
        impl IntLaneOp for $name {
            fn arity(&self) -> usize {
                2
            }
            fn concrete_lane(&self, a: &[u128], _elem_width: u32) -> u128 {
                a[0].$wrapping(a[1])
            }
            fn symbolic_lane(&self, a: &[RustBV], _elem_width: u32, ctx: &SymContext) -> RustBV {
                a[0].clone().$method(a[1].clone(), ctx)
            }
        }
    };
}

impl_int_lane_arith!(IAdd, wrapping_add, add_into);
impl_int_lane_arith!(ISub, wrapping_sub, sub_into);
impl_int_lane_arith!(IMul, wrapping_mul, mul_into);

impl IntLaneOp for ICmpEq {
    fn arity(&self) -> usize {
        2
    }
    fn concrete_lane(&self, a: &[u128], elem_width: u32) -> u128 {
        if a[0] == a[1] {
            VEXOps::low_bit_mask_u128(elem_width)
        } else {
            0
        }
    }
    fn symbolic_lane(&self, a: &[RustBV], elem_width: u32, ctx: &SymContext) -> RustBV {
        a[0].clone()
            .eq_into(a[1].clone(), ctx)
            .sign_extend_into(elem_width, ctx)
    }
}

impl IntLaneOp for ICmpGt {
    fn arity(&self) -> usize {
        2
    }
    fn concrete_lane(&self, a: &[u128], elem_width: u32) -> u128 {
        // Lanes arrive zero-extended into u128, so the unsigned compare is the
        // raw one; only the signed form needs the sign-extend first.
        let gt = if self.signed {
            let l = VEXOps::sign_extend_low_to_i128(a[0], elem_width);
            let r = VEXOps::sign_extend_low_to_i128(a[1], elem_width);
            l > r
        } else {
            a[0] > a[1]
        };
        if gt {
            VEXOps::low_bit_mask_u128(elem_width)
        } else {
            0
        }
    }
    fn symbolic_lane(&self, a: &[RustBV], elem_width: u32, ctx: &SymContext) -> RustBV {
        let cmp = if self.signed {
            a[0].clone().sgt_into(a[1].clone(), ctx)
        } else {
            a[0].clone().ugt_into(a[1].clone(), ctx)
        };
        cmp.sign_extend_into(elem_width, ctx)
    }
}

impl IntLaneOp for IMinMax {
    fn arity(&self) -> usize {
        2
    }
    fn concrete_lane(&self, a: &[u128], elem_width: u32) -> u128 {
        let pick_left = if self.signed {
            let l = VEXOps::sign_extend_low_to_i128(a[0], elem_width);
            let r = VEXOps::sign_extend_low_to_i128(a[1], elem_width);
            if self.is_max { l >= r } else { l <= r }
        } else if self.is_max {
            a[0] >= a[1]
        } else {
            a[0] <= a[1]
        };
        if pick_left { a[0] } else { a[1] }
    }
    fn symbolic_lane(&self, a: &[RustBV], _elem_width: u32, ctx: &SymContext) -> RustBV {
        let l = a[0].clone();
        let r = a[1].clone();
        let cond = match (self.signed, self.is_max) {
            (true, true) => l.clone().sge_into(r.clone(), ctx),
            (true, false) => l.clone().sle_into(r.clone(), ctx),
            (false, true) => l.clone().uge_into(r.clone(), ctx),
            (false, false) => l.clone().ule_into(r.clone(), ctx),
        };
        cond.ite_into(l, r, ctx)
    }
}

impl IntLaneOp for IAbs {
    fn arity(&self) -> usize {
        1
    }
    fn concrete_lane(&self, a: &[u128], elem_width: u32) -> u128 {
        let v = a[0];
        let sign_bit: u128 = 1u128 << (elem_width - 1);
        // |x| = (~x + 1) when negative (two's complement), else x.
        if v & sign_bit != 0 {
            (!v).wrapping_add(1)
        } else {
            v
        }
    }
    fn symbolic_lane(&self, a: &[RustBV], elem_width: u32, ctx: &SymContext) -> RustBV {
        let elem_val = a[0].clone();
        let zero = RustBV::concrete(0, elem_width);
        let neg = elem_val.clone().neg_into(ctx);
        let is_neg = elem_val.clone().slt_into(zero, ctx);
        is_neg.ite_into(neg, elem_val, ctx)
    }
}
