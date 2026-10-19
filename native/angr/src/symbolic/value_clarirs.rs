//! `RustBV` → clarirs: [`build_term`], the clarirs term for an op over
//! values, which every constructor builds through (angr-mx00p.12), and
//! [`RustBV::to_clarirs`], the term a value is. The encodings:
//!
//! - a 1-bit comparison result is `ite(cmp, 1, 0)`, the encoding
//!   [`RustBV::from_clarirs`] uses for a Bool, and 1-bit `And` / `Or` /
//!   `Xor` / `Not` are the Bool ops under the same encoding;
//! - `Ite`'s condition is "nonzero";
//! - `Reverse` of a width that is not a multiple of 8, or is under 16, is the
//!   identity;
//! - `Clz` / `Ctz` / `Popcount` expand to an ite ladder and a bit sum, since
//!   clarirs has no such ops;
//! - a float op reads its operands as IEEE bit patterns and returns one.
//!
//! A zero-width value has no term.

use clarirs_core::prelude::{AstFactory, AstOp, AstRef, BitVec, ClarirsError, FPRM, FSort};
use num_bigint::BigUint;

use super::value::{BVOp, FloatOpKind, FloatPrec, RustBV};
use crate::claripy::ast::GLOBAL_CONTEXT;

type Term = AstRef<'static>;

fn bvv(value: u128, width: u32) -> Result<Term, ClarirsError> {
    GLOBAL_CONTEXT.bvv(BitVec::from((BigUint::from(value), width)))
}

/// A Bool as the engine's 1-bit encoding, `ite(b, 1, 0)`.
pub(crate) fn bit_of(b: Term) -> Result<Term, ClarirsError> {
    GLOBAL_CONTEXT.ite(b, bvv(1, 1)?, bvv(0, 1)?)
}

/// The Bool a bitvector stands for as a condition: "nonzero". Peels the
/// `ite(c, 1, 0)` encoding back to `c`.
pub(crate) fn bool_of(bits: &Term) -> Result<Term, ClarirsError> {
    if let Some(c) = bool_inside(bits) {
        return Ok(c);
    }
    GLOBAL_CONTEXT.neq(bits.clone(), bvv(0, bits.size())?)
}

/// The `c` of a 1-bit `ite(c, 1, 0)`: the Bool a value holds in the engine's
/// encoding, or `None` for any other term.
pub(crate) fn bool_inside(bits: &Term) -> Option<Term> {
    match bits.op() {
        AstOp::ITE(c, t, e) if bits.size() == 1 && is_bvv(t, 1) && is_bvv(e, 0) => Some(c.clone()),
        _ => None,
    }
}

fn is_bvv(term: &Term, value: u64) -> bool {
    matches!(term.op(), AstOp::BVV(v) if v.to_biguint() == BigUint::from(value))
}

impl RustBV {
    /// The clarirs term this value denotes. See the module doc for the
    /// encodings; `Err` for a float op or a zero-width value.
    pub(crate) fn to_clarirs(&self) -> Result<Term, ClarirsError> {
        match self {
            RustBV::Concrete { width: 0, .. } => Err(ClarirsError::InvalidArguments(
                "a zero-width value has no clarirs term".into(),
            )),
            RustBV::Concrete { value, width } => bvv(*value, *width),
            RustBV::Ast { ast, .. } => Ok(ast.clone()),
        }
    }
}

/// The clarirs term for `op` over `operands`, with the encodings in the
/// module doc. Constructors call it to build a value as a clarirs node; `Err`
/// when an operand has no term (a float op) or the op is a float op.
pub(crate) fn build_term(op: &BVOp, operands: &[RustBV], width: u32) -> Result<Term, ClarirsError> {
    let gc = &*GLOBAL_CONTEXT;
    let arg = |i: usize| -> Result<Term, ClarirsError> {
        operands
            .get(i)
            .ok_or_else(|| ClarirsError::InvalidArguments(format!("{op:?}: missing operand {i}")))?
            .to_clarirs()
    };
    // 1-bit logic is built on the Bools the bits stand for, so a condition
    // stays `ite(c, 1, 0)` and reaches Z3 and Python as a Bool.
    if width == 1 {
        let cond = |i: usize| bool_of(&arg(i)?);
        match op {
            BVOp::And => return bit_of(gc.and2(cond(0)?, cond(1)?)?),
            BVOp::Or => return bit_of(gc.or2(cond(0)?, cond(1)?)?),
            BVOp::Xor => return bit_of(gc.xor2(cond(0)?, cond(1)?)?),
            BVOp::Not => return bit_of(gc.not(cond(0)?)?),
            _ => {}
        }
    }
    Ok(match op {
        BVOp::Add => gc.add(arg(0)?, arg(1)?)?,
        BVOp::Sub => gc.sub(arg(0)?, arg(1)?)?,
        BVOp::Mul => gc.mul(arg(0)?, arg(1)?)?,
        BVOp::UDiv => gc.udiv(arg(0)?, arg(1)?)?,
        BVOp::SDiv => gc.sdiv(arg(0)?, arg(1)?)?,
        BVOp::URem => gc.urem(arg(0)?, arg(1)?)?,
        BVOp::SRem => gc.srem(arg(0)?, arg(1)?)?,
        BVOp::Neg => gc.neg(arg(0)?)?,
        BVOp::And => gc.and2(arg(0)?, arg(1)?)?,
        BVOp::Or => gc.or2(arg(0)?, arg(1)?)?,
        BVOp::Xor => gc.xor2(arg(0)?, arg(1)?)?,
        BVOp::Not => gc.not(arg(0)?)?,
        BVOp::Shl => gc.shl(arg(0)?, arg(1)?)?,
        BVOp::Lshr => gc.lshr(arg(0)?, arg(1)?)?,
        BVOp::Ashr => gc.ashr(arg(0)?, arg(1)?)?,
        BVOp::RotL => gc.rotate_left(arg(0)?, arg(1)?)?,
        BVOp::RotR => gc.rotate_right(arg(0)?, arg(1)?)?,
        BVOp::ZeroExt(bits) => gc.zero_ext(arg(0)?, *bits)?,
        BVOp::SignExt(bits) => gc.sign_ext(arg(0)?, *bits)?,
        BVOp::Extract(high, low) => gc.extract(arg(0)?, *high, *low)?,
        BVOp::Concat => gc.concat2(arg(0)?, arg(1)?)?,
        BVOp::Eq => bit_of(gc.eq_(arg(0)?, arg(1)?)?)?,
        BVOp::Ne => bit_of(gc.neq(arg(0)?, arg(1)?)?)?,
        BVOp::Ult => bit_of(gc.ult(arg(0)?, arg(1)?)?)?,
        BVOp::Ule => bit_of(gc.ule(arg(0)?, arg(1)?)?)?,
        BVOp::Ugt => bit_of(gc.ugt(arg(0)?, arg(1)?)?)?,
        BVOp::Uge => bit_of(gc.uge(arg(0)?, arg(1)?)?)?,
        BVOp::Slt => bit_of(gc.slt(arg(0)?, arg(1)?)?)?,
        BVOp::Sle => bit_of(gc.sle(arg(0)?, arg(1)?)?)?,
        BVOp::Sgt => bit_of(gc.sgt(arg(0)?, arg(1)?)?)?,
        BVOp::Sge => bit_of(gc.sge(arg(0)?, arg(1)?)?)?,
        BVOp::Ite => gc.ite(bool_of(&arg(0)?)?, arg(1)?, arg(2)?)?,
        BVOp::Reverse => {
            let inner = arg(0)?;
            let w = inner.size();
            if w.is_multiple_of(8) && w >= 16 {
                gc.byte_reverse(inner)?
            } else {
                inner
            }
        }
        BVOp::Clz | BVOp::Ctz => bit_scan(op, &arg(0)?, width)?,
        BVOp::Popcount => popcount(&arg(0)?, width)?,
        BVOp::Float { kind, prec } => float_term(*kind, *prec, operands)?,
    })
}

/// The clarirs sort of a float precision.
fn fsort(prec: FloatPrec) -> FSort {
    match prec {
        FloatPrec::F32 => FSort::f32(),
        FloatPrec::F64 => FSort::f64(),
    }
}

/// A VEX rounding mode (its low two bits) as a clarirs one, the mapping
/// `value_z3::vex_rm_to_z3` uses.
fn vex_rm(mode: u128) -> FPRM {
    match mode & 0x3 {
        0 => FPRM::NearestTiesToEven,
        1 => FPRM::TowardNegative,
        2 => FPRM::TowardPositive,
        _ => FPRM::TowardZero,
    }
}

/// `build` under the VEX rounding mode `rm` holds: one term for a concrete
/// mode, otherwise an `ite` over all four on `rm`'s low two bits, nested as
/// `value_z3::dispatch_symbolic_rm` nests them.
fn with_vex_rm(
    rm: &RustBV,
    build: impl Fn(FPRM) -> Result<Term, ClarirsError>,
) -> Result<Term, ClarirsError> {
    if let Some(mode) = rm.as_u128() {
        return build(vex_rm(mode));
    }
    let gc = &*GLOBAL_CONTEXT;
    let low = gc.extract(rm.to_clarirs()?, 1, 0)?;
    let is = |mode: u128| gc.eq_(low.clone(), bvv(mode, 2)?);
    let pick23 = gc.ite(is(2)?, build(vex_rm(2))?, build(vex_rm(3))?)?;
    let pick123 = gc.ite(is(1)?, build(vex_rm(1))?, pick23)?;
    gc.ite(is(0)?, build(vex_rm(0))?, pick123)
}

/// A float op over IEEE bit patterns, as `value_z3::build_fp_z3_ast_cached`
/// lowers it: each float operand is reinterpreted from its bits
/// (`bv_to_fp`), the result is converted back to bits (`fp_to_ieeebv`), and
/// a comparison is the 1-bit `ite(cmp, 1, 0)`. The ops without a rounding
/// operand round to nearest, ties to even; for the `*Rm` ops operand 0 is the
/// VEX rounding mode.
fn float_term(
    kind: FloatOpKind,
    prec: FloatPrec,
    operands: &[RustBV],
) -> Result<Term, ClarirsError> {
    use FloatOpKind as K;
    let gc = &*GLOBAL_CONTEXT;
    let sort = fsort(prec);
    let rne = FPRM::NearestTiesToEven;
    let arg = |i: usize| -> Result<Term, ClarirsError> {
        operands
            .get(i)
            .ok_or_else(|| {
                ClarirsError::InvalidArguments(format!("{kind:?}: missing operand {i}"))
            })?
            .to_clarirs()
    };
    let fp_in = |i: usize, sort: FSort| gc.bv_to_fp(arg(i)?, sort);
    let fp = |i: usize| fp_in(i, sort);
    let rm_of = || {
        operands.first().ok_or_else(|| {
            ClarirsError::InvalidArguments(format!("{kind:?}: missing rounding mode"))
        })
    };
    let to_int = |value: Term, dst_bits: u8, signed: bool, rm: FPRM| {
        if signed {
            gc.fp_to_sbv(value, u32::from(dst_bits), rm)
        } else {
            gc.fp_to_ubv(value, u32::from(dst_bits), rm)
        }
    };
    let float = match kind {
        K::CmpEq => return bit_of(gc.fp_eq(fp(0)?, fp(1)?)?),
        K::CmpLt => return bit_of(gc.fp_lt(fp(0)?, fp(1)?)?),
        K::CmpLe => return bit_of(gc.fp_leq(fp(0)?, fp(1)?)?),
        K::IsNaN => return bit_of(gc.fp_is_nan(fp(0)?)?),
        K::ConvertFtoI { dst_bits, signed } => return to_int(fp(0)?, dst_bits, signed, rne),
        K::ConvertFtoIRm { dst_bits, signed } => {
            let value = fp(1)?;
            return with_vex_rm(rm_of()?, |rm| to_int(value.clone(), dst_bits, signed, rm));
        }
        K::Add => gc.fp_add(fp(0)?, fp(1)?, rne)?,
        K::Sub => gc.fp_sub(fp(0)?, fp(1)?, rne)?,
        K::Mul => gc.fp_mul(fp(0)?, fp(1)?, rne)?,
        K::Div => gc.fp_div(fp(0)?, fp(1)?, rne)?,
        K::MaxNum => gc.fp_max(fp(0)?, fp(1)?)?,
        K::MinNum => gc.fp_min(fp(0)?, fp(1)?)?,
        K::Sqrt => gc.fp_sqrt(fp(0)?, rne)?,
        K::Neg => gc.fp_neg(fp(0)?)?,
        K::Abs => gc.fp_abs(fp(0)?)?,
        K::Fma => gc.fp_fma(fp(0)?, fp(1)?, fp(2)?, rne)?,
        K::Fms => gc.fp_fma(fp(0)?, fp(1)?, gc.fp_neg(fp(2)?)?, rne)?,
        K::ConvertItoF { signed, .. } => {
            if signed {
                gc.bv_to_fp_signed(arg(0)?, sort, rne)?
            } else {
                gc.bv_to_fp_unsigned(arg(0)?, sort, rne)?
            }
        }
        K::ConvertItoFRm { signed, .. } => {
            let value = arg(1)?;
            with_vex_rm(rm_of()?, |rm| {
                if signed {
                    gc.bv_to_fp_signed(value.clone(), sort, rm)
                } else {
                    gc.bv_to_fp_unsigned(value.clone(), sort, rm)
                }
            })?
        }
        K::ConvertFtoF { src_prec } => gc.fp_to_fp(fp_in(0, fsort(src_prec))?, sort, rne)?,
        K::ConvertFtoFRm { src_prec } => {
            let value = fp_in(1, fsort(src_prec))?;
            with_vex_rm(rm_of()?, |rm| gc.fp_to_fp(value.clone(), sort, rm))?
        }
        K::RoundToInt => {
            let value = fp(1)?;
            with_vex_rm(rm_of()?, |rm| gc.fp_round_to_integral(value.clone(), rm))?
        }
        K::AddRm | K::SubRm | K::MulRm | K::DivRm => {
            let (a, b) = (fp(1)?, fp(2)?);
            with_vex_rm(rm_of()?, |rm| match kind {
                K::AddRm => gc.fp_add(a.clone(), b.clone(), rm),
                K::SubRm => gc.fp_sub(a.clone(), b.clone(), rm),
                K::MulRm => gc.fp_mul(a.clone(), b.clone(), rm),
                _ => gc.fp_div(a.clone(), b.clone(), rm),
            })?
        }
        K::SqrtRm => {
            let a = fp(1)?;
            with_vex_rm(rm_of()?, |rm| gc.fp_sqrt(a.clone(), rm))?
        }
        K::FmaRm | K::FmsRm => {
            let (a, b, c) = (fp(1)?, fp(2)?, fp(3)?);
            let c = if kind == K::FmsRm { gc.fp_neg(c)? } else { c };
            with_vex_rm(rm_of()?, |rm| {
                gc.fp_fma(a.clone(), b.clone(), c.clone(), rm)
            })?
        }
    };
    gc.fp_to_ieeebv(float)
}

/// Accumulator width for a bitcount over `width` bits: enough for the value
/// `width`, the same as `value_z3::bitcount_acc_width`.
fn bitcount_acc_width(width: u32) -> u32 {
    (u32::BITS - width.leading_zeros()).min(width)
}

fn widen(acc: Term, acc_w: u32, width: u32) -> Result<Term, ClarirsError> {
    if acc_w < width {
        GLOBAL_CONTEXT.zero_ext(acc, width - acc_w)
    } else {
        Ok(acc)
    }
}

/// `Clz` / `Ctz` as the ite ladder `value_z3` builds: the bit tested last is
/// the outermost, so the most significant set bit (clz) or the least
/// significant one (ctz) wins.
fn bit_scan(op: &BVOp, x: &Term, width: u32) -> Result<Term, ClarirsError> {
    let gc = &*GLOBAL_CONTEXT;
    let acc_w = bitcount_acc_width(width);
    let one = bvv(1, 1)?;
    let mut result = bvv(u128::from(width), acc_w)?;
    let positions: Vec<u32> = match op {
        BVOp::Clz => (0..width).collect(),
        _ => (0..width).rev().collect(),
    };
    for pos in positions {
        let set = gc.eq_(gc.extract(x.clone(), pos, pos)?, one.clone())?;
        let count = match op {
            BVOp::Clz => width - 1 - pos,
            _ => pos,
        };
        result = gc.ite(set, bvv(u128::from(count), acc_w)?, result)?;
    }
    widen(result, acc_w, width)
}

/// `Popcount` as the sum of the zero-extended bits, as `value_z3` builds it.
fn popcount(x: &Term, width: u32) -> Result<Term, ClarirsError> {
    let gc = &*GLOBAL_CONTEXT;
    let acc_w = bitcount_acc_width(width);
    let bit = |pos: u32| -> Result<Term, ClarirsError> {
        let b = gc.extract(x.clone(), pos, pos)?;
        if acc_w > 1 {
            gc.zero_ext(b, acc_w - 1)
        } else {
            Ok(b)
        }
    };
    let mut acc = bit(0)?;
    for pos in 1..width {
        acc = gc.add(acc, bit(pos)?)?;
    }
    widen(acc, acc_w, width)
}

test_submod!(z3 "value_clarirs_tests.rs" => tests);
