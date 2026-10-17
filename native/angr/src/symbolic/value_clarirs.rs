//! `RustBV` → clarirs: the clarirs term a value denotes (angr-mx00p.12).
//!
//! [`RustBV::to_clarirs`] is the bridge that lets a constructor build a
//! [`RustBV::Ast`] over operands that are still in the older arms. The term
//! has the semantics the value's Z3 lowering (`value_z3.rs`) has:
//!
//! - a 1-bit comparison result is `ite(cmp, 1, 0)`, the encoding
//!   [`RustBV::from_clarirs`] uses for a Bool;
//! - `Ite`'s condition is "nonzero", as `build_z3_ast_cached` reads it;
//! - `Reverse` of a width that is not a multiple of 8, or is under 16, is the
//!   identity;
//! - `Clz` / `Ctz` / `Popcount` expand to the same ite ladder and bit sum the
//!   Z3 builder emits, since clarirs has no such ops.
//!
//! Float ops have no clarirs form yet and are refused, as is a zero-width
//! value; callers keep the older arm for those.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use clarirs_core::prelude::{AstFactory, AstOp, AstRef, BitVec, ClarirsError};
use num_bigint::BigUint;

use super::value::{BVOp, RustBV};
use crate::claripy::ast::GLOBAL_CONTEXT;

type Term = AstRef<'static>;

thread_local! {
    /// The term each `Expression` converted on this thread denotes, while the
    /// expression lives. Keyed by the operands `Arc`, which identifies an
    /// `Expression` node and is shared by its clones; confirmed by the `Weak`,
    /// which keeps the allocation from being reused. Without it, a chain of
    /// ops whose operands are still `Expression`s would reconvert the whole
    /// subtree at every op.
    static CONVERTED: RefCell<ConvertedCache> = RefCell::new(ConvertedCache::default());
}

#[derive(Default)]
struct ConvertedCache {
    map: HashMap<usize, (Weak<[RustBV]>, Term)>,
    sweep_at: usize,
}

impl ConvertedCache {
    const MIN_SWEEP: usize = 1024;

    fn key(operands: &Arc<[RustBV]>) -> usize {
        Arc::as_ptr(operands).cast::<RustBV>() as usize
    }

    fn get(&self, operands: &Arc<[RustBV]>) -> Option<Term> {
        let (node, term) = self.map.get(&Self::key(operands))?;
        std::ptr::addr_eq(node.as_ptr(), Arc::as_ptr(operands)).then(|| term.clone())
    }

    fn insert(&mut self, operands: &Arc<[RustBV]>, term: &Term) {
        if self.map.len() >= self.sweep_at {
            self.map.retain(|_, (node, _)| node.strong_count() > 0);
            self.sweep_at = (self.map.len() * 2).max(Self::MIN_SWEEP);
        }
        self.map.insert(
            Self::key(operands),
            (Arc::downgrade(operands), term.clone()),
        );
    }
}

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
    if let AstOp::ITE(c, t, e) = bits.op()
        && is_bvv(t, 1)
        && is_bvv(e, 0)
    {
        return Ok(c.clone());
    }
    GLOBAL_CONTEXT.neq(bits.clone(), bvv(0, bits.size())?)
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
            RustBV::Concrete { value, width } | RustBV::Constrained { value, width, .. } => {
                bvv(*value, *width)
            }
            RustBV::Symbolic { name, width, .. } => {
                match super::registry::strip_bool_symbol_name(name) {
                    Some(bool_name) => bit_of(GLOBAL_CONTEXT.bools(bool_name)?),
                    None => GLOBAL_CONTEXT.bvs(&**name, *width),
                }
            }
            RustBV::Ast { ast, .. } => Ok(ast.clone()),
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                if let Some(done) = CONVERTED.with_borrow(|c| c.get(operands)) {
                    return Ok(done);
                }
                let term = expression_term(op, operands, *width)?;
                #[cfg(test)]
                EXPRESSIONS_CONVERTED.set(EXPRESSIONS_CONVERTED.get() + 1);
                CONVERTED.with_borrow_mut(|c| c.insert(operands, &term));
                Ok(term)
            }
        }
    }
}

fn expression_term(op: &BVOp, operands: &[RustBV], width: u32) -> Result<Term, ClarirsError> {
    let gc = &*GLOBAL_CONTEXT;
    let arg = |i: usize| -> Result<Term, ClarirsError> {
        operands
            .get(i)
            .ok_or_else(|| ClarirsError::InvalidArguments(format!("{op:?}: missing operand {i}")))?
            .to_clarirs()
    };
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
        BVOp::Float { .. } => {
            return Err(ClarirsError::UnsupportedOperation(format!(
                "{op:?} has no clarirs form"
            )));
        }
    })
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

#[cfg(test)]
thread_local! {
    /// `Expression` nodes converted (cache misses) on this thread.
    static EXPRESSIONS_CONVERTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

test_submod!(z3 "value_clarirs_tests.rs" => tests);
