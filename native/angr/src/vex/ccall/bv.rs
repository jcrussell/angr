//! `Bv`: the bitvector the symbolic ccall builders compute with, a clarirs
//! term from the global context.
//!
//! The builders chain many operations, so a `Bv` is `None` once any step
//! fails instead of failing at each one: every operation on a `None`
//! operand stays `None`, and [`Bv::into_ast`] hands that back to the caller
//! as "Rust does not serve this call". Comparisons return the 1-bit
//! `ite(c, 1, 0)` encoding the helpers combine with bitwise operations.
//!
//! Terms are built unsimplified; [`Bv::into_ast`] simplifies the result once.

use clarirs_core::prelude::*;
use num_bigint::BigUint;

use crate::claripy::ast::GLOBAL_CONTEXT;

type Term = AstRef<'static>;

#[derive(Clone, Debug)]
pub(crate) struct Bv(Option<Term>);

impl Bv {
    pub(crate) fn from_ast(ast: Term) -> Self {
        Bv(Some(ast))
    }

    /// A `width`-bit constant holding the low `width` bits of `value`.
    pub(crate) fn concrete(value: u128, width: u32) -> Self {
        Bv(GLOBAL_CONTEXT
            .bvv(BitVec::from((BigUint::from(value), width)))
            .ok())
    }

    /// The simplified term, or `None` if any step failed.
    pub(crate) fn into_ast(self) -> Option<Term> {
        self.0.and_then(|ast| ast.simplify().ok())
    }

    /// The width in bits; 0 once a step has failed.
    pub(crate) fn width(&self) -> u32 {
        match self.0.as_ref().map(|ast| ast.ast_type()) {
            Some(AstType::BitVec(width)) => width,
            _ => 0,
        }
    }

    /// The value of a constant of at most 64 bits, as written (no folding).
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self.0.as_ref()?.op() {
            AstOp::BVV(bv) => bv.to_u64(),
            _ => None,
        }
    }

    fn map(&self, f: impl FnOnce(&Term) -> Result<Term, ClarirsError>) -> Self {
        Bv(self.0.as_ref().and_then(|a| f(a).ok()))
    }

    fn zip(
        &self,
        other: &Self,
        f: impl FnOnce(&Term, &Term) -> Result<Term, ClarirsError>,
    ) -> Self {
        Bv(match (&self.0, &other.0) {
            (Some(a), Some(b)) => f(a, b).ok(),
            _ => None,
        })
    }

    /// A Bool term as a 1-bit bitvector.
    fn bit(b: Result<Term, ClarirsError>) -> Result<Term, ClarirsError> {
        let ctx = &*GLOBAL_CONTEXT;
        ctx.ite(
            b?,
            ctx.bvv(BitVec::from((1u64, 1)))?,
            ctx.bvv(BitVec::from((0u64, 1)))?,
        )
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        self.zip(other, |a, b| GLOBAL_CONTEXT.add(a, b))
    }

    pub(crate) fn sub(&self, other: &Self) -> Self {
        self.zip(other, |a, b| GLOBAL_CONTEXT.sub(a, b))
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        self.zip(other, |a, b| GLOBAL_CONTEXT.mul(a, b))
    }

    pub(crate) fn and(&self, other: &Self) -> Self {
        self.zip(other, |a, b| GLOBAL_CONTEXT.and2(a, b))
    }

    pub(crate) fn or(&self, other: &Self) -> Self {
        self.zip(other, |a, b| GLOBAL_CONTEXT.or2(a, b))
    }

    pub(crate) fn xor(&self, other: &Self) -> Self {
        self.zip(other, |a, b| GLOBAL_CONTEXT.xor2(a, b))
    }

    pub(crate) fn not(&self) -> Self {
        self.map(|a| GLOBAL_CONTEXT.not(a))
    }

    /// Shift left by `amount`, which has the same width.
    pub(crate) fn shl(&self, amount: &Self) -> Self {
        self.zip(amount, |a, b| GLOBAL_CONTEXT.shl(a, b))
    }

    pub(crate) fn eq(&self, other: &Self) -> Self {
        self.zip(other, |a, b| Self::bit(GLOBAL_CONTEXT.eq_(a, b)))
    }

    pub(crate) fn ne(&self, other: &Self) -> Self {
        self.zip(other, |a, b| Self::bit(GLOBAL_CONTEXT.neq(a, b)))
    }

    pub(crate) fn ult(&self, other: &Self) -> Self {
        self.zip(other, |a, b| Self::bit(GLOBAL_CONTEXT.ult(a, b)))
    }

    pub(crate) fn ule(&self, other: &Self) -> Self {
        self.zip(other, |a, b| Self::bit(GLOBAL_CONTEXT.ule(a, b)))
    }

    pub(crate) fn ugt(&self, other: &Self) -> Self {
        self.zip(other, |a, b| Self::bit(GLOBAL_CONTEXT.ugt(a, b)))
    }

    pub(crate) fn uge(&self, other: &Self) -> Self {
        self.zip(other, |a, b| Self::bit(GLOBAL_CONTEXT.uge(a, b)))
    }

    /// Bits `high..=low`.
    pub(crate) fn extract(&self, high: u32, low: u32) -> Self {
        self.map(|a| GLOBAL_CONTEXT.extract(a, high, low))
    }

    /// Zero-extend to `to_width` bits (not by `to_width` bits).
    pub(crate) fn zero_extend(&self, to_width: u32) -> Self {
        match to_width.checked_sub(self.width()) {
            Some(0) => self.clone(),
            Some(extra) => self.map(|a| GLOBAL_CONTEXT.zero_ext(a, extra)),
            None => Bv(None),
        }
    }

    /// Sign-extend to `to_width` bits (not by `to_width` bits).
    pub(crate) fn sign_extend(&self, to_width: u32) -> Self {
        match to_width.checked_sub(self.width()) {
            Some(0) => self.clone(),
            Some(extra) => self.map(|a| GLOBAL_CONTEXT.sign_ext(a, extra)),
            None => Bv(None),
        }
    }

    /// `self ? then : else_`, where `self` is a 1-bit condition.
    pub(crate) fn ite(&self, then: &Self, else_: &Self) -> Self {
        Bv(match (&self.0, &then.0, &else_.0) {
            (Some(c), Some(t), Some(e)) => (|| {
                let ctx = &*GLOBAL_CONTEXT;
                let cond = ctx.eq_(c, ctx.bvv(BitVec::from((1u64, 1)))?)?;
                ctx.ite(cond, t, e)
            })()
            .ok(),
            _ => None,
        })
    }

    /// A fresh `width`-bit symbol.
    #[cfg(test)]
    pub(crate) fn symbolic(name: impl AsRef<str>, width: u32) -> Self {
        Bv(GLOBAL_CONTEXT.bvs(name, width).ok())
    }

    /// The term as a clarirs AST, unsimplified.
    #[cfg(test)]
    pub(crate) fn ast(&self) -> Option<Term> {
        self.0.clone()
    }

    /// The value, once simplification folds the term to a constant.
    #[cfg(test)]
    pub(crate) fn const_value(&self) -> Option<u64> {
        Bv(self.clone().into_ast()).as_u64()
    }

    /// The value with `symbol` replaced by the constant `value`.
    #[cfg(test)]
    pub(crate) fn eval_with(&self, symbol: &Bv, value: u64) -> Option<u128> {
        let (ast, sym) = (self.0.as_ref()?, symbol.0.as_ref()?);
        let constant = GLOBAL_CONTEXT
            .bvv(BitVec::from((value, symbol.width())))
            .ok()?;
        let replaced = ast.replace(sym, &constant).ok()?;
        Bv(Some(replaced)).const_value().map(u128::from)
    }
}
