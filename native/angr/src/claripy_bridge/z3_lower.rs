//! Lower a clarirs AST into the engine's thread-local Z3 context.
//!
//! claripy is clarirs: a Python claripy AST wraps an [`AstRef`], and its own
//! Z3 backend (`clarirs-z3`) builds terms in a private per-thread context the
//! engine cannot see. This module is the engine's equivalent of that backend's
//! `to_z3`, targeting the context `z3::Context::thread_local()` names instead,
//! so the result can be asserted on, or evaluated against, a `SymContext`'s
//! solver. It is lossless for every Bool / bitvector / float op, which is what
//! lets `RustSolverContext` and the constraint-import tiers accept the ops
//! `claripy_to_rustbv` has no `RustBV` form for (fp comparisons, Bool `Xor`).
//!
//! Op semantics mirror `clarirs-z3`'s `AstExtZ3::to_z3` one for one (`SRem` is
//! `bvsrem`, `Neq` on floats is `not(fp.eq)`, a float sort has `mantissa + 1`
//! significand bits). Leaves become Z3 constants named exactly by the clarirs
//! symbol, which is also the name `RustBV::Symbolic` uses for a BV leaf
//! (`SymbolKind::rust_symbol_name` is the identity on BV names) — so a lowered
//! constraint binds the same Z3 constants as the engine's own terms.
//!
//! Strings and the VSA ops (`Union` / `Intersection` / `Widen`) are refused
//! with [`BridgeError::UnsupportedOp`]; callers treat that like any other
//! failed lowering and fall back to `claripy_to_rustbv`.

#![cfg(feature = "vex-engine-z3")]
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::CString;

use clarirs_core::prelude::*;
use pyo3::prelude::*;
use z3_sys::*;

use crate::claripy::ast::base::Base;
use crate::symbolic::Z3AstPtr;

use super::BridgeError;
use super::import::MAX_IMPORT_RECURSION_DEPTH;

/// Lower the clarirs AST behind a Python claripy object.
///
/// `Err(TypeMismatch)` when `obj` is not a claripy AST of *this* extension's
/// type — notably under `cargo test`, where the embedded interpreter imports
/// the installed `angr.rustylib`, whose `Base` is a different type object.
pub(crate) fn lower_claripy_ast(obj: &Bound<'_, PyAny>) -> Result<Z3AstPtr, BridgeError> {
    let base = obj
        .cast::<Base>()
        .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
    lower(&base.get().ast())
}

/// Lower a clarirs AST, sharing the lowering of repeated subtrees.
pub(crate) fn lower(ast: &AstRef<'static>) -> Result<Z3AstPtr, BridgeError> {
    // `depth` is cached on every node, so this bounds the recursion below
    // before it starts (same budget as the claripy import walk, angr-2a3i9).
    if ast.depth() > MAX_IMPORT_RECURSION_DEPTH {
        return Err(BridgeError::RecursionLimit(format!(
            "z3 lowering: AST depth {} exceeds {MAX_IMPORT_RECURSION_DEPTH}",
            ast.depth()
        )));
    }
    let ctx = z3::Context::thread_local();
    let mut memo = HashMap::new();
    Lowerer {
        ctx: &ctx,
        memo: &mut memo,
    }
    .lower(ast)
}

struct Lowerer<'a> {
    ctx: &'a z3::Context,
    /// Keyed by node address: an `AstRef` is interned, so equal subtrees are
    /// one node, and every key's node is kept alive by the root being lowered.
    memo: &'a mut HashMap<*const AstNode<'static>, Z3AstPtr>,
}

fn conversion(msg: impl Into<String>) -> BridgeError {
    BridgeError::InvalidArgs(msg.into())
}

impl Lowerer<'_> {
    fn raw_ctx(&self) -> Z3_context {
        self.ctx.get_z3_context()
    }

    /// Take ownership of a freshly built term: `Z3_mk_*` returns it with
    /// refcount 0, and `from_borrowed_raw` takes the first reference before
    /// any other Z3 call can run.
    fn own(&self, raw: Option<Z3_ast>) -> Result<Z3AstPtr, BridgeError> {
        let raw = raw.ok_or_else(|| conversion("Z3 returned a null term"))?;
        // SAFETY: `raw` was just returned by a `Z3_mk_*` call on `self.ctx`.
        unsafe { Z3AstPtr::from_borrowed_raw(self.ctx, raw.as_ptr() as usize) }
            .ok_or_else(|| conversion("Z3 returned a null term"))
    }

    fn symbol(&self, name: &str) -> Result<Z3_symbol, BridgeError> {
        let name = CString::new(name).map_err(|e| conversion(e.to_string()))?;
        // SAFETY: `name` outlives the call; Z3 copies the string.
        unsafe { Z3_mk_string_symbol(self.raw_ctx(), name.as_ptr()) }
            .ok_or_else(|| conversion("Z3_mk_string_symbol returned null"))
    }

    fn constant(&self, name: &str, sort: Z3_sort) -> Result<Z3AstPtr, BridgeError> {
        let sym = self.symbol(name)?;
        // SAFETY: `sym` and `sort` belong to this context.
        self.own(unsafe { Z3_mk_const(self.raw_ctx(), sym, sort) })
    }

    fn bv_sort(&self, width: u32) -> Result<Z3_sort, BridgeError> {
        if width == 0 {
            return Err(conversion("zero-width bitvector"));
        }
        // SAFETY: plain sort constructor on a live context.
        unsafe { Z3_mk_bv_sort(self.raw_ctx(), width) }
            .ok_or_else(|| conversion("Z3_mk_bv_sort returned null"))
    }

    fn fp_sort(&self, sort: FSort) -> Result<Z3_sort, BridgeError> {
        // SAFETY: plain sort constructor on a live context.
        unsafe { Z3_mk_fpa_sort(self.raw_ctx(), sort.exponent, sort.mantissa + 1) }
            .ok_or_else(|| conversion("Z3_mk_fpa_sort returned null"))
    }

    fn rounding_mode(&self, rm: FPRM) -> Result<Z3AstPtr, BridgeError> {
        let c = self.raw_ctx();
        // SAFETY: nullary constructors on a live context.
        self.own(unsafe {
            match rm {
                FPRM::NearestTiesToEven => Z3_mk_fpa_rne(c),
                FPRM::TowardPositive => Z3_mk_fpa_rtp(c),
                FPRM::TowardNegative => Z3_mk_fpa_rtn(c),
                FPRM::TowardZero => Z3_mk_fpa_rtz(c),
                FPRM::NearestTiesToAway => Z3_mk_fpa_rna(c),
            }
        })
    }

    fn numeral(&self, value: &str, sort: Z3_sort) -> Result<Z3AstPtr, BridgeError> {
        let value = CString::new(value).map_err(|e| conversion(e.to_string()))?;
        // SAFETY: `value` outlives the call; `sort` belongs to this context.
        self.own(unsafe { Z3_mk_numeral(self.raw_ctx(), value.as_ptr(), sort) })
    }

    /// Left-fold a binary constructor over `children` (n-ary bitvector ops,
    /// `Concat`, Bool `Xor`).
    fn fold(
        &self,
        children: &[Z3AstPtr],
        op: unsafe extern "C" fn(Z3_context, Z3_ast, Z3_ast) -> Option<Z3_ast>,
    ) -> Result<Z3AstPtr, BridgeError> {
        let (first, rest) = children
            .split_first()
            .ok_or_else(|| conversion("n-ary op with no operands"))?;
        let mut acc = first.clone_ref();
        for next in rest {
            // SAFETY: both operands are live terms of this context, held by
            // `acc` / `children` across the call.
            acc = self.own(unsafe { op(self.raw_ctx(), acc.as_z3_ast(), next.as_z3_ast()) })?;
        }
        Ok(acc)
    }

    fn lower(&mut self, ast: &AstRef<'static>) -> Result<Z3AstPtr, BridgeError> {
        let key = std::sync::Arc::as_ptr(ast);
        if let Some(done) = self.memo.get(&key) {
            return Ok(done.clone_ref());
        }
        let children = ast
            .child_iter()
            .map(|child| self.lower(&child))
            .collect::<Result<Vec<_>, _>>()?;
        let term = self.node(ast, &children)?;
        self.memo.insert(key, term.clone_ref());
        Ok(term)
    }

    /// Build one node from its already-lowered children.
    #[allow(clippy::too_many_lines, reason = "one arm per AstOp variant")]
    fn node(&self, ast: &AstRef<'static>, ch: &[Z3AstPtr]) -> Result<Z3AstPtr, BridgeError> {
        let c = self.raw_ctx();
        let arg = |i: usize| -> Result<Z3_ast, BridgeError> {
            ch.get(i)
                .map(Z3AstPtr::as_z3_ast)
                .ok_or_else(|| conversion(format!("missing operand {i}")))
        };
        let is_bool = ast.ast_type().is_bool();
        // SAFETY (every `unsafe` below): operands come from `ch`, which holds
        // a reference to each for the whole of this call, and every sort or
        // rounding-mode term is built on, and owned in, the same context.
        let raw = match ast.op() {
            AstOp::BoolS(name) => {
                let sort = unsafe { Z3_mk_bool_sort(c) }
                    .ok_or_else(|| conversion("Z3_mk_bool_sort returned null"))?;
                return self.constant(name.as_str(), sort);
            }
            AstOp::BoolV(b) => unsafe { if *b { Z3_mk_true(c) } else { Z3_mk_false(c) } },
            AstOp::BVS(name, width) => {
                let sort = self.bv_sort(*width)?;
                return self.constant(name.as_str(), sort);
            }
            AstOp::BVV(value) => {
                let sort = self.bv_sort(value.len())?;
                return self.numeral(&value.to_biguint().to_string(), sort);
            }
            AstOp::FPS(name, sort) => {
                let sort = self.fp_sort(*sort)?;
                return self.constant(name.as_str(), sort);
            }
            AstOp::FPV(value) => {
                let sort = self.fp_sort(value.fsort())?;
                unsafe {
                    match value {
                        Float::F32(v) => Z3_mk_fpa_numeral_float(c, *v, sort),
                        Float::F64(v) => Z3_mk_fpa_numeral_double(c, *v, sort),
                    }
                }
            }

            // Polymorphic Bool / bitvector ops.
            AstOp::Not(..) if is_bool => unsafe { Z3_mk_not(c, arg(0)?) },
            AstOp::Not(..) => unsafe { Z3_mk_bvnot(c, arg(0)?) },
            AstOp::And(..) | AstOp::Or(..) if is_bool => {
                let args: Vec<Z3_ast> = ch.iter().map(Z3AstPtr::as_z3_ast).collect();
                let n = u32::try_from(args.len()).map_err(|e| conversion(e.to_string()))?;
                unsafe {
                    if matches!(ast.op(), AstOp::And(..)) {
                        Z3_mk_and(c, n, args.as_ptr())
                    } else {
                        Z3_mk_or(c, n, args.as_ptr())
                    }
                }
            }
            AstOp::And(..) => return self.fold(ch, Z3_mk_bvand),
            AstOp::Or(..) => return self.fold(ch, Z3_mk_bvor),
            AstOp::Xor(..) if is_bool => return self.fold(ch, Z3_mk_xor),
            AstOp::Xor(..) => return self.fold(ch, Z3_mk_bvxor),
            AstOp::ITE(..) => unsafe { Z3_mk_ite(c, arg(0)?, arg(1)?, arg(2)?) },

            // Equality on any sort: IEEE semantics for floats.
            AstOp::Eq(a, _) if a.ast_type().is_float() => unsafe {
                Z3_mk_fpa_eq(c, arg(0)?, arg(1)?)
            },
            AstOp::Eq(..) => unsafe { Z3_mk_eq(c, arg(0)?, arg(1)?) },
            AstOp::Neq(a, _) if a.ast_type().is_float() => {
                let eq = self.own(unsafe { Z3_mk_fpa_eq(c, arg(0)?, arg(1)?) })?;
                unsafe { Z3_mk_not(c, eq.as_z3_ast()) }
            }
            AstOp::Neq(..) => {
                let pair = [arg(0)?, arg(1)?];
                unsafe { Z3_mk_distinct(c, 2, pair.as_ptr()) }
            }
            AstOp::ULT(..) => unsafe { Z3_mk_bvult(c, arg(0)?, arg(1)?) },
            AstOp::ULE(..) => unsafe { Z3_mk_bvule(c, arg(0)?, arg(1)?) },
            AstOp::UGT(..) => unsafe { Z3_mk_bvugt(c, arg(0)?, arg(1)?) },
            AstOp::UGE(..) => unsafe { Z3_mk_bvuge(c, arg(0)?, arg(1)?) },
            AstOp::SLT(..) => unsafe { Z3_mk_bvslt(c, arg(0)?, arg(1)?) },
            AstOp::SLE(..) => unsafe { Z3_mk_bvsle(c, arg(0)?, arg(1)?) },
            AstOp::SGT(..) => unsafe { Z3_mk_bvsgt(c, arg(0)?, arg(1)?) },
            AstOp::SGE(..) => unsafe { Z3_mk_bvsge(c, arg(0)?, arg(1)?) },

            // Bitvector arithmetic, shifts and structure.
            AstOp::Neg(..) => unsafe { Z3_mk_bvneg(c, arg(0)?) },
            AstOp::Add(..) => return self.fold(ch, Z3_mk_bvadd),
            AstOp::Mul(..) => return self.fold(ch, Z3_mk_bvmul),
            AstOp::Sub(..) => unsafe { Z3_mk_bvsub(c, arg(0)?, arg(1)?) },
            AstOp::UDiv(..) => unsafe { Z3_mk_bvudiv(c, arg(0)?, arg(1)?) },
            AstOp::SDiv(..) => unsafe { Z3_mk_bvsdiv(c, arg(0)?, arg(1)?) },
            AstOp::URem(..) => unsafe { Z3_mk_bvurem(c, arg(0)?, arg(1)?) },
            AstOp::SRem(..) => unsafe { Z3_mk_bvsrem(c, arg(0)?, arg(1)?) },
            AstOp::ShL(..) => unsafe { Z3_mk_bvshl(c, arg(0)?, arg(1)?) },
            AstOp::LShR(..) => unsafe { Z3_mk_bvlshr(c, arg(0)?, arg(1)?) },
            AstOp::AShR(..) => unsafe { Z3_mk_bvashr(c, arg(0)?, arg(1)?) },
            AstOp::RotateLeft(..) => unsafe { Z3_mk_ext_rotate_left(c, arg(0)?, arg(1)?) },
            AstOp::RotateRight(..) => unsafe { Z3_mk_ext_rotate_right(c, arg(0)?, arg(1)?) },
            AstOp::ZeroExt(_, bits) => unsafe { Z3_mk_zero_ext(c, *bits, arg(0)?) },
            AstOp::SignExt(_, bits) => unsafe { Z3_mk_sign_ext(c, *bits, arg(0)?) },
            AstOp::Extract(a, high, low) => {
                if *high >= a.size() || low > high {
                    return Err(conversion(format!(
                        "Extract({high}, {low}) out of range for a {}-bit operand",
                        a.size()
                    )));
                }
                unsafe { Z3_mk_extract(c, *high, *low, arg(0)?) }
            }
            AstOp::Concat(..) => return self.fold(ch, Z3_mk_concat),
            AstOp::ByteReverse(a) => {
                let size = a.size();
                if size == 0 || !size.is_multiple_of(8) {
                    return Err(conversion(format!(
                        "Reverse of a {size}-bit operand (needs a nonzero multiple of 8)"
                    )));
                }
                let src = arg(0)?;
                let bytes = (0..size / 8)
                    .map(|i| self.own(unsafe { Z3_mk_extract(c, i * 8 + 7, i * 8, src) }))
                    .collect::<Result<Vec<_>, _>>()?;
                return self.fold(&bytes, Z3_mk_concat);
            }

            // Floats.
            AstOp::FpNeg(..) => unsafe { Z3_mk_fpa_neg(c, arg(0)?) },
            AstOp::FpAbs(..) => unsafe { Z3_mk_fpa_abs(c, arg(0)?) },
            AstOp::FpLt(..) => unsafe { Z3_mk_fpa_lt(c, arg(0)?, arg(1)?) },
            AstOp::FpLeq(..) => unsafe { Z3_mk_fpa_leq(c, arg(0)?, arg(1)?) },
            AstOp::FpGt(..) => unsafe { Z3_mk_fpa_gt(c, arg(0)?, arg(1)?) },
            AstOp::FpGeq(..) => unsafe { Z3_mk_fpa_geq(c, arg(0)?, arg(1)?) },
            AstOp::FpIsNan(..) => unsafe { Z3_mk_fpa_is_nan(c, arg(0)?) },
            AstOp::FpIsInf(..) => unsafe { Z3_mk_fpa_is_infinite(c, arg(0)?) },
            AstOp::FpToIEEEBV(..) => unsafe { Z3_mk_fpa_to_ieee_bv(c, arg(0)?) },
            AstOp::FpFP(..) => unsafe { Z3_mk_fpa_fp(c, arg(0)?, arg(1)?, arg(2)?) },
            AstOp::BvToFp(_, sort) => {
                let sort = self.fp_sort(*sort)?;
                unsafe { Z3_mk_fpa_to_fp_bv(c, arg(0)?, sort) }
            }
            AstOp::FpAdd(_, _, rm)
            | AstOp::FpSub(_, _, rm)
            | AstOp::FpMul(_, _, rm)
            | AstOp::FpDiv(_, _, rm) => {
                let rm = self.rounding_mode(*rm)?;
                let (r, a, b) = (rm.as_z3_ast(), arg(0)?, arg(1)?);
                unsafe {
                    match ast.op() {
                        AstOp::FpAdd(..) => Z3_mk_fpa_add(c, r, a, b),
                        AstOp::FpSub(..) => Z3_mk_fpa_sub(c, r, a, b),
                        AstOp::FpMul(..) => Z3_mk_fpa_mul(c, r, a, b),
                        _ => Z3_mk_fpa_div(c, r, a, b),
                    }
                }
            }
            AstOp::FpSqrt(_, rm) => {
                let rm = self.rounding_mode(*rm)?;
                unsafe { Z3_mk_fpa_sqrt(c, rm.as_z3_ast(), arg(0)?) }
            }
            AstOp::FpToUBV(_, bits, rm) => {
                let rm = self.rounding_mode(*rm)?;
                unsafe { Z3_mk_fpa_to_ubv(c, rm.as_z3_ast(), arg(0)?, *bits) }
            }
            AstOp::FpToSBV(_, bits, rm) => {
                let rm = self.rounding_mode(*rm)?;
                unsafe { Z3_mk_fpa_to_sbv(c, rm.as_z3_ast(), arg(0)?, *bits) }
            }
            AstOp::FpToFp(_, sort, rm) => {
                let (sort, rm) = (self.fp_sort(*sort)?, self.rounding_mode(*rm)?);
                unsafe { Z3_mk_fpa_to_fp_float(c, rm.as_z3_ast(), arg(0)?, sort) }
            }
            AstOp::BvToFpSigned(_, sort, rm) => {
                let (sort, rm) = (self.fp_sort(*sort)?, self.rounding_mode(*rm)?);
                unsafe { Z3_mk_fpa_to_fp_signed(c, rm.as_z3_ast(), arg(0)?, sort) }
            }
            AstOp::BvToFpUnsigned(_, sort, rm) => {
                let (sort, rm) = (self.fp_sort(*sort)?, self.rounding_mode(*rm)?);
                unsafe { Z3_mk_fpa_to_fp_unsigned(c, rm.as_z3_ast(), arg(0)?, sort) }
            }

            AstOp::StrContains(..)
            | AstOp::StrPrefixOf(..)
            | AstOp::StrSuffixOf(..)
            | AstOp::StrIsDigit(..)
            | AstOp::StrLen(..)
            | AstOp::StrIndexOf(..)
            | AstOp::StrToBV(..)
            | AstOp::StringS(..)
            | AstOp::StringV(..)
            | AstOp::StrConcat(..)
            | AstOp::StrSubstr(..)
            | AstOp::StrReplace(..)
            | AstOp::BVToStr(..)
            | AstOp::Union(..)
            | AstOp::Intersection(..)
            | AstOp::Widen(..) => {
                use crate::claripy::ast::opstring::ToOpString;
                return Err(BridgeError::UnsupportedOp(ast.to_opstring()));
            }
        };
        self.own(raw)
    }
}

test_submod!("z3_lower_tests.rs" => tests);
