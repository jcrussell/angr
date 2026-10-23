//! Constant folds checked against Z3.
//!
//! Each test builds terms over constants, folds them with `simplify`, and
//! asserts that the result is what Z3 computes for the same unsimplified
//! term. Z3 is the reference because it is what clarirs lowers symbolic terms
//! to, so a fold that disagrees with it makes concrete and symbolic
//! evaluation disagree.

use clarirs_core::prelude::*;
use z3_sys::*;

use super::AstExtZ3;
use crate::{Z3_CONTEXT, rc::RcAst};

const ITERATIONS: usize = 200;

/// A xorshift generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.next() as usize % items.len()]
    }

    /// A `width`-bit value: an edge case, a small number, or a random pattern.
    fn bv(&mut self, width: u32) -> BitVec {
        let signed_max = (BitVec::ones(width) >> 1).unwrap();
        match self.next() % 9 {
            0 => BitVec::zeros(width),
            1 => BitVec::ones(width),
            2 => (!signed_max).unwrap(),
            3 => signed_max,
            4 => BitVec::from((u64::from(width), width)),
            5 | 6 => BitVec::from((self.next() % 70, width)),
            _ => {
                let random = BitVec::from((self.next(), 64))
                    .concat(&BitVec::from((self.next(), 64)))
                    .unwrap();
                random.extract(0, width - 1).unwrap()
            }
        }
    }

    /// A value of `fsort`: a special value, a value near 1, or any bit pattern.
    fn float(&mut self, fsort: FSort) -> Float {
        let specials = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.0,
            -1.0,
            0.5,
            2.5,
            -2.5,
        ];
        let near_one = f64::from_bits(self.next() >> 12 | 0x3ff << 52 | (self.next() & 1) << 63);
        let (choice, bits) = (self.next() % 3, self.next());
        let f = match choice {
            0 => self.pick(&specials),
            1 => near_one,
            _ => f64::from_bits(bits),
        };
        match (fsort == FSort::f32(), choice) {
            (true, 2) => Float::F32(f32::from_bits(bits as u32)),
            (true, _) => Float::F32(f as f32),
            (false, _) => Float::F64(f),
        }
    }

    fn string(&mut self) -> &'static str {
        self.pick(&["", "a", "ab", "abc", "bc", "abcabc", "xyz", "é"])
    }
}

/// What Z3 computes for `term`, read back as a clarirs AST.
fn z3_value<'c>(ctx: &'c Context<'c>, term: &AstRef<'c>) -> AstRef<'c> {
    Z3_CONTEXT.with(|z3_ctx| unsafe {
        let simplified = Z3_simplify(*z3_ctx, *term.to_z3().unwrap());
        AstRef::from_z3(ctx, RcAst::try_from(simplified).unwrap()).unwrap()
    })
}

/// Asserts that `term` folds to the value Z3 computes for it. Floats are
/// compared as IEEE bits, so the sign of a zero counts, and any NaN matches
/// any NaN.
fn assert_folds_like_z3<'c>(ctx: &'c Context<'c>, term: AstRef<'c>) {
    let folded = term.simplify().unwrap();
    if let AstOp::FPV(value) = folded.op() {
        if value.is_nan() {
            let is_nan = z3_value(ctx, &ctx.fp_is_nan(&term).unwrap());
            assert!(is_nan.is_true(), "{term:?} folds to NaN, Z3 disagrees");
        } else {
            let bits = z3_value(ctx, &ctx.fp_to_ieeebv(&term).unwrap());
            assert_eq!(
                ctx.bvv(value.to_ieee_bits()).unwrap(),
                bits,
                "{term:?} folds to {value:?}"
            );
        }
        return;
    }
    assert_eq!(folded, z3_value(ctx, &term), "{term:?}");
}

#[test]
fn bitvector_folds_match_z3() {
    let ctx = Context::new();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for width in [1, 8, 13, 64, 65, 128] {
        for _ in 0..ITERATIONS {
            let (a, b) = (rng.bv(width), rng.bv(width));
            let (x, y) = (ctx.bvv(a).unwrap(), ctx.bvv(b.clone()).unwrap());
            let check = |term| assert_folds_like_z3(&ctx, term);
            check(ctx.add(&x, &y).unwrap());
            check(ctx.sub(&x, &y).unwrap());
            check(ctx.mul(&x, &y).unwrap());
            check(ctx.urem(&x, &y).unwrap());
            check(ctx.srem(&x, &y).unwrap());
            // A division by zero is left unfolded on purpose: claripy raises
            if !b.is_zero() {
                check(ctx.udiv(&x, &y).unwrap());
                check(ctx.sdiv(&x, &y).unwrap());
            }
            check(ctx.shl(&x, &y).unwrap());
            check(ctx.lshr(&x, &y).unwrap());
            check(ctx.ashr(&x, &y).unwrap());
            check(ctx.rotate_left(&x, &y).unwrap());
            check(ctx.rotate_right(&x, &y).unwrap());
            check(ctx.and2(&x, &y).unwrap());
            check(ctx.or2(&x, &y).unwrap());
            check(ctx.xor2(&x, &y).unwrap());
            check(ctx.not(&x).unwrap());
            check(ctx.neg(&x).unwrap());
            check(ctx.concat2(&x, &y).unwrap());
            check(ctx.zero_ext(&x, 7).unwrap());
            check(ctx.sign_ext(&x, 7).unwrap());
            let low = (rng.next() % u64::from(width)) as u32;
            let high = low + (rng.next() % u64::from(width - low)) as u32;
            check(ctx.extract(&x, high, low).unwrap());
            if width % 8 == 0 {
                check(ctx.byte_reverse(&x).unwrap());
            }
            check(ctx.eq_(&x, &y).unwrap());
            check(ctx.neq(&x, &y).unwrap());
            check(ctx.ult(&x, &y).unwrap());
            check(ctx.ule(&x, &y).unwrap());
            check(ctx.ugt(&x, &y).unwrap());
            check(ctx.uge(&x, &y).unwrap());
            check(ctx.slt(&x, &y).unwrap());
            check(ctx.sle(&x, &y).unwrap());
            check(ctx.sgt(&x, &y).unwrap());
            check(ctx.sge(&x, &y).unwrap());
        }
    }
}

#[test]
fn float_folds_match_z3() {
    let ctx = Context::new();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let rne = FPRM::NearestTiesToEven;
    for fsort in [FSort::f32(), FSort::f64()] {
        for _ in 0..ITERATIONS {
            let (a, b) = (rng.float(fsort), rng.float(fsort));
            let (x, y) = (ctx.fpv(a).unwrap(), ctx.fpv(b).unwrap());
            let check = |term| assert_folds_like_z3(&ctx, term);
            check(ctx.fp_add(&x, &y, rne).unwrap());
            check(ctx.fp_sub(&x, &y, rne).unwrap());
            check(ctx.fp_mul(&x, &y, rne).unwrap());
            check(ctx.fp_div(&x, &y, rne).unwrap());
            check(ctx.fp_sqrt(&x, rne).unwrap());
            check(ctx.fp_neg(&x).unwrap());
            check(ctx.fp_abs(&x).unwrap());
            check(ctx.fp_eq(&x, &y).unwrap());
            check(ctx.fp_neq(&x, &y).unwrap());
            check(ctx.fp_lt(&x, &y).unwrap());
            check(ctx.fp_leq(&x, &y).unwrap());
            check(ctx.fp_gt(&x, &y).unwrap());
            check(ctx.fp_geq(&x, &y).unwrap());
            check(ctx.fp_is_nan(&x).unwrap());
            check(ctx.fp_is_inf(&x).unwrap());
            // SMT-LIB leaves the bits of a NaN unspecified
            if !a.is_nan() {
                check(ctx.fp_to_ieeebv(&x).unwrap());
            }
            // Out-of-range conversions to an integer are unspecified
            let v = a.to_f64().unwrap();
            if v.is_finite() && v.abs() < 2f64.powi(62) {
                check(ctx.fp_to_sbv(&x, 64, rne).unwrap());
                if v >= 0.0 {
                    check(ctx.fp_to_ubv(&x, 64, rne).unwrap());
                }
            }
        }
    }
}

#[test]
fn string_folds_match_z3() {
    let ctx = Context::new();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let index = |v: u64, width: u32| ctx.bvv(BitVec::from((v, width))).unwrap();
    for _ in 0..ITERATIONS {
        let (a, b, c) = (rng.string(), rng.string(), rng.string());
        let (x, y, z) = (
            ctx.stringv(a).unwrap(),
            ctx.stringv(b).unwrap(),
            ctx.stringv(c).unwrap(),
        );
        let check = |term| assert_folds_like_z3(&ctx, term);
        check(ctx.str_len(&x).unwrap());
        check(ctx.str_concat(&x, &y).unwrap());
        check(ctx.str_contains(&x, &y).unwrap());
        check(ctx.str_prefix_of(&x, &y).unwrap());
        check(ctx.str_suffix_of(&x, &y).unwrap());
        check(ctx.str_replace(&x, &y, &z).unwrap());
        check(ctx.str_eq(&x, &y).unwrap());
        let (start, length) = (
            rng.pick(&[0, 1, 2, 7, u64::MAX]),
            rng.pick(&[0, 1, 2, 9, u64::MAX]),
        );
        let width = rng.pick(&[64, 128]);
        // A length of u64::MAX is left out: the str.substr fold's start + length overflows
        if length != u64::MAX {
            check(
                ctx.str_substr(&x, index(start, width), index(length, width))
                    .unwrap(),
            );
        }
        // str_index_of is not checked here: from_z3 cannot read back the
        // int2bv(ite ...) term Z3 simplifies str.indexof to
    }
}
