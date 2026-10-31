//! Oracles for the ops: each `SymBV` op against the clarirs term it stands
//! for, spelled with the raw constructors over `GLOBAL_CONTEXT` rather than
//! `build_term`, so the reference shares none of `SymBV`'s encodings. In the
//! reference, 1-bit logic is bitwise, a comparison is `ite(cmp, 1, 0)` and
//! `ite`'s condition is `c != 0`.

use clarirs_core::GLOBAL_CONTEXT;
use clarirs_core::prelude::*;
use num_bigint::BigUint;

use crate::SymBV;

type Term = AstRef<'static>;

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

    /// A number in `lo..=hi`.
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % u64::from(hi - lo + 1)) as u32
    }

    /// A `width`-bit value: an edge case, a small number, or a random pattern.
    fn bv(&mut self, width: u32) -> BitVec {
        let ones = BitVec::ones(width);
        let signed_max = (ones.clone() >> 1).unwrap();
        match self.next() % 9 {
            0 => BitVec::zeros(width),
            1 => ones,
            2 => (!signed_max).unwrap(),
            3 => signed_max,
            4 => BitVec::from((BigUint::from(width), width)),
            5 | 6 => BitVec::from((BigUint::from(self.next() % 70), width)),
            _ => {
                let mut random = BigUint::from(0u8);
                for _ in 0..width.div_ceil(64) {
                    random = (random << 64) | BigUint::from(self.next());
                }
                BitVec::from((random, width))
            }
        }
    }
}

fn bvv(value: BitVec) -> Term {
    GLOBAL_CONTEXT.bvv(value).unwrap()
}

fn bvv_u(value: u64, width: u32) -> Term {
    bvv(BitVec::from((BigUint::from(value), width)))
}

/// The value a literal term is, through `from_clarirs`'s literal fold.
fn value(term: &Term) -> SymBV {
    SymBV::from_clarirs(term.clone()).unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Neg,
    And,
    Or,
    Xor,
    Not,
    Shl,
    Lshr,
    Ashr,
    RotL,
    RotR,
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
    /// Extend to this width.
    ZeroExt(u32),
    SignExt(u32),
    /// Keep this many low bits.
    Truncate(u32),
    Extract(u32, u32),
    Concat,
    Ite,
    Reverse,
    Clz,
    Ctz,
    Popcount,
}

use Op::*;

const BINARY: [Op; 15] = [
    Add, Sub, Mul, UDiv, SDiv, URem, SRem, And, Or, Xor, Shl, Lshr, Ashr, RotL, RotR,
];
const UNARY: [Op; 6] = [Neg, Not, Reverse, Clz, Ctz, Popcount];
const COMPARE: [Op; 10] = [Eq, Ne, Ult, Ule, Ugt, Uge, Slt, Sle, Sgt, Sge];

impl Op {
    fn is_shift(self) -> bool {
        matches!(self, Shl | Lshr | Ashr)
    }

    /// The `SymBV` op.
    fn sym(self, args: &[SymBV]) -> SymBV {
        let a = &args[0];
        let b = || &args[1];
        match self {
            Add => a.add(b()),
            Sub => a.sub(b()),
            Mul => a.mul(b()),
            UDiv => a.udiv(b()),
            SDiv => a.sdiv(b()),
            URem => a.urem(b()),
            SRem => a.srem(b()),
            Neg => a.neg(),
            And => a.and(b()),
            Or => a.or(b()),
            Xor => a.xor(b()),
            Not => a.not(),
            Shl => a.shl(b()),
            Lshr => a.lshr(b()),
            Ashr => a.ashr(b()),
            RotL => a.rotl(b()),
            RotR => a.rotr(b()),
            Eq => a.eq(b()),
            Ne => a.ne(b()),
            Ult => a.ult(b()),
            Ule => a.ule(b()),
            Ugt => a.ugt(b()),
            Uge => a.uge(b()),
            Slt => a.slt(b()),
            Sle => a.sle(b()),
            Sgt => a.sgt(b()),
            Sge => a.sge(b()),
            ZeroExt(to) => a.zero_extend(to),
            SignExt(to) => a.sign_extend(to),
            Truncate(to) => a.truncate(to),
            Extract(high, low) => a.extract(high, low),
            Concat => a.concat(b()),
            Ite => a.ite(b(), &args[2]),
            Reverse => a.reverse(),
            Clz => a.clz(),
            Ctz => a.ctz(),
            Popcount => a.popcount(),
        }
    }

    /// The reference term.
    fn raw(self, args: &[Term]) -> Term {
        let ctx = &*GLOBAL_CONTEXT;
        let a = args[0].clone();
        let b = || args[1].clone();
        let w = a.size();
        let bit = |cond: Term| ctx.ite(cond, bvv_u(1, 1), bvv_u(0, 1)).unwrap();
        let sum = |terms: Vec<Term>| {
            terms
                .into_iter()
                .reduce(|acc, t| ctx.add(acc, t).unwrap())
                .unwrap()
        };
        match self {
            Add => ctx.add(a, b()).unwrap(),
            Sub => ctx.sub(a, b()).unwrap(),
            Mul => ctx.mul(a, b()).unwrap(),
            UDiv => ctx.udiv(a, b()).unwrap(),
            SDiv => ctx.sdiv(a, b()).unwrap(),
            URem => ctx.urem(a, b()).unwrap(),
            SRem => ctx.srem(a, b()).unwrap(),
            Neg => ctx.neg(a).unwrap(),
            And => ctx.and2(a, b()).unwrap(),
            Or => ctx.or2(a, b()).unwrap(),
            Xor => ctx.xor2(a, b()).unwrap(),
            Not => ctx.not(a).unwrap(),
            Shl => ctx.shl(a, b()).unwrap(),
            Lshr => ctx.lshr(a, b()).unwrap(),
            Ashr => ctx.ashr(a, b()).unwrap(),
            RotL => ctx.rotate_left(a, b()).unwrap(),
            RotR => ctx.rotate_right(a, b()).unwrap(),
            Eq => bit(ctx.eq_(a, b()).unwrap()),
            Ne => bit(ctx.neq(a, b()).unwrap()),
            Ult => bit(ctx.ult(a, b()).unwrap()),
            Ule => bit(ctx.ule(a, b()).unwrap()),
            Ugt => bit(ctx.ugt(a, b()).unwrap()),
            Uge => bit(ctx.uge(a, b()).unwrap()),
            Slt => bit(ctx.slt(a, b()).unwrap()),
            Sle => bit(ctx.sle(a, b()).unwrap()),
            Sgt => bit(ctx.sgt(a, b()).unwrap()),
            Sge => bit(ctx.sge(a, b()).unwrap()),
            ZeroExt(to) | SignExt(to) if to == w => a,
            ZeroExt(to) => ctx.zero_ext(a, to - w).unwrap(),
            SignExt(to) => ctx.sign_ext(a, to - w).unwrap(),
            Truncate(to) => ctx.extract(a, to - 1, 0).unwrap(),
            Extract(high, low) => ctx.extract(a, high, low).unwrap(),
            Concat => ctx.concat2(a, b()).unwrap(),
            Ite => {
                let zero = bvv(BitVec::zeros(w));
                ctx.ite(ctx.neq(a, zero).unwrap(), b(), args[2].clone())
                    .unwrap()
            }
            // Byte 0 moves to the top; widths that are not whole bytes, or a
            // single byte, are left as they are.
            Reverse if !w.is_multiple_of(8) || w < 16 => a,
            Reverse => (0..w / 8)
                .map(|i| ctx.extract(a.clone(), 8 * i + 7, 8 * i).unwrap())
                .reduce(|acc, byte| ctx.concat2(acc, byte).unwrap())
                .unwrap(),
            // clz(x) counts the k in 0..w with x < 2^k.
            Clz => sum((0..w)
                .map(|k| {
                    let pow = bvv(BitVec::from((BigUint::from(1u8) << k, w)));
                    ctx.zero_ext(bit(ctx.ult(a.clone(), pow).unwrap()), w - 1)
                        .unwrap()
                })
                .collect()),
            // ctz(x) counts the k in 1..=w whose low k bits are all zero.
            Ctz => sum((1..=w)
                .map(|k| {
                    let low = ctx.extract(a.clone(), k - 1, 0).unwrap();
                    let zero = ctx.eq_(low, bvv(BitVec::zeros(k))).unwrap();
                    ctx.zero_ext(bit(zero), w - 1).unwrap()
                })
                .collect()),
            Popcount => sum((0..w)
                .map(|i| {
                    let bit = ctx.extract(a.clone(), i, i).unwrap();
                    ctx.zero_ext(bit, w - 1).unwrap()
                })
                .collect()),
        }
    }
}

// ---------------------------------------------------------------------------
// Suite 1: on constants, each op's fold equals clarirs's simplify of its term.
// ---------------------------------------------------------------------------

const FOLD_WIDTHS: [u32; 10] = [1, 8, 13, 63, 64, 65, 127, 128, 129, 256];
const FOLD_ITERATIONS: usize = 200;

fn assert_folds_like_clarirs(op: Op, args: &[Term]) {
    let values: Vec<SymBV> = args.iter().map(value).collect();
    let folded = op.sym(&values).to_clarirs().unwrap().simplify().unwrap();
    let reference = op.raw(args).simplify().unwrap();
    assert_eq!(folded, reference, "{op:?} over {args:?}");
}

fn shift_amount_is_large(amount: &BitVec) -> bool {
    amount.to_biguint() >= BigUint::from(1u64 << 32)
}

#[test]
fn fold_matches_clarirs_simplify() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for width in FOLD_WIDTHS {
        for i in 0..FOLD_ITERATIONS {
            let (a, b) = (rng.bv(width), rng.bv(width));
            let divisor_is_zero = b.is_zero();
            let b_value = b.clone();
            let (x, y) = (bvv(a), bvv(b));
            let check = |op: Op, args: &[Term]| assert_folds_like_clarirs(op, args);
            for op in BINARY {
                // A division by zero is left unfolded on purpose: claripy raises.
                if divisor_is_zero && matches!(op, UDiv | SDiv | URem | SRem) {
                    continue;
                }
                // clarirs's shift folds misread an amount of 2^32 or more, and
                // any amount wider than 64 bits (angr-mx00p.105).
                if op.is_shift() && (width > 64 || shift_amount_is_large(&b_value)) {
                    continue;
                }
                check(op, &[x.clone(), y.clone()]);
            }
            for op in UNARY.into_iter().chain(COMPARE) {
                // The bit-count references are a term per bit; a sample is enough.
                if matches!(op, Clz | Ctz | Popcount) && i % 8 != 0 {
                    continue;
                }
                let args = [x.clone(), y.clone()];
                check(op, &args[..if COMPARE.contains(&op) { 2 } else { 1 }]);
            }
            let to = width + rng.range(0, 70);
            check(ZeroExt(to), std::slice::from_ref(&x));
            check(SignExt(to), std::slice::from_ref(&x));
            check(Truncate(rng.range(1, width)), std::slice::from_ref(&x));
            let low = rng.range(0, width - 1);
            check(
                Extract(rng.range(low, width - 1), low),
                std::slice::from_ref(&x),
            );
            let z_width = rng.range(1, 70);
            let z = bvv(rng.bv(z_width));
            check(Concat, &[x.clone(), z.clone()]);
            check(Concat, &[z, x.clone()]);
            let cond_width = rng.pick(&[1, width]);
            let cond = bvv(rng.bv(cond_width));
            check(Ite, &[cond, x.clone(), y.clone()]);
        }
    }
}

/// The empty bitvector has one value, so every op on it is that value, and a
/// concatenation with it is the other operand.
#[test]
fn zero_width_ops_are_the_empty_value() {
    let empty = SymBV::Concrete { value: 0, width: 0 };
    assert_eq!(SymBV::symbol("zero_width_leaf", 0), empty);
    for op in BINARY.into_iter().chain(UNARY) {
        let args = [empty.clone(), empty.clone()];
        let got = op.sym(&args[..if UNARY.contains(&op) { 1 } else { 2 }]);
        assert_eq!(got, empty, "{op:?}");
    }
    for op in COMPARE {
        let got = op.sym(&[empty.clone(), empty.clone()]);
        assert_eq!(got.width(), 1, "{op:?}");
        assert!(got.is_concrete(), "{op:?}");
    }
    for x in [SymBV::concrete(0x5a, 8), SymBV::symbol("zero_width_x", 8)] {
        assert_eq!(x.concat(&empty), x);
        assert_eq!(empty.concat(&x), x);
    }
}
