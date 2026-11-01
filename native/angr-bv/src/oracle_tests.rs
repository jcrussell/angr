//! Oracles for the ops: each `SymBV` op against the clarirs term it stands
//! for, spelled with the raw constructors over `GLOBAL_CONTEXT` rather than
//! `build_term`, so the reference shares none of `SymBV`'s encodings. In the
//! reference, 1-bit logic is bitwise, a comparison is `ite(cmp, 1, 0)` and
//! `ite`'s condition is `c != 0`.

use clarirs_core::GLOBAL_CONTEXT;
use clarirs_core::prelude::*;
use clarirs_z3::Z3Solver;
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
    fn is_mul_div(self) -> bool {
        matches!(self, Mul | UDiv | SDiv | URem | SRem)
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

#[test]
fn fold_matches_clarirs_simplify() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for width in FOLD_WIDTHS {
        for i in 0..FOLD_ITERATIONS {
            let (a, b) = (rng.bv(width), rng.bv(width));
            let divisor_is_zero = b.is_zero();
            let (x, y) = (bvv(a), bvv(b));
            let check = |op: Op, args: &[Term]| assert_folds_like_clarirs(op, args);
            for op in BINARY {
                // A division by zero is left unfolded on purpose: claripy raises.
                if divisor_is_zero && matches!(op, UDiv | SDiv | URem | SRem) {
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

// ---------------------------------------------------------------------------
// Suite 2: random op trees over symbols, proved equal to their raw terms by Z3.
// ---------------------------------------------------------------------------

const TREE_WIDTHS: [u32; 6] = [1, 8, 13, 64, 65, 128];
const TREES_PER_WIDTH: usize = 200;
const MAX_DEPTH: u32 = 3;

/// An op tree, built as a `SymBV` and as its raw term from the same plan.
#[derive(Clone, Debug)]
enum Plan {
    /// Symbol `i` of this width.
    Leaf(u32, u32),
    Const(BitVec),
    Op(Op, Vec<Plan>),
}

impl Plan {
    fn name(i: u32, width: u32) -> String {
        format!("x{i}_{width}")
    }

    /// The tree as a `SymBV`, with `leaf` for each symbol.
    fn sym(&self, leaf: &dyn Fn(u32, u32) -> SymBV) -> SymBV {
        match self {
            Plan::Leaf(i, w) => leaf(*i, *w),
            Plan::Const(v) => value(&bvv(v.clone())),
            Plan::Op(op, kids) => {
                let args: Vec<SymBV> = kids.iter().map(|k| k.sym(leaf)).collect();
                op.sym(&args)
            }
        }
    }

    fn raw(&self) -> Term {
        match self {
            Plan::Leaf(i, w) => GLOBAL_CONTEXT.bvs(Self::name(*i, *w), *w).unwrap(),
            Plan::Const(v) => bvv(v.clone()),
            Plan::Op(op, kids) => {
                let args: Vec<Term> = kids.iter().map(Plan::raw).collect();
                op.raw(&args)
            }
        }
    }

    fn leaves(&self, out: &mut Vec<(u32, u32)>) {
        match self {
            Plan::Leaf(i, w) if !out.contains(&(*i, *w)) => out.push((*i, *w)),
            Plan::Leaf(..) | Plan::Const(_) => {}
            Plan::Op(_, kids) => kids.iter().for_each(|k| k.leaves(out)),
        }
    }
}

struct Gen(Rng);

impl Gen {
    fn leaf(&mut self, width: u32) -> Plan {
        if self.0.next().is_multiple_of(4) {
            Plan::Const(self.0.bv(width))
        } else {
            Plan::Leaf(self.0.range(0, 2), width)
        }
    }

    /// A tree of `width` bits and at most `depth` levels of ops.
    fn tree(&mut self, width: u32, depth: u32) -> Plan {
        if depth == 0 || (depth < MAX_DEPTH && self.0.next().is_multiple_of(5)) {
            return self.leaf(width);
        }
        let d = depth - 1;
        let rng = &mut self.0;
        let src_wider = |rng: &mut Rng| width + rng.range(1, 16);
        match rng.next() % 12 {
            // Wide multiply/divide is left to the fold suite: Z3 bit-blasts it slowly.
            0..=2 => {
                let op = rng.pick(&BINARY);
                if op.is_mul_div() && width > 64 {
                    return self.tree(width, depth);
                }
                // Z3 merges nested rotates by adding their amounts mod 2^w, not
                // mod w, so off powers of two a rotate takes a constant amount.
                if matches!(op, RotL | RotR) && !width.is_power_of_two() {
                    let amount = Plan::Const(self.0.bv(width));
                    return Plan::Op(op, vec![self.tree(width, d), amount]);
                }
                Plan::Op(op, vec![self.tree(width, d), self.tree(width, d)])
            }
            // Wide bit counts are left to the fold suite: Z3 proves a sum of
            // bits against an ite ladder slowly.
            3 if width > 16 => Plan::Op(rng.pick(&[Neg, Not, Reverse]), vec![self.tree(width, d)]),
            3 => Plan::Op(rng.pick(&UNARY), vec![self.tree(width, d)]),
            4 if width == 1 => {
                let op = rng.pick(&COMPARE);
                let w = rng.pick(&TREE_WIDTHS);
                Plan::Op(op, vec![self.tree(w, d), self.tree(w, d)])
            }
            // The zext-compare fold: a widened value against a constant.
            5 if width == 1 => {
                let op = rng.pick(&COMPARE);
                let (narrow, wide) = (rng.range(1, 32), rng.pick(&[33, 64, 65]));
                let zext = Plan::Op(ZeroExt(wide), vec![self.tree(narrow, d)]);
                let constant = Plan::Const(self.0.bv(wide));
                let kids = if self.0.next().is_multiple_of(2) {
                    vec![zext, constant]
                } else {
                    vec![constant, zext]
                };
                Plan::Op(op, kids)
            }
            4 | 5 => self.tree(width, depth),
            6 if width > 1 => {
                let src = rng.range(1, width - 1);
                let op = if rng.next().is_multiple_of(2) {
                    ZeroExt(width)
                } else {
                    SignExt(width)
                };
                Plan::Op(op, vec![self.tree(src, d)])
            }
            6 => {
                let src = src_wider(rng);
                Plan::Op(Truncate(width), vec![self.tree(src, d)])
            }
            // Extract, often of a concat, an extract or a reverse.
            7 | 8 => {
                let src = src_wider(rng);
                let low = rng.range(0, src - width);
                let inner = match rng.next() % 4 {
                    0 if d > 0 => {
                        let high_w = rng.range(1, src - 1);
                        Plan::Op(
                            Concat,
                            vec![self.tree(high_w, d - 1), self.tree(src - high_w, d - 1)],
                        )
                    }
                    1 if d > 0 => {
                        let outer = src + self.0.range(1, 8);
                        let l = self.0.range(0, outer - src);
                        Plan::Op(Extract(l + src - 1, l), vec![self.tree(outer, d - 1)])
                    }
                    2 if d > 0 => {
                        let bytes = src.div_ceil(8).max(2) * 8;
                        let inner = Plan::Op(Reverse, vec![self.tree(bytes, d - 1)]);
                        Plan::Op(Truncate(src), vec![inner])
                    }
                    _ => self.tree(src, d),
                };
                Plan::Op(Extract(low + width - 1, low), vec![inner])
            }
            9 if width > 1 => {
                let high_w = rng.range(1, width - 1);
                Plan::Op(
                    Concat,
                    vec![self.tree(high_w, d), self.tree(width - high_w, d)],
                )
            }
            9 => self.tree(width, depth),
            10 => Plan::Op(
                Ite,
                vec![self.tree(1, d), self.tree(width, d), self.tree(width, d)],
            ),
            // An identity with 0 or all-ones, or a commutative op built both ways.
            _ => {
                let op = rng.pick(&[Add, Mul, And, Or, Xor, Sub, Shl, Lshr, Ashr]);
                if op.is_mul_div() && width > 64 {
                    return self.tree(width, depth);
                }
                let x = self.tree(width, d);
                if self.0.next().is_multiple_of(2) {
                    let edge = if self.0.next().is_multiple_of(2) {
                        BitVec::zeros(width)
                    } else {
                        BitVec::ones(width)
                    };
                    let kids = if self.0.next().is_multiple_of(2) {
                        vec![x, Plan::Const(edge)]
                    } else {
                        vec![Plan::Const(edge), x]
                    };
                    return Plan::Op(op, kids);
                }
                let op = self.0.pick(&[Add, Mul, And, Or, Xor]);
                if op == Mul && width > 64 {
                    return self.tree(width, depth);
                }
                let y = self.tree(width, d);
                let both = |a: &Plan, b: &Plan| Plan::Op(op, vec![a.clone(), b.clone()]);
                Plan::Op(Xor, vec![both(&x, &y), both(&y, &x)])
            }
        }
    }
}

/// Asserts that no assignment satisfies `constraints`. Unknown fails.
fn assert_unsat(constraints: &[Term], what: &str) {
    let mut solver = Z3Solver::new(&GLOBAL_CONTEXT);
    for c in constraints {
        solver.add(c).unwrap();
    }
    match solver.satisfiable() {
        Ok(false) => {}
        Ok(true) => panic!("{what}: satisfiable"),
        Err(e) => panic!("{what}: {e}"),
    }
}

#[test]
fn op_trees_match_their_raw_terms() {
    let ctx = &*GLOBAL_CONTEXT;
    let mut generator = Gen(Rng(0x2545_f491_4f6c_dd1d));
    for width in TREE_WIDTHS {
        for _ in 0..TREES_PER_WIDTH {
            let plan = generator.tree(width, MAX_DEPTH);
            let raw = plan.raw();
            let sym = plan.sym(&|i, w| SymBV::symbol(Plan::name(i, w), w));
            let term = sym.to_clarirs().unwrap();
            assert_unsat(
                &[ctx.neq(term.clone(), raw.clone()).unwrap()],
                &format!("(a) {plan:?}"),
            );

            let simplified = term.simplify().unwrap();
            assert_unsat(
                &[ctx.neq(simplified, raw.clone()).unwrap()],
                &format!("(b) {plan:?}"),
            );

            let mut leaves = Vec::new();
            plan.leaves(&mut leaves);
            let sigma: Vec<((u32, u32), BitVec)> = leaves
                .iter()
                .map(|&(i, w)| ((i, w), generator.0.bv(w)))
                .collect();
            let folded = plan.sym(&|i, w| {
                let (_, v) = sigma.iter().find(|(key, _)| *key == (i, w)).unwrap();
                value(&bvv(v.clone()))
            });
            let mut constraints = vec![ctx.neq(raw, folded.to_clarirs().unwrap()).unwrap()];
            for ((i, w), v) in &sigma {
                let leaf = ctx.bvs(Plan::name(*i, *w), *w).unwrap();
                constraints.push(ctx.eq_(leaf, bvv(v.clone())).unwrap());
            }
            assert_unsat(&constraints, &format!("(c) {plan:?} at {sigma:?}"));
        }
    }
}

// ---------------------------------------------------------------------------
// Float ops. `SymBV` does not fold them (`float_term` only builds the term), so
// the fold suite has none.
// ---------------------------------------------------------------------------

mod float {
    use super::*;
    use crate::{FloatOpKind as K, FloatPrec};

    fn sort(prec: FloatPrec) -> FSort {
        match prec {
            FloatPrec::F32 => FSort::f32(),
            FloatPrec::F64 => FSort::f64(),
        }
    }

    fn other(prec: FloatPrec) -> FloatPrec {
        match prec {
            FloatPrec::F32 => FloatPrec::F64,
            FloatPrec::F64 => FloatPrec::F32,
        }
    }

    pub(super) fn kinds(prec: FloatPrec) -> Vec<K> {
        let mut kinds = vec![
            K::Add,
            K::Sub,
            K::Mul,
            K::Div,
            K::MaxNum,
            K::MinNum,
            K::Sqrt,
            K::Neg,
            K::Abs,
            K::Fma,
            K::Fms,
            K::CmpEq,
            K::CmpLt,
            K::CmpLe,
            K::IsNaN,
            K::RoundToInt,
            K::AddRm,
            K::SubRm,
            K::MulRm,
            K::DivRm,
            K::SqrtRm,
            K::FmaRm,
            K::FmsRm,
            K::ConvertFtoF {
                src_prec: other(prec),
            },
            K::ConvertFtoFRm {
                src_prec: other(prec),
            },
        ];
        for bits in [32, 64] {
            for signed in [false, true] {
                kinds.push(K::ConvertItoF {
                    src_bits: bits,
                    signed,
                });
                kinds.push(K::ConvertItoFRm {
                    src_bits: bits,
                    signed,
                });
                kinds.push(K::ConvertFtoI {
                    dst_bits: bits,
                    signed,
                });
                kinds.push(K::ConvertFtoIRm {
                    dst_bits: bits,
                    signed,
                });
            }
        }
        kinds
    }

    pub(super) fn takes_rm(kind: K) -> bool {
        matches!(
            kind,
            K::RoundToInt
                | K::AddRm
                | K::SubRm
                | K::MulRm
                | K::DivRm
                | K::SqrtRm
                | K::FmaRm
                | K::FmsRm
                | K::ConvertItoFRm { .. }
                | K::ConvertFtoIRm { .. }
                | K::ConvertFtoFRm { .. }
        )
    }

    /// The width of each operand, the rounding mode first where there is one.
    pub(super) fn operand_widths(kind: K, prec: FloatPrec) -> Vec<u32> {
        let value = match kind {
            K::ConvertItoF { src_bits, .. } | K::ConvertItoFRm { src_bits, .. } => {
                u32::from(src_bits)
            }
            K::ConvertFtoF { src_prec } | K::ConvertFtoFRm { src_prec } => src_prec.bits(),
            _ => prec.bits(),
        };
        let values = kind.arity() - usize::from(takes_rm(kind));
        let mut widths = vec![value; values];
        if takes_rm(kind) {
            widths.insert(0, 32);
        }
        widths
    }

    /// Whether the result is a float's IEEE bits, which SMT-LIB leaves
    /// unspecified for a NaN.
    pub(super) fn returns_float(kind: K) -> bool {
        !kind.is_compare() && !matches!(kind, K::ConvertFtoI { .. } | K::ConvertFtoIRm { .. })
    }

    const MODES: [FPRM; 5] = [
        FPRM::NearestTiesToEven,
        FPRM::TowardNegative,
        FPRM::TowardPositive,
        FPRM::TowardZero,
        FPRM::NearestTiesToAway,
    ];

    /// `build` under the VEX rounding mode `rm`: one term for a literal, and
    /// for a symbol a ladder testing the modes from the last, with the last
    /// mode also standing for every value outside them.
    fn under_rm(rm: &Term, build: impl Fn(FPRM) -> Term) -> Term {
        let ctx = &*GLOBAL_CONTEXT;
        if let AstOp::BVV(mode) = rm.op() {
            return build(MODES[mode.to_u64().unwrap() as usize]);
        }
        let mut term = build(MODES[0]);
        for (i, &mode) in MODES.iter().enumerate().skip(1).take(3) {
            let is_mode = ctx.eq_(rm.clone(), bvv_u(i as u64, 32)).unwrap();
            term = ctx.ite(is_mode, build(mode), term).unwrap();
        }
        let in_range = ctx.ult(rm.clone(), bvv_u(4, 32)).unwrap();
        ctx.ite(in_range, term, build(MODES[4])).unwrap()
    }

    /// The reference term for `kind` over IEEE bit patterns.
    pub(super) fn raw(kind: K, prec: FloatPrec, args: &[Term]) -> Term {
        let ctx = &*GLOBAL_CONTEXT;
        let s = sort(prec);
        let rne = FPRM::NearestTiesToEven;
        let fp = |t: &Term, s: FSort| ctx.bv_to_fp(t.clone(), s).unwrap();
        let bit = |b: Term| ctx.ite(b, bvv_u(1, 1), bvv_u(0, 1)).unwrap();
        let bits = |f: Term| ctx.fp_to_ieeebv(f).unwrap();
        let to_int = |f: Term, dst: u8, signed: bool, rm: FPRM| {
            if signed {
                ctx.fp_to_sbv(f, u32::from(dst), rm).unwrap()
            } else {
                ctx.fp_to_ubv(f, u32::from(dst), rm).unwrap()
            }
        };
        let from_int = |i: &Term, signed: bool, rm: FPRM| {
            if signed {
                ctx.bv_to_fp_signed(i.clone(), s, rm).unwrap()
            } else {
                ctx.bv_to_fp_unsigned(i.clone(), s, rm).unwrap()
            }
        };
        let a = |i: usize| fp(&args[i], s);
        let rm = &args[0];
        match kind {
            K::Add => bits(ctx.fp_add(a(0), a(1), rne).unwrap()),
            K::Sub => bits(ctx.fp_sub(a(0), a(1), rne).unwrap()),
            K::Mul => bits(ctx.fp_mul(a(0), a(1), rne).unwrap()),
            K::Div => bits(ctx.fp_div(a(0), a(1), rne).unwrap()),
            K::MaxNum => bits(ctx.fp_max(a(0), a(1)).unwrap()),
            K::MinNum => bits(ctx.fp_min(a(0), a(1)).unwrap()),
            K::Sqrt => bits(ctx.fp_sqrt(a(0), rne).unwrap()),
            K::Neg => bits(ctx.fp_neg(a(0)).unwrap()),
            K::Abs => bits(ctx.fp_abs(a(0)).unwrap()),
            K::Fma => bits(ctx.fp_fma(a(0), a(1), a(2), rne).unwrap()),
            K::Fms => bits(
                ctx.fp_fma(a(0), a(1), ctx.fp_neg(a(2)).unwrap(), rne)
                    .unwrap(),
            ),
            K::CmpEq => bit(ctx.fp_eq(a(0), a(1)).unwrap()),
            K::CmpLt => bit(ctx.fp_lt(a(0), a(1)).unwrap()),
            K::CmpLe => bit(ctx.fp_leq(a(0), a(1)).unwrap()),
            K::IsNaN => bit(ctx.fp_is_nan(a(0)).unwrap()),
            K::RoundToInt => under_rm(rm, |m| bits(ctx.fp_round_to_integral(a(1), m).unwrap())),
            K::AddRm => under_rm(rm, |m| bits(ctx.fp_add(a(1), a(2), m).unwrap())),
            K::SubRm => under_rm(rm, |m| bits(ctx.fp_sub(a(1), a(2), m).unwrap())),
            K::MulRm => under_rm(rm, |m| bits(ctx.fp_mul(a(1), a(2), m).unwrap())),
            K::DivRm => under_rm(rm, |m| bits(ctx.fp_div(a(1), a(2), m).unwrap())),
            K::SqrtRm => under_rm(rm, |m| bits(ctx.fp_sqrt(a(1), m).unwrap())),
            K::FmaRm => under_rm(rm, |m| bits(ctx.fp_fma(a(1), a(2), a(3), m).unwrap())),
            K::FmsRm => under_rm(rm, |m| {
                bits(
                    ctx.fp_fma(a(1), a(2), ctx.fp_neg(a(3)).unwrap(), m)
                        .unwrap(),
                )
            }),
            K::ConvertFtoF { src_prec } => {
                bits(ctx.fp_to_fp(fp(&args[0], sort(src_prec)), s, rne).unwrap())
            }
            K::ConvertFtoFRm { src_prec } => under_rm(rm, |m| {
                bits(ctx.fp_to_fp(fp(&args[1], sort(src_prec)), s, m).unwrap())
            }),
            K::ConvertItoF { signed, .. } => bits(from_int(&args[0], signed, rne)),
            K::ConvertItoFRm { signed, .. } => {
                under_rm(rm, |m| bits(from_int(&args[1], signed, m)))
            }
            K::ConvertFtoI { dst_bits, signed } => to_int(a(0), dst_bits, signed, rne),
            K::ConvertFtoIRm { dst_bits, signed } => {
                under_rm(rm, |m| to_int(a(1), dst_bits, signed, m))
            }
        }
    }

    /// Special and ordinary values of `prec`, as IEEE bits.
    pub(super) fn specials(prec: FloatPrec) -> Vec<u64> {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.5,
            0.1,
            2.5,
            1e6,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut out: Vec<u64> = match prec {
            FloatPrec::F32 => values
                .iter()
                .map(|&v| u64::from((v as f32).to_bits()))
                .collect(),
            FloatPrec::F64 => values.iter().map(|v| v.to_bits()).collect(),
        };
        out.push(match prec {
            FloatPrec::F32 => u64::from(f32::NAN.to_bits()),
            FloatPrec::F64 => f64::NAN.to_bits(),
        });
        out.push(1); // the smallest subnormal
        out
    }

    /// An in-range operand of a float-to-integer conversion: out of range
    /// the result is unspecified.
    pub(super) fn in_range(kind: K, prec: FloatPrec, bits: u64) -> bool {
        let value = match prec {
            FloatPrec::F32 => f64::from(f32::from_bits(bits as u32)),
            FloatPrec::F64 => f64::from_bits(bits),
        };
        match kind {
            K::ConvertFtoI { signed, .. } | K::ConvertFtoIRm { signed, .. } => {
                value.is_finite() && value.abs() < 1e9 && (signed || value > -1.0)
            }
            _ => true,
        }
    }
}

/// Each float op against an independently spelled raw term: (a) over
/// symbols, and (b) after clarirs's simplify, over constants.
#[test]
fn float_ops_match_their_raw_terms() {
    use crate::FloatPrec;
    let ctx = &*GLOBAL_CONTEXT;
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    for prec in [FloatPrec::F32, FloatPrec::F64] {
        for kind in float::kinds(prec) {
            let widths = float::operand_widths(kind, prec);
            let not_nan = |result: &Term| {
                let sort = match prec {
                    FloatPrec::F32 => FSort::f32(),
                    FloatPrec::F64 => FSort::f64(),
                };
                let f = ctx.bv_to_fp(result.clone(), sort).unwrap();
                ctx.not(ctx.fp_is_nan(f).unwrap()).unwrap()
            };
            let check = |sym: Term, raw: Term, what: &str| {
                let mut constraints = vec![ctx.neq(sym, raw.clone()).unwrap()];
                if float::returns_float(kind) {
                    constraints.push(not_nan(&raw));
                }
                assert_unsat(&constraints, what);
            };

            // The symbolic-mode ladder is checked in f32: Z3 bit-blasts an f64
            // divide under every mode slowly.
            let constant_rm = float::takes_rm(kind) && prec == FloatPrec::F64;
            let symbols: Vec<SymBV> = (0..widths.len())
                .map(|i| match i {
                    0 if constant_rm => SymBV::concrete(u128::from(rng.range(0, 4)), 32),
                    _ => SymBV::symbol(format!("f{i}_{}", widths[i]), widths[i]),
                })
                .collect();
            let terms: Vec<Term> = symbols.iter().map(|s| s.to_clarirs().unwrap()).collect();
            let sym = SymBV::float_node(kind, prec, symbols).to_clarirs().unwrap();
            check(
                sym,
                float::raw(kind, prec, &terms),
                &format!("(a) {kind:?} {prec:?}"),
            );

            for _ in 0..12 {
                let mut values: Vec<u64> = Vec::new();
                for (i, &w) in widths.iter().enumerate() {
                    values.push(if i == 0 && float::takes_rm(kind) {
                        u64::from(rng.range(0, 4))
                    } else if matches!(
                        kind,
                        crate::FloatOpKind::ConvertItoF { .. }
                            | crate::FloatOpKind::ConvertItoFRm { .. }
                    ) {
                        rng.bv(w).to_u64().unwrap()
                    } else {
                        let src = match kind {
                            crate::FloatOpKind::ConvertFtoF { src_prec }
                            | crate::FloatOpKind::ConvertFtoFRm { src_prec } => src_prec,
                            _ => prec,
                        };
                        rng.pick(&float::specials(src))
                    });
                }
                let floats = &values[usize::from(float::takes_rm(kind))..];
                if !floats.iter().all(|&v| float::in_range(kind, prec, v)) {
                    continue;
                }
                let terms: Vec<Term> = values
                    .iter()
                    .zip(&widths)
                    .map(|(&v, &w)| bvv_u(v, w))
                    .collect();
                let operands: Vec<SymBV> = terms.iter().map(value).collect();
                let folded = SymBV::float_node(kind, prec, operands)
                    .to_clarirs()
                    .unwrap()
                    .simplify()
                    .unwrap();
                check(
                    folded,
                    float::raw(kind, prec, &terms),
                    &format!("(b) {kind:?} {prec:?} over {values:x?}"),
                );
            }
        }
    }
}
