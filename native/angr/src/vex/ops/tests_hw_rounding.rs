// angr-mx00p.101.7: FP arithmetic under every VEX rounding mode against x86
// hardware.
//
// VEX's rounding-mode encoding (0 nearest-even, 1 toward -inf, 2 toward
// +inf, 3 toward zero) is MXCSR.RC's, so modes 0-3 run on the SSE unit with
// RC set and compare bit-exact with `VEXOps::binop_with_rm` /
// `unop_with_rm`. Mode 4 (ties-away) has no x86 rounding control; its
// reference is Z3 over a raw clarirs term, spelled here with the factory
// constructors and evaluated through `clarirs_z3`, never through
// `clarirs_num::Float`'s rounding helpers (Rust's folds may share them). The
// same Z3 reference is checked against the hardware for modes 0-3, so the
// mode-4 oracle is itself seen to agree with silicon where silicon exists.
//
// A Rust result that is not already a literal is read back two ways: through
// clarirs's simplify fold (what a served op's export folds to) and through Z3
// on the unsimplified term. Both must match.
//
// NaN results compare as "both NaN": x86 returns its negative default NaN or
// the first quieted input, Rust and Z3 a canonical one. When both sides
// return an input NaN the bits must match.

use std::arch::x86_64::{
    __m128, __m128d, _mm_add_pd, _mm_add_ps, _mm_add_sd, _mm_add_ss, _mm_div_pd, _mm_div_ps,
    _mm_div_sd, _mm_div_ss, _mm_mul_pd, _mm_mul_ps, _mm_mul_sd, _mm_mul_ss, _mm_sqrt_pd,
    _mm_sqrt_ps, _mm_sqrt_sd, _mm_sqrt_ss, _mm_sub_pd, _mm_sub_ps, _mm_sub_sd, _mm_sub_ss,
};
use std::hint::black_box;

use clarirs_core::GLOBAL_CONTEXT;
use clarirs_core::prelude::*;
use clarirs_z3::Z3Solver;
use num_bigint::BigUint;

use super::*;
use crate::vex::ir::IRType;

type Term = AstRef<'static>;

/// MXCSR's rounding-control field (bits 13-14).
const MXCSR_RC: u32 = 0b11 << 13;

/// Sets MXCSR.RC to a VEX mode 0-3 and restores the saved MXCSR on drop, so
/// a failing assertion cannot leave the test thread rounding wrongly.
struct RoundingGuard(u32);

impl RoundingGuard {
    #[allow(
        deprecated,
        reason = "_mm_getcsr/_mm_setcsr are deprecated in favour of inline asm, but they are the documented way to drive MXCSR from Rust and this is test-only code"
    )]
    fn set(mode: u32) -> Self {
        assert!(mode < 4, "x86 has no rounding control for VEX mode {mode}");
        // SAFETY: SSE is baseline on x86_64; writing RC only changes how this
        // thread rounds, and `Drop` restores the saved register.
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !MXCSR_RC) | (mode << 13));
            Self(saved)
        }
    }
}

impl Drop for RoundingGuard {
    #[allow(deprecated, reason = "see RoundingGuard::set")]
    fn drop(&mut self) {
        // SAFETY: restores the value read in `set`.
        unsafe { std::arch::x86_64::_mm_setcsr(self.0) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
}

const OPS: [Op; 5] = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Sqrt];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Prec {
    F32,
    F64,
}

impl Prec {
    fn bits(self) -> u32 {
        match self {
            Prec::F32 => 32,
            Prec::F64 => 64,
        }
    }

    fn ty(self) -> IRType {
        match self {
            Prec::F32 => IRType::F32,
            Prec::F64 => IRType::F64,
        }
    }

    fn sort(self) -> FSort {
        match self {
            Prec::F32 => FSort::f32(),
            Prec::F64 => FSort::f64(),
        }
    }

    fn mask(self) -> u128 {
        (1u128 << self.bits()) - 1
    }

    fn is_nan(self, bits: u128) -> bool {
        match self {
            Prec::F32 => f32::from_bits(bits as u32).is_nan(),
            Prec::F64 => f64::from_bits(bits as u64).is_nan(),
        }
    }

    /// The listed edges: +-0, +-inf, quiet NaN, the smallest subnormal, the
    /// largest finite, and the rounding probes 1.0 (with -1.0 and 3.0),
    /// 2^-60 / 2^-30 (far below 1.0's last place), 1.5 ulp(1.0) and half an
    /// ulp(1.0) (the ties that separate nearest-even from ties-away).
    fn edges(self) -> Vec<u128> {
        match self {
            Prec::F32 => [
                0.0f32,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                f32::from_bits(1),
                f32::MAX,
                1.0,
                -1.0,
                3.0,
                2f32.powi(-30),
                1.5 * f32::EPSILON,
                0.5 * f32::EPSILON,
            ]
            .iter()
            .map(|f| u128::from(f.to_bits()))
            .collect(),
            Prec::F64 => [
                0.0f64,
                -0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::from_bits(1),
                f64::MAX,
                1.0,
                -1.0,
                3.0,
                2f64.powi(-60),
                1.5 * f64::EPSILON,
                0.5 * f64::EPSILON,
            ]
            .iter()
            .map(|f| u128::from(f.to_bits()))
            .collect(),
        }
    }

    /// The lane count of a 128-bit vector.
    fn lanes(self) -> u8 {
        (128 / self.bits()) as u8
    }
}

/// A xorshift generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// An edge or a random bit pattern, half the time each.
    fn operand(&mut self, prec: Prec, edges: &[u128]) -> u128 {
        if self.next().is_multiple_of(2) {
            edges[self.next() as usize % edges.len()]
        } else {
            u128::from(self.next()) & prec.mask()
        }
    }
}

// ---- the hardware ----

fn m128d(bits: u128) -> __m128d {
    // SAFETY: both types are 16 plain bytes; lane 0 is the low half.
    unsafe { std::mem::transmute::<u128, __m128d>(black_box(bits)) }
}

fn m128(bits: u128) -> __m128 {
    // SAFETY: as `m128d`.
    unsafe { std::mem::transmute::<u128, __m128>(black_box(bits)) }
}

fn from_m128d(v: __m128d) -> u128 {
    // SAFETY: as `m128d`.
    black_box(unsafe { std::mem::transmute::<__m128d, u128>(v) })
}

fn from_m128(v: __m128) -> u128 {
    // SAFETY: as `m128d`.
    black_box(unsafe { std::mem::transmute::<__m128, u128>(v) })
}

/// `op` on the SSE unit under VEX mode `mode`, over whole 128-bit vectors
/// (`scalar` uses the `_sd`/`_ss` form, whose lane 0 is the result). The
/// operands and the result pass through `black_box`, which orders them
/// between the guard's two MXCSR writes and keeps LLVM from folding the
/// arithmetic under its default rounding.
fn hardware(op: Op, prec: Prec, scalar: bool, mode: u32, a: u128, b: u128) -> u128 {
    let _guard = RoundingGuard::set(mode);
    match prec {
        Prec::F64 => {
            let (x, y) = (m128d(a), m128d(b));
            // SAFETY: SSE2 is baseline on x86_64.
            from_m128d(unsafe { match (op, scalar) {
                (Op::Add, true) => _mm_add_sd(x, y),
                (Op::Sub, true) => _mm_sub_sd(x, y),
                (Op::Mul, true) => _mm_mul_sd(x, y),
                (Op::Div, true) => _mm_div_sd(x, y),
                (Op::Sqrt, true) => _mm_sqrt_sd(x, x),
                (Op::Add, false) => _mm_add_pd(x, y),
                (Op::Sub, false) => _mm_sub_pd(x, y),
                (Op::Mul, false) => _mm_mul_pd(x, y),
                (Op::Div, false) => _mm_div_pd(x, y),
                (Op::Sqrt, false) => _mm_sqrt_pd(x),
            } })
        }
        Prec::F32 => {
            let (x, y) = (m128(a), m128(b));
            // SAFETY: SSE is baseline on x86_64.
            from_m128(unsafe { match (op, scalar) {
                (Op::Add, true) => _mm_add_ss(x, y),
                (Op::Sub, true) => _mm_sub_ss(x, y),
                (Op::Mul, true) => _mm_mul_ss(x, y),
                (Op::Div, true) => _mm_div_ss(x, y),
                (Op::Sqrt, true) => _mm_sqrt_ss(x),
                (Op::Add, false) => _mm_add_ps(x, y),
                (Op::Sub, false) => _mm_sub_ps(x, y),
                (Op::Mul, false) => _mm_mul_ps(x, y),
                (Op::Div, false) => _mm_div_ps(x, y),
                (Op::Sqrt, false) => _mm_sqrt_ps(x),
            } })
        }
    }
}

// ---- the Z3 reference ----

fn bvv(value: u128, width: u32) -> Term {
    GLOBAL_CONTEXT
        .bvv(BitVec::from((BigUint::from(value), width)))
        .unwrap()
}

fn literal(term: &Term) -> Option<u128> {
    match term.op() {
        AstOp::BVV(bv) => u128::try_from(bv.to_biguint()).ok(),
        _ => None,
    }
}

/// The `width`-bit value of `term` once Z3 binds each `(symbol, value)` pin.
/// The operands stay symbols because `Z3Solver::eval` runs clarirs's
/// simplify first, which would fold a variable-free term without Z3.
fn z3_value(term: &Term, width: u32, pins: &[(Term, u128)]) -> u128 {
    let ctx = &*GLOBAL_CONTEXT;
    let mut solver = Z3Solver::new(ctx);
    for (symbol, value) in pins {
        let eq = ctx.eq_(symbol.clone(), bvv(*value, symbol.size())).unwrap();
        solver.add(&eq).unwrap();
    }
    let result = ctx.bvs("hw_rounding_result", width).unwrap();
    solver.add(&ctx.eq_(result.clone(), term.clone()).unwrap()).unwrap();
    literal(&solver.eval(&result).unwrap()).expect("Z3 returned a non-literal")
}

const FPRMS: [FPRM; 5] = [
    FPRM::NearestTiesToEven,
    FPRM::TowardNegative,
    FPRM::TowardPositive,
    FPRM::TowardZero,
    FPRM::NearestTiesToAway,
];

/// One lane of `op` under VEX mode `mode`, as Z3 rounds it.
fn reference(op: Op, prec: Prec, mode: u32, a: u128, b: u128) -> u128 {
    let ctx = &*GLOBAL_CONTEXT;
    let w = prec.bits();
    let (sa, sb) = (ctx.bvs("hw_a", w).unwrap(), ctx.bvs("hw_b", w).unwrap());
    let fp = |v: &Term| ctx.bv_to_fp(v.clone(), prec.sort()).unwrap();
    let rm = FPRMS[mode as usize];
    let f = match op {
        Op::Add => ctx.fp_add(fp(&sa), fp(&sb), rm),
        Op::Sub => ctx.fp_sub(fp(&sa), fp(&sb), rm),
        Op::Mul => ctx.fp_mul(fp(&sa), fp(&sb), rm),
        Op::Div => ctx.fp_div(fp(&sa), fp(&sb), rm),
        Op::Sqrt => ctx.fp_sqrt(fp(&sa), rm),
    }
    .unwrap();
    z3_value(&ctx.fp_to_ieeebv(f).unwrap(), w, &[(sa, a), (sb, b)])
}

// ---- Rust ----

/// `op` through the rm-taking `VEXOps` entry the interpreter calls.
fn rust_op(op: Op, prec: Prec, count: Option<u8>, mode: u32, x: RustBV, y: RustBV) -> RustBV {
    let rm = RustBV::concrete(u128::from(mode), 32);
    let ty = prec.ty();
    match (op, count) {
        (Op::Add, None) => VEXOps::binop_with_rm(IROp::FAdd(ty), rm, x, y),
        (Op::Sub, None) => VEXOps::binop_with_rm(IROp::FSub(ty), rm, x, y),
        (Op::Mul, None) => VEXOps::binop_with_rm(IROp::FMul(ty), rm, x, y),
        (Op::Div, None) => VEXOps::binop_with_rm(IROp::FDiv(ty), rm, x, y),
        (Op::Sqrt, None) => VEXOps::unop_with_rm(IROp::FSqrt(ty), rm, x),
        (Op::Add, Some(count)) => {
            VEXOps::binop_with_rm(IROp::VFAdd { elem: ty, count }, rm, x, y)
        }
        (Op::Sub, Some(count)) => {
            VEXOps::binop_with_rm(IROp::VFSub { elem: ty, count }, rm, x, y)
        }
        (Op::Mul, Some(count)) => {
            VEXOps::binop_with_rm(IROp::VFMul { elem: ty, count }, rm, x, y)
        }
        (Op::Div, Some(count)) => {
            VEXOps::binop_with_rm(IROp::VFDiv { elem: ty, count }, rm, x, y)
        }
        (Op::Sqrt, Some(count)) => {
            VEXOps::unop_with_rm(IROp::VFSqrt { elem: ty, count }, rm, x)
        }
    }
    .unwrap()
}

/// `op` in Rust, read back both ways a served op can be: on literal
/// operands (the native fast path, or the simplify fold of the rm term), and
/// on symbols Z3 pins to the same bits (the term's Z3 lowering).
fn rust(op: Op, prec: Prec, count: Option<u8>, mode: u32, a: u128, b: u128) -> [(&'static str, u128); 2] {
    let width = count.map_or(prec.bits(), |_| 128);
    let concrete = rust_op(op, prec, count, mode, RustBV::concrete(a, width), RustBV::concrete(b, width));
    assert_eq!(concrete.width(), width, "{op:?} {prec:?} mode {mode}");
    let folded = match concrete.as_u128() {
        Some(v) => v,
        None => literal(&concrete.to_clarirs().unwrap().simplify().unwrap())
            .unwrap_or_else(|| panic!("{op:?} {prec:?} mode {mode}: the fold left a term")),
    };
    let (sa, sb) = (RustBV::symbol("hw_x", width), RustBV::symbol("hw_y", width));
    let pins = [
        (sa.to_clarirs().unwrap(), a),
        (sb.to_clarirs().unwrap(), b),
    ];
    let term = rust_op(op, prec, count, mode, sa, sb).to_clarirs().unwrap();
    [("fold", folded), ("z3", z3_value(&term, width, &pins))]
}

/// Whether lane result `got` agrees with oracle lane `want`, under the NaN
/// rule in the file header.
fn lane_agrees(prec: Prec, inputs: &[u128], want: u128, got: u128) -> bool {
    if !prec.is_nan(want) || !prec.is_nan(got) {
        return want == got;
    }
    !(inputs.contains(&want) && inputs.contains(&got)) || want == got
}

fn assert_lanes_agree(
    what: &str,
    prec: Prec,
    lanes: usize,
    inputs: (u128, u128),
    want: u128,
    got: u128,
) {
    let w = prec.bits();
    for i in 0..lanes {
        let lane = |v: u128| (v >> (i as u32 * w)) & prec.mask();
        let ins = [lane(inputs.0), lane(inputs.1)];
        assert!(
            lane_agrees(prec, &ins, lane(want), lane(got)),
            "{what} lane {i}: inputs {:#x} {:#x}, want {:#x}, got {:#x}",
            ins[0],
            ins[1],
            lane(want),
            lane(got)
        );
    }
}

/// Every scalar case: edge x edge pairs plus a seeded random batch.
fn scalar_cases(prec: Prec, seed: u64) -> Vec<(u128, u128)> {
    let edges = prec.edges();
    let mut cases: Vec<(u128, u128)> = edges
        .iter()
        .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
        .collect();
    let mut rng = Rng(seed);
    cases.extend((0..48).map(|_| (rng.operand(prec, &edges), rng.operand(prec, &edges))));
    cases
}

/// Every packed case: vectors whose lanes are drawn from the edges and the
/// random patterns.
fn packed_cases(prec: Prec, seed: u64) -> Vec<(u128, u128)> {
    let edges = prec.edges();
    let mut rng = Rng(seed);
    let mut vector = || {
        (0..u32::from(prec.lanes())).fold(0u128, |acc, i| {
            acc | (rng.operand(prec, &edges) << (i * prec.bits()))
        })
    };
    (0..48).map(|_| (vector(), vector())).collect()
}

/// The oracle for one case: the SSE unit for modes 0-3, which the Z3
/// reference must match lane by lane, and the Z3 reference alone for mode 4.
fn oracle(op: Op, prec: Prec, scalar: bool, mode: u32, a: u128, b: u128) -> u128 {
    let lanes = if scalar { 1 } else { usize::from(prec.lanes()) };
    let w = prec.bits();
    let lane = |v: u128, i: usize| (v >> (i as u32 * w)) & prec.mask();
    let z3 = (0..lanes).fold(0u128, |acc, i| {
        acc | (reference(op, prec, mode, lane(a, i), lane(b, i)) << (i as u32 * w))
    });
    if mode == 4 {
        return z3;
    }
    let hw = hardware(op, prec, scalar, mode, a, b);
    let hw = if scalar { hw & prec.mask() } else { hw };
    assert_lanes_agree(
        &format!("Z3 reference vs x86: {op:?} {prec:?} scalar={scalar} mode {mode}"),
        prec,
        lanes,
        (a, b),
        hw,
        z3,
    );
    hw
}

fn check(prec: Prec, packed: bool) {
    let cases = if packed {
        packed_cases(prec, 0x510e_527f_ade6_82d1)
    } else {
        scalar_cases(prec, 0x9b05_688c_2b3e_6c1f)
    };
    let lanes = if packed { usize::from(prec.lanes()) } else { 1 };
    let count = packed.then(|| prec.lanes());
    for op in OPS {
        for mode in 0..5u32 {
            for &(a, b) in &cases {
                let b = if op == Op::Sqrt { a } else { b };
                let want = oracle(op, prec, !packed, mode, a, b);
                for (how, got) in rust(op, prec, count, mode, a, b) {
                    assert_lanes_agree(
                        &format!("Rust ({how}) vs oracle: {op:?} {prec:?} packed={packed} mode {mode}"),
                        prec,
                        lanes,
                        (a, b),
                        want,
                        got,
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_f32_matches_hardware_in_every_rounding_mode() {
    check(Prec::F32, false);
}

#[test]
fn scalar_f64_matches_hardware_in_every_rounding_mode() {
    check(Prec::F64, false);
}

#[test]
fn packed_f32x4_matches_hardware_in_every_rounding_mode() {
    check(Prec::F32, true);
}

#[test]
fn packed_f64x2_matches_hardware_in_every_rounding_mode() {
    check(Prec::F64, true);
}

/// The guard really changes how the SSE unit rounds, and restores it: 1/3
/// rounds differently toward -inf and +inf, and the default mode is back
/// once the guard drops.
#[test]
fn rounding_guard_drives_mxcsr() {
    let one = u128::from(1.0f64.to_bits());
    let three = u128::from(3.0f64.to_bits());
    let down = hardware(Op::Div, Prec::F64, true, 1, one, three) & Prec::F64.mask();
    let up = hardware(Op::Div, Prec::F64, true, 2, one, three) & Prec::F64.mask();
    assert_eq!(up, down + 1, "1/3 toward +inf is one ulp above toward -inf");
    // 1/3 = 0x3fd5_5555_5555_5555.5..., so nearest-even rounds down.
    assert_eq!(down, 0x3fd5_5555_5555_5555);
    let third = black_box(1.0f64) / black_box(3.0f64);
    assert_eq!(u128::from(third.to_bits()), down, "MXCSR restored to nearest-even");
}
