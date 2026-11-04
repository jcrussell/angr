//! The `*_rm` op registry the rounding-mode sweep drives: every rm-taking
//! `VEXOps` dispatch, its mode-sensitive operand cases, and the VEX shape it
//! arrives as. Its consumers are the Z3 sweep in `ops/tests_rounding_mode_sweep.rs`
//! (whose module comment explains the sections and the exclusions) and the
//! engine's dispatcher-selection half, `interpreter::rm_dispatch_tests`, which
//! reaches it through the `testing` feature (angr-mx00p.51.10).
//!
//! `all_rm_ops()` mirrors, not reinvents, the real dispatch surface: every
//! entry invokes the same public `VEXOps` methods the interpreter calls.

use super::*;
use crate::ir::IRType;
use angr_bv::VexRm;

/// Every VEX rounding mode `VexRm::decode` accepts, by encoding: 0=nearest
/// (RNE), 1=down/-inf (RD), 2=up/+inf (RU), 3=zero/truncate (RZ),
/// 4=nearest, ties away from zero (RNA; angr-vmi2l.56).
pub const VEX_ROUNDING_MODES: [u32; 5] = [
    VexRm::NearestEven.encoding(),
    VexRm::TowardNegative.encoding(),
    VexRm::TowardPositive.encoding(),
    VexRm::TowardZero.encoding(),
    VexRm::NearestAway.encoding(),
];

/// One non-rm operand of a candidate test case: its raw bit pattern plus its
/// width, so the sweep can build the right-width `RustBV` for it without the
/// entry needing a separate parallel width list.
pub type Operand = (u128, u32);

/// Which real VEX-encoded shape an entry's op arrives as, and — for Family
/// B — which interpreter-level `eval_*` dispatcher (`interpreter/
/// expressions_arith.rs`) is responsible for picking its `VEXOps` method. Drives
/// sections 2 and 3 (see module doc comment): both are Family-B only,
/// because Family A is reached via plain `VEXOps::binop` with no separate
/// rm-less sibling to mis-route to, and no rm-concreteness branch to
/// exercise symbolically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RmDispatchKind {
    /// `VEXOps::binop`'s rm-taking `binop_float` arms (F64toF32, the
    /// float<->int conversions, RoundF{32,64}toInt). No interpreter-level
    /// dispatcher-selection risk: `eval_binop` unconditionally calls
    /// `VEXOps::binop`, and there is no `_with_rm` sibling it could
    /// mis-route to instead.
    FamilyA,
    /// VEX `Binop(rm, value)` -> `eval_binop` -> `VEXOps::binop` ->
    /// `binop_misc` -> `VEXOps::unop_with_rm` (FSqrt only — VEX encodes
    /// `Iop_SqrtF{32,64}` as a Binop, not a Unop).
    BinopToUnopWithRm,
    /// VEX `Triop(rm, a, b)` -> `eval_triop` -> `VEXOps::binop_with_rm`
    /// (FAdd/FSub/FMul/FDiv).
    TriopWithRm,
    /// VEX `Qop(rm, a, b, c)` -> `eval_qop` -> `VEXOps::qop_with_rm`
    /// (FMAdd/FMSub) — the exact caller-dispatch shape (angr-03vl4.30).
    QopWithRm,
}

/// Signature every dispatch entry's `call` closure shares, regardless of the
/// underlying `VEXOps` method's arity (`binop`/`binop_with_rm`/
/// `unop_with_rm`/`qop_with_rm` all take a different number of non-rm
/// operands, flattened here into a slice).
pub type RmDispatchFn = dyn Fn(RustBV, &[RustBV]) -> Result<RustBV, OpError>;

/// One `*_rm` op's dispatch entry. `call` invokes the *public* `VEXOps`
/// method the real interpreter uses (never a private/`pub(super)` helper),
/// exactly mirroring the call pattern each existing per-op test already uses.
pub struct RmOpEntry {
    pub name: &'static str,
    /// A handful of operand tuples known (and, independently, asserted by
    /// this sweep itself — see the mode-sensitivity check below) to round
    /// differently under different VEX modes, so the equivalence check below
    /// is non-vacuous.
    pub cases: Vec<Vec<Operand>>,
    /// Operand tuples every mode rounds identically (the sign of zero:
    /// `RoundF{32,64}toInt(-0.0)` is `-0.0` in every mode). Section 1 checks
    /// them for concrete/symbolic agreement but exempts them from the
    /// mode-sensitivity check, which they would fail by construction.
    pub mode_invariant_cases: Vec<Vec<Operand>>,
    pub call: Box<RmDispatchFn>,
    /// The `IROp` this entry dispatches, for sections 2/3's interpreter- and
    /// rm-less-sibling calls, which need the concrete `IROp` value rather
    /// than a pre-baked closure.
    pub op: IROp,
    /// Which dispatch shape `op` arrives as — see `RmDispatchKind`.
    pub kind: RmDispatchKind,
}

fn f32_bits(v: f32) -> u128 {
    v.to_bits() as u128
}
/// `lane` repeated `count` times, lane 0 lowest.
fn splat(lane: u128, width: u32, count: u32) -> u128 {
    (0..count).fold(0, |acc, i| acc | (lane << (i * width)))
}

fn f64_bits(v: f64) -> u128 {
    v.to_bits() as u128
}

/// The exact midpoint of 1.0f32 and its successor, as an f64: RNE and RD/RZ
/// give 1.0 (even mantissa), RU and RNA the successor.
fn f64_to_f32_tie(sign: f64) -> f64 {
    let lo = 1.0f32;
    sign * (f64::from(lo) + f64::from(lo.next_up())) / 2.0
}

/// F64→F32 boundary value straddling two adjacent f32s at the 0.75-fraction
/// point, adapted from tests_conversions.rs's
/// `test_f64_to_f32_rm_straddles_boundary`: RNE/RU round up to `hi`, RD/RZ
/// round down to `lo`.
fn f64_to_f32_straddle(sign: f64) -> f64 {
    let lo = 1.0f32;
    let hi = lo.next_up();
    let v = f64::from(lo) + 0.75 * (f64::from(hi) - f64::from(lo));
    sign * v
}

/// Registers one `binop_float`-family op (Family A: F64toF32 and the
/// float<->int conversions with an explicit rm, plus the RoundF{32,64}toInt
/// ops) — all reached via the ordinary `VEXOps::binop(op, rm, value)`
/// dispatch, per `ops/mod.rs::binop_float`.
fn family_a(name: &'static str, op: IROp, src_bits: u32, values: Vec<u128>) -> RmOpEntry {
    RmOpEntry {
        name,
        cases: values.into_iter().map(|v| vec![(v, src_bits)]).collect(),
        mode_invariant_cases: Vec::new(),
        call: Box::new(move |rm, operands| VEXOps::binop(op, rm, operands[0].clone())),
        op,
        kind: RmDispatchKind::FamilyA,
    }
}

/// Registers a `binop_with_rm` (Triop-shaped: rm + 2 operands) entry —
/// `FAdd`/`FSub`/`FMul`/`FDiv`.
fn binop_with_rm_entry(name: &'static str, op: IROp, case: Vec<Operand>) -> RmOpEntry {
    RmOpEntry {
        name,
        cases: vec![case],
        mode_invariant_cases: Vec::new(),
        call: Box::new(move |rm, operands| {
            VEXOps::binop_with_rm(op, rm, operands[0].clone(), operands[1].clone())
        }),
        op,
        kind: RmDispatchKind::TriopWithRm,
    }
}

/// Registers an `unop_with_rm` (rm + 1 operand) entry — `FSqrt`.
fn unop_with_rm_entry(name: &'static str, op: IROp, case: Vec<Operand>) -> RmOpEntry {
    RmOpEntry {
        name,
        cases: vec![case],
        mode_invariant_cases: Vec::new(),
        call: Box::new(move |rm, operands| VEXOps::unop_with_rm(op, rm, operands[0].clone())),
        op,
        kind: RmDispatchKind::BinopToUnopWithRm,
    }
}

/// Registers a `qop_with_rm` (rm + 3 operands) entry — `FMAdd`/`FMSub`.
fn qop_with_rm_entry(name: &'static str, op: IROp, case: Vec<Operand>) -> RmOpEntry {
    RmOpEntry {
        name,
        cases: vec![case],
        mode_invariant_cases: Vec::new(),
        call: Box::new(move |rm, operands| {
            VEXOps::qop_with_rm(
                op,
                rm,
                operands[0].clone(),
                operands[1].clone(),
                operands[2].clone(),
            )
        }),
        op,
        kind: RmDispatchKind::QopWithRm,
    }
}

/// The full registry of `*_rm`-flavored ops, read off the real dispatch
/// match arms (see the module doc comment for exactly which ones).
pub fn all_rm_ops() -> Vec<RmOpEntry> {
    // 1-ULP-scale ties, generalized from the exact bit patterns
    // tests_float_arith.rs's `test_float_add_with_rm_rz_f32`
    // (0x3F800001/0x3F800002 == 1.0+1ulp/1.0+2ulp) already proved
    // mode-sensitive for f32; `f32::EPSILON`/`f64::EPSILON` *is* ulp(1.0)
    // for each precision, so this construction is precision-generic.
    let add_a_f32 = 1.0f32 + f32::EPSILON;
    let add_b_f32 = 1.0f32 + 2.0 * f32::EPSILON;
    let add_a_f64 = 1.0f64 + f64::EPSILON;
    let add_b_f64 = 1.0f64 + 2.0 * f64::EPSILON;

    // FMA tie, generalized from tests_float_arith.rs's `fma_tie_operands`
    // (c = 3.0 * 2^-54 == 0.75 * f64::EPSILON for f64).
    let fma_c_f32 = 0.75 * f32::EPSILON;
    let fma_c_f64 = 0.75 * f64::EPSILON;

    vec![
        // -- Family A: rm-taking arms of `binop_float`, reached through
        // plain `VEXOps::binop` (VEX encodes these as Binops whose first
        // operand is the rm). Ties (±2.5) reused from
        // tests_conversions.rs's `test_round_f32_to_int_symbolic_rm` /
        // `test_round_f64_to_int_symbolic_rm`.
        // -0.3 rounds to -0.0 under every mode but RD: IEEE roundToIntegral
        // keeps the sign of zero, which the concrete path once dropped
        // (angr-vmi2l.57). -0.0 itself is mode-invariant.
        RmOpEntry {
            mode_invariant_cases: vec![vec![(f32_bits(-0.0), 32)]],
            ..family_a(
                "RoundF32toInt",
                IROp::RoundF32toInt,
                32,
                vec![f32_bits(2.5), f32_bits(-2.5), f32_bits(-0.3)],
            )
        },
        RmOpEntry {
            mode_invariant_cases: vec![vec![(f64_bits(-0.0), 64)]],
            ..family_a(
                "RoundF64toInt",
                IROp::RoundF64toInt,
                64,
                vec![f64_bits(2.5), f64_bits(-2.5), f64_bits(-0.3)],
            )
        },
        family_a(
            "F64toF32",
            IROp::F64toF32,
            64,
            vec![
                f64_bits(f64_to_f32_straddle(1.0)),
                f64_bits(f64_to_f32_straddle(-1.0)),
                f64_bits(f64_to_f32_tie(1.0)),
                f64_bits(f64_to_f32_tie(-1.0)),
            ],
        ),
        family_a(
            "F32toI32S",
            IROp::F32toI32S,
            32,
            vec![f32_bits(2.5), f32_bits(-2.5)],
        ),
        family_a(
            "F64toI32S",
            IROp::F64toI32S,
            64,
            vec![f64_bits(2.5), f64_bits(-2.5)],
        ),
        family_a(
            "F64toI16S",
            IROp::F64toI16S,
            64,
            vec![f64_bits(2.5), f64_bits(-2.5)],
        ),
        family_a(
            "F32toI64S",
            IROp::F32toI64S,
            32,
            vec![f32_bits(2.5), f32_bits(-2.5)],
        ),
        family_a(
            "F64toI64S",
            IROp::F64toI64S,
            64,
            vec![f64_bits(2.5), f64_bits(-2.5)],
        ),
        // Unsigned destinations: positive tie only.
        family_a("F32toI32U", IROp::F32toI32U, 32, vec![f32_bits(2.5)]),
        family_a("F64toI32U", IROp::F64toI32U, 64, vec![f64_bits(2.5)]),
        family_a("F32toI64U", IROp::F32toI64U, 32, vec![f32_bits(2.5)]),
        family_a("F64toI64U", IROp::F64toI64U, 64, vec![f64_bits(2.5)]),
        // Int-to-float: 2^p + 1 is the tie between 2^p and 2^p + 2 when the
        // destination has a p-bit significand (24 for F32, 53 for F64), so
        // RNE/RD/RZ and RU/RNA land on different neighbours.
        family_a(
            "I32StoF32",
            IROp::I32StoF32,
            32,
            vec![(1 << 24) + 1, (-(1i32 << 24) - 1) as u32 as u128],
        ),
        family_a("I32UtoF32", IROp::I32UtoF32, 32, vec![(1 << 24) + 1]),
        family_a(
            "I64StoF32",
            IROp::I64StoF32,
            64,
            vec![(1 << 24) + 1, (-(1i64 << 24) - 1) as u64 as u128],
        ),
        family_a("I64UtoF32", IROp::I64UtoF32, 64, vec![(1 << 24) + 1]),
        family_a(
            "I64StoF64",
            IROp::I64StoF64,
            64,
            vec![(1 << 53) + 1, (-(1i64 << 53) - 1) as u64 as u128],
        ),
        family_a("I64UtoF64", IROp::I64UtoF64, 64, vec![(1 << 53) + 1]),
        // -- Family B: `binop_with_rm` arms (rm + 2 operands). --
        binop_with_rm_entry(
            "FAdd(F32)",
            IROp::FAdd(IRType::F32),
            vec![(f32_bits(add_a_f32), 32), (f32_bits(add_b_f32), 32)],
        ),
        binop_with_rm_entry(
            "FAdd(F64)",
            IROp::FAdd(IRType::F64),
            vec![(f64_bits(add_a_f64), 64), (f64_bits(add_b_f64), 64)],
        ),
        // a - b == a + (-b): reuse the same tie sum by negating b.
        binop_with_rm_entry(
            "FSub(F32)",
            IROp::FSub(IRType::F32),
            vec![(f32_bits(add_a_f32), 32), (f32_bits(-add_b_f32), 32)],
        ),
        binop_with_rm_entry(
            "FSub(F64)",
            IROp::FSub(IRType::F64),
            vec![(f64_bits(add_a_f64), 64), (f64_bits(-add_b_f64), 64)],
        ),
        // (1+eps)^2 == 1 + 2eps + eps^2: eps^2 is far below half a ulp, so
        // this always lands just above the 1+2eps grid point.
        binop_with_rm_entry(
            "FMul(F32)",
            IROp::FMul(IRType::F32),
            vec![(f32_bits(add_a_f32), 32), (f32_bits(add_a_f32), 32)],
        ),
        binop_with_rm_entry(
            "FMul(F64)",
            IROp::FMul(IRType::F64),
            vec![(f64_bits(add_a_f64), 64), (f64_bits(add_a_f64), 64)],
        ),
        // 1/10 is not exactly representable in binary at any precision, so
        // it always needs rounding (reused verbatim from
        // tests_float_arith.rs's `test_float_div_with_rm_*_f32`).
        binop_with_rm_entry(
            "FDiv(F32)",
            IROp::FDiv(IRType::F32),
            vec![(f32_bits(1.0), 32), (f32_bits(10.0), 32)],
        ),
        binop_with_rm_entry(
            "FDiv(F64)",
            IROp::FDiv(IRType::F64),
            vec![(f64_bits(1.0), 64), (f64_bits(10.0), 64)],
        ),
        // -- Family B: `unop_with_rm` arm (rm + 1 operand). --
        // sqrt(2) is irrational, reused from
        // tests_float_arith.rs's `test_float_sqrt_with_rm_ru_f32`.
        unop_with_rm_entry(
            "FSqrt(F32)",
            IROp::FSqrt(IRType::F32),
            vec![(f32_bits(2.0), 32)],
        ),
        unop_with_rm_entry(
            "FSqrt(F64)",
            IROp::FSqrt(IRType::F64),
            vec![(f64_bits(2.0), 64)],
        ),
        // -- Packed forms: the same tie cases in every lane (angr-mx00p.130). --
        binop_with_rm_entry(
            "VFAdd(F32x4)",
            IROp::VFAdd {
                elem: IRType::F32,
                count: 4,
            },
            vec![
                (splat(f32_bits(add_a_f32), 32, 4), 128),
                (splat(f32_bits(add_b_f32), 32, 4), 128),
            ],
        ),
        binop_with_rm_entry(
            "VFSub(F64x2)",
            IROp::VFSub {
                elem: IRType::F64,
                count: 2,
            },
            vec![
                (splat(f64_bits(add_a_f64), 64, 2), 128),
                (splat(f64_bits(-add_b_f64), 64, 2), 128),
            ],
        ),
        binop_with_rm_entry(
            "VFMul(F32x4)",
            IROp::VFMul {
                elem: IRType::F32,
                count: 4,
            },
            vec![
                (splat(f32_bits(add_a_f32), 32, 4), 128),
                (splat(f32_bits(add_a_f32), 32, 4), 128),
            ],
        ),
        binop_with_rm_entry(
            "VFDiv(F64x2)",
            IROp::VFDiv {
                elem: IRType::F64,
                count: 2,
            },
            vec![
                (splat(f64_bits(1.0), 64, 2), 128),
                (splat(f64_bits(10.0), 64, 2), 128),
            ],
        ),
        unop_with_rm_entry(
            "VFSqrt(F32x4)",
            IROp::VFSqrt {
                elem: IRType::F32,
                count: 4,
            },
            vec![(splat(f32_bits(2.0), 32, 4), 128)],
        ),
        // -- Family B: `qop_with_rm` arms (rm + 3 operands). --
        qop_with_rm_entry(
            "FMAdd(F32)",
            IROp::FMAdd(IRType::F32),
            vec![
                (f32_bits(1.0), 32),
                (f32_bits(1.0), 32),
                (f32_bits(fma_c_f32), 32),
            ],
        ),
        qop_with_rm_entry(
            "FMAdd(F64)",
            IROp::FMAdd(IRType::F64),
            vec![
                (f64_bits(1.0), 64),
                (f64_bits(1.0), 64),
                (f64_bits(fma_c_f64), 64),
            ],
        ),
        // FMSub computes a*b - c; negate c to land on the same FMAdd tie sum.
        qop_with_rm_entry(
            "FMSub(F32)",
            IROp::FMSub(IRType::F32),
            vec![
                (f32_bits(1.0), 32),
                (f32_bits(1.0), 32),
                (f32_bits(-fma_c_f32), 32),
            ],
        ),
        qop_with_rm_entry(
            "FMSub(F64)",
            IROp::FMSub(IRType::F64),
            vec![
                (f64_bits(1.0), 64),
                (f64_bits(1.0), 64),
                (f64_bits(-fma_c_f64), 64),
            ],
        ),
    ]
}
