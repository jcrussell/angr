// Generic *_rm rounding-mode coverage sweep, in three sections.
//
// The two bug shapes it guards against:
//   * the inner-rm shape: a `_with_rm`-family implementation's concrete path
//     ignores the VEX rounding mode, as `f64_to_f32_rm`'s once did
//     (angr-c7xno.85).
//   * the caller-dispatch shape: a dispatcher such as `eval_qop`
//     (interpreter/expressions_arith.rs) drops the Qop's rm and calls the
//     rm-less `VEXOps::qop` instead of `VEXOps::qop_with_rm` — a bug in the
//     CALLER's dispatch choice, not in `qop_with_rm` itself (angr-03vl4.30).
// Both were caught reactively with hand-written per-op tests (see
// `test_f64_to_f32_rm_straddles_boundary` in tests_conversions.rs and
// `test_qop_with_rm_rz_f64_differs_from_rne` in tests_float_arith.rs). This
// file is the proactive, DRY counterpart, split into three sections that
// each prove a different thing:
//
// 1. `test_rounding_mode_concrete_symbolic_equivalence` — Family A (the
//    rm-taking arms of `binop_float`, reached through plain `VEXOps::binop`:
//    F64toF32 and the float<->int conversions) AND Family B (`FAdd`/`FSub`/
//    `FMul`/`FDiv`/`FSqrt`/`FMAdd`/`FMSub`, reached through `binop_with_rm`/
//    `unop_with_rm`/`qop_with_rm`). Builds each op's non-rm operands two
//    ways (fully concrete, and symbolic vars pinned via a Z3 constraint to
//    the same bits) and asserts the SAME dispatch method — called directly,
//    never through the interpreter — gives the same bit-exact answer either
//    way. This catches the inner-rm bug shape (a `_with_rm` method's
//    concrete-fast-path branch silently dropping rm) for every op, both
//    families. It does **not** catch the caller-dispatch bug shape for Family
//    B: `binop_with_rm`/`unop_with_rm`/`qop_with_rm` branch on the RM
//    OPERAND's concreteness, not the value operands', so this section's
//    "concrete" and "symbolic" legs call the identical dispatch method and
//    are Z3-vs-Z3 self-consistency for all 14 Family-B ops — mutation
//    testing the caller-dispatch shape back in (interpreter calls the rm-less
//    sibling instead of the `_with_rm` variant) leaves this section green.
//    Section 2 exists to close that gap.
// 2. `test_dispatcher_selection_routes_through_rm_threaded_sibling`, which
//    lives in `interpreter::rm_dispatch_tests` so that vex/ops tests name no
//    interpreter type, and reads this file's `all_rm_ops()` — Family B only. Drives evaluation through the REAL interpreter-level
//    dispatch entry points (`VEXInterpreter::eval_binop`/`eval_triop`/
//    `eval_qop` in `interpreter/expressions_arith.rs`) — the layer the
//    caller-dispatch bug actually lived in — for one representative op per
//    Family-B kind, and asserts the interpreter's answer matches the
//    correctly-rm-threaded `*_with_rm` sibling and NOT the rm-less
//    (RNE-only) sibling. This is the section that would catch a
//    caller-picks-wrong-method regression.
// 3. `test_symbolic_rm_dispatch_matches_concrete_rm` — Family B only.
//    Exercises `with_vex_rm` (`value_clarirs.rs`), the branch
//    `binop_with_rm`/`unop_with_rm`/`qop_with_rm` take when the rm OPERAND
//    ITSELF is symbolic (as opposed to section 1's symbolic-VALUE/
//    concrete-rm legs) — never exercised before this section, since
//    `all_rm_ops()`'s cases only ever build a concrete rm.
//
// Deliberate exclusion, not an oversight: the `IROp::Raw` x87 transcendental
// ops (`Iop_AtanF64`, `Iop_Yl2xF64`, `Iop_Yl2xp1F64`, `Iop_ScaleF64`, routed
// through the `IROp::Raw` arm of `binop_with_rm` into `transcendentals.rs`)
// take an rm operand that is consumed but ignored by design — Rust libm has
// no rounding-mode-parameterized transcendental, so
// `try_concrete_triop_rm`'s `_rm` parameter is intentionally unused (see
// `transcendentals.rs`'s module doc comment). There is no Z3 FP
// transcendental theory to compare against either. None of this file's three
// sections cover them.
//
// `all_rm_ops()` mirrors — not reinvents — the real dispatch surface: every
// entry below invokes the same *public* `VEXOps` methods the interpreter
// calls (`binop`, `binop_with_rm`, `unop_with_rm`, `qop_with_rm`), and the
// `IROp` variant list was read directly off their match arms (`ops/mod.rs`:
// the rm-taking arms of `binop_float`, reached through plain `binop` since
// VEX encodes e.g. `Iop_F64toF32(rm, value)` as a Binop, not a `_with_rm`
// suffixed shape; and the match arms inside `binop_with_rm` / `unop_with_rm`
// / `qop_with_rm` themselves) rather than guessed from memory. Rust has no
// runtime reflection over `match` arms, so this list is not *compile-time*
// guaranteed to stay in sync if a new rm-taking arm is added to those
// dispatchers — but it is a faithful, freshly-verified transcription of the
// arms as they exist now, not a hand-typed guess.
//
// Complements (does not replace) `tools/audit_rounding_mode_threading.py`,
// which only checks that the rm parameter is *touched* in the source; it
// cannot tell whether the rounding math itself agrees with Z3.

use super::test_helpers::{pinned_symbol, z3_eval};
use super::*;
use angr_bv::VexRm;
use crate::vex::ir::IRType;

/// Every VEX rounding mode `VexRm::decode` accepts, by encoding: 0=nearest
/// (RNE), 1=down/-inf (RD), 2=up/+inf (RU), 3=zero/truncate (RZ),
/// 4=nearest, ties away from zero (RNA; angr-vmi2l.56).
pub(crate) const VEX_ROUNDING_MODES: [u32; 5] = [
    VexRm::NearestEven.encoding(),
    VexRm::TowardNegative.encoding(),
    VexRm::TowardPositive.encoding(),
    VexRm::TowardZero.encoding(),
    VexRm::NearestAway.encoding(),
];

/// Concrete rm operands `VexRm::decode` rejects: libVEX's 5..=7, the AVX-512
/// embedded-rounding 8..=11, and values whose low bits alone would decode.
const UNSUPPORTED_ROUNDING_MODES: [u128; 9] = [5, 6, 7, 8, 9, 10, 11, 0xFFFF_FFFC, 1 << 32];

/// One non-rm operand of a candidate test case: its raw bit pattern plus its
/// width, so the sweep can build the right-width `RustBV` for it without the
/// entry needing a separate parallel width list.
pub(crate) type Operand = (u128, u32);

/// Which real VEX-encoded shape an entry's op arrives as, and — for Family
/// B — which interpreter-level `eval_*` dispatcher (`interpreter/
/// expressions_arith.rs`) is responsible for picking its `VEXOps` method. Drives
/// sections 2 and 3 (see module doc comment): both are Family-B only,
/// because Family A is reached via plain `VEXOps::binop` with no separate
/// rm-less sibling to mis-route to, and no rm-concreteness branch to
/// exercise symbolically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RmDispatchKind {
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
pub(crate) type RmDispatchFn = dyn Fn(RustBV, &[RustBV]) -> Result<RustBV, OpError>;

/// One `*_rm` op's dispatch entry. `call` invokes the *public* `VEXOps`
/// method the real interpreter uses (never a private/`pub(super)` helper),
/// exactly mirroring the call pattern each existing per-op test already uses.
pub(crate) struct RmOpEntry {
    pub(crate) name: &'static str,
    /// A handful of operand tuples known (and, independently, asserted by
    /// this sweep itself — see the mode-sensitivity check below) to round
    /// differently under different VEX modes, so the equivalence check below
    /// is non-vacuous.
    pub(crate) cases: Vec<Vec<Operand>>,
    /// Operand tuples every mode rounds identically (the sign of zero:
    /// `RoundF{32,64}toInt(-0.0)` is `-0.0` in every mode). Section 1 checks
    /// them for concrete/symbolic agreement but exempts them from the
    /// mode-sensitivity check, which they would fail by construction.
    pub(crate) mode_invariant_cases: Vec<Vec<Operand>>,
    pub(crate) call: Box<RmDispatchFn>,
    /// The `IROp` this entry dispatches, for sections 2/3's interpreter- and
    /// rm-less-sibling calls, which need the concrete `IROp` value rather
    /// than a pre-baked closure.
    pub(crate) op: IROp,
    /// Which dispatch shape `op` arrives as — see `RmDispatchKind`.
    pub(crate) kind: RmDispatchKind,
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
        call: Box::new(move |rm, operands| {
            VEXOps::unop_with_rm(op, rm, operands[0].clone())
        }),
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
pub(crate) fn all_rm_ops() -> Vec<RmOpEntry> {
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
            IROp::VFAdd { elem: IRType::F32, count: 4 },
            vec![(splat(f32_bits(add_a_f32), 32, 4), 128), (splat(f32_bits(add_b_f32), 32, 4), 128)],
        ),
        binop_with_rm_entry(
            "VFSub(F64x2)",
            IROp::VFSub { elem: IRType::F64, count: 2 },
            vec![(splat(f64_bits(add_a_f64), 64, 2), 128), (splat(f64_bits(-add_b_f64), 64, 2), 128)],
        ),
        binop_with_rm_entry(
            "VFMul(F32x4)",
            IROp::VFMul { elem: IRType::F32, count: 4 },
            vec![(splat(f32_bits(add_a_f32), 32, 4), 128), (splat(f32_bits(add_a_f32), 32, 4), 128)],
        ),
        binop_with_rm_entry(
            "VFDiv(F64x2)",
            IROp::VFDiv { elem: IRType::F64, count: 2 },
            vec![(splat(f64_bits(1.0), 64, 2), 128), (splat(f64_bits(10.0), 64, 2), 128)],
        ),
        unop_with_rm_entry(
            "VFSqrt(F32x4)",
            IROp::VFSqrt { elem: IRType::F32, count: 4 },
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

/// The sweep: for every `*_rm` op, every candidate operand case, and every
/// VEX rounding mode — build the operands two ways (fully concrete, and
/// symbolic vars pinned via a Z3 constraint to the same bits) and assert the
/// dispatcher gives the same bit-exact answer either way. A `*_rm`
/// implementation that silently drops the rm operand (the exact
/// inner-rm / caller-dispatch bug shape) would still agree with itself on
/// the concrete side — the divergence only shows up against the Z3 path,
/// which unconditionally honors the mode.
#[test]
fn test_rounding_mode_concrete_symbolic_equivalence() {
    for entry in all_rm_ops() {
        let sensitive = entry.cases.iter().map(|case| (case, true));
        let invariant = entry.mode_invariant_cases.iter().map(|case| (case, false));
        for (case, must_be_mode_sensitive) in sensitive.chain(invariant) {
            let mut mode_results = Vec::with_capacity(VEX_ROUNDING_MODES.len());

            for &rm in &VEX_ROUNDING_MODES {
                let rm_bv = RustBV::concrete(rm as u128, 32);

                // Concrete path: every operand is a plain concrete RustBV.
                let concrete_operands: Vec<RustBV> = case
                    .iter()
                    .map(|&(bits, width)| RustBV::concrete(bits, width))
                    .collect();
                let concrete_result = (entry.call)(rm_bv.clone(), &concrete_operands)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{}: concrete call failed for case {case:?} rm={rm}: {e:?}",
                            entry.name
                        )
                    });
                let concrete_bits = z3_eval(&concrete_result, &[]).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(concrete_result) returned None for case {case:?} rm={rm}",
                        entry.name
                    )
                });

                // Symbolic path: fresh symbolic vars pinned to the exact
                // same bits via a Z3 constraint, then the *same* dispatcher
                // call and an eval of the result through the model.
                let (symbolic_operands, pins): (Vec<RustBV>, Vec<RustBV>) = case
                    .iter()
                    .enumerate()
                    .map(|(i, &(bits, width))| pinned_symbol(&format!("operand{i}"), bits, width))
                    .unzip();
                let symbolic_result = (entry.call)(rm_bv.clone(), &symbolic_operands)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{}: symbolic call failed for case {case:?} rm={rm}: {e:?}",
                            entry.name
                        )
                    });
                let symbolic_bits = z3_eval(&symbolic_result, &pins).unwrap_or_else(|| {
                    panic!(
                        "{}: pinned operands unsatisfiable for case {case:?} rm={rm}",
                        entry.name
                    )
                });

                assert_eq!(
                    concrete_bits, symbolic_bits,
                    "{}: concrete/symbolic mismatch at rm={rm} for case {case:?} — the \
                     concrete path likely dropped the rounding mode",
                    entry.name
                );

                mode_results.push(concrete_bits);
            }

            // Non-vacuousness: if every mode produced the same bits, this
            // operand case doesn't actually exercise rounding, and the
            // concrete/symbolic equality above would pass even for a
            // *_rm implementation that silently ignores rm and always
            // computes RNE.
            assert!(
                !must_be_mode_sensitive || mode_results.iter().any(|&r| r != mode_results[0]),
                "{}: case {case:?} is not mode-sensitive — every VEX rounding mode gave \
                 {:#x}; pick a different boundary operand",
                entry.name,
                mode_results[0]
            );
        }
    }
}

// =============================================================================
// Section 3: symbolic-rm dispatch coverage (Family B only).
// =============================================================================

/// Section 3 (see module doc comment): pins a genuinely symbolic rm
/// (`RustBV::symbol`, not `RustBV::concrete`) to each VEX mode via a Z3
/// equality constraint, forcing `binop_with_rm`/`unop_with_rm`/`qop_with_rm`
/// past their `rm.as_u128()` concrete fast path into `build_float_expr` ->
/// `with_vex_rm` (`value_clarirs.rs`) — the ITE-over-`VexRm::ALL`
/// branch that section 1's `RustBV::concrete(rm, ...)` rm never reaches.
/// Asserts the path-constrained symbolic-rm answer matches the
/// fully-concrete-rm answer for the same mode.
#[test]
fn test_symbolic_rm_dispatch_matches_concrete_rm() {
    for entry in all_rm_ops() {
        if entry.kind == RmDispatchKind::FamilyA {
            continue;
        }
        for case in &entry.cases {
            let operands: Vec<RustBV> = case
                .iter()
                .map(|&(bits, width)| RustBV::concrete(bits, width))
                .collect();

            for &rm in &VEX_ROUNDING_MODES {
                // Fully-concrete-rm answer for this mode (same shape as
                // section 1's concrete leg).
                let concrete_rm = RustBV::concrete(rm as u128, 32);
                let concrete_result =
                    (entry.call)(concrete_rm, &operands).unwrap_or_else(|e| {
                        panic!("{}: concrete-rm call failed rm={rm}: {e:?}", entry.name)
                    });
                let concrete_bits = z3_eval(&concrete_result, &[]).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(concrete-rm result) returned None rm={rm}",
                        entry.name
                    )
                });

                // Symbolic rm, path-constrained to the same mode via a Z3
                // equality — this is what makes `rm.as_u128()` return `None`
                // inside `binop_with_rm`/`unop_with_rm`/`qop_with_rm`,
                // routing through `with_vex_rm` instead of the
                // concrete fast path.
                let (rm_sym, pin) = pinned_symbol("rm_sym", rm as u128, 32);
                let symbolic_result = (entry.call)(rm_sym, &operands).unwrap_or_else(|e| {
                    panic!("{}: symbolic-rm call failed rm={rm}: {e:?}", entry.name)
                });
                let symbolic_bits = z3_eval(&symbolic_result, &[pin]).unwrap_or_else(|| {
                    panic!("{}: pinned rm unsatisfiable (rm={rm})", entry.name)
                });

                assert_eq!(
                    concrete_bits, symbolic_bits,
                    "{}: concrete-rm/symbolic-rm mismatch at rm={rm} for case {case:?} — \
                     with_vex_rm likely disagrees with the concrete-rm fast path",
                    entry.name
                );
            }
        }
    }
}

// =============================================================================
// Section 4: unsupported concrete rounding modes are errors (both families).
// =============================================================================

/// Every `*_rm` entry point rejects a concrete rm `VexRm::decode` does not
/// accept with `OpError::UnsupportedRoundingMode`, as the Python engine's
/// `translate_rm` raises `UnsupportedIROpError`, whether the value operands
/// are concrete (the native fold) or symbolic (the lowering it would build).
/// Masking the mode to its low bits used to turn 4 into RNE (angr-vmi2l.56).
#[test]
fn test_unsupported_rounding_mode_is_an_error() {
    for entry in all_rm_ops() {
        let case = &entry.cases[0];
        for &bad in &UNSUPPORTED_ROUNDING_MODES {
            let concrete: Vec<RustBV> = case
                .iter()
                .map(|&(bits, width)| RustBV::concrete(bits, width))
                .collect();
            let symbolic: Vec<RustBV> = case
                .iter()
                .enumerate()
                .map(|(i, &(_, width))| RustBV::symbol(format!("operand{i}"), width))
                .collect();
            for (label, operands) in [("concrete", &concrete), ("symbolic", &symbolic)] {
                let rm = RustBV::concrete(bad, 64);
                match (entry.call)(rm, operands) {
                    Err(OpError::UnsupportedRoundingMode(e)) => assert_eq!(e.0, bad),
                    other => panic!(
                        "{}: rm={bad:#x} with {label} operands must be \
                         UnsupportedRoundingMode, got {other:?}",
                        entry.name
                    ),
                }
            }
        }
    }
}
