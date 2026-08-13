// Generic *_rm rounding-mode coverage sweep, in three sections.
//
// Historical bug shapes:
//   * angr-c7xno.85: `f64_to_f32_rm`'s concrete path ignored the VEX
//     rounding mode — a bug *inside* a `_with_rm`-family implementation.
//   * angr-03vl4.30: `eval_qop` (interpreter/expressions.rs) dropped the
//     Qop's rm and called the rm-less `VEXOps::qop` instead of
//     `VEXOps::qop_with_rm` — a bug in the CALLER's dispatch choice, not in
//     `qop_with_rm` itself.
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
//    way. This catches the angr-c7xno.85 bug shape (a `_with_rm` method's
//    concrete-fast-path branch silently dropping rm) for every op, both
//    families. It does **not** catch the angr-03vl4.30 bug shape for Family
//    B: `binop_with_rm`/`unop_with_rm`/`qop_with_rm` branch on the RM
//    OPERAND's concreteness, not the value operands', so this section's
//    "concrete" and "symbolic" legs call the identical dispatch method and
//    are Z3-vs-Z3 self-consistency for all 14 Family-B ops — mutation
//    testing the angr-03vl4.30 shape back in (interpreter calls the rm-less
//    sibling instead of the `_with_rm` variant) leaves this section green.
//    Section 2 exists to close that gap.
// 2. `test_dispatcher_selection_routes_through_rm_threaded_sibling` —
//    Family B only. Drives evaluation through the REAL interpreter-level
//    dispatch entry points (`VEXInterpreter::eval_binop`/`eval_triop`/
//    `eval_qop` in `interpreter/expressions.rs`) — the layer the
//    angr-03vl4.30 bug actually lived in — for one representative op per
//    Family-B kind, and asserts the interpreter's answer matches the
//    correctly-rm-threaded `*_with_rm` sibling and NOT the rm-less
//    (RNE-only) sibling. This is the section that would catch a
//    caller-picks-wrong-method regression.
// 3. `test_symbolic_rm_dispatch_matches_concrete_rm` — Family B only.
//    Exercises `dispatch_symbolic_rm` (`symbolic/value_z3.rs`), the branch
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

use super::*;
use crate::vex::ir::IRType;
use pyo3::prelude::*;

/// The 4 VEX rounding modes (low 2 bits of the rm operand), per the doc
/// comments in conversions.rs (`round_f32_to_int_with_mode` et al.):
/// 0=nearest (RNE), 1=down/-inf (RD), 2=up/+inf (RU), 3=zero/truncate (RZ).
const VEX_ROUNDING_MODES: [u32; 4] = [0, 1, 2, 3];

/// One non-rm operand of a candidate test case: its raw bit pattern plus its
/// width, so the sweep can build the right-width `RustBV` for it without the
/// entry needing a separate parallel width list.
type Operand = (u128, u32);

/// Which real VEX-encoded shape an entry's op arrives as, and — for Family
/// B — which interpreter-level `eval_*` dispatcher (`interpreter/
/// expressions.rs`) is responsible for picking its `VEXOps` method. Drives
/// sections 2 and 3 (see module doc comment): both are Family-B only,
/// because Family A is reached via plain `VEXOps::binop` with no separate
/// rm-less sibling to mis-route to, and no rm-concreteness branch to
/// exercise symbolically.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RmDispatchKind {
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
    /// (FMAdd/FMSub) — the exact shape angr-03vl4.30 broke.
    QopWithRm,
}

/// Signature every dispatch entry's `call` closure shares, regardless of the
/// underlying `VEXOps` method's arity (`binop`/`binop_with_rm`/
/// `unop_with_rm`/`qop_with_rm` all take a different number of non-rm
/// operands, flattened here into a slice).
type RmDispatchFn = dyn Fn(RustBV, &[RustBV], &SymContext) -> Result<RustBV, OpError>;

/// One `*_rm` op's dispatch entry. `call` invokes the *public* `VEXOps`
/// method the real interpreter uses (never a private/`pub(super)` helper),
/// exactly mirroring the call pattern each existing per-op test already uses.
struct RmOpEntry {
    name: &'static str,
    /// A handful of operand tuples known (and, independently, asserted by
    /// this sweep itself — see the mode-sensitivity check below) to round
    /// differently under different VEX modes, so the equivalence check below
    /// is non-vacuous.
    cases: Vec<Vec<Operand>>,
    call: Box<RmDispatchFn>,
    /// The `IROp` this entry dispatches, for sections 2/3's interpreter- and
    /// rm-less-sibling calls, which need the concrete `IROp` value rather
    /// than a pre-baked closure.
    op: IROp,
    /// Which dispatch shape `op` arrives as — see `RmDispatchKind`.
    kind: RmDispatchKind,
}

fn f32_bits(v: f32) -> u128 {
    v.to_bits() as u128
}
fn f64_bits(v: f64) -> u128 {
    v.to_bits() as u128
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
/// ops) — all reached via the ordinary `VEXOps::binop(op, rm, value, ctx)`
/// dispatch, per `ops/mod.rs::binop_float`.
fn family_a(name: &'static str, op: IROp, src_bits: u32, values: Vec<u128>) -> RmOpEntry {
    RmOpEntry {
        name,
        cases: values.into_iter().map(|v| vec![(v, src_bits)]).collect(),
        call: Box::new(move |rm, operands, ctx| VEXOps::binop(op, rm, operands[0].clone(), ctx)),
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
        call: Box::new(move |rm, operands, ctx| {
            VEXOps::binop_with_rm(op, rm, operands[0].clone(), operands[1].clone(), ctx)
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
        call: Box::new(move |rm, operands, ctx| {
            VEXOps::unop_with_rm(op, rm, operands[0].clone(), ctx)
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
        call: Box::new(move |rm, operands, ctx| {
            VEXOps::qop_with_rm(
                op,
                rm,
                operands[0].clone(),
                operands[1].clone(),
                operands[2].clone(),
                ctx,
            )
        }),
        op,
        kind: RmDispatchKind::QopWithRm,
    }
}

/// The full registry of `*_rm`-flavored ops, read off the real dispatch
/// match arms (see the module doc comment for exactly which ones).
fn all_rm_ops() -> Vec<RmOpEntry> {
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
        family_a(
            "RoundF32toInt",
            IROp::RoundF32toInt,
            32,
            vec![f32_bits(2.5), f32_bits(-2.5)],
        ),
        family_a(
            "RoundF64toInt",
            IROp::RoundF64toInt,
            64,
            vec![f64_bits(2.5), f64_bits(-2.5)],
        ),
        family_a(
            "F64toF32",
            IROp::F64toF32,
            64,
            vec![
                f64_bits(f64_to_f32_straddle(1.0)),
                f64_bits(f64_to_f32_straddle(-1.0)),
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
/// angr-c7xno.85 / angr-03vl4.30 bug shape) would still agree with itself on
/// the concrete side — the divergence only shows up against the Z3 path,
/// which unconditionally honors the mode.
#[test]
fn test_rounding_mode_concrete_symbolic_equivalence() {
    for entry in all_rm_ops() {
        for case in &entry.cases {
            let mut mode_results = Vec::with_capacity(VEX_ROUNDING_MODES.len());

            for &rm in &VEX_ROUNDING_MODES {
                let ctx = SymContext::new_mock();
                let rm_bv = RustBV::concrete(rm as u128, 32);

                // Concrete path: every operand is a plain concrete RustBV.
                let concrete_operands: Vec<RustBV> = case
                    .iter()
                    .map(|&(bits, width)| RustBV::concrete(bits, width))
                    .collect();
                let concrete_result = (entry.call)(rm_bv.clone(), &concrete_operands, &ctx)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{}: concrete call failed for case {case:?} rm={rm}: {e:?}",
                            entry.name
                        )
                    });
                let concrete_bits = ctx.eval(&concrete_result).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(concrete_result) returned None for case {case:?} rm={rm}",
                        entry.name
                    )
                });

                // Symbolic path: fresh symbolic vars pinned to the exact
                // same bits via a Z3 constraint, then the *same* dispatcher
                // call and an eval of the result through the model.
                let symbolic_operands: Vec<RustBV> = case
                    .iter()
                    .enumerate()
                    .map(|(i, &(bits, width))| {
                        let sym = RustBV::symbolic(&ctx, format!("operand{i}"), width);
                        let pin = sym
                            .to_z3_ast()
                            .eq(RustBV::concrete(bits, width).to_z3_ast());
                        ctx.add_constraint(pin);
                        sym
                    })
                    .collect();
                let symbolic_result = (entry.call)(rm_bv.clone(), &symbolic_operands, &ctx)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{}: symbolic call failed for case {case:?} rm={rm}: {e:?}",
                            entry.name
                        )
                    });
                assert!(
                    ctx.is_sat(),
                    "{}: pinned-operand context must stay SAT (case {case:?} rm={rm})",
                    entry.name
                );
                let symbolic_bits = ctx.eval(&symbolic_result).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(symbolic_result) returned None for case {case:?} rm={rm}",
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
                mode_results.iter().any(|&r| r != mode_results[0]),
                "{}: case {case:?} is not mode-sensitive — all 4 VEX rounding modes gave \
                 {:#x}; pick a different boundary operand",
                entry.name,
                mode_results[0]
            );
        }
    }
}

// =============================================================================
// Section 2: dispatcher-selection coverage (Family B only).
// =============================================================================

/// The angr-03vl4.30 bug shape, reconstructed directly: call the rm-LESS
/// sibling `VEXOps` method (always RNE) instead of the `*_with_rm` variant —
/// what `eval_binop`/`eval_triop`/`eval_qop` would compute if they picked
/// the wrong method. `operands` excludes rm (same shape as `RmOpEntry::cases`).
fn eval_rne_only_sibling(
    kind: RmDispatchKind,
    op: IROp,
    operands: &[RustBV],
    ctx: &SymContext,
) -> Result<RustBV, OpError> {
    match kind {
        RmDispatchKind::BinopToUnopWithRm => VEXOps::unop(op, operands[0].clone(), ctx),
        RmDispatchKind::TriopWithRm => {
            VEXOps::binop(op, operands[0].clone(), operands[1].clone(), ctx)
        }
        RmDispatchKind::QopWithRm => VEXOps::qop(
            op,
            operands[0].clone(),
            operands[1].clone(),
            operands[2].clone(),
            ctx,
        ),
        RmDispatchKind::FamilyA => {
            unreachable!("eval_rne_only_sibling is Family-B only; callers must filter FamilyA")
        }
    }
}

/// Build an `IRConst` carrying `bits` at `width`, for the interpreter-level
/// dispatch call below. Only the widths `all_rm_ops()`'s Family-B cases
/// actually use (rm is always 32-bit; FP operands are 32 or 64-bit).
fn irconst_for(bits: u128, width: u32) -> crate::vex::ir::IRConst {
    use crate::vex::ir::IRConst;
    match width {
        32 => IRConst::U32(bits as u32),
        64 => IRConst::U64(bits as u64),
        other => {
            panic!("irconst_for: unsupported operand width {other} in dispatcher-selection sweep")
        }
    }
}

/// Evaluate one Family-B op through the REAL interpreter-level dispatch
/// entry point (`VEXInterpreter::eval_binop`/`eval_triop`/`eval_qop` in
/// `interpreter/expressions.rs`) — the layer the angr-03vl4.30 bug actually
/// lived in (the caller picked `VEXOps::qop` instead of `qop_with_rm`),
/// as opposed to the rest of this file, which calls
/// `VEXOps::binop_with_rm`/`unop_with_rm`/`qop_with_rm` directly and so
/// cannot see a caller-picks-wrong-method regression.
///
/// `rm` and `case` are passed as raw bits/widths (not `RustBV`) so the IR
/// tree can be built entirely from `IRExpr::Const` — `VEXInterpreter::temps`
/// is module-private to `interpreter` and unreachable from this file, but
/// `eval_binop`/`eval_triop`/`eval_qop` themselves are `pub(crate)`
/// specifically so this sweep can call them (see their doc comments in
/// `interpreter/expressions.rs`).
fn eval_through_interpreter(
    kind: RmDispatchKind,
    op: IROp,
    rm: u32,
    case: &[Operand],
    ctx: &SymContext,
) -> Result<RustBV, String> {
    use crate::callbacks::PythonCallbacks;
    use crate::interpreter::VEXInterpreter;
    use crate::vex::ir::{IRExpr, TypeEnv, VexArch};

    let rm_expr = IRExpr::Const(crate::vex::ir::IRConst::U32(rm));
    let operand_exprs: Vec<IRExpr> = case
        .iter()
        .map(|&(bits, width)| IRExpr::Const(irconst_for(bits, width)))
        .collect();

    Python::initialize();
    Python::attach(|_py| {
        let callbacks = PythonCallbacks::new();
        let mut interp = VEXInterpreter::new(VexArch::AMD64, ctx);
        let env = TypeEnv::new();
        let result = match kind {
            RmDispatchKind::BinopToUnopWithRm => {
                interp.eval_binop(&callbacks, op, &rm_expr, &operand_exprs[0], &env)
            }
            RmDispatchKind::TriopWithRm => interp.eval_triop(
                &callbacks,
                op,
                &rm_expr,
                &operand_exprs[0],
                &operand_exprs[1],
                &env,
            ),
            RmDispatchKind::QopWithRm => interp.eval_qop(
                &callbacks,
                op,
                [
                    &rm_expr,
                    &operand_exprs[0],
                    &operand_exprs[1],
                    &operand_exprs[2],
                ],
                &env,
            ),
            RmDispatchKind::FamilyA => unreachable!(
                "eval_through_interpreter is Family-B only; callers must filter FamilyA"
            ),
        };
        result.map_err(|e| format!("{e:?}"))
    })
}

/// Section 2 (see module doc comment): drives evaluation through the real
/// interpreter dispatch entry points and cross-checks the result against
/// BOTH the correctly-rm-threaded `*_with_rm` sibling (must match) and the
/// rm-less RNE-only sibling (must NOT match) — directly reconstructing the
/// angr-03vl4.30 regression shape at the layer it actually lived in,
/// generalized across all 7 Family-B op kinds.
#[test]
fn test_dispatcher_selection_routes_through_rm_threaded_sibling() {
    // RNE (mode 0) takes the identical native fast path in both `*_with_rm`
    // and its rm-less sibling (the `m & 0x3 == 0` short-circuit every
    // `*_with_rm` opens with — see e.g. `binop_with_rm` in `ops/mod.rs`), so
    // it cannot distinguish correct dispatch from the bug shape. RD/RU/RZ
    // are the candidates where the historical bug could diverge — but not
    // every case's *_with_rm answer differs from RNE at every one of those
    // three (e.g. FAdd(F32)'s tie case happens to round the same way under
    // RU as under RNE), so a mode is only asserted on when it actually is
    // divergent for this case; the non-vacuousness check after the mode
    // loop requires at least one to be.
    let non_rne_modes: [u32; 3] = [1, 2, 3];

    for entry in all_rm_ops() {
        if entry.kind == RmDispatchKind::FamilyA {
            continue;
        }
        for case in &entry.cases {
            let operands: Vec<RustBV> = case
                .iter()
                .map(|&(bits, width)| RustBV::concrete(bits, width))
                .collect();
            let mut divergent_modes_checked = 0;

            for &rm in &non_rne_modes {
                let ctx = SymContext::new_mock();
                let rm_bv = RustBV::concrete(rm as u128, 32);

                // The correctly-rm-threaded answer: the same `*_with_rm`
                // call section 1 already exercises directly.
                let threaded = (entry.call)(rm_bv, &operands, &ctx).unwrap_or_else(|e| {
                    panic!("{}: with_rm call failed rm={rm}: {e:?}", entry.name)
                });
                let threaded_bits = ctx.eval(&threaded).unwrap_or_else(|| {
                    panic!("{}: eval(with_rm result) returned None rm={rm}", entry.name)
                });

                // The angr-03vl4.30 bug shape: the rm-less sibling, always RNE.
                let rne_only = eval_rne_only_sibling(entry.kind, entry.op, &operands, &ctx)
                    .unwrap_or_else(|e| {
                        panic!("{}: rm-less sibling call failed: {e:?}", entry.name)
                    });
                let rne_only_bits = ctx.eval(&rne_only).unwrap_or_else(|| {
                    panic!("{}: eval(rm-less sibling result) returned None", entry.name)
                });

                if threaded_bits == rne_only_bits {
                    // This mode happens to round identically to RNE for this
                    // case — can't distinguish correct dispatch from the bug
                    // shape here, so skip it (a different mode may still be
                    // useful; enforced non-vacuous below).
                    continue;
                }
                divergent_modes_checked += 1;

                // The real interpreter entry point.
                let interp_result = eval_through_interpreter(entry.kind, entry.op, rm, case, &ctx)
                    .unwrap_or_else(|e| {
                        panic!("{}: interpreter dispatch failed rm={rm}: {e}", entry.name)
                    });
                let interp_bits = ctx.eval(&interp_result).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(interpreter result) returned None rm={rm}",
                        entry.name
                    )
                });

                assert_eq!(
                    interp_bits, threaded_bits,
                    "{}: interpreter dispatch (eval_binop/eval_triop/eval_qop) at rm={rm} \
                     disagrees with the correctly-rm-threaded VEXOps::*_with_rm answer for \
                     case {case:?} — the interpreter likely routed through the rm-less \
                     sibling (the angr-03vl4.30 bug shape)",
                    entry.name
                );
                assert_ne!(
                    interp_bits, rne_only_bits,
                    "{}: interpreter dispatch at rm={rm} for case {case:?} matches the \
                     RNE-only rm-less-sibling answer — this IS the angr-03vl4.30 bug shape \
                     (the interpreter picked VEXOps::{{binop,unop,qop}} instead of the \
                     *_with_rm variant)",
                    entry.name
                );
            }

            assert!(
                divergent_modes_checked > 0,
                "{}: case {case:?} is not mode-sensitive vs RNE under ANY of RD/RU/RZ, so it \
                 cannot distinguish correct interpreter dispatch from the angr-03vl4.30 bug \
                 shape; pick a different operand case",
                entry.name
            );
        }
    }
}

// =============================================================================
// Section 3: symbolic-rm dispatch coverage (Family B only).
// =============================================================================

/// Section 3 (see module doc comment): pins a genuinely symbolic rm
/// (`RustBV::symbolic`, not `RustBV::concrete`) to each VEX mode via a Z3
/// equality constraint, forcing `binop_with_rm`/`unop_with_rm`/`qop_with_rm`
/// past their `rm.as_u128()` concrete fast path into `build_float_expr` ->
/// `dispatch_symbolic_rm` (`symbolic/value_z3.rs`) — the ITE-over-4-modes
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
                let ctx = SymContext::new_mock();

                // Fully-concrete-rm answer for this mode (same shape as
                // section 1's concrete leg).
                let concrete_rm = RustBV::concrete(rm as u128, 32);
                let concrete_result =
                    (entry.call)(concrete_rm, &operands, &ctx).unwrap_or_else(|e| {
                        panic!("{}: concrete-rm call failed rm={rm}: {e:?}", entry.name)
                    });
                let concrete_bits = ctx.eval(&concrete_result).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(concrete-rm result) returned None rm={rm}",
                        entry.name
                    )
                });

                // Symbolic rm, path-constrained to the same mode via a Z3
                // equality — this is what makes `rm.as_u128()` return `None`
                // inside `binop_with_rm`/`unop_with_rm`/`qop_with_rm`,
                // routing through `dispatch_symbolic_rm` instead of the
                // concrete fast path.
                let rm_sym = RustBV::symbolic(&ctx, "rm_sym", 32);
                let pin = rm_sym
                    .to_z3_ast()
                    .eq(RustBV::concrete(rm as u128, 32).to_z3_ast());
                ctx.add_constraint(pin);
                let symbolic_result = (entry.call)(rm_sym, &operands, &ctx).unwrap_or_else(|e| {
                    panic!("{}: symbolic-rm call failed rm={rm}: {e:?}", entry.name)
                });
                assert!(
                    ctx.is_sat(),
                    "{}: pinned-rm context must stay SAT (rm={rm})",
                    entry.name
                );
                let symbolic_bits = ctx.eval(&symbolic_result).unwrap_or_else(|| {
                    panic!(
                        "{}: eval(symbolic-rm result) returned None rm={rm}",
                        entry.name
                    )
                });

                assert_eq!(
                    concrete_bits, symbolic_bits,
                    "{}: concrete-rm/symbolic-rm mismatch at rm={rm} for case {case:?} — \
                     dispatch_symbolic_rm likely disagrees with the concrete-rm fast path",
                    entry.name
                );
            }
        }
    }
}
