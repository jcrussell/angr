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
//    interpreter type, and reads `rm_op_registry::all_rm_ops()` — Family B only. Drives evaluation through the REAL interpreter-level
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
use super::rm_op_registry::{RmDispatchKind, VEX_ROUNDING_MODES, all_rm_ops};
use super::*;


/// Concrete rm operands `VexRm::decode` rejects: libVEX's 5..=7, the AVX-512
/// embedded-rounding 8..=11, and values whose low bits alone would decode.
const UNSUPPORTED_ROUNDING_MODES: [u128; 9] = [5, 6, 7, 8, 9, 10, 11, 0xFFFF_FFFC, 1 << 32];

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
