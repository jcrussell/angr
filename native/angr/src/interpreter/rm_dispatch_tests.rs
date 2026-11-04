//! Dispatcher-selection half of the `*_rm` rounding-mode sweep (section 2 of
//! `vex::ops::tests_rounding_mode_sweep`'s module doc). The sweep's op
//! registry, `all_rm_ops()`, lives with the ops; this half drives the
//! interpreter, so it lives here and vex/ops tests name no interpreter type
//! (angr-mx00p.101.8).

use crate::interpreter::{TestHost, VEXInterpreter};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::{IRConst, IRExpr, IROp, TypeEnv, VexArch};
use crate::vex::ops::{OpError, VEXOps};
use crate::vex::ops::rm_op_registry::{Operand, RmDispatchKind, VEX_ROUNDING_MODES, all_rm_ops};

/// The caller-dispatch bug shape, reconstructed directly: call the rm-LESS
/// sibling `VEXOps` method (always RNE) instead of the `*_with_rm` variant —
/// what `eval_binop`/`eval_triop`/`eval_qop` would compute if they picked
/// the wrong method. `operands` excludes rm (same shape as `RmOpEntry::cases`).
fn eval_rne_only_sibling(
    kind: RmDispatchKind,
    op: IROp,
    operands: &[RustBV],
    ) -> Result<RustBV, OpError> {
    match kind {
        RmDispatchKind::BinopToUnopWithRm => VEXOps::unop(op, operands[0].clone()),
        RmDispatchKind::TriopWithRm => {
            VEXOps::binop(op, operands[0].clone(), operands[1].clone())
        }
        RmDispatchKind::QopWithRm => VEXOps::qop(
            op,
            operands[0].clone(),
            operands[1].clone(),
            operands[2].clone(),
        ),
        RmDispatchKind::FamilyA => {
            unreachable!("eval_rne_only_sibling is Family-B only; callers must filter FamilyA")
        }
    }
}

/// Build an `IRConst` carrying `bits` at `width`, for the interpreter-level
/// dispatch call below. Only the widths `all_rm_ops()`'s Family-B cases
/// actually use (rm is always 32-bit; FP operands are 32 or 64-bit, packed
/// ones 128).
fn irconst_for(bits: u128, width: u32) -> IRConst {
    match width {
        32 => IRConst::U32(bits as u32),
        64 => IRConst::U64(bits as u64),
        128 => IRConst::U128(bits),
        other => {
            panic!("irconst_for: unsupported operand width {other} in dispatcher-selection sweep")
        }
    }
}

/// Evaluate one Family-B op through the REAL interpreter-level dispatch
/// entry point (`VEXInterpreter::eval_binop`/`eval_triop`/`eval_qop` in
/// `interpreter/expressions_arith.rs`) — the layer the caller-dispatch bug actually
/// lived in (the caller picked `VEXOps::qop` instead of `qop_with_rm`),
/// as opposed to the rest of the sweep, which calls
/// `VEXOps::binop_with_rm`/`unop_with_rm`/`qop_with_rm` directly and so
/// cannot see a caller-picks-wrong-method regression.
///
/// `rm` and `case` are passed as raw bits/widths (not `RustBV`) so the IR
/// tree can be built entirely from `IRExpr::Const`, with no temps to seed.
/// `eval_binop`/`eval_triop`/`eval_qop` are `pub(super)` so this sweep can
/// call them (see their doc comments in `interpreter/expressions_arith.rs`).
fn eval_through_interpreter(
    kind: RmDispatchKind,
    op: IROp,
    rm: u32,
    case: &[Operand],
    ctx: &SymContext,
) -> Result<RustBV, String> {
    let rm_expr = IRExpr::Const(IRConst::U32(rm));
    let operand_exprs: Vec<IRExpr> = case
        .iter()
        .map(|&(bits, width)| IRExpr::Const(irconst_for(bits, width)))
        .collect();

    let callbacks = TestHost::default();
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
    }

/// Section 2 of the sweep: drives evaluation through the real
/// interpreter dispatch entry points and cross-checks the result against
/// BOTH the correctly-rm-threaded `*_with_rm` sibling (must match) and the
/// rm-less RNE-only sibling (must NOT match) — directly reconstructing the
/// caller-dispatch regression shape at the layer it actually lived in,
/// generalized across all 7 Family-B op kinds.
#[test]
fn test_dispatcher_selection_routes_through_rm_threaded_sibling() {
    // RNE (mode 0) takes the identical native fast path in both `*_with_rm`
    // and its rm-less sibling (the `VexRm::NearestEven` short-circuit every
    // `*_with_rm` opens with — see e.g. `binop_with_rm` in `ops/mod.rs`), so
    // it cannot distinguish correct dispatch from the bug shape. The other
    // modes are the candidates where the historical bug could diverge — but
    // not every case's *_with_rm answer differs from RNE at every one of them
    // (e.g. FAdd(F32)'s tie case happens to round the same way under RU as
    // under RNE), so a mode is only asserted on when it actually is
    // divergent for this case; the non-vacuousness check after the mode
    // loop requires at least one to be.
    let non_rne_modes = &VEX_ROUNDING_MODES[1..];

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

            for &rm in non_rne_modes {
                let ctx = SymContext::new_mock();
                let rm_bv = RustBV::concrete(rm as u128, 32);

                // The correctly-rm-threaded answer: the same `*_with_rm`
                // call section 1 already exercises directly.
                let threaded = (entry.call)(rm_bv, &operands).unwrap_or_else(|e| {
                    panic!("{}: with_rm call failed rm={rm}: {e:?}", entry.name)
                });
                let threaded_bits = ctx.eval(&threaded).unwrap_or_else(|| {
                    panic!("{}: eval(with_rm result) returned None rm={rm}", entry.name)
                });

                // The caller-dispatch bug shape: the rm-less sibling, always RNE.
                let rne_only = eval_rne_only_sibling(entry.kind, entry.op, &operands)
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
                     sibling (the rm-dropping bug shape)",
                    entry.name
                );
                assert_ne!(
                    interp_bits, rne_only_bits,
                    "{}: interpreter dispatch at rm={rm} for case {case:?} matches the \
                     RNE-only rm-less-sibling answer — this IS the rm-dropping bug shape \
                     (the interpreter picked VEXOps::{{binop,unop,qop}} instead of the \
                     *_with_rm variant)",
                    entry.name
                );
            }

            assert!(
                divergent_modes_checked > 0,
                "{}: case {case:?} is not mode-sensitive vs RNE under ANY non-RNE mode, so it \
                 cannot distinguish correct interpreter dispatch from the rm-dropping bug \
                 shape; pick a different operand case",
                entry.name
            );
        }
    }
}

