//! The `IROp` arithmetic dispatchers — `Unop`/`Binop`/`Triop`/`Qop`/`CCall`.
//!
//! Split out of `expressions.rs` (angr-fs8kb.66). Each `eval_*` here unwraps
//! its operands, hands the op to [`VEXOps`], and on an unsupported op routes
//! through `vex_op_fallback` — which either defers the whole block to Python
//! (the default) or, under the `fabricate_unsupported_irop` escape hatch,
//! fabricates a fresh symbolic of `fabricated_result_width`. The load paths are
//! in `expressions_loads.rs`; the dispatch entry point and the remaining
//! non-arithmetic expression forms (`Ite`, `GetI`, `Const`) stay in
//! `expressions.rs`.

use super::*;
use crate::vex::ir::IRCallee;

/// The three binop families that parse to a concrete `IROp` but have no native
/// dispatch arm (`VEXOps::binop` returns `OpError::NotBinary`):
/// `Iop_Perm{8,32}x*` (=> `VPerm`), `Iop_Pclmul*`, and `Iop_Crc32C`. All deterministic — Python's
/// VEX engine models them exactly — so `eval_binop` routes them to Python
/// fallback rather than fabricating a wrong fresh symbolic. Keep this in lockstep
/// with `opcode_map.rs` if a new must-fallback family is added.
/// See bd `angr-s6miz` and the "Parse-succeeds / dispatch-fabricates (silent
/// BYPASS)" section of `docs/extending-angr/rust_vex_ops.rst`.
pub(super) fn is_dispatch_fabricate_family(op: &IROp) -> bool {
    matches!(
        op,
        IROp::VPerm { .. }
            | IROp::PclmulLQLQ
            | IROp::PclmulHQHQ
            | IROp::PclmulLQHQ
            | IROp::PclmulHQLQ
            | IROp::Crc32C
    )
}

/// Width of the fresh symbolic [`VEXInterpreter::vex_op_fallback`] fabricates
/// for an unsupported op. Normally the op's own `result_type()`; when that is
/// `None` — an unmapped VEX op, the only way to reach the fallback — the widest
/// *value* operand stands in. A hardcoded 64 (what `eval_unop`/`eval_triop`/
/// `eval_qop` each used before angr-0jh0j.28, while `eval_binop` already did
/// this) fabricates a wrongly-narrow placeholder for a 128-bit+ float/vector
/// op, and a wrongly-wide one for a byte/short op.
///
/// `value_widths` deliberately excludes a Triop/Qop rounding-mode operand: `rm`
/// is I32 metadata, not a value, so letting it participate would be a no-op at
/// best and (for a sub-32-bit op) a widening at worst.
pub(super) fn fabricated_result_width(op: &IROp, value_widths: &[u32]) -> u32 {
    op.result_type().map(|t| t.bits()).unwrap_or_else(|| {
        // SILENT(cat-a): every caller passes at least one operand width, so the
        // empty-slice arm is unreachable; 64 keeps the helper total.
        value_widths.iter().copied().max().unwrap_or(64)
    })
}

/// Opt-in escape hatch (`ANGR_RUST_FABRICATE_UNSUPPORTED_IROP`) for the
/// symbolic-operand arm of [`VEXInterpreter::vex_op_fallback`], the
/// condition-flag arm of [`VEXInterpreter::eval_ccall`], and the
/// no-handler-anywhere arm of `VEXInterpreter::handle_dirty_call`
/// (`statements.rs`). When set (to any
/// non-empty, non-`"0"` value) an unsupported op with a symbolic operand
/// fabricates a fresh unconstrained symbolic (the pre-angr-oyzvj behavior)
/// instead of routing the block to Python. Default (unset): route to Python —
/// the correct behavior, since fabricating an unconstrained value silently
/// explores both branches of any downstream condition. Read once per process
/// (cached), matching the `OnceLock` env pattern in `engine.rs`.
///
/// Named `FABRICATE_*`, deliberately NOT `BYPASS_*`: the existing
/// `BYPASS_UNSUPPORTED_IROP` SimOption means the opposite (route to Python's
/// resilience mixin), so reusing that name would invert its sense.
pub(super) fn fabricate_unsupported_irop() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("ANGR_RUST_FABRICATE_UNSUPPORTED_IROP")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

impl<'a> VEXInterpreter<'a> {
    /// Shared tail for the unsupported-op fallback in
    /// `eval_unop`/`eval_binop`/`eval_triop`/`eval_qop`. Symbolic operands route
    /// the block to Python's VEX engine (the reference implementation for the
    /// float / vector-float conversions that land here), mirroring the
    /// dispatch-fabricate family (angr-s6miz). Fabricating a fresh unconstrained
    /// symbolic instead — the pre-angr-oyzvj default — silently diverges: any
    /// downstream condition over the fabricated value explores both branches
    /// unconstrained. That behavior is retained only as an explicit opt-in via
    /// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` (visibility via
    /// `vex_bypass_fabricate_count`). Concrete operands propagate the typed
    /// `OpError` so the failure surfaces a typed `RustUnsupportedVexOpError` on
    /// the test path (and routes the state to the errored stash, op + arch in
    /// the message, in live exploration — see the taxonomy note in `errors.rs`)
    /// rather than fabricating a silently-wrong value (angr-sa3j). The per-op
    /// `python_vex_{unop,binop,triop,qop}_fallback_count` is bumped at the call
    /// site; this bumps the aggregate `python_vex_op_fallback_count`.
    fn vex_op_fallback(
        &mut self,
        e: OpError,
        any_sym: bool,
        width: u32,
        name: String,
    ) -> Result<RustBV, CbExecutionError> {
        self.stats.python_vex_op_fallback_count += 1;
        if any_sym {
            if fabricate_unsupported_irop() {
                self.stats.vex_bypass_fabricate_count += 1;
                Ok(RustBV::symbolic(self.ctx, name, width))
            } else {
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "unsupported symbolic IROp routed to Python ({name}): {e}"
                )))
            }
        } else {
            Err(CbExecutionError::Op(e))
        }
    }

    pub(super) fn eval_unop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        arg: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_unop(iropclass(&op));
        let arg_val = self.eval_expr_with_callbacks(callbacks, arg, tyenv)?;
        let arg_is_sym = arg_val.is_symbolic();
        let fallback_width = fabricated_result_width(&op, &[arg_val.width()]);
        match VEXOps::unop(op, arg_val, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly rather than letting the
            // fresh-symbolic fallback below swallow it. See the rustdoc on
            // `OpError::UnsupportedNeon` (vex/ops/error.rs).
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            // Fallback for unsupported unary ops (e.g., float conversions).
            Err(e) => {
                self.stats.python_vex_unop_fallback_count += 1;
                self.vex_op_fallback(
                    e,
                    arg_is_sym,
                    fallback_width,
                    format!("unsup_unop_{:x}", self.pc),
                )
            }
        }
    }

    /// `pub(crate)` (rather than the module-private default every sibling
    /// `eval_*` dispatcher uses) so `vex::ops::tests_rounding_mode_sweep`'s
    /// dispatcher-selection sweep can drive evaluation through the real
    /// entry point that had the angr-03vl4.30 bug, instead of calling
    /// `VEXOps::binop_with_rm`/`qop_with_rm` directly the way the rest of
    /// that file does — see the module doc comment there.
    pub(crate) fn eval_binop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        left: &IRExpr,
        right: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_binop(iropclass(&op));
        let left_val = self.eval_expr_with_callbacks(callbacks, left, tyenv)?;
        let right_val = self.eval_expr_with_callbacks(callbacks, right, tyenv)?;
        let fallback_width = fabricated_result_width(&op, &[left_val.width(), right_val.width()]);
        let any_sym = left_val.is_symbolic() || right_val.is_symbolic();
        match VEXOps::binop(op, left_val, right_val, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly rather than letting the
            // fresh-symbolic fallback below swallow it. See the rustdoc on
            // `OpError::UnsupportedNeon` (vex/ops/error.rs).
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            // angr-s6miz: the three dispatch-fabricate families
            // (Iop_Perm{8,32}x* => VPerm, Iop_Pclmul*, Iop_Crc32C) parse to a
            // concrete IROp but have no native dispatch arm, so `binop` returns
            // `NotBinary`. They are deterministic ops Python models exactly, so
            // route the block to Python's VEX engine rather than fabricating a
            // wrong fresh symbolic (the BYPASS arm below). Routed for BOTH
            // symbolic and concrete args — strictly better than the old
            // fabricate-on-sym / hard-error-on-concrete split.
            Err(_) if is_dispatch_fabricate_family(&op) => {
                // angr-9ke6b.85: this IS a Python-VEX-fallback event, so it
                // bumps the same two counters the ordinary `Err(e)` arm below
                // does (per-op subset + aggregate). It cannot route through
                // `vex_op_fallback`: that helper hard-errors on all-concrete
                // args and honors the fabricate opt-in, both of which this
                // family deliberately bypasses.
                self.stats.python_vex_binop_fallback_count += 1;
                self.stats.python_vex_op_fallback_count += 1;
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "{} ({:?})",
                    crate::interpreter::DISPATCH_FABRICATE_REASON,
                    op
                )))
            }
            // Fallback for unsupported binary ops (e.g., vector float ops).
            Err(e) => {
                self.stats.python_vex_binop_fallback_count += 1;
                self.vex_op_fallback(
                    e,
                    any_sym,
                    fallback_width,
                    format!("unsup_binop_{:x}", self.pc),
                )
            }
        }
    }

    /// `pub(crate)` — see the doc comment on `eval_binop` above; same
    /// dispatcher-selection-sweep reason applies here.
    pub(crate) fn eval_triop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        arg1: &IRExpr,
        arg2: &IRExpr,
        arg3: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_triop(iropclass(&op));
        // VEX Triops are float arithmetic with a rounding mode:
        // (rm, a, b). For FAdd/FSub/FMul/FDiv we route through
        // `binop_with_rm` which honors the VEX rm bits when non-RNE;
        // RNE keeps the native-f{32,64} fast path. Other Triops
        // ignore rm and fall through to `binop`.
        let rm = self.eval_expr_with_callbacks(callbacks, arg1, tyenv)?;
        let v2 = self.eval_expr_with_callbacks(callbacks, arg2, tyenv)?;
        let v3 = self.eval_expr_with_callbacks(callbacks, arg3, tyenv)?;
        let any_sym = v2.is_symbolic() || v3.is_symbolic() || rm.is_symbolic();
        let width = fabricated_result_width(&op, &[v2.width(), v3.width()]);
        match VEXOps::binop_with_rm(op, rm, v2, v3, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly rather than letting the
            // fresh-symbolic fallback below swallow it. See the rustdoc on
            // `OpError::UnsupportedNeon` (vex/ops/error.rs).
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            Err(e) => {
                self.stats.python_vex_triop_fallback_count += 1;
                self.vex_op_fallback(e, any_sym, width, format!("unsup_triop_{:x}", self.pc))
            }
        }
    }

    /// `args` is the Qop's four operands in IR order (`arg1..arg4`).
    ///
    /// `pub(crate)` — see the doc comment on `eval_binop` above; this is the
    /// exact dispatcher the angr-03vl4.30 bug lived in (routed to
    /// `VEXOps::qop` instead of `qop_with_rm`), so the dispatcher-selection
    /// sweep needs to call it directly.
    pub(crate) fn eval_qop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        args: [&IRExpr; 4],
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_qop(iropclass(&op));
        // VEX Qops are fused multiply-add/sub with a rounding mode:
        // (rm, a, b, c). Route through `qop_with_rm`, which honors the VEX
        // rm bits when non-RNE and keeps the native mul_add fast path for
        // RNE — same split as `binop_with_rm` in the Triop arm above.
        let rm = self.eval_expr_with_callbacks(callbacks, args[0], tyenv)?;
        let v2 = self.eval_expr_with_callbacks(callbacks, args[1], tyenv)?;
        let v3 = self.eval_expr_with_callbacks(callbacks, args[2], tyenv)?;
        let v4 = self.eval_expr_with_callbacks(callbacks, args[3], tyenv)?;
        let any_sym = v2.is_symbolic() || v3.is_symbolic() || v4.is_symbolic() || rm.is_symbolic();
        let width = fabricated_result_width(&op, &[v2.width(), v3.width(), v4.width()]);
        match VEXOps::qop_with_rm(op, rm, v2, v3, v4, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly rather than letting the
            // fresh-symbolic fallback below swallow it. See the rustdoc on
            // `OpError::UnsupportedNeon` (vex/ops/error.rs).
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            Err(e) => {
                self.stats.python_vex_qop_fallback_count += 1;
                self.vex_op_fallback(e, any_sym, width, format!("unsup_qop_{:x}", self.pc))
            }
        }
    }

    pub(super) fn eval_ccall(
        &mut self,
        callbacks: &PythonCallbacks,
        cee: &IRCallee,
        retty: IRType,
        args: &[IRExpr],
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let mut arg_vals = Vec::with_capacity(args.len());
        for arg in args {
            arg_vals.push(self.eval_expr_with_callbacks(callbacks, arg, tyenv)?);
        }

        if let Some(result) =
            ccall::handle_ccall_with_ctx(&cee.name, &arg_vals, retty.bits(), Some(self.ctx))
        {
            return Ok(result);
        }

        // eflags/rflags condition-calculation CCalls that we couldn't handle
        // symbolically follow the same policy as `vex_op_fallback` (angr-oyzvj,
        // angr-9ke6b.88): route the block to Python by default. Fabricating a
        // fresh unconstrained symbolic here is *worse* than for an ordinary op,
        // because the result of a `calculate_condition` CCall IS a branch
        // guard — an unconstrained one explores both directions regardless of
        // the real flag semantics. Retained only as an explicit opt-in via
        // `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` (same gate, same
        // `vex_bypass_fabricate_count` visibility).
        let is_cond_ccall = cee.name.contains("calculate_condition")
            || cee.name.contains("calculate_eflags")
            || cee.name.contains("calculate_rflags");
        if is_cond_ccall && fabricate_unsupported_irop() {
            log::debug!(
                "CCall '{}' not handled symbolically at 0x{:x}, fabricating symbolic variable \
                 (ANGR_RUST_FABRICATE_UNSUPPORTED_IROP)",
                cee.name,
                self.pc
            );
            self.stats.vex_bypass_fabricate_count += 1;
            return Ok(RustBV::symbolic(
                self.ctx,
                format!("ccall_unsupported_{:x}", self.pc),
                retty.bits(),
            ));
        }

        // Every other unsupported CCall must defer to Python's VEX engine.
        // Returning concrete(0) would silently corrupt the result and let
        // execution continue with bad data.
        Err(CbExecutionError::NeedPythonFallback(format!(
            "unsupported CCall '{}' at 0x{:x}",
            cee.name, self.pc
        )))
    }
}
