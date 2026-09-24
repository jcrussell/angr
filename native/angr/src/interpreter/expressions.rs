//! `IRExpr` evaluation — the read half of VEX execution: the dispatch entry
//! point plus the expression forms that are neither a load nor an op.
//!
//! `eval_expr_with_callbacks` is the general entry point (`eval_expr_simple`
//! the no-callback variant used where re-entering Python is not allowed), and
//! `eval_expr_with_callbacks_inner` is the per-`IRExpr`-variant match every
//! other group hangs off. Staying here are the forms with no sibling family of
//! their own: `eval_ite`, `eval_geti` (+ `regarray_offset`), `eval_const`, and
//! the `concretize_and_pin` address helper.
//!
//! The three groups split out of this file by angr-fs8kb.66:
//!
//! * `expressions_loads.rs` — the load-resolution ladder (`eval_load` and
//!   everything under it, the multi-address/ITE callback helpers, `LoadG`).
//! * `expressions_arith.rs` — the `IROp` dispatchers
//!   (`eval_unop`/`eval_binop`/`eval_triop`/`eval_qop`/`eval_ccall`) and the
//!   `vex_op_fallback` unsupported-op path.
//! * `expressions_inspect.rs` — the read-side `state.inspect` breakpoint
//!   dispatchers this file and both siblings call into, mirror-image of the
//!   write-side `statements_inspect.rs`.
use super::*;
use crate::vex::ir::IRRegArray;

impl<'a> VEXInterpreter<'a> {
    /// Evaluate an IR expression using Python callbacks for memory loads.
    pub(super) fn eval_expr_with_callbacks(
        &mut self,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let expr_start = profile_start!(self);
        let result = self.eval_expr_with_callbacks_inner(callbacks, expr, tyenv);
        profile_add!(expr_start, self.stats.expr_eval_time_ns);
        if self.profiling_enabled {
            self.stats.expr_eval_count += 1;
        }
        // state.inspect expr event (angr-lge2) — fires `when='after'` after
        // every IRExpr evaluation. Gated on `inspect_event_enabled(InspectBit::Expr)` so
        // the no-BP case is one `AtomicU32::load + AND` per call. This is
        // the highest-frequency dispatch site in the engine (every binop
        // arg, store data, exit guard, etc. comes through here).
        if let Ok(ref value) = result {
            self.dispatch_expr_inspect(callbacks, value);
        }
        result
    }

    fn eval_expr_with_callbacks_inner(
        &mut self,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),

            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    let value = val.clone();
                    self.dispatch_tmp_read_inspect(callbacks, *tmp, &value);
                    Ok(value)
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }

            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                let value = self.registers.get(*offset, size, self.ctx);
                self.dispatch_reg_read_inspect(callbacks, *offset, size, &value);
                Ok(value)
            }

            IRExpr::Load { addr, ty, endness } => {
                self.eval_load(callbacks, addr, *ty, *endness, tyenv)
            }

            IRExpr::Unop { op, arg } => self.eval_unop(callbacks, *op, arg, tyenv),

            IRExpr::Binop { op, left, right } => {
                self.eval_binop(callbacks, *op, left, right, tyenv)
            }

            IRExpr::ITE {
                cond,
                iftrue,
                iffalse,
            } => self.eval_ite(callbacks, cond, iftrue, iffalse, tyenv),

            IRExpr::GetI { descr, ix, bias } => self.eval_geti(callbacks, *descr, ix, *bias, tyenv),

            IRExpr::Triop {
                op,
                arg1,
                arg2,
                arg3,
            } => self.eval_triop(callbacks, *op, arg1, arg2, arg3, tyenv),

            IRExpr::Qop {
                op,
                arg1,
                arg2,
                arg3,
                arg4,
            } => self.eval_qop(callbacks, *op, [arg1, arg2, arg3, arg4], tyenv),

            IRExpr::CCall { cee, retty, args } => {
                self.eval_ccall(callbacks, cee, *retty, args, tyenv)
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                // Request Python fallback instead of failing —
                // these special expressions require Python's VEX handling.
                // Reason string carries `VECRET_GSPTR_REASON` so
                // `exploration::run_loop` can bump a per-category counter
                // (angr-2iow prevalence measurement).
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "{} (special expr {:?} requires Python)",
                    crate::interpreter::VECRET_GSPTR_REASON,
                    expr
                )))
            }
        }
    }

    fn eval_ite(
        &mut self,
        callbacks: &PythonCallbacks,
        cond: &IRExpr,
        iftrue: &IRExpr,
        iffalse: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let cond_val = self.eval_expr_with_callbacks(callbacks, cond, tyenv)?;
        // Short-circuit: skip evaluating the dead branch when condition is concrete
        if let Some(v) = cond_val.as_u128() {
            return if v != 0 {
                self.eval_expr_with_callbacks(callbacks, iftrue, tyenv)
            } else {
                self.eval_expr_with_callbacks(callbacks, iffalse, tyenv)
            };
        }
        let true_val = self.eval_expr_with_callbacks(callbacks, iftrue, tyenv)?;
        let false_val = self.eval_expr_with_callbacks(callbacks, iffalse, tyenv)?;
        Ok(cond_val.ite(&true_val, &false_val, self.ctx))
    }

    /// Eagerly concretize a symbolic value via the solver and pin the choice
    /// with an equality constraint so a later solve cannot pick a different
    /// value (which would make the dependent read/arg inconsistent with the
    /// path constraints — unsound). Returns the pinned concrete as a u64, or
    /// None when the solver cannot produce a model (e.g. an UNSAT path). The
    /// caller maps None to its own error/fallback. This is the shared
    /// soundness primitive behind GetI/PutI index concretization and the
    /// dirty-call arg loops.
    pub(super) fn concretize_and_pin(&self, v: &RustBV) -> Option<u64> {
        let concrete = self.ctx.eval(v)?;
        let conc_bv = RustBV::concrete(concrete, v.width());
        let constraint = v.eq(&conc_bv, self.ctx);
        self.ctx.assume_true(&constraint);
        Some(concrete as u64)
    }

    fn eval_geti(
        &mut self,
        callbacks: &PythonCallbacks,
        descr: IRRegArray,
        ix: &IRExpr,
        bias: u32,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        // Evaluate the index expression
        let ix_val = self.eval_expr_with_callbacks(callbacks, ix, tyenv)?;
        let (offset, elem_size) = self.regarray_offset(&descr, &ix_val, bias, "GetI")?;

        // Read from the register file
        Ok(self.registers.get(offset, elem_size, self.ctx))
    }

    /// Resolve a VEX register-array access (`GetI`/`PutI`) to a flat
    /// register-file offset, returning `(offset, elem_size)`.
    ///
    /// Shared by `eval_geti` and the `IRStmt::PutI` arm in `statements.rs`:
    /// both need the same rotating-offset formula and the same
    /// concretize-and-pin treatment of a symbolic index, so a fix to either
    /// must land for both. `site` names the caller for the
    /// concretization-failure message.
    pub(super) fn regarray_offset(
        &self,
        descr: &IRRegArray,
        ix_val: &RustBV,
        bias: u32,
        site: &str,
    ) -> Result<(u32, u32), CbExecutionError> {
        // GetI/PutI require a concrete index to compute the register offset
        let idx = if let Some(idx) = ix_val.as_u64() {
            idx
        } else {
            // Symbolic index - concretize using the solver and pin the choice
            // with an equality constraint so a later solve cannot pick a
            // different index, which would make this register access
            // inconsistent with the path constraints (unsound).
            self.concretize_and_pin(ix_val).ok_or_else(|| {
                CbExecutionError::Unsupported(format!("{site} index concretization failed"))
            })?
        };

        // Calculate the rotating register offset:
        // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
        //
        // `nElems` is lifter-derived (deserialized from pyvex or marshalled
        // from the libVEX FFI) and is never validated on the way in, so a
        // malformed/hand-crafted IRSB can carry 0 here. Real libVEX only ever
        // emits a small nonzero nElems, so this is a trust-boundary check in
        // the same class as the LoadG/CAS/LLSC ones in `statements.rs`: fail
        // loud with `InvalidIR` rather than panic on divide-by-zero.
        if descr.nElems == 0 {
            return Err(CbExecutionError::InvalidIR(format!(
                "{site} register array descriptor has nElems == 0"
            )));
        }
        let elem_size = descr.elemTy.bytes();
        let index = ((idx as u32).wrapping_add(bias)) % descr.nElems;
        // A register-file offset is an identity, not an address: wrapping it
        // would silently name a *different* register, so refuse instead
        // (bd memory `invariant-overflow-fix-refuse-not-saturate-identities`).
        // Same trust boundary as the `nElems == 0` check above — `base`,
        // `nElems` and `elemTy` all arrive unvalidated from the lifter.
        let offset = index
            .checked_mul(elem_size)
            .and_then(|delta| descr.base.checked_add(delta))
            .ok_or_else(|| {
                CbExecutionError::InvalidIR(format!(
                    "{site} register array descriptor overflows the register file: \
                     base={} nElems={} elem_size={elem_size}",
                    descr.base, descr.nElems
                ))
            })?;
        Ok((offset, elem_size))
    }

    /// Evaluate an IR constant.
    fn eval_const(&self, c: &IRConst) -> RustBV {
        match c {
            IRConst::U1(v) => RustBV::concrete(*v as u128, 1),
            IRConst::U8(v) => RustBV::concrete(*v as u128, 8),
            IRConst::U16(v) => RustBV::concrete(*v as u128, 16),
            IRConst::U32(v) => RustBV::concrete(*v as u128, 32),
            IRConst::U64(v) => RustBV::concrete(*v as u128, 64),
            IRConst::U128(v) => RustBV::concrete(*v, 128),
            IRConst::F32(v) => RustBV::concrete(v.to_bits() as u128, 32),
            IRConst::F64(v) => RustBV::concrete(v.to_bits() as u128, 64),
            IRConst::V128(v) => RustBV::concrete(*v, 128),
            // A `Concrete` only ever stores its low 128 bits, so packing all four
            // u64 lanes into one would drop `v[2]`/`v[3]` *and* mistag the width as
            // 128 (every sibling arm tags its true IRType width). Assemble the two
            // halves as a `Concat` instead — the wide-concrete rule from bd memory
            // `invariant-concrete-bv-u128-16-byte-limit`, same shape as
            // `bv_utils::bytes_to_bv`. `a.concat(b)` puts `a` above `b`, so the
            // high half leads. Ico_V256's halves genuinely differ (see bd memory
            // `invariant-libvex-restricted-vector-consts`), so this is reachable.
            IRConst::V256(v) => {
                let high = RustBV::concrete(v[2] as u128 | ((v[3] as u128) << 64), 128);
                let low = RustBV::concrete(v[0] as u128 | ((v[1] as u128) << 64), 128);
                high.concat_no_ctx(&low)
            }
        }
    }

    pub(super) fn eval_expr_simple(
        &self,
        expr: &IRExpr,
        _tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),
            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }
            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }
            _ => Err(CbExecutionError::Unsupported(
                "complex expr in default exit".to_string(),
            )),
        }
    }
}

test_submod!("expressions_tests.rs" => expressions_tests);
