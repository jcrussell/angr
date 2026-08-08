//! `state.inspect` breakpoint dispatch methods for `PythonCallbacks`.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).
//! Inherent-impl block only; the single `#[pymethods]` block stays in `mod.rs`.

use super::*;

/// Materialize an optional claripy AST argument for a breakpoint call.
///
/// Every `call_inspect_*` method that forwards a `value` / `guard` / `expr`
/// attr passes `None` through as Python `None` rather than omitting the
/// argument, so the Python endpoint sees a uniform arity.
fn opt_ast_or_none(py: Python<'_>, ast: Option<&Py<PyAny>>) -> Py<PyAny> {
    match ast {
        Some(v) => v.clone_ref(py),
        None => py.None(),
    }
}

impl PythonCallbacks {
    /// Shared prologue for every `call_inspect_*` method: attach to the
    /// interpreter, open a GIL-profiling span attributed to the inspect site,
    /// and invoke `f` with the registered callback.
    ///
    /// `absent` is returned when no callback is registered for the slot — the
    /// no-breakpoint case is not an error. Note the attach and the profiling
    /// guard deliberately happen *before* the slot check, so an unregistered
    /// slot still shows up in the `CallbackSite::Inspect` GIL accounting the
    /// way it did when each method open-coded this.
    fn with_inspect_cb<R>(
        &self,
        slot: Option<&Py<PyAny>>,
        absent: R,
        f: impl FnOnce(Python<'_>, &Py<PyAny>) -> PyResult<R>,
    ) -> PyResult<R> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::Inspect,
            );
            match slot {
                Some(cb) => f(py, cb),
                None => Ok(absent),
            }
        })
    }

    /// Fast O(1) check for whether an inspect event is enabled.
    ///
    /// Takes an [`InspectBit`] rather than a bare bit number so a dispatch
    /// site names the event it gates; the bit assignment lives in the single
    /// `inspect_events!` table (angr-12jjk.21).
    #[inline(always)]
    pub(crate) fn inspect_event_enabled(&self, event: InspectBit) -> bool {
        self.inspect_bit_enabled(event.bit())
    }

    /// Raw-bit form of [`Self::inspect_event_enabled`]. Only the enabled-mask
    /// plumbing (and its tests) should reach for this — engine code names its
    /// event via [`InspectBit`]. Bits up to 31 are valid since the underlying
    /// bitmask is `AtomicU32` (widened in angr-lge2 to fit `expr` at bit 16).
    #[inline(always)]
    fn inspect_bit_enabled(&self, event_bit: u8) -> bool {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
            & (1u32 << event_bit)
            != 0
    }

    /// Invoke the Python inspect mem_read callback.
    ///
    /// Caller is expected to gate this on `inspect_event_enabled(InspectBit::MemRead)` for
    /// the common no-breakpoint case. Errors propagate so the engine can
    /// surface user-action failures rather than swallowing them.
    ///
    /// Returns the value the user's BP action left in `state.inspect.
    /// mem_read_expr` when it differs from what we passed in (value
    /// injection — angr-uy32); `None` when unchanged, no breakpoint fired,
    /// or no callback is registered. The caller converts a returned AST
    /// back to a `RustBV` and substitutes it for the loaded value.
    pub(crate) fn call_inspect_mem_read(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.with_inspect_cb(self.inspect_mem_read.as_ref(), None, |py, cb| {
            let value_obj = opt_ast_or_none(py, value_ast);
            let ret = cb.call1(py, (state_id, when, addr, size, value_obj, endness))?;
            if ret.is_none(py) {
                Ok(None)
            } else {
                Ok(Some(ret))
            }
        })
    }

    /// Invoke the Python inspect mem_write callback. See `call_inspect_mem_read`.
    /// Fires `when='before'` (pre-store) — the returned AST, if any, is the
    /// user's `mem_write_expr` override to substitute for the stored value
    /// (angr-inh0) — and `when='after'` (post-store), where the return is
    /// informational only. See `dispatch_mem_write_inspect`.
    pub(crate) fn call_inspect_mem_write(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.with_inspect_cb(self.inspect_mem_write.as_ref(), None, |py, cb| {
            let value_obj = opt_ast_or_none(py, value_ast);
            let ret = cb.call1(py, (state_id, when, addr, size, value_obj, endness))?;
            if ret.is_none(py) {
                Ok(None)
            } else {
                Ok(Some(ret))
            }
        })
    }

    /// Invoke the Python inspect reg_read callback.
    pub(crate) fn call_inspect_reg_read(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_reg_read.as_ref(), (), |py, cb| {
            let value_obj = opt_ast_or_none(py, value_ast);
            cb.call1(py, (state_id, when, offset, size, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect reg_write callback.
    pub(crate) fn call_inspect_reg_write(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_reg_write.as_ref(), (), |py, cb| {
            let value_obj = opt_ast_or_none(py, value_ast);
            cb.call1(py, (state_id, when, offset, size, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect instruction callback.
    pub(crate) fn call_inspect_instruction(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_instruction.as_ref(), (), |py, cb| {
            cb.call1(py, (state_id, when, addr))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect irsb (block) callback.
    pub(crate) fn call_inspect_irsb(&self, state_id: i64, when: &str, addr: u64) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_irsb.as_ref(), (), |py, cb| {
            cb.call1(py, (state_id, when, addr))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect exit (conditional branch) callback.
    pub(crate) fn call_inspect_exit(
        &self,
        state_id: i64,
        when: &str,
        target: u64,
        jumpkind: &str,
        guard_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_exit.as_ref(), (), |py, cb| {
            let guard_obj = opt_ast_or_none(py, guard_ast);
            cb.call1(py, (state_id, when, target, jumpkind, guard_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect call (function-entry) callback.
    pub(crate) fn call_inspect_call(
        &self,
        state_id: i64,
        when: &str,
        function_address: u64,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_call.as_ref(), (), |py, cb| {
            cb.call1(py, (state_id, when, function_address))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect return (function-exit) callback.
    pub(crate) fn call_inspect_return(
        &self,
        state_id: i64,
        when: &str,
        function_address: u64,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_return.as_ref(), (), |py, cb| {
            cb.call1(py, (state_id, when, function_address))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect tmp_read (VEX `RdTmp`) callback.
    pub(crate) fn call_inspect_tmp_read(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_tmp_read.as_ref(), (), |py, cb| {
            let value_obj = opt_ast_or_none(py, value_ast);
            cb.call1(py, (state_id, when, tmp_num, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect tmp_write (VEX `WrTmp`) callback.
    pub(crate) fn call_inspect_tmp_write(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_tmp_write.as_ref(), (), |py, cb| {
            let value_obj = opt_ast_or_none(py, value_ast);
            cb.call1(py, (state_id, when, tmp_num, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect statement (per VEX IR statement) callback.
    pub(crate) fn call_inspect_statement(
        &self,
        state_id: i64,
        when: &str,
        stmt_idx: u32,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_statement.as_ref(), (), |py, cb| {
            cb.call1(py, (state_id, when, stmt_idx))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect expr (per VEX IR expression eval) callback.
    /// Fires `when='after'` with the computed expression value reconstructed
    /// as a claripy AST. `expr` itself is passed as `None` (Rust IRExpr
    /// doesn't round-trip cleanly into a `pyvex.IRExpr`).
    pub(crate) fn call_inspect_expr(
        &self,
        state_id: i64,
        when: &str,
        expr_result: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_expr.as_ref(), (), |py, cb| {
            let value_obj = opt_ast_or_none(py, expr_result);
            cb.call1(py, (state_id, when, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect address_concretization callback.
    /// `addr_ast` is the symbolic address AST (claripy reconstruction);
    /// `result` is the list of concrete addresses produced by the concretizer
    /// (None on `when='before'`).
    pub(crate) fn call_inspect_address_concretization(
        &self,
        state_id: i64,
        when: &str,
        action: &str,
        addr_ast: &Py<PyAny>,
        result: Option<Vec<u64>>,
    ) -> PyResult<()> {
        self.with_inspect_cb(
            self.inspect_address_concretization.as_ref(),
            (),
            |py, cb| {
                let addr_obj = addr_ast.clone_ref(py);
                let result_obj: Py<PyAny> = match result {
                    Some(addrs) => pyo3::types::PyList::new(py, addrs)?.into_any().unbind(),
                    None => py.None(),
                };
                cb.call1(py, (state_id, when, action, addr_obj, result_obj))?;
                Ok(())
            },
        )
    }

    /// Invoke the Python inspect symbolic_variable callback.
    /// Fires `when='after'` when the Rust engine mints a fresh BVS for
    /// an unconstrained memory load. `expr_ast` is the claripy reconstruction
    /// of the freshly-minted BVS.
    pub(crate) fn call_inspect_symbolic_variable(
        &self,
        state_id: i64,
        when: &str,
        name: &str,
        size: u32,
        expr_ast: &Py<PyAny>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_symbolic_variable.as_ref(), (), |py, cb| {
            let expr_obj = expr_ast.clone_ref(py);
            cb.call1(py, (state_id, when, name, size, expr_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect fork callback.
    /// Fires `when='after'` for each forked state created by the
    /// deferred-fork processing in `exploration/stepping.rs`.
    /// No attrs — the BP just sees the forked state's id.
    pub(crate) fn call_inspect_fork(&self, state_id: i64, when: &str) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_fork.as_ref(), (), |py, cb| {
            cb.call1(py, (state_id, when))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect constraints callback for a natively-added
    /// branch guard (angr-op0dn.14.4.1).
    ///
    /// `guard` is the 1-bit condition BV as passed to `assume_true` /
    /// `assume_false`; `is_true` is that polarity. The added constraint is
    /// materialized for Python exactly the way `_export_state_constraints`
    /// does it — `rustbv_to_claripy(guard)`, wrapped in `claripy.Not(..)`
    /// when the polarity is false — and handed to the BP as a one-element
    /// `added_constraints` list.
    ///
    /// Caller gates on `inspect_event_enabled(InspectBit::Constraints)`. A claripy import or
    /// export failure drops the event (returns `Ok(())`): a BP that cannot
    /// be materialized must not halt exploration.
    pub(crate) fn call_inspect_constraints(
        &self,
        state_id: i64,
        when: &str,
        guard: &crate::symbolic::RustBV,
        is_true: bool,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_constraints.as_ref(), (), |py, cb| {
            let claripy = py.import("claripy")?;
            let constraint = match crate::claripy_bridge::assumed_guard_to_claripy(
                py,
                guard,
                claripy.as_any(),
                is_true,
            ) {
                Ok(c) => c,
                // SILENT(cat-b): a guard that will not export to claripy costs
                // the user this one `constraints` BP fire and nothing else —
                // the guard itself is already lowered into the state's `RustBV`
                // constraint set, and this native path never honors a BP's
                // mutated `added_constraints` anyway (see the doc comment), so
                // the analysis result is unaffected. Loud failure is the wrong
                // trade: an observability hook must not abort exploration.
                // Stays at `debug!` rather than `warn!` because the fork-guard
                // add site is hot and a systematically-unexportable guard shape
                // would repeat the message per fork.
                Err(e) => {
                    log::debug!("constraints inspect export failed (state {state_id}): {e}");
                    return Ok(());
                }
            };
            cb.call1(py, (state_id, when, vec![constraint]))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect vex_lift callback for a block served by the
    /// native in-process libVEX lifter (angr-op0dn.14.4.2).
    ///
    /// `size` is `None` on the BEFORE fire and the lifted IRSB's byte size on
    /// the AFTER fire; `buff` carries the bytes handed to libVEX (BEFORE only),
    /// mirroring what `_cb_lift_block` passes on the Python-lift path.
    /// `state_id` is `-1` there too — a lift is state-independent, so the
    /// Python endpoint attributes it to a representative active state.
    ///
    /// Caller gates on `inspect_event_enabled(InspectBit::VexLift)`. That
    /// production caller lives in the native libVEX lift path, so in a build
    /// without `libvex-ffi` (default-OFF in Cargo.toml, default-ON via
    /// setup.py) the only way in is `py_call_inspect_vex_lift`, the test entry
    /// point in `callbacks/mod.rs`.
    pub(crate) fn call_inspect_vex_lift(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: Option<u32>,
        buff: Option<&[u8]>,
    ) -> PyResult<()> {
        self.with_inspect_cb(self.inspect_vex_lift.as_ref(), (), |py, cb| {
            let buff_obj = buff.map(|b| PyBytes::new(py, b));
            cb.call1(py, (state_id, when, addr, size, buff_obj))?;
            Ok(())
        })
    }
}

#[cfg(test)]
#[path = "inspect_tests.rs"]
mod inspect_tests;
