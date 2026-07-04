//! `state.inspect` breakpoint dispatch methods for `PythonCallbacks`.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).
//! Inherent-impl block only; the single `#[pymethods]` block stays in `mod.rs`.

use super::*;

impl PythonCallbacks {
    /// Fast O(1) check for whether an inspect event is enabled.
    /// Bit N = `crate::state::InspectEvent` variant N (MemRead=0, MemWrite=1, …).
    /// `event_bit` is taken as `u8` for ergonomics; values up to 31 are valid
    /// since the underlying bitmask is `AtomicU32` (widened in angr-lge2 to
    /// fit `expr` at bit 16).
    #[inline(always)]
    pub fn inspect_event_enabled(&self, event_bit: u8) -> bool {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
            & (1u32 << event_bit)
            != 0
    }

    /// Debug-only: read the raw bitmask. Used by eprintln traces.
    pub fn get_inspect_enabled_for_debug(&self) -> u32 {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Invoke the Python inspect mem_read callback.
    ///
    /// Caller is expected to gate this on `inspect_event_enabled(0)` for
    /// the common no-breakpoint case. Errors propagate so the engine can
    /// surface user-action failures rather than swallowing them.
    ///
    /// Returns the value the user's BP action left in `state.inspect.
    /// mem_read_expr` when it differs from what we passed in (value
    /// injection — angr-uy32); `None` when unchanged, no breakpoint fired,
    /// or no callback is registered. The caller converts a returned AST
    /// back to a `RustBV` and substitutes it for the loaded value.
    pub fn call_inspect_mem_read(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_mem_read.as_ref() {
                Some(cb) => cb,
                None => return Ok(None),
            };
            let value_obj: Py<PyAny> = match value_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
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
    pub fn call_inspect_mem_write(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_mem_write.as_ref() {
                Some(cb) => cb,
                None => return Ok(None),
            };
            let value_obj: Py<PyAny> = match value_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            let ret = cb.call1(py, (state_id, when, addr, size, value_obj, endness))?;
            if ret.is_none(py) {
                Ok(None)
            } else {
                Ok(Some(ret))
            }
        })
    }

    /// Invoke the Python inspect reg_read callback.
    pub fn call_inspect_reg_read(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_reg_read.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let value_obj: Py<PyAny> = match value_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, offset, size, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect reg_write callback.
    pub fn call_inspect_reg_write(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_reg_write.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let value_obj: Py<PyAny> = match value_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, offset, size, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect instruction callback.
    pub fn call_inspect_instruction(&self, state_id: i64, when: &str, addr: u64) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_instruction.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            cb.call1(py, (state_id, when, addr))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect irsb (block) callback.
    pub fn call_inspect_irsb(&self, state_id: i64, when: &str, addr: u64) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_irsb.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            cb.call1(py, (state_id, when, addr))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect exit (conditional branch) callback.
    pub fn call_inspect_exit(
        &self,
        state_id: i64,
        when: &str,
        target: u64,
        jumpkind: &str,
        guard_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_exit.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let guard_obj: Py<PyAny> = match guard_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, target, jumpkind, guard_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect call (function-entry) callback.
    pub fn call_inspect_call(
        &self,
        state_id: i64,
        when: &str,
        function_address: u64,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_call.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            cb.call1(py, (state_id, when, function_address))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect return (function-exit) callback.
    pub fn call_inspect_return(
        &self,
        state_id: i64,
        when: &str,
        function_address: u64,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_return.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            cb.call1(py, (state_id, when, function_address))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect tmp_read (VEX `RdTmp`) callback.
    pub fn call_inspect_tmp_read(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_tmp_read.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let value_obj: Py<PyAny> = match value_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, tmp_num, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect tmp_write (VEX `WrTmp`) callback.
    pub fn call_inspect_tmp_write(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_tmp_write.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let value_obj: Py<PyAny> = match value_ast {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, tmp_num, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect statement (per VEX IR statement) callback.
    pub fn call_inspect_statement(&self, state_id: i64, when: &str, stmt_idx: u32) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_statement.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            cb.call1(py, (state_id, when, stmt_idx))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect expr (per VEX IR expression eval) callback.
    /// Fires `when='after'` with the computed expression value reconstructed
    /// as a claripy AST. `expr` itself is passed as `None` (Rust IRExpr
    /// doesn't round-trip cleanly into a `pyvex.IRExpr`).
    pub fn call_inspect_expr(
        &self,
        state_id: i64,
        when: &str,
        expr_result: Option<&Py<PyAny>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_expr.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let value_obj: Py<PyAny> = match expr_result {
                Some(v) => v.clone_ref(py),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, value_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect address_concretization callback.
    /// `addr_ast` is the symbolic address AST (claripy reconstruction);
    /// `result` is the list of concrete addresses produced by the concretizer
    /// (None on `when='before'`).
    pub fn call_inspect_address_concretization(
        &self,
        state_id: i64,
        when: &str,
        action: &str,
        addr_ast: &Py<PyAny>,
        result: Option<Vec<u64>>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_address_concretization.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let addr_obj = addr_ast.clone_ref(py);
            let result_obj: Py<PyAny> = match result {
                Some(addrs) => pyo3::types::PyList::new(py, addrs)?.into_any().unbind(),
                None => py.None(),
            };
            cb.call1(py, (state_id, when, action, addr_obj, result_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect symbolic_variable callback.
    /// Fires `when='after'` when the Rust engine mints a fresh BVS for
    /// an unconstrained memory load. `expr_ast` is the claripy reconstruction
    /// of the freshly-minted BVS.
    pub fn call_inspect_symbolic_variable(
        &self,
        state_id: i64,
        when: &str,
        name: &str,
        size: u32,
        expr_ast: &Py<PyAny>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_symbolic_variable.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            let expr_obj = expr_ast.clone_ref(py);
            cb.call1(py, (state_id, when, name, size, expr_obj))?;
            Ok(())
        })
    }

    /// Invoke the Python inspect fork callback.
    /// Fires `when='after'` for each forked state created by the
    /// deferred-fork processing in `exploration/stepping.rs`.
    /// No attrs — the BP just sees the forked state's id.
    pub fn call_inspect_fork(&self, state_id: i64, when: &str) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter();
            let cb = match self.inspect_fork.as_ref() {
                Some(cb) => cb,
                None => return Ok(()),
            };
            cb.call1(py, (state_id, when))?;
            Ok(())
        })
    }
}
