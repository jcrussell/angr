//! Pending callback API methods.
//!
//! Bodies for the cluster of pyclass-exposed methods that read or mutate the
//! `pending_callback` state — registers, memory, history, jumpkind, dirty
//! pages, constraints, snapshots, and solver fork/borrow helpers — plus the
//! related skip-hook stack. The pyclass-facing thin wrappers live in `mod.rs`
//! and forward to the `pub(crate)` bodies in this module.
//!
//! PyO3 0.27.2 in this project does not enable `multiple-pymethods`, so each
//! pyclass is limited to a single `#[pymethods]` impl block — see
//! `invariant-pyo3-single-pymethods-impl`. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` extension-impl
//! pattern used elsewhere in `exploration/`.

use super::*;

impl RustExplorationManager {
    // -------------------------------------------------------------------------
    // Pending state initialization / memory mapping
    // -------------------------------------------------------------------------

    pub(crate) fn _set_pending_state_pc(&mut self, pc: u64) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending.state.set_pc(pc);
            Ok(())
        })
    }

    pub(crate) fn _pending_state_map_memory(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending
                .state
                .map_memory_data(addr, data, Permission::from_bits(permissions));
            Ok(())
        })
    }

    pub(crate) fn _active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        if let Some(stash) = self.sm.get_mut(STASH_ACTIVE) {
            for state in stash.iter_mut() {
                state.map_memory_data(addr, data, Permission::from_bits(permissions));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Branch condition / register inspection
    // -------------------------------------------------------------------------

    pub(crate) fn _get_pending_branch_condition(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_pending(|pending| {
            let condition_id = match &pending.reason {
                CallbackReason::SymbolicBranch { condition_id, .. } => *condition_id,
                _ => {
                    return Err(PyValueError::new_err(
                        "pending callback is not a symbolic branch",
                    ));
                }
            };

            let condition = pending
                .stored_conditions
                .get(&condition_id)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "condition {} not found in stored_conditions",
                        condition_id
                    ))
                })?;

            let claripy = py.import("claripy")?;
            rustbv_to_claripy(py, condition, claripy.as_any())
                .map_err(|e| PyRuntimeError::new_err(format!("failed to convert condition: {}", e)))
        })
    }

    pub(crate) fn _get_pending_register(&self, name: &str) -> PyResult<Option<u128>> {
        self.with_pending(|pending| {
            pending
                .state
                .get_register(name)
                .map(|bv| bv.as_u128())
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))
        })
    }

    pub(crate) fn _get_pending_register_ast(
        &self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        self.with_pending(|pending| {
            let bv = pending
                .state
                .get_register(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let claripy = py.import("claripy")?;
            rustbv_to_claripy(py, &bv, claripy.as_any())
                .map_err(|e| PyRuntimeError::new_err(format!("register conversion: {}", e)))
        })
    }

    pub(crate) fn _get_pending_history(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| Ok(pending.state.history().to_vec()))
    }

    pub(crate) fn _get_pending_jumpkind(&self) -> PyResult<String> {
        self.with_pending(|pending| {
            Ok(pending
                .jumpkind
                .clone()
                .unwrap_or_else(|| "Ijk_Boring".to_string()))
        })
    }

    pub(crate) fn _get_pending_history_and_jumpkind(&self) -> PyResult<(Vec<u64>, String)> {
        self.with_pending(|pending| {
            let history = pending.state.history().to_vec();
            let jumpkind = pending
                .jumpkind
                .clone()
                .unwrap_or_else(|| "Ijk_Boring".to_string());
            Ok((history, jumpkind))
        })
    }

    pub(crate) fn _set_pending_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let size = pending
                .state
                .arch()
                .register_size(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let bv = crate::symbolic::RustBV::concrete(value, size * 8);
            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "failed to set register: {}",
                    name
                )))
            }
        })
    }

    pub(crate) fn _set_pending_register_symbolic(
        &mut self,
        name: &str,
        handle_id: u64,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let bv = if let Some(ref solver) = pending.solver_ctx {
                solver
                    .symbol_table()
                    .get(handle_id)
                    .ok_or_else(|| crate::solver::invalid_handle_id(&[handle_id]))?
            } else {
                return Err(PyRuntimeError::new_err(
                    "no solver context in pending state",
                ));
            };

            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "failed to set register: {}",
                    name
                )))
            }
        })
    }

    pub(crate) fn _set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &sym_ctx;

            let bv = claripy_to_rustbv(py, ast, ctx_ref)
                .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;

            drop(sym_ctx);

            if pending.state.set_register(reg_name, bv) {
                log::debug!("Set symbolic register {} from claripy AST", reg_name);
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "failed to set register: {}",
                    reg_name
                )))
            }
        })
    }

    // -------------------------------------------------------------------------
    // Symbolic memory import
    // -------------------------------------------------------------------------

    pub(crate) fn _import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let sym_ctx = solver_ref.borrow();
            let bv = claripy_to_rustbv(py, ast, &sym_ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion: {}", e)))?;
            drop(sym_ctx);
            state.memory_mut().import_symbolic_value(addr, bv, None);
            Ok(())
        })
    }

    pub(crate) fn _import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let bv = claripy_to_rustbv(py, ast, &sym_ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;
            drop(sym_ctx);

            pending
                .state
                .memory_mut()
                .import_symbolic_value(addr, bv, None);
            log::debug!("Imported symbolic memory at 0x{:x}", addr);
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Pending memory get/set (high-level API, used by Python init)
    // -------------------------------------------------------------------------

    pub(crate) fn _get_pending_memory(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self.with_pending(|pending| {
            let bv = pending
                .state
                .memory_load(addr, size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let value = bv.as_u128().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "pending memory at 0x{addr:x} is symbolic; cannot convert to concrete bytes"
                ))
            })?;
            Ok(super::helpers::u128_to_le_bytes(value, size as usize))
        })
    }

    pub(crate) fn _set_pending_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        // angr-5aj8: route through the shared 16-byte-chunk helper. The prior
        // single-pack implementation here silently truncated/overflowed for
        // data.len() > 16 (RustBV::concrete is u128-backed); chunking fixes it.
        self.with_pending_mut(|pending| {
            super::helpers::store_concrete_bytes_chunked(addr, data, |chunk_addr, bv| {
                pending
                    .state
                    .memory_store(chunk_addr, bv)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
        })
    }

    pub(crate) fn _get_pending_dirty_pages(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| Ok(pending.state.get_dirty_pages()))
    }

    pub(crate) fn _clear_pending_dirty_tracking(&mut self) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending.state.clear_dirty_pages();
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Pending constraints / handles / snapshots
    // -------------------------------------------------------------------------

    pub(crate) fn _export_pending_constraints(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.with_pending(|pending| {
            let mut result = Vec::new();

            let claripy_mod = py.import("claripy")?;

            for rustbv in pending.stored_conditions.values() {
                match rustbv_to_claripy(py, rustbv, &claripy_mod) {
                    Ok(ast) => {
                        result.push(ast);
                    }
                    Err(e) => {
                        log::debug!("Could not convert stored condition to claripy: {}", e);
                    }
                }
            }

            log::debug!("Exported {} pending constraints", result.len());
            Ok(result)
        })
    }

    pub(crate) fn _get_active_handle_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        if let Some(ref pending) = self.pending_callback {
            for id in pending.stored_conditions.keys() {
                ids.push(*id);
            }
            for fork in &pending.deferred_forks {
                ids.push(fork.condition_id);
            }
        }
        ids
    }

    pub(crate) fn _export_pending_state(&self) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self.with_pending(|pending| Ok(pending.state.export_full()))
    }

    pub(crate) fn _get_pending_root_state_id(&self) -> PyResult<Option<u64>> {
        self.with_pending(|pending| {
            let state_id = pending.state.state_id();
            Ok(self.sm.roots().get(&state_id).copied())
        })
    }

    pub(crate) fn _get_pending_ancestry(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| {
            let mut ancestry = vec![pending.state.state_id()];

            if let Some(parent_id) = pending.state.parent_id() {
                ancestry.push(parent_id);
            }

            let state_id = pending.state.state_id();
            if let Some(&root_id) = self.sm.roots().get(&state_id)
                && !ancestry.contains(&root_id)
            {
                ancestry.push(root_id);
            }

            Ok(ancestry)
        })
    }

    pub(crate) fn _export_callback_bundle<'py>(
        &self,
        py: Python<'py>,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.with_pending(|pending| {
            let dict = PyDict::new(py);

            let reg_dict = PyDict::new(py);
            for name in &register_names {
                match pending.state.get_register(name) {
                    Some(bv) => {
                        if let Some(val) = bv.as_u128() {
                            reg_dict.set_item(name, val)?;
                        } else {
                            reg_dict.set_item(name, py.None())?;
                        }
                    }
                    None => {
                        reg_dict.set_item(name, py.None())?;
                    }
                }
            }
            dict.set_item("registers", reg_dict)?;

            let solver_ref = pending.state.solver();
            if shared_solver {
                let rust_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
                let constraint_count = rust_ctx.num_constraints();
                dict.set_item("solver", Py::new(py, rust_ctx)?)?;
                dict.set_item("constraint_count", constraint_count)?;
            } else {
                let forked_ctx = solver_ref.borrow().fork();
                let rust_ctx = RustSolverContext::from_sym_context(forked_ctx);
                let constraint_count = rust_ctx.num_constraints();
                dict.set_item("solver", Py::new(py, rust_ctx)?)?;
                dict.set_item("constraint_count", constraint_count)?;
            }

            dict.set_item("history", pending.state.history().to_vec())?;

            dict.set_item(
                "jumpkind",
                pending
                    .jumpkind
                    .clone()
                    .unwrap_or_else(|| "Ijk_Boring".to_string()),
            )?;

            dict.set_item("stdout", pending.state.stdout_buffer().to_vec())?;

            Ok(dict)
        })
    }

    // -------------------------------------------------------------------------
    // Pending solver fork / borrow / constraint sync
    // -------------------------------------------------------------------------

    pub(crate) fn _fork_pending_solver(&self) -> PyResult<RustSolverContext> {
        self.with_pending(|pending| {
            let solver_ref = pending.state.solver();
            let forked_ctx = solver_ref.borrow().fork();
            Ok(RustSolverContext::from_sym_context(forked_ctx))
        })
    }

    pub(crate) fn _borrow_pending_solver(&self) -> PyResult<RustSolverContext> {
        self.with_pending(|pending| {
            let solver_rc = pending.state.solver().clone();
            Ok(RustSolverContext::from_shared_sym_context(solver_rc))
        })
    }

    pub(crate) fn _add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        use pyo3::types::PyListMethods;

        self.with_pending_mut(|pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &sym_ctx;

            let len = constraints.len();
            for i in 0..len {
                if let Ok(constraint) = constraints.get_item(i) {
                    if let Ok(bv) = claripy_to_rustbv(py, &constraint, ctx_ref) {
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                            } else {
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                            }
                        }
                    } else {
                        log::debug!("Could not convert constraint {} from Python", i);
                    }
                }
            }
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Pending memory low-level (page / address access used by callbacks)
    // -------------------------------------------------------------------------

    pub(crate) fn _get_pending_mapped_pages(&self) -> PyResult<Vec<u64>> {
        self.with_pending(|pending| {
            Ok(pending
                .state
                .memory()
                .pages()
                .keys()
                .map(|&pn| pn << 12)
                .collect())
        })
    }

    pub(crate) fn _pending_memory_load_page(&self, page_addr: u64) -> PyResult<Vec<u8>> {
        self.with_pending(|pending| {
            pending
                .state
                .memory()
                .load_page_concrete(page_addr)
                .map_err(|e| PyValueError::new_err(format!("page load failed: {}", e)))
        })
    }

    /// Return every multi-byte symbolic object whose base address lies on
    /// `page_addr`'s page, as `(addr, claripy_ast)` pairs.
    ///
    /// Used by Python's `_create_state_for_callback` to replay symbolic
    /// stores that native SimProcedures (NativeRead, etc.) made into the
    /// cached Python SimState after `_install_rust_memory_proxy` has
    /// already written concrete defaults from the SP page.
    pub(crate) fn _pending_memory_load_symbolic_page<'py>(
        &self,
        py: Python<'py>,
        page_addr: u64,
    ) -> PyResult<Vec<(u64, Py<PyAny>)>> {
        self.with_pending(|pending| {
            let claripy_mod = py.import("claripy")?;
            let target_page = page_addr >> 12;
            let mut out = Vec::new();
            for (addr, bv) in pending.state.memory().symbolic_objects_iter() {
                if addr.page_num() != target_page {
                    continue;
                }
                match rustbv_to_claripy(py, bv, claripy_mod.as_any()) {
                    Ok(ast) => out.push((addr.raw(), ast)),
                    Err(e) => {
                        log::debug!(
                            "symbolic-page replay: failed to convert AST at 0x{:x}: {}",
                            addr.raw(),
                            e
                        );
                    }
                }
            }
            Ok(out)
        })
    }

    pub(crate) fn _pending_memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self.with_pending(|pending| {
            let solver_ref = pending.state.solver();
            let ctx = solver_ref.borrow();
            match pending.state.memory().load_concrete(addr, size, &ctx) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u128() {
                        let byte_count = (size as usize).min(16);
                        Ok(val.to_le_bytes()[..byte_count].to_vec())
                    } else {
                        let solver = pending.state.solver();
                        let ctx = solver.borrow();
                        if let Some(val) = ctx.eval(&bv) {
                            let byte_count = (size as usize).min(16);
                            Ok(val.to_le_bytes()[..byte_count].to_vec())
                        } else {
                            Ok(vec![0u8; size as usize])
                        }
                    }
                }
                Err(_) => Ok(vec![0u8; size as usize]),
            }
        })
    }

    pub(crate) fn _pending_memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        // angr-5aj8: split into 16-byte chunks. RustBV::Concrete is u128-backed;
        // packing more than 16 bytes (the prior implementation silently truncated
        // and constructed an oversized concrete BV) leaves store_concrete to emit
        // a 16-byte-cycle pattern across the entire claimed width. Use the safe
        // pattern from RustSimState::apply_changes.
        self.with_pending_mut(|pending| {
            super::helpers::store_concrete_bytes_chunked(addr, data, |chunk_addr, bv| {
                pending
                    .state
                    .memory_mut()
                    .store_concrete(chunk_addr, bv)
                    .map_err(|e| PyRuntimeError::new_err(format!("memory store error: {}", e)))
            })
        })
    }

    pub(crate) fn _pending_memory_map_data(
        &mut self,
        addr: u64,
        data: &[u8],
        perm: u8,
    ) -> PyResult<()> {
        self.with_pending_mut(|pending| {
            pending
                .state
                .map_memory_data(addr, data, crate::memory::Permission::from_bits(perm));
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Skip-hook stack (zero-length hook anti-loop)
    // -------------------------------------------------------------------------

    pub(crate) fn _set_skip_hook_addr(&mut self, addr: u64) {
        let expiry = self.steps + 2;
        self.skip_hook_stack.push((addr, expiry));
        log::debug!("Added skip hook 0x{:x} with expiry step {}", addr, expiry);
    }

    pub(crate) fn _clear_skip_hook_addr(&mut self) {
        self.skip_hook_stack.clear();
    }

    pub(crate) fn _clear_skip_hook_for_addr(&mut self, addr: u64) {
        self.skip_hook_stack.retain(|&(a, _)| a != addr);
    }
}
