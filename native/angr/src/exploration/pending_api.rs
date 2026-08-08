//! Pending callback API methods.
//!
//! Bodies for the cluster of pyclass-exposed methods that read or mutate the
//! `pending_callback` state — registers, memory, history, jumpkind, dirty
//! pages, constraints, snapshots, and solver fork/borrow helpers — plus the
//! related skip-hook stack. The pyclass-facing thin wrappers live in `mod.rs`
//! and forward to the `pub(crate)` bodies in this module.
//!
//! The extension-impl split here predates PyO3's `multiple-pymethods`
//! feature, which is now enabled (angr-9ke6b.50, see
//! `invariant-pyo3-multiple-pymethods-enabled`), so the single-block rule no
//! longer forces it. It is kept as a style choice: thin `#[pyo3]` wrappers
//! stay next to their siblings while the substantial bodies live here. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` extension-impl
//! pattern used elsewhere in `exploration/`.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::errors::MapPyErr;
use crate::vex::ir::JumpKind;

impl RustExplorationManager {
    // -------------------------------------------------------------------------
    // Pending state initialization / memory mapping
    // -------------------------------------------------------------------------

    pub(crate) fn _set_pending_state_pc(&mut self, state_id: u64, pc: u64) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
            pending.state.set_pc(pc);
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

    pub(crate) fn _get_pending_branch_condition(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Py<PyAny>> {
        self.with_pending(state_id, |pending| {
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
                        "condition {condition_id} not found in stored_conditions"
                    ))
                })?;

            let claripy = py.import("claripy")?;
            rustbv_to_claripy(py, condition, claripy.as_any())
                .map_err(|e| ast_export_err(&format!("condition {condition_id}"), e))
        })
    }

    /// Read a register from a parked pending callback's state as a concrete
    /// `u128`.
    ///
    /// An unknown register name is a hard `PyValueError`, so `Ok(None)` means
    /// exactly one thing: the register exists but holds a symbolic value.
    /// That is the opposite of the mirrored `_get_state_register`, which
    /// folds both cases into `None`; the rationale for the split lives in
    /// `state_api::_get_state_register`'s doc comment (angr-sqfj8.56). Short
    /// version: this API's callers are our own callback-arg plumbing passing
    /// arch-derived names, so a miss is a bug — not an angr register that
    /// Rust's `RegisterFile` declines to model.
    pub(crate) fn _get_pending_register(
        &self,
        state_id: u64,
        name: &str,
    ) -> PyResult<Option<u128>> {
        self.with_pending(state_id, |pending| {
            pending
                .state
                .get_register(name)
                .map(|bv| bv.as_u128())
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))
        })
    }

    pub(crate) fn _get_pending_register_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        self.with_pending(state_id, |pending| {
            let bv = pending
                .state
                .get_register(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))?;
            let claripy = py.import("claripy")?;
            rustbv_to_claripy(py, &bv, claripy.as_any())
                .map_err(|e| PyRuntimeError::new_err(format!("register conversion: {e}")))
        })
    }

    pub(crate) fn _get_pending_history_and_jumpkind(
        &self,
        state_id: u64,
    ) -> PyResult<(Vec<u64>, String)> {
        self.with_pending(state_id, |pending| {
            let history = pending.state.history().iter().copied().collect::<Vec<_>>();
            let jumpkind = pending
                .jumpkind
                .clone()
                .unwrap_or_else(|| JumpKind::Boring.ijk_name().to_string());
            Ok((history, jumpkind))
        })
    }

    pub(crate) fn _set_pending_register(
        &mut self,
        state_id: u64,
        name: &str,
        value: u128,
    ) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
            let size = pending
                .state
                .arch()
                .register_size(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))?;
            let bv = crate::symbolic::RustBV::concrete(value, size * 8);
            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "failed to set register: {name}"
                )))
            }
        })
    }

    pub(crate) fn _set_pending_register_symbolic(
        &mut self,
        state_id: u64,
        name: &str,
        handle_id: u64,
    ) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
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
                    "failed to set register: {name}"
                )))
            }
        })
    }

    pub(crate) fn _set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &sym_ctx;

            let bv = claripy_to_rustbv(py, ast, ctx_ref)
                .map_err(|e| ast_import_err(&format!("register {reg_name}"), e))?;

            drop(sym_ctx);

            if pending.state.set_register(reg_name, bv) {
                log::debug!("Set symbolic register {reg_name} from claripy AST");
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "failed to set register: {reg_name}"
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
                .map_err(|e| ast_import_err(&format!("memory 0x{addr:x}"), e))?;
            drop(sym_ctx);
            state
                .memory_mut()
                .import_symbolic_value(addr, bv, None)
                .py_value_err()
        })
    }

    // -------------------------------------------------------------------------
    // Symbolic file content export (angr-0xyq2 Phase 3)
    // -------------------------------------------------------------------------

    pub(crate) fn _set_fs_cwd(&mut self, state_id: u64, cwd: &str) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            state.file_system().set_cwd(cwd.as_bytes().to_vec());
            Ok(())
        })
    }

    pub(crate) fn _register_file_content(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        path: &str,
        byte_asts: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            let bytes = import_byte_asts(py, state, byte_asts)?;
            state.file_system().register_file_content(path, bytes);
            Ok(())
        })
    }

    /// Attach `byte_asts` to fd 0 as bounded symbolic content (angr-mb09c).
    /// Seed-time channel for a Python-filled `posix.stdin.content`; see
    /// `FileSystem::set_fd_content_sym` for why the path registry cannot
    /// reach an already-open fd.
    pub(crate) fn _seed_stdin_content(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        byte_asts: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            let bytes = import_byte_asts(py, state, byte_asts)?;
            state.file_system().set_fd_content_sym(0, bytes);
            Ok(())
        })
    }

    pub(crate) fn _get_demoted_paths(&self, state_id: u64) -> PyResult<Vec<String>> {
        self.with_state(state_id, |state| {
            Ok(state.file_system_ref().demoted_paths())
        })
    }

    pub(crate) fn _demote_file_path(&mut self, state_id: u64, path: &str) -> PyResult<bool> {
        self.with_state_mut(state_id, |state| Ok(state.file_system().demote_path(path)))
    }

    pub(crate) fn _import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let bv = claripy_to_rustbv(py, ast, &sym_ctx)
                .map_err(|e| ast_import_err(&format!("memory 0x{addr:x}"), e))?;
            drop(sym_ctx);

            pending
                .state
                .memory_mut()
                .import_symbolic_value(addr, bv, None)
                .py_value_err()?;
            log::debug!("Imported symbolic memory at 0x{addr:x}");
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Pending memory get/set (high-level API, used by Python init)
    // -------------------------------------------------------------------------

    /// Concrete bytes for `size` bytes of memory on a pending callback state.
    ///
    /// Errors (rather than concretizing) when any chunk of the range is
    /// symbolic — angr-04tw3.2 deliberately keeps this path eval-free, which is
    /// the one behavioral difference from `_get_state_memory`.
    pub(crate) fn _get_pending_memory(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Vec<u8>> {
        self.with_pending(state_id, |pending| {
            crate::symbolic::load_concrete_bytes_chunked(addr, size, |a, n| {
                let bv = pending.state.memory_load(a, n).py_value_err()?;

                let value = bv.as_u128().ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "pending memory at 0x{a:x} is symbolic; cannot convert to concrete bytes"
                    ))
                })?;
                Ok(crate::symbolic::u128_to_le_bytes(value, n as usize))
            })
        })
    }

    pub(crate) fn _get_pending_dirty_pages(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self.with_pending(state_id, |pending| Ok(pending.state.get_dirty_pages()))
    }

    pub(crate) fn _clear_pending_dirty_tracking(&mut self, state_id: u64) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
            pending.state.clear_dirty_pages();
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Pending constraints / handles / snapshots
    // -------------------------------------------------------------------------

    pub(crate) fn _export_pending_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self.with_pending(state_id, |pending| {
            let mut result = Vec::new();

            let claripy_mod = py.import("claripy")?;

            for rustbv in pending.stored_conditions.values() {
                match rustbv_to_claripy(py, rustbv, &claripy_mod) {
                    Ok(ast) => {
                        result.push(ast);
                    }
                    Err(e) => {
                        log::debug!("Could not convert stored condition to claripy: {e}");
                    }
                }
            }

            log::debug!("Exported {} pending constraints", result.len());
            Ok(result)
        })
    }

    pub(crate) fn _get_active_handle_ids(&self) -> Vec<u64> {
        // Union the actively-referenced handle ids across ALL pending callbacks
        // (not keyed by state_id): the sole caller `get_active_handle_ids` has no
        // state_id to pass, and the result feeds AST-handle-cache eviction
        // protection, which must keep every live handle regardless of which
        // pending state owns it. With one entry on the single-threaded path this
        // is byte-identical to the former single-slot behaviour.
        let mut ids = Vec::new();
        for pending in self.pending_callbacks.values() {
            for id in pending.stored_conditions.keys() {
                ids.push(*id);
            }
            for fork in &pending.deferred_forks {
                ids.push(fork.condition_id);
            }
        }
        ids
    }

    /// angr-sqfj8.27: uses `flush_and_export_full` (not the unflushed
    /// `export_full`), matching the fixed `_export_state`/`_export_stash`
    /// sites (angr-9ke6b.101) — otherwise Multi-covered bytes installed by
    /// the lazy symbolic-address store path are silently absent from the
    /// exported pending state.
    pub(crate) fn _export_pending_state(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self.with_pending_mut(state_id, |pending| {
            Ok(pending.state.flush_and_export_full())
        })
    }

    pub(crate) fn _get_pending_root_state_id(&self, state_id: u64) -> PyResult<Option<u64>> {
        self.with_pending(state_id, |pending| {
            let self_id = pending.state.state_id();
            Ok(self.sm.roots().get(&self_id).copied())
        })
    }

    /// Look up the parent id of an arbitrary state the manager still holds.
    ///
    /// Only two places hold states: `pending_callbacks` and the stashes, and
    /// `find_state` already searches both (pending first, as a direct hash
    /// lookup). A state that is in neither has been consumed by a fork or
    /// dropped, so its parent link is unrecoverable — hence the best-effort
    /// walk in `_get_pending_ancestry`.
    fn _parent_of(&self, state_id: u64) -> Option<u64> {
        self.find_state(state_id).and_then(RustSimState::parent_id)
    }

    /// Depth cap for the ancestry walk. Fork chains this deep do not occur in
    /// practice; the cap exists so a corrupted parent link cannot make the walk
    /// scan every stash an unbounded number of times.
    const MAX_ANCESTRY_DEPTH: usize = 64;

    pub(crate) fn _get_pending_ancestry(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self.with_pending(state_id, |pending| {
            let self_id = pending.state.state_id();
            let mut ancestry = vec![self_id];

            // Walk the parent chain transitively rather than a single hop, so a
            // >=3-level fork chain does not drop its intermediate ancestors (the
            // Python cache walks in rust_callback_dispatch.py / rust_state_sync.py
            // key on exactly these ids). Best-effort: the walk stops at the first
            // ancestor the manager no longer holds.
            let mut next = pending.state.parent_id();
            while let Some(pid) = next {
                if ancestry.contains(&pid) || ancestry.len() >= Self::MAX_ANCESTRY_DEPTH {
                    break;
                }
                ancestry.push(pid);
                next = self._parent_of(pid);
            }

            if let Some(&root_id) = self.sm.roots().get(&self_id)
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
        state_id: u64,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.with_pending(state_id, |pending| {
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

            dict.set_item(
                "history",
                pending.state.history().iter().copied().collect::<Vec<_>>(),
            )?;

            dict.set_item(
                "jumpkind",
                pending
                    .jumpkind
                    .clone()
                    .unwrap_or_else(|| JumpKind::Boring.ijk_name().to_string()),
            )?;

            dict.set_item("stdout", pending.state.stdout_buffer().to_vec())?;

            Ok(dict)
        })
    }

    // -------------------------------------------------------------------------
    // Pending solver fork / borrow / constraint sync
    // -------------------------------------------------------------------------

    pub(crate) fn _fork_pending_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self.with_pending(state_id, |pending| {
            let solver_ref = pending.state.solver();
            let forked_ctx = solver_ref.borrow().fork();
            Ok(RustSolverContext::from_sym_context(forked_ctx))
        })
    }

    pub(crate) fn _borrow_pending_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self.with_pending(state_id, |pending| {
            let solver_rc = pending.state.solver().clone();
            Ok(RustSolverContext::from_shared_sym_context(solver_rc))
        })
    }

    pub(crate) fn _add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self.with_pending_mut(state_id, |pending| {
            let solver_ref = pending.state.solver();
            // Shares `import_python_constraints` with `_add_constraints_to_state`
            // so the raw-pointer fast path (which keeps an unconvertible-but-live
            // AST bound instead of silently vanishing — angr-ph300.20) can only
            // ever be fixed in one place (angr-9ke6b.71).
            let added = {
                let sym_ctx = solver_ref.borrow();
                import_python_constraints(py, &sym_ctx, constraints, "pending")
            };
            log::debug!("Added {added} constraints to pending state {state_id}");
            Ok(())
        })
    }

    // -------------------------------------------------------------------------
    // Pending memory low-level (page / address access used by callbacks)
    // -------------------------------------------------------------------------

    pub(crate) fn _get_pending_mapped_pages(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self.with_pending(state_id, |pending| {
            Ok(pending
                .state
                .memory()
                .pages()
                .keys()
                .map(|&pn| pn << 12)
                .collect())
        })
    }

    pub(crate) fn _pending_memory_load_page(
        &self,
        state_id: u64,
        page_addr: u64,
    ) -> PyResult<Vec<u8>> {
        self.with_pending(state_id, |pending| {
            pending
                .state
                .memory()
                .load_page_concrete(page_addr)
                .map_err(|e| PyValueError::new_err(format!("page load failed: {e}")))
        })
    }

    /// Return every multi-byte symbolic object whose base address lies on
    /// `page_addr`'s page, as `(addr, claripy_ast)` pairs.
    ///
    /// Used by Python's `_create_state_for_callback` to replay symbolic
    /// stores that native SimProcedures (NativeRead, etc.) made into the
    /// cached Python SimState after `_install_rust_memory_proxy` has
    /// already written concrete defaults from the SP page.
    ///
    /// angr-sqfj8.26: flushes memory before reading `symbolic_objects_iter()`
    /// — otherwise a byte still living in a Multi cell is silently absent
    /// from every pending-callback sync.
    pub(crate) fn _pending_memory_load_symbolic_page<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        page_addr: u64,
    ) -> PyResult<Vec<(u64, Py<PyAny>)>> {
        self.with_pending_mut(state_id, |pending| {
            pending.state.flush_memory();
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

    /// Concrete bytes for a pending callback state's memory, evaluating
    /// symbolic bytes against the state's live solver.
    ///
    /// Unlike `_get_pending_memory` (which is eval-free per angr-04tw3.2) this
    /// concretizes a symbolic load to any SAT witness; it errors only when the
    /// load fails outright or no witness exists. Both refusals are errors
    /// rather than a zero fill: callbacks consume the result by length and
    /// mistook full-size zeros for real data (angr-ph300.19).
    pub(crate) fn _pending_memory_load(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Vec<u8>> {
        self.with_pending(state_id, |pending| {
            // One solver borrow for the whole range, not one per 16-byte chunk.
            let solver_ref = pending.state.solver();
            let ctx = solver_ref.borrow();
            crate::symbolic::load_concrete_bytes_chunked(addr, size, |a, n| {
                match pending.state.memory().load_concrete(a, n, &ctx) {
                    Ok(bv) => {
                        if let Some(val) = bv.as_u128() {
                            Ok(crate::symbolic::u128_to_le_bytes(val, n as usize))
                        } else if let Some(val) = ctx.eval(&bv) {
                            Ok(crate::symbolic::u128_to_le_bytes(val, n as usize))
                        } else {
                            Err(PyValueError::new_err(format!(
                                "pending memory load at 0x{a:x} is symbolic \
                                 and not concretizable"
                            )))
                        }
                    }
                    Err(e) => Err(PyValueError::new_err(format!(
                        "pending memory load at 0x{a:x} failed: {e}"
                    ))),
                }
            })
        })
    }

    // -------------------------------------------------------------------------
    // Skip-hook stack (zero-length hook anti-loop)
    // -------------------------------------------------------------------------

    pub(crate) fn _set_skip_hook_addr(&mut self, addr: u64) {
        let expiry = self.steps + 2;
        self.skip_hook_stack.push((addr, expiry));
        log::debug!("Added skip hook 0x{addr:x} with expiry step {expiry}");
    }

    pub(crate) fn _clear_skip_hook_addr(&mut self) {
        self.skip_hook_stack.clear();
    }

    pub(crate) fn _clear_skip_hook_for_addr(&mut self, addr: u64) {
        self.skip_hook_stack.retain(|&(a, _)| a != addr);
    }

    /// Consume one skip token for `pc`, returning whether the hook at `pc`
    /// should be skipped this step (GAP 6, zero-length-hook anti-loop).
    ///
    /// Expired entries (`expiry <= self.steps`) are dropped first, then the
    /// **topmost** matching entry — and only that one — is popped. Two nested
    /// zero-length hooks at the same address push two tokens
    /// (`_set_skip_hook_addr` does not dedup), and each occurrence must get
    /// its own; removing every match at once would leave the second
    /// occurrence unskipped and re-arm the infinite loop GAP 6 prevents
    /// (angr-9ke6b.47).
    pub(crate) fn consume_skip_hook(&mut self, pc: u64) -> bool {
        let steps = self.steps;
        self.skip_hook_stack.retain(|&(_, expiry)| expiry > steps);
        match self.skip_hook_stack.iter().rposition(|&(a, _)| a == pc) {
            Some(idx) => {
                self.skip_hook_stack.remove(idx);
                log::debug!("Skipping hook at 0x{pc:x} (zero-length hook, step {steps})");
                true
            }
            None => false,
        }
    }
}

/// Import a list of claripy byte ASTs into `state`'s solver context as 8-bit
/// `RustBV`s, preserving BVS identity (`claripy_to_rustbv` interns leaves).
/// Shared by the symbolic-file registry and the stdin seed channel.
fn import_byte_asts(
    py: Python<'_>,
    state: &crate::state::RustSimState,
    byte_asts: &Bound<'_, pyo3::types::PyList>,
) -> PyResult<Vec<crate::symbolic::RustBV>> {
    let solver_ref = state.solver();
    let sym_ctx = solver_ref.borrow();
    let mut bytes = Vec::with_capacity(byte_asts.len());
    for (idx, item) in byte_asts.iter().enumerate() {
        let bv = claripy_to_rustbv(py, &item, &sym_ctx)
            .map_err(|e| ast_import_err(&format!("content byte {idx}"), e))?;
        if bv.width() != 8 {
            return Err(PyValueError::new_err(format!(
                "content byte {idx} has width {} (expected 8)",
                bv.width()
            )));
        }
        bytes.push(bv);
    }
    Ok(bytes)
}

test_submod!("pending_api_tests.rs" => tests);
