//! State inspection API methods.
//!
//! Bodies for the cluster of pyclass-exposed methods that take a `state_id`
//! and read or mutate that specific state — register/memory/solver inspection,
//! constraint export/import, mmap/brk pointer plumbing, per-state Python-AST
//! metadata (`symbolic_pages` / `hook_symbolic_memory` / `addr_to_ast`),
//! state export, eval, history/heap/fd inspection, and inspection events.
//! The pyclass-facing thin wrappers live in `mod.rs` and forward to the
//! `pub(crate)` bodies in this module.
//!
//! PyO3 0.27.2 in this project does not enable `multiple-pymethods`, so each
//! pyclass is limited to a single `#[pymethods]` impl block — see
//! `invariant-pyo3-single-pymethods-impl`. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` /
//! `pending_api.rs` extension-impl pattern used elsewhere in `exploration/`.

use super::*;

impl RustExplorationManager {
    // -------------------------------------------------------------------------
    // Constraint sync (state-keyed)
    // -------------------------------------------------------------------------

    pub(crate) fn _add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            // Pre-fetch Z3 backend for fast path
            #[cfg(feature = "vex-engine-z3")]
            let z3_backend = py.import("claripy")
                .and_then(|c| c.getattr("backends"))
                .and_then(|b| b.getattr("z3"))
                .ok();

            let mut added = 0u32;
            for item in constraints.iter() {
                // Fast path: extract raw Z3 AST and assert directly
                #[cfg(feature = "vex-engine-z3")]
                {
                    if let Some(ref backend) = z3_backend {
                        if let Ok(z3_obj) = backend.call_method1("convert", (&item,)) {
                            if let Ok(ast_ref) = z3_obj.call_method0("as_ast") {
                                if let Ok(ptr) = ast_ref.getattr("value").and_then(|v| v.extract::<usize>()) {
                                    if ptr != 0 {
                                        unsafe { ctx_ref.add_constraint_raw(ptr); }
                                        if let Ok(bv) = claripy_to_rustbv(py, &item, ctx_ref) {
                                            ctx_ref.assumed_constraints_push(bv, true);
                                        }
                                        added += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                // Slow path: convert via RustBV
                match claripy_to_rustbv(py, &item, ctx_ref) {
                    Ok(bv) => {
                        if bv.width() == 1 {
                            sym_ctx.assume_true(&bv);
                        } else {
                            let zero = RustBV::concrete(0, bv.width());
                            let neq = bv.ne(&zero, ctx_ref);
                            sym_ctx.assume_true(&neq);
                        }
                        added += 1;
                    }
                    Err(e) => {
                        log::debug!("Could not convert initial constraint: {}", e);
                    }
                }
            }
            log::debug!("Added {} initial constraints to state {}", added, state_id);
            Ok(state.satisfiable())
        })
    }

    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn _export_z3_constraint_ptrs(&self, state_id: u64) -> PyResult<Vec<usize>> {
        self.with_state(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            Ok(ctx.export_z3_assertion_ptrs())
        })
    }

    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn _import_z3_constraint_ptrs(&mut self, state_id: u64, ptrs: Vec<usize>) -> PyResult<bool> {
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            for ptr in &ptrs {
                if *ptr != 0 {
                    unsafe { ctx.add_constraint_raw(*ptr); }
                }
            }
            log::debug!("Imported {} Z3 constraints to state {}", ptrs.len(), state_id);
            Ok(state.satisfiable())
        })
    }

    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn _debug_solver_info(&self, state_id: u64) -> PyResult<String> {
        self.with_state(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let push_level = ctx.debug_push_level();
            let n_constraints = ctx.num_constraints();
            let ptrs = ctx.export_z3_assertion_ptrs();
            Ok(format!("push_level={}, num_constraints={}, exported_ptrs={}", push_level, n_constraints, ptrs.len()))
        })
    }

    pub(crate) fn _export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let claripy = py.import("claripy")?;
        self.with_state(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let assumed = ctx.get_assumed_constraints();
            let mut results = Vec::new();
            for (bv, is_true) in &assumed {
                match rustbv_to_claripy(py, bv, claripy.as_any()) {
                    Ok(ast) => {
                        if *is_true {
                            results.push(ast);
                        } else {
                            match claripy.call_method1("Not", (ast,)) {
                                Ok(negated) => results.push(negated.unbind()),
                                Err(_) => {}
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Ok(results)
        })
    }

    // -------------------------------------------------------------------------
    // Solver timeout / mmap / brk plumbing (state-keyed)
    // -------------------------------------------------------------------------

    pub(crate) fn _set_state_solver_timeout(&self, state_id: u64, timeout_ms: u32) -> PyResult<()> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_solver_timeout: state {} not found",
                state_id
            ))
        })?;
        state.solver().borrow().set_timeout(timeout_ms);
        Ok(())
    }

    pub(crate) fn _get_state_solver_timeout(&self, state_id: u64) -> PyResult<u32> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "get_state_solver_timeout: state {} not found",
                state_id
            ))
        })?;
        Ok(state.solver().borrow().timeout_ms())
    }

    pub(crate) fn _get_state_mmap_base(&self, state_id: u64) -> PyResult<u64> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "get_state_mmap_base: state {} not found",
                state_id
            ))
        })?;
        Ok(state.mmap_base())
    }

    pub(crate) fn _set_state_mmap_base(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_mmap_base: state {} not found",
                state_id
            ))
        })?;
        state.set_mmap_base(addr);
        Ok(())
    }

    pub(crate) fn _get_state_posix_brk(&self, state_id: u64) -> PyResult<u64> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "get_state_posix_brk: state {} not found",
                state_id
            ))
        })?;
        Ok(state.posix_brk())
    }

    pub(crate) fn _set_state_posix_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_posix_brk: state {} not found",
                state_id
            ))
        })?;
        state.set_posix_brk(addr);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Per-state Python-AST metadata
    // -------------------------------------------------------------------------

    pub(crate) fn _set_state_symbolic_pages<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        pages: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_symbolic_pages: state {} not found",
                state_id
            ))
        })?;
        let mut map: HashMap<u64, Py<PyAny>> = HashMap::with_capacity(pages.len());
        for (key, value) in pages.iter() {
            let addr: u64 = key.extract()?;
            map.insert(addr, value.unbind());
        }
        let _ = py;
        state.replace_symbolic_pages(map);
        Ok(())
    }

    pub(crate) fn _get_state_symbolic_pages<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(state) = self.find_state(state_id) {
            for (addr, ast) in state.symbolic_pages() {
                dict.set_item(*addr, ast.clone_ref(py))?;
            }
        }
        Ok(dict)
    }

    pub(crate) fn _set_state_hook_symbolic_memory(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_hook_symbolic_memory: state {} not found",
                state_id
            ))
        })?;
        state.set_hook_symbolic_memory(addr, ast, size);
        Ok(())
    }

    pub(crate) fn _get_state_hook_symbolic_memory<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(state) = self.find_state(state_id) {
            for (addr, (ast, size)) in state.hook_symbolic_memory() {
                dict.set_item(*addr, (ast.clone_ref(py), *size))?;
            }
        }
        Ok(dict)
    }

    pub(crate) fn _set_state_addr_to_ast(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        let state = self.find_state_mut(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_state_addr_to_ast: state {} not found",
                state_id
            ))
        })?;
        state.set_addr_to_ast(addr, ast, size);
        Ok(())
    }

    pub(crate) fn _get_state_addr_to_ast<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(state) = self.find_state(state_id) {
            for (addr, (ast, size)) in state.addr_to_ast() {
                dict.set_item(*addr, (ast.clone_ref(py), *size))?;
            }
        }
        Ok(dict)
    }

    pub(crate) fn _clear_state_metadata(&mut self, state_id: u64) -> PyResult<()> {
        if let Some(state) = self.find_state_mut(state_id) {
            state.clear_state_metadata();
        }
        Ok(())
    }

    pub(crate) fn _fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        let state = self.find_state(state_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "fork_state_solver: state {} not found",
                state_id
            ))
        })?;
        let solver_ref = state.solver();
        let forked_ctx = solver_ref.borrow().fork();
        Ok(RustSolverContext::from_sym_context(forked_ctx))
    }

    // -------------------------------------------------------------------------
    // State export
    // -------------------------------------------------------------------------

    pub(crate) fn _export_state(&self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self.with_state(state_id, |state| Ok(state.export_full()))
    }

    pub(crate) fn _export_state_flushed(&mut self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        // find_state_mut only checks stashes; we still need an explicit pending fallback.
        if let Some(state) = self.find_state_mut(state_id) {
            return Ok(state.flush_and_export_full());
        }
        if let Some(ref mut pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Ok(pending.state.flush_and_export_full());
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    pub(crate) fn _export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.sm.get(stash)
            .map(|s| s.iter().map(|state| state.export_full()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn _export_found_states_flushed(&mut self) -> Vec<crate::state::ExplorationStateSnapshot> {
        if let Some(states) = self.sm.get_mut(STASH_FOUND) {
            states.iter_mut().map(|s| s.flush_and_export_full()).collect()
        } else {
            Vec::new()
        }
    }

    pub(crate) fn _export_found_states(&self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self._export_stash(STASH_FOUND)
    }

    // -------------------------------------------------------------------------
    // Eval / satisfiability / register / memory inspection
    // -------------------------------------------------------------------------

    pub(crate) fn _eval_in_state(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        self.with_state(state_id, |state| {
            match state.memory_load(addr, size) {
                Ok(bv) => {
                    if let Some(val) = state.eval(&bv) {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    Ok(None)
                }
                Err(_) => Ok(None),
            }
        })
    }

    pub(crate) fn _state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        self.with_state(state_id, |state| {
            let mem = state.memory();
            let total = mem.symbolic_object_count();
            let has_at_addr = mem.get_symbolic_object(addr).map(|bv| bv.width());
            let page_num = addr >> 12;
            let offset = (addr & 0xFFF) as u16;
            let page_info = if let Some(page) = mem.pages().get(&page_num) {
                format!("page=mapped sym_at_offset={}", page.is_symbolic(offset))
            } else {
                "page=unmapped".to_string()
            };
            Ok(format!(
                "total_sym_objs={} at_0x{:x}={:?} {}",
                total, addr, has_at_addr, page_info
            ))
        })
    }

    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn _get_state_symbolic_z3_asts(&self, state_id: u64) -> PyResult<Vec<(u64, usize, u32)>> {
        use z3::ast::Ast;
        self.with_state(state_id, |state| {
            let mem = state.memory();
            let mut result = Vec::new();
            for (&addr, bv) in mem.symbolic_objects_iter() {
                // Only export Expression values (Rust-computed).
                // Skip Symbolic values (imported from Python) — Python already
                // has those with proper claripy identity.
                // Also skip addresses that were originally imported from Python,
                // even if the binary modified them (Symbolic→Expression).
                // Python's memory has the correct original value; overwriting
                // it would break post-exploration constraint solving (flareon5).
                if mem.is_imported_addr(addr) {
                    continue;
                }
                if matches!(bv, RustBV::Expression { .. }) {
                    let z3_ast = bv.to_z3_ast();
                    let raw_ptr = z3_ast.get_z3_ast().as_ptr() as usize;
                    // Prevent z3::ast::BV destructor from decrementing the ref count.
                    // Python takes ownership of this pointer.
                    std::mem::forget(z3_ast);
                    result.push((addr, raw_ptr, bv.width()));
                }
            }
            Ok(result)
        })
    }

    pub(crate) fn _state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.satisfiable()))
    }

    pub(crate) fn _state_enforce_permissions(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.enforce_permissions()))
    }

    pub(crate) fn _get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self.with_state(state_id, |state| {
            Ok(state.get_register(name).and_then(|bv| bv.as_u128()))
        })
    }

    pub(crate) fn _get_state_registers_batch(&self, state_id: u64, names: Vec<String>) -> PyResult<Vec<Option<u128>>> {
        self.with_state(state_id, |state| {
            Ok(names.iter()
                .map(|name| state.get_register(name).and_then(|bv| bv.as_u128()))
                .collect())
        })
    }

    pub(crate) fn _get_state_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        self.with_state(state_id, |state| {
            match state.memory_load(addr, size) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u128() {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    if let Some(val) = state.eval(&bv) {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    Ok(None)
                }
                Err(_) => Ok(None),
            }
        })
    }

    // -------------------------------------------------------------------------
    // stdout / stdin / file descriptor inspection
    // -------------------------------------------------------------------------

    pub(crate) fn _has_state_stdout(&self, state_id: u64) -> bool {
        // find_state already checks pending_callback first.
        self.find_state(state_id).map_or(false, |s| s.has_stdout())
    }

    pub(crate) fn _get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self._get_state_fd_output(state_id, 1)
    }

    pub(crate) fn _get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self.with_state(state_id, |state| Ok(state.fd_buffer(fd).to_vec()))
    }

    pub(crate) fn _has_state_stdin_symbols(&self, state_id: u64) -> bool {
        self.find_state(state_id).map_or(false, |s| s.has_stdin_symbols())
    }

    pub(crate) fn _get_state_stdin_symbols(&self, state_id: u64) -> PyResult<Vec<(String, u32)>> {
        self.with_state(state_id, |state| Ok(state.stdin_symbols().to_vec()))
    }

    // -------------------------------------------------------------------------
    // Call stack / history / heap / fds
    // -------------------------------------------------------------------------

    pub(crate) fn _get_state_call_stack(&self, state_id: u64) -> PyResult<Vec<(u64, u64, u64, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.call_stack().iter()
                .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
                .collect())
        })
    }

    pub(crate) fn _get_state_call_stack_depth(&self, state_id: u64) -> PyResult<usize> {
        self.with_state(state_id, |state| Ok(state.call_stack_depth()))
    }

    pub(crate) fn _get_state_detailed_history(&self, state_id: u64) -> PyResult<Vec<(u64, u8, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.detailed_history().iter()
                .map(|e| (e.addr, e.jumpkind, e.jump_target))
                .collect())
        })
    }

    pub(crate) fn _get_state_heap_metadata(&self, state_id: u64) -> PyResult<(Vec<(u64, u64)>, Vec<u64>)> {
        self.with_state(state_id, |state| {
            let meta = state.heap_metadata();
            let allocated: Vec<(u64, u64)> = meta.allocated.iter()
                .map(|(&addr, &size)| (addr, size))
                .collect();
            let freed = meta.freed.clone();
            Ok((allocated, freed))
        })
    }

    pub(crate) fn _get_state_open_fds(&self, state_id: u64) -> PyResult<Vec<(u32, String, u64, u32, usize, bool)>> {
        self.with_state(state_id, |state| {
            Ok(state.file_system_ref().all_fds().iter().filter_map(|&fd| {
                let info = state.file_system_ref().fd_info(fd)?;
                Some((fd, info.0.to_string(), info.1, info.2, info.3, info.4))
            }).collect())
        })
    }

    pub(crate) fn _get_state_fd_content(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self.with_state(state_id, |state| {
            Ok(state.file_system_ref().fd_content(fd).to_vec())
        })
    }

    // -------------------------------------------------------------------------
    // Inspection events
    // -------------------------------------------------------------------------

    pub(crate) fn _enable_state_inspection(&mut self, state_id: u64, event_type: u8) -> PyResult<()> {
        let event = crate::state::InspectEvent::from_u8(event_type)
            .ok_or_else(|| PyValueError::new_err(format!("invalid event type: {}", event_type)))?;
        self.with_state_mut(state_id, |state| {
            state.inspection_mut().enable(event);
            Ok(())
        })
    }

    pub(crate) fn _enable_all_inspections(&mut self, state_id: u64) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            state.inspection_mut().enable_all();
            Ok(())
        })
    }

    pub(crate) fn _get_state_inspection_counts(&self, state_id: u64) -> PyResult<Vec<(String, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.inspection().event_counts().iter().enumerate()
                .filter(|&(_, &count)| count > 0)
                .filter_map(|(i, &count)| {
                    let event = crate::state::InspectEvent::from_u8(i as u8)?;
                    Some((event.name().to_string(), count))
                })
                .collect())
        })
    }

    pub(crate) fn _get_state_inspection_events(&self, state_id: u64) -> PyResult<Vec<(u8, String, u64, u32, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state.inspection().events().iter().map(|e| {
                (e.event as u8, e.event.name().to_string(), e.addr, e.size, e.block_addr)
            }).collect())
        })
    }

    pub(crate) fn _eval_stdin_symbol(&self, state_id: u64, name: &str) -> Option<u64> {
        let state = self.find_state(state_id)?;
        let ctx = state.solver().borrow();
        // Find the symbol by name in the solver context
        let sym = crate::symbolic::RustBV::symbolic(&ctx, name, 8);
        ctx.eval(&sym).map(|v| v as u64)
    }
}
