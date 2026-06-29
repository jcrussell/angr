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
            let ctx_ref: &SymContext = &sym_ctx;

            // Pre-fetch Z3 backend for fast path
            #[cfg(feature = "vex-engine-z3")]
            let z3_backend = py
                .import("claripy")
                .and_then(|c| c.getattr("backends"))
                .and_then(|b| b.getattr("z3"))
                .ok();

            let mut added = 0u32;
            for item in constraints.iter() {
                // Fast path: extract typed Z3 AST handle and assert directly
                #[cfg(feature = "vex-engine-z3")]
                {
                    if let Some(ref backend) = z3_backend
                        && let Ok(z3_obj) = backend.call_method1("convert", (&item,))
                        && let Ok(ast_ref) = z3_obj.call_method0("as_ast")
                        && let Ok(ptr) = ast_ref.getattr("value").and_then(|v| v.extract::<usize>())
                    {
                        let z3_ctx = z3::Context::thread_local();
                        // SAFETY: claripy's z3 backend returned
                        // this pointer for a live AST it caches;
                        // matches our thread-local Z3 context.
                        if let Some(z3_ast) = unsafe { Z3AstPtr::from_borrowed_raw(&z3_ctx, ptr) } {
                            ctx_ref.add_constraint_raw(z3_ast);
                            if let Ok(bv) = claripy_to_rustbv(py, &item, ctx_ref) {
                                ctx_ref.assumed_constraints_push(bv, true);
                            }
                            added += 1;
                            continue;
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
    pub(crate) fn _import_z3_constraint_ptrs(
        &mut self,
        state_id: u64,
        ptrs: Vec<usize>,
    ) -> PyResult<bool> {
        // angr-33t9: validate every pointer is a Bool-sorted AST in the active
        // thread-local Z3 context BEFORE handing it to `add_constraint_raw`'s
        // unsafe wrap. This catches the realistic misuse cases — null in the
        // middle of a list, a BV ptr exported by mistake, a foreign-context
        // AST — and converts them to PyValueError. It does NOT defend against
        // arbitrary integers (e.g. 0xdeadbeef): `Z3_get_sort` dereferences the
        // pointer, so truly garbage values may still segfault before Z3 has a
        // chance to signal an error. The SAFETY contract in `add_constraint_raw`
        // is unchanged; this check just rejects the cheap-to-detect failures.
        // See PyO3 trust-model audit in docs/advanced-topics/rust_engine.rst.
        use z3_sys::{SortKind, Z3_get_sort, Z3_get_sort_kind};
        let z3_ctx_handle = z3::Context::thread_local();
        let raw_ctx = z3_ctx_handle.get_z3_context();
        for (idx, ptr) in ptrs.iter().enumerate() {
            let raw_ast =
                std::ptr::NonNull::new(*ptr as *mut z3_sys::_Z3_ast).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "import_z3_constraint_ptrs: null pointer at index {idx}"
                    ))
                })?;
            // SAFETY: `raw_ctx` is the active thread-local context for the
            // duration of this call; `raw_ast` is non-null (checked above).
            // Z3 returns None when `raw_ast` is not a valid AST belonging to
            // `raw_ctx`; the binding's Option wrapping converts this safely.
            // If `raw_ast` is non-AST garbage, the deref inside Z3 may
            // segfault — that's the residual UB documented above.
            let sort = unsafe { Z3_get_sort(raw_ctx, raw_ast) }.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "import_z3_constraint_ptrs: pointer at index {idx} is not a valid Z3 AST in the active context"
                ))
            })?;
            // SAFETY: `sort` came from `Z3_get_sort` on the same context, so
            // it is a live `Z3_sort` in `raw_ctx`. `Z3_get_sort_kind` is a
            // pure metadata read.
            let kind = unsafe { Z3_get_sort_kind(raw_ctx, sort) };
            if kind != SortKind::Bool {
                return Err(PyValueError::new_err(format!(
                    "import_z3_constraint_ptrs: pointer at index {idx} has sort kind {kind:?}, expected Bool"
                )));
            }
        }
        self.with_state_mut(state_id, |state| {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let z3_ctx = z3::Context::thread_local();
            for ptr in &ptrs {
                // SAFETY: every ptr was validated above by the
                // Z3_get_sort/sort_kind probe to be a live Bool-sorted AST
                // in `z3_ctx`. `Z3AstPtr::from_borrowed_raw` takes its own
                // ref via `Z3_inc_ref`.
                if let Some(z3_ast) = unsafe { Z3AstPtr::from_borrowed_raw(&z3_ctx, *ptr) } {
                    ctx.add_constraint_raw(z3_ast);
                }
            }
            log::debug!(
                "Imported {} Z3 constraints to state {}",
                ptrs.len(),
                state_id
            );
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
            Ok(format!(
                "push_level={}, num_constraints={}, exported_ptrs={}",
                push_level,
                n_constraints,
                ptrs.len()
            ))
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
                if let Ok(ast) = rustbv_to_claripy(py, bv, claripy.as_any()) {
                    if *is_true {
                        results.push(ast);
                    } else {
                        if let Ok(negated) = claripy.call_method1("Not", (ast,)) {
                            results.push(negated.unbind())
                        }
                    }
                }
            }
            Ok(results)
        })
    }

    // -------------------------------------------------------------------------
    // Solver timeout / mmap / brk plumbing (state-keyed)
    // -------------------------------------------------------------------------

    pub(crate) fn _set_state_solver_timeout(&self, state_id: u64, timeout_ms: u32) -> PyResult<()> {
        self.with_state(state_id, |state| {
            state.solver().borrow().set_timeout(timeout_ms);
            Ok(())
        })
    }

    pub(crate) fn _get_state_solver_timeout(&self, state_id: u64) -> PyResult<u32> {
        self.with_state(state_id, |state| Ok(state.solver().borrow().timeout_ms()))
    }

    pub(crate) fn _get_state_mmap_base(&self, state_id: u64) -> PyResult<u64> {
        self.with_state(state_id, |state| Ok(state.mmap_base()))
    }

    pub(crate) fn _set_state_mmap_base(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            state.set_mmap_base(addr);
            Ok(())
        })
    }

    pub(crate) fn _get_state_posix_brk(&self, state_id: u64) -> PyResult<u64> {
        self.with_state(state_id, |state| Ok(state.posix_brk()))
    }

    pub(crate) fn _set_state_posix_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            state.set_posix_brk(addr);
            Ok(())
        })
    }

    pub(crate) fn _get_state_heap_brk(&self, state_id: u64) -> PyResult<u64> {
        self.with_state(state_id, |state| Ok(state.heap_brk()))
    }

    pub(crate) fn _set_state_heap_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            state.set_heap_brk(addr);
            Ok(())
        })
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
        let _ = py;
        let mut map: HashMap<u64, Py<PyAny>> = HashMap::with_capacity(pages.len());
        for (key, value) in pages.iter() {
            let addr: u64 = key.extract()?;
            map.insert(addr, value.unbind());
        }
        self.with_state_mut(state_id, |state| {
            state.replace_symbolic_pages(map);
            Ok(())
        })
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
        self.with_state_mut(state_id, |state| {
            state.set_hook_symbolic_memory(addr, ast, size);
            Ok(())
        })
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
        self.with_state_mut(state_id, |state| {
            state.set_addr_to_ast(addr, ast, size);
            Ok(())
        })
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
            PyValueError::new_err(format!("fork_state_solver: state {} not found", state_id))
        })?;
        let solver_ref = state.solver();
        let forked_ctx = solver_ref.borrow().fork();
        Ok(RustSolverContext::from_sym_context(forked_ctx))
    }

    // -------------------------------------------------------------------------
    // State export
    // -------------------------------------------------------------------------

    pub(crate) fn _export_state(
        &self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self.with_state(state_id, |state| Ok(state.export_full()))
    }

    pub(crate) fn _export_state_flushed(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        // find_state_mut only checks stashes; we still need an explicit pending fallback.
        if let Some(state) = self.find_state_mut(state_id) {
            return Ok(state.flush_and_export_full());
        }
        if let Some(pending) = self.pending_callbacks.get_mut(&StateId::new(state_id)) {
            return Ok(pending.state.flush_and_export_full());
        }
        Err(PyValueError::new_err(format!(
            "state {} not found",
            state_id
        )))
    }

    pub(crate) fn _export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.sm
            .get(stash)
            .map(|s| s.iter().map(|state| state.export_full()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn _export_found_states_flushed(
        &mut self,
    ) -> Vec<crate::state::ExplorationStateSnapshot> {
        if let Some(states) = self.sm.get_mut(STASH_FOUND) {
            states
                .iter_mut()
                .map(|s| s.flush_and_export_full())
                .collect()
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

    pub(crate) fn _eval_in_state(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Vec<u8>>> {
        self.with_state(state_id, |state| match state.memory_load(addr, size) {
            Ok(bv) => {
                if let Some(val) = state.eval(&bv) {
                    return Ok(Some(super::helpers::u128_to_le_bytes(val, size as usize)));
                }
                Ok(None)
            }
            Err(_) => Ok(None),
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
    pub(crate) fn _get_state_symbolic_z3_asts(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<(u64, usize, u32)>> {
        use z3::ast::Ast;
        self.with_state(state_id, |state| {
            let mem = state.memory();
            let mut result = Vec::new();
            for (addr, bv) in mem.symbolic_objects_iter() {
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
                    result.push((addr.raw(), raw_ptr, bv.width()));
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

    pub(crate) fn _state_enforce_nx(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.enforce_nx()))
    }

    pub(crate) fn _state_no_ip_concretization(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.no_ip_concretization()))
    }

    pub(crate) fn _state_no_symbolic_jump_resolution(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.no_symbolic_jump_resolution()))
    }

    pub(crate) fn _state_keep_ip_symbolic(&self, state_id: u64) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.keep_ip_symbolic()))
    }

    /// Whether the named symex-relevant SimOption is active on the state
    /// (angr-kzjv6). Lets Python tests assert the option subset threaded in
    /// `_add_rust_state` reached the Rust state.
    pub(crate) fn _state_has_option(&self, state_id: u64, name: &str) -> PyResult<bool> {
        self.with_state(state_id, |state| Ok(state.has_option(name)))
    }

    pub(crate) fn _get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self.with_state(state_id, |state| {
            Ok(state.get_register(name).and_then(|bv| bv.as_u128()))
        })
    }

    pub(crate) fn _get_state_registers_batch(
        &self,
        state_id: u64,
        names: Vec<String>,
    ) -> PyResult<Vec<Option<u128>>> {
        self.with_state(state_id, |state| {
            Ok(names
                .iter()
                .map(|name| state.get_register(name).and_then(|bv| bv.as_u128()))
                .collect())
        })
    }

    /// Get the claripy AST for a register on a specific state (angr-4pm1).
    ///
    /// Mirrors `_get_pending_register_ast` for an arbitrary `state_id`. Returns
    /// the claripy AST built from Rust's stored `RustBV` (symbolic or
    /// concrete) via `rustbv_to_claripy`, preserving identity for symbolic
    /// values so `state.solver.add(state.regs.<sym_reg> == K)` actually
    /// constrains the Rust-side symbol.
    ///
    /// Returns `None` when the register name is unknown to the architecture
    /// or the state holds no value for it — callers can decide whether that
    /// is a hard error or fall back to minting a fresh BVS.
    pub(crate) fn _get_state_register_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        name: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.with_state(state_id, |state| {
            let bv = match state.get_register(name) {
                Some(bv) => bv,
                None => return Ok(None),
            };
            let claripy = py.import("claripy")?;
            let ast = rustbv_to_claripy(py, &bv, claripy.as_any()).map_err(|e| {
                PyRuntimeError::new_err(format!("register {} AST conversion failed: {}", name, e))
            })?;
            Ok(Some(ast))
        })
    }

    /// Set a register on a specific state to a symbolic value from a claripy
    /// AST (angr-4pm1). Mirrors `_set_pending_register_symbolic_ast` for an
    /// arbitrary `state_id`.
    ///
    /// Routes through `claripy_to_rustbv` so the symbol gets registered in
    /// the shared cache. That makes the inverse `get_state_register_ast`
    /// round-trip return the original claripy AST verbatim (identity
    /// preservation), which is what the proxy needs so constraints land on
    /// the same Z3 symbol Rust is tracking.
    pub(crate) fn _set_state_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            let bv = {
                let solver_ref = state.solver();
                let sym_ctx = solver_ref.borrow();
                let ctx_ref: &SymContext = &sym_ctx;
                claripy_to_rustbv(py, ast, ctx_ref)
                    .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?
            };
            if state.set_register(reg_name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "failed to set register: {}",
                    reg_name
                )))
            }
        })
    }

    pub(crate) fn _get_state_memory(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Vec<u8>>> {
        // The u128 reconstruction below round-trips at most 16 bytes; a wider
        // request would truncate (`val >> (i*8)` wraps mod 128 for i >= 16).
        // Read in <=16-byte chunks and concatenate so any size is exact.
        if size > 16 {
            let mut out = Vec::with_capacity(size as usize);
            let mut off = 0u32;
            while off < size {
                let chunk = (size - off).min(16);
                match self._get_state_memory(state_id, addr + off as u64, chunk)? {
                    Some(bytes) => out.extend_from_slice(&bytes),
                    None => return Ok(None),
                }
                off += chunk;
            }
            return Ok(Some(out));
        }
        self.with_state(state_id, |state| match state.memory_load(addr, size) {
            Ok(bv) => {
                if let Some(val) = bv.as_u128() {
                    return Ok(Some(super::helpers::u128_to_le_bytes(val, size as usize)));
                }
                if let Some(val) = state.eval(&bv) {
                    return Ok(Some(super::helpers::u128_to_le_bytes(val, size as usize)));
                }
                Ok(None)
            }
            Err(_) => Ok(None),
        })
    }

    /// Return the claripy AST for `size` bytes of memory at `addr` on
    /// `state_id` (angr-8dop.1). Unlike `_get_state_memory`, this never
    /// concretizes — a symbolic load returns the symbolic AST verbatim so
    /// downstream callers (SimProcedures via the `RustMemoryProxy` gate)
    /// can build constraints on the actual symbolic bytes.
    ///
    /// Returns `None` when the memory load errors (typically unmapped /
    /// missing permission) so the caller can fall back to a zero-fill.
    ///
    /// Byte-order convention: the returned AST has byte `i` of memory at
    /// bit positions `[i*8+7 : i*8]` (LSB-first), matching how
    /// `_set_state_memory_concrete` lays out concrete bytes. The proxy
    /// reverses for `Iend_BE`, returns verbatim for `Iend_LE`.
    pub(crate) fn _get_state_memory_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.with_state(state_id, |state| match state.memory_load(addr, size) {
            Ok(bv) => {
                let claripy = py.import("claripy")?;
                let ast = rustbv_to_claripy(py, &bv, claripy.as_any()).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "memory AST conversion at addr 0x{:x} size {}: {}",
                        addr, size, e
                    ))
                })?;
                Ok(Some(ast))
            }
            Err(_) => Ok(None),
        })
    }

    /// Concrete-address concrete-value memory store on `state_id`
    /// (angr-j28e write-through). Mirrors `_set_pending_memory` for an
    /// arbitrary state. Used by the Python-side `RustMemoryProxy.store`
    /// when a hook callback or `state.inspect` user code mutates memory
    /// through the proxy with a concrete address + concrete bytes.
    pub(crate) fn _set_state_memory_concrete(
        &mut self,
        state_id: u64,
        addr: u64,
        data: &[u8],
    ) -> PyResult<()> {
        // angr-5aj8: split into 16-byte chunks because RustBV::Concrete is
        // backed by a u128. A single packed value would shift-overflow for
        // byte indices >= 16, and the downstream store_concrete page-fill
        // loop would emit a 16-byte-cycle pattern across the entire
        // data.len() range — corrupting memory wholesale. Mirrors the safe
        // pattern in RustSimState::apply_changes (state.rs).
        self.with_state_mut(state_id, |state| {
            super::helpers::store_concrete_bytes_chunked(addr, data, |chunk_addr, bv| {
                state
                    .memory_store(chunk_addr, bv)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
        })
    }

    /// Concrete-address symbolic-value memory store on `state_id`
    /// (angr-j28e write-through). The value is supplied as a claripy AST;
    /// routes through `claripy_to_rustbv` so the symbol is registered in
    /// the shared cache and downstream reads see the same Z3 AST.
    pub(crate) fn _set_state_memory_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_state_mut(state_id, |state| {
            let bv = {
                let solver_ref = state.solver();
                let sym_ctx = solver_ref.borrow();
                let ctx_ref: &SymContext = &sym_ctx;
                claripy_to_rustbv(py, ast, ctx_ref)
                    .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?
            };
            state
                .memory_store(addr, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Phase 1.4 (angr-5zw8): perform a symbolic-address store on `state_id`
    /// via the lazy Multi-cell path. Mirrors the eager fallback in
    /// `_cb_memory_store_symbolic_full` but installs Multi alternatives at
    /// each concretized byte instead of folding eager ITE chains.
    ///
    /// Returns `true` if the store landed in Rust memory. Returns `false`
    /// when conversion fails or the address concretization yields
    /// TooLarge / Failed — caller should fall back to the Python path so
    /// the write is not silently lost.
    pub(crate) fn _state_memory_store_symbolic_multi<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        addr_ast: &Bound<'py, PyAny>,
        data_ast: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        self.with_state_mut(state_id, |state| {
            // Scope the immutable solver borrow so the subsequent mutable
            // memory_store_symbolic_multi call can take its own borrow.
            let (addr_bv, data_bv) = {
                let solver_ref = state.solver();
                let sym_ctx = solver_ref.borrow();
                let ctx: &SymContext = &sym_ctx;
                let addr_bv = match claripy_to_rustbv(py, addr_ast, ctx) {
                    Ok(bv) => bv,
                    Err(e) => {
                        log::debug!(
                            "state_memory_store_symbolic_multi: addr convert failed: {}",
                            e
                        );
                        return Ok(false);
                    }
                };
                let data_bv = match claripy_to_rustbv(py, data_ast, ctx) {
                    Ok(bv) => bv,
                    Err(e) => {
                        log::debug!(
                            "state_memory_store_symbolic_multi: data convert failed: {}",
                            e
                        );
                        return Ok(false);
                    }
                };
                (addr_bv, data_bv)
            };
            match state.memory_store_symbolic_multi(addr_bv, data_bv) {
                Ok(()) => Ok(true),
                Err(e) => {
                    log::debug!("state_memory_store_symbolic_multi: store failed: {:?}", e);
                    Ok(false)
                }
            }
        })
    }

    // -------------------------------------------------------------------------
    // stdout / stdin / file descriptor inspection
    // -------------------------------------------------------------------------

    pub(crate) fn _has_state_stdout(&self, state_id: u64) -> bool {
        // find_state already checks pending_callback first.
        self.find_state(state_id).is_some_and(|s| s.has_stdout())
    }

    pub(crate) fn _get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self._get_state_fd_output(state_id, 1)
    }

    pub(crate) fn _get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self.with_state(state_id, |state| Ok(state.fd_buffer(fd).to_vec()))
    }

    pub(crate) fn _has_state_stdin_symbols(&self, state_id: u64) -> bool {
        self.find_state(state_id)
            .is_some_and(|s| s.has_stdin_symbols())
    }

    pub(crate) fn _get_state_stdin_symbols(&self, state_id: u64) -> PyResult<Vec<(String, u32)>> {
        self.with_state(state_id, |state| Ok(state.stdin_symbols().to_vec()))
    }

    // -------------------------------------------------------------------------
    // Call stack / history / heap / fds
    // -------------------------------------------------------------------------

    pub(crate) fn _get_state_call_stack(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<(u64, u64, u64, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state
                .call_stack()
                .iter()
                .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
                .collect())
        })
    }

    pub(crate) fn _get_state_call_stack_depth(&self, state_id: u64) -> PyResult<usize> {
        self.with_state(state_id, |state| Ok(state.call_stack_depth()))
    }

    pub(crate) fn _get_state_detailed_history(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<(u64, u8, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state
                .detailed_history()
                .iter()
                .map(|e| (e.addr, e.jumpkind, e.jump_target))
                .collect())
        })
    }

    pub(crate) fn _get_state_heap_metadata(&self, state_id: u64) -> PyResult<HeapMetadataReturn> {
        self.with_state(state_id, |state| {
            let meta = state.heap_metadata();
            let allocated: Vec<(u64, u64)> = meta
                .allocated
                .iter()
                .map(|(&addr, &size)| (addr, size))
                .collect();
            let freed = meta.freed.clone();
            Ok((allocated, freed))
        })
    }

    pub(crate) fn _get_state_open_fds(&self, state_id: u64) -> PyResult<Vec<OpenFdInfo>> {
        self.with_state(state_id, |state| {
            Ok(state
                .file_system_ref()
                .all_fds()
                .iter()
                .filter_map(|&fd| {
                    let info = state.file_system_ref().fd_info(fd)?;
                    Some((fd, info.0.to_string(), info.1, info.2, info.3, info.4))
                })
                .collect())
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

    pub(crate) fn _enable_state_inspection(
        &mut self,
        state_id: u64,
        event_type: u8,
    ) -> PyResult<()> {
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

    pub(crate) fn _get_state_inspection_counts(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<(String, u64)>> {
        self.with_state(state_id, |state| {
            Ok(state
                .inspection()
                .event_counts()
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .filter_map(|(i, &count)| {
                    let event = crate::state::InspectEvent::from_u8(i as u8)?;
                    Some((event.name().to_string(), count))
                })
                .collect())
        })
    }

    pub(crate) fn _get_state_inspection_events(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<InspectionEventInfo>> {
        self.with_state(state_id, |state| {
            Ok(state
                .inspection()
                .events()
                .iter()
                .map(|e| {
                    (
                        e.event as u8,
                        e.event.name().to_string(),
                        e.addr,
                        e.size,
                        e.block_addr,
                    )
                })
                .collect())
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
