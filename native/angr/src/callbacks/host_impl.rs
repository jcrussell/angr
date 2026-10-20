//! [`EngineHost`] for `PythonCallbacks`: the interpreter's services, served
//! from Python.
//!
//! Every Python object the interpreter used to build or unpack itself is
//! built and unpacked here instead: claripy ASTs for breakpoints and for
//! symbolic loads, and the `PyErr`s that become `String` messages.

use super::*;
use crate::claripy_bridge::{claripy_module, claripy_to_rustbv, is_claripy_ast, rustbv_to_claripy};
use crate::interpreter::{EngineHost, HostLoad, HostPage};
use crate::symbolic::{RustBV, SymContext};

/// `value` as a claripy AST, or `None` when the export fails (the breakpoint
/// is then skipped: a missing claripy or an unexportable value must not halt
/// exploration).
fn to_claripy(value: &RustBV) -> Option<Py<PyAny>> {
    Python::attach(|py| {
        let claripy_mod = claripy_module(py).ok()?;
        rustbv_to_claripy(py, value, &claripy_mod).ok()
    })
}

/// A claripy AST from Python as a `RustBV`, or `None` when it is not one or
/// does not convert (the caller then mints a fresh symbol or keeps its value).
fn from_claripy(ctx: &SymContext, ast: &Py<PyAny>) -> Option<RustBV> {
    Python::attach(|py| {
        let ast = ast.bind(py);
        if is_claripy_ast(ast) {
            claripy_to_rustbv(py, ast, ctx).ok()
        } else {
            None
        }
    })
}

fn host_load(ctx: &SymContext, (bytes, symbolic, ast): BatchLoadEntry) -> HostLoad {
    let value = if symbolic {
        ast.as_ref().and_then(|ast| from_claripy(ctx, ast))
    } else {
        None
    };
    HostLoad {
        bytes,
        symbolic,
        value,
    }
}

fn message(e: &PyErr) -> String {
    e.to_string()
}

impl EngineHost for PythonCallbacks {
    fn has_memory_store_symbolic_value(&self) -> bool {
        PythonCallbacks::has_memory_store_symbolic_value(self)
    }

    fn has_memory_store_symbolic_full(&self) -> bool {
        PythonCallbacks::has_memory_store_symbolic_full(self)
    }

    fn has_memory_load_symbolic_full(&self) -> bool {
        PythonCallbacks::has_memory_load_symbolic_full(self)
    }

    fn has_fetch_page(&self) -> bool {
        PythonCallbacks::has_fetch_page(self)
    }

    fn has_dirty_call(&self) -> bool {
        PythonCallbacks::has_dirty_call(self)
    }

    fn can_serve_page(&self, page_addr: u64) -> bool {
        self.python_can_serve_page(page_addr)
    }

    fn has_page(&self, page_addr: u64) -> bool {
        self.python_has_page(page_addr)
    }

    #[inline]
    fn inspect_event_enabled(&self, event: InspectBit) -> bool {
        PythonCallbacks::inspect_event_enabled(self, event)
    }

    fn memory_load(&self, ctx: &SymContext, addr: u64, size: u32) -> Result<HostLoad, String> {
        let entry = self.call_memory_load(addr, size).map_err(|e| message(&e))?;
        Ok(host_load(ctx, entry))
    }

    fn memory_load_batch(
        &self,
        ctx: &SymContext,
        loads: &[(u64, u32)],
    ) -> Result<Vec<HostLoad>, String> {
        let entries = self
            .call_memory_load_batch(loads)
            .map_err(|e| message(&e))?;
        Ok(entries
            .into_iter()
            .map(|entry| host_load(ctx, entry))
            .collect())
    }

    fn memory_load_symbolic_full(
        &self,
        ctx: &SymContext,
        addr: &RustBV,
        size: u32,
    ) -> Result<Option<RustBV>, String> {
        let ast = self
            .call_memory_load_symbolic_full(addr, size)
            .map_err(|e| message(&e))?;
        Ok(from_claripy(ctx, &ast))
    }

    fn memory_store(&self, addr: u64, data: &[u8]) -> Result<(), String> {
        self.call_memory_store(addr, data).map_err(|e| message(&e))
    }

    fn memory_store_batch(&self, stores: &[(u64, Vec<u8>)]) -> Result<(), String> {
        self.call_memory_store_batch(stores)
            .map_err(|e| message(&e))
    }

    fn memory_store_symbolic(
        &self,
        addrs: &[u64],
        data: &RustBV,
        addr_ast: &RustBV,
    ) -> Result<(), String> {
        self.call_memory_store_symbolic(addrs, data, addr_ast)
            .map_err(|e| message(&e))
    }

    fn memory_store_symbolic_value(&self, addr: u64, value: &RustBV) -> Result<(), String> {
        self.call_memory_store_symbolic_value(addr, value)
            .map_err(|e| message(&e))
    }

    fn memory_store_symbolic_full(&self, addr: &RustBV, data: &RustBV) -> Result<(), String> {
        self.call_memory_store_symbolic_full(addr, data)
            .map_err(|e| message(&e))
    }

    fn fetch_page(&self, page_addr: u64) -> Result<HostPage, String> {
        self.call_fetch_page(page_addr).map_err(|e| message(&e))
    }

    fn batch_fetch_pages(&self, page_addrs: &[u64]) -> Result<Vec<HostPage>, String> {
        self.call_batch_fetch_pages(page_addrs)
            .map_err(|e| message(&e))
    }

    fn lift_block(
        &self,
        addr: u64,
        opt_level: Option<i32>,
        dirty_bytes: Option<&[u8]>,
    ) -> Result<LiftBlockOutcome, String> {
        self.call_lift_block(addr, opt_level, dirty_bytes)
            .map_err(|e| message(&e))
    }

    fn dirty_call(
        &self,
        name: &str,
        args: &[u64],
        ret_ty_bits: u32,
    ) -> Result<(Vec<u8>, bool), String> {
        let (bytes, symbolic, _ast) = self
            .call_dirty_call(name, args, ret_ty_bits)
            .map_err(|e| message(&e))?;
        Ok((bytes, symbolic))
    }

    fn inspect_mem_read(
        &self,
        ctx: &SymContext,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value: &RustBV,
        endness: &str,
    ) -> Option<RustBV> {
        let value_ast = to_claripy(value)?;
        let mutated = self
            .call_inspect_mem_read(state_id, when, addr, size, Some(&value_ast), endness)
            .ok()??;
        from_claripy_any(ctx, &mutated)
    }

    fn inspect_mem_write(
        &self,
        ctx: &SymContext,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value: &RustBV,
        endness: &str,
    ) -> Option<RustBV> {
        let value_ast = to_claripy(value)?;
        let mutated = self
            .call_inspect_mem_write(state_id, when, addr, size, Some(&value_ast), endness)
            .ok()??;
        from_claripy_any(ctx, &mutated)
    }

    fn inspect_reg_read(&self, state_id: i64, when: &str, offset: u32, size: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value) else { return };
        note_inspect_error(
            self.call_inspect_reg_read(state_id, when, offset, size, Some(&ast)),
            "reg_read",
        );
    }

    fn inspect_reg_write(&self, state_id: i64, when: &str, offset: u32, size: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value) else { return };
        note_inspect_error(
            self.call_inspect_reg_write(state_id, when, offset, size, Some(&ast)),
            "reg_write",
        );
    }

    fn inspect_tmp_read(&self, state_id: i64, when: &str, tmp: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value) else { return };
        note_inspect_error(
            self.call_inspect_tmp_read(state_id, when, tmp, Some(&ast)),
            "tmp_read",
        );
    }

    fn inspect_tmp_write(&self, state_id: i64, when: &str, tmp: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value) else { return };
        note_inspect_error(
            self.call_inspect_tmp_write(state_id, when, tmp, Some(&ast)),
            "tmp_write",
        );
    }

    fn inspect_expr(&self, state_id: i64, when: &str, value: &RustBV) {
        let Some(ast) = to_claripy(value) else { return };
        note_inspect_error(self.call_inspect_expr(state_id, when, Some(&ast)), "expr");
    }

    fn inspect_exit(&self, state_id: i64, when: &str, target: u64, jumpkind: &str, guard: &RustBV) {
        let Some(ast) = to_claripy(guard) else { return };
        note_inspect_error(
            self.call_inspect_exit(state_id, when, target, jumpkind, Some(&ast)),
            "exit",
        );
    }

    fn inspect_address_concretization(
        &self,
        state_id: i64,
        when: &str,
        action: &str,
        addr: &RustBV,
        result: Option<Vec<u64>>,
    ) {
        let Some(ast) = to_claripy(addr) else { return };
        note_inspect_error(
            self.call_inspect_address_concretization(state_id, when, action, &ast, result),
            "address_concretization",
        );
    }

    fn inspect_symbolic_variable(
        &self,
        state_id: i64,
        when: &str,
        name: &str,
        size_bits: u32,
        value: &RustBV,
    ) {
        let Some(ast) = to_claripy(value) else { return };
        note_inspect_error(
            self.call_inspect_symbolic_variable(state_id, when, name, size_bits, &ast),
            "symbolic_variable",
        );
    }

    fn inspect_instruction(&self, state_id: i64, when: &str, addr: u64) {
        note_inspect_error(
            self.call_inspect_instruction(state_id, when, addr),
            "instruction",
        );
    }

    fn inspect_irsb(&self, state_id: i64, when: &str, addr: u64) {
        note_inspect_error(self.call_inspect_irsb(state_id, when, addr), "irsb");
    }

    fn inspect_statement(&self, state_id: i64, when: &str, stmt_idx: u32) {
        note_inspect_error(
            self.call_inspect_statement(state_id, when, stmt_idx),
            "statement",
        );
    }

    fn inspect_call(&self, state_id: i64, when: &str, function_address: u64) {
        note_inspect_error(
            self.call_inspect_call(state_id, when, function_address),
            "call",
        );
    }

    fn inspect_return(&self, state_id: i64, when: &str, function_address: u64) {
        note_inspect_error(
            self.call_inspect_return(state_id, when, function_address),
            "return",
        );
    }

    fn inspect_vex_lift(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: Option<u32>,
        bytes: Option<&[u8]>,
    ) {
        if let Err(e) = self.call_inspect_vex_lift(state_id, when, addr, size, bytes) {
            log::debug!("vex_lift inspect ({when}) dispatch failed at 0x{addr:x}: {e}");
        }
    }
}

/// A breakpoint's override value as a `RustBV`. Unlike [`from_claripy`] it
/// does not check `is_claripy_ast` first, matching the conversion the
/// interpreter used for `mem_read_expr` / `mem_write_expr` overrides.
fn from_claripy_any(ctx: &SymContext, ast: &Py<PyAny>) -> Option<RustBV> {
    Python::attach(|py| claripy_to_rustbv(py, ast.bind(py), ctx).ok())
}
