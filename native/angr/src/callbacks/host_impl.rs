//! [`EngineHost`] for `PythonCallbacks`: the interpreter's services, served
//! from Python.
//!
//! Every Python object the interpreter used to build or unpack itself is
//! built and unpacked here instead: claripy ASTs for breakpoints and for
//! symbolic loads, and the `PyErr`s that become `String` messages.

use super::*;
use crate::claripy_bridge::{claripy_module, claripy_to_rustbv, rustbv_to_claripy};
use crate::interpreter::{
    EngineHost, HostLoad, HostPage, InspectBit, InspectView, LiftBlockOutcome,
};
use crate::symbolic::{RustBV, SymContext};

/// `value` as a claripy AST for the `event` breakpoint, or `None` when the
/// export fails. The breakpoint is then skipped, since a missing claripy or an
/// unexportable value must not halt exploration, but the skip is logged at
/// warn: the user's breakpoint silently not firing is the wrong answer
/// (angr-vmi2l.205).
fn to_claripy(value: &RustBV, event: &str) -> Option<Py<PyAny>> {
    Python::attach(|py| {
        let exported =
            claripy_module(py).and_then(|claripy_mod| rustbv_to_claripy(py, value, &claripy_mod));
        Some(silent_default!(
            cat_c,
            exported,
            return None,
            |err| "state.inspect '{event}' breakpoint skipped: exporting its value to claripy failed: {err}"
        ))
    })
}

/// The override a value-injecting breakpoint (`mem_read` / `mem_write`) left
/// in its `<event>_expr` attr, imported as a `RustBV`.
///
/// `None` when the breakpoint left the value alone, when the dispatch raised
/// (logged through [`note_inspect_error`] like every sibling event), or when
/// the importer rejects the override (a Python int, an FP AST, an op it does
/// not take). That last case logs at warn naming the event and the override,
/// since the user asked for a value the engine then does not use
/// (angr-vmi2l.205). The caller keeps its original value whenever this is
/// `None`.
fn inspect_override(
    ctx: &SymContext,
    dispatched: PyResult<Option<Py<PyAny>>>,
    event: &str,
) -> Option<RustBV> {
    let mutated = match dispatched {
        Ok(mutated) => mutated?,
        Err(err) => {
            note_inspect_error(Err(err), event);
            return None;
        }
    };
    Python::attach(|py| {
        let ast = mutated.bind(py);
        Some(silent_default!(
            cat_c,
            claripy_to_rustbv(py, ast, ctx),
            return None,
            |err| "state.inspect '{event}' override {ast} rejected ({err}); keeping the original value"
        ))
    })
}

/// A claripy AST from Python as a `RustBV`, or `None` when it is not one or
/// does not convert (the caller then mints a fresh symbol or keeps its value).
fn from_claripy(ctx: &SymContext, ast: &Py<PyAny>) -> Option<RustBV> {
    // SILENT(cat-b): a host load the importer rejects falls back to its
    // bytes or a fresh symbol, the caller's documented `None` arm.
    Python::attach(|py| claripy_to_rustbv(py, ast.bind(py), ctx).ok())
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

    fn inspect_wants_view(&self) -> bool {
        self.inspect_wants_view
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn publish_inspect_view(&self, view: Option<InspectView>) {
        super::inspect_view::publish(view);
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
        // The callback's pyvex lift enters the same libVEX as a native lift on
        // another worker; hold the shared lift lock across it (angr-mx00p.45).
        let _lift_guard = crate::libvex_lifter::lift_lock();
        self.call_lift_block(addr, opt_level, dirty_bytes)
            .map_err(|e| message(&e))
    }

    fn dirty_call(
        &self,
        ctx: &SymContext,
        name: &str,
        args: &[u64],
        ret_ty_bits: u32,
    ) -> Result<HostLoad, String> {
        let entry = self
            .call_dirty_call(name, args, ret_ty_bits)
            .map_err(|e| message(&e))?;
        Ok(host_load(ctx, entry))
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
        condition: Option<&RustBV>,
    ) -> Option<RustBV> {
        let value_ast = to_claripy(value, "mem_read")?;
        let condition_ast = match condition {
            Some(guard) => Some(to_claripy(guard, "mem_read")?),
            None => None,
        };
        let dispatched = self.call_inspect_mem_read(
            state_id,
            when,
            addr,
            size,
            Some(&value_ast),
            endness,
            condition_ast.as_ref(),
        );
        inspect_override(ctx, dispatched, "mem_read")
    }

    fn inspect_mem_write(
        &self,
        ctx: &SymContext,
        state_id: i64,
        when: &str,
        addr: &RustBV,
        size: u32,
        value: &RustBV,
        endness: &str,
        condition: Option<&RustBV>,
    ) -> Option<RustBV> {
        let value_ast = to_claripy(value, "mem_write")?;
        let condition_ast = match condition {
            Some(guard) => Some(to_claripy(guard, "mem_write")?),
            None => None,
        };
        let addr_obj = match addr.as_u64() {
            // u64 -> Python int cannot fail: its error type is `Infallible`.
            Some(addr) => Python::attach(|py| match addr.into_pyobject(py) {
                Ok(int) => int.into_any().unbind(),
                Err(never) => match never {},
            }),
            None => to_claripy(addr, "mem_write")?,
        };
        let dispatched = self.call_inspect_mem_write(
            state_id,
            when,
            &addr_obj,
            size,
            Some(&value_ast),
            endness,
            condition_ast.as_ref(),
        );
        inspect_override(ctx, dispatched, "mem_write")
    }

    fn inspect_reg_read(&self, state_id: i64, when: &str, offset: u32, size: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value, "reg_read") else {
            return;
        };
        note_inspect_error(
            self.call_inspect_reg_read(state_id, when, offset, size, Some(&ast)),
            "reg_read",
        );
    }

    fn inspect_reg_write(&self, state_id: i64, when: &str, offset: u32, size: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value, "reg_write") else {
            return;
        };
        note_inspect_error(
            self.call_inspect_reg_write(state_id, when, offset, size, Some(&ast)),
            "reg_write",
        );
    }

    fn inspect_tmp_read(&self, state_id: i64, when: &str, tmp: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value, "tmp_read") else {
            return;
        };
        note_inspect_error(
            self.call_inspect_tmp_read(state_id, when, tmp, Some(&ast)),
            "tmp_read",
        );
    }

    fn inspect_tmp_write(&self, state_id: i64, when: &str, tmp: u32, value: &RustBV) {
        let Some(ast) = to_claripy(value, "tmp_write") else {
            return;
        };
        note_inspect_error(
            self.call_inspect_tmp_write(state_id, when, tmp, Some(&ast)),
            "tmp_write",
        );
    }

    fn inspect_expr(&self, state_id: i64, when: &str, value: &RustBV) {
        let Some(ast) = to_claripy(value, "expr") else {
            return;
        };
        note_inspect_error(self.call_inspect_expr(state_id, when, Some(&ast)), "expr");
    }

    fn inspect_exit(&self, state_id: i64, when: &str, target: u64, jumpkind: &str, guard: &RustBV) {
        let Some(ast) = to_claripy(guard, "exit") else {
            return;
        };
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
        let Some(ast) = to_claripy(addr, "address_concretization") else {
            return;
        };
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
        let Some(ast) = to_claripy(value, "symbolic_variable") else {
            return;
        };
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
