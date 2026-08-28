//! Python-visible test entry points for the `call_inspect_*` dispatch family.
//!
//! Split out of `callbacks/mod.rs` (angr-5mnx3.8), which had grown to be the
//! largest file in the directory while the module's own header already
//! documented the angr-zel8z.2 decomposition into `config`/`dispatch`/
//! `events`/`inspect`. Nothing here is reachable from the engine: every item
//! exists so the marshalling round-trip of an `inspect` breakpoint can be
//! exercised from Python without driving a VEX dispatch site (uq4n.3). The
//! dispatch methods these forward to live in `callbacks/inspect.rs`; their
//! behavioural tests are in `callbacks/inspect_tests.rs`.
//!
//! Every entry is written against a `self.forward(..)` placeholder that
//! [`angr_macros::inspect_test_entries`] rewrites to
//! `call_inspect_<entry name>`, so the Rust method name, the Python-visible
//! name and the dispatch method under test all derive from one ident — a
//! copy-paste between the identically-typed pairs (mem_read/mem_write,
//! reg_read/reg_write, tmp_read/tmp_write, call/return) can no longer test the
//! sibling (angr-0jh0j.6). Each entry returns whatever the registered
//! breakpoint returned.

use angr_macros::inspect_test_entries;
use pyo3::prelude::*;

use super::PythonCallbacks;

inspect_test_entries! {
    PythonCallbacks =>

    /// Test entry point: invoke the registered mem_read callback directly.
    #[pyo3(signature = (state_id, when, addr, size, value_ast, endness))]
    pub fn mem_read(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.forward(state_id, when, addr, size, value_ast.as_ref(), endness)
    }

    /// Test entry point: invoke the registered mem_write callback directly.
    #[pyo3(signature = (state_id, when, addr, size, value_ast, endness))]
    pub fn mem_write(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.forward(state_id, when, addr, size, value_ast.as_ref(), endness)
    }

    /// Test entry point: invoke the registered reg_read callback directly.
    #[pyo3(signature = (state_id, when, offset, size, value_ast))]
    pub fn reg_read(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, offset, size, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered reg_write callback directly.
    #[pyo3(signature = (state_id, when, offset, size, value_ast))]
    pub fn reg_write(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, offset, size, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered instruction callback directly.
    pub fn instruction(&self, state_id: i64, when: &str, addr: u64) -> PyResult<()> {
        self.forward(state_id, when, addr)
    }

    /// Test entry point: invoke the registered irsb callback directly.
    pub fn irsb(&self, state_id: i64, when: &str, addr: u64) -> PyResult<()> {
        self.forward(state_id, when, addr)
    }

    /// Test entry point: invoke the registered exit callback directly.
    #[pyo3(signature = (state_id, when, target, jumpkind, guard_ast))]
    pub fn exit(
        &self,
        state_id: i64,
        when: &str,
        target: u64,
        jumpkind: &str,
        guard_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, target, jumpkind, guard_ast.as_ref())
    }

    /// Test entry point: invoke the registered call callback directly.
    pub fn call(&self, state_id: i64, when: &str, function_address: u64) -> PyResult<()> {
        self.forward(state_id, when, function_address)
    }

    /// Test entry point: invoke the registered return callback directly.
    pub fn r#return(&self, state_id: i64, when: &str, function_address: u64) -> PyResult<()> {
        self.forward(state_id, when, function_address)
    }

    /// Test entry point: invoke the registered tmp_read callback directly.
    #[pyo3(signature = (state_id, when, tmp_num, value_ast))]
    pub fn tmp_read(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, tmp_num, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered tmp_write callback directly.
    #[pyo3(signature = (state_id, when, tmp_num, value_ast))]
    pub fn tmp_write(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, tmp_num, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered statement callback directly.
    pub fn statement(&self, state_id: i64, when: &str, stmt_idx: u32) -> PyResult<()> {
        self.forward(state_id, when, stmt_idx)
    }

    /// Test entry point: invoke the registered expr callback directly.
    #[pyo3(signature = (state_id, when, expr_result))]
    pub fn expr(
        &self,
        state_id: i64,
        when: &str,
        expr_result: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, expr_result.as_ref())
    }

    /// Test entry point: invoke the address_concretization callback directly.
    #[pyo3(signature = (state_id, when, action, addr_ast, result))]
    pub fn address_concretization(
        &self,
        state_id: i64,
        when: &str,
        action: &str,
        addr_ast: Py<PyAny>,
        result: Option<Vec<u64>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, action, &addr_ast, result)
    }

    /// Test entry point: invoke the symbolic_variable callback directly.
    #[pyo3(signature = (state_id, when, name, size, expr_ast))]
    pub fn symbolic_variable(
        &self,
        state_id: i64,
        when: &str,
        name: &str,
        size: u32,
        expr_ast: Py<PyAny>,
    ) -> PyResult<()> {
        self.forward(state_id, when, name, size, &expr_ast)
    }

    /// Test entry point: invoke the registered fork callback directly.
    pub fn fork(&self, state_id: i64, when: &str) -> PyResult<()> {
        self.forward(state_id, when)
    }

    /// Test entry point: invoke the registered constraints callback directly.
    ///
    /// `guard_ast` is a claripy AST standing in for the branch guard the
    /// production caller passes as the state's own `RustBV` — a type Python
    /// cannot construct — so it is imported into a throwaway `SymContext`
    /// first. What that leaves under test is the marshalling half of
    /// `call_inspect_constraints`: the `assumed_guard_to_claripy` export
    /// (including the `is_true == false` `claripy.Not(..)` wrap) and the
    /// `(state_id, when, [constraint])` call shape.
    #[pyo3(signature = (state_id, when, guard_ast, is_true))]
    pub fn constraints(
        &self,
        py: Python<'_>,
        state_id: i64,
        when: &str,
        guard_ast: &Bound<'_, PyAny>,
        is_true: bool,
    ) -> PyResult<()> {
        let ctx = crate::symbolic::SymContext::new();
        let guard = crate::claripy_bridge::claripy_to_rustbv(py, guard_ast, &ctx)
            .map_err(|e| crate::claripy_bridge::ast_import_err("inspect constraints guard", e))?;
        self.forward(state_id, when, &guard, is_true)
    }

    /// Test entry point: invoke the registered vex_lift callback directly.
    ///
    /// Mirrors the native libVEX lift path's two fires: BEFORE passes
    /// `size=None` plus the byte buffer handed to libVEX, AFTER passes the
    /// lifted IRSB's size and no buffer.
    #[pyo3(signature = (state_id, when, addr, size=None, buff=None))]
    pub fn vex_lift(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: Option<u32>,
        buff: Option<Vec<u8>>,
    ) -> PyResult<()> {
        self.forward(state_id, when, addr, size, buff.as_deref())
    }
}
