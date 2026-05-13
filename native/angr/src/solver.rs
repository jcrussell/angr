//! PyO3-exposed Rust solver context.
//!
//! This module provides a Python-accessible constraint solver that uses
//! Z3 under the hood. It bridges claripy ASTs to Rust's SymContext.
//!
//! With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
//! to manage explicit context lifetimes.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use crate::claripy_bridge::{BridgeError, claripy_to_rustbv, try_extract_bvv};
use crate::symbolic::{RustBV, RustBVHandle, RustSymbolTable, SymContext};

/// Try to extract raw Z3_ast pointer from a claripy AST's z3 backend.
/// Returns the pointer as a non-zero usize, or an error if unavailable
/// or null. Guaranteeing non-null at the source means callers can pass
/// the value directly to `NonNull::new_unchecked` without re-checking.
/// This preserves claripy's original Z3 AST structure.
#[cfg(feature = "vex-engine-z3")]
fn extract_z3_ast_ptr(py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<usize> {
    let claripy = py.import("claripy")?;
    let z3_backend = claripy.getattr("backends")?.getattr("z3")?;
    let z3_obj = z3_backend.call_method1("convert", (ast,))?;
    let ast_ref = z3_obj.call_method0("as_ast")?;
    let ptr: usize = ast_ref.getattr("value")?.extract()?;
    if ptr == 0 {
        return Err(PyRuntimeError::new_err(
            "claripy z3 backend returned null Z3_ast pointer",
        ));
    }
    Ok(ptr)
}

/// Convert a BridgeError to a PyErr.
impl From<BridgeError> for PyErr {
    fn from(err: BridgeError) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Storage for SymContext: either owned or shared via Rc.
/// Shared mode allows Python callbacks to use the pending state's solver
/// directly (O(1)) instead of forking it (~3ms Z3 clone per callback).
enum SolverCtxStorage {
    Owned(SymContext),
    Shared(Rc<RefCell<SymContext>>),
}

/// Guard for borrowing SymContext from either storage variant.
/// Implements Deref<Target=SymContext> for transparent access.
enum SolverCtxGuard<'a> {
    Ref(&'a SymContext),
    Borrowed(Ref<'a, SymContext>),
}

impl<'a> std::ops::Deref for SolverCtxGuard<'a> {
    type Target = SymContext;
    fn deref(&self) -> &SymContext {
        match self {
            SolverCtxGuard::Ref(r) => r,
            SolverCtxGuard::Borrowed(b) => b,
        }
    }
}

/// Python-accessible Rust solver context.
///
/// This wraps a Z3-backed SymContext and provides solver operations
/// that can be called from Python with claripy ASTs.
///
/// Supports two modes:
/// - Owned: holds its own SymContext (normal case, forked contexts)
/// - Shared: borrows a state's solver via Rc (zero-cost callback solver)
///
/// With z3-rs 0.19+, we don't need to manage a separate Z3 context -
/// it's automatically handled via thread-local storage.
#[pyclass(unsendable)]
pub struct RustSolverContext {
    // Inner context (same structure whether Z3 is enabled or not)
    inner: Box<SolverInner>,
}

struct SolverInner {
    sym_ctx: SolverCtxStorage,
    /// Symbol table for handle-based API (claripy bypass).
    symbol_table: RustSymbolTable,
}

impl SolverInner {
    /// Get a guard for the SymContext, works for both owned and shared.
    fn ctx(&self) -> SolverCtxGuard<'_> {
        match &self.sym_ctx {
            SolverCtxStorage::Owned(ctx) => SolverCtxGuard::Ref(ctx),
            SolverCtxStorage::Shared(rc) => SolverCtxGuard::Borrowed(rc.borrow()),
        }
    }
}

#[pymethods]
impl RustSolverContext {
    /// Create a new Rust solver context.
    #[new]
    pub fn new() -> Self {
        // With z3-rs 0.19+, the Z3 context is thread-local
        // SymContext::new() handles the setup
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(SymContext::new()),
                symbol_table: RustSymbolTable::new(),
            }),
        }
    }

    /// Add a constraint from a claripy AST.
    ///
    /// Uses fast path when Z3 context is shared: extracts the raw Z3_ast
    /// from claripy's z3 backend and asserts it directly, preserving the
    /// original Z3 AST structure. Falls back to RustBV conversion otherwise.
    pub fn add_constraint_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.inner.ctx();

        // Fast path: try to extract raw Z3 AST from claripy's z3 backend.
        // This preserves the original AST structure (Python's claripy creates
        // different Z3 trees than our build_z3_ast), avoiding the structural
        // divergence that causes 3-7x slower Z3 solving.
        #[cfg(feature = "vex-engine-z3")]
        {
            if let Ok(z3_ast_ptr) = extract_z3_ast_ptr(py, ast) {
                // SAFETY: extract_z3_ast_ptr guarantees non-null on Ok.
                unsafe {
                    ctx.add_constraint_raw(z3_ast_ptr);
                }
                // Also track in RustBV for export (best-effort, non-critical)
                if let Ok(bv) = claripy_to_rustbv(py, ast, &*ctx) {
                    ctx.assumed_constraints_push(bv, true);
                }
                return Ok(());
            }
        }

        // Slow path: convert claripy AST to RustBV, then build Z3 AST
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;

        #[cfg(feature = "vex-engine-z3")]
        {
            if bv.width() == 1 {
                ctx.assume_true(&bv);
            } else {
                let zero = RustBV::concrete(0, bv.width());
                let neq = bv.ne(&zero, &*ctx);
                ctx.assume_true(&neq);
            }
        }

        Ok(())
    }

    /// Add multiple constraints from claripy ASTs.
    pub fn add_constraints(&self, py: Python<'_>, asts: &Bound<'_, PyList>) -> PyResult<()> {
        for ast in asts.iter() {
            self.add_constraint_ast(py, &ast)?;
        }
        Ok(())
    }

    /// Check if the current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.inner.ctx().is_sat()
    }

    /// Set the Z3 solver timeout in milliseconds.
    pub fn set_timeout(&self, timeout_ms: u32) {
        self.inner.ctx().set_timeout(timeout_ms);
    }

    /// Get the Z3 solver timeout in milliseconds.
    pub fn timeout_ms(&self) -> u32 {
        self.inner.ctx().timeout_ms()
    }

    /// Evaluate a claripy AST to a single concrete value.
    ///
    /// Returns None if unsatisfiable or the expression cannot be evaluated.
    /// For values > 128 bits, use eval_wide which returns a Python int.
    pub fn eval(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        // Fast path: concrete BVV doesn't need solver
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value.into_pyobject(py)?.into()));
        }

        let ctx = self.inner.ctx();

        // Try standard claripy → RustBV conversion first
        match claripy_to_rustbv(py, ast, &*ctx) {
            Ok(bv) => {
                let width = bv.width();
                if width <= 128 {
                    match ctx.eval(&bv) {
                        Some(v) => return Ok(Some(v.into_pyobject(py)?.into())),
                        None => {} // fall through to Z3 AST pointer path
                    }
                } else {
                    match ctx.eval_wide(&bv) {
                        Some(bytes) => {
                            let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                            let int_class = py.get_type::<pyo3::types::PyInt>();
                            let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                            return Ok(Some(py_int.into()));
                        }
                        None => {} // fall through to Z3 AST pointer path
                    }
                }
            }
            Err(_) => {}
        }

        // Z3 AST pointer fast path via shared context.
        // This handles: (1) expressions where claripy_to_rustbv creates a new
        // Z3 variable that has no constraints (e.g., stdin BVS from Python),
        // (2) complex expressions containing imported Z3 ASTs.
        // Using the original Z3 AST pointer preserves identity with
        // constraints already in the solver.
        #[cfg(feature = "vex-engine-z3")]
        {
            if let Ok(z3_ptr) = extract_z3_ast_ptr(py, ast) {
                return self.eval_z3_ast_ptr(py, z3_ptr, ast);
            }
        }
        Ok(None)
    }

    /// Evaluate a Z3 AST pointer directly in the solver context.
    #[cfg(feature = "vex-engine-z3")]
    fn eval_z3_ast_ptr(
        &self,
        py: Python<'_>,
        z3_ptr: usize,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        use z3::ast::Ast;
        let ctx = self.inner.ctx();

        // Get the bit width from claripy
        let width: u32 = match ast.getattr("length") {
            Ok(l) => l.extract().unwrap_or(64),
            Err(_) => 64,
        };

        // Build a RustBV::Symbolic wrapping this Z3 AST
        let z3_bv = unsafe {
            let raw = std::ptr::NonNull::new_unchecked(z3_ptr as *mut _);
            let z3_ctx = z3::Context::thread_local();
            z3::ast::BV::wrap(&z3_ctx, raw)
        };
        let bv = RustBV::Symbolic {
            id: 0,
            ast: z3_bv,
            width,
            name: Arc::from(""),
        };

        if width <= 128 {
            match ctx.eval(&bv) {
                Some(v) => Ok(Some(v.into_pyobject(py)?.into())),
                None => Ok(None),
            }
        } else {
            match ctx.eval_wide(&bv) {
                Some(bytes) => {
                    let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                    let int_class = py.get_type::<pyo3::types::PyInt>();
                    let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                    Ok(Some(py_int.into()))
                }
                None => Ok(None),
            }
        }
    }

    /// Evaluate a claripy AST and return up to n solutions.
    pub fn eval_upto(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        n: usize,
    ) -> PyResult<Py<PyAny>> {
        let result_list = pyo3::types::PyList::empty(py);

        // Fast path: concrete BVV has exactly one solution
        if n > 0 {
            if let Some((value, _width)) = try_extract_bvv(ast) {
                result_list.append(value.into_pyobject(py)?)?;
                return Ok(result_list.into());
            }
        }

        let width: u32 = ast
            .getattr("length")
            .and_then(|l| l.extract())
            .unwrap_or(64);
        let is_wide = width > 128;

        let ctx = self.inner.ctx();
        let bv = match claripy_to_rustbv(py, ast, &*ctx) {
            Ok(bv) => bv,
            Err(_) => {
                // Z3 fast path for complex expressions
                #[cfg(feature = "vex-engine-z3")]
                {
                    use z3::ast::Ast;
                    if let Ok(z3_ptr) = extract_z3_ast_ptr(py, ast) {
                        // SAFETY: extract_z3_ast_ptr guarantees non-null on Ok.
                        let z3_bv = unsafe {
                            let raw = std::ptr::NonNull::new_unchecked(z3_ptr as *mut _);
                            let z3_ctx = z3::Context::thread_local();
                            z3::ast::BV::wrap(&z3_ctx, raw)
                        };
                        RustBV::Symbolic {
                            id: 0,
                            ast: z3_bv,
                            width,
                            name: Arc::from(""),
                        }
                    } else {
                        return Ok(result_list.into());
                    }
                }
                #[cfg(not(feature = "vex-engine-z3"))]
                {
                    return Ok(result_list.into());
                }
            }
        };

        if is_wide {
            // Wide values: use eval_upto_wide to get full-precision bytes
            let results = ctx.eval_upto_wide(&bv, n);
            let int_class = py.get_type::<pyo3::types::PyInt>();
            for bytes in results {
                let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                result_list.append(py_int)?;
            }
        } else {
            // Standard path: u128 values
            let results = ctx.eval_upto(&bv, n);
            for v in results {
                result_list.append(v.into_pyobject(py)?)?;
            }
        }
        Ok(result_list.into())
    }

    /// Get the minimum value of a claripy AST.
    #[pyo3(signature = (ast, signed=false))]
    pub fn min(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        signed: bool,
    ) -> PyResult<Option<u128>> {
        // Fast path: concrete BVV
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value));
        }
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        Ok(ctx.min(&bv, signed))
    }

    /// Get the maximum value of a claripy AST.
    #[pyo3(signature = (ast, signed=false))]
    pub fn max(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        signed: bool,
    ) -> PyResult<Option<u128>> {
        // Fast path: concrete BVV
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value));
        }
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        Ok(ctx.max(&bv, signed))
    }

    /// Check if the given AST is definitely true.
    pub fn is_true(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        if bv.width() != 1 {
            return Ok(false);
        }

        // Definitely true if it can only be true
        let can_be_false = ctx.can_be_false(&bv);
        Ok(!can_be_false)
    }

    /// Check if the given AST is definitely false.
    pub fn is_false(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        if bv.width() != 1 {
            return Ok(false);
        }

        // Definitely false if it can only be false
        let can_be_true = ctx.can_be_true(&bv);
        Ok(!can_be_true)
    }

    /// Check if a specific value is a valid solution for an AST.
    pub fn solution(&self, py: Python<'_>, ast: &Bound<'_, PyAny>, value: u128) -> PyResult<bool> {
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        Ok(ctx.solution(&bv, value))
    }

    /// Save solver state for temporary constraints.
    pub fn push(&self) {
        self.inner.ctx().push();
    }

    /// Restore solver state.
    pub fn pop(&self) {
        self.inner.ctx().pop();
    }

    /// Fork the solver context.
    ///
    /// Returns a new RustSolverContext with the same constraints and symbol table.
    /// With z3-rs 0.19+, all solvers on the same thread share the
    /// thread-local Z3 context.
    pub fn fork(&self) -> Self {
        let ctx = self.inner.ctx();
        let forked_sym_ctx = ctx.fork();
        let forked_symbol_table = self.inner.symbol_table.fork();
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(forked_sym_ctx),
                symbol_table: forked_symbol_table,
            }),
        }
    }

    /// Pop solver state multiple times.
    ///
    /// This is used for deferred fork processing to restore solver state
    /// to before specific branch constraints were added.
    pub fn pop_to_level(&self, target_level: u32, current_level: u32) {
        let pops = current_level.saturating_sub(target_level);
        for _ in 0..pops {
            self.inner.ctx().pop();
        }
    }

    /// Add a constraint that a 1-bit value is true.
    ///
    /// Used for applying branch constraints during fork processing.
    pub fn assume_true_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        if bv.width() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "constraint must be a 1-bit value, got {} bits",
                bv.width()
            )));
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            ctx.assume_true(&bv);
        }
        Ok(())
    }

    /// Add a constraint that a 1-bit value is false.
    ///
    /// Used for applying negated branch constraints during fork processing.
    pub fn assume_false_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        if bv.width() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "constraint must be a 1-bit value, got {} bits",
                bv.width()
            )));
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            ctx.assume_false(&bv);
        }
        Ok(())
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.inner.ctx().num_constraints()
    }

    /// Create a new symbolic bitvector name.
    ///
    /// Returns a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        self.inner.ctx().unique_name(base)
    }

    /// Check if Z3 is available.
    #[staticmethod]
    pub fn z3_available() -> bool {
        cfg!(feature = "vex-engine-z3")
    }

    /// Check if this solver context is shared (borrowed from a pending state).
    /// Shared solvers write constraints directly to the state, so constraint
    /// sync after callbacks can be skipped.
    pub fn is_shared(&self) -> bool {
        matches!(self.inner.sym_ctx, SolverCtxStorage::Shared(_))
    }

    /// Get the unsat core as indices of constraints added.
    ///
    /// Returns the indices of constraints that form the unsatisfiable core.
    /// Call this after checking satisfiability and finding UNSAT.
    pub fn unsat_core(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.ctx().unsat_core())
    }

    /// Get all Z3 solver assertions as strings.
    ///
    /// Returns string representations of all active constraints in the Z3 solver.
    /// Useful for debugging and for verifying constraint sync between Rust and Python.
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        self.inner.ctx().get_all_constraints_str()
    }

    /// Get the number of assertions in the Z3 solver.
    ///
    /// Returns the total count of active constraints.
    pub fn z3_assertion_count(&self) -> usize {
        self.inner.ctx().z3_assertion_count()
    }

    /// Export constraints as serialized data for Python sync.
    ///
    /// This returns a list of (description, is_trackable) tuples for each
    /// constraint in the solver. The descriptions can be used for debugging
    /// and the is_trackable flag indicates if the constraint could be
    /// reconstructed from tracked handles.
    ///
    /// Note: Full Z3->claripy AST conversion is complex. This method provides
    /// constraint info for debugging. The primary sync mechanism is through
    /// the bidirectional constraint flow via add_constraints_to_pending.
    pub fn export_constraint_info(&self) -> Vec<(String, bool)> {
        self.inner
            .ctx()
            .get_all_constraints_str()
            .into_iter()
            .map(|s| (s, true)) // All Z3 constraints are trackable
            .collect()
    }

    /// Get the number of new constraints added since last sync.
    ///
    /// This helps track constraint growth during callbacks.
    pub fn constraint_delta(&self, baseline: usize) -> usize {
        let current = self.inner.ctx().num_constraints();
        current.saturating_sub(baseline)
    }

    // =========================================================================
    // Handle-based API (Claripy Bypass)
    // These methods allow Python to perform symbolic operations without
    // converting to/from claripy ASTs, providing significant speedups.
    // =========================================================================

    /// Create a new symbolic bitvector and return a handle.
    ///
    /// This bypasses claripy.BVS() for native Rust symbolic value creation.
    pub fn create_symbolic(&self, name: &str, width: u32) -> RustBVHandle {
        let ctx = self.inner.ctx();
        self.inner.symbol_table.create_symbolic(&*ctx, name, width)
    }

    /// Create a new concrete bitvector and return a handle.
    ///
    /// This bypasses claripy.BVV() for native Rust concrete value creation.
    pub fn create_concrete(&self, value: u128, width: u32) -> RustBVHandle {
        self.inner.symbol_table.create_concrete(value, width)
    }

    /// Evaluate a handle to get a concrete value.
    ///
    /// Returns None if unsatisfiable or the value cannot be evaluated.
    pub fn eval_handle(&self, handle_id: u64) -> Option<u128> {
        let bv = self.inner.symbol_table.get(handle_id)?;
        self.inner.ctx().eval(&bv)
    }

    /// Get the minimum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn min_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.inner.symbol_table.get(handle_id)?;
        self.inner.ctx().min(&bv, signed)
    }

    /// Get the maximum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn max_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.inner.symbol_table.get(handle_id)?;
        self.inner.ctx().max(&bv, signed)
    }

    /// Evaluate a handle and return up to n solutions.
    pub fn eval_upto_handle(&self, handle_id: u64, n: usize) -> Vec<u128> {
        if let Some(bv) = self.inner.symbol_table.get(handle_id) {
            self.inner.ctx().eval_upto(&bv, n)
        } else {
            Vec::new()
        }
    }

    /// Add a constraint from a handle (must be 1-bit).
    pub fn add_constraint_handle(&self, handle_id: u64) -> PyResult<()> {
        let bv =
            self.inner.symbol_table.get(handle_id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("invalid handle id: {}", handle_id))
            })?;

        #[cfg(feature = "vex-engine-z3")]
        {
            let ctx = self.inner.ctx();
            if bv.width() == 1 {
                ctx.assume_true(&bv);
            } else {
                let zero = RustBV::concrete(0, bv.width());
                let neq = bv.ne(&zero, &*ctx);
                ctx.assume_true(&neq);
            }
        }

        Ok(())
    }

    /// Check if a specific value is a valid solution for a handle.
    pub fn solution_handle(&self, handle_id: u64, value: u128) -> bool {
        if let Some(bv) = self.inner.symbol_table.get(handle_id) {
            self.inner.ctx().solution(&bv, value)
        } else {
            false
        }
    }

    /// Get the number of handles in the symbol table.
    pub fn handle_count(&self) -> usize {
        self.inner.symbol_table.len()
    }

    /// Convert a claripy AST to a handle.
    ///
    /// This performs a one-time conversion of a claripy AST to a RustBV,
    /// storing it in the symbol table and returning a handle. Subsequent
    /// operations can use the handle directly, avoiding repeated AST traversal.
    ///
    /// This is the key optimization for the callback return path: instead of
    /// returning a claripy AST that must be traversed on every use, we convert
    /// once and return a handle for O(1) lookups.
    pub fn claripy_ast_to_handle(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        let bv = claripy_to_rustbv(py, ast, &*ctx)?;
        Ok(self.inner.symbol_table.insert(bv))
    }

    // =========================================================================
    // Handle-based Arithmetic Operations
    // =========================================================================

    /// Add two handles and return a new handle.
    pub fn op_add(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_add(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Subtract two handles and return a new handle.
    pub fn op_sub(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_sub(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Multiply two handles and return a new handle.
    pub fn op_mul(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_mul(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned division of two handles.
    pub fn op_udiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_udiv(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed division of two handles.
    pub fn op_sdiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_sdiv(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned remainder of two handles.
    pub fn op_urem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_urem(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed remainder of two handles.
    pub fn op_srem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_srem(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Negation of a handle.
    pub fn op_neg(&self, a_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_neg(a_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Bitwise Operations
    // =========================================================================

    /// Bitwise AND of two handles.
    pub fn op_and(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_and(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Bitwise OR of two handles.
    pub fn op_or(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_or(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Bitwise XOR of two handles.
    pub fn op_xor(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_xor(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Bitwise NOT of a handle.
    pub fn op_not(&self, a_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_not(a_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Shift Operations
    // =========================================================================

    /// Left shift.
    pub fn op_shl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_shl(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Logical right shift.
    pub fn op_lshr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_lshr(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Arithmetic right shift.
    pub fn op_ashr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_ashr(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Rotate left.
    pub fn op_rotl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_rotl(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Rotate right.
    pub fn op_rotr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_rotr(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit handle).
    pub fn op_eq(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_eq(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Inequality comparison (returns 1-bit handle).
    pub fn op_ne(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_ne(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned less than.
    pub fn op_ult(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_ult(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned less than or equal.
    pub fn op_ule(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_ule(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned greater than.
    pub fn op_ugt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_ugt(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned greater than or equal.
    pub fn op_uge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_uge(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed less than.
    pub fn op_slt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_slt(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed less than or equal.
    pub fn op_sle(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_sle(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed greater than.
    pub fn op_sgt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_sgt(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed greater than or equal.
    pub fn op_sge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_sge(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn op_zero_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_zero_extend(a_id, to_width, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Sign-extend to a wider width.
    pub fn op_sign_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_sign_extend(a_id, to_width, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Truncate to a narrower width.
    pub fn op_truncate(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_truncate(a_id, to_width, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Extract bits [high:low] (inclusive).
    pub fn op_extract(&self, a_id: u64, high: u32, low: u32) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_extract(a_id, high, low, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Concatenate two values (a becomes high bits).
    pub fn op_concat(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_concat(a_id, b_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// If-then-else: if cond then then_val else else_val.
    pub fn op_ite(&self, cond_id: u64, then_id: u64, else_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.inner.ctx();
        self.inner
            .symbol_table
            .op_ite(cond_id, then_id, else_id, &*ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Solver Profiling Stats
    // =========================================================================

    /// Get global Z3 solver profiling stats as a dict.
    #[staticmethod]
    pub fn get_solver_stats() -> std::collections::HashMap<String, u64> {
        crate::symbolic::get_solver_stats()
    }

    /// Reset global Z3 solver profiling stats to zero.
    #[staticmethod]
    pub fn reset_solver_stats() {
        crate::symbolic::reset_solver_stats()
    }
}

impl Default for RustSolverContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RustSolverContext {
    /// Create a RustSolverContext from an existing SymContext.
    ///
    /// This is used when forking solver contexts during callback handling,
    /// allowing Python callbacks to inherit the full constraint context
    /// from Rust exploration.
    pub fn from_sym_context(sym_ctx: SymContext) -> Self {
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(sym_ctx),
                symbol_table: RustSymbolTable::new(),
            }),
        }
    }

    /// Create a RustSolverContext from an existing SymContext with a forked symbol table.
    ///
    /// This preserves both constraints and symbolic variable mappings.
    pub fn from_sym_context_with_symbols(
        sym_ctx: SymContext,
        symbol_table: RustSymbolTable,
    ) -> Self {
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(sym_ctx),
                symbol_table,
            }),
        }
    }

    /// Create a RustSolverContext that shares the solver from a state's Rc<RefCell<SymContext>>.
    ///
    /// This is O(1) — just an Rc clone (reference count increment) instead of
    /// a full Z3 solver clone (~3ms). The shared solver writes constraints directly
    /// to the pending state, eliminating the need for post-callback constraint sync.
    ///
    /// Safety: Only use when Rust exploration is suspended (during Python callbacks).
    pub fn from_shared_sym_context(shared: Rc<RefCell<SymContext>>) -> Self {
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Shared(shared),
                symbol_table: RustSymbolTable::new(),
            }),
        }
    }

    /// Get a reference to the inner SymContext.
    ///
    /// This is used by the Rust VEX engine to share the solver context,
    /// ensuring branch constraints are properly tracked during execution.
    /// Only works for owned contexts; returns None for shared contexts.
    pub fn sym_context(&self) -> Option<&SymContext> {
        match &self.inner.sym_ctx {
            SolverCtxStorage::Owned(ctx) => Some(ctx),
            SolverCtxStorage::Shared(_) => None,
        }
    }

    /// Get a reference to the symbol table.
    ///
    /// This is used by the interpreter to look up handles returned from Python.
    pub fn symbol_table(&self) -> &RustSymbolTable {
        &self.inner.symbol_table
    }
}

/// Register the solver module with Python.
pub fn solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustSolverContext>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let _ctx = RustSolverContext::new();
    }

    #[test]
    fn test_solver_fork() {
        let ctx = RustSolverContext::new();
        let _forked = ctx.fork();
    }

    #[test]
    fn test_z3_available() {
        let available = RustSolverContext::z3_available();
        #[cfg(feature = "vex-engine-z3")]
        assert!(available);
        #[cfg(not(feature = "vex-engine-z3"))]
        assert!(!available);
    }
}
