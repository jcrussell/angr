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

use crate::claripy_bridge::{claripy_to_rustbv, BridgeError};
use crate::symbolic::{RustBV, RustBVHandle, RustSymbolTable, SymContext};

/// Convert a BridgeError to a PyErr.
impl From<BridgeError> for PyErr {
    fn from(err: BridgeError) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}

/// Python-accessible Rust solver context.
///
/// This wraps a Z3-backed SymContext and provides solver operations
/// that can be called from Python with claripy ASTs.
///
/// With z3-rs 0.19+, we don't need to manage a separate Z3 context -
/// it's automatically handled via thread-local storage.
#[pyclass(unsendable)]
pub struct RustSolverContext {
    // Inner context (same structure whether Z3 is enabled or not)
    inner: Box<SolverInner>,
}

struct SolverInner {
    sym_ctx: SymContext,
    /// Symbol table for handle-based API (claripy bypass).
    symbol_table: RustSymbolTable,
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
                sym_ctx: SymContext::new(),
                symbol_table: RustSymbolTable::new(),
            }),
        }
    }

    /// Add a constraint from a claripy AST.
    ///
    /// The constraint should be a 1-bit (boolean) value. For wider values,
    /// we interpret them as "value != 0" to maintain compatibility with
    /// claripy's flexible constraint handling.
    pub fn add_constraint_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;

        #[cfg(feature = "vex-engine-z3")]
        {
            if bv.width() == 1 {
                // Standard boolean constraint
                self.inner.sym_ctx.assume_true(&bv);
            } else {
                // For wider values, interpret as "value != 0"
                // This is consistent with how claripy handles such constraints
                let zero = RustBV::concrete(0, bv.width());
                let neq = bv.ne(&zero, &self.inner.sym_ctx);
                self.inner.sym_ctx.assume_true(&neq);
            }
        }

        Ok(())
    }

    /// Add multiple constraints from claripy ASTs.
    pub fn add_constraints(
        &self,
        py: Python<'_>,
        asts: &Bound<'_, PyList>,
    ) -> PyResult<()> {
        for ast in asts.iter() {
            self.add_constraint_ast(py, &ast)?;
        }
        Ok(())
    }

    /// Check if the current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.inner.sym_ctx.is_sat()
    }

    /// Evaluate a claripy AST to a single concrete value.
    ///
    /// Returns None if unsatisfiable or the expression cannot be evaluated.
    /// For values > 128 bits, use eval_wide which returns a Python int.
    pub fn eval(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Option<PyObject>> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        let width = bv.width();

        // For narrow values (<= 128 bits), use the fast path
        if width <= 128 {
            match self.inner.sym_ctx.eval(&bv) {
                Some(v) => Ok(Some(v.into_pyobject(py)?.into())),
                None => Ok(None),
            }
        } else {
            // For wide values, use the wide path that returns bytes
            match self.inner.sym_ctx.eval_wide(&bv) {
                Some(bytes) => {
                    // Convert bytes to Python int using int.from_bytes
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
    ) -> PyResult<Vec<u128>> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        Ok(self.inner.sym_ctx.eval_upto(&bv, n))
    }

    /// Get the minimum value of a claripy AST.
    #[pyo3(signature = (ast, signed=false))]
    pub fn min(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        signed: bool,
    ) -> PyResult<Option<u128>> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        Ok(self.inner.sym_ctx.min(&bv, signed))
    }

    /// Get the maximum value of a claripy AST.
    #[pyo3(signature = (ast, signed=false))]
    pub fn max(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        signed: bool,
    ) -> PyResult<Option<u128>> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        Ok(self.inner.sym_ctx.max(&bv, signed))
    }

    /// Check if the given AST is definitely true.
    pub fn is_true(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<bool> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        if bv.width() != 1 {
            return Ok(false);
        }

        // Definitely true if it can only be true
        let can_be_false = self.inner.sym_ctx.can_be_false(&bv);
        Ok(!can_be_false)
    }

    /// Check if the given AST is definitely false.
    pub fn is_false(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<bool> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        if bv.width() != 1 {
            return Ok(false);
        }

        // Definitely false if it can only be false
        let can_be_true = self.inner.sym_ctx.can_be_true(&bv);
        Ok(!can_be_true)
    }

    /// Check if a specific value is a valid solution for an AST.
    pub fn solution(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        value: u128,
    ) -> PyResult<bool> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        Ok(self.inner.sym_ctx.solution(&bv, value))
    }

    /// Save solver state for temporary constraints.
    pub fn push(&self) {
        self.inner.sym_ctx.push();
    }

    /// Restore solver state.
    pub fn pop(&self) {
        self.inner.sym_ctx.pop();
    }

    /// Fork the solver context.
    ///
    /// Returns a new RustSolverContext with the same constraints and symbol table.
    /// With z3-rs 0.19+, all solvers on the same thread share the
    /// thread-local Z3 context.
    pub fn fork(&self) -> Self {
        let forked_sym_ctx = self.inner.sym_ctx.fork();
        let forked_symbol_table = self.inner.symbol_table.fork();
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: forked_sym_ctx,
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
            self.inner.sym_ctx.pop();
        }
    }

    /// Add a constraint that a 1-bit value is true.
    ///
    /// Used for applying branch constraints during fork processing.
    pub fn assume_true_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        if bv.width() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "constraint must be a 1-bit value, got {} bits",
                bv.width()
            )));
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            self.inner.sym_ctx.assume_true(&bv);
        }
        Ok(())
    }

    /// Add a constraint that a 1-bit value is false.
    ///
    /// Used for applying negated branch constraints during fork processing.
    pub fn assume_false_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        if bv.width() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "constraint must be a 1-bit value, got {} bits",
                bv.width()
            )));
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            self.inner.sym_ctx.assume_false(&bv);
        }
        Ok(())
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.inner.sym_ctx.num_constraints()
    }

    /// Create a new symbolic bitvector name.
    ///
    /// Returns a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        self.inner.sym_ctx.unique_name(base)
    }

    /// Check if Z3 is available.
    #[staticmethod]
    pub fn z3_available() -> bool {
        cfg!(feature = "vex-engine-z3")
    }

    /// Get the unsat core as indices of constraints added.
    ///
    /// Returns the indices of constraints that form the unsatisfiable core.
    /// Call this after checking satisfiability and finding UNSAT.
    pub fn unsat_core(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.sym_ctx.unsat_core())
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
        self.inner.symbol_table.create_symbolic(&self.inner.sym_ctx, name, width)
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
        self.inner.sym_ctx.eval(&bv)
    }

    /// Get the minimum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn min_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.inner.symbol_table.get(handle_id)?;
        self.inner.sym_ctx.min(&bv, signed)
    }

    /// Get the maximum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn max_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.inner.symbol_table.get(handle_id)?;
        self.inner.sym_ctx.max(&bv, signed)
    }

    /// Evaluate a handle and return up to n solutions.
    pub fn eval_upto_handle(&self, handle_id: u64, n: usize) -> Vec<u128> {
        if let Some(bv) = self.inner.symbol_table.get(handle_id) {
            self.inner.sym_ctx.eval_upto(&bv, n)
        } else {
            Vec::new()
        }
    }

    /// Add a constraint from a handle (must be 1-bit).
    pub fn add_constraint_handle(&self, handle_id: u64) -> PyResult<()> {
        let bv = self.inner.symbol_table.get(handle_id).ok_or_else(|| {
            PyRuntimeError::new_err(format!("invalid handle id: {}", handle_id))
        })?;

        #[cfg(feature = "vex-engine-z3")]
        {
            if bv.width() == 1 {
                self.inner.sym_ctx.assume_true(&bv);
            } else {
                let zero = RustBV::concrete(0, bv.width());
                let neq = bv.ne(&zero, &self.inner.sym_ctx);
                self.inner.sym_ctx.assume_true(&neq);
            }
        }

        Ok(())
    }

    /// Check if a specific value is a valid solution for a handle.
    pub fn solution_handle(&self, handle_id: u64, value: u128) -> bool {
        if let Some(bv) = self.inner.symbol_table.get(handle_id) {
            self.inner.sym_ctx.solution(&bv, value)
        } else {
            false
        }
    }

    /// Get the number of handles in the symbol table.
    pub fn handle_count(&self) -> usize {
        self.inner.symbol_table.len()
    }

    // =========================================================================
    // Handle-based Arithmetic Operations
    // =========================================================================

    /// Add two handles and return a new handle.
    pub fn op_add(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_add(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Subtract two handles and return a new handle.
    pub fn op_sub(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_sub(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Multiply two handles and return a new handle.
    pub fn op_mul(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_mul(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned division of two handles.
    pub fn op_udiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_udiv(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed division of two handles.
    pub fn op_sdiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_sdiv(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned remainder of two handles.
    pub fn op_urem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_urem(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed remainder of two handles.
    pub fn op_srem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_srem(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Negation of a handle.
    pub fn op_neg(&self, a_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_neg(a_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Bitwise Operations
    // =========================================================================

    /// Bitwise AND of two handles.
    pub fn op_and(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_and(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Bitwise OR of two handles.
    pub fn op_or(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_or(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Bitwise XOR of two handles.
    pub fn op_xor(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_xor(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Bitwise NOT of a handle.
    pub fn op_not(&self, a_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_not(a_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Shift Operations
    // =========================================================================

    /// Left shift.
    pub fn op_shl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_shl(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Logical right shift.
    pub fn op_lshr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_lshr(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Arithmetic right shift.
    pub fn op_ashr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_ashr(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Rotate left.
    pub fn op_rotl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_rotl(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Rotate right.
    pub fn op_rotr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_rotr(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit handle).
    pub fn op_eq(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_eq(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Inequality comparison (returns 1-bit handle).
    pub fn op_ne(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_ne(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned less than.
    pub fn op_ult(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_ult(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned less than or equal.
    pub fn op_ule(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_ule(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned greater than.
    pub fn op_ugt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_ugt(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Unsigned greater than or equal.
    pub fn op_uge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_uge(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed less than.
    pub fn op_slt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_slt(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed less than or equal.
    pub fn op_sle(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_sle(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed greater than.
    pub fn op_sgt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_sgt(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Signed greater than or equal.
    pub fn op_sge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_sge(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    // =========================================================================
    // Handle-based Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn op_zero_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_zero_extend(a_id, to_width, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Sign-extend to a wider width.
    pub fn op_sign_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_sign_extend(a_id, to_width, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Truncate to a narrower width.
    pub fn op_truncate(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_truncate(a_id, to_width, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Extract bits [high:low] (inclusive).
    pub fn op_extract(&self, a_id: u64, high: u32, low: u32) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_extract(a_id, high, low, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// Concatenate two values (a becomes high bits).
    pub fn op_concat(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_concat(a_id, b_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }

    /// If-then-else: if cond then then_val else else_val.
    pub fn op_ite(&self, cond_id: u64, then_id: u64, else_id: u64) -> PyResult<RustBVHandle> {
        self.inner.symbol_table.op_ite(cond_id, then_id, else_id, &self.inner.sym_ctx)
            .ok_or_else(|| PyRuntimeError::new_err("invalid handle id"))
    }
}

impl Default for RustSolverContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RustSolverContext {
    /// Get a reference to the inner SymContext.
    ///
    /// This is used by the Rust VEX engine to share the solver context,
    /// ensuring branch constraints are properly tracked during execution.
    pub fn sym_context(&self) -> &SymContext {
        &self.inner.sym_ctx
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
