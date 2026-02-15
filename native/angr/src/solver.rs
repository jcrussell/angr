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
use crate::symbolic::{RustBV, SymContext};

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
    pub fn eval(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Option<u128>> {
        let bv = claripy_to_rustbv(py, ast, &self.inner.sym_ctx)?;
        Ok(self.inner.sym_ctx.eval(&bv))
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
    /// Returns a new RustSolverContext with the same constraints.
    /// With z3-rs 0.19+, all solvers on the same thread share the
    /// thread-local Z3 context.
    pub fn fork(&self) -> Self {
        let forked_sym_ctx = self.inner.sym_ctx.fork();
        RustSolverContext {
            inner: Box::new(SolverInner {
                sym_ctx: forked_sym_ctx,
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
