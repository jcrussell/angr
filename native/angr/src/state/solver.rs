//! Solver/constraint-access cluster for `RustSimState`.
//!
//! The thin accessors that forward to the shared `Rc<RefCell<SymContext>>`:
//! the `solver()` handle itself, constraint accumulation (`add_constraint`),
//! satisfiability (`satisfiable` / `set_sat_cache`), and the concrete-value
//! queries (`eval` / `min` / `max`). Split out of `mod.rs` per the god-object
//! decomposition (angr-0mqkc.5); mirrors the `construction.rs` / `options.rs`
//! / `registers.rs` / `memory.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    /// Get a reference to the solver context.
    pub fn solver(&self) -> &Rc<RefCell<SymContext>> {
        &self.solver
    }

    /// Add a constraint.
    pub fn add_constraint(&self, constraint: RustBV) {
        let ctx = self.solver.borrow();
        ctx.assume_true(&constraint);
    }

    /// Check if current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        let ctx = self.solver.borrow();
        ctx.is_sat()
    }

    /// Prime the SAT cache (avoids redundant Z3 checks after branch forking).
    pub fn set_sat_cache(&self, value: bool) {
        self.solver.borrow().set_sat_cache(value);
    }

    /// Evaluate an expression to a concrete value.
    pub fn eval(&self, expr: &RustBV) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.eval(expr)
    }

    /// Get minimum value of an expression.
    pub fn min(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.min(expr, signed)
    }

    /// Get maximum value of an expression.
    pub fn max(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.max(expr, signed)
    }
}
