//! Z3 solver context and constraint management.
//!
//! The `SymContext` manages:
//! - Symbolic variable creation and ID assignment
//! - Constraint tracking (when Z3 is available)
//! - Satisfiability checking (when Z3 is available)

use std::cell::Cell;
use std::collections::HashMap;

use super::RustBV;

/// Solver context for symbolic execution.
///
/// Manages symbolic variable creation and, when Z3 is available,
/// constraint solving and satisfiability checking.
pub struct SymContext<'ctx> {
    /// Counter for generating unique symbol IDs.
    next_id: Cell<u64>,
    /// Named symbolic variables for debugging.
    symbol_table: HashMap<String, u64>,
    /// Phantom data for lifetime (used when Z3 is enabled).
    _phantom: std::marker::PhantomData<&'ctx ()>,

    // Z3-specific fields (when feature is enabled)
    #[cfg(feature = "vex-engine-z3")]
    z3_ctx: &'ctx z3::Context,
    #[cfg(feature = "vex-engine-z3")]
    solver: z3::Solver<'ctx>,
    #[cfg(feature = "vex-engine-z3")]
    constraints: Vec<z3::ast::Bool<'ctx>>,
}

impl<'ctx> SymContext<'ctx> {
    /// Create a new mock solver context (without Z3).
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new_mock() -> Self {
        SymContext {
            next_id: Cell::new(0),
            symbol_table: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Alias for new_mock when Z3 is not available.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new(_: &'ctx ()) -> Self {
        Self::new_mock()
    }

    /// Create a new solver context with Z3.
    #[cfg(feature = "vex-engine-z3")]
    pub fn new(z3_ctx: &'ctx z3::Context) -> Self {
        SymContext {
            next_id: Cell::new(0),
            symbol_table: HashMap::new(),
            _phantom: std::marker::PhantomData,
            z3_ctx,
            solver: z3::Solver::new(z3_ctx),
            constraints: Vec::new(),
        }
    }

    /// Create a mock context for testing (when Z3 is enabled but not needed).
    #[cfg(feature = "vex-engine-z3")]
    pub fn new_mock() -> Self {
        // This leaks a Z3 context - only for testing
        let z3_ctx: &'static z3::Context = Box::leak(Box::new(z3::Context::new(&z3::Config::new())));
        Self::new(z3_ctx)
    }

    /// Get the Z3 context reference (when available).
    #[cfg(feature = "vex-engine-z3")]
    pub fn z3_ctx(&self) -> &'ctx z3::Context {
        self.z3_ctx
    }

    /// Get the next unique ID for a symbolic variable.
    pub fn next_id(&self) -> u64 {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }

    /// Get the number of constraints.
    #[cfg(feature = "vex-engine-z3")]
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn num_constraints(&self) -> usize {
        0
    }

    // =========================================================================
    // Symbol Management
    // =========================================================================

    /// Create a new symbolic bitvector with a unique name.
    pub fn new_bv(&'ctx self, name: &str, width: u32) -> RustBV<'ctx> {
        let unique_name = self.unique_name(name);
        RustBV::symbolic(self, &unique_name, width)
    }

    /// Create a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        let id = self.next_id();
        format!("{}_{}", base, id)
    }

    // =========================================================================
    // Constraint Management (Z3-backed)
    // =========================================================================

    /// Add a constraint.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint(&mut self, constraint: z3::ast::Bool<'ctx>) {
        self.solver.assert(&constraint);
        self.constraints.push(constraint);
    }

    /// Add a constraint that the bitvector equals a specific value.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_bv_constraint(&mut self, bv: &RustBV<'ctx>, value: u128) {
        use z3::ast::Ast;
        let ast = bv.to_z3_ast(self);
        let val_ast = if bv.width() <= 64 {
            z3::ast::BV::from_u64(self.z3_ctx, value as u64, bv.width())
        } else {
            let lo = z3::ast::BV::from_u64(self.z3_ctx, value as u64, 64);
            let hi = z3::ast::BV::from_u64(self.z3_ctx, (value >> 64) as u64, bv.width() - 64);
            hi.concat(&lo)
        };
        let constraint = ast._eq(&val_ast);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is true (non-zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_true(&mut self, cond: &RustBV<'ctx>) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let ast = cond.to_z3_ast(self);
        let one = z3::ast::BV::from_u64(self.z3_ctx, 1, 1);
        let constraint = ast._eq(&one);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_false(&mut self, cond: &RustBV<'ctx>) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let ast = cond.to_z3_ast(self);
        let zero = z3::ast::BV::from_u64(self.z3_ctx, 0, 1);
        let constraint = ast._eq(&zero);
        self.add_constraint(constraint);
    }

    // =========================================================================
    // Satisfiability & Evaluation (Z3-backed)
    // =========================================================================

    /// Check if the current constraints are satisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_sat(&self) -> bool {
        matches!(self.solver.check(), z3::SatResult::Sat)
    }

    /// Check if a bitvector condition can be true.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_true(&self, cond: &RustBV<'ctx>) -> bool {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v != 0;
        }
        let ast = cond.to_z3_ast(self);
        let one = z3::ast::BV::from_u64(self.z3_ctx, 1, 1);
        let constraint = ast._eq(&one);
        self.solver.push();
        self.solver.assert(&constraint);
        let result = matches!(self.solver.check(), z3::SatResult::Sat);
        self.solver.pop(1);
        result
    }

    /// Check if a bitvector condition can be false.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_false(&self, cond: &RustBV<'ctx>) -> bool {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v == 0;
        }
        let ast = cond.to_z3_ast(self);
        let zero = z3::ast::BV::from_u64(self.z3_ctx, 0, 1);
        let constraint = ast._eq(&zero);
        self.solver.push();
        self.solver.assert(&constraint);
        let result = matches!(self.solver.check(), z3::SatResult::Sat);
        self.solver.pop(1);
        result
    }

    /// Evaluate a bitvector to a concrete value if possible.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval(&self, bv: &RustBV<'ctx>) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }
        // Need to solve
        if !self.is_sat() {
            return None;
        }
        let model = self.solver.get_model()?;
        let ast = bv.to_z3_ast(self);
        let result = model.eval(&ast, true)?;
        result.as_u64().map(|v| v as u128)
    }

    // =========================================================================
    // Mock implementations when Z3 is not available
    // =========================================================================

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn is_sat(&self) -> bool {
        // Without Z3, we assume everything is satisfiable
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_true(&self, cond: &RustBV<'ctx>) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v != 0;
        }
        // Without Z3, assume symbolic can be true
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_false(&self, cond: &RustBV<'ctx>) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v == 0;
        }
        // Without Z3, assume symbolic can be false
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval(&self, bv: &RustBV<'ctx>) -> Option<u128> {
        // Without Z3, can only evaluate concrete values
        bv.as_u128()
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the context, creating a copy with the same constraints.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork(&self) -> Self {
        let new_solver = z3::Solver::new(self.z3_ctx);
        for constraint in &self.constraints {
            new_solver.assert(constraint);
        }
        SymContext {
            next_id: Cell::new(self.next_id.get()),
            symbol_table: self.symbol_table.clone(),
            _phantom: std::marker::PhantomData,
            z3_ctx: self.z3_ctx,
            solver: new_solver,
            constraints: self.constraints.clone(),
        }
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork(&self) -> Self {
        SymContext {
            next_id: Cell::new(self.next_id.get()),
            symbol_table: self.symbol_table.clone(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Fork with an additional constraint on the true branch.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork_true(&self, cond: &RustBV<'ctx>) -> Self {
        let mut forked = self.fork();
        forked.assume_true(cond);
        forked
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork_true(&self, _cond: &RustBV<'ctx>) -> Self {
        self.fork()
    }

    /// Fork with an additional constraint on the false branch.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork_false(&self, cond: &RustBV<'ctx>) -> Self {
        let mut forked = self.fork();
        forked.assume_false(cond);
        forked
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork_false(&self, _cond: &RustBV<'ctx>) -> Self {
        self.fork()
    }
}

impl<'ctx> Clone for SymContext<'ctx> {
    fn clone(&self) -> Self {
        self.fork()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generation() {
        let ctx = SymContext::new_mock();
        assert_eq!(ctx.next_id(), 0);
        assert_eq!(ctx.next_id(), 1);
        assert_eq!(ctx.next_id(), 2);
    }

    #[test]
    fn test_unique_names() {
        let ctx = SymContext::new_mock();
        let name1 = ctx.unique_name("x");
        let name2 = ctx.unique_name("x");
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_concrete_eval() {
        let ctx = SymContext::new_mock();
        let bv = RustBV::concrete(42, 32);
        assert_eq!(ctx.eval(&bv), Some(42));
    }

    #[test]
    fn test_fork() {
        let ctx = SymContext::new_mock();
        let id1 = ctx.next_id();

        let forked = ctx.fork();
        let id2 = forked.next_id();

        // Forked context should continue from same ID
        assert_eq!(id2, id1 + 1);
    }
}
