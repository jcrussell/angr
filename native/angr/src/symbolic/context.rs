//! Z3 solver context and constraint management.
//!
//! The `SymContext` manages:
//! - Symbolic variable creation and ID assignment
//! - Constraint tracking (when Z3 is available)
//! - Satisfiability checking (when Z3 is available)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::{Mutex, RwLock};

use super::RustBV;

/// Solver context for symbolic execution.
///
/// Manages symbolic variable creation and, when Z3 is available,
/// constraint solving and satisfiability checking.
///
/// With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
/// to store a reference to it. All Z3 operations on a thread share
/// the same context automatically.
pub struct SymContext {
    /// Counter for generating unique symbol IDs.
    next_id: AtomicU64,
    /// Number of constraints added (for tracking).
    constraint_count: AtomicUsize,
    /// Named symbolic variables for debugging.
    symbol_table: RwLock<HashMap<String, u64>>,

    // Z3-specific fields (when feature is enabled)
    #[cfg(feature = "vex-engine-z3")]
    solver: Mutex<z3::Solver>,
}

impl SymContext {
    /// Create a new mock solver context (without Z3).
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new_mock() -> Self {
        SymContext {
            next_id: AtomicU64::new(0),
            constraint_count: AtomicUsize::new(0),
            symbol_table: RwLock::new(HashMap::new()),
        }
    }

    /// Alias for new_mock when Z3 is not available.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new() -> Self {
        Self::new_mock()
    }

    /// Create a new solver context with Z3.
    ///
    /// With z3-rs 0.19+, the Z3 context is thread-local.
    /// All Z3 operations on this thread will use the same context.
    #[cfg(feature = "vex-engine-z3")]
    pub fn new() -> Self {
        SymContext {
            next_id: AtomicU64::new(0),
            constraint_count: AtomicUsize::new(0),
            symbol_table: RwLock::new(HashMap::new()),
            solver: Mutex::new(z3::Solver::new()),
        }
    }

    /// Create a mock context for testing (when Z3 is enabled but not needed).
    #[cfg(feature = "vex-engine-z3")]
    pub fn new_mock() -> Self {
        Self::new()
    }

    /// Get the next unique ID for a symbolic variable.
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraint_count.load(Ordering::SeqCst)
    }

    // =========================================================================
    // Symbol Management
    // =========================================================================

    /// Create a new symbolic bitvector with a unique name.
    pub fn new_bv(&self, name: &str, width: u32) -> RustBV {
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
    pub fn add_constraint(&self, constraint: z3::ast::Bool) {
        self.solver.lock().assert(&constraint);
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Add a constraint that the bitvector equals a specific value.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_bv_constraint(&self, bv: &RustBV, value: u128) {
        use z3::ast::Ast;
        let ast = bv.to_z3_ast();
        let val_ast = if bv.width() <= 64 {
            z3::ast::BV::from_u64(value as u64, bv.width())
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, bv.width() - 64);
            hi.concat(&lo)
        };
        let constraint = ast._eq(&val_ast);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is true (non-zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_true(&self, cond: &RustBV) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let ast = cond.to_z3_ast();
        let one = z3::ast::BV::from_u64(1, 1);
        let constraint = ast._eq(&one);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_false(&self, cond: &RustBV) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let ast = cond.to_z3_ast();
        let zero = z3::ast::BV::from_u64(0, 1);
        let constraint = ast._eq(&zero);
        self.add_constraint(constraint);
    }

    // =========================================================================
    // Satisfiability & Evaluation (Z3-backed)
    // =========================================================================

    /// Check if the current constraints are satisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_sat(&self) -> bool {
        matches!(self.solver.lock().check(), z3::SatResult::Sat)
    }

    /// Check if a bitvector condition can be true.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_true(&self, cond: &RustBV) -> bool {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v != 0;
        }
        let ast = cond.to_z3_ast();
        let one = z3::ast::BV::from_u64(1, 1);
        let constraint = ast._eq(&one);
        self.solver.lock().push();
        self.solver.lock().assert(&constraint);
        let result = matches!(self.solver.lock().check(), z3::SatResult::Sat);
        self.solver.lock().pop(1);
        result
    }

    /// Check if a bitvector condition can be false.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_false(&self, cond: &RustBV) -> bool {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v == 0;
        }
        let ast = cond.to_z3_ast();
        let zero = z3::ast::BV::from_u64(0, 1);
        let constraint = ast._eq(&zero);
        self.solver.lock().push();
        self.solver.lock().assert(&constraint);
        let result = matches!(self.solver.lock().check(), z3::SatResult::Sat);
        self.solver.lock().pop(1);
        result
    }

    /// Evaluate a bitvector to a concrete value if possible.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }
        // Need to solve
        if !self.is_sat() {
            return None;
        }
        let model = self.solver.lock().get_model()?;
        let ast = bv.to_z3_ast();
        let result = model.eval(&ast, true)?;
        Self::extract_bv_value(&result)
    }

    /// Extract a u128 value from a Z3 BV result.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value(bv: &z3::ast::BV) -> Option<u128> {
        // Try as u64 first (fast path for <= 64-bit)
        if let Some(v) = bv.as_u64() {
            return Some(v as u128);
        }
        // For larger values, try to extract as i64 and convert
        if let Some(v) = bv.as_i64() {
            return Some(v as u64 as u128);
        }
        // For very wide values, we'd need to get the string repr and parse
        // For now, return None for > 64-bit non-trivial values
        None
    }

    /// Evaluate a bitvector and return up to n solutions.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return vec![v];
        }

        if n == 0 {
            return vec![];
        }

        let mut results = Vec::with_capacity(n);
        let ast = bv.to_z3_ast();

        // Hold lock for entire operation to avoid lifetime issues
        let solver = self.solver.lock();
        solver.push();

        for _ in 0..n {
            match solver.check() {
                z3::SatResult::Sat => {
                    if let Some(model) = solver.get_model() {
                        if let Some(result) = model.eval(&ast, true) {
                            if let Some(value) = Self::extract_bv_value(&result) {
                                results.push(value);
                                // Add constraint to exclude this value
                                let val_ast = if bv.width() <= 64 {
                                    z3::ast::BV::from_u64(value as u64, bv.width())
                                } else {
                                    let lo = z3::ast::BV::from_u64(value as u64, 64);
                                    let hi = z3::ast::BV::from_u64(
                                        (value >> 64) as u64,
                                        bv.width() - 64,
                                    );
                                    hi.concat(&lo)
                                };
                                solver.assert(&ast._eq(&val_ast).not());
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        solver.pop(1);
        results
    }

    /// Get the minimum value of a bitvector.
    #[cfg(feature = "vex-engine-z3")]
    pub fn min(&self, bv: &RustBV, signed: bool) -> Option<u128> {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        // Hold lock for entire operation to avoid lifetime issues
        let solver = self.solver.lock();
        solver.push();

        let mut result = None;

        // Keep constraining to find minimum
        loop {
            match solver.check() {
                z3::SatResult::Sat => {
                    if let Some(model) = solver.get_model() {
                        if let Some(eval_result) = model.eval(&ast, true) {
                            if let Some(value) = Self::extract_bv_value(&eval_result) {
                                result = Some(value);
                                // Constrain to be strictly less
                                let val_ast = if width <= 64 {
                                    z3::ast::BV::from_u64(value as u64, width)
                                } else {
                                    let lo = z3::ast::BV::from_u64(value as u64, 64);
                                    let hi = z3::ast::BV::from_u64(
                                        (value >> 64) as u64,
                                        width - 64,
                                    );
                                    hi.concat(&lo)
                                };
                                // Use direct comparison methods from z3-rs 0.19+
                                let constraint = if signed {
                                    ast.bvslt(&val_ast)
                                } else {
                                    ast.bvult(&val_ast)
                                };
                                solver.assert(&constraint);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        solver.pop(1);
        result
    }

    /// Get the maximum value of a bitvector.
    #[cfg(feature = "vex-engine-z3")]
    pub fn max(&self, bv: &RustBV, signed: bool) -> Option<u128> {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        // Hold lock for entire operation to avoid lifetime issues
        let solver = self.solver.lock();
        solver.push();

        let mut result = None;

        // Keep constraining to find maximum
        loop {
            match solver.check() {
                z3::SatResult::Sat => {
                    if let Some(model) = solver.get_model() {
                        if let Some(eval_result) = model.eval(&ast, true) {
                            if let Some(value) = Self::extract_bv_value(&eval_result) {
                                result = Some(value);
                                // Constrain to be strictly greater
                                let val_ast = if width <= 64 {
                                    z3::ast::BV::from_u64(value as u64, width)
                                } else {
                                    let lo = z3::ast::BV::from_u64(value as u64, 64);
                                    let hi = z3::ast::BV::from_u64(
                                        (value >> 64) as u64,
                                        width - 64,
                                    );
                                    hi.concat(&lo)
                                };
                                // Use direct comparison methods from z3-rs 0.19+
                                let constraint = if signed {
                                    ast.bvsgt(&val_ast)
                                } else {
                                    ast.bvugt(&val_ast)
                                };
                                solver.assert(&constraint);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        solver.pop(1);
        result
    }

    /// Check if a specific value is a valid solution for a bitvector.
    #[cfg(feature = "vex-engine-z3")]
    pub fn solution(&self, bv: &RustBV, value: u128) -> bool {
        use z3::ast::Ast;

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return v == value;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        let val_ast = if width <= 64 {
            z3::ast::BV::from_u64(value as u64, width)
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, width - 64);
            hi.concat(&lo)
        };

        let constraint = ast._eq(&val_ast);

        self.solver.lock().push();
        self.solver.lock().assert(&constraint);
        let result = matches!(self.solver.lock().check(), z3::SatResult::Sat);
        self.solver.lock().pop(1);

        result
    }

    /// Save solver state for temporary constraints.
    #[cfg(feature = "vex-engine-z3")]
    pub fn push(&self) {
        self.solver.lock().push();
    }

    /// Restore solver state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn pop(&self) {
        self.solver.lock().pop(1);
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
    pub fn can_be_true(&self, cond: &RustBV) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v != 0;
        }
        // Without Z3, assume symbolic can be true
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_false(&self, cond: &RustBV) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v == 0;
        }
        // Without Z3, assume symbolic can be false
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Without Z3, can only evaluate concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        if n == 0 {
            return vec![];
        }
        // Without Z3, can only return concrete values
        bv.as_u128().map(|v| vec![v]).unwrap_or_default()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn min(&self, bv: &RustBV, _signed: bool) -> Option<u128> {
        // Without Z3, can only return concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn max(&self, bv: &RustBV, _signed: bool) -> Option<u128> {
        // Without Z3, can only return concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn solution(&self, bv: &RustBV, value: u128) -> bool {
        // Without Z3, can only check concrete values
        bv.as_u128().map(|v| v == value).unwrap_or(true)
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn push(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn pop(&self) {
        // No-op without Z3
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the context, creating a new context.
    /// Note: Constraints must be re-added by the caller.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork(&self) -> Self {
        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(0),  // Start fresh - caller must re-add constraints
            symbol_table: RwLock::new(self.symbol_table.read().clone()),
            solver: Mutex::new(z3::Solver::new()),
        }
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork(&self) -> Self {
        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(0),
            symbol_table: RwLock::new(self.symbol_table.read().clone()),
        }
    }

    /// Fork with an additional constraint on the true branch.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork_true(&self, cond: &RustBV) -> Self {
        let forked = self.fork();
        forked.assume_true(cond);
        forked
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork_true(&self, _cond: &RustBV) -> Self {
        self.fork()
    }

    /// Fork with an additional constraint on the false branch.
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork_false(&self, cond: &RustBV) -> Self {
        let forked = self.fork();
        forked.assume_false(cond);
        forked
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork_false(&self, _cond: &RustBV) -> Self {
        self.fork()
    }
}

impl Clone for SymContext {
    fn clone(&self) -> Self {
        self.fork()
    }
}

impl Default for SymContext {
    fn default() -> Self {
        Self::new()
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
