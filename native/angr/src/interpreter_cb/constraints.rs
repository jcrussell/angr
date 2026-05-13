use super::*;

impl<'a> CallbackInterpreter<'a> {
    /// Track an address concretization constraint for Python sync.
    ///
    /// When Rust concretizes a symbolic address to a concrete value, this
    /// constraint needs to be communicated to Python's claripy solver.
    pub fn track_concretization_constraint(&mut self, addr_expr: &RustBV, concrete_addr: u64) {
        // Only track if the address was actually symbolic
        if addr_expr.is_symbolic() {
            self.pending_python_constraints
                .push(PendingConstraint::address_concretization(
                    addr_expr.clone(),
                    concrete_addr,
                ));
        }
    }

    /// Track a branch constraint for Python sync.
    pub fn track_branch_constraint(&mut self, cond: &RustBV, took_true_branch: bool) {
        if cond.is_symbolic() {
            if took_true_branch {
                self.pending_python_constraints
                    .push(PendingConstraint::branch_true(cond.clone()));
            } else {
                self.pending_python_constraints
                    .push(PendingConstraint::branch_false(cond.clone()));
            }
        }
    }

    /// Check if there are pending constraints to sync.
    pub fn has_pending_constraints(&self) -> bool {
        !self.pending_python_constraints.is_empty()
    }

    /// Get the number of pending constraints.
    pub fn pending_constraint_count(&self) -> usize {
        self.pending_python_constraints.len()
    }

    /// Get pending constraints for export to Python.
    ///
    /// Returns a list of (width, concrete_value) tuples that can be converted
    /// to claripy constraints. The caller should use these to add constraints
    /// to Python's state before performing Python-based operations.
    ///
    /// Note: Full export to claripy ASTs would require storing the original
    /// symbolic expressions, which is complex. This simplified approach exports
    /// the constraint info so Python can reconstruct them if needed.
    pub fn get_pending_constraints(&self) -> &[PendingConstraint] {
        &self.pending_python_constraints
    }

    /// Clear pending constraints after sync.
    pub fn clear_pending_constraints(&mut self) {
        self.pending_python_constraints.clear();
    }

    /// Export pending constraints as a list that Python can process.
    ///
    /// Returns a list of tuples: (description, width, concrete_value)
    /// Python can use these to add constraints to its solver state.
    pub fn export_constraints_for_python(&self) -> Vec<(String, u32, u128, Option<u64>)> {
        self.pending_python_constraints
            .iter()
            .map(|c| {
                (
                    c.description.clone(),
                    c.expression.width(),
                    c.concrete_value,
                    c.handle_id,
                )
            })
            .collect()
    }

    /// Sync pending constraints to Python before making a callback.
    ///
    /// This ensures that Python's claripy solver has all the constraints
    /// that Rust has accumulated, which is critical for operations that
    /// depend on solver state (e.g., symbolic memory operations, SimProcedures).
    ///
    /// Call this before any Python callback that may need solver context.
    pub fn sync_before_callback(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
    ) -> Result<(), CbExecutionError> {
        if self.has_pending_constraints() {
            let constraints = self.export_constraints_for_python();
            callbacks
                .call_sync_constraints(py, &constraints)
                .map_err(|e| {
                    CbExecutionError::Callback(format!("constraint sync failed: {}", e))
                })?;
            self.clear_pending_constraints();
        }
        Ok(())
    }
}
