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

    /// Clear any pending constraints before making a callback into Python.
    ///
    /// angr-h0dv: this used to push the constraints into Python's claripy
    /// solver via `callbacks.call_sync_constraints`, but a 20-bench soak
    /// proved that path was dead — Path A (rust_solver_ctx attach in
    /// `rust_callback_dispatch.py::_install_rust_solver_on_callback_state`)
    /// covers every live callback site, so the Python side already shares
    /// solver context with Rust. The Rust-internal constraint tracker is
    /// retained for in-process tests; we just drop the accumulated set so
    /// the next exploration window starts clean.
    pub fn sync_before_callback(
        &mut self,
        _py: Python<'_>,
        _callbacks: &PythonCallbacks,
    ) -> Result<(), CbExecutionError> {
        if self.has_pending_constraints() {
            self.clear_pending_constraints();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_interp(ctx: &SymContext) -> CallbackInterpreter<'_> {
        CallbackInterpreter::new(VexArch::AMD64, ctx)
    }

    #[test]
    fn concrete_address_does_not_get_tracked() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let concrete = RustBV::concrete(0x4000, 64);
        interp.track_concretization_constraint(&concrete, 0x4000);
        assert_eq!(interp.pending_constraint_count(), 0);
        assert!(!interp.has_pending_constraints());
    }

    #[test]
    fn symbolic_address_gets_tracked() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let sym = RustBV::symbolic(&ctx, "addr", 64);
        interp.track_concretization_constraint(&sym, 0xdead_beef);
        assert_eq!(interp.pending_constraint_count(), 1);
        assert!(interp.has_pending_constraints());
        let exported = interp.export_constraints_for_python();
        assert_eq!(exported.len(), 1);
        let (desc, width, value, handle_id) = &exported[0];
        assert_eq!(*width, 64);
        assert_eq!(*value, 0xdead_beef_u128);
        assert_eq!(*handle_id, None);
        assert!(desc.contains("addr_concretize"));
        assert!(desc.contains("deadbeef"));
    }

    #[test]
    fn branch_true_on_concrete_is_dropped() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let cond = RustBV::concrete(1, 1);
        interp.track_branch_constraint(&cond, true);
        assert_eq!(interp.pending_constraint_count(), 0);
    }

    #[test]
    fn branch_true_on_symbolic_is_tracked() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let cond = RustBV::symbolic(&ctx, "cond", 1);
        interp.track_branch_constraint(&cond, true);
        assert_eq!(interp.pending_constraint_count(), 1);
        let (desc, width, value, _) = &interp.export_constraints_for_python()[0];
        assert_eq!(desc, "branch_true");
        assert_eq!(*width, 1);
        assert_eq!(*value, 1);
    }

    #[test]
    fn branch_false_on_symbolic_records_zero() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let cond = RustBV::symbolic(&ctx, "cond", 1);
        interp.track_branch_constraint(&cond, false);
        let (desc, _width, value, _) = &interp.export_constraints_for_python()[0];
        assert_eq!(desc, "branch_false");
        assert_eq!(*value, 0);
    }

    #[test]
    fn clear_drops_all_pending() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let sym = RustBV::symbolic(&ctx, "x", 64);
        interp.track_concretization_constraint(&sym, 0x1000);
        interp.track_branch_constraint(&sym, true);
        assert_eq!(interp.pending_constraint_count(), 2);
        interp.clear_pending_constraints();
        assert_eq!(interp.pending_constraint_count(), 0);
        assert!(!interp.has_pending_constraints());
    }

    #[test]
    fn multiple_constraints_preserve_order() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let a = RustBV::symbolic(&ctx, "a", 64);
        let b = RustBV::symbolic(&ctx, "b", 1);
        interp.track_concretization_constraint(&a, 0x10);
        interp.track_branch_constraint(&b, true);
        interp.track_concretization_constraint(&a, 0x20);
        let exported = interp.export_constraints_for_python();
        assert_eq!(exported.len(), 3);
        assert_eq!(exported[0].2, 0x10);
        assert_eq!(exported[1].0, "branch_true");
        assert_eq!(exported[2].2, 0x20);
    }

    #[test]
    fn pending_constraint_branch_helpers_set_concrete_values() {
        let ctx = SymContext::new_mock();
        let cond = RustBV::symbolic(&ctx, "c", 1);
        let t = PendingConstraint::branch_true(cond.clone());
        let f = PendingConstraint::branch_false(cond);
        assert_eq!(t.concrete_value, 1);
        assert_eq!(t.description, "branch_true");
        assert_eq!(f.concrete_value, 0);
        assert_eq!(f.description, "branch_false");
    }

    #[test]
    fn address_concretization_with_handle_preserves_id() {
        let ctx = SymContext::new_mock();
        let addr = RustBV::symbolic(&ctx, "p", 64);
        let pc = PendingConstraint::address_concretization_with_handle(addr, 0x1234, Some(42));
        assert_eq!(pc.handle_id, Some(42));
        assert_eq!(pc.concrete_value, 0x1234);
        assert!(pc.description.contains("1234"));
    }

    #[test]
    fn get_pending_constraints_returns_same_as_export() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let sym = RustBV::symbolic(&ctx, "z", 32);
        interp.track_concretization_constraint(&sym, 0xabcd);
        let pending = interp.get_pending_constraints();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].concrete_value, 0xabcd);
        assert_eq!(pending[0].expression.width(), 32);
    }
}
