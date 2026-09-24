//! The no-Z3 (`--no-default-features`) half of [`SymContext`].
//!
//! Split out of `context.rs` (angr-fs8kb.35): the whole file is
//! `#[cfg(not(feature = "vex-engine-z3"))]` at the `mod` declaration in
//! `symbolic/mod.rs`, so it vanishes from the default build. It holds the mock
//! constructors and the mock twins of `assume_true` / `assume_false` /
//! `check_branch_feasibility`, whose Z3 halves live in `constraint_ops.rs` and
//! `solving_ops.rs`. Keeping them together is what lets `make check-no-z3`'s
//! combos be read as one file rather than as cfg islands scattered through the
//! struct's main `impl`.
//!
//! **Panic policy:** carries `#![deny(clippy::unwrap_used,
//! clippy::expect_used)]`, same as the rest of the `SymContext` slice files.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

use parking_lot::Mutex;

use super::local_constraints::LocalConstraints;
use super::{RustBV, SymContext};

impl SymContext {
    /// Create a new mock solver context (without Z3).
    pub fn new_mock() -> Self {
        SymContext {
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::new(HashMap::new()),
            assumed_constraints_shared: Mutex::new(Arc::new(Vec::new())),
            assume_class_reconstructible: AtomicBool::new(true),
            local_constraints: Mutex::new(LocalConstraints::new()),
            mock_scope_savepoints: Mutex::new(Vec::new()),
        }
    }

    /// Alias for new_mock when Z3 is not available.
    pub fn new() -> Self {
        Self::new_mock()
    }

    // =========================================================================
    // Mock implementations of the constraint/solving surface
    // =========================================================================

    pub fn assume_true(&self, cond: &RustBV) {
        self.assume(cond, true);
    }

    pub fn assume_false(&self, cond: &RustBV) {
        self.assume(cond, false);
    }

    /// Non-Z3 mirror of `constraint_ops.rs::assume` (angr-12jjk.11): the two
    /// polarities share one body here too, so a change to the export contract
    /// can't land in one twin and miss the other.
    fn assume(&self, cond: &RustBV, want_true: bool) {
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python; no Z3 to assert against.
        self.local_constraints
            .lock()
            .assumed
            .push((cond.clone(), want_true));
    }

    pub fn check_branch_feasibility(&self, cond: &RustBV) -> (bool, bool) {
        debug_assert_eq!(cond.width(), 1);
        if let Some(v) = cond.as_u128() {
            return (v != 0, v == 0);
        }
        // Without Z3, assume both directions are feasible — matches the
        // can_be_true/can_be_false stubs.
        (true, true)
    }
}
