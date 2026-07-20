//! Transaction / scoping `&self` methods for [`SymContext`].
//!
//! Slice 10 of the `symbolic/context.rs` split (angr-a2br.2.8). These are the
//! solver-scope and transactional-sync entry points: the timeout accessors
//! (`set_timeout` / `timeout_ms`), the SAT-cache primer (`set_sat_cache`), the
//! raw scope save/restore (`push` / `pop`), the transaction lifecycle
//! (`transaction_begin` / `transaction_commit` / `transaction_rollback`,
//! `current_push_level`, `in_transaction`), and the constraint-introspection
//! read paths (`unsat_core`, `get_all_constraints_str`, `z3_assertion_count`).
//!
//! Unlike the fully Z3-gated `constraint_ops`, this slice carries both the
//! `#[cfg(feature = "vex-engine-z3")]` implementations and their non-Z3 mock
//! counterparts, so the module decl in `mod.rs` is NOT feature-gated (mirrors
//! `solving_ops` / `lineage_ops`).
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! the transaction-bookkeeping fields it mutates (`push_level`,
//! `push_constraint_counts`, `push_local_cache_lengths`,
//! `push_assumed_local_lengths`, `solver`, `timeout_ms`) are promoted to
//! `pub(super)` (== `pub(in crate::symbolic)`) so this sibling module can reach
//! them without a public API leak. The caches `sat_cache`/`model_cache` and the
//! `constraint_count`/`local_constraints`/`constraint_trackers` fields were
//! already `pub(super)` from earlier slices. See bd memory
//! `a2br2-context-split-impl-block-plan` for the slice plan.

use super::SymContext;

#[cfg(feature = "vex-engine-z3")]
use super::ConstraintSyncError;
#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::Ordering;

#[cfg(not(feature = "vex-engine-z3"))]
use super::ConstraintSyncError;
#[cfg(not(feature = "vex-engine-z3"))]
use std::sync::atomic::Ordering;

impl SymContext {
    /// Set the Z3 solver timeout in milliseconds.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_timeout(&self, timeout_ms: u32) {
        self.timeout_ms.store(timeout_ms, Ordering::SeqCst);
        if let Some(solver) = self.solver.lock().as_ref() {
            solver.set_params(&build_solver_params(timeout_ms));
        }
    }

    /// Get the Z3 solver timeout in milliseconds.
    #[cfg(feature = "vex-engine-z3")]
    pub fn timeout_ms(&self) -> u32 {
        self.timeout_ms.load(Ordering::SeqCst)
    }

    /// Prime the SAT cache with a known value.
    /// Used after symbolic branch forking where the interpreter already
    /// proved feasibility — avoids redundant Z3 check() calls.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_sat_cache(&self, value: bool) {
        self.sat_cache.set(Some(value));
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn set_sat_cache(&self, _value: bool) {}

    /// Save solver state for temporary constraints.
    ///
    /// Dispatches the Z3-side scope save through
    /// `scope_savepoint_push` (angr-v5a5
    /// slice 4b.2): in the None lineage branch this is the previous
    /// `self.solver().push()`; in the Some (shared-lineage) branch it
    /// records the current `scope_path` length onto `scope_savepoints`
    /// and defers the Z3-side maintenance to the lazy `switch_to` in
    /// `with_z3_solver`. Cache invalidation
    /// (`sat_cache`, `model_cache`) is owned here rather than by the
    /// helper, since different future callers of the savepoint helpers
    /// may want different invalidation policies.
    #[cfg(feature = "vex-engine-z3")]
    pub fn push(&self) {
        self.scope_savepoint_push();
        // Invalidate caches since constraint set may change
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    /// Restore solver state.
    ///
    /// Dispatches the Z3-side scope restore through
    /// `scope_savepoint_pop` (angr-v5a5
    /// slice 4b.3): in the None lineage branch this is the previous
    /// `self.solver().pop(1)`; in the Some (shared-lineage) branch it
    /// pops the most-recent savepoint off `scope_savepoints` and
    /// truncates `scope_path` back to that length. Cache invalidation
    /// (`sat_cache`, `model_cache`) is owned here rather than by the
    /// helper, since different future callers of the savepoint helpers
    /// may want different invalidation policies.
    #[cfg(feature = "vex-engine-z3")]
    pub fn pop(&self) {
        self.scope_savepoint_pop();
        // Invalidate caches since constraint set has changed
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    /// Attempt a scope [`pop()`](Self::pop), refusing an unbalanced under-pop
    /// (angr-ph300.48).
    ///
    /// In the **None** lineage branch a bare `pop()` forwards to
    /// `z3::Solver::pop(1)`, which **panics** on under-pop — a Python caller
    /// invoking `RustSolverContext.pop()` with no matching `push()` would
    /// otherwise abort the interpreter. This gate consults
    /// [`bare_z3_push_depth()`](Self::bare_z3_push_depth) (the counter
    /// `scope_savepoint_push`/`pop` keep in lockstep with the per-context Z3
    /// scope stack) and returns `false` instead of popping when it is `0`.
    ///
    /// In the **Some** (shared-lineage) branch a mismatched pop is already
    /// silently ignored (`scope_savepoints` empties out harmlessly), so it is
    /// always safe — `try_pop` returns `true` and delegates to `pop()`.
    ///
    /// Returns `true` when a pop was performed (or would have been safe),
    /// `false` when it was refused to avoid a bare-scope underflow.
    #[cfg(feature = "vex-engine-z3")]
    pub fn try_pop(&self) -> bool {
        let has_lineage = self.lineage.lock().is_some();
        if !has_lineage && self.bare_z3_push_depth() == 0 {
            return false;
        }
        self.pop();
        true
    }

    /// Mock `try_pop` — without Z3 a `pop()` is a no-op that never panics, so
    /// every pop is trivially "safe".
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn try_pop(&self) -> bool {
        self.pop();
        true
    }

    // =========================================================================
    // Transactional Constraint Sync
    // =========================================================================

    /// Begin a new transaction.
    ///
    /// This pushes a new solver frame and records the constraint count,
    /// allowing rollback on failure via `transaction_rollback()`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_begin(&self) {
        self.push();
        let current_count = self.constraint_count.load(Ordering::SeqCst);
        self.push_constraint_counts.lock().push(current_count);
        let (local_len, assumed_local_len) = {
            let local = self.local_constraints.lock();
            (local.z3_assertions.len(), local.assumed.len())
        };
        self.push_local_cache_lengths.lock().push(local_len);
        self.push_assumed_local_lengths
            .lock()
            .push(assumed_local_len);
        self.push_level.fetch_add(1, Ordering::SeqCst);
    }

    /// Commit the current transaction.
    ///
    /// This validates that constraints are satisfiable before committing.
    /// Returns an error if constraints became unsatisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_commit(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }

        // Validate constraints are satisfiable before committing
        if !self.is_sat() {
            // Rollback on failure
            self.transaction_rollback()?;
            return Err(ConstraintSyncError::Unsatisfiable);
        }

        // Pop the solver frame but keep the constraints
        // Note: We don't actually pop here since we want to keep constraints
        // The push was just for protection during sync
        self.push_constraint_counts.lock().pop();
        self.push_level.fetch_sub(1, Ordering::SeqCst);

        Ok(())
    }

    /// Rollback the current transaction.
    ///
    /// This restores the solver state to before `transaction_begin()` was called.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_rollback(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }

        // Pop the solver frame (discards constraints added since begin)
        self.pop();

        // Restore constraint count
        if let Some(prev_count) = self.push_constraint_counts.lock().pop() {
            self.constraint_count.store(prev_count, Ordering::SeqCst);
        }

        // Truncate local Z3 cache and assumed_constraints to pre-transaction length
        let prev_z3_len = self.push_local_cache_lengths.lock().pop();
        let prev_assumed_len = self.push_assumed_local_lengths.lock().pop();
        if prev_z3_len.is_some() || prev_assumed_len.is_some() {
            let mut local = self.local_constraints.lock();
            if let Some(prev_len) = prev_z3_len {
                local.z3_assertions.truncate(prev_len);
            }
            if let Some(prev_len) = prev_assumed_len {
                local.assumed.truncate(prev_len);
            }
            // angr-sfp9: stale dedup_set entries from the rolled-back
            // assertions could falsely dedup a re-assert. Drop the side-
            // table and let the next add_constraint_raw call rebuild it
            // from shared + post-truncate local.
            local.dedup_set.clear();
            local.dedup_set_seeded = false;
        }

        self.push_level.fetch_sub(1, Ordering::SeqCst);

        Ok(())
    }

    /// Get the current transaction level.
    ///
    /// Returns 0 if no transaction is active.
    pub fn current_push_level(&self) -> usize {
        self.push_level.load(Ordering::SeqCst)
    }

    /// Check if currently in a transaction.
    pub fn in_transaction(&self) -> bool {
        self.current_push_level() > 0
    }

    // Non-Z3 versions of transaction methods
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_begin(&self) {
        let current_count = self.constraint_count.load(Ordering::SeqCst);
        self.push_constraint_counts.lock().push(current_count);
        self.push_level.fetch_add(1, Ordering::SeqCst);
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_commit(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }
        self.push_constraint_counts.lock().pop();
        self.push_level.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_rollback(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }
        if let Some(prev_count) = self.push_constraint_counts.lock().pop() {
            self.constraint_count.store(prev_count, Ordering::SeqCst);
        }
        self.push_level.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    /// Get the unsat core as indices of constraints added.
    ///
    /// Returns the indices of constraints that form the unsatisfiable core.
    /// Call this after checking satisfiability and finding UNSAT.
    #[cfg(feature = "vex-engine-z3")]
    pub fn unsat_core(&self) -> Vec<usize> {
        let core_strs: Vec<String> = self.with_z3_solver(|solver| {
            solver
                .get_unsat_core()
                .iter()
                .map(|ast| format!("{ast}"))
                .collect()
        });

        let trackers = self.constraint_trackers.lock();
        let mut result = Vec::new();

        // Match core tracking booleans to stored tracker indices by string representation
        for core_str in &core_strs {
            for (i, tracker) in trackers.iter().enumerate() {
                if format!("{tracker}") == *core_str {
                    result.push(i);
                    break;
                }
            }
        }

        result
    }

    /// Compute an unsat core over the *assumed-constraint* list, on demand.
    ///
    /// Complements [`Self::unsat_core`], which only reports constraints the
    /// caller opted into tracking at add time via
    /// `add_constraint_tracked_indexed`. The engine's own path constraints
    /// (fork guards, SimProc adds) go in untracked — tracking booleans cost
    /// a fresh Bool symbol + `assert_and_track` per constraint on the hot
    /// path — so a core read off the live solver would be *silently
    /// incomplete*, which is the one failure mode `CONSTRAINT_TRACKING_IN_SOLVER`
    /// exists to prevent.
    ///
    /// Instead of tracking eagerly, rebuild a throwaway solver at query time
    /// from the constraint IR the context already keeps: `assumed_constraints`
    /// (the `(RustBV, is_true)` pairs that `get_assumed_constraints` exports as
    /// `state.solver.constraints`) asserted *tracked*, plus the residual
    /// no-RustBV assertions (`non_bv_assertions`) and `extra` asserted
    /// *untracked*. That is the same shared ∪ local pair `to_snapshot` rebuilds
    /// a context from, so the core is complete; untracked asserts are excluded
    /// from the core by construction, exactly like Python's claripy tracking
    /// solver excludes `extra_constraints`.
    ///
    /// The returned indices are positions in `get_assumed_constraints()` — so
    /// they index `state.solver.constraints` 1:1. Empty when the constraint set
    /// is SAT (Z3 produces no core), matching `SimSolver.unsat_core`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn unsat_core_assumed(&self, extra: &[z3::ast::Bool]) -> Vec<usize> {
        use std::collections::HashSet;
        use z3::SatResult;

        let assumed = self.get_assumed_constraints();
        // Assumption literals + `check_assumptions`, NOT `assert_and_track` +
        // `check`: the latter only produces a core when the solver's
        // `unsat_core` param is armed before the first assert, while
        // `check_assumptions` always produces one over the literals it is
        // handed, so it needs no solver configuration. The solver is also a
        // plain `z3::Solver`, not `build_solver`: the tactic-backed variant
        // `build_solver` can return (ANGR_Z3_TACTIC) cannot produce cores.
        let solver = z3::Solver::new();
        let mut params = z3::Params::new();
        params.set_u32("timeout", self.timeout_ms.load(Ordering::SeqCst));
        solver.set_params(&params);

        let mut trackers = Vec::with_capacity(assumed.len());
        let mut guarded: HashSet<z3::ast::Bool> = HashSet::with_capacity(assumed.len());
        for (idx, (bv, is_true)) in assumed.iter().enumerate() {
            let cond = self.assumed_pair_to_z3_bool(bv, *is_true);
            let tracker = z3::ast::Bool::new_const(format!("__core_{idx}").as_str());
            solver.assert(tracker.implies(&cond));
            guarded.insert(cond);
            trackers.push(tracker);
        }
        // Residual (no-RustBV) assertions: needed for a faithful SAT/UNSAT
        // verdict, but they have no claripy AST to report, so assert
        // unconditionally — with no literal guarding them they can never
        // appear in the core.
        //
        // EXCEPT the ones that also have an `assumed` entry. A constraint
        // imported from Python (`_add_constraints_to_state`'s Z3-ptr fast path)
        // lands in BOTH lists: `add_constraint_raw` logs it as residual and the
        // same call site pushes an `assumed` pair for it. Asserting the residual
        // copy unguarded would pin the constraint outside the assumption
        // literals, so an all-Python contradiction comes back Unsat with an
        // EMPTY core — the constraints that caused it are all unguarded. Z3
        // hash-conses ASTs, so an `Eq`/`Hash` hit against the guarded set is
        // exact structural identity, not a heuristic.
        let shared_non_bv = self.non_bv_assertions_shared.lock().clone();
        for c in shared_non_bv.iter() {
            if !guarded.contains(c) {
                solver.assert(c);
            }
        }
        {
            let local = self.local_constraints.lock();
            for c in local.non_bv_assertions.iter() {
                if !guarded.contains(c) {
                    solver.assert(c);
                }
            }
        }
        for c in extra {
            solver.assert(c);
        }

        if solver.check_assumptions(&trackers) != SatResult::Unsat {
            return Vec::new();
        }
        let mut core: Vec<usize> = solver
            .get_unsat_core()
            .iter()
            .filter_map(|ast| {
                format!("{ast}")
                    .strip_prefix("__core_")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .collect();
        core.sort_unstable();
        core
    }

    /// Rebuild the Z3 Bool for one `(RustBV, is_true)` assumed-constraint pair.
    ///
    /// Mirrors `assume_true` / `assume_false`'s symbolic path: a width-1 guard
    /// becomes a Bool directly; a wider BV is a C-style truth test (`bv != 0`).
    /// `is_true == false` negates. The concrete fast paths in `assume_*` are
    /// deliberately NOT replicated — a concrete guard round-trips through Z3 as
    /// a constant Bool, which the solver folds away.
    #[cfg(feature = "vex-engine-z3")]
    fn assumed_pair_to_z3_bool(&self, bv: &super::RustBV, is_true: bool) -> z3::ast::Bool {
        let cond = if bv.width() == 1 {
            bv.to_z3_bool()
        } else {
            let zero = super::RustBV::concrete(0, bv.width());
            bv.ne(&zero, self).to_z3_bool()
        };
        if is_true { cond } else { cond.not() }
    }

    /// Get all solver assertions as strings.
    ///
    /// Returns string representations of all Z3 constraints. Useful for debugging
    /// and for syncing constraint state to Python. While not a full AST export,
    /// this allows Python to understand what constraints are active.
    #[cfg(feature = "vex-engine-z3")]
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        self.with_z3_solver(|solver| {
            solver
                .get_assertions()
                .iter()
                .map(|a| format!("{a}"))
                .collect()
        })
    }

    /// Check the total number of assertions in the Z3 solver.
    ///
    /// This can be used to verify constraint sync between Rust and Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn z3_assertion_count(&self) -> usize {
        self.with_z3_solver(|solver| solver.get_assertions().len())
    }

    // =========================================================================
    // Mock implementations when Z3 is not available
    // =========================================================================

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn push(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn pop(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn set_timeout(&self, _timeout_ms: u32) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn timeout_ms(&self) -> u32 {
        0
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn unsat_core(&self) -> Vec<usize> {
        // Without Z3, no unsat core available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn unsat_core_assumed(&self, _extra: &[()]) -> Vec<usize> {
        // Without Z3, no unsat core available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        // Without Z3, no constraints available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn z3_assertion_count(&self) -> usize {
        // Without Z3, no assertions
        0
    }
}
