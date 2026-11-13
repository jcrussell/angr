//! Lineage / shared-solver `&self` methods for [`SymContext`].
//!
//! Slice 6 of the `symbolic/context.rs` split (angr-a2br.2.5). These were the
//! most field-isolated methods of the original monolithic `impl SymContext`
//! block. The core of the slice is the lineage cells (`lineage`, `scope_path`,
//! `scope_savepoints`, `use_shared_lineage_solver`, `bare_z3_push_depth`) —
//! promoted to `pub(super)` so this sibling module can reach them.
//!
//! Three methods reach **outside** those cells, into constraint-log state
//! owned by `context.rs` / `constraint_ops.rs`. A change here can therefore
//! break constraint bookkeeping, not just lineage bookkeeping:
//!
//! - [`SymContext::fold_sharing_walk`] calls `get_assumed_constraints()`
//!   (`context.rs`), which reads `assumed_constraints_shared` and
//!   `local_constraints`.
//! - [`SymContext::scope_savepoint_push`] / [`SymContext::scope_savepoint_pop`]
//!   record and replay `local_constraints` log lengths through
//!   `bare_local_savepoints` (angr-ph300.41/.42), truncating
//!   `local_constraints` on the pop side.
//! - The test-only `local_z3_assertions_len` reads `local_constraints`.
//!
//! See `rust_z3_sharing.rst` "Interaction with `SharedLineageSolver`" for the
//! lineage invariants that gate which methods stay in `context.rs`.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! `pub(super)` (== `pub(in crate::symbolic)`) keeps the promoted fields
//! module-private — no public API leak.

use super::SymContext;
use super::sharing::ConstraintSharingWalk;

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl SymContext {
    /// Fold this context's assumed constraints into the in-progress sharing
    /// walk (angr-zdho). Each (RustBV, _) is treated as a top-level constraint
    /// tree and walked recursively; pointer-keyed dedup matches today's
    /// per-conversion cache, structural-keyed dedup answers what
    /// construction-level hash-cons (angr-behq) would dedupe to.
    ///
    /// The cloned vector is handed to the walk *by value*: it would otherwise
    /// be dropped here, and the next context's clone could reuse its addresses
    /// and be scored as already-seen (see the `sharing` module doc)
    /// (angr-gkcxh).
    pub fn fold_sharing_walk(&self, walk: &mut ConstraintSharingWalk) {
        walk.visit_batch(
            self.get_assumed_constraints()
                .into_iter()
                .map(|(bv, _)| bv)
                .collect(),
        );
    }

    /// Clone of this context's lineage Arc, if any (angr-v5a5 spike).
    ///
    /// `None` unless some ancestor opted into the shared-lineage feature
    /// via `set_use_shared_lineage_solver` — the default-off case, where
    /// [`fork`](SymContext::fork) only propagates an Arc the parent
    /// already had (i.e. `None`). Under an opt-in, `fork` mints a
    /// [`SharedLineageSolver`](super::lineage::SharedLineageSolver) and
    /// every descendant returns a clone of that Arc.
    pub fn lineage_arc(&self) -> Option<Arc<Mutex<super::lineage::SharedLineageSolver>>> {
        self.lineage.lock().as_ref().map(Arc::clone)
    }

    /// Current per-state scope-path depth (angr-v5a5 spike).
    ///
    /// Always 0 while `lineage` is `None` (the default-off case) — frames
    /// are minted only when `assume_*` routes through the lineage solver.
    /// Exposed as a telemetry surface for both modes.
    pub fn scope_path_len(&self) -> usize {
        self.scope_path.lock().len()
    }

    /// Current size of this context's scope-savepoint stack
    /// (angr-v5a5 slice 4b).
    ///
    /// Always 0 while `lineage` is `None` (the default-off case): that
    /// branch keeps its scopes in `bare_local_savepoints`
    /// instead. Exposed for telemetry and test assertions.
    pub fn scope_savepoint_depth(&self) -> usize {
        self.scope_savepoints.lock().len()
    }

    /// Current count of outstanding bare Z3 pushes (angr-3ms1 step 1a).
    ///
    /// Returns the number of `scope_savepoint_push`
    /// calls on the **None** lineage branch that have not yet been
    /// balanced by a matching `scope_savepoint_pop`.
    /// Always 0 immediately after construction and when every push has
    /// been popped. Always 0 along the Some (shared-lineage) branch —
    /// that branch records on `scope_savepoints` rather than touching
    /// the Z3 stack directly.
    ///
    /// Exposed for telemetry and read by the fork-time materialization
    /// gate in [`fork`](SymContext::fork), which refuses to mint a fresh
    /// `SharedLineageSolver` frame when the parent's
    /// `bare_z3_push_depth` is non-zero.
    pub fn bare_z3_push_depth(&self) -> usize {
        self.bare_z3_push_depth.load(Ordering::Relaxed)
    }

    /// Current length of the local `z3_assertions` log (angr-ph300.41).
    /// Test-only white-box accessor for asserting that a bare `pop()`
    /// truncates constraints added inside the popped scope.
    #[cfg(test)]
    pub(crate) fn local_z3_assertions_len(&self) -> usize {
        self.local_constraints.lock().z3_assertions.len()
    }

    /// Whether fork-time `SharedLineageSolver` materialization is opted
    /// in for this context (angr-3ms1 step 1b).
    ///
    /// Returns `false` by default. When `true`, the fork-time gate in
    /// [`fork`](SymContext::fork) mints a fresh `SharedLineageSolver` on
    /// every fork (subject to the `bare_z3_push_depth == 0` correctness
    /// gate from step 1a and the lineage-dismantle detector). Inherited
    /// from parent to child by `fork`.
    pub fn use_shared_lineage_solver(&self) -> bool {
        self.use_shared_lineage_solver.load(Ordering::Relaxed)
    }

    /// Set the fork-time `SharedLineageSolver` materialization opt-in
    /// (angr-3ms1 step 1b).
    ///
    /// Takes effect at the next [`fork`](Self::fork) call — slice-1c's
    /// gate reads this on every fork to decide whether to mint a fresh
    /// `SharedLineageSolver`. Existing in-flight lineages on this
    /// context are unaffected; flipping the flag off does NOT tear down
    /// an already-installed lineage.
    ///
    /// Default is `false`. Wired from Python via the
    /// `use_shared_lineage_solver=` kwarg on
    /// `RustExplorationManager.__init__`; the manager calls this on
    /// each seed state's solver context so descendants inherit the
    /// opt-in through `fork()`.
    ///
    /// **Do NOT default this on for any single exploration strategy.**
    /// Flipping it default-on for `strategy='dfs'` is
    /// the wrong direction (angr-ua1i): both the canonical WIN canary (ais3_crackme)
    /// and LOSE canary (defcon2016quals_baby-re) run
    /// BFS in the bench harness, so strategy is not the discriminator.
    /// The replacement is runtime thrash detection in
    /// [`super::lineage::sample_for_thrash`], which is strategy-agnostic
    /// and ships in `tick_and_sample_for_thrash` from `run_loop`.
    pub fn set_use_shared_lineage_solver(&self, v: bool) {
        self.use_shared_lineage_solver.store(v, Ordering::Relaxed);
    }

    /// Save a scope-path savepoint, dispatching by lineage
    /// (angr-v5a5 slice 4b).
    ///
    /// In the **None** dispatch path (today's only production path),
    /// the recorded savepoint is the whole scope: the cached solver pushes
    /// a frame for it at the next query (`SymContext::sync_solver`), and a
    /// solver rebuilt after an eviction pushes it again. In the **Some**
    /// (shared-lineage) dispatch path, this records the current
    /// `scope_path.len()` on `scope_savepoints` so a later
    /// [`scope_savepoint_pop()`](Self::scope_savepoint_pop) can truncate
    /// `scope_path` back to this point — no Z3 op is performed against
    /// the shared solver, because the shared solver's stack reflects the
    /// most-recently-loaded sibling's scope path and a bare `push()`
    /// would put assertions in the wrong scope.
    ///
    /// The shared-lineage Z3 push is performed lazily by
    /// `with_z3_solver` the next time a query
    /// fires for this state — via `SharedLineageSolver::switch_to`,
    /// which pushes whatever frames the state has accumulated.
    ///
    /// Does **not** invalidate `sat_cache` / `model_cache` on its own —
    /// the public wrapper (`push()`) owns that.
    ///
    /// Wired in by slice 4b.2: [`push()`](Self::push) is the public caller.
    pub(super) fn scope_savepoint_push(&self) {
        // angr-ph300.41/.42: record the local-constraint log lengths so the
        // matching bare `pop()` can truncate everything added inside this
        // scope. Captured before the lineage dispatch so both branches share it.
        {
            static NEXT_SAVEPOINT: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(1);
            let local = self.local_constraints.lock();
            self.bare_local_savepoints
                .lock()
                .push(super::context::Savepoint {
                    id: NEXT_SAVEPOINT.fetch_add(1, Ordering::Relaxed),
                    z3_len: local.z3_assertions.len(),
                    assumed_len: local.assumed.len(),
                    non_bv_len: local.non_bv_assertions.len(),
                });
        }
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                // The savepoint above is the scope. The cached solver takes
                // its Z3 frame at the next query (`sync_solver`).
                // angr-3ms1 step 1a: track the bare push so the slice-1c
                // fork-time materialization gate can refuse to mint a
                // lineage while bare pushes are outstanding.
                self.bare_z3_push_depth.fetch_add(1, Ordering::Relaxed);
            }
            Some(_) => {
                let depth = self.scope_path.lock().len();
                self.scope_savepoints.lock().push(depth);
            }
        }
    }

    /// Restore the most-recently-saved scope-path savepoint, dispatching
    /// by lineage (angr-v5a5 slice 4b).
    ///
    /// In the **None** dispatch path, removing the savepoint and truncating
    /// the logs is the whole pop; the cached solver drops its frame at the
    /// next query. In the **Some** (shared-lineage) dispatch
    /// path, this pops the most-recent savepoint off `scope_savepoints`
    /// and truncates `scope_path` back to that length, discarding any
    /// frames added after the matching
    /// [`scope_savepoint_push()`](Self::scope_savepoint_push).
    ///
    /// Symmetric with `scope_savepoint_push`:
    /// when the call stack is balanced (every push has a matching pop),
    /// `scope_savepoints` empties out and `scope_path` returns to its
    /// pre-push length.
    ///
    /// Mismatched pops (no preceding push) are ignored in both branches:
    /// there is no savepoint to remove.
    ///
    /// Does **not** invalidate `sat_cache` / `model_cache` on its own —
    /// the public wrapper (`pop()`) owns that.
    pub(super) fn scope_savepoint_pop(&self) {
        // angr-ph300.41/.42: discard any local constraints logged inside the
        // scope this pop closes, so they neither dedup-suppress a re-add nor
        // get replayed into a fork as permanent asserts. Balanced with the
        // record in `scope_savepoint_push`. A mismatched pop (no matching push)
        // leaves the logs untouched.
        // Pop into a local first so the `bare_local_savepoints` guard drops
        // before we acquire `local_constraints` — never hold both hot locks at
        // once (clippy nursery significant_drop_in_scrutinee, angr-zi35f.12).
        let popped = self.bare_local_savepoints.lock().pop();
        if let Some(super::context::Savepoint {
            z3_len,
            assumed_len,
            non_bv_len,
            ..
        }) = popped
        {
            let mut local = self.local_constraints.lock();
            if local.z3_assertions.len() > z3_len
                || local.assumed.len() > assumed_len
                || local.non_bv_assertions.len() > non_bv_len
            {
                local.z3_assertions.truncate(z3_len);
                local.assumed.truncate(assumed_len);
                local.non_bv_assertions.truncate(non_bv_len);
                // Stale dedup ptrs from the truncated assertions would falsely
                // dedup a re-assert (angr-ph300.41); drop the side-table and
                // let the next add_constraint_lowered rebuild it from shared+local.
                local.dedup_set.clear();
                local.dedup_set_seeded = false;
            }
        }
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                // Removing the savepoint above closed the scope; the cached
                // solver pops its frame at the next query (`sync_solver`).
                // A pop with no matching push leaves the counter at zero.
                let _unmatched_pop = self.bare_z3_push_depth.try_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |depth| depth.checked_sub(1),
                );
            }
            Some(_) => {
                if let Some(depth) = self.scope_savepoints.lock().pop() {
                    self.scope_path.lock().truncate(depth);
                }
            }
        }
    }
}
