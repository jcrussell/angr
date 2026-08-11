//! Lineage / shared-solver `&self` methods for [`SymContext`].
//!
//! Slice 6 of the `symbolic/context.rs` split (angr-a2br.2.5). These are the
//! most field-isolated methods of the original monolithic `impl SymContext`
//! block: they touch only the lineage cells (`lineage`, `scope_path`,
//! `scope_savepoints`, `use_shared_lineage_solver`, `bare_z3_push_depth`) —
//! promoted to `pub(super)` so this sibling module can reach them — plus the
//! `solver()` accessor (also `pub(super)`) on the savepoint push/pop path.
//! See bead angr-a2br.2.4 for the slice plan and
//! `rust_z3_sharing.rst` for the fork/merge atomic-snapshot invariant that
//! gates which methods stay in `context.rs`.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! `pub(super)` (== `pub(in crate::symbolic)`) keeps the promoted fields
//! module-private — no public API leak.

use super::SymContext;
use super::sharing::ConstraintSharingWalk;

#[cfg(feature = "vex-engine-z3")]
use parking_lot::Mutex;
#[cfg(feature = "vex-engine-z3")]
use std::sync::Arc;
#[cfg(feature = "vex-engine-z3")]
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
    /// and be scored as already-seen (angr-gkcxh — see the `sharing` module
    /// doc).
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
    #[cfg(feature = "vex-engine-z3")]
    pub fn lineage_arc(&self) -> Option<Arc<Mutex<super::lineage::SharedLineageSolver>>> {
        self.lineage.lock().as_ref().map(Arc::clone)
    }

    /// Current per-state scope-path depth (angr-v5a5 spike).
    ///
    /// Always 0 while `lineage` is `None` (the default-off case) — frames
    /// are minted only when `assume_*` routes through the lineage solver.
    /// Exposed as a telemetry surface for both modes.
    #[cfg(feature = "vex-engine-z3")]
    pub fn scope_path_len(&self) -> usize {
        self.scope_path.lock().len()
    }

    /// Current size of this context's scope-savepoint stack (angr-v5a5
    /// slice 4b).
    ///
    /// Always 0 while `lineage` is `None` (the default-off case): that
    /// branch uses the per-context Z3 solver's native `push()/pop()`
    /// instead. Exposed for telemetry and test assertions.
    #[cfg(feature = "vex-engine-z3")]
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
    #[cfg(feature = "vex-engine-z3")]
    pub fn bare_z3_push_depth(&self) -> usize {
        self.bare_z3_push_depth.load(Ordering::Relaxed)
    }

    /// Current length of the local `z3_assertions` log (angr-ph300.41).
    /// Test-only white-box accessor for asserting that a bare `pop()`
    /// truncates constraints added inside the popped scope.
    #[cfg(all(test, feature = "vex-engine-z3"))]
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
    #[cfg(feature = "vex-engine-z3")]
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
    /// **Do NOT default this on for any single exploration strategy.** The
    /// angr-ua1i proposal — flip default-on for `strategy='dfs'` — is
    /// the wrong direction: both the canonical WIN canary (ais3_crackme
    /// 1.24x) and LOSE canary (defcon2016quals_baby-re 6.79x slower) run
    /// BFS in the bench harness, so strategy is not the discriminator.
    /// The replacement is runtime thrash detection in
    /// [`super::lineage::sample_for_thrash`], which is strategy-agnostic
    /// and ships in `tick_and_sample_for_thrash` from `run_loop`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_use_shared_lineage_solver(&self, v: bool) {
        self.use_shared_lineage_solver.store(v, Ordering::Relaxed);
    }

    /// Save a scope-path savepoint, dispatching by lineage (angr-v5a5
    /// slice 4b).
    ///
    /// In the **None** dispatch path (today's only production path),
    /// this directly pushes the per-context Z3 solver — preserving the
    /// pre-slice behavior of `self.solver().push()`. In the **Some**
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
    #[cfg(feature = "vex-engine-z3")]
    pub(super) fn scope_savepoint_push(&self) {
        // angr-ph300.41/.42: record the local-constraint log lengths so the
        // matching bare `pop()` can truncate everything added inside this
        // scope. Captured before the lineage dispatch so both branches share it.
        {
            let local = self.local_constraints.lock();
            self.bare_local_savepoints.lock().push((
                local.z3_assertions.len(),
                local.assumed.len(),
                local.non_bv_assertions.len(),
            ));
        }
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                let solver = self.solver();
                solver.push();
                // Release the per-context z3::Solver guard before the
                // lock-independent atomic so the hottest lock in the engine
                // is not held across the fetch_add (clippy nursery
                // significant_drop_tightening).
                drop(solver);
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
    /// In the **None** dispatch path, this directly pops the per-context
    /// Z3 solver — preserving the pre-slice behavior of
    /// `self.solver().pop(1)`. In the **Some** (shared-lineage) dispatch
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
    /// Mismatched pops (no preceding push) are silently ignored in the
    /// Some branch — `scope_savepoints` simply empties out. The None branch
    /// inherits z3-rs's behavior (a panic on under-popping the solver),
    /// which `try_pop()` guards against via `bare_z3_push_depth`.
    ///
    /// Does **not** invalidate `sat_cache` / `model_cache` on its own —
    /// the public wrapper (`pop()`) owns that.
    #[cfg(feature = "vex-engine-z3")]
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
        if let Some((z3_len, assumed_len, non_bv_len)) = popped {
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
                // let the next add_constraint_raw rebuild it from shared+local.
                local.dedup_set.clear();
                local.dedup_set_seeded = false;
            }
        }
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                let solver = self.solver();
                solver.pop(1);
                // Release the per-context z3::Solver guard before the
                // lock-independent atomic + debug_assert so the hottest lock
                // in the engine is not held across them (clippy nursery
                // significant_drop_tightening).
                drop(solver);
                // angr-3ms1 step 1a: decrement after the Z3 pop succeeds.
                // z3-rs panics on under-pop, so we never reach this on
                // an unbalanced sequence — the counter stays in sync
                // with the per-context solver's actual push depth. Kept
                // debug-only (angr-9ke6b.220) for exactly that reason: the
                // failure is already caught loudly one line up.
                let prev = self.bare_z3_push_depth.fetch_sub(1, Ordering::Relaxed);
                debug_assert!(
                    prev > 0,
                    "bare_z3_push_depth underflowed — pop without matching push"
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
