//! Constraint-mutation `&self` methods for [`SymContext`].
//!
//! Slice 9 of the `symbolic/context.rs` split (angr-a2br.2.7). These are the
//! Z3-backed constraint-addition paths: the public mutators
//! (`add_constraint`, `add_constraint_raw`, `add_constraints_raw_batch`,
//! `add_constraint_tracked_indexed`, `add_bv_constraint`, `assume_true`,
//! `assume_false`) plus their private helpers — the ptr-keyed dedup pair
//! (`seed_and_check_z3_dedup` / `check_z3_dedup_if_seeded`) and the
//! model-cache invalidators (`invalidate_model_if_inconsistent` and its
//! batch variant).
//!
//! Every method here is `#[cfg(feature = "vex-engine-z3")]`, so the whole
//! module is gated behind the feature in `mod.rs` (no empty `impl` block in
//! the non-Z3 build). Lives as a second `impl SymContext` block in a child
//! module of `symbolic`; the fields these mutate (`local_constraints`,
//! `z3_assertions_shared`, `constraint_trackers`) plus the
//! [`LocalConstraints`] struct and its fields/`extend_assertions` method are
//! promoted to `pub(super)` (== `pub(in crate::symbolic)`) so this sibling
//! module can reach them without a public API leak. The read-path caches
//! `sat_cache`/`model_cache` and the `constraint_count`/`lineage`/`scope_path`
//! fields were already `pub(super)` from earlier slices. See bd memory
//! `a2br2-context-split-impl-block-plan` for the slice plan.

use super::RustBV;
use super::SymContext;

#[cfg(feature = "vex-engine-z3")]
use super::context::LocalConstraints;
#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;
#[cfg(feature = "vex-engine-z3")]
use super::stats::*;
#[cfg(feature = "vex-engine-z3")]
use std::sync::Arc;
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::Ordering;

impl SymContext {
    /// Lazily seed `local.dedup_set` from `shared + local z3_assertions`,
    /// then check whether `constraint`'s Z3_ast ptr is already known.
    ///
    /// Returns `true` if the constraint is a duplicate (caller should skip
    /// the Z3 assert + lineage push); the ptr is left out of `dedup_set` and
    /// `local.z3_assertions` is unchanged. Returns `false` otherwise, after
    /// inserting the ptr into `dedup_set` and pushing `constraint.clone()`
    /// onto `local.z3_assertions`.
    ///
    /// Caller holds `local` (the `local_constraints` lock guard inner) and
    /// is responsible for any SCANNED/HIT counter bookkeeping — this helper
    /// stays counter-agnostic so each call site (`add_constraint_raw`,
    /// `assume_true`, `assume_false`) attributes hits to its own counter
    /// pair (angr-mwbp).
    #[cfg(feature = "vex-engine-z3")]
    fn seed_and_check_z3_dedup(
        &self,
        local: &mut LocalConstraints,
        constraint: &z3::ast::Bool,
    ) -> bool {
        use z3::ast::Ast;
        if !local.dedup_set_seeded {
            let shared = Arc::clone(&self.z3_assertions_shared.lock());
            let local_len = local.z3_assertions.len();
            local.dedup_set.reserve(shared.len() + local_len);
            for c in shared.iter() {
                local.dedup_set.insert(c.get_z3_ast().as_ptr() as usize);
            }
            // Collect local ptrs into a separate Vec to avoid the
            // simultaneous mut+ref borrow on `local`.
            let local_ptrs: Vec<usize> = local
                .z3_assertions
                .iter()
                .map(|c| c.get_z3_ast().as_ptr() as usize)
                .collect();
            local.dedup_set.extend(local_ptrs);
            local.dedup_set_seeded = true;
        }
        Self::contains_or_insert_ptr(local, constraint)
    }

    /// Shared dedup tail: returns `true` if `constraint`'s Z3_ast ptr is
    /// already in `local.dedup_set` (duplicate — leave state untouched);
    /// otherwise inserts the ptr, pushes `constraint.clone()` onto
    /// `local.z3_assertions`, and returns `false`. Counter bookkeeping stays
    /// with the caller so each site attributes hits to its own counter.
    #[cfg(feature = "vex-engine-z3")]
    fn contains_or_insert_ptr(local: &mut LocalConstraints, constraint: &z3::ast::Bool) -> bool {
        use z3::ast::Ast;
        let new_ptr = constraint.get_z3_ast().as_ptr() as usize;
        if local.dedup_set.contains(&new_ptr) {
            true
        } else {
            local.dedup_set.insert(new_ptr);
            local.z3_assertions.push(constraint.clone());
            false
        }
    }

    /// Like [`Self::seed_and_check_z3_dedup`] but does NOT trigger seeding —
    /// only consults the dedup_set when already populated. If unseeded,
    /// falls through to the legacy [`LocalConstraints::push_assertion`]
    /// behavior (push to z3_assertions, conditionally track in dedup_set if
    /// seeded) and returns `false` so callers fall through to
    /// [`Self::add_constraint`].
    ///
    /// Used by `assume_true`/`assume_false` (angr-mwbp): seeding from these
    /// sites would walk `shared + local z3_assertions` on every fresh
    /// context's first call, an O(N) regression on branch-heavy benches
    /// without `add_constraint_raw` traffic. Piggy-backing on an existing
    /// seed (most commonly placed by `add_constraint_raw` in the bridge)
    /// gets the dedup benefit when it's free and avoids cost otherwise.
    #[cfg(feature = "vex-engine-z3")]
    fn check_z3_dedup_if_seeded(
        &self,
        local: &mut LocalConstraints,
        constraint: &z3::ast::Bool,
    ) -> bool {
        if !local.dedup_set_seeded {
            // No seed yet — match the legacy push_assertion behavior and
            // let the caller fall through to add_constraint.
            local.z3_assertions.push(constraint.clone());
            return false;
        }
        Z3_ASSUME_DEDUP_SCANNED_COUNT.fetch_add(1, Ordering::Relaxed);
        let dup = Self::contains_or_insert_ptr(local, constraint);
        // angr-gmad2 diagnostic: a dedup HIT should always be backed by a
        // live, structurally-equal Bool in shared+local z3_assertions —
        // Z3 hash-consing returns an existing ptr only for a live equal AST.
        // If a HIT ptr is NOT backed by any live assertion, it is a stale-ptr
        // false positive (a freed AST's address reused) and we would have
        // dropped an INTENDED constraint. Gated behind Debug so it is
        // zero-cost in production; run with RUST_LOG=debug to audit.
        if dup && log::log_enabled!(log::Level::Debug) {
            self.debug_verify_dedup_backing(local, constraint);
        }
        dup
    }

    /// angr-gmad2 diagnostic (Debug-gated): verify a dedup HIT's Z3_ast ptr is
    /// backed by a live structurally-equal Bool in `z3_assertions_shared` or
    /// `local.z3_assertions`. Emits `debug!` when backed (sound true-positive)
    /// and `warn!` when unbacked (stale-ptr false positive — a dropped
    /// constraint). O(N) scan, so only invoked under Debug logging.
    #[cfg(feature = "vex-engine-z3")]
    fn debug_verify_dedup_backing(&self, local: &LocalConstraints, constraint: &z3::ast::Bool) {
        use z3::ast::Ast;
        let ptr = constraint.get_z3_ast().as_ptr() as usize;
        let in_local = local
            .z3_assertions
            .iter()
            .any(|c| c.get_z3_ast().as_ptr() as usize == ptr);
        let in_shared = {
            let shared = Arc::clone(&self.z3_assertions_shared.lock());
            shared
                .iter()
                .any(|c| c.get_z3_ast().as_ptr() as usize == ptr)
        };
        if in_local || in_shared {
            log::debug!(
                target: "rustylib::symbolic",
                "assume-dedup HIT ptr={ptr:#x} backed (local={in_local} shared={in_shared}) -- sound true-positive"
            );
        } else {
            log::warn!(
                target: "rustylib::symbolic",
                "assume-dedup HIT ptr={ptr:#x} UNBACKED by any live z3_assertion -- STALE-PTR FALSE POSITIVE (intended constraint dropped)"
            );
        }
    }

    /// Add a constraint from a typed Z3 AST handle (shared context fast path).
    ///
    /// This bypasses the RustBV → build_z3_ast_cached() conversion, preserving
    /// the original Z3 AST structure from Python's claripy/z3 backend.
    ///
    /// The `Z3AstPtr` handle carries its own refcount; on entry, this
    /// function wraps the pointer as a [`z3::ast::Bool`] (which takes its
    /// own ref via `Z3_inc_ref`) and the handle's `Drop` releases the
    /// extraction-time ref before return — net zero change to the AST's
    /// refcount across the call.
    ///
    /// Dispatches on `self.lineage` (angr-v5a5 slice 4c.2; mirrors the
    /// pattern landed in slice 4c.1 for [`Self::add_constraint`]):
    ///
    /// - **None** (today's only production path): asserts the constraint
    ///   on the per-context Z3 solver — byte-identical to the pre-slice
    ///   `self.with_z3_solver(|s| s.assert(&constraint))` call.
    /// - **Some** (shared-lineage): mints a fresh
    ///   [`ScopeFrame`](super::lineage::ScopeFrame) carrying the constraint,
    ///   appends it to `self.scope_path`, and calls
    ///   [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
    ///   to push the new frame onto the shared solver. The
    ///   `constraint.clone()` is a ref-bump on the same Z3 AST that came
    ///   in via `Ast::wrap` — no additional Z3 allocations.
    ///
    /// See [`Self::add_constraint`] for the full rationale on why the
    /// Some branch can't route through `with_z3_solver` (would put the
    /// assert at scope 0 = lineage base = leak to all siblings).
    ///
    /// **Caller contract:** the wrapped pointer must denote a Bool-sorted
    /// AST. The constructor of `Z3AstPtr` is `unsafe` precisely so this
    /// invariant is checked at extraction time; once a `Z3AstPtr` exists,
    /// this method is safe to call.
    /// Install one constraint via the active lineage mode, then run the
    /// standard post-assert bookkeeping (constraint_count bump, sat_cache
    /// clear, model-consistency invalidation).
    ///
    /// `assert_none` runs only on the non-lineage (`None`) path with the
    /// per-context solver guard already acquired — it is where the single
    /// real per-caller difference lives (`assert` vs `assert_and_track`).
    /// The `Some` (shared-lineage) path mints a fresh
    /// [`ScopeFrame`](super::lineage::ScopeFrame) carrying the constraint,
    /// pushes it onto `scope_path`, and calls
    /// [`switch_to`](super::lineage::SharedLineageSolver::switch_to); see
    /// [`Self::add_constraint`] for why the `Some` branch can't route
    /// through `with_z3_solver`'s closure form. Shared by
    /// [`Self::add_constraint_raw`], [`Self::add_constraint`], and
    /// [`Self::add_constraint_tracked_indexed`].
    #[cfg(feature = "vex-engine-z3")]
    fn install_constraint(
        &self,
        constraint: &z3::ast::Bool,
        assert_none: impl FnOnce(&z3::Solver),
    ) {
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                let solver = self.solver();
                assert_none(&solver);
            }
            Some(lin) => {
                let frame = super::lineage::ScopeFrame::new(true, constraint.clone());
                let path_snapshot = {
                    let mut sp = self.scope_path.lock();
                    sp.push(frame);
                    sp.clone()
                };
                let mut guard = lin.lock();
                guard.switch_to(&path_snapshot);
            }
        }
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        self.sat_cache.set(None);
        self.invalidate_model_if_inconsistent(constraint);
    }

    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint_raw(&self, ast: super::Z3AstPtr) {
        let ctx = z3::Context::thread_local();
        // SAFETY: `ast` is a live `Z3_ast` (the `Z3AstPtr` holds an active
        // ref via `Z3_inc_ref`). The pointer denotes a Bool by the
        // documented caller contract. `Ast::wrap` performs its own
        // `Z3_inc_ref` so the wrapped `Bool` is independent of `ast`'s
        // ref, which drops at end of function.
        let constraint: z3::ast::Bool = unsafe { z3::ast::Ast::wrap(&ctx, ast.as_z3_ast()) };
        ADD_CONSTRAINT_RAW_TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);
        sample_simplify_skip(&constraint);
        // angr-sfp9: ptr-keyed dedup against the side-table. Z3 hash-cons
        // makes ptr-equality == structural-equality among live ASTs, so a
        // hit means this exact Bool is already asserted on the solver via
        // an earlier assert (in shared+local z3_assertions) — skipping the
        // re-assert is correct (Z3 internally treats repeated asserts as a
        // single fact) and avoids growing `z3_assertions` with a duplicate
        // that would scale constraint-export work for nothing.
        ADD_CONSTRAINT_RAW_DEDUP_SCANNED_COUNT.fetch_add(1, Ordering::Relaxed);
        let was_dup = {
            let mut local = self.local_constraints.lock();
            let dup = self.seed_and_check_z3_dedup(&mut local, &constraint);
            if !dup {
                // angr-t3l5o Phase 1: residual sink #1. This no-RustBV
                // constraint (Python claripy-sync fallback / cross-process
                // ptr import) has no `assumed` entry to reconstruct it from,
                // so record it in the residual log to round-trip via
                // `residual_smtlib2`. Only on the non-dup path — a dup is
                // already counted on its first add. `seed_and_check_z3_dedup`
                // pushed the same Bool to `z3_assertions` just above, so the
                // residual log stays a subset of `z3_assertions`.
                local.non_bv_assertions.push(constraint.clone());
            }
            dup
        };
        if was_dup {
            ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
            // Already asserted on the solver and tracked in z3_assertions —
            // no solver work, no cache invalidation, no constraint_count
            // bump (the logical assertion was already counted on the
            // initial add).
            return;
        }
        self.install_constraint(&constraint, |s| s.assert(&constraint));
    }

    /// Add a constraint (fast path: no tracking overhead).
    ///
    /// Dispatches on `self.lineage` (angr-v5a5 slice 4c.1):
    ///
    /// - **None** (today's only production path): asserts the constraint
    ///   on the per-context Z3 solver — byte-identical to the pre-slice
    ///   `self.with_z3_solver(|s| s.assert(&constraint))` call.
    /// - **Some** (shared-lineage): mints a fresh
    ///   [`ScopeFrame`](super::lineage::ScopeFrame) carrying the constraint,
    ///   appends it to `self.scope_path`, and calls
    ///   [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
    ///   to push the new frame onto the shared solver. The constraint
    ///   lands in its own scope, visible only to states whose
    ///   `scope_path` includes this frame's id — preserving per-state
    ///   isolation across siblings.
    ///
    /// Why not just use `with_z3_solver` in the Some path: `with_z3_solver`
    /// switches the shared solver to the caller's *current* `scope_path`
    /// and then runs `solver.assert(&c)`. That puts `c` at scope
    /// `scope_path.len()` (the most-recently-pushed level). For a freshly
    /// forked state with empty `scope_path`, that scope is 0 — the
    /// lineage base — so the constraint would leak to every sibling
    /// instead of being state-private. Mint-frame-then-switch keeps the
    /// assert inside a fresh push that only this state holds in its
    /// `scope_path`.
    ///
    /// The `is_true` field on the new frame is set to `true` because
    /// callers (`assume_true`/`assume_false`) have already done the
    /// negation in the Z3 Bool passed in. The flag is metadata for
    /// [`switch_to`](super::lineage::SharedLineageSolver::switch_to) — it
    /// only reads `z3_assertion`, so the flag doesn't affect Z3 semantics.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint(&self, constraint: z3::ast::Bool) {
        // Use plain assert for fast path (no unsat_core tracking overhead).
        // This avoids creating tracking booleans, string formatting, and
        // mutex acquisition on constraint_trackers for every constraint.
        // NOTE: Don't cache here — callers (assume_true, assume_false,
        // add_constraint_raw) cache before calling this to avoid double-cache.
        self.install_constraint(&constraint, |s| s.assert(&constraint));
    }

    /// Batched fast path for `add_constraint_raw`: asserts N constraints under
    /// one `local_constraints` lock, one solver/lineage transition, and one
    /// model invalidation pass. Each `(z3_ast, bv, is_true)` tuple
    /// corresponds to the per-constraint metadata that the single-shot path
    /// stores in `local_constraints.{z3_assertions, assumed}`.
    ///
    /// Same precondition as [`Self::add_constraint_raw`]: every `Z3AstPtr`
    /// must denote a Bool-sorted AST in the active thread-local Z3 context.
    /// The constructor of `Z3AstPtr` is `unsafe` so this is checked at
    /// extraction time.
    ///
    /// Dispatches on `self.lineage` (angr-v5a5 slice 4c.2c; mirrors the
    /// pattern landed in slice 4c.1 for [`Self::add_constraint`], slice 4c.2
    /// for [`Self::add_constraint_raw`], and slice 4c.2b for
    /// [`Self::add_constraint_tracked_indexed`]):
    ///
    /// - **None** (today's only production path): asserts all N constraints
    ///   on the per-context Z3 solver under a single solver guard —
    ///   byte-identical to the pre-slice `self.with_z3_solver(|s| { for c
    ///   in &constraints { s.assert(c); } })` call.
    /// - **Some** (shared-lineage): mints N fresh
    ///   [`ScopeFrame`](super::lineage::ScopeFrame)s carrying clones of the
    ///   constraints, appends them all to `self.scope_path` under one
    ///   `scope_path.lock()` acquisition, snapshots the new path, drops the
    ///   `scope_path` lock, and calls
    ///   [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
    ///   once. `switch_to`'s tail walks the divergent suffix (here, the N
    ///   new frames) and asserts each one under the shared solver — N
    ///   ref-bumps but only a single round-trip through the lineage Mutex.
    ///   Cheaper than N separate `add_constraint_raw` calls would be
    ///   (each of those is one lock + one switch_to call).
    ///
    /// Why not just route through `with_z3_solver` in the Some branch: same
    /// rationale as [`Self::add_constraint`] — `with_z3_solver`'s Some path
    /// switches to the caller's *current* `scope_path` and then asserts at
    /// scope_path.len() (the most-recently-pushed level). For a freshly
    /// forked state with empty `scope_path`, that is scope 0 = the lineage
    /// base, so the N constraints would leak to all siblings. Mint-N-frames-
    /// then-switch keeps every constraint inside its own fresh push that
    /// only this state holds in its `scope_path`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraints_raw_batch(&self, entries: Vec<(super::Z3AstPtr, RustBV, bool)>) {
        if entries.is_empty() {
            return;
        }
        let z3_ctx = z3::Context::thread_local();
        let mut constraints: Vec<z3::ast::Bool> = Vec::with_capacity(entries.len());
        let mut assumed: Vec<(RustBV, bool)> = Vec::with_capacity(entries.len());
        for (ast, bv, is_true) in entries {
            // SAFETY: `ast` is a live `Z3_ast` (the `Z3AstPtr` holds an
            // active ref via `Z3_inc_ref`). The pointer denotes a Bool by
            // the documented caller contract. `Ast::wrap` performs its
            // own `Z3_inc_ref` so the wrapped `Bool` is independent of
            // `ast`'s ref, which drops at end of this iteration.
            let constraint: z3::ast::Bool = unsafe { z3::ast::Ast::wrap(&z3_ctx, ast.as_z3_ast()) };
            constraints.push(constraint);
            assumed.push((bv, is_true));
        }
        // Single lock on local_constraints for both vectors.
        {
            let mut local = self.local_constraints.lock();
            local.extend_assertions(constraints.iter().cloned());
            local.assumed.extend(assumed);
        }
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                // Single solver guard for all assertions — byte-identical to
                // the pre-slice with_z3_solver(|s| { for c { s.assert(c); }})
                // call.
                let solver = self.solver();
                for c in &constraints {
                    solver.assert(c);
                }
            }
            Some(lin) => {
                // Mint N frames under one scope_path lock, snapshot, drop,
                // then a single switch_to call walks the divergent suffix
                // and asserts each new frame on the shared solver.
                let path_snapshot = {
                    let mut sp = self.scope_path.lock();
                    for c in &constraints {
                        sp.push(super::lineage::ScopeFrame::new(true, c.clone()));
                    }
                    sp.clone()
                };
                let mut guard = lin.lock();
                guard.switch_to(&path_snapshot);
            }
        }
        self.constraint_count
            .fetch_add(constraints.len(), Ordering::SeqCst);
        self.sat_cache.set(None);
        self.invalidate_model_if_inconsistent_batch(&constraints);
    }

    /// Batched model invalidation: if the cached model fails to satisfy any
    /// constraint in `constraints`, drop it. Short-circuits on the first
    /// inconsistency. If the cache is already empty, returns immediately.
    #[cfg(feature = "vex-engine-z3")]
    fn invalidate_model_if_inconsistent_batch(&self, constraints: &[z3::ast::Bool]) {
        let mut cache = self.model_cache.borrow_mut();
        if cache.is_none() {
            return;
        }
        let mut still_valid = true;
        if let Some(model) = cache.as_ref() {
            for constraint in constraints {
                let ok = model
                    .eval(constraint, true)
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);
                if !ok {
                    still_valid = false;
                    break;
                }
            }
        }
        if !still_valid {
            *cache = None;
        }
    }

    /// Add a constraint with tracking for unsat_core extraction.
    /// Use this only when unsat_core analysis is needed.
    ///
    /// Returns the tracker index assigned to this constraint, which is also
    /// the index that will appear in [`Self::unsat_core`] output if the
    /// constraint participates in the unsat core.
    ///
    /// Dispatches on `self.lineage` (angr-v5a5 slice 4c.2b; mirrors the
    /// pattern landed in slice 4c.1 for [`Self::add_constraint`] and slice
    /// 4c.2 for [`Self::add_constraint_raw`]):
    ///
    /// - **None** (today's only production path): calls
    ///   `solver.assert_and_track(&constraint, &track_bool)` on the
    ///   per-context Z3 solver — byte-identical to the pre-slice
    ///   `self.with_z3_solver(|s| s.assert_and_track(...))` call. Full
    ///   unsat-core fidelity preserved.
    /// - **Some** (shared-lineage): mints a fresh
    ///   [`ScopeFrame`](super::lineage::ScopeFrame) carrying the bare
    ///   constraint, appends it to `self.scope_path`, and calls
    ///   [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
    ///   to push the new frame onto the shared solver. The tracker is
    ///   still registered in `constraint_trackers` so the returned index
    ///   stays stable, but the frame's `z3_assertion` is asserted via
    ///   plain `assert` inside `switch_to` — Z3's `get_unsat_core()` will
    ///   NOT report this constraint's tracker if it participates in an
    ///   unsat core under a lineage-installed context. Lineage-mode
    ///   unsat-core fidelity is intentionally deferred: switch_to would
    ///   need an `assert_and_track`-aware variant (and a tracker field on
    ///   `ScopeFrame`) to re-register the tracker on every state load.
    ///   See `add_constraint`'s rationale for why the Some branch can't
    ///   route through `with_z3_solver`'s closure form (it would land
    ///   the assert at scope 0 = the lineage base = sibling leak).
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint_tracked_indexed(&self, constraint: z3::ast::Bool) -> usize {
        let idx = self.constraint_count.load(Ordering::SeqCst);
        let track_name = format!("__track_{idx}");
        let track_bool = z3::ast::Bool::new_const(track_name.as_str());

        let tracker_idx = {
            let mut trackers = self.constraint_trackers.lock();
            let i = trackers.len();
            trackers.push(track_bool.clone());
            i
        };

        self.install_constraint(&constraint, |s| {
            s.assert_and_track(&constraint, &track_bool)
        });
        tracker_idx
    }

    /// If a cached model exists, drop it unless it still satisfies the new
    /// constraint. Models that satisfy a superset of constraints stay valid;
    /// this lets check_branch_feasibility reuse a model across consecutive
    /// assume_true/assume_false calls in deferred-fork mode.
    #[cfg(feature = "vex-engine-z3")]
    fn invalidate_model_if_inconsistent(&self, constraint: &z3::ast::Bool) {
        let mut cache = self.model_cache.borrow_mut();
        if cache.is_none() {
            return;
        }
        let still_valid = cache
            .as_ref()
            .and_then(|m| m.eval(constraint, true))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        if !still_valid {
            *cache = None;
        }
    }

    /// Add a constraint that the bitvector equals a specific value.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_bv_constraint(&self, bv: &RustBV, value: u128) {
        // Fast path: if bv is already concrete, the constraint is either
        // trivially true (skip) or trivially false (makes UNSAT).
        if let Some(v) = bv.as_u128()
            && v == value
        {
            return; // Tautology — skip Z3
        }
        // Falls through to add False constraint (UNSAT)
        let ast = bv.to_z3_ast();
        let val_ast = super::bv_codec::make_bv_const(value, bv.width());
        let constraint = ast.eq(&val_ast);
        // angr-t3l5o Phase 1: residual sink #2 (address concretization).
        // `add_constraint` (below) only asserts on the live solver and does
        // NOT seed the `z3_assertions` log, so without these pushes the
        // constraint would be lost on fork (solver rebuild replays
        // `z3_assertions`) and on snapshot. Push to BOTH logs: `z3_assertions`
        // so a forked/rematerialized solver replays it, and `non_bv_assertions`
        // so it round-trips via `residual_smtlib2` (it has no `assumed` entry).
        {
            let mut local = self.local_constraints.lock();
            local.push_assertion(constraint.clone());
            local.non_bv_assertions.push(constraint.clone());
        }
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is true (non-zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_true(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Fast path: concrete true is a tautology — skip Z3 entirely (still
        // record it for Python export).
        if let Some(v) = cond.as_u128()
            && v != 0
        {
            self.local_constraints
                .lock()
                .assumed
                .push((cond.clone(), true));
            Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
            return; // Asserting True is a no-op
        }
        // v == 0: asserting False makes solver UNSAT — still add it
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Use to_z3_bool() to produce native Z3 Bool for comparison ops,
        // avoiding ITE(cmp, BV(1,1), BV(0,1)).eq(BV(1,1)) round-trip.
        let constraint = cond.to_z3_bool();
        // angr-1joc measurement: sampled simplify-skip check.
        sample_simplify_skip(&constraint);
        // angr-mwbp: piggy-back dedup on the side-table when it's already
        // seeded by an earlier `add_constraint_raw` call in this context.
        // Triggering the seed from here would walk shared+local on every
        // assume_*/false call in fresh contexts — a real bench regression
        // on heavily-branched workloads where add_constraint_raw isn't
        // hot (flareon2015_2 timed out at 30s under unconditional seeding).
        // When seeded, `assumed` still grows unconditionally to preserve
        // Python-visible duplicate constraints (claripy `solver.add(c)`
        // semantics) — dedup only short-circuits the redundant
        // `z3_assertions.push` + `add_constraint(c)` round-trip.
        let was_dup = {
            let mut local = self.local_constraints.lock();
            local.assumed.push((cond.clone(), true));
            self.check_z3_dedup_if_seeded(&mut local, &constraint)
        };
        if was_dup {
            Z3_ASSUME_DEDUP_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_false(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Fast path: concrete false (== 0) means not(False) = True — skip Z3
        if let Some(v) = cond.as_u128()
            && v == 0
        {
            self.local_constraints
                .lock()
                .assumed
                .push((cond.clone(), false));
            Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
            return; // Asserting not(False) = True is a no-op
        }
        // v != 0: asserting not(True) = False makes solver UNSAT — still add it
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Negate the bool directly
        let constraint = cond.to_z3_bool().not();
        // angr-1joc measurement: sampled simplify-skip check.
        sample_simplify_skip(&constraint);
        // angr-mwbp: see assume_true above for the dedup contract.
        let was_dup = {
            let mut local = self.local_constraints.lock();
            local.assumed.push((cond.clone(), false));
            self.check_z3_dedup_if_seeded(&mut local, &constraint)
        };
        if was_dup {
            Z3_ASSUME_DEDUP_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.add_constraint(constraint);
    }
}
