//! Constraint-mutation `&self` methods for [`SymContext`].
//!
//! Slice 9 of the `symbolic/context.rs` split (angr-a2br.2.7). These are the
//! Z3-backed constraint-addition paths: the public mutators
//! (`add_constraint_lowered`, `add_constraint_lowered_assumed`,
//! `add_constraints_raw_batch`, `add_constraint_tracked_lowered`,
//! `add_bv_constraint`, `assume_true`, `assume_false`) plus their private helpers — the ptr-keyed dedup pair
//! (`seed_and_check_z3_dedup` / `check_z3_dedup_if_seeded`) and the
//! model-cache invalidators (`invalidate_model_if_inconsistent` and its
//! batch variant).
//!
//! Lives as a second `impl SymContext` block in a child
//! module of `symbolic`; the fields these mutate (`local_constraints`,
//! `z3_assertions_shared`, `constraint_trackers`) plus the
//! [`LocalConstraints`] struct and its fields are promoted to `pub(super)`
//! (== `pub(in crate::symbolic)`) so this sibling module can reach them
//! without a public API leak. The read-path caches
//! `sat_cache`/`model_cache` and the `constraint_count`/`lineage`/`scope_path`
//! fields were already `pub(super)` (angr-a2br.2.4).

use super::RustBV;
use super::SymContext;
use super::ToZ3;

use super::local_constraints::{LocalConstraints, LoweredConstraint, Origin};
use super::model_cache::ModelRef;
use super::solver_build::*;
use super::stats::*;
use std::sync::Arc;
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
    /// stays counter-agnostic so each call site (`add_constraint_lowered`,
    /// `assume_true`, `assume_false`) attributes hits to its own counter
    /// pair (angr-mwbp).
    fn seed_and_check_z3_dedup(
        &self,
        local: &mut LocalConstraints,
        constraint: &z3::ast::Bool,
        origin: Origin,
    ) -> bool {
        use z3::ast::Ast;
        if !local.dedup_set_seeded {
            let shared = Arc::clone(&self.z3_assertions_shared.lock());
            let local_len = local.z3_assertions.len();
            local.dedup_set.reserve(shared.len() + local_len);
            for c in shared.iter() {
                local.dedup_set.insert(c.z3.get_z3_ast().as_ptr() as usize);
            }
            // Collect local ptrs into a separate Vec to avoid the
            // simultaneous mut+ref borrow on `local`.
            let local_ptrs: Vec<usize> = local
                .z3_assertions
                .iter()
                .map(|c| c.z3.get_z3_ast().as_ptr() as usize)
                .collect();
            local.dedup_set.extend(local_ptrs);
            local.dedup_set_seeded = true;
        }
        Self::contains_or_insert_ptr(local, constraint, origin)
    }

    /// Shared dedup tail: returns `true` if `constraint`'s Z3_ast ptr is
    /// already in `local.dedup_set` (duplicate — leave state untouched);
    /// otherwise inserts the ptr, pushes `constraint.clone()` onto
    /// `local.z3_assertions`, and returns `false`. Counter bookkeeping stays
    /// with the caller so each site attributes hits to its own counter.
    fn contains_or_insert_ptr(
        local: &mut LocalConstraints,
        constraint: &z3::ast::Bool,
        origin: Origin,
    ) -> bool {
        use z3::ast::Ast;
        let new_ptr = constraint.get_z3_ast().as_ptr() as usize;
        if local.dedup_set.contains(&new_ptr) {
            true
        } else {
            local.dedup_set.insert(new_ptr);
            local.log_asserted(constraint.clone(), origin);
            false
        }
    }

    /// Like [`Self::seed_and_check_z3_dedup`] but does NOT trigger seeding —
    /// only consults the dedup_set when already populated. If unseeded,
    /// falls through to the legacy [`LocalConstraints::push_assertion`]
    /// behavior (push to z3_assertions, conditionally track in dedup_set if
    /// seeded) and returns `false` so callers fall through to
    /// `install_constraint`.
    ///
    /// Used by `assume_true`/`assume_false` (angr-mwbp): seeding from these
    /// sites would walk `shared + local z3_assertions` on every fresh
    /// context's first call, an O(N) regression on branch-heavy benches
    /// without `add_constraint_lowered` traffic. Piggy-backing on an existing
    /// seed (most commonly placed by `add_constraint_lowered` in the bridge)
    /// gets the dedup benefit when it's free and avoids cost otherwise.
    fn check_z3_dedup_if_seeded(
        &self,
        local: &mut LocalConstraints,
        constraint: &z3::ast::Bool,
        origin: Origin,
    ) -> bool {
        if !local.dedup_set_seeded {
            // No seed yet — match the legacy push_assertion behavior and
            // let the caller fall through to install_constraint.
            local.log_asserted(constraint.clone(), origin);
            return false;
        }
        Z3_ASSUME_DEDUP_SCANNED_COUNT.fetch_add(1, Ordering::Relaxed);
        let dup = Self::contains_or_insert_ptr(local, constraint, origin);
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

    /// The backing lookup behind [`Self::debug_verify_dedup_backing`]: does
    /// `ptr` name a live Bool in `local.z3_assertions` / `z3_assertions_shared`?
    ///
    /// Returns `(in_local, in_shared)`; `(false, false)` is the STALE-PTR
    /// condition — a dedup HIT on a ptr no live assertion holds, i.e. an
    /// intended constraint about to be dropped. Split out from the `log!`
    /// wrapper so the detector's decision is testable without installing a
    /// competing process-global `log` subscriber (`log::set_logger` is a
    /// one-shot singleton already claimed elsewhere in this test binary — see
    /// `constraint_ops_tests::test_dedup_backing_of_ptr_flags_stale_ptr_hit`).
    fn dedup_backing_of_ptr(&self, local: &LocalConstraints, ptr: usize) -> (bool, bool) {
        use z3::ast::Ast;
        let in_local = local
            .z3_assertions
            .iter()
            .any(|c| c.z3.get_z3_ast().as_ptr() as usize == ptr);
        let in_shared = {
            let shared = Arc::clone(&self.z3_assertions_shared.lock());
            shared
                .iter()
                .any(|c| c.z3.get_z3_ast().as_ptr() as usize == ptr)
        };
        (in_local, in_shared)
    }

    /// angr-gmad2 diagnostic (Debug-gated): verify a dedup HIT's Z3_ast ptr is
    /// backed by a live structurally-equal Bool in `z3_assertions_shared` or
    /// `local.z3_assertions`. Emits `debug!` when backed (sound true-positive)
    /// and `warn!` when unbacked (stale-ptr false positive — a dropped
    /// constraint). O(N) scan, so only invoked under Debug logging.
    fn debug_verify_dedup_backing(&self, local: &LocalConstraints, constraint: &z3::ast::Bool) {
        use z3::ast::Ast;
        let ptr = constraint.get_z3_ast().as_ptr() as usize;
        let (in_local, in_shared) = self.dedup_backing_of_ptr(local, ptr);
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

    /// Install one constraint via the active lineage mode, then run the
    /// standard post-assert bookkeeping (constraint_count bump, sat_cache
    /// clear, model-consistency invalidation).
    ///
    /// The caller has already put the constraint in the assertion log. On
    /// the non-lineage (`None`) path that is the install: the context's
    /// solver takes it from the log at the next query
    /// (`SymContext::sync_solver`), so an add costs no Z3 call and builds no
    /// solver (angr-az6ru.2).
    /// The `Some` (shared-lineage) path mints a fresh
    /// [`ScopeFrame`](super::lineage::ScopeFrame) carrying the constraint,
    /// appends it to `self.scope_path`, and calls
    /// [`switch_to`](super::lineage::SharedLineageSolver::switch_to) to push
    /// the new frame onto the shared solver. The constraint lands in its own
    /// scope, visible only to states whose `scope_path` includes this frame's
    /// id, which keeps siblings isolated (angr-v5a5 slice 4c.1).
    ///
    /// Why the `Some` path does not use `with_z3_solver`: that switches the
    /// shared solver to the caller's *current* `scope_path` and then asserts
    /// at scope `scope_path.len()`. For a freshly forked state with an empty
    /// `scope_path` that is scope 0, the lineage base, so the constraint
    /// would leak to every sibling. Mint-frame-then-switch keeps the assert
    /// inside a fresh push that only this state holds in its `scope_path`.
    ///
    /// The frame's `is_true` is always `true`: the callers have already
    /// negated the Z3 Bool they pass, and `switch_to` reads only
    /// `z3_assertion`.
    ///
    /// Shared by every adder in this file.
    pub(super) fn install_constraint(&self, constraint: &z3::ast::Bool) {
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {}
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

    /// Add a constraint a caller lowered from a clarirs Bool it still holds
    /// (the Python-origin sinks, a snapshot's residual terms).
    ///
    /// The Z3 form is the caller's lowering, so the structure Python's
    /// claripy built is what the solver sees. The constraint is deduped
    /// against the side table and, on a miss, installed via
    /// `install_constraint`, which dispatches on `self.lineage` (angr-v5a5
    /// slice 4c.2).
    ///
    /// A non-dup constraint is also pushed to the residual log
    /// (`non_bv_assertions`), since it has no `assumed` pair to be rebuilt
    /// from; [`Self::add_constraint_lowered_assumed`] is the variant that
    /// skips that. The entry keeps the term, so the constraint can be rebuilt
    /// without Z3 (angr-az6ru.2).
    pub fn add_constraint_lowered(&self, constraint: LoweredConstraint) {
        let (term, z3) = constraint.into_parts();
        self.add_constraint_raw_inner(z3, Origin::Term(term), true);
    }

    /// Assert every constraint of `log`, another context's assertion log, on
    /// this one. Each lands in the residual log (this context has no
    /// `assumed` pair for it) with the origin it had, so one that could be
    /// rebuilt from clarirs there can be here.
    pub fn import_constraint_log(&self, log: &super::ConstraintLog) {
        for entry in &log.0 {
            self.add_constraint_raw_inner(entry.z3.clone(), entry.origin.clone(), true);
        }
    }

    /// [`Self::add_constraint_lowered`] for a caller that records an `assumed`
    /// pair for the same constraint (`_add_constraints_to_state`'s lowered
    /// fast path).
    ///
    /// Skips the residual log: the residual sink exists so constraints with no
    /// RustBV form can still be round-tripped, and this one HAS a RustBV form.
    /// Logging it in both places is not just redundant, it is wrong for
    /// `unsat_core_assumed`, which asserts residuals *unguarded* (they have no
    /// claripy AST to blame) — a constraint sitting in both lists would be
    /// pinned outside its assumption literal, and an all-Python contradiction
    /// would come back Unsat with an EMPTY core (angr-op0dn.14.2).
    pub fn add_constraint_lowered_assumed(&self, constraint: LoweredConstraint) {
        let (term, z3) = constraint.into_parts();
        self.add_constraint_raw_inner(z3, Origin::Term(term), false);
    }

    /// Record an *already-logged* constraint in the residual log.
    ///
    /// [`Self::add_constraint_lowered`] writes both logs; this is the residual
    /// half on its own, for a caller that put the constraint in the
    /// assertion log by some other route — today only
    /// [`Self::add_constraint_tracked_lowered`], whose entry carries a
    /// tracker and so cannot reuse `add_constraint_lowered`.
    ///
    /// Same "exactly one log" contract as [`Self::add_constraint_lowered_assumed`],
    /// from the other side: call this only for a constraint with NO RustBV
    /// form (nothing to reconstruct it from in the assumed log). See that
    /// method's doc comment for why double-listing corrupts
    /// `unsat_core_assumed`.
    ///
    /// The caller must already have pushed the same `Bool` into
    /// `z3_assertions`, keeping the residual log a subset of it — the
    /// invariant `to_snapshot`'s `debug_assert!` pair checks.
    pub fn push_residual_assertion(&self, constraint: LoweredConstraint) {
        let (term, z3) = constraint.into_parts();
        self.local_constraints
            .lock()
            .push_residual(z3, Origin::Term(term));
    }

    /// How many entries the residual log holds, shared prefix and local
    /// additions together.
    pub fn residual_len(&self) -> usize {
        // overflow-ok: two live `Vec::len`s; their sum cannot near `usize::MAX`.
        self.non_bv_assertions_shared.lock().len()
            + self.local_constraints.lock().non_bv_assertions.len()
    }

    fn add_constraint_raw_inner(
        &self,
        constraint: z3::ast::Bool,
        origin: Origin,
        log_residual: bool,
    ) {
        self.close_branch_scope_before_permanent_add();
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
            let dup = self.seed_and_check_z3_dedup(&mut local, &constraint, origin.clone());
            if !dup && log_residual {
                // angr-t3l5o Phase 1: residual sink #1. This no-RustBV
                // constraint (Python claripy-sync fallback / a constraint
                // log copied from another state) has no `assumed` entry to
                // reconstruct it from,
                // so record it in the residual log to round-trip via
                // `residual_terms`. Only on the non-dup path — a dup is
                // already counted on its first add. `seed_and_check_z3_dedup`
                // pushed the same Bool to `z3_assertions` just above, so the
                // residual log stays a subset of `z3_assertions`.
                local.push_residual(constraint.clone(), origin);
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
        self.install_constraint(&constraint);
    }

    /// Install a constraint the caller has already put in the assertion log,
    /// without closing the branch scope.
    fn add_constraint_unscoped(&self, constraint: z3::ast::Bool) {
        self.install_constraint(&constraint);
    }

    /// Batched [`Self::add_constraint_lowered_assumed`]: asserts N constraints
    /// under one `local_constraints` lock, one solver/lineage transition, and
    /// one model invalidation pass. Each `(constraint, bv, is_true)` tuple is
    /// the lowered constraint the solver takes and the `assumed` pair that
    /// stands for it in the export log.
    ///
    /// Entries are deduped exactly as [`Self::add_constraint_lowered`] dedupes
    /// its single constraint — via `seed_and_check_z3_dedup` (private, so no
    /// intra-doc link from this public method), which
    /// also covers duplicates *within* one batch since each miss inserts its
    /// ptr before the next entry is checked. A dup contributes nothing but
    /// its `assumed` pair: no `z3_assertions` push, no solver assert, no
    /// `ScopeFrame`, no `constraint_count` bump. The three
    /// `ADD_CONSTRAINT_RAW_*` counters are bumped in bulk so batch traffic
    /// shows up in `get_solver_stats` alongside the single-shot path.
    ///
    /// Dispatches on `self.lineage` (angr-v5a5 slice 4c.2c; the batch form
    /// of `install_constraint`):
    ///
    /// - **None** (today's only production path): logs all N constraints;
    ///   the context's solver takes them at the next query.
    /// - **Some** (shared-lineage): mints N fresh
    ///   [`ScopeFrame`](super::lineage::ScopeFrame)s carrying clones of the
    ///   constraints, appends them all to `self.scope_path` under one
    ///   `scope_path.lock()` acquisition, snapshots the new path, drops the
    ///   `scope_path` lock, and calls
    ///   [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
    ///   once. `switch_to`'s tail walks the divergent suffix (here, the N
    ///   new frames) and asserts each one under the shared solver — N
    ///   ref-bumps but only a single round-trip through the lineage Mutex.
    ///   Cheaper than N separate `add_constraint_lowered` calls would be
    ///   (each of those is one lock + one switch_to call).
    ///
    /// Why not route through `with_z3_solver` in the Some branch: see
    /// `install_constraint`.
    pub fn add_constraints_raw_batch(&self, entries: Vec<(LoweredConstraint, RustBV, bool)>) {
        if entries.is_empty() {
            return;
        }
        self.close_branch_scope_before_permanent_add();
        // Each entry's assertion-log origin is the term it was lowered from,
        // as on the single-shot path, so the log rebuilds the Z3 form the
        // caller asserted.
        let mut constraints = Vec::with_capacity(entries.len());
        let mut origins = Vec::with_capacity(entries.len());
        let mut assumed = Vec::with_capacity(entries.len());
        for (constraint, bv, is_true) in entries {
            let (term, z3) = constraint.into_parts();
            constraints.push(z3);
            origins.push(Origin::Term(term));
            assumed.push((bv, is_true));
        }
        // The ptr-keyed dedup on the batch path (angr-sfp9, angr-5mnx3.43):
        // without it this API would push/assert/count every entry unconditionally, so a
        // constraint already asserted (a duplicate inside this very batch,
        // or one added by an earlier `add_constraint_lowered`) grew
        // `z3_assertions`, inflated `constraint_count`, and — under shared
        // lineage — minted a spurious `ScopeFrame`. Filter first, under the
        // same single `local_constraints` lock, so the solver/lineage
        // dispatch below only ever sees genuinely new constraints.
        let total = constraints.len();
        ADD_CONSTRAINT_RAW_TOTAL_COUNT.fetch_add(total as u64, Ordering::Relaxed);
        ADD_CONSTRAINT_RAW_DEDUP_SCANNED_COUNT.fetch_add(total as u64, Ordering::Relaxed);
        // Single lock on local_constraints for both vectors.
        let constraints: Vec<z3::ast::Bool> = {
            let mut local = self.local_constraints.lock();
            // `assumed` is grown for every entry, dup or not: it is the
            // Python-visible constraint list where duplicates are meaningful.
            // Same contract as the assume_* dedup — see the
            // `Z3_ASSUME_DEDUP_*` group comment in `stats.rs`.
            local.assumed.extend(assumed);
            // `seed_and_check_z3_dedup` pushes each non-dup onto
            // `local.z3_assertions` itself, so no separate bulk push.
            constraints
                .into_iter()
                .zip(origins)
                .filter(|(c, origin)| !self.seed_and_check_z3_dedup(&mut local, c, origin.clone()))
                .map(|(c, _)| c)
                .collect()
        };
        // overflow-ok: `constraints` was just rebound by a `filter` over the
        // `total`-element vec, so its length can only have shrunk.
        let dedup_hits = total - constraints.len();
        if dedup_hits > 0 {
            ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.fetch_add(dedup_hits as u64, Ordering::Relaxed);
        }
        if constraints.is_empty() {
            // Every entry was already asserted on the solver and tracked in
            // `z3_assertions` — no solver work, no scope frames, no cache
            // invalidation, no `constraint_count` bump (each was counted on
            // its initial add). Mirrors `add_constraint_raw_inner`'s
            // `was_dup` early return.
            return;
        }
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                // Logged above; the solver takes them at the next query.
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

    /// Batched model invalidation: drop each cached witness (`model_cache`,
    /// `alt_model_cache`) that fails any constraint in `constraints`.
    fn invalidate_model_if_inconsistent_batch(&self, constraints: &[z3::ast::Bool]) {
        self.retain_models_satisfying(|model| {
            constraints
                .iter()
                .all(|constraint| model_satisfies(model, constraint))
        });
    }

    /// Keep each cached witness that `satisfies` the constraint just added and
    /// drop the rest; a surviving `alt_model_cache` is promoted when the
    /// primary fails, so the taken side of a branch keeps the witness
    /// `check_branch_feasibility` found for it.
    fn retain_models_satisfying(&self, satisfies: impl Fn(&z3::Model) -> bool) {
        let mut cache = self.model_cache.borrow_mut();
        let mut alt = self.alt_model_cache.borrow_mut();
        if cache.is_none() && alt.is_none() {
            return;
        }
        // A token with no model on this thread is dropped with the stale ones.
        let keep = |slot: Option<ModelRef>| {
            slot.filter(|token| token.get().is_some_and(|model| satisfies(&model)))
        };
        let primary = keep(cache.take());
        let secondary = keep(alt.take());
        match primary {
            Some(m) => {
                *cache = Some(m);
                *alt = secondary;
            }
            None => *cache = secondary,
        }
    }

    /// Add a constraint with tracking for unsat_core extraction.
    /// Use this only when unsat_core analysis is needed.
    ///
    /// Returns the tracker index assigned to this constraint, which is also
    /// the index that will appear in [`Self::unsat_core`] output if the
    /// constraint participates in the unsat core.
    ///
    /// The log entry keeps the clarirs term the caller lowered. Dispatches on
    /// `self.lineage` (angr-v5a5 slice 4c.2b):
    ///
    /// - **None** (today's only production path): logs the constraint with
    ///   its tracker, and the context's solver asserts it with
    ///   `assert_and_track` at the next query. Full unsat-core fidelity.
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
    ///   See `install_constraint` for why the Some branch can't
    ///   route through `with_z3_solver`'s closure form.
    pub fn add_constraint_tracked_lowered(&self, constraint: &LoweredConstraint) -> usize {
        self.add_constraint_tracked_inner(
            constraint.z3().clone(),
            Origin::Term(constraint.term().clone()),
        )
    }

    fn add_constraint_tracked_inner(&self, constraint: z3::ast::Bool, origin: Origin) -> usize {
        self.close_branch_scope_before_permanent_add();
        let idx = self.constraint_count.load(Ordering::SeqCst);
        let track_name = format!("__track_{idx}");
        let track_bool = z3::ast::Bool::new_const(track_name.as_str());

        // angr-sqfj8.101: every other adder (`add_constraint_raw_inner`,
        // `add_constraints_raw_batch`) logs into `local.z3_assertions`, so
        // `fork()` and the context's solver (`SymContext::sync_solver`) see
        // the constraint. This path used to skip that entirely. Push
        // unconditionally (no dedup-skip) rather than routing through
        // `seed_and_check_z3_dedup`/`contains_or_insert_ptr`: those
        // conditionally omit the push on a dup, which would leave
        // `tracker_idx` out of sync with `unsat_core()`'s by-position match
        // against `constraint_trackers` (pinned by
        // `tests/engines/rust/test_solver_ops.py`'s `tracker_idx == call
        // order` expectation) — every call here mints a tracker and asserts
        // regardless of dedup, so the bookkeeping must do the same.
        self.install_constraint(&constraint);

        // The log entry carries the tracker, so the solver asserts this
        // copy under it whenever it is built (`SymContext::sync_solver`),
        // and a fork's solver does too (angr-vmi2l.91).
        let tracker_idx = {
            let mut trackers = self.constraint_trackers.lock();
            let i = trackers.len();
            trackers.push((track_bool.clone(), constraint.clone()));
            i
        };
        {
            use z3::ast::Ast;
            let mut local = self.local_constraints.lock();
            local.log_tracked(constraint.clone(), origin, track_bool);
            if local.dedup_set_seeded {
                local
                    .dedup_set
                    .insert(constraint.get_z3_ast().as_ptr() as usize);
            }
        }
        tracker_idx
    }

    /// Drop each cached witness that no longer satisfies the new constraint.
    /// Models that satisfy a superset of constraints stay valid; this lets
    /// check_branch_feasibility reuse a model across consecutive
    /// assume_true/assume_false calls in deferred-fork mode.
    fn invalidate_model_if_inconsistent(&self, constraint: &z3::ast::Bool) {
        self.retain_models_satisfying(|model| model_satisfies(model, constraint));
    }

    /// Add a constraint that the bitvector equals a specific value.
    pub fn add_bv_constraint(&self, bv: &RustBV, value: u128) {
        self.close_branch_scope_before_permanent_add();
        // Fast path: if bv is already concrete, the constraint is either
        // trivially true (skip) or trivially false (makes UNSAT).
        if let Some(v) = bv.as_u128()
            && v == value
        {
            return; // Tautology — skip Z3
        }
        // Falls through to add False constraint (UNSAT). A term with no Z3
        // form cannot reach the solver: the constraint is dropped with a warn,
        // as `assume` drops one (angr-mx00p.116.2).
        let origin = Origin::BvEq(bv.clone(), value);
        let Some(constraint) = super::lowered_or_warn(origin.lower(), "add_bv_constraint") else {
            return;
        };
        // angr-t3l5o Phase 1: residual sink #2 (address concretization).
        // Push to BOTH logs: `z3_assertions`, the only place a solver reads
        // constraints from, so this context's solver and every fork's hold
        // it, and `non_bv_assertions` so it round-trips via `residual_terms`
        // (it has no `assumed` entry).
        {
            let mut local = self.local_constraints.lock();
            local.push_assertion(constraint.clone(), origin.clone());
            local.push_residual(constraint.clone(), origin);
        }
        self.add_constraint_unscoped(constraint);
    }

    /// Add a constraint that the bitvector is true (non-zero for 1-bit).
    pub fn assume_true(&self, cond: &RustBV) {
        self.close_branch_scope_before_permanent_add();
        self.assume_unscoped(cond, true);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    pub fn assume_false(&self, cond: &RustBV) {
        self.close_branch_scope_before_permanent_add();
        self.assume_unscoped(cond, false);
    }

    /// Shared body of `assume_true` / `assume_false` (angr-12jjk.11).
    ///
    /// The two entry points differ only in polarity: which concrete value is
    /// the tautology, the `bool` recorded in the `assumed` export log, and
    /// whether the Z3 bool is negated. Everything else — the concrete/symbolic
    /// counter split, the sampled simplify-skip probe, the dedup contract and
    /// the `install_constraint` handoff — is identical, so it lives here once
    /// rather than in two hand-synced copies.
    pub(super) fn assume_unscoped(&self, cond: &RustBV, want_true: bool) {
        debug_assert_eq!(cond.width(), 1);
        // Fast path: the concrete value matching `want_true` is a tautology —
        // skip Z3 entirely (still record it for Python export).
        if let Some(v) = cond.as_u128()
            && (v != 0) == want_true
        {
            self.local_constraints
                .lock()
                .assumed
                .push((cond.clone(), want_true));
            Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
            return; // Asserting a tautology is a no-op
        }
        // Concrete but opposite polarity: asserting False makes the solver
        // UNSAT — still add it.
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Use to_z3_bool() to produce a native Z3 Bool for comparison ops,
        // avoiding the ITE(cmp, BV(1,1), BV(0,1)).eq(BV(1,1)) round-trip; the
        // false polarity negates that bool directly.
        //
        // A cond with no Z3 form (an op `z3_lower` refuses) cannot reach
        // the solver. It stays in the `assumed` export log, but the solver
        // goes without it, with a warn: the state is under-constrained,
        // never pruned, the side a Z3 Unknown errs on
        // (`invariant-z3-unknown-not-unsat`, angr-mx00p.116.2).
        let Some(bool_ast) = super::lowered_or_warn(cond.to_z3_bool(), "assume") else {
            self.local_constraints
                .lock()
                .assumed
                .push((cond.clone(), want_true));
            return;
        };
        let constraint = if want_true { bool_ast } else { bool_ast.not() };
        // angr-1joc measurement: sampled simplify-skip check.
        sample_simplify_skip(&constraint);
        // angr-mwbp: piggy-back dedup on the side-table when it's already
        // seeded by an earlier `add_constraint_lowered` call in this context.
        // Triggering the seed from here would walk shared+local on every
        // assume_true/false call in fresh contexts — a real bench regression
        // on heavily-branched workloads where add_constraint_lowered isn't
        // hot (flareon2015_2 timed out at 30s under unconditional seeding).
        // When seeded, `assumed` still grows unconditionally to preserve
        // Python-visible duplicate constraints (claripy `solver.add(c)`
        // semantics) — dedup only short-circuits the redundant
        // `z3_assertions.push` + `install_constraint(c)` round-trip.
        let was_dup = {
            let mut local = self.local_constraints.lock();
            local.assumed.push((cond.clone(), want_true));
            let origin = Origin::Assume(cond.clone(), want_true);
            self.check_z3_dedup_if_seeded(&mut local, &constraint, origin)
        };
        if was_dup {
            Z3_ASSUME_DEDUP_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.add_constraint_unscoped(constraint);
    }
}

test_submod!("constraint_ops_tests.rs" => constraint_ops_tests);

/// Whether `model` shows `constraint` true.
///
/// SILENT(cat-a): an eval/extraction failure means the model cannot be shown
/// to satisfy the constraint, which is exactly the `false` case — the model is
/// dropped and the next query re-solves. Never a wrong answer, only a lost
/// cache hit.
fn model_satisfies(model: &z3::Model, constraint: &z3::ast::Bool) -> bool {
    model
        .eval(constraint, true)
        .and_then(|b| b.as_bool())
        .unwrap_or(false)
}
