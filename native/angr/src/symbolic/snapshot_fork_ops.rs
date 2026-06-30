//! Fork / merge / snapshot lifecycle `&self` methods for [`SymContext`].
//!
//! Slice 11 of the `symbolic/context.rs` split (angr-a2br.2.9). These are the
//! state-cloning and serialization entry points: snapshot round-trip
//! (`to_snapshot` / `restore_from_snapshot` plus the Z3-only
//! `dump_solver_smtlib2` helper), structural cloning (`fork`, and the
//! branch-conditioned `fork_true` / `fork_false`), and state combination
//! (`merge`).
//!
//! Like `transaction_ops` / `solving_ops`, this slice carries BOTH the
//! `#[cfg(feature = "vex-engine-z3")]` implementations and their non-Z3 mock
//! counterparts, so the module decl in `mod.rs` is NOT feature-gated.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`.
//! `fork`/`merge` reconstruct a full `SymContext { .. }`, so every struct field
//! they touch must be reachable from this sibling module. All but two were
//! already promoted by earlier slices; this slice promotes the last holdouts
//! (`symbol_table`, `assumed_constraints_shared`) and the private `PushStack`
//! type alias to `pub(super)` (== `pub(in crate::symbolic)`). See bd memory
//! `a2br2-context-split-impl-block-plan` for the slice plan.

use super::context::{LocalConstraints, PushStack, freeze_into_shared};
use super::{RustBV, SymContext, SymContextSnapshot};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::Mutex;

#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;
#[cfg(feature = "vex-engine-z3")]
use std::cell::{Cell, RefCell};
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::{AtomicBool, AtomicU32};

impl SymContext {
    /// Build a serializable snapshot of this context's path-constraint state
    /// (angr-x04s.1.2).
    ///
    /// Captures `assumed_constraints` — the canonical record from which all
    /// per-context Z3 cache state (solver, sat_cache, model_cache, lineage
    /// scope_path, push stacks) re-derives. Other [`SymContext`] fields are
    /// either runtime-only caches (rebuild on first query) or runtime
    /// counters (`next_id`, `constraint_count`) that the loader can leave
    /// at defaults — the IDs carried inside each `RustBV` are already
    /// globally unique and survive the round-trip.
    ///
    /// See `snapshot-serialization-design` for the broader plan and
    /// `rustsimstate-field-buckets` for which surrounding state buckets
    /// are covered (RegisterFile / MemoryPage) versus deferred
    /// (Python-side `Py<PyAny>` overlays).
    pub fn to_snapshot(&self) -> SymContextSnapshot {
        let assumed_constraints = self.get_assumed_constraints();
        // angr-82g6: also dump the full Z3 solver state in SMT-LIB2 so
        // every assertion — including constraints added via
        // `add_constraint_raw` that have no [`RustBV`] form in
        // `assumed_constraints` (the Python claripy-sync fallback path
        // in `_add_constraints_to_state` and the cross-process pointer-
        // import path in `_import_z3_constraint_ptrs`) — survives the
        // round-trip.
        //
        // Restore replays this dump through `add_constraint_raw` and
        // separately writes the `assumed_constraints` log via
        // `assumed_constraints_push` (no solver re-assert), so the two
        // captures are independent and don't need ptr-level dedup. This
        // sidesteps the pitfall that Python-claripy ASTs and our
        // `claripy_to_rustbv`-rebuilt ASTs are structurally different
        // (hash-cons to different pointers) — both forms land back where
        // they came from.
        #[cfg(feature = "vex-engine-z3")]
        let solver_smtlib2 = {
            // angr-t3l5o Phase 0b: time the SMT-LIB2 text emit and record the
            // residual (no-RustBV) assertion count, both gated on the env flag.
            if crate::migrate_phase_timers::count_armed() {
                // Approximate residual count as total Z3 assertions minus the
                // reconstructible assume class (see field doc on
                // MIGRATE_RAW_CONSTRAINT_COUNT). Computed before the emit so the
                // emit timer measures the dump alone. `z3_assertion_count`
                // materializes the lazy solver — acceptable in measurement mode.
                let total = self.z3_assertion_count() as u64;
                let assumed = assumed_constraints.len() as u64;
                crate::migrate_phase_timers::add_raw_constraint_count(
                    total.saturating_sub(assumed),
                );
            }
            crate::migrate_phase_timers::time_phase(
                &crate::migrate_phase_timers::MIGRATE_SMTLIB2_EMIT_NS,
                || self.dump_solver_smtlib2(),
            )
        };
        #[cfg(not(feature = "vex-engine-z3"))]
        let solver_smtlib2 = String::new();
        SymContextSnapshot {
            assumed_constraints,
            solver_smtlib2,
        }
    }

    /// Helper for [`Self::to_snapshot`]: dump the union of
    /// `z3_assertions_shared` and the local Z3 assertion vector into a
    /// temp [`z3::Solver`] and emit SMT-LIB2.
    ///
    /// Returns the empty string when no Z3 assertions exist, so the
    /// snapshot envelope stays minimal in the no-constraints case.
    #[cfg(feature = "vex-engine-z3")]
    fn dump_solver_smtlib2(&self) -> String {
        let temp = z3::Solver::new();
        let mut any = false;
        let shared = Arc::clone(&self.z3_assertions_shared.lock());
        for c in shared.iter() {
            temp.assert(c);
            any = true;
        }
        let local = self.local_constraints.lock();
        for c in local.z3_assertions.iter() {
            temp.assert(c);
            any = true;
        }
        if any {
            format!("{}", temp)
        } else {
            String::new()
        }
    }

    /// Restore the path-constraint state captured by `to_snapshot` into
    /// a fresh context. The caller is responsible for constructing the
    /// `SymContext` (typically via [`SymContext::new`] or `new_mock`); this
    /// method replays the captured constraints through `assume_true` /
    /// `assume_false` so the underlying Z3 solver, fast-path caches, and
    /// `assumed_constraints` log all rebuild consistently.
    ///
    /// Replay matches what `fork()` does after deserialization: each
    /// captured `(constraint, is_true)` is re-asserted via the public
    /// assume APIs. The shared/local split is collapsed — the restored
    /// context starts with all entries on the local side and can be
    /// re-shared by a subsequent `fork()`.
    ///
    /// angr-82g6: the Z3 solver state is rebuilt from `solver_smtlib2`
    /// (every assertion replayed through [`Self::add_constraint_raw`])
    /// so constraints added via the raw path — which has no [`RustBV`]
    /// form in `assumed_constraints` — round-trip faithfully. The
    /// `assumed_constraints` log is then populated directly via
    /// [`Self::assumed_constraints_push`] (no second solver assert) so
    /// the BV-export log matches the original without double-counting
    /// against `constraint_count`. Snapshots written before this field
    /// existed (`solver_smtlib2` empty by `#[serde(default)]`) fall
    /// through to the original `assume_*` replay path.
    ///
    /// Must run inside an active Z3 thread-local context when the
    /// `vex-engine-z3` feature is enabled (the same rule as `RustBV`
    /// deserialization — see `snapshot-rustbv-shadow-type-pattern`).
    pub fn restore_from_snapshot(&self, snap: &SymContextSnapshot) {
        #[cfg(feature = "vex-engine-z3")]
        {
            if !snap.solver_smtlib2.is_empty() {
                use z3::ast::Ast;
                // angr-t3l5o Phase 0b: time the SMT-LIB2 re-parse + re-assert
                // loop (the reattach-side text round-trip). The assumed_constraints
                // BV-log replay below is intentionally NOT counted as parse.
                crate::migrate_phase_timers::time_phase(
                    &crate::migrate_phase_timers::MIGRATE_SMTLIB2_PARSE_NS,
                    || {
                        let tmp = z3::Solver::new();
                        tmp.from_string(snap.solver_smtlib2.as_str());
                        let z3_ctx = z3::Context::thread_local();
                        for assertion in tmp.get_assertions() {
                            let ptr = assertion.get_z3_ast().as_ptr() as usize;
                            // SAFETY: `ptr` came from a Bool returned by
                            // `get_assertions()` parsed into the thread-local
                            // Z3 context (the same one `add_constraint_raw`
                            // will use). `from_borrowed_raw` takes its own ref
                            // via `Z3_inc_ref`, independent of the temp
                            // solver's reference.
                            if let Some(z3_ast) =
                                unsafe { super::Z3AstPtr::from_borrowed_raw(&z3_ctx, ptr) }
                            {
                                self.add_constraint_raw(z3_ast);
                            }
                        }
                    },
                );
                for (cond, is_true) in &snap.assumed_constraints {
                    self.assumed_constraints_push(cond.clone(), *is_true);
                }
                return;
            }
        }
        // Legacy / non-Z3 path: replay assumed_constraints via the
        // public assume APIs. This is the only path when
        // `solver_smtlib2` is empty (older snapshots, or vex-engine-z3
        // disabled at build time).
        for (cond, is_true) in &snap.assumed_constraints {
            if *is_true {
                self.assume_true(cond);
            } else {
                self.assume_false(cond);
            }
        }
    }

    /// angr-t3l5o Phase 0a bench hook: expose the private
    /// [`Self::dump_solver_smtlib2`] so the per-phase micro-bench in
    /// `benches/vex_engine.rs` can time the SMT-LIB2 *emit* in isolation
    /// without copy-pasting its body.
    #[doc(hidden)]
    #[cfg(feature = "vex-engine-z3")]
    pub fn bench_dump_solver_smtlib2(&self) -> String {
        self.dump_solver_smtlib2()
    }

    /// angr-t3l5o Phase 0a bench hook: re-parse an SMT-LIB2 text dump into a
    /// fresh context's solver via the same `from_string` + `get_assertions` +
    /// `add_constraint_raw` loop that [`Self::restore_from_snapshot`] uses,
    /// isolating the SMT-LIB2 *parse* cost. Returns the number of re-asserted
    /// constraints. Must run inside an active thread-local Z3 context.
    #[doc(hidden)]
    #[cfg(feature = "vex-engine-z3")]
    pub fn bench_parse_smtlib2(smtlib2: &str) -> usize {
        use z3::ast::Ast;
        let fresh = SymContext::new();
        let tmp = z3::Solver::new();
        tmp.from_string(smtlib2);
        let z3_ctx = z3::Context::thread_local();
        let mut n = 0usize;
        for assertion in tmp.get_assertions() {
            let ptr = assertion.get_z3_ast().as_ptr() as usize;
            // SAFETY: same contract as `restore_from_snapshot` — `ptr` is a
            // Bool from `get_assertions()` in the active thread-local context.
            if let Some(z3_ast) = unsafe { super::Z3AstPtr::from_borrowed_raw(&z3_ctx, ptr) } {
                fresh.add_constraint_raw(z3_ast);
                n += 1;
            }
        }
        n
    }

    /// angr-t3l5o Phase 0a bench hook: assert a width-1 `RustBV` predicate via
    /// the *raw* path (`add_constraint_raw`) so it lands in the residual
    /// (no-`RustBV`) class — i.e. it is NOT recorded in `assumed_constraints`
    /// and so survives the round-trip only through the SMT-LIB2 text dump.
    /// Lets the micro-bench populate the `raw_fraction` axis.
    #[doc(hidden)]
    #[cfg(feature = "vex-engine-z3")]
    pub fn bench_add_constraint_raw_from_bool(&self, pred: &RustBV) {
        use z3::ast::Ast;
        let bool_ast = pred.to_z3_bool();
        let ctx = z3::Context::thread_local();
        let ptr = bool_ast.get_z3_ast().as_ptr() as usize;
        // SAFETY: `bool_ast` keeps the AST alive across the inc_ref; the
        // resulting Z3AstPtr holds its own ref (mirrors the `raw_entry` test
        // helper in context_tests/constraints.rs).
        if let Some(z3_ast) = unsafe { super::Z3AstPtr::from_borrowed_raw(&ctx, ptr) } {
            self.add_constraint_raw(z3_ast);
        }
    }

    /// Cross-context twin of [`Self::fork`] (angr-ahypj): build a fresh
    /// `SymContext` whose path-constraint state lives in `target_ctx`, using
    /// the [`z3::Translate`] (`Z3_translate`) primitive instead of the
    /// SMT-LIB2 string round-trip that [`Self::restore_from_snapshot`] uses.
    ///
    /// Mirrors `restore_from_snapshot`'s two-part replay so the result is
    /// byte-for-byte equivalent:
    ///
    /// 1. **Solver state** — every Z3 assertion (`z3_assertions_shared` +
    ///    local) is `Z3_translate`d into `target_ctx` and re-asserted via
    ///    [`Self::add_constraint`]. This covers constraints with no [`RustBV`]
    ///    form (the raw Python-sync / cross-process import paths), exactly as
    ///    the SMT-LIB2 dump does in the snapshot path.
    /// 2. **Assumed-constraint BV log** — every `(bv, is_true)` is deep-
    ///    translated via [`RustBV::translate_into`] and pushed through
    ///    [`Self::assumed_constraints_push`] (no second solver assert), so the
    ///    BV-export log round-trips without double-counting `constraint_count`.
    ///
    /// # Preconditions / panics
    ///
    /// `target_ctx` must be the **active thread-local Z3 context** — the fresh
    /// `SymContext`'s lazy solver and `add_constraint` both build against
    /// `z3::Context::thread_local()`, so the caller must
    /// `set_thread_local(target_ctx)` first (the production Option-A model
    /// runs this on the target worker's thread). It must also be a *different*
    /// context from the source's, or the underlying `Z3_translate` panics
    /// (see [`RustBV::translate_into`]).
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_into(&self, target_ctx: &z3::Context) -> Self {
        use z3::Translate;
        use z3::ast::Ast;
        let new = SymContext::new();
        // (1) Full solver state: `Z3_translate` every assertion into the target
        // context, then route through `add_constraint_raw` (NOT `add_constraint`)
        // so the `z3_assertions` LOG is seeded — `add_constraint`/`install_constraint`
        // only asserts on the live solver and would leave the log empty, breaking
        // any subsequent re-translate (the A->B->A path). Mirrors the replay loop
        // in `restore_from_snapshot`.
        let translate_one = |c: &z3::ast::Bool| {
            let translated = c.translate(target_ctx);
            let ptr = translated.get_z3_ast().as_ptr() as usize;
            // SAFETY: `ptr` denotes a Bool freshly `Z3_translate`d into
            // `target_ctx` (== the active thread-local, per the precondition);
            // `from_borrowed_raw` takes its own ref via `Z3_inc_ref`,
            // independent of `translated`'s ref.
            if let Some(z3_ast) = unsafe { super::Z3AstPtr::from_borrowed_raw(target_ctx, ptr) } {
                new.add_constraint_raw(z3_ast);
            }
        };
        let shared = Arc::clone(&self.z3_assertions_shared.lock());
        for c in shared.iter() {
            translate_one(c);
        }
        {
            let local = self.local_constraints.lock();
            for c in local.z3_assertions.iter() {
                translate_one(c);
            }
        }
        // (2) Assumed-constraint BV log: translate, push without re-asserting.
        for (bv, is_true) in self.get_assumed_constraints() {
            new.assumed_constraints_push(bv.translate_into(target_ctx), is_true);
        }
        new
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the context, creating a new context with all constraints preserved.
    ///
    /// The Z3 solver is NOT created eagerly — it starts as None and is
    /// materialized on first access (lazy). This avoids the O(n) assertion
    /// replay cost for forked states that are pruned/avoided/deadended
    /// without ever querying the solver.
    ///
    /// When local additions exist and we are not inside a push/pop transaction,
    /// fork "freezes self": the local additions are drained into self's shared
    /// Arc so subsequent forks of self with empty local become O(1) Arc::clone.
    /// When self.shared is uniquely owned, this avoids cloning every Bool
    /// (each Bool::clone is a Z3_inc_ref FFI call).
    ///
    /// **Cross-cutting invariants enforced here** (see module-level
    /// "Lineage + solver invariants" for the full set):
    ///
    /// - `fork-freeze-self-invariant`: freeze only fires when
    ///   `push_level == 0` (`in_transaction == false`). Inside a transaction,
    ///   draining local would leak rolled-back constraints into shared.
    /// - `invariant-v5a5-slice-1c-mint-semantics`: the three-gate check
    ///   (`use_shared_lineage_solver` opt-in, zero bare pushes, not
    ///   dismantled) is load-bearing. Each gate guards a different failure
    ///   mode — see inline comments below.
    /// - `invariant-bare-z3-push-depth`: gate (b) reads the **parent's**
    ///   counter, not the child's. The child inherits the parent's value
    ///   but `fork()` resets `push_level` to 0, so any future push
    ///   accounting on the child starts fresh.
    /// - `invariant-v5ht-dismantle-child-none`: when dismantled, child
    ///   gets `None` (NOT `Arc::clone(parent.lineage)`). Cloning would
    ///   give the child a stale base — see the inline comment for the
    ///   baby-re chr() repro this prevents.
    // Arc<Mutex<SharedLineageSolver>>: SharedLineageSolver wraps a z3::Solver (not Send/Sync),
    // but the Arc is load-bearing — every sibling SymContext minted from a common fork shares
    // the same Arc so cross-state lineage queries hit the same Z3 solver (see field doc on
    // `lineage`). Engine runs single-threaded under Python's GIL; Rc would force the fork
    // signature to surface non-Send, breaking the Arc<Mutex<...>> sharing contract.
    #[allow(clippy::arc_with_non_send_sync)]
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork(&self) -> Self {
        let in_transaction = self.push_level.load(Ordering::Relaxed) > 0;
        let (frozen_shared, frozen_assumed) = {
            let mut local = self.local_constraints.lock();
            let frozen_shared = freeze_into_shared(
                &self.z3_assertions_shared,
                &mut local.z3_assertions,
                in_transaction,
            );
            let frozen_assumed = freeze_into_shared(
                &self.assumed_constraints_shared,
                &mut local.assumed,
                in_transaction,
            );
            (frozen_shared, frozen_assumed)
        };
        let assumed_total_len = frozen_assumed.len();

        // angr-3ms1 step 1c: fork-time SharedLineageSolver materialization
        // gate. When BOTH (a) the parent opted in via
        // `use_shared_lineage_solver` AND (b) no bare Z3 pushes are
        // outstanding on the parent's per-context solver, mint a fresh
        // `SharedLineageSolver` for the child seeded with `frozen_shared`
        // as base assertions (scope 0, never popped). Otherwise, keep the
        // pre-1c behavior of Arc::cloning the parent's lineage Arc — which
        // is `None` by default in production today.
        //
        // The two-gate check is load-bearing. Condition (a) keeps the
        // BFS-thrash regression (defcon2016quals_baby-re ~10x;
        // `v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental`) out of
        // the default CI gate by holding minting OFF until a caller
        // explicitly opts in via the kwarg on `RustExplorationManager`.
        // Condition (b) protects the per-context solver's bare-push frames
        // from being clobbered by a sibling that takes over Z3 stack
        // ownership through the new lineage (the exact correctness bug
        // that `test_fork_inside_push_isolation` exposes in earlier
        // attempts; see `v5a5-bare-z3-push-depth-counter-design`).
        //
        // Seeding via `assert_base` puts the parent's frozen constraints
        // at scope 0 of the lineage's solver, so the child's first query
        // (which runs `switch_to(empty scope_path)`) sees the parent's
        // constraints without needing to also walk `frozen_shared`
        // separately. Subsequent constraints added on the child go
        // through the slice-4c migrated `add_constraint*` paths, which
        // mint per-state ScopeFrames on top of the lineage base.
        // angr-v5ht adds the third gate (c): if the runtime thrash
        // detector has dismantled lineage minting (the hot-cache hit
        // ratio dropped below threshold over a recent sampling window),
        // skip lineage minting AND drop the lineage Arc on the child.
        //
        // Note we cannot simply Arc::clone the parent's lineage into
        // the dismantled child: the parent's lineage's base assertions
        // were frozen at the moment the lineage was first minted (an
        // earlier ancestor), but the parent has since accumulated more
        // constraints (in its per-state scope_path). A child starting
        // with an empty scope_path would call switch_to(empty), popping
        // the shared solver back to its base — missing every constraint
        // the parent added post-mint. That stale constraint set leaks
        // unconstrained SAT solutions on the find state (see the
        // baby-re chr() repro from this iter). Setting child_lineage
        // to None puts the child on the per-context solver path, which
        // builds from frozen_shared = the parent's FULL constraint set,
        // preserving correctness at the cost of the lineage win for
        // this child. Existing in-flight lineage Arcs on ancestors keep
        // working — the simple dismantle variant only suppresses
        // minting on FUTURE forks.
        let dismantled = super::lineage::is_lineage_dismantled();
        let child_lineage = if self.use_shared_lineage_solver.load(Ordering::Relaxed)
            && self.bare_z3_push_depth.load(Ordering::Relaxed) == 0
            && !dismantled
        {
            // angr-a2br.1.1: debug-assert the gate is consistent at the
            // moment of mint. Tautological with the `if` condition but
            // documents the `invariant-v5a5-slice-1c-mint-semantics`
            // contract for readers tracing this branch.
            debug_assert!(
                self.bare_z3_push_depth.load(Ordering::Relaxed) == 0,
                "minting lineage with non-zero bare_z3_push_depth violates \
                 invariant-bare-z3-push-depth"
            );
            debug_assert!(
                !super::lineage::is_lineage_dismantled(),
                "minting lineage after dismantle violates \
                 invariant-v5ht-dismantle-child-none"
            );
            let lineage_solver = super::lineage::SharedLineageSolver::new(build_solver(
                self.timeout_ms.load(Ordering::SeqCst),
            ));
            for constraint in frozen_shared.iter() {
                lineage_solver.assert_base(constraint);
            }
            Some(Arc::new(Mutex::new(lineage_solver)))
        } else if dismantled {
            None
        } else {
            self.lineage.lock().as_ref().map(Arc::clone)
        };

        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(assumed_total_len),
            symbol_table: Arc::clone(&self.symbol_table),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            push_local_cache_lengths: Mutex::new(PushStack::new()),
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(frozen_assumed),
            z3_assertions_shared: Mutex::new(frozen_shared),
            local_constraints: Mutex::new(LocalConstraints::new()),
            solver: Mutex::new(None),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(Vec::new()),
            timeout_ms: AtomicU32::new(self.timeout_ms.load(Ordering::SeqCst)),
            lineage: Mutex::new(child_lineage),
            scope_path: Mutex::new(super::lineage::ScopePath::new()),
            scope_savepoints: Mutex::new(Vec::new()),
            // angr-3ms1 step 1a: child inherits parent's bare-push depth
            // so a fork inside a `push()` region keeps a consistent
            // accounting of outstanding bare pushes. The slice-1c
            // materialization gate reads the *parent's* value at the
            // moment of fork to decide whether to mint a lineage; copying
            // it into the child also keeps post-fork pop accounting
            // consistent if a child somehow inherits a pushed region
            // (today's fork semantics reset push_level, so in practice
            // the child observes 0 unless future code threads bare pushes
            // through fork).
            bare_z3_push_depth: AtomicUsize::new(self.bare_z3_push_depth.load(Ordering::Relaxed)),
            // angr-3ms1 step 1b: inherit the opt-in flag from parent so
            // a lineage opt-in on a seed state propagates to every
            // descendant without per-fork plumbing on the Python side.
            use_shared_lineage_solver: AtomicBool::new(
                self.use_shared_lineage_solver.load(Ordering::Relaxed),
            ),
        }
    }

    /// Run a closure against this context's Z3 solver, dispatching through
    /// the shared-lineage solver when one is attached (angr-v5a5 slice 3b).
    ///
    /// When `self.lineage` is `None` (today's only production path), this
    /// is a thin wrapper over [`solver()`](Self::solver): the closure sees
    /// the per-context lazy-materialized solver exactly as a direct
    /// `let solver = self.solver();` would. When `self.lineage` is `Some`
    /// (currently only `set_lineage_for_testing` installs one), the call
    /// routes through [`SharedLineageSolver::with_solver`], which switches
    /// the shared solver to this context's `scope_path` before invoking
    /// `f`.
    ///
    /// The dispatcher exists in this slice so subsequent slices can migrate
    /// individual solver call sites (`is_sat`, `eval`, `min`, `max`, …)
    /// one at a time without each migration also having to inline the
    /// dispatch logic.
    ///
    /// # Locking
    ///
    /// - **None path:** holds the per-context `solver` mutex via
    ///   [`solver()`](Self::solver) for the duration of `f`.
    /// - **Some path:** snapshots `scope_path` under its own mutex
    ///   (released before `f` runs), then holds the lineage mutex for
    ///   the duration of `f`. The lineage mutex serializes sibling
    ///   states sharing the same `SharedLineageSolver`.
    ///
    /// In either case, `f` must not re-enter into `with_z3_solver` or
    /// any SymContext method that would re-acquire the same lock — the
    /// existing direct `self.solver()` callers have the same invariant.
    ///
    /// Slice 3c migrated the first caller (`debug_solver_string`);
    /// slice 3d migrated `add_constraint`, the assume_true/assume_false/
    /// add_bv_constraint hot path; slice 3e migrated `add_constraint_raw`
    /// (the unsafe Python-side raw-Z3-AST bridge); slice 3f migrated
    /// `add_constraint_tracked_indexed` (the unsat-core-tracked variant);
    /// slice 3g migrated `is_sat` (first migration with a return value
    /// and an in-closure side effect — the post-check `get_model()` that
    /// populates `model_cache` runs inside the closure so it shares the
    /// solver lock with the `check()` call); slice 3h migrated `eval`
    /// (three Z3 operations under one lock — `check()`, `get_model()`,
    /// and `model.eval(ast, true)` — with `?`-propagation on the inner
    /// `Option<u128>` so a None model or extraction returns from the
    /// closure cleanly while still letting `sat_cache.set(Some(true))`
    /// have fired); slice 3i migrated `eval_wide` (same three Z3 ops as
    /// `eval` but returning `Option<Vec<u8>>` via `extract_bv_value_wide`;
    /// no `model_cache`/`sat_cache` writes since the original didn't have
    /// them); slice 3j migrated `check_branch_feasibility` (first
    /// migration with balanced `push`/`pop` pairs inside the closure and
    /// a three-armed match on the model-cache prediction — the predicted
    /// `Option<bool>` is computed inside the closure so the `model_cache`
    /// borrow and the solver lock are acquired in the same order as the
    /// pre-slice code, and the early `return (false, true)` in the None
    /// arm returns from the closure cleanly since the closure return
    /// type matches the function return type); slice 3k batch-migrated
    /// the four remaining flat (no-scope-stack) callers in one commit:
    /// `add_constraints_raw_batch` (assert N constraints under one lock),
    /// `unsat_core` (read `solver.get_unsat_core()` and stringify before
    /// taking the `constraint_trackers` lock — keeps the two locks from
    /// nesting), `get_all_constraints_str`, and `z3_assertion_count`
    /// (one-line reads of `solver.get_assertions()`). Bundled into one
    /// commit because each migration is the same one-line wrap pattern
    /// as slice 3c/d/e and individually noise-level. With 3k the entire
    /// "no scope-stack" subset of direct-solver callers is migrated —
    /// every remaining `self.solver()` call site manipulates Z3's scope
    /// stack across multiple operations. Slice 4a.1 begins the scope-
    /// stack-caller migration with `eval_upto`: the outer push/pop is
    /// balanced inside the closure (same pattern slice 3j proved with
    /// `check_branch_feasibility`), so the Z3 scope stack returns to
    /// its pre-closure depth before `f` returns. Slice 4a.2 extends the
    /// same wrap to `eval_upto_wide` (the byte-array sibling of
    /// `eval_upto`). Slice 4a.3 extends it to `min`: the outer push/pop
    /// brackets a binary-search loop with nested per-iteration push/
    /// check/pop pairs (and, in the signed case, an additional
    /// has_negative pre-check that also push/pops). All push/pop pairs
    /// remain balanced when the closure returns. Slice 4a.4 extends
    /// the same wrap to `max` (the dual of `min`: bvsge/bvuge binary
    /// search with a has_non_negative pre-check, same nested-push/pop
    /// shape). Slice 4a.5 extends the same wrap to `range_seeded`:
    /// outer push/pop brackets two binary-search loops (seeded min in
    /// [0, hi_seed] and seeded max in [lo_seed, max_val]) each with
    /// nested per-iteration push/check/pop pairs. All push/pop pairs
    /// remain balanced when the closure returns. Slice 4a.6 extends
    /// the same wrap to `solution`: a single push/assert/check/pop
    /// inside the closure (the simplest of the slice-4a migrations).
    /// With 4a.6 the balanced-in-one-call subset of scope-stack
    /// callers is complete; the remaining spans-multiple-calls
    /// subset (push/pop, transaction_*) is queued for slice 4b and
    /// likely needs a separate scope_path API to migrate. Slice 4b.1
    /// lands the scope-savepoint infrastructure:
    /// `scope_savepoint_push` and
    /// `scope_savepoint_pop` dispatch on
    /// lineage (None → bare per-context Z3 push/pop; Some → record/
    /// restore `scope_path.len()` on the new `scope_savepoints`
    /// stack). No public-API callers migrated yet — those come in
    /// slice 4b.2 (push) and 4b.3 (pop). Slice 4c.2b migrates
    /// `add_constraint_tracked_indexed` off the `with_z3_solver`
    /// dispatcher onto its own lineage-aware inline match — same shape
    /// as 4c.1/4c.2. The None branch keeps full unsat-core fidelity
    /// (`assert_and_track` on the per-context solver); the Some branch
    /// mints a `ScopeFrame` carrying the bare constraint and routes
    /// through `switch_to`, intentionally deferring unsat-core fidelity
    /// (`switch_to` uses plain `assert`, so the tracker is not
    /// re-registered on lineage reloads). Slice 4c.2c migrates
    /// `add_constraints_raw_batch` off the `with_z3_solver` dispatcher
    /// onto its own lineage-aware inline match — same shape as 4c.1/
    /// 4c.2/4c.2b but with N constraints per call. The None branch
    /// holds the per-context solver guard once and asserts all N
    /// constraints inside the guard. The Some branch mints N
    /// `ScopeFrame`s under one `scope_path.lock()` acquisition, then
    /// makes a single `switch_to` call whose tail walks the divergent
    /// suffix and asserts each new frame on the shared solver.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn with_z3_solver<R>(&self, f: impl FnOnce(&z3::Solver) -> R) -> R {
        let lineage = self.lineage.lock().as_ref().map(Arc::clone);
        match lineage {
            None => {
                let solver = self.solver();
                f(&solver)
            }
            Some(lin) => {
                let path = self.scope_path.lock().clone();
                let mut guard = lin.lock();
                guard.with_solver(&path, f)
            }
        }
    }

    /// Install a lineage Arc on this context (test-only, angr-v5a5 spike).
    ///
    /// Lets unit tests verify the fork-propagation invariant — that a
    /// child's `lineage_arc()` returns the same Arc as the parent's —
    /// without yet wiring up the production lineage-creation path in
    /// [`fork()`](Self::fork). Removed when integration takes ownership
    /// of lineage creation.
    #[cfg(all(test, feature = "vex-engine-z3"))]
    pub(crate) fn set_lineage_for_testing(
        &self,
        lin: Arc<Mutex<super::lineage::SharedLineageSolver>>,
    ) {
        *self.lineage.lock() = Some(lin);
    }

    /// Append a `ScopeFrame` to this context's `scope_path` (test-only,
    /// angr-v5a5 slice 4b).
    ///
    /// Lets unit tests exercise the truncation behavior of
    /// `scope_savepoint_pop` before the
    /// production path that mints frames lands in slice 4c.
    #[cfg(all(test, feature = "vex-engine-z3"))]
    pub(crate) fn push_scope_frame_for_testing(&self, frame: super::lineage::ScopeFrame) {
        self.scope_path.lock().push(frame);
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork(&self) -> Self {
        let in_transaction = self.push_level.load(Ordering::Relaxed) > 0;
        let frozen_assumed = {
            let mut local = self.local_constraints.lock();
            freeze_into_shared(
                &self.assumed_constraints_shared,
                &mut local.assumed,
                in_transaction,
            )
        };

        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::clone(&self.symbol_table),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(frozen_assumed),
            local_constraints: Mutex::new(LocalConstraints::new()),
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

    /// Merge multiple solver contexts into one.
    ///
    /// Creates a new context whose constraint set is the disjunction of the
    /// input contexts' constraints, each guarded by its merge condition.
    /// The merge conditions are 1-bit RustBV values; the merged context
    /// asserts `Or(all merge conditions)` to ensure at least one path holds.
    ///
    /// # Arguments
    /// * `others` - The other contexts to merge with `self`
    /// * `merge_conditions` - One condition per context (`self` first, then `others`)
    ///
    /// # Returns
    /// A new SymContext containing the merged constraints.
    #[cfg(feature = "vex-engine-z3")]
    pub fn merge(&self, others: &[&SymContext], merge_conditions: &[RustBV]) -> Self {
        assert_eq!(
            others.len() + 1,
            merge_conditions.len(),
            "merge_conditions must have one entry per context (self + others)"
        );

        // Start with a fresh context
        let mut merged = Self::with_timeout(self.timeout_ms.load(Ordering::SeqCst));

        // Merge symbol tables — merged is freshly constructed (Arc count == 1),
        // so Arc::make_mut returns a unique mutable reference without cloning.
        {
            let merged_table = Arc::make_mut(&mut merged.symbol_table);
            merged_table.extend(self.symbol_table.iter().map(|(k, v)| (k.clone(), *v)));
            for other in others {
                for (k, v) in other.symbol_table.iter() {
                    merged_table.entry(k.clone()).or_insert(*v);
                }
            }
        }

        // Set next_id to max across all contexts
        let mut max_id = self.next_id.load(Ordering::SeqCst);
        for other in others {
            max_id = max_id.max(other.next_id.load(Ordering::SeqCst));
        }
        merged.next_id.store(max_id, Ordering::SeqCst);

        // For each input context, guard its constraints with the merge condition:
        //   merge_cond_i => (constraint_1 AND constraint_2 AND ...)
        // Which is equivalent to: NOT(merge_cond_i) OR (constraint_1 AND constraint_2 AND ...)
        let all_contexts: Vec<&SymContext> = std::iter::once(self)
            .chain(others.iter().copied())
            .collect();
        let mut all_z3_conditions = Vec::new();

        for (ctx, cond) in all_contexts.iter().zip(merge_conditions.iter()) {
            // Compute NOT(cond) up front so cond_bool can be moved into
            // all_z3_conditions without cloning the Z3 AST.
            let cond_bool = cond.to_z3_bool();
            let not_cond = cond_bool.not();
            all_z3_conditions.push(cond_bool);

            // Collect all Z3 assertions and assumed pairs from this context.
            let shared = Arc::clone(&ctx.z3_assertions_shared.lock());
            let assumed_shared = Arc::clone(&ctx.assumed_constraints_shared.lock());
            let ctx_local = ctx.local_constraints.lock();

            // For each constraint c_j in context i:
            //   assert (NOT merge_cond_i OR c_j)
            // This means: if this merge path is active, all its constraints hold
            for assertion in shared.iter().chain(ctx_local.z3_assertions.iter()) {
                let guarded = z3::ast::Bool::or(&[&not_cond, assertion]);
                merged
                    .local_constraints
                    .lock()
                    .push_assertion(guarded.clone());
                merged.add_constraint(guarded);
            }

            // Also merge assumed_constraints for Python export
            {
                let mut merged_local = merged.local_constraints.lock();
                merged_local.assumed.extend(assumed_shared.iter().cloned());
                merged_local
                    .assumed
                    .extend(ctx_local.assumed.iter().cloned());
            }
        }

        // Assert that at least one merge condition is true
        let cond_refs: Vec<&z3::ast::Bool> = all_z3_conditions.iter().collect();
        let or_conds = z3::ast::Bool::or(&cond_refs);
        merged
            .local_constraints
            .lock()
            .push_assertion(or_conds.clone());
        merged.add_constraint(or_conds);

        merged
    }

    /// Merge without Z3 — just combines assumed constraints.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn merge(&self, others: &[&SymContext], merge_conditions: &[RustBV]) -> Self {
        let _ = merge_conditions;
        let mut merged = Self::new();

        // Merge symbol tables — see Z3 path comment about Arc::make_mut on
        // a freshly constructed Arc (refcount 1 → uniquely owned).
        {
            let merged_table = Arc::make_mut(&mut merged.symbol_table);
            merged_table.extend(self.symbol_table.iter().map(|(k, v)| (k.clone(), *v)));
            for other in others {
                for (k, v) in other.symbol_table.iter() {
                    merged_table.entry(k.clone()).or_insert(*v);
                }
            }
        }

        let mut max_id = self.next_id.load(Ordering::SeqCst);
        for other in others {
            max_id = max_id.max(other.next_id.load(Ordering::SeqCst));
        }
        merged.next_id.store(max_id, Ordering::SeqCst);

        // Merge assumed constraints (shared + local from each context).
        {
            let mut merged_local = merged.local_constraints.lock();
            let self_shared = Arc::clone(&self.assumed_constraints_shared.lock());
            merged_local.assumed.extend(self_shared.iter().cloned());
            merged_local
                .assumed
                .extend(self.local_constraints.lock().assumed.iter().cloned());
            for other in others {
                let other_shared = Arc::clone(&other.assumed_constraints_shared.lock());
                merged_local.assumed.extend(other_shared.iter().cloned());
                merged_local
                    .assumed
                    .extend(other.local_constraints.lock().assumed.iter().cloned());
            }
        }

        merged
    }
}
