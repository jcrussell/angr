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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::Mutex;

#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;
#[cfg(feature = "vex-engine-z3")]
use std::cell::{Cell, RefCell};
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::AtomicU32;

impl SymContext {
    /// Build a serializable snapshot of this context's path-constraint state
    /// (angr-x04s.1.2).
    ///
    /// Captures `assumed_constraints` — the canonical record from which all
    /// per-context Z3 cache state (solver, sat_cache, model_cache, lineage
    /// scope_path, push stacks) re-derives. Other [`SymContext`] fields are
    /// runtime-only caches that rebuild on first query.
    ///
    /// The two runtime counters — `next_id` and `constraint_count` — ARE
    /// captured and must be: the ids the restored `RustBV` leaves carry were
    /// minted by the *source process*'s allocator, which this process's
    /// allocator knows nothing about, so a resume would re-mint ids the
    /// restored leaves already own (angr-op0dn.13.14). Restore raises the
    /// global allocator past the captured watermark instead of pinning it, so
    /// it can never hand back an id this process already issued.
    /// See [`SymContextSnapshot::next_id`].
    ///
    /// See `snapshot-serialization-design` for the broader plan and
    /// `rustsimstate-field-buckets` for which surrounding state buckets
    /// are covered (RegisterFile / MemoryPage) versus deferred
    /// (Python-side `Py<PyAny>` overlays).
    pub fn to_snapshot(&self) -> SymContextSnapshot {
        let assumed_constraints = self.get_assumed_constraints();
        // angr-t3l5o Phase 1: two-class capture. The assume class is carried
        // as `assumed_constraints` IR and rebuilt by re-asserting via
        // `assume_*` on restore (no text). The RESIDUAL class (no-RustBV
        // assertions — raw / bv-eq / merge-guard) is carried as an SMT-LIB2
        // text dump in `residual_smtlib2`, EMPTY in the common case.
        //
        // The merge fallback: when `assume_class_reconstructible` is false the
        // context's `assumed` pairs are export-only (a `merge()` left the
        // guarded `Or` disjunctions on the solver, not the unconditional
        // pairs). Re-asserting them would over-constrain, so we instead dump
        // the FULL solver and tell restore (via `reassert_assumed = false`)
        // not to re-assert the assume class — the pre-Phase-1 behavior,
        // preserved for correctness on merged lineages.
        #[cfg(feature = "vex-engine-z3")]
        let (residual_smtlib2, reassert_assumed) = {
            let reassert_assumed = self
                .assume_class_reconstructible
                .load(std::sync::atomic::Ordering::Relaxed);
            // angr-t3l5o Phase 1: self-checking invariant. `z3_assertions`
            // (and thus `z3_assertion_count`) is the disjoint union of the
            // assume-class asserts (reconstructible from `assumed_constraints`)
            // and the residual log. We use the WEAKER bounded form rather than
            // an exact equality because the exact reconstructible-assume count
            // is hard to recover here: an `assumed` entry produces a distinct
            // solver assertion only when it is neither a concrete-true
            // tautology nor a ptr-dedup hit (see `assume_true`/`assume_false`),
            // and Z3 may also drop/merge structurally-identical asserts in
            // `get_assertions()`. The bounds catch the real failure mode this
            // guard targets — a future caller asserting on the solver without
            // recording into a log — while tolerating those benign sub-counts.
            #[cfg(debug_assertions)]
            {
                let non_bv_count = self.non_bv_assertions_shared.lock().len()
                    + self.local_constraints.lock().non_bv_assertions.len();
                let z3_count = self.z3_assertion_count();
                let assumed_len = assumed_constraints.len();
                debug_assert!(
                    z3_count >= non_bv_count,
                    "residual log ({non_bv_count}) exceeds solver assertion \
                     count ({z3_count}) — a residual sink recorded a Bool that \
                     never reached the solver"
                );
                debug_assert!(
                    z3_count <= assumed_len + non_bv_count,
                    "solver assertion count ({z3_count}) exceeds assume \
                     ({assumed_len}) + residual ({non_bv_count}) — a caller \
                     asserted on the solver without recording into a log"
                );
            }
            // angr-t3l5o Phase 0b: record the EXACT residual (no-RustBV)
            // assertion count, env-gated. Phase 1 makes the exact count
            // available (the `non_bv_assertions` log) so we no longer
            // approximate it as `z3_assertion_count - assumed.len()`.
            if crate::migrate_phase_timers::count_armed() {
                let non_bv_count = (self.non_bv_assertions_shared.lock().len()
                    + self.local_constraints.lock().non_bv_assertions.len())
                    as u64;
                crate::migrate_phase_timers::add_raw_constraint_count(non_bv_count);
            }
            // Time the EMIT phase around the dump that actually runs. On the
            // common reconstructible path this is `dump_non_bv_smtlib2`, which
            // returns "" when the residual log is empty — so post-Phase-1
            // attribution shows the text emit collapse toward ~0.
            let residual = crate::migrate_phase_timers::time_phase(
                &crate::migrate_phase_timers::MIGRATE_SMTLIB2_EMIT_NS,
                || {
                    if reassert_assumed {
                        self.dump_non_bv_smtlib2()
                    } else {
                        self.dump_solver_smtlib2()
                    }
                },
            );
            (residual, reassert_assumed)
        };
        #[cfg(not(feature = "vex-engine-z3"))]
        let (residual_smtlib2, reassert_assumed) = (
            String::new(),
            self.assume_class_reconstructible
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        SymContextSnapshot {
            assumed_constraints,
            residual_smtlib2,
            reassert_assumed,
            // angr-kenpr: capture the authoritative live assertion count so
            // restore can report the SAME `num_constraints` the source had.
            // The assume-class IR replay (Phase 1) rebuilds from the `assumed`
            // LOG, which preserves entries that were live-deduped away on the
            // source solver (ptr-dedup in `check_z3_dedup_if_seeded`). Those
            // entries re-assert on restore under fresh ptrs, so a naive replay
            // over-counts. Pinning `constraint_count` to the source value keeps
            // the round-trip contract (`state_constraint_count` preserved) that
            // the pre-Phase-1 full-solver dump gave for free. See bd memory
            // `snapshot-constraint-count-pin`.
            constraint_count: self.num_constraints(),
            // angr-op0dn.13.14: capture the id watermark. Restore builds a
            // fresh SymContext (next_id = 0) whose restored leaves already own
            // ids 0..watermark, so without this every symbol minted during the
            // resume aliases one of them.
            next_id: crate::symbolic::symbol_id_watermark(),
        }
    }

    /// Helper for [`Self::to_snapshot`]: dump the union of
    /// `z3_assertions_shared` and the local Z3 assertion vector (the FULL
    /// solver state) into a temp [`z3::Solver`] and emit SMT-LIB2.
    ///
    /// Returns the empty string when no Z3 assertions exist. Used only on the
    /// merge fallback path (`assume_class_reconstructible == false`); the
    /// common path uses [`Self::dump_non_bv_smtlib2`]. Still reachable from
    /// the Phase-0 bench hook `bench_dump_solver_smtlib2`.
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
            format!("{temp}")
        } else {
            String::new()
        }
    }

    /// Helper for [`Self::to_snapshot`]: dump ONLY the residual (no-[`RustBV`])
    /// assertions — `non_bv_assertions_shared ∪ local.non_bv_assertions` —
    /// into a temp [`z3::Solver`] and emit SMT-LIB2 (angr-t3l5o Phase 1).
    ///
    /// Returns the empty string when the residual log is empty, which is the
    /// common case (pure assume-class contexts). This is the lever that
    /// collapses the old full-solver text round-trip: the assume class is
    /// rebuilt from `assumed_constraints` IR instead of from text.
    #[cfg(feature = "vex-engine-z3")]
    fn dump_non_bv_smtlib2(&self) -> String {
        let temp = z3::Solver::new();
        let mut any = false;
        let shared = Arc::clone(&self.non_bv_assertions_shared.lock());
        for c in shared.iter() {
            temp.assert(c);
            any = true;
        }
        let local = self.local_constraints.lock();
        for c in local.non_bv_assertions.iter() {
            temp.assert(c);
            any = true;
        }
        if any {
            format!("{temp}")
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
    /// angr-t3l5o Phase 1: two-class reattach. When `reassert_assumed` is
    /// true (the common case) the assume class is rebuilt from
    /// `assumed_constraints` IR by re-asserting each pair through the public
    /// `assume_*` APIs — repopulating the Z3 solver, the BV-export log, and
    /// `constraint_count` with NO SMT-LIB2 text emit/parse — then the
    /// (usually empty) residual is replayed. When false (a merged context)
    /// the full residual dump is replayed and the `assumed` pairs are pushed
    /// to the BV-export log WITHOUT asserting (re-asserting would
    /// over-constrain the guarded merge state); `assume_class_reconstructible`
    /// is set false so a re-migration of the restored context stays correct.
    ///
    /// Must run inside an active Z3 thread-local context when the
    /// `vex-engine-z3` feature is enabled (the same rule as `RustBV`
    /// deserialization — see `snapshot-rustbv-shadow-type-pattern`).
    pub fn restore_from_snapshot(&self, snap: &SymContextSnapshot) {
        // angr-op0dn.13.14: raise the global id allocator past every id the
        // restored leaves already own. The `RustBV::Symbolic` leaves
        // deserialized into this context's registers / memory / constraints
        // carry ids minted by the SOURCE process's allocator, which ours never
        // issued. Without the reserve, the first symbol minted during the
        // resume re-uses an id a restored leaf holds, and every id-keyed lookup
        // (claripy export registry, `stored_conditions`, symbol table) aliases
        // the two. A symbol aliased onto another collapses a downstream
        // compare-and-branch to a CONCRETE guard, so `IRStmt::Exit` stops
        // forking and whole search subtrees vanish with no visible error.
        // `fetch_max`, so a legacy snapshot's 0 is a no-op.
        //
        // angr-euw28: when the envelope came from a FOREIGN process the whole
        // id space is rebased by an offset (see `SymbolIdRebase`), so the
        // restored leaves top out at `next_id - 1 + offset`, not `next_id - 1`.
        crate::symbolic::reserve_symbol_id(
            snap.next_id
                .saturating_add(crate::symbolic::symbol_id_rebase_offset()),
        );
        #[cfg(feature = "vex-engine-z3")]
        {
            if snap.reassert_assumed {
                // (i) Rebuild the assume class from IR — no text round-trip.
                for (cond, is_true) in &snap.assumed_constraints {
                    if *is_true {
                        self.assume_true(cond);
                    } else {
                        self.assume_false(cond);
                    }
                }
                // (ii) Replay the residual (raw / bv-eq) class, if any. Routes
                // through `add_constraint_raw` (residual sink #1) so the
                // restored context's `non_bv_assertions` log is rebuilt too.
                if !snap.residual_smtlib2.is_empty() {
                    self.replay_residual_smtlib2(&snap.residual_smtlib2);
                }
                // (iii) angr-kenpr: pin `constraint_count` back to the source's
                // authoritative value. The assume-class IR replay above can
                // re-assert `assumed`-log entries that were live-deduped away on
                // the source solver (ptr-dedup fires against a since-freed AST
                // that is not reproduced under the fresh restore ptrs), which
                // inflates the counter. The source count is the round-trip
                // contract `state_constraint_count` must preserve; the extra
                // re-asserted Bool is a logically-redundant no-op on the solver.
                // Legacy snapshots (pre-kenpr) carry 0 here → skip the pin and
                // keep the replayed count. See bd `snapshot-constraint-count-pin`.
                if snap.constraint_count > 0 {
                    self.set_constraint_count(snap.constraint_count);
                }
            } else {
                // Merge fallback: the `assumed` pairs are export-only. Replay
                // the full residual dump, then push the pairs to the BV log
                // without asserting.
                self.assume_class_reconstructible
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                if !snap.residual_smtlib2.is_empty() {
                    self.replay_residual_smtlib2(&snap.residual_smtlib2);
                }
                for (cond, is_true) in &snap.assumed_constraints {
                    self.assumed_constraints_push(cond.clone(), *is_true);
                }
            }
        }
        // Non-Z3 (mock) path: no solver to assert against, so replay the
        // assume pairs through the mock `assume_*` (which only repopulates the
        // BV-export log). The residual class does not exist without Z3.
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            for (cond, is_true) in &snap.assumed_constraints {
                if *is_true {
                    self.assume_true(cond);
                } else {
                    self.assume_false(cond);
                }
            }
        }
    }

    /// Replay an SMT-LIB2 residual dump onto this context's solver via the
    /// `from_string` + `get_assertions` + [`Self::add_constraint_raw`] loop
    /// (angr-t3l5o Phase 1). Each re-asserted Bool lands in the residual log
    /// (`add_constraint_raw` is residual sink #1). Times the parse phase into
    /// `MIGRATE_SMTLIB2_PARSE_NS` when migration phase timers are armed.
    #[cfg(feature = "vex-engine-z3")]
    fn replay_residual_smtlib2(&self, residual: &str) {
        use z3::ast::Ast;
        crate::migrate_phase_timers::time_phase(
            &crate::migrate_phase_timers::MIGRATE_SMTLIB2_PARSE_NS,
            || {
                let tmp = z3::Solver::new();
                tmp.from_string(residual);
                let z3_ctx = z3::Context::thread_local();
                for assertion in tmp.get_assertions() {
                    let ptr = assertion.get_z3_ast().as_ptr() as usize;
                    // SAFETY: `ptr` came from a Bool returned by
                    // `get_assertions()` parsed into the thread-local Z3
                    // context (the same one `add_constraint_raw` will use).
                    // `from_borrowed_raw` takes its own ref via `Z3_inc_ref`,
                    // independent of the temp solver's reference.
                    if let Some(z3_ast) =
                        unsafe { super::Z3AstPtr::from_borrowed_raw(&z3_ctx, ptr) }
                    {
                        self.add_constraint_raw(z3_ast);
                    }
                }
            },
        );
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
        // angr-t3l5o Phase 1: translate_into routes EVERY assertion (assume
        // class included) through `add_constraint_raw`, so the rebuilt
        // context's residual log holds the whole constraint set while
        // `assumed` holds export-only duplicates of the assume class. Mark it
        // non-reconstructible so a later `to_snapshot` dumps the full solver
        // and does not also re-assert `assumed` (which would double the assume
        // class). This path is not snapshotted in production, but the flag
        // keeps the two transports composable.
        new.assume_class_reconstructible
            .store(false, std::sync::atomic::Ordering::Relaxed);
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
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork(&self) -> Self {
        let in_transaction = self.push_level.load(Ordering::Relaxed) > 0;
        let (frozen_shared, frozen_assumed, frozen_non_bv) = {
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
            // angr-t3l5o Phase 1: the residual log follows the IDENTICAL
            // shared/local freeze lifecycle as `z3_assertions` so fork/merge
            // never drop residual constraints.
            let frozen_non_bv = freeze_into_shared(
                &self.non_bv_assertions_shared,
                &mut local.non_bv_assertions,
                in_transaction,
            );
            (frozen_shared, frozen_assumed, frozen_non_bv)
        };

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
            Some(crate::arc_shared(Mutex::new(lineage_solver)))
        } else if dismantled {
            None
        } else {
            self.lineage.lock().as_ref().map(Arc::clone)
        };

        SymContext {
            // angr-ph300.47: seed the child count from the parent's live
            // `num_constraints()`, NOT `assumed_total_len` (frozen_assumed.len()).
            // The child inherits the parent's FULL constraint set via
            // frozen_shared/frozen_non_bv, which includes entries with no
            // `assumed` pair — `add_constraint_raw` residuals and
            // `add_bv_constraint` (address concretization) both bump
            // `constraint_count` but push no assumed pair. Seeding from the
            // assumed length alone erased those from the child's count, so a
            // concretize-then-fork shrank `num_constraints()` below the parent's
            // for an identical effective set. This mirrors the angr-kenpr
            // snapshot-restore pin (`create_snapshot` pins to `num_constraints()`),
            // keeping the `state_constraint_count` round-trip contract intact.
            constraint_count: AtomicUsize::new(self.num_constraints()),
            symbol_table: Arc::clone(&self.symbol_table),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            push_local_cache_lengths: Mutex::new(PushStack::new()),
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            bare_local_savepoints: Mutex::new(Vec::new()),
            assumed_constraints_shared: Mutex::new(frozen_assumed),
            z3_assertions_shared: Mutex::new(frozen_shared),
            non_bv_assertions_shared: Mutex::new(frozen_non_bv),
            // angr-t3l5o Phase 1: inherit reconstructibility parent→child so a
            // merge's export-only-assume marking propagates down the lineage.
            assume_class_reconstructible: AtomicBool::new(
                self.assume_class_reconstructible.load(Ordering::Relaxed),
            ),
            local_constraints: Mutex::new(LocalConstraints::new()),
            solver: Mutex::new(None),
            // angr-gorvf.16: carry the parent's SAT/model witness into the
            // child. The child's frozen constraint set is exactly the parent's
            // full set at fork time (freeze_into_shared moved every local
            // constraint into `frozen_shared`/`frozen_assumed`), so any model
            // that satisfies the parent satisfies the child. Keeping it warm
            // means the child's first `eval()` — e.g. the Rust-redirected
            // `state.solver.eval` a Python callback SimProcedure runs on a
            // freshly materialized state — hits `model.eval()` instead of
            // paying a fresh `solver.check()` (CheckSite::Eval). Any constraint
            // the child later adds runs `invalidate_model_if_inconsistent`,
            // which drops a now-stale model, so this stays sound. In
            // deterministic mode `eval` ignores the cache (uses `min`), so this
            // only changes the non-deterministic witness — from an arbitrary
            // fresh model to the parent's equally-valid one.
            sat_cache: Cell::new(self.sat_cache.get()),
            model_cache: RefCell::new(self.model_cache.borrow().clone()),
            constraint_trackers: Mutex::new(Vec::new()),
            timeout_ms: AtomicU32::new(self.timeout_ms.load(Ordering::SeqCst)),
            // angr-op0dn.10.2: a whole lineage stays in one witness-selection
            // mode — a child forked from a deterministic parent is deterministic.
            deterministic: AtomicBool::new(self.deterministic.load(Ordering::Relaxed)),
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
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::clone(&self.symbol_table),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(frozen_assumed),
            assume_class_reconstructible: AtomicBool::new(
                self.assume_class_reconstructible.load(Ordering::Relaxed),
            ),
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

        // No id reconciliation needed: symbol ids come from the process-global
        // allocator (`symbolic::bv_id_ops::NEXT_SYMBOL_ID`), so every input
        // context's ids are already disjoint from every other's.

        // For each input context, guard its constraints with the merge condition:
        //   merge_cond_i => (constraint_1 AND constraint_2 AND ...)
        // Which is equivalent to: NOT(merge_cond_i) OR (constraint_1 AND constraint_2 AND ...)
        let all_contexts: Vec<&SymContext> = std::iter::once(self)
            .chain(others.iter().copied())
            .collect();
        let mut all_z3_conditions = Vec::new();

        // angr-op0dn.11.3: shared-prefix-aware merge. `fork()` freezes self's
        // local constraints into `z3_assertions_shared` and hands the SAME Arc
        // to the child (`freeze_into_shared`), so sibling diamond arms hold a
        // POINTER-EQUAL frozen prefix all the way to this merge point. When
        // every input context shares that one Arc, the prefix holds on EVERY
        // path reaching the merge (all arms descend from the fork point), so we
        // assert it ONCE, UNGUARDED, and guard only each arm's divergent local
        // suffix — the guarded-`Or` count becomes proportional to the
        // DIVERGENCE (Σ local) instead of the TOTAL (Σ shared+local).
        //
        // Soundness: the pre-11.3 shape already forces the prefix on every
        // model — `Or(conds)` makes some `cond_i` true, whose guard then
        // requires that arm's (identical) shared prefix — so lifting the prefix
        // out of the guards removes no models and adds none. The rewrite fires
        // only when the Arcs are ptr-equal (byte-identical content), so there
        // is no shadowing: `local` is strictly additive to the frozen prefix
        // (`freeze_into_shared` only appends; an arm never retracts a shared
        // constraint — the S5b spike's `test_local_never_shadows_shared` checks
        // this). If the prefixes are NOT all ptr-equal (arms with no common
        // frozen ancestor), fall back to guarding every constraint of every arm.
        let common_prefix = Arc::clone(&all_contexts[0].z3_assertions_shared.lock());
        let prefix_shared_by_all = all_contexts[1..]
            .iter()
            .all(|ctx| Arc::ptr_eq(&ctx.z3_assertions_shared.lock(), &common_prefix));
        let use_prefix_fast_path = prefix_shared_by_all && !common_prefix.is_empty();

        if use_prefix_fast_path {
            // Assert the common frozen prefix exactly once, unguarded. Record
            // each in the residual log so the `to_snapshot` self-invariant
            // (z3_count <= assumed + non_bv) holds regardless of how the prefix
            // constraint originated; the merged context is dumped full-solver on
            // snapshot anyway (`assume_class_reconstructible = false` below).
            {
                let mut ml = merged.local_constraints.lock();
                for assertion in common_prefix.iter() {
                    ml.push_assertion(assertion.clone());
                    ml.non_bv_assertions.push(assertion.clone());
                }
            }
            for assertion in common_prefix.iter() {
                merged.add_constraint(assertion.clone());
            }
        }

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

            // On the shared-prefix fast path guard ONLY the divergent local
            // suffix (the prefix was asserted unguarded above); otherwise guard
            // every assertion of the arm (shared prefix + local suffix).
            let to_guard: Vec<&z3::ast::Bool> = if use_prefix_fast_path {
                ctx_local.z3_assertions.iter().collect()
            } else {
                shared
                    .iter()
                    .chain(ctx_local.z3_assertions.iter())
                    .collect()
            };

            // For each constraint c_j guarded for context i:
            //   assert (NOT merge_cond_i OR c_j)
            // This means: if this merge path is active, all its constraints hold
            for assertion in to_guard {
                let guarded = z3::ast::Bool::or(&[&not_cond, assertion]);
                {
                    // angr-t3l5o Phase 1: residual sink #3 (merge guard). The
                    // guarded `Or` has no RustBV form and is not reconstructible
                    // from `assumed`, so record it in the residual log too.
                    let mut ml = merged.local_constraints.lock();
                    ml.push_assertion(guarded.clone());
                    ml.non_bv_assertions.push(guarded.clone());
                }
                merged.add_constraint(guarded);
                #[cfg(test)]
                super::context::merge_instrument::note_guarded();
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
        {
            // angr-t3l5o Phase 1: residual sink #3 (merge `Or` guard).
            let mut ml = merged.local_constraints.lock();
            ml.push_assertion(or_conds.clone());
            ml.non_bv_assertions.push(or_conds.clone());
        }
        merged.add_constraint(or_conds);
        #[cfg(test)]
        super::context::merge_instrument::note_guarded();

        // angr-t3l5o Phase 1: a merged context's `assumed` pairs are
        // export-only — the solver holds the guarded `Or`s above, not the
        // unconditional pairs. Mark it non-reconstructible so `to_snapshot`
        // dumps the full solver and restore does NOT re-assert the pairs.
        merged
            .assume_class_reconstructible
            .store(false, Ordering::Relaxed);

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

        // angr-t3l5o Phase 1: keep the flag consistent with the Z3 merge path
        // (inert for the mock restore, which always replays via `assume_*`).
        merged
            .assume_class_reconstructible
            .store(false, Ordering::Relaxed);

        merged
    }
}
