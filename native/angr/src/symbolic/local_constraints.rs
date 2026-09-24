//! [`LocalConstraints`] — the post-fork local constraint additions of a
//! [`SymContext`], behind that context's single hot-path Mutex.
//!
//! Split out of `context.rs` (angr-fs8kb.35). The readers/mutators live in
//! `constraint_ops.rs` (add/assume), `snapshot_fork_ops.rs` (freeze on fork)
//! and `lineage_ops.rs` (savepoint truncation) — this file is the type
//! definition plus the two constructors/inserters those slices share.
//!
//! **Panic policy:** carries `#![deny(clippy::unwrap_used,
//! clippy::expect_used)]`, same as the rest of the `SymContext` slice files.
#![deny(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "vex-engine-z3")]
use std::collections::HashSet;

use super::RustBV;
#[expect(
    unused_imports,
    reason = "referenced only by the intra-doc links below; rustc's unused_imports pass does not see doc-link uses"
)]
use super::SymContext;

/// Local-only constraint state added after fork.
///
/// Combines `assumed` (RustBV pairs for Python export) and `z3_assertions`
/// (cached Z3 Bool nodes for fast fork replay) under a single Mutex so that
/// the hot path (`assume_true`/`assume_false`/`add_constraint_raw`) only
/// acquires one lock instead of two.
///
/// Also carries a `dedup_set` side-table of Z3_ast ptrs (angr-sfp9) used
/// by [`SymContext::add_constraint_raw`] to skip the push+assert work when
/// the incoming constraint is already asserted on the current solver. Z3's
/// hash-cons gives `ptr-equality == structural-equality` for live ASTs, so
/// the raw ptr is a valid identity key. The set is lazily seeded on first
/// access from `z3_assertions_shared` + `z3_assertions`, then maintained
/// incrementally by every path that pushes into `z3_assertions`.
pub(super) struct LocalConstraints {
    /// Local assumed (RustBV, is_assumed_true) pairs added after fork.
    /// `pub(super)` for the constraint-mutation methods in `constraint_ops.rs`
    /// (slice 9, angr-a2br.2.7).
    pub(super) assumed: Vec<(RustBV, bool)>,
    /// Local Z3 Bool assertions added after fork — only these are cloned on fork.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) z3_assertions: Vec<z3::ast::Bool>,
    /// Local residual (no-[`RustBV`]) Z3 Bool assertions added after fork
    /// (angr-t3l5o Phase 1). A strict subset of `z3_assertions`: the entries
    /// pushed by the three residual sinks — `add_constraint_raw` (non-dup),
    /// `add_bv_constraint`, and the `merge` guard/`Or` asserts. Mirrors the
    /// `z3_assertions` shared/local lifecycle so `fork`'s `freeze_into_shared`
    /// and `merge` carry it without new logic. Dumped to `residual_smtlib2`
    /// in `to_snapshot`; the assume class is reconstructed from
    /// `assumed_constraints` IR instead.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) non_bv_assertions: Vec<z3::ast::Bool>,
    /// HashSet of Z3_ast ptrs for O(1) dedup in `add_constraint_raw`.
    /// Holds ptrs for every assertion known to be currently asserted on the
    /// solver (i.e. everything in `z3_assertions_shared` + `z3_assertions`).
    /// Lazily seeded — `dedup_set_seeded == false` means the set is stale
    /// and must be rebuilt from the shared+local vecs before consultation.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) dedup_set: HashSet<usize>,
    /// True once `dedup_set` has been populated from shared+local for this
    /// context. Reset to false by `fork()`, `merge()` (via `new()`), and a
    /// bare `pop()` that closes a scope which added assertions
    /// (`scope_savepoint_pop`, angr-ph300.41) — all of which truncate
    /// `z3_assertions`.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) dedup_set_seeded: bool,
}

impl LocalConstraints {
    pub(super) fn new() -> Self {
        LocalConstraints {
            assumed: Vec::new(),
            #[cfg(feature = "vex-engine-z3")]
            z3_assertions: Vec::new(),
            #[cfg(feature = "vex-engine-z3")]
            non_bv_assertions: Vec::new(),
            #[cfg(feature = "vex-engine-z3")]
            dedup_set: HashSet::new(),
            #[cfg(feature = "vex-engine-z3")]
            dedup_set_seeded: false,
        }
    }

    /// Insert a Z3 Bool into `z3_assertions` and, when the dedup set is
    /// already seeded, record its ptr too.
    ///
    /// Callers must already know `b` is not present: this records the ptr
    /// but never *consults* the set. There is deliberately no bulk sibling —
    /// the one that existed (`extend_assertions`) was the whole of
    /// angr-5mnx3.43, letting `add_constraints_raw_batch` re-push and
    /// re-assert already-asserted constraints. A bulk caller wanting dedup
    /// should loop over [`SymContext::seed_and_check_z3_dedup`], which
    /// pushes on a miss, exactly as `add_constraints_raw_batch` now does.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) fn push_assertion(&mut self, b: z3::ast::Bool) {
        if self.dedup_set_seeded {
            use z3::ast::Ast;
            let ptr = b.get_z3_ast().as_ptr() as usize;
            self.dedup_set.insert(ptr);
        }
        self.z3_assertions.push(b);
    }
}
