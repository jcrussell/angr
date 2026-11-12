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

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use clarirs_core::GLOBAL_CONTEXT;
use clarirs_core::prelude::*;

use super::RustBV;
#[expect(
    unused_imports,
    reason = "referenced only by the intra-doc links below; rustc's unused_imports pass does not see doc-link uses"
)]
use super::SymContext;
use super::stats::{
    ASSERTED_WITHOUT_TERM_COUNT, RESIDUAL_LOGGED_COUNT, RESIDUAL_WITHOUT_TERM_COUNT,
};
use super::z3_lower::{LowerError, lower_bool};

/// A clarirs Bool together with its Z3 lowering.
///
/// The only constructors lower the term themselves, so the two halves cannot
/// disagree: a residual sink handed one of these records a constraint that can
/// be rebuilt from clarirs alone (angr-az6ru.2).
pub struct LoweredConstraint {
    term: AstRef<'static>,
    z3: z3::ast::Bool,
}

impl LoweredConstraint {
    /// Lower a Bool-sorted clarirs term. A term of any other sort is `Err`,
    /// as it is for `z3_lower::expect_bool`.
    pub fn lower(term: AstRef<'static>) -> Result<Self, LowerError> {
        let z3 = lower_bool(&term)?;
        Ok(Self { term, z3 })
    }

    /// The constraint a 1-bit value stands for: the Bool inside its
    /// `ite(c, 1, 0)` encoding, or `bit != 0`.
    ///
    /// Counts as one `to_z3_bool` in the solver stats, which it replaces at
    /// its call sites.
    pub fn of_bit(bit: &RustBV) -> Result<Self, LowerError> {
        super::stats::record_z3_ast_build();
        super::stats::record_z3_bool_build();
        Self::lower(bool_term(bit)?)
    }

    /// The Z3 form.
    pub fn z3(&self) -> &z3::ast::Bool {
        &self.z3
    }

    /// The clarirs form.
    pub fn term(&self) -> &AstRef<'static> {
        &self.term
    }

    /// The Z3 form alone, for a caller that records the constraint in the
    /// `assumed` log instead of the residual one.
    pub fn into_z3(self) -> z3::ast::Bool {
        self.z3
    }

    pub(super) fn into_parts(self) -> (AstRef<'static>, z3::ast::Bool) {
        (self.term, self.z3)
    }
}

/// The clarirs Bool a 1-bit value stands for.
pub(super) fn bool_term(bit: &RustBV) -> Result<AstRef<'static>, LowerError> {
    let term = match bit {
        RustBV::Concrete { value, .. } => GLOBAL_CONTEXT.boolv(*value != 0),
        RustBV::Ast { ast, .. } => angr_bv::bool_of(ast),
    };
    term.map_err(|err| LowerError::InvalidArgs(format!("no Bool view: {err}")))
}

/// Where an asserted constraint came from, which is also how its clarirs form
/// is recovered.
#[derive(Clone)]
pub(super) enum Origin {
    /// `assume_true` / `assume_false` of this 1-bit value, or a batch entry
    /// standing for the same pair. The term is derived on demand, so the
    /// branch path pays one refcount bump and builds no clarirs node.
    Assume(RustBV, bool),
    /// A sink that held the clarirs Bool.
    Term(AstRef<'static>),
    /// A sink that held only the Z3 term; `asserted_without_term` in the
    /// solver stats counts these.
    Z3Only,
}

/// One entry of the assertion log: a constraint asserted on the solver, with
/// what is needed to rebuild it from clarirs alone (angr-az6ru.2).
#[derive(Clone)]
pub(super) struct Asserted {
    pub(super) z3: z3::ast::Bool,
    pub(super) origin: Origin,
}

impl Asserted {
    /// The constraint as a clarirs Bool, or `None` for a [`Origin::Z3Only`]
    /// entry.
    pub(super) fn term(&self) -> Option<AstRef<'static>> {
        self.origin.term()
    }
}

/// A copy of one context's assertion log, for asserting the same constraints
/// on another context on the same thread ([`SymContext::export_constraint_log`],
/// [`SymContext::import_constraint_log`]). Opaque: it can only come from a
/// context, so every entry is a constraint some solver already holds.
pub struct ConstraintLog(pub(super) Vec<Asserted>);

impl ConstraintLog {
    /// How many constraints the log holds.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the log holds no constraint.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Origin {
    /// The constraint as a clarirs Bool, or `None` for [`Origin::Z3Only`].
    pub(super) fn term(&self) -> Option<AstRef<'static>> {
        match self {
            Origin::Assume(cond, want_true) => {
                // SILENT(cat-a): an assume whose cond has no Bool view never
                // reaches the log (`assume_unscoped` returns before pushing),
                // so `None` here is unreachable in practice and reads as an
                // entry without a term, which is counted.
                let term = bool_term(cond).ok()?;
                if *want_true {
                    Some(term)
                } else {
                    GLOBAL_CONTEXT.not(term).ok() // SILENT(cat-a): same as above
                }
            }
            Origin::Term(term) => Some(term.clone()),
            Origin::Z3Only => None,
        }
    }
}

/// One entry of the residual log: a constraint with no `assumed` pair.
///
/// `term` is its clarirs form. It is `None` for the sinks that hold only a Z3
/// term: an SMT-LIB2 residual replayed from a snapshot, and a `merge` guard
/// whose condition has no Z3 form. `residual_without_term` in the solver
/// stats counts those.
#[derive(Clone)]
pub(super) struct Residual {
    pub(super) z3: z3::ast::Bool,
    pub(super) term: Option<AstRef<'static>>,
}

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
    pub(super) z3_assertions: Vec<Asserted>,
    /// Local residual (no-[`RustBV`]) Z3 Bool assertions added after fork
    /// (angr-t3l5o Phase 1). A strict subset of `z3_assertions`: the entries
    /// pushed by the three residual sinks — `add_constraint_raw` (non-dup),
    /// `add_bv_constraint`, and the `merge` guard/`Or` asserts. Mirrors the
    /// `z3_assertions` shared/local lifecycle so `fork`'s `freeze_into_shared`
    /// and `merge` carry it without new logic. Dumped to `residual_smtlib2`
    /// in `to_snapshot`; the assume class is reconstructed from
    /// `assumed_constraints` IR instead.
    pub(super) non_bv_assertions: Vec<Residual>,
    /// HashSet of Z3_ast ptrs for O(1) dedup in `add_constraint_raw`.
    /// Holds ptrs for every assertion known to be currently asserted on the
    /// solver (i.e. everything in `z3_assertions_shared` + `z3_assertions`).
    /// Lazily seeded — `dedup_set_seeded == false` means the set is stale
    /// and must be rebuilt from the shared+local vecs before consultation.
    pub(super) dedup_set: HashSet<usize>,
    /// True once `dedup_set` has been populated from shared+local for this
    /// context. Reset to false by `fork()`, `merge()` (via `new()`), and a
    /// bare `pop()` that closes a scope which added assertions
    /// (`scope_savepoint_pop`, angr-ph300.41) — all of which truncate
    /// `z3_assertions`.
    pub(super) dedup_set_seeded: bool,
}

impl LocalConstraints {
    pub(super) fn new() -> Self {
        LocalConstraints {
            assumed: Vec::new(),
            z3_assertions: Vec::new(),
            non_bv_assertions: Vec::new(),
            dedup_set: HashSet::new(),
            dedup_set_seeded: false,
        }
    }

    /// Insert a Z3 Bool into `z3_assertions` and, when the dedup set is
    /// already seeded, record its ptr too.
    ///
    /// Callers must already know `b` is not present: this records the ptr
    /// but never *consults* the set. There is deliberately no bulk sibling —
    /// a bulk `extend_assertions` let `add_constraints_raw_batch` re-push and
    /// re-assert already-asserted constraints (angr-5mnx3.43). A bulk caller wanting dedup
    /// should loop over [`SymContext::seed_and_check_z3_dedup`], which
    /// pushes on a miss, exactly as `add_constraints_raw_batch` now does.
    pub(super) fn push_assertion(&mut self, b: z3::ast::Bool, origin: Origin) {
        if self.dedup_set_seeded {
            use z3::ast::Ast;
            let ptr = b.get_z3_ast().as_ptr() as usize;
            self.dedup_set.insert(ptr);
        }
        self.log_asserted(b, origin);
    }

    /// Append to `z3_assertions` without touching the dedup set. Every push
    /// onto the log goes through here, so `asserted_without_term` sees each
    /// [`Origin::Z3Only`] entry.
    pub(super) fn log_asserted(&mut self, z3: z3::ast::Bool, origin: Origin) {
        if matches!(origin, Origin::Z3Only) {
            ASSERTED_WITHOUT_TERM_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        self.z3_assertions.push(Asserted { z3, origin });
    }

    /// Record a constraint in the residual log. Every residual sink goes
    /// through here, so the `residual_*` counters see each one.
    pub(super) fn push_residual(&mut self, z3: z3::ast::Bool, term: Option<AstRef<'static>>) {
        RESIDUAL_LOGGED_COUNT.fetch_add(1, Ordering::Relaxed);
        if term.is_none() {
            RESIDUAL_WITHOUT_TERM_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        self.non_bv_assertions.push(Residual { z3, term });
    }
}
