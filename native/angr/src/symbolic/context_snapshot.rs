//! [`SymContextSnapshot`] — the serde capture of a [`SymContext`]'s
//! path-constraint state.
//!
//! Split out of `context.rs` (angr-fs8kb.35). Pure data + its serde defaults;
//! the capture/restore logic that produces and consumes it lives beside
//! `fork`/`merge` in `snapshot_fork_ops.rs`.
//!
//! **Panic policy:** carries `#![deny(clippy::unwrap_used,
//! clippy::expect_used)]`, same as the rest of the `SymContext` slice files.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use serde::{Deserialize, Serialize};

use super::RustBV;
#[expect(
    unused_imports,
    reason = "referenced only by the intra-doc links below; rustc's unused_imports pass does not see doc-link uses"
)]
use super::SymContext;

/// Snapshot of a [`SymContext`]'s path-constraint state.
///
/// Two-class capture (angr-t3l5o Phase 1):
///
/// * The **assume class** is captured as context-free [`RustBV`] IR in
///   `assumed_constraints`. On restore it is rebuilt by re-asserting each
///   pair through `assume_true`/`assume_false` — no SMT-LIB2 text emit or
///   parse. This is the common, hot path (path constraints from branch
///   forking) and is what makes cross-Z3-context migration cheap.
/// * The **residual class** is the set of solver assertions that have no
///   [`RustBV`] form and so are not reconstructible from `assumed_constraints`
///   — the raw entries from [`SymContext::add_constraint_raw`] (Python
///   claripy-sync fallback + cross-process pointer import), the
///   address-concretization equalities from [`SymContext::add_bv_constraint`],
///   and the `merge()` guard/`Or` disjunctions. These are carried as an
///   SMT-LIB2 text dump in `residual_smtlib2`, which is **empty** in the
///   common case (no raw/bv/merge constraints) — the win path that collapses
///   the old full-solver text round-trip.
///
/// Per-Z3-context cache state (solver, model_cache, sat_cache, lineage
/// scope_path, push stacks) is NOT included — these are runtime caches that
/// the loader rebuilds on first query against the restored constraints.
///
/// # Quiescence precondition (angr-9ke6b.135)
///
/// Because the push stacks are dropped, capture and restore are only valid at
/// a point with **no open push/pop scope**: `bare_z3_push_depth == 0`,
/// `scope_savepoints` empty, `scope_path` at its base. All real callers
/// ([`RustSimState::to_snapshot`](crate::state::RustSimState::to_snapshot) and
/// the stash-manager round-trip) are top-level state-persistence points that
/// satisfy this, and restore always runs against a freshly-constructed
/// [`SymContext`] whose depth is 0 by construction.
///
/// A snapshot taken mid-scope would silently reset `bare_z3_push_depth` to 0
/// on restore, which would let a later [`fork`](SymContext::fork) mint a
/// `SharedLineageSolver` frame in exactly the situation that counter's gate
/// exists to prevent (the parent's unbalanced bare pushes leaking into the
/// child's base — see bd memory `invariant-bare-z3-push-depth`).
/// `to_snapshot` / `restore_from_snapshot` carry a `debug_assert!` for this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymContextSnapshot {
    /// `(constraint, is_assumed_true)` pairs in insertion order.
    pub assumed_constraints: Vec<(RustBV, bool)>,
    /// Residual Z3 solver state as an SMT-LIB2 dump (angr-t3l5o Phase 1,
    /// renamed from `solver_smtlib2`).
    ///
    /// Carries ONLY the residual class — the solver assertions with no
    /// [`RustBV`] form (raw / bv-eq / merge-guard). Empty in the common
    /// assume-only case, so no `format!` text emit happens on the hot path.
    ///
    /// When `reassert_assumed` is false (a merged context, see that field)
    /// this instead carries the **full** solver dump and the `assumed`
    /// pairs are NOT re-asserted on restore.
    ///
    /// `#[serde(default)]` keeps deserialization tolerant of a missing field
    /// (empty residual → assume-only replay).
    #[serde(default)]
    pub residual_smtlib2: String,
    /// Whether `assumed_constraints` should be re-asserted on the solver at
    /// restore time (angr-t3l5o Phase 1).
    ///
    /// `true` (the common case): the assume class was directly asserted, so
    /// restore rebuilds the solver by re-asserting each pair via
    /// `assume_true`/`assume_false`, then replays `residual_smtlib2` (the
    /// raw/bv residual). `false`: the context came from a `merge()`, whose
    /// `assumed` pairs are export-only (the solver holds the guarded `Or`
    /// disjunctions, not the unconditional pairs). For those, restore replays
    /// the full `residual_smtlib2` dump and pushes the pairs to the BV-export
    /// log WITHOUT asserting — re-asserting would over-constrain the merged
    /// state. `#[serde(default = "default_true")]` keeps a missing field
    /// (legacy / mock) on the common re-assert path.
    #[serde(default = "default_true")]
    pub reassert_assumed: bool,
    /// The source context's authoritative `constraint_count` at capture time
    /// (angr-kenpr). Restore pins `constraint_count` back to this after the
    /// assume-class IR replay so `state_constraint_count` round-trips exactly —
    /// the replay can re-assert `assumed`-log entries that were live-deduped
    /// away on the source solver, inflating the counter otherwise.
    ///
    /// `#[serde(default)]` yields `0` for legacy snapshots, which restore reads
    /// as "not captured" and skips the pin (keeps the replayed count).
    #[serde(default)]
    pub constraint_count: usize,
    /// The source context's `next_id` watermark at capture time
    /// (angr-op0dn.13.14).
    ///
    /// Every `RustBV::Symbolic` / `Constrained` leaf carries an id minted from
    /// this counter. A restored state builds a **fresh** `SymContext`, whose
    /// counter would otherwise start at 0 — so the first symbol minted during
    /// the resume re-uses an id a restored leaf already owns, and every
    /// id-keyed lookup (the claripy export registry in
    /// [`SymbolicIdentityRegistry`](super::SymbolicIdentityRegistry),
    /// `stored_conditions`, the symbol table) silently aliases the two.
    /// Restore seeds the counter back to this watermark.
    ///
    /// `#[serde(default)]` yields `0` for legacy snapshots — restore reads that
    /// as "not captured" and falls back to the max leaf id seen in
    /// `assumed_constraints`.
    #[serde(default)]
    pub next_id: u64,
    /// Whether the source context was in deterministic (unsigned-minimum
    /// witness) `eval` mode at capture time (angr-ph300.46).
    ///
    /// A whole lineage stays in one witness-selection mode (`fork` inherits
    /// it), so a snapshot must round-trip it too — otherwise a restored
    /// deterministic state silently reverts to arbitrary-Z3-model witnesses
    /// and run-to-run nondeterminism reappears. `#[serde(default)]` yields
    /// `false` for legacy snapshots (the historical default).
    #[serde(default)]
    pub deterministic: bool,
    /// Whether the source context had the `SharedLineageSolver`
    /// materialization opt-in set at capture time (angr-ph300.46).
    ///
    /// Inherited across `fork` like `deterministic`; round-tripped so a
    /// restored descendant keeps minting shared lineages. `#[serde(default)]`
    /// yields `false` for legacy snapshots.
    #[serde(default)]
    pub use_shared_lineage_solver: bool,
}

/// serde default for [`SymContextSnapshot::reassert_assumed`] — the common
/// assume-reconstructible path.
fn default_true() -> bool {
    true
}
