//! Z3 solver context and constraint management.
//!
//! The `SymContext` manages:
//! - Symbolic variable creation and ID assignment
//! - Constraint tracking (when Z3 is available)
//! - Satisfiability checking (when Z3 is available)
//!
//! ## Lineage + solver invariants
//!
//! The shared-lineage Z3 solver path (angr-v5a5 / angr-3ms1 / angr-v5ht)
//! introduces several cross-cutting invariants that future refactors must
//! preserve when splitting this module (see angr-a2br for the planned
//! split). Invariants that have a bd memory carrying the long-form
//! rationale cite its key (recall via `bd recall <key>`); the rest are
//! stated in full here.
//!
//! - **`lineage` mutex shape**:
//!   `Mutex<Option<Arc<Mutex<SharedLineageSolver>>>>`. The outer `Mutex` is
//!   load-bearing — [`fork`](SymContext::fork) takes `&self`, not `&mut
//!   self`, and must be able to install a fresh lineage. Collapsing to a
//!   plain `Option<Arc<…>>` or to `OnceLock` would force the fork
//!   signature to change.
//! - **FrameId minting**: per-state scope-path
//!   identity uses a globally-monotonic [`FrameId`](super::lineage::FrameId)
//!   minted at constraint-add time, NOT `Arc::ptr_eq` on the RustBV. This
//!   buys [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
//!   a cheap by-id prefix comparison while staying robust across fork
//!   boundaries where RustBV identity can split.
//! - **Three-gate materialization** (`invariant-bare-z3-push-depth`,
//!   `invariant-v5ht-dismantle-child-none`):
//!   [`fork`](SymContext::fork) only mints a fresh `SharedLineageSolver`
//!   when (a) the parent opted in via
//!   [`set_use_shared_lineage_solver`](SymContext::set_use_shared_lineage_solver),
//!   (b) the parent's [`bare_z3_push_depth`](SymContext::bare_z3_push_depth)
//!   is zero, and (c) the runtime thrash detector
//!   ([`super::lineage::is_lineage_dismantled`]) has not fired. When (c) is
//!   true `child_lineage` is set to `None` rather than `Arc::clone`'d —
//!   `Arc::clone` would give the child a stale base. Regression guard:
//!   `tests/engines/rust/ :: test_lineage_minted_only_when_opted_in`
//!   and `test_lineage_not_minted_under_bare_push`.
//! - **fork-freeze under push**:
//!   [`fork`](SymContext::fork) only drains local→shared in place when
//!   `push_level == 0`. Inside an open scope a `pop()` truncates `local`
//!   back to its pre-push length; draining would leak popped constraints
//!   into `shared`. (`push_level` is now always 0 — the transaction API that
//!   raised it was removed in angr-ph300.44 — but the guard is retained.)
//! - **Z3 construction canonicalization** (`invariant-z3-construction-canonicalization`):
//!   Z3 hash-cons applies at construction time, but commutative operands
//!   are NOT normalized (`mk_bvadd(x, y)` and `mk_bvadd(y, x)` produce
//!   distinct AST pointers). Concrete numerals and named symbol leaves ARE
//!   canonical. Regression guard:
//!   `super::value::tests::z3_already_dedupes_structurally_equal_rustbv_trees`.
//! - **No `parallel.enable`** (`avoid-z3-parallel-enable`): setting
//!   `parallel.enable=true` on solver params is correctness-breaking on
//!   this codebase (downstream consumers do not handle Z3 `Unknown`
//!   results). Do NOT re-enable in [`build_solver_params`].
//! - **No full lineage teardown**: the
//!   angr-0dgq teardown variant (walk all stashes, drop each state's
//!   lineage Arc, invalidate per-context solvers) is a net loss vs the
//!   v5ht simple variant. [`super::lineage::set_lineage_dismantled`] only
//!   suppresses future mints; in-flight lineages keep working.
//! - **No DFS-only opt-in**: do
//!   NOT default `use_shared_lineage_solver` on for `strategy='dfs'`. Both
//!   the canonical WIN (ais3_crackme) and LOSE (defcon2016quals_baby-re)
//!   canaries run BFS — strategy is not the discriminator.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. It owns the Z3
//! solver behind guest-derived constraints; the one surviving `expect` in
//! [`SymContext::solver`] reads back a lazily-materialized solver the same
//! function just stored under the same held guard.
#![deny(clippy::unwrap_used, clippy::expect_used)]

// The Z3-only half of the std imports: every consumer of `Cell`/`RefCell`
// (`sat_cache`, `model_cache`), `HashSet` (`LocalConstraints::dedup_set`),
// `AtomicU32` (`timeout_ms`) and `Ordering` is itself behind
// `#[cfg(feature = "vex-engine-z3")]`, so importing them unconditionally warns
// in the no-z3 combos `make check-no-z3` gates (angr-sqfj8.139).
#[cfg(feature = "vex-engine-z3")]
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
#[cfg(feature = "vex-engine-z3")]
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::RustBV;
// Constraint-sharing analysis types live in `sharing.rs` (angr-a2br.2 slice 3).
// `fold_sharing_walk` below takes a `&mut ConstraintSharingWalk`.
// Solver/engine profiling counters live in `stats.rs` (angr-ugc2). The glob
// brings every counter static plus `CheckSite` / `VexOpFamily` and the
// `record_*` helpers into scope so the solving paths below read and bump them
// unchanged. Z3-gated: every read/bump site is itself behind
// `vex-engine-z3` (angr-sqfj8.139).
#[cfg(feature = "vex-engine-z3")]
use super::stats::*;

// Z3 solver construction + per-check timing/sampling wrappers live in
// `solver_build.rs` (angr-a2br.2 slice 2). The glob brings `build_solver`,
// `build_solver_params`, `timed_check`, and `sample_simplify_skip` into scope
// for the solving paths below (and, via `use super::*`, the test module).
#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;

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

/// Default Z3 solver timeout in milliseconds.
///
/// 30 seconds — chosen to match claripy's historical default and to cap the
/// occasional Z3 outlier on bimodal-SAT benches. Overridable per-state via
/// `RustExplorationManager::set_solver_timeout`.
pub const DEFAULT_SOLVER_TIMEOUT_MS: u32 = 30_000;

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
    #[cfg(feature = "vex-engine-z3")]
    pub(super) fn push_assertion(&mut self, b: z3::ast::Bool) {
        if self.dedup_set_seeded {
            use z3::ast::Ast;
            let ptr = b.get_z3_ast().as_ptr() as usize;
            self.dedup_set.insert(ptr);
        }
        self.z3_assertions.push(b);
    }

    /// Bulk variant of [`Self::push_assertion`].
    ///
    /// `pub(super)` for `add_constraints_raw_batch` in `constraint_ops.rs`
    /// (slice 9, angr-a2br.2.7).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) fn extend_assertions<I: IntoIterator<Item = z3::ast::Bool>>(&mut self, iter: I) {
        if self.dedup_set_seeded {
            use z3::ast::Ast;
            for b in iter {
                let ptr = b.get_z3_ast().as_ptr() as usize;
                self.dedup_set.insert(ptr);
                self.z3_assertions.push(b);
            }
        } else {
            self.z3_assertions.extend(iter);
        }
    }
}

/// Solver context for symbolic execution.
///
/// Manages symbolic variable creation and, when Z3 is available,
/// constraint solving and satisfiability checking.
///
/// With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
/// to store a reference to it. All Z3 operations on a thread share
/// the same context automatically.
///
/// Bare scope save/restore is available via `push()` / `pop()` / `try_pop()`
/// (see `transaction_ops.rs`); the higher-level `transaction_begin/commit/
/// rollback` API was removed in angr-ph300.44 (dead code + latent corruption).
pub struct SymContext {
    /// Number of constraints added (for tracking).
    /// `pub(super)` for the `num_constraints` accessor in `bv_id_ops.rs`.
    pub(super) constraint_count: AtomicUsize,
    /// Named symbolic variables for debugging.
    /// Arc-shared on fork (O(1) clone). Only mutated when constructing
    /// a fresh merged context — Arc::make_mut works because the merged
    /// SymContext is freshly created with a unique Arc.
    pub(super) symbol_table: Arc<HashMap<String, u64>>,
    /// Transaction push level, read by `fork()`'s `in_transaction` gate
    /// (`snapshot_fork_ops`) and exposed for diagnostics via `debug_push_level`.
    /// The only incrementer was `transaction_begin`, removed in angr-ph300.44
    /// (dead API + latent corruption), so this is now always 0 and the fork
    /// gate consequently always sees no transaction. Kept as a field so the
    /// gate and `debug_push_level` stay structurally intact.
    pub(super) push_level: AtomicUsize,
    /// Local-constraint savepoints for **bare** `push()`/`pop()` (angr-ph300.41/.42).
    ///
    /// Each entry records `(z3_assertions.len(), assumed.len(),
    /// non_bv_assertions.len())` captured by `scope_savepoint_push()` at the
    /// moment of a bare `push()`. `scope_savepoint_pop()` truncates all three
    /// local logs back to the saved lengths and drops the dedup side-table,
    /// so constraints added inside a bare push/pop scope do not leak past the
    /// matching `pop()`. Without this, a `pop()` popped the Z3 frame but left the
    /// `z3_assertions` log (and `dedup_set` ptrs) intact: re-adding the same
    /// constraint dedup-hit and was skipped (.41), and `fork()` replayed the
    /// stale log into the child as permanent asserts (.42).
    ///
    /// Pairs push↔pop exactly like `scope_savepoints`.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) bare_local_savepoints: Mutex<Vec<(usize, usize, usize)>>,
    /// Track assumed RustBV constraints for export to Python.
    /// Each entry is (constraint, is_assumed_true). The shared prefix is an
    /// Arc<Vec<...>> for O(1) clone on fork; local additions live alongside
    /// `z3_assertions` in `local_constraints` so the hot path only takes one
    /// lock for both vectors.
    /// Wrapped in Mutex so fork() can freeze local into shared in-place when safe.
    pub(super) assumed_constraints_shared: Mutex<Arc<Vec<(RustBV, bool)>>>,

    /// Shared (frozen) Z3 Bool assertions from parent — O(1) clone via Arc.
    /// Wrapped in Mutex so fork() can freeze local into shared in-place when safe
    /// (avoids cloning every Bool — each Bool::clone would call Z3_inc_ref).
    /// `pub(super)` for the dedup-seed path in `constraint_ops.rs`
    /// (slice 9, angr-a2br.2.7).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) z3_assertions_shared: Mutex<Arc<Vec<z3::ast::Bool>>>,

    /// Shared (frozen) residual (no-[`RustBV`]) Z3 Bool assertions from
    /// parent (angr-t3l5o Phase 1) — mirrors `z3_assertions_shared` exactly so
    /// `fork`'s `freeze_into_shared` carries it with no new logic. Holds only
    /// the residual subset (raw / bv-eq / merge-guard); the assume class is
    /// reconstructed from `assumed_constraints` IR. Dumped to
    /// `residual_smtlib2` in `to_snapshot`.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) non_bv_assertions_shared: Mutex<Arc<Vec<z3::ast::Bool>>>,

    /// Whether this context's `assumed` pairs are directly asserted on the
    /// solver (angr-t3l5o Phase 1). `true` for normal contexts — re-asserting
    /// the assume class via `assume_*` reconstructs the assume-class solver
    /// assertions. Set `false` by `merge()`: a merged context's `assumed`
    /// pairs are export-only (the solver holds guarded `Or` disjunctions,
    /// not the unconditional pairs), so re-asserting them would over-constrain.
    /// Inherited parent→child on `fork`. Drives the `reassert_assumed` flag on
    /// the snapshot and whether `to_snapshot` dumps only the residual (cheap)
    /// or the full solver (merge fallback).
    pub(super) assume_class_reconstructible: AtomicBool,

    /// Local additions (assumed pairs + Z3 Bool cache) added after fork.
    /// Combined under one Mutex so the assume_*/add_constraint_raw hot path
    /// only acquires a single lock instead of two.
    /// `pub(super)` for the constraint-mutation methods in `constraint_ops.rs`
    /// (slice 9, angr-a2br.2.7).
    pub(super) local_constraints: Mutex<LocalConstraints>,

    // Z3-specific fields (when feature is enabled)
    /// Z3 solver — lazy: starts as None on fork(), materialized on first access.
    /// This avoids O(n) assertion replay for forked states that are
    /// pruned/avoided/deadended without ever querying the solver.
    /// `pub(super)` so `set_timeout` in `transaction_ops.rs` (slice 10,
    /// angr-a2br.2.8) can re-apply params without forcing lazy materialization.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) solver: Mutex<Option<z3::Solver>>,
    /// Cached SAT result, invalidated on constraint addition.
    /// `pub(super)` so the read-path query methods in `solving_ops.rs`
    /// (slice 7, angr-wf6f) can read/write it.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) sat_cache: Cell<Option<bool>>,
    /// Cached Z3 model, invalidated on constraint addition.
    /// `pub(super)` so the read-path query methods in `solving_ops.rs`
    /// (slice 7, angr-wf6f) can read/write it.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) model_cache: RefCell<Option<z3::Model>>,
    /// List of Z3 tracking boolean constants for unsat core mapping.
    /// Each entry is a (track_bool, constraint_ast) pair.
    /// `pub(super)` for `add_constraint_tracked_indexed` in `constraint_ops.rs`
    /// (slice 9, angr-a2br.2.7).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) constraint_trackers: Mutex<Vec<z3::ast::Bool>>,
    /// Z3 solver timeout in milliseconds (default: [`DEFAULT_SOLVER_TIMEOUT_MS`]).
    /// `pub(super)` for `set_timeout`/`timeout_ms` in `transaction_ops.rs`
    /// (slice 10, angr-a2br.2.8).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) timeout_ms: AtomicU32,

    /// Strict-deterministic witness selection (angr-op0dn.10.2, M2.2).
    ///
    /// Off by default; opt-in via [`set_deterministic`](Self::set_deterministic)
    /// because it trades Z3 checks for reproducibility (each witness costs an
    /// `O(log width)` binary search instead of one `get_model`). When on,
    /// `eval` / `eval_upto` return the unsigned-minimum witness and the
    /// ascending prefix of the feasible set rather than whatever model Z3
    /// happened to build — see `solving_ops.rs`. Inherited parent→child on
    /// `fork` so a whole lineage stays in the same mode.
    /// `pub(super)` for the query methods in `solving_ops.rs`.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) deterministic: AtomicBool,

    /// Shared-lineage Z3 solver (angr-v5a5 / angr-3ms1) — working, default-off.
    ///
    /// `None` for seed states and for every state in a lineage that never
    /// opted in, which is the production default. Minted by
    /// [`fork()`](Self::fork) when `use_shared_lineage_solver` is set (see
    /// that field for the three-gate materialization rule); otherwise `fork`
    /// propagates whatever Arc the parent already had. When `Some`, every
    /// state descended from that fork shares the same Arc; the inner Mutex
    /// serializes solver access across sibling states, and constraint
    /// installs route through `ScopeFrame` + `SharedLineageSolver::switch_to`
    /// (see `constraint_ops.rs`) instead of the per-context Z3 solver.
    ///
    /// **Mutex shape is load-bearing:**
    /// the outer `Mutex<Option<...>>` lets [`fork`](Self::fork) install a
    /// lineage through `&self` (fork's signature). The inner
    /// `Mutex<SharedLineageSolver>` serializes sibling-state queries
    /// against the shared Z3 solver. Both layers are required; do not
    /// collapse to `OnceLock` or `Option<Arc<...>>` without first changing
    /// the fork signature.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) lineage: Mutex<Option<Arc<Mutex<super::lineage::SharedLineageSolver>>>>,

    /// Per-state scope path: the ordered list of constraint frames this
    /// state has added since its lineage's base. Empty when `lineage` is
    /// `None` or when this state sits exactly at the lineage base.
    ///
    /// Mirrors the `local_constraints.z3_assertions` Vec in shape but
    /// stamps each entry with a globally-unique `FrameId` so sibling
    /// scope paths can share a prefix without RustBV-identity tricks (see
    /// the "FrameId, not pointer identity" rule on [`super::lineage`]).
    ///
    /// Stays empty whenever `lineage` is `None` — which is the production
    /// default, since the shared-lineage feature is opt-in. Once a lineage
    /// is installed, `assume_*` mints a frame here per constraint and routes
    /// the query through `SharedLineageSolver::switch_to`.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) scope_path: Mutex<super::lineage::ScopePath>,

    /// Per-state stack of saved `scope_path` lengths (angr-v5a5 slice 4b).
    ///
    /// Each entry is the value of `scope_path.len()` at the moment a
    /// matching `scope_savepoint_push()` was called. `scope_savepoint_pop()`
    /// truncates `scope_path` back to the most-recently-saved length.
    ///
    /// Only consulted on the `Some` (shared-lineage) dispatch branch —
    /// the `None` branch keeps using the per-context Z3 solver's native
    /// `push()/pop()`, so `scope_savepoints` stays empty unless a lineage
    /// is installed (the default, since the feature is opt-in). Under a
    /// lineage this stack is the per-state savepoint mechanism that lets
    /// the transactional plumbing (push/pop and transaction_*) coexist
    /// with the shared Z3 stack — bare Z3 push/pop on the shared solver
    /// would corrupt sibling state.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) scope_savepoints: Mutex<Vec<usize>>,

    /// Outstanding bare Z3 pushes on the per-context solver (angr-3ms1
    /// step 1a).
    ///
    /// Tracks calls to `scope_savepoint_push`
    /// that took the **None** dispatch branch — i.e. those that issued
    /// `solver.push()` directly on the per-context Z3 solver and have not
    /// yet been balanced by a matching pop. The **Some** branch records
    /// on `scope_savepoints` instead and leaves this counter alone, so
    /// with no lineage installed (the production default) the counter
    /// mirrors the per-context solver's push depth exactly.
    ///
    /// Exposed via [`bare_z3_push_depth`](Self::bare_z3_push_depth) for
    /// telemetry and read by the fork-time materialization gate in
    /// [`fork`](Self::fork), which refuses to mint a fresh
    /// `SharedLineageSolver` frame when this
    /// counter is non-zero: the child's lineage would otherwise steal
    /// ownership of the Z3 stack and the parent's unbalanced bare pushes
    /// would leak into the child's base (see bd memory
    /// `invariant-bare-z3-push-depth` for the failure mode this gates
    /// against).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) bare_z3_push_depth: AtomicUsize,

    /// Opt-in flag for fork-time `SharedLineageSolver` materialization
    /// (angr-3ms1 step 1b) — the master switch for the whole feature.
    ///
    /// When `true`, [`fork`](Self::fork) mints a fresh
    /// `SharedLineageSolver` on every fork, subject to two further gates:
    /// `bare_z3_push_depth == 0` (step 1a's correctness gate) and the
    /// lineage-dismantle detector not having fired. When `false` (the
    /// default), `fork()` propagates the parent's lineage Arc unchanged —
    /// `None` unless something opted in upstream, so no lineage is
    /// installed and the per-context solver path is used throughout.
    ///
    /// Set per-state via [`set_use_shared_lineage_solver`](Self::set_use_shared_lineage_solver)
    /// and inherited from parent to child in [`fork`](Self::fork) so a
    /// lineage opt-in on a seed state propagates to every descendant
    /// without per-fork plumbing on the Python side.
    ///
    /// Kept default-off because the v5a5 spike found that
    /// unconditional fork-time materialization regresses
    /// defcon2016quals_baby-re ~10x under default BFS exploration. The
    /// opt-in lets the slice-2 canary measure the lineage win on
    /// DFS/per-state-batched workloads without touching the default CI
    /// gate.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) use_shared_lineage_solver: AtomicBool,
}

impl SymContext {
    /// Create a new mock solver context (without Z3).
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new_mock() -> Self {
        SymContext {
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::new(HashMap::new()),
            push_level: AtomicUsize::new(0),
            assumed_constraints_shared: Mutex::new(Arc::new(Vec::new())),
            assume_class_reconstructible: AtomicBool::new(true),
            local_constraints: Mutex::new(LocalConstraints::new()),
        }
    }

    /// Alias for new_mock when Z3 is not available.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn new() -> Self {
        Self::new_mock()
    }

    /// Create a new solver context with Z3.
    ///
    /// With z3-rs 0.19+, the Z3 context is thread-local.
    /// All Z3 operations on this thread will use the same context.
    #[cfg(feature = "vex-engine-z3")]
    pub fn new() -> Self {
        Self::with_timeout(DEFAULT_SOLVER_TIMEOUT_MS)
    }

    /// Create a new solver context with Z3 and a custom timeout.
    // Arc<Vec<RustBV/z3::ast::Bool>> for assumed_constraints_shared / z3_assertions_shared:
    // the inner types are not Send/Sync (z3 AST handles, Python-backed RustBV), but the Arc
    // is correct — fork() shares the constraint prefix across sibling SymContexts via
    // Arc::clone for O(1) copy. The missing Send/Sync is not a real constraint because these
    // Arcs never cross a thread: SymContext is !Send by design and the only cross-thread
    // transport is StateMigrationPayload (state/migration.rs), which is Send-by-construction
    // and compile-time asserted. Switching to Rc would propagate non-Send through the
    // SymContext API surface. See `arc_shared` in lib.rs for the full rationale.
    #[cfg(feature = "vex-engine-z3")]
    pub fn with_timeout(timeout_ms: u32) -> Self {
        // unsat_core disabled for performance — tracking booleans add
        // significant overhead per constraint.
        let solver = build_solver(timeout_ms);

        SymContext {
            push_level: AtomicUsize::new(0),
            bare_local_savepoints: Mutex::new(Vec::new()),
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::new(HashMap::new()),
            assumed_constraints_shared: Mutex::new(crate::arc_shared(Vec::new())),
            z3_assertions_shared: Mutex::new(crate::arc_shared(Vec::new())),
            non_bv_assertions_shared: Mutex::new(crate::arc_shared(Vec::new())),
            assume_class_reconstructible: AtomicBool::new(true),
            local_constraints: Mutex::new(LocalConstraints::new()),
            solver: Mutex::new(Some(solver)),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(Vec::new()),
            timeout_ms: AtomicU32::new(timeout_ms),
            deterministic: AtomicBool::new(false),
            lineage: Mutex::new(None),
            scope_path: Mutex::new(super::lineage::ScopePath::new()),
            scope_savepoints: Mutex::new(Vec::new()),
            bare_z3_push_depth: AtomicUsize::new(0),
            use_shared_lineage_solver: AtomicBool::new(false),
        }
    }

    /// Create a mock context for testing (when Z3 is enabled but not needed).
    #[cfg(feature = "vex-engine-z3")]
    pub fn new_mock() -> Self {
        Self::new()
    }

    /// Get or lazily create the Z3 solver.
    ///
    /// Forked contexts start with `solver = None` to avoid O(n) assertion
    /// replay for states that are pruned/avoided without querying the solver.
    /// On first access, a fresh solver is created and cached assertions are
    /// replayed.
    #[cfg(feature = "vex-engine-z3")]
    #[allow(
        clippy::expect_used,
        reason = "`guard` is `Some` here by construction: the block directly above stores `*guard = Some(new_solver)` on the `None` path and the guard is held across both, so no other thread can clear it in between. `MutexGuard::map` must yield a `&mut Solver`, so there is no `Option` return to widen into"
    )]
    pub(super) fn solver(&self) -> parking_lot::MappedMutexGuard<'_, z3::Solver> {
        let mut guard = self.solver.lock();
        if guard.is_none() {
            let start = std::time::Instant::now();
            let new_solver = build_solver(self.timeout_ms.load(Ordering::SeqCst));

            // Replay cached Z3 assertions: shared prefix then local additions
            let shared = Arc::clone(&self.z3_assertions_shared.lock());
            for constraint in shared.iter() {
                new_solver.assert(constraint);
            }
            let local = self.local_constraints.lock();
            for constraint in local.z3_assertions.iter() {
                new_solver.assert(constraint);
            }

            *guard = Some(new_solver);
            Z3_MATERIALIZE_COUNT.fetch_add(1, Ordering::Relaxed);
            Z3_MATERIALIZE_TIME_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        parking_lot::MutexGuard::map(guard, |opt| {
            opt.as_mut()
                .expect("solver was just initialized in the None branch above")
        })
    }

    // next_id / num_constraints / new_bv / unique_name moved to bv_id_ops.rs
    // (slice 8, angr-a2br.2.6).

    // =========================================================================
    // Constraint Management (Z3-backed)
    // =========================================================================

    /// Push an entry to assumed_constraints for export tracking.
    /// Used by the fast path that bypasses assume_true.
    #[cfg(feature = "vex-engine-z3")]
    pub fn assumed_constraints_push(&self, bv: RustBV, is_true: bool) {
        self.local_constraints.lock().assumed.push((bv, is_true));
    }

    /// Current length of the *local* assumed-constraints export log.
    ///
    /// Paired with [`Self::truncate_assumed_local`] to bracket transient
    /// `assume_true`/`assume_false` calls whose Z3 assertions live inside a
    /// bare `push()`/`pop()` scope but whose export-log entries must NOT
    /// persist. The plain `pop()` restores the Z3 solver frame but this
    /// explicit truncation is what clears `local.assumed`, so
    /// feasibility-check assumes would otherwise leak into the log that
    /// `to_snapshot`/`restore_from_snapshot` faithfully re-assert on a wave
    /// migration — the ype54 concrete-guard poison (angr-ype54).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assumed_local_len(&self) -> usize {
        self.local_constraints.lock().assumed.len()
    }

    /// Truncate the *local* assumed-constraints export log back to `len`,
    /// discarding transient entries appended since an
    /// [`Self::assumed_local_len`] savepoint. Does not touch the Z3 solver
    /// (the enclosing `pop()` owns that) — only the export/re-assert log.
    #[cfg(feature = "vex-engine-z3")]
    pub fn truncate_assumed_local(&self, len: usize) {
        let mut local = self.local_constraints.lock();
        if len < local.assumed.len() {
            local.assumed.truncate(len);
        }
    }

    /// Export all Z3 assertion pointers from the assertion cache.
    /// Uses z3_assertions_shared + the local z3_assertions vector which track
    /// every assertion made via assume_true/assume_false/add_constraint_raw.
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_assertion_ptrs(&self) -> Vec<usize> {
        use z3::ast::Ast;
        let mut ptrs = Vec::new();
        let shared = Arc::clone(&self.z3_assertions_shared.lock());
        for constraint in shared.iter() {
            ptrs.push(constraint.get_z3_ast().as_ptr() as usize);
        }
        let local = self.local_constraints.lock();
        for constraint in local.z3_assertions.iter() {
            ptrs.push(constraint.get_z3_ast().as_ptr() as usize);
        }
        ptrs
    }

    /// Debug: get the Z3 solver's internal push level.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_push_level(&self) -> usize {
        self.push_level.load(std::sync::atomic::Ordering::SeqCst)
    }

    // Constraint-mutation &self methods (add_constraint / add_constraint_raw /
    // add_constraints_raw_batch / add_constraint_tracked_indexed /
    // add_bv_constraint / assume_true / assume_false, plus the dedup helpers
    // seed_and_check_z3_dedup / check_z3_dedup_if_seeded and the model-cache
    // invalidators) moved to constraint_ops.rs (slice 9, angr-a2br.2.7).

    // Scoping &self methods (set_timeout / timeout_ms / set_sat_cache /
    // push / pop / try_pop / unsat_core / get_all_constraints_str /
    // z3_assertion_count, plus their non-Z3 mock variants) moved to
    // transaction_ops.rs (slice 10, angr-a2br.2.8). The transaction_begin/
    // commit/rollback lifecycle was removed in angr-ph300.44.

    // =========================================================================
    // Mock implementations when Z3 is not available
    // =========================================================================

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn assume_true(&self, cond: &RustBV) {
        self.assume(cond, true);
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn assume_false(&self, cond: &RustBV) {
        self.assume(cond, false);
    }

    /// Non-Z3 mirror of `constraint_ops.rs::assume` (angr-12jjk.11): the two
    /// polarities share one body here too, so a change to the export contract
    /// can't land in one twin and miss the other.
    #[cfg(not(feature = "vex-engine-z3"))]
    fn assume(&self, cond: &RustBV, want_true: bool) {
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python; no Z3 to assert against.
        self.local_constraints
            .lock()
            .assumed
            .push((cond.clone(), want_true));
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn check_branch_feasibility(&self, cond: &RustBV) -> (bool, bool) {
        debug_assert_eq!(cond.width(), 1);
        if let Some(v) = cond.as_u128() {
            return (v != 0, v == 0);
        }
        // Without Z3, assume both directions are feasible — matches the
        // can_be_true/can_be_false stubs.
        (true, true)
    }

    // =========================================================================
    // Constraint Export
    // =========================================================================

    /// Get the assumed constraints as (RustBV, is_assumed_true) pairs.
    ///
    /// This exports the tracked path constraints that can be converted to claripy
    /// ASTs for Python constraint sync. Each constraint is a 1-bit RustBV that was
    /// either assumed true or false during symbolic execution.
    pub fn get_assumed_constraints(&self) -> Vec<(RustBV, bool)> {
        let shared = Arc::clone(&self.assumed_constraints_shared.lock());
        let local = self.local_constraints.lock();
        let mut out = Vec::with_capacity(shared.len() + local.assumed.len());
        out.extend_from_slice(&shared);
        out.extend_from_slice(&local.assumed);
        out
    }

    /// Get the number of assumed constraints.
    pub fn assumed_constraint_count(&self) -> usize {
        self.assumed_constraints_shared.lock().len() + self.local_constraints.lock().assumed.len()
    }
}

impl Clone for SymContext {
    fn clone(&self) -> Self {
        self.fork()
    }
}

// =============================================================================
// Fork freeze helpers
// =============================================================================

/// Freeze a local additions vector into the shared `Arc<Vec<T>>`.
///
/// Outside an open push/pop scope (when `in_transaction` is false) this drains
/// `local` into `shared` in place — when shared has unique ownership the move
/// avoids the per-element clones (e.g. each `z3::ast::Bool::clone` is a
/// `Z3_inc_ref` FFI call). When `in_transaction` is true we must preserve
/// `local` so a later `pop()` can truncate it; in that case we fall back to
/// allocating a fresh Vec by cloning shared and copying local's elements.
///
/// (Callers derive `in_transaction` from `push_level > 0`, which is always 0
/// since the transaction API was removed in angr-ph300.44 — so the copy branch
/// is currently unreachable, but retained as generic fork infrastructure.)
pub(super) fn freeze_into_shared<T: Clone>(
    shared: &Mutex<Arc<Vec<T>>>,
    local: &mut Vec<T>,
    in_transaction: bool,
) -> Arc<Vec<T>> {
    if local.is_empty() {
        return Arc::clone(&shared.lock());
    }
    let mut shared_guard = shared.lock();
    if in_transaction {
        // Cannot mutate local — rollback expects it intact.
        let mut merged = Vec::with_capacity(shared_guard.len() + local.len());
        merged.extend_from_slice(&shared_guard);
        merged.extend_from_slice(local);
        return Arc::new(merged);
    }
    if let Some(inner) = Arc::get_mut(&mut *shared_guard) {
        // Unique ownership: in-place append, no element clones either side.
        inner.reserve(local.len());
        inner.append(local);
    } else {
        // Aliased: allocate new Vec, but move local's elements (no local clones).
        let mut merged = Vec::with_capacity(shared_guard.len() + local.len());
        merged.extend_from_slice(&shared_guard);
        merged.append(local);
        *shared_guard = Arc::new(merged);
    }
    Arc::clone(&shared_guard)
}

impl Default for SymContext {
    fn default() -> Self {
        Self::new()
    }
}

// a2br.2.11: SymContext unit tests, split by theme out of the former
// monolithic `context_tests.rs` (2340 lines) to keep each file under the
// <2000-line epic acceptance criterion. Declared as direct children of
// `context` so `use super::*` reaches `context`'s private items.
/// Test-only instrumentation for `SymContext::merge`'s guarded-assertion count
/// (angr-op0dn.11.3). The production merge increments this for every guarded
/// `Or(!cond, c)` it emits plus the final `Or` of merge flags. On the
/// shared-prefix (CoW) path the guarded count collapses to the divergent
/// (local) constraint count + 1; on the fallback path it stays the total
/// constraint count + 1. Tests reset it, run a merge, and assert the count.
#[cfg(test)]
pub(crate) mod merge_instrument {
    use std::cell::Cell;

    thread_local! {
        static GUARDED_EMITTED: Cell<u64> = const { Cell::new(0) };
    }

    /// Called from `SymContext::merge` for each guarded `Or` (and the flag `Or`).
    #[inline]
    pub(crate) fn note_guarded() {
        GUARDED_EMITTED.with(|c| c.set(c.get() + 1));
    }

    /// Reset the counter to zero before a measured merge.
    pub(crate) fn reset() {
        GUARDED_EMITTED.with(|c| c.set(0));
    }

    /// Read the number of guarded `Or`s emitted since the last [`reset`].
    pub(crate) fn emitted() -> u64 {
        GUARDED_EMITTED.with(|c| c.get())
    }
}

// Gated on vex-engine-z3 (bd angr-cagbn): every test here drives
// `SymContext::add_constraint` / Z3AstPtr, which only exist with z3. Keeps the
// no-z3 nightly `cargo test` combos compiling; default build runs them all.
test_submod!(z3 "context_tests/constraints.rs" => context_tests_constraints);
test_submod!("context_tests/lineage.rs" => context_tests_lineage);
test_submod!("context_tests/merge_prefix.rs" => context_tests_merge_prefix);
test_submod!("context_tests/merge_shape_spike.rs" => context_tests_merge_shape_spike);
test_submod!("context_tests/smtlib2_snapshot.rs" => context_tests_smtlib2_snapshot);
test_submod!("context_tests/solver.rs" => context_tests_solver);
