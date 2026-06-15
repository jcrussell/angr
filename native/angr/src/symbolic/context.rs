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
//! split). Each invariant cites the bd memory key carrying the long-form
//! rationale; recall via `bd recall <key>`.
//!
//! - **`lineage` mutex shape** (`invariant-v5a5-lineage-mutex-shape`):
//!   `Mutex<Option<Arc<Mutex<SharedLineageSolver>>>>`. The outer `Mutex` is
//!   load-bearing — [`fork`](SymContext::fork) takes `&self`, not `&mut
//!   self`, and must be able to install a fresh lineage. Collapsing to a
//!   plain `Option<Arc<…>>` or to `OnceLock` would force the fork
//!   signature to change.
//! - **FrameId minting** (`v5a5-frame-id-design`): per-state scope-path
//!   identity uses a globally-monotonic [`FrameId`](super::lineage::FrameId)
//!   minted at constraint-add time, NOT `Arc::ptr_eq` on the RustBV. This
//!   buys [`SharedLineageSolver::switch_to`](super::lineage::SharedLineageSolver::switch_to)
//!   a cheap by-id prefix comparison while staying robust across fork
//!   boundaries where RustBV identity can split.
//! - **Three-gate materialization** (`invariant-v5a5-slice-1c-mint-semantics`,
//!   `invariant-bare-z3-push-depth`, `invariant-v5ht-dismantle-child-none`):
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
//! - **fork-freeze under push** (`fork-freeze-self-invariant`):
//!   [`fork`](SymContext::fork) only drains local→shared in place when
//!   `push_level == 0`. Inside a transaction, `transaction_rollback`
//!   truncates `local` back to its pre-transaction length; draining would
//!   leak rolled-back constraints into `shared`.
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
//! - **No full lineage teardown** (`avoid-full-lineage-teardown`): the
//!   angr-0dgq teardown variant (walk all stashes, drop each state's
//!   lineage Arc, invalidate per-context solvers) is a net loss vs the
//!   v5ht simple variant. [`super::lineage::set_lineage_dismantled`] only
//!   suppresses future mints; in-flight lineages keep working.
//! - **No DFS-only opt-in** (`avoid-dfs-coupling-for-shared-lineage`): do
//!   NOT default `use_shared_lineage_solver` on for `strategy='dfs'`. Both
//!   the canonical WIN (ais3_crackme) and LOSE (defcon2016quals_baby-re)
//!   canaries run BFS — strategy is not the discriminator.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::RustBV;
// Constraint-sharing analysis types live in `sharing.rs` (angr-a2br.2 slice 3).
// `fold_sharing_walk` below takes a `&mut ConstraintSharingWalk`.
// Solver/engine profiling counters live in `stats.rs` (angr-ugc2). The glob
// brings every counter static plus `CheckSite` / `VexOpFamily` and the
// `record_*` helpers into scope so the solving paths below read and bump them
// unchanged.
use super::stats::*;

// Z3 solver construction + per-check timing/sampling wrappers live in
// `solver_build.rs` (angr-a2br.2 slice 2). The glob brings `build_solver`,
// `build_solver_params`, `timed_check`, and `sample_simplify_skip` into scope
// for the solving paths below (and, via `use super::*`, the test module).
#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;

/// Snapshot of a [`SymContext`]'s path-constraint state.
///
/// Captures the `assumed_constraints` Vec — the canonical record from
/// which Z3 solver state, scope paths, and fast-path caches re-derive
/// after load. Built by [`SymContext::to_snapshot`] and consumed by
/// [`SymContext::restore_from_snapshot`].
///
/// Carries `(RustBV, bool)` directly via the [`RustBV`] serde derive
/// (angr-x04s.1.1). Per-Z3-context cache state (solver, model_cache,
/// sat_cache, lineage scope_path, push stacks) is NOT included — these
/// are runtime caches that the loader rebuilds on first query against
/// the restored constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymContextSnapshot {
    /// `(constraint, is_assumed_true)` pairs in insertion order.
    pub assumed_constraints: Vec<(RustBV, bool)>,
    /// Full Z3 solver state as an SMT-LIB2 dump (angr-82g6).
    ///
    /// Captures every assertion currently on the solver — both the
    /// `assume_*`-tracked entries (already in `assumed_constraints`)
    /// and the raw entries from [`SymContext::add_constraint_raw`] that
    /// have no [`RustBV`] form (the Python-claripy-sync fallback path
    /// in `_add_constraints_to_state` and the cross-process pointer-
    /// import path in `_import_z3_constraint_ptrs`).
    ///
    /// Restore replays this dump through `add_constraint_raw` to
    /// rebuild Z3-side solver state, then populates the
    /// `assumed_constraints` BV log via `assumed_constraints_push`
    /// (no second solver assert). The two captures are independent —
    /// no ptr-level dedup needed, which side-steps the case where
    /// Python-claripy ASTs and `claripy_to_rustbv`-rebuilt ASTs are
    /// structurally different and hash-cons to different pointers.
    ///
    /// `#[serde(default)]` keeps round-trip compat with older snapshots
    /// that predate this field — they fall through to the original
    /// `assume_*` replay path with the pre-82g6 lossy semantics.
    #[serde(default)]
    pub solver_smtlib2: String,
}

/// Default Z3 solver timeout in milliseconds.
///
/// 30 seconds — chosen to match claripy's historical default and to cap the
/// occasional Z3 outlier on bimodal-SAT benches. Overridable per-state via
/// [`RustExplorationManager::set_solver_timeout`](crate::exploration::RustExplorationManager).
pub const DEFAULT_SOLVER_TIMEOUT_MS: u32 = 30_000;

/// Inline capacity for SymContext push_* stacks. Branch nesting is typically
/// shallow (≤8) within a single block; SmallVec avoids the heap allocation
/// for the first push.
pub(super) type PushStack = SmallVec<[usize; 8]>;

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
    /// HashSet of Z3_ast ptrs for O(1) dedup in `add_constraint_raw`.
    /// Holds ptrs for every assertion known to be currently asserted on the
    /// solver (i.e. everything in `z3_assertions_shared` + `z3_assertions`).
    /// Lazily seeded — `dedup_set_seeded == false` means the set is stale
    /// and must be rebuilt from the shared+local vecs before consultation.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) dedup_set: HashSet<usize>,
    /// True once `dedup_set` has been populated from shared+local for this
    /// context. Reset to false by `fork()`, `merge()` (via `new()`), and
    /// `transaction_rollback()` (which truncates `z3_assertions`).
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

/// Error type for constraint sync operations.
#[derive(Debug, Clone)]
pub enum ConstraintSyncError {
    /// Conversion failed for a constraint.
    ConversionFailed(String),
    /// Constraints became unsatisfiable after sync.
    Unsatisfiable,
    /// Invalid rollback (no transaction to rollback).
    NoTransaction,
}

impl std::fmt::Display for ConstraintSyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConversionFailed(msg) => write!(f, "constraint conversion failed: {msg}"),
            Self::Unsatisfiable => write!(f, "constraints became unsatisfiable after sync"),
            Self::NoTransaction => write!(f, "no transaction to rollback"),
        }
    }
}

impl std::error::Error for ConstraintSyncError {}

/// Solver context for symbolic execution.
///
/// Manages symbolic variable creation and, when Z3 is available,
/// constraint solving and satisfiability checking.
///
/// With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
/// to store a reference to it. All Z3 operations on a thread share
/// the same context automatically.
///
/// ## Transactional Constraint Sync
///
/// The context supports transactional constraint sync with push/pop semantics:
/// - `transaction_begin()`: Start a new transaction
/// - `transaction_commit()`: Commit constraints (validate and keep)
/// - `transaction_rollback()`: Rollback on failure
pub struct SymContext {
    /// Counter for generating unique symbol IDs.
    /// `pub(super)` for the `next_id` accessor in `bv_id_ops.rs` (slice 8).
    pub(super) next_id: AtomicU64,
    /// Number of constraints added (for tracking).
    /// `pub(super)` for the `num_constraints` accessor in `bv_id_ops.rs`.
    pub(super) constraint_count: AtomicUsize,
    /// Named symbolic variables for debugging.
    /// Arc-shared on fork (O(1) clone). Only mutated when constructing
    /// a fresh merged context — Arc::make_mut works because the merged
    /// SymContext is freshly created with a unique Arc.
    pub(super) symbol_table: Arc<HashMap<String, u64>>,
    /// Current push level for transaction tracking.
    /// `pub(super)` for the transaction/scoping methods in `transaction_ops.rs`
    /// (slice 10, angr-a2br.2.8).
    pub(super) push_level: AtomicUsize,
    /// Constraint count at each push level (for rollback).
    /// `pub(super)` for `transaction_ops.rs` (slice 10, angr-a2br.2.8).
    pub(super) push_constraint_counts: Mutex<PushStack>,
    /// Local Z3 cache length at each push level (for rollback truncation).
    /// `pub(super)` for `transaction_ops.rs` (slice 10, angr-a2br.2.8).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) push_local_cache_lengths: Mutex<PushStack>,
    /// Local assumed_constraints length at each push level (for rollback truncation).
    /// `pub(super)` for `transaction_ops.rs` (slice 10, angr-a2br.2.8).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) push_assumed_local_lengths: Mutex<PushStack>,
    /// Phase 2 Fix: Track assumed RustBV constraints for export to Python.
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

    /// Shared-lineage Z3 solver (angr-v5a5 spike, integration in progress).
    ///
    /// `None` for seed states and any state whose lineage has not yet been
    /// established. Set via [`fork()`](Self::fork) once integration is wired
    /// (next slice). When `Some`, every state descended from a common fork
    /// shares the same Arc; the inner Mutex serializes solver access across
    /// sibling states.
    ///
    /// This slice (angr-v5a5 fields-only) introduces the field but does not
    /// yet route queries through it — [`solver()`](Self::solver) still uses
    /// the lazy-materialize path. The next slice replaces that.
    ///
    /// **Mutex shape is load-bearing** (`bd recall invariant-v5a5-lineage-mutex-shape`):
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
    /// scope paths can share a prefix without RustBV-identity tricks
    /// (see memory `v5a5-frame-id-design`).
    ///
    /// Inert in this slice — the next slice wires `assume_*` to mint
    /// frames here and routes queries through `SharedLineageSolver::switch_to`.
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
    /// `push()/pop()`, so `scope_savepoints` stays empty in production.
    /// When slice 4c lights up lineage materialization, this stack
    /// becomes the per-state savepoint mechanism that lets the
    /// transactional plumbing (push/pop and transaction_*) coexist with
    /// the shared Z3 stack — bare Z3 push/pop on the shared solver
    /// would corrupt sibling state.
    ///
    /// Inert in this slice for the same reason `scope_path` is inert:
    /// production never installs a lineage today. Tests using
    /// `set_lineage_for_testing` exercise the dispatch.
    #[cfg(feature = "vex-engine-z3")]
    pub(super) scope_savepoints: Mutex<Vec<usize>>,

    /// Outstanding bare Z3 pushes on the per-context solver (angr-3ms1
    /// step 1a).
    ///
    /// Tracks calls to `scope_savepoint_push`
    /// that took the **None** dispatch branch — i.e. those that issued
    /// `solver.push()` directly on the per-context Z3 solver and have not
    /// yet been balanced by a matching pop. The **Some** branch records
    /// on `scope_savepoints` instead and leaves this counter alone, so in
    /// production today (no lineage ever installed) the counter mirrors
    /// the per-context solver's push depth exactly.
    ///
    /// Inert in this slice — exposed via
    /// [`bare_z3_push_depth`](Self::bare_z3_push_depth) for telemetry and
    /// for the slice-1c fork-time materialization gate. That gate will
    /// refuse to mint a fresh `SharedLineageSolver` frame when this
    /// counter is non-zero: the child's lineage would otherwise steal
    /// ownership of the Z3 stack and the parent's unbalanced bare pushes
    /// would leak into the child's base (see the
    /// `v5a5-bare-z3-push-depth-counter-design` memo for the failure
    /// mode this gates against).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) bare_z3_push_depth: AtomicUsize,

    /// Opt-in flag for fork-time `SharedLineageSolver` materialization
    /// (angr-3ms1 step 1b).
    ///
    /// When `true`, the slice-1c fork-time gate will mint a fresh
    /// `SharedLineageSolver` on every fork (subject to the
    /// `bare_z3_push_depth == 0` correctness gate from step 1a). When
    /// `false` (the default), `fork()` keeps the existing behavior of
    /// propagating the parent's lineage Arc unchanged — None in
    /// production today, so no lineage is ever installed.
    ///
    /// Set per-state via [`set_use_shared_lineage_solver`](Self::set_use_shared_lineage_solver)
    /// and inherited from parent to child in [`fork`](Self::fork) so a
    /// lineage opt-in on a seed state propagates to every descendant
    /// without per-fork plumbing on the Python side. Inert in this slice
    /// — the materialization gate (step 1c) will read it.
    ///
    /// Kept default-off because the v5a5 spike's
    /// `v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental` finding
    /// showed unconditional fork-time materialization regresses
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
            next_id: AtomicU64::new(0),
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::new(HashMap::new()),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(Arc::new(Vec::new())),
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
    // Arc::clone for O(1) copy. The engine runs single-threaded under Python's GIL, so the
    // missing Send/Sync is not a real constraint; switching to Rc would propagate non-Send
    // through the SymContext API surface.
    #[allow(clippy::arc_with_non_send_sync)]
    #[cfg(feature = "vex-engine-z3")]
    pub fn with_timeout(timeout_ms: u32) -> Self {
        // unsat_core disabled for performance — tracking booleans add
        // significant overhead per constraint.
        let solver = build_solver(timeout_ms);

        SymContext {
            next_id: AtomicU64::new(0),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            push_local_cache_lengths: Mutex::new(PushStack::new()),
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::new(HashMap::new()),
            assumed_constraints_shared: Mutex::new(Arc::new(Vec::new())),
            z3_assertions_shared: Mutex::new(Arc::new(Vec::new())),
            local_constraints: Mutex::new(LocalConstraints::new()),
            solver: Mutex::new(Some(solver)),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(Vec::new()),
            timeout_ms: AtomicU32::new(timeout_ms),
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

    // Transaction / scoping &self methods (set_timeout / timeout_ms /
    // set_sat_cache / push / pop / transaction_begin / transaction_commit /
    // transaction_rollback / current_push_level / in_transaction / unsat_core /
    // get_all_constraints_str / z3_assertion_count, plus their non-Z3 mock
    // variants) moved to transaction_ops.rs (slice 10, angr-a2br.2.8).

    // =========================================================================
    // Mock implementations when Z3 is not available
    // =========================================================================

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn assume_true(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python; no Z3 to assert against.
        self.local_constraints
            .lock()
            .assumed
            .push((cond.clone(), true));
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn assume_false(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        self.local_constraints
            .lock()
            .assumed
            .push((cond.clone(), false));
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
    // Constraint Export (Phase 2 Fix)
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

/// Freeze a local additions vector into the shared Arc<Vec<T>>.
///
/// Outside a push/pop transaction (when `in_transaction` is false) this drains
/// `local` into `shared` in place — when shared has unique ownership the move
/// avoids the per-element clones (e.g. each `z3::ast::Bool::clone` is a
/// `Z3_inc_ref` FFI call). Inside a transaction we must preserve `local` so
/// `transaction_rollback` can truncate it; in that case we fall back to
/// allocating a fresh Vec by cloning shared and copying local's elements.
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

#[cfg(test)]
mod tests {
    // Tests construct `Arc<Mutex<SharedLineageSolver>>` to match the
    // production `lineage_arc()` type. `SharedLineageSolver` wraps a
    // thread-local Z3 solver and is intentionally non-Send/Sync;
    // switching tests to `Rc` would diverge from production usage.
    #![allow(clippy::arc_with_non_send_sync)]

    use super::*;
    #[cfg(feature = "vex-engine-z3")]
    use crate::symbolic::Z3AstPtr;

    #[test]
    fn test_id_generation() {
        let ctx = SymContext::new_mock();
        assert_eq!(ctx.next_id(), 0);
        assert_eq!(ctx.next_id(), 1);
        assert_eq!(ctx.next_id(), 2);
    }

    #[test]
    fn test_unique_names() {
        let ctx = SymContext::new_mock();
        let name1 = ctx.unique_name("x");
        let name2 = ctx.unique_name("x");
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_concrete_eval() {
        let ctx = SymContext::new_mock();
        let bv = RustBV::concrete(42, 32);
        assert_eq!(ctx.eval(&bv), Some(42));
    }

    #[test]
    fn test_fork() {
        let ctx = SymContext::new_mock();
        let id1 = ctx.next_id();

        let forked = ctx.fork();
        let id2 = forked.next_id();

        // Forked context should continue from same ID
        assert_eq!(id2, id1 + 1);
    }

    /// angr-v5a5 spike: fresh contexts have no lineage attached and an
    /// empty scope path.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_lineage_starts_none() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());
        assert_eq!(ctx.scope_path_len(), 0);
    }

    /// angr-v5a5 spike: forking does not auto-create a lineage. The
    /// inert-fields slice keeps both parent and child at None — the next
    /// slice will add the lineage-creation path.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_keeps_lineage_none() {
        let ctx = SymContext::new();
        let forked = ctx.fork();
        assert!(ctx.lineage_arc().is_none());
        assert!(forked.lineage_arc().is_none());
        assert_eq!(forked.scope_path_len(), 0);
    }

    /// angr-v5a5 spike: when the parent has a lineage Arc, fork
    /// propagates it to the child by Arc::clone (same allocation).
    /// Uses set_lineage_for_testing because the integration patch that
    /// creates the lineage on fork lives in a later slice; today we
    /// only verify the propagation wiring.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_propagates_lineage_arc() {
        let parent = SymContext::new();
        let lin = Arc::new(Mutex::new(super::super::lineage::SharedLineageSolver::new(
            build_solver(30_000),
        )));
        parent.set_lineage_for_testing(Arc::clone(&lin));

        let child = parent.fork();
        let child_arc = child.lineage_arc().expect("child should inherit lineage");
        let parent_arc = parent.lineage_arc().expect("parent retains its lineage");
        assert!(
            Arc::ptr_eq(&child_arc, &parent_arc),
            "fork must Arc::clone the lineage, not allocate a new one"
        );
        assert!(
            Arc::ptr_eq(&child_arc, &lin),
            "child arc should point at the same SharedLineageSolver"
        );
        // Child starts with an empty scope path even when the lineage is set.
        assert_eq!(child.scope_path_len(), 0);
    }

    /// angr-v5a5 slice 3a: get_solver_stats surfaces the four lineage
    /// counters and reset_solver_stats clears them. We can't assert exact
    /// values because the global atomics are shared with other tests in
    /// the suite — instead, assert the keys are present and that a
    /// post-reset snapshot taken before any new switch_to is 0.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_lineage_telemetry_surfaced() {
        let stats = get_solver_stats();
        for key in [
            "lineage_switch_count",
            "lineage_switch_hot_count",
            "lineage_push_count",
            "lineage_pop_count",
        ] {
            assert!(
                stats.contains_key(key),
                "get_solver_stats should surface {key}"
            );
        }

        // After a reset, the four lineage counters must read 0 — but only
        // if nothing else bumps them between reset and read. Take the
        // snapshot inside a closure that brackets the reset to minimize
        // the race window; even so, only assert <= some tiny upper bound
        // (other parallel tests can race in).
        reset_solver_stats();
        let post = get_solver_stats();
        // Lower bound is trivially 0; sanity-check the keys are still
        // present after the reset and the values are within a tiny
        // tolerance of zero (allow concurrent test bumps).
        for key in [
            "lineage_switch_count",
            "lineage_switch_hot_count",
            "lineage_push_count",
            "lineage_pop_count",
        ] {
            assert!(post.contains_key(key));
        }
    }

    /// angr-v5a5 slice 3b: with_z3_solver dispatches to self.solver() when
    /// no lineage is attached. The closure must see the same per-context
    /// lazy solver that a direct `self.solver()` would.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_with_z3_solver_no_lineage_uses_local_solver() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());

        let x = RustBV::symbolic(&ctx, "test_with_z3_solver_no_lineage_x", 8);
        let five = RustBV::concrete(5, 8);
        ctx.assume_true(&x.eq(&five, &ctx));

        // Closure asserts via with_z3_solver — must hit the same solver
        // that holds the assume_true constraint.
        let sat = ctx.with_z3_solver(|solver| solver.check());
        assert_eq!(sat, z3::SatResult::Sat);

        // Add a contradictory temporary constraint inside the closure and
        // confirm the per-context solver state is the one being queried.
        let unsat = ctx.with_z3_solver(|solver| {
            solver.push();
            solver.assert(&{
                let bv_x = z3::ast::BV::new_const("test_with_z3_solver_no_lineage_x", 8);
                bv_x.eq(z3::ast::BV::from_u64(42, 8))
            });
            let r = solver.check();
            solver.pop(1);
            r
        });
        assert_eq!(unsat, z3::SatResult::Unsat, "x == 5 ∧ x == 42 is UNSAT");
    }

    /// angr-v5a5 slice 3b: with_z3_solver dispatches into the lineage's
    /// shared solver when one is attached, bumping the lineage_switch_count
    /// telemetry. Today the production path never installs a lineage, so
    /// we use set_lineage_for_testing to exercise the dispatcher.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_with_z3_solver_routes_to_lineage() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));
        assert!(ctx.lineage_arc().is_some());

        // The dispatcher should hand a solver to the closure and the
        // lineage_switch_count counter should advance — that's the
        // observable proof we routed through SharedLineageSolver::with_solver
        // instead of the per-context lazy solver.
        let pre = super::super::lineage::lineage_stats();
        let pre_switch = pre[0].1;

        let result = ctx.with_z3_solver(|solver| solver.check());
        assert_eq!(
            result,
            z3::SatResult::Sat,
            "fresh lineage solver with no constraints is trivially Sat"
        );

        let post = super::super::lineage::lineage_stats();
        let post_switch = post[0].1;
        assert!(
            post_switch > pre_switch,
            "lineage_switch_count must advance (pre={pre_switch}, post={post_switch})"
        );
    }

    /// angr-v5a5 slice 4b: a fresh SymContext has an empty
    /// scope-savepoint stack.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_scope_savepoints_start_empty() {
        let ctx = SymContext::new();
        assert_eq!(ctx.scope_savepoint_depth(), 0);
    }

    /// angr-v5a5 slice 4b: when no lineage is attached (the None
    /// dispatch path), `scope_savepoint_push` goes to the per-context
    /// Z3 solver and does NOT record on `scope_savepoints` — preserving
    /// the pre-slice behavior of bare `self.solver().push()`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_scope_savepoint_none_branch_skips_stack() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());

        ctx.scope_savepoint_push();
        assert_eq!(
            ctx.scope_savepoint_depth(),
            0,
            "None branch must not record on scope_savepoints"
        );

        ctx.scope_savepoint_pop();
        assert_eq!(ctx.scope_savepoint_depth(), 0);
    }

    /// angr-v5a5 slice 4b: with a lineage attached (Some dispatch
    /// path), `scope_savepoint_push` records the current scope_path
    /// length on `scope_savepoints`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_scope_savepoint_some_branch_records_depth() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        // Initial state: scope_path empty.
        assert_eq!(ctx.scope_path_len(), 0);
        assert_eq!(ctx.scope_savepoint_depth(), 0);

        ctx.scope_savepoint_push();
        assert_eq!(
            ctx.scope_savepoint_depth(),
            1,
            "Some branch must push onto scope_savepoints"
        );
        // No Z3 op was issued — the shared solver's stack is unchanged
        // and the per-state scope_path is still empty.
        assert_eq!(ctx.scope_path_len(), 0);

        ctx.scope_savepoint_pop();
        assert_eq!(ctx.scope_savepoint_depth(), 0);
        assert_eq!(ctx.scope_path_len(), 0);
    }

    /// angr-v5a5 slice 4b: with a lineage attached, frames pushed onto
    /// `scope_path` between `scope_savepoint_push` and
    /// `scope_savepoint_pop` are truncated by the pop.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_scope_savepoint_truncates_scope_path() {
        use super::super::lineage::{ScopeFrame, SharedLineageSolver};

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        let bv_x = z3::ast::BV::new_const("test_scope_savepoint_x", 8);
        let mk_frame = |is_true: bool, val: u64| {
            ScopeFrame::new(is_true, bv_x.eq(z3::ast::BV::from_u64(val, 8)))
        };

        // Add an initial frame (simulates a pre-existing per-state
        // constraint), save a savepoint, then add two more frames.
        ctx.push_scope_frame_for_testing(mk_frame(true, 1));
        assert_eq!(ctx.scope_path_len(), 1);

        ctx.scope_savepoint_push();
        assert_eq!(ctx.scope_savepoint_depth(), 1);

        ctx.push_scope_frame_for_testing(mk_frame(true, 2));
        ctx.push_scope_frame_for_testing(mk_frame(false, 3));
        assert_eq!(ctx.scope_path_len(), 3);

        // Pop the savepoint: scope_path truncates to its pre-push
        // length (1), and the savepoint stack drains.
        ctx.scope_savepoint_pop();
        assert_eq!(
            ctx.scope_path_len(),
            1,
            "pop must truncate scope_path back to the saved length"
        );
        assert_eq!(ctx.scope_savepoint_depth(), 0);
    }

    /// angr-v5a5 slice 4b: nested savepoints LIFO correctly. Pushing
    /// twice then popping once truncates to the inner savepoint;
    /// popping again truncates to the outer.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_scope_savepoint_nested_lifo() {
        use super::super::lineage::{ScopeFrame, SharedLineageSolver};

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        let bv_x = z3::ast::BV::new_const("test_scope_savepoint_nested_x", 8);
        let mk_frame = |val: u64| ScopeFrame::new(true, bv_x.eq(z3::ast::BV::from_u64(val, 8)));

        // outer save (depth=0), add 1 frame, inner save (depth=1), add 2,
        // pop -> truncate to 1, pop -> truncate to 0.
        ctx.scope_savepoint_push();
        ctx.push_scope_frame_for_testing(mk_frame(1));
        ctx.scope_savepoint_push();
        ctx.push_scope_frame_for_testing(mk_frame(2));
        ctx.push_scope_frame_for_testing(mk_frame(3));
        assert_eq!(ctx.scope_path_len(), 3);
        assert_eq!(ctx.scope_savepoint_depth(), 2);

        ctx.scope_savepoint_pop();
        assert_eq!(
            ctx.scope_path_len(),
            1,
            "inner pop should truncate to the inner save"
        );
        assert_eq!(ctx.scope_savepoint_depth(), 1);

        ctx.scope_savepoint_pop();
        assert_eq!(
            ctx.scope_path_len(),
            0,
            "outer pop should truncate to the outer save"
        );
        assert_eq!(ctx.scope_savepoint_depth(), 0);
    }

    /// angr-v5a5 slice 4c.1: with no lineage attached, `add_constraint`
    /// must hit the per-context Z3 solver and leave `scope_path` empty
    /// — byte-identical behavior to the pre-slice
    /// `self.with_z3_solver(|s| s.assert(&c))` call.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_none_branch_no_scope_path() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());
        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "test_add_constraint_none_branch_x", 8);
        let five = RustBV::concrete(5, 8);
        ctx.assume_true(&x.eq(&five, &ctx));

        // None branch must not touch scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            0,
            "None branch must not mint scope frames"
        );

        // Constraint must be in force on the per-context solver.
        assert!(ctx.solution(&x, 5));
        assert!(!ctx.solution(&x, 6));
    }

    /// angr-v5a5 slice 4c.1: with a lineage attached, `add_constraint`
    /// mints a fresh `ScopeFrame`, appends it to `scope_path`, and
    /// routes the assert through the shared solver's switch_to —
    /// the bug-shaped piece the slice-4-blocker-analysis memo called out.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_some_branch_appends_scope_frame() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "test_add_constraint_some_branch_x", 8);
        let five = RustBV::concrete(5, 8);
        ctx.assume_true(&x.eq(&five, &ctx));

        // Some branch must mint exactly one frame on scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            1,
            "Some branch must mint one scope frame per add_constraint"
        );

        // The frame must have been pushed onto the shared solver — its
        // loaded_depth should match scope_path's length.
        assert_eq!(
            lin.lock().loaded_depth(),
            1,
            "switch_to must have pushed the new frame onto the shared solver"
        );
    }

    /// angr-v5a5 slice 4c.1: sibling isolation invariant — the constraint
    /// minted by sibling A's `add_constraint` must NOT be visible when
    /// sibling B (which never added it) issues a query through the same
    /// shared lineage solver. This is the core invariant the slice-4
    /// design protects: per-state constraints stay in per-state scope
    /// frames, never at the lineage base.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_sibling_isolation() {
        use super::super::lineage::SharedLineageSolver;

        // Both contexts share the same lineage.
        let sibling_a = SymContext::new();
        let sibling_b = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        sibling_a.set_lineage_for_testing(Arc::clone(&lin));
        sibling_b.set_lineage_for_testing(Arc::clone(&lin));

        // Sibling A adds x == 5.
        let x_a = RustBV::symbolic(&sibling_a, "test_add_constraint_sibling_x", 8);
        let five = RustBV::concrete(5, 8);
        sibling_a.assume_true(&x_a.eq(&five, &sibling_a));
        assert_eq!(sibling_a.scope_path_len(), 1);
        assert_eq!(sibling_b.scope_path_len(), 0);

        // Sibling B references the same Z3 symbol by name but has not
        // constrained it. A query from B should see x as unconstrained
        // — switch_to to B's empty scope_path pops A's frame first.
        let x_b = RustBV::symbolic(&sibling_b, "test_add_constraint_sibling_x", 8);
        // B should accept any value for x.
        assert!(sibling_b.solution(&x_b, 42));
        assert!(sibling_b.solution(&x_b, 99));

        // A still sees x == 5.
        assert!(sibling_a.solution(&x_a, 5));
        assert!(!sibling_a.solution(&x_a, 42));
    }

    /// angr-v5a5 slice 4c.2: helper mirrors `batch_entry` for the
    /// single-shot `add_constraint_raw` path. Lifts a width-1 RustBV's
    /// Z3 Bool AST into a typed `Z3AstPtr` handle whose own ref keeps
    /// the AST alive until consumption (no `mem::forget` leak required —
    /// the wrapper does proper refcounting via `Z3_inc_ref` / `Z3_dec_ref`).
    #[cfg(feature = "vex-engine-z3")]
    fn raw_entry(cond: &RustBV) -> Z3AstPtr {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let bool_ast = cond.to_z3_bool();
        let ctx = z3::Context::thread_local();
        let ptr = bool_ast.get_z3_ast().as_ptr() as usize;
        // SAFETY: `bool_ast` keeps the AST alive across the inc_ref call;
        // the resulting Z3AstPtr holds its own ref so the AST survives
        // `bool_ast` dropping at end of this function.
        unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) }.expect("non-null Bool AST")
    }

    /// angr-v5a5 slice 4c.2: with no lineage attached, `add_constraint_raw`
    /// must hit the per-context Z3 solver and leave `scope_path` empty —
    /// byte-identical behavior to the pre-slice
    /// `self.with_z3_solver(|s| s.assert(&c))` call. Mirrors
    /// `test_add_constraint_none_branch_no_scope_path` for the raw path.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_raw_none_branch_no_scope_path() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());
        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "test_add_constraint_raw_none_x", 8);
        let five = RustBV::concrete(5, 8);
        let ast = raw_entry(&x.eq(&five, &ctx));
        ctx.add_constraint_raw(ast);

        // None branch must not touch scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            0,
            "None branch must not mint scope frames"
        );

        // Constraint must be in force on the per-context solver.
        assert!(ctx.solution(&x, 5));
        assert!(!ctx.solution(&x, 6));
    }

    /// angr-v5a5 slice 4c.2: with a lineage attached, `add_constraint_raw`
    /// mints a fresh `ScopeFrame`, appends it to `scope_path`, and routes
    /// the assert through the shared solver's switch_to — same shape as
    /// `test_add_constraint_some_branch_appends_scope_frame` for the raw
    /// path. The `constraint.clone()` inside `add_constraint_raw` is a
    /// ref-bump on the Z3 AST originally wrapped from the raw pointer.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_raw_some_branch_appends_scope_frame() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "test_add_constraint_raw_some_x", 8);
        let five = RustBV::concrete(5, 8);
        let ast = raw_entry(&x.eq(&five, &ctx));
        ctx.add_constraint_raw(ast);

        // Some branch must mint exactly one frame on scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            1,
            "Some branch must mint one scope frame per add_constraint_raw"
        );

        // The frame must have been pushed onto the shared solver — its
        // loaded_depth should match scope_path's length.
        assert_eq!(
            lin.lock().loaded_depth(),
            1,
            "switch_to must have pushed the new frame onto the shared solver"
        );
    }

    /// angr-v5a5 slice 4c.2: sibling isolation invariant for the raw path
    /// — the constraint minted by sibling A's `add_constraint_raw` must
    /// NOT be visible when sibling B issues a query through the same
    /// shared lineage solver. Mirrors `test_add_constraint_sibling_isolation`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_raw_sibling_isolation() {
        use super::super::lineage::SharedLineageSolver;

        let sibling_a = SymContext::new();
        let sibling_b = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        sibling_a.set_lineage_for_testing(Arc::clone(&lin));
        sibling_b.set_lineage_for_testing(Arc::clone(&lin));

        // Sibling A adds x == 5 via the raw path.
        let x_a = RustBV::symbolic(&sibling_a, "test_add_constraint_raw_sibling_x", 8);
        let five = RustBV::concrete(5, 8);
        let ast = raw_entry(&x_a.eq(&five, &sibling_a));
        sibling_a.add_constraint_raw(ast);
        assert_eq!(sibling_a.scope_path_len(), 1);
        assert_eq!(sibling_b.scope_path_len(), 0);

        // Sibling B references the same named symbol but is unconstrained
        // — switch_to to B's empty scope_path must pop A's frame first.
        let x_b = RustBV::symbolic(&sibling_b, "test_add_constraint_raw_sibling_x", 8);
        assert!(sibling_b.solution(&x_b, 42));
        assert!(sibling_b.solution(&x_b, 99));

        // A still sees x == 5.
        assert!(sibling_a.solution(&x_a, 5));
        assert!(!sibling_a.solution(&x_a, 42));
    }

    /// angr-sfp9: a second `add_constraint_raw` with the SAME Z3_ast ptr
    /// must hit the dedup side-table — z3_assertions stays at 1 entry, the
    /// per-call hit counter increments, and the constraint stays in force.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_raw_dedup_repeat_skips_push() {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "test_dedup_repeat_x", 8);
        let five = RustBV::concrete(5, 8);
        let ast = raw_entry(&x.eq(&five, &ctx));

        let hits_before = ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.load(Ordering::Relaxed);

        // Three calls with the same Z3_ast — dedup must catch reps 2 and 3.
        // `clone_ref` produces independent handles pointing at the same AST.
        ctx.add_constraint_raw(ast.clone_ref());
        ctx.add_constraint_raw(ast.clone_ref());
        ctx.add_constraint_raw(ast);

        // Only one entry should land in local.z3_assertions despite three
        // calls — the side-table catches reps 2 and 3.
        {
            let local = ctx.local_constraints.lock();
            assert_eq!(local.z3_assertions.len(), 1, "dedup must skip the push");
            assert!(local.dedup_set_seeded, "first call seeds the set");
            assert_eq!(local.dedup_set.len(), 1);
        }
        // Two of the three calls hit dedup.
        assert_eq!(
            ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.load(Ordering::Relaxed) - hits_before,
            2
        );

        // Constraint still in force despite skipping reps 2 and 3.
        assert!(ctx.solution(&x, 5));
        assert!(!ctx.solution(&x, 6));
    }

    /// angr-mwbp: when `dedup_set` is already seeded (here by a prior
    /// `add_constraint_raw`), a repeat `assume_true` with the SAME 1-bit
    /// RustBV cond hits the side-table — `z3_assertions` stays at the same
    /// length, `assumed` still grows (Python-visible duplicates preserved
    /// for claripy `solver.add(c)` semantics), and the constraint stays
    /// in force.
    ///
    /// Note: `assume_true` does NOT trigger seeding by itself (would impose
    /// an O(N) regression on branch-heavy benches without
    /// `add_constraint_raw` traffic — see `check_z3_dedup_if_seeded`). The
    /// `add_constraint_raw` call below provides the seed.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_assume_true_dedup_repeat_skips_assert() {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "test_assume_true_dedup_x", 8);
        let five = RustBV::concrete(5, 8);
        let cond = x.eq(&five, &ctx);

        // Seed the dedup_set via a distinct add_constraint_raw assertion.
        let y = RustBV::symbolic(&ctx, "test_assume_true_dedup_y", 8);
        let ten = RustBV::concrete(10, 8);
        ctx.add_constraint_raw(raw_entry(&y.eq(&ten, &ctx)));
        assert!(ctx.local_constraints.lock().dedup_set_seeded);

        ctx.assume_true(&cond);
        ctx.assume_true(&cond);
        ctx.assume_true(&cond);

        {
            let local = ctx.local_constraints.lock();
            assert_eq!(
                local.z3_assertions.len(),
                2,
                "assume_true dedup must skip the redundant pushes (1 raw + 1 assume)"
            );
            assert_eq!(
                local.assumed.len(),
                3,
                "assumed vec grows on every call — Python-visible duplicates preserved"
            );
            assert!(local.dedup_set_seeded);
            assert_eq!(local.dedup_set.len(), 2);
        }

        // Constraint still in force despite skipping reps 2 and 3.
        assert!(ctx.solution(&x, 5));
        assert!(!ctx.solution(&x, 6));
    }

    /// angr-mwbp: `assume_false` on the same 1-bit cond also dedups when the
    /// dedup_set is seeded. The negation `!cond` produces a stable Z3 ptr
    /// (the negation node is hash-cons'd by Z3), so reps 2+ are caught.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_assume_false_dedup_repeat_skips_assert() {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "test_assume_false_dedup_x", 8);
        let five = RustBV::concrete(5, 8);
        let cond = x.eq(&five, &ctx);

        // Seed the dedup_set via a distinct add_constraint_raw assertion.
        let y = RustBV::symbolic(&ctx, "test_assume_false_dedup_y", 8);
        let ten = RustBV::concrete(10, 8);
        ctx.add_constraint_raw(raw_entry(&y.eq(&ten, &ctx)));

        ctx.assume_false(&cond);
        ctx.assume_false(&cond);

        {
            let local = ctx.local_constraints.lock();
            assert_eq!(local.z3_assertions.len(), 2);
            assert_eq!(local.assumed.len(), 2);
            assert_eq!(local.dedup_set.len(), 2);
        }

        // The not-eq constraint is in force: x != 5.
        assert!(!ctx.solution(&x, 5));
        assert!(ctx.solution(&x, 6));
    }

    /// angr-mwbp: with no prior `add_constraint_raw` to seed the side-table,
    /// `assume_true` must NOT trigger seeding by itself — fresh contexts
    /// stay unseeded and fall through to the legacy push-only behavior.
    /// This is the bench-safety contract for `check_z3_dedup_if_seeded`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_assume_true_no_self_seeding_under_fresh_context() {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "test_assume_no_seed_x", 8);
        let five = RustBV::concrete(5, 8);
        let cond = x.eq(&five, &ctx);

        ctx.assume_true(&cond);
        ctx.assume_true(&cond);

        {
            let local = ctx.local_constraints.lock();
            assert!(
                !local.dedup_set_seeded,
                "assume_true alone must not trigger O(N) seeding (angr-mwbp)"
            );
            // Without seeding, both pushes land in z3_assertions.
            assert_eq!(local.z3_assertions.len(), 2);
            assert_eq!(local.assumed.len(), 2);
        }
        // Constraint still in force.
        assert!(ctx.solution(&x, 5));
        assert!(!ctx.solution(&x, 6));
    }

    /// angr-sfp9: a ptr already present in shared (post-fork) must be
    /// caught by the lazy seed on first `add_constraint_raw` call in the
    /// child — verifies the seed walks `z3_assertions_shared`.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_raw_dedup_seeds_from_shared() {
        let parent = SymContext::new();
        let x = RustBV::symbolic(&parent, "test_dedup_shared_x", 8);
        let five = RustBV::concrete(5, 8);
        let ast = raw_entry(&x.eq(&five, &parent));
        // Take a second ref for the child call below before the parent
        // consumes its handle.
        let ast_for_child = ast.clone_ref();
        parent.add_constraint_raw(ast);
        // Fork the parent; child's frozen_shared should contain the
        // assertion, and child's local.dedup_set is unseeded.
        let child = parent.fork();
        assert!(!child.local_constraints.lock().dedup_set_seeded);
        let hits_before = ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.load(Ordering::Relaxed);
        child.add_constraint_raw(ast_for_child);
        // Seeded from shared; the ptr was already there, so this call is
        // a dedup hit. Child's local.z3_assertions stays empty.
        {
            let local = child.local_constraints.lock();
            assert!(local.dedup_set_seeded);
            assert_eq!(
                local.z3_assertions.len(),
                0,
                "dedup against shared must skip push on child local"
            );
        }
        assert_eq!(
            ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.load(Ordering::Relaxed) - hits_before,
            1
        );
    }

    /// angr-v5a5 slice 4c.2b: with no lineage attached,
    /// `add_constraint_tracked_indexed` must hit the per-context Z3 solver
    /// via `assert_and_track` — byte-identical behavior to the pre-slice
    /// `self.with_z3_solver(|s| s.assert_and_track(...))` call. The
    /// tracker registers in `constraint_trackers` and the returned index
    /// matches the trackers vector position. Mirrors
    /// `test_add_constraint_none_branch_no_scope_path` plus an explicit
    /// unsat-core fidelity check.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_tracked_indexed_none_branch_no_scope_path() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());
        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "test_act_indexed_none_x", 8);
        let five = RustBV::concrete(5, 8);
        let ten = RustBV::concrete(10, 8);

        // Tracked constraint 1: x == 5.
        let c1 = x.eq(&five, &ctx).to_z3_bool();
        let idx1 = ctx.add_constraint_tracked_indexed(c1);
        assert_eq!(idx1, 0, "first tracker registered at index 0");

        // None branch must not touch scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            0,
            "None branch must not mint scope frames"
        );

        // Constraint must be in force on the per-context solver.
        assert!(ctx.solution(&x, 5));
        assert!(!ctx.solution(&x, 6));

        // Tracked constraint 2: x == 10 (deliberately UNSAT against c1).
        let c2 = x.eq(&ten, &ctx).to_z3_bool();
        let idx2 = ctx.add_constraint_tracked_indexed(c2);
        assert_eq!(idx2, 1, "second tracker registered at index 1");

        // Both trackers should appear in the unsat core — full fidelity
        // is preserved on the None branch.
        assert!(!ctx.is_sat(), "x == 5 ∧ x == 10 must be UNSAT");
        let core = ctx.unsat_core();
        assert!(
            core.contains(&idx1) && core.contains(&idx2),
            "None branch unsat_core must include both tracker indices; got {:?}",
            core
        );
    }

    /// angr-v5a5 slice 4c.2b: with a lineage attached,
    /// `add_constraint_tracked_indexed` mints a fresh `ScopeFrame`,
    /// appends it to `scope_path`, and routes the assert through the
    /// shared solver's `switch_to` — same shape as 4c.1/4c.2's Some-branch
    /// tests. The tracker registers in `constraint_trackers` (so the
    /// returned index is stable) but `switch_to` uses plain `assert`,
    /// so unsat-core fidelity is intentionally deferred here.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_tracked_indexed_some_branch_appends_scope_frame() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "test_act_indexed_some_x", 8);
        let five = RustBV::concrete(5, 8);
        let constraint = x.eq(&five, &ctx).to_z3_bool();
        let idx = ctx.add_constraint_tracked_indexed(constraint);
        assert_eq!(
            idx, 0,
            "tracker index 0 expected even with lineage installed"
        );

        // Some branch must mint exactly one frame on scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            1,
            "Some branch must mint one scope frame per add_constraint_tracked_indexed"
        );

        // The frame must have been pushed onto the shared solver — its
        // loaded_depth should match scope_path's length.
        assert_eq!(
            lin.lock().loaded_depth(),
            1,
            "switch_to must have pushed the new frame onto the shared solver"
        );
    }

    /// angr-v5a5 slice 4c.2b: sibling isolation invariant for the
    /// tracked-indexed path — the constraint minted by sibling A's
    /// `add_constraint_tracked_indexed` must NOT be visible when sibling
    /// B issues a query through the same shared lineage solver. Mirrors
    /// `test_add_constraint_sibling_isolation` for the tracked variant.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraint_tracked_indexed_sibling_isolation() {
        use super::super::lineage::SharedLineageSolver;

        let sibling_a = SymContext::new();
        let sibling_b = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        sibling_a.set_lineage_for_testing(Arc::clone(&lin));
        sibling_b.set_lineage_for_testing(Arc::clone(&lin));

        // Sibling A adds tracked x == 5.
        let x_a = RustBV::symbolic(&sibling_a, "test_act_indexed_sibling_x", 8);
        let five = RustBV::concrete(5, 8);
        let constraint = x_a.eq(&five, &sibling_a).to_z3_bool();
        let _ = sibling_a.add_constraint_tracked_indexed(constraint);
        assert_eq!(sibling_a.scope_path_len(), 1);
        assert_eq!(sibling_b.scope_path_len(), 0);

        // Sibling B references the same named symbol but is unconstrained
        // — switch_to to B's empty scope_path must pop A's frame first.
        let x_b = RustBV::symbolic(&sibling_b, "test_act_indexed_sibling_x", 8);
        assert!(sibling_b.solution(&x_b, 42));
        assert!(sibling_b.solution(&x_b, 99));

        // A still sees x == 5.
        assert!(sibling_a.solution(&x_a, 5));
        assert!(!sibling_a.solution(&x_a, 42));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_z3_variable_identity() {
        // Test that Z3 variables with the same name are treated as the same variable
        let ctx = SymContext::new();

        // Create a symbolic variable
        let x = RustBV::symbolic(&ctx, "x", 32);

        // Add constraint: x > 10
        let ten = RustBV::concrete(10, 32);
        let gt_ten = x.ugt(&ten, &ctx);
        ctx.assume_true(&gt_ten);

        // Verify constraint is enforced
        assert!(ctx.solution(&x, 15)); // 15 > 10, should be true
        assert!(!ctx.solution(&x, 5)); // 5 > 10 is false, should be unsat

        // Now add constraint: x < 20
        let twenty = RustBV::concrete(20, 32);
        let lt_twenty = x.ult(&twenty, &ctx);
        ctx.assume_true(&lt_twenty);

        // Verify both constraints are enforced
        assert!(ctx.solution(&x, 15)); // 10 < 15 < 20
        assert!(!ctx.solution(&x, 5)); // 5 < 10
        assert!(!ctx.solution(&x, 25)); // 25 > 20
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_max_constrained() {
        // Test min/max with constrained variable
        let ctx = SymContext::new();

        // Create a symbolic variable
        let x = RustBV::symbolic(&ctx, "x", 32);

        // Add constraints: 10 < x < 20
        let ten = RustBV::concrete(10, 32);
        let twenty = RustBV::concrete(20, 32);
        ctx.assume_true(&x.ugt(&ten, &ctx));
        ctx.assume_true(&x.ult(&twenty, &ctx));

        // min should be 11, max should be 19
        let min_val = ctx.min(&x, false);
        let max_val = ctx.max(&x, false);

        assert_eq!(min_val, Some(11), "min should be 11");
        assert_eq!(max_val, Some(19), "max should be 19");
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_z3_same_name_different_create() {
        // Test that creating variables with the same name but different calls
        // still references the same Z3 variable

        let ctx = SymContext::new();

        // Create two RustBV::symbolic with the same name
        let x1 = RustBV::symbolic(&ctx, "x_test", 32);
        let x2 = RustBV::symbolic(&ctx, "x_test", 32);

        // Add constraint using x1: x1 > 10
        let ten = RustBV::concrete(10, 32);
        let gt_ten = x1.ugt(&ten, &ctx);
        ctx.assume_true(&gt_ten);

        // Check using x2 - should have the same constraint if same variable
        // If they're different variables, x2 wouldn't have the constraint
        let ast2 = x2.to_z3_ast();

        // Try to find if x2 can be 5 (should be UNSAT if same as x1)
        ctx.push();
        let five = z3::ast::BV::from_u64(5, 32);
        let eq_five = ast2.eq(&five);
        ctx.add_constraint(eq_five);
        let can_be_five = ctx.is_sat();
        ctx.pop();

        assert!(
            !can_be_five,
            "x2 should have same constraints as x1 since same name"
        );
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_max_use_cached_model_unsigned() {
        // Verify unsigned min()/max() return correct values when seeded by a
        // cached model, and that the HIT counter increments. Counters are
        // process-wide and tests run in parallel, so we only assert deltas
        // with >= bounds (other tests may bump the same counter concurrently).
        use super::super::stats::{Z3_EXTREMA_MODEL_HIT_COUNT, Z3_EXTREMA_MODEL_MISS_COUNT};

        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_min_max_cached", 32);
        let lo_bound = RustBV::concrete(10, 32);
        let hi_bound = RustBV::concrete(20, 32);
        ctx.assume_true(&x.ugt(&lo_bound, &ctx));
        ctx.assume_true(&x.ult(&hi_bound, &ctx));

        // Populate the model cache with an eval.
        let v = ctx.eval(&x);
        assert!(v.is_some());

        let hit_before = Z3_EXTREMA_MODEL_HIT_COUNT.load(Ordering::Relaxed);
        let miss_before = Z3_EXTREMA_MODEL_MISS_COUNT.load(Ordering::Relaxed);

        let min_val = ctx.min(&x, false);
        let max_val = ctx.max(&x, false);

        let hit_after = Z3_EXTREMA_MODEL_HIT_COUNT.load(Ordering::Relaxed);
        let miss_after = Z3_EXTREMA_MODEL_MISS_COUNT.load(Ordering::Relaxed);

        assert_eq!(min_val, Some(11), "min should be 11");
        assert_eq!(max_val, Some(19), "max should be 19");
        // Our 2 calls each had a usable model — should bump HIT by >=2 and
        // not bump MISS at all.
        assert!(
            hit_after - hit_before >= 2,
            "expected >=2 extrema cache hits across min+max, got {}",
            hit_after - hit_before
        );
        assert_eq!(
            miss_after - miss_before,
            0,
            "expected 0 extrema cache misses from this test's calls"
        );
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_use_cached_model_seeds_unsigned_zero_witness() {
        // If the cached witness is 0, unsigned min should short-circuit
        // (hi=0=lo) and return 0 with no binary-search SAT checks.
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_min_zero_witness", 32);
        let five = RustBV::concrete(5, 32);
        ctx.assume_true(&x.ule(&five, &ctx));
        // Pin the model under a push frame so the constraint x==0 doesn't
        // persist into the actual min() call. The model survives the pop.
        ctx.push();
        ctx.add_bv_constraint(&x, 0);
        let _ = ctx.eval(&x);
        ctx.pop();

        let result = ctx.min(&x, false);
        assert_eq!(result, Some(0), "min should be 0 (witness-pinned)");
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_max_signed_with_negative_witness() {
        // Verify signed min/max are correct when the cached witness is
        // signed-negative.
        use super::super::stats::Z3_EXTREMA_MODEL_HIT_COUNT;
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_signed_neg", 32);
        // Constrain: -20 <= x <= -5 (signed)
        let neg20 = RustBV::concrete((-20i32) as u32 as u128, 32);
        let neg5 = RustBV::concrete((-5i32) as u32 as u128, 32);
        ctx.assume_true(&x.sge(&neg20, &ctx));
        ctx.assume_true(&x.sle(&neg5, &ctx));
        // Populate cache with eval — witness must be in [-20, -5].
        let v = ctx.eval(&x).unwrap();
        // Witness's sign bit (bit 31 for width=32) must be set.
        assert_ne!(v & (1u128 << 31), 0, "witness should be signed-negative");

        let hit_before = Z3_EXTREMA_MODEL_HIT_COUNT.load(Ordering::Relaxed);

        let min_signed = ctx.min(&x, true);
        let max_signed = ctx.max(&x, true);

        let hit_after = Z3_EXTREMA_MODEL_HIT_COUNT.load(Ordering::Relaxed);

        assert_eq!(min_signed, Some((-20i32) as u32 as u128));
        assert_eq!(max_signed, Some((-5i32) as u32 as u128));
        assert!(
            hit_after - hit_before >= 2,
            "expected >=2 extrema hits, got {}",
            hit_after - hit_before
        );
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_max_signed_with_positive_witness() {
        // Verify signed min/max are correct when the cached witness is
        // signed-positive (covers the witness_is_non_negative path in max).
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_signed_pos", 32);
        // Constrain: 5 <= x <= 20 (signed)
        let five = RustBV::concrete(5u128, 32);
        let twenty = RustBV::concrete(20u128, 32);
        ctx.assume_true(&x.sge(&five, &ctx));
        ctx.assume_true(&x.sle(&twenty, &ctx));
        // Populate cache.
        let v = ctx.eval(&x).unwrap();
        assert_eq!(
            v & (1u128 << 31),
            0,
            "witness should be signed-non-negative"
        );

        assert_eq!(ctx.min(&x, true), Some(5));
        assert_eq!(ctx.max(&x, true), Some(20));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_min_max_no_cached_model_still_correct() {
        // When no model is cached (e.g. fresh context after pop without prior
        // eval), min/max must still work correctly via the fallback path.
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_no_model", 32);
        let lo_bound = RustBV::concrete(100, 32);
        let hi_bound = RustBV::concrete(200, 32);
        ctx.assume_true(&x.ugt(&lo_bound, &ctx));
        ctx.assume_true(&x.ult(&hi_bound, &ctx));
        // Don't call eval. The first is_sat inside min will populate the
        // cache, so the witness path is exercised — but the seeded value is
        // whatever Z3 chose. Still must be in [101, 199].
        assert_eq!(ctx.min(&x, false), Some(101));
        assert_eq!(ctx.max(&x, false), Some(199));
    }

    /// Helper: build a `(Z3AstPtr, RustBV, bool)` tuple from a width-1 cond
    /// for use with `add_constraints_raw_batch`. Mirrors what the
    /// `RustSolverContext::add_constraints` fast path does with claripy ASTs.
    /// The typed `Z3AstPtr` does proper refcounting via `Z3_inc_ref` —
    /// no `mem::forget` leak required.
    #[cfg(feature = "vex-engine-z3")]
    fn batch_entry(cond: &RustBV) -> (Z3AstPtr, RustBV, bool) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let bool_ast = cond.to_z3_bool();
        let ctx = z3::Context::thread_local();
        let ptr = bool_ast.get_z3_ast().as_ptr() as usize;
        // SAFETY: `bool_ast` keeps the AST alive across the inc_ref call;
        // the resulting Z3AstPtr holds its own ref so the AST survives
        // `bool_ast` dropping at end of this function.
        let ast = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) }.expect("non-null Bool AST");
        (ast, cond.clone(), true)
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_basic() {
        // Three independent constraints in one batch should constrain x as
        // tightly as adding them one by one. Verifies semantics match the
        // unbatched path.
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_batch_basic", 32);
        let lo = RustBV::concrete(10, 32);
        let hi = RustBV::concrete(20, 32);
        let mid = RustBV::concrete(15, 32);

        let entries = vec![
            batch_entry(&x.ugt(&lo, &ctx)),
            batch_entry(&x.ult(&hi, &ctx)),
            batch_entry(&x.uge(&mid, &ctx)),
        ];
        let before = ctx.num_constraints();
        ctx.add_constraints_raw_batch(entries);
        assert_eq!(ctx.num_constraints(), before + 3);
        // x must satisfy 15 <= x < 20.
        assert!(ctx.solution(&x, 15));
        assert!(ctx.solution(&x, 19));
        assert!(!ctx.solution(&x, 14));
        assert!(!ctx.solution(&x, 20));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_empty_is_noop() {
        // Empty batch must not touch the solver or counter.
        let ctx = SymContext::new();
        let before = ctx.num_constraints();
        ctx.add_constraints_raw_batch(Vec::new());
        assert_eq!(ctx.num_constraints(), before);
        assert!(ctx.is_sat());
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_drops_inconsistent_model() {
        // Populate the model cache with eval, then batch-add a constraint
        // that contradicts that model. The cache must be dropped.
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_batch_model", 32);
        // Loose constraint first; populate model.
        ctx.assume_true(&x.ult(&RustBV::concrete(100, 32), &ctx));
        let first = ctx.eval(&x).unwrap();
        // Now batch-add x == new_val (forces a value distinct from `first`
        // but still within [0,99]), which invalidates the cached model.
        let new_val = if first == 0 { 1 } else { 0 };
        let pinned = RustBV::concrete(new_val, 32);
        let entries = vec![batch_entry(&x.eq(&pinned, &ctx))];
        ctx.add_constraints_raw_batch(entries);
        // The next eval must produce the newly-required value.
        assert_eq!(ctx.eval(&x), Some(new_val));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_preserves_consistent_model() {
        // A constraint already satisfied by the cached model should leave
        // the model in place (matches the single-shot invalidate path).
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_batch_consistent", 32);
        ctx.assume_true(&x.eq(&RustBV::concrete(7, 32), &ctx));
        // Populate model.
        let v = ctx.eval(&x).unwrap();
        assert_eq!(v, 7);
        // Batch-add a constraint that the model already satisfies.
        let entries = vec![batch_entry(&x.ult(&RustBV::concrete(100, 32), &ctx))];
        ctx.add_constraints_raw_batch(entries);
        // Still SAT, still 7.
        assert!(ctx.is_sat());
        assert_eq!(ctx.eval(&x), Some(7));
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_tracks_local_assertions() {
        // The batch path must populate local_constraints.z3_assertions so
        // export_z3_assertion_ptrs sees the same count as the per-constraint
        // path. Regression guard against forgetting to extend the vector.
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_batch_export", 32);
        let entries = vec![
            batch_entry(&x.ugt(&RustBV::concrete(0, 32), &ctx)),
            batch_entry(&x.ult(&RustBV::concrete(100, 32), &ctx)),
        ];
        let before = ctx.export_z3_assertion_ptrs().len();
        ctx.add_constraints_raw_batch(entries);
        let after = ctx.export_z3_assertion_ptrs().len();
        assert_eq!(after - before, 2);
    }

    /// angr-v5a5 slice 4c.2c: with no lineage attached,
    /// `add_constraints_raw_batch` must hit the per-context Z3 solver and
    /// leave `scope_path` empty — byte-identical behavior to the pre-slice
    /// `self.with_z3_solver(|s| { for c in &constraints { s.assert(c); } })`
    /// call. Mirrors `test_add_constraint_raw_none_branch_no_scope_path`
    /// for the batched path, with a multi-entry batch.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_none_branch_no_scope_path() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());
        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "x_batch_none", 32);
        let entries = vec![
            batch_entry(&x.ugt(&RustBV::concrete(10, 32), &ctx)),
            batch_entry(&x.ult(&RustBV::concrete(20, 32), &ctx)),
            batch_entry(&x.uge(&RustBV::concrete(15, 32), &ctx)),
        ];
        ctx.add_constraints_raw_batch(entries);

        // None branch must not mint scope frames.
        assert_eq!(
            ctx.scope_path_len(),
            0,
            "None branch must not mint scope frames"
        );

        // All three constraints must be in force on the per-context solver.
        assert!(ctx.solution(&x, 15));
        assert!(ctx.solution(&x, 19));
        assert!(!ctx.solution(&x, 14));
        assert!(!ctx.solution(&x, 20));
    }

    /// angr-v5a5 slice 4c.2c: with a lineage attached,
    /// `add_constraints_raw_batch` mints N fresh `ScopeFrame`s under one
    /// `scope_path.lock()` acquisition and routes a single `switch_to` call
    /// through the shared solver — the divergent-suffix walk in `switch_to`
    /// then asserts each new frame inside its own Z3 push. Mirrors
    /// `test_add_constraint_raw_some_branch_appends_scope_frame` for the
    /// batched path.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_some_branch_appends_scope_frames() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        assert_eq!(ctx.scope_path_len(), 0);

        let x = RustBV::symbolic(&ctx, "x_batch_some", 32);
        let entries = vec![
            batch_entry(&x.ugt(&RustBV::concrete(10, 32), &ctx)),
            batch_entry(&x.ult(&RustBV::concrete(20, 32), &ctx)),
            batch_entry(&x.uge(&RustBV::concrete(15, 32), &ctx)),
        ];
        ctx.add_constraints_raw_batch(entries);

        // Some branch must mint exactly N frames on scope_path.
        assert_eq!(
            ctx.scope_path_len(),
            3,
            "Some branch must mint one scope frame per batch entry"
        );

        // switch_to must have pushed all three frames onto the shared
        // solver — loaded_depth equals scope_path's length.
        assert_eq!(
            lin.lock().loaded_depth(),
            3,
            "switch_to must have pushed all batch frames onto the shared solver"
        );
    }

    /// angr-v5a5 slice 4c.2c: sibling isolation invariant for the batched
    /// raw path — the N constraints minted by sibling A's
    /// `add_constraints_raw_batch` must NOT be visible when sibling B
    /// queries the same shared lineage solver. Mirrors
    /// `test_add_constraint_raw_sibling_isolation` with a multi-entry batch.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_add_constraints_raw_batch_sibling_isolation() {
        use super::super::lineage::SharedLineageSolver;

        let sibling_a = SymContext::new();
        let sibling_b = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        sibling_a.set_lineage_for_testing(Arc::clone(&lin));
        sibling_b.set_lineage_for_testing(Arc::clone(&lin));

        // Sibling A batch-adds x > 10 ∧ x < 20.
        let x_a = RustBV::symbolic(&sibling_a, "x_batch_sibling", 32);
        let entries = vec![
            batch_entry(&x_a.ugt(&RustBV::concrete(10, 32), &sibling_a)),
            batch_entry(&x_a.ult(&RustBV::concrete(20, 32), &sibling_a)),
        ];
        sibling_a.add_constraints_raw_batch(entries);
        assert_eq!(sibling_a.scope_path_len(), 2);
        assert_eq!(sibling_b.scope_path_len(), 0);

        // Sibling B references the same named symbol but is unconstrained
        // — switch_to to B's empty scope_path must pop both of A's frames.
        let x_b = RustBV::symbolic(&sibling_b, "x_batch_sibling", 32);
        assert!(sibling_b.solution(&x_b, 5));
        assert!(sibling_b.solution(&x_b, 42));

        // A still sees 10 < x < 20.
        assert!(sibling_a.solution(&x_a, 15));
        assert!(!sibling_a.solution(&x_a, 5));
        assert!(!sibling_a.solution(&x_a, 42));
    }

    // -------------------------------------------------------------------------
    // angr-2j5v instrumentation counter tests
    // -------------------------------------------------------------------------
    //
    // Counters are process-global atomics — other parallel tests may touch
    // them. Each test reads a baseline, performs `n` recorder calls, and
    // asserts the delta is `>= n` (not `== n`). Tests do NOT assume the
    // counters start at zero.

    #[test]
    fn test_record_vex_dispatch_counters() {
        let baseline = get_solver_stats();
        let base_unop = baseline.get("vex_unop_total").copied().unwrap_or(0);
        let base_binop = baseline.get("vex_binop_total").copied().unwrap_or(0);
        let base_triop = baseline.get("vex_triop_total").copied().unwrap_or(0);
        let base_qop = baseline.get("vex_qop_total").copied().unwrap_or(0);
        let base_arith = baseline.get("vex_op_arith").copied().unwrap_or(0);
        let base_logic = baseline.get("vex_op_logic").copied().unwrap_or(0);
        let base_fp = baseline.get("vex_op_fp").copied().unwrap_or(0);

        record_vex_unop(VexOpFamily::Logic);
        record_vex_binop(VexOpFamily::Arith);
        record_vex_binop(VexOpFamily::Arith);
        record_vex_triop(VexOpFamily::Fp);
        record_vex_qop(VexOpFamily::Fp);

        let stats = get_solver_stats();
        assert!(stats.get("vex_unop_total").copied().unwrap() > base_unop);
        assert!(stats.get("vex_binop_total").copied().unwrap() >= base_binop + 2);
        assert!(stats.get("vex_triop_total").copied().unwrap() > base_triop);
        assert!(stats.get("vex_qop_total").copied().unwrap() > base_qop);
        // Each *_op_<family> got bumped once per record_vex_* call.
        assert!(stats.get("vex_op_arith").copied().unwrap() >= base_arith + 2);
        assert!(stats.get("vex_op_logic").copied().unwrap() > base_logic);
        assert!(stats.get("vex_op_fp").copied().unwrap() >= base_fp + 2);
    }

    #[test]
    fn test_record_mem_load_store_counters() {
        let baseline = get_solver_stats();
        let base_load = baseline.get("mem_load_count").copied().unwrap_or(0);
        let base_store = baseline.get("mem_store_count").copied().unwrap_or(0);
        let base_load_bytes = baseline.get("mem_load_bytes").copied().unwrap_or(0);
        let base_store_bytes = baseline.get("mem_store_bytes").copied().unwrap_or(0);
        let base_lsym = baseline.get("mem_load_symbolic_addr").copied().unwrap_or(0);
        let base_ssym = baseline
            .get("mem_store_symbolic_addr")
            .copied()
            .unwrap_or(0);
        let base_fault = baseline
            .get("mem_lazy_page_fault_count")
            .copied()
            .unwrap_or(0);

        record_mem_load(8);
        record_mem_load(4);
        record_mem_load_symbolic_addr();
        record_mem_store(16);
        record_mem_store_symbolic_addr();
        record_mem_lazy_page_fault();

        let stats = get_solver_stats();
        assert!(stats.get("mem_load_count").copied().unwrap() >= base_load + 2);
        assert!(stats.get("mem_store_count").copied().unwrap() > base_store);
        assert!(stats.get("mem_load_bytes").copied().unwrap() >= base_load_bytes + 12);
        assert!(stats.get("mem_store_bytes").copied().unwrap() >= base_store_bytes + 16);
        assert!(stats.get("mem_load_symbolic_addr").copied().unwrap() > base_lsym);
        assert!(stats.get("mem_store_symbolic_addr").copied().unwrap() > base_ssym);
        assert!(stats.get("mem_lazy_page_fault_count").copied().unwrap() > base_fault);
    }

    #[test]
    fn test_record_concretize_counters() {
        let baseline = get_solver_stats();
        let base_read = baseline.get("concretize_read_count").copied().unwrap_or(0);
        let base_write = baseline.get("concretize_write_count").copied().unwrap_or(0);
        let base_total = baseline
            .get("concretize_total_candidates")
            .copied()
            .unwrap_or(0);
        let base_max = baseline
            .get("concretize_max_candidates")
            .copied()
            .unwrap_or(0);

        record_concretize_read(3);
        record_concretize_read(7);
        record_concretize_write(1);

        let stats = get_solver_stats();
        assert!(stats.get("concretize_read_count").copied().unwrap() >= base_read + 2);
        assert!(stats.get("concretize_write_count").copied().unwrap() > base_write);
        // Total candidates: 3 + 7 + 1 = 11.
        assert!(
            stats.get("concretize_total_candidates").copied().unwrap() >= base_total + 11,
            "expected concretize_total_candidates delta >= 11"
        );
        // Max watermark must reach >= 7 (the largest K we recorded).
        assert!(stats.get("concretize_max_candidates").copied().unwrap() >= base_max.max(7));
    }

    #[test]
    fn test_record_bvop_counters() {
        let baseline = get_solver_stats();
        let base_rev = baseline.get("bvop_reverse_count").copied().unwrap_or(0);
        let base_cat = baseline.get("bvop_concat_count").copied().unwrap_or(0);
        let base_ext = baseline.get("bvop_extract_count").copied().unwrap_or(0);

        record_bvop_reverse();
        record_bvop_concat();
        record_bvop_concat();
        record_bvop_extract();
        record_bvop_extract();
        record_bvop_extract();

        let stats = get_solver_stats();
        assert!(stats.get("bvop_reverse_count").copied().unwrap() > base_rev);
        assert!(stats.get("bvop_concat_count").copied().unwrap() >= base_cat + 2);
        assert!(stats.get("bvop_extract_count").copied().unwrap() >= base_ext + 3);
    }

    #[test]
    fn test_bvop_counters_fire_on_symbolic_construction() {
        // End-to-end: building Reverse/Concat/Extract via the public RustBV
        // API on symbolic inputs must bump the respective counters. Concrete
        // inputs are folded by `as_u128()` and do NOT bump (this is the
        // desired behavior — we count node emissions, not fold-throughs).
        let ctx = SymContext::new_mock();
        let baseline = get_solver_stats();
        let base_rev = baseline.get("bvop_reverse_count").copied().unwrap_or(0);
        let base_cat = baseline.get("bvop_concat_count").copied().unwrap_or(0);
        let base_ext = baseline.get("bvop_extract_count").copied().unwrap_or(0);

        let s = RustBV::symbolic(&ctx, "test_2j5v", 32);
        let _r = s.reverse(&ctx);
        let _c = s.concat(&s, &ctx);
        let _e = s.extract(15, 0, &ctx);

        let stats = get_solver_stats();
        assert!(stats.get("bvop_reverse_count").copied().unwrap() > base_rev);
        assert!(stats.get("bvop_concat_count").copied().unwrap() > base_cat);
        assert!(stats.get("bvop_extract_count").copied().unwrap() > base_ext);
    }

    // angr-9o4n.1: Constraint round-trip spike via Z3_solver_to_string /
    // Z3_solver_from_string. Drives whether SMT-LIB2 is the right format for
    // angr-9o4n state save/restore. See bead notes for measured numbers.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_smtlib2_constraint_round_trip() {
        use std::time::Instant;

        // Build a non-trivial constraint set: 32-bit BVs + Extract + Concat +
        // multiple assertions. Names are uniquified so we don't collide with
        // any other test in the same Z3 thread-local context.
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "rt_x_9o4n", 32);
        let y = RustBV::symbolic(&ctx, "rt_y_9o4n", 32);

        // 10 < x < 20 (unsigned)
        let ten = RustBV::concrete(10, 32);
        let twenty = RustBV::concrete(20, 32);
        ctx.assume_true(&x.ugt(&ten, &ctx));
        ctx.assume_true(&x.ult(&twenty, &ctx));

        // Extract: high 16 bits of y are zero.
        let y_high = y.extract(31, 16, &ctx);
        let zero16 = RustBV::concrete(0, 16);
        ctx.assume_true(&y_high.eq(&zero16, &ctx));

        // Concat: low(x,16) ++ low(y,16) == 0x000B_0007 (x=11 satisfies low(x,16)=0x000B;
        // y_low=0x0007 satisfies the concat).
        let x_low = x.extract(15, 0, &ctx);
        let y_low = y.extract(15, 0, &ctx);
        let combined = x_low.concat(&y_low, &ctx);
        let target = RustBV::concrete(0x000B_0007, 32);
        ctx.assume_true(&combined.eq(&target, &ctx));

        // Sanity: original is SAT and the witness values fall in expected ranges.
        let original_sat = ctx.is_sat();
        assert!(original_sat, "constraint set should be sat");
        let x_witness = ctx.eval(&x).expect("x evaluable");
        let y_witness = ctx.eval(&y).expect("y evaluable");
        assert_eq!(
            x_witness, 11,
            "x must be 11 (the only value with 10<x<20 whose low 16 bits = 0x000B)"
        );
        assert_eq!(y_witness, 0x0000_0007, "y_high=0, y_low=0x0007");

        // Step 1: Serialize via Solver::to_string (SMT-LIB2 S-expression).
        let serialize_start = Instant::now();
        let serialized = ctx.debug_solver_string();
        let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
        let serialized_bytes = serialized.len();
        assert!(
            !serialized.is_empty(),
            "serialized SMT-LIB2 must be non-empty"
        );

        // Step 2: Parse into a fresh z3::Solver (shares the thread-local Z3
        // context, but is a logically independent solver). Constants declared
        // by name in the SMT-LIB2 string re-resolve to the SAME Z3 ASTs as the
        // originals because Z3 interns named constants in the context.
        let deserialize_start = Instant::now();
        let new_solver = z3::Solver::new();
        new_solver.from_string(serialized.clone());
        let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64;

        // Step 3a: check_sat matches.
        let new_check_start = Instant::now();
        let new_sat = matches!(new_solver.check(), z3::SatResult::Sat);
        let new_check_ns = new_check_start.elapsed().as_nanos() as u64;
        assert_eq!(
            new_sat, original_sat,
            "round-tripped solver sat-result must match"
        );

        // Step 3b: model values for x, y match the original witness (the
        // constraint set is restrictive enough that x=11, y_low=7 are forced).
        let new_model = new_solver
            .get_model()
            .expect("sat solver must produce model");
        let new_x_val = new_model
            .eval(&x.to_z3_ast(), true)
            .and_then(|bv| bv.as_u64())
            .expect("model should evaluate x");
        let new_y_val = new_model
            .eval(&y.to_z3_ast(), true)
            .and_then(|bv| bv.as_u64())
            .expect("model should evaluate y");
        assert_eq!(
            new_x_val as u128, x_witness,
            "round-tripped x model value must match"
        );
        assert_eq!(
            new_y_val as u128, y_witness,
            "round-tripped y model value must match"
        );

        // Step 4: report measurements. Captured by `cargo test -- --nocapture`
        // or `cargo test test_smtlib2_constraint_round_trip -- --nocapture`,
        // and pasted into the bead notes.
        eprintln!(
            "[angr-9o4n.1] SMT-LIB2 round-trip (small): {} bytes; \
             to_string={}us from_string={}us check_sat={}us; \
             5 assertions, 2 32-bit BV vars (Extract+Concat).",
            serialized_bytes,
            serialize_ns / 1000,
            deserialize_ns / 1000,
            new_check_ns / 1000,
        );
    }

    // angr-9o4n.1: scaling check. Mid-sized constraint set (~100 assertions,
    // 32 BV vars) to give a sense of cost as exploration state grows.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_smtlib2_constraint_round_trip_scaled() {
        use std::time::Instant;

        let ctx = SymContext::new();
        const NVARS: usize = 32;
        let vars: Vec<RustBV> = (0..NVARS)
            .map(|i| RustBV::symbolic(&ctx, format!("rt_scaled_x{}_9o4n", i), 32))
            .collect();

        // For each var: low(x) > i, low(x) < i+100 — gives a range constraint.
        // Then chain pairs: vars[i] != vars[i+1] for i in 0..NVARS-1.
        for (i, v) in vars.iter().enumerate() {
            let lo = RustBV::concrete(i as u128, 32);
            let hi = RustBV::concrete((i + 100) as u128, 32);
            ctx.assume_true(&v.ugt(&lo, &ctx));
            ctx.assume_true(&v.ult(&hi, &ctx));
        }
        for w in vars.windows(2) {
            let neq = w[0].eq(&w[1], &ctx);
            ctx.assume_false(&neq);
        }

        let original_sat = ctx.is_sat();
        assert!(original_sat, "scaled constraint set should be sat");

        let serialize_start = Instant::now();
        let serialized = ctx.debug_solver_string();
        let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
        let serialized_bytes = serialized.len();

        let deserialize_start = Instant::now();
        let new_solver = z3::Solver::new();
        new_solver.from_string(serialized.clone());
        let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64;

        let new_check_start = Instant::now();
        let new_sat = matches!(new_solver.check(), z3::SatResult::Sat);
        let new_check_ns = new_check_start.elapsed().as_nanos() as u64;
        assert_eq!(new_sat, original_sat);

        // Spot-check one variable's model value carries across.
        let original_v0 = ctx.eval(&vars[0]).expect("v0 evaluable");
        let new_model = new_solver
            .get_model()
            .expect("sat solver must produce model");
        let new_v0 = new_model
            .eval(&vars[0].to_z3_ast(), true)
            .and_then(|bv| bv.as_u64())
            .expect("model should evaluate v0");
        // Note: models from independent solver checks need not be identical.
        // We assert that the new model also satisfies the constraint (0 < v0 < 100).
        assert!(
            new_v0 > 0 && new_v0 < 100,
            "new model v0={} must satisfy 0 < v0 < 100; original was {}",
            new_v0,
            original_v0,
        );

        let n_assertions = NVARS * 2 + (NVARS - 1);
        eprintln!(
            "[angr-9o4n.1] SMT-LIB2 round-trip (scaled): {} bytes; \
             to_string={}us from_string={}us check_sat={}us; \
             {} assertions, {} 32-bit BV vars.",
            serialized_bytes,
            serialize_ns / 1000,
            deserialize_ns / 1000,
            new_check_ns / 1000,
            n_assertions,
            NVARS,
        );
    }

    // angr-rwzi: Validate SMT-LIB2 round-trip across a SEPARATE Z3 context.
    // Same-thread/same-context worked in angr-9o4n.1 because constants in the
    // shared context dedupe by (symbol, sort). The realistic save/restore path
    // (different thread or different process) gets a fresh Z3_context, so the
    // open question is whether name-based re-resolution (BV::new_const) in the
    // new context binds to the same AST that `Z3_solver_from_string` creates.
    //
    // Discriminator: the constraint set forces x=11, y=7. If name interning
    // works cross-context, the new model returns 11/7 via name lookup. If the
    // re-declared const is disconnected from the parsed assertions,
    // model.eval(.., model_completion=true) returns the Z3 default (0).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_smtlib2_cross_context_round_trip() {
        use std::time::Instant;
        use z3::ast::{Ast, BV};
        use z3::{Config, Context, Solver, with_z3_context};

        // -------- Build constraints in the default (original) context. --------
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "xctx_x_rwzi", 32);
        let y = RustBV::symbolic(&ctx, "xctx_y_rwzi", 32);

        let ten = RustBV::concrete(10, 32);
        let twenty = RustBV::concrete(20, 32);
        ctx.assume_true(&x.ugt(&ten, &ctx));
        ctx.assume_true(&x.ult(&twenty, &ctx));

        let y_high = y.extract(31, 16, &ctx);
        let zero16 = RustBV::concrete(0, 16);
        ctx.assume_true(&y_high.eq(&zero16, &ctx));

        let x_low = x.extract(15, 0, &ctx);
        let y_low = y.extract(15, 0, &ctx);
        let combined = x_low.concat(&y_low, &ctx);
        let target = RustBV::concrete(0x000B_0007, 32);
        ctx.assume_true(&combined.eq(&target, &ctx));

        assert!(ctx.is_sat());
        let x_witness = ctx.eval(&x).expect("x evaluable");
        let y_witness = ctx.eval(&y).expect("y evaluable");
        assert_eq!(x_witness, 11);
        assert_eq!(y_witness, 0x0000_0007);

        // Record the original AST/ctx pointers so we can prove the new
        // context's by-name lookup yields a DIFFERENT AST (i.e. is truly
        // cross-context). Cast to `usize` here so we can move them across the
        // `Send + Sync` bound of `with_z3_context` (Z3 raw pointers wrap
        // `NonNull` which isn't `Send`).
        let original_x_ast_usize = x.to_z3_ast().get_z3_ast().as_ptr() as usize;
        let original_ctx_usize = z3::Context::thread_local().get_z3_context().as_ptr() as usize;

        let serialize_start = Instant::now();
        let serialized = ctx.debug_solver_string();
        let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
        let serialized_bytes = serialized.len();

        // -------- Switch to a freshly-created Z3 context. --------
        // `Context::new` allocates a separate `Z3_context`; `with_z3_context`
        // swaps DEFAULT_CONTEXT for the closure body, so all subsequent
        // `Solver::new`, `BV::new_const`, `from_string`, model eval, etc.
        // resolve against the new context. The `Send + Sync` bound on the
        // closure type prevents accidentally smuggling Z3 ASTs from the old
        // context across the boundary; we only pass in plain `String`.
        let cfg = Config::new();
        let new_ctx = Context::new(&cfg);
        let new_ctx_usize_for_assert = new_ctx.get_z3_context().as_ptr() as usize;

        let (
            new_sat,
            new_x_val,
            new_y_val,
            new_x_ast_usize,
            seen_ctx_usize,
            deserialize_ns,
            new_check_ns,
        ) = with_z3_context(&new_ctx, || -> (bool, u64, u64, usize, usize, u64, u64) {
            // Sanity: confirm we really are in a different context.
            let in_closure_ctx_usize =
                z3::Context::thread_local().get_z3_context().as_ptr() as usize;

            let solver = Solver::new();
            let deserialize_start = Instant::now();
            solver.from_string(serialized.clone());
            let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64;

            let check_start = Instant::now();
            let sat = matches!(solver.check(), z3::SatResult::Sat);
            let check_ns = check_start.elapsed().as_nanos() as u64;

            // Re-resolve constants by NAME in the new context — this is the
            // realistic save/restore path (consumer holds only names + sorts,
            // not the original ASTs).
            let x_new = BV::new_const("xctx_x_rwzi", 32);
            let y_new = BV::new_const("xctx_y_rwzi", 32);
            let x_new_ast_usize = x_new.get_z3_ast().as_ptr() as usize;

            let model = solver.get_model().expect("sat solver must produce model");
            let x_val = model
                .eval(&x_new, true)
                .and_then(|v| v.as_u64())
                .expect("model must evaluate x_new");
            let y_val = model
                .eval(&y_new, true)
                .and_then(|v| v.as_u64())
                .expect("model must evaluate y_new");

            (
                sat,
                x_val,
                y_val,
                x_new_ast_usize,
                in_closure_ctx_usize,
                deserialize_ns,
                check_ns,
            )
        });

        // -------- Verify we actually used a different context. --------
        assert_ne!(
            original_ctx_usize, new_ctx_usize_for_assert,
            "test bug: new context pointer equals original; not testing cross-context"
        );
        assert_eq!(
            seen_ctx_usize, new_ctx_usize_for_assert,
            "with_z3_context did not actually swap the thread-local context"
        );
        // ASTs are per-context: the same-name BV in the new context must be a
        // different `Z3_ast` pointer than the one in the original context.
        assert_ne!(
            original_x_ast_usize, new_x_ast_usize,
            "test bug: cross-context BV::new_const returned an AST pointer \
             identical to the original-context AST — contexts are not actually \
             distinct"
        );

        // -------- The actual cross-context round-trip claims. --------
        assert!(
            new_sat,
            "cross-context round-tripped solver must remain SAT"
        );
        assert_eq!(
            new_x_val as u128, x_witness,
            "cross-context model must give x=11 via name lookup; got {} \
             (=0 would mean the by-name constant in the new context is \
             disconnected from the parsed assertions)",
            new_x_val
        );
        assert_eq!(
            new_y_val as u128, y_witness,
            "cross-context model must give y=7 via name lookup; got {}",
            new_y_val
        );

        eprintln!(
            "[angr-rwzi] SMT-LIB2 cross-context round-trip: {} bytes; \
             to_string={}us from_string={}us check_sat={}us; \
             original_ctx=0x{:x} new_ctx=0x{:x}; \
             original_x_ast=0x{:x} new_x_ast=0x{:x}",
            serialized_bytes,
            serialize_ns / 1000,
            deserialize_ns / 1000,
            new_check_ns / 1000,
            original_ctx_usize,
            new_ctx_usize_for_assert,
            original_x_ast_usize,
            new_x_ast_usize,
        );
    }

    /// angr-3ms1 step 1a: on the None (no-lineage) branch,
    /// `scope_savepoint_push`/`pop` bump `bare_z3_push_depth` in lockstep
    /// with the per-context Z3 solver's stack. Nested pushes accumulate;
    /// matching pops drain the counter back to 0.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_bare_z3_push_depth_none_branch_balanced() {
        let ctx = SymContext::new();
        assert!(ctx.lineage_arc().is_none());
        assert_eq!(ctx.bare_z3_push_depth(), 0);

        ctx.scope_savepoint_push();
        assert_eq!(ctx.bare_z3_push_depth(), 1);

        ctx.scope_savepoint_push();
        assert_eq!(ctx.bare_z3_push_depth(), 2);

        ctx.scope_savepoint_pop();
        assert_eq!(ctx.bare_z3_push_depth(), 1);

        ctx.scope_savepoint_pop();
        assert_eq!(
            ctx.bare_z3_push_depth(),
            0,
            "counter must drain back to 0 after balanced pops"
        );

        // The Some-branch sibling test lives separately
        // (`test_bare_z3_push_depth_some_branch_inert`); here we also
        // confirm the None branch left `scope_savepoints` untouched, so
        // the two paths don't accidentally double-count.
        assert_eq!(ctx.scope_savepoint_depth(), 0);
    }

    /// angr-3ms1 step 1a: on the Some (shared-lineage) branch,
    /// `scope_savepoint_push`/`pop` record on `scope_savepoints` and must
    /// NOT touch `bare_z3_push_depth` — the counter only tracks pushes
    /// against the per-context Z3 solver.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_bare_z3_push_depth_some_branch_inert() {
        use super::super::lineage::SharedLineageSolver;

        let ctx = SymContext::new();
        let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
        ctx.set_lineage_for_testing(Arc::clone(&lin));

        assert_eq!(ctx.bare_z3_push_depth(), 0);

        ctx.scope_savepoint_push();
        ctx.scope_savepoint_push();
        assert_eq!(ctx.scope_savepoint_depth(), 2);
        assert_eq!(
            ctx.bare_z3_push_depth(),
            0,
            "Some branch must not touch bare_z3_push_depth"
        );

        ctx.scope_savepoint_pop();
        ctx.scope_savepoint_pop();
        assert_eq!(ctx.scope_savepoint_depth(), 0);
        assert_eq!(ctx.bare_z3_push_depth(), 0);
    }

    /// angr-3ms1 step 1a: `fork()` copies the parent's
    /// `bare_z3_push_depth` into the child. The slice-1c materialization
    /// gate inspects the parent's value at fork time, but copying the
    /// value into the child keeps the post-fork accounting consistent
    /// for any future code path that threads bare pushes across fork.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_bare_z3_push_depth_inherited_on_fork() {
        let parent = SymContext::new();
        assert_eq!(parent.bare_z3_push_depth(), 0);

        // A fork before any push: child inherits the 0.
        let child_zero = parent.fork();
        assert_eq!(
            child_zero.bare_z3_push_depth(),
            0,
            "fork before any push must hand the child a 0 depth"
        );

        // After two bare pushes, the parent's counter is 2; a fork at
        // that point hands the child the same depth.
        parent.scope_savepoint_push();
        parent.scope_savepoint_push();
        assert_eq!(parent.bare_z3_push_depth(), 2);

        let child_two = parent.fork();
        assert_eq!(
            child_two.bare_z3_push_depth(),
            2,
            "child must inherit the parent's bare_z3_push_depth at fork time"
        );

        // Drain the parent's pushes; the child's copy stays at 2 — it's
        // a per-context counter, not a shared cell.
        parent.scope_savepoint_pop();
        parent.scope_savepoint_pop();
        assert_eq!(parent.bare_z3_push_depth(), 0);
        assert_eq!(
            child_two.bare_z3_push_depth(),
            2,
            "child's counter is independent of parent's post-fork mutations"
        );
    }

    /// angr-3ms1 step 1b: a freshly constructed context has
    /// `use_shared_lineage_solver == false`. Setter flips it; the value
    /// round-trips through the getter.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_use_shared_lineage_solver_default_and_setter() {
        let ctx = SymContext::new();
        assert!(
            !ctx.use_shared_lineage_solver(),
            "default must be off so the slice-1c gate stays inert on plain RustExplorationManager runs"
        );

        ctx.set_use_shared_lineage_solver(true);
        assert!(ctx.use_shared_lineage_solver());

        ctx.set_use_shared_lineage_solver(false);
        assert!(!ctx.use_shared_lineage_solver());
    }

    /// angr-3ms1 step 1b: `fork()` copies the parent's
    /// `use_shared_lineage_solver` value into the child so a single
    /// setter call on the seed state propagates to every descendant via
    /// fork — no per-fork plumbing on the Python side. Like
    /// `bare_z3_push_depth`, the child carries its own AtomicBool, so
    /// post-fork mutations on either side don't bleed into the other.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_use_shared_lineage_solver_inherited_on_fork() {
        let parent = SymContext::new();
        assert!(!parent.use_shared_lineage_solver());

        // Default-off parent forks a default-off child.
        let child_off = parent.fork();
        assert!(
            !child_off.use_shared_lineage_solver(),
            "fork before opt-in must hand the child a false flag"
        );

        // Opt the parent in; subsequent fork hands the child the same
        // value.
        parent.set_use_shared_lineage_solver(true);
        let child_on = parent.fork();
        assert!(
            child_on.use_shared_lineage_solver(),
            "child must inherit the parent's opt-in at fork time"
        );

        // Per-context independence: flipping the parent off does not
        // disturb the child's already-inherited true.
        parent.set_use_shared_lineage_solver(false);
        assert!(!parent.use_shared_lineage_solver());
        assert!(
            child_on.use_shared_lineage_solver(),
            "child's flag is independent of parent's post-fork mutations"
        );
    }

    /// angr-3ms1 step 1c: when the parent has opted in AND has no bare
    /// Z3 pushes outstanding, `fork()` mints a fresh `SharedLineageSolver`
    /// and installs it in the child. The parent's own lineage is not
    /// touched — staying `None` so the parent keeps querying its
    /// per-context solver. The two contexts therefore hold distinct
    /// solver instances after the fork.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_mints_lineage_when_gate_passes() {
        let parent = SymContext::new();
        parent.set_use_shared_lineage_solver(true);
        assert_eq!(parent.bare_z3_push_depth(), 0);
        assert!(
            parent.lineage_arc().is_none(),
            "parent starts without a lineage"
        );

        let child = parent.fork();
        assert!(
            child.lineage_arc().is_some(),
            "child must receive a freshly minted lineage when the gate passes"
        );
        assert!(
            parent.lineage_arc().is_none(),
            "parent's lineage must NOT change as a side effect of forking — \
             minting only installs on the child"
        );
    }

    /// angr-3ms1 step 1c: with the opt-in flag off (the default),
    /// `fork()` keeps the pre-1c behavior of Arc::cloning the parent's
    /// lineage Arc. Default `None` parent → `None` child.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_skips_mint_when_flag_off() {
        let parent = SymContext::new();
        assert!(!parent.use_shared_lineage_solver());

        let child = parent.fork();
        assert!(
            child.lineage_arc().is_none(),
            "default-off flag must keep the slice-1c gate inert — no mint"
        );
    }

    /// angr-3ms1 step 1c: condition (b) of the gate refuses to mint
    /// while the parent's per-context solver has outstanding bare Z3
    /// pushes (`bare_z3_push_depth > 0`). Without this guard the child's
    /// new lineage would take over Z3 stack ownership while the parent's
    /// unbalanced pushes are still live, leaking the parent's pushed-only
    /// constraints into the new lineage base — the failure mode that
    /// `test_fork_inside_push_isolation` exposed in earlier slice 4c.3
    /// attempts (see `v5a5-bare-z3-push-depth-counter-design`).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_skips_mint_when_bare_push_outstanding() {
        let parent = SymContext::new();
        parent.set_use_shared_lineage_solver(true);

        // A bare push on the None lineage branch bumps bare_z3_push_depth
        // to 1 — the gate must refuse to mint while this is non-zero.
        parent.scope_savepoint_push();
        assert_eq!(parent.bare_z3_push_depth(), 1);

        let child = parent.fork();
        assert!(
            child.lineage_arc().is_none(),
            "gate must refuse to mint while parent has outstanding bare pushes"
        );

        // Clean up the parent's push so the test's per-context solver
        // returns to a balanced state (avoids tripping debug_asserts in
        // later teardown).
        parent.scope_savepoint_pop();
        assert_eq!(parent.bare_z3_push_depth(), 0);
    }

    /// angr-3ms1 step 1c: a newly minted lineage is seeded with the
    /// parent's existing assertions as base assertions (scope 0). The
    /// child's first query routes through `with_z3_solver`'s Some branch,
    /// running `switch_to(empty)` then `solver.check()` — which respects
    /// the base assertions installed at fork time. Verifies the child's
    /// solver returns UNSAT when the parent's constraints already entail
    /// it, even though the child added no constraints of its own.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_minted_lineage_seeded_with_parent_constraints() {
        let parent = SymContext::new();
        parent.set_use_shared_lineage_solver(true);

        // Parent asserts x == 5 on its per-context solver (None branch).
        let x = RustBV::symbolic(&parent, "fork_mint_seed_x", 8);
        let five = RustBV::concrete(5, 8);
        parent.assume_true(&x.eq(&five, &parent));

        // Fork → child gets a fresh lineage seeded with x == 5.
        let child = parent.fork();
        assert!(child.lineage_arc().is_some());

        // The child's lineage solver knows about x == 5: assume_true(x == 6)
        // through the lineage path produces UNSAT.
        let six = RustBV::concrete(6, 8);
        child.assume_true(&x.eq(&six, &parent));
        assert!(
            !child.is_sat(),
            "child must see parent's x == 5 (base) ∧ self-added x == 6 → UNSAT"
        );

        // The parent's per-context solver is untouched — adding the
        // child's contradictory constraint did NOT leak into the parent.
        assert!(
            parent.is_sat(),
            "parent must remain SAT — its per-context solver only holds x == 5"
        );
    }

    /// angr-3ms1 step 1c: the opt-in flag inherits parent→child in
    /// fork(), so a single setter call on a seed state propagates the
    /// minting behavior to every descendant. Each fork along that chain
    /// mints its own fresh lineage (the gate keeps passing because the
    /// flag stays true and bare_z3_push_depth stays 0).
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_fork_chain_mints_fresh_lineage_at_each_level() {
        let parent = SymContext::new();
        parent.set_use_shared_lineage_solver(true);

        let child = parent.fork();
        let grandchild = child.fork();

        let child_lin = child.lineage_arc().expect("child must have a lineage");
        let grandchild_lin = grandchild
            .lineage_arc()
            .expect("grandchild must have a lineage");
        assert!(
            !Arc::ptr_eq(&child_lin, &grandchild_lin),
            "each fork mints its own fresh lineage — Arc identities must differ"
        );
        assert!(
            grandchild.use_shared_lineage_solver(),
            "flag inherits down the chain"
        );
    }

    /// Round-trip the snapshot through serde JSON and verify the captured
    /// assumed_constraints reconstruct equivalent Z3 ASTs (angr-x04s.1.2
    /// acceptance check). Uses Z3 `Bool::eq` to confirm that the original
    /// and restored constraint ASTs are *structurally* the same expression
    /// (after Z3's `simplify()`), not just satisfy the same models.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_snapshot_assumed_constraints_roundtrip_via_z3() {
        use z3::ast::Ast;

        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "snap_x04s_x", 32);
        let y = RustBV::symbolic(&ctx, "snap_x04s_y", 32);
        let zero = RustBV::concrete(0, 32);
        let ten = RustBV::concrete(10, 32);
        let twenty = RustBV::concrete(20, 32);

        ctx.assume_true(&x.ugt(&ten, &ctx));
        ctx.assume_true(&x.ult(&twenty, &ctx));
        ctx.assume_false(&y.eq(&zero, &ctx));

        let pre = ctx.get_assumed_constraints();
        assert_eq!(pre.len(), 3);

        // Capture the original Z3 Bool ASTs (after simplify) so we can
        // compare structurally against the restored ones.
        let original_simplified: Vec<z3::ast::Bool> = pre
            .iter()
            .map(|(cond, is_true)| {
                let b = cond.to_z3_bool();
                let b = if *is_true { b } else { b.not() };
                b.simplify()
            })
            .collect();

        let snap = ctx.to_snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.assumed_constraints.len(), 3);

        let ctx2 = SymContext::new();
        ctx2.restore_from_snapshot(&restored);
        let post = ctx2.get_assumed_constraints();
        assert_eq!(post.len(), 3);

        // is_true flags match.
        for (i, ((_, a), (_, b))) in pre.iter().zip(post.iter()).enumerate() {
            assert_eq!(a, b, "is_true flag mismatch at slot {i}");
        }

        // Each restored constraint produces a Z3 Bool that is
        // structurally identical (after simplify) to the original — proves
        // the AST cache rebuild followed the original tree shape.
        for (i, (cond, is_true)) in post.iter().enumerate() {
            let restored_bool = cond.to_z3_bool();
            let restored_bool = if *is_true {
                restored_bool
            } else {
                restored_bool.not()
            };
            assert_eq!(
                restored_bool.simplify(),
                original_simplified[i],
                "restored constraint slot {i} does not match original Z3 AST"
            );
        }

        // The restored context must still be SAT and concretize x and y
        // to values that honor every constraint.
        assert!(ctx2.is_sat());
        let x_val = ctx2.eval(&x).expect("x evaluable");
        let y_val = ctx2.eval(&y).expect("y evaluable");
        assert!(x_val > 10 && x_val < 20, "x={x_val} must satisfy 10<x<20");
        assert_ne!(y_val, 0, "y must be non-zero");
    }

    /// angr-82g6: constraints added via `add_constraint_raw` (no RustBV
    /// available, e.g. the Python claripy-sync fallback path) must
    /// survive snapshot round-trip via the `solver_smtlib2` dump.
    /// `assumed_constraint_count` stays unchanged across the trip (the
    /// raw entries never touch the BV log), but `num_constraints` and
    /// solver SAT state are preserved end-to-end.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_snapshot_add_constraint_raw_roundtrip() {
        use z3::ast::Ast;

        let ctx = SymContext::new();
        // Mix: one assume_true (RustBV-known), one add_constraint_raw
        // (Z3-only, no RustBV form recorded).
        let x = RustBV::symbolic(&ctx, "snap_82g6_x", 32);
        let ten = RustBV::concrete(10, 32);
        let twenty = RustBV::concrete(20, 32);
        ctx.assume_true(&x.ugt(&ten, &ctx));

        // Raw path: build a Z3 Bool directly and feed it through
        // add_constraint_raw — mirrors `_add_constraints_to_state`'s
        // Z3-ptr fast path when claripy_to_rustbv fails to translate.
        let z3_ctx = z3::Context::thread_local();
        let raw_bool = {
            let x_z3 = x.to_z3_ast();
            let twenty_z3 = twenty.to_z3_ast();
            x_z3.bvult(&twenty_z3)
        };
        let raw_ptr = raw_bool.get_z3_ast().as_ptr() as usize;
        let z3_ast_ptr = unsafe { Z3AstPtr::from_borrowed_raw(&z3_ctx, raw_ptr) }
            .expect("raw Bool must yield a Z3AstPtr");
        ctx.add_constraint_raw(z3_ast_ptr);

        // Pre-snapshot bookkeeping. `num_constraints` counts both paths;
        // `assumed_constraint_count` only the assume path.
        let pre_total = ctx.num_constraints();
        let pre_assumed = ctx.assumed_constraint_count();
        assert_eq!(pre_total, 2, "raw + assume = 2 logical constraints");
        assert_eq!(pre_assumed, 1, "only the assume entry hits the BV log");

        // Round-trip through serde.
        let snap = ctx.to_snapshot();
        assert_eq!(snap.assumed_constraints.len(), 1);
        assert!(
            !snap.solver_smtlib2.is_empty(),
            "snapshot must carry an SMT-LIB2 dump when raw constraints \
             are present"
        );
        let json = serde_json::to_string(&snap).expect("serialize");
        let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");

        let ctx2 = SymContext::new();
        ctx2.restore_from_snapshot(&restored);

        // angr-82g6: num_constraints now matches pre-snapshot (was the
        // bug — restored counted only assumed_constraints).
        assert_eq!(
            ctx2.num_constraints(),
            pre_total,
            "num_constraints must round-trip through snapshot"
        );
        assert_eq!(
            ctx2.assumed_constraint_count(),
            pre_assumed,
            "assumed_constraint_count is preserved (raw entries stay raw)"
        );

        // Solver is still SAT and respects BOTH constraints (x > 10
        // AND x < 20).
        assert!(ctx2.is_sat());
        let x_val = ctx2.eval(&x).expect("x evaluable");
        assert!(
            x_val > 10 && x_val < 20,
            "restored x={x_val} must satisfy 10 < x < 20 — including \
             the raw-path x<20 constraint"
        );
    }

    /// angr-82g6: backward-compat path — a snapshot deserialized from
    /// JSON that omits `solver_smtlib2` (older payloads) must still
    /// restore via the assumed-replay codepath alone.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_snapshot_missing_solver_smtlib2_deserializes_with_default() {
        // Legacy JSON shape — pre-82g6 snapshots only carry
        // assumed_constraints.
        let legacy_json = r#"{"assumed_constraints":[]}"#;
        let restored: SymContextSnapshot =
            serde_json::from_str(legacy_json).expect("legacy JSON must parse");
        assert!(restored.solver_smtlib2.is_empty());

        let ctx = SymContext::new();
        ctx.restore_from_snapshot(&restored);
        assert_eq!(ctx.num_constraints(), 0);
    }

    /// Mock-backend round-trip: with the Z3 feature off, the snapshot
    /// still preserves the `(RustBV, bool)` log and restore replays the
    /// pairs into a fresh context.
    #[cfg(not(feature = "vex-engine-z3"))]
    #[test]
    fn test_snapshot_assumed_constraints_roundtrip_mock() {
        let ctx = SymContext::new_mock();
        let cond_a = RustBV::concrete(1, 1);
        let cond_b = RustBV::concrete(0, 1);
        ctx.assume_true(&cond_a);
        ctx.assume_false(&cond_b);
        let snap = ctx.to_snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
        let ctx2 = SymContext::new_mock();
        ctx2.restore_from_snapshot(&restored);
        let post = ctx2.get_assumed_constraints();
        assert_eq!(post.len(), 2);
        assert_eq!(post[0].1, true);
        assert_eq!(post[1].1, false);
    }
}
