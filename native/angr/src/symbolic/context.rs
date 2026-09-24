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
//! introduces several cross-cutting invariants that cut across the slice files
//! this module was split into (angr-a2br: `bv_id_ops.rs`, `constraint_ops.rs`,
//! `solving_ops.rs`, `transaction_ops.rs`, `snapshot_fork_ops.rs`,
//! `lineage_ops.rs`; angr-fs8kb.35: `context_snapshot.rs`,
//! `local_constraints.rs`, `context_mock.rs`, `merge_instrument.rs` — see the
//! file index in `symbolic/mod.rs` for what landed where). The splits preserved
//! all of them, and any further refactor must too;
//! because no single slice owns them, they stay documented here beside the
//! struct definition. Invariants that have a bd memory carrying the long-form
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
//!   `Arc::clone` would give the child a stale base. Regression guards, one
//!   per gate: `context_tests_smtlib2_snapshot::test_fork_skips_mint_when_flag_off`
//!   (a), `::test_fork_skips_mint_when_bare_push_outstanding` (b), and
//!   `super::lineage_tests::test_fork_drops_lineage_when_dismantled` (c) —
//!   the last lives beside the sampler tests because it mutates the global
//!   dismantle flag and must hold their serializing lock. The passing case is
//!   `context_tests_smtlib2_snapshot::test_fork_mints_lineage_when_gate_passes`.
//! - **fork-freeze under push** (angr-c7xno.75):
//!   [`fork`](SymContext::fork) only drains local→shared in place when NO
//!   bare push scope is open (`bare_local_savepoints` is empty). Inside an
//!   open scope a `pop()` truncates `local` back to its pre-push length;
//!   draining would leak popped constraints into `shared`, which has no
//!   removal path — a later fork then inherits a constraint the parent
//!   itself no longer believes, and gets spuriously pruned as UNSAT. The
//!   child still sees the in-scope constraints (freeze returns the merged
//!   set either way); only the parent's ability to retract them survives.
//!   Regression guard:
//!   `context_tests_constraints::test_fork_under_bare_push_does_not_freeze_popped_constraints`.
//! - **Z3 construction canonicalization** (`invariant-z3-construction-canonicalization`):
//!   Z3 hash-cons applies at construction time, but commutative operands
//!   are NOT normalized (`mk_bvadd(x, y)` and `mk_bvadd(y, x)` produce
//!   distinct AST pointers). Concrete numerals and named symbol leaves ARE
//!   canonical. Regression guard:
//!   `super::value::value_zext_cmp_tests::z3_already_dedupes_structurally_equal_rustbv_trees`.
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
//! solver behind guest-derived constraints; the one surviving non-test
//! `expect` in [`SymContext::solver`] reads back a lazily-materialized solver
//! the same function just stored under the same held guard. (The deny also
//! reaches the `#[cfg(test)]` children — `merge_instrument` and the
//! `context_tests/` files — which opt out wholesale via a reasoned
//! `#[allow]`, so count non-test sites only when checking that claim.)
#![deny(clippy::unwrap_used, clippy::expect_used)]

// The Z3-only half of the std imports: every consumer of `Cell`/`RefCell`
// (`sat_cache`, `model_cache`), `AtomicU32` (`timeout_ms`) and `Ordering` is
// itself behind `#[cfg(feature = "vex-engine-z3")]`, so importing them
// unconditionally warns in the no-z3 combos `make check-no-z3` gates
// (angr-sqfj8.139).
#[cfg(feature = "vex-engine-z3")]
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::Mutex;

use super::RustBV;
use super::local_constraints::LocalConstraints;
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

/// Default Z3 solver timeout in milliseconds.
///
/// 30 seconds — chosen to match claripy's historical default and to cap the
/// occasional Z3 outlier on bimodal-SAT benches. Overridable per-state via
/// `RustExplorationManager::set_solver_timeout`.
pub const DEFAULT_SOLVER_TIMEOUT_MS: u32 = 30_000;

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

    /// No-Z3 analogue of [`bare_local_savepoints`](Self::bare_local_savepoints)
    /// (angr-c7xno.100).
    ///
    /// Each entry is `local_constraints.assumed.len()` at the moment of a
    /// `push()` — the only local log this build has, since `z3_assertions` /
    /// `non_bv_assertions` / `dedup_set` are all Z3-gated out of
    /// [`LocalConstraints`]. `pop()` truncates back to it, so the mock arm
    /// honours the same "constraints added inside a scope do not survive the
    /// matching pop" contract, and its length is the scope depth
    /// [`try_pop`](Self::try_pop) needs to refuse an under-pop. Before this
    /// existed the mock `push()`/`pop()` were no-ops and `try_pop()` returned
    /// `true` unconditionally, so `RustSolverContext::pop` accepted an
    /// unbalanced pop that the Z3 arm rejects with a `ValueError`.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub(super) mock_scope_savepoints: Mutex<Vec<usize>>,
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
            Z3_MATERIALIZE_TIME_NS.fetch_add(crate::elapsed_ns(start), Ordering::Relaxed);
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

impl Default for SymContext {
    fn default() -> Self {
        Self::new()
    }
}

// a2br.2.11: SymContext unit tests, split by theme out of the former
// monolithic `context_tests.rs` (2340 lines) to keep each file under the
// <2000-line epic acceptance criterion. Declared as direct children of
// `context` so `use super::*` reaches `context`'s private items.
//
// angr-fs8kb.35: `merge_instrument` — the guarded-assertion counter
// `SymContext::merge` bumps and `context_tests_merge_prefix` reads — moved to
// its own file but stays a child module of `context` via `#[path]`, so every
// caller's path is unchanged.
#[cfg(test)]
#[path = "merge_instrument.rs"]
pub(crate) mod merge_instrument;

// Gated on vex-engine-z3 (bd angr-cagbn): every test here drives
// `SymContext::add_constraint` / Z3AstPtr, which only exist with z3. Keeps the
// no-z3 nightly `cargo test` combos compiling; default build runs them all.
// The *file* paths below are nested under context_tests/, but the modules they
// create are FLAT siblings of each other directly under `context` — there is no
// `context_tests` parent module. Cite a test as
// `context_tests_constraints::test_name`, never `context_tests::constraints::…`
// (angr-6cp06.48 found four citations that took the file layout for the module
// layout; neither grep nor rust-analyzer resolves that form).
test_submod!(z3 "context_tests/constraints.rs" => context_tests_constraints);
test_submod!("context_tests/lineage.rs" => context_tests_lineage);
test_submod!("context_tests/merge_prefix.rs" => context_tests_merge_prefix);
test_submod!("context_tests/merge_shape_spike.rs" => context_tests_merge_shape_spike);
test_submod!("context_tests/smtlib2_snapshot.rs" => context_tests_smtlib2_snapshot);
test_submod!("context_tests/solver_queries.rs" => context_tests_solver_queries);
