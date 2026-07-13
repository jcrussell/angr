//! Shared-lineage Z3 solver with push/pop scope tracking (angr-v5a5 spike).
//!
//! Implements the data structures and core mechanics for Option A from the
//! angr-hk7k research spike (see memory `invariant-hk7k-design-options`).
//! The plan: all states descended from a common ancestor share one Z3
//! [`z3::Solver`] wrapped in an `Arc<Mutex<SharedLineageSolver>>`. The
//! shared solver carries the lineage's base assertions at scope 0; each
//! state's local-diff is tracked as a [`ScopePath`] of frames pushed on
//! top of the base.
//!
//! On context switch (when a different state in the lineage issues a
//! query), [`SharedLineageSolver::switch_to`] computes the longest common
//! prefix between the currently-loaded path and the target path by frame
//! `id`, pops the divergent suffix off the solver, and pushes the target
//! tail. Cost is `O(local_diff)`, not `O(total_assertions)` — the win the
//! hk7k research spike targets.
//!
//! ## Status
//!
//! This module is **not yet wired into [`crate::symbolic::SymContext`]**.
//! This is the v5a5 skeleton: structs, switch_to mechanics, telemetry
//! counters, and unit tests verifying the invariants. The next slice will
//! teach `SymContext::fork()` to thread a `SharedLineageSolver` through
//! the lineage and replace the per-state lazy-materialize path. Keeping
//! the skeleton standalone keeps this iteration's blast radius bounded
//! (no behavior change in any existing call site).
//!
//! ## Invariants
//!
//! - `SharedLineageSolver::loaded_path.len()` always equals the number of
//!   `z3.push()` frames currently outstanding on the inner solver.
//! - Every frame in `loaded_path` has been asserted on `z3` under its own
//!   `push()` frame, so `pop(1)` removes exactly that frame's assertion.
//! - Base assertions installed via [`SharedLineageSolver::assert_base`]
//!   sit at scope 0 — they are never push/pop scoped.
//! - [`FrameId`]s are globally unique and monotonically assigned, so two
//!   `ScopePath`s share a prefix iff their first N frames have identical
//!   ids — the cheap, side-effect-free prefix comparison the design hinges
//!   on.
//!
//! ## Cross-cutting design memories
//!
//! Recall via `bd recall <key>`:
//!
//! - `invariant-hk7k-design-options` — why Option A (shared-lineage push/pop
//!   tracking) was picked over solver-translate / fork-boundary push/pop /
//!   per-state-context strategies.
//! - `v5a5-frame-id-design` — why prefix matching uses a monotonic
//!   [`FrameId`] minted at constraint-add time, not `Arc::ptr_eq` on the
//!   underlying RustBV. Identity stays stable across fork boundaries.
//! - `invariant-v5a5-lineage-mutex-shape` — why `SymContext.lineage` is
//!   typed `Mutex<Option<Arc<Mutex<SharedLineageSolver>>>>` (the outer
//!   Mutex makes `fork(&self)` legal).
//! - `avoid-z3-parallel-enable` — `parallel.enable=true` is
//!   correctness-breaking; do NOT set it in `build_solver_params`.
//! - `avoid-full-lineage-teardown` — the angr-0dgq teardown variant is a
//!   net loss vs the v5ht "simple" dismantle (which only suppresses
//!   future mints). Do NOT retry without first making per-context solver
//!   rebuild incremental.
//! - `avoid-dfs-coupling-for-shared-lineage` — do NOT default this on for
//!   `strategy='dfs'`. Workload shape, not strategy, predicts the win.
//! - `v5ht-sampler-tick-bottleneck` — the sampler hook MUST be at the TOP
//!   of `run_loop` (before any callback-path early-return) so
//!   callback-heavy workloads still tick. See the comment at the call
//!   site in `crate::exploration::run_loop`.
//! - `v5ht-threshold-justification-2026-05-25` — 35% hot threshold is
//!   calibrated on N=4 workloads; widen the dataset before changing it.

#![cfg(feature = "vex-engine-z3")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use z3::ast::Bool;

/// Globally-unique scope-frame id. Assigned monotonically when a state
/// adds a constraint. Sibling states descended from a common ancestor
/// share the same id for any constraint added by the ancestor — the basis
/// for [`switch_to`](SharedLineageSolver::switch_to)'s common-prefix
/// optimization.
pub type FrameId = u64;

static NEXT_FRAME_ID: AtomicU64 = AtomicU64::new(1);

/// Mint a fresh globally-unique [`FrameId`].
pub fn mint_frame_id() -> FrameId {
    NEXT_FRAME_ID.fetch_add(1, Ordering::Relaxed)
}

/// One constraint frame on a state's scope path.
///
/// `is_true` mirrors `SymContext::assumed_constraints`'s assumed/negated
/// flag; `z3_assertion` is the already-derived Z3 [`Bool`] (i.e. the
/// `cond.to_z3_bool()` for `is_true == true`, or its `.not()` for
/// `is_true == false`). Push-time logic does not need to re-derive the
/// negation, keeping `switch_to` purely Z3-side bookkeeping.
#[derive(Clone)]
pub struct ScopeFrame {
    pub id: FrameId,
    pub is_true: bool,
    pub z3_assertion: Bool,
}

impl ScopeFrame {
    /// Construct a fresh frame with a globally-unique id.
    pub fn new(is_true: bool, z3_assertion: Bool) -> Self {
        ScopeFrame {
            id: mint_frame_id(),
            is_true,
            z3_assertion,
        }
    }
}

/// Ordered sequence of scope frames a state has accumulated since its
/// lineage's base. State A and state B share a prefix iff the first N
/// frames have identical [`FrameId`]s — they descend from the same fork
/// point.
pub type ScopePath = Vec<ScopeFrame>;

// =============================================================================
// Telemetry
// =============================================================================
// Lightweight atomic counters mirroring the Z3_* counter style in
// `context.rs`. They are read by the unit tests for now; the integration
// patch will route them out through `get_solver_stats()`.

static LINEAGE_SWITCH_COUNT: AtomicU64 = AtomicU64::new(0);
static LINEAGE_SWITCH_HOT_COUNT: AtomicU64 = AtomicU64::new(0);
static LINEAGE_SWITCH_FAST_PATH_COUNT: AtomicU64 = AtomicU64::new(0);
static LINEAGE_PUSH_COUNT: AtomicU64 = AtomicU64::new(0);
static LINEAGE_POP_COUNT: AtomicU64 = AtomicU64::new(0);

// =============================================================================
// Runtime thrash detection (angr-v5ht)
// =============================================================================
// Global kill switch for fork-time lineage minting. Set when
// `sample_for_thrash` detects the hot-cache hit ratio has fallen below
// the configured threshold over a sampling window. The fork-gate in
// `SymContext::fork` consults `is_lineage_dismantled()` and, when true,
// falls through to the pre-lineage behavior (Arc-clone the parent's
// lineage, which is None in production without opt-in). Existing
// in-flight lineage Arcs are NOT torn down — the SIMPLE variant of the
// design (see angr-v5ht bead description).

static LINEAGE_DISMANTLED: AtomicBool = AtomicBool::new(false);
static LINEAGE_DISMANTLE_COUNT: AtomicU64 = AtomicU64::new(0);
// Counts every `sample_for_thrash` invocation that lands on a sample
// step (i.e. is not short-circuited by the off-step / interval-zero /
// already-dismantled fast paths). Lets us distinguish 'sampler never
// ran' from 'sampler ran but never crossed the threshold' when
// debugging benchmark misses.
static LINEAGE_SAMPLE_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
// Counts every sample call that crossed the `min_switches` gate and
// produced a verdict (whether dismantle or keep-on).
static LINEAGE_SAMPLE_DECISION_COUNT: AtomicU64 = AtomicU64::new(0);
// Counters tracking the value at the last sampling window boundary.
// Used to compute the per-window delta in `sample_for_thrash`.
static LAST_SAMPLE_SWITCH_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_SAMPLE_HOT_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_SAMPLE_STEP: AtomicU64 = AtomicU64::new(0);

/// Return whether the runtime thrash detector has dismantled lineage
/// minting. When true, `SymContext::fork` takes the pre-lineage path.
pub fn is_lineage_dismantled() -> bool {
    LINEAGE_DISMANTLED.load(Ordering::Relaxed)
}

/// Force the dismantle flag to `v`. Tests use this to exercise both
/// states without going through the sampler. Production code should
/// reach the dismantled state only via [`sample_for_thrash`].
pub fn set_lineage_dismantled(v: bool) {
    LINEAGE_DISMANTLED.store(v, Ordering::Relaxed);
}

/// Sample the lineage counters and flip [`is_lineage_dismantled`] on if
/// the hot-cache hit ratio over the current window is below
/// `hot_threshold_pct`. Returns true iff the dismantle flag transitioned
/// from false to true on this call.
///
/// `step` is the current step counter; the sampler only does work on
/// steps where `step.is_multiple_of(sample_interval)`. `min_switches` is
/// the minimum number of lineage_switch events that must have occurred
/// in the window before the threshold can fire — protects against very
/// early dismantle on tiny windows. Suggested defaults (see angr-v5ht):
/// `sample_interval=10`, `min_switches=20`, `hot_threshold_pct=35`.
///
/// Idempotent once dismantled: subsequent calls see the flag already
/// set and exit cheaply without re-sampling. Cheap on the off-step
/// fast path (one mod, one branch).
pub fn sample_for_thrash(
    tick: u64,
    sample_interval: u64,
    min_switches: u64,
    hot_threshold_pct: u32,
) -> bool {
    if LINEAGE_DISMANTLED.load(Ordering::Relaxed) {
        return false;
    }
    if sample_interval == 0 || tick == 0 || !tick.is_multiple_of(sample_interval) {
        return false;
    }
    LINEAGE_SAMPLE_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    let switch_now = LINEAGE_SWITCH_COUNT.load(Ordering::Relaxed);
    let hot_now = LINEAGE_SWITCH_HOT_COUNT.load(Ordering::Relaxed);
    let switch_prev = LAST_SAMPLE_SWITCH_COUNT.load(Ordering::Relaxed);
    let hot_prev = LAST_SAMPLE_HOT_COUNT.load(Ordering::Relaxed);

    let switch_delta = switch_now.saturating_sub(switch_prev);
    let hot_delta = hot_now.saturating_sub(hot_prev);

    // Accumulate across sample windows until min_switches is met —
    // workloads with low switch volume per N-tick window (e.g.
    // google2016_unbreakable_0: ~110 switches over the whole run)
    // would otherwise never collect enough samples to fire. Reset the
    // window snapshot ONLY when a decision is made; before that, keep
    // accumulating into the same window across multiple sampler calls.
    if switch_delta < min_switches {
        LAST_SAMPLE_STEP.store(tick, Ordering::Relaxed);
        return false;
    }

    LINEAGE_SAMPLE_DECISION_COUNT.fetch_add(1, Ordering::Relaxed);
    LAST_SAMPLE_SWITCH_COUNT.store(switch_now, Ordering::Relaxed);
    LAST_SAMPLE_HOT_COUNT.store(hot_now, Ordering::Relaxed);
    LAST_SAMPLE_STEP.store(tick, Ordering::Relaxed);

    // hot_delta * 100 < threshold * switch_delta -> ratio below threshold.
    // u64 multiplication is safe at any plausible counter scale: a
    // workload exceeding 2^57 switches per window dwarfs anything
    // observed in the v5ht analysis.
    let threshold = hot_threshold_pct as u64;
    if hot_delta.saturating_mul(100) < threshold.saturating_mul(switch_delta) {
        LINEAGE_DISMANTLED.store(true, Ordering::Relaxed);
        LINEAGE_DISMANTLE_COUNT.fetch_add(1, Ordering::Relaxed);
        return true;
    }
    false
}

/// Internal tick counter for [`tick_and_sample_for_thrash`]: bumped
/// once per call. Lets the sampler fire on a fixed cadence of
/// invocations, independent of the manager-level `self.steps` counter
/// (which only advances on a state-step without a Python-callback
/// return — workloads heavy in SimProcedure callbacks like
/// google2016_unbreakable_0 never bump it and would never trigger
/// step-count-based sampling).
static SAMPLER_TICK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Convenience wrapper that bumps an internal tick counter and calls
/// [`sample_for_thrash`] with the bumped value. Hook from the run loop
/// (one call per for-loop iteration, BEFORE any early returns) so the
/// sampler sees a monotonic per-iteration clock.
pub fn tick_and_sample_for_thrash(
    sample_interval: u64,
    min_switches: u64,
    hot_threshold_pct: u32,
) -> bool {
    let tick = SAMPLER_TICK_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    sample_for_thrash(tick, sample_interval, min_switches, hot_threshold_pct)
}

// =============================================================================
// Per-lineage-tree census (angr-g1fev)
// =============================================================================
// The v5ht sampler decides ONCE, GLOBALLY, from a >=20-switch window
// summed over every tree. The follow-up proposal is to decide per tree
// instead. That is only buildable if individual trees actually accumulate
// enough switches to support a decision window — these counters measure
// exactly that, and whether a tree's early hot ratio predicts its later
// one (the rev250 mispredict from tools/decisions/solver_pool_design.md
// sec 7).
//
// A "tree" is one `SharedLineageSolver` (one Arc minted by a fork-gate
// hit; see `SymContext::fork`). Windows are counted in switches ON THAT
// TREE.

/// Switch count at which a tree's early hot ratio is first evaluated.
const CENSUS_EARLY_WINDOW: u64 = 8;
/// Switch count at which the same tree's later hot ratio is evaluated,
/// so early-vs-late disagreement (mispredict) can be counted.
const CENSUS_LATE_WINDOW: u64 = 24;
/// Hot-ratio percentage above which a tree is classified "hot" (the
/// ratio at which the switch_to trade is believed to pay). Matches the
/// v5ht global sampler's default threshold.
const CENSUS_HOT_PCT: u64 = 35;

/// Bumped by [`reset_lineage_stats`]. Trees outlive a stats reset (the
/// manager resets counters at the start of every `run()`, but the state
/// graph — and its lineage Arcs — can predate it), so a tree carries the
/// epoch it last counted under and zeroes its own switch/hot counters
/// when it sees a newer one. Without this, a tree minted before the
/// reset reports a switch count the per-run globals cannot explain.
static CENSUS_EPOCH: AtomicU64 = AtomicU64::new(0);

static TREES_MINTED: AtomicU64 = AtomicU64::new(0);
static TREES_WITH_SWITCH: AtomicU64 = AtomicU64::new(0);
static TREES_REACHING_EARLY: AtomicU64 = AtomicU64::new(0);
static TREES_HOT_AT_EARLY: AtomicU64 = AtomicU64::new(0);
static TREES_REACHING_LATE: AtomicU64 = AtomicU64::new(0);
static TREES_HOT_AT_LATE: AtomicU64 = AtomicU64::new(0);
/// Trees judged cold at [`CENSUS_EARLY_WINDOW`] but hot at
/// [`CENSUS_LATE_WINDOW`] — i.e. a per-tree early decision would have
/// dismantled a tree that goes on to pay. This is the rev250 failure
/// mode, measured per tree rather than globally.
static TREES_MISPREDICT_COLD: AtomicU64 = AtomicU64::new(0);
/// Largest switch count reached by any single tree.
static TREE_SWITCH_MAX: AtomicU64 = AtomicU64::new(0);

fn note_tree_minted() {
    TREES_MINTED.fetch_add(1, Ordering::Relaxed);
}

/// Fold one tree's running (switches, hots) into the census at the
/// window boundaries. Called from `SharedLineageSolver::census_switch`,
/// so it runs on every switch — kept to a couple of compares plus one
/// `fetch_max` off the boundaries.
fn note_tree_switch(switches: u64, hots: u64, hots_at_early: u64) {
    TREE_SWITCH_MAX.fetch_max(switches, Ordering::Relaxed);
    if switches == 1 {
        TREES_WITH_SWITCH.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let hot = |h: u64, n: u64| h * 100 >= CENSUS_HOT_PCT * n;
    if switches == CENSUS_EARLY_WINDOW {
        TREES_REACHING_EARLY.fetch_add(1, Ordering::Relaxed);
        if hot(hots, switches) {
            TREES_HOT_AT_EARLY.fetch_add(1, Ordering::Relaxed);
        }
    } else if switches == CENSUS_LATE_WINDOW {
        TREES_REACHING_LATE.fetch_add(1, Ordering::Relaxed);
        let hot_late = hot(hots, switches);
        if hot_late {
            TREES_HOT_AT_LATE.fetch_add(1, Ordering::Relaxed);
        }
        // The per-tree analogue of the rev250 failure: an early-window
        // decision would have dismantled this tree, yet it turns hot.
        if hot_late && !hot(hots_at_early, CENSUS_EARLY_WINDOW) {
            TREES_MISPREDICT_COLD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Snapshot the per-lineage-tree census counters (angr-g1fev).
pub fn tree_census_stats() -> [(&'static str, u64); 8] {
    [
        ("lineage_trees_minted", TREES_MINTED.load(Ordering::Relaxed)),
        (
            "lineage_trees_with_switch",
            TREES_WITH_SWITCH.load(Ordering::Relaxed),
        ),
        (
            "lineage_trees_reaching_early_window",
            TREES_REACHING_EARLY.load(Ordering::Relaxed),
        ),
        (
            "lineage_trees_hot_at_early_window",
            TREES_HOT_AT_EARLY.load(Ordering::Relaxed),
        ),
        (
            "lineage_trees_reaching_late_window",
            TREES_REACHING_LATE.load(Ordering::Relaxed),
        ),
        (
            "lineage_trees_hot_at_late_window",
            TREES_HOT_AT_LATE.load(Ordering::Relaxed),
        ),
        (
            "lineage_trees_mispredict_cold_early",
            TREES_MISPREDICT_COLD.load(Ordering::Relaxed),
        ),
        (
            "lineage_tree_switch_max",
            TREE_SWITCH_MAX.load(Ordering::Relaxed),
        ),
    ]
}

/// Snapshot the runtime-thrash-detection counters. Exposed alongside
/// [`lineage_stats`] so the integration patch can fold them into
/// `get_solver_stats`.
pub fn dismantle_stats() -> [(&'static str, u64); 4] {
    [
        (
            "lineage_dismantled",
            LINEAGE_DISMANTLED.load(Ordering::Relaxed) as u64,
        ),
        (
            "lineage_dismantle_count",
            LINEAGE_DISMANTLE_COUNT.load(Ordering::Relaxed),
        ),
        (
            "lineage_sample_call_count",
            LINEAGE_SAMPLE_CALL_COUNT.load(Ordering::Relaxed),
        ),
        (
            "lineage_sample_decision_count",
            LINEAGE_SAMPLE_DECISION_COUNT.load(Ordering::Relaxed),
        ),
    ]
}

/// Test helper: clear only the runtime-thrash-detection state without
/// touching the global LINEAGE_SWITCH_*/HOT_* counters. The broader
/// [`reset_lineage_stats`] is unsafe to call from a single test in a
/// multi-test module because it resets the workload counters that other
/// parallel tests have already snapshotted; this narrower variant lets
/// the sampler tests start from a clean dismantle baseline without
/// disturbing peer tests. Also primes LAST_SAMPLE_SWITCH_COUNT /
/// LAST_SAMPLE_HOT_COUNT to the current global values so that
/// [`sample_for_thrash`]'s next call sees only the delta the test
/// generates, plus any contemporaneous pollution from parallel tests.
#[doc(hidden)]
pub fn reset_dismantle_state_for_test() {
    LINEAGE_DISMANTLED.store(false, Ordering::Relaxed);
    LINEAGE_DISMANTLE_COUNT.store(0, Ordering::Relaxed);
    LAST_SAMPLE_SWITCH_COUNT.store(
        LINEAGE_SWITCH_COUNT.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    LAST_SAMPLE_HOT_COUNT.store(
        LINEAGE_SWITCH_HOT_COUNT.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    LAST_SAMPLE_STEP.store(0, Ordering::Relaxed);
}

/// Snapshot of the lineage-solver counters.
///
/// Returned as a fixed-size array of `(name, value)` so the integration
/// patch can fold it into the existing `HashMap<String, u64>` shape used
/// by [`crate::symbolic::get_solver_stats`].
///
/// `lineage_switch_hot_count` is the broader hot-no-op count (includes
/// both the O(1) tail-id+depth fast path and the prefix-walk-then-(0,0)
/// fallback). `lineage_switch_fast_path_count` is the strict subset that
/// hit the O(1) check — useful for measuring how often consecutive
/// queries land on the same state (the hot-cache win the BFS-thrash
/// motivation in angr-v5a5 design targets).
pub fn lineage_stats() -> [(&'static str, u64); 5] {
    [
        (
            "lineage_switch_count",
            LINEAGE_SWITCH_COUNT.load(Ordering::Relaxed),
        ),
        (
            "lineage_switch_hot_count",
            LINEAGE_SWITCH_HOT_COUNT.load(Ordering::Relaxed),
        ),
        (
            "lineage_switch_fast_path_count",
            LINEAGE_SWITCH_FAST_PATH_COUNT.load(Ordering::Relaxed),
        ),
        (
            "lineage_push_count",
            LINEAGE_PUSH_COUNT.load(Ordering::Relaxed),
        ),
        (
            "lineage_pop_count",
            LINEAGE_POP_COUNT.load(Ordering::Relaxed),
        ),
    ]
}

/// Reset lineage-solver counters. Pairs with [`lineage_stats`] when the
/// integration patch wires resets into `reset_solver_stats`. Also
/// clears the runtime-thrash-detection state (dismantle flag, sample
/// window, dismantle count) so a fresh exploration starts with lineage
/// minting enabled.
pub fn reset_lineage_stats() {
    LINEAGE_SWITCH_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_SWITCH_HOT_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_SWITCH_FAST_PATH_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_PUSH_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_POP_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_DISMANTLED.store(false, Ordering::Relaxed);
    LINEAGE_DISMANTLE_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_SAMPLE_CALL_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_SAMPLE_DECISION_COUNT.store(0, Ordering::Relaxed);
    SAMPLER_TICK_COUNT.store(0, Ordering::Relaxed);
    LAST_SAMPLE_SWITCH_COUNT.store(0, Ordering::Relaxed);
    LAST_SAMPLE_HOT_COUNT.store(0, Ordering::Relaxed);
    LAST_SAMPLE_STEP.store(0, Ordering::Relaxed);
    TREES_MINTED.store(0, Ordering::Relaxed);
    TREES_WITH_SWITCH.store(0, Ordering::Relaxed);
    TREES_REACHING_EARLY.store(0, Ordering::Relaxed);
    TREES_HOT_AT_EARLY.store(0, Ordering::Relaxed);
    TREES_REACHING_LATE.store(0, Ordering::Relaxed);
    TREES_HOT_AT_LATE.store(0, Ordering::Relaxed);
    TREES_MISPREDICT_COLD.store(0, Ordering::Relaxed);
    TREE_SWITCH_MAX.store(0, Ordering::Relaxed);
    CENSUS_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// Z3 solver shared by all states in one lineage, with a scope-tracked
/// stack of pushes corresponding to whichever state most recently issued
/// a query through it.
///
/// Not thread-safe on its own — the integration wraps it in
/// `Arc<Mutex<SharedLineageSolver>>`. Cross-state queries take the mutex,
/// `switch_to` runs under the lock, and the solver is released before the
/// caller does any Python or per-state mutation.
pub struct SharedLineageSolver {
    z3: z3::Solver,
    loaded_path: ScopePath,
    /// Switch/hot counts for THIS tree only (angr-g1fev census). The
    /// global `LINEAGE_SWITCH_*` counters sum these across every tree,
    /// which is exactly what makes the global sampler unable to decide
    /// per-tree.
    switches: u64,
    hots: u64,
    /// `hots` snapshotted at [`CENSUS_EARLY_WINDOW`] switches, so the
    /// late window can tell whether an early per-tree decision would
    /// have mispredicted this tree.
    hots_at_early: u64,
    /// [`CENSUS_EPOCH`] value these per-tree counters were last counted
    /// under; a mismatch means a stats reset happened and they are stale.
    epoch: u64,
}

impl SharedLineageSolver {
    /// Construct from a fresh Z3 solver. Base assertions for the lineage
    /// should be installed via [`assert_base`](Self::assert_base) before
    /// any [`switch_to`](Self::switch_to)/[`with_solver`](Self::with_solver)
    /// calls — they sit at scope 0 and are never popped.
    pub fn new(z3: z3::Solver) -> Self {
        note_tree_minted();
        SharedLineageSolver {
            z3,
            loaded_path: ScopePath::new(),
            switches: 0,
            hots: 0,
            hots_at_early: 0,
            epoch: CENSUS_EPOCH.load(Ordering::Relaxed),
        }
    }

    /// Record one switch on this tree and feed the per-tree census
    /// (angr-g1fev). Called at the top of [`switch_to`]; `hot` says
    /// whether the switch was a no-op (0 pops, 0 pushes).
    fn census_switch(&mut self, hot: bool) {
        let epoch = CENSUS_EPOCH.load(Ordering::Relaxed);
        if self.epoch != epoch {
            self.epoch = epoch;
            self.switches = 0;
            self.hots = 0;
            self.hots_at_early = 0;
        }
        self.switches += 1;
        if hot {
            self.hots += 1;
        }
        if self.switches == CENSUS_EARLY_WINDOW {
            self.hots_at_early = self.hots;
        }
        note_tree_switch(self.switches, self.hots, self.hots_at_early);
    }

    /// Assert a base constraint at scope 0 (unscoped — never popped).
    ///
    /// Debug-asserts that no scope frames are currently loaded; calling
    /// after a `switch_to` would put the assertion at the wrong scope.
    pub fn assert_base(&self, assertion: &Bool) {
        debug_assert!(
            self.loaded_path.is_empty(),
            "assert_base called while a scope path is loaded — would land at the wrong scope"
        );
        self.z3.assert(assertion);
    }

    /// Number of frames currently pushed on the solver. For invariant
    /// checks and tests.
    pub fn loaded_depth(&self) -> usize {
        self.loaded_path.len()
    }

    /// Switch the solver to match `target_path`, returning `(pops, pushes)`.
    ///
    /// Two fast paths sit ahead of the general prefix walk:
    ///
    /// 1. **O(1) hot-cache short-circuit.** When `target_path` has the
    ///    same length as `loaded_path` AND the same tail [`FrameId`], the
    ///    two paths are necessarily identical: [`FrameId`]s are globally
    ///    unique and minted only at constraint-add time, so a frame with
    ///    id `K` was pushed exactly once on one specific scope path. Every
    ///    state that holds frame `K` inherited it from that pushing state,
    ///    so every path ending in id `K` at depth `D` shares the same
    ///    `D-1` ancestor frames. This O(1) check avoids the O(min(|loaded|,
    ///    |target|)) walk through `common_prefix_len` when consecutive
    ///    queries come from the same state (the BFS-step intra-state query
    ///    burst that motivates the hot-state cache in the angr-v5a5 design).
    ///
    /// 2. **General prefix walk.** If the O(1) cache misses, fall back to
    ///    `common_prefix_len` for the full prefix calculation. Pops the
    ///    divergent suffix off the solver and pushes the target tail.
    pub fn switch_to(&mut self, target_path: &ScopePath) -> (usize, usize) {
        LINEAGE_SWITCH_COUNT.fetch_add(1, Ordering::Relaxed);

        // O(1) hot-cache fast path. See doc comment for the FrameId
        // uniqueness argument that justifies skipping the prefix walk.
        if target_path.len() == self.loaded_path.len()
            && target_path.last().map(|f| f.id) == self.loaded_path.last().map(|f| f.id)
        {
            LINEAGE_SWITCH_HOT_COUNT.fetch_add(1, Ordering::Relaxed);
            LINEAGE_SWITCH_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            self.census_switch(true);
            return (0, 0);
        }

        let prefix = common_prefix_len(&self.loaded_path, target_path);
        let pops = self.loaded_path.len() - prefix;
        let pushes = target_path.len() - prefix;

        if pops == 0 && pushes == 0 {
            // Paths share a prefix that covers both fully but the O(1)
            // tail-id check missed — should be unreachable in practice
            // (tail+depth equality is iff identity). Counted under
            // SWITCH_HOT but NOT SWITCH_FAST_PATH to keep the fast-path
            // counter a strict measure of the O(1) short-circuit.
            LINEAGE_SWITCH_HOT_COUNT.fetch_add(1, Ordering::Relaxed);
            self.census_switch(true);
            return (0, 0);
        }

        self.census_switch(false);

        if pops > 0 {
            self.z3.pop(pops as u32);
            LINEAGE_POP_COUNT.fetch_add(pops as u64, Ordering::Relaxed);
            self.loaded_path.truncate(prefix);
        }

        for frame in &target_path[prefix..] {
            self.z3.push();
            self.z3.assert(&frame.z3_assertion);
            self.loaded_path.push(frame.clone());
        }
        if pushes > 0 {
            LINEAGE_PUSH_COUNT.fetch_add(pushes as u64, Ordering::Relaxed);
        }

        (pops, pushes)
    }

    /// Switch to `target_path` and run a closure against the solver.
    ///
    /// The planned query API for the integration: every solver access
    /// from a `SymContext` goes through this method, ensuring the solver
    /// is in the right scope for the calling state before the closure
    /// runs.
    pub fn with_solver<R>(
        &mut self,
        target_path: &ScopePath,
        f: impl FnOnce(&z3::Solver) -> R,
    ) -> R {
        self.switch_to(target_path);
        f(&self.z3)
    }
}

/// Longest common prefix length between two scope paths, compared by
/// [`FrameId`].
fn common_prefix_len(a: &ScopePath, b: &ScopePath) -> usize {
    a.iter()
        .zip(b.iter())
        .take_while(|(x, y)| x.id == y.id)
        .count()
}

#[cfg(test)]
#[path = "lineage_tests.rs"]
mod lineage_tests;
