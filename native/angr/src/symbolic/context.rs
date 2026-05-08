//! Z3 solver context and constraint management.
//!
//! The `SymContext` manages:
//! - Symbolic variable creation and ID assignment
//! - Constraint tracking (when Z3 is available)
//! - Satisfiability checking (when Z3 is available)

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use smallvec::SmallVec;

use super::RustBV;

/// Inline capacity for SymContext push_* stacks. Branch nesting is typically
/// shallow (≤8) within a single block; SmallVec avoids the heap allocation
/// for the first push.
type PushStack = SmallVec<[usize; 8]>;

// =============================================================================
// Global Z3 Solver Profiling Counters
// =============================================================================
// Zero-cost when not read: atomic fetch_add is ~1ns on x86.

/// Total number of Z3 solver.check() calls.
static Z3_CHECK_COUNT: AtomicU64 = AtomicU64::new(0);
/// Total time (nanoseconds) spent in Z3 solver.check() calls.
static Z3_CHECK_TIME_NS: AtomicU64 = AtomicU64::new(0);
/// Number of solver materializations (lazy fork → first access).
static Z3_MATERIALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Total time (nanoseconds) spent materializing solvers.
static Z3_MATERIALIZE_TIME_NS: AtomicU64 = AtomicU64::new(0);
/// Number of assume_true/assume_false calls that hit the concrete fast path.
static Z3_ASSUME_CONCRETE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of assume_true/assume_false calls that went to Z3.
static Z3_ASSUME_SYMBOLIC_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of check_branch_feasibility calls.
static Z3_BRANCH_CHECK_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of branch checks where condition was concrete.
static Z3_BRANCH_CONCRETE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of branch checks where the cached parent model predicted one direction.
static Z3_BRANCH_MODEL_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of branch checks where no usable cached model was available.
static Z3_BRANCH_MODEL_MISS_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of to_z3_ast() / to_z3_bool() calls (AST construction).
static Z3_AST_BUILD_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Sat.
static Z3_SAT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Unsat.
static Z3_UNSAT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Unknown (timeout / resource limit).
static Z3_TIMEOUT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Per-site counters and timers for solver.check() calls.
/// Indexed by `CheckSite as usize`.
const NUM_CHECK_SITES: usize = 9;
static Z3_CHECK_SITE_COUNT: [AtomicU64; NUM_CHECK_SITES] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
static Z3_CHECK_SITE_TIME_NS: [AtomicU64; NUM_CHECK_SITES] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

/// Distinguishes which call site invoked solver.check() for profiling.
#[cfg(feature = "vex-engine-z3")]
#[derive(Clone, Copy)]
pub enum CheckSite {
    Satisfiable = 0,
    BranchTrue = 1,
    BranchFalse = 2,
    Eval = 3,
    EvalUpto = 4,
    MinInit = 5,
    MinSearch = 6,
    MaxInit = 7,
    MaxSearch = 8,
}

#[cfg(feature = "vex-engine-z3")]
const SITE_NAMES: [&str; NUM_CHECK_SITES] = [
    "satisfiable",
    "branch_true",
    "branch_false",
    "eval",
    "eval_upto",
    "min_init",
    "min_search",
    "max_init",
    "max_search",
];

/// Get all solver profiling stats as a HashMap.
pub fn get_solver_stats() -> HashMap<String, u64> {
    let mut stats = HashMap::new();
    stats.insert("z3_check_count".into(), Z3_CHECK_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_check_time_ns".into(), Z3_CHECK_TIME_NS.load(Ordering::Relaxed));
    stats.insert("z3_materialize_count".into(), Z3_MATERIALIZE_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_materialize_time_ns".into(), Z3_MATERIALIZE_TIME_NS.load(Ordering::Relaxed));
    stats.insert("z3_assume_concrete".into(), Z3_ASSUME_CONCRETE_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_assume_symbolic".into(), Z3_ASSUME_SYMBOLIC_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_branch_check".into(), Z3_BRANCH_CHECK_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_branch_concrete".into(), Z3_BRANCH_CONCRETE_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_branch_model_hit".into(), Z3_BRANCH_MODEL_HIT_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_branch_model_miss".into(), Z3_BRANCH_MODEL_MISS_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_ast_build".into(), Z3_AST_BUILD_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_sat_count".into(), Z3_SAT_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_unsat_count".into(), Z3_UNSAT_COUNT.load(Ordering::Relaxed));
    stats.insert("z3_timeout_count".into(), Z3_TIMEOUT_COUNT.load(Ordering::Relaxed));
    #[cfg(feature = "vex-engine-z3")]
    for i in 0..NUM_CHECK_SITES {
        let count = Z3_CHECK_SITE_COUNT[i].load(Ordering::Relaxed);
        let time_ns = Z3_CHECK_SITE_TIME_NS[i].load(Ordering::Relaxed);
        if count > 0 {
            stats.insert(format!("z3_site_{}_count", SITE_NAMES[i]), count);
            stats.insert(format!("z3_site_{}_time_ns", SITE_NAMES[i]), time_ns);
        }
    }
    stats
}

/// Reset all solver profiling stats to zero.
pub fn reset_solver_stats() {
    Z3_CHECK_COUNT.store(0, Ordering::Relaxed);
    Z3_CHECK_TIME_NS.store(0, Ordering::Relaxed);
    Z3_MATERIALIZE_COUNT.store(0, Ordering::Relaxed);
    Z3_MATERIALIZE_TIME_NS.store(0, Ordering::Relaxed);
    Z3_ASSUME_CONCRETE_COUNT.store(0, Ordering::Relaxed);
    Z3_ASSUME_SYMBOLIC_COUNT.store(0, Ordering::Relaxed);
    Z3_BRANCH_CHECK_COUNT.store(0, Ordering::Relaxed);
    Z3_BRANCH_CONCRETE_COUNT.store(0, Ordering::Relaxed);
    Z3_BRANCH_MODEL_HIT_COUNT.store(0, Ordering::Relaxed);
    Z3_BRANCH_MODEL_MISS_COUNT.store(0, Ordering::Relaxed);
    Z3_AST_BUILD_COUNT.store(0, Ordering::Relaxed);
    Z3_SAT_COUNT.store(0, Ordering::Relaxed);
    Z3_UNSAT_COUNT.store(0, Ordering::Relaxed);
    Z3_TIMEOUT_COUNT.store(0, Ordering::Relaxed);
    for i in 0..NUM_CHECK_SITES {
        Z3_CHECK_SITE_COUNT[i].store(0, Ordering::Relaxed);
        Z3_CHECK_SITE_TIME_NS[i].store(0, Ordering::Relaxed);
    }
}

/// Increment Z3 AST build counter (called from value.rs).
#[inline]
pub fn record_z3_ast_build() {
    Z3_AST_BUILD_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Timed wrapper around solver.check() — records count, total time, and per-site stats.
#[cfg(feature = "vex-engine-z3")]
#[inline]
fn timed_check(solver: &z3::Solver, site: CheckSite) -> z3::SatResult {
    let start = std::time::Instant::now();
    let result = solver.check();
    let elapsed_ns = start.elapsed().as_nanos() as u64;
    Z3_CHECK_COUNT.fetch_add(1, Ordering::Relaxed);
    Z3_CHECK_TIME_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
    let idx = site as usize;
    Z3_CHECK_SITE_COUNT[idx].fetch_add(1, Ordering::Relaxed);
    Z3_CHECK_SITE_TIME_NS[idx].fetch_add(elapsed_ns, Ordering::Relaxed);
    match result {
        z3::SatResult::Sat => Z3_SAT_COUNT.fetch_add(1, Ordering::Relaxed),
        z3::SatResult::Unsat => Z3_UNSAT_COUNT.fetch_add(1, Ordering::Relaxed),
        z3::SatResult::Unknown => Z3_TIMEOUT_COUNT.fetch_add(1, Ordering::Relaxed),
    };
    result
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
    next_id: AtomicU64,
    /// Number of constraints added (for tracking).
    constraint_count: AtomicUsize,
    /// Named symbolic variables for debugging.
    /// Arc-shared on fork (O(1) clone). Only mutated when constructing
    /// a fresh merged context — Arc::make_mut works because the merged
    /// SymContext is freshly created with a unique Arc.
    symbol_table: Arc<HashMap<String, u64>>,
    /// Current push level for transaction tracking.
    push_level: AtomicUsize,
    /// Constraint count at each push level (for rollback).
    push_constraint_counts: Mutex<PushStack>,
    /// Local Z3 cache length at each push level (for rollback truncation).
    #[cfg(feature = "vex-engine-z3")]
    push_local_cache_lengths: Mutex<PushStack>,
    /// Local assumed_constraints length at each push level (for rollback truncation).
    push_assumed_local_lengths: Mutex<PushStack>,
    /// Phase 2 Fix: Track assumed RustBV constraints for export to Python.
    /// Each entry is (constraint, is_assumed_true). Split into Arc-shared frozen
    /// prefix (O(1) clone on fork) and a local Vec for additions after fork —
    /// mirrors the z3_assertions_shared/local layout.
    /// Wrapped in Mutex so fork() can freeze local into shared in-place when safe.
    assumed_constraints_shared: Mutex<Arc<Vec<(RustBV, bool)>>>,
    assumed_constraints_local: Mutex<Vec<(RustBV, bool)>>,

    /// Shared (frozen) Z3 Bool assertions from parent — O(1) clone via Arc.
    /// Wrapped in Mutex so fork() can freeze local into shared in-place when safe
    /// (avoids cloning every Bool — each Bool::clone would call Z3_inc_ref).
    #[cfg(feature = "vex-engine-z3")]
    z3_assertions_shared: Mutex<Arc<Vec<z3::ast::Bool>>>,
    /// Local Z3 Bool assertions added after fork — only these are cloned.
    #[cfg(feature = "vex-engine-z3")]
    z3_assertions_local: Mutex<Vec<z3::ast::Bool>>,

    // Z3-specific fields (when feature is enabled)
    /// Z3 solver — lazy: starts as None on fork(), materialized on first access.
    /// This avoids O(n) assertion replay for forked states that are
    /// pruned/avoided/deadended without ever querying the solver.
    #[cfg(feature = "vex-engine-z3")]
    solver: Mutex<Option<z3::Solver>>,
    /// Cached SAT result, invalidated on constraint addition.
    #[cfg(feature = "vex-engine-z3")]
    sat_cache: Cell<Option<bool>>,
    /// Cached Z3 model, invalidated on constraint addition.
    #[cfg(feature = "vex-engine-z3")]
    model_cache: RefCell<Option<z3::Model>>,
    /// List of Z3 tracking boolean constants for unsat core mapping.
    /// Each entry is a (track_bool, constraint_ast) pair.
    #[cfg(feature = "vex-engine-z3")]
    constraint_trackers: Mutex<Vec<z3::ast::Bool>>,
    /// Z3 solver timeout in milliseconds (default: 30000).
    #[cfg(feature = "vex-engine-z3")]
    timeout_ms: AtomicU32,
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
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(Arc::new(Vec::new())),
            assumed_constraints_local: Mutex::new(Vec::new()),
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
        Self::with_timeout(30000)
    }

    /// Create a new solver context with Z3 and a custom timeout.
    #[cfg(feature = "vex-engine-z3")]
    pub fn with_timeout(timeout_ms: u32) -> Self {
        // Create solver with timeout (unsat_core disabled for performance —
        // tracking booleans add significant overhead per constraint)
        let solver = z3::Solver::new();
        let mut params = z3::Params::new();
        params.set_u32("timeout", timeout_ms);
        // Propagate extraction inward through arithmetic — reduces constraint
        // structure before bit-blasting, especially for Rust-generated constraints
        // that use nested Extract/SignExt patterns.
        params.set_bool("bv_extract_prop", true);
        solver.set_params(&params);

        SymContext {
            next_id: AtomicU64::new(0),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            push_local_cache_lengths: Mutex::new(PushStack::new()),
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::new(HashMap::new()),
            assumed_constraints_shared: Mutex::new(Arc::new(Vec::new())),
            assumed_constraints_local: Mutex::new(Vec::new()),
            z3_assertions_shared: Mutex::new(Arc::new(Vec::new())),
            z3_assertions_local: Mutex::new(Vec::new()),
            solver: Mutex::new(Some(solver)),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(Vec::new()),
            timeout_ms: AtomicU32::new(timeout_ms),
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
    fn solver(&self) -> parking_lot::MappedMutexGuard<'_, z3::Solver> {
        let mut guard = self.solver.lock();
        if guard.is_none() {
            let start = std::time::Instant::now();
            let new_solver = z3::Solver::new();
            let mut params = z3::Params::new();
            params.set_u32("timeout", self.timeout_ms.load(Ordering::SeqCst));
            params.set_bool("bv_extract_prop", true);
            new_solver.set_params(&params);

            // Replay cached Z3 assertions: shared prefix then local additions
            let shared = Arc::clone(&self.z3_assertions_shared.lock());
            for constraint in shared.iter() {
                new_solver.assert(constraint);
            }
            let local = self.z3_assertions_local.lock();
            for constraint in local.iter() {
                new_solver.assert(constraint);
            }

            *guard = Some(new_solver);
            Z3_MATERIALIZE_COUNT.fetch_add(1, Ordering::Relaxed);
            Z3_MATERIALIZE_TIME_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        parking_lot::MutexGuard::map(guard, |opt| opt.as_mut().expect("solver was just initialized in the None branch above"))
    }

    /// Get the next unique ID for a symbolic variable.
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraint_count.load(Ordering::SeqCst)
    }

    // =========================================================================
    // Symbol Management
    // =========================================================================

    /// Create a new symbolic bitvector with a unique name.
    pub fn new_bv(&self, name: &str, width: u32) -> RustBV {
        let unique_name = self.unique_name(name);
        RustBV::symbolic(self, &unique_name, width)
    }

    /// Create a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        let id = self.next_id();
        format!("{}_{}", base, id)
    }

    // =========================================================================
    // Constraint Management (Z3-backed)
    // =========================================================================

    /// Push an entry to assumed_constraints for export tracking.
    /// Used by the fast path that bypasses assume_true.
    #[cfg(feature = "vex-engine-z3")]
    pub fn assumed_constraints_push(&self, bv: RustBV, is_true: bool) {
        self.assumed_constraints_local.lock().push((bv, is_true));
    }

    /// Export all Z3 assertion pointers from the assertion cache.
    /// Uses z3_assertions_shared + z3_assertions_local which track
    /// every assertion made via assume_true/assume_false/add_constraint_raw.
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_assertion_ptrs(&self) -> Vec<usize> {
        use z3::ast::Ast;
        let mut ptrs = Vec::new();
        let shared = Arc::clone(&self.z3_assertions_shared.lock());
        for constraint in shared.iter() {
            ptrs.push(constraint.get_z3_ast().as_ptr() as usize);
        }
        let local = self.z3_assertions_local.lock();
        for constraint in local.iter() {
            ptrs.push(constraint.get_z3_ast().as_ptr() as usize);
        }
        ptrs
    }

    /// Debug: dump solver state as string for comparison.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_string(&self) -> String {
        let solver = self.solver();
        format!("{}", solver)
    }

    /// Debug: get the Z3 solver's internal push level.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_push_level(&self) -> usize {
        self.push_level.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Add a constraint from a raw Z3_ast pointer (shared context fast path).
    ///
    /// This bypasses the RustBV → build_z3_ast() conversion, preserving the
    /// original Z3 AST structure from Python's claripy/z3 backend.
    /// SAFETY: The pointer must be a valid Z3_ast Bool in the same Z3 context.
    #[cfg(feature = "vex-engine-z3")]
    pub unsafe fn add_constraint_raw(&self, z3_ast_ptr: usize) {
        let ctx = z3::Context::thread_local();
        let constraint: z3::ast::Bool = unsafe {
            let raw_ast = std::ptr::NonNull::new_unchecked(z3_ast_ptr as *mut _);
            z3::ast::Ast::wrap(&ctx, raw_ast)
        };
        // Cache Z3 Bool for fast fork replay
        self.z3_assertions_local.lock().push(constraint.clone());
        self.solver().assert(&constraint);
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        self.sat_cache.set(None);
        // Keep the cached model if it still satisfies the new constraint —
        // otherwise it's invalidated. This lets check_branch_feasibility
        // reuse the model across consecutive assume_true/assume_false calls.
        self.invalidate_model_if_inconsistent(&constraint);
    }

    /// Add a constraint (fast path: no tracking overhead).
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint(&self, constraint: z3::ast::Bool) {
        // Use plain assert for fast path (no unsat_core tracking overhead).
        // This avoids creating tracking booleans, string formatting, and
        // mutex acquisition on constraint_trackers for every constraint.
        // NOTE: Don't cache here — callers (assume_true, assume_false,
        // add_constraint_raw) cache before calling this to avoid double-cache.
        self.solver().assert(&constraint);
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        // Invalidate sat_cache - constraint set has changed.
        self.sat_cache.set(None);
        // Preserve model_cache when consistent with the new constraint.
        self.invalidate_model_if_inconsistent(&constraint);
    }

    /// Add a constraint with tracking for unsat_core extraction.
    /// Use this only when unsat_core analysis is needed.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint_tracked(&self, constraint: z3::ast::Bool) {
        let idx = self.constraint_count.load(Ordering::SeqCst);
        let track_name = format!("__track_{}", idx);
        let track_bool = z3::ast::Bool::new_const(track_name.as_str());

        {
            let mut trackers = self.constraint_trackers.lock();
            trackers.push(track_bool.clone());
        }

        self.solver().assert_and_track(&constraint, &track_bool);
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        self.sat_cache.set(None);
        self.invalidate_model_if_inconsistent(&constraint);
    }

    /// If a cached model exists, drop it unless it still satisfies the new
    /// constraint. Models that satisfy a superset of constraints stay valid;
    /// this lets check_branch_feasibility reuse a model across consecutive
    /// assume_true/assume_false calls in deferred-fork mode.
    #[cfg(feature = "vex-engine-z3")]
    fn invalidate_model_if_inconsistent(&self, constraint: &z3::ast::Bool) {
        let mut cache = self.model_cache.borrow_mut();
        if cache.is_none() {
            return;
        }
        let still_valid = cache
            .as_ref()
            .and_then(|m| m.eval(constraint, true))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        if !still_valid {
            *cache = None;
        }
    }

    /// Add a constraint that the bitvector equals a specific value.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_bv_constraint(&self, bv: &RustBV, value: u128) {
        // Fast path: if bv is already concrete, the constraint is either
        // trivially true (skip) or trivially false (makes UNSAT).
        if let Some(v) = bv.as_u128() {
            if v == value {
                return; // Tautology — skip Z3
            }
            // Falls through to add False constraint (UNSAT)
        }
        let ast = bv.to_z3_ast();
        let val_ast = if bv.width() <= 64 {
            z3::ast::BV::from_u64(value as u64, bv.width())
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, bv.width() - 64);
            hi.concat(&lo)
        };
        let constraint = ast.eq(&val_ast);
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is true (non-zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_true(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python
        self.assumed_constraints_local.lock().push((cond.clone(), true));
        // Fast path: concrete true is a tautology — skip Z3 entirely
        if let Some(v) = cond.as_u128() {
            if v != 0 {
                Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
                return; // Asserting True is a no-op
            }
            // v == 0: asserting False makes solver UNSAT — still add it
        }
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Use to_z3_bool() to produce native Z3 Bool for comparison ops,
        // avoiding ITE(cmp, BV(1,1), BV(0,1))._eq(BV(1,1)) round-trip.
        let constraint = cond.to_z3_bool();
        // Cache Z3 Bool for fast fork replay
        self.z3_assertions_local.lock().push(constraint.clone());
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_false(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python
        self.assumed_constraints_local.lock().push((cond.clone(), false));
        // Fast path: concrete false (== 0) means not(False) = True — skip Z3
        if let Some(v) = cond.as_u128() {
            if v == 0 {
                Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
                return; // Asserting not(False) = True is a no-op
            }
            // v != 0: asserting not(True) = False makes solver UNSAT — still add it
        }
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Negate the bool directly
        let constraint = cond.to_z3_bool().not();
        // Cache Z3 Bool for fast fork replay
        self.z3_assertions_local.lock().push(constraint.clone());
        self.add_constraint(constraint);
    }

    // =========================================================================
    // Satisfiability & Evaluation (Z3-backed)
    // =========================================================================

    /// Check if the current constraints are satisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_sat(&self) -> bool {
        // Check cache first
        if let Some(cached) = self.sat_cache.get() {
            return cached;
        }
        // Perform actual SAT check
        let solver = self.solver();
        let result = matches!(timed_check(&solver, CheckSite::Satisfiable), z3::SatResult::Sat);
        self.sat_cache.set(Some(result));
        // Populate model_cache if SAT — get_model is essentially free after
        // a successful check, and the model lets check_branch_feasibility
        // skip one of two Z3 checks.
        if result {
            let mut cache = self.model_cache.borrow_mut();
            if cache.is_none() {
                if let Some(m) = solver.get_model() {
                    *cache = Some(m);
                }
            }
        }
        result
    }

    /// Set the Z3 solver timeout in milliseconds.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_timeout(&self, timeout_ms: u32) {
        self.timeout_ms.store(timeout_ms, Ordering::SeqCst);
        if let Some(solver) = self.solver.lock().as_ref() {
            let mut params = z3::Params::new();
            params.set_u32("timeout", timeout_ms);
            params.set_bool("bv_extract_prop", true);
            solver.set_params(&params);
        }
    }

    /// Get the Z3 solver timeout in milliseconds.
    #[cfg(feature = "vex-engine-z3")]
    pub fn timeout_ms(&self) -> u32 {
        self.timeout_ms.load(Ordering::SeqCst)
    }

    /// Prime the SAT cache with a known value.
    /// Used after symbolic branch forking where the interpreter already
    /// proved feasibility — avoids redundant Z3 check() calls.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_sat_cache(&self, value: bool) {
        self.sat_cache.set(Some(value));
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn set_sat_cache(&self, _value: bool) {}

    /// Check if a bitvector condition can be true.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_true(&self, cond: &RustBV) -> bool {
        self.check_branch_feasibility(cond).0
    }

    /// Check if a bitvector condition can be false.
    #[cfg(feature = "vex-engine-z3")]
    pub fn can_be_false(&self, cond: &RustBV) -> bool {
        self.check_branch_feasibility(cond).1
    }

    /// Check both branch directions in a single solver session.
    /// Returns (can_be_true, can_be_false). Holds the solver lock once
    /// for both checks, reducing lock acquisitions from 8 to 2.
    /// When only one direction is feasible, skips the second Z3 check.
    ///
    /// Optimization: if a parent model is cached (from a prior is_sat/eval
    /// or carried across add_constraint calls), evaluate cond on it. The
    /// model satisfies the parent constraints C, so M(cond)=true proves
    /// can_be_true without Z3 (and symmetrically for false). Only the
    /// other direction needs a Z3 check.
    ///
    /// Note: In deferred fork mode, this is NOT called — the interpreter
    /// skips feasibility checks and assumes both branches are feasible.
    /// This method is only used in non-deferred mode and for explicit checks.
    #[cfg(feature = "vex-engine-z3")]
    pub fn check_branch_feasibility(&self, cond: &RustBV) -> (bool, bool) {
        debug_assert_eq!(cond.width(), 1);
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            Z3_BRANCH_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
            return (v != 0, v == 0);
        }
        Z3_BRANCH_CHECK_COUNT.fetch_add(1, Ordering::Relaxed);
        // Use native Bool to avoid ITE wrapping overhead
        let bool_ast = cond.to_z3_bool();

        let solver = self.solver();

        // Try to predict one direction with the cached parent model.
        let predicted: Option<bool> = self
            .model_cache
            .borrow()
            .as_ref()
            .and_then(|m| m.eval(&bool_ast, true))
            .and_then(|b| b.as_bool());

        match predicted {
            Some(true) => {
                Z3_BRANCH_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                // can_be_true=true is proven by the model. Check the other
                // direction (¬cond) with Z3.
                solver.push();
                solver.assert(&bool_ast.not());
                let can_false =
                    matches!(timed_check(&solver, CheckSite::BranchFalse), z3::SatResult::Sat);
                solver.pop(1);
                (true, can_false)
            }
            Some(false) => {
                Z3_BRANCH_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                // can_be_false=true is proven by the model. Check cond.
                solver.push();
                solver.assert(&bool_ast);
                let can_true =
                    matches!(timed_check(&solver, CheckSite::BranchTrue), z3::SatResult::Sat);
                solver.pop(1);
                (can_true, true)
            }
            None => {
                Z3_BRANCH_MODEL_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
                // No cached model: do the original two-check flow.
                solver.push();
                solver.assert(&bool_ast);
                let can_true =
                    matches!(timed_check(&solver, CheckSite::BranchTrue), z3::SatResult::Sat);
                solver.pop(1);

                if !can_true {
                    return (false, true); // Must be false-only
                }

                solver.push();
                solver.assert(&bool_ast.not());
                let can_false =
                    matches!(timed_check(&solver, CheckSite::BranchFalse), z3::SatResult::Sat);
                solver.pop(1);

                (can_true, can_false)
            }
        }
    }

    /// Evaluate a bitvector to a concrete value if possible.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        // Try to use cached model first (avoids re-checking SAT)
        {
            let cache = self.model_cache.borrow();
            if let Some(ref model) = *cache {
                let ast = bv.to_z3_ast();
                if let Some(result) = model.eval(&ast, true) {
                    return Self::extract_bv_value(&result);
                }
            }
        }

        // Need to get a fresh model - must call check() first for Z3
        let solver = self.solver();
        match timed_check(&solver, CheckSite::Eval) {
            z3::SatResult::Sat => {
                self.sat_cache.set(Some(true));
            }
            _ => {
                self.sat_cache.set(Some(false));
                return None;
            }
        }

        // Get model from the check we just did
        let model = solver.get_model()?;
        let ast = bv.to_z3_ast();
        let result = model.eval(&ast, true)?;
        let value = Self::extract_bv_value(&result);
        drop(solver); // Release lock before borrowing model_cache
        *self.model_cache.borrow_mut() = Some(model);
        value
    }

    /// Evaluate a bitvector to bytes (for values > 128 bits).
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_wide(&self, bv: &RustBV) -> Option<Vec<u8>> {
        let width = bv.width();

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            let byte_len = ((width + 7) / 8) as usize;
            let mut result = vec![0u8; byte_len];
            let bytes = v.to_be_bytes();
            let offset = byte_len.saturating_sub(16);
            for (i, &_b) in bytes.iter().enumerate() {
                let src_idx = 16 - byte_len.min(16) + i;
                if src_idx < 16 && offset + i < byte_len {
                    result[offset + i] = bytes[src_idx];
                }
            }
            // Handle case where byte_len <= 16
            if byte_len <= 16 {
                result = bytes[16 - byte_len..].to_vec();
            }
            return Some(result);
        }

        // Need to get a model from Z3
        let solver = self.solver();
        match timed_check(&solver, CheckSite::Eval) {
            z3::SatResult::Sat => {}
            _ => return None,
        }

        let model = solver.get_model()?;
        let ast = bv.to_z3_ast();
        let result = model.eval(&ast, true)?;
        Self::extract_bv_value_wide(&result, width)
    }

    /// Extract a u128 value from a Z3 BV result.
    /// For values > 128 bits, returns the low 128 bits (caller should use
    /// extract_bv_value_wide for arbitrarily large values).
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value(bv: &z3::ast::BV) -> Option<u128> {
        // Try as u64 first (fast path for <= 64-bit)
        if let Some(v) = bv.as_u64() {
            return Some(v as u128);
        }
        // For larger values, parse the string representation
        Self::extract_bv_value_from_string(bv)
    }

    /// Extract a BV value by parsing its string representation.
    /// Handles arbitrarily large values, returns low 128 bits.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value_from_string(bv: &z3::ast::BV) -> Option<u128> {
        let s = format!("{}", bv);
        // Z3 uses formats: #xHEXDIGITS, #bBINARY, or decimal
        if let Some(hex_str) = s.strip_prefix("#x") {
            // Parse as hex, taking low 128 bits
            parse_wide_hex_low128(hex_str)
        } else if let Some(bin_str) = s.strip_prefix("#b") {
            // Parse as binary, taking low 128 bits
            parse_wide_binary_low128(bin_str)
        } else {
            // Try decimal
            s.parse::<u128>().ok()
        }
    }

    /// Extract an arbitrarily large BV value as a Vec<u8> (big-endian).
    /// Used for values > 128 bits where we need the full value.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_bv_value_wide(bv: &z3::ast::BV, width: u32) -> Option<Vec<u8>> {
        let s = format!("{}", bv);
        if let Some(hex_str) = s.strip_prefix("#x") {
            // Parse full hex value to bytes
            parse_hex_to_bytes(hex_str, width)
        } else if let Some(bin_str) = s.strip_prefix("#b") {
            // Parse full binary value to bytes
            parse_binary_to_bytes(bin_str, width)
        } else {
            // Decimal - parse and convert
            parse_decimal_to_bytes(&s, width)
        }
    }

    /// Evaluate a bitvector and return up to n solutions.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return vec![v];
        }

        if n == 0 {
            return vec![];
        }

        let mut results = Vec::with_capacity(n);
        let ast = bv.to_z3_ast();

        // Hold lock for entire operation to avoid lifetime issues
        let solver = self.solver();
        solver.push();

        for _ in 0..n {
            match timed_check(&solver, CheckSite::EvalUpto) {
                z3::SatResult::Sat => {
                    if let Some(model) = solver.get_model() {
                        if let Some(result) = model.eval(&ast, true) {
                            if let Some(value) = Self::extract_bv_value(&result) {
                                results.push(value);
                                // Add constraint to exclude this value
                                let val_ast = if bv.width() <= 64 {
                                    z3::ast::BV::from_u64(value as u64, bv.width())
                                } else {
                                    let lo = z3::ast::BV::from_u64(value as u64, 64);
                                    let hi = z3::ast::BV::from_u64(
                                        (value >> 64) as u64,
                                        bv.width() - 64,
                                    );
                                    hi.concat(&lo)
                                };
                                solver.assert(&ast.eq(&val_ast).not());
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        solver.pop(1);
        results
    }

    /// Evaluate a bitvector and return up to n solutions as byte arrays (big-endian).
    /// Handles values of any width without truncation.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto_wide(&self, bv: &RustBV, n: usize) -> Vec<Vec<u8>> {
        let width = bv.width();

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            let byte_len = ((width + 7) / 8) as usize;
            let bytes = v.to_be_bytes();
            let result = if byte_len <= 16 {
                bytes[16 - byte_len..].to_vec()
            } else {
                let mut r = vec![0u8; byte_len];
                r[byte_len - 16..].copy_from_slice(&bytes);
                r
            };
            return vec![result];
        }

        if n == 0 {
            return vec![];
        }

        let mut results = Vec::with_capacity(n);
        let ast = bv.to_z3_ast();

        let solver = self.solver();
        solver.push();

        for _ in 0..n {
            match timed_check(&solver, CheckSite::EvalUpto) {
                z3::SatResult::Sat => {
                    if let Some(model) = solver.get_model() {
                        if let Some(result) = model.eval(&ast, true) {
                            if let Some(bytes) = Self::extract_bv_value_wide(&result, width) {
                                // Exclude this value from future solutions
                                // Build Z3 constant from bytes for full-precision exclusion
                                let val_ast = Self::make_bv_from_bytes(&bytes, width);
                                solver.assert(&ast.eq(&val_ast).not());
                                results.push(bytes);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        solver.pop(1);
        results
    }

    /// Create a Z3 BV constant from a u128 value.
    #[cfg(feature = "vex-engine-z3")]
    fn make_bv_const(value: u128, width: u32) -> z3::ast::BV {
        if width <= 64 {
            z3::ast::BV::from_u64(value as u64, width)
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, width - 64);
            hi.concat(&lo)
        }
    }

    /// Create a Z3 BV constant from big-endian bytes.
    /// Handles arbitrary widths by building 64-bit chunks and concatenating.
    #[cfg(feature = "vex-engine-z3")]
    fn make_bv_from_bytes(bytes: &[u8], width: u32) -> z3::ast::BV {
        if width <= 64 {
            let mut val: u64 = 0;
            for &b in bytes {
                val = (val << 8) | (b as u64);
            }
            return z3::ast::BV::from_u64(val, width);
        }

        // Build from 64-bit chunks (big-endian)
        let byte_len = bytes.len();
        let mut result: Option<z3::ast::BV> = None;
        let mut bits_remaining = width;
        let mut pos = 0;

        while bits_remaining > 0 {
            let chunk_bits = std::cmp::min(bits_remaining, 64);
            let chunk_bytes = ((chunk_bits + 7) / 8) as usize;
            let mut val: u64 = 0;
            for i in 0..chunk_bytes {
                if pos + i < byte_len {
                    val = (val << 8) | (bytes[pos + i] as u64);
                } else {
                    val <<= 8;
                }
            }
            let chunk = z3::ast::BV::from_u64(val, chunk_bits);
            result = Some(match result {
                Some(prev) => prev.concat(&chunk),
                None => chunk,
            });
            pos += chunk_bytes;
            bits_remaining -= chunk_bits;
        }

        result.unwrap_or_else(|| z3::ast::BV::from_u64(0, width))
    }

    /// Get the minimum value of a bitvector using binary search (O(log N)).
    ///
    /// This implementation uses pure SAT checks without model value extraction,
    /// which allows it to work with bitvectors of any width (including >64 bits).
    /// Based on claripy's _extrema algorithm.
    #[cfg(feature = "vex-engine-z3")]
    pub fn min(&self, bv: &RustBV, signed: bool) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        // Set initial bounds based on signedness
        // For unsigned: [0, 2^width - 1]
        // For signed: [-(2^(width-1)), 2^(width-1) - 1] represented in two's complement
        let (mut lo, mut hi): (u128, u128) = if signed {
            // Signed: lo is most negative (0x8000...), hi is most positive (0x7FFF...)
            let sign_bit = 1u128 << (width - 1);
            let max_positive = sign_bit - 1;
            // In two's complement ordering for binary search, we search [0, max_positive] then [sign_bit, max_val]
            // But for signed comparison, Z3 handles this correctly with bvsle/bvsge
            // Start with the full signed range in two's complement representation
            (sign_bit, max_positive)
        } else {
            // Unsigned: [0, 2^width - 1]
            let max_val = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            (0, max_val)
        };

        let solver = self.solver();
        solver.push();

        if signed {
            // For signed values, we need to handle the two's complement ordering
            // First check if a negative value (sign bit set) is possible
            solver.push();
            let zero = Self::make_bv_const(0, width);
            solver.assert(&ast.bvslt(&zero)); // bv < 0 (signed)
            let has_negative = matches!(timed_check(&solver, CheckSite::MinInit), z3::SatResult::Sat);
            solver.pop(1);

            if has_negative {
                // Minimum is negative, search in [sign_bit, all_ones] range
                let sign_bit = 1u128 << (width - 1);
                let max_val = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                lo = sign_bit;
                hi = max_val;

                // Binary search for minimum negative value (smallest = most negative)
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be <= mid (signed comparison)
                    solver.assert(&ast.bvsle(&mid_ast));
                    let can_be_le_mid = matches!(timed_check(&solver, CheckSite::MinSearch), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_le_mid {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
            } else {
                // Minimum is non-negative, search in [0, max_positive] range
                let max_positive = (1u128 << (width - 1)) - 1;
                lo = 0;
                hi = max_positive;

                // Binary search for minimum non-negative value
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be <= mid (signed comparison)
                    solver.assert(&ast.bvsle(&mid_ast));
                    let can_be_le_mid = matches!(timed_check(&solver, CheckSite::MinSearch), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_le_mid {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
            }
        } else {
            // Unsigned binary search
            while lo < hi {
                let mid = lo + (hi - lo) / 2;

                solver.push();
                let mid_ast = Self::make_bv_const(mid, width);
                // Check if bv can be <= mid (unsigned comparison)
                solver.assert(&ast.bvule(&mid_ast));
                let can_be_le_mid = matches!(timed_check(&solver, CheckSite::MinSearch), z3::SatResult::Sat);
                solver.pop(1);

                if can_be_le_mid {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
        }

        solver.pop(1);
        Some(lo)
    }

    /// Get the maximum value of a bitvector using binary search (O(log N)).
    ///
    /// This implementation uses pure SAT checks without model value extraction,
    /// which allows it to work with bitvectors of any width (including >64 bits).
    /// Based on claripy's _extrema algorithm.
    #[cfg(feature = "vex-engine-z3")]
    pub fn max(&self, bv: &RustBV, signed: bool) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        let solver = self.solver();
        solver.push();

        let (mut lo, mut hi): (u128, u128);

        if signed {
            // For signed values, we need to handle the two's complement ordering
            // First check if a non-negative value (sign bit clear) is possible
            let zero = Self::make_bv_const(0, width);

            solver.push();
            solver.assert(&ast.bvsge(&zero)); // bv >= 0 (signed)
            let has_non_negative = matches!(timed_check(&solver, CheckSite::MaxInit), z3::SatResult::Sat);
            solver.pop(1);

            if has_non_negative {
                // Maximum is non-negative, search in [0, max_positive] range
                let max_positive = (1u128 << (width - 1)) - 1;
                lo = 0;
                hi = max_positive;

                // Binary search for maximum non-negative value
                while lo < hi {
                    // Use ceiling division to avoid infinite loop when lo + 1 == hi
                    let mid = lo + (hi - lo + 1) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be >= mid (signed comparison)
                    solver.assert(&ast.bvsge(&mid_ast));
                    let can_be_ge_mid = matches!(timed_check(&solver, CheckSite::MaxSearch), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_ge_mid {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
            } else {
                // Maximum is negative, search in [sign_bit, all_ones] range
                let sign_bit = 1u128 << (width - 1);
                let max_val = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                lo = sign_bit;
                hi = max_val;

                // Binary search for maximum negative value (largest = least negative = closest to 0)
                while lo < hi {
                    // Use ceiling division to avoid infinite loop when lo + 1 == hi
                    let mid = lo + (hi - lo + 1) / 2;

                    solver.push();
                    let mid_ast = Self::make_bv_const(mid, width);
                    // Check if bv can be >= mid (signed comparison)
                    solver.assert(&ast.bvsge(&mid_ast));
                    let can_be_ge_mid = matches!(timed_check(&solver, CheckSite::MaxSearch), z3::SatResult::Sat);
                    solver.pop(1);

                    if can_be_ge_mid {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
            }
        } else {
            // Unsigned binary search
            let max_val = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            lo = 0;
            hi = max_val;

            while lo < hi {
                // Use ceiling division to avoid infinite loop when lo + 1 == hi
                let mid = lo + (hi - lo + 1) / 2;

                solver.push();
                let mid_ast = Self::make_bv_const(mid, width);
                // Check if bv can be >= mid (unsigned comparison)
                solver.assert(&ast.bvuge(&mid_ast));
                let can_be_ge_mid = matches!(timed_check(&solver, CheckSite::MaxSearch), z3::SatResult::Sat);
                solver.pop(1);

                if can_be_ge_mid {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
        }

        solver.pop(1);
        Some(lo)
    }

    /// Get the range [min, max] of possible values for a bitvector.
    ///
    /// Returns None if the constraints are unsatisfiable or evaluation fails.
    #[cfg(feature = "vex-engine-z3")]
    pub fn range(&self, bv: &RustBV) -> Option<(u128, u128)> {
        let min = self.min(bv, false)?;
        let max = self.max(bv, false)?;
        Some((min, max))
    }

    /// Get the unsigned range [min, max] of a bitvector, using known valid
    /// solutions to seed the binary search.
    ///
    /// `smallest_known` and `largest_known` must be values that satisfy the
    /// current constraints (e.g., from `solutions()`). They are used as
    /// initial bounds: `min` is searched in `[0, smallest_known]` and `max`
    /// is searched in `[largest_known, 2^width - 1]`. This roughly halves the
    /// SAT calls when `smallest_known` is well below `2^width`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn range_seeded(
        &self,
        bv: &RustBV,
        smallest_known: u128,
        largest_known: u128,
    ) -> Option<(u128, u128)> {
        if let Some(v) = bv.as_u128() {
            return Some((v, v));
        }
        if !self.is_sat() {
            return None;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();
        let max_val: u128 = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };

        // Clamp seeds to the bitvector range. Seeds must already be valid
        // solutions, so smallest_known ≤ true_max and largest_known ≥ true_min.
        let hi_seed = smallest_known.min(max_val);
        let lo_seed = largest_known.min(max_val);

        let solver = self.solver();
        solver.push();

        // Binary search for min in [0, hi_seed].
        let mut lo: u128 = 0;
        let mut hi: u128 = hi_seed;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            solver.push();
            let mid_ast = Self::make_bv_const(mid, width);
            solver.assert(&ast.bvule(&mid_ast));
            let can_be_le_mid = matches!(
                timed_check(&solver, CheckSite::MinSearch),
                z3::SatResult::Sat
            );
            solver.pop(1);
            if can_be_le_mid {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let min_val = lo;

        // Binary search for max in [lo_seed, max_val].
        let mut lo: u128 = lo_seed;
        let mut hi: u128 = max_val;
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            solver.push();
            let mid_ast = Self::make_bv_const(mid, width);
            solver.assert(&ast.bvuge(&mid_ast));
            let can_be_ge_mid = matches!(
                timed_check(&solver, CheckSite::MaxSearch),
                z3::SatResult::Sat
            );
            solver.pop(1);
            if can_be_ge_mid {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let max_val = lo;

        solver.pop(1);
        Some((min_val, max_val))
    }

    /// Get up to n concrete solutions for a bitvector.
    ///
    /// This is a convenience wrapper around eval_upto.
    #[cfg(feature = "vex-engine-z3")]
    pub fn solutions(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        self.eval_upto(bv, n)
    }

    /// Check if a specific value is a valid solution for a bitvector.
    #[cfg(feature = "vex-engine-z3")]
    pub fn solution(&self, bv: &RustBV, value: u128) -> bool {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return v == value;
        }

        let ast = bv.to_z3_ast();
        let width = bv.width();

        let val_ast = if width <= 64 {
            z3::ast::BV::from_u64(value as u64, width)
        } else {
            let lo = z3::ast::BV::from_u64(value as u64, 64);
            let hi = z3::ast::BV::from_u64((value >> 64) as u64, width - 64);
            hi.concat(&lo)
        };

        let constraint = ast.eq(&val_ast);

        let solver = self.solver();
        solver.push();
        solver.assert(&constraint);
        let result = matches!(timed_check(&solver, CheckSite::Satisfiable), z3::SatResult::Sat);
        solver.pop(1);

        result
    }

    /// Save solver state for temporary constraints.
    #[cfg(feature = "vex-engine-z3")]
    pub fn push(&self) {
        self.solver().push();
        // Invalidate caches since constraint set may change
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    /// Restore solver state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn pop(&self) {
        self.solver().pop(1);
        // Invalidate caches since constraint set has changed
        self.sat_cache.set(None);
        *self.model_cache.borrow_mut() = None;
    }

    // =========================================================================
    // Transactional Constraint Sync
    // =========================================================================

    /// Begin a new transaction.
    ///
    /// This pushes a new solver frame and records the constraint count,
    /// allowing rollback on failure via `transaction_rollback()`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_begin(&self) {
        self.push();
        let current_count = self.constraint_count.load(Ordering::SeqCst);
        self.push_constraint_counts.lock().push(current_count);
        let local_len = self.z3_assertions_local.lock().len();
        self.push_local_cache_lengths.lock().push(local_len);
        let assumed_local_len = self.assumed_constraints_local.lock().len();
        self.push_assumed_local_lengths.lock().push(assumed_local_len);
        self.push_level.fetch_add(1, Ordering::SeqCst);
    }

    /// Commit the current transaction.
    ///
    /// This validates that constraints are satisfiable before committing.
    /// Returns an error if constraints became unsatisfiable.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_commit(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }

        // Validate constraints are satisfiable before committing
        if !self.is_sat() {
            // Rollback on failure
            self.transaction_rollback()?;
            return Err(ConstraintSyncError::Unsatisfiable);
        }

        // Pop the solver frame but keep the constraints
        // Note: We don't actually pop here since we want to keep constraints
        // The push was just for protection during sync
        self.push_constraint_counts.lock().pop();
        self.push_level.fetch_sub(1, Ordering::SeqCst);

        Ok(())
    }

    /// Rollback the current transaction.
    ///
    /// This restores the solver state to before `transaction_begin()` was called.
    #[cfg(feature = "vex-engine-z3")]
    pub fn transaction_rollback(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }

        // Pop the solver frame (discards constraints added since begin)
        self.pop();

        // Restore constraint count
        if let Some(prev_count) = self.push_constraint_counts.lock().pop() {
            self.constraint_count.store(prev_count, Ordering::SeqCst);
        }

        // Truncate local Z3 cache to pre-transaction length
        if let Some(prev_len) = self.push_local_cache_lengths.lock().pop() {
            self.z3_assertions_local.lock().truncate(prev_len);
        }

        // Truncate local assumed_constraints to pre-transaction length
        if let Some(prev_len) = self.push_assumed_local_lengths.lock().pop() {
            self.assumed_constraints_local.lock().truncate(prev_len);
        }

        self.push_level.fetch_sub(1, Ordering::SeqCst);

        Ok(())
    }

    /// Get the current transaction level.
    ///
    /// Returns 0 if no transaction is active.
    pub fn current_push_level(&self) -> usize {
        self.push_level.load(Ordering::SeqCst)
    }

    /// Check if currently in a transaction.
    pub fn in_transaction(&self) -> bool {
        self.current_push_level() > 0
    }

    // Non-Z3 versions of transaction methods
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_begin(&self) {
        let current_count = self.constraint_count.load(Ordering::SeqCst);
        self.push_constraint_counts.lock().push(current_count);
        self.push_level.fetch_add(1, Ordering::SeqCst);
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_commit(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }
        self.push_constraint_counts.lock().pop();
        self.push_level.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn transaction_rollback(&self) -> Result<(), ConstraintSyncError> {
        let level = self.push_level.load(Ordering::SeqCst);
        if level == 0 {
            return Err(ConstraintSyncError::NoTransaction);
        }
        if let Some(prev_count) = self.push_constraint_counts.lock().pop() {
            self.constraint_count.store(prev_count, Ordering::SeqCst);
        }
        self.push_level.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    /// Get the unsat core as indices of constraints added.
    ///
    /// Returns the indices of constraints that form the unsatisfiable core.
    /// Call this after checking satisfiability and finding UNSAT.
    #[cfg(feature = "vex-engine-z3")]
    pub fn unsat_core(&self) -> Vec<usize> {
        let solver = self.solver();
        let core = solver.get_unsat_core();

        let trackers = self.constraint_trackers.lock();
        let mut result = Vec::new();

        // Match core tracking booleans to stored tracker indices by string representation
        for core_ast in core.iter() {
            let core_str = format!("{}", core_ast);
            for (i, tracker) in trackers.iter().enumerate() {
                if format!("{}", tracker) == core_str {
                    result.push(i);
                    break;
                }
            }
        }

        result
    }

    /// Get all solver assertions as strings.
    ///
    /// Returns string representations of all Z3 constraints. Useful for debugging
    /// and for syncing constraint state to Python. While not a full AST export,
    /// this allows Python to understand what constraints are active.
    #[cfg(feature = "vex-engine-z3")]
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        let solver = self.solver();
        solver.get_assertions().iter().map(|a| format!("{}", a)).collect()
    }

    /// Check the total number of assertions in the Z3 solver.
    ///
    /// This can be used to verify constraint sync between Rust and Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn z3_assertion_count(&self) -> usize {
        let solver = self.solver();
        solver.get_assertions().len()
    }

    // =========================================================================
    // Mock implementations when Z3 is not available
    // =========================================================================

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn is_sat(&self) -> bool {
        // Without Z3, we assume everything is satisfiable
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_true(&self, cond: &RustBV) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v != 0;
        }
        // Without Z3, assume symbolic can be true
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn can_be_false(&self, cond: &RustBV) -> bool {
        // Quick check for concrete values
        if let Some(v) = cond.as_u128() {
            return v == 0;
        }
        // Without Z3, assume symbolic can be false
        true
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Without Z3, can only evaluate concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_wide(&self, bv: &RustBV) -> Option<Vec<u8>> {
        // Without Z3, can only evaluate concrete values
        let width = bv.width();
        bv.as_u128().map(|v| {
            let byte_len = ((width + 7) / 8) as usize;
            let bytes = v.to_be_bytes();
            if byte_len <= 16 {
                bytes[16 - byte_len..].to_vec()
            } else {
                let mut result = vec![0u8; byte_len];
                result[byte_len - 16..].copy_from_slice(&bytes);
                result
            }
        })
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        if n == 0 {
            return vec![];
        }
        // Without Z3, can only return concrete values
        bv.as_u128().map(|v| vec![v]).unwrap_or_default()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto_wide(&self, bv: &RustBV, n: usize) -> Vec<Vec<u8>> {
        if n == 0 {
            return vec![];
        }
        self.eval_wide(bv).map(|v| vec![v]).unwrap_or_default()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn min(&self, bv: &RustBV, _signed: bool) -> Option<u128> {
        // Without Z3, can only return concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn max(&self, bv: &RustBV, _signed: bool) -> Option<u128> {
        // Without Z3, can only return concrete values
        bv.as_u128()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn solution(&self, bv: &RustBV, value: u128) -> bool {
        // Without Z3, can only check concrete values
        bv.as_u128().map(|v| v == value).unwrap_or(true)
    }

    /// Get the range [min, max] of possible values for a bitvector.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn range(&self, bv: &RustBV) -> Option<(u128, u128)> {
        // Without Z3, can only return range for concrete values
        bv.as_u128().map(|v| (v, v))
    }

    /// Seeded range — without Z3, equivalent to range().
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn range_seeded(
        &self,
        bv: &RustBV,
        _smallest_known: u128,
        _largest_known: u128,
    ) -> Option<(u128, u128)> {
        bv.as_u128().map(|v| (v, v))
    }

    /// Get up to n concrete solutions for a bitvector.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn solutions(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        if n == 0 {
            return vec![];
        }
        // Without Z3, can only return concrete values
        bv.as_u128().map(|v| vec![v]).unwrap_or_default()
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn push(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn pop(&self) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn assume_true(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Track for export to Python; no Z3 to assert against.
        self.assumed_constraints_local.lock().push((cond.clone(), true));
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn assume_false(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        self.assumed_constraints_local.lock().push((cond.clone(), false));
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

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn set_timeout(&self, _timeout_ms: u32) {
        // No-op without Z3
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn timeout_ms(&self) -> u32 {
        0
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn unsat_core(&self) -> Vec<usize> {
        // Without Z3, no unsat core available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        // Without Z3, no constraints available
        vec![]
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn z3_assertion_count(&self) -> usize {
        // Without Z3, no assertions
        0
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
        let local = self.assumed_constraints_local.lock();
        let mut out = Vec::with_capacity(shared.len() + local.len());
        out.extend_from_slice(&shared);
        out.extend_from_slice(&local);
        out
    }

    /// Get the number of assumed constraints.
    pub fn assumed_constraint_count(&self) -> usize {
        self.assumed_constraints_shared.lock().len() + self.assumed_constraints_local.lock().len()
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
    #[cfg(feature = "vex-engine-z3")]
    pub fn fork(&self) -> Self {
        let in_transaction = self.push_level.load(Ordering::Relaxed) > 0;
        let frozen_shared = freeze_z3_assertions(
            &self.z3_assertions_shared,
            &self.z3_assertions_local,
            in_transaction,
        );
        let frozen_assumed = freeze_assumed_constraints(
            &self.assumed_constraints_shared,
            &self.assumed_constraints_local,
            in_transaction,
        );
        let assumed_total_len = frozen_assumed.len();

        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(assumed_total_len),
            symbol_table: Arc::clone(&self.symbol_table),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            push_local_cache_lengths: Mutex::new(PushStack::new()),
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(frozen_assumed),
            assumed_constraints_local: Mutex::new(Vec::new()),
            z3_assertions_shared: Mutex::new(frozen_shared),
            z3_assertions_local: Mutex::new(Vec::new()),
            solver: Mutex::new(None),
            sat_cache: Cell::new(None),
            model_cache: RefCell::new(None),
            constraint_trackers: Mutex::new(Vec::new()),
            timeout_ms: AtomicU32::new(self.timeout_ms.load(Ordering::SeqCst)),
        }
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn fork(&self) -> Self {
        let in_transaction = self.push_level.load(Ordering::Relaxed) > 0;
        let frozen_assumed = freeze_assumed_constraints(
            &self.assumed_constraints_shared,
            &self.assumed_constraints_local,
            in_transaction,
        );

        SymContext {
            next_id: AtomicU64::new(self.next_id.load(Ordering::SeqCst)),
            constraint_count: AtomicUsize::new(0),
            symbol_table: Arc::clone(&self.symbol_table),
            push_level: AtomicUsize::new(0),
            push_constraint_counts: Mutex::new(PushStack::new()),
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
            assumed_constraints_shared: Mutex::new(frozen_assumed),
            assumed_constraints_local: Mutex::new(Vec::new()),
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
        let all_contexts: Vec<&SymContext> = std::iter::once(self).chain(others.iter().copied()).collect();
        let mut all_z3_conditions = Vec::new();

        for (ctx, cond) in all_contexts.iter().zip(merge_conditions.iter()) {
            // Compute NOT(cond) up front so cond_bool can be moved into
            // all_z3_conditions without cloning the Z3 AST.
            let cond_bool = cond.to_z3_bool();
            let not_cond = cond_bool.not();
            all_z3_conditions.push(cond_bool);

            // Collect all Z3 assertions from this context
            let shared = Arc::clone(&ctx.z3_assertions_shared.lock());
            let local = ctx.z3_assertions_local.lock();

            // For each constraint c_j in context i:
            //   assert (NOT merge_cond_i OR c_j)
            // This means: if this merge path is active, all its constraints hold
            for assertion in shared.iter().chain(local.iter()) {
                let guarded = z3::ast::Bool::or(&[&not_cond, assertion]);
                merged.z3_assertions_local.lock().push(guarded.clone());
                merged.add_constraint(guarded);
            }

            // Also merge assumed_constraints for Python export
            let assumed_shared = Arc::clone(&ctx.assumed_constraints_shared.lock());
            let assumed_local = ctx.assumed_constraints_local.lock();
            let mut merged_assumed_local = merged.assumed_constraints_local.lock();
            merged_assumed_local.extend(assumed_shared.iter().cloned());
            merged_assumed_local.extend(assumed_local.iter().cloned());
        }

        // Assert that at least one merge condition is true
        let cond_refs: Vec<&z3::ast::Bool> = all_z3_conditions.iter().collect();
        let or_conds = z3::ast::Bool::or(&cond_refs);
        merged.z3_assertions_local.lock().push(or_conds.clone());
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
            let mut merged_assumed = merged.assumed_constraints_local.lock();
            let self_shared = Arc::clone(&self.assumed_constraints_shared.lock());
            merged_assumed.extend(self_shared.iter().cloned());
            merged_assumed.extend(self.assumed_constraints_local.lock().iter().cloned());
            for other in others {
                let other_shared = Arc::clone(&other.assumed_constraints_shared.lock());
                merged_assumed.extend(other_shared.iter().cloned());
                merged_assumed.extend(other.assumed_constraints_local.lock().iter().cloned());
            }
        }

        merged
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

/// Freeze the local Z3 assertions vector into the shared Arc.
///
/// Outside a push/pop transaction (when `in_transaction` is false) this drains
/// `local` into `shared` in place — when shared has unique ownership the move
/// avoids the per-element Bool clones (each clone is a Z3_inc_ref FFI call).
/// Inside a transaction we must preserve `local` so `transaction_rollback` can
/// truncate it; in that case we fall back to allocating a fresh Vec by cloning
/// shared and moving local's elements without draining the original local.
#[cfg(feature = "vex-engine-z3")]
fn freeze_z3_assertions(
    shared: &Mutex<Arc<Vec<z3::ast::Bool>>>,
    local: &Mutex<Vec<z3::ast::Bool>>,
    in_transaction: bool,
) -> Arc<Vec<z3::ast::Bool>> {
    let mut local_guard = local.lock();
    if local_guard.is_empty() {
        return Arc::clone(&shared.lock());
    }
    let mut shared_guard = shared.lock();
    if in_transaction {
        // Cannot mutate local — rollback expects it intact.
        let mut merged = Vec::with_capacity(shared_guard.len() + local_guard.len());
        merged.extend_from_slice(&shared_guard);
        merged.extend_from_slice(&local_guard);
        return Arc::new(merged);
    }
    if let Some(inner) = Arc::get_mut(&mut *shared_guard) {
        // Unique ownership: in-place append, no element clones either side.
        inner.reserve(local_guard.len());
        inner.append(&mut *local_guard);
    } else {
        // Aliased: allocate new Vec, but move local's elements (no local clones).
        let mut merged = Vec::with_capacity(shared_guard.len() + local_guard.len());
        merged.extend_from_slice(&shared_guard);
        merged.append(&mut *local_guard);
        *shared_guard = Arc::new(merged);
    }
    Arc::clone(&shared_guard)
}

/// Freeze the local assumed-constraints vector into the shared Arc.
/// See `freeze_z3_assertions` for invariants.
fn freeze_assumed_constraints(
    shared: &Mutex<Arc<Vec<(RustBV, bool)>>>,
    local: &Mutex<Vec<(RustBV, bool)>>,
    in_transaction: bool,
) -> Arc<Vec<(RustBV, bool)>> {
    let mut local_guard = local.lock();
    if local_guard.is_empty() {
        return Arc::clone(&shared.lock());
    }
    let mut shared_guard = shared.lock();
    if in_transaction {
        let mut merged = Vec::with_capacity(shared_guard.len() + local_guard.len());
        merged.extend_from_slice(&shared_guard);
        merged.extend_from_slice(&local_guard);
        return Arc::new(merged);
    }
    if let Some(inner) = Arc::get_mut(&mut *shared_guard) {
        inner.reserve(local_guard.len());
        inner.append(&mut *local_guard);
    } else {
        let mut merged = Vec::with_capacity(shared_guard.len() + local_guard.len());
        merged.extend_from_slice(&shared_guard);
        merged.append(&mut *local_guard);
        *shared_guard = Arc::new(merged);
    }
    Arc::clone(&shared_guard)
}

// =============================================================================
// Helper functions for parsing Z3 BV string representations
// =============================================================================

/// Parse a hex string to u128, taking low 128 bits if larger.
#[cfg(feature = "vex-engine-z3")]
fn parse_wide_hex_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 32 hex chars), take low 32 chars
    let low_hex = if s.len() > 32 {
        &s[s.len() - 32..]
    } else {
        s
    };
    u128::from_str_radix(low_hex, 16).ok()
}

/// Parse a binary string to u128, taking low 128 bits if larger.
#[cfg(feature = "vex-engine-z3")]
fn parse_wide_binary_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 128 bin chars), take low 128 chars
    let low_bin = if s.len() > 128 {
        &s[s.len() - 128..]
    } else {
        s
    };
    u128::from_str_radix(low_bin, 2).ok()
}

/// Parse a hex string to full bytes (big-endian).
#[cfg(feature = "vex-engine-z3")]
fn parse_hex_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    let byte_len = ((width + 7) / 8) as usize;
    let mut result = vec![0u8; byte_len];

    // Pad hex string to even length
    let padded = if s.len() % 2 == 1 {
        format!("0{}", s)
    } else {
        s.to_string()
    };

    // Parse hex pairs from right to left (big-endian output)
    let hex_bytes: Vec<u8> = (0..padded.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&padded[i..i + 2], 16).ok())
        .collect();

    // Copy to result (right-aligned, big-endian)
    let offset = byte_len.saturating_sub(hex_bytes.len());
    for (i, &b) in hex_bytes.iter().enumerate() {
        if offset + i < byte_len {
            result[offset + i] = b;
        }
    }

    Some(result)
}

/// Parse a binary string to full bytes (big-endian).
#[cfg(feature = "vex-engine-z3")]
fn parse_binary_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    let byte_len = ((width + 7) / 8) as usize;
    let mut result = vec![0u8; byte_len];

    // Parse bits from right to left
    let bits: Vec<u8> = s.chars().filter_map(|c| match c {
        '0' => Some(0),
        '1' => Some(1),
        _ => None,
    }).collect();

    // Build bytes from bits (big-endian)
    let bit_offset = byte_len * 8 - bits.len();
    for (i, &bit) in bits.iter().enumerate() {
        let bit_pos = bit_offset + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = 7 - (bit_pos % 8);
        if byte_idx < byte_len {
            result[byte_idx] |= bit << bit_idx;
        }
    }

    Some(result)
}

/// Parse a decimal string to bytes (big-endian).
#[cfg(feature = "vex-engine-z3")]
fn parse_decimal_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    // For small values, parse and convert
    if let Ok(v) = s.parse::<u128>() {
        let byte_len = ((width + 7) / 8) as usize;
        let mut result = vec![0u8; byte_len];
        let bytes = v.to_be_bytes();
        let offset = byte_len.saturating_sub(16);
        for (i, &b) in bytes.iter().enumerate() {
            if offset + i < byte_len {
                result[offset + i] = b;
            }
        }
        return Some(result);
    }

    // For very large decimals, we'd need big integer parsing
    // This is rare in practice as Z3 typically uses hex format
    None
}

impl Default for SymContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!ctx.solution(&x, 5));  // 5 > 10 is false, should be unsat

        // Now add constraint: x < 20
        let twenty = RustBV::concrete(20, 32);
        let lt_twenty = x.ult(&twenty, &ctx);
        ctx.assume_true(&lt_twenty);

        // Verify both constraints are enforced
        assert!(ctx.solution(&x, 15));  // 10 < 15 < 20
        assert!(!ctx.solution(&x, 5));   // 5 < 10
        assert!(!ctx.solution(&x, 25));  // 25 > 20
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
        use z3::ast::Ast;

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

        assert!(!can_be_five, "x2 should have same constraints as x1 since same name");
    }
}
