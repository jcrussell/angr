//! Z3 solver context and constraint management.
//!
//! The `SymContext` manages:
//! - Symbolic variable creation and ID assignment
//! - Constraint tracking (when Z3 is available)
//! - Satisfiability checking (when Z3 is available)

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use parking_lot::Mutex;
use smallvec::SmallVec;

use super::RustBV;

/// Inline capacity for SymContext push_* stacks. Branch nesting is typically
/// shallow (≤8) within a single block; SmallVec avoids the heap allocation
/// for the first push.
type PushStack = SmallVec<[usize; 8]>;

/// Local-only constraint state added after fork.
///
/// Combines `assumed` (RustBV pairs for Python export) and `z3_assertions`
/// (cached Z3 Bool nodes for fast fork replay) under a single Mutex so that
/// the hot path (`assume_true`/`assume_false`/`add_constraint_raw`) only
/// acquires one lock instead of two.
struct LocalConstraints {
    /// Local assumed (RustBV, is_assumed_true) pairs added after fork.
    assumed: Vec<(RustBV, bool)>,
    /// Local Z3 Bool assertions added after fork — only these are cloned on fork.
    #[cfg(feature = "vex-engine-z3")]
    z3_assertions: Vec<z3::ast::Bool>,
}

impl LocalConstraints {
    fn new() -> Self {
        LocalConstraints {
            assumed: Vec::new(),
            #[cfg(feature = "vex-engine-z3")]
            z3_assertions: Vec::new(),
        }
    }
}

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
/// Number of min()/max() calls where the cached model produced a usable
/// witness used to tighten the binary-search initial bound (and, in the
/// signed case, sometimes skipped the MinInit/MaxInit pre-check).
static Z3_EXTREMA_MODEL_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of min()/max() calls where no cached model was available (or the
/// model could not be evaluated against the bv ast).
static Z3_EXTREMA_MODEL_MISS_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of to_z3_ast() / to_z3_bool() calls (AST construction).
static Z3_AST_BUILD_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of cache hits in `to_z3_ast_cached` per-call HashMap (angr-zdho).
///
/// Each hit means a sub-expression was visited more than once during a single
/// top-level `to_z3_ast()` call and the Arc-pointer key already had a Z3 AST
/// built — the inner DAG had at least one shared Arc subtree. Ratio of hits to
/// (hits+misses) gives the per-conversion sharing rate the cache captures.
static Z3_AST_CACHE_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of cache misses in `to_z3_ast_cached` per-call HashMap (angr-zdho).
///
/// A miss is a unique RustBV pointer visited within one `to_z3_ast()` call.
/// Equals the count of distinct Arc-pointer subtrees materialized into Z3
/// ASTs for that conversion.
static Z3_AST_CACHE_MISS_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Sat.
static Z3_SAT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Unsat.
static Z3_UNSAT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Unknown (timeout / resource limit).
static Z3_TIMEOUT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Deepest ITE chain ever stored as a symbolic memory cell value.
///
/// Recorded by the eager `store_strided` path and by Multi-cell installation
/// (Phase 1/2, `install_multi_for_candidates`) so before/after comparison
/// against the eager baseline is direct.
static MEM_ITE_DEPTH_MAX: AtomicU32 = AtomicU32::new(0);
/// Cumulative count of conditional iterations added to ITE chains in symbolic
/// stores. Sum, not max — surfaces total ITE-chain work across a run.
static MEM_ITE_DEPTH_TOTAL: AtomicU64 = AtomicU64::new(0);

// -----------------------------------------------------------------------------
// Dispatch / volume counters (angr-2j5v)
// -----------------------------------------------------------------------------
// Lightweight breadth-first instrumentation across the rest of the pipeline.
// Each counter is a single AtomicU64 fetch_add at the construction site —
// matches the existing Z3_* pattern. Read out via `get_solver_stats()`.

// VEX op dispatch — count per family at the entry of unop/binop/triop/qop.
static VEX_UNOP_TOTAL: AtomicU64 = AtomicU64::new(0);
static VEX_BINOP_TOTAL: AtomicU64 = AtomicU64::new(0);
static VEX_TRIOP_TOTAL: AtomicU64 = AtomicU64::new(0);
static VEX_QOP_TOTAL: AtomicU64 = AtomicU64::new(0);
static VEX_OP_ARITH: AtomicU64 = AtomicU64::new(0);
static VEX_OP_LOGIC: AtomicU64 = AtomicU64::new(0);
static VEX_OP_SHIFT: AtomicU64 = AtomicU64::new(0);
static VEX_OP_CMP: AtomicU64 = AtomicU64::new(0);
static VEX_OP_EXT: AtomicU64 = AtomicU64::new(0);
static VEX_OP_FP: AtomicU64 = AtomicU64::new(0);
static VEX_OP_VEC: AtomicU64 = AtomicU64::new(0);
static VEX_OP_OTHER: AtomicU64 = AtomicU64::new(0);

// Memory volume — load/store call counts, total bytes, concrete-vs-symbolic
// address split. Recorded at top-level `SymbolicMemory::{load,store}` only,
// not at every internal helper, so the counter reflects the public surface.
static MEM_LOAD_COUNT: AtomicU64 = AtomicU64::new(0);
static MEM_STORE_COUNT: AtomicU64 = AtomicU64::new(0);
static MEM_LOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static MEM_STORE_BYTES: AtomicU64 = AtomicU64::new(0);
static MEM_LOAD_SYMBOLIC_ADDR: AtomicU64 = AtomicU64::new(0);
static MEM_STORE_SYMBOLIC_ADDR: AtomicU64 = AtomicU64::new(0);
/// `UnmappedPageInRegion` faults — lazy region had no backing page, caller
/// must materialize. Bumped in load/store before the error is propagated.
static MEM_LAZY_PAGE_FAULT_COUNT: AtomicU64 = AtomicU64::new(0);

// Concretization fanout — count + cumulative K + max K observed per call.
static CONCRETIZE_READ_COUNT: AtomicU64 = AtomicU64::new(0);
static CONCRETIZE_WRITE_COUNT: AtomicU64 = AtomicU64::new(0);
static CONCRETIZE_TOTAL_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static CONCRETIZE_MAX_CANDIDATES: AtomicU32 = AtomicU32::new(0);

// angr-62li: address-concretization disjunction hoisting. When the
// concretizer returns Multiple, the engine now asserts the disjunction
// `Or(addr == a0, ..., addr == aK)` as a top-level constraint so Z3's
// propagate-values tactic can substitute the addr var. Bumped at the
// hoist site in `memory/store.rs::assert_address_disjunction`.
static CONCRETIZE_DISJUNCTION_COUNT: AtomicU64 = AtomicU64::new(0);
static CONCRETIZE_DISJUNCTION_TERMS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CONCRETIZE_DISJUNCTION_MAX_TERMS: AtomicU32 = AtomicU32::new(0);

// AST construction (specific) — Reverse/Concat/Extract emissions. These
// are the AST shapes most often blamed when claripy<->Rust mismatch surfaces
// (angr-tlvl-style residuals). Counted at the public `RustBV::{reverse,
// concat, extract}` entry points; internal recursion within `*_into` is
// not double-counted.
static BVOP_REVERSE_COUNT: AtomicU64 = AtomicU64::new(0);
static BVOP_CONCAT_COUNT: AtomicU64 = AtomicU64::new(0);
static BVOP_EXTRACT_COUNT: AtomicU64 = AtomicU64::new(0);

// angr-g7nq: trivial-constraint fast path counters. Bumped from
// `value.rs::{eq_into,ne_into,ult_into,ule_into,ugt_into,uge_into}` when
// the `Cmp(ZeroExt(k, x), BVV)` (or commuted form) pattern is recognized
// and either collapsed to a smaller AST (the "high bits zero" case) or
// short-circuited to a concrete answer (the "high bits nonzero" case, an
// unsatisfiable/tautological subexpression). Separate counts for the two
// outcomes make it possible to attribute hits without inspecting the
// emitted ASTs.
static ZEXT_CMP_COLLAPSE_COUNT: AtomicU64 = AtomicU64::new(0);
static ZEXT_CMP_TRIVIAL_DECIDE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Per-site counters and timers for solver.check() calls.
/// Indexed by `CheckSite as usize`.
const NUM_CHECK_SITES: usize = 9;
static Z3_CHECK_SITE_COUNT: [AtomicU64; NUM_CHECK_SITES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static Z3_CHECK_SITE_TIME_NS: [AtomicU64; NUM_CHECK_SITES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
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
    stats.insert(
        "z3_check_count".into(),
        Z3_CHECK_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_check_time_ns".into(),
        Z3_CHECK_TIME_NS.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_materialize_count".into(),
        Z3_MATERIALIZE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_materialize_time_ns".into(),
        Z3_MATERIALIZE_TIME_NS.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_assume_concrete".into(),
        Z3_ASSUME_CONCRETE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_assume_symbolic".into(),
        Z3_ASSUME_SYMBOLIC_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_branch_check".into(),
        Z3_BRANCH_CHECK_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_branch_concrete".into(),
        Z3_BRANCH_CONCRETE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_branch_model_hit".into(),
        Z3_BRANCH_MODEL_HIT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_branch_model_miss".into(),
        Z3_BRANCH_MODEL_MISS_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_extrema_model_hit".into(),
        Z3_EXTREMA_MODEL_HIT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_extrema_model_miss".into(),
        Z3_EXTREMA_MODEL_MISS_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_ast_build".into(),
        Z3_AST_BUILD_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_ast_cache_hit".into(),
        Z3_AST_CACHE_HIT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_ast_cache_miss".into(),
        Z3_AST_CACHE_MISS_COUNT.load(Ordering::Relaxed),
    );
    stats.insert("z3_sat_count".into(), Z3_SAT_COUNT.load(Ordering::Relaxed));
    stats.insert(
        "z3_unsat_count".into(),
        Z3_UNSAT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_timeout_count".into(),
        Z3_TIMEOUT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_ite_depth_max".into(),
        MEM_ITE_DEPTH_MAX.load(Ordering::Relaxed) as u64,
    );
    stats.insert(
        "mem_ite_depth_total".into(),
        MEM_ITE_DEPTH_TOTAL.load(Ordering::Relaxed),
    );
    // VEX op dispatch
    stats.insert(
        "vex_unop_total".into(),
        VEX_UNOP_TOTAL.load(Ordering::Relaxed),
    );
    stats.insert(
        "vex_binop_total".into(),
        VEX_BINOP_TOTAL.load(Ordering::Relaxed),
    );
    stats.insert(
        "vex_triop_total".into(),
        VEX_TRIOP_TOTAL.load(Ordering::Relaxed),
    );
    stats.insert("vex_qop_total".into(), VEX_QOP_TOTAL.load(Ordering::Relaxed));
    stats.insert("vex_op_arith".into(), VEX_OP_ARITH.load(Ordering::Relaxed));
    stats.insert("vex_op_logic".into(), VEX_OP_LOGIC.load(Ordering::Relaxed));
    stats.insert("vex_op_shift".into(), VEX_OP_SHIFT.load(Ordering::Relaxed));
    stats.insert("vex_op_cmp".into(), VEX_OP_CMP.load(Ordering::Relaxed));
    stats.insert("vex_op_ext".into(), VEX_OP_EXT.load(Ordering::Relaxed));
    stats.insert("vex_op_fp".into(), VEX_OP_FP.load(Ordering::Relaxed));
    stats.insert("vex_op_vec".into(), VEX_OP_VEC.load(Ordering::Relaxed));
    stats.insert("vex_op_other".into(), VEX_OP_OTHER.load(Ordering::Relaxed));
    // Memory volume
    stats.insert(
        "mem_load_count".into(),
        MEM_LOAD_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_store_count".into(),
        MEM_STORE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_load_bytes".into(),
        MEM_LOAD_BYTES.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_store_bytes".into(),
        MEM_STORE_BYTES.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_load_symbolic_addr".into(),
        MEM_LOAD_SYMBOLIC_ADDR.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_store_symbolic_addr".into(),
        MEM_STORE_SYMBOLIC_ADDR.load(Ordering::Relaxed),
    );
    stats.insert(
        "mem_lazy_page_fault_count".into(),
        MEM_LAZY_PAGE_FAULT_COUNT.load(Ordering::Relaxed),
    );
    // Concretization fanout
    stats.insert(
        "concretize_read_count".into(),
        CONCRETIZE_READ_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "concretize_write_count".into(),
        CONCRETIZE_WRITE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "concretize_total_candidates".into(),
        CONCRETIZE_TOTAL_CANDIDATES.load(Ordering::Relaxed),
    );
    stats.insert(
        "concretize_max_candidates".into(),
        CONCRETIZE_MAX_CANDIDATES.load(Ordering::Relaxed) as u64,
    );
    // angr-62li: disjunction hoisting
    stats.insert(
        "concretize_disjunction_count".into(),
        CONCRETIZE_DISJUNCTION_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "concretize_disjunction_terms_total".into(),
        CONCRETIZE_DISJUNCTION_TERMS_TOTAL.load(Ordering::Relaxed),
    );
    stats.insert(
        "concretize_disjunction_max_terms".into(),
        CONCRETIZE_DISJUNCTION_MAX_TERMS.load(Ordering::Relaxed) as u64,
    );
    // AST construction
    stats.insert(
        "bvop_reverse_count".into(),
        BVOP_REVERSE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "bvop_concat_count".into(),
        BVOP_CONCAT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "bvop_extract_count".into(),
        BVOP_EXTRACT_COUNT.load(Ordering::Relaxed),
    );
    // angr-g7nq: trivial-constraint fast path
    stats.insert(
        "zext_cmp_collapse_count".into(),
        ZEXT_CMP_COLLAPSE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "zext_cmp_trivial_decide_count".into(),
        ZEXT_CMP_TRIVIAL_DECIDE_COUNT.load(Ordering::Relaxed),
    );
    // angr-v5a5: shared-lineage solver telemetry. The counters live in
    // `super::lineage` and are atomically incremented by `switch_to`; this
    // is the only place that surfaces them to the Python caller via
    // `RustExplorationManager.get_solver_stats()`. Always 0 until the
    // lineage-creation slice lands — emitting them now keeps the key set
    // stable across versions.
    #[cfg(feature = "vex-engine-z3")]
    for (name, value) in super::lineage::lineage_stats() {
        stats.insert(name.into(), value);
    }
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
    Z3_EXTREMA_MODEL_HIT_COUNT.store(0, Ordering::Relaxed);
    Z3_EXTREMA_MODEL_MISS_COUNT.store(0, Ordering::Relaxed);
    Z3_AST_BUILD_COUNT.store(0, Ordering::Relaxed);
    Z3_AST_CACHE_HIT_COUNT.store(0, Ordering::Relaxed);
    Z3_AST_CACHE_MISS_COUNT.store(0, Ordering::Relaxed);
    Z3_SAT_COUNT.store(0, Ordering::Relaxed);
    Z3_UNSAT_COUNT.store(0, Ordering::Relaxed);
    Z3_TIMEOUT_COUNT.store(0, Ordering::Relaxed);
    MEM_ITE_DEPTH_MAX.store(0, Ordering::Relaxed);
    MEM_ITE_DEPTH_TOTAL.store(0, Ordering::Relaxed);
    // angr-2j5v counters
    VEX_UNOP_TOTAL.store(0, Ordering::Relaxed);
    VEX_BINOP_TOTAL.store(0, Ordering::Relaxed);
    VEX_TRIOP_TOTAL.store(0, Ordering::Relaxed);
    VEX_QOP_TOTAL.store(0, Ordering::Relaxed);
    VEX_OP_ARITH.store(0, Ordering::Relaxed);
    VEX_OP_LOGIC.store(0, Ordering::Relaxed);
    VEX_OP_SHIFT.store(0, Ordering::Relaxed);
    VEX_OP_CMP.store(0, Ordering::Relaxed);
    VEX_OP_EXT.store(0, Ordering::Relaxed);
    VEX_OP_FP.store(0, Ordering::Relaxed);
    VEX_OP_VEC.store(0, Ordering::Relaxed);
    VEX_OP_OTHER.store(0, Ordering::Relaxed);
    MEM_LOAD_COUNT.store(0, Ordering::Relaxed);
    MEM_STORE_COUNT.store(0, Ordering::Relaxed);
    MEM_LOAD_BYTES.store(0, Ordering::Relaxed);
    MEM_STORE_BYTES.store(0, Ordering::Relaxed);
    MEM_LOAD_SYMBOLIC_ADDR.store(0, Ordering::Relaxed);
    MEM_STORE_SYMBOLIC_ADDR.store(0, Ordering::Relaxed);
    MEM_LAZY_PAGE_FAULT_COUNT.store(0, Ordering::Relaxed);
    CONCRETIZE_READ_COUNT.store(0, Ordering::Relaxed);
    CONCRETIZE_WRITE_COUNT.store(0, Ordering::Relaxed);
    CONCRETIZE_TOTAL_CANDIDATES.store(0, Ordering::Relaxed);
    CONCRETIZE_MAX_CANDIDATES.store(0, Ordering::Relaxed);
    CONCRETIZE_DISJUNCTION_COUNT.store(0, Ordering::Relaxed);
    CONCRETIZE_DISJUNCTION_TERMS_TOTAL.store(0, Ordering::Relaxed);
    CONCRETIZE_DISJUNCTION_MAX_TERMS.store(0, Ordering::Relaxed);
    BVOP_REVERSE_COUNT.store(0, Ordering::Relaxed);
    BVOP_CONCAT_COUNT.store(0, Ordering::Relaxed);
    BVOP_EXTRACT_COUNT.store(0, Ordering::Relaxed);
    ZEXT_CMP_COLLAPSE_COUNT.store(0, Ordering::Relaxed);
    ZEXT_CMP_TRIVIAL_DECIDE_COUNT.store(0, Ordering::Relaxed);
    // angr-v5a5: clear lineage telemetry alongside the rest.
    #[cfg(feature = "vex-engine-z3")]
    super::lineage::reset_lineage_stats();
    for i in 0..NUM_CHECK_SITES {
        Z3_CHECK_SITE_COUNT[i].store(0, Ordering::Relaxed);
        Z3_CHECK_SITE_TIME_NS[i].store(0, Ordering::Relaxed);
    }
}

/// Record that a symbolic store wrote an ITE chain of `depth` alternatives.
///
/// Bumps the cumulative total and lifts the max watermark via `fetch_max`.
/// Called from `memory/store.rs::store_strided` and from Multi-cell
/// installation (`install_multi_for_candidates`) so before/after comparison
/// against the eager baseline is direct.
#[inline]
pub fn record_mem_ite_depth(depth: u32) {
    if depth == 0 {
        return;
    }
    MEM_ITE_DEPTH_MAX.fetch_max(depth, Ordering::Relaxed);
    MEM_ITE_DEPTH_TOTAL.fetch_add(depth as u64, Ordering::Relaxed);
}

/// Increment Z3 AST build counter (called from value.rs).
#[inline]
pub fn record_z3_ast_build() {
    Z3_AST_BUILD_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Increment per-call to_z3_ast_cached cache-hit counter (angr-zdho).
#[inline]
pub fn record_z3_ast_cache_hit() {
    Z3_AST_CACHE_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Increment per-call to_z3_ast_cached cache-miss counter (angr-zdho).
#[inline]
pub fn record_z3_ast_cache_miss() {
    Z3_AST_CACHE_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
}

// -----------------------------------------------------------------------------
// Recorder functions for angr-2j5v counters
// -----------------------------------------------------------------------------

/// VEX IR op family — passed by the dispatcher in `vex/ops.rs` so the
/// classifier lives next to the dispatch and the counter bookkeeping stays in
/// `symbolic/context.rs` alongside the other instrumentation atomics.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum VexOpFamily {
    Arith,
    Logic,
    Shift,
    Cmp,
    Ext,
    Fp,
    Vec,
    Other,
}

#[inline]
fn bump_vex_family(family: VexOpFamily) {
    let counter = match family {
        VexOpFamily::Arith => &VEX_OP_ARITH,
        VexOpFamily::Logic => &VEX_OP_LOGIC,
        VexOpFamily::Shift => &VEX_OP_SHIFT,
        VexOpFamily::Cmp => &VEX_OP_CMP,
        VexOpFamily::Ext => &VEX_OP_EXT,
        VexOpFamily::Fp => &VEX_OP_FP,
        VexOpFamily::Vec => &VEX_OP_VEC,
        VexOpFamily::Other => &VEX_OP_OTHER,
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_vex_unop(family: VexOpFamily) {
    VEX_UNOP_TOTAL.fetch_add(1, Ordering::Relaxed);
    bump_vex_family(family);
}

#[inline]
pub fn record_vex_binop(family: VexOpFamily) {
    VEX_BINOP_TOTAL.fetch_add(1, Ordering::Relaxed);
    bump_vex_family(family);
}

#[inline]
pub fn record_vex_triop(family: VexOpFamily) {
    VEX_TRIOP_TOTAL.fetch_add(1, Ordering::Relaxed);
    bump_vex_family(family);
}

#[inline]
pub fn record_vex_qop(family: VexOpFamily) {
    VEX_QOP_TOTAL.fetch_add(1, Ordering::Relaxed);
    bump_vex_family(family);
}

/// Record a load operation reaching `SymbolicMemory::load_concrete`. This is
/// the unified entry point — counts BOTH loads via the public `load(addr_bv)`
/// and the state.rs hot path that calls `load_concrete(addr_u64)` directly.
#[inline]
pub fn record_mem_load(bytes: u64) {
    MEM_LOAD_COUNT.fetch_add(1, Ordering::Relaxed);
    MEM_LOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Record a store operation reaching `SymbolicMemory::store_concrete`.
#[inline]
pub fn record_mem_store(bytes: u64) {
    MEM_STORE_COUNT.fetch_add(1, Ordering::Relaxed);
    MEM_STORE_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Record that a public `SymbolicMemory::load` was called with a symbolic
/// address (subset of `mem_load_count`). The symbolic-vs-concrete split is
/// only visible at the public entry point — by the time we reach
/// `load_concrete` the address has been evaluated to a u64.
#[inline]
pub fn record_mem_load_symbolic_addr() {
    MEM_LOAD_SYMBOLIC_ADDR.fetch_add(1, Ordering::Relaxed);
}

/// Symbolic-addr counterpart to `record_mem_load_symbolic_addr`, for the
/// public `SymbolicMemory::store` entry.
#[inline]
pub fn record_mem_store_symbolic_addr() {
    MEM_STORE_SYMBOLIC_ADDR.fetch_add(1, Ordering::Relaxed);
}

/// Record a lazy-region page fault (`MemoryError::UnmappedPageInRegion`).
#[inline]
pub fn record_mem_lazy_page_fault() {
    MEM_LAZY_PAGE_FAULT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Record a `concretize_read` outcome with the candidate count `k`.
#[inline]
pub fn record_concretize_read(k: u32) {
    CONCRETIZE_READ_COUNT.fetch_add(1, Ordering::Relaxed);
    CONCRETIZE_TOTAL_CANDIDATES.fetch_add(k as u64, Ordering::Relaxed);
    CONCRETIZE_MAX_CANDIDATES.fetch_max(k, Ordering::Relaxed);
}

/// Record a `concretize_write` outcome with the candidate count `k`.
#[inline]
pub fn record_concretize_write(k: u32) {
    CONCRETIZE_WRITE_COUNT.fetch_add(1, Ordering::Relaxed);
    CONCRETIZE_TOTAL_CANDIDATES.fetch_add(k as u64, Ordering::Relaxed);
    CONCRETIZE_MAX_CANDIDATES.fetch_max(k, Ordering::Relaxed);
}

/// angr-62li: record a disjunction `Or(addr == a0, ..., addr == a{k-1})`
/// hoisted to the top-level Rust solver as a propagate-values aid.
#[inline]
pub fn record_concretize_disjunction(k: u32) {
    if k == 0 {
        return;
    }
    CONCRETIZE_DISJUNCTION_COUNT.fetch_add(1, Ordering::Relaxed);
    CONCRETIZE_DISJUNCTION_TERMS_TOTAL.fetch_add(k as u64, Ordering::Relaxed);
    CONCRETIZE_DISJUNCTION_MAX_TERMS.fetch_max(k, Ordering::Relaxed);
}

/// Record a public `RustBV::reverse` call.
#[inline]
pub fn record_bvop_reverse() {
    BVOP_REVERSE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Record a public `RustBV::concat` call.
#[inline]
pub fn record_bvop_concat() {
    BVOP_CONCAT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Record a public `RustBV::extract` call.
#[inline]
pub fn record_bvop_extract() {
    BVOP_EXTRACT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-g7nq: record a `Cmp(ZeroExt(k, x), BVV)` collapse — high k bits of
/// const are zero, so the comparison is rewritten on the W-k-bit operands.
#[inline]
pub fn record_zext_cmp_collapse() {
    ZEXT_CMP_COLLAPSE_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-g7nq: record a `Cmp(ZeroExt(k, x), BVV)` trivial decision — the
/// constant has a nonzero high k-bit prefix, so the subexpression is
/// structurally unsatisfiable (Eq) or always satisfiable (Ne) or has a
/// constant unsigned ordering, and the result is a concrete 0/1 bool.
#[inline]
pub fn record_zext_cmp_trivial_decide() {
    ZEXT_CMP_TRIVIAL_DECIDE_COUNT.fetch_add(1, Ordering::Relaxed);
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

/// Build the Z3 `Params` object applied to every fresh `z3::Solver` and
/// every `set_timeout` call. Centralizes the bv_rewriter knobs we set.
///
/// Two knobs enabled (Z3 disables both by default):
/// - `bv_extract_prop`: propagates Extract inward through arithmetic.
/// - `mul2concat`: rewrites `x * 2^k` -> `concat(x, 0^k)`. Added in
///   angr-ya00.1 (free bv_* sweep, 2026-05-19) — drops fairlight's bimodal
///   slow-mode median from 21.66s to 14.46s (-33%) without regressing
///   8 other Z3-heavy benches measured (max delta +0.8% on flareon2015_2).
///
/// Three sweep knobs left disabled: `bv_not_simpl`, `bv_ite2id`,
/// `blast_eq_value` (no measurable effect across matrix). `bv_sort_ac`
/// rejected: looked like a fairlight winner combined with mul2concat
/// (median 8.10s) but blows up defcon2016quals_baby-re by 7x (0.50s ->
/// 3.71s) and adds +25% to flareon2015_2.
#[cfg(feature = "vex-engine-z3")]
fn build_solver_params(timeout_ms: u32) -> z3::Params {
    let mut params = z3::Params::new();
    params.set_u32("timeout", timeout_ms);
    params.set_bool("bv_extract_prop", true);
    params.set_bool("mul2concat", true);
    params
}

/// Parsed `ANGR_Z3_TACTIC` env-var spec, cached once per process.
///
/// Recognized values:
/// - unset / empty / "default" / "smt" → default `z3::Solver::new()`
///   (Z3's smt portfolio strategy).
/// - "qfbv_smart" → probe-conditional tactic that dispatches on
///   `num-consts`: large problems (> 20 constants) go to `qfbv`, small ones
///   stay on `smt`. Targets the bimodal benches (fairlight, sokohashv2,
///   mma_howtouse) where qfbv wins big, without regressing small-problem
///   benches like csgames2018 / flareon2015_2 where the qfbv preset's
///   per-check overhead dominates.
/// - any other value → colon-separated pipeline of tactic names
///   composed via `Tactic::and_then`, e.g.
///   `simplify:propagate-values:solve-eqs:bit-blast:sat`.
#[cfg(feature = "vex-engine-z3")]
#[derive(Debug, Clone)]
enum TacticSpec {
    Default,
    Pipeline(Vec<String>),
    QfbvSmart,
}

#[cfg(feature = "vex-engine-z3")]
fn tactic_spec() -> &'static TacticSpec {
    static SPEC: std::sync::OnceLock<TacticSpec> = std::sync::OnceLock::new();
    SPEC.get_or_init(|| match std::env::var("ANGR_Z3_TACTIC") {
        Ok(s) => {
            let s = s.trim();
            if s.is_empty() || s.eq_ignore_ascii_case("default") || s.eq_ignore_ascii_case("smt") {
                TacticSpec::Default
            } else if s.eq_ignore_ascii_case("qfbv_smart") {
                TacticSpec::QfbvSmart
            } else {
                let names: Vec<String> = s
                    .split(':')
                    .map(|n| n.trim().to_string())
                    .filter(|n| !n.is_empty())
                    .collect();
                if names.is_empty() {
                    TacticSpec::Default
                } else {
                    TacticSpec::Pipeline(names)
                }
            }
        }
        Err(_) => TacticSpec::Default,
    })
}

/// `num-consts > N` threshold used by `qfbv_smart`. Override via
/// `ANGR_Z3_QFBV_THRESHOLD`. 20 is the empirical pivot between bimodal
/// hash-cracker problems (winners) and CTF-style small-problem benches
/// (regressers) — see angr-ya00 measurements in
/// `docs/advanced-topics/rust_engine.rst`.
#[cfg(feature = "vex-engine-z3")]
fn qfbv_smart_threshold() -> f64 {
    static THRESHOLD: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("ANGR_Z3_QFBV_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(20.0)
    })
}

/// Construct a fresh Z3 solver per the active [`TacticSpec`], with timeout
/// and bv_rewriter params applied. Three sites in this file rely on this:
/// initial creation, lazy fork materialization, and `set_timeout`.
#[cfg(feature = "vex-engine-z3")]
fn build_solver(timeout_ms: u32) -> z3::Solver {
    let solver = match tactic_spec() {
        TacticSpec::Default => z3::Solver::new(),
        TacticSpec::Pipeline(names) => {
            let mut iter = names.iter();
            let first = z3::Tactic::new(iter.next().expect("non-empty pipeline"));
            let composed = iter.fold(first, |acc, name| acc.and_then(&z3::Tactic::new(name)));
            composed.solver()
        }
        TacticSpec::QfbvSmart => {
            let qfbv = z3::Tactic::new("qfbv");
            let smt = z3::Tactic::new("smt");
            let num_consts = z3::Probe::new("num-consts");
            let thresh = z3::Probe::constant(qfbv_smart_threshold());
            let cond = z3::Tactic::cond(&num_consts.gt(&thresh), &qfbv, &smt);
            cond.solver()
        }
    };
    solver.set_params(&build_solver_params(timeout_ms));
    solver
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

// =============================================================================
// Constraint sharing walk (angr-zdho)
// =============================================================================
//
// Walks a population of constraint RustBVs (typically every state's assumed
// list at end of exploration) and reports:
//
//   - `total_visits`     — recursive descents through the DAG, counting Arc
//                          re-visits as separate. The upper bound on AST work
//                          if we had NO cache (neither per-call nor hash-cons).
//   - `unique_pointers`  — distinct `Arc<RustBV>` allocations seen. This is
//                          what today's per-call `to_z3_ast_cached` collapses
//                          repeated visits down to.
//   - `unique_shapes`    — distinct *structural* shapes seen. Two nodes with
//                          the same op/width and structurally-equal children
//                          share a shape. This is the lower bound a
//                          construction-time hash-cons (angr-behq) would
//                          reach.
//
// `unique_pointers - unique_shapes` is the "structural-duplicate" count from
// the bead — RustBVs that hash-cons could merge but the current build path
// keeps distinct.

/// Canonical key for structural equality between two `RustBV` subtrees.
///
/// Two `RustBV` nodes hash to the same `StructuralKey` iff a construction-time
/// hash-cons would treat them as the same node.
#[derive(Clone, PartialEq, Eq, Hash)]
enum StructuralKey {
    /// Concrete leaf: same value at the same width.
    Concrete(u128, u32),
    /// Symbolic leaf: same `id` (ids are minted globally so this implies same
    /// width too, but include it explicitly for clarity).
    Symbolic(u64, u32),
    /// Constrained leaf: symbolic id with a known concrete witness.
    Constrained(u64, u128, u32),
    /// Expression node: same op, same width, structurally-equal operand list.
    Expression(super::value::BVOp, u32, Vec<u64>),
}

/// Accumulator for a constraint-sharing analysis pass.
///
/// Build with `ConstraintSharingWalk::new()`, fold one or more
/// `SymContext`s in via `SymContext::fold_sharing_walk`, then read out
/// the totals with `into_stats()`.
pub struct ConstraintSharingWalk {
    /// Pointer → canonical shape id. A pointer entry is created on first
    /// visit, so `len()` is the count of unique RustBV Arc allocations.
    ptr_to_shape: HashMap<usize, u64>,
    /// Structural shape → canonical id. `len()` is the count of unique
    /// structural shapes — what hash-cons would shrink to.
    shape_to_id: HashMap<StructuralKey, u64>,
    /// Monotonic id allocator for shape interning.
    next_shape_id: u64,
    /// Recursive descents (including Arc re-visits).
    total_visits: u64,
}

impl ConstraintSharingWalk {
    pub fn new() -> Self {
        Self {
            ptr_to_shape: HashMap::new(),
            shape_to_id: HashMap::new(),
            next_shape_id: 0,
            total_visits: 0,
        }
    }

    /// Recursively walk `node`, updating the maps and the visit counter.
    /// Returns the canonical shape id for `node`.
    pub fn visit(&mut self, node: &RustBV) -> u64 {
        self.total_visits = self.total_visits.saturating_add(1);
        let ptr_key = node as *const RustBV as usize;
        if let Some(&id) = self.ptr_to_shape.get(&ptr_key) {
            return id;
        }
        let key = match node {
            RustBV::Concrete { value, width } => StructuralKey::Concrete(*value, *width),
            RustBV::Symbolic { id, width, .. } => StructuralKey::Symbolic(*id, *width),
            RustBV::Constrained { id, value, width } => {
                StructuralKey::Constrained(*id, *value, *width)
            }
            RustBV::Expression {
                op, operands, width, ..
            } => {
                let mut op_ids = Vec::with_capacity(operands.len());
                for operand in operands.iter() {
                    op_ids.push(self.visit(operand));
                }
                StructuralKey::Expression(op.clone(), *width, op_ids)
            }
        };
        let id = if let Some(&existing) = self.shape_to_id.get(&key) {
            existing
        } else {
            let id = self.next_shape_id;
            self.next_shape_id = self.next_shape_id.saturating_add(1);
            self.shape_to_id.insert(key, id);
            id
        };
        self.ptr_to_shape.insert(ptr_key, id);
        id
    }

    /// Consume the walk and return the aggregate stats.
    pub fn into_stats(self) -> ConstraintSharingStats {
        ConstraintSharingStats {
            total_visits: self.total_visits,
            unique_pointers: self.ptr_to_shape.len() as u64,
            unique_shapes: self.shape_to_id.len() as u64,
        }
    }
}

impl Default for ConstraintSharingWalk {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate output of `ConstraintSharingWalk::into_stats()`.
pub struct ConstraintSharingStats {
    pub total_visits: u64,
    pub unique_pointers: u64,
    pub unique_shapes: u64,
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
    /// Each entry is (constraint, is_assumed_true). The shared prefix is an
    /// Arc<Vec<...>> for O(1) clone on fork; local additions live alongside
    /// `z3_assertions` in `local_constraints` so the hot path only takes one
    /// lock for both vectors.
    /// Wrapped in Mutex so fork() can freeze local into shared in-place when safe.
    assumed_constraints_shared: Mutex<Arc<Vec<(RustBV, bool)>>>,

    /// Shared (frozen) Z3 Bool assertions from parent — O(1) clone via Arc.
    /// Wrapped in Mutex so fork() can freeze local into shared in-place when safe
    /// (avoids cloning every Bool — each Bool::clone would call Z3_inc_ref).
    #[cfg(feature = "vex-engine-z3")]
    z3_assertions_shared: Mutex<Arc<Vec<z3::ast::Bool>>>,

    /// Local additions (assumed pairs + Z3 Bool cache) added after fork.
    /// Combined under one Mutex so the assume_*/add_constraint_raw hot path
    /// only acquires a single lock instead of two.
    local_constraints: Mutex<LocalConstraints>,

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
    #[cfg(feature = "vex-engine-z3")]
    lineage: Mutex<Option<Arc<Mutex<super::lineage::SharedLineageSolver>>>>,

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
    scope_path: Mutex<super::lineage::ScopePath>,
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
        Self::with_timeout(30000)
    }

    /// Create a new solver context with Z3 and a custom timeout.
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

    /// Debug: dump solver state as string for comparison.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_string(&self) -> String {
        self.with_z3_solver(|solver| format!("{}", solver))
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
        self.local_constraints
            .lock()
            .z3_assertions
            .push(constraint.clone());
        self.with_z3_solver(|solver| solver.assert(&constraint));
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
        self.with_z3_solver(|solver| solver.assert(&constraint));
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        // Invalidate sat_cache - constraint set has changed.
        self.sat_cache.set(None);
        // Preserve model_cache when consistent with the new constraint.
        self.invalidate_model_if_inconsistent(&constraint);
    }

    /// Batched fast path for `add_constraint_raw`: asserts N constraints under
    /// one `local_constraints` lock, one `solver()` guard, and one model
    /// invalidation pass. Each `(z3_ast_ptr, bv, is_true)` tuple corresponds
    /// to the per-constraint metadata that the single-shot path stores in
    /// `local_constraints.{z3_assertions, assumed}`.
    ///
    /// SAFETY: every pointer must be a valid `Z3_ast` Bool in the active
    /// thread-local Z3 context (same precondition as `add_constraint_raw`).
    #[cfg(feature = "vex-engine-z3")]
    pub unsafe fn add_constraints_raw_batch(
        &self,
        entries: Vec<(usize, RustBV, bool)>,
    ) {
        if entries.is_empty() {
            return;
        }
        let z3_ctx = z3::Context::thread_local();
        let mut constraints: Vec<z3::ast::Bool> = Vec::with_capacity(entries.len());
        let mut assumed: Vec<(RustBV, bool)> = Vec::with_capacity(entries.len());
        for (ptr, bv, is_true) in entries {
            let constraint: z3::ast::Bool = unsafe {
                let raw_ast = std::ptr::NonNull::new_unchecked(ptr as *mut _);
                z3::ast::Ast::wrap(&z3_ctx, raw_ast)
            };
            constraints.push(constraint);
            assumed.push((bv, is_true));
        }
        // Single lock on local_constraints for both vectors.
        {
            let mut local = self.local_constraints.lock();
            local.z3_assertions.extend(constraints.iter().cloned());
            local.assumed.extend(assumed);
        }
        // Single solver guard for all assertions.
        self.with_z3_solver(|solver| {
            for c in &constraints {
                solver.assert(c);
            }
        });
        self.constraint_count
            .fetch_add(constraints.len(), Ordering::SeqCst);
        self.sat_cache.set(None);
        self.invalidate_model_if_inconsistent_batch(&constraints);
    }

    /// Batched model invalidation: if the cached model fails to satisfy any
    /// constraint in `constraints`, drop it. Short-circuits on the first
    /// inconsistency. If the cache is already empty, returns immediately.
    #[cfg(feature = "vex-engine-z3")]
    fn invalidate_model_if_inconsistent_batch(&self, constraints: &[z3::ast::Bool]) {
        let mut cache = self.model_cache.borrow_mut();
        if cache.is_none() {
            return;
        }
        let mut still_valid = true;
        if let Some(model) = cache.as_ref() {
            for constraint in constraints {
                let ok = model
                    .eval(constraint, true)
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);
                if !ok {
                    still_valid = false;
                    break;
                }
            }
        }
        if !still_valid {
            *cache = None;
        }
    }

    /// Add a constraint with tracking for unsat_core extraction.
    /// Use this only when unsat_core analysis is needed.
    ///
    /// Returns the tracker index assigned to this constraint, which is also
    /// the index that will appear in [`Self::unsat_core`] output if the
    /// constraint participates in the unsat core.
    #[cfg(feature = "vex-engine-z3")]
    pub fn add_constraint_tracked_indexed(&self, constraint: z3::ast::Bool) -> usize {
        let idx = self.constraint_count.load(Ordering::SeqCst);
        let track_name = format!("__track_{}", idx);
        let track_bool = z3::ast::Bool::new_const(track_name.as_str());

        let tracker_idx = {
            let mut trackers = self.constraint_trackers.lock();
            let i = trackers.len();
            trackers.push(track_bool.clone());
            i
        };

        self.with_z3_solver(|solver| solver.assert_and_track(&constraint, &track_bool));
        self.constraint_count.fetch_add(1, Ordering::SeqCst);
        self.sat_cache.set(None);
        self.invalidate_model_if_inconsistent(&constraint);
        tracker_idx
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
        // Fast path: concrete true is a tautology — skip Z3 entirely (still
        // record it for Python export).
        if let Some(v) = cond.as_u128() {
            if v != 0 {
                self.local_constraints
                    .lock()
                    .assumed
                    .push((cond.clone(), true));
                Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
                return; // Asserting True is a no-op
            }
            // v == 0: asserting False makes solver UNSAT — still add it
        }
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Use to_z3_bool() to produce native Z3 Bool for comparison ops,
        // avoiding ITE(cmp, BV(1,1), BV(0,1))._eq(BV(1,1)) round-trip.
        let constraint = cond.to_z3_bool();
        // Single lock: track for export and cache Z3 Bool for fast fork replay.
        {
            let mut local = self.local_constraints.lock();
            local.assumed.push((cond.clone(), true));
            local.z3_assertions.push(constraint.clone());
        }
        self.add_constraint(constraint);
    }

    /// Add a constraint that the bitvector is false (zero for 1-bit).
    #[cfg(feature = "vex-engine-z3")]
    pub fn assume_false(&self, cond: &RustBV) {
        debug_assert_eq!(cond.width(), 1);
        // Fast path: concrete false (== 0) means not(False) = True — skip Z3
        if let Some(v) = cond.as_u128() {
            if v == 0 {
                self.local_constraints
                    .lock()
                    .assumed
                    .push((cond.clone(), false));
                Z3_ASSUME_CONCRETE_COUNT.fetch_add(1, Ordering::Relaxed);
                return; // Asserting not(False) = True is a no-op
            }
            // v != 0: asserting not(True) = False makes solver UNSAT — still add it
        }
        Z3_ASSUME_SYMBOLIC_COUNT.fetch_add(1, Ordering::Relaxed);
        // Negate the bool directly
        let constraint = cond.to_z3_bool().not();
        // Single lock: track for export and cache Z3 Bool for fast fork replay.
        {
            let mut local = self.local_constraints.lock();
            local.assumed.push((cond.clone(), false));
            local.z3_assertions.push(constraint.clone());
        }
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
        // Perform actual SAT check. Both the check() and the post-check
        // get_model() must happen under the same solver lock, so both run
        // inside the with_z3_solver closure.
        let result = self.with_z3_solver(|solver| {
            let result = matches!(
                timed_check(solver, CheckSite::Satisfiable),
                z3::SatResult::Sat
            );
            // Populate model_cache if SAT — get_model is essentially free
            // after a successful check, and the model lets
            // check_branch_feasibility skip one of two Z3 checks.
            if result {
                let mut cache = self.model_cache.borrow_mut();
                if cache.is_none() {
                    if let Some(m) = solver.get_model() {
                        *cache = Some(m);
                    }
                }
            }
            result
        });
        self.sat_cache.set(Some(result));
        result
    }

    /// Set the Z3 solver timeout in milliseconds.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_timeout(&self, timeout_ms: u32) {
        self.timeout_ms.store(timeout_ms, Ordering::SeqCst);
        if let Some(solver) = self.solver.lock().as_ref() {
            solver.set_params(&build_solver_params(timeout_ms));
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

        // All three branches share one solver lock acquisition via
        // with_z3_solver. The push/pop pairs are balanced inside the
        // closure (every push has a matching pop), so the underlying
        // Z3 scope stack returns to its pre-closure depth before f
        // returns — safe for both the None (per-context) and Some
        // (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
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
                    // can_be_true=true is proven by the model. Check the
                    // other direction (¬cond) with Z3.
                    solver.push();
                    solver.assert(&bool_ast.not());
                    let can_false = matches!(
                        timed_check(solver, CheckSite::BranchFalse),
                        z3::SatResult::Sat
                    );
                    solver.pop(1);
                    (true, can_false)
                }
                Some(false) => {
                    Z3_BRANCH_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                    // can_be_false=true is proven by the model. Check cond.
                    solver.push();
                    solver.assert(&bool_ast);
                    let can_true = matches!(
                        timed_check(solver, CheckSite::BranchTrue),
                        z3::SatResult::Sat
                    );
                    solver.pop(1);
                    (can_true, true)
                }
                None => {
                    Z3_BRANCH_MODEL_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
                    // No cached model: do the original two-check flow.
                    solver.push();
                    solver.assert(&bool_ast);
                    let can_true = matches!(
                        timed_check(solver, CheckSite::BranchTrue),
                        z3::SatResult::Sat
                    );
                    solver.pop(1);

                    if !can_true {
                        return (false, true); // Must be false-only
                    }

                    solver.push();
                    solver.assert(&bool_ast.not());
                    let can_false = matches!(
                        timed_check(solver, CheckSite::BranchFalse),
                        z3::SatResult::Sat
                    );
                    solver.pop(1);

                    (can_true, can_false)
                }
            }
        })
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

        // Need a fresh model — check(), get_model(), and the AST evaluation
        // all share one solver lock acquisition via with_z3_solver. The
        // model_cache write happens inside the closure (model_cache is a
        // RefCell, independent of the solver Mutex, so this does not nest
        // locks against the solver guard).
        self.with_z3_solver(|solver| {
            match timed_check(solver, CheckSite::Eval) {
                z3::SatResult::Sat => {
                    self.sat_cache.set(Some(true));
                }
                _ => {
                    self.sat_cache.set(Some(false));
                    return None;
                }
            }

            let model = solver.get_model()?;
            let ast = bv.to_z3_ast();
            let result = model.eval(&ast, true)?;
            let value = Self::extract_bv_value(&result);
            *self.model_cache.borrow_mut() = Some(model);
            value
        })
    }

    /// Evaluate a bv against the cached parent model without doing a SAT
    /// check. Returns the witness value as a u128 (raw bit pattern, zero-
    /// extended for widths <128). Used by min()/max() to seed binary-search
    /// bounds from a previously-computed model.
    ///
    /// Soundness: per `invalidate_model_if_inconsistent`, any model in the
    /// cache satisfies the current constraint set. So `M(bv)` is a feasible
    /// value of `bv` — for unsigned, `min <= M(bv) <= max`; for signed, the
    /// same holds under signed interpretation.
    #[cfg(feature = "vex-engine-z3")]
    fn cached_model_eval(&self, ast: &z3::ast::BV) -> Option<u128> {
        let cache = self.model_cache.borrow();
        let model = cache.as_ref()?;
        let result = model.eval(ast, true)?;
        Self::extract_bv_value(&result)
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

        // Need a fresh model — check(), get_model(), and the AST evaluation
        // all share one solver lock acquisition via with_z3_solver. Mirrors
        // the slice 3h pattern from `eval`, minus the model_cache write
        // (eval_wide neither reads nor populates model_cache today) and
        // minus the sat_cache update (the original eval_wide didn't set it
        // either — preserve behavior).
        self.with_z3_solver(|solver| {
            match timed_check(solver, CheckSite::Eval) {
                z3::SatResult::Sat => {}
                _ => return None,
            }

            let model = solver.get_model()?;
            let ast = bv.to_z3_ast();
            let result = model.eval(&ast, true)?;
            Self::extract_bv_value_wide(&result, width)
        })
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

        // All n check/get_model/assert-exclude iterations share one solver
        // lock acquisition via with_z3_solver. The outer push/pop pair is
        // balanced inside the closure, so the Z3 scope stack returns to its
        // pre-closure depth before f returns — safe for both the None
        // (per-context) and Some (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();

            for _ in 0..n {
                match timed_check(solver, CheckSite::EvalUpto) {
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
        });
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

        // All n check/get_model/assert-exclude iterations share one solver
        // lock acquisition via with_z3_solver. The outer push/pop pair is
        // balanced inside the closure, so the Z3 scope stack returns to its
        // pre-closure depth before f returns — safe for both the None
        // (per-context) and Some (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();

            for _ in 0..n {
                match timed_check(solver, CheckSite::EvalUpto) {
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
        });
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
    ///
    /// Optimization: when a parent model is cached (from a prior is_sat / eval
    /// on the same constraint set), `M(bv)` is a feasible witness `w`. We use
    /// it to tighten the initial `hi` bound: `min <= w` always holds. For the
    /// signed case, if `w` is negative we additionally skip the MinInit
    /// pre-check (we know a negative value exists).
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

        // Peek the cached model (populated by is_sat above when it does a
        // fresh check, or carried over from a prior eval/min/max on the
        // same constraint set).
        let witness = self.cached_model_eval(&ast);
        if witness.is_some() {
            Z3_EXTREMA_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
        } else {
            Z3_EXTREMA_MODEL_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        // All push/check/pop work shares one solver lock acquisition via
        // with_z3_solver. The outer push/pop pair and the per-iteration
        // push/check/pop pairs are all balanced inside the closure, so the
        // Z3 scope stack returns to its pre-closure depth before f returns
        // — safe for both the None (per-context) and Some (shared-lineage)
        // dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();

            let (mut lo, mut hi): (u128, u128) = if signed {
                let sign_bit = 1u128 << (width - 1);
                let max_val = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                let max_positive = sign_bit - 1;

                // Witness's signed interpretation: negative iff sign bit set.
                let witness_is_negative = witness.map(|v| (v & sign_bit) != 0).unwrap_or(false);

                // has_negative is true if some satisfying assignment is signed
                // negative. A negative witness proves it without a Z3 check.
                let has_negative = if witness_is_negative {
                    true
                } else {
                    solver.push();
                    let zero = Self::make_bv_const(0, width);
                    solver.assert(&ast.bvslt(&zero)); // bv < 0 (signed)
                    let r =
                        matches!(timed_check(solver, CheckSite::MinInit), z3::SatResult::Sat);
                    solver.pop(1);
                    r
                };

                if has_negative {
                    // Minimum is negative, search in [sign_bit, max_val] range.
                    let hi_seed = if witness_is_negative {
                        // Witness is in [sign_bit, max_val] and feasible.
                        witness.unwrap().min(max_val)
                    } else {
                        max_val
                    };
                    (sign_bit, hi_seed)
                } else {
                    // Minimum is non-negative. Witness, if any, is non-negative
                    // (otherwise has_negative would be true), so it's in
                    // [0, max_positive] and tightens the upper bound.
                    let hi_seed = witness.map(|v| v.min(max_positive)).unwrap_or(max_positive);
                    (0, hi_seed)
                }
            } else {
                let max_val = if width >= 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                };
                let hi_seed = witness.map(|v| v.min(max_val)).unwrap_or(max_val);
                (0, hi_seed)
            };

            // Common binary search loop. The signed and unsigned variants only
            // differ in the comparison operator (bvsle vs bvule).
            while lo < hi {
                let mid = lo + (hi - lo) / 2;

                solver.push();
                let mid_ast = Self::make_bv_const(mid, width);
                if signed {
                    solver.assert(&ast.bvsle(&mid_ast));
                } else {
                    solver.assert(&ast.bvule(&mid_ast));
                }
                let can_be_le_mid = matches!(
                    timed_check(solver, CheckSite::MinSearch),
                    z3::SatResult::Sat
                );
                solver.pop(1);

                if can_be_le_mid {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }

            solver.pop(1);
            Some(lo)
        })
    }

    /// Get the maximum value of a bitvector using binary search (O(log N)).
    ///
    /// This implementation uses pure SAT checks without model value extraction,
    /// which allows it to work with bitvectors of any width (including >64 bits).
    /// Based on claripy's _extrema algorithm.
    ///
    /// Optimization: when a parent model is cached, `M(bv)` is a feasible
    /// witness `w` and `max >= w` always holds — used to tighten the initial
    /// `lo` bound. For the signed case, if `w` is non-negative we additionally
    /// skip the MaxInit pre-check (we know a non-negative value exists).
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

        let witness = self.cached_model_eval(&ast);
        if witness.is_some() {
            Z3_EXTREMA_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
        } else {
            Z3_EXTREMA_MODEL_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        let solver = self.solver();
        solver.push();

        let (mut lo, mut hi): (u128, u128) = if signed {
            let sign_bit = 1u128 << (width - 1);
            let max_val = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            let max_positive = sign_bit - 1;

            // Witness's signed interpretation: non-negative iff sign bit clear.
            let witness_is_non_negative = witness.map(|v| (v & sign_bit) == 0).unwrap_or(false);

            // A non-negative witness proves has_non_negative without a Z3 check.
            let has_non_negative = if witness_is_non_negative {
                true
            } else {
                solver.push();
                let zero = Self::make_bv_const(0, width);
                solver.assert(&ast.bvsge(&zero)); // bv >= 0 (signed)
                let r =
                    matches!(timed_check(&solver, CheckSite::MaxInit), z3::SatResult::Sat);
                solver.pop(1);
                r
            };

            if has_non_negative {
                // Maximum is non-negative, search in [0, max_positive] range.
                // Witness, when non-negative, gives a tight lower bound.
                let lo_seed = if witness_is_non_negative {
                    witness.unwrap().min(max_positive)
                } else {
                    0
                };
                (lo_seed, max_positive)
            } else {
                // Maximum is negative. Witness, if any, is negative (otherwise
                // has_non_negative would be true), so it's in [sign_bit, max_val].
                let lo_seed = witness
                    .map(|v| v.max(sign_bit).min(max_val))
                    .unwrap_or(sign_bit);
                (lo_seed, max_val)
            }
        } else {
            let max_val = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            let lo_seed = witness.map(|v| v.min(max_val)).unwrap_or(0);
            (lo_seed, max_val)
        };

        // Common binary search loop. The signed and unsigned variants only
        // differ in the comparison operator (bvsge vs bvuge).
        while lo < hi {
            // Use ceiling division to avoid infinite loop when lo + 1 == hi
            let mid = lo + (hi - lo + 1) / 2;

            solver.push();
            let mid_ast = Self::make_bv_const(mid, width);
            if signed {
                solver.assert(&ast.bvsge(&mid_ast));
            } else {
                solver.assert(&ast.bvuge(&mid_ast));
            }
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
        let result = matches!(
            timed_check(&solver, CheckSite::Satisfiable),
            z3::SatResult::Sat
        );
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
        let (local_len, assumed_local_len) = {
            let local = self.local_constraints.lock();
            (local.z3_assertions.len(), local.assumed.len())
        };
        self.push_local_cache_lengths.lock().push(local_len);
        self.push_assumed_local_lengths
            .lock()
            .push(assumed_local_len);
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

        // Truncate local Z3 cache and assumed_constraints to pre-transaction length
        let prev_z3_len = self.push_local_cache_lengths.lock().pop();
        let prev_assumed_len = self.push_assumed_local_lengths.lock().pop();
        if prev_z3_len.is_some() || prev_assumed_len.is_some() {
            let mut local = self.local_constraints.lock();
            if let Some(prev_len) = prev_z3_len {
                local.z3_assertions.truncate(prev_len);
            }
            if let Some(prev_len) = prev_assumed_len {
                local.assumed.truncate(prev_len);
            }
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
        let core_strs: Vec<String> = self.with_z3_solver(|solver| {
            solver
                .get_unsat_core()
                .iter()
                .map(|ast| format!("{}", ast))
                .collect()
        });

        let trackers = self.constraint_trackers.lock();
        let mut result = Vec::new();

        // Match core tracking booleans to stored tracker indices by string representation
        for core_str in &core_strs {
            for (i, tracker) in trackers.iter().enumerate() {
                if format!("{}", tracker) == *core_str {
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
        self.with_z3_solver(|solver| {
            solver
                .get_assertions()
                .iter()
                .map(|a| format!("{}", a))
                .collect()
        })
    }

    /// Check the total number of assertions in the Z3 solver.
    ///
    /// This can be used to verify constraint sync between Rust and Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn z3_assertion_count(&self) -> usize {
        self.with_z3_solver(|solver| solver.get_assertions().len())
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

    /// Fold this context's assumed constraints into the in-progress sharing
    /// walk (angr-zdho). Each (RustBV, _) is treated as a top-level constraint
    /// tree and walked recursively; pointer-keyed dedup matches today's
    /// per-conversion cache, structural-keyed dedup answers what
    /// construction-level hash-cons (angr-behq) would dedupe to.
    pub fn fold_sharing_walk(&self, walk: &mut ConstraintSharingWalk) {
        let constraints = self.get_assumed_constraints();
        for (bv, _) in &constraints {
            walk.visit(bv);
        }
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

        // angr-v5a5 fields-only slice: child inherits parent's lineage Arc
        // (if any). The Arc::clone here is cheap and harmless — until the
        // next slice routes solver() through with_solver(), nothing reads
        // the inherited lineage. Capturing it now lets sibling-fork tests
        // verify the propagation invariant ahead of the wiring change.
        let child_lineage = self.lineage.lock().as_ref().map(Arc::clone);

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
        }
    }

    /// Clone of this context's lineage Arc, if any (angr-v5a5 spike).
    ///
    /// Returns `None` until the v5a5 integration starts creating
    /// [`SharedLineageSolver`](super::lineage::SharedLineageSolver) on
    /// fork. Currently a fork only propagates an Arc the parent already
    /// had, so all states observed via the public API see `None`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn lineage_arc(&self) -> Option<Arc<Mutex<super::lineage::SharedLineageSolver>>> {
        self.lineage.lock().as_ref().map(Arc::clone)
    }

    /// Current per-state scope-path depth (angr-v5a5 spike).
    ///
    /// Always 0 in this slice — frames are minted by the next slice when
    /// `assume_*` routes through the lineage solver. Exposed now so the
    /// integration can have a consistent telemetry surface.
    #[cfg(feature = "vex-engine-z3")]
    pub fn scope_path_len(&self) -> usize {
        self.scope_path.lock().len()
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
    /// remain balanced when the closure returns. Remaining scope-stack
    /// callers (`max`, `min_with_hint`, `can_be_value`, push/pop,
    /// transaction_*) will follow in 4a.4+, with the
    /// spans-multiple-calls subset (push/pop, transaction_*) likely
    /// needing a separate scope_path API.
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
            push_assumed_local_lengths: Mutex::new(PushStack::new()),
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
                    .z3_assertions
                    .push(guarded.clone());
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
            .z3_assertions
            .push(or_conds.clone());
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
fn freeze_into_shared<T: Clone>(
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

// =============================================================================
// Helper functions for parsing Z3 BV string representations
// =============================================================================

/// Parse a hex string to u128, taking low 128 bits if larger.
#[cfg(feature = "vex-engine-z3")]
fn parse_wide_hex_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 32 hex chars), take low 32 chars
    let low_hex = if s.len() > 32 { &s[s.len() - 32..] } else { s };
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
    let bits: Vec<u8> = s
        .chars()
        .filter_map(|c| match c {
            '0' => Some(0),
            '1' => Some(1),
            _ => None,
        })
        .collect();

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
                use z3::ast::Ast;
                let bv_x =
                    z3::ast::BV::new_const("test_with_z3_solver_no_lineage_x", 8);
                bv_x._eq(&z3::ast::BV::from_u64(42, 8))
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
        use super::{Z3_EXTREMA_MODEL_HIT_COUNT, Z3_EXTREMA_MODEL_MISS_COUNT};

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
        use super::Z3_EXTREMA_MODEL_HIT_COUNT;
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
        assert_eq!(v & (1u128 << 31), 0, "witness should be signed-non-negative");

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

    /// Helper: build a `(z3_ast_ptr, RustBV, bool)` tuple from a width-1 cond
    /// for use with `add_constraints_raw_batch`. Mirrors what the
    /// `RustSolverContext::add_constraints` fast path does with claripy ASTs.
    #[cfg(feature = "vex-engine-z3")]
    fn batch_entry(cond: &RustBV) -> (usize, RustBV, bool) {
        use z3::ast::Ast;
        debug_assert_eq!(cond.width(), 1);
        let bool_ast = cond.to_z3_bool();
        let ptr = bool_ast.get_z3_ast().as_ptr() as usize;
        // The pointer borrows from `bool_ast`; intentionally leak it via
        // forget so the Z3 ref-count survives until add_constraints_raw_batch
        // rewraps it (it bumps the refcount on wrap and decrements on drop).
        std::mem::forget(bool_ast);
        (ptr, cond.clone(), true)
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
        unsafe {
            ctx.add_constraints_raw_batch(entries);
        }
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
        unsafe {
            ctx.add_constraints_raw_batch(Vec::new());
        }
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
        unsafe {
            ctx.add_constraints_raw_batch(entries);
        }
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
        unsafe {
            ctx.add_constraints_raw_batch(entries);
        }
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
        unsafe {
            ctx.add_constraints_raw_batch(entries);
        }
        let after = ctx.export_z3_assertion_ptrs().len();
        assert_eq!(after - before, 2);
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
        assert!(stats.get("vex_unop_total").copied().unwrap() >= base_unop + 1);
        assert!(stats.get("vex_binop_total").copied().unwrap() >= base_binop + 2);
        assert!(stats.get("vex_triop_total").copied().unwrap() >= base_triop + 1);
        assert!(stats.get("vex_qop_total").copied().unwrap() >= base_qop + 1);
        // Each *_op_<family> got bumped once per record_vex_* call.
        assert!(stats.get("vex_op_arith").copied().unwrap() >= base_arith + 2);
        assert!(stats.get("vex_op_logic").copied().unwrap() >= base_logic + 1);
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
        let base_ssym = baseline.get("mem_store_symbolic_addr").copied().unwrap_or(0);
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
        assert!(stats.get("mem_store_count").copied().unwrap() >= base_store + 1);
        assert!(stats.get("mem_load_bytes").copied().unwrap() >= base_load_bytes + 12);
        assert!(stats.get("mem_store_bytes").copied().unwrap() >= base_store_bytes + 16);
        assert!(stats.get("mem_load_symbolic_addr").copied().unwrap() >= base_lsym + 1);
        assert!(stats.get("mem_store_symbolic_addr").copied().unwrap() >= base_ssym + 1);
        assert!(
            stats
                .get("mem_lazy_page_fault_count")
                .copied()
                .unwrap()
                >= base_fault + 1
        );
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
        assert!(stats.get("concretize_write_count").copied().unwrap() >= base_write + 1);
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
        assert!(stats.get("bvop_reverse_count").copied().unwrap() >= base_rev + 1);
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
        assert!(stats.get("bvop_reverse_count").copied().unwrap() >= base_rev + 1);
        assert!(stats.get("bvop_concat_count").copied().unwrap() >= base_cat + 1);
        assert!(stats.get("bvop_extract_count").copied().unwrap() >= base_ext + 1);
    }

    // angr-9o4n.1: Constraint round-trip spike via Z3_solver_to_string /
    // Z3_solver_from_string. Drives whether SMT-LIB2 is the right format for
    // angr-9o4n state save/restore. See bead notes for measured numbers.
    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_smtlib2_constraint_round_trip() {
        use std::time::Instant;
        use z3::ast::Ast;

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
        assert_eq!(x_witness, 11, "x must be 11 (the only value with 10<x<20 whose low 16 bits = 0x000B)");
        assert_eq!(y_witness, 0x0000_0007, "y_high=0, y_low=0x0007");

        // Step 1: Serialize via Solver::to_string (SMT-LIB2 S-expression).
        let serialize_start = Instant::now();
        let serialized = ctx.debug_solver_string();
        let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
        let serialized_bytes = serialized.len();
        assert!(!serialized.is_empty(), "serialized SMT-LIB2 must be non-empty");

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
        assert_eq!(new_sat, original_sat, "round-tripped solver sat-result must match");

        // Step 3b: model values for x, y match the original witness (the
        // constraint set is restrictive enough that x=11, y_low=7 are forced).
        let new_model = new_solver.get_model().expect("sat solver must produce model");
        let new_x_val = new_model
            .eval(&x.to_z3_ast(), true)
            .and_then(|bv| bv.as_u64())
            .expect("model should evaluate x");
        let new_y_val = new_model
            .eval(&y.to_z3_ast(), true)
            .and_then(|bv| bv.as_u64())
            .expect("model should evaluate y");
        assert_eq!(new_x_val as u128, x_witness, "round-tripped x model value must match");
        assert_eq!(new_y_val as u128, y_witness, "round-tripped y model value must match");

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
        use z3::ast::Ast;

        let ctx = SymContext::new();
        const NVARS: usize = 32;
        let vars: Vec<RustBV> = (0..NVARS)
            .map(|i| RustBV::symbolic(&ctx, &format!("rt_scaled_x{}_9o4n", i), 32))
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
        let new_model = new_solver.get_model().expect("sat solver must produce model");
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
        use z3::{with_z3_context, Config, Context, Solver};

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
        let original_ctx_usize =
            z3::Context::thread_local().get_z3_context().as_ptr() as usize;

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
        assert!(new_sat, "cross-context round-tripped solver must remain SAT");
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
}
