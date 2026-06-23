//! Global Z3 solver / engine profiling counters.
//!
//! Extracted from `context.rs` (angr-ugc2, first slice of the angr-a2br.2
//! split). This block is self-contained: every counter is a module-level
//! atomic with no `SymContext` or `self.` reference. The aggregation
//! (`get_solver_stats`), reset (`reset_solver_stats`), and per-event
//! `record_*` helpers live here; the solving code in `context.rs`
//! (`timed_check`, `sample_simplify_skip`, the `add_constraint` /
//! `assume_*` / extrema paths) reads and bumps these via `use super::stats::*`.
//!
//! Zero-cost when not read: an atomic `fetch_add` is ~1ns on x86.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// =============================================================================
// Global Z3 Solver Profiling Counters
// =============================================================================
// Zero-cost when not read: atomic fetch_add is ~1ns on x86.

/// Total number of Z3 solver.check() calls.
pub(crate) static Z3_CHECK_COUNT: AtomicU64 = AtomicU64::new(0);
/// Total time (nanoseconds) spent in Z3 solver.check() calls.
pub(crate) static Z3_CHECK_TIME_NS: AtomicU64 = AtomicU64::new(0);
/// Number of solver materializations (lazy fork → first access).
pub(crate) static Z3_MATERIALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Total time (nanoseconds) spent materializing solvers.
pub(crate) static Z3_MATERIALIZE_TIME_NS: AtomicU64 = AtomicU64::new(0);
/// Number of assume_true/assume_false calls that hit the concrete fast path.
pub(crate) static Z3_ASSUME_CONCRETE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of assume_true/assume_false calls that went to Z3.
pub(crate) static Z3_ASSUME_SYMBOLIC_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of check_branch_feasibility calls.
pub(crate) static Z3_BRANCH_CHECK_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of branch checks where condition was concrete.
pub(crate) static Z3_BRANCH_CONCRETE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of branch checks where the cached parent model predicted one direction.
pub(crate) static Z3_BRANCH_MODEL_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of branch checks where no usable cached model was available.
pub(crate) static Z3_BRANCH_MODEL_MISS_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of min()/max() calls where the cached model produced a usable
/// witness used to tighten the binary-search initial bound (and, in the
/// signed case, sometimes skipped the MinInit/MaxInit pre-check).
pub(crate) static Z3_EXTREMA_MODEL_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of min()/max() calls where no cached model was available (or the
/// model could not be evaluated against the bv ast).
pub(crate) static Z3_EXTREMA_MODEL_MISS_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of to_z3_ast() / to_z3_bool() calls (AST construction).
pub(crate) static Z3_AST_BUILD_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of cache hits in `to_z3_ast_cached` per-call HashMap (angr-zdho).
///
/// Each hit means a sub-expression was visited more than once during a single
/// top-level `to_z3_ast()` call and the Arc-pointer key already had a Z3 AST
/// built — the inner DAG had at least one shared Arc subtree. Ratio of hits to
/// (hits+misses) gives the per-conversion sharing rate the cache captures.
pub(crate) static Z3_AST_CACHE_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of cache misses in `to_z3_ast_cached` per-call HashMap (angr-zdho).
///
/// A miss is a unique RustBV pointer visited within one `to_z3_ast()` call.
/// Equals the count of distinct Arc-pointer subtrees materialized into Z3
/// ASTs for that conversion.
pub(crate) static Z3_AST_CACHE_MISS_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of persistent per-`Expression` memo hits in `to_z3_ast_cached`
/// (angr-ovqja.3).
///
/// A hit means the same `RustBV::Expression` value was converted to a Z3 AST
/// in a *prior* top-level `to_z3_ast()` call (eval→min→max, `range()`, address
/// concretize) and the whole compound tree rebuild was avoided — the cached
/// `z3::ast::BV` was returned via a refcount bump. Distinct from
/// `z3_ast_cache_hit`, which only dedups shared subtrees *within* one call.
pub(crate) static Z3_AST_MEMO_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Sat.
pub(crate) static Z3_SAT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Unsat.
pub(crate) static Z3_UNSAT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Number of Z3 solver.check() calls that returned Unknown (timeout / resource limit).
pub(crate) static Z3_TIMEOUT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Deepest ITE chain ever stored as a symbolic memory cell value.
///
/// Recorded by the eager `store_strided` path and by Multi-cell installation
/// (Phase 1/2, `install_multi_for_candidates`) so before/after comparison
/// against the eager baseline is direct.
pub(crate) static MEM_ITE_DEPTH_MAX: AtomicU32 = AtomicU32::new(0);
/// Cumulative count of conditional iterations added to ITE chains in symbolic
/// stores. Sum, not max — surfaces total ITE-chain work across a run.
pub(crate) static MEM_ITE_DEPTH_TOTAL: AtomicU64 = AtomicU64::new(0);

// -----------------------------------------------------------------------------
// Dispatch / volume counters (angr-2j5v)
// -----------------------------------------------------------------------------
// Lightweight breadth-first instrumentation across the rest of the pipeline.
// Each counter is a single AtomicU64 fetch_add at the construction site —
// matches the existing Z3_* pattern. Read out via `get_solver_stats()`.

// VEX op dispatch — count per family at the entry of unop/binop/triop/qop.
pub(crate) static VEX_UNOP_TOTAL: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_BINOP_TOTAL: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_TRIOP_TOTAL: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_QOP_TOTAL: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_ARITH: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_LOGIC: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_SHIFT: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_CMP: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_EXT: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_FP: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_VEC: AtomicU64 = AtomicU64::new(0);
pub(crate) static VEX_OP_OTHER: AtomicU64 = AtomicU64::new(0);

// Memory volume — load/store call counts, total bytes, concrete-vs-symbolic
// address split. Recorded at top-level `SymbolicMemory::{load,store}` only,
// not at every internal helper, so the counter reflects the public surface.
pub(crate) static MEM_LOAD_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static MEM_STORE_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static MEM_LOAD_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static MEM_STORE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static MEM_LOAD_SYMBOLIC_ADDR: AtomicU64 = AtomicU64::new(0);
pub(crate) static MEM_STORE_SYMBOLIC_ADDR: AtomicU64 = AtomicU64::new(0);
/// `UnmappedPageInRegion` faults — lazy region had no backing page, caller
/// must materialize. Bumped in load/store before the error is propagated.
pub(crate) static MEM_LAZY_PAGE_FAULT_COUNT: AtomicU64 = AtomicU64::new(0);

// Concretization fanout — count + cumulative K + max K observed per call.
pub(crate) static CONCRETIZE_READ_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static CONCRETIZE_WRITE_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static CONCRETIZE_TOTAL_CANDIDATES: AtomicU64 = AtomicU64::new(0);
pub(crate) static CONCRETIZE_MAX_CANDIDATES: AtomicU32 = AtomicU32::new(0);

// angr-62li: address-concretization disjunction hoisting. When the
// concretizer returns Multiple, the engine now asserts the disjunction
// `Or(addr == a0, ..., addr == aK)` as a top-level constraint so Z3's
// propagate-values tactic can substitute the addr var. Bumped at the
// hoist site in `memory/store.rs::assert_address_disjunction`.
pub(crate) static CONCRETIZE_DISJUNCTION_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static CONCRETIZE_DISJUNCTION_TERMS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub(crate) static CONCRETIZE_DISJUNCTION_MAX_TERMS: AtomicU32 = AtomicU32::new(0);

// AST construction (specific) — Reverse/Concat/Extract emissions. These
// are the AST shapes most often blamed when claripy<->Rust mismatch surfaces
// (angr-tlvl-style residuals). Counted at the public `RustBV::{reverse,
// concat, extract}` entry points; internal recursion within `*_into` is
// not double-counted.
pub(crate) static BVOP_REVERSE_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static BVOP_CONCAT_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static BVOP_EXTRACT_COUNT: AtomicU64 = AtomicU64::new(0);

// angr-g7nq: trivial-constraint fast path counters. Bumped from
// `value.rs::{eq_into,ne_into,ult_into,ule_into,ugt_into,uge_into}` when
// the `Cmp(ZeroExt(k, x), BVV)` (or commuted form) pattern is recognized
// and either collapsed to a smaller AST (the "high bits zero" case) or
// short-circuited to a concrete answer (the "high bits nonzero" case, an
// unsatisfiable/tautological subexpression). Separate counts for the two
// outcomes make it possible to attribute hits without inspecting the
// emitted ASTs.
pub(crate) static ZEXT_CMP_COLLAPSE_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static ZEXT_CMP_TRIVIAL_DECIDE_COUNT: AtomicU64 = AtomicU64::new(0);

// angr-acoq: claripy-export soundness counters for symbolic Clz/Ctz/Popcount
// and Float results. Bumped from `claripy_bridge::rustbv_to_claripy_memo`.
//   - SOUND_CLZ: width<=64 clz/ctz/popcount exported as a sound ITE/sum
//     encoding tied to the operand (no constraint relationship lost).
//   - UNCONSTRAINED_CLZ: width>64 clz/ctz/popcount fell back to a fresh BVS
//     (sound encoding declined for width).
//   - UNCONSTRAINED_FP: Float result exported as a fresh BVS (Z3 FP term kept
//     in-engine; the claripy-side value is unconstrained by design).
// The unconstrained counters surface "Python eval may see a Rust-infeasible
// value" risk in --dump-counters.
pub(crate) static EXPORT_SOUND_CLZ_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static EXPORT_UNCONSTRAINED_CLZ_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static EXPORT_UNCONSTRAINED_FP_COUNT: AtomicU64 = AtomicU64::new(0);

// angr-1joc: constraint-dedup measurement counters. Three signals to decide
// whether canonicalization/dedup at add_constraint_raw is worth implementing
// (threshold: any >=10% wins a follow-up bead). See bd memory
// invariant-z3-construction-canonicalization for the underlying Z3 contract.
//
// (a) commutative-arg-order: how often the RustBV-level
// `canonicalize_commutative` actually swaps operands. Swap rate ≈ how often
// the un-canonicalized RustBV order WOULD have produced a distinct Z3 AST.
// Bumped from `value.rs::canonicalize_commutative`.
pub(crate) static RUSTBV_COMMUTATIVE_CANONICALIZE_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static RUSTBV_COMMUTATIVE_SWAP_COUNT: AtomicU64 = AtomicU64::new(0);
// (b) branch-condition simplify-skip: sampled across assume_true /
// assume_false / add_constraint_raw. `BRANCH_COND_SIMPLIFY_REDUCED_COUNT`
// increments only when Z3 simplify() returns a STRUCTURALLY-DIFFERENT AST
// (different Z3_ast pointer). Population total = Z3_ASSUME_SYMBOLIC_COUNT +
// ADD_CONSTRAINT_RAW_TOTAL_COUNT — both pre-existing/added below.
pub(crate) static ADD_CONSTRAINT_RAW_TOTAL_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static BRANCH_COND_SIMPLIFY_SAMPLED_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static BRANCH_COND_SIMPLIFY_REDUCED_COUNT: AtomicU64 = AtomicU64::new(0);
/// Every Nth assertion runs Z3 simplify() for measurement.  Driven by a
/// shared atomic counter across all three assert sites.  N=64 gives ~1.5%
/// overhead with a single simplify call per sample. Override via
/// `ANGR_Z3_SIMPLIFY_STRIDE` (e.g. `=1` for full-population sampling
/// during a profiling spike); see `simplify_sample_stride()` below.
#[cfg(feature = "vex-engine-z3")]
pub(crate) const SIMPLIFY_SAMPLE_STRIDE_DEFAULT: u64 = 64;
pub(crate) static SIMPLIFY_SAMPLE_TICKER: AtomicU64 = AtomicU64::new(0);
// (c) full-list assertion-dedup: how often the incoming Z3_ast ptr ALREADY
// appears in shared+local `z3_assertions` (angr-sfp9). Always-on via the
// `dedup_set` HashSet side-table inside `LocalConstraints` — O(1) ptr lookup
// per call after a one-time lazy seed from shared+local.
// SCANNED bumps once per `add_constraint_raw` call. HIT bumps when the ptr
// was already present, in which case the push+assert are skipped.
pub(crate) static ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static ADD_CONSTRAINT_RAW_DEDUP_SCANNED_COUNT: AtomicU64 = AtomicU64::new(0);
// angr-mwbp: dedup hit/scanned for the assume_true/assume_false hot path,
// surfaced separately from the add_constraint_raw counters so the impact of
// extending dedup to the symbolic-branch sites is attributable. Same Z3 ptr
// keying as `dedup_set`; the `assumed` vec is still grown on every call
// (preserves Python-visible duplicate constraints) — the dedup only skips
// the redundant push into `z3_assertions` + `add_constraint` round-trip.
pub(crate) static Z3_ASSUME_DEDUP_SCANNED_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static Z3_ASSUME_DEDUP_HIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Per-site counters and timers for solver.check() calls.
/// Indexed by `CheckSite as usize`.
pub(crate) const NUM_CHECK_SITES: usize = 9;
pub(crate) static Z3_CHECK_SITE_COUNT: [AtomicU64; NUM_CHECK_SITES] = [
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
pub(crate) static Z3_CHECK_SITE_TIME_NS: [AtomicU64; NUM_CHECK_SITES] = [
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
pub(crate) const SITE_NAMES: [&str; NUM_CHECK_SITES] = [
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
    stats.insert(
        "z3_ast_memo_hit".into(),
        Z3_AST_MEMO_HIT_COUNT.load(Ordering::Relaxed),
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
    stats.insert(
        "vex_qop_total".into(),
        VEX_QOP_TOTAL.load(Ordering::Relaxed),
    );
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
    // angr-acoq: claripy-export soundness counters.
    stats.insert(
        "rust_export_sound_clz".into(),
        EXPORT_SOUND_CLZ_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "rust_export_unconstrained_clz".into(),
        EXPORT_UNCONSTRAINED_CLZ_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "rust_export_unconstrained_fp".into(),
        EXPORT_UNCONSTRAINED_FP_COUNT.load(Ordering::Relaxed),
    );
    // angr-1joc: constraint-dedup measurement counters.
    stats.insert(
        "rustbv_commutative_canonicalize_count".into(),
        RUSTBV_COMMUTATIVE_CANONICALIZE_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "rustbv_commutative_swap_count".into(),
        RUSTBV_COMMUTATIVE_SWAP_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "add_constraint_raw_total".into(),
        ADD_CONSTRAINT_RAW_TOTAL_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "branch_cond_simplify_sampled".into(),
        BRANCH_COND_SIMPLIFY_SAMPLED_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "branch_cond_simplify_reduced".into(),
        BRANCH_COND_SIMPLIFY_REDUCED_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "add_constraint_raw_dedup_scanned".into(),
        ADD_CONSTRAINT_RAW_DEDUP_SCANNED_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "add_constraint_raw_dedup_hit".into(),
        ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_assume_dedup_scanned".into(),
        Z3_ASSUME_DEDUP_SCANNED_COUNT.load(Ordering::Relaxed),
    );
    stats.insert(
        "z3_assume_dedup_hit".into(),
        Z3_ASSUME_DEDUP_HIT_COUNT.load(Ordering::Relaxed),
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
    // angr-v5ht: runtime thrash detector state alongside the lineage
    // counters. `lineage_dismantled` is the bool kill switch (0/1);
    // `lineage_dismantle_count` is the number of times the sampler has
    // flipped the switch on across resets (typically 0 or 1 per
    // exploration since dismantle is sticky until reset).
    #[cfg(feature = "vex-engine-z3")]
    for (name, value) in super::lineage::dismantle_stats() {
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
    Z3_AST_MEMO_HIT_COUNT.store(0, Ordering::Relaxed);
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
    EXPORT_SOUND_CLZ_COUNT.store(0, Ordering::Relaxed);
    EXPORT_UNCONSTRAINED_CLZ_COUNT.store(0, Ordering::Relaxed);
    EXPORT_UNCONSTRAINED_FP_COUNT.store(0, Ordering::Relaxed);
    // angr-1joc constraint-dedup measurement counters.
    RUSTBV_COMMUTATIVE_CANONICALIZE_COUNT.store(0, Ordering::Relaxed);
    RUSTBV_COMMUTATIVE_SWAP_COUNT.store(0, Ordering::Relaxed);
    ADD_CONSTRAINT_RAW_TOTAL_COUNT.store(0, Ordering::Relaxed);
    BRANCH_COND_SIMPLIFY_SAMPLED_COUNT.store(0, Ordering::Relaxed);
    BRANCH_COND_SIMPLIFY_REDUCED_COUNT.store(0, Ordering::Relaxed);
    SIMPLIFY_SAMPLE_TICKER.store(0, Ordering::Relaxed);
    ADD_CONSTRAINT_RAW_DEDUP_SCANNED_COUNT.store(0, Ordering::Relaxed);
    ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT.store(0, Ordering::Relaxed);
    Z3_ASSUME_DEDUP_SCANNED_COUNT.store(0, Ordering::Relaxed);
    Z3_ASSUME_DEDUP_HIT_COUNT.store(0, Ordering::Relaxed);
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
#[cfg(feature = "vex-engine-z3")]
#[inline]
pub fn record_z3_ast_build() {
    Z3_AST_BUILD_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Increment per-call to_z3_ast_cached cache-hit counter (angr-zdho).
#[cfg(feature = "vex-engine-z3")]
#[inline]
pub fn record_z3_ast_cache_hit() {
    Z3_AST_CACHE_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Increment per-call to_z3_ast_cached cache-miss counter (angr-zdho).
#[cfg(feature = "vex-engine-z3")]
#[inline]
pub fn record_z3_ast_cache_miss() {
    Z3_AST_CACHE_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Increment the persistent per-`Expression` memo-hit counter (angr-ovqja.3).
#[cfg(feature = "vex-engine-z3")]
#[inline]
pub fn record_z3_ast_memo_hit() {
    Z3_AST_MEMO_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
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

/// angr-acoq: record a sound clz/ctz/popcount export (width<=64, encoded as
/// an ITE/sum tied to the operand).
#[inline]
pub fn record_export_sound_clz() {
    EXPORT_SOUND_CLZ_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-acoq: record an unconstrained clz/ctz/popcount export (width>64 fresh
/// BVS fallback — constraint relationship lost).
#[inline]
pub fn record_export_unconstrained_clz() {
    EXPORT_UNCONSTRAINED_CLZ_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-acoq: record an unconstrained Float export (fresh BVS — Z3 FP term
/// kept in-engine, claripy-side value unconstrained).
#[inline]
pub fn record_export_unconstrained_fp() {
    EXPORT_UNCONSTRAINED_FP_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// angr-1joc: record a `canonicalize_commutative` invocation. `swapped` is
/// true when the canonical order required swapping the operand pair.
#[inline]
pub fn record_commutative_canonicalize(swapped: bool) {
    RUSTBV_COMMUTATIVE_CANONICALIZE_COUNT.fetch_add(1, Ordering::Relaxed);
    if swapped {
        RUSTBV_COMMUTATIVE_SWAP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}
