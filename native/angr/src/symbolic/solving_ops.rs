//! Solving / evaluation read-path `&self` methods for [`SymContext`].
//!
//! Slice 7 of the `symbolic/context.rs` split (angr-wf6f). These are the
//! read-only solver query methods: satisfiability (`is_sat`), branch
//! feasibility (`can_be_true` / `can_be_false` / `check_branch_feasibility`),
//! model evaluation (`eval` / `eval_wide` / `eval_upto` / `eval_upto_wide`),
//! extrema (`min` / `max` / `range` / `range_seeded`), solution enumeration
//! (`solutions` / `solution`), and the `debug_solver_string` dump.
//!
//! All Z3 access is routed through the still-in-`context.rs`
//! `with_z3_solver` dispatcher (`pub(crate)`), so the only private fields
//! these touch are the two query caches `sat_cache` and `model_cache` —
//! promoted to `pub(super)` so this sibling module can reach them. The
//! solver/lineage/timeout mutators (`set_timeout` / `timeout_ms` /
//! `set_sat_cache`) stay in `context.rs`: they write `solver`/`timeout_ms`
//! directly and are not part of the read path. See bd memory
//! `a2br2-context-split-impl-block-plan` for the slice plan.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! `pub(super)` (== `pub(in crate::symbolic)`) keeps the promoted caches
//! module-private — no public API leak.

use super::RustBV;
use super::SymContext;

#[cfg(feature = "vex-engine-z3")]
use super::bv_codec::*;
#[cfg(feature = "vex-engine-z3")]
use super::query_class;
#[cfg(feature = "vex-engine-z3")]
use super::solver_build::*;
#[cfg(feature = "vex-engine-z3")]
use super::stats::*;
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::Ordering;

/// Render a concrete `u128` as a big-endian byte vector sized for `width`
/// bits (`ceil(width/8)` bytes). For widths ≤ 128 the low `byte_len` bytes
/// of the value are taken; wider widths left-pad with zeros. Shared by the
/// concrete fast paths of `eval_wide` / `eval_upto_wide`.
fn u128_to_be_bytes_width(value: u128, width: u32) -> Vec<u8> {
    let byte_len = width.div_ceil(8) as usize;
    let bytes = value.to_be_bytes();
    if byte_len <= 16 {
        bytes[16 - byte_len..].to_vec()
    } else {
        let mut result = vec![0u8; byte_len];
        result[byte_len - 16..].copy_from_slice(&bytes);
        result
    }
}

/// Read every bv in `bvs` off a single Z3 model. All-or-nothing: `None` when
/// any part fails to evaluate, so a caller never sees a half-model. Shared by
/// `eval_many`'s cached-model and fresh-model paths (angr-ue4ro).
#[cfg(feature = "vex-engine-z3")]
fn eval_all_in_model(model: &z3::Model, bvs: &[RustBV]) -> Option<Vec<u128>> {
    let mut values = Vec::with_capacity(bvs.len());
    for bv in bvs {
        if let Some(v) = bv.as_u128() {
            values.push(v);
            continue;
        }
        let result = model.eval(&bv.to_z3_ast(), true)?;
        values.push(extract_bv_value(&result)?);
    }
    Some(values)
}

/// Binary search for the minimum feasible value of `ast` in `[lo, hi]`.
///
/// Floor-mid bisection: assert `cmp(ast, mid)` (i.e. `ast <= mid`) and keep the
/// lower half when satisfiable. `cmp` selects the signed (`bvsle`) vs unsigned
/// (`bvule`) ordering. Shared by `min` and the min half of `range_seeded`; the
/// caller owns the surrounding `with_z3_solver` push/pop frame.
#[cfg(feature = "vex-engine-z3")]
fn bsearch_min(
    solver: &z3::Solver,
    ast: &z3::ast::BV,
    width: u32,
    mut lo: u128,
    mut hi: u128,
    cmp: impl Fn(&z3::ast::BV, &z3::ast::BV) -> z3::ast::Bool,
) -> u128 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        solver.push();
        let mid_ast = make_bv_const(mid, width);
        solver.assert(cmp(ast, &mid_ast));
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
    lo
}

/// Binary search for the maximum feasible value of `ast` in `[lo, hi]`.
///
/// Ceil-mid bisection (avoids an infinite loop when `lo + 1 == hi`): assert
/// `cmp(ast, mid)` (i.e. `ast >= mid`) and keep the upper half when
/// satisfiable. `cmp` selects the signed (`bvsge`) vs unsigned (`bvuge`)
/// ordering. Shared by `max` and the max half of `range_seeded`; the caller
/// owns the surrounding `with_z3_solver` push/pop frame.
#[cfg(feature = "vex-engine-z3")]
fn bsearch_max(
    solver: &z3::Solver,
    ast: &z3::ast::BV,
    width: u32,
    mut lo: u128,
    mut hi: u128,
    cmp: impl Fn(&z3::ast::BV, &z3::ast::BV) -> z3::ast::Bool,
) -> u128 {
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        solver.push();
        let mid_ast = make_bv_const(mid, width);
        solver.assert(cmp(ast, &mid_ast));
        let can_be_ge_mid = matches!(
            timed_check(solver, CheckSite::MaxSearch),
            z3::SatResult::Sat
        );
        solver.pop(1);
        if can_be_ge_mid {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

impl SymContext {
    /// Debug: dump solver state as string for comparison.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_string(&self) -> String {
        self.with_z3_solver(|solver| format!("{solver}"))
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
        let _class =
            query_class::scope(|| query_class::classify_sat(&self.get_assumed_constraints()));
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
                if cache.is_none()
                    && let Some(m) = solver.get_model()
                {
                    *cache = Some(m);
                }
            }
            result
        });
        self.sat_cache.set(Some(result));
        result
    }

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
        let _class = query_class::scope(|| {
            query_class::classify_bool(cond, &self.get_assumed_constraints())
        });
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
                    solver.assert(bool_ast.not());
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
                    solver.assert(bool_ast.not());
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
                    return extract_bv_value(&result);
                }
            }
        }

        // Need a fresh model — check(), get_model(), and the AST evaluation
        // all share one solver lock acquisition via with_z3_solver. The
        // model_cache write happens inside the closure (model_cache is a
        // RefCell, independent of the solver Mutex, so this does not nest
        // locks against the solver guard).
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));
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
            let value = extract_bv_value(&result);
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
        extract_bv_value(&result)
    }

    /// Evaluate a bitvector to bytes (for values > 128 bits).
    ///
    /// Reads and populates `model_cache` exactly like `eval` (angr-ue4ro): a
    /// wide value must come out of ONE model, and a caller that evals the same
    /// 256-bit stdin BVS twice — or evals it once and then `posix.dumps(0)`s
    /// the same bytes — must get the same answer both times. The old
    /// fresh-check-every-call version handed back a different (still
    /// satisfying) model each time, which reads as nondeterminism to a user.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_wide(&self, bv: &RustBV) -> Option<Vec<u8>> {
        let width = bv.width();

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(u128_to_be_bytes_width(v, width));
        }

        {
            let cache = self.model_cache.borrow();
            if let Some(ref model) = *cache {
                let ast = bv.to_z3_ast();
                if let Some(result) = model.eval(&ast, true) {
                    return extract_bv_value_wide(&result, width);
                }
            }
        }

        // Need a fresh model — check(), get_model(), and the AST evaluation
        // all share one solver lock acquisition via with_z3_solver. Mirrors
        // the slice 3h pattern from `eval`, minus the sat_cache update (the
        // original eval_wide didn't set it — preserve behavior).
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));
        self.with_z3_solver(|solver| {
            match timed_check(solver, CheckSite::Eval) {
                z3::SatResult::Sat => {}
                _ => return None,
            }

            let model = solver.get_model()?;
            let ast = bv.to_z3_ast();
            let result = model.eval(&ast, true)?;
            let value = extract_bv_value_wide(&result, width);
            *self.model_cache.borrow_mut() = Some(model);
            value
        })
    }

    /// Evaluate several bitvectors against ONE model (angr-ue4ro).
    ///
    /// The wide-eval fallback in `RustSolverFallback._rust_eval` decomposes an
    /// expression Rust cannot convert whole into byte-sized `Extract`s. Solving
    /// each byte independently mixes models: every byte satisfies its own
    /// byte-local constraints, but a cross-byte constraint (a checksum, or the
    /// `(acc & 0xff) == 0xee` find gate on the pbounce synthetic) can be
    /// violated by the concatenation. Route that decomposition through here so
    /// all parts are read off a single satisfying assignment.
    ///
    /// Returns `None` when the constraints are unsat or any part fails to
    /// evaluate — never a partially-filled vector.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_many(&self, bvs: &[RustBV]) -> Option<Vec<u128>> {
        if bvs.is_empty() {
            return Some(Vec::new());
        }

        {
            let cache = self.model_cache.borrow();
            if let Some(ref model) = *cache
                && let Some(values) = eval_all_in_model(model, bvs)
            {
                return Some(values);
            }
        }

        let _class = query_class::scope(|| {
            query_class::classify_eval_many(bvs, &self.get_assumed_constraints())
        });
        self.with_z3_solver(|solver| {
            match timed_check(solver, CheckSite::Eval) {
                z3::SatResult::Sat => self.sat_cache.set(Some(true)),
                _ => {
                    self.sat_cache.set(Some(false));
                    return None;
                }
            }

            let model = solver.get_model()?;
            let values = eval_all_in_model(&model, bvs);
            *self.model_cache.borrow_mut() = Some(model);
            values
        })
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
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));

        // Seed iteration 0 from a warm cached model when present (angr-ovqja.4).
        // Soundness: per `invalidate_model_if_inconsistent`, a surviving cached
        // model satisfies all permanent constraints, and iteration-0 runs under
        // a fresh empty push scope (no exclude constraints yet), so M(ast) with
        // completion is a genuine feasible solution. Seeding it lets us skip
        // exactly one check+get_model. The exclude-loop below then enumerates
        // the remaining distinct values exactly as before. eval_upto is treated
        // as unordered by callers (e.g. `solutions()`), and the result count is
        // unchanged (min(n, #feasible)), so the solution set is faithful.
        let seeded = self.cached_model_eval(&ast);

        // The remaining check/get_model/assert-exclude iterations share one
        // solver lock acquisition via with_z3_solver. The outer push/pop pair
        // is balanced inside the closure, so the Z3 scope stack returns to its
        // pre-closure depth before f returns — safe for both the None
        // (per-context) and Some (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();

            let mut remaining = n;
            if let Some(value) = seeded {
                Z3_EVAL_UPTO_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                results.push(value);
                let val_ast = make_bv_const(value, bv.width());
                solver.assert(ast.eq(&val_ast).not());
                remaining -= 1;
            }

            for _ in 0..remaining {
                match timed_check(solver, CheckSite::EvalUpto) {
                    z3::SatResult::Sat => {
                        if let Some(model) = solver.get_model() {
                            if let Some(result) = model.eval(&ast, true) {
                                if let Some(value) = extract_bv_value(&result) {
                                    results.push(value);
                                    // Add constraint to exclude this value
                                    let val_ast = make_bv_const(value, bv.width());
                                    solver.assert(ast.eq(&val_ast).not());
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
        // Canonical presentation order (angr-op0dn.10.1). Pure post-processing
        // on the enumerated set: the exclude-loop above decides WHICH values are
        // returned, this only decides in what order. Makes the exhaustive case
        // (n >= #feasible, e.g. `solutions()`) bit-for-bit reproducible across
        // runs regardless of which witness Z3 or the warm-model seed found
        // first. Any future short-circuit must land ABOVE this sort.
        results.sort_unstable();
        results
    }

    /// Evaluate a bitvector and return up to n solutions as byte arrays (big-endian).
    /// Handles values of any width without truncation.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto_wide(&self, bv: &RustBV, n: usize) -> Vec<Vec<u8>> {
        let width = bv.width();

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return vec![u128_to_be_bytes_width(v, width)];
        }

        if n == 0 {
            return vec![];
        }

        let mut results = Vec::with_capacity(n);
        let ast = bv.to_z3_ast();
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));

        // Seed iteration 0 from a warm cached model when present (angr-ovqja.4).
        // Same soundness argument as eval_upto: the cached model satisfies all
        // permanent constraints, and iteration-0 runs under a fresh empty push
        // scope, so the wide witness is a genuine feasible solution. Saves one
        // check+get_model; the exclude-loop enumerates the rest unchanged.
        let seeded: Option<Vec<u8>> = {
            let cache = self.model_cache.borrow();
            cache
                .as_ref()
                .and_then(|m| m.eval(&ast, true))
                .and_then(|r| extract_bv_value_wide(&r, width))
        };

        // The remaining check/get_model/assert-exclude iterations share one
        // solver lock acquisition via with_z3_solver. The outer push/pop pair
        // is balanced inside the closure, so the Z3 scope stack returns to its
        // pre-closure depth before f returns — safe for both the None
        // (per-context) and Some (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();

            let mut remaining = n;
            if let Some(bytes) = seeded {
                Z3_EVAL_UPTO_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                let val_ast = make_bv_from_bytes(&bytes, width);
                solver.assert(ast.eq(&val_ast).not());
                results.push(bytes);
                remaining -= 1;
            }

            for _ in 0..remaining {
                match timed_check(solver, CheckSite::EvalUpto) {
                    z3::SatResult::Sat => {
                        if let Some(model) = solver.get_model() {
                            if let Some(result) = model.eval(&ast, true) {
                                if let Some(bytes) = extract_bv_value_wide(&result, width) {
                                    // Exclude this value from future solutions
                                    // Build Z3 constant from bytes for full-precision exclusion
                                    let val_ast = make_bv_from_bytes(&bytes, width);
                                    solver.assert(ast.eq(&val_ast).not());
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
        // Canonical ascending numeric order (angr-op0dn.10.1). Every witness is
        // `width` bits wide, so all byte vectors have the same length and
        // big-endian lexicographic order coincides with numeric order.
        results.sort_unstable();
        results
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

        // The binary search below tracks bounds in a u128. For widths above 128
        // the true extremum can exceed u128::MAX, and the old `u128::MAX` cap
        // plus the low-128-truncated witness returned a wrong value. Report
        // unknown rather than a truncated extremum (angr-cxw7). 128-bit BVs are
        // fine: their range is exactly [0, u128::MAX].
        if width > 128 {
            return None;
        }

        // Peek the cached model (populated by is_sat above when it does a
        // fresh check, or carried over from a prior eval/min/max on the
        // same constraint set).
        let _class = query_class::scope(|| {
            query_class::classify_extrema(bv, &self.get_assumed_constraints())
        });
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

            let (lo, hi): (u128, u128) = if signed {
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
                    let zero = make_bv_const(0, width);
                    solver.assert(ast.bvslt(&zero)); // bv < 0 (signed)
                    let r = matches!(timed_check(solver, CheckSite::MinInit), z3::SatResult::Sat);
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
            let result = if signed {
                bsearch_min(solver, &ast, width, lo, hi, |a, m| a.bvsle(m))
            } else {
                bsearch_min(solver, &ast, width, lo, hi, |a, m| a.bvule(m))
            };

            solver.pop(1);
            Some(result)
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

        // See min(): widths above 128 cannot be represented in the u128 binary
        // search bounds, so a truncated extremum would be returned. Report
        // unknown instead (angr-cxw7).
        if width > 128 {
            return None;
        }

        let _class = query_class::scope(|| {
            query_class::classify_extrema(bv, &self.get_assumed_constraints())
        });
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

            let (lo, hi): (u128, u128) = if signed {
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
                    let zero = make_bv_const(0, width);
                    solver.assert(ast.bvsge(&zero)); // bv >= 0 (signed)
                    let r = matches!(timed_check(solver, CheckSite::MaxInit), z3::SatResult::Sat);
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
            let result = if signed {
                bsearch_max(solver, &ast, width, lo, hi, |a, m| a.bvsge(m))
            } else {
                bsearch_max(solver, &ast, width, lo, hi, |a, m| a.bvuge(m))
            };

            solver.pop(1);
            Some(result)
        })
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
        let _class = query_class::scope(|| {
            query_class::classify_extrema(bv, &self.get_assumed_constraints())
        });
        let max_val: u128 = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };

        // Clamp seeds to the bitvector range. Seeds must already be valid
        // solutions, so smallest_known ≤ true_max and largest_known ≥ true_min.
        let hi_seed = smallest_known.min(max_val);
        let lo_seed = largest_known.min(max_val);

        // All push/check/pop work shares one solver lock acquisition via
        // with_z3_solver. The outer push/pop brackets two binary-search loops
        // (min in [0, hi_seed], then max in [lo_seed, max_val]) each with
        // nested per-iteration push/check/pop pairs. All push/pop pairs are
        // balanced when the closure returns, so the Z3 scope stack returns to
        // its pre-closure depth — safe for both the None (per-context) and
        // Some (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();

            // Binary search for min in [0, hi_seed], then max in [lo_seed, max_val].
            let min_val = bsearch_min(solver, &ast, width, 0, hi_seed, |a, m| a.bvule(m));
            let max_result = bsearch_max(solver, &ast, width, lo_seed, max_val, |a, m| a.bvuge(m));

            solver.pop(1);
            Some((min_val, max_result))
        })
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

        let val_ast = make_bv_const(value, width);

        let constraint = ast.eq(&val_ast);
        // `solution` asks whether a specific value is feasible: the query set
        // is the target's symbols, same shape as an eval.
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));

        // All push/assert/check/pop work shares one solver lock acquisition via
        // with_z3_solver. The push/pop pair is balanced inside the closure, so
        // the Z3 scope stack returns to its pre-closure depth — safe for both
        // the None (per-context) and Some (shared-lineage) dispatch paths.
        self.with_z3_solver(|solver| {
            solver.push();
            solver.assert(&constraint);
            let result = matches!(
                timed_check(solver, CheckSite::Satisfiable),
                z3::SatResult::Sat
            );
            solver.pop(1);
            result
        })
    }

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
        bv.as_u128().map(|v| u128_to_be_bytes_width(v, width))
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_many(&self, bvs: &[RustBV]) -> Option<Vec<u128>> {
        // Without Z3, can only evaluate concrete values
        bvs.iter().map(|bv| bv.as_u128()).collect()
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
}
