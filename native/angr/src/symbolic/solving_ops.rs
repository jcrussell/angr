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
//! directly and are not part of the read path. See bead angr-a2br.2.4
//! for the slice plan.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. Every query
//! here runs over guest-derived constraints, and an undecided Z3 result is
//! propagated as `None` (invariant `invariant-z3-unknown-not-unsat`), never
//! collapsed or unwrapped.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! `pub(super)` (== `pub(in crate::symbolic)`) keeps the promoted caches
//! module-private — no public API leak.
#![deny(clippy::unwrap_used, clippy::expect_used)]

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

/// Result of a bounded solution enumeration ([`SymContext::eval_upto_checked`],
/// [`SymContext::solutions_checked`]).
///
/// The plain `Vec` returned by `eval_upto`/`solutions` cannot distinguish "these
/// are all the feasible values" from "Z3 stopped deciding after `k` of them" —
/// both come back as a short `Vec` (angr-03vl4.85). `undecided` carries that
/// distinction so a caller whose correctness depends on exhaustiveness (see
/// `AddressConcretizer::concretize_internal`) can refuse the partial set.
///
/// Note that `undecided == false` does **not** mean the set is complete: a
/// decided enumeration that returns exactly `n` values may simply have hit the
/// caller's cap. The idiom for detecting *that* is to request `n + 1` and treat
/// `len > n` as an overflow; both checks are needed.
pub struct Enumeration<T> {
    /// The values found, in canonical ascending order.
    pub values: Vec<T>,
    /// True when enumeration stopped on an undecided Z3 result (timeout, or a
    /// model/extraction failure) rather than on a decided Unsat or the `n` cap.
    pub undecided: bool,
}

impl<T> Enumeration<T> {
    /// A fully decided enumeration (concrete fast paths, `n == 0`, and the
    /// non-Z3 stubs, none of which ever consult a solver).
    fn decided(values: Vec<T>) -> Self {
        Self {
            values,
            undecided: false,
        }
    }
}

/// Render a concrete `u128` as a big-endian byte vector sized for `width`
/// bits (`ceil(width/8)` bytes). For widths ≤ 128 the low `byte_len` bytes
/// of the value are taken; wider widths left-pad with zeros. Shared by the
/// concrete fast paths of `eval_wide` / `eval_upto_wide`.
fn u128_to_be_bytes_width(value: u128, width: u32) -> Vec<u8> {
    let byte_len = width.div_ceil(8) as usize;
    let bytes = value.to_be_bytes();
    if byte_len <= 16 {
        // overflow-ok: guarded by this branch's `byte_len <= 16`.
        bytes[16 - byte_len..].to_vec()
    } else {
        let mut result = vec![0u8; byte_len];
        // overflow-ok: guarded by the `else`, i.e. `byte_len > 16`.
        result[byte_len - 16..].copy_from_slice(&bytes);
        result
    }
}

/// Largest unsigned value representable in `width` bits, saturating at
/// `u128::MAX`.
///
/// The saturation matters: `1u128 << 128` overflows (panic under debug,
/// wrap under release), and callers that binary-search in a `u128` cannot
/// represent anything above `u128::MAX` anyway — the `width > 128` guard in
/// `min` / `max` / `range_seeded` bails out before the search rather than
/// returning a truncated extremum (angr-cxw7, extended to the seeded path in
/// angr-sqfj8.103). Shared by `lex_min_witness`,
/// `eval_upto_ascending`, both arms of `min` and `max`, and `range_seeded`
/// so a future width-boundary fix lands in one place (angr-9ke6b.141).
///
/// The computation itself lives in [`RustBV::all_ones_mask`] — a max value at
/// `width` bits and that width's all-ones mask are the same number, and the
/// engine had grown three independent copies of the saturating shift
/// (angr-0jh0j.55). This stays a named wrapper because the solving path reads
/// its result as a search bound, not as a mask; it carries no logic of its own.
#[cfg(feature = "vex-engine-z3")]
#[inline]
fn max_val_for_width(width: u32) -> u128 {
    RustBV::all_ones_mask(width)
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
///
/// Returns `None` when any mid-bisection check comes back Z3 `Unknown` (a
/// timeout under a tight `set_timeout`, angr-ph300.43): the search cannot tell
/// "nothing at or below mid" from "gave up", and swallowing the Unknown as the
/// former would move `lo` past the true minimum and return a confidently wrong
/// bound. Aborting lets the caller degrade to `None` rather than fabricate one.
#[cfg(feature = "vex-engine-z3")]
fn bsearch_min(
    solver: &z3::Solver,
    ast: &z3::ast::BV,
    width: u32,
    mut lo: u128,
    mut hi: u128,
    cmp: impl Fn(&z3::ast::BV, &z3::ast::BV) -> z3::ast::Bool,
) -> Option<u128> {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        solver.push();
        let mid_ast = make_bv_const(mid, width);
        solver.assert(cmp(ast, &mid_ast));
        let check = timed_check(solver, CheckSite::MinSearch);
        solver.pop(1);
        // Unknown (timeout) aborts with None — never move a bound (angr-ph300.43).
        match check.decided()? {
            true => hi = mid,
            false => lo = mid + 1,
        }
    }
    Some(lo)
}

/// Binary search for the maximum feasible value of `ast` in `[lo, hi]`.
///
/// Ceil-mid bisection (avoids an infinite loop when `lo + 1 == hi`): assert
/// `cmp(ast, mid)` (i.e. `ast >= mid`) and keep the upper half when
/// satisfiable. `cmp` selects the signed (`bvsge`) vs unsigned (`bvuge`)
/// ordering. Shared by `max` and the max half of `range_seeded`; the caller
/// owns the surrounding `with_z3_solver` push/pop frame.
///
/// Returns `None` on a Z3 `Unknown` mid-bisection check for the same reason as
/// [`bsearch_min`] (angr-ph300.43): swallowing the timeout would drop `hi`
/// below the true maximum and return a wrong extremum.
#[cfg(feature = "vex-engine-z3")]
fn bsearch_max(
    solver: &z3::Solver,
    ast: &z3::ast::BV,
    width: u32,
    mut lo: u128,
    mut hi: u128,
    cmp: impl Fn(&z3::ast::BV, &z3::ast::BV) -> z3::ast::Bool,
) -> Option<u128> {
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        solver.push();
        let mid_ast = make_bv_const(mid, width);
        solver.assert(cmp(ast, &mid_ast));
        let check = timed_check(solver, CheckSite::MaxSearch);
        solver.pop(1);
        // Unknown (timeout) aborts with None — never move a bound (angr-ph300.43).
        match check.decided()? {
            true => lo = mid,
            false => hi = mid - 1,
        }
    }
    Some(lo)
}

impl SymContext {
    /// Debug: dump solver state as string for comparison.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_string(&self) -> String {
        self.with_z3_solver(|solver| format!("{solver}"))
    }

    /// Test-only: pin the solver's `rlimit` (resource budget) so the next
    /// check aborts to a Z3 Unknown deterministically — the stand-in for a
    /// timeout that a wall-clock `timeout_ms` cannot give a test without
    /// flakiness (see `test_pin_rlimit_reaches_the_solver`). `rlimit = 0`
    /// restores an unbounded budget.
    ///
    /// Lives on `SymContext` rather than beside `build_solver_params` because
    /// the `solver_build` module is private to `symbolic/`, and the prune-gate
    /// tests in `state/tests/solver_gate.rs` need the same rig.
    #[cfg(all(test, feature = "vex-engine-z3"))]
    pub(crate) fn pin_rlimit_for_test(&self, rlimit: u32) {
        self.with_z3_solver(|solver| {
            let mut params = super::solver_build::build_solver_params(self.timeout_ms());
            params.set_u32("rlimit", rlimit);
            solver.set_params(&params);
        });
    }

    // =========================================================================
    // Satisfiability & Evaluation (Z3-backed)
    // =========================================================================

    /// Check if the current constraints are satisfiable, reporting an
    /// undecided (Z3 Unknown / timeout) query as `None` rather than collapsing
    /// it into `false`.
    ///
    /// This is the honest form of `is_sat`: a caller that would *prune* on a
    /// `false` must use this one, because "Z3 gave up" and "proven
    /// contradictory" have opposite consequences for soundness — dropping a
    /// state on the former loses a feasible path (angr-03vl4.62). The same
    /// distinction `check_branch_feasibility` already makes for branch
    /// pruning (`invariant-z3-unknown-not-unsat`).
    ///
    /// `None` is never cached: a later call retries the query.
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_sat_checked(&self) -> Option<bool> {
        // Check cache first
        if let Some(cached) = self.sat_cache.get() {
            return Some(cached);
        }
        let _class =
            query_class::scope(|| query_class::classify_sat(&self.get_assumed_constraints()));
        // Perform actual SAT check. Both the check() and the post-check
        // get_model() must happen under the same solver lock, so both run
        // inside the with_z3_solver closure.
        //
        // Returns None on a Z3 Unknown (timeout): the constraint set's
        // satisfiability is genuinely undetermined, so we must NOT cache it.
        // Caching Some(false) after a transient timeout would pin the context
        // unsatisfiable permanently — every later satisfiable()/eval gate on it
        // returns the stale false even once the timeout budget would allow a
        // real answer (angr-ph300.43).
        let result: Option<bool> = self.with_z3_solver(|solver| {
            let decided = timed_check(solver, CheckSite::Satisfiable).decided();
            if decided == Some(true) {
                // Populate model_cache — get_model is essentially free
                // after a successful check, and the model lets
                // check_branch_feasibility skip one of two Z3 checks.
                let mut cache = self.model_cache.borrow_mut();
                if cache.is_none()
                    && let Some(m) = solver.get_model()
                {
                    *cache = Some(m);
                }
            }
            // None (Unknown) propagates untouched: the caller leaves sat_cache
            // unset so a later query retries (angr-ph300.43).
            decided
        });
        // Only a decided result updates the cache; Unknown leaves it unset so a
        // later call retries.
        if let Some(v) = result {
            self.sat_cache.set(Some(v));
        }
        result
    }

    /// Check if the current constraints are satisfiable.
    ///
    /// Lenient form: an undecided query reports `false`. Correct only for
    /// callers whose `false` branch is "give up on producing an answer"
    /// (`min`/`max`/`range` return `None`); a caller that *prunes* on `false`
    /// must use `is_sat_checked` instead (angr-03vl4.62).
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_sat(&self) -> bool {
        silent_default!(
            cat_c,
            self.is_sat_checked(),
            false,
            "is_sat: Z3 returned Unknown (timeout); reporting not-satisfiable without caching it"
        )
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
    /// Returns (can_be_true, can_be_false). One `with_z3_solver`
    /// acquisition covers both directions, rather than one per direction as a
    /// naive `can_be_true()` + `can_be_false()` pair would take.
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
                    // A Z3 Unknown (timeout) here must NOT prune the false
                    // branch: only a decided Unsat proves ¬cond infeasible
                    // (angr-ph300.43). Conservatively keep the branch on Unknown
                    // (decided() == None) — only Some(false) prunes.
                    let can_false =
                        timed_check(solver, CheckSite::BranchFalse).decided() != Some(false);
                    solver.pop(1);
                    (true, can_false)
                }
                Some(false) => {
                    Z3_BRANCH_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                    // can_be_false=true is proven by the model. Check cond.
                    solver.push();
                    solver.assert(&bool_ast);
                    // Unknown must not prune the true branch (angr-ph300.43):
                    // only a decided Unsat (Some(false)) makes it infeasible.
                    let can_true =
                        timed_check(solver, CheckSite::BranchTrue).decided() != Some(false);
                    solver.pop(1);
                    (can_true, true)
                }
                None => {
                    Z3_BRANCH_MODEL_MISS_COUNT.fetch_add(1, Ordering::Relaxed);
                    // No cached model: do the original two-check flow. A Z3
                    // Unknown (timeout) on either check must not prune that
                    // branch — only a decided Unsat does (angr-ph300.43). The
                    // pre-fix `matches!(_, Sat)` treated a timeout on the cond
                    // check as "cond infeasible" and returned (false, true),
                    // silently killing the feasible true branch.
                    solver.push();
                    solver.assert(&bool_ast);
                    let cond_check = timed_check(solver, CheckSite::BranchTrue).decided();
                    solver.pop(1);
                    let can_true = cond_check != Some(false);

                    // Only short-circuit on a proven-infeasible cond (decided
                    // Unsat), never on an undecided Unknown (None).
                    if cond_check == Some(false) {
                        return (false, true); // Must be false-only
                    }

                    solver.push();
                    solver.assert(bool_ast.not());
                    let can_false =
                        timed_check(solver, CheckSite::BranchFalse).decided() != Some(false);
                    solver.pop(1);

                    (can_true, can_false)
                }
            }
        })
    }

    /// Whether strict-deterministic witness selection is on (angr-op0dn.10.2).
    #[cfg(feature = "vex-engine-z3")]
    pub fn is_deterministic(&self) -> bool {
        self.deterministic.load(Ordering::Relaxed)
    }

    /// Turn strict-deterministic witness selection on or off (angr-op0dn.10.2).
    ///
    /// Default `false` — the flag is opt-in because it buys reproducibility
    /// with Z3 checks: every witness costs an `O(log width)` binary search
    /// (see [`min`](Self::min)) instead of one `get_model`, and the warm
    /// model-cache seed (angr-ovqja.4) is bypassed. With it on,
    /// [`eval`](Self::eval) returns the unsigned minimum of the feasible set
    /// and [`eval_upto`](Self::eval_upto) returns its ascending prefix, so a
    /// truncated result is a canonical prefix of the sorted full set rather
    /// than an arbitrary Z3-chosen subset.
    ///
    /// Carve-out: [`eval_upto_wide`](Self::eval_upto_wide) is unaffected. Its
    /// widths exceed the `u128` bounds the binary search tracks (`min` itself
    /// returns `None` above 128 bits, angr-cxw7); it keeps the
    /// enumerate-then-sort canonicalization from angr-op0dn.10.1, which is
    /// canonical only when `n >= #feasible`.
    ///
    /// Inherited parent→child on [`fork`](Self::fork), so setting it on a seed
    /// context covers the whole lineage.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_deterministic(&self, v: bool) {
        self.deterministic.store(v, Ordering::Relaxed);
    }

    /// Evaluate a bitvector to a concrete value if possible.
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval(&self, bv: &RustBV) -> Option<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Some(v);
        }

        // Strict-deterministic mode (angr-op0dn.10.2): the witness is the
        // unsigned minimum of the feasible set rather than whatever model Z3
        // built. Reuses the `min` binary search wholesale — including its
        // is_sat gate, so an unsat context still yields None. Deliberately
        // skips the model-cache read below: a cached model is a history-
        // dependent arbitrary witness, which is exactly the nondeterminism
        // this mode exists to remove.
        if self.is_deterministic() {
            return self.min(bv, false);
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
            match timed_check(solver, CheckSite::Eval).decided() {
                Some(true) => {
                    self.sat_cache.set(Some(true));
                }
                // Only a decided Unsat pins sat_cache=false; a Z3 Unknown
                // (None, timeout) leaves it unset so a later query can retry,
                // rather than pinning the context unsatisfiable forever
                // (angr-ph300.43).
                Some(false) => {
                    self.sat_cache.set(Some(false));
                    return None;
                }
                None => return None,
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
        self.cached_model_eval_with(ast, extract_bv_value)
    }

    /// Generic core of `cached_model_eval`: read `ast`'s value out of the warm
    /// cached model (if any) with a caller-supplied extractor. The wide
    /// (`Vec<u8>`) enumeration path in `enumerate_distinct` needs the same
    /// warm-model read with `extract_bv_value_wide` instead.
    #[cfg(feature = "vex-engine-z3")]
    fn cached_model_eval_with<T>(
        &self,
        ast: &z3::ast::BV,
        extract: impl Fn(&z3::ast::BV) -> Option<T>,
    ) -> Option<T> {
        let cache = self.model_cache.borrow();
        let model = cache.as_ref()?;
        let result = model.eval(ast, true)?;
        extract(&result)
    }

    /// Enumerate up to `n` distinct satisfying values of `ast`, in canonical
    /// ascending order. Shared core of `eval_upto` and `eval_upto_wide` — see
    /// those for the per-width fast paths and the soundness argument; this is
    /// only the enumeration machinery, so a fix here lands on both.
    ///
    /// Iteration 0 is seeded from the warm `model_cache` when present
    /// (angr-ovqja.4), skipping exactly one check+get_model. Each subsequent
    /// iteration does check → get_model → `extract` → assert
    /// `ast != exclusion_of(value)`. Anything but a decided Sat stops the loop:
    /// Unsat means the set is exhausted, Unknown (timeout) means undetermined —
    /// either way the prefix gathered so far is a valid set (angr-ph300.43). A
    /// missing model or failed extraction stops it the same way.
    ///
    /// The two stop reasons are *not* interchangeable to every caller, so they
    /// are reported apart in [`Enumeration::undecided`]: only the Unsat stop (or
    /// exhausting `n`) leaves a prefix a caller may treat as the whole feasible
    /// set (angr-03vl4.85).
    ///
    /// The trailing sort is canonical *presentation* order (angr-op0dn.10.1):
    /// the exclude-loop decides WHICH values come back, this only decides in
    /// what order. Any future short-circuit must land ABOVE that sort.
    ///
    /// The check/get_model/assert-exclude iterations share one solver lock
    /// acquisition via `with_z3_solver`. The push/pop pair is balanced inside
    /// the closure, so the Z3 scope stack returns to its pre-closure depth
    /// before `f` returns — safe for both the None (per-context) and Some
    /// (shared-lineage) dispatch paths.
    #[cfg(feature = "vex-engine-z3")]
    fn enumerate_distinct<T: Ord>(
        &self,
        ast: &z3::ast::BV,
        n: usize,
        extract: impl Fn(&z3::ast::BV) -> Option<T>,
        exclusion_of: impl Fn(&T) -> z3::ast::BV,
    ) -> Enumeration<T> {
        let mut results = Vec::with_capacity(n);
        let mut undecided = false;
        let seeded = self.cached_model_eval_with(ast, &extract);

        self.with_z3_solver(|solver| {
            solver.push();

            let mut remaining = n;
            if let Some(value) = seeded {
                Z3_EVAL_UPTO_MODEL_HIT_COUNT.fetch_add(1, Ordering::Relaxed);
                solver.assert(ast.eq(exclusion_of(&value)).not());
                results.push(value);
                remaining -= 1;
            }

            for _ in 0..remaining {
                // Only a decided Unsat means "no more solutions"; Unknown, and
                // likewise a missing model or a failed extraction, leave the
                // remainder of the feasible set unknown.
                match timed_check(solver, CheckSite::EvalUpto).decided() {
                    Some(true) => {}
                    Some(false) => break,
                    None => {
                        undecided = true;
                        break;
                    }
                }
                let Some(model) = solver.get_model() else {
                    undecided = true;
                    break;
                };
                let Some(result) = model.eval(ast, true) else {
                    undecided = true;
                    break;
                };
                let Some(value) = extract(&result) else {
                    undecided = true;
                    break;
                };
                solver.assert(ast.eq(exclusion_of(&value)).not());
                results.push(value);
            }

            solver.pop(1);
        });

        results.sort_unstable();
        Enumeration {
            values: results,
            undecided,
        }
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

        // Strict-deterministic mode (angr-op0dn.10.7): the wide witness is the
        // unsigned minimum of the feasible set, the same rule `eval` follows in
        // this mode (angr-op0dn.10.2) — `min` itself is unusable above 128 bits
        // (angr-cxw7), so the minimum is built byte-wise instead. Deliberately
        // skips the model-cache read below for the reason `eval` does: a cached
        // model is a history-dependent arbitrary witness.
        if self.is_deterministic() {
            let _class = query_class::scope(|| {
                query_class::classify_eval(bv, &self.get_assumed_constraints())
            });
            return self.min_wide(bv);
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
            // Proceed only on a decided Sat; both Unsat and Unknown yield None
            // (this path never wrote sat_cache — preserve that, angr-ph300.43).
            if timed_check(solver, CheckSite::Eval).decided() != Some(true) {
                return None;
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

        // Strict-deterministic mode (angr-op0dn.10.7): minimize the parts in
        // order under each other's pinned values, so the joint witness is a
        // function of the constraints alone instead of whatever model Z3 built.
        // The width gate is all-or-nothing at the *batch* level, not a per-part
        // carve-out: one part wider than 128 bits drops the whole call to the
        // arbitrary-model path, narrow parts included. `bsearch_min` tracks its
        // bounds in a u128 and has nothing canonical to return for the wide part
        // (angr-cxw7, the same carve-out `eval_upto` makes on its single BV), and
        // the parts must all come off one witness, so they cannot be split across
        // the two paths — a mixed batch loses strict determinism wholesale. Skips
        // the model cache
        // below for the reason `eval` does: a cached model is an arbitrary,
        // history-dependent witness.
        if self.is_deterministic() && bvs.iter().all(|bv| bv.width() <= 128) {
            let _class = query_class::scope(|| {
                query_class::classify_eval_many(bvs, &self.get_assumed_constraints())
            });
            let symbolic: Vec<(z3::ast::BV, u32)> = bvs
                .iter()
                .filter(|bv| bv.as_u128().is_none())
                .map(|bv| (bv.to_z3_ast(), bv.width()))
                .collect();
            let mins = if symbolic.is_empty() {
                Vec::new()
            } else {
                self.lex_min_witness(&symbolic)?
            };
            let mut mins = mins.into_iter();
            return bvs
                .iter()
                .map(|bv| bv.as_u128().or_else(|| mins.next()))
                .collect();
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
            match timed_check(solver, CheckSite::Eval).decided() {
                Some(true) => self.sat_cache.set(Some(true)),
                // Unknown (None, timeout) must not pin the context unsat
                // (angr-ph300.43); only a decided Unsat does.
                Some(false) => {
                    self.sat_cache.set(Some(false));
                    return None;
                }
                None => return None,
            }

            let model = solver.get_model()?;
            let values = eval_all_in_model(&model, bvs);
            *self.model_cache.borrow_mut() = Some(model);
            values
        })
    }

    /// Evaluate a bitvector and return up to n solutions.
    ///
    /// Drops the "did enumeration finish, or did Z3 give up?" signal — use
    /// [`eval_upto_checked`](Self::eval_upto_checked) when the answer's
    /// correctness depends on the set being complete (angr-03vl4.85).
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        self.eval_upto_checked(bv, n).values
    }

    /// [`eval_upto`](Self::eval_upto) plus the stop reason — see
    /// [`Enumeration`].
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto_checked(&self, bv: &RustBV, n: usize) -> Enumeration<u128> {
        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Enumeration::decided(vec![v]);
        }

        if n == 0 {
            return Enumeration::decided(vec![]);
        }

        // Strict-deterministic mode (angr-op0dn.10.2): ascending enumeration
        // from the minimum, so a truncated result (n < #feasible) is the
        // canonical prefix of the sorted feasible set instead of an arbitrary
        // Z3-chosen subset. Subsumes the 10.1 sort for this mode.
        // Widths above 128 fall through to the default path: the binary search
        // tracks bounds in a u128 and `min` reports None above 128 bits
        // (angr-cxw7), so there is nothing canonical to build there — those
        // keep the 10.1 enumerate-then-sort guarantee.
        if self.is_deterministic() && bv.width() <= 128 {
            return self.eval_upto_ascending(bv, n);
        }

        let ast = bv.to_z3_ast();
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));

        // Soundness of the warm-model seed `enumerate_distinct` applies: per
        // `invalidate_model_if_inconsistent`, a surviving cached model satisfies
        // all permanent constraints, and iteration-0 runs under a fresh empty
        // push scope (no exclude constraints yet), so M(ast) with completion is
        // a genuine feasible solution (angr-ovqja.4). eval_upto is treated as
        // unordered by callers (e.g. `solutions()`), and the result count is
        // unchanged (min(n, #feasible)), so the solution set is faithful.
        //
        // The canonical ascending order the helper returns makes the exhaustive
        // case (n >= #feasible) bit-for-bit reproducible across runs regardless
        // of which witness Z3 or the warm-model seed found first.
        let width = bv.width();
        self.enumerate_distinct(&ast, n, extract_bv_value, |value| {
            make_bv_const(*value, width)
        })
    }

    /// Canonical joint witness: the lexicographic minimum over `parts`, in the
    /// order given (angr-op0dn.10.7). `parts` must be non-empty and every width
    /// must be <= 128 (the `bsearch_min` bound; see `min`).
    ///
    /// Each part is minimized with every earlier part pinned to its own chosen
    /// value, so all parts still come off ONE satisfying assignment (the
    /// angr-ue4ro invariant) while that assignment is now determined by the
    /// constraints rather than by Z3's search. For the big-endian byte
    /// decomposition both callers hand it, the lexicographic minimum over the
    /// parts IS the unsigned minimum of the reassembled value — the witness rule
    /// `eval` uses in this mode (angr-op0dn.10.2).
    ///
    /// `None` when the constraint set is unsat.
    #[cfg(feature = "vex-engine-z3")]
    fn lex_min_witness(&self, parts: &[(z3::ast::BV, u32)]) -> Option<Vec<u128>> {
        self.with_z3_solver(|solver| {
            solver.push();
            let mut values = Vec::with_capacity(parts.len());
            for (ast, width) in parts {
                // The pins asserted below are satisfiable by construction, so an
                // unsat here can only come from the base constraint set. A Z3
                // Unknown (timeout) aborts without pinning sat_cache=false — the
                // base set's satisfiability stays undetermined (angr-ph300.43).
                match timed_check(solver, CheckSite::Eval).decided() {
                    Some(true) => {}
                    Some(false) => {
                        solver.pop(1);
                        self.sat_cache.set(Some(false));
                        return None;
                    }
                    // Unknown (None): base set's satisfiability undetermined —
                    // abort without pinning sat_cache (angr-ph300.43).
                    None => {
                        solver.pop(1);
                        return None;
                    }
                }
                let max_val = max_val_for_width(*width);
                let value = match bsearch_min(solver, ast, *width, 0, max_val, |a, m| a.bvule(m)) {
                    Some(v) => v,
                    // Mid-bisection Z3 timeout (angr-ph300.43): no canonical
                    // minimum for this part, so abort the whole joint witness
                    // rather than return one built on a fabricated value. Leave
                    // sat_cache unset — the base set's satisfiability is unknown.
                    None => {
                        solver.pop(1);
                        return None;
                    }
                };
                solver.assert(ast.eq(make_bv_const(value, *width)));
                values.push(value);
            }
            solver.pop(1);
            self.sat_cache.set(Some(true));
            Some(values)
        })
    }

    /// Unsigned minimum of a bitvector of ANY width, as big-endian bytes
    /// (angr-op0dn.10.7). The strict-deterministic witness for `eval_wide`,
    /// where `min` cannot be reused because it reports `None` above 128 bits
    /// (angr-cxw7). Minimizing the big-endian bytes most-significant-first is
    /// exactly minimizing the value, and each byte fits `bsearch_min`'s u128.
    #[cfg(feature = "vex-engine-z3")]
    fn min_wide(&self, bv: &RustBV) -> Option<Vec<u8>> {
        let ast = bv.to_z3_ast();
        let mut parts = Vec::with_capacity(bv.width().div_ceil(8) as usize);
        let mut hi = bv.width();
        while hi > 0 {
            // The ragged chunk is the FIRST (most-significant) one, not the
            // last: `make_bv_from_bytes` (bv_codec.rs) documents the byte array
            // as a right-aligned big-endian value, so a width that is not a
            // multiple of 8 has a short TOP byte. Slicing full bytes off the
            // top instead would land the ragged chunk in the least-significant
            // byte and misalign every other byte by `width % 8` bits
            // (angr-6cp06.45).
            let chunk = if hi.is_multiple_of(8) { 8 } else { hi % 8 };
            let lo = hi - chunk;
            parts.push((ast.extract(hi - 1, lo), chunk));
            hi = lo;
        }
        Some(
            self.lex_min_witness(&parts)?
                .into_iter()
                .map(|v| v as u8)
                .collect(),
        )
    }

    /// Canonical ascending enumeration for [`eval_upto`](Self::eval_upto) under strict-
    /// deterministic mode (angr-op0dn.10.2). Caller guarantees `n > 0`,
    /// `width <= 128`, and a symbolic `bv`.
    ///
    /// Each iteration binary-searches the minimum feasible value at or above
    /// `lo` (reusing [`bsearch_min`], the same machinery [`min`](Self::min)
    /// drives), records it, then raises the floor to `v + 1`. So witness `i` is
    /// the `i`-th smallest feasible value and the returned Vec is the ascending
    /// prefix of the sorted feasible set — identical across runs regardless of
    /// which model Z3 would have produced.
    ///
    /// Cost is why this is opt-in: `n * (1 + log2(width))` checks against the
    /// default path's `n` checks. No `get_model` call happens, so the warm
    /// model-cache seed is bypassed and `model_cache` is left untouched (a
    /// history-dependent seed is precisely the nondeterminism being removed).
    #[cfg(feature = "vex-engine-z3")]
    fn eval_upto_ascending(&self, bv: &RustBV, n: usize) -> Enumeration<u128> {
        let width = bv.width();
        let ast = bv.to_z3_ast();
        let max_val = max_val_for_width(width);
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));

        let mut results = Vec::with_capacity(n);
        let mut undecided = false;
        self.with_z3_solver(|solver| {
            solver.push();

            let mut lo: u128 = 0;
            for _ in 0..n {
                // Is anything left at or above the current floor? The floor is
                // an asserted `ast >= lo` (below), so this check also covers an
                // unsat base constraint set on iteration 0.
                // Stop on anything but a decided Sat (Unsat = exhausted,
                // Unknown = undetermined); the prefix stays a valid set, and the
                // two reasons are reported apart (angr-03vl4.85).
                match timed_check(solver, CheckSite::EvalUpto).decided() {
                    Some(true) => {}
                    Some(false) => break,
                    None => {
                        undecided = true;
                        break;
                    }
                }
                // Feasible min in [lo, max_val] — `ast >= lo` is already
                // asserted, so bsearch_min never returns below the floor.
                // A mid-bisection Z3 timeout (angr-ph300.43) yields None: stop
                // enumerating rather than append a fabricated value. The prefix
                // gathered so far is still a valid ascending set.
                let Some(value) = bsearch_min(solver, &ast, width, lo, max_val, |a, m| a.bvule(m))
                else {
                    undecided = true;
                    break;
                };
                results.push(value);
                if value == max_val {
                    break; // No room above the top of the range.
                }
                lo = value + 1;
                solver.assert(ast.bvuge(make_bv_const(lo, width)));
            }

            solver.pop(1);
        });
        Enumeration {
            values: results,
            undecided,
        }
    }

    /// Evaluate a bitvector and return up to n solutions as byte arrays (big-endian).
    /// Handles values of any width without truncation.
    ///
    /// Lenient form: like [`eval_upto`](Self::eval_upto) the bare `Vec` cannot
    /// say *why* enumeration stopped. Callers whose correctness depends on the
    /// set being complete want
    /// [`eval_upto_wide_checked`](Self::eval_upto_wide_checked).
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto_wide(&self, bv: &RustBV, n: usize) -> Vec<Vec<u8>> {
        self.eval_upto_wide_checked(bv, n).values
    }

    /// [`eval_upto_wide`](Self::eval_upto_wide) plus the stop reason — see
    /// [`Enumeration`].
    #[cfg(feature = "vex-engine-z3")]
    pub fn eval_upto_wide_checked(&self, bv: &RustBV, n: usize) -> Enumeration<Vec<u8>> {
        let width = bv.width();

        // Fast path for concrete values
        if let Some(v) = bv.as_u128() {
            return Enumeration::decided(vec![u128_to_be_bytes_width(v, width)]);
        }

        if n == 0 {
            return Enumeration::decided(vec![]);
        }

        let ast = bv.to_z3_ast();
        let _class =
            query_class::scope(|| query_class::classify_eval(bv, &self.get_assumed_constraints()));

        // Same warm-seed soundness argument as eval_upto, and the exclusion
        // constants are built from the full byte string so the exclusion is
        // full-precision at any width.
        //
        // The helper's canonical order is ascending *numeric* order here too:
        // every witness is `width` bits wide, so all byte vectors have the same
        // length and big-endian lexicographic order coincides with numeric
        // order (angr-op0dn.10.1).
        self.enumerate_distinct(
            &ast,
            n,
            |result| extract_bv_value_wide(result, width),
            |bytes| make_bv_from_bytes(bytes, width),
        )
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
                let max_val = max_val_for_width(width);
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
                    let check = timed_check(solver, CheckSite::MinInit);
                    solver.pop(1);
                    match check.decided() {
                        // Some(true)=has_negative, Some(false)=no negative value.
                        Some(v) => v,
                        // The sign probe timed out (None): whether the feasible
                        // set reaches below zero is undetermined. Collapsing
                        // Unknown to false (has_negative=false) would confine the
                        // search to [0, max_positive] and converge on a fabricated
                        // non-negative minimum even when the true minimum is
                        // negative. Propagate the unknown instead — the same rule
                        // bsearch_min already follows on a mid-bisection Unknown
                        // (angr-n0irt.1, invariant-z3-unknown-not-unsat).
                        None => {
                            solver.pop(1); // balance the outer push() before abort
                            return None;
                        }
                    }
                };

                if has_negative {
                    // Minimum is negative, search in [sign_bit, max_val] range.
                    // Binding the witness in the arm that requires it keeps
                    // `witness_is_negative` (itself `witness.map(..)`-derived)
                    // from being re-proved with an unwrap.
                    let hi_seed = match witness {
                        // Witness is in [sign_bit, max_val] and feasible.
                        Some(w) if witness_is_negative => w.min(max_val),
                        _ => max_val,
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
                let max_val = max_val_for_width(width);
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
            // `result` is None on a mid-bisection Z3 timeout (angr-ph300.43):
            // propagate the unknown rather than a fabricated extremum.
            result
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
                let max_val = max_val_for_width(width);
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
                    let check = timed_check(solver, CheckSite::MaxInit);
                    solver.pop(1);
                    match check.decided() {
                        // Some(true)=has_non_negative, Some(false)=all negative.
                        Some(v) => v,
                        // The sign probe timed out (None): whether the feasible
                        // set reaches at or above zero is undetermined. Collapsing
                        // Unknown to false (has_non_negative=false) would confine
                        // the search to [sign_bit, max_val] and converge on a
                        // fabricated negative maximum even when the true maximum
                        // is non-negative. Propagate the unknown instead — the
                        // same rule bsearch_max follows on a mid-bisection Unknown
                        // (angr-n0irt.1, invariant-z3-unknown-not-unsat).
                        None => {
                            solver.pop(1); // balance the outer push() before abort
                            return None;
                        }
                    }
                };

                if has_non_negative {
                    // Maximum is non-negative, search in [0, max_positive] range.
                    // Witness, when non-negative, gives a tight lower bound.
                    // Same shape as `bsearch_min`'s `hi_seed`: bind the
                    // witness in the arm that needs it rather than unwrapping.
                    let lo_seed = match witness {
                        Some(w) if witness_is_non_negative => w.min(max_positive),
                        _ => 0,
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
                let max_val = max_val_for_width(width);
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
            // None on a mid-bisection Z3 timeout (angr-ph300.43).
            result
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
    ///
    /// Returns `None` for widths above 128 — the bounds are tracked in a
    /// `u128`, so a wider extremum cannot be represented (same guard as
    /// `min` / `max`).
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

        // Same u128-bound guard `min` and `max` carry: the two binary searches
        // below track bounds in a u128, so for widths above 128 the true
        // extremum can exceed `u128::MAX` and `max_val_for_width` saturates.
        // Report unknown rather than a truncated range (angr-sqfj8.103, the
        // angr-cxw7 fix applied to the seeded path). 128-bit BVs are fine:
        // their range is exactly [0, u128::MAX].
        if width > 128 {
            return None;
        }

        let _class = query_class::scope(|| {
            query_class::classify_extrema(bv, &self.get_assumed_constraints())
        });
        let max_val: u128 = max_val_for_width(width);

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
            // None if either half hit a Z3 timeout mid-bisection (angr-ph300.43):
            // a half-known range would be a wrong bound, so report unknown.
            Some((min_val?, max_result?))
        })
    }

    /// Get up to n concrete solutions for a bitvector.
    ///
    /// This is a convenience wrapper around eval_upto — and inherits its
    /// inability to report *why* enumeration stopped. Callers that need the
    /// complete feasible set want [`solutions_checked`](Self::solutions_checked).
    #[cfg(feature = "vex-engine-z3")]
    pub fn solutions(&self, bv: &RustBV, n: usize) -> Vec<u128> {
        self.eval_upto(bv, n)
    }

    /// [`solutions`](Self::solutions) plus the stop reason — see
    /// [`Enumeration`].
    #[cfg(feature = "vex-engine-z3")]
    pub fn solutions_checked(&self, bv: &RustBV, n: usize) -> Enumeration<u128> {
        self.eval_upto_checked(bv, n)
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
            // `value` is a solution only if pinning it is a decided Sat; a
            // timeout (None) cannot prove feasibility, so it is conservatively
            // "not a solution" — preserving this query's original behavior while
            // making the Unknown handling explicit (angr-qwyti.3).
            let result = timed_check(solver, CheckSite::Satisfiable).decided() == Some(true);
            solver.pop(1);
            result
        })
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn is_sat(&self) -> bool {
        // Without Z3, we assume everything is satisfiable
        true
    }

    /// Without Z3 there is no query to time out, so the answer is always decided.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn is_sat_checked(&self) -> Option<bool> {
        Some(true)
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
        self.eval_upto_checked(bv, n).values
    }

    /// Without Z3 nothing is ever undecided: a concrete value enumerates to
    /// itself and a symbolic one to the empty set, both definitively.
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto_checked(&self, bv: &RustBV, n: usize) -> Enumeration<u128> {
        if n == 0 {
            return Enumeration::decided(vec![]);
        }
        // Without Z3, can only return concrete values
        Enumeration::decided(bv.as_u128().map(|v| vec![v]).unwrap_or_default())
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto_wide(&self, bv: &RustBV, n: usize) -> Vec<Vec<u8>> {
        self.eval_upto_wide_checked(bv, n).values
    }

    /// See the Z3 twin — without Z3 the stop reason is always "decided".
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn eval_upto_wide_checked(&self, bv: &RustBV, n: usize) -> Enumeration<Vec<u8>> {
        if n == 0 {
            return Enumeration::decided(vec![]);
        }
        Enumeration::decided(self.eval_wide(bv).map(|v| vec![v]).unwrap_or_default())
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
        self.eval_upto(bv, n)
    }

    /// See the Z3 twin — without Z3 the stop reason is always "decided".
    #[cfg(not(feature = "vex-engine-z3"))]
    pub fn solutions_checked(&self, bv: &RustBV, n: usize) -> Enumeration<u128> {
        self.eval_upto_checked(bv, n)
    }
}

// `test_submod!`, not a hand-written `mod`: this file's `#![deny(clippy::
// unwrap_used, clippy::expect_used)]` above propagates into the child test
// module, and the macro is what re-`allow`s the two lints there (angr-03vl4.83).
test_submod!(z3 "solving_ops_tests.rs" => solving_ops_tests);
