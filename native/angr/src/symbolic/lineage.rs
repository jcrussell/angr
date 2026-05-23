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

#![cfg(feature = "vex-engine-z3")]

use std::sync::atomic::{AtomicU64, Ordering};

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
/// integration patch wires resets into `reset_solver_stats`.
pub fn reset_lineage_stats() {
    LINEAGE_SWITCH_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_SWITCH_HOT_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_SWITCH_FAST_PATH_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_PUSH_COUNT.store(0, Ordering::Relaxed);
    LINEAGE_POP_COUNT.store(0, Ordering::Relaxed);
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
}

impl SharedLineageSolver {
    /// Construct from a fresh Z3 solver. Base assertions for the lineage
    /// should be installed via [`assert_base`](Self::assert_base) before
    /// any [`switch_to`](Self::switch_to)/[`with_solver`](Self::with_solver)
    /// calls — they sit at scope 0 and are never popped.
    pub fn new(z3: z3::Solver) -> Self {
        SharedLineageSolver {
            z3,
            loaded_path: ScopePath::new(),
        }
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
    ///    |target|)) walk through [`common_prefix_len`] when consecutive
    ///    queries come from the same state (the BFS-step intra-state query
    ///    burst that motivates the hot-state cache in the angr-v5a5 design).
    ///
    /// 2. **General prefix walk.** If the O(1) cache misses, fall back to
    ///    [`common_prefix_len`] for the full prefix calculation. Pops the
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
            return (0, 0);
        }

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
mod tests {
    use super::*;
    use z3::ast::Ast;

    fn make_solver() -> z3::Solver {
        // Use the default tactic — these tests don't depend on the
        // QF_BV-vs-SMT routing in context.rs's `build_solver`. The
        // thread-local Z3 context is implicitly initialized.
        z3::Solver::new()
    }

    fn bvconst(name: &str, width: u32) -> z3::ast::BV {
        z3::ast::BV::new_const(name, width)
    }

    fn eq_bv_const(bv: &z3::ast::BV, value: u64) -> Bool {
        bv._eq(&z3::ast::BV::from_u64(value, bv.get_size()))
    }

    /// Frame ids are unique within a single run.
    #[test]
    fn test_frame_ids_unique() {
        let a = mint_frame_id();
        let b = mint_frame_id();
        let c = mint_frame_id();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    /// Switching to an empty path on a fresh solver is a no-op.
    #[test]
    fn test_switch_empty_is_noop() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let empty = ScopePath::new();
        let (pops, pushes) = lin.switch_to(&empty);
        assert_eq!((pops, pushes), (0, 0));
        assert_eq!(lin.loaded_depth(), 0);
    }

    /// Switching to a path of length k pushes k frames; switching back
    /// to empty pops them.
    #[test]
    fn test_switch_push_then_pop() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_push_then_pop_x", 8);

        let path = vec![
            ScopeFrame::new(true, eq_bv_const(&x, 5)),
            ScopeFrame::new(true, x.bvugt(&z3::ast::BV::from_u64(0, 8))),
        ];

        let (pops, pushes) = lin.switch_to(&path);
        assert_eq!((pops, pushes), (0, 2));
        assert_eq!(lin.loaded_depth(), 2);

        let (pops, pushes) = lin.switch_to(&ScopePath::new());
        assert_eq!((pops, pushes), (2, 0));
        assert_eq!(lin.loaded_depth(), 0);
    }

    /// Switching to the same path twice is a no-op the second time —
    /// the hot-path optimization the BFS-thrashing heuristic in the
    /// parent bead's description depends on.
    #[test]
    fn test_switch_same_path_is_hot_noop() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_same_path_is_hot_noop_x", 8);

        let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];

        lin.switch_to(&path);
        let depth_after_first = lin.loaded_depth();
        let (pops, pushes) = lin.switch_to(&path);
        assert_eq!((pops, pushes), (0, 0), "second switch should be a no-op");
        assert_eq!(lin.loaded_depth(), depth_after_first);
    }

    /// Counters increment by the expected amount on switch/push/pop.
    /// Uses delta comparisons (post - pre) because the `LINEAGE_*`
    /// atomics are global and other tests in this module run in parallel
    /// and may bump them concurrently — we just need to verify our own
    /// operations contributed the right deltas.
    #[test]
    fn test_counters_increment() {
        let pre = lineage_stats();
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_counters_increment_x", 8);
        let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];

        // Switch into path (1 switch, 1 push), then switch to same path
        // (1 switch, 1 hot no-op via O(1) fast path), then back to empty
        // (1 switch, 1 pop).
        lin.switch_to(&path);
        lin.switch_to(&path);
        lin.switch_to(&ScopePath::new());

        let post = lineage_stats();
        let delta = |name: &str| {
            let pre_v = pre.iter().find(|(n, _)| *n == name).map_or(0, |(_, v)| *v);
            let post_v = post
                .iter()
                .find(|(n, _)| *n == name)
                .map_or(0, |(_, v)| *v);
            post_v.saturating_sub(pre_v)
        };
        // Lower bounds: other parallel tests can only ADD to these
        // counters, never subtract — but we must contribute at least
        // this many ourselves.
        assert!(
            delta("lineage_switch_count") >= 3,
            "≥3 switches (got {})",
            delta("lineage_switch_count")
        );
        assert!(
            delta("lineage_switch_hot_count") >= 1,
            "≥1 hot no-op (got {})",
            delta("lineage_switch_hot_count")
        );
        assert!(
            delta("lineage_switch_fast_path_count") >= 1,
            "≥1 fast-path hit (got {})",
            delta("lineage_switch_fast_path_count")
        );
        assert!(
            delta("lineage_push_count") >= 1,
            "≥1 push (got {})",
            delta("lineage_push_count")
        );
        assert!(
            delta("lineage_pop_count") >= 1,
            "≥1 pop (got {})",
            delta("lineage_pop_count")
        );
    }

    /// Sibling paths sharing a 2-frame prefix only pop+push the diverging
    /// 1-frame tail (the cost optimization the design targets).
    #[test]
    fn test_switch_common_prefix_only_diffs() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_common_prefix_only_diffs_x", 8);

        // Shared prefix: two frames the siblings would have inherited
        // from a common ancestor.
        let prefix = [
            ScopeFrame::new(true, x.bvugt(&z3::ast::BV::from_u64(0, 8))),
            ScopeFrame::new(true, x.bvult(&z3::ast::BV::from_u64(100, 8))),
        ];

        let mut sibling_a = prefix.to_vec();
        sibling_a.push(ScopeFrame::new(true, eq_bv_const(&x, 5)));

        let mut sibling_b = prefix.to_vec();
        sibling_b.push(ScopeFrame::new(true, eq_bv_const(&x, 42)));

        let (pops, pushes) = lin.switch_to(&sibling_a);
        assert_eq!((pops, pushes), (0, 3));

        // Switching to sibling_b should reuse the 2-frame prefix —
        // exactly one pop, exactly one push.
        let (pops, pushes) = lin.switch_to(&sibling_b);
        assert_eq!(
            (pops, pushes),
            (1, 1),
            "only the diverging tail frame should move"
        );
        assert_eq!(lin.loaded_depth(), 3);
    }

    /// Loaded constraints actually constrain the solver: an SMT query
    /// against `x == 5` finds x = 5; switching to a sibling with
    /// `x == 42` finds 42.
    #[test]
    fn test_loaded_constraint_actually_holds() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_loaded_constraint_actually_holds_x", 32);

        let path_5 = vec![ScopeFrame::new(true, eq_bv_const(&x, 5))];
        let val = lin.with_solver(&path_5, |solver| {
            assert_eq!(solver.check(), z3::SatResult::Sat);
            let m = solver.get_model().expect("model after Sat");
            m.eval(&x, true).and_then(|bv| bv.as_u64())
        });
        assert_eq!(val, Some(5));

        let path_42 = vec![ScopeFrame::new(true, eq_bv_const(&x, 42))];
        let val = lin.with_solver(&path_42, |solver| {
            assert_eq!(solver.check(), z3::SatResult::Sat);
            let m = solver.get_model().expect("model after Sat");
            m.eval(&x, true).and_then(|bv| bv.as_u64())
        });
        assert_eq!(val, Some(42));
    }

    /// Base assertions persist across switch_to/pop cycles.
    #[test]
    fn test_base_assertions_persist() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_base_assertions_persist_x", 8);

        // x > 10 at scope 0
        lin.assert_base(&x.bvugt(&z3::ast::BV::from_u64(10, 8)));

        // Push a frame constraining x < 20, query, then pop back.
        let scoped = vec![ScopeFrame::new(
            true,
            x.bvult(&z3::ast::BV::from_u64(20, 8)),
        )];
        lin.switch_to(&scoped);
        lin.switch_to(&ScopePath::new());

        // The base constraint must still be in force after pop.
        let val = lin.with_solver(&ScopePath::new(), |solver| {
            // Add a temporary constraint forcing x == 5 (violates base),
            // verify the solver reports UNSAT.
            solver.push();
            solver.assert(&eq_bv_const(&x, 5));
            let r = solver.check();
            solver.pop(1);
            r
        });
        assert_eq!(
            val,
            z3::SatResult::Unsat,
            "base x > 10 must still hold after pop"
        );
    }

    /// Three-deep sibling: A=[a,b,c], B=[a,b,d], C=[a,e]. Switching
    /// A → B reuses 2 frames; A → C reuses 1.
    #[test]
    fn test_switch_three_deep_siblings() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_three_deep_siblings_x", 16);

        let a = ScopeFrame::new(true, x.bvugt(&z3::ast::BV::from_u64(0, 16)));
        let b = ScopeFrame::new(true, x.bvult(&z3::ast::BV::from_u64(1000, 16)));
        let c = ScopeFrame::new(true, eq_bv_const(&x, 5));
        let d = ScopeFrame::new(true, eq_bv_const(&x, 7));
        let e = ScopeFrame::new(true, x.bvugt(&z3::ast::BV::from_u64(500, 16)));

        let path_abc = vec![a.clone(), b.clone(), c];
        let path_abd = vec![a.clone(), b, d];
        let path_ae = vec![a, e];

        let (pops, pushes) = lin.switch_to(&path_abc);
        assert_eq!((pops, pushes), (0, 3));

        // A→B: 2-frame prefix (a, b) reused; pop 1, push 1.
        let (pops, pushes) = lin.switch_to(&path_abd);
        assert_eq!((pops, pushes), (1, 1));

        // B→C: 1-frame prefix (a) reused; pop 2 (b, d), push 1 (e).
        let (pops, pushes) = lin.switch_to(&path_ae);
        assert_eq!((pops, pushes), (2, 1));
        assert_eq!(lin.loaded_depth(), 2);
    }

    /// O(1) hot-cache fast path: a same-path re-switch must return
    /// (0, 0) AND increment `lineage_switch_fast_path_count`. Divergent
    /// switches return non-zero pop/push counts. Counter is a lower-bound
    /// check because other parallel tests in the module also bump it.
    #[test]
    fn test_switch_fast_path_counter_only_fires_on_same_path() {
        let read = |name: &str, snapshot: &[(&'static str, u64)]| {
            snapshot
                .iter()
                .find(|(n, _)| *n == name)
                .map_or(0, |(_, v)| *v)
        };

        let pre = lineage_stats();
        let fp_pre = read("lineage_switch_fast_path_count", &pre);

        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_fast_path_counter_x", 8);
        let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];

        // First switch: divergent (empty → 1-frame). NOT a fast-path hit;
        // the return value reflects that 1 frame was pushed.
        let (pops, pushes) = lin.switch_to(&path);
        assert_eq!((pops, pushes), (0, 1), "first switch pushes 1 frame");

        // Second switch: same path → O(1) fast path fires, returns (0, 0).
        let (pops, pushes) = lin.switch_to(&path);
        assert_eq!(
            (pops, pushes),
            (0, 0),
            "second same-path switch must be a no-op"
        );

        // Third switch: back to empty → divergent, returns (1, 0).
        let (pops, pushes) = lin.switch_to(&ScopePath::new());
        assert_eq!((pops, pushes), (1, 0), "third switch pops 1 frame");

        // Lower-bound check: at least one fast-path hit was contributed
        // by us (the second switch). Other parallel tests can only ADD
        // to the global counter, never subtract.
        let post = lineage_stats();
        let fp_post = read("lineage_switch_fast_path_count", &post);
        assert!(
            fp_post.saturating_sub(fp_pre) >= 1,
            "at least 1 fast-path hit expected (got {})",
            fp_post.saturating_sub(fp_pre)
        );
    }

    /// Repeated re-entry on the SAME deep path (the BFS intra-state
    /// query-burst pattern the hot-cache targets) hits the fast path on
    /// every call after the first.
    #[test]
    fn test_switch_fast_path_deep_path_repeated_reentry() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_fast_path_deep_path_x", 32);

        // 10-frame deep path representing many accumulated constraints
        // (e.g., from a long-running state).
        let path: ScopePath = (0..10)
            .map(|i| {
                ScopeFrame::new(
                    true,
                    x.bvugt(&z3::ast::BV::from_u64(i as u64, 32)),
                )
            })
            .collect();

        // Initial switch into the deep path.
        lin.switch_to(&path);
        assert_eq!(lin.loaded_depth(), 10);

        // Capture fast-path counter, then re-enter 100 times. Every
        // re-entry must hit the O(1) cache (same path, same tail id,
        // same depth) — no Z3 push/pop should fire.
        let before = lineage_stats();
        let fp_before = before
            .iter()
            .find(|(n, _)| *n == "lineage_switch_fast_path_count")
            .map_or(0, |(_, v)| *v);
        let push_before = before
            .iter()
            .find(|(n, _)| *n == "lineage_push_count")
            .map_or(0, |(_, v)| *v);
        let pop_before = before
            .iter()
            .find(|(n, _)| *n == "lineage_pop_count")
            .map_or(0, |(_, v)| *v);

        let mut local_push = 0u64;
        let mut local_pop = 0u64;
        for _ in 0..100 {
            let (pops, pushes) = lin.switch_to(&path);
            assert_eq!(
                (pops, pushes),
                (0, 0),
                "every same-path re-entry must be a no-op"
            );
            local_push += pushes as u64;
            local_pop += pops as u64;
        }
        // Behavioural proof, race-free: this caller's own switch_to
        // calls returned 0 pops and 0 pushes for every re-entry.
        assert_eq!(local_push, 0);
        assert_eq!(local_pop, 0);

        let after = lineage_stats();
        let fp_after = after
            .iter()
            .find(|(n, _)| *n == "lineage_switch_fast_path_count")
            .map_or(0, |(_, v)| *v);
        // Counter check: at least 100 fast-path hits contributed by us.
        // Lower-bound rather than equality because other parallel tests
        // in this module also share the global counter.
        assert!(
            fp_after.saturating_sub(fp_before) >= 100,
            "expected ≥100 fast-path hits, got {}",
            fp_after.saturating_sub(fp_before)
        );
        // push_before/pop_before remain unused snapshots in the race-safe
        // version — kept as documentation of the invariant being
        // demonstrated. The behavioural assertions above (local_push,
        // local_pop) are the authoritative check.
        let _ = (push_before, pop_before);
    }

    /// Two distinct paths that happen to share the same length but
    /// diverge in the tail must NOT collide on the fast path (different
    /// tail FrameIds).
    #[test]
    fn test_switch_fast_path_does_not_collide_on_different_tails() {
        let mut lin = SharedLineageSolver::new(make_solver());
        let x = bvconst("test_switch_fast_path_no_collide_x", 16);

        // Two paths of identical length 2, sharing the first frame `a`
        // but diverging in the tail (`c` vs `d`).
        let a = ScopeFrame::new(true, x.bvugt(&z3::ast::BV::from_u64(0, 16)));
        let c = ScopeFrame::new(true, eq_bv_const(&x, 5));
        let d = ScopeFrame::new(true, eq_bv_const(&x, 42));

        let path_ac = vec![a.clone(), c];
        let path_ad = vec![a, d];

        lin.switch_to(&path_ac);

        // Switch to a different path that has the same length but a
        // different tail id. Must NOT fire the fast path; the (pops,
        // pushes) return value of (1, 1) is the race-free behavioural
        // proof that the prefix walk did execute and identified the
        // tail-only divergence. (Global counter delta cannot prove
        // negative "did not fire" because other parallel tests in this
        // module also bump the same atomic.)
        let (pops, pushes) = lin.switch_to(&path_ad);
        assert_eq!(
            (pops, pushes),
            (1, 1),
            "different-tail switch must pop+push the diverging frame"
        );
        assert_eq!(lin.loaded_depth(), 2);
    }
}
