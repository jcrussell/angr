//! Assumption-based shared-lineage Z3 solver — Alternative (d) spike for
//! [`angr-3ms1`] (v5a5 follow-up).
//!
//! Where [`super::lineage::SharedLineageSolver`] uses push/pop to scope
//! per-state constraints, this variant asserts every constraint at the
//! solver base as `(tag => assertion)` where `tag` is a per-frame fresh
//! [`Bool`]. Each state carries a `path` of these frames. A query runs
//! `solver.check_assumptions(&state.tags)`, which tells Z3 to consider
//! only the assertions whose tags are in the assumption list.
//!
//! ## Why this might beat push/pop on BFS-thrash workloads
//!
//! The push/pop architecture suffers two costs on BFS-style cross-state
//! query interleaving (see `v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental`):
//!
//! 1. **Push/pop overhead** — every state switch pops the divergent
//!    suffix off the loaded path and pushes the new tail.
//! 2. **Learned-clause loss** — clauses derived inside a pushed scope
//!    are discarded on pop, since they may depend on assertions in the
//!    popped frame.
//!
//! Assumption-based solving sidesteps both: assertions stay permanent at
//! the base, so Z3 retains learned clauses across queries. State switch
//! is O(1) (no Z3 operation at all). Per-query cost gains a small
//! assumption-vector argument.
//!
//! ## Scope of this module
//!
//! This is the **spike skeleton** (angr-3ms1, 2026-05-23). It is NOT
//! wired into [`super::context::SymContext`]. The fork-time integration
//! would land in a follow-up bead, gated on positive microbench results
//! against the push/pop variant.
//!
//! See `benches/vex_engine.rs::bench_lineage_assumption_vs_push_pop` for
//! the head-to-head measurement.

#![cfg(feature = "vex-engine-z3")]

use std::collections::HashSet;

use z3::ast::Bool;

use super::lineage::{FrameId, mint_frame_id};

/// One constraint frame on a state's path, tagged with a fresh [`Bool`].
///
/// The `tag` is the assumption Z3 uses to gate the constraint: the solver
/// holds `(tag => z3_assertion)` permanently at the base; supplying `tag`
/// in `check_assumptions` activates the constraint for that one query.
#[derive(Clone)]
pub struct AssumptionFrame {
    pub id: FrameId,
    pub tag: Bool,
    pub z3_assertion: Bool,
}

impl AssumptionFrame {
    /// Construct a fresh frame with a unique [`FrameId`] and a fresh
    /// tag [`Bool`] named after the id.
    pub fn new(z3_assertion: Bool) -> Self {
        let id = mint_frame_id();
        let tag = Bool::new_const(format!("lin_assume_tag_{}", id));
        AssumptionFrame {
            id,
            tag,
            z3_assertion,
        }
    }
}

/// Ordered sequence of [`AssumptionFrame`]s representing one state's
/// accumulated path constraints. Mirrors `super::lineage::ScopePath` in
/// shape — only the per-frame data differs.
pub type AssumptionPath = Vec<AssumptionFrame>;

/// Z3 solver shared by all states in one lineage. Per-state constraints
/// are asserted exactly once at the base as `(tag => constraint)`. A
/// state queries via [`with_solver`](Self::with_solver), which passes
/// the state's tag list as Z3 assumptions.
pub struct SharedLineageSolverAssumptions {
    z3: z3::Solver,
    /// Frame ids already asserted at the base. Ensures each
    /// `(tag => assertion)` is asserted once — additional states sharing
    /// the same frame (e.g., siblings inheriting a parent's constraint)
    /// will encounter the id already present and skip.
    asserted_frames: HashSet<FrameId>,
}

impl SharedLineageSolverAssumptions {
    pub fn new(z3: z3::Solver) -> Self {
        SharedLineageSolverAssumptions {
            z3,
            asserted_frames: HashSet::new(),
        }
    }

    /// Assert a base constraint at scope 0 (unconditional — no tag).
    pub fn assert_base(&self, assertion: &Bool) {
        self.z3.assert(assertion);
    }

    /// Number of distinct frames asserted at the base. For tests.
    pub fn asserted_frame_count(&self) -> usize {
        self.asserted_frames.len()
    }

    /// Ensure every frame in `path` has its `(tag => assertion)`
    /// implication asserted at the base. Idempotent on FrameId — calling
    /// twice with the same frame asserts only once.
    pub fn ensure_asserted(&mut self, path: &[AssumptionFrame]) {
        for f in path {
            if self.asserted_frames.insert(f.id) {
                let implication = f.tag.implies(&f.z3_assertion);
                self.z3.assert(&implication);
            }
        }
    }

    /// Activate `path`'s tags and run a closure against the solver.
    ///
    /// The closure receives the inner [`z3::Solver`] and the slice of
    /// tags for the calling state. Callers wanting to invoke
    /// [`z3::Solver::check_assumptions`] should pass the slice directly;
    /// callers wanting to mix in extra one-shot assumptions (e.g. a
    /// branch-feasibility check) can extend the slice locally.
    pub fn with_solver<R>(
        &mut self,
        path: &[AssumptionFrame],
        f: impl FnOnce(&z3::Solver, &[Bool]) -> R,
    ) -> R {
        self.ensure_asserted(path);
        let tags: Vec<Bool> = path.iter().map(|f| f.tag.clone()).collect();
        f(&self.z3, &tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solver() -> z3::Solver {
        z3::Solver::new()
    }

    fn bvconst(name: &str, width: u32) -> z3::ast::BV {
        z3::ast::BV::new_const(name, width)
    }

    fn eq_bv_const(bv: &z3::ast::BV, value: u64) -> Bool {
        bv._eq(&z3::ast::BV::from_u64(value, bv.get_size()))
    }

    /// Single-state happy path: one constraint, query with the right tag
    /// yields SAT and the model satisfies the constraint.
    #[test]
    fn test_single_state_sat() {
        let mut lin = SharedLineageSolverAssumptions::new(make_solver());
        let x = bvconst("test_single_state_sat_x", 32);
        let path = vec![AssumptionFrame::new(eq_bv_const(&x, 7))];

        let val = lin.with_solver(&path, |solver, tags| {
            assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            let m = solver.get_model().expect("model after Sat");
            m.eval(&x, true).and_then(|bv| bv.as_u64())
        });
        assert_eq!(val, Some(7));
    }

    /// Sibling isolation: two states share a parent constraint and each
    /// adds a different leaf. Querying one state with the other's leaf
    /// tag activates the wrong constraint.
    #[test]
    fn test_sibling_isolation() {
        let mut lin = SharedLineageSolverAssumptions::new(make_solver());
        let x = bvconst("test_sibling_isolation_x", 32);

        let shared = AssumptionFrame::new(x.bvugt(&z3::ast::BV::from_u64(0, 32)));
        let leaf_a = AssumptionFrame::new(eq_bv_const(&x, 5));
        let leaf_b = AssumptionFrame::new(eq_bv_const(&x, 42));

        let path_a = vec![shared.clone(), leaf_a];
        let path_b = vec![shared, leaf_b];

        // Query state A — expect x = 5.
        let val_a = lin.with_solver(&path_a, |solver, tags| {
            assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            solver
                .get_model()
                .and_then(|m| m.eval(&x, true).and_then(|bv| bv.as_u64()))
        });
        assert_eq!(val_a, Some(5));

        // Query state B — expect x = 42. The leaf_a tag is NOT in B's
        // path, so its constraint stays dormant.
        let val_b = lin.with_solver(&path_b, |solver, tags| {
            assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            solver
                .get_model()
                .and_then(|m| m.eval(&x, true).and_then(|bv| bv.as_u64()))
        });
        assert_eq!(val_b, Some(42));
    }

    /// Each frame is asserted at the base exactly once. Re-calling
    /// ensure_asserted with the same path leaves the count unchanged.
    #[test]
    fn test_idempotent_ensure_asserted() {
        let mut lin = SharedLineageSolverAssumptions::new(make_solver());
        let x = bvconst("test_idempotent_x", 16);

        let path = vec![
            AssumptionFrame::new(x.bvugt(&z3::ast::BV::from_u64(0, 16))),
            AssumptionFrame::new(x.bvult(&z3::ast::BV::from_u64(100, 16))),
        ];

        lin.ensure_asserted(&path);
        assert_eq!(lin.asserted_frame_count(), 2);
        lin.ensure_asserted(&path);
        assert_eq!(lin.asserted_frame_count(), 2);

        // A new path sharing one frame and adding one new frame bumps
        // the count by exactly one.
        let path_plus = vec![
            path[0].clone(),
            AssumptionFrame::new(eq_bv_const(&x, 5)),
        ];
        lin.ensure_asserted(&path_plus);
        assert_eq!(lin.asserted_frame_count(), 3);
    }

    /// UNSAT example: x == 5 and x == 42 together are UNSAT. The state
    /// with both leaves' tags active reports UNSAT.
    #[test]
    fn test_unsat_when_contradictory_tags_active() {
        let mut lin = SharedLineageSolverAssumptions::new(make_solver());
        let x = bvconst("test_unsat_x", 8);

        let f5 = AssumptionFrame::new(eq_bv_const(&x, 5));
        let f42 = AssumptionFrame::new(eq_bv_const(&x, 42));

        // Assert both at the base via ensure_asserted on each, then
        // query with BOTH tags. The state-wise path itself wouldn't
        // naturally hold both; the test just exercises that the tag
        // mechanism activates them.
        let path_both = vec![f5, f42];
        let res = lin.with_solver(&path_both, |solver, tags| {
            solver.check_assumptions(tags)
        });
        assert_eq!(res, z3::SatResult::Unsat);

        // With no tags at all, the empty assumption set leaves both
        // constraints dormant — solver should be SAT (only the trivially
        // true implications are active).
        let res_empty = lin.with_solver(&[], |solver, tags| {
            assert!(tags.is_empty());
            solver.check_assumptions(tags)
        });
        assert_eq!(res_empty, z3::SatResult::Sat);
    }

    /// Base assertion persists alongside per-frame implications.
    #[test]
    fn test_base_assertion_holds() {
        let mut lin = SharedLineageSolverAssumptions::new(make_solver());
        let x = bvconst("test_base_assertion_x", 8);

        // Base: x > 10
        lin.assert_base(&x.bvugt(&z3::ast::BV::from_u64(10, 8)));

        // Frame: x < 20
        let path = vec![AssumptionFrame::new(
            x.bvult(&z3::ast::BV::from_u64(20, 8)),
        )];

        let val = lin.with_solver(&path, |solver, tags| {
            assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            solver
                .get_model()
                .and_then(|m| m.eval(&x, true).and_then(|bv| bv.as_u64()))
        });
        let val = val.expect("model present");
        assert!(val > 10 && val < 20, "got {val}, expected 11..=19");

        // Empty path — base alone — still SAT and still > 10.
        let val_base_only = lin.with_solver(&[], |solver, tags| {
            assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            solver
                .get_model()
                .and_then(|m| m.eval(&x, true).and_then(|bv| bv.as_u64()))
        });
        let v = val_base_only.expect("model present");
        assert!(v > 10, "base x > 10 must hold without any tags, got {v}");
    }

    /// Switching between two sibling paths repeatedly is the BFS-thrash
    /// motif. Each switch is a single check_assumptions call; assertion
    /// table never grows beyond the union of the two paths.
    #[test]
    fn test_bfs_thrash_does_not_grow_assertions() {
        let mut lin = SharedLineageSolverAssumptions::new(make_solver());
        let x = bvconst("test_bfs_thrash_x", 16);

        let shared = AssumptionFrame::new(x.bvugt(&z3::ast::BV::from_u64(0, 16)));
        let leaf_a = AssumptionFrame::new(eq_bv_const(&x, 5));
        let leaf_b = AssumptionFrame::new(eq_bv_const(&x, 42));

        let path_a = vec![shared.clone(), leaf_a];
        let path_b = vec![shared, leaf_b];

        for _ in 0..50 {
            lin.with_solver(&path_a, |solver, tags| {
                assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            });
            lin.with_solver(&path_b, |solver, tags| {
                assert_eq!(solver.check_assumptions(tags), z3::SatResult::Sat);
            });
        }

        // 3 distinct frames — shared, leaf_a, leaf_b. Never grows
        // beyond the union, regardless of how many switches happen.
        assert_eq!(lin.asserted_frame_count(), 3);
    }
}
