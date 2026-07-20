#![allow(clippy::arc_with_non_send_sync)]
use super::*;
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;

/// Serializes the two tests that delta-assert the process-global
/// `ADD_CONSTRAINT_RAW_DEDUP_HIT_COUNT`. Under the default parallel test
/// runner their measurement windows can interleave and inflate the observed
/// delta (a sibling test's dedup hit lands between `hits_before` and the
/// assertion). Holding this lock across each window makes the deltas exact.
/// Poison-tolerant: a panic in one test must not cascade into the other.
#[cfg(feature = "vex-engine-z3")]
static DEDUP_HIT_COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    let mk_frame =
        |is_true: bool, val: u64| ScopeFrame::new(is_true, bv_x.eq(z3::ast::BV::from_u64(val, 8)));

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

    let _serial = DEDUP_HIT_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let _serial = DEDUP_HIT_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        "None branch unsat_core must include both tracker indices; got {core:?}"
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

/// angr-op0dn.14.2: `unsat_core_assumed` rebuilds a tracked solver from the
/// assumed-constraint IR at query time, so constraints the engine added
/// UNTRACKED (the `assume_true`/`assume_false` fork-guard path) are still
/// nameable. The returned indices index `get_assumed_constraints()`, which is
/// the list `state.solver.constraints` exports.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_unsat_core_assumed_names_untracked_engine_constraints() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "test_uca_x", 8);
    let five = RustBV::concrete(5, 8);
    let six = RustBV::concrete(6, 8);
    let seven = RustBV::concrete(7, 8);

    // Added via the engine's untracked path — no assumption literals.
    ctx.assume_true(&x.eq(&five, &ctx));
    ctx.assume_true(&x.eq(&six, &ctx));
    ctx.assume_false(&x.eq(&seven, &ctx));

    assert_eq!(ctx.get_assumed_constraints().len(), 3);
    assert!(!ctx.is_sat());
    // The live solver has no trackers, so the eager API reports nothing...
    assert!(ctx.unsat_core().is_empty());
    // ...but the on-demand core blames the contradicting pair (0, 1) and
    // leaves the innocent x != 7 (index 2) out.
    let core = ctx.unsat_core_assumed(&[]);
    assert!(
        core.contains(&0) && core.contains(&1),
        "expected indices 0 and 1 in core, got {core:?}"
    );
    assert!(!core.contains(&2), "innocent constraint blamed: {core:?}");
}

/// angr-op0dn.14.2: a constraint imported from Python lands in BOTH constraint
/// lists — `_add_constraints_to_state`'s Z3-ptr fast path calls
/// `add_constraint_raw` (which logs it as a residual `non_bv_assertion`) and
/// then pushes an `assumed` pair for the same constraint. `unsat_core_assumed`
/// must guard such a constraint by its assumption literal ONLY: asserting the
/// residual copy unguarded as well pins it outside the literals, so an
/// all-Python contradiction comes back Unsat with an EMPTY core.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_unsat_core_assumed_ignores_residual_copy_of_assumed_constraint() {
    use z3::ast::Ast;

    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "test_uca_dup_x", 32);
    let z3_ctx = z3::Context::thread_local();

    // Mirror the Python import path for `x > 10` and `x < 5`: raw Z3 assert
    // plus an assumed pair for the very same constraint.
    for bound in [(10u128, true), (5, false)] {
        let (val, greater) = bound;
        let rhs = RustBV::concrete(val, 32);
        let guard = if greater {
            x.ugt(&rhs, &ctx)
        } else {
            x.ult(&rhs, &ctx)
        };
        let raw = guard.to_z3_bool();
        let ptr =
            unsafe { Z3AstPtr::from_borrowed_raw(&z3_ctx, raw.get_z3_ast().as_ptr() as usize) }
                .expect("Bool must yield a Z3AstPtr");
        ctx.add_constraint_raw(ptr);
        ctx.assumed_constraints_push(guard, true);
    }

    assert_eq!(ctx.get_assumed_constraints().len(), 2);
    assert!(!ctx.is_sat());
    let core = ctx.unsat_core_assumed(&[]);
    assert_eq!(
        core,
        vec![0, 1],
        "both Python-imported constraints must be nameable, got {core:?}"
    );
}

/// angr-ph300.41: a bare `pop()` must truncate the local `z3_assertions`
/// log back to its pre-`push()` length. Before the fix, `pop()` popped the
/// Z3 frame but left the log intact, so the stale ptr dedup-suppressed a
/// re-assert and `fork()` replayed the popped constraint as a permanent
/// assert.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bare_pop_truncates_local_z3_assertions_log() {
    let ctx = SymContext::new();
    let z = RustBV::symbolic(&ctx, "test_bare_pop_trunc_z", 32);
    let twenty = RustBV::concrete(20, 32);

    let before = ctx.local_z3_assertions_len();
    ctx.push();
    ctx.assume_true(&z.eq(&twenty, &ctx));
    assert!(
        ctx.local_z3_assertions_len() > before,
        "assume_true inside the scope must grow the local z3_assertions log"
    );
    ctx.pop();
    assert_eq!(
        ctx.local_z3_assertions_len(),
        before,
        "bare pop must truncate the local z3_assertions log to its pre-push length"
    );
}

/// angr-ph300.42: forking after `push(); add(z==20); pop()` must not
/// resurrect `z == 20` in the child — the popped constraint was discarded,
/// so the child is free to assert `z != 20` and stay SAT. Before the fix,
/// the stale `z3_assertions` entry was frozen into the child on fork and
/// replayed as a permanent assert, pruning the feasible `z != 20` path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_after_bare_pop_drops_popped_constraint() {
    let ctx = SymContext::new();
    let z = RustBV::symbolic(&ctx, "test_fork_after_pop_z", 32);
    let twenty = RustBV::concrete(20, 32);

    ctx.push();
    ctx.assume_true(&z.eq(&twenty, &ctx)); // z == 20, inside the scope
    assert!(ctx.is_sat());
    ctx.pop(); // discard z == 20

    let child = ctx.fork();
    child.assume_false(&z.eq(&twenty, &child)); // z != 20
    assert!(
        child.is_sat(),
        "popped z == 20 must not resurrect in the fork; z != 20 is feasible"
    );

    // The parent stays consistent too: after the pop, z is unconstrained.
    ctx.assume_false(&z.eq(&twenty, &ctx));
    assert!(
        ctx.is_sat(),
        "parent must also be free of the popped z == 20"
    );
}
