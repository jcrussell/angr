#![allow(clippy::arc_with_non_send_sync)]
use super::*;
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;

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
    use super::super::stats::Z3_EXTREMA_MODEL_HIT_COUNT;

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

    let min_val = ctx.min(&x, false);
    let max_val = ctx.max(&x, false);

    let hit_after = Z3_EXTREMA_MODEL_HIT_COUNT.load(Ordering::Relaxed);

    assert_eq!(min_val, Some(11), "min should be 11");
    assert_eq!(max_val, Some(19), "max should be 19");
    // Our 2 calls each had a usable model — should bump HIT by >=2.
    //
    // There is deliberately NO companion `MISS delta == 0` assertion: MISS is
    // the same process-wide counter, so any *other* test in the binary that
    // calls min()/max() without a cached model bumps it concurrently and the
    // exact-equality form fails at random (it did, ~25% of runs). Only >=
    // bounds are sound against a global counter.
    assert!(
        hit_after - hit_before >= 2,
        "expected >=2 extrema cache hits across min+max, got {}",
        hit_after - hit_before
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
    use super::super::stats::Z3_EXTREMA_MODEL_HIT_COUNT;
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
    assert_eq!(
        v & (1u128 << 31),
        0,
        "witness should be signed-non-negative"
    );

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

/// Helper: build a `(Z3AstPtr, RustBV, bool)` tuple from a width-1 cond
/// for use with `add_constraints_raw_batch`. Mirrors what the
/// `RustSolverContext::add_constraints_ast` fast path does with claripy ASTs.
/// The typed `Z3AstPtr` does proper refcounting via `Z3_inc_ref` —
/// no `mem::forget` leak required.
#[cfg(feature = "vex-engine-z3")]
fn batch_entry(cond: &RustBV) -> (Z3AstPtr, RustBV, bool) {
    use z3::ast::Ast;
    debug_assert_eq!(cond.width(), 1);
    let bool_ast = cond.to_z3_bool();
    let ctx = z3::Context::thread_local();
    let ptr = bool_ast.get_z3_ast().as_ptr() as usize;
    // SAFETY: `bool_ast` keeps the AST alive across the inc_ref call;
    // the resulting Z3AstPtr holds its own ref so the AST survives
    // `bool_ast` dropping at end of this function.
    let ast = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) }.expect("non-null Bool AST");
    (ast, cond.clone(), true)
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
    ctx.add_constraints_raw_batch(entries);
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
    ctx.add_constraints_raw_batch(Vec::new());
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
    let new_val = u128::from(first == 0);
    let pinned = RustBV::concrete(new_val, 32);
    let entries = vec![batch_entry(&x.eq(&pinned, &ctx))];
    ctx.add_constraints_raw_batch(entries);
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
    ctx.add_constraints_raw_batch(entries);
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
    ctx.add_constraints_raw_batch(entries);
    let after = ctx.export_z3_assertion_ptrs().len();
    assert_eq!(after - before, 2);
}

/// angr-v5a5 slice 4c.2c: with no lineage attached,
/// `add_constraints_raw_batch` must hit the per-context Z3 solver and
/// leave `scope_path` empty — byte-identical behavior to the pre-slice
/// `self.with_z3_solver(|s| { for c in &constraints { s.assert(c); } })`
/// call. Mirrors `test_add_constraint_raw_none_branch_no_scope_path`
/// for the batched path, with a multi-entry batch.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_add_constraints_raw_batch_none_branch_no_scope_path() {
    let ctx = SymContext::new();
    assert!(ctx.lineage_arc().is_none());
    assert_eq!(ctx.scope_path_len(), 0);

    let x = RustBV::symbolic(&ctx, "x_batch_none", 32);
    let entries = vec![
        batch_entry(&x.ugt(&RustBV::concrete(10, 32), &ctx)),
        batch_entry(&x.ult(&RustBV::concrete(20, 32), &ctx)),
        batch_entry(&x.uge(&RustBV::concrete(15, 32), &ctx)),
    ];
    ctx.add_constraints_raw_batch(entries);

    // None branch must not mint scope frames.
    assert_eq!(
        ctx.scope_path_len(),
        0,
        "None branch must not mint scope frames"
    );

    // All three constraints must be in force on the per-context solver.
    assert!(ctx.solution(&x, 15));
    assert!(ctx.solution(&x, 19));
    assert!(!ctx.solution(&x, 14));
    assert!(!ctx.solution(&x, 20));
}

/// angr-v5a5 slice 4c.2c: with a lineage attached,
/// `add_constraints_raw_batch` mints N fresh `ScopeFrame`s under one
/// `scope_path.lock()` acquisition and routes a single `switch_to` call
/// through the shared solver — the divergent-suffix walk in `switch_to`
/// then asserts each new frame inside its own Z3 push. Mirrors
/// `test_add_constraint_raw_some_branch_appends_scope_frame` for the
/// batched path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_add_constraints_raw_batch_some_branch_appends_scope_frames() {
    use super::super::lineage::SharedLineageSolver;

    let ctx = SymContext::new();
    let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
    ctx.set_lineage_for_testing(Arc::clone(&lin));

    assert_eq!(ctx.scope_path_len(), 0);

    let x = RustBV::symbolic(&ctx, "x_batch_some", 32);
    let entries = vec![
        batch_entry(&x.ugt(&RustBV::concrete(10, 32), &ctx)),
        batch_entry(&x.ult(&RustBV::concrete(20, 32), &ctx)),
        batch_entry(&x.uge(&RustBV::concrete(15, 32), &ctx)),
    ];
    ctx.add_constraints_raw_batch(entries);

    // Some branch must mint exactly N frames on scope_path.
    assert_eq!(
        ctx.scope_path_len(),
        3,
        "Some branch must mint one scope frame per batch entry"
    );

    // switch_to must have pushed all three frames onto the shared
    // solver — loaded_depth equals scope_path's length.
    assert_eq!(
        lin.lock().loaded_depth(),
        3,
        "switch_to must have pushed all batch frames onto the shared solver"
    );
}

/// angr-v5a5 slice 4c.2c: sibling isolation invariant for the batched
/// raw path — the N constraints minted by sibling A's
/// `add_constraints_raw_batch` must NOT be visible when sibling B
/// queries the same shared lineage solver. Mirrors
/// `test_add_constraint_raw_sibling_isolation` with a multi-entry batch.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_add_constraints_raw_batch_sibling_isolation() {
    use super::super::lineage::SharedLineageSolver;

    let sibling_a = SymContext::new();
    let sibling_b = SymContext::new();
    let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
    sibling_a.set_lineage_for_testing(Arc::clone(&lin));
    sibling_b.set_lineage_for_testing(Arc::clone(&lin));

    // Sibling A batch-adds x > 10 ∧ x < 20.
    let x_a = RustBV::symbolic(&sibling_a, "x_batch_sibling", 32);
    let entries = vec![
        batch_entry(&x_a.ugt(&RustBV::concrete(10, 32), &sibling_a)),
        batch_entry(&x_a.ult(&RustBV::concrete(20, 32), &sibling_a)),
    ];
    sibling_a.add_constraints_raw_batch(entries);
    assert_eq!(sibling_a.scope_path_len(), 2);
    assert_eq!(sibling_b.scope_path_len(), 0);

    // Sibling B references the same named symbol but is unconstrained
    // — switch_to to B's empty scope_path must pop both of A's frames.
    let x_b = RustBV::symbolic(&sibling_b, "x_batch_sibling", 32);
    assert!(sibling_b.solution(&x_b, 5));
    assert!(sibling_b.solution(&x_b, 42));

    // A still sees 10 < x < 20.
    assert!(sibling_a.solution(&x_a, 15));
    assert!(!sibling_a.solution(&x_a, 5));
    assert!(!sibling_a.solution(&x_a, 42));
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
    assert!(stats.get("vex_unop_total").copied().unwrap() > base_unop);
    assert!(stats.get("vex_binop_total").copied().unwrap() >= base_binop + 2);
    assert!(stats.get("vex_triop_total").copied().unwrap() > base_triop);
    assert!(stats.get("vex_qop_total").copied().unwrap() > base_qop);
    // Each *_op_<family> got bumped once per record_vex_* call.
    assert!(stats.get("vex_op_arith").copied().unwrap() >= base_arith + 2);
    assert!(stats.get("vex_op_logic").copied().unwrap() > base_logic);
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
    let base_ssym = baseline
        .get("mem_store_symbolic_addr")
        .copied()
        .unwrap_or(0);
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
    assert!(stats.get("mem_store_count").copied().unwrap() > base_store);
    assert!(stats.get("mem_load_bytes").copied().unwrap() >= base_load_bytes + 12);
    assert!(stats.get("mem_store_bytes").copied().unwrap() >= base_store_bytes + 16);
    assert!(stats.get("mem_load_symbolic_addr").copied().unwrap() > base_lsym);
    assert!(stats.get("mem_store_symbolic_addr").copied().unwrap() > base_ssym);
    assert!(stats.get("mem_lazy_page_fault_count").copied().unwrap() > base_fault);
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
    assert!(stats.get("concretize_write_count").copied().unwrap() > base_write);
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
    assert!(stats.get("bvop_reverse_count").copied().unwrap() > base_rev);
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
    assert!(stats.get("bvop_reverse_count").copied().unwrap() > base_rev);
    assert!(stats.get("bvop_concat_count").copied().unwrap() > base_cat);
    assert!(stats.get("bvop_extract_count").copied().unwrap() > base_ext);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_range_seeded_constrained_interval() {
    // range_seeded must recover the true [min, max] of a constrained BV
    // from valid seeds, leaving the Z3 scope stack balanced afterward.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x_range_seeded", 32);
    // 10 <= x <= 200
    ctx.assume_true(&x.uge(&RustBV::concrete(10, 32), &ctx));
    ctx.assume_true(&x.ule(&RustBV::concrete(200, 32), &ctx));

    // Interior seeds (both valid solutions): min searched in [0, 50],
    // max searched in [150, 2^32-1]. Must still find the true bounds.
    assert_eq!(ctx.range_seeded(&x, 50, 150), Some((10, 200)));

    // Seeds exactly at the extrema must also work.
    assert_eq!(ctx.range_seeded(&x, 10, 200), Some((10, 200)));

    // Scope balance: follow-up min/max must still see the same bounds,
    // proving range_seeded left no stray push frames on the solver.
    assert_eq!(ctx.min(&x, false), Some(10));
    assert_eq!(ctx.max(&x, false), Some(200));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_range_seeded_concrete_fast_path() {
    // A concrete BV short-circuits to (v, v) regardless of the seeds and
    // without consulting the solver.
    let ctx = SymContext::new();
    let c = RustBV::concrete(42, 32);
    assert_eq!(ctx.range_seeded(&c, 0, 1000), Some((42, 42)));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_range_seeded_unsat_returns_none() {
    // An UNSAT context returns None before running either binary search.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x_range_unsat", 32);
    ctx.assume_true(&x.ugt(&RustBV::concrete(200, 32), &ctx));
    ctx.assume_true(&x.ult(&RustBV::concrete(10, 32), &ctx));
    assert!(!ctx.is_sat());
    assert_eq!(ctx.range_seeded(&x, 5, 5), None);
}

/// `range_seeded` must carry the same `width > 128` bail-out as `min` / `max`
/// (angr-sqfj8.103). Its bounds live in a `u128`, so above 128 bits the true
/// extrema are unrepresentable and the binary search would report a truncated
/// range with full confidence. 128 itself is still fine.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_range_seeded_above_128_bits_returns_none() {
    for width in [129u32, 192, 256] {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x_range_wide", width);
        // Constrain to a narrow interval whose extrema *are* u128-representable,
        // so a missing guard would return Some((10, 200)) rather than garbage —
        // the guard is about the search space, not the answer.
        ctx.assume_true(&x.uge(&RustBV::concrete(10, width), &ctx));
        ctx.assume_true(&x.ule(&RustBV::concrete(200, width), &ctx));
        assert_eq!(
            ctx.range_seeded(&x, 50, 150),
            None,
            "width {width} must report unknown, matching min/max"
        );
        // Same contract as min/max at this width, and the early return must
        // not have left a stray push frame behind.
        assert_eq!(ctx.min(&x, false), None, "width {width}");
        assert_eq!(ctx.max(&x, false), None, "width {width}");
        assert!(ctx.is_sat(), "width {width} context still usable");
    }

    // The 128-bit boundary itself still searches normally.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x_range_128", 128);
    ctx.assume_true(&x.uge(&RustBV::concrete(10, 128), &ctx));
    ctx.assume_true(&x.ule(&RustBV::concrete(200, 128), &ctx));
    assert_eq!(ctx.range_seeded(&x, 50, 150), Some((10, 200)));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_guards_each_branch_under_its_condition() {
    // Two contexts with disjoint constraints (x==5, x==9). The merged
    // context must admit each branch's value under its own merge flag and
    // reject a value satisfying neither (Or(merge_conditions) forces at
    // least one branch active).
    let s1 = SymContext::new();
    let x1 = RustBV::symbolic(&s1, "x_merge", 32);
    s1.assume_true(&x1.eq(&RustBV::concrete(5, 32), &s1));

    let s2 = SymContext::new();
    // Same name => same Z3 variable, mirroring forked-state identity.
    let x2 = RustBV::symbolic(&s2, "x_merge", 32);
    s2.assume_true(&x2.eq(&RustBV::concrete(9, 32), &s2));

    // One 1-bit merge flag per context (self first, then others).
    let f0 = RustBV::symbolic(&s1, "merge_flag_0", 1);
    let f1 = RustBV::symbolic(&s1, "merge_flag_1", 1);
    let merged = s1.merge(&[&s2], &[f0, f1]);

    let x = RustBV::symbolic(&merged, "x_merge", 32);
    assert!(
        merged.solution(&x, 5),
        "branch s1 (x==5) must be admissible"
    );
    assert!(
        merged.solution(&x, 9),
        "branch s2 (x==9) must be admissible"
    );
    assert!(
        !merged.solution(&x, 7),
        "value in neither branch must be rejected"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
#[should_panic(expected = "merge_conditions must have one entry per context")]
fn test_merge_panics_on_condition_count_mismatch() {
    let s1 = SymContext::new();
    let s2 = SymContext::new();
    // others.len() + 1 == 2, but only one condition is supplied.
    let f0 = RustBV::symbolic(&s1, "merge_flag_bad", 1);
    let _ = s1.merge(&[&s2], &[f0]);
}

/// angr-ue4ro: `eval_many` must read every part off ONE model. Solving each
/// part with a separate `eval` can mix models, so parts tied by a cross-part
/// constraint come back mutually inconsistent.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_many_is_single_model() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let y = RustBV::symbolic(&ctx, "y", 8);

    // Cross-part constraint: x + y == 10 (both parts individually free).
    let sum = x.add(&y, &ctx);
    let ten = RustBV::concrete(10, 8);
    let is_ten = sum.eq(&ten, &ctx);
    ctx.assume_true(&is_ten);

    let vals = ctx
        .eval_many(&[x, y])
        .expect("sat constraint set should yield a model");
    assert_eq!(vals.len(), 2);
    assert_eq!(
        (vals[0] + vals[1]) & 0xff,
        10,
        "parts must satisfy the cross-part constraint (x={}, y={})",
        vals[0],
        vals[1]
    );
}

/// An unsat context yields no model at all — never a partially-filled vector.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_many_unsat_returns_none() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let one = RustBV::concrete(1, 8);
    let two = RustBV::concrete(2, 8);
    let is_one = x.eq(&one, &ctx);
    let is_two = x.eq(&two, &ctx);
    ctx.assume_true(&is_one);
    ctx.assume_true(&is_two);

    assert!(ctx.eval_many(&[x]).is_none());
}

/// angr-op0dn.10.1 (M2.1): `eval_upto` must present the enumerated witnesses in
/// canonical ascending order, so an exhaustive enumeration (n >= #feasible, the
/// `solutions()` pattern) is bit-for-bit reproducible across runs. Without the
/// sort the order is whatever Z3's model-generation happened to produce, which
/// is stable within a process but not something we may rely on.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_upto_canonical_order() {
    // 5 <= x < 10 over 8 bits => exactly {5, 6, 7, 8, 9}.
    let expected: Vec<u128> = vec![5, 6, 7, 8, 9];

    for _ in 0..10 {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x", 8);
        let lo = RustBV::concrete(5, 8);
        let hi = RustBV::concrete(10, 8);
        let ge_lo = x.uge(&lo, &ctx);
        let lt_hi = x.ult(&hi, &ctx);
        ctx.assume_true(&ge_lo);
        ctx.assume_true(&lt_hi);

        // n == #feasible and n > #feasible must both give the identical Vec.
        assert_eq!(ctx.eval_upto(&x, 5), expected);
        assert_eq!(ctx.eval_upto(&x, 16), expected);
    }
}

/// The wide (byte-vector) boundary carries the same guarantee. Every witness is
/// `width` bits wide, so big-endian lexicographic order == numeric order.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_upto_wide_canonical_order() {
    const WIDTH: u32 = 192; // > 128 bits: exercises the genuinely-wide path.
    let byte_len = (WIDTH / 8) as usize;
    let expected: Vec<Vec<u8>> = (0u8..4)
        .map(|v| {
            let mut bytes = vec![0u8; byte_len];
            bytes[byte_len - 1] = v;
            bytes
        })
        .collect();

    for _ in 0..10 {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x", WIDTH);
        let four = RustBV::concrete(4, WIDTH);
        let lt_four = x.ult(&four, &ctx);
        ctx.assume_true(&lt_four);

        assert_eq!(ctx.eval_upto_wide(&x, 4), expected);
        assert_eq!(ctx.eval_upto_wide(&x, 8), expected);
    }
}

/// Strict-deterministic mode (angr-op0dn.10.2): `eval` picks the unsigned
/// minimum of the feasible set, not an arbitrary Z3 model value. Ten fresh
/// contexts must agree, and the value must equal `min`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_deterministic_is_min() {
    for _ in 0..10 {
        let ctx = SymContext::new();
        ctx.set_deterministic(true);
        let x = RustBV::symbolic(&ctx, "x", 8);
        let lo = RustBV::concrete(5, 8);
        let hi = RustBV::concrete(10, 8);
        let ge_lo = x.uge(&lo, &ctx);
        let lt_hi = x.ult(&hi, &ctx);
        ctx.assume_true(&ge_lo);
        ctx.assume_true(&lt_hi);

        // A prior is_sat warms the model cache — under the flag that seed must
        // NOT leak into the witness choice.
        assert!(ctx.is_sat());
        assert_eq!(ctx.eval(&x), Some(5));
        assert_eq!(ctx.eval(&x), ctx.min(&x, false));
    }
}

/// Truncated `eval_upto` under the flag returns the ascending prefix of the
/// sorted feasible set (the hard half of M2.2 — 10.1's post-sort only made the
/// exhaustive case canonical).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_upto_deterministic_truncated_prefix() {
    for _ in 0..10 {
        let ctx = SymContext::new();
        ctx.set_deterministic(true);
        let x = RustBV::symbolic(&ctx, "x", 8);
        let lo = RustBV::concrete(5, 8);
        let hi = RustBV::concrete(10, 8);
        let ge_lo = x.uge(&lo, &ctx);
        let lt_hi = x.ult(&hi, &ctx);
        ctx.assume_true(&ge_lo);
        ctx.assume_true(&lt_hi);

        // Feasible set is {5,6,7,8,9}: truncation takes the smallest k.
        assert_eq!(ctx.eval_upto(&x, 1), vec![5]);
        assert_eq!(ctx.eval_upto(&x, 2), vec![5, 6]);
        assert_eq!(ctx.eval_upto(&x, 5), vec![5, 6, 7, 8, 9]);
        // Exhaustive request stops at the feasible set (no over-enumeration).
        assert_eq!(ctx.eval_upto(&x, 16), vec![5, 6, 7, 8, 9]);
    }
}

/// The ascending walk must terminate cleanly when the feasible set includes the
/// top of the range (the `value == max_val` guard, where `lo + 1` would wrap).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_upto_deterministic_saturating_top() {
    let ctx = SymContext::new();
    ctx.set_deterministic(true);
    let x = RustBV::symbolic(&ctx, "x", 8);
    let bound = RustBV::concrete(0xfe, 8);
    let ge = x.uge(&bound, &ctx);
    ctx.assume_true(&ge);

    assert_eq!(ctx.eval_upto(&x, 4), vec![0xfe, 0xff]);
    assert_eq!(ctx.eval(&x), Some(0xfe));
}

/// Unsat contexts and the flag's default-off state.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_deterministic_flag_default_and_unsat() {
    let ctx = SymContext::new();
    assert!(!ctx.is_deterministic(), "flag must be opt-in");

    ctx.set_deterministic(true);
    let x = RustBV::symbolic(&ctx, "x", 8);
    let three = RustBV::concrete(3, 8);
    let lt = x.ult(&three, &ctx);
    let gt = x.ugt(&three, &ctx);
    ctx.assume_true(&lt);
    ctx.assume_true(&gt);

    assert_eq!(ctx.eval(&x), None);
    assert!(ctx.eval_upto(&x, 4).is_empty());

    // Children inherit the mode so a whole lineage stays canonical.
    let child = ctx.fork();
    assert!(child.is_deterministic());
}

/// Strict-deterministic mode (angr-op0dn.10.7): `eval_many` returns the
/// lexicographic minimum over the parts *in order*, not an arbitrary joint
/// model. This is the path `posix.dumps(0)` on an exported found state takes
/// (byte-`Extract` decomposition → `eval_batch` → `eval_many`), and until 10.7
/// it read whatever assignment Z3 happened to build — which made a partially
/// constrained stdin print a different-but-valid answer on nearly every run.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_many_deterministic_is_lex_min() {
    for _ in 0..10 {
        let ctx = SymContext::new();
        ctx.set_deterministic(true);
        // hi is pinned; lo is free below 4 — so only a canonical rule fixes it.
        let hi = RustBV::symbolic(&ctx, "hi", 8);
        let lo = RustBV::symbolic(&ctx, "lo", 8);
        let seven = RustBV::concrete(7, 8);
        let four = RustBV::concrete(4, 8);
        let hi_eq = hi.eq(&seven, &ctx);
        let lo_lt = lo.ult(&four, &ctx);
        ctx.assume_true(&hi_eq);
        ctx.assume_true(&lo_lt);

        // Warm the model cache first: under the flag its arbitrary witness must
        // not leak into the choice (same guard as `eval`).
        assert!(ctx.is_sat());
        assert_eq!(
            ctx.eval_many(&[hi.clone(), lo.clone(), RustBV::concrete(9, 8)]),
            Some(vec![7, 0, 9]),
        );
    }
}

/// The joint witness must still come off ONE assignment (angr-ue4ro): a
/// cross-part constraint has to hold for the values `eval_many` hands back.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_many_deterministic_respects_cross_part_constraint() {
    let ctx = SymContext::new();
    ctx.set_deterministic(true);
    let a = RustBV::symbolic(&ctx, "a", 8);
    let b = RustBV::symbolic(&ctx, "b", 8);
    let ten = RustBV::concrete(10, 8);
    // a > b, so minimizing `a` first still forces b < a — lex order matters.
    let gt = a.ugt(&b, &ctx);
    let a_lt = a.ult(&ten, &ctx);
    ctx.assume_true(&gt);
    ctx.assume_true(&a_lt);

    let vals = ctx.eval_many(&[a, b]).expect("sat");
    assert_eq!(vals, vec![1, 0], "lex-min under a > b");
}

/// `eval_wide` under the flag returns the unsigned minimum as big-endian bytes,
/// for widths past the 128-bit ceiling where `min` reports None (angr-cxw7).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_wide_deterministic_is_unsigned_min() {
    const WIDTH: u32 = 192;
    let byte_len = (WIDTH / 8) as usize;
    let mut expected = vec![0u8; byte_len];
    expected[byte_len - 1] = 3;

    for _ in 0..5 {
        let ctx = SymContext::new();
        ctx.set_deterministic(true);
        let x = RustBV::symbolic(&ctx, "x", WIDTH);
        let three = RustBV::concrete(3, WIDTH);
        let ge = x.uge(&three, &ctx);
        ctx.assume_true(&ge);

        assert_eq!(ctx.eval_wide(&x), Some(expected.clone()));
    }
}

// ---------------------------------------------------------------------------
// angr-ph300.43: Z3 Unknown (timeout) must not be conflated with Unsat in the
// binary-search extrema or the branch-feasibility checks. A mid-bisection
// timeout used to be swallowed as "nothing at/below mid" (bsearch) or
// "branch infeasible" (check_branch_feasibility), yielding a confidently wrong
// extremum / pruning a feasible branch. The fix propagates Unknown distinctly.
//
// Both tests build a hard-but-satisfiable factoring instance (product of two
// ~63-bit primes) under a 1ms solver budget so every Z3 check on it reliably
// returns Unknown, then assert the conservative outcome.

/// Product of two 63-bit primes — a genuine semiprime whose factorization
/// exists (so priming sat_cache(true) is legitimate) but that Z3 cannot crack
/// within a 1ms budget.
#[cfg(feature = "vex-engine-z3")]
const HARD_SEMIPRIME: u128 = 9_223_372_036_854_775_783u128 * 9_223_372_036_854_775_643u128;

/// Deterministic search budget layered under the 1ms wall-clock timeout — see
/// [`pin_rlimit`]. Sized far above what 1ms of Z3 consumes (so the tests still
/// exercise the *timeout* path they were written for) but far below anything
/// that could run long: worst case the check aborts in well under a second.
#[cfg(feature = "vex-engine-z3")]
const HARD_FACTORING_RLIMIT: u32 = 100_000;

/// Deterministic termination bound for the three hard-factoring timeout tests
/// (angr-x8ocv / angr-zdakq).
///
/// Z3's `timeout` param is enforced by a background `scoped_timer` thread that
/// flips the context's cancel flag. Under a heavily parallel `cargo test
/// --release` run that timer has been observed *not* to fire: a gdb backtrace
/// on a wedged run showed the thread genuinely grinding inside
/// `Z3_solver_check -> smt::context::search -> bounded_search -> propagate`
/// for 20.5h under a 1ms budget. `rlimit` is a *deterministic* budget polled by
/// the search loop itself (`reslimit::inc`) with no thread involved, so it
/// bounds the check regardless of timer behavior.
///
/// Both are kept: `timeout` is what these tests are actually about (they assert
/// the conservative Unknown handling), `rlimit` only guarantees the check
/// terminates so a missed timer degrades to a fast Unknown instead of a hang.
/// Applied *after* the constraints are asserted, because `set_params` on a
/// fresh solver would be clobbered by a later solver rebuild.
#[cfg(feature = "vex-engine-z3")]
fn pin_rlimit(ctx: &SymContext, rlimit: u32) {
    ctx.with_z3_solver(|solver| {
        let mut params = crate::symbolic::solver_build::build_solver_params(ctx.timeout_ms());
        params.set_u32("rlimit", rlimit);
        solver.set_params(&params);
    });
}

/// Guard that [`pin_rlimit`] actually reaches the solver (angr-zdakq). The
/// three hard-factoring tests above cannot detect a silently-ignored `rlimit`
/// param — they would just fall back to the flaky wall-clock timeout, and on
/// the run where that timer misfires the suite wedges again.
///
/// So exercise it on a query that is *easy*: `10 < x < 20` under an rlimit of 1
/// resource unit and no wall-clock timeout. If `rlimit` is honored the check
/// aborts Unknown and `min()` returns None; if it is ignored the check runs to
/// completion and returns `Some(11)` — a fast, non-hanging failure.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pin_rlimit_reaches_the_solver() {
    let ctx = SymContext::with_timeout(u32::MAX);
    let x = RustBV::symbolic(&ctx, "rlimit_probe_x", 32);
    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);
    ctx.assume_true(&x.ugt(&ten, &ctx));
    ctx.assume_true(&x.ult(&twenty, &ctx));
    pin_rlimit(&ctx, 1);
    // Skip min()'s is_sat gate so the bisection checks are what we observe.
    ctx.set_sat_cache(true);
    assert_eq!(
        ctx.min(&x, false),
        None,
        "rlimit=1 must abort every check to Unknown — a Some(_) here means the \
         rlimit param never reached the solver and the hard-factoring timeout \
         tests have lost their deterministic termination bound"
    );
}

/// Assert `x * y == N`, `x > 1`, `y > 1` on a 1ms-budget context, returning the
/// 64-bit factor `x`. The multiply is widened to 128 bits so the product does
/// not overflow.
#[cfg(feature = "vex-engine-z3")]
fn build_hard_factoring(ctx: &SymContext) -> RustBV {
    let x = RustBV::symbolic(ctx, "fac_x", 64);
    let y = RustBV::symbolic(ctx, "fac_y", 64);
    let one = RustBV::concrete(1, 64);
    ctx.assume_true(&x.ugt(&one, ctx));
    ctx.assume_true(&y.ugt(&one, ctx));
    let zx = x.zero_extend(128, ctx);
    let zy = y.zero_extend(128, ctx);
    let prod = zx.mul(&zy, ctx);
    let n = RustBV::concrete(HARD_SEMIPRIME, 128);
    ctx.assume_true(&prod.eq(&n, ctx));
    x
}

/// min() must abort to None on a mid-bisection Z3 timeout rather than return a
/// fabricated extremum. sat_cache is primed true (the constraints ARE
/// satisfiable — a factorization exists) so min() skips its is_sat gate and
/// drives straight into bsearch_min, where every check times out at 1ms.
/// Pre-fix: bsearch swallowed each Unknown as "not <= mid", moved lo up, and
/// returned a confident wrong value. Post-fix: None.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_min_aborts_to_none_on_bisection_timeout() {
    let ctx = SymContext::with_timeout(1);
    let x = build_hard_factoring(&ctx);
    pin_rlimit(&ctx, HARD_FACTORING_RLIMIT);
    // Legitimate: the semiprime has a factorization, so the set is satisfiable.
    ctx.set_sat_cache(true);
    assert_eq!(
        ctx.min(&x, false),
        None,
        "min() must return None on a bisection timeout, not a fabricated extremum"
    );
}

/// max() likewise aborts to None on a bisection timeout (angr-ph300.43).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_max_aborts_to_none_on_bisection_timeout() {
    let ctx = SymContext::with_timeout(1);
    let x = build_hard_factoring(&ctx);
    pin_rlimit(&ctx, HARD_FACTORING_RLIMIT);
    ctx.set_sat_cache(true);
    assert_eq!(
        ctx.max(&x, false),
        None,
        "max() must return None on a bisection timeout, not a fabricated extremum"
    );
}

/// Build a 64-bit `x` whose signed sign-probe (`x <_s 0` / `x >=_s 0`) is as
/// hard as factoring `HARD_SEMIPRIME`, so that probe times out under the pinned
/// rlimit (angr-n0irt.1).
///
/// Encodes `sign(x) == p`, where `p` is the hard predicate `a*b == N` (with
/// `a>1, b>1`) and `sign` is `x <_s 0` when `hard_when_negative`, else
/// `x >=_s 0`. Asserting that sign forces `p == true`, driving the solver into
/// the factorization until it blows the rlimit → Unknown.
///
/// NOTE ON DISCRIMINATION: the rlimit is a *cumulative* per-solver budget, so
/// once the sign probe exhausts it every later check (including the fall-through
/// `bsearch`) also returns Unknown → None. That means these tests pin the
/// *contract* — signed `min`/`max` must return None on a timeout, a path the
/// unsigned bisection-timeout tests never exercised — but they cannot isolate
/// the n0irt.1 fix from the pre-fix code: pre-fix, the probe's Unknown collapsed
/// to `has_(non_)negative=false` and the search fell through to `bsearch`, which
/// then hit the same exhausted budget and *also* returned None. Isolating the
/// fabricated-`Some(0)` would require the probe to time out while a fresh
/// `bsearch` still decides — only reachable via the wall-clock-only timeout,
/// which is deliberately avoided here because its timer is flaky under parallel
/// `cargo test` (see `pin_rlimit`). The fix stays as a defensive correctness
/// alignment with `invariant-z3-unknown-not-unsat` / angr-ph300.43.
#[cfg(feature = "vex-engine-z3")]
fn build_hard_sign_probe(ctx: &SymContext, hard_when_negative: bool) -> RustBV {
    let x = RustBV::symbolic(ctx, "sgn_x", 64);
    let a = RustBV::symbolic(ctx, "sgn_a", 64);
    let b = RustBV::symbolic(ctx, "sgn_b", 64);
    let one = RustBV::concrete(1, 64);
    ctx.assume_true(&a.ugt(&one, ctx));
    ctx.assume_true(&b.ugt(&one, ctx));
    let za = a.zero_extend(128, ctx);
    let zb = b.zero_extend(128, ctx);
    let prod = za.mul(&zb, ctx);
    let n = RustBV::concrete(HARD_SEMIPRIME, 128);
    let p = prod.eq(&n, ctx); // 1-bit: hard-to-decide factoring predicate
    let zero = RustBV::concrete(0, 64);
    let sign = if hard_when_negative {
        x.slt(&zero, ctx) // x <_s 0 — the min() probe
    } else {
        x.sge(&zero, ctx) // x >=_s 0 — the max() probe
    };
    ctx.assume_true(&sign.eq(&p, ctx));
    x
}

/// The `SatOutcome::decided` forcing-function (angr-qwyti.3) must map a Z3
/// `Unknown` to `None` — never fold it into a sat/unsat boolean. This is the
/// single mapping every solver query now routes through, so the mapping itself
/// is pinned here directly (deterministic, no timeout needed), complementing the
/// end-to-end min/max probe tests below that exercise the None-propagation path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_satoutcome_decided_maps_unknown_to_none() {
    use crate::symbolic::solver_build::SatOutcome;
    assert_eq!(z3::SatResult::Sat.decided(), Some(true));
    assert_eq!(z3::SatResult::Unsat.decided(), Some(false));
    assert_eq!(
        z3::SatResult::Unknown.decided(),
        None,
        "Unknown (timeout) must map to None, not a boolean — the angr-ph300.43 / \
         angr-n0irt.1 bug class"
    );
}

/// Signed min() must return None when the sign probe (`x <_s 0`) times out,
/// never a fabricated extremum. The unsigned bisection-timeout tests never
/// exercised `signed=true`, so the MinInit-probe arm was previously uncovered;
/// this pins the None-on-timeout contract for the signed path. See
/// `build_hard_sign_probe` for why this cannot isolate the n0irt.1 fix from the
/// pre-fix behavior (cumulative rlimit makes both return None).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_min_signed_aborts_to_none_when_sign_probe_times_out() {
    let ctx = SymContext::with_timeout(1);
    let x = build_hard_sign_probe(&ctx, true);
    pin_rlimit(&ctx, HARD_FACTORING_RLIMIT);
    // The constraint set is satisfiable (via p == false), so skip the is_sat
    // gate and drive straight into the signed sign probe.
    ctx.set_sat_cache(true);
    assert_eq!(
        ctx.min(&x, true),
        None,
        "signed min() must return None when the x<0 sign probe times out, not \
         collapse Unknown->has_negative=false and fabricate Some(0)"
    );
}

/// Signed max() twin of the above: when the `x >=_s 0` sign probe times out,
/// max() must return None, never a fabricated extremum (angr-n0irt.1). Pins the
/// signed-path None-on-timeout contract; see `build_hard_sign_probe` for the
/// discrimination caveat.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_max_signed_aborts_to_none_when_sign_probe_times_out() {
    let ctx = SymContext::with_timeout(1);
    let x = build_hard_sign_probe(&ctx, false);
    pin_rlimit(&ctx, HARD_FACTORING_RLIMIT);
    ctx.set_sat_cache(true);
    assert_eq!(
        ctx.max(&x, true),
        None,
        "signed max() must return None when the x>=0 sign probe times out, not \
         collapse Unknown->has_non_negative=false and fabricate a value"
    );
}

/// check_branch_feasibility must not prune a branch on a Z3 timeout — only a
/// decided Unsat prunes. The condition `x * y == N` is genuinely feasible but
/// unsolvable at 1ms, so both checks return Unknown. Pre-fix the None-arm
/// first check timed out, was read as "cond infeasible", and returned
/// (false, true) — silently killing the feasible true branch. Post-fix both
/// directions stay live: (true, true).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_check_branch_feasibility_keeps_branch_on_timeout() {
    let ctx = SymContext::with_timeout(1);
    let x = RustBV::symbolic(&ctx, "cbf_x", 64);
    let y = RustBV::symbolic(&ctx, "cbf_y", 64);
    let one = RustBV::concrete(1, 64);
    ctx.assume_true(&x.ugt(&one, &ctx));
    ctx.assume_true(&y.ugt(&one, &ctx));
    let zx = x.zero_extend(128, &ctx);
    let zy = y.zero_extend(128, &ctx);
    let prod = zx.mul(&zy, &ctx);
    let n = RustBV::concrete(HARD_SEMIPRIME, 128);
    // 1-bit condition whose feasibility is as hard as the factoring itself.
    let cond = prod.eq(&n, &ctx);
    pin_rlimit(&ctx, HARD_FACTORING_RLIMIT);
    let (can_true, can_false) = ctx.check_branch_feasibility(&cond);
    assert!(
        can_true,
        "an undecided (timeout) cond must not prune the true branch"
    );
    assert!(
        can_false,
        "an undecided (timeout) cond must not prune the false branch"
    );
}

// angr-ph300.29: the internal Z3 arm for Clz/Ctz/Popcount must build a sound
// encoding tied to the operand, not a shared unconstrained new_const. With the
// old `BV::new_const("clz", w)`, every symbolic clz of a width lowered to ONE
// hash-consed variable, so distinct operands' clz aliased.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_clz_distinct_operands_not_aliased() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "clz_alias_x", 32);
    let y = RustBV::symbolic(&ctx, "clz_alias_y", 32);

    let zero = RustBV::concrete(0, 32);
    let five = RustBV::concrete(5, 32);
    // clz(x) == 0  (top bit of x set)  &&  clz(y) == 5
    ctx.assume_true(&x.clz(&ctx).eq(&zero, &ctx));
    ctx.assume_true(&y.clz(&ctx).eq(&five, &ctx));

    // With aliased consts this is UNSAT (one var can't be both 0 and 5); with
    // the sound operand-tied encoding it is clearly satisfiable.
    assert!(
        ctx.is_sat(),
        "clz(x)==0 && clz(y)==5 must be SAT — distinct operands must not alias"
    );
}

// angr-ph300.29: Rust-side eval through the internal Z3 arm must respect the
// clz constraint on the operand. With the shared new_const, eval(x) was
// unconstrained by `clz(x)==k` and could return a value whose real leading
// zero count differs from k.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_clz_constraint_binds_operand_eval() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "clz_bind_x", 32);
    let zero = RustBV::concrete(0, 32);
    // clz(x) == 0  =>  x's MSB is set  =>  x >= 0x8000_0000.
    ctx.assume_true(&x.clz(&ctx).eq(&zero, &ctx));

    let v = ctx.eval(&x).expect("x must be evaluable");
    assert_eq!(
        (v as u32).leading_zeros(),
        0,
        "eval(x) under clz(x)==0 must have zero leading zeros, got {v:#x}"
    );
}

// angr-ph300.29: symbolic popcount must equal the concrete population count
// when the operand is pinned, exercising the sum-of-bits Z3 encoding.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_popcount_matches_concrete_when_operand_pinned() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "popcnt_x", 32);
    let val = RustBV::concrete(0xF0F0_00FF, 32);
    ctx.assume_true(&x.eq(&val, &ctx));

    let pc = ctx
        .eval(&x.popcount(&ctx))
        .expect("popcount must be evaluable");
    assert_eq!(
        pc,
        0xF0F0_00FFu32.count_ones() as u128,
        "symbolic popcount of a pinned operand must match concrete count_ones"
    );
}
