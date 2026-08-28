//! Tests for the ptr-keyed dedup diagnostics in `constraint_ops.rs`
//! (angr-5mnx3.46).
//!
//! The subject is [`SymContext::dedup_backing_of_ptr`] and its `log!` wrapper
//! [`SymContext::debug_verify_dedup_backing`]: the angr-gmad2 detector that
//! decides whether a dedup HIT is a sound true-positive (Z3 hash-consing
//! handed back the ptr of an assertion that is still live) or a STALE-PTR
//! false positive (a freed AST's address got reused, so an *intended*
//! constraint is about to be dropped).
//!
//! These assert on the predicate rather than on the emitted log line:
//! `log::set_logger` is a process-global one-shot singleton that
//! `engine_tests.rs`'s `set_rust_log_level_accepts_levels_and_specs` already
//! claims in this same test binary, so a capture logger installed here would
//! race it non-deterministically — the same reasoning
//! `state/tests/merge_scalars.rs`'s
//! `test_native_resume_stack_diverges_detects_depth_mismatch` records. The
//! wrapper itself gets a smoke call so both of its arms stay compiled and
//! panic-free.
//!
//! Everything here reaches `SymContext`'s private dedup helpers directly:
//! this module is a child of `constraint_ops`, so no visibility has to be
//! widened for the tests.

// `z3::ast::Bool` is neither Send nor Sync, and the shared-assertion vec this
// file stands up mirrors `z3_assertions_shared`'s own `Arc<Vec<Bool>>` shape --
// same opt-out `context_tests/constraints.rs` takes for the same reason.
#![allow(clippy::arc_with_non_send_sync)]

use super::*;
use z3::ast::Ast;

/// Build a 1-bit `x == value` condition plus the exact `z3::ast::Bool` the
/// dedup path would key on, and return it together with that Bool's Z3_ast
/// ptr. Keeping the Bool alive in the caller is what keeps the ptr valid.
fn cond_bool(ctx: &SymContext, name: &str, value: u64) -> (z3::ast::Bool, usize) {
    let x = RustBV::symbolic(ctx, name, 8);
    let cond = x.eq(&RustBV::concrete(u128::from(value), 8), ctx);
    let b = cond.to_z3_bool();
    let ptr = b.get_z3_ast().as_ptr() as usize;
    (b, ptr)
}

/// A dedup HIT on a ptr still held by `local.z3_assertions` is the sound
/// true-positive case — the `debug!` arm of the detector.
#[test]
fn test_dedup_backing_of_ptr_sees_live_local_assertion() {
    let ctx = SymContext::new();
    let (b, ptr) = cond_bool(&ctx, "gmad2_local_x", 5);

    let mut local = ctx.local_constraints.lock();
    assert!(
        !ctx.seed_and_check_z3_dedup(&mut local, &b),
        "first sighting of the ptr is a miss, and pushes onto z3_assertions"
    );
    assert!(
        ctx.seed_and_check_z3_dedup(&mut local, &b),
        "the repeat must be a dedup HIT"
    );

    assert_eq!(
        ctx.dedup_backing_of_ptr(&local, ptr),
        (true, false),
        "the HIT ptr is still in local.z3_assertions -- sound true-positive"
    );
    // Smoke the log wrapper's backed arm.
    ctx.debug_verify_dedup_backing(&local, &b);
}

/// The same HIT is equally sound when the backing assertion has been frozen
/// into `z3_assertions_shared` (what `fork`'s `freeze_into_shared` does) and
/// `local.z3_assertions` no longer holds it.
#[test]
fn test_dedup_backing_of_ptr_sees_frozen_shared_assertion() {
    let ctx = SymContext::new();
    let (b, ptr) = cond_bool(&ctx, "gmad2_shared_x", 7);

    // Stand in for freeze_into_shared: the assertion lives only in shared.
    *ctx.z3_assertions_shared.lock() = Arc::new(vec![b.clone()]);

    let mut local = ctx.local_constraints.lock();
    assert!(
        ctx.seed_and_check_z3_dedup(&mut local, &b),
        "seeding walks shared too, so the shared ptr reads as a HIT"
    );
    assert_eq!(
        ctx.dedup_backing_of_ptr(&local, ptr),
        (false, true),
        "backed by shared alone is still a sound true-positive"
    );
    ctx.debug_verify_dedup_backing(&local, &b);
}

/// The regression the detector exists for: a truncation of `z3_assertions`
/// that forgets to clear `dedup_set_seeded` (the contract documented on
/// `LocalConstraints::dedup_set_seeded`) leaves ptrs in `dedup_set` that no
/// live assertion backs. Dedup then reports a HIT and the *intended*
/// constraint is silently dropped — `dedup_backing_of_ptr` must report
/// `(false, false)` so the `warn!` arm fires.
#[test]
fn test_dedup_backing_of_ptr_flags_stale_ptr_hit() {
    let ctx = SymContext::new();
    let (b, ptr) = cond_bool(&ctx, "gmad2_stale_x", 9);

    let mut local = ctx.local_constraints.lock();
    assert!(!ctx.seed_and_check_z3_dedup(&mut local, &b));
    assert_eq!(local.z3_assertions.len(), 1);

    // Simulate the broken pop(): drop the assertions, keep the seed flag.
    local.z3_assertions.clear();
    assert!(
        local.dedup_set_seeded,
        "the whole point of the scenario is that the stale set is still trusted"
    );

    assert!(
        ctx.seed_and_check_z3_dedup(&mut local, &b),
        "the stale set still reports a HIT -- this is the constraint being dropped"
    );
    assert_eq!(
        ctx.dedup_backing_of_ptr(&local, ptr),
        (false, false),
        "no live assertion backs the HIT ptr -- STALE-PTR false positive"
    );
    // Smoke the log wrapper's unbacked (warn!) arm.
    ctx.debug_verify_dedup_backing(&local, &b);
}

/// A ptr that was never asserted at all is unbacked too — guards against a
/// `dedup_backing_of_ptr` that answered "backed" for anything (e.g. by
/// scanning lengths instead of ptrs).
#[test]
fn test_dedup_backing_of_ptr_unknown_ptr_is_unbacked() {
    let ctx = SymContext::new();
    let (asserted, _) = cond_bool(&ctx, "gmad2_unknown_a", 1);
    let (_other, other_ptr) = cond_bool(&ctx, "gmad2_unknown_b", 2);

    let mut local = ctx.local_constraints.lock();
    assert!(!ctx.seed_and_check_z3_dedup(&mut local, &asserted));

    assert_eq!(
        ctx.dedup_backing_of_ptr(&local, other_ptr),
        (false, false),
        "a ptr from a never-asserted Bool must not read as backed"
    );
}
