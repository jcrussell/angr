//! angr-op0dn.11.3 — acceptance tests for the shared-prefix-aware
//! `SymContext::merge` (`snapshot_fork_ops.rs`).
//!
//! The S5b spike (`merge_shape_spike.rs`) prototyped the CoW-cheap merge in a
//! throwaway `cow_merge` helper and proved feasibility. These tests exercise
//! the SHIPPED `SymContext::merge` and assert its productionized shape:
//!   * on a diamond whose arms share a ptr-equal frozen prefix, the guarded-`Or`
//!     count == divergent-constraint count + 1 (the `Or` of merge flags), with
//!     the common prefix asserted exactly once, unguarded;
//!   * the merged state's SAT verdict and value admissibility are unchanged from
//!     the pre-11.3 behaviour (semantics-preserving);
//!   * when the arms do NOT share a ptr-equal prefix, merge falls back to
//!     guarding every constraint of every arm (total + 1).
//!
//! The guarded count is observed via the `merge_instrument` thread-local
//! counter (`context.rs`), which the production merge increments per guarded
//! `Or` plus the final flag `Or`.

#![cfg(feature = "vex-engine-z3")]

use super::*;
use std::sync::Arc;

const SHARED_CONSTRAINTS: usize = 8;
const DIVERGENT_PER_ARM: usize = 2;
const N_ARMS: usize = 2;

/// Base context with `SHARED_CONSTRAINTS` constraints on `x` (`x > 7`, seeded
/// as 8 distinct assertions so the prefix has measurable length).
fn build_base() -> (SymContext, RustBV) {
    let base = SymContext::new();
    let x = RustBV::symbolic(&base, "x_diamond", 32);
    for i in 0..SHARED_CONSTRAINTS as u128 {
        base.assume_true(&x.ugt(&RustBV::concrete(i, 32), &base));
    }
    (base, x)
}

/// Fork the base into two diamond arms, each with a divergent local suffix.
/// Arm A pins `x` into (150, 250); arm B into (250, 350).
fn build_arms(base: &SymContext, x: &RustBV) -> (SymContext, SymContext) {
    let arm_a = base.fork();
    arm_a.assume_true(&x.ugt(&RustBV::concrete(150, 32), &arm_a));
    arm_a.assume_true(&x.ult(&RustBV::concrete(250, 32), &arm_a));

    let arm_b = base.fork();
    arm_b.assume_true(&x.ugt(&RustBV::concrete(250, 32), &arm_b));
    arm_b.assume_true(&x.ult(&RustBV::concrete(350, 32), &arm_b));

    (arm_a, arm_b)
}

/// Run the SHIPPED merge under the guarded-count instrument.
fn merge_measured(
    arm_a: &SymContext,
    others: &[&SymContext],
    conds: &[RustBV],
) -> (SymContext, u64) {
    merge_instrument::reset();
    let merged = arm_a.merge(others, conds);
    (merged, merge_instrument::emitted())
}

/// ACCEPTANCE: on a diamond with a ptr-equal shared prefix, the production
/// merge guards exactly the divergent (local) constraints + 1 flag `Or`, and
/// the prefix is asserted once (not once per arm).
#[test]
fn test_shared_prefix_merge_guards_divergence_only() {
    let (base, x) = build_base();
    let (arm_a, arm_b) = build_arms(&base, &x);

    // Precondition: siblings really do share the frozen prefix Arc.
    assert!(Arc::ptr_eq(
        &arm_a.z3_assertions_shared.lock(),
        &arm_b.z3_assertions_shared.lock()
    ));

    let f0 = RustBV::symbolic(&base, "merge_flag_0", 1);
    let f1 = RustBV::symbolic(&base, "merge_flag_1", 1);
    let (_merged, guarded) = merge_measured(&arm_a, &[&arm_b], &[f0, f1]);

    // guarded == Σ divergent local + 1 (the flag Or), NOT Σ total + 1.
    let divergent = (N_ARMS * DIVERGENT_PER_ARM) as u64;
    assert_eq!(
        guarded,
        divergent + 1,
        "shared-prefix merge must guard only the divergent suffix + the flag Or"
    );
    // Sanity: the pre-11.3 shape would have been Σ total + 1.
    let total = (N_ARMS * (SHARED_CONSTRAINTS + DIVERGENT_PER_ARM)) as u64;
    assert!(guarded < total + 1, "must beat the total-guard shape");
}

/// ACCEPTANCE: the shared-prefix merge is semantics-preserving — the merged
/// state stays SAT and admits exactly the union of arm values (x=200 ∈ arm A,
/// x=300 ∈ arm B), rejecting values in neither arm (x=100) and values that
/// violate the shared prefix (x=5).
#[test]
fn test_shared_prefix_merge_preserves_semantics() {
    let (base, x) = build_base();
    let (arm_a, arm_b) = build_arms(&base, &x);

    let f0 = RustBV::symbolic(&base, "merge_flag_0", 1);
    let f1 = RustBV::symbolic(&base, "merge_flag_1", 1);
    let merged = arm_a.merge(&[&arm_b], &[f0, f1]);

    assert!(merged.is_sat(), "shared-prefix merge must stay SAT");
    let xm = RustBV::symbolic(&merged, "x_diamond", 32);
    assert!(merged.solution(&xm, 200), "arm A value must be admissible");
    assert!(merged.solution(&xm, 300), "arm B value must be admissible");
    assert!(!merged.solution(&xm, 100), "value in neither arm rejected");
    assert!(
        !merged.solution(&xm, 5),
        "value violating the prefix rejected"
    );
}

/// ACCEPTANCE (fallback): when the arms do NOT share a ptr-equal frozen prefix
/// (independently constructed contexts), merge falls back to guarding every
/// constraint of every arm. Two fresh bases each get their own constraints on
/// disjoint symbols, so no common frozen ancestor exists.
#[test]
fn test_no_shared_prefix_falls_back_to_full_guard() {
    let ctx_a = SymContext::new();
    let a = RustBV::symbolic(&ctx_a, "a_indep", 32);
    ctx_a.assume_true(&a.ugt(&RustBV::concrete(10, 32), &ctx_a));
    ctx_a.assume_true(&a.ult(&RustBV::concrete(20, 32), &ctx_a));

    let ctx_b = SymContext::new();
    let b = RustBV::symbolic(&ctx_b, "b_indep", 32);
    ctx_b.assume_true(&b.ugt(&RustBV::concrete(100, 32), &ctx_b));
    ctx_b.assume_true(&b.ult(&RustBV::concrete(200, 32), &ctx_b));

    // Their frozen prefixes are NOT ptr-equal (both empty Arcs, distinct
    // allocations — no fork tie): the fast path must NOT engage. To make the
    // fallback observable, freeze each context's constraints into `shared` via
    // fork so they carry a non-empty (but distinct) prefix.
    let arm_a = ctx_a.fork();
    let arm_b = ctx_b.fork();
    assert!(!Arc::ptr_eq(
        &arm_a.z3_assertions_shared.lock(),
        &arm_b.z3_assertions_shared.lock()
    ));

    let f0 = RustBV::symbolic(&ctx_a, "mf0", 1);
    let f1 = RustBV::symbolic(&ctx_a, "mf1", 1);
    let (_merged, guarded) = merge_measured(&arm_a, &[&arm_b], &[f0, f1]);

    // No shared prefix → every constraint of every arm is guarded. Each arm
    // froze 2 constraints into `shared` and has 0 local, so total guarded is
    // 2 + 2 = 4, plus the flag Or = 5.
    let total = (2 + 2) as u64 + 1;
    assert_eq!(
        guarded, total,
        "no-shared-prefix merge must guard every constraint of every arm"
    );
}

/// SINGLE-ARM edge: merging a context with itself-only (no others) still asserts
/// the prefix once and guards only the local suffix + flag Or, and stays SAT.
#[test]
fn test_single_arm_merge_shared_prefix() {
    let (base, x) = build_base();
    let arm = base.fork();
    arm.assume_true(&x.ugt(&RustBV::concrete(150, 32), &arm));

    let f0 = RustBV::symbolic(&base, "solo_flag", 1);
    let (merged, guarded) = merge_measured(&arm, &[], &[f0]);

    // 1 divergent local + 1 flag Or.
    assert_eq!(guarded, 2);
    assert!(merged.is_sat());
}

/// angr-ph300.46: a merge of deterministic arms must stay deterministic.
/// `with_timeout` (merge's fresh base) hardcodes both per-lineage flags to
/// false; before the fix the merged context silently reverted to arbitrary-Z3
/// witnesses even when every input was deterministic.
#[test]
fn test_merge_preserves_deterministic_and_shared_lineage_flags() {
    let (base, x) = build_base();
    base.set_deterministic(true);
    base.set_use_shared_lineage_solver(true);

    let arm_a = base.fork();
    arm_a.assume_true(&x.ugt(&RustBV::concrete(150, 32), &arm_a));
    let arm_b = base.fork();
    arm_b.assume_true(&x.ult(&RustBV::concrete(250, 32), &arm_b));

    // Both arms inherited the flags via fork; confirm the merge OR keeps them.
    let f0 = RustBV::symbolic(&base, "det_flag_0", 1);
    let f1 = RustBV::symbolic(&base, "det_flag_1", 1);
    let merged = arm_a.merge(&[&arm_b], &[f0, f1]);
    assert!(
        merged.is_deterministic(),
        "merged context must inherit deterministic witness mode"
    );
    assert!(
        merged.use_shared_lineage_solver(),
        "merged context must inherit the shared-lineage opt-in"
    );
}

/// angr-ph300.46: the OR semantics — a single deterministic arm makes the
/// whole merge deterministic even when the others are not.
#[test]
fn test_merge_flags_or_across_arms() {
    let ctx_a = SymContext::new();
    ctx_a.set_deterministic(true);
    let ctx_b = SymContext::new(); // non-deterministic

    let f0 = RustBV::symbolic(&ctx_a, "or_flag_0", 1);
    let f1 = RustBV::symbolic(&ctx_a, "or_flag_1", 1);
    let merged = ctx_a.merge(&[&ctx_b], &[f0, f1]);
    assert!(
        merged.is_deterministic(),
        "any deterministic arm makes the merge deterministic"
    );
    assert!(
        !merged.use_shared_lineage_solver(),
        "no arm opted into shared lineage → merged stays opted out"
    );
}

/// angr-ph300.46: the per-lineage flags round-trip through a snapshot. A
/// restored deterministic context must stay deterministic.
#[test]
fn test_snapshot_roundtrips_lineage_flags() {
    let ctx = SymContext::new();
    ctx.set_deterministic(true);
    ctx.set_use_shared_lineage_solver(true);
    let x = RustBV::symbolic(&ctx, "snap_x", 32);
    ctx.assume_true(&x.ugt(&RustBV::concrete(5, 32), &ctx));

    let snap = ctx.to_snapshot();
    assert!(snap.deterministic);
    assert!(snap.use_shared_lineage_solver);

    let restored = SymContext::new();
    assert!(!restored.is_deterministic());
    restored.restore_from_snapshot(&snap);
    assert!(
        restored.is_deterministic(),
        "restored context must recover deterministic mode"
    );
    assert!(
        restored.use_shared_lineage_solver(),
        "restored context must recover the shared-lineage opt-in"
    );
}
