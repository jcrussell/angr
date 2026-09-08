//! S5b — constraint-merge cost-shape spike (angr-op0dn.11.1.2).
//!
//! Contingent moonshot spike (parent angr-op0dn.11.1, human-GO 2026-07-15,
//! decoupled from S6). Prototype/measurement code — may be throwaway; it exists
//! to answer the M3 merge GO/NO-GO's *constraint* cost-shape question and to
//! prototype the CoW-cheap merge, not to ship an impl (that is
//! angr-op0dn.11.2.x). Sibling of the memory-side spike
//! `memory/tests/merge_cost_shape.rs` (S5a).
//!
//! Question: on a diamond CFG, does `SymContext::merge`
//! (`snapshot_fork_ops.rs::merge`) cost scale with the *divergence* between the
//! arms, or with their *total* constraint count?
//!
//! Answer, established here against the merge as it stood in 2026-07 — the
//! rewrite this spike argued for has since SHIPPED, see "Status" below:
//!   * That merge guarded EVERY constraint of EVERY arm: it walked
//!     `shared.iter().chain(ctx_local.z3_assertions.iter())` and emitted one
//!     `Or(not merge_cond, assertion)` per constraint. So the guarded-`Or`
//!     count was the TOTAL constraint count (Σ over arms of shared+local),
//!     even though every arm's `shared` prefix is byte-for-byte identical.
//!   * The primitive for a divergence-proportional merge ALREADY SHIPS: the
//!     fork-freeze invariant. `fork()` drains self's local constraints
//!     into `z3_assertions_shared` via `freeze_into_shared` and then hands the
//!     SAME `Arc<Vec<Bool>>` to the child, so two arms forked from one base
//!     hold a **ptr-equal** frozen prefix Arc all the way to the merge point;
//!     each arm's post-fork divergence lives only in `local.z3_assertions`.
//!   * Because every diamond arm descends from the fork point, the shared
//!     prefix holds on EVERY path reaching the merge — so it can be asserted
//!     ONCE, UNGUARDED, guarding only the local (divergent) suffixes. The
//!     prototype below does exactly that and emits guarded assertions ==
//!     divergent-constraint count + 1 `Or` of the merge flags, while producing
//!     the identical SAT/eval answers as the production merge.
//!
//! Soundness input for angr-op0dn.11.3's kill clause: the unguarded-prefix
//! rewrite is sound ONLY because `local.z3_assertions` is strictly ADDITIVE to
//! the frozen `shared` prefix — a fork never lets an arm retract or shadow a
//! shared constraint (the Arc is frozen; new constraints only append to local).
//! `test_local_never_shadows_shared` checks that invariant directly.
//!
//! ## Status: the prototyped rewrite shipped (angr-op0dn.11.3)
//! `SymContext::merge` now takes exactly this shape: when every input arm's
//! `z3_assertions_shared` Arc is ptr-equal it asserts that prefix once,
//! UNGUARDED, and guards only each arm's divergent local suffix, falling back
//! to guard-every-constraint when the arms share no frozen ancestor.
//! `context_tests/merge_prefix.rs::test_shared_prefix_merge_guards_divergence_only`
//! pins the live count at divergent+1 rather than total+1.
//!
//! So this module is now a historical record: the pre-11.3 baseline shape, the
//! feasibility/soundness argument for replacing it, and a standalone
//! reimplementation (`cow_merge`) kept as an independent oracle that the
//! shipped merge is checked against
//! (`test_cow_merge_matches_production_semantics`). It is NOT a description of
//! work still to be done.
//!
//! ## Fidelity gaps NOT covered by this constraint-only prototype
//! (enumerated per the S5b acceptance; these were the M3-2b impl surface, not
//! feasibility blockers — the first two are settled by the shipped merge):
//!   * `assumed` export pairs — the merge marks itself non-reconstructible
//!     (`assume_class_reconstructible = false`) so `to_snapshot` dumps the full
//!     solver. The shipped unguarded-prefix path keeps that flag; the prefix
//!     pairs are still export-only and not re-asserted on restore.
//!   * `non_bv_assertions` residual log — the guarded `Or`s have no RustBV form
//!     (angr-t3l5o residual sink #3); the unguarded prefix DOES have one and
//!     could in principle stay reconstructible, but the shipped merge records
//!     it in the residual log too, so the whole merged context stays
//!     full-solver on snapshot.
//!   * Memory-side fidelity (un-merged `multi_objects`, concatenated
//!     `pending_writes`, longest-stdout heuristic) is the S5a/memory spike's
//!     domain, not the constraint solver's.

#![cfg(feature = "vex-engine-z3")]

use super::*;
use std::sync::Arc;

/// Shared pre-header constraints seeded on the base before the fork. Every arm
/// inherits these as a byte-identical frozen prefix.
const SHARED_CONSTRAINTS: usize = 8;
/// Divergent local constraints each arm adds after the fork.
const DIVERGENT_PER_ARM: usize = 2;
const N_ARMS: usize = 2;

/// Build the diamond pre-header: a base context with `SHARED_CONSTRAINTS`
/// constraints on `x` (`x u> 0 .. x u> 7`, i.e. `x > 7`, but 8 distinct
/// assertions so the prefix has a measurable length).
fn build_base() -> (SymContext, RustBV) {
    let base = SymContext::new();
    let x = RustBV::symbolic(&base, "x_diamond", 32);
    for i in 0..SHARED_CONSTRAINTS as u128 {
        base.assume_true(&x.ugt(&RustBV::concrete(i, 32), &base));
    }
    (base, x)
}

/// Fork the base into the two diamond arms and give each its divergent local
/// suffix. Arm A pins `x` into (150, 250); arm B into (250, 350). Both ranges
/// are compatible with the shared prefix (`x > 7`) and mutually disjoint.
fn build_arms(base: &SymContext, x: &RustBV) -> (SymContext, SymContext) {
    let arm_a = base.fork();
    arm_a.assume_true(&x.ugt(&RustBV::concrete(150, 32), &arm_a));
    arm_a.assume_true(&x.ult(&RustBV::concrete(250, 32), &arm_a));

    let arm_b = base.fork();
    arm_b.assume_true(&x.ugt(&RustBV::concrete(250, 32), &arm_b));
    arm_b.assume_true(&x.ult(&RustBV::concrete(350, 32), &arm_b));

    (arm_a, arm_b)
}

/// Per-arm constraint counts as (shared_prefix_len, local_len).
fn arm_shape(arm: &SymContext) -> (usize, usize) {
    let shared = arm.z3_assertions_shared.lock().len();
    let local = arm.local_constraints.lock().z3_assertions.len();
    (shared, local)
}

/// Cost of the *pre-angr-op0dn.11.3* merge shape: guard EVERY constraint of
/// EVERY arm. Kept as the baseline the shipped shared-prefix merge is measured
/// against — it is arithmetic over `arm_shape`, not a call into `merge`.
/// Returns the guarded-`Or` count (excludes the final `Or` of merge flags).
fn baseline_guarded_count(arms: &[&SymContext]) -> u64 {
    arms.iter()
        .map(|a| {
            let (shared, local) = arm_shape(a);
            (shared + local) as u64
        })
        .sum()
}

/// Cost of the *CoW-aware* merge shape: the shared prefix is asserted ONCE
/// unguarded, so only the divergent local suffixes are guarded. Returns the
/// guarded-`Or` count (excludes the final `Or` of merge flags).
fn cow_guarded_count(arms: &[&SymContext]) -> u64 {
    arms.iter().map(|a| arm_shape(a).1 as u64).sum()
}

/// Build a CoW-aware merged context: assert the (ptr-shared) prefix ONCE
/// unguarded, guard only each arm's local suffix, then `Or` the merge flags.
/// Returns `(merged, guarded_count)` where `guarded_count` is the number of
/// guarded `Or`s emitted (NOT counting the flag `Or`), i.e. the divergent
/// constraint count.
fn cow_merge(arms: &[&SymContext], merge_conditions: &[RustBV]) -> (SymContext, u64) {
    assert_eq!(arms.len(), merge_conditions.len());
    let merged = SymContext::new();
    let mut guarded = 0u64;

    // The frozen prefix is ptr-equal across all sibling arms (fork-freeze
    // invariant), so any arm's `shared` vector is THE common prefix. Assert it
    // once, unguarded — sound because every diamond arm descends from the fork
    // point, so the prefix holds on every path reaching the merge.
    let prefix = Arc::clone(&arms[0].z3_assertions_shared.lock());
    for assertion in prefix.iter() {
        merged.add_constraint(assertion.clone());
    }

    // Guard ONLY the divergent local suffixes.
    let mut flags = Vec::with_capacity(merge_conditions.len());
    for (arm, cond) in arms.iter().zip(merge_conditions.iter()) {
        let cond_bool = cond.to_z3_bool();
        let not_cond = cond_bool.not();
        flags.push(cond_bool);
        let local = arm.local_constraints.lock();
        for assertion in local.z3_assertions.iter() {
            let g = z3::ast::Bool::or(&[&not_cond, assertion]);
            merged.add_constraint(g);
            guarded += 1;
        }
    }

    // At least one merge path must be active.
    let flag_refs: Vec<&z3::ast::Bool> = flags.iter().collect();
    merged.add_constraint(z3::ast::Bool::or(&flag_refs));

    (merged, guarded)
}

/// MEASUREMENT: the pre-11.3 merge shape guarded the TOTAL constraint count;
/// the CoW shape (now shipped) guards only the divergent count. On this
/// diamond that is a (2*(8+2))=20 vs (2*2)=4 = 5x reduction, and — the key
/// claim — the CoW count equals the divergent-constraint count exactly.
#[test]
fn test_merge_guarded_count_is_divergence_proportional() {
    let (base, x) = build_base();
    let (arm_a, arm_b) = build_arms(&base, &x);
    let arms: [&SymContext; N_ARMS] = [&arm_a, &arm_b];

    // Each arm: 8 shared + 2 local.
    for a in arms {
        assert_eq!(arm_shape(a), (SHARED_CONSTRAINTS, DIVERGENT_PER_ARM));
    }

    let baseline = baseline_guarded_count(&arms);
    let cow = cow_guarded_count(&arms);
    let divergent_total = (N_ARMS * DIVERGENT_PER_ARM) as u64;

    // The pre-11.3 shape guarded every constraint of every arm.
    assert_eq!(
        baseline,
        (N_ARMS * (SHARED_CONSTRAINTS + DIVERGENT_PER_ARM)) as u64,
        "baseline merge shape should guard the total constraint count"
    );
    // CoW guards exactly the divergent constraints — the S5b acceptance's
    // "guarded assertions == divergent-constraint count".
    assert_eq!(
        cow, divergent_total,
        "CoW merge should guard exactly the divergent-constraint count"
    );
    // Guarded-count delta: the baseline GUARDS the prefix once per arm; CoW
    // guards it zero times (asserts it unguarded instead).
    assert_eq!(baseline - cow, (N_ARMS * SHARED_CONSTRAINTS) as u64);

    // Total assertions emitted (guarded + the unguarded prefix + the flag Or).
    // The prefix collapses from N_ARMS copies to one, so the whole-merge saving
    // is (N_ARMS - 1) * SHARED_CONSTRAINTS.
    let baseline_total = baseline + 1; // + final Or
    let cow_total = cow + SHARED_CONSTRAINTS as u64 + 1; // + unguarded prefix + Or
    assert_eq!(
        baseline_total - cow_total,
        ((N_ARMS - 1) * SHARED_CONSTRAINTS) as u64,
        "prefix collapse saves (N_ARMS-1) * SHARED_CONSTRAINTS total assertions"
    );
}

/// FORK-FREEZE INVARIANT: two arms forked from one base hold a ptr-equal
/// `z3_assertions_shared` Arc (the frozen prefix), and post-fork divergence
/// lives only in `local`. This ptr-sharing survives to the merge point — the
/// premise the unguarded-prefix rewrite keys off.
#[test]
fn test_fork_freeze_prefix_arc_is_ptr_shared_across_arms() {
    let (base, x) = build_base();
    let (arm_a, arm_b) = build_arms(&base, &x);

    let base_arc = Arc::clone(&base.z3_assertions_shared.lock());
    let a_arc = Arc::clone(&arm_a.z3_assertions_shared.lock());
    let b_arc = Arc::clone(&arm_b.z3_assertions_shared.lock());

    assert!(
        Arc::ptr_eq(&a_arc, &b_arc),
        "sibling arms must share the frozen prefix Arc"
    );
    assert!(
        Arc::ptr_eq(&base_arc, &a_arc),
        "the frozen prefix must be the base's post-fork shared Arc"
    );
    // Divergence is local-only: the shared prefix stayed at its frozen length.
    assert_eq!(a_arc.len(), SHARED_CONSTRAINTS);
    assert_eq!(
        arm_a.local_constraints.lock().z3_assertions.len(),
        DIVERGENT_PER_ARM
    );
}

/// SHADOWING CHECK (soundness input for angr-op0dn.11.3's kill clause): an
/// arm's `local` constraints must be strictly ADDITIVE — never a duplicate or
/// override of a `shared` prefix constraint. Verified by Z3 string identity:
/// no local assertion's textual form appears in the shared prefix.
#[test]
fn test_local_never_shadows_shared() {
    let (base, x) = build_base();
    let (arm_a, _arm_b) = build_arms(&base, &x);

    let shared = Arc::clone(&arm_a.z3_assertions_shared.lock());
    let shared_forms: std::collections::HashSet<String> =
        shared.iter().map(|b| b.to_string()).collect();

    let local = arm_a.local_constraints.lock();
    for assertion in local.z3_assertions.iter() {
        assert!(
            !shared_forms.contains(&assertion.to_string()),
            "local constraint {assertion} shadows a shared-prefix constraint"
        );
    }
}

/// PROTOTYPE SOLVABILITY + EVAL UNION (and the KILL clause): the CoW-merged
/// context must remain SAT at small divergence and admit exactly the union of
/// arm values — x=200 (arm A) and x=300 (arm B) admissible, x=100 (neither
/// arm) and x=5 (violates the shared prefix) rejected. If the unguarded-prefix
/// rewrite made merged states go unsat at small divergence, this trips and
/// S5's merge line is KILLED.
#[test]
fn test_cow_merge_stays_sat_and_evals_arm_union() {
    let (base, x) = build_base();
    let (arm_a, arm_b) = build_arms(&base, &x);

    let f0 = RustBV::symbolic(&base, "merge_flag_0", 1);
    let f1 = RustBV::symbolic(&base, "merge_flag_1", 1);
    let (merged, guarded) = cow_merge(&[&arm_a, &arm_b], &[f0, f1]);

    assert_eq!(guarded, (N_ARMS * DIVERGENT_PER_ARM) as u64);
    assert!(merged.is_sat(), "CoW-merged diamond must stay SAT");

    let xm = RustBV::symbolic(&merged, "x_diamond", 32);
    assert!(
        merged.solution(&xm, 200),
        "arm A value (x=200) must be admissible"
    );
    assert!(
        merged.solution(&xm, 300),
        "arm B value (x=300) must be admissible"
    );
    assert!(
        !merged.solution(&xm, 100),
        "x=100 satisfies neither arm and must be rejected"
    );
    assert!(
        !merged.solution(&xm, 5),
        "x=5 violates the shared prefix (x>7) and must be rejected"
    );
}

/// PRODUCTION PARITY: the CoW prototype is behaviour-equivalent to the shipped
/// `SymContext::merge` — same SAT verdict and same admit/reject on the union of
/// arm values. Since angr-op0dn.11.3 shipped the unguarded-prefix shape into
/// `merge` itself, this is no longer a cost comparison but an independent
/// reimplementation kept as a semantic oracle for it.
#[test]
fn test_cow_merge_matches_production_semantics() {
    let (base, x) = build_base();
    let (arm_a, arm_b) = build_arms(&base, &x);

    let f0 = RustBV::symbolic(&base, "merge_flag_0", 1);
    let f1 = RustBV::symbolic(&base, "merge_flag_1", 1);

    let prod = arm_a.merge(&[&arm_b], &[f0.clone(), f1.clone()]);
    let (cow, _) = cow_merge(&[&arm_a, &arm_b], &[f0, f1]);

    let xp = RustBV::symbolic(&prod, "x_diamond", 32);
    let xc = RustBV::symbolic(&cow, "x_diamond", 32);

    assert_eq!(prod.is_sat(), cow.is_sat());
    for v in [200u128, 300, 100, 5] {
        assert_eq!(
            prod.solution(&xp, v),
            cow.solution(&xc, v),
            "prod vs CoW disagree on x={v}"
        );
    }
}
