// Tests for symbolic/sharing.rs (see rust-mod-tests-sibling-extraction for why
// these live in a sibling file rather than an inline `mod tests`).
//
// The manager-level fold on top of this walk is pinned separately in
// `exploration/manager_methods_diagnostics_tests.rs`; what is pinned here is
// the walk's own accounting — in particular that node addresses stay
// comparable ACROSS batches, which is the angr-gkcxh miscount.
use super::*;
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::SymContext;

/// One batch, two separately-built but structurally identical leaves: two
/// nodes, one shape. The baseline every cross-batch assertion below is a
/// generalization of.
#[test]
fn one_batch_separates_pointer_identity_from_structural_identity() {
    let mut walk = ConstraintSharingWalk::new();
    walk.visit_batch(vec![RustBV::concrete(0x1234, 64), RustBV::concrete(0x1234, 64)]);
    let stats = walk.into_stats();
    assert_eq!(stats.total_visits, 2);
    assert_eq!(stats.unique_pointers, 2, "two distinct allocations");
    assert_eq!(stats.unique_shapes, 1, "...of one structural shape");
}

/// The angr-gkcxh regression: batches handed in one at a time must each be
/// counted, not collapsed onto a freed predecessor's address.
///
/// Before the fix, `fold_sharing_walk` walked a `Vec` it dropped on return, so
/// the allocator handed batch N+1 the address batch N had just freed; the walk
/// saw a pointer hit and returned the *previous* batch's shape id — leaving
/// both `unique_pointers` and `unique_shapes` pinned near 1 no matter how many
/// distinct constraints were folded in.
#[test]
fn distinct_constraints_in_separate_batches_are_all_counted() {
    const N: u128 = 64;
    let mut walk = ConstraintSharingWalk::new();
    for value in 0..N {
        walk.visit_batch(vec![RustBV::concrete(value, 64)]);
    }
    let stats = walk.into_stats();
    assert_eq!(stats.total_visits, N as u64);
    assert_eq!(
        stats.unique_pointers, N as u64,
        "each batch's node must key on an address no earlier batch can have reused"
    );
    assert_eq!(
        stats.unique_shapes, N as u64,
        "the values are all distinct, so no two share a shape"
    );
}

/// `unique_pointers >= unique_shapes` is what makes `structural_duplicates`
/// (their difference) meaningful. It can only be violated by counting one node
/// as several shapes or several nodes as one pointer — i.e. by exactly the
/// address-reuse bug above, which drove pointers *below* shapes.
#[test]
fn pointers_never_undercount_shapes_across_many_batches() {
    let mut walk = ConstraintSharingWalk::new();
    for value in 0..32u128 {
        // Mixed batch sizes and a repeated shape so the two counters diverge.
        walk.visit_batch(vec![
            RustBV::concrete(value, 64),
            RustBV::concrete(value, 64),
            RustBV::concrete(0, 32),
        ]);
    }
    let stats = walk.into_stats();
    assert!(
        stats.unique_pointers >= stats.unique_shapes,
        "pointers {} must not fall below shapes {}",
        stats.unique_pointers,
        stats.unique_shapes
    );
    assert_eq!(stats.unique_pointers, 96, "3 nodes x 32 batches");
    assert_eq!(
        stats.unique_shapes, 33,
        "32 distinct 64-bit values plus the one repeated 32-bit zero"
    );
}

/// The same guarantee through the real caller: `fold_sharing_walk` clones the
/// context's assumed list, and that clone is the temporary angr-gkcxh was
/// about. Two contexts holding different constraints must contribute two
/// pointers and two shapes.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn fold_sharing_walk_keeps_each_contexts_clone_alive() {
    let mut walk = ConstraintSharingWalk::new();
    for value in 0..16u128 {
        let ctx = SymContext::new();
        ctx.assumed_constraints_push(RustBV::concrete(value, 64), true);
        ctx.fold_sharing_walk(&mut walk);
    }
    let stats = walk.into_stats();
    assert_eq!(stats.unique_pointers, 16);
    assert_eq!(stats.unique_shapes, 16);
}
