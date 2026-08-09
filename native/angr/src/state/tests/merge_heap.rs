//! Merging the heap and CGC allocator state: min CGC base with sinkhole
//! intersection, and heap-metadata union across branches.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

// angr-n0irt.2: RustSimState::merge must extend the angr-ph300.51 max-merge fix
// to the CGC allocator state. cgc_allocation_base grows DOWNWARD, so the anti-
// alias combinator is min (furthest-advanced base across branches), NOT the max
// used for the up-growing brk/mmap watermarks. cgc_sinkholes must INTERSECT (a
// freed region is only safe to reuse if it was freed in every branch — a region
// freed in one branch but live in another survives the page-unioning merge, so
// unioning sinkholes would reintroduce aliasing).
#[test]
fn test_merge_takes_min_cgc_base_and_intersects_sinkholes() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();
    let mut c = RustSimState::new("amd64").unwrap();

    // Downward-growing base: a is the default high-water, b and c bumped lower
    // (allocated more). min over all three is b's 0xB700_0000 — which is neither
    // self's value nor a plain "take other", so this distinguishes min from max
    // (max would be a's 0xB800_0000) and from "last other" (c's 0xB750_0000).
    a.set_cgc_allocation_base(0xB800_0000);
    b.set_cgc_allocation_base(0xB700_0000);
    c.set_cgc_allocation_base(0xB750_0000);

    // Only (0x1000, 0x100) is present in ALL three branches -> survives.
    a.cgc_add_sinkhole(0x1000, 0x100);
    a.cgc_add_sinkhole(0x2000, 0x100); // self-only -> dropped
    b.cgc_add_sinkhole(0x1000, 0x100);
    b.cgc_add_sinkhole(0x4000, 0x100); // other-only -> dropped
    c.cgc_add_sinkhole(0x1000, 0x100);
    c.cgc_add_sinkhole(0x5000, 0x100); // other-only -> dropped

    let m0 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "n0irt2_m0", 1)
    };
    let m1 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "n0irt2_m1", 1)
    };
    let m2 = {
        let s = a.solver().borrow();
        RustBV::symbolic(&s, "n0irt2_m2", 1)
    };
    let merged = a.merge(&[&b, &c], &[m0, m1, m2]);

    assert_eq!(
        merged.cgc_allocation_base(),
        0xB700_0000,
        "cgc_allocation_base must be min across all branches (grows downward)"
    );
    assert_eq!(
        merged.cgc_sinkholes(),
        &[(0x1000, 0x100)],
        "cgc_sinkholes must intersect: only the region freed in every branch survives"
    );
}

// angr-n0irt.3: RustSimState::merge must union heap_metadata across every
// branch, not keep only self's. heap_brk is maxed on merge, so a branch-only
// malloc stays reachable in the unioned memory — but if its alloc_size entry
// lived only on the dropped branch, NativeRealloc's copy length defaults to the
// FULL new size (old_size.map_or(size, ..)), over-reading past the true old
// allocation. Unioning allocations closes that; freed addresses union as a set.
#[test]
fn test_merge_unions_heap_metadata_across_branches() {
    Python::initialize();

    let mut a = RustSimState::new("amd64").unwrap();
    let mut b = RustSimState::new("amd64").unwrap();

    // self (a) allocates one block. other (b) allocates the same base block
    // (shared, as if inherited from a common fork point) plus a *second*,
    // higher block that lives only on its branch — that branch-only block must
    // survive the merge carrying its true size. a and b are independent states
    // starting from the same heap_brk, so b's first alloc aliases a's address.
    let a_addr = a.heap_alloc(0x10);
    let b_shared = b.heap_alloc(0x10);
    assert_eq!(a_addr, b_shared, "both start at the same heap_brk");
    let b_only = b.heap_alloc(0x40);
    assert_ne!(a_addr, b_only, "branch-only block sits at a higher address");

    // b frees the shared-address block only on its branch; the free must still
    // be recorded in the merged state for double-free / leak analysis.
    b.heap_free(b_shared);

    let (m0, m1) = {
        let s = a.solver().borrow();
        (
            RustBV::symbolic(&s, "n0irt3_m0", 1),
            RustBV::symbolic(&s, "n0irt3_m1", 1),
        )
    };
    let merged = a.merge(&[&b], &[m0, m1]);

    let hm = merged.heap_metadata();
    assert_eq!(
        hm.alloc_size(a_addr),
        Some(0x10),
        "self's allocation must survive the merge (a still holds this address)"
    );
    assert_eq!(
        hm.alloc_size(b_only),
        Some(0x40),
        "branch-only allocation must be unioned in with its true size, so \
         NativeRealloc copies min(new, 0x40) rather than over-reading"
    );
    assert!(
        hm.freed.contains(&b_shared),
        "other-branch free must be recorded in the merged state"
    );
}
