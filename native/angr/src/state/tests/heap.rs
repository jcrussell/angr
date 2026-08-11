//! The heap and CGC allocators themselves (not their merge): `union_from`'s
//! freed-allocation and dedup rules, `cgc_take_max_sinkhole`'s fit/split
//! selection, and `heap_alloc_aligned`'s rounding and bookkeeping.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

// angr-n0irt.3 (unit): HeapMetadata::union_from keeps self's size on an
// address collision and unions `freed` as a set (no double-count of a free both
// branches inherited from a pre-fork allocation).
#[test]
fn test_heap_metadata_union_from() {
    let mut a = HeapMetadata::default();
    let mut b = HeapMetadata::default();

    a.record_alloc(0x1000, 0x10); // shared addr, self size wins
    a.record_alloc(0x2000, 0x20); // self-only
    b.record_alloc(0x1000, 0x99); // conflict -> dropped in favor of self's 0x10
    b.record_alloc(0x3000, 0x30); // other-only -> unioned in

    a.record_free(0x9000); // inherited free present on both branches
    b.record_free(0x9000); // dup -> not double-counted
    b.record_free(0x8000); // other-only free -> unioned in

    a.union_from(&b);

    assert_eq!(
        a.alloc_size(0x1000),
        Some(0x10),
        "self size wins on collision"
    );
    assert_eq!(
        a.alloc_size(0x2000),
        Some(0x20),
        "self-only allocation kept"
    );
    assert_eq!(
        a.alloc_size(0x3000),
        Some(0x30),
        "other-only allocation unioned"
    );
    assert_eq!(
        a.free_count(),
        2,
        "freed unioned as a set: {{0x9000, 0x8000}}"
    );
    assert!(a.freed.contains(&0x8000), "other-only free must be present");
}

// angr-sqfj8.85: `union_from` must not resurrect an address `self` already
// freed just because `other` never freed it on its branch — the check
// against `self.freed` has to gate the `or_insert`, not run after.
#[test]
fn test_heap_metadata_union_from_does_not_resurrect_freed_allocation() {
    let mut a = HeapMetadata::default();
    let mut b = HeapMetadata::default();

    a.record_alloc(0x4000, 0x10);
    a.record_free(0x4000); // self already freed this address
    b.record_alloc(0x4000, 0x10); // other's branch never freed it

    a.union_from(&b);

    assert_eq!(
        a.alloc_size(0x4000),
        None,
        "self's free must win — union must not resurrect a freed allocation"
    );
    assert!(
        a.freed.contains(&0x4000),
        "the address stays recorded as freed"
    );
}

// angr-9ke6b.127 (unit): `freed` is a set on the single-path side too — a
// double free of the same pointer records one entry, so `free_count` means the
// same thing whether the duplication arrived via `union_from` or via two
// `record_free` calls on one path.
#[test]
fn test_heap_metadata_record_free_dedups() {
    let mut hm = HeapMetadata::default();
    hm.record_alloc(0x1000, 0x10);
    hm.record_alloc(0x2000, 0x20);

    assert_eq!(
        hm.record_free(0x1000),
        Some(0x10),
        "first free returns size"
    );
    assert_eq!(
        hm.record_free(0x1000),
        None,
        "double free is already distinguishable by the None return"
    );
    hm.record_free(0x2000);
    hm.record_free(0x3000); // never allocated here; still recorded once
    hm.record_free(0x3000);

    assert_eq!(
        hm.free_count(),
        3,
        "freed is a set: {{0x1000, 0x2000, 0x3000}}"
    );
    assert_eq!(
        hm.freed,
        vec![0x1000, 0x2000, 0x3000],
        "first-free order preserved (export determinism)"
    );

    // A merge of a double-freeing path into another is idempotent too.
    let mut other = HeapMetadata::default();
    other.record_free(0x1000);
    other.record_free(0x4000);
    hm.union_from(&other);
    assert_eq!(hm.freed, vec![0x1000, 0x2000, 0x3000, 0x4000]);
}

// angr-c7xno.70: direct coverage for the two process.rs allocators that had
// none. `cgc_take_max_sinkhole` was only reached transitively through
// `syscalls/cgc.rs`'s allocate handler, and `heap_alloc_aligned` only through
// `procedures/malloc.rs`'s memalign/posix_memalign — neither exercised the
// documented split / alignment-fallback edges at the state layer.

/// `cgc_take_max_sinkhole` returns `None` (caller bumps `allocation_base`) when
/// no sinkhole is large enough, and leaves the freelist untouched.
#[test]
fn test_cgc_take_max_sinkhole_returns_none_when_nothing_fits() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.cgc_add_sinkhole(0x1000, 0x40);
    s.cgc_add_sinkhole(0x2000, 0x80);

    assert_eq!(s.cgc_take_max_sinkhole(0x100), None);
    let mut remaining = s.cgc_sinkholes().to_vec();
    remaining.sort_unstable();
    assert_eq!(
        remaining,
        vec![(0x1000, 0x40), (0x2000, 0x80)],
        "a failed take must not consume anything"
    );
}

/// The scan is by *descending address among the sinkholes that fit*, not by
/// insertion order and not by size: the highest-address fitting entry wins even
/// when a lower-address entry is bigger and was inserted first.
#[test]
fn test_cgc_take_max_sinkhole_prefers_highest_fitting_address() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.cgc_add_sinkhole(0x1000, 0x400); // bigger, but lower
    s.cgc_add_sinkhole(0x3000, 0x100); // highest, still fits
    s.cgc_add_sinkhole(0x9000, 0x10); // highest overall, too small

    // Exact fit on the 0x3000 entry: it is consumed whole, nothing is pushed back.
    assert_eq!(s.cgc_take_max_sinkhole(0x100), Some(0x3000));
    let mut remaining = s.cgc_sinkholes().to_vec();
    remaining.sort_unstable();
    assert_eq!(
        remaining,
        vec![(0x1000, 0x400), (0x9000, 0x10)],
        "an exact fit must be removed outright, and the too-small entry kept"
    );
}

/// Split invariant: when the chosen sinkhole is larger than the request, the
/// leftover at the LOW end stays on the freelist and the HIGH end is returned.
/// Taking twice from the same region must therefore walk downward.
#[test]
fn test_cgc_take_max_sinkhole_splits_low_end_stays_high_end_returned() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.cgc_add_sinkhole(0x4000, 0x1000);

    // 0x4000 + (0x1000 - 0x200) == 0x4E00, leftover [0x4000, 0x4E00).
    assert_eq!(s.cgc_take_max_sinkhole(0x200), Some(0x4E00));
    assert_eq!(s.cgc_sinkholes(), &[(0x4000, 0xE00)]);

    // Second take carves the next-highest slice out of the leftover.
    assert_eq!(s.cgc_take_max_sinkhole(0x200), Some(0x4C00));
    assert_eq!(s.cgc_sinkholes(), &[(0x4000, 0xC00)]);
}

/// `heap_alloc_aligned` with `alignment` 0 or 1 is exactly `heap_alloc`: the
/// address is the unmodified brk even when that brk is oddly aligned.
#[test]
fn test_heap_alloc_aligned_falls_back_to_heap_alloc_for_alignment_0_and_1() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_heap_brk(0xC000_0003);

    let a0 = s.heap_alloc_aligned(8, 0);
    assert_eq!(a0, 0xC000_0003, "alignment 0 must not move the address");
    // Size bump is still rounded up to 16.
    assert_eq!(s.heap_brk(), 0xC000_0013);

    let a1 = s.heap_alloc_aligned(8, 1);
    assert_eq!(a1, 0xC000_0013, "alignment 1 must not move the address");
    assert_eq!(s.heap_brk(), 0xC000_0023);
}

/// A misaligned brk is rounded UP to the requested power-of-2 alignment, and
/// the bump from there is the 16-rounded size, so the next allocation stays
/// 16-aligned relative to the new base.
#[test]
fn test_heap_alloc_aligned_rounds_brk_up_and_bumps_by_rounded_size() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_heap_brk(0xC000_0008);

    let addr = s.heap_alloc_aligned(24, 0x100);
    assert_eq!(addr, 0xC000_0100, "brk must round up to the next 0x100");
    assert_eq!(
        s.heap_brk(),
        0xC000_0120,
        "size 24 rounds up to 32 for the bump"
    );

    // Already aligned: the address must not skip a whole alignment unit.
    let addr2 = s.heap_alloc_aligned(1, 0x20);
    assert_eq!(addr2, 0xC000_0120, "an already-aligned brk stays put");
    assert_eq!(s.heap_brk(), 0xC000_0130);
}

/// Alignment does not change the *metadata* bookkeeping: `record_alloc` stores
/// the requested size (not the 16-rounded bump) against the aligned address,
/// and the allocation is freeable at exactly that address.
#[test]
fn test_heap_alloc_aligned_records_requested_size_at_aligned_address() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_heap_brk(0xC000_0001);

    let addr = s.heap_alloc_aligned(24, 0x40);
    assert_eq!(addr, 0xC000_0040);
    assert!(s.heap_metadata().is_allocated(addr));
    assert_eq!(
        s.heap_metadata().alloc_size(addr),
        Some(24),
        "the requested size is recorded, not the 16-rounded bump"
    );
    assert_eq!(s.heap_metadata().alloc_count(), 1);
    assert!(
        !s.heap_metadata().is_allocated(0xC000_0001),
        "the pre-alignment brk is not a live allocation"
    );

    assert_eq!(s.heap_free(addr), Some(24));
    assert_eq!(s.heap_metadata().free_count(), 1);
}

/// angr-03vl4.53: the 16-byte size rounding is saturating, so a near-`u64::MAX`
/// size bumps the brk *forward* rather than wrapping the rounding down to a
/// small (or zero) bump. `procedures/malloc.rs` caps guest sizes long before
/// this, but the state-layer allocator must not depend on that to stay sane —
/// the wrapping form panicked under `[profile.release-checked]` and produced
/// aliasing allocations under `[profile.release]`.
#[test]
fn test_heap_alloc_saturates_instead_of_wrapping_the_size_rounding() {
    for size in [u64::MAX, u64::MAX - 1, u64::MAX - 14] {
        let mut s = RustSimState::new("amd64").unwrap();
        s.set_heap_brk(0x1000);

        let addr = s.heap_alloc(size);
        assert_eq!(addr, 0x1000);
        assert_ne!(
            s.heap_brk(),
            0x1000,
            "heap_alloc({size:#x}) must not leave the brk in place"
        );
        assert!(
            s.heap_brk() < 0x1000,
            "a >2^63 bump wraps the address space, but only after saturating \
             the rounding: brk={:#x}",
            s.heap_brk()
        );
        assert_eq!(s.heap_metadata().alloc_size(addr), Some(size));

        // Same for the aligned entry point.
        let mut s = RustSimState::new("amd64").unwrap();
        s.set_heap_brk(0x1000);
        let addr = s.heap_alloc_aligned(size, 0x40);
        assert_eq!(addr, 0x1000);
        assert_ne!(s.heap_brk(), 0x1000);
    }
}
