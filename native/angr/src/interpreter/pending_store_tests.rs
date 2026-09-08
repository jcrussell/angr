// Tests for pending_store.rs (PendingStore).
// Split out of the parent module's inline `mod tests` (see rust-mod-tests-sibling-extraction).
use super::*;

#[test]
fn empty_buffer_misses() {
    let buf = PendingStoreBuffer::with_capacity(8);
    assert!(buf.try_load(0x100, 4).is_none());
}

#[test]
fn single_store_full_cover() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]);
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 3, 4]);
    assert_eq!(buf.try_load(0x101, 2).unwrap(), &[2, 3]);
}

#[test]
fn miss_when_no_overlap() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x200, vec![1, 2, 3, 4]);
    assert!(buf.try_load(0x100, 4).is_none());
    assert!(buf.try_load(0x205, 4).is_none()); // beyond end
}

#[test]
fn most_recent_full_cover_wins() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]);
    buf.push(0x100, vec![9, 8, 7, 6]);
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[9, 8, 7, 6]);
}

#[test]
fn small_store_patches_earlier_store() {
    // Larger covering store followed by smaller store at same addr.
    // The reverse-scan fallback (idx 1 doesn't fully cover a 4-byte load, so
    // we fall back to idx 0) must see the patched byte, not the stale
    // pre-overwrite value: last-write-wins per byte, not per whole store.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]); // idx 0
    buf.push(0x100, vec![9]); // idx 1, partial: overwrites byte 0
    // Load size 4 at 0x100: idx 1 doesn't fully cover; reverse scan finds idx
    // 0, whose buffer was patched in place when idx 1 was pushed.
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[9, 2, 3, 4]);
}

#[test]
fn small_store_in_middle_patches_earlier_store() {
    // Partial overlap strictly inside the earlier store's range (not at its
    // base address).
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]); // idx 0: covers 0x100..0x104
    buf.push(0x102, vec![9]); // idx 1: overwrites byte at 0x102
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 9, 4]);
}

#[test]
fn store_extending_past_earlier_end_patches_overlap_only() {
    // New store starts inside the earlier store's range and extends past its
    // end. Only the overlapping byte should be patched into the earlier
    // store; the earlier store's own trailing byte (untouched) is preserved,
    // and the fast indexed path (not the fallback) returns it directly.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]); // idx 0: covers 0x100..0x104
    buf.push(0x103, vec![7, 8]); // idx 1: covers 0x103..0x105, overlaps last byte
    // 0x100 is still indexed to idx 0 (untouched by the idx 1 push), and idx
    // 0 fully covers a 4-byte load -- returned directly, but must reflect
    // the patched last byte.
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 3, 7]);
    // The new store itself is unaffected and still returns its own bytes.
    assert_eq!(buf.try_load(0x103, 2).unwrap(), &[7, 8]);
}

#[test]
fn store_before_earlier_start_patches_overlap_only() {
    // New store starts before the earlier store's range and overlaps only
    // its leading bytes.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x104, vec![1, 2, 3, 4]); // idx 0: covers 0x104..0x108
    buf.push(0x102, vec![9, 10, 11, 12]); // idx 1: covers 0x102..0x106
    // 0x104 is now indexed to idx 1, which doesn't fully cover a 4-byte load
    // (idx 1 ends at 0x106); reverse scan falls back to idx 0, whose first
    // two bytes were patched by the idx 1 push.
    assert_eq!(buf.try_load(0x104, 4).unwrap(), &[11, 12, 3, 4]);
}

#[test]
fn single_wide_store_patches_multiple_earlier_stores() {
    // One new store overlapping two disjoint earlier stores must patch both.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]); // idx 0: covers 0x100..0x104
    buf.push(0x108, vec![5, 6, 7, 8]); // idx 1: covers 0x108..0x10c
    // idx 2: covers 0x102..0x10a, overlapping the tail of idx 0 (0x102,0x103)
    // and the head of idx 1 (0x108, 0x109).
    buf.push(0x102, vec![10, 11, 12, 13, 14, 15, 16, 17]);
    // 0x100 still indexed to idx 0, which fully covers a 4-byte load.
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 10, 11]);
    // 0x108 is indexed to idx 2, which does NOT fully cover a 4-byte load
    // (idx 2 ends at 0x10a); reverse scan falls back to idx 1, whose first
    // two bytes were patched by the idx 2 push.
    assert_eq!(buf.try_load(0x108, 4).unwrap(), &[16, 17, 7, 8]);
}

#[test]
fn drain_clears_index() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]);
    drop(buf.drain().collect::<Vec<_>>());
    assert!(buf.is_empty());
    assert!(buf.try_load(0x100, 4).is_none());
}

#[test]
fn clear_clears_index() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]);
    buf.clear();
    assert!(buf.is_empty());
    assert!(buf.try_load(0x100, 4).is_none());
}

#[test]
fn try_load_exact_requires_base_match() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        buf.try_load_exact(0x100, 8).unwrap(),
        &[1, 2, 3, 4, 5, 6, 7, 8]
    );
    // Loading at offset within the store should fail (not an exact base match).
    assert!(buf.try_load_exact(0x101, 4).is_none());
}

#[test]
fn store_wrapping_past_the_top_of_the_address_space() {
    // A store based at u64::MAX - 3 covers MAX-3..=MAX and then 0..=3.
    // `push` indexes those bytes with `wrapping_add` and `try_load` recovers
    // the offset with the matching `wrapping_sub`; under
    // `--profile release-checked` the pre-angr-xloth.3 `addr + offset` /
    // `addr - store_addr` pair panicked instead (angr-xloth.3).
    let base = u64::MAX - 3;
    let mut buf = PendingStoreBuffer::with_capacity(4);
    buf.push(base, vec![10, 11, 12, 13, 14, 15, 16, 17]);

    assert_eq!(buf.try_load(base, 4).unwrap(), &[10, 11, 12, 13]);
    // The wrapped tail: address 0 is the 5th byte of the store.
    assert_eq!(buf.try_load(0, 4).unwrap(), &[14, 15, 16, 17]);
    assert_eq!(buf.try_load(2, 2).unwrap(), &[16, 17]);
    // Past the end of the wrapped range is still a miss.
    assert!(buf.try_load(4, 1).is_none());
    // ...and so is a load that would run off the end of the store.
    assert!(buf.try_load(3, 2).is_none());
}

#[test]
fn load_at_top_of_address_space_does_not_overflow() {
    // The reverse-scan fallback used to compute `addr + size` directly, which
    // overflows for a load abutting u64::MAX (angr-xloth.3).
    let mut buf = PendingStoreBuffer::with_capacity(4);
    buf.push(u64::MAX - 7, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // Narrower, later store over the same base forces the reverse scan.
    buf.push(u64::MAX - 7, vec![9, 10]);
    assert_eq!(buf.try_load(u64::MAX - 7, 8).unwrap(), &[9, 10, 3, 4, 5, 6, 7, 8]);
    assert!(buf.try_load(u64::MAX, 8).is_none());
}

// --- try_load_assembled: multi-store union coverage (angr-6cp06.69) ---

#[test]
fn adjacent_narrow_stores_assemble_into_wider_load() {
    // The shape the bug was filed for: a byte-wise write loop, then a wider
    // read-back before the buffer is flushed. No single store covers the
    // 2-byte load, so `try_load` misses and the assembly path must answer.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![0xaa]);
    buf.push(0x101, vec![0xbb]);
    assert!(buf.try_load(0x100, 2).is_none());
    assert_eq!(buf.try_load_assembled(0x100, 2).unwrap(), vec![0xaa, 0xbb]);
}

#[test]
fn assembled_load_spans_four_single_byte_stores() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    for (i, b) in [1u8, 2, 3, 4].into_iter().enumerate() {
        buf.push(0x100 + i as u64, vec![b]);
    }
    assert_eq!(
        buf.try_load_assembled(0x100, 4).unwrap(),
        vec![1, 2, 3, 4]
    );
    // Sub-ranges assemble too.
    assert_eq!(buf.try_load_assembled(0x101, 2).unwrap(), vec![2, 3]);
}

#[test]
fn assembled_load_is_last_write_wins_per_byte() {
    // Overlapping (not merely adjacent) stores: the newest store owns each
    // byte it covers, even where an older store also covers it.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3]); // 0x100..0x103
    buf.push(0x102, vec![9, 9]); // 0x102..0x104, overwrites 0x102
    assert!(buf.try_load(0x100, 4).is_none()); // neither store covers 4 bytes
    assert_eq!(
        buf.try_load_assembled(0x100, 4).unwrap(),
        vec![1, 2, 9, 9]
    );
}

#[test]
fn partially_covered_load_does_not_assemble() {
    // A hole in the middle and a hole past the end both fall through to the
    // caller's lower layers rather than fabricating bytes.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1]);
    buf.push(0x102, vec![3]);
    assert!(buf.try_load_assembled(0x100, 3).is_none()); // 0x101 unbuffered
    assert!(buf.try_load_assembled(0x102, 2).is_none()); // 0x103 unbuffered
    // Load starting outside the buffer misses without allocating.
    assert!(buf.try_load_assembled(0x101, 1).is_none());
}

#[test]
fn assembled_load_wraps_at_top_of_address_space() {
    // `push` indexes bytes with `wrapping_add`; the assembly path must
    // recover the offsets with the matching `wrapping_sub`.
    let mut buf = PendingStoreBuffer::with_capacity(4);
    buf.push(u64::MAX, vec![0xde]);
    buf.push(0, vec![0xad]);
    assert_eq!(
        buf.try_load_assembled(u64::MAX, 2).unwrap(),
        vec![0xde, 0xad]
    );
}

#[test]
fn assembled_load_after_drain_misses() {
    let mut buf = PendingStoreBuffer::with_capacity(4);
    buf.push(0x100, vec![1]);
    buf.push(0x101, vec![2]);
    let drained: Vec<_> = buf.drain().collect();
    assert_eq!(drained.len(), 2);
    assert!(buf.try_load_assembled(0x100, 2).is_none());
}

#[test]
fn partial_load_reports_covered_bytes_and_holes() {
    // The shape angr-6cp06.88 is about: 1 byte stored at 0x100, 2 bytes read
    // from 0x100. `try_load`/`try_load_assembled` both miss; the partial
    // gather reports the buffered byte and a hole the caller fills from the
    // layers below.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![0xaa]);
    assert!(buf.try_load(0x100, 2).is_none());
    assert!(buf.try_load_assembled(0x100, 2).is_none());
    assert_eq!(buf.try_load_partial(0x100, 2).unwrap(), vec![Some(0xaa), None]);
}

#[test]
fn partial_load_covers_hole_in_the_middle_and_leading_hole() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1]);
    buf.push(0x102, vec![3]);
    // Hole in the middle.
    assert_eq!(
        buf.try_load_partial(0x100, 3).unwrap(),
        vec![Some(1), None, Some(3)]
    );
    // A leading hole: the load's *first* byte is unbuffered, which is exactly
    // what `try_load_assembled`'s first-byte fast-skip cannot see.
    assert_eq!(
        buf.try_load_partial(0x101, 2).unwrap(),
        vec![None, Some(3)]
    );
}

#[test]
fn partial_load_misses_when_no_byte_is_buffered() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    assert!(buf.try_load_partial(0x100, 4).is_none());
    buf.push(0x200, vec![7]);
    // Buffer non-empty but disjoint from the load: still a miss, so the
    // caller hands back the lower-layer result untouched.
    assert!(buf.try_load_partial(0x100, 4).is_none());
}

#[test]
fn partial_load_is_last_write_wins() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2]);
    buf.push(0x101, vec![0x99]);
    assert_eq!(
        buf.try_load_partial(0x100, 3).unwrap(),
        vec![Some(1), Some(0x99), None]
    );
}

#[test]
fn partial_load_wraps_at_top_of_address_space() {
    let mut buf = PendingStoreBuffer::with_capacity(4);
    buf.push(u64::MAX, vec![0xde]);
    assert_eq!(
        buf.try_load_partial(u64::MAX, 2).unwrap(),
        vec![Some(0xde), None]
    );
}

#[test]
fn partial_load_after_drain_misses() {
    let mut buf = PendingStoreBuffer::with_capacity(4);
    buf.push(0x100, vec![1]);
    let _ = buf.drain().count();
    assert!(buf.try_load_partial(0x100, 2).is_none());
}
