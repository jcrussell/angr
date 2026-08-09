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
