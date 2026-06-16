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
fn small_store_falls_back_to_earlier() {
    // Larger covering store followed by smaller store at same addr.
    // Old semantics: reverse scan finds earlier fully-covering store first.
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]); // idx 0
    buf.push(0x100, vec![9]); // idx 1, partial
    // Load size 4 at 0x100: idx 1 doesn't fully cover; reverse scan finds idx 0.
    assert_eq!(buf.try_load(0x100, 4).unwrap(), &[1, 2, 3, 4]);
}

#[test]
fn drain_clears_index() {
    let mut buf = PendingStoreBuffer::with_capacity(8);
    buf.push(0x100, vec![1, 2, 3, 4]);
    let _: Vec<_> = buf.drain().collect();
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
