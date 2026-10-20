use super::*;

#[test]
fn serde_roundtrip_concrete_page() {
    let mut p = MemoryPage::new(0x1000, Permission::RW);
    p.store_concrete(0, &[1, 2, 3, 4]);
    p.store_concrete(PAGE_SIZE as u16 - 4, &[0xde, 0xad, 0xbe, 0xef]);

    let s = serde_json::to_string(&p).expect("serialize");
    let restored: MemoryPage = serde_json::from_str(&s).expect("deserialize");

    assert_eq!(restored.base_addr(), 0x1000);
    assert_eq!(restored.permissions(), Permission::RW);
    assert!(!restored.has_symbolic());
    assert_eq!(restored.load_concrete(0, 4), vec![1, 2, 3, 4]);
    assert_eq!(
        restored.load_concrete(PAGE_SIZE as u16 - 4, 4),
        vec![0xde, 0xad, 0xbe, 0xef]
    );
}

#[test]
fn serde_roundtrip_symbolic_bitmap() {
    let mut p = MemoryPage::new(0x2000, Permission::RWX);
    p.store_concrete(0, &[0xaa; 16]);
    // Mark a scattered set of bytes as symbolic.
    for off in [0u16, 7, 63, 64, 100, 4095] {
        p.mark_symbolic(off, 1);
    }
    assert!(p.has_symbolic());
    let before = p.symbolic_offsets();

    let s = serde_json::to_string(&p).expect("serialize");
    let restored: MemoryPage = serde_json::from_str(&s).expect("deserialize");

    assert_eq!(restored.base_addr(), 0x2000);
    assert_eq!(restored.permissions(), Permission::RWX);
    assert!(restored.has_symbolic());
    assert_eq!(restored.symbolic_offsets(), before);
    // Concrete bytes survive too (sanity check on the data side).
    assert_eq!(restored.load_concrete(0, 16), vec![0xaa; 16]);
}

#[test]
fn serde_roundtrip_multi_bitmap() {
    let mut p = MemoryPage::new(0x3000, Permission::R);
    p.mark_multi(5);
    p.mark_multi(2050);

    let s = serde_json::to_string(&p).expect("serialize");
    let restored: MemoryPage = serde_json::from_str(&s).expect("deserialize");

    assert!(restored.is_multi(5));
    assert!(restored.is_multi(2050));
    assert!(!restored.is_multi(6));
    assert!(!restored.has_symbolic());
}

#[test]
fn serde_malformed_bitmap_length_drops_to_none() {
    // A snapshot that names a symbolic bitmap with the wrong word
    // count should not crash; instead the page should come back as
    // fully concrete (the safer fallback).
    let bad = MemoryPageData {
        data: vec![0u8; PAGE_SIZE as usize],
        permissions: Permission::RW,
        base_addr: 0x4000,
        symbolic_bitmap: Some(vec![0u64; BITMAP_WORDS - 1]),
        multi_bitmap: None,
    };
    let s = serde_json::to_string(&bad).expect("serialize");
    let restored: MemoryPage = serde_json::from_str(&s).expect("deserialize");
    assert!(!restored.has_symbolic());
}

/// The wire form is `BITMAP_WORDS` u64 words with byte `i` at bit `i % 64` of
/// word `i / 64`, independent of the in-memory `FixedBitSet` form, so
/// existing snapshots keep decoding (angr-mx00p.30).
#[test]
fn serde_bitmap_word_layout_is_stable() {
    let mut p = MemoryPage::new(0x5000, Permission::RW);
    for off in [0u16, 63, 64, 4095] {
        p.mark_symbolic(off, 1);
    }
    let data = MemoryPageData::from(p);
    let words = data.symbolic_bitmap.clone().expect("symbolic bitmap present");
    assert_eq!(words.len(), BITMAP_WORDS);
    assert_eq!(words[0], 1 | (1u64 << 63));
    assert_eq!(words[1], 1);
    assert_eq!(words[BITMAP_WORDS - 1], 1u64 << 63);
    assert!(words[2..BITMAP_WORDS - 1].iter().all(|&w| w == 0));
    assert_eq!(data.multi_bitmap, None);

    let restored = MemoryPage::from(data);
    assert_eq!(restored.symbolic_offsets(), vec![0, 63, 64, 4095]);
}

/// An all-zero word vector loads as "no bits set", keeping `has_symbolic`'s
/// O(1) drop-when-empty invariant.
#[test]
fn serde_all_zero_bitmap_loads_unallocated() {
    let zero = MemoryPageData {
        data: vec![0u8; PAGE_SIZE as usize],
        permissions: Permission::RW,
        base_addr: 0x6000,
        symbolic_bitmap: Some(vec![0u64; BITMAP_WORDS]),
        multi_bitmap: Some(vec![0u64; BITMAP_WORDS]),
    };
    let restored = MemoryPage::from(zero);
    assert!(!restored.has_symbolic());
    assert!(!restored.has_multi());
}

/// Range set / clear across word boundaries, the empty range, and the drop
/// once the last bit clears.
#[test]
fn bitmap_ranges_and_drop_when_empty() {
    let mut p = MemoryPage::new(0x7000, Permission::RW);
    p.mark_symbolic(10, 0);
    assert!(!p.has_symbolic(), "an empty range must not allocate");

    p.mark_symbolic(60, 10); // spans words 0 and 1
    assert_eq!(p.symbolic_offsets(), (60..70).collect::<Vec<u16>>());
    p.store_concrete(62, &[0; 4]);
    assert_eq!(
        p.symbolic_offsets(),
        vec![60, 61, 66, 67, 68, 69],
        "a concrete store clears exactly its bytes"
    );
    p.store_concrete(60, &[0; 10]);
    assert!(!p.has_symbolic(), "the bitmap drops once the last bit clears");

    p.mark_multi(4095);
    assert!(p.has_multi() && p.is_multi(4095));
    p.clear_multi(4095);
    assert!(!p.has_multi());
}
