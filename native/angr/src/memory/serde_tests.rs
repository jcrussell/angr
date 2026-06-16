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
