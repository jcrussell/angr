// Tests for concrete_memory.rs (ConcreteMemoryRegion).
// Sibling-file layout per rust-mod-tests-sibling-extraction.
use super::*;

fn region(base: u64, len: usize) -> ConcreteMemoryRegion {
    ConcreteMemoryRegion {
        base,
        size: len as u64,
        data: Arc::new((0..len as u8).collect()),
    }
}

#[test]
fn read_inside_and_outside() {
    let r = region(0x1000, 8);
    assert_eq!(r.read(0x1000, 4).unwrap(), &[0, 1, 2, 3]);
    assert_eq!(r.read(0x1004, 4).unwrap(), &[4, 5, 6, 7]);
    // Before the base, past the end, and straddling the end all miss.
    assert!(r.read(0xfff, 1).is_none());
    assert!(r.read(0x1008, 1).is_none());
    assert!(r.read(0x1006, 4).is_none());
}

#[test]
fn contains_matches_read() {
    let r = region(0x1000, 8);
    assert!(r.contains(0x1000));
    assert!(r.contains(0x1007));
    assert!(!r.contains(0xfff));
    assert!(!r.contains(0x1008));
}

#[test]
fn region_at_top_of_address_space_does_not_overflow() {
    // `base + size` overflows here, so the old `addr < base + size` form of
    // `contains` (and `is_in_binary`) panicked under `--profile
    // release-checked` and wrapped to `0` in the shipped .so, which made every
    // in-region address test false (angr-xloth.3).
    let base = u64::MAX - 3;
    let r = region(base, 4);
    assert!(r.contains(base));
    assert!(r.contains(u64::MAX));
    assert!(!r.contains(0));
    assert!(!r.contains(base - 1));
    assert_eq!(r.read(base, 4).unwrap(), &[0, 1, 2, 3]);
    assert_eq!(r.read(u64::MAX, 1).unwrap(), &[3]);
    // A read whose end would wrap back into range must still miss rather than
    // index the slice out of bounds.
    assert!(r.read(u64::MAX, 2).is_none());
    assert!(r.read(0, 1).is_none());
}
