//! Unit tests for [`super`] — `StateSet` operations (insert/union/intersection/singleton).
//!
//! Extracted from the parent module body to keep `state.rs` focused on the
//! implementation; see the `rust-mod-tests-sibling-extraction` bd memory.

use super::*;

#[test]
fn test_state_set_basic() {
    let mut set = StateSet::with_capacity(10);
    assert!(set.is_empty());

    set.insert(3);
    set.insert(7);
    assert!(!set.is_empty());
    assert_eq!(set.len(), 2);
    assert!(set.contains(3));
    assert!(set.contains(7));
    assert!(!set.contains(5));
}

#[test]
fn test_state_set_union() {
    let mut set1 = StateSet::with_capacity(10);
    set1.insert(1);
    set1.insert(3);

    let mut set2 = StateSet::with_capacity(10);
    set2.insert(2);
    set2.insert(3);

    set1.union_with(&set2);
    assert_eq!(set1.len(), 3);
    assert!(set1.contains(1));
    assert!(set1.contains(2));
    assert!(set1.contains(3));
}

#[test]
fn test_state_set_intersection() {
    let mut set1 = StateSet::with_capacity(10);
    set1.insert(1);
    set1.insert(3);
    set1.insert(5);

    let mut set2 = StateSet::with_capacity(10);
    set2.insert(2);
    set2.insert(3);
    set2.insert(5);

    let inter = set1.intersection(&set2);
    assert_eq!(inter.len(), 2);
    assert!(inter.contains(3));
    assert!(inter.contains(5));
}

#[test]
fn test_state_set_singleton() {
    let set = StateSet::singleton(5, 10);
    assert_eq!(set.len(), 1);
    assert!(set.contains(5));
}

/// Regression: `PartialEq`/`Hash` must reflect logical contents only. The
/// derived impls delegated to `FixedBitSet`, whose own derives include the
/// capacity, so equal sets built at different capacities compared unequal.
#[test]
fn test_state_set_eq_hash_ignore_capacity() {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of(set: &StateSet) -> u64 {
        let mut hasher = DefaultHasher::new();
        set.hash(&mut hasher);
        hasher.finish()
    }

    let mut small = StateSet::with_capacity(16);
    small.insert(3);
    let large = StateSet::singleton(3, 100);

    assert_eq!(small, large);
    assert_eq!(hash_of(&small), hash_of(&large));

    // Grown-then-shrunk sets stay equal to a freshly built one.
    let mut grown = StateSet::with_capacity(4);
    grown.insert(3);
    grown.insert(64);
    grown.remove(64);
    assert_eq!(grown, small);
    assert_eq!(hash_of(&grown), hash_of(&small));

    // Empty sets at any capacity are one key.
    let empty_small = StateSet::with_capacity(0);
    let empty_large = StateSet::with_capacity(256);
    assert_eq!(empty_small, empty_large);
    assert_eq!(hash_of(&empty_small), hash_of(&empty_large));

    // Distinct contents still differ.
    assert_ne!(small, StateSet::singleton(4, 100));
    assert_ne!(small, StateSet::from_iter([3, 4]));

    // The point of the fix: usable directly as a map key.
    let mut map: HashMap<StateSet, u32> = HashMap::new();
    map.insert(small.clone(), 1);
    map.insert(large.clone(), 2);
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&small), Some(&2));
}
