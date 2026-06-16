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
