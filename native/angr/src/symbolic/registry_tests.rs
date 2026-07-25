// Tests for the symbolic identity registry (SymbolicIdentityRegistry).
// Extracted from registry.rs as a sibling #[path]-included module.

use super::*;

#[test]
fn test_registry_basic() {
    let registry = SymbolicIdentityRegistry::new();

    // Register a symbol
    Python::initialize();
    Python::attach(|py| {
        let obj = py.None();
        registry.register(12345, 1, "x", 32, obj);

        // Lookup should succeed
        assert_eq!(registry.lookup_by_hash(12345), Some(1));
        // D2 Fix: lookup_by_name_and_width uses width-qualified names
        assert!(registry.lookup_by_name_and_width("x", 32).is_some());
        assert!(registry.has_original(1));
    });
}

#[test]
fn test_registry_clear() {
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        let obj = py.None();
        registry.register(12345, 1, "x", 32, obj);

        assert!(!registry.is_empty());

        registry.clear();

        assert!(registry.is_empty());
        assert!(registry.lookup_by_hash(12345).is_none());
    });
}

#[test]
fn test_registry_remove_prunes_name_map() {
    // remove() must drop the id→name entry too, keeping all four maps
    // symmetric (angr-4xaga.3).
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        registry.register(12345, 1, "x", 32, py.None());
        assert_eq!(registry.lookup_name_by_id(1).as_deref(), Some("x"));

        registry.remove(1);

        assert!(registry.lookup_name_by_id(1).is_none());
        assert!(!registry.has_original(1));
    });
}

#[test]
fn test_registry_retain_prunes_name_map() {
    // retain() must prune the id→name entry for pruned ids (angr-4xaga.3).
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        registry.register(111, 1, "keep", 32, py.None());
        registry.register(222, 2, "drop", 32, py.None());

        let mut active = std::collections::HashSet::new();
        active.insert(1u64);
        registry.retain(&active);

        assert_eq!(registry.lookup_name_by_id(1).as_deref(), Some("keep"));
        assert!(registry.lookup_name_by_id(2).is_none());
    });
}

#[test]
fn test_registry_allocate_id() {
    let registry = SymbolicIdentityRegistry::new();

    let id1 = registry.allocate_id();
    let id2 = registry.allocate_id();
    let id3 = registry.allocate_id();

    assert!(id1 < id2);
    assert!(id2 < id3);
}

#[test]
fn test_registry_ensure_id_at_least() {
    let registry = SymbolicIdentityRegistry::new();

    registry.ensure_id_at_least(100);

    let next = registry.allocate_id();
    assert!(next >= 101);
}
