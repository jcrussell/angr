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

// --- unbounded-growth warning policy (angr-9ke6b.40) ---
//
// The registry has no GC caller, so growth over a long run is surfaced by a
// one-shot warning per threshold rather than collected. These exercise the
// policy directly (`maybe_warn_growth`) so they cost nothing — driving it
// through `register` would mean minting 100k symbols.

#[test]
fn growth_warn_index_does_not_advance_below_the_first_threshold() {
    let registry = SymbolicIdentityRegistry::new();

    registry.maybe_warn_growth(GROWTH_WARN_THRESHOLDS[0] - 1);

    assert_eq!(registry.growth_warn_idx.load(Ordering::SeqCst), 0);
}

#[test]
fn growth_warn_fires_once_per_threshold_not_once_per_registration() {
    let registry = SymbolicIdentityRegistry::new();

    // Many registrations sitting between thresholds 0 and 1 must advance the
    // index exactly once — otherwise every symbol past 100k logs a warning.
    for _ in 0..5 {
        registry.maybe_warn_growth(GROWTH_WARN_THRESHOLDS[0]);
    }
    assert_eq!(registry.growth_warn_idx.load(Ordering::SeqCst), 1);

    // Crossing the next threshold arms the next warning.
    registry.maybe_warn_growth(GROWTH_WARN_THRESHOLDS[1]);
    assert_eq!(registry.growth_warn_idx.load(Ordering::SeqCst), 2);
}

#[test]
fn growth_warn_saturates_past_the_last_threshold() {
    let registry = SymbolicIdentityRegistry::new();

    // A count above every threshold must not index past the array.
    for _ in 0..GROWTH_WARN_THRESHOLDS.len() + 3 {
        registry.maybe_warn_growth(usize::MAX);
    }

    assert_eq!(
        registry.growth_warn_idx.load(Ordering::SeqCst),
        GROWTH_WARN_THRESHOLDS.len()
    );
}

#[test]
fn clear_rearms_the_growth_warning() {
    let registry = SymbolicIdentityRegistry::new();

    registry.maybe_warn_growth(usize::MAX);
    assert_ne!(registry.growth_warn_idx.load(Ordering::SeqCst), 0);

    // `clear` is the exploration-start reset; a fresh run must be able to warn
    // again rather than inherit the previous run's exhausted thresholds.
    registry.clear();

    assert_eq!(registry.growth_warn_idx.load(Ordering::SeqCst), 0);
}

#[test]
fn register_tracks_live_count_for_the_growth_warning() {
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        registry.register(1, 1, "a", 32, py.None());
        registry.register(2, 2, "b", 32, py.None());
        // Re-registering an existing id must not double-count: the warning is
        // driven by live entries, not by cumulative registrations.
        registry.register(3, 2, "b", 32, py.None());

        assert_eq!(registry.len(), 2);
        assert_eq!(registry.stats().new_registrations, 3);
    });
}
