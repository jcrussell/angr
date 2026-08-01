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
        registry.register(12345, 1, "x", 32, SymbolKind::BitVector, obj);

        // Lookup should succeed
        assert_eq!(registry.lookup_by_hash(12345), Some(1));
        // D2 Fix: lookup_by_name_and_width uses width-qualified names
        assert!(
            registry
                .lookup_by_name_and_width("x", 32, SymbolKind::BitVector)
                .is_some()
        );
        assert!(registry.has_original(1));
    });
}

#[test]
fn test_registry_clear() {
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        let obj = py.None();
        registry.register(12345, 1, "x", 32, SymbolKind::BitVector, obj);

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
        registry.register(12345, 1, "x", 32, SymbolKind::BitVector, py.None());
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
        registry.register(111, 1, "keep", 32, SymbolKind::BitVector, py.None());
        registry.register(222, 2, "drop", 32, SymbolKind::BitVector, py.None());

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
        registry.register(1, 1, "a", 32, SymbolKind::BitVector, py.None());
        registry.register(2, 2, "b", 32, SymbolKind::BitVector, py.None());
        // Re-registering an existing id must not double-count: the warning is
        // driven by live entries, not by cumulative registrations.
        registry.register(3, 2, "b", 32, SymbolKind::BitVector, py.None());

        assert_eq!(registry.len(), 2);
        assert_eq!(registry.stats().new_registrations, 3);
    });
}

#[test]
fn bvs_width_1_and_bools_of_the_same_name_get_distinct_slots() {
    // angr-9ke6b.38: `BVS("flag", 1)` and `BoolS("flag")` are both width-1
    // Symbolics on the Rust side. Keyed on name+width alone they collided, and
    // the second import resolved to the first's rust_id — after which export
    // answered `get_original_ast` with the wrong-sorted claripy AST.
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        registry.register(0xB7, 7, "flag", 1, SymbolKind::BitVector, py.None());
        registry.register(0xB8, 8, "flag", 1, SymbolKind::Bool, py.None());

        let bv = registry
            .lookup_by_name_and_width("flag", 1, SymbolKind::BitVector)
            .expect("BV slot");
        let boolean = registry
            .lookup_by_name_and_width("flag", 1, SymbolKind::Bool)
            .expect("Bool slot");

        assert_eq!(bv.rust_id, 7);
        assert_eq!(boolean.rust_id, 8);
        assert_eq!(bv.kind, SymbolKind::BitVector);
        assert_eq!(boolean.kind, SymbolKind::Bool);
        assert_eq!(registry.len(), 2);
    });
}

#[test]
fn lookup_by_name_and_width_does_not_cross_sorts() {
    // A symbol registered under one sort must be invisible to a lookup under
    // the other, so the importer mints a fresh id instead of aliasing.
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        registry.register(0xC1, 1, "only_bool", 1, SymbolKind::Bool, py.None());

        assert!(
            registry
                .lookup_by_name_and_width("only_bool", 1, SymbolKind::BitVector)
                .is_none()
        );
        assert!(
            registry
                .lookup_by_name_and_width("only_bool", 1, SymbolKind::Bool)
                .is_some()
        );
    });
}

#[test]
fn remove_prunes_only_the_matching_sort_slot() {
    // The name map holds one entry per (name, width, sort); removing one id
    // must not evict its same-named sibling of the other sort.
    let registry = SymbolicIdentityRegistry::new();

    Python::initialize();
    Python::attach(|py| {
        registry.register(0xD1, 1, "dual", 1, SymbolKind::BitVector, py.None());
        registry.register(0xD2, 2, "dual", 1, SymbolKind::Bool, py.None());

        registry.remove(1);

        assert!(
            registry
                .lookup_by_name_and_width("dual", 1, SymbolKind::BitVector)
                .is_none()
        );
        assert_eq!(
            registry
                .lookup_by_name_and_width("dual", 1, SymbolKind::Bool)
                .map(|i| i.rust_id),
            Some(2)
        );
    });
}

#[test]
fn rust_symbol_name_is_identity_for_bitvectors() {
    // angr-9ke6b.223: every BV name — and every Z3 constant already built from
    // one — must be untouched by the Bool mangling, so this arm has to stay a
    // borrow of the input.
    assert_eq!(SymbolKind::BitVector.rust_symbol_name("flag"), "flag");
    assert!(matches!(
        SymbolKind::BitVector.rust_symbol_name("flag"),
        Cow::Borrowed(_)
    ));
}

#[test]
fn rust_symbol_name_separates_bool_from_same_named_bitvector() {
    // The registry sort tag splits their identity; this splits the Z3 constant
    // `RustBV::from_parts` builds from the name (angr-9ke6b.223).
    let bv = SymbolKind::BitVector.rust_symbol_name("flag");
    let boolean = SymbolKind::Bool.rust_symbol_name("flag");

    assert_ne!(bv, boolean);
    assert!(boolean.ends_with("flag"));
}

#[test]
fn rust_symbol_name_is_injective_over_bool_names() {
    // Distinct claripy names must stay distinct after mangling — a prefix keeps
    // that trivially true, but pin it so a future "sanitize the name" change
    // cannot quietly collapse two symbols into one Z3 constant.
    assert_ne!(
        SymbolKind::Bool.rust_symbol_name("a"),
        SymbolKind::Bool.rust_symbol_name("b")
    );
}

#[test]
fn strip_bool_symbol_name_inverts_the_bool_tag() {
    // `RustBV::from_parts` decodes the tag to decide whether to build a Bool-
    // sorted Z3 constant, so the two halves must round-trip exactly — and the
    // recovered string must be claripy's own name, since that is what makes the
    // constant identical to the one claripy's z3 backend emits (angr-9ke6b.223).
    let tagged = SymbolKind::Bool.rust_symbol_name("flag");
    assert_eq!(strip_bool_symbol_name(&tagged), Some("flag"));
}

#[test]
fn strip_bool_symbol_name_ignores_bitvector_names() {
    // A BV leaf must never be decoded as a Bool, or its Z3 term changes sort.
    let untagged = SymbolKind::BitVector.rust_symbol_name("flag");
    assert_eq!(strip_bool_symbol_name(&untagged), None);
    // Rust-minted names (stdin bytes, unconstrained fills) are BVs too.
    assert_eq!(strip_bool_symbol_name("stdin_0_8"), None);
}
