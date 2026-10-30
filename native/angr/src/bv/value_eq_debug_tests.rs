//! `PartialEq` / `Debug` identity semantics for every `RustBV` variant
//! (angr-sqfj8.96).

use super::*;

/// angr-sqfj8.96: `PartialEq` used to short-circuit to `false` whenever
/// either side had no `as_u128()`, so `x == x` was false for every
/// symbolic value — a latent trap for any `contains`/dedup caller.
#[test]
fn test_partial_eq_is_reflexive_for_every_variant() {
    let concrete = RustBV::concrete(5, 32);
    let symbolic = RustBV::symbol("reflexive_sym", 32);
    let ast = symbolic.add(&RustBV::concrete(1, 32));
    assert!(
        matches!(ast, RustBV::Ast { .. }),
        "test setup: symbolic + concrete should build a clarirs term, got {ast:?}"
    );
    for bv in [&concrete, &symbolic, &ast] {
        assert_eq!(bv, bv, "PartialEq must be reflexive for {bv:?}");
        assert_eq!(
            bv,
            &bv.clone(),
            "a clone must compare equal to its source for {bv:?}"
        );
    }
}

/// The equality classes from the `PartialEq for RustBV` doc: known value
/// (Concrete), leaf identity by name, and node identity for clarirs terms,
/// whose nodes are interned so identity is structure.
/// Nothing compares equal across a class boundary.
#[test]
fn test_partial_eq_classes_are_disjoint_and_identity_based() {
    // Class 1: value-based.
    let concrete = RustBV::concrete(5, 32);
    assert_eq!(concrete, RustBV::concrete(5, 32));
    assert_ne!(concrete, RustBV::concrete(5, 64), "width must be compared");
    assert_ne!(concrete, RustBV::concrete(6, 32));

    // Class 2: a named leaf is its name and width (angr-mx00p.12): the id
    // only feeds the name counter, so it no longer tells two leaves apart —
    // as in Z3, where the name is the constant.
    let s1 = RustBV::symbol("alpha", 32);
    let s1_other_id = RustBV::symbol("alpha", 32);
    let s2 = RustBV::symbol("beta", 32);
    let s1_wide = RustBV::symbol("alpha", 64);
    assert_eq!(s1, s1_other_id, "same name + width is the same leaf symbol");
    assert_ne!(s1, s2, "distinct names are distinct symbols");
    assert_ne!(s1, s1_wide, "width must be compared");

    let sym = RustBV::symbol("classes_sym", 32);
    // Class 3: an Ast is node identity, and clarirs interns, so two
    // independently built terms with the same structure are one node.
    let a1 = sym.add(&RustBV::concrete(1, 32));
    let a2 = sym.add(&RustBV::concrete(1, 32));
    assert_eq!(a1, a2, "an interned term is one node");
    assert_ne!(a1, sym.add(&RustBV::concrete(2, 32)));

    // Cross-class: never equal, in either direction.
    for (a, b) in [(&concrete, &s1), (&concrete, &a1), (&s1, &a1)] {
        assert_ne!(a, b, "{a:?} and {b:?} are in different equality classes");
        assert_ne!(b, a, "{b:?} and {a:?} are in different equality classes");
    }
}

/// `Debug` must surface each value's identity — otherwise two distinct values
/// print identically in logs and assertion failures (angr-sqfj8.99): for a
/// named leaf, its name.
#[test]
fn test_debug_includes_symbol_id_for_leaf_variants() {
    // A named leaf prints its name: that is its identity (angr-mx00p.12).
    let a = RustBV::symbol("alpha", 32);
    let b = RustBV::symbol("beta", 32);
    assert_eq!(format!("{a:?}"), "Ast(alpha, 32)");
    assert_ne!(format!("{a:?}"), format!("{b:?}"));

    // Display is the user-facing `<BVn ...>` rendering and deliberately stays
    // id-free; only Debug carries identity.
    assert_eq!(a.to_string(), RustBV::symbol("alpha", 32).to_string());
}
