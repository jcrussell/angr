//! `PartialEq` / `Debug` identity semantics for every `RustBV` variant
//! (angr-sqfj8.96).

use super::value_tests_support::bv_shape;
use super::*;

/// angr-sqfj8.96: `PartialEq` used to short-circuit to `false` whenever
/// either side had no `as_u128()`, so `x == x` was false for every
/// `Symbolic`/`Expression` — a latent trap for any `contains`/dedup caller.
#[test]
fn test_partial_eq_is_reflexive_for_every_variant() {
    let ctx = SymContext::new_mock();
    let concrete = RustBV::concrete(5, 32);
    let symbolic = RustBV::symbolic(&ctx, "reflexive_sym", 32);
    let constrained = RustBV::Constrained {
        id: 7,
        value: 5,
        width: 32,
    };
    let expression = symbolic.add(&RustBV::concrete(1, 32), &ctx);
    assert!(
        matches!(expression, RustBV::Expression { .. }),
        "test setup: symbolic + concrete should build an Expression, got {expression:?}"
    );

    for bv in [&concrete, &symbolic, &constrained, &expression] {
        assert_eq!(bv, bv, "PartialEq must be reflexive for {bv:?}");
        assert_eq!(
            bv,
            &bv.clone(),
            "a clone must compare equal to its source for {bv:?}"
        );
    }
}

/// The three equality classes from the `PartialEq for RustBV` doc: known-value
/// (Concrete/Constrained), leaf-symbol identity (Symbolic), node identity
/// (Expression). Nothing compares equal across a class boundary.
#[test]
fn test_partial_eq_classes_are_disjoint_and_identity_based() {
    let ctx = SymContext::new_mock();

    // Class 1: value-based. A Constrained equals the Concrete holding its value.
    let concrete = RustBV::concrete(5, 32);
    let constrained = RustBV::Constrained {
        id: 7,
        value: 5,
        width: 32,
    };
    assert_eq!(concrete, constrained);
    assert_eq!(constrained, concrete);
    assert_ne!(concrete, RustBV::concrete(5, 64), "width must be compared");
    assert_ne!(concrete, RustBV::concrete(6, 32));

    // Class 2: Symbolic is keyed by allocated id + width; `name` is debug-only.
    let s1 = RustBV::symbolic_with_id(11, "alpha", 32);
    let s1_other_name = RustBV::symbolic_with_id(11, "beta", 32);
    let s2 = RustBV::symbolic_with_id(12, "alpha", 32);
    let s1_wide = RustBV::symbolic_with_id(11, "alpha", 64);
    assert_eq!(s1, s1_other_name, "same id + width is the same leaf symbol");
    assert_ne!(s1, s2, "distinct ids are distinct symbols");
    assert_ne!(s1, s1_wide, "width must be compared");

    // Class 3: Expression is node identity, NOT a structural walk — two
    // independently built but structurally identical trees compare unequal.
    let sym = RustBV::symbolic(&ctx, "classes_sym", 32);
    let e1 = sym.add(&RustBV::concrete(1, 32), &ctx);
    let e2 = sym.add(&RustBV::concrete(1, 32), &ctx);
    assert_eq!(
        bv_shape(&e1),
        bv_shape(&e2),
        "test setup: both expressions should be structurally identical"
    );
    assert_ne!(
        e1, e2,
        "Expression equality is node identity, not structure"
    );
    assert_eq!(e1, e1.clone(), "clone shares the operand Arc");

    // Cross-class: never equal, in either direction.
    for (a, b) in [
        (&concrete, &s1),
        (&concrete, &e1),
        (&constrained, &s1),
        (&constrained, &e1),
        (&s1, &e1),
    ] {
        assert_ne!(a, b, "{a:?} and {b:?} are in different equality classes");
        assert_ne!(b, a, "{b:?} and {a:?} are in different equality classes");
    }
}

/// `id` is the identity key for the two leaf variants (`query_class` matches on
/// it, and it is what the claripy registry keys off), so `Debug` must surface
/// it — otherwise two distinct symbols that share a name/value + width print
/// identically in logs and assertion failures (angr-sqfj8.99).
#[test]
fn test_debug_includes_symbol_id_for_leaf_variants() {
    let a = RustBV::symbolic_with_id(11, "alpha", 32);
    let b = RustBV::symbolic_with_id(12, "alpha", 32);
    let da = format!("{a:?}");
    let db = format!("{b:?}");
    assert_eq!(da, "Symbolic(#11, alpha, 32)");
    assert_ne!(da, db, "distinct symbol ids must print differently");

    let c = RustBV::Constrained {
        id: 7,
        value: 0x2a,
        width: 32,
    };
    let d = RustBV::Constrained {
        id: 8,
        value: 0x2a,
        width: 32,
    };
    assert_eq!(format!("{c:?}"), "Constrained(#7, 0x2a, 32)");
    assert_ne!(
        format!("{c:?}"),
        format!("{d:?}"),
        "distinct constrained ids must print differently"
    );

    // Display is the user-facing `<BVn ...>` rendering and deliberately stays
    // id-free; only Debug carries identity.
    assert_eq!(a.to_string(), b.to_string());
}
