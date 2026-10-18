//! The Bool tag on Rust-side symbol names.

use std::borrow::Cow;

use super::*;

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
    // This splits the variable `RustBV::from_parts` builds from the name
    // (angr-9ke6b.223).
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
    // `RustBV::from_parts` decodes the tag to decide whether to build a BoolS,
    // so the two halves must round-trip exactly — and the recovered string
    // must be claripy's own name, the BoolS Python holds (angr-9ke6b.223).
    let tagged = SymbolKind::Bool.rust_symbol_name("flag");
    assert_eq!(strip_bool_symbol_name(&tagged), Some("flag"));
}

#[test]
fn strip_bool_symbol_name_ignores_bitvector_names() {
    // A BV leaf must never be decoded as a Bool, or its term changes sort.
    let untagged = SymbolKind::BitVector.rust_symbol_name("flag");
    assert_eq!(strip_bool_symbol_name(&untagged), None);
    // Rust-minted names (stdin bytes, unconstrained fills) are BVs too.
    assert_eq!(strip_bool_symbol_name("stdin_0_8"), None);
}
