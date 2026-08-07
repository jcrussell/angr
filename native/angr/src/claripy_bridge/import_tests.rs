//! In-module unit tests for `claripy_bridge/import.rs` (angr-sqfj8.18).
//!
//! Scoped to `import_symbolic_leaf`'s hash-hit arm — the step-1 resolution
//! that reuses an already-registered symbol id. Both tests mutate the
//! process-global `SymbolicIdentityRegistry`, so every name/hash/id below is
//! unique to this file: cargo runs the suite multi-threaded and a shared key
//! would let a sibling test's registration decide the outcome.

use super::*;

use crate::symbolic::global_registry;

/// The canonical name wins over the name the incoming claripy AST carries.
///
/// This is the angr-izov2 contract: the Z3 constant behind a `Symbolic` is
/// built from its *name*, so rebinding a known id under claripy's renamed
/// string (`x_42_64`) would produce a variable Z3 considers unrelated to the
/// one every existing constraint mentions.
#[test]
fn test_import_symbolic_leaf_hash_hit_rebinds_under_canonical_name() {
    const SYMBOL_ID: u64 = 0x5F18_0001;
    const AST_HASH: i64 = 0x5F18_0001;
    const CANONICAL: &str = "sqfj8_18_canonical";

    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();
        store_claripy_ast_with_info(
            AST_HASH,
            SYMBOL_ID,
            CANONICAL,
            64,
            SymbolKind::BitVector,
            py.None(),
        );

        // Same hash, but the AST carries claripy's renamed string.
        let ast = py.None().into_bound(py);
        let bv = import_symbolic_leaf(
            &ast,
            AST_HASH,
            "sqfj8_18_canonical_42_64",
            64,
            SymbolKind::BitVector,
            &ctx,
        );

        let RustBV::Symbolic { id, name, .. } = &bv else {
            panic!("expected a Symbolic leaf, got {bv:?}");
        };
        assert_eq!(*id, SYMBOL_ID, "hash hit must reuse the registered id");
        assert_eq!(
            &**name, CANONICAL,
            "hash hit must rebuild under the canonical Rust name, not the \
             claripy-renamed one (angr-izov2)",
        );
    });
}

/// A hash mapping whose id has no canonical name must NOT rebind that id.
///
/// The torn state (`py_hash_to_rust_id` hit, `rust_id_to_name` miss) is what
/// `SymbolicIdentityRegistry::remove` transiently produces; it is unreachable
/// today because both sides run under the GIL, but the old
/// `unwrap_or(&rust_name)` fallback would have rebound the id under a
/// different Z3 constant if it ever were. Resolution now falls through to the
/// mint path, which registers a fresh id consistently — same Z3 constant,
/// no torn registry left behind.
#[test]
fn test_import_symbolic_leaf_falls_through_when_id_has_no_canonical_name() {
    const ORPHAN_ID: u64 = 0x5F18_0002;
    const AST_HASH: i64 = 0x5F18_0002;
    const NAME: &str = "sqfj8_18_orphan";

    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();

        // Build the torn state without reaching into private fields:
        // `register_by_id` populates only `rust_id_to_py`, and
        // `update_hash_mapping` only `py_hash_to_rust_id`. Neither writes
        // `rust_id_to_name`.
        global_registry().register_by_id(ORPHAN_ID, py.None());
        global_registry().update_hash_mapping(AST_HASH, ORPHAN_ID);
        assert!(
            global_registry().lookup_name_by_id(ORPHAN_ID).is_none(),
            "setup: the orphan id must have no canonical name",
        );

        let ast = py.None().into_bound(py);
        let bv = import_symbolic_leaf(&ast, AST_HASH, NAME, 64, SymbolKind::BitVector, &ctx);

        let RustBV::Symbolic { id, name, .. } = &bv else {
            panic!("expected a Symbolic leaf, got {bv:?}");
        };
        assert_ne!(
            *id, ORPHAN_ID,
            "an id with no canonical name must not be rebound",
        );
        assert_eq!(&**name, NAME, "fallback resolution keeps the Rust name");
        assert_eq!(
            global_registry().lookup_name_by_id(*id).as_deref(),
            Some(NAME),
            "the fallback path must re-register the leaf so the two maps agree \
             again",
        );
    });
}
