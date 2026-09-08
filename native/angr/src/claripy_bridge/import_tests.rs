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

/// angr-0jh0j.8: `ZeroExt`/`SignExt` derive their result width from a
/// caller-chosen `extend_bits`, so the derivation itself is the guard site —
/// there is no width *argument* for `check_bv_width` to sit on. Both halves
/// must refuse: the `u32` sum wrapping (which would come out narrower than the
/// source) and a non-wrapping sum past `MAX_BV_WIDTH`.
#[test]
fn test_extended_width_refuses_overflow_and_absurd_widths() {
    use crate::symbolic::MAX_BV_WIDTH;

    let msg = |r: Result<u32, BridgeError>| match r {
        Ok(w) => panic!("expected a refusal, got width {w}"),
        Err(e) => e.to_string(),
    };

    // Wraparound: 8 + (u32::MAX - 3) is 4 in wrapping arithmetic, i.e. a
    // *narrowing* "extension" the release profile would accept silently.
    let wrapped = msg(extended_width("ZeroExt", 8, u32::MAX - 3));
    assert!(wrapped.contains("ZeroExt"), "{wrapped}");
    assert!(wrapped.contains("overflows u32"), "{wrapped}");

    // No wraparound, but still an absurd sort for Z3 to allocate.
    let absurd = msg(extended_width("SignExt", 8, MAX_BV_WIDTH));
    assert!(absurd.contains("SignExt"), "{absurd}");
    assert!(absurd.contains(&MAX_BV_WIDTH.to_string()), "{absurd}");

    // The cap itself stays constructible, and a plain extension is untouched.
    assert_eq!(extended_width("ZeroExt", 8, MAX_BV_WIDTH - 8).unwrap(), MAX_BV_WIDTH);
    assert_eq!(extended_width("SignExt", 32, 32).unwrap(), 64);
}

/// angr-5mnx3.9: `Concat` derives its result width from the operand list, so —
/// like `ZeroExt`/`SignExt` — the derivation is the guard site, and the guard
/// has to run on the *accumulator* at every step. Two operands that are each
/// individually under `MAX_BV_WIDTH` can sum past it; without the guard
/// `value_ops::concat_owned`'s unchecked `u32` add would hand back a `RustBV`
/// whose stored `width()` every later extract-bounds/eval/export check trusts.
#[test]
fn test_concat_arm_refuses_oversized_derived_width() {
    use crate::symbolic::MAX_BV_WIDTH;

    // Each half is a legal BVV width; their sum is not.
    let half = MAX_BV_WIDTH / 2 + 1;

    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();
        let module = pyo3::types::PyModule::from_code(
            py,
            &std::ffi::CString::new(
                "class Fake:\n\
                 \x20   def __init__(self, op, args, length):\n\
                 \x20       self.op = op\n\
                 \x20       self.args = args\n\
                 \x20       self.length = length\n\
                 \x20   def __hash__(self):\n\
                 \x20       return id(self)\n\
                 \n\
                 def concat(half):\n\
                 \x20   leaf = lambda: Fake('BVV', (0, half), half)\n\
                 \x20   return Fake('Concat', (leaf(), leaf()), 2 * half)\n",
            )
            .expect("cstring"),
            &std::ffi::CString::new("fake_ast.py").expect("cstring"),
            &std::ffi::CString::new("fake_ast").expect("cstring"),
        )
        .expect("module");
        let ast = module
            .getattr("concat")
            .expect("concat")
            .call1((half,))
            .expect("build fake Concat AST");

        let err = claripy_to_rustbv(py, &ast, &ctx)
            .expect_err("a Concat past MAX_BV_WIDTH must be refused, not silently built");
        let msg = err.to_string();
        assert!(msg.contains("Concat"), "{msg}");
        assert!(msg.contains(&MAX_BV_WIDTH.to_string()), "{msg}");

        // Control: the same shape inside the cap still converts, so the
        // assertion above cannot pass just because the fake AST is unusable.
        let ok_ast = module
            .getattr("concat")
            .expect("concat")
            .call1((8u32,))
            .expect("build fake Concat AST");
        let bv = claripy_to_rustbv(py, &ok_ast, &ctx).expect("a 16-bit Concat is legal");
        assert_eq!(bv.width(), 16);
    });
}

/// angr-6cp06.81: unlike `Extract`/`Concat`, the ~25 binary-op arms and the
/// `If` arm had no width guard of their own — `value_ops.rs` only
/// `debug_assert_eq!`s operand-width equality, and its module doc names *this*
/// boundary as the reason that is enough. So in the shipped `release` profile
/// the check existed nowhere, and the "claripy validated it already" backstop
/// is gated behind claripy's `_d._DEBUG` performance toggle.
#[test]
fn test_check_same_width_refuses_mismatched_operands() {
    let a = RustBV::concrete(0, 32);
    let b = RustBV::concrete(0, 64);

    let err = match check_same_width("__add__", &a, &b) {
        Ok(()) => panic!("expected a refusal for 32-bit vs 64-bit operands"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("__add__"), "{err}");
    assert!(err.contains("32-bit"), "{err}");
    assert!(err.contains("64-bit"), "{err}");

    // Equal widths pass, including the 1-bit Bool operands the comparison and
    // `If` arms feed it.
    assert!(check_same_width("ULT", &a, &RustBV::concrete(1, 32)).is_ok());
    assert!(
        check_same_width("If", &RustBV::concrete(1, 1), &RustBV::concrete(0, 1)).is_ok()
    );
}
