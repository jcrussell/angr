//! In-module unit tests for `claripy_bridge/export.rs`.
//!
//! The width/coercion decision-seam tests that used to live here moved to
//! `export/width_decisions_tests.rs` alongside their subject (angr-5mnx3.11).

/// angr-fs8kb.81: `compare_dunder` (`BVOp::Eq` / `BVOp::Ne`) must reject a
/// dunder that answers `NotImplemented` — or `None` — instead of returning it
/// verbatim as if it were a claripy AST, the same way its `bitwise_dunder`
/// sibling already did.
///
/// Driven through the real export path by registering a stand-in object for a
/// symbolic leaf in the identity cache: `ensure_claripy_ast` passes anything
/// with an `op` attribute through untouched (it is tagged `SILENT(cat-c)` for
/// exactly this reason), so the stand-in reaches the dispatch as `args[0]` and
/// its `__eq__` / `__ne__` are what the closure calls.
///
/// The `None` half is the one the bool unwrap could swallow: `extract::<bool>`
/// is truthiness-shaped, so an unguarded `None` becomes `BoolV(False)` — a
/// wrong answer rather than a bogus object. Hence the rejection runs first.
#[test]
fn compare_dunder_rejects_not_implemented_and_none() {
    use crate::claripy_bridge::cache::{evict_claripy_ast, store_claripy_ast_with_info};
    use crate::symbolic::{RustBV, SymContext, SymbolKind};
    use pyo3::prelude::*;

    pyo3::Python::initialize();
    Python::attach(|py| {
        let Ok(claripy) = py.import("angr.rustylib.claripy") else {
            return; // claripy not importable in this env -- skip
        };
        // Two stand-ins, one per non-AST return value the guard must catch.
        let fakes = py
            .eval(
                c"[type('NotImpl', (), {'op': 'BVS', 'length': 32,
                     '__eq__': lambda s, o: NotImplemented,
                     '__ne__': lambda s, o: NotImplemented,
                     '__hash__': None})(),
                   type('NoneRet', (), {'op': 'BVS', 'length': 32,
                     '__eq__': lambda s, o: None,
                     '__ne__': lambda s, o: None,
                     '__hash__': None})()]",
                None,
                None,
            )
            .expect("build stand-in objects");

        for (i, fake) in fakes.try_iter().unwrap().enumerate() {
            let fake = fake.unwrap();
            let ctx = SymContext::new_mock();
            let name = format!("fs8kb81_x{i}");
            let x = RustBV::symbolic(&ctx, name.clone(), 32);
            let RustBV::Symbolic { id, .. } = &x else {
                panic!("RustBV::symbolic must produce a Symbolic leaf");
            };
            let id = *id;
            store_claripy_ast_with_info(
                // A hash that cannot collide with a real claripy AST's.
                -(i as i64) - 1,
                id,
                &name,
                32,
                SymbolKind::BitVector,
                fake.unbind(),
            );

            let five = RustBV::concrete(5, 32);
            for (label, expr) in [("Eq", x.eq(&five, &ctx)), ("Ne", x.ne(&five, &ctx))] {
                let err = super::rustbv_to_claripy(py, &expr, claripy.as_any())
                    .expect_err("a non-AST dunder result must not export as an AST");
                let msg = err.to_string();
                assert!(
                    msg.contains("returned NotImplemented"),
                    "{label} (stand-in {i}) must raise the shared non-AST diagnostic, got: {msg}"
                );
            }
            evict_claripy_ast(id);
        }
    });
}
