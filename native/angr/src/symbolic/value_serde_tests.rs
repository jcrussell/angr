//! Serde round-trip, expression-memo and cross-context translation tests for
//! the op-tree primitives (angr-x04s.1.1, angr-panhl.2).

use super::value_tests_support::raw_extract_node;
use super::*;

// -----------------------------------------------------------------
// Serde round-trip tests for the op-tree primitives (angr-x04s.1.1).
// Each variant is constructed inside a fresh SymContext, serialized
// to JSON, deserialized back, and structurally compared to the
// original. The Symbolic variant's Z3 AST is reconstructed lazily
// — we verify that round-tripped Symbolic values still produce a
// valid Z3 AST via to_z3_ast().
// -----------------------------------------------------------------

#[test]
fn serde_roundtrip_concrete() {
    let bv = RustBV::concrete(0xdead_beef, 32);
    let json = serde_json::to_string(&bv).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match (&bv, &back) {
        (
            RustBV::Concrete {
                value: v1,
                width: w1,
            },
            RustBV::Concrete {
                value: v2,
                width: w2,
            },
        ) => {
            assert_eq!(v1, v2);
            assert_eq!(w1, w2);
        }
        _ => panic!("variant changed across round-trip"),
    }
}

#[test]
fn serde_roundtrip_symbolic() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let bv = RustBV::symbolic(&ctx, "x_serde", 32);
    let json = serde_json::to_string(&bv).expect("serialize");
    // Deserialize under the same context (thread-local is still active).
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.leaf_var(), bv.leaf_var());
    assert_eq!(back, bv, "the leaf round-trips to the same interned node");
    // The lazily-rebuilt AST must be usable: it should produce a Z3 AST
    // pointer (sanity check that the lowering succeeds under the active
    // thread-local context).
    let _ptr = back.to_z3_ast().get_z3_ast().as_ptr();
}

#[test]
fn serde_roundtrip_expression_tree() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x_expr_serde", 32);
    let c = RustBV::concrete(7, 32);
    let expr = x.add(&c, &ctx);
    // expr is Add(x, 7) after commutative canonicalization (concrete sorts
    // to the right).
    let json = serde_json::to_string(&expr).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, expr, "the decoded term is the same interned node");
    assert_eq!(back.width(), 32);
    let node = back.node().expect("a compound value");
    assert_eq!(node.op, BVOp::Add);
    assert_eq!(node.operands.len(), 2);
    assert_eq!(node.operands[0].leaf_var(), x.leaf_var());
    assert!(matches!(
        node.operands[1],
        RustBV::Concrete {
            value: 7,
            width: 32
        }
    ));
}

#[test]
fn serde_roundtrip_float_op() {
    // BVOp::Float carries kind + prec; verify it survives JSON
    // round-trip without losing fields (covers FloatOpKind +
    // FloatPrec derive correctness).
    let op = BVOp::Float {
        kind: FloatOpKind::ConvertItoF {
            src_bits: 32,
            signed: true,
        },
        prec: FloatPrec::F64,
    };
    let json = serde_json::to_string(&op).expect("serialize");
    let back: BVOp = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(op, back);
}

/// Regression (angr-ovqja.3): the persistent per-value Z3 AST memo must not
/// leak a stale-context AST. If the thread-local Z3 context is swapped
/// (`with_z3_config`), `to_z3_ast()` must rebuild
/// against the active context rather than return the cached BV from the
/// previous context.
#[test]
fn test_expression_memo_no_stale_context_leak() {
    use z3::ast::Ast as Z3AstTrait;
    use z3::{Config, Context};

    // The stale-memo path requires converting the SAME `RustBV` in two
    // contexts, but `with_z3_config`'s `Send + Sync` closure bound forbids
    // capturing a Z3-bearing value. `with_z3_config` runs the closure
    // synchronously on the calling thread, so wrapping the borrow to satisfy
    // the bound moves nothing across threads.
    struct SameThread<'a>(&'a RustBV);
    // SAFETY: only ever handed to `with_z3_config`, which calls its closure on
    // this thread before returning; the borrow never reaches another thread.
    unsafe impl Send for SameThread<'_> {}
    // SAFETY: as above.
    unsafe impl Sync for SameThread<'_> {}
    impl<'a> SameThread<'a> {
        // A method call makes the closure capture the whole wrapper; naming
        // the field would capture only the inner `&RustBV`.
        fn get(&self) -> &'a RustBV {
            self.0
        }
    }

    let expr = raw_extract_node(RustBV::concrete(0x11223344, 32), 7, 0);

    // First conversion in the default thread-local context populates the memo.
    let ast1 = expr.to_z3_ast();
    let default_ctx = ast1.get_ctx().get_z3_context().as_ptr() as usize;

    // Second conversion in the SAME context is served from the memo and must
    // still belong to that context.
    let ast1b = expr.to_z3_ast();
    assert_eq!(
        ast1b.get_ctx().get_z3_context().as_ptr() as usize,
        default_ctx,
        "same-context repeat conversion must stay in the default context",
    );

    // Swap to a freshly-allocated context. The memo holds a default-context BV,
    // so the guard must reject it and rebuild against the new context.
    let shared = SameThread(&expr);
    let (new_ctx_ptr, in_new_ctx) = z3::with_z3_config(&Config::new(), || {
        let expr = shared.get();
        (
            Context::thread_local().get_z3_context().as_ptr() as usize,
            expr.to_z3_ast().get_ctx().get_z3_context().as_ptr() as usize,
        )
    });
    assert_ne!(
        default_ctx, new_ctx_ptr,
        "test bug: new context equals default; not testing cross-context",
    );
    assert_eq!(
        in_new_ctx, new_ctx_ptr,
        "stale-context leak: memo returned a BV from the old context",
    );

    // Back in the default context the guard rejects the now-new-context memo
    // and rebuilds against the default context again.
    let ast3 = expr.to_z3_ast();
    assert_eq!(
        ast3.get_ctx().get_z3_context().as_ptr() as usize,
        default_ctx,
        "returning to the default context must rebuild against it",
    );
}
