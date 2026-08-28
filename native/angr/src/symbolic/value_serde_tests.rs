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
fn serde_roundtrip_constrained() {
    let bv = RustBV::Constrained {
        id: 42,
        value: 7,
        width: 64,
    };
    let json = serde_json::to_string(&bv).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match back {
        RustBV::Constrained { id, value, width } => {
            assert_eq!(id, 42);
            assert_eq!(value, 7);
            assert_eq!(width, 64);
        }
        _ => panic!("variant changed across round-trip"),
    }
}

/// Rebuild a `Constrained` the only way live code ever does — snapshot
/// deserialization (`RustBVData::Constrained` → `RustBV::Constrained`).
fn constrained_via_snapshot(id: u64, value: u128, width: u32) -> RustBV {
    let json = serde_json::to_string(&RustBV::Constrained { id, value, width }).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

fn constrained_id(bv: &RustBV) -> Option<u64> {
    match bv {
        RustBV::Constrained { id, .. } => Some(*id),
        _ => None,
    }
}

/// angr-9ke6b.128: a same-width `truncate` and a full-width `extract` are the
/// identity, so they must hand back the `Constrained` untouched. Folding them
/// through `as_u128()` (which is `Some` for `Constrained`) would produce a bare
/// `Concrete` and drop the symbol `id` that `query_class` /
/// `memory::ite_builder` / `interpreter::concretize_cache` key off — the value
/// would silently start reading as fully concrete.
#[test]
fn identity_truncate_extract_preserve_constrained_id() {
    let ctx = SymContext::new_mock();
    let bv = constrained_via_snapshot(42, 7, 64);

    let truncated = bv.truncate(64, &ctx);
    assert_eq!(constrained_id(&truncated), Some(42), "same-width truncate");
    assert!(truncated.is_symbolic());
    assert_eq!(truncated.width(), 64);

    let extracted = bv.extract(63, 0, &ctx);
    assert_eq!(constrained_id(&extracted), Some(42), "full-width extract");
    assert!(extracted.is_symbolic());
    assert_eq!(extracted.width(), 64);

    // Same for the consuming variants, which are the ones the interpreter calls.
    let truncated = bv.clone().truncate_into(64, &ctx);
    assert_eq!(constrained_id(&truncated), Some(42), "truncate_into");
    let extracted = bv.extract_into(63, 0, &ctx);
    assert_eq!(constrained_id(&extracted), Some(42), "extract_into");
}

/// The flip side of the invariant above: a *narrowing* truncate / *partial*
/// extract produces a different value, so it is a new concrete leaf rather than
/// the same symbol — matching `zero_extend_into` / `sign_extend_into`, which
/// also fold a `Constrained` away once they actually change the width.
#[test]
fn narrowing_truncate_extract_fold_constrained_to_concrete() {
    let ctx = SymContext::new_mock();
    let bv = constrained_via_snapshot(42, 0xdead_beef, 64);

    let truncated = bv.truncate(16, &ctx);
    assert!(truncated.is_concrete());
    assert_eq!(truncated.as_u64(), Some(0xbeef));

    let extracted = bv.extract(31, 16, &ctx);
    assert!(extracted.is_concrete());
    assert_eq!(extracted.as_u64(), Some(0xdead));

    let extended = bv.zero_extend(96, &ctx);
    assert!(extended.is_concrete());
    assert_eq!(extended.as_u64(), Some(0xdead_beef));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn serde_roundtrip_symbolic() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let bv = RustBV::symbolic(&ctx, "x_serde", 32);
    let (orig_id, orig_width, orig_name) = match &bv {
        RustBV::Symbolic {
            id, width, name, ..
        } => (*id, *width, name.to_string()),
        _ => panic!("expected Symbolic"),
    };
    let json = serde_json::to_string(&bv).expect("serialize");
    // Deserialize under the same context (thread-local is still active).
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match &back {
        RustBV::Symbolic {
            id, width, name, ..
        } => {
            assert_eq!(*id, orig_id);
            assert_eq!(*width, orig_width);
            assert_eq!(name.as_ref(), orig_name.as_str());
        }
        _ => panic!("variant changed across round-trip"),
    }
    // The lazily-rebuilt AST must be usable: it should produce a Z3 AST
    // pointer (sanity check that BV::new_const succeeded under the
    // active thread-local context).
    let _ptr = back.to_z3_ast().get_z3_ast().as_ptr();
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn serde_roundtrip_expression_tree() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x_expr_serde", 32);
    let c = RustBV::concrete(7, 32);
    let expr = x.add(&c, &ctx);
    // expr is Expression(Add, [Symbolic(x), Concrete(7)]) after
    // commutative canonicalization (concrete sorts to the right).
    let json = serde_json::to_string(&expr).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match &back {
        RustBV::Expression {
            op,
            operands,
            width,
            ..
        } => {
            assert_eq!(*op, BVOp::Add);
            assert_eq!(*width, 32);
            assert_eq!(operands.len(), 2);
            assert!(matches!(operands[0], RustBV::Symbolic { .. }));
            assert!(matches!(
                operands[1],
                RustBV::Concrete {
                    value: 7,
                    width: 32
                }
            ));
        }
        _ => panic!("variant changed across round-trip"),
    }
}

#[cfg(feature = "vex-engine-z3")]
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

/// Regression (angr-ovqja.3): the persistent per-`Expression` Z3 AST memo must
/// not leak a stale-context AST. The memo caches the built `z3::ast::BV` for an
/// `Expression`; if the thread-local Z3 context is swapped (only `with_z3_context`
/// in tests — never production), `to_z3_ast()` must rebuild against the active
/// context rather than return the cached BV from the previous context.
///
/// Uses an Extract-over-concrete node so the rebuilt AST resolves to a fresh
/// context-local constant with NO `Symbolic` leaves (whose own `ast` cache would
/// confound the context check — that staleness predates this memo).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_expression_memo_no_stale_context_leak() {
    use z3::ast::Ast as Z3AstTrait;
    use z3::{Config, Context};

    // `with_z3_context` cannot be used here: its `Send + Sync` closure bound
    // exists precisely to forbid smuggling a Z3-bearing value (our `RustBV`)
    // across the boundary — yet the stale-memo path requires converting the
    // SAME `RustBV` in two contexts. Swap the thread-local context manually
    // and restore it on the way out (single-threaded test, so this is sound).
    let original = Context::thread_local();

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
    let cfg = Config::new();
    let new_ctx = Context::new(&cfg);
    let new_ctx_ptr = new_ctx.get_z3_context().as_ptr() as usize;
    assert_ne!(
        default_ctx, new_ctx_ptr,
        "test bug: new context equals default; not testing cross-context",
    );

    Context::set_thread_local(&new_ctx);
    let in_new_ctx = expr.to_z3_ast().get_ctx().get_z3_context().as_ptr() as usize;
    // Restore before asserting so a failure does not poison sibling tests.
    Context::set_thread_local(&original);
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

// --- Cross-context AST translation (angr-panhl.2 parallel kill-gate) ---
//
// `translate_into` is the per-task `Z3_translate` primitive for the
// shared-nothing parallel design (Option A). It must move a `RustBV` from
// one thread-local Z3 context to another with every constraint re-checking
// identical, and at a per-node cost in the spike's ballpark.

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_cross_context_eval() {
    use z3::ast::Ast as Z3AstTrait;
    use z3::{Config, Context};

    // Source context = default thread-local. Build a memory-load-shaped tree.
    let src = SymContext::new_mock();
    let x = RustBV::symbolic(&src, "x", 32);
    let rev = x.reverse(&src); // Expression { Reverse, [x] }
    let pin = x.eq(&RustBV::concrete(0x11223344, 32), &src);

    // Translate into a freshly-allocated, independent context. `Z3_translate`
    // takes explicit source/dest contexts, so translation does NOT depend on
    // which context is currently thread-local.
    let original = Context::thread_local();
    let cfg = Config::new();
    let target = Context::new(&cfg);
    let target_ptr = target.get_z3_context().as_ptr() as usize;
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target_ptr,
        "test bug: target equals source context",
    );
    let rev_t = rev.translate_into(&target);
    let pin_t = pin.translate_into(&target);

    // Every translated leaf AST must now belong to the target context.
    let leaf_ctx = match &rev_t {
        RustBV::Expression { operands, .. } => match &operands[0] {
            RustBV::Symbolic { ast, .. } => ast.get_ctx().get_z3_context().as_ptr() as usize,
            other => panic!("expected symbolic leaf, got {other:?}"),
        },
        other => panic!("expected expression, got {other:?}"),
    };

    // Evaluate the translated tree under the target context and restore the
    // thread-local before asserting (a failure must not poison sibling tests).
    Context::set_thread_local(&target);
    let ctx_b = SymContext::new_mock();
    ctx_b.add_constraint(pin_t.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    let got = ctx_b.eval(&rev_t);
    Context::set_thread_local(&original);

    assert_eq!(
        leaf_ctx, target_ptr,
        "translated leaf AST must live in the target context",
    );
    assert_eq!(
        got,
        Some(0x44332211),
        "cross-context Reverse(x) eval must match the single-context result",
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_roundtrips_through_fresh_context() {
    // The bead's no-op correctness gate: round-trip A -> B -> A through a
    // fresh independent context; every constraint must still re-check equal.
    // (Translating into the *same* context is unsupported — `Z3_translate`
    // returns null — so a genuine round trip must bounce through a distinct
    // context, exactly as a worker hand-off and hand-back would.)
    let ctx = SymContext::new_mock(); // values live in default thread-local A
    let x = RustBV::symbolic(&ctx, "x", 64);
    let rev = x.reverse(&ctx);
    let pin = x.eq(&RustBV::concrete(0x0123456789ABCDEF, 64), &ctx);

    let a = z3::Context::thread_local(); // handle to context A
    let b = z3::Context::new(&z3::Config::new());
    let rev_back = rev.translate_into(&b).translate_into(&a);
    let pin_back = pin.translate_into(&b).translate_into(&a);

    // Thread-local is still A, so the round-tripped values eval directly.
    ctx.add_constraint(pin_back.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev_back), Some(0xEFCDAB8967452301));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_concrete_identity() {
    // Concrete / Constrained carry no AST: translation is a cheap value clone
    // that stays valid in any context (it never references one).
    let target = z3::Context::new(&z3::Config::new());
    let c = RustBV::concrete(0xDEADBEEF, 32);
    let t = c.translate_into(&target);
    assert_eq!(t.as_u64(), Some(0xDEAD_BEEF));
    assert_eq!(t.width(), 32);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_per_node_cost() {
    use std::time::Instant;

    // Mimic a real mid-run memory-load constraint: Reverse(Concat(64 byte
    // BVSes)). Only the 64 Symbolic leaves carry context-bound ASTs that need
    // `Z3_translate`; the Concat/Reverse Expression nodes recurse and reset
    // their lazy memo (rebuilt under whichever context is active), so they are
    // nearly free. This is the favorable shape the kill-gate predicted.
    let ctx = SymContext::new_mock();
    let leaves: Vec<RustBV> = (0..64u32)
        .map(|i| RustBV::symbolic(&ctx, format!("b{i}"), 8))
        .collect();
    let mut acc = leaves[0].clone();
    for b in &leaves[1..] {
        acc = acc.concat(b, &ctx);
    }
    let expr = acc.reverse(&ctx);
    let n_leaves = leaves.len();
    let n_nodes = n_leaves + (n_leaves - 1) + 1; // leaves + concats + reverse

    let target = z3::Context::new(&z3::Config::new());
    let iters = 200u32;
    // Warm up one translation (allocator / first-touch) before timing.
    let _ = expr.translate_into(&target);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = expr.translate_into(&target);
    }
    let elapsed = start.elapsed();
    let per_node = elapsed.as_nanos() as f64 / (iters as f64 * n_nodes as f64);
    let per_leaf = elapsed.as_nanos() as f64 / (iters as f64 * n_leaves as f64);
    eprintln!(
        "translate_into cost: {n_nodes} nodes ({n_leaves} Z3 leaves), {iters} iters in {elapsed:?} \
         => {per_node:.0} ns/node, {per_leaf:.0} ns/leaf-translate",
    );
    // Loose sanity ceiling only — the precise number is the kill-gate
    // measurement (recorded in the bd note), not a tight CI assertion.
    assert!(
        per_node < 50_000.0,
        "translate cost {per_node:.0} ns/node is absurdly high — likely a regression",
    );
}
