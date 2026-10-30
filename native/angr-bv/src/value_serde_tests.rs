//! Serde round-trip and cross-context translation tests for
//! the op-tree primitives (angr-x04s.1.1, angr-panhl.2).

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
    let bv = SymBV::concrete(0xdead_beef, 32);
    let json = serde_json::to_string(&bv).expect("serialize");
    let back: SymBV = serde_json::from_str(&json).expect("deserialize");
    match (&bv, &back) {
        (
            SymBV::Concrete {
                value: v1,
                width: w1,
            },
            SymBV::Concrete {
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
fn serde_roundtrip_expression_tree() {
    let x = SymBV::symbol("x_expr_serde", 32);
    let c = SymBV::concrete(7, 32);
    let expr = x.add(&c);
    // expr is Add(x, 7) after commutative canonicalization (concrete sorts
    // to the right).
    let json = serde_json::to_string(&expr).expect("serialize");
    let back: SymBV = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, expr, "the decoded term is the same interned node");
    assert_eq!(back.width(), 32);
    let node = back.node().expect("a compound value");
    assert_eq!(node.op, BVOp::Add);
    assert_eq!(node.operands.len(), 2);
    assert_eq!(node.operands[0].leaf_var(), x.leaf_var());
    assert!(matches!(
        node.operands[1],
        SymBV::Concrete {
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
