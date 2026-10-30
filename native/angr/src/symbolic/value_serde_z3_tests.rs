//! The solver-facing half of `bv/value_serde_tests.rs`: these tests lower or solve, so
//! they stay in `symbolic` (angr-mx00p.116.7).

use crate::bv::*;
// Serde round-trip and cross-context translation tests for
// the op-tree primitives (angr-x04s.1.1, angr-panhl.2).

use crate::symbolic::ToZ3;
#[test]
fn serde_roundtrip_symbolic() {
    use z3::ast::Ast as Z3AstTrait;
    let bv = RustBV::symbol("x_serde", 32);
    let json = serde_json::to_string(&bv).expect("serialize");
    // Deserialize under the same context (thread-local is still active).
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.leaf_var(), bv.leaf_var());
    assert_eq!(back, bv, "the leaf round-trips to the same interned node");
    // The lazily-rebuilt AST must be usable: it should produce a Z3 AST
    // pointer (sanity check that the lowering succeeds under the active
    // thread-local context).
    let _ptr = back.to_z3_ast().unwrap().get_z3_ast().as_ptr();
}
