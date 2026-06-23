// Unit tests for claripy_bridge.rs (claripy AST <-> RustBV/RustFloat bridge).
// Split from the parent module per rust-mod-tests-sibling-extraction.

use std::sync::Arc;

use super::*;
use crate::symbolic::{RustBV, SymContext};

#[test]
fn test_extract_int_value_small() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let val = 42i64.into_pyobject(py).unwrap();
        assert_eq!(extract_int_value(val.into_any().clone()).unwrap(), 42);
    });
}

/// angr-cxw7: an integer needing more than 128 bits must be rejected, not
/// silently truncated to its low 128 bits. `1 << 200` previously imported
/// as 0.
#[test]
fn test_extract_int_value_above_128_bits_errors() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        // 2**200, built in Python so it stays an arbitrary-precision int.
        let big = py
            .eval(c"1 << 200", None, None)
            .expect("eval 1<<200")
            .into_any();
        let err = extract_int_value(big).expect_err("must reject >128-bit int");
        assert!(matches!(err, BridgeError::InvalidArgs(_)));

        // A 256-bit-wide value whose magnitude still fits in 128 bits is OK.
        let small = py.eval(c"1 << 100", None, None).expect("eval 1<<100");
        assert_eq!(extract_int_value(small.into_any()).unwrap(), 1u128 << 100);
    });
}

/// Regression for angr-c3rd: exporting `And(Eq(x,5), Eq(y,7))` (two
/// Bool-converting operands) must keep BOTH operands. The old (None,None)
/// arm returned only `args[0]`, silently dropping the second comparison.
#[test]
fn test_export_and_of_two_bools_keeps_both_operands() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return, // claripy not importable in this env — skip
        };
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let y = RustBV::symbolic(&ctx, "y", 32);
        let eq1 = x.eq(&RustBV::concrete(5, 32), &ctx);
        let eq2 = y.eq(&RustBV::concrete(7, 32), &ctx);
        let expr = eq1.and(&eq2, &ctx);

        let ast = rustbv_to_claripy(py, &expr, claripy.as_any()).unwrap();
        let ast = ast.bind(py);

        // Both x and y must survive the export (bug dropped y entirely).
        let vars = ast.getattr("variables").unwrap();
        assert_eq!(
            vars.len().unwrap(),
            2,
            "And(Eq,Eq) export must reference both operands' variables"
        );
        // And of two 1-bit BVs is itself a 1-bit BV.
        let length: Option<u32> = ast.getattr("length").ok().and_then(|l| l.extract().ok());
        assert_eq!(length, Some(1), "And of two Bool->BV(1) is a 1-bit BV");
    });
}

/// Regression for angr-c3rd: a non-And/Or/Xor op with one Bool operand
/// (here `Eq(Eq(x,5), BVV(1,1))`) must dispatch through the real op, not
/// be rebuilt as `__add__`. An `__add__` rebuild yields a BV (length 1);
/// the correct `__eq__` yields a Bool (length None).
#[test]
fn test_export_eq_with_bool_operand_not_rebuilt_as_add() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "x", 32);
        let inner = x.eq(&RustBV::concrete(5, 32), &ctx); // 1-bit Eq (Bool on export)
        let expr = inner.eq(&RustBV::concrete(1, 1), &ctx); // Eq(Bool-ish, BV1)

        let ast = rustbv_to_claripy(py, &expr, claripy.as_any()).unwrap();
        let ast = ast.bind(py);

        // A Bool has length None; an erroneous __add__ rebuild would be BV(1).
        let length: Option<u32> = ast.getattr("length").ok().and_then(|l| l.extract().ok());
        assert_eq!(
            length, None,
            "Eq with a Bool operand must export as a Bool (__eq__), not a BV (__add__)"
        );
        // x must still be present.
        let vars = ast.getattr("variables").unwrap();
        assert_eq!(vars.len().unwrap(), 1);
    });
}

/// angr-acoq soundness: a symbolic clz/ctz/popcount with width<=64 must
/// export as an encoding tied to the operand, so Python-side eval matches
/// the true bit-count for every concrete operand value. Pre-fix this was a
/// fresh unconstrained BVS that could yield Rust-infeasible values.
#[test]
fn test_export_symbolic_bitcount_is_sound() {
    use crate::symbolic::BVOp;
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let ctx = SymContext::new_mock();
        let width: u32 = 8;

        // Evaluate `ast` with its single BVS leaf pinned to `v`.
        let eval_with = |ast: &Bound<'_, PyAny>, leaf: &Bound<'_, PyAny>, v: u128| -> u128 {
            let bvv = claripy.call_method1("BVV", (v as i64, width)).unwrap();
            let cond = leaf.call_method1("__eq__", (&bvv,)).unwrap();
            let s = claripy.call_method0("Solver").unwrap();
            s.call_method1("add", (cond,)).unwrap();
            let res = s.call_method1("eval", (ast, 1u32)).unwrap();
            res.get_item(0).unwrap().extract().unwrap()
        };
        // Pull the single BVS leaf out of an exported encoding.
        let bvs_leaf = |ast: &Bound<'_, PyAny>| -> Py<PyAny> {
            let leaves = ast.call_method0("leaf_asts").unwrap();
            for leaf in leaves.try_iter().unwrap() {
                let leaf = leaf.unwrap();
                let op: String = leaf.getattr("op").unwrap().extract().unwrap();
                if op == "BVS" {
                    return leaf.unbind();
                }
            }
            panic!("no BVS leaf found in exported encoding");
        };

        for op in [BVOp::Clz, BVOp::Ctz, BVOp::Popcount] {
            let x = RustBV::symbolic(&ctx, "x", width);
            let expr = match op {
                BVOp::Clz => x.clz(&ctx),
                BVOp::Ctz => x.ctz(&ctx),
                BVOp::Popcount => x.popcount(&ctx),
                _ => unreachable!(),
            };
            let ast = rustbv_to_claripy(py, &expr, claripy.as_any()).unwrap();
            let ast = ast.bind(py);
            let leaf = bvs_leaf(ast);
            let leaf = leaf.bind(py);
            for v in [0u128, 1, 2, 0x80, 0x0F, 0xF0, 0xAA, 0xFF] {
                let got = eval_with(ast, leaf, v);
                let b = v as u8;
                let expected = match op {
                    BVOp::Clz => b.leading_zeros() as u128,
                    BVOp::Ctz => {
                        if b == 0 {
                            8
                        } else {
                            b.trailing_zeros() as u128
                        }
                    }
                    BVOp::Popcount => b.count_ones() as u128,
                    _ => unreachable!(),
                };
                assert_eq!(got, expected, "{:?}({:#x}) width=8", op, v);
            }
        }
    });
}

/// angr-acoq identity: exporting the same symbolic clz/popcount or Float
/// RustBV twice returns the identical claripy AST (stabilized via the
/// EXPRESSION_BY_OPERANDS_PTR cache). Pre-fix each call minted a fresh BVS.
#[test]
fn test_export_symbolic_clz_and_fp_identity_stable() {
    use crate::symbolic::{BVOp, FloatOpKind, FloatPrec};
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = match py.import("claripy") {
            Ok(m) => m,
            Err(_) => return,
        };
        let ctx = SymContext::new_mock();

        // popcount (sound encoding path)
        let x = RustBV::symbolic(&ctx, "x", 32);
        let pc = x.popcount(&ctx);
        let a1 = rustbv_to_claripy(py, &pc, claripy.as_any()).unwrap();
        let a2 = rustbv_to_claripy(py, &pc, claripy.as_any()).unwrap();
        assert!(
            a1.bind(py).is(a2.bind(py)),
            "repeated popcount export must return the identical claripy AST"
        );

        // Float result (unconstrained-but-stable BVS path)
        let y = RustBV::symbolic(&ctx, "y", 64);
        let fp = RustBV::Expression {
            id: RustBV::EXPRESSION_ID,
            width: 64,
            op: BVOp::Float {
                kind: FloatOpKind::Add,
                prec: FloatPrec::F64,
            },
            operands: Arc::<[RustBV]>::from(vec![y.clone(), y]),
            memo: Default::default(),
        };
        let f1 = rustbv_to_claripy(py, &fp, claripy.as_any()).unwrap();
        let f2 = rustbv_to_claripy(py, &fp, claripy.as_any()).unwrap();
        assert!(
            f1.bind(py).is(f2.bind(py)),
            "repeated Float export must return the identical claripy AST"
        );
    });
}
