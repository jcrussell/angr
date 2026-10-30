// Unit tests for claripy_bridge.rs (claripy AST <-> RustBV/RustFloat bridge).
// Split from the parent module per rust-module-root-split-recipe.


use crate::symbolic::ToZ3;
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

/// Regression (angr-c3rd): exporting `And(Eq(x,5), Eq(y,7))` (two
/// Bool-converting operands) must keep BOTH operands. The old (None,None)
/// arm returned only `args[0]`, silently dropping the second comparison.
#[test]
fn test_export_and_of_two_bools_keeps_both_operands() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
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
        // And of two comparisons exports as a claripy Bool.
        assert!(
            ast.is_instance_of::<crate::claripy::ast::bool::Bool>(),
            "And(Eq,Eq) must export as a Bool, got {ast}"
        );
    });
}

/// Regression (angr-c3rd): a non-And/Or/Xor op with one Bool operand
/// (here `Eq(Eq(x,5), BVV(1,1))`) must dispatch through the real op, not
/// be rebuilt as `__add__`. An `__add__` rebuild yields a BV (length 1);
/// the correct `__eq__` yields a Bool (length None).
#[test]
fn test_export_eq_with_bool_operand_not_rebuilt_as_add() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
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
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
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
            // Export the leaf itself: it is the same clarirs node the
            // encoding references.
            let leaf = rustbv_to_claripy(py, &x, claripy.as_any()).unwrap();
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
                assert_eq!(got, expected, "{op:?}({v:#x}) width=8");
            }
        }
    });
}

/// angr-acoq identity: exporting the same symbolic clz/popcount or Float
/// RustBV twice returns the identical claripy AST. Pre-fix each call minted a
/// fresh BVS.
#[test]
fn test_export_symbolic_clz_and_fp_identity_stable() {
    use crate::symbolic::{FloatOpKind, FloatPrec};
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
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
        let fp = RustBV::float_node(FloatOpKind::Add, FloatPrec::F64, vec![y.clone(), y]);
        let f1 = rustbv_to_claripy(py, &fp, claripy.as_any()).unwrap();
        let f2 = rustbv_to_claripy(py, &fp, claripy.as_any()).unwrap();
        assert!(
            f1.bind(py).is(f2.bind(py)),
            "repeated Float export must return the identical claripy AST"
        );
    });
}

/// angr-ph300.2: round-trip `import(export(bv))` must be semantically identical
/// to `bv` for every node kind the bridge supports. Proven with a Z3
/// universality check — `bv != roundtrip` must be unsatisfiable — plus a width
/// guard so a silent width drift (the class of bug the AST_CACHE width check
/// defends against) is caught even when it happens to stay satisfiable-clean.
#[test]
fn test_roundtrip_import_export_preserves_semantics() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
        let ctx = SymContext::new_mock();

        // export bv -> claripy AST -> import back to a fresh RustBV
        let roundtrip = |bv: &RustBV| -> RustBV {
            let ast = rustbv_to_claripy(py, bv, claripy.as_any()).unwrap();
            claripy_to_rustbv(py, ast.bind(py), &ctx).unwrap()
        };
        let assert_equiv = |bv: &RustBV, label: &str| {
            let rt = roundtrip(bv);
            assert_eq!(bv.width(), rt.width(), "{label}: round-trip width drift");
            ctx.push();
            ctx.add_constraint(bv.to_z3_ast().unwrap().eq(rt.to_z3_ast().unwrap()).not());
            assert!(
                !ctx.is_sat(),
                "{label}: round-trip changed semantics (counter-example exists)"
            );
            ctx.pop();
        };

        let x = RustBV::symbolic(&ctx, "x", 32);
        let y = RustBV::symbolic(&ctx, "y", 32);
        let c5 = RustBV::concrete(5, 32);

        // BVV (concrete) across widths incl. zero + all-ones.
        assert_equiv(&RustBV::concrete(0, 8), "bvv_zero_8");
        assert_equiv(&RustBV::concrete(0xFF, 8), "bvv_ones_8");
        assert_equiv(&RustBV::concrete(0xDEAD_BEEF, 32), "bvv_32");
        assert_equiv(&RustBV::concrete(0x1122_3344_5566_7788, 64), "bvv_64");
        // BVS leaf.
        assert_equiv(&x, "bvs");
        // Arithmetic.
        assert_equiv(&x.add(&y, &ctx), "add");
        assert_equiv(&x.sub(&c5, &ctx), "sub");
        assert_equiv(&x.mul(&y, &ctx), "mul");
        assert_equiv(&x.neg(&ctx), "neg");
        // Division / remainder (signed + unsigned). Guards the asymmetric
        // claripy op-name mapping on both bridge directions: export emits
        // UDiv/SDiv/URem/SMod, import maps them back to udiv/sdiv/urem/srem.
        assert_equiv(&x.udiv(&y, &ctx), "udiv");
        assert_equiv(&x.sdiv(&y, &ctx), "sdiv");
        assert_equiv(&x.urem(&y, &ctx), "urem");
        assert_equiv(&x.srem(&y, &ctx), "srem");
        // Bitwise.
        assert_equiv(&x.and(&y, &ctx), "and");
        assert_equiv(&x.or(&y, &ctx), "or");
        assert_equiv(&x.xor(&y, &ctx), "xor");
        assert_equiv(&x.not(&ctx), "not");
        // Shifts (concrete and symbolic amount).
        assert_equiv(&x.shl(&RustBV::concrete(3, 32), &ctx), "shl");
        assert_equiv(&x.lshr(&RustBV::concrete(3, 32), &ctx), "lshr");
        assert_equiv(&x.ashr(&y, &ctx), "ashr_sym");
        // Rotates. Symbolic amount keeps the concrete-fold arm of
        // `define_rotate_pair!` from collapsing the node before export, so the
        // RotL/RotR -> RotateLeft/RotateRight op-name mapping is exercised.
        assert_equiv(&x.rotl(&y, &ctx), "rotl_sym");
        assert_equiv(&x.rotr(&y, &ctx), "rotr_sym");
        assert_equiv(&x.rotl(&RustBV::concrete(7, 32), &ctx), "rotl_const");
        // Byte reversal. Requires a byte-aligned width on the import side.
        assert_equiv(&x.reverse(&ctx), "reverse");
        // Comparisons -> width-1 Bool.
        assert_equiv(&x.eq(&c5, &ctx), "eq");
        assert_equiv(&x.ne(&y, &ctx), "ne");
        // Ordered comparisons, both signednesses and both directions. These are
        // the op names most prone to a mixed-up mapping (ULT vs UGT, SLT vs
        // ULT); an inverted or sign-swapped entry on either bridge direction
        // makes the round-trip semantically different, which the
        // counter-example check below catches.
        assert_equiv(&x.ult(&y, &ctx), "ult");
        assert_equiv(&x.ule(&y, &ctx), "ule");
        assert_equiv(&x.ugt(&y, &ctx), "ugt");
        assert_equiv(&x.uge(&y, &ctx), "uge");
        assert_equiv(&x.slt(&y, &ctx), "slt");
        assert_equiv(&x.sle(&y, &ctx), "sle");
        assert_equiv(&x.sgt(&y, &ctx), "sgt");
        assert_equiv(&x.sge(&y, &ctx), "sge");
        // Same ops against a concrete operand: `c5` has a clear high bit of 0,
        // so a signed/unsigned mix-up in the mapping is separable from an
        // operand-order mix-up.
        assert_equiv(&x.ult(&c5, &ctx), "ult_const");
        assert_equiv(&x.slt(&c5, &ctx), "slt_const");
        // Bool combinators (And/Or/Not over two comparisons).
        let b1 = x.eq(&c5, &ctx);
        let b2 = y.eq(&RustBV::concrete(7, 32), &ctx);
        assert_equiv(&b1.and(&b2, &ctx), "bool_and");
        assert_equiv(&b1.or(&b2, &ctx), "bool_or");
        assert_equiv(&b1.not(&ctx), "bool_not");
        // ITE selecting between two 32-bit operands on a Bool condition.
        assert_equiv(&b1.ite(&x, &y, &ctx), "ite");
        // Extract / Concat / ZeroExt / SignExt.
        assert_equiv(&x.extract(15, 8, &ctx), "extract");
        assert_equiv(&x.concat(&y, &ctx), "concat");
        assert_equiv(&x.clone().extend_into(64, false, &ctx), "zero_ext");
        assert_equiv(&x.clone().extend_into(64, true, &ctx), "sign_ext");
        // Nested compound expression across several node kinds.
        let nested = x.add(&y, &ctx).mul(&c5, &ctx).xor(&x, &ctx);
        assert_equiv(&nested, "nested");
    });
}

/// angr-ph300.2: a concrete BVV must survive the round-trip with its exact
/// value and width preserved (the Z3 check above proves equivalence, this
/// pins the literal so a value-corrupting import bug is unambiguous).
#[test]
fn test_roundtrip_concrete_value_exact() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
        let ctx = SymContext::new_mock();
        for (val, width) in [
            (0u128, 8u32),
            (0xFF, 8),
            (0x7F, 8),
            (0x1234, 16),
            (0xDEAD_BEEF, 32),
            (0x1122_3344_5566_7788, 64),
        ] {
            let bv = RustBV::concrete(val, width);
            let ast = rustbv_to_claripy(py, &bv, claripy.as_any()).unwrap();
            let rt = claripy_to_rustbv(py, ast.bind(py), &ctx).unwrap();
            assert_eq!(rt.width(), width, "width for {val:#x}/{width}");
            assert_eq!(
                rt.as_u128(),
                Some(val),
                "concrete value for {val:#x}/{width}"
            );
        }
    });
}

/// angr-n0irt.19: claripy's `BV.__floordiv__` (`//`) and `BV.__mod__` (`%`)
/// are UNSIGNED (counter to Python `int` semantics); only the `SDiv` / `SMod`
/// op-names are signed. `import.rs::claripy_to_rustbv` encodes that mapping in
/// a comment but no test locked it in. Pin it with the exact values cited in
/// that comment so a future op-name remap regresses loudly. The operand is a
/// pinned symbolic BVS (not two concrete BVVs) so claripy keeps the div/mod op
/// node instead of const-folding it away before it reaches the import arm.
#[test]
fn test_import_floordiv_mod_are_unsigned() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
        let ctx = SymContext::new_mock();

        // Symbolic operand v, pinned to 0xFFFF_FFFE (== -2 as signed i32).
        let bvs = claripy.call_method1("BVS", ("v", 32u32)).unwrap();
        let three = claripy.call_method1("BVV", (3u64, 32u32)).unwrap();
        let v = claripy_to_rustbv(py, &bvs, &ctx).unwrap();
        ctx.add_constraint(v.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(0xFFFF_FFFE, 32)));

        // Import each op AST and evaluate under the pinned constraint. Same
        // BVS "v" maps to the same z3 var, so eval resolves to a single value.
        let eval_op = |ast: &Bound<'_, PyAny>| -> u128 {
            let rt = claripy_to_rustbv(py, ast, &ctx).unwrap();
            ctx.eval(&rt).expect("pinned operand -> concrete eval")
        };

        // __floordiv__ / __mod__ are UNSIGNED: 0xFFFF_FFFE / 3 == 0x5555_5554,
        // 0xFFFF_FFFE % 3 == 2.
        let floordiv = bvs.call_method1("__floordiv__", (&three,)).unwrap();
        assert_eq!(eval_op(&floordiv), 0x5555_5554, "__floordiv__ must be udiv");
        let modv = bvs.call_method1("__mod__", (&three,)).unwrap();
        assert_eq!(eval_op(&modv), 2, "__mod__ must be urem");

        // SDiv / SMod are SIGNED: (-2) sdiv 3 == 0 (trunc toward zero),
        // (-2) srem 3 == -2 == 0xFFFF_FFFE. These must differ from the
        // unsigned results above, proving the arms are not conflated.
        let sdiv = claripy.call_method1("SDiv", (&bvs, &three)).unwrap();
        assert_eq!(eval_op(&sdiv), 0, "SDiv must be signed sdiv");
        let smod = claripy.call_method1("SMod", (&bvs, &three)).unwrap();
        assert_eq!(eval_op(&smod), 0xFFFF_FFFE, "SMod must be signed srem");
    });
}

/// angr-2a3i9: a long, non-shared claripy AST chain (repeated `ZeroExt`, no
/// common subexpressions) gets no help from the per-call import memo, so it
/// recurses to full tree depth in `claripy_to_rustbv`. Before the depth guard this class of input was the
/// reachable trigger for a raw-SIGSEGV failure mode (angr-2a3i9), mirroring
/// the real worker-thread crash (angr-h92bx) but on the
/// *unguarded* main thread. Build a chain well past
/// `MAX_IMPORT_RECURSION_DEPTH` and assert a catchable
/// `BridgeError::RecursionLimit` instead of a crash.
///
/// Runs on an explicit 8 MiB thread to mirror the CPython main-thread stack
/// the depth guard is calibrated against — the
/// default `cargo test` harness thread has a much smaller stack (~2 MiB,
/// the same undersized default that caused the worker-thread crash) and
/// would overflow *before* the ~4096-deep guarded
/// descent completes, which would fail this test for an unrelated reason.
#[test]
fn test_claripy_to_rustbv_long_chain_hits_depth_guard() {
    pyo3::Python::initialize();
    std::thread::Builder::new()
        // 256 MiB, not 8: the MAX_*_RECURSION_DEPTH=4096 guard was calibrated
        // against release frame sizes, which fit 4096 frames in 8 MiB (this test
        // passes in release at 8 MiB). A debug build's frames are several times
        // larger — the import direction's especially — so `cargo test` (debug)
        // overflowed 8 MiB *before* reaching the guard. The
        // reservation is virtual: only the ~4096 frames actually descended
        // commit. Sizing for the debug frame lets the guard fire (returning
        // RecursionLimit) in both profiles; the test still proves the guard,
        // not the stack (angr-rk5tw).
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            Python::attach(|py| {
                let ctx = SymContext::new_mock();

                // Chain depth intentionally well past MAX_IMPORT_RECURSION_DEPTH.
                // `ZeroExt` because clarirs flattens associative ops (`Add`,
                // `Xor`, `Concat`) into one n-ary node, which would make the
                // chain shallow and the test vacuous. Built in clarirs and
                // wrapped with `claripy_object` so the Python-side simplifier
                // never sees the chain.
                use crate::claripy::ast::GLOBAL_CONTEXT;
                use clarirs_core::prelude::*;
                let mut node = GLOBAL_CONTEXT.bvs("chain", 8).expect("BVS");
                for _ in 0..4300 {
                    node = GLOBAL_CONTEXT.zero_ext(&node, 1).expect("ZeroExt");
                }
                let ast = crate::claripy_bridge::claripy_object(py, node);

                let err = claripy_to_rustbv(py, &ast, &ctx).expect_err(
                    "a claripy AST chain deeper than MAX_IMPORT_RECURSION_DEPTH must return an \
                     error, not overflow the native stack",
                );
                assert!(
                    matches!(err, BridgeError::RecursionLimit(_)),
                    "expected BridgeError::RecursionLimit, got {err:?}"
                );
            });
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Regression (angr-9ke6b.38): `BVS(name, 1)` and `BoolS(name)` sharing an
/// explicit name must import to DISTINCT rust ids and export back to their own
/// claripy AST.
///
/// The symbol registry keys `name_to_info` by name+width; a `BoolS` leaf
/// imports with a hardcoded width of 1, so before the `SymbolKind` tag was
/// added to the key the second import resolved to the first's id and export
/// answered both with whichever AST registered first — a `BVS` where the
/// caller had a `Bool`, or vice versa.
#[test]
fn test_bvs_width1_and_bools_same_name_do_not_alias() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
        let ctx = SymContext::new_mock();

        // `explicit_name` is what makes the collision reachable: without it
        // claripy renames to `name_<counter>_<width>` and the two never share
        // a name in the first place.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("explicit_name", true).unwrap();
        let bvs = claripy
            .call_method("BVS", ("ke6b38_dual", 1u32), Some(&kwargs))
            .expect("BVS");
        let bools = claripy
            .call_method("BoolS", ("ke6b38_dual",), Some(&kwargs))
            .expect("BoolS");

        let bv_rust = claripy_to_rustbv(py, &bvs, &ctx).expect("import BVS");
        let bool_rust = claripy_to_rustbv(py, &bools, &ctx).expect("import BoolS");

        assert_ne!(
            bv_rust, bool_rust,
            "BVS(name, 1) and BoolS(name) must not import as one leaf"
        );

        // Each leaf must round-trip to its OWN claripy leaf, not the sibling's.
        let bv_back = rustbv_to_claripy(py, &bv_rust, claripy.as_any()).expect("export BVS");
        let bool_back = rustbv_to_claripy(py, &bool_rust, claripy.as_any()).expect("export BoolS");
        let bv_op: String = bv_back.bind(py).getattr("op").unwrap().extract().unwrap();
        let bool_op: String = bool_back.bind(py).getattr("op").unwrap().extract().unwrap();
        assert_eq!(bv_op, "BVS");
        assert_eq!(bool_op, "BoolS");
    });
}

/// angr-9ke6b.222: a purely Rust-minted leaf must export under its Rust name
/// verbatim, so that losing its registry entry is a *fidelity* loss (object
/// identity, annotations) rather than a soundness one.
///
/// `claripy.BVS(name, w)` without `explicit_name` renames to
/// `name_<counter>_<w>`. Re-importing such an AST after a registry miss mints a
/// Rust leaf named `name_<counter>_<w>`, and because `RustBV::from_parts`
/// derives the Z3 constant from the NAME, that leaf is a brand-new variable no
/// existing constraint binds — silently wrong (angr-izov2). The
/// assertions below pin both halves: the exported claripy name is the Rust name,
/// and a leaf re-minted from that name denotes the same Z3 constant despite
/// carrying a different `rust_id`.
#[test]
fn test_rust_minted_leaf_exports_under_its_rust_name() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
        let ctx = SymContext::new_mock();

        // A leaf that never came from claripy, e.g. `stdin_<n>_<i>` from the
        // native read proc: nothing has registered a claripy AST for it, so
        // export takes the mint path.
        let rust_name = "ke6b222_native_leaf";
        let minted = RustBV::symbolic(&ctx, rust_name, 8);

        let ast = rustbv_to_claripy(py, &minted, claripy.as_any()).expect("export");
        let exported_name: String = ast
            .bind(py)
            .getattr("args")
            .unwrap()
            .get_item(0)
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(
            exported_name, rust_name,
            "export must not let claripy rename the leaf to {rust_name}_<counter>_8"
        );

        // What a re-import builds from the exported name: the same leaf, since
        // a leaf is its name (angr-mx00p.12).
        let reminted = RustBV::symbolic(&ctx, exported_name.as_str(), 8);
        assert_eq!(minted, reminted, "the re-imported name is the same leaf");
        assert_eq!(
            minted.to_z3_ast().unwrap(),
            reminted.to_z3_ast().unwrap(),
            "a registry miss must not change which Z3 variable the leaf denotes"
        );
    });
}

/// angr-c7xno.20: direct coverage for `assumed_guard_to_claripy`, which had
/// none — its callers (`push_assumed_constraint_or_log` in
/// `exploration/state_api.rs`, `call_inspect_constraints` in
/// `callbacks/inspect.rs`) only exercised it indirectly, from outside this
/// module's test visibility.
///
/// Pins all four branch combinations (Bool guard / 1-bit-BV guard x
/// `is_true` true / false). The invariant that matters in every one of them:
/// the result is a claripy **Bool**, never the `NotImplemented` singleton —
/// `claripy.Not()` on a BV returns `NotImplemented` rather than raising, which
/// is the bug the BV branch exists to prevent (angr-op0dn.14.4.1).
#[test]
fn test_assumed_guard_to_claripy_all_four_branches() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let claripy = crate::claripy_bridge::claripy_module(py).expect("claripy module");
        let ctx = SymContext::new_mock();
        let x = RustBV::symbolic(&ctx, "c7xno20_x", 32);

        // A comparison exports as a claripy Bool (no `.length`); a 1-bit
        // Extract exports as a 1-bit BV. Those are the two shapes the
        // `.length` probe in `assumed_guard_to_claripy` discriminates.
        use crate::claripy::ast::bool::Bool;
        let bool_guard = x.eq(&RustBV::concrete(5, 32), &ctx);
        let bv_guard = x.extract(0, 0, &ctx);
        assert!(
            rustbv_to_claripy(py, &bool_guard, claripy.as_any())
                .unwrap()
                .bind(py)
                .is_instance_of::<Bool>(),
            "precondition: an eq guard must export as a Bool"
        );
        assert!(
            !rustbv_to_claripy(py, &bv_guard, claripy.as_any())
                .unwrap()
                .bind(py)
                .getattr("length")
                .unwrap()
                .is_none(),
            "precondition: a 1-bit extract guard must export as a BV"
        );

        // Pin x = 5 so every result below evaluates to a single concrete bit:
        // the Bool guard holds, and x's bit 0 is 1.
        ctx.add_constraint(x.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(5, 32)));

        // `want_op` is a set, not a single name: claripy's simplifier folds
        // `Not(a == b)` down to `a != b`, so the Bool/false branch is allowed
        // to surface either spelling. The semantic check below is what actually
        // pins the truth value.
        for (guard, shape, is_true, want_op, _want_eval) in [
            (&bool_guard, "Bool", true, &["__eq__"][..], 1u128),
            (&bool_guard, "Bool", false, &["Not", "__ne__"][..], 0),
            (&bv_guard, "BV", true, &["__eq__"][..], 1),
            (&bv_guard, "BV", false, &["__eq__"][..], 0),
        ] {
            let label = format!("{shape} guard, is_true={is_true}");
            let out = assumed_guard_to_claripy(py, guard, claripy.as_any(), is_true)
                .unwrap_or_else(|e| panic!("{label}: conversion failed: {e}"));
            let out = out.bind(py);

            // Every branch must produce a claripy Bool, never the
            // NotImplemented singleton or a BV.
            assert!(out.is_instance_of::<Bool>(), "{label}: result must be a Bool, got {out}");

            let op: String = out.getattr("op").unwrap().extract().unwrap();
            assert!(
                want_op.contains(&op.as_str()),
                "{label}: unexpected top-level op {op}, want one of {want_op:?}"
            );

            // The BV branch must compare against `BVV(is_true as u64, 1)` —
            // `guard != 0` semantics — rather than reaching for `Not`.
            if shape == "BV" {
                let rhs = out.getattr("args").unwrap().get_item(1).unwrap();
                assert_eq!(
                    rhs.getattr("length").unwrap().extract::<u32>().unwrap(),
                    1,
                    "{label}: rhs must be a 1-bit BVV"
                );
                assert_eq!(
                    rhs.getattr("args")
                        .unwrap()
                        .get_item(0)
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    u64::from(is_true),
                    "{label}: rhs bit must encode is_true"
                );
            }

            // Semantic check under the pinned x: a true-assumed guard must
            // evaluate to 1 and a false-assumed one to 0, in both shapes.
            {
                let back = claripy_to_rustbv(py, out, &ctx)
                    .unwrap_or_else(|e| panic!("{label}: re-import failed: {e}"));
                assert_eq!(
                    ctx.eval(&back),
                    Some(_want_eval),
                    "{label}: wrong truth value under pinned x = 5"
                );
            }
        }
    });
}
