//! In-module unit tests for `claripy_bridge/import.rs` (angr-sqfj8.18): the
//! width and arity guards of the import walk.

use super::*;

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

/// Python source for the `Fake` stand-in AST class shared by the
/// `PyModule::from_code` fixtures below.
const FAKE_AST_CLASS: &str = "class Fake:\n\
     \x20   def __init__(self, op, args, length):\n\
     \x20       self.op = op\n\
     \x20       self.args = args\n\
     \x20       self.length = length\n\
     \n";

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
            &std::ffi::CString::new(format!(
                "{FAKE_AST_CLASS}\
                 def concat(half):\n\
                 \x20   leaf = lambda: Fake('BVV', (0, half), half)\n\
                 \x20   return Fake('Concat', (leaf(), leaf()), 2 * half)\n"
            ))
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

/// angr-fs8kb.80: the `And` / `Or` arms folded their operand list by reaching
/// `RustBV::and` / `RustBV::or` directly, so the sweep that gave every other
/// multi-operand arm a [`check_same_width`] call never touched them
/// (angr-6cp06.81). Both now fold through `import_nary_op`, which performs the
/// check.
#[test]
fn test_bool_fold_refuses_mismatched_operand_widths() {
    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();
        let module = pyo3::types::PyModule::from_code(
            py,
            &std::ffi::CString::new(format!(
                "{FAKE_AST_CLASS}\
                 def bvv(width):\n\
                 \x20   return Fake('BVV', (0, width), width)\n\
                 \n\
                 def fold(op, widths):\n\
                 \x20   return Fake(op, tuple(bvv(w) for w in widths), max(widths))\n\
                 \n\
                 def empty(op):\n\
                 \x20   return Fake(op, (), 1)\n"
            ))
            .expect("cstring"),
            &std::ffi::CString::new("fake_bool_ast.py").expect("cstring"),
            &std::ffi::CString::new("fake_bool_ast").expect("cstring"),
        )
        .expect("module");
        let fold = module.getattr("fold").expect("fold");

        for op in ["And", "Or"] {
            let ast = fold
                .call1((op, vec![1u32, 8u32]))
                .expect("build mismatched fake AST");
            let msg = claripy_to_rustbv(py, &ast, &ctx)
                .expect_err("mismatched bool operand widths must be refused")
                .to_string();
            assert!(msg.contains(op), "{msg}");
            assert!(msg.contains("1-bit"), "{msg}");
            assert!(msg.contains("8-bit"), "{msg}");

            // Control: the same shape at equal widths still converts, so the
            // refusal above cannot be an artifact of the fake AST.
            let ok = fold
                .call1((op, vec![1u32, 1u32, 1u32]))
                .expect("build matched fake AST");
            let bv = claripy_to_rustbv(py, &ok, &ctx).expect("same-width bool fold is legal");
            assert_eq!(bv.width(), 1);
        }

        // Empty-operand identities are unchanged by the refactor: And -> 1, Or -> 0.
        let empty = module.getattr("empty").expect("empty");

        for (op, want) in [("And", 1u64), ("Or", 0u64)] {
            let ast = empty.call1((op,)).expect("build empty fake AST");
            let bv = claripy_to_rustbv(py, &ast, &ctx).expect("empty bool fold is legal");
            assert_eq!(bv.width(), 1, "{op}");
            assert_eq!(bv.as_u64(), Some(want), "{op}");
        }
    });
}

/// clarirs flattens `x + y + z` into one 3-argument `__add__` (and likewise
/// `__mul__`, `__and__`, `__or__`, `__xor__`, Bool `Xor`). The importer used to
/// require exactly two operands, so a longer chain failed to import and a
/// register synced from Python silently became its default instead.
#[test]
fn test_nary_assoc_ops_import_every_operand() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();
        let bvv = |v: u64| GLOBAL_CONTEXT.bvv(BitVec::from((v, 32))).expect("BVV");
        // Concrete operands so the folded result can be checked exactly.
        // `make_ast_exact` keeps the n-ary node rather than letting the
        // simplifier constant-fold it before the import ever sees it.
        let nary = |op: AstOp<'static>| {
            GLOBAL_CONTEXT
                .make_ast_exact(op, Default::default())
                .expect("n-ary node")
        };
        let cases = [
            (AstOp::Add(vec![bvv(1), bvv(2), bvv(3)]), 6u128),
            (AstOp::Mul(vec![bvv(2), bvv(3), bvv(4)]), 24),
            (AstOp::And(vec![bvv(0xff), bvv(0x0f), bvv(0x3c)]), 0x0c),
            (AstOp::Or(vec![bvv(1), bvv(2), bvv(4)]), 7),
            (AstOp::Xor(vec![bvv(1), bvv(3), bvv(4)]), 6),
        ];
        for (op, expected) in cases {
            let ast = nary(op);
            assert_eq!(ast.op().child_iter().count(), 3, "fixture must stay 3-ary: {ast:?}");
            let obj = claripy_object(py, ast.clone());
            let bv = claripy_to_rustbv(py, &obj, &ctx).expect("n-ary import");
            assert_eq!(bv.as_u128(), Some(expected), "{ast:?}");
        }
    });
}

/// The native clarirs walk (`import_clarirs`) and the attribute walk must
/// build equivalent values. A Python shim exposing only `op` / `args` /
/// `length` forces the attribute walk all the way down; Z3 then checks the two
/// results cannot differ.
#[test]
fn test_native_and_attribute_walks_agree() {
    use clarirs_core::prelude::*;
    use pyo3::types::PyDict;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"class Duck:
    def __init__(self, ast):
        self._ast = ast
        self.op = ast.op
        self.args = tuple(Duck(a) if hasattr(a, 'op') else a for a in ast.args)
        self.length = getattr(ast, 'length', None)
",
            Some(&globals),
            None,
        )
        .expect("define Duck");
        let duck = globals.get_item("Duck").expect("lookup").expect("Duck");

        let g = &*GLOBAL_CONTEXT;
        let x = g.bvs("walks_x", 32).expect("x");
        let y = g.bvs("walks_y", 32).expect("y");
        let k = |v: u64| g.bvv(BitVec::from((v, 32))).expect("k");
        let exprs = [
            g.add(g.mul(x.clone(), k(3)).expect("mul"), y.clone()).expect("add"),
            g.concat([g.extract(x.clone(), 15, 0).expect("ex"), g.extract(y.clone(), 31, 16).expect("ex")])
                .expect("concat"),
            g.ite(g.ult(x.clone(), y.clone()).expect("ult"), g.lshr(x.clone(), k(2)).expect("lshr"), g.sub(y.clone(), x.clone()).expect("sub"))
                .expect("ite"),
            g.zero_ext(g.extract(x.clone(), 7, 0).expect("ex"), 24).expect("zext"),
            g.byte_reverse(g.xor([x.clone(), y.clone(), k(0xff)]).expect("xor")).expect("rev"),
        ];
        for expr in exprs {
            let native_obj = claripy_object(py, expr.clone());
            let duck_obj = duck.call1((native_obj.clone(),)).expect("Duck(ast)");

            let ctx = SymContext::new();
            let native = claripy_to_rustbv(py, &native_obj, &ctx).expect("native walk");
            let walked = claripy_to_rustbv(py, &duck_obj, &ctx).expect("attribute walk");
            assert_eq!(native.width(), walked.width(), "{expr:?}");
            ctx.assume_true(&native.ne(&walked, &ctx));
            assert!(!ctx.is_sat(), "the two walks disagree on {expr:?}");
        }
    });
}

/// A node reached along many paths imports once (angr-mx00p.89): each level
/// of this chain uses the level below twice, so a walk that re-imports shared
/// subterms visits 2^LEVELS nodes and never finishes, where the memo visits
/// each of them once. The result is still the right value.
#[test]
fn test_shared_subterms_import_once() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    const LEVELS: u64 = 48;
    Python::initialize();
    Python::attach(|py| {
        let g = &*GLOBAL_CONTEXT;
        let x = g.bvs("shared_x", 32).expect("x");
        let y = g.bvs("shared_y", 32).expect("y");
        let k = |v: u64| g.bvv(BitVec::from((v, 32))).expect("k");
        let mut v = y.clone();
        for i in 1..=LEVELS {
            let up = g.add(v.clone(), k(1)).expect("add");
            let down = g.sub(v.clone(), k(1)).expect("sub");
            v = g.ite(g.ult(x.clone(), k(i)).expect("ult"), up, down).expect("ite");
        }
        let ctx = SymContext::new();
        let bv = claripy_to_rustbv(py, &claripy_object(py, v), &ctx).expect("shared import");
        // x == 0 takes every `up` branch.
        let x_bv = claripy_to_rustbv(py, &claripy_object(py, x), &ctx).expect("x");
        let y_bv = claripy_to_rustbv(py, &claripy_object(py, y), &ctx).expect("y");
        ctx.assume_true(&x_bv.eq(&RustBV::concrete(0, 32), &ctx));
        let expected = y_bv.add(&RustBV::concrete(u128::from(LEVELS), 32), &ctx);
        ctx.assume_true(&bv.ne(&expected, &ctx));
        assert!(!ctx.is_sat(), "the memoized import changed the value");
    });
}

/// A float predicate imports whole as a `RustBV::Ast` (angr-mx00p.11): its
/// operands are never turned into IEEE bit patterns, so `fp.eq(x, x)` keeps
/// IEEE semantics (false for NaN) instead of becoming a bit equality, which is
/// always true. Exporting the value hands back the very same clarirs node.
#[test]
fn test_float_predicate_imports_whole_and_exports_unchanged() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy::ast::base::Base;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();
        let x = GLOBAL_CONTEXT.fps("import_nan_x", FSort::f64()).expect("FPS");
        let eq = GLOBAL_CONTEXT.eq_(&x, &x).expect("fp.eq");
        let bv = claripy_to_rustbv(py, &claripy_object(py, eq.clone()), &ctx).expect("import");
        assert!(matches!(bv, RustBV::Ast { .. }), "expected RustBV::Ast, got {bv:?}");
        assert_eq!(bv.width(), 1);
        assert!(ctx.can_be_false(&bv), "NaN must make fp.eq(x, x) false");

        let claripy_mod = crate::claripy_bridge::claripy_module(py).expect("claripy");
        let exported = crate::claripy_bridge::rustbv_to_claripy(py, &bv, &claripy_mod).expect("export");
        let base = exported.bind(py).cast::<Base>().expect("a claripy AST");
        // The Bool is held in the 1-bit encoding and exported as the Bool:
        // the very node Python handed in.
        assert!(std::sync::Arc::ptr_eq(&base.get().ast(), &eq));
    });
}
