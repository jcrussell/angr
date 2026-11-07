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

/// angr-5mnx3.9: `Concat` derives its result width from the operand list, so —
/// like `ZeroExt`/`SignExt` — the derivation is the guard site, and the guard
/// has to run on the *accumulator* at every step. Two operands that are each
/// individually under `MAX_BV_WIDTH` can sum past it; without the guard
/// `value_ops::concat_owned`'s unchecked `u32` add would hand back a `RustBV`
/// whose stored `width()` every later extract-bounds/eval/export check trusts.
#[test]
fn test_concat_arm_refuses_oversized_derived_width() {
    use clarirs_core::prelude::*;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;
    use crate::symbolic::MAX_BV_WIDTH;

    Python::initialize();
    Python::attach(|py| {
        let ctx = SymContext::new_mock();
        let concat = |half: u32| {
            let leaf = |n: &str| GLOBAL_CONTEXT.bvs(format!("concat_cap_{n}_{half}"), half).expect("BVS");
            let node = GLOBAL_CONTEXT
                .make_ast_exact(AstOp::Concat(vec![leaf("a"), leaf("b")]), Default::default())
                .expect("Concat");
            claripy_object(py, node)
        };

        // Each half is a legal BVS width; their sum is not.
        let err = claripy_to_rustbv(py, &concat(MAX_BV_WIDTH / 2 + 1), &ctx)
            .expect_err("a Concat past MAX_BV_WIDTH must be refused, not silently built");
        let msg = err.to_string();
        assert!(msg.contains("Concat"), "{msg}");
        assert!(msg.contains(&MAX_BV_WIDTH.to_string()), "{msg}");

        // Control: the same shape inside the cap still converts, so the
        // assertion above cannot pass just because the fixture is unusable.
        let bv = claripy_to_rustbv(py, &concat(8), &ctx).expect("a 16-bit Concat is legal");
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

/// Only a claripy AST of this binary's type imports (angr-vmi2l.222): an
/// object that merely exposes `op` / `args` is a type mismatch, not walked by
/// attribute, while an object wrapping a claripy AST in `.ast` (a
/// `SimActionObject`) imports as that AST (angr-vmi2l.221).
#[test]
fn test_only_claripy_asts_import() {
    use clarirs_core::prelude::*;
    use pyo3::types::PyDict;

    use crate::claripy::ast::GLOBAL_CONTEXT;
    use crate::claripy_bridge::claripy_object;

    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"class Duck:
    def __init__(self, op, args):
        self.op = op
        self.args = args

class Wrapper:
    def __init__(self, ast):
        self.ast = ast
",
            Some(&globals),
            None,
        )
        .expect("define fixtures");
        let ctx = SymContext::new_mock();

        let duck = globals
            .get_item("Duck")
            .expect("lookup")
            .expect("Duck")
            .call1(("BVV", (5u64, 32u32)))
            .expect("Duck(...)");
        let err = claripy_to_rustbv(py, &duck, &ctx).expect_err("a duck-typed AST is refused");
        assert!(matches!(err, BridgeError::TypeMismatch(_)), "{err}");
        assert_eq!(try_extract_bvv(&duck), None);

        let five = claripy_object(py, GLOBAL_CONTEXT.bvv(BitVec::from((5u64, 32))).expect("BVV"));
        let wrapped = globals
            .get_item("Wrapper")
            .expect("lookup")
            .expect("Wrapper")
            .call1((five,))
            .expect("Wrapper(...)");
        let bv = claripy_to_rustbv(py, &wrapped, &ctx).expect("a wrapped AST imports");
        assert_eq!(bv.as_u128(), Some(5));
        assert_eq!(try_extract_bvv(&wrapped), Some((5, 32)));
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
        ctx.assume_true(&x_bv.eq(&RustBV::concrete(0, 32)));
        let expected = y_bv.add(&RustBV::concrete(u128::from(LEVELS), 32));
        ctx.assume_true(&bv.ne(&expected));
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
