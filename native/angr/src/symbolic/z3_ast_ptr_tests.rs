use super::*;
use z3::ast::{Ast, Bool};

/// Constructing and dropping a `Z3AstPtr` should not crash and should
/// not affect the underlying AST's liveness while the original holder
/// still keeps a ref.
#[test]
fn test_from_borrowed_raw_inc_dec_balanced() {
    let ctx = Context::thread_local();
    let original = Bool::from_bool(true);
    let raw_ptr = original.get_z3_ast().as_ptr() as usize;

    // SAFETY: `original` keeps the AST alive; raw_ptr is its Z3_ast.
    let wrapped = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, raw_ptr) }.expect("non-null ptr");

    assert_eq!(wrapped.as_z3_ast().as_ptr() as usize, raw_ptr);
    drop(wrapped);

    // `original` still in scope — AST must still be valid. Reading its
    // boolean value would crash if Drop had taken the last ref.
    assert_eq!(original.as_bool(), Some(true));
}

#[test]
fn test_from_borrowed_raw_null_returns_none() {
    let ctx = Context::thread_local();
    // SAFETY: passing zero is the documented null case.
    let result = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, 0) };
    assert!(result.is_none(), "null ptr must yield None");
}

#[test]
fn test_clone_ref_independently_droppable() {
    let ctx = Context::thread_local();
    let original = Bool::from_bool(false);
    let raw_ptr = original.get_z3_ast().as_ptr() as usize;

    // SAFETY: `original` keeps the AST alive; raw_ptr is its Z3_ast.
    let a = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, raw_ptr) }.unwrap();
    let b = a.clone_ref();
    assert_eq!(a.as_z3_ast(), b.as_z3_ast());

    // Both handles should drop independently without underflow.
    drop(a);
    drop(b);

    // `original` still keeps the AST alive.
    assert_eq!(original.as_bool(), Some(false));
}

/// `bv_width` must report the Z3 sort's own size — this is the width the
/// solver-side `BV::wrap` callers use, so it has to be exact for every
/// width, not just the pointer-sized ones (angr-9ke6b.202).
#[test]
fn test_bv_width_reports_z3_sort_size() {
    let ctx = Context::thread_local();
    for width in [1u32, 8, 32, 64, 96, 128, 256, 512] {
        let original = z3::ast::BV::new_const(format!("bv_width_probe_{width}"), width);
        let raw_ptr = original.get_z3_ast().as_ptr() as usize;
        // SAFETY: `original` keeps the AST alive; raw_ptr is its Z3_ast.
        let wrapped = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, raw_ptr) }.unwrap();
        assert_eq!(
            wrapped.bv_width(),
            Some(width),
            "bv_width must match the Z3 sort size for a {width}-bit BV"
        );
    }
}

/// A Bool-sorted AST has no bit width. Callers rely on `None` here to route
/// to the Bool→1-bit lowering (or to bail) rather than BV-wrapping a Bool,
/// which trips Z3's process-aborting error handler (angr-58ks).
#[test]
fn test_bv_width_none_for_bool_sort() {
    let ctx = Context::thread_local();
    let original = Bool::from_bool(true);
    let raw_ptr = original.get_z3_ast().as_ptr() as usize;
    // SAFETY: `original` keeps the AST alive; raw_ptr is its Z3_ast.
    let wrapped = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, raw_ptr) }.unwrap();
    assert_eq!(wrapped.bv_width(), None, "Bool sort has no BV width");
    assert!(wrapped.is_bool());
}
