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

    assert_eq!(wrapped.as_usize(), raw_ptr);
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
    assert_eq!(a.as_usize(), b.as_usize());

    // Both handles should drop independently without underflow.
    drop(a);
    drop(b);

    // `original` still keeps the AST alive.
    assert_eq!(original.as_bool(), Some(false));
}
