use super::*;

#[test]
fn test_symbol_table_creation() {
    let table = RustSymbolTable::new();
    assert!(table.is_empty());
}

#[test]
fn test_create_concrete() {
    let table = RustSymbolTable::new();
    let handle = table.create_concrete(42, 32);
    assert_eq!(handle.width(), 32);
    assert!(handle.is_concrete());
    assert_eq!(handle.concrete(), Some(42));
}

#[test]
fn test_create_symbolic() {
    let table = RustSymbolTable::new();
    let ctx = SymContext::new_mock();
    let handle = table.create_symbolic(&ctx, "x", 32);
    assert_eq!(handle.width(), 32);
    assert!(!handle.is_concrete());
}

#[test]
fn test_operations() {
    let table = RustSymbolTable::new();
    let ctx = SymContext::new_mock();

    let h1 = table.create_concrete(10, 32);
    let h2 = table.create_concrete(32, 32);

    let h3 = table.op_add(h1.id(), h2.id(), &ctx).unwrap();
    assert_eq!(h3.width(), 32);
    assert!(h3.is_concrete());
    assert_eq!(h3.concrete(), Some(42));
}

#[test]
fn test_fork() {
    let table = RustSymbolTable::new();
    let h1 = table.create_concrete(42, 32);

    let forked = table.fork();
    assert_eq!(forked.len(), 1);
    assert!(forked.get(h1.id()).is_some());
}
