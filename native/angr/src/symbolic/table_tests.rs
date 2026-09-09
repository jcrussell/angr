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

/// The copy-on-write map behind `fork` must stay invisible: a write on either
/// side unshares, so neither table observes the other's mutations.
#[test]
fn test_fork_is_copy_on_write_isolated() {
    let table = RustSymbolTable::new();
    let h1 = table.create_concrete(42, 32);

    let forked = table.fork();

    // Child insert doesn't leak into the parent...
    let h2 = forked.create_concrete(7, 32);
    assert_eq!(forked.len(), 2);
    assert_eq!(table.len(), 1);
    assert!(table.get(h2.id()).is_none());

    // ...and a parent remove doesn't tear the entry out from under the child.
    assert!(table.remove(h1.id()).is_some());
    assert_eq!(table.len(), 0);
    assert_eq!(forked.get(h1.id()).unwrap().as_u128(), Some(42));

    // `clear` unshares too, rather than emptying the sibling's map.
    forked.clear();
    let fresh = forked.fork();
    assert!(fresh.is_empty());
}

/// A child must not reissue an id the parent already bound — handles minted
/// before the fork stay resolvable to their original value in both tables.
/// (Sibling tables are independent id namespaces; only parent-inherited ids
/// have to agree.)
#[test]
fn test_fork_child_does_not_reissue_parent_ids() {
    let table = RustSymbolTable::new();
    let h1 = table.create_concrete(1, 32);
    let h2 = table.create_concrete(2, 32);

    let child = table.fork();
    let h3 = child.create_concrete(3, 32);

    assert_ne!(h3.id(), h1.id());
    assert_ne!(h3.id(), h2.id());
    assert_eq!(child.get(h1.id()).unwrap().as_u128(), Some(1));
    assert_eq!(child.get(h2.id()).unwrap().as_u128(), Some(2));
}

/// The flip side of the test above, and the reason `RustBVHandle`'s docstring
/// warns against mixing handles across states: two *siblings* of the same
/// parent both continue the parent's counter, so they mint identical ids for
/// unrelated values. Pinned because the docs now promise this is possible —
/// partitioning the id space later would make that warning a lie.
#[test]
fn test_sibling_forks_mint_colliding_ids() {
    let table = RustSymbolTable::new();
    table.create_concrete(1, 32);

    let left = table.fork();
    let right = table.fork();
    let l = left.create_concrete(0xaa, 32);
    let r = right.create_concrete(0xbb, 32);

    assert_eq!(l.id(), r.id(), "sibling tables share an id namespace");
    assert_eq!(left.get(l.id()).unwrap().as_u128(), Some(0xaa));
    assert_eq!(right.get(r.id()).unwrap().as_u128(), Some(0xbb));
}
