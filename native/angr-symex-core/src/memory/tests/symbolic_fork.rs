//! Fork isolation for `SymbolicMemory`'s symbolic sidecars: mutating one side
//! of a fork must not be visible in the other for `symbolic_spans`,
//! `imported_addrs` or permission flags.
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56).

use super::super::*;

/// angr-24e7: writes to symbolic_spans in a forked memory must not leak
/// back into the parent. Regression test for fork field-by-field cloning.
#[test]
fn test_fork_symbolic_spans_isolation() {
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);
    parent.map(0x2000, 0x1000, Permission::RWX);

    // Parent imports a 64-bit (8-byte) wide symbolic value at 0x1000.
    // import_symbolic_value populates symbolic_spans for bytes 1..8.
    let parent_sym = RustBV::symbol("parent_wide", 64);
    parent
        .import_symbolic_value(0x1000, parent_sym, None)
        .unwrap();
    // Sanity: spans for 0x1001..0x1008 exist on parent.
    for off in 1..8u64 {
        assert!(
            parent.symbolic_spans.contains_key(&Address(0x1000 + off)),
            "parent must have span entry for byte 0x{:x}",
            0x1000 + off
        );
    }
    let parent_span_count_before = parent.symbolic_spans.len();

    // Fork; then write a fresh wide symbolic in the child at a different
    // base. This must NOT add 0x2001..0x2008 to the parent's spans.
    let mut child = parent.fork();
    let child_sym = RustBV::symbol("child_wide", 64);
    child
        .import_symbolic_value(0x2000, child_sym, None)
        .unwrap();

    // Parent's symbolic_spans is unchanged.
    assert_eq!(
        parent.symbolic_spans.len(),
        parent_span_count_before,
        "parent symbolic_spans grew after child mutation"
    );
    for off in 1..8u64 {
        assert!(
            !parent.symbolic_spans.contains_key(&Address(0x2000 + off)),
            "parent leaked span entry for child-only byte 0x{:x}",
            0x2000 + off
        );
        assert!(
            child.symbolic_spans.contains_key(&Address(0x2000 + off)),
            "child must own span entry for byte 0x{:x}",
            0x2000 + off
        );
    }
}

/// angr-24e7: imported_addrs must be cloned (not shared) on fork so that
/// child-side imports do not appear in the parent.
#[test]
fn test_fork_imported_addrs_isolation() {
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);
    parent.map(0x2000, 0x1000, Permission::RWX);

    let parent_sym = RustBV::symbol("parent_imp", 32);
    parent
        .import_symbolic_value(0x1000, parent_sym, None)
        .unwrap();
    assert!(parent.is_imported_addr(0x1000));
    assert!(!parent.is_imported_addr(0x2000));

    let mut child = parent.fork();
    let child_sym = RustBV::symbol("child_imp", 32);
    child
        .import_symbolic_value(0x2000, child_sym, None)
        .unwrap();

    // Child sees both; parent must only see its own.
    assert!(child.is_imported_addr(0x1000));
    assert!(child.is_imported_addr(0x2000));
    assert!(parent.is_imported_addr(0x1000));
    assert!(
        !parent.is_imported_addr(0x2000),
        "parent leaked child-only imported_addr 0x2000"
    );
}

/// angr-24e7: enforce_permissions is a per-state flag. Mutating it on
/// the child after fork must not flip the parent's flag.
/// (Complements test_permission_enforcement_propagates_through_fork
/// which checks the *initial* propagation; this guards the converse —
/// that the flag is owned, not aliased.)
#[test]
fn test_fork_perm_flag_isolation() {
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::R);
    // Parent starts with enforcement OFF.
    assert!(!parent.enforce_permissions());

    let mut child = parent.fork();
    // Flip child's flag; parent must remain OFF.
    child.set_enforce_permissions(true);
    assert!(child.enforce_permissions());
    assert!(
        !parent.enforce_permissions(),
        "child enabling enforce_permissions leaked into parent"
    );

    // Now parent: enable enforcement, fork, then disable on child.
    // Parent's flag must remain ON.
    parent.set_enforce_permissions(true);
    let mut child2 = parent.fork();
    assert!(child2.enforce_permissions());
    child2.set_enforce_permissions(false);
    assert!(
        parent.enforce_permissions(),
        "child disabling enforce_permissions leaked into parent"
    );
}
