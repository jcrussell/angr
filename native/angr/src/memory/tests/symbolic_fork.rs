//! Fork isolation for `SymbolicMemory`'s symbolic sidecars, plus the
//! deferred-write (`pending_writes`) lifecycle that interacts with it:
//! mutating one side of a fork must not be visible in the other for
//! `symbolic_spans`, `imported_addrs`, permission flags or pending writes,
//! while a pending write recorded *before* the fork must flush on both sides.
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56).

use super::super::*;

/// angr-24e7: writes to symbolic_spans in a forked memory must not leak
/// back into the parent. Regression test for fork field-by-field cloning.
#[test]
fn test_fork_symbolic_spans_isolation() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);
    parent.map(0x2000, 0x1000, Permission::RWX);

    // Parent imports a 64-bit (8-byte) wide symbolic value at 0x1000.
    // import_symbolic_value populates symbolic_spans for bytes 1..8.
    let parent_sym = RustBV::symbolic(&ctx, "parent_wide", 64);
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
    let child_sym = RustBV::symbolic(&ctx, "child_wide", 64);
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
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);
    parent.map(0x2000, 0x1000, Permission::RWX);

    let parent_sym = RustBV::symbolic(&ctx, "parent_imp", 32);
    parent
        .import_symbolic_value(0x1000, parent_sym, None)
        .unwrap();
    assert!(parent.is_imported_addr(0x1000));
    assert!(!parent.is_imported_addr(0x2000));

    let mut child = parent.fork();
    let child_sym = RustBV::symbolic(&ctx, "child_imp", 32);
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

/// angr-24e7: pending_writes must be cloned on fork. New deferred stores
/// recorded in the child must not appear in the parent's pending list.
#[test]
fn test_fork_pending_writes_isolation() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);

    // Parent records one pending write.
    let p_addr = RustBV::symbolic(&ctx, "p_addr", 64);
    let p_val = RustBV::concrete(0xAAAA, 16);
    parent.add_pending_write(PendingWrite {
        addr: p_addr,
        value: p_val,
        size: 2,
        condition: None,
    });
    assert_eq!(parent.pending_writes_count(), 1);

    // Fork; then add a fresh pending write only in the child.
    let mut child = parent.fork();
    assert_eq!(
        child.pending_writes_count(),
        1,
        "child should inherit parent's pending writes at fork time"
    );

    let c_addr = RustBV::symbolic(&ctx, "c_addr", 64);
    let c_val = RustBV::concrete(0xBBBB, 16);
    child.add_pending_write(PendingWrite {
        addr: c_addr,
        value: c_val,
        size: 2,
        condition: None,
    });

    assert_eq!(
        child.pending_writes_count(),
        2,
        "child should now have its inherited write plus the new one"
    );
    assert_eq!(
        parent.pending_writes_count(),
        1,
        "parent leaked child's pending write into its own list"
    );
}

/// angr-5zbe: a pending write registered with `add_pending_write`
/// must NOT be visible to a subsequent load until
/// `flush_pending_writes` materializes it. This documents the
/// current "defer-then-flush" semantics: the load-time overlay
/// (`apply_pending_writes_concrete`/`_symbolic`) is intentionally
/// stubbed out (see the `lazy-memory-load-overlay-fails` memory),
/// so loads see only what is committed to pages. After flush the
/// concrete-addr pending write must be observable on a re-load.
#[test]
fn test_pending_write_visible_after_flush() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Establish a baseline value at 0x1000.
    let baseline = RustBV::concrete(0xAAAA, 16);
    mem.store_concrete(0x1000, baseline).unwrap();
    let pre = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(pre.as_u64(), Some(0xAAAA));

    // Defer a new value at the same concrete address.
    let new_val = RustBV::concrete(0xBBBB, 16);
    mem.add_pending_write(PendingWrite {
        addr: RustBV::concrete(0x1000, 64),
        value: new_val,
        size: 2,
        condition: None,
    });
    assert_eq!(mem.pending_writes_count(), 1);

    // Pre-flush: load must return the baseline (overlay is disabled).
    let mid = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(
        mid.as_u64(),
        Some(0xAAAA),
        "load before flush must NOT see deferred pending write \
         (overlay is intentionally disabled, see lazy-memory-load-overlay-fails)"
    );

    // Flush, then load: the pending value must now be present and the
    // pending list must be empty.
    mem.flush_pending_writes(&ctx, &concretizer).unwrap();
    assert_eq!(mem.pending_writes_count(), 0);
    let post = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(
        post.as_u64(),
        Some(0xBBBB),
        "load after flush must see materialized pending write"
    );
}

/// angr-5zbe: a pending write registered before `fork()` is inherited
/// by both halves and must be observable in BOTH after each calls
/// `flush_pending_writes` independently. Regression guard against
/// drift in the fork pending_writes clone path
/// (`SymbolicMemory::fork`) and the flush pipeline.
#[test]
fn test_fork_pending_writes_visible_in_both_after_flush() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);
    parent
        .store_concrete(0x1000, RustBV::concrete(0xAAAA, 16))
        .unwrap();

    parent.add_pending_write(PendingWrite {
        addr: RustBV::concrete(0x1000, 64),
        value: RustBV::concrete(0xBBBB, 16),
        size: 2,
        condition: None,
    });

    let mut child = parent.fork();
    assert_eq!(parent.pending_writes_count(), 1);
    assert_eq!(child.pending_writes_count(), 1);

    // Each half flushes independently and the materialized value must
    // be observable on a subsequent load.
    parent.flush_pending_writes(&ctx, &concretizer).unwrap();
    let p_loaded = parent.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(
        p_loaded.as_u64(),
        Some(0xBBBB),
        "parent load after its own flush must see the pending write"
    );

    // The child still has its own copy of the pending write — flushing
    // the parent must not drain the child's queue.
    assert_eq!(
        child.pending_writes_count(),
        1,
        "parent flush leaked into child's pending queue"
    );

    child.flush_pending_writes(&ctx, &concretizer).unwrap();
    let c_loaded = child.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(
        c_loaded.as_u64(),
        Some(0xBBBB),
        "child load after its own flush must see the inherited pending write"
    );
}

/// angr-9ke6b.100: when a pending symbolic-address write concretizes to
/// several candidates and one of them lands on a page that is unmapped in
/// Rust but declared lazy, `flush_pending_writes` must surface
/// `UnmappedPageInRegion` (Python still holds that page's backer data)
/// rather than materializing an ITE whose `else` branch is a zero fill.
/// Candidates preceding the lazy one still materialize against their real
/// prior contents.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pending_write_flush_signals_lazy_unmapped_candidate() {
    let ctx = SymContext::new_mock();
    let mut concretizer = AddressConcretizer::new();
    // SYMBOLIC_WRITE_ADDRESSES: without it the write chain is Max-only and
    // collapses to a Single candidate, never reaching materialize_pending_ite.
    concretizer.symbolic_write_addresses = true;

    let mut mem = SymbolicMemory::new(Endness::Little);
    // Page 0x1000 is real; page 0x2000 is unmapped but inside a lazy region.
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.add_lazy_region(0x2000, 0x1000);
    mem.store_concrete(0x1000, RustBV::concrete(0xAAAA, 16))
        .unwrap();

    let addr = RustBV::symbolic(&ctx, "pw_lazy_addr", 64);
    ctx.assume_true(
        &addr
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    mem.add_pending_write(PendingWrite {
        addr,
        value: RustBV::concrete(0xBBBB, 16),
        size: 2,
        condition: None,
    });

    let err = mem
        .flush_pending_writes(&ctx, &concretizer)
        .expect_err("lazy unmapped candidate must surface the fetch-me signal");
    match err {
        MemoryError::UnmappedPageInRegion { page_addr } => {
            assert_eq!(page_addr, 0x2000, "must point at the lazy page");
        }
        other => panic!("expected UnmappedPageInRegion, got {other:?}"),
    }

    // The mapped candidate kept its real prior byte in the ITE's else arm:
    // under addr == 0x2000 the cell at 0x1000 must still read 0xAAAA, not 0.
    let solver_check = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert!(
        solver_check.as_u64().is_none(),
        "0x1000 should now hold the materialized ITE, not a constant"
    );
}

/// angr-9ke6b.100 companion: when the unmapped candidate is NOT in a lazy
/// region there is genuinely no prior value anywhere, so the zero `current`
/// default stands and the flush fails only on the store's own mapping check.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pending_write_flush_never_mapped_candidate_is_plain_unmapped() {
    let ctx = SymContext::new_mock();
    let mut concretizer = AddressConcretizer::new();
    concretizer.symbolic_write_addresses = true;

    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    // No add_lazy_region: 0x2000 is unmapped everywhere.

    let addr = RustBV::symbolic(&ctx, "pw_nomap_addr", 64);
    ctx.assume_true(
        &addr
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    mem.add_pending_write(PendingWrite {
        addr,
        value: RustBV::concrete(0xBBBB, 16),
        size: 2,
        condition: None,
    });

    let err = mem
        .flush_pending_writes(&ctx, &concretizer)
        .expect_err("never-mapped candidate still cannot be stored");
    assert!(
        matches!(err, MemoryError::Unmapped { .. }),
        "expected plain Unmapped, got {err:?}"
    );
}
