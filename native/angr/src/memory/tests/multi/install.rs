//! Phase 1.3 (angr-aija) and Phase 2 (angr-qh5u): installing Multi cells from
//! the store side — first the `store_concrete_multi` /
//! `store_symbolic_unified_multi` helpers, then the end-to-end
//! `install_multi_for_candidates` path with its lazy-region safety,
//! permission and wrapping-candidate guards.

use super::*;

// ============================================================================
// Phase 1.3 (angr-aija): store_concrete_multi / store_symbolic_unified_multi
// helpers. These exercise the store -> Multi -> load round-trip.
// ============================================================================

/// Round-trip a multi-byte LE store through `store_concrete_multi` and
/// the Phase 1.2 load path. With two candidates {A, B}, the load at A
/// should yield the full stored value, and the load at B likewise.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_concrete_multi_le_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_addr", 64);
    let value = RustBV::concrete(0xDEAD_BEEF, 32);

    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .expect("store_concrete_multi must succeed");

    // Multi cells were installed at every byte of both candidates.
    assert_eq!(mem.multi_cell_count(), 8);

    // Read back under each candidate concretization.
    let loaded_a = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("LE load at candidate A must succeed");
    let loaded_b = mem
        .load_concrete_lazy(0x2000, 4, &ctx)
        .expect("LE load at candidate B must succeed");

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded_a), Some(0xDEAD_BEEF));
    // The B cell was also written under the cond addr==B; under addr==A
    // its load should fall back to the default else (concrete 0).
    assert_eq!(probe_a.eval(&loaded_b), Some(0x0));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x2000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded_b), Some(0xDEAD_BEEF));
    assert_eq!(probe_b.eval(&loaded_a), Some(0x0));
}

/// Big-endian variant of the round-trip. byte 0 is the MSB so the
/// per-byte split must mirror that orientation.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_concrete_multi_be_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_be_addr", 64);
    let value = RustBV::concrete(0xCAFE_BABE, 32);

    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .expect("BE store_concrete_multi must succeed");

    let loaded_a = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded_a), Some(0xCAFE_BABE));
}

/// Fork independence: installing Multi cells in a child must not bleed
/// into the parent, even when the child later mutates them again.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_concrete_multi_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_fork_addr", 64);
    let value = RustBV::concrete(0x11_22_33_44, 32);
    parent
        .store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    let parent_count_before = parent.multi_cell_count();
    assert_eq!(parent_count_before, 8);

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), parent_count_before);

    // Child overwrites byte 0 of the first candidate with a concrete byte.
    // store_concrete clears the corresponding multi bit (pinned by
    // collapse.rs::test_concrete_overwrite_clears_multi_bit) AND the
    // multi_objects entry (angr-1tes; pinned by
    // cache.rs::test_concrete_overwrite_clears_multi_cell), so the
    // clear_multi_at below is belt-and-suspenders for this fork test.
    child
        .store_concrete(0x1000, RustBV::concrete(0xFF, 8))
        .unwrap();
    child.clear_multi_at(0x1000);

    assert_eq!(parent.multi_cell_count(), parent_count_before);
    assert_eq!(child.multi_cell_count(), parent_count_before - 1);

    // Parent's load still sees the original Multi alts.
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));
    let parent_loaded = parent.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(probe.eval(&parent_loaded), Some(0x11_22_33_44));
}

/// End-to-end: `store_symbolic_unified_multi` with an address constrained
/// to two solutions concretizes to Multiple, installs Multi cells, and
/// `load_concrete_lazy` materializes the correct value for each candidate.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_symbolic_unified_multi_multiple_round_trip() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "ssm_addr", 64);
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let value = RustBV::concrete(0x1122_3344, 32);
    let conc_result = mem
        .store_symbolic_unified_multi(addr_var.clone(), value, &ctx, &concretizer)
        .expect("Multi unified store must succeed");
    // Two candidates with a regular delta get detected as Strided by
    // the concretizer; either Multiple or Strided routes through
    // install_multi_for_candidates and must install per-byte Multi cells.
    assert!(matches!(
        conc_result,
        Some(ConcretizationResult::Multiple(_)) | Some(ConcretizationResult::Strided { .. })
    ));

    // Per-byte Multi cells should exist for both candidates.
    assert_eq!(mem.multi_cell_count(), 8);

    let loaded_a = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded_a), Some(0x1122_3344));
}

/// `store_symbolic_unified_multi` with a fully-concrete address must
/// short-circuit to the eager `store_concrete_lazy` path (no Multi
/// cells installed). Counter the lazy-vs-eager distinction at the entry.
#[test]
fn test_store_symbolic_unified_multi_concrete_addr_no_multi() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr = RustBV::concrete(0x1000, 64);
    let value = RustBV::concrete(0xAABB_CCDD, 32);

    let pre_multi = mem.multi_cell_count();
    let result = mem
        .store_symbolic_unified_multi(addr, value, &ctx, &concretizer)
        .expect("concrete-address unified-multi store must succeed");
    assert!(matches!(result, Some(ConcretizationResult::Single(_))));
    assert_eq!(
        mem.multi_cell_count(),
        pre_multi,
        "concrete-address path must NOT install Multi cells"
    );

    // Standard load must read the concretely-stored value.
    let loaded = mem.load_concrete(0x1000, 4, &ctx).unwrap();
    assert_eq!(loaded.as_u64(), Some(0xAABB_CCDD));
}

// ============================================================================
// Phase 2 (angr-qh5u): Multi-cell store install, lazy-region safety, and Multi
// flush on export. Phase 4.3 (angr-mmdh.3) made Multi cells the only path —
// `store_symbolic_unified` always routes Multiple/Strided through
// `install_multi_for_candidates_safe`.
// ============================================================================

/// `store_symbolic_unified` routes Multiple to Multi cells — same end-state as
/// `store_symbolic_unified_multi`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_gate_on_installs_multi() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_on_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0xCAFEBABE, 32);

    mem.store_symbolic_unified(addr_var.clone(), value, &ctx, &concretizer)
        .expect("gated-on Multi store must succeed");
    assert_eq!(mem.multi_cell_count(), 8);

    // Load under the addr==A constraint reads back the value.
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    let loaded = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(probe.eval(&loaded), Some(0xCAFEBABE));
}

/// The Phase 2 safe installer must return `UnmappedPageInRegion` when a
/// candidate page lives in a declared lazy region — the interpreter
/// fetches the page from Python rather than letting Rust auto-map a
/// zero page that diverges from Python's backer data.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_safe_install_lazy_region_signals() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Map page 0x1000; leave page 0x2000 unmapped but inside a lazy region.
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.add_lazy_region(0x2000, 0x1000);

    let addr_var = RustBV::symbolic(&ctx, "p2_lazy_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0x11, 8);

    let err = mem
        .store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect_err("lazy unmapped candidate must error to interpreter");
    match err {
        MemoryError::UnmappedPageInRegion { page_addr } => {
            assert_eq!(page_addr, 0x2000, "must point at the lazy page");
        }
        other => panic!("expected UnmappedPageInRegion, got {other:?}"),
    }
    // No Multi cells installed on the partial run.
    assert_eq!(mem.multi_cell_count(), 0);
}

/// The Phase 2 safe installer must silently skip candidates whose pages
/// are unmapped and NOT in any lazy region (matches
/// `prepare_addresses_for_ite`'s skip-unmapped behavior).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_safe_install_skips_unmapped_non_lazy() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Only page 0x1000 is mapped. Page 0x2000 is unmapped and NOT lazy.
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_skip_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0xAB, 8);

    mem.store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect("non-lazy unmapped candidate must be silently skipped");

    // Only the mapped candidate (0x1000) got a Multi cell.
    assert_eq!(mem.multi_cell_count(), 1);
    assert!(mem.get_multi_alternatives(0x1000).is_some());
    assert!(mem.get_multi_alternatives(0x2000).is_none());
}

/// angr-sqfj8.74: when the non-lazy-unmapped filter above empties the
/// candidate list entirely, `install_multi_for_candidates_safe` drops the
/// whole store and returns `Ok` rather than erroring — the `SILENT(cat-b)`
/// case documented at that site. Pins the boundary so a future change that
/// turns "all candidates unreachable" into an error (diverging from eager
/// `prepare_addresses_for_ite`, which stores nothing in the same situation)
/// fails loudly here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_safe_install_drops_store_when_every_candidate_unmapped() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Neither candidate page is mapped, and neither is in a lazy region.
    // An unrelated mapped page keeps the memory non-empty.
    mem.map(0x5000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_all_unmapped_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0xAB, 8);

    mem.store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect("all-unreachable candidate set must drop the store, not error");

    // Nothing was installed and no page was auto-mapped by the drop.
    assert_eq!(mem.multi_cell_count(), 0);
    assert!(mem.get_multi_alternatives(0x1000).is_none());
    assert!(mem.get_multi_alternatives(0x2000).is_none());
}

/// angr-9ke6b.94: with `enforce_permissions` on, a Multiple-concretized
/// symbolic-address store into a read-only page must be rejected exactly like
/// the Single-concretized store `store_concrete` already rejects. Before the
/// fix `install_multi_for_candidates` never consulted `check_perms_range`, so
/// the write silently mutated the read-only page.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_install_enforces_write_permission() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Candidate 0x1000 is read-only; 0x2000 is writable.
    mem.map(0x1000, 0x1000, Permission::R);
    mem.map(0x2000, 0x1000, Permission::RW);
    mem.set_enforce_permissions(true);

    let addr_var = RustBV::symbolic(&ctx, "perm_multi_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );

    let err = mem
        .store_symbolic_unified(addr_var, RustBV::concrete(0xAB, 8), &ctx, &concretizer)
        .expect_err("Multiple-concretized store into an R-only page must be rejected");
    match err {
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
            assert_eq!(addr, 0x1000);
            assert!(required.write);
            assert!(!actual.write);
        }
        other => panic!("expected Permission error, got {other:?}"),
    }
    // The check runs before any mutation, so nothing was installed — not even
    // for the writable candidate.
    assert_eq!(mem.multi_cell_count(), 0);

    // Symmetry: the Single-concretized store on the same page is rejected the
    // same way (this is the behavior the Multi path was diverging from).
    let single_err = mem
        .store_concrete(0x1000, RustBV::concrete(0xAB, 8))
        .expect_err("Single-concretized store into an R-only page must be rejected");
    assert!(matches!(single_err, MemoryError::Permission { .. }));
}

/// The permission gate must be inert when `enforce_permissions` is off (the
/// default), so ordinary Multi installs into R-only or auto-mapped pages keep
/// working.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_install_permission_check_off_by_default() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    mem.map(0x1000, 0x1000, Permission::R);
    mem.map(0x2000, 0x1000, Permission::R);
    assert!(!mem.enforce_permissions());

    let addr_var = RustBV::symbolic(&ctx, "perm_off_multi_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );

    mem.store_symbolic_unified(addr_var, RustBV::concrete(0xAB, 8), &ctx, &concretizer)
        .expect("permission enforcement off: R-only page must still accept the store");
    assert_eq!(mem.multi_cell_count(), 2);
}

/// angr-03vl4.37: a candidate whose byte range runs off the top of the address
/// space must be rejected outright, never wrapped onto a low page. The
/// candidates come from Z3 solutions for a guest-supplied pointer, and
/// `install_multi_for_candidates` computed the per-byte address with a bare
/// `cand + b` — a panic under debug-assertions, a silent redirect in release.
/// Two guards now cover it: the page-range pass (`end_page_inclusive`) and the
/// `checked_add` in the install loop itself; this pins the observable contract
/// either one must deliver.
#[test]
fn test_multi_install_rejects_wrapping_candidate() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // The page a wrapped byte address would land on.
    mem.map(0x0u64, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "wrap_multi_addr", 64);
    let wrapping = u64::MAX - 1; // [u64::MAX-1, u64::MAX+2) for a 4-byte value
    let err = mem
        .store_concrete_multi(
            &addr_var,
            &RustBV::concrete(0xdead_beef, 32),
            &[0x100, wrapping],
            &ctx,
        )
        .expect_err("a candidate whose range wraps past u64::MAX must be rejected");
    assert!(matches!(err, MemoryError::OutOfBounds { .. }));

    // The rejection precedes every mutation, so not even the well-formed
    // candidate installed — and page 0 is untouched.
    assert_eq!(mem.multi_cell_count(), 0);
    for byte in 0..4u64 {
        assert!(mem.get_multi_alternatives(byte).is_none());
    }
}

/// Harness 6 boundary sweep for the fix above: every entry in the shared
/// `test_boundary_values` table, not just the one `u64::MAX - 1` candidate
/// `test_multi_install_rejects_wrapping_candidate` pins, must be rejected
/// exactly when its 4-byte range would overflow — and must still install
/// cleanly when it does not, so the guard isn't over-broad.
#[test]
fn install_multi_for_candidates_boundary_sweep_rejects_wrapping_candidates() {
    let ctx = SymContext::new_mock();
    for &addr in &crate::test_boundary_values::boundary_addresses() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x0u64, 0x1000, Permission::RWX);

        let addr_var = RustBV::symbolic(&ctx, "boundary_multi_addr", 64);
        let value = RustBV::concrete(0xdead_beef, 32);
        let overflows = addr.checked_add(3).is_none();

        let result = mem.store_concrete_multi(&addr_var, &value, &[addr], &ctx);
        if overflows {
            let err = result.expect_err("wrapping candidate must be rejected");
            assert!(
                matches!(err, MemoryError::OutOfBounds { .. }),
                "addr={addr:#x}: expected OutOfBounds, got {err:?}"
            );
            assert_eq!(
                mem.multi_cell_count(),
                0,
                "addr={addr:#x}: rejection must install nothing"
            );
        } else {
            result.unwrap_or_else(|e| panic!("addr={addr:#x} must install cleanly, got {e:?}"));
            assert_eq!(
                mem.multi_cell_count(),
                4,
                "addr={addr:#x}: a 4-byte value must install exactly 4 Multi cells"
            );
        }
    }
}

/// `flush_multi_cells` must collapse every Multi byte into a per-byte
/// symbolic_objects entry and mark the page-level symbolic bit so the
/// state export pipeline picks it up. This is the export-correctness
/// invariant called out in `rust_lazy_memory_design.rst` Phase 2.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_flush_multi_to_symbolic_objects() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_flush_addr", 64);
    let value = RustBV::concrete(0xAA, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .expect("test setup install must succeed");
    assert_eq!(mem.multi_cell_count(), 2);

    mem.flush_multi_cells(&ctx);

    // Multi cells are gone, symbolic_objects has 1-byte entries at each
    // flushed address.
    assert_eq!(mem.multi_cell_count(), 0);
    assert!(mem.get_symbolic_object(0x1000).is_some());
    assert!(mem.get_symbolic_object(0x2000).is_some());

    // The 1-byte symbolic_object at 0x1000 evaluates to 0xAA under
    // addr==0x1000 (because cond=(addr==0x1000) is true, picking value 0xAA).
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    let sym = mem.get_symbolic_object(0x1000).unwrap();
    assert_eq!(probe.eval(sym), Some(0xAA));

    // The 1-byte symbolic_object at 0x2000 evaluates to 0 (concrete else)
    // under addr==0x1000 because its cond fires only when addr==0x2000.
    let sym2 = mem.get_symbolic_object(0x2000).unwrap();
    assert_eq!(probe.eval(sym2), Some(0));
}

/// After fork, mutating Multi cells in the parent must not leak into
/// the child via the new safe-install path. Pairs with
/// `test_store_concrete_multi_fork_independence` but exercises the
/// Phase 2 entry point.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_fork_independence_via_safe_install() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_fork_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    parent
        .store_symbolic_unified(
            addr_var.clone(),
            RustBV::concrete(0x12, 8),
            &ctx,
            &concretizer,
        )
        .unwrap();
    let parent_count = parent.multi_cell_count();
    assert_eq!(parent_count, 2);

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), parent_count);

    // Child stores another value at the same symbolic addr — appends
    // a second alternative to each Multi cell. Parent must not see it.
    child
        .store_symbolic_unified(addr_var, RustBV::concrete(0x34, 8), &ctx, &concretizer)
        .unwrap();
    let child_a = child.get_multi_alternatives(0x1000).unwrap();
    let parent_a = parent.get_multi_alternatives(0x1000).unwrap();
    assert_eq!(parent_a.len(), 1, "parent must keep its single alternative");
    assert_eq!(child_a.len(), 2, "child accumulates appended alternative");
}
