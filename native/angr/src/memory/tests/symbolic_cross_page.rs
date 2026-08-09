//! Accesses that span more than one page: concretizing a symbolic address
//! with solutions on either side of a page boundary, and permission
//! enforcement when an unaligned or wide access straddles pages whose
//! permissions differ (the middle page being read-only / write-only must
//! reject the whole access).
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56).

use super::super::*;

/// angr-jdz9: wide symbolic load whose two pinned solutions each
/// straddle a different 4 KiB page boundary.
///
/// Setup: addr ∈ {0x1FFC, 0x2FFC}; both pages flanking each boundary
/// are mapped with distinct concrete bytes near the seam.
/// - 0x1FFC..0x2003 → page 0x1000 last 4 bytes + page 0x2000 first 4
/// - 0x2FFC..0x3003 → page 0x2000 last 4 bytes + page 0x3000 first 4
///
/// Each cross-page slice yields a distinct 8-byte value. After
/// `load_symbolic_unified`, evaluating the result under each pinned
/// addr (in a forked context to isolate from the multi-solution
/// constraint) must reproduce the LE concatenation. If the engine
/// were to apply a permission check or page lookup for only one
/// branch — or were to materialise the ITE only against the first
/// page's bytes — the eval under the *other* solution would diverge
/// from the expected value.
///
/// Locks down current behaviour for angr-jdz9. Two pinned solutions
/// with stride 0x1000 hit the Strided concretization branch in
/// `load_symbolic_unified`, so this also covers the strided ITE
/// path's per-leaf cross-page handling.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_load_cross_page_multiple_solutions() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.map(0x2000, 0x1000, Permission::RWX);
    mem.map(0x3000, 0x1000, Permission::RWX);

    // Distinct bytes around each page boundary so the two straddling
    // 8-byte slices yield distinguishable LE values.
    // 0x1FFC..0x1FFF on page 0x1000:
    mem.store_concrete(0x1FFC, RustBV::concrete(0x44_33_22_11, 32))
        .expect("store 0x1FFC");
    // 0x2000..0x2003 on page 0x2000:
    mem.store_concrete(0x2000, RustBV::concrete(0x88_77_66_55, 32))
        .expect("store 0x2000");
    // 0x2FFC..0x2FFF on page 0x2000:
    mem.store_concrete(0x2FFC, RustBV::concrete(0xCC_BB_AA_99, 32))
        .expect("store 0x2FFC");
    // 0x3000..0x3003 on page 0x3000:
    mem.store_concrete(0x3000, RustBV::concrete(0x00_FF_EE_DD, 32))
        .expect("store 0x3000");

    // Symbolic 64-bit address constrained to {0x1FFC, 0x2FFC} via
    // `(addr == a) | (addr == b)`. Sat solver enumeration in the
    // concretizer should expose both solutions.
    let addr = RustBV::symbolic(&ctx, "load_addr", 64);
    let a = RustBV::concrete(0x1FFC, 64);
    let b = RustBV::concrete(0x2FFC, 64);
    let eq_a = addr.eq(&a, &ctx);
    let eq_b = addr.eq(&b, &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));
    assert!(ctx.is_sat(), "two-solution constraint must be SAT");

    // Wide load with the symbolic address.
    let loaded = mem
        .load_symbolic_unified(addr.clone(), 8, &ctx, &concretizer)
        .expect("symbolic load must succeed when both solutions are mapped");
    assert!(ctx.is_sat(), "context must remain SAT after the load");

    let expected_at_1ffc: u128 = 0x88_77_66_55_44_33_22_11;
    let expected_at_2ffc: u128 = 0x00_FF_EE_DD_CC_BB_AA_99;
    assert_ne!(
        expected_at_1ffc, expected_at_2ffc,
        "test fixture: pinned values must differ to detect ITE collapse"
    );

    // Pin addr to 0x1FFC in a forked context and evaluate the load.
    // The eval must reproduce the LE concatenation of the bytes
    // straddling page 0x1000 / page 0x2000. If the loaded ITE was
    // built only against page 0x2000's bytes (collapsing the cross-
    // page slice), the eval would be wrong here.
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr.eq(&RustBV::concrete(0x1FFC, 64), &probe_a));
    assert!(probe_a.is_sat(), "addr == 0x1FFC must remain SAT");
    assert_eq!(
        probe_a.eval(&loaded),
        Some(expected_at_1ffc),
        "load under addr==0x1FFC: expected LE of page-0x1000 last 4 \
         bytes followed by page-0x2000 first 4 bytes"
    );

    // Pin addr to 0x2FFC; the eval must reproduce the slice across
    // page 0x2000 / page 0x3000. If the ITE branch for the second
    // solution were missing (e.g. permission check applied only to
    // the first concretized address), this eval would diverge.
    let probe_b = ctx.fork();
    probe_b.assume_true(&addr.eq(&RustBV::concrete(0x2FFC, 64), &probe_b));
    assert!(probe_b.is_sat(), "addr == 0x2FFC must remain SAT");
    assert_eq!(
        probe_b.eval(&loaded),
        Some(expected_at_2ffc),
        "load under addr==0x2FFC: expected LE of page-0x2000 last 4 \
         bytes followed by page-0x3000 first 4 bytes"
    );

    // Sanity: both solutions are reachable from the loaded value
    // (the solver must see two distinct results).
    let solutions = ctx.eval_upto(&loaded, 4);
    assert!(
        solutions.contains(&expected_at_1ffc),
        "solver must enumerate the 0x1FFC slice in loaded value; \
         got {solutions:?}"
    );
    assert!(
        solutions.contains(&expected_at_2ffc),
        "solver must enumerate the 0x2FFC slice in loaded value; \
         got {solutions:?}"
    );
}

/// angr-xok8: unaligned 32-byte store starting at 0x1FF0 crosses pages
/// 0x1000 (RW) and 0x2000 (R-only). The multi-page slow path in
/// store_concrete must defer to check_perms_range, which must reject
/// the W check on the middle page.
///
/// (The bead originally said "32 bytes ... 3 pages", but a 32-byte
/// access can span at most 2 pages. See companion test
/// test_permission_enforcement_wide_store_three_pages_middle_readonly
/// for the true 3-page case.)
#[test]
fn test_permission_enforcement_unaligned_store_two_pages_middle_readonly() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RW);
    mem.map(0x2000, 0x1000, Permission::R);
    mem.map(0x3000, 0x1000, Permission::RW);
    mem.set_enforce_permissions(true);

    let value = RustBV::concrete(0xDEADBEEFCAFEBABE, 32 * 8);
    let err = mem.store_concrete(0x1FF0, value).unwrap_err();
    match err {
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
            assert_eq!(
                addr, 0x2000,
                "Permission error should point at R-only middle page"
            );
            assert_eq!(required, Permission::W);
            assert_eq!(actual, Permission::R);
        }
        other => panic!("expected Permission error, got {other:?}"),
    }
}

/// angr-xok8: a >4096-byte store at 0x1FF0 truly spans 3 pages
/// (0x1000, 0x2000, 0x3000). With middle R-only, check_perms_range
/// must visit page 0x2000 and reject the W check.
#[test]
fn test_permission_enforcement_wide_store_three_pages_middle_readonly() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RW);
    mem.map(0x2000, 0x1000, Permission::R);
    mem.map(0x3000, 0x1000, Permission::RW);
    mem.set_enforce_permissions(true);

    // 8208 bytes from 0x1FF0 → last byte at 0x3FFF (page 0x3),
    // touching pages 0x1, 0x2, 0x3. Width must be width_bytes * 8.
    let width_bits = 8208u32 * 8;
    let value = RustBV::concrete(0xCAFEBABE, width_bits);
    let err = mem.store_concrete(0x1FF0, value).unwrap_err();
    match err {
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
            assert_eq!(
                addr, 0x2000,
                "Permission error should point at R-only middle page \
                 when iterating 3-page range"
            );
            assert_eq!(required, Permission::W);
            assert_eq!(actual, Permission::R);
        }
        other => panic!("expected Permission error, got {other:?}"),
    }
}

/// angr-xok8: dual of the wide-store test for loads. With a W-only
/// middle page, a 3-page load must surface a Permission error on
/// the middle page (R required, W actual).
#[test]
fn test_permission_enforcement_wide_load_three_pages_middle_writeonly() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::R);
    mem.map(0x2000, 0x1000, Permission::W);
    mem.map(0x3000, 0x1000, Permission::R);
    mem.set_enforce_permissions(true);

    let err = mem.load_concrete(0x1FF0, 8208, &ctx).unwrap_err();
    match err {
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
            assert_eq!(
                addr, 0x2000,
                "Permission error should point at W-only middle page"
            );
            assert_eq!(required, Permission::R);
            assert_eq!(actual, Permission::W);
        }
        other => panic!("expected Permission error, got {other:?}"),
    }
}
