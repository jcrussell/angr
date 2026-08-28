//! Phase 3 (angr-j0n4) and Phase 4.1 (angr-mmdh.1): the per-load and
//! wider-load Multi-cell collapse caches, plus the angr-1tes invariant that a
//! concrete overwrite clears the `multi_objects` entry rather than only the
//! page bitmap bit.

use super::*;

// ============================================================================
// Phase 3 (angr-j0n4): per-load Multi-cell collapse cache
// ============================================================================

/// After a single load of a Multi byte, the payload's collapse cache must
/// be populated. A second load returns the cached BV unchanged.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase3_collapse_cache_hit_after_load() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p3_hit_addr", 64);
    let value = RustBV::concrete(0x77, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();

    // Before any load, the cache is empty.
    let payload = mem.get_multi_alternatives(0x1000).unwrap();
    assert!(!payload.has_cached_collapse(), "cache must start empty");

    // First load populates the cache.
    let first = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    let payload = mem.get_multi_alternatives(0x1000).unwrap();
    assert!(payload.has_cached_collapse(), "first load must cache");

    // Second load returns the same BV (we can't compare Z3 AST identity
    // directly, but the model-eval result must match).
    let second = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(probe.eval(&first), Some(0x77));
    assert_eq!(probe.eval(&second), Some(0x77));
}

/// Appending an alternative via `MultiPayload::push` must invalidate the
/// collapse cache so the next load rebuilds the ITE.
#[test]
fn test_phase3_collapse_cache_invalidated_on_push() {
    let ctx = SymContext::new_mock();
    let mut payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::concrete(1, 1),
        RustBV::concrete(0xAA, 8),
    )]);
    let _ = payload.collapse(0, &ctx);
    assert!(
        payload.has_cached_collapse(),
        "collapse must populate cache"
    );

    payload.push(MultiAlternative::new(
        RustBV::concrete(0, 1),
        RustBV::concrete(0xBB, 8),
    ));
    assert!(
        !payload.has_cached_collapse(),
        "push must invalidate cached collapse"
    );
}

/// `MultiPayload::collapse` must rebuild when the page's concrete default
/// byte changes between loads. Otherwise a concrete overwrite of the cell's
/// page byte (which does not currently clear the Multi marker) would serve
/// a stale ITE.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase3_collapse_cache_invalidated_on_default_byte_change() {
    let ctx = SymContext::new_mock();
    let payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::symbolic(&ctx, "p3_default_change_cond", 1),
        RustBV::concrete(0xAA, 8),
    )]);

    let collapsed_0 = payload.collapse(0x00, &ctx);
    assert!(payload.has_cached_collapse());
    let collapsed_again = payload.collapse(0x00, &ctx);
    assert_eq!(
        ctx.eval(&collapsed_0),
        ctx.eval(&collapsed_again),
        "cache hit must return equivalent BV"
    );

    // A different default byte must produce a different ELSE leaf.
    let collapsed_ff = payload.collapse(0xFF, &ctx);
    // Force the cond=false branch so the ELSE leaf is observable.
    let probe = ctx.fork();
    probe.assume_true(
        &payload.alternatives()[0]
            .cond
            .eq(&RustBV::concrete(0, 1), &probe),
    );
    assert_eq!(
        probe.eval(&collapsed_ff),
        Some(0xFF),
        "rebuilt collapse must reflect the new default byte"
    );
}

/// After fork, the parent and child each hold an independent payload. If
/// the parent's cache is populated, the child's clone carries it forward
/// (the BV is referentially safe — Z3 ASTs are immutable / refcounted).
#[test]
fn test_phase3_collapse_cache_clones_with_payload() {
    let ctx = SymContext::new_mock();
    let payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::concrete(1, 1),
        RustBV::concrete(0x33, 8),
    )]);
    let _ = payload.collapse(0x00, &ctx);
    assert!(payload.has_cached_collapse());

    let cloned = payload.clone();
    assert!(
        cloned.has_cached_collapse(),
        "clone must carry the cached collapse forward"
    );

    // Independence: pushing on the clone does not touch the original.
    let mut cloned_mut = cloned;
    cloned_mut.push(MultiAlternative::new(
        RustBV::concrete(0, 1),
        RustBV::concrete(0x44, 8),
    ));
    assert!(!cloned_mut.has_cached_collapse());
    assert!(
        payload.has_cached_collapse(),
        "original payload must retain its cache after clone mutation"
    );
}

// ============================================================================
// Phase 4.1 (angr-mmdh.1): wider-load collapse cache in
// `assemble_load_with_multi`
// ============================================================================

/// A wider load that touches a Multi byte must populate the wider-load
/// cache. A second identical load must hit and return an equivalent BV.
#[test]
fn test_phase4_wider_load_cache_hit() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_hit_addr", 64);
    let value = RustBV::concrete(0xAA, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();

    assert_eq!(mem.wider_load_cache_len(), 0, "cache starts empty");

    // First load (size 4 = wider than 1): populates the cache.
    let first = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(mem.wider_load_cache_len(), 1, "first load populates cache");

    // Second identical load: returns from cache.
    let second = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(probe.eval(&first), probe.eval(&second));
    assert_eq!(mem.wider_load_cache_len(), 1, "second load reuses entry");
}

/// Size==1 loads bypass the wider-load cache (no concat to amortize).
#[test]
fn test_phase4_wider_load_cache_skips_size_one() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_size1_addr", 64);
    mem.store_concrete_multi(
        &addr_var,
        &RustBV::concrete(0x55, 8),
        &[0x1000, 0x2000],
        &ctx,
    )
    .unwrap();

    let _ = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        0,
        "size==1 must not populate the wider-load cache"
    );
}

/// Installing a new Multi alternative at a byte covered by a cached load
/// must invalidate the cached entry (fingerprint mismatch on next read).
/// Uses two independent address vars so the second store contributes an
/// alternative whose cond can be made true while the first is false —
/// letting eval pin the rebuilt result to the new value and fail loudly
/// if the cache returned the stale BV.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase4_wider_load_cache_invalidated_on_multi_install() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var1 = RustBV::symbolic(&ctx, "p41_inval_addr1", 64);
    let addr_var2 = RustBV::symbolic(&ctx, "p41_inval_addr2", 64);

    // First install: byte 0x1000 gets alt (addr_var1 == 0x1000, 0xAA).
    mem.store_concrete_multi(
        &addr_var1,
        &RustBV::concrete(0xAA, 8),
        &[0x1000, 0x2000],
        &ctx,
    )
    .unwrap();

    // Prime the cache under a probe where addr_var1==0x1000 (alt 0 fires
    // → result byte is 0xAA).
    let first_probe = ctx.fork();
    first_probe.assume_true(&addr_var1.eq(&RustBV::concrete(0x1000, 64), &first_probe));
    let first = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(mem.wider_load_cache_len(), 1);
    assert_eq!(first_probe.eval(&first), Some(0xAA));

    // Second install at byte 0x1000 — independent cond (addr_var2 == 0x1000)
    // bumps the per-byte version so the cached fingerprint mismatches.
    mem.store_concrete_multi(
        &addr_var2,
        &RustBV::concrete(0xBB, 8),
        &[0x1000, 0x3000],
        &ctx,
    )
    .unwrap();

    // Probe where alt 0's cond is false (addr_var1==0x9999) but alt 1's
    // cond is true (addr_var2==0x1000). Right-fold ITE:
    //   alt 0 (outermost): cond false → fall to ELSE
    //   alt 1: cond true → 0xBB
    // If the cache returns the stale BV from the first load (which lacks
    // the alt 1 branch), eval here would NOT be 0xBB.
    let probe = ctx.fork();
    probe.assume_true(&addr_var1.eq(&RustBV::concrete(0x9999, 64), &probe));
    probe.assume_true(&addr_var2.eq(&RustBV::concrete(0x1000, 64), &probe));
    let after = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        1,
        "rebuilt entry replaces the stale one, cache size unchanged"
    );
    assert_eq!(
        probe.eval(&after),
        Some(0xBB),
        "rebuilt load must include the alt installed after the cache prime"
    );
}

/// Loads that touch any plain Symbolic byte must not be cached.
#[test]
fn test_phase4_wider_load_cache_skips_symbolic_bytes() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_sym_byte_addr", 64);
    // One Multi byte at 0x1000…
    mem.store_concrete_multi(
        &addr_var,
        &RustBV::concrete(0xAA, 8),
        &[0x1000, 0x2000],
        &ctx,
    )
    .unwrap();

    // …and a plain Symbolic byte at 0x1001 (via concrete store with a
    // symbolic value).
    let sym_val = RustBV::symbolic(&ctx, "p41_sym_byte_val", 8);
    mem.store_concrete(0x1001, sym_val).unwrap();

    // A 4-byte load at 0x1000 covers both — must NOT cache.
    let _ = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        0,
        "loads touching plain Symbolic bytes must not be cached"
    );
}

/// Forks start with a COLD wider-load cache (angr-6t8z3.2, commit
/// 94c015db5): `fork()` deliberately does not deep-clone the parent's
/// cache — the child repopulates lazily on its first wider load, and a
/// child-side rebuild must not disturb the parent's own cached entry.
#[test]
fn test_phase4_wider_load_cache_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_fork_addr", 64);
    parent
        .store_concrete_multi(
            &addr_var,
            &RustBV::concrete(0xAA, 8),
            &[0x1000, 0x2000],
            &ctx,
        )
        .unwrap();
    let _ = parent.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(parent.wider_load_cache_len(), 1);

    let mut child = parent.fork();
    assert_eq!(
        child.wider_load_cache_len(),
        0,
        "fork starts with a cold cache (no deep-clone; repopulates lazily)"
    );

    // Child mutation must not affect parent.
    child
        .store_concrete_multi(
            &addr_var,
            &RustBV::concrete(0xBB, 8),
            &[0x1000, 0x3000],
            &ctx,
        )
        .unwrap();
    let _ = child.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    // Parent's cache still valid (size unchanged, fingerprint still matches
    // its own copy of multi_versions).
    let _ = parent.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(parent.wider_load_cache_len(), 1);
    assert_eq!(child.wider_load_cache_len(), 1);
}

// ============================================================================
// angr-1tes: cache + multi_objects invariant under concrete overwrite.
// ============================================================================

/// A concrete store at a byte covered by a previously-installed Multi cell
/// must produce a re-load that reflects the new concrete byte, not the stale
/// Multi alternative. Pre-fix, `SymbolicMemory::store_concrete` cleared the
/// page-level `multi_bitmap` bit but left the `multi_objects` entry (and the
/// `multi_versions` counter) untouched. The next load's dispatcher (see
/// `range_has_multi`, the shared gate `load_concrete_common` consults ahead
/// of every symbolic fast path) checks `multi_objects.contains_key`
/// and hands the load to `assemble_load_with_multi`, which folds the orphaned
/// alternatives over the new page-byte default — returning the old Multi
/// value when the original cond is satisfiable.
#[test]
fn test_concrete_overwrite_clears_multi_cell() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_overwrite_addr", 64);
    // Install Multi at 0x1000..0x1004 (4 bytes) with one candidate (0x1000)
    // and value 0xAABBCCDD. Pre-fix this leaves an orphaned `multi_objects`
    // entry that survives the concrete store below.
    mem.store_concrete_multi(
        &addr_var,
        &RustBV::concrete(0xAABBCCDD, 32),
        &[0x1000],
        &ctx,
    )
    .unwrap();
    assert_eq!(mem.multi_cell_count(), 4);

    // Prime the wider-load cache.
    let _ = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(mem.wider_load_cache_len(), 1);

    // Concrete overwrite of byte 0x1000.
    mem.store_concrete(0x1000, RustBV::concrete(0x11, 8))
        .unwrap();

    // The Multi cell at 0x1000 must be gone — both the sidecar map AND the
    // page bit. The cache entry may remain (eviction is lazy) but a refetch
    // must not see the stale alternative.
    assert!(
        mem.get_multi_alternatives(0x1000).is_none(),
        "concrete overwrite must drop the orphaned Multi cell at 0x1000"
    );
    let page = mem.pages.get(&(0x1000 >> 12)).expect("page mapped");
    assert!(
        !page.is_multi(0),
        "page Multi bit at offset 0 must be cleared by concrete store"
    );

    // Re-read: under the cond that previously fired the Multi alt
    // (addr_var == 0x1000), the new concrete byte 0x11 must dominate.
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    let after = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    assert_eq!(
        probe.eval(&after),
        Some(0x11),
        "re-load after concrete overwrite must reflect the new byte, \
         not the orphaned Multi alternative"
    );
}
