//! Unit tests for [`super::concretize_cache`] — the per-block address
//! concretization cache and its read/write fallback-on-cache-HIT branches.
//!
//! Split out per the rust-mod-tests-sibling-extraction convention. GIL-free:
//! no PyO3. These exercise the cache-HIT `TooLarge` paths in
//! `concretize_cached_read` / `concretize_cached_write` that are not reachable
//! through `AddressConcretizer::concretize_read`/`concretize_write` (those
//! already collapse `TooLarge` to `Single` before the value is cached). We seed
//! a raw `TooLarge` directly into the cache, then assert the cache-lookup
//! fallback fires (or is suppressed when the fallback flag is off).

use super::*;

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

/// Build a concretizer with the read/write fallback flags set explicitly so a
/// test controls whether a cached `TooLarge` collapses to `Single`.
fn concretizer_with_fallbacks(read_any: bool, write_max: bool) -> AddressConcretizer {
    let mut c = AddressConcretizer::new();
    c.read_fallback_any = read_any;
    c.write_fallback_max = write_max;
    c
}

#[test]
fn concretize_cached_read_short_circuits_concrete_addr() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let addr = RustBV::concrete(0x4000, 64);
    let result = interp.concretize_cached_read(&addr);
    match &*result {
        ConcretizationResult::Single(a) => assert_eq!(*a, 0x4000),
        other => panic!("expected Single(0x4000), got {other:?}"),
    }
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn concretize_cached_read_applies_any_fallback_on_cached_toolarge() {
    // angr-1c88c gap 6/7: cache-HIT TooLarge -> read_fallback_any (eval).
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_concretizer(concretizer_with_fallbacks(true, true));

    let addr = RustBV::symbolic(&ctx, "read_addr", 64);
    // Seed the cache with a raw TooLarge for this address's key.
    let key = VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr);
    interp.concretize_cache.insert(
        key,
        Arc::new(ConcretizationResult::TooLarge {
            min: 0,
            max: u64::MAX,
            limit: 1024,
        }),
    );

    let result = interp.concretize_cached_read(&addr);
    // read_fallback_any -> ctx.eval(addr) -> Single(model value).
    match &*result {
        ConcretizationResult::Single(_) => {}
        other => panic!("expected Single from Any fallback, got {other:?}"),
    }
}

#[test]
fn concretize_cached_read_keeps_toolarge_when_fallback_disabled() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_concretizer(concretizer_with_fallbacks(false, true));

    let addr = RustBV::symbolic(&ctx, "read_addr_nofb", 64);
    let key = VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr);
    interp.concretize_cache.insert(
        key,
        Arc::new(ConcretizationResult::TooLarge {
            min: 0,
            max: u64::MAX,
            limit: 1024,
        }),
    );

    let result = interp.concretize_cached_read(&addr);
    // Fallback off: the raw TooLarge is returned unchanged.
    match &*result {
        ConcretizationResult::TooLarge { min, max, limit } => {
            assert_eq!((*min, *max, *limit), (0, u64::MAX, 1024));
        }
        other => panic!("expected TooLarge unchanged, got {other:?}"),
    }
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn concretize_cached_write_applies_max_fallback_on_cached_toolarge() {
    // angr-1c88c gap 5/7: cache-HIT TooLarge -> write_fallback_max (range.max).
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_concretizer(concretizer_with_fallbacks(true, true));

    let addr = RustBV::symbolic(&ctx, "write_addr", 64);
    let key = VEXInterpreter::bv_cache_key(ConcretizeNs::Write, &addr);
    interp.concretize_cache.insert(
        key,
        Arc::new(ConcretizationResult::TooLarge {
            min: 0,
            max: u64::MAX,
            limit: 128,
        }),
    );

    let result = interp.concretize_cached_write(&addr);
    // write_fallback_max -> ctx.range(addr) -> Single(max) (or eval fallback).
    match &*result {
        ConcretizationResult::Single(_) => {}
        other => panic!("expected Single from Max fallback, got {other:?}"),
    }
}

#[test]
fn concretize_cached_write_keeps_toolarge_when_fallback_disabled() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_concretizer(concretizer_with_fallbacks(true, false));

    let addr = RustBV::symbolic(&ctx, "write_addr_nofb", 64);
    let key = VEXInterpreter::bv_cache_key(ConcretizeNs::Write, &addr);
    interp.concretize_cache.insert(
        key,
        Arc::new(ConcretizationResult::TooLarge {
            min: 0,
            max: u64::MAX,
            limit: 128,
        }),
    );

    let result = interp.concretize_cached_write(&addr);
    match &*result {
        ConcretizationResult::TooLarge { min, max, limit } => {
            assert_eq!((*min, *max, *limit), (0, u64::MAX, 128));
        }
        other => panic!("expected TooLarge unchanged, got {other:?}"),
    }
}

#[test]
fn concretize_cached_jump_short_circuits_concrete_addr() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let addr = RustBV::concrete(0x401000, 64);
    match &*interp.concretize_cached_jump(&addr) {
        ConcretizationResult::Single(a) => assert_eq!(*a, 0x401000),
        other => panic!("expected Single(0x401000), got {other:?}"),
    }
}

/// angr-9ke6b.92: the jump variant has no fallback of its own, so a cached
/// `TooLarge` must come back unchanged even when the read/write fallback flags
/// are on — `eval_next_addr` turns any non-`Single` into an `Unsupported`
/// deferral to Python rather than pinning an arbitrary target.
#[test]
fn concretize_cached_jump_keeps_toolarge_verbatim_on_cache_hit() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_concretizer(concretizer_with_fallbacks(true, true));

    let addr = RustBV::symbolic(&ctx, "jump_target", 64);
    let key = VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr);
    interp.concretize_cache.insert(
        key,
        Arc::new(ConcretizationResult::TooLarge {
            min: 0,
            max: u64::MAX,
            limit: 1024,
        }),
    );

    match &*interp.concretize_cached_jump(&addr) {
        ConcretizationResult::TooLarge { min, max, limit } => {
            assert_eq!((*min, *max, *limit), (0, u64::MAX, 1024));
        }
        other => panic!("expected TooLarge unchanged, got {other:?}"),
    }
}

/// The cache is shared with the read variant (same `Range(read_range_limit)`
/// chain, `ConcretizeNs::ReadJump`): a `Single` seeded by an earlier load in
/// the same block is reused for a later jump target with the same key, so the
/// block pays one solver query rather than one per exit.
#[test]
fn concretize_cached_jump_reuses_shared_cache_entry() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);

    let addr = RustBV::symbolic(&ctx, "shared_target", 64);
    let key = VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr);
    interp
        .concretize_cache
        .insert(key, Arc::new(ConcretizationResult::Single(0x4141)));

    match &*interp.concretize_cached_jump(&addr) {
        ConcretizationResult::Single(a) => assert_eq!(*a, 0x4141),
        other => panic!("expected the cached Single(0x4141), got {other:?}"),
    }
}

/// angr-360gt: the read/jump chain and the write chain run *different* range
/// limits, so their cache entries must live in different namespaces.
#[test]
fn bv_cache_key_separates_read_and_write_namespaces() {
    let ctx = SymContext::new_mock();
    let addr = RustBV::symbolic(&ctx, "xover", 64);
    assert_ne!(
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr),
        VEXInterpreter::bv_cache_key(ConcretizeNs::Write, &addr),
        "a read/jump entry must not be reachable from the write cache lookup"
    );
}

/// angr-360gt: a `Multiple` produced under `read_range_limit` (1024) must not
/// satisfy a later store on the same expression, whose `write_range_limit`
/// (128) would have rejected it. Pre-fix, `concretize_cached_write` hit the
/// shared entry and — since `Multiple` is not `TooLarge` — returned it
/// verbatim, fanning the store out across every cached address instead of
/// collapsing to one Max-pinned cell. Whether that happened depended purely on
/// whether a load / mid-block `Ist_Exit` touched the expression first.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn write_does_not_reuse_a_read_range_multiple() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut concretizer = concretizer_with_fallbacks(true, true);
    // SYMBOLIC_WRITE_ADDRESSES on: the write chain is Range(128) -> Max, i.e.
    // the exact configuration in which the crossover changes the outcome.
    concretizer.symbolic_write_addresses = true;
    interp.set_concretizer(concretizer);

    let addr = RustBV::symbolic(&ctx, "xover_addr", 64);
    // 200 candidates: inside read_range_limit=1024, outside write_range_limit=128.
    let read_side: Vec<u64> = (0..200).map(|i| 0x5000 + i).collect();
    interp.concretize_cache.insert(
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr),
        Arc::new(ConcretizationResult::Multiple(read_side)),
    );

    // The store must run its own chain: the unconstrained 64-bit address is
    // TooLarge for Range(128), so write_fallback_max collapses it to one cell.
    match &*interp.concretize_cached_write(&addr) {
        ConcretizationResult::Single(_) => {}
        other => panic!("write must not inherit the read-range Multiple, got {other:?}"),
    }
}

/// The converse direction of angr-360gt: a store's Max-collapsed `Single` also
/// lives in its own namespace, so a later load re-runs its own chain. That is
/// still consistent — `pin_fallback_addr` asserted `addr == max` when the write
/// fallback fired, so the fresh read concretization sees one solution — but the
/// entry must not be *reused* verbatim, since a write `Single` can equally come
/// from a Max-only chain whose limit the read side never applied.
#[test]
fn read_does_not_reuse_a_write_namespace_entry() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_concretizer(concretizer_with_fallbacks(false, true));

    let addr = RustBV::symbolic(&ctx, "write_then_read", 64);
    interp.concretize_cache.insert(
        VEXInterpreter::bv_cache_key(ConcretizeNs::Write, &addr),
        Arc::new(ConcretizationResult::Multiple(vec![0xdead, 0xbeef])),
    );

    let before = interp.concretize_cache.len();
    let result = interp.concretize_cached_read(&addr);
    assert!(
        !matches!(&*result, ConcretizationResult::Multiple(a) if a == &[0xdead, 0xbeef]),
        "read must not inherit the write-namespace entry, got {result:?}"
    );
    assert_eq!(
        interp.concretize_cache.len(),
        before + 1,
        "the read must have missed and inserted its own ReadJump entry"
    );
}

/// angr-owr37: two addresses whose differing leaf symbol sits >= 2 levels deep
/// — `Add(And(x,0xf),c)` vs `Add(And(y,0xf),c)` — must NOT share a cache key.
/// The original 1-level `bv_cache_key` hashed only `(discriminant, width)` for a
/// nested Expression operand, so both keyed identically and the second
/// symbolic-address store in a block cache-hit the first's target — silent
/// wrong-address memory corruption.
#[test]
fn bv_cache_key_distinguishes_nested_leaf_symbols() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 64);
    let y = RustBV::symbolic(&ctx, "y", 64);
    let mask = RustBV::concrete(0xf, 64);
    let off = RustBV::concrete(0x1800, 64);

    let addr_x = x.and(&mask, &ctx).add(&off, &ctx);
    let addr_y = y.and(&mask, &ctx).add(&off, &ctx);

    // Sanity: these are genuinely nested Expression trees, not simplified leaves.
    assert!(matches!(addr_x, RustBV::Expression { .. }));
    assert!(matches!(addr_y, RustBV::Expression { .. }));

    assert_ne!(
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr_x),
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &addr_y),
        "distinct nested leaf symbols must produce distinct cache keys"
    );
}

/// angr-owr37 hardening: `BVOp` carries inline payload (`Extract(hi,lo)`,
/// `ZeroExt(n)`, `Float{..}`). Hashing only the op discriminant collapsed
/// `Extract(7,0,x)` and `Extract(15,8,x)` to the same key even though they read
/// disjoint bits; the fix hashes the whole op.
#[test]
fn bv_cache_key_distinguishes_op_payload() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 64);

    let lo = x.extract(7, 0, &ctx);
    let hi = x.extract(15, 8, &ctx);
    assert!(matches!(lo, RustBV::Expression { .. }));
    assert!(matches!(hi, RustBV::Expression { .. }));

    assert_ne!(
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &lo),
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &hi),
        "Extract ops reading disjoint bit ranges must produce distinct cache keys"
    );
}

/// angr-9ke6b.89: leaf keys must be variant-tagged. `bv_cache_key` used to
/// shortcut `Concrete` to its raw `value` and `Symbolic`/`Constrained` to their
/// raw `id`, so a `Concrete { value: N }` and a `Symbolic { id: N }` collided on
/// the same key. The three production call sites all early-return on
/// `as_u64()` and never hand a `Concrete`/`Constrained` to `bv_cache_key`, so
/// this was latent — but a future caller that skips that filter would silently
/// cross-wire a concrete address with an unrelated symbol.
#[test]
fn bv_cache_key_distinguishes_leaf_variants_with_equal_scalars() {
    let ctx = SymContext::new_mock();
    let sym = RustBV::symbolic(&ctx, "x", 64);
    let RustBV::Symbolic { id, width, .. } = &sym else {
        panic!("RustBV::symbolic must produce a Symbolic leaf");
    };
    let (id, width) = (*id, *width);

    let concrete = RustBV::concrete(id as u128, width);
    let constrained = RustBV::Constrained {
        id,
        value: id as u128,
        width,
    };

    let keys = [
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &sym),
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &concrete),
        VEXInterpreter::bv_cache_key(ConcretizeNs::ReadJump, &constrained),
    ];
    assert_ne!(keys[0], keys[1], "Symbolic{{id:N}} vs Concrete{{value:N}}");
    assert_ne!(keys[0], keys[2], "Symbolic{{id:N}} vs Constrained{{id:N}}");
    assert_ne!(
        keys[1], keys[2],
        "Concrete{{value:N}} vs Constrained{{id:N}}"
    );
}
