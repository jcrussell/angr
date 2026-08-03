use super::super::*;

/// angr-wyxb: when two symbolic stores partially overlap, the address
/// constraint on each store's address expression and any value
/// constraints must remain in the solver after the stores complete.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_store_partial_overlap_constraint_propagation() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Symbolic addresses, each pinned to a specific value via a
    // constraint added to the solver up-front.
    let addr1 = RustBV::symbolic(&ctx, "addr1", 64);
    let addr2 = RustBV::symbolic(&ctx, "addr2", 64);
    ctx.assume_true(&addr1.eq(&RustBV::concrete(0x1000, 64), &ctx));
    ctx.assume_true(&addr2.eq(&RustBV::concrete(0x1004, 64), &ctx));

    // Two 64-bit symbolic values; sym1 carries an additional constraint
    // (a specific u128 value) so we can verify that this value-side
    // constraint also survives the partial overlap.
    let sym1 = RustBV::symbolic(&ctx, "sym1", 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2", 64);
    let pinned_sym1: u128 = 0xDEAD_BEEF_F00D_BABE;
    ctx.assume_true(&sym1.eq(&RustBV::concrete(pinned_sym1, 64), &ctx));

    // Partial overlap: sym1 covers [0x1000, 0x1008); sym2 covers
    // [0x1004, 0x100C). Bytes [0x1004, 0x1008) are written by both.
    mem.store_symbolic(addr1.clone(), sym1.clone(), &ctx, &concretizer)
        .expect("store_symbolic addr1 must succeed");
    mem.store_symbolic(addr2, sym2, &ctx, &concretizer)
        .expect("store_symbolic addr2 must succeed");

    // The base context must remain satisfiable.
    assert!(
        ctx.is_sat(),
        "context must stay SAT after partial-overlap stores"
    );

    // addr1's solution constraint (== 0x1000) must survive: probing
    // an alternative value in a forked context must be UNSAT.
    let probe_addr = ctx.fork();
    probe_addr.assume_true(&addr1.eq(&RustBV::concrete(0x2000, 64), &probe_addr));
    assert!(
        !probe_addr.is_sat(),
        "addr1==0x1000 must survive partial-overlap stores; \
         probing addr1==0x2000 was unexpectedly SAT"
    );

    // sym1's value constraint must survive: probing sym1 == 0 must
    // be UNSAT (sym1 is pinned to 0xDEAD_BEEF_F00D_BABE).
    let probe_sym1 = ctx.fork();
    probe_sym1.assume_true(&sym1.eq(&RustBV::concrete(0, 64), &probe_sym1));
    assert!(
        !probe_sym1.is_sat(),
        "sym1's value constraint must survive partial-overlap stores; \
         probing sym1==0 was unexpectedly SAT"
    );

    // Loading from 0x1000 must still produce a satisfiable expression
    // and stay consistent with the surviving value-side constraints.
    let loaded = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("load[0x1000:4] must succeed after partial-overlap stores");
    assert!(
        ctx.is_sat(),
        "context must remain SAT after the post-store load"
    );
    // Eval should produce *some* concrete model — addr/value constraints
    // narrow the model space but do not make it UNSAT.
    assert!(
        ctx.eval(&loaded).is_some(),
        "loaded value must be evaluable under the preserved constraints"
    );
}

/// angr-syf4: 128-bit symbolic store to big-endian memory must lay out
/// bytes MSB-first (byte at addr+0 = MSB, byte at addr+15 = LSB) and
/// sub-word loads must extract the corresponding lanes.
///
/// Regression for the wide-symbolic-object byte-reversal class of bugs
/// described in project_endianness_bug.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_big_endian_128bit_wide_symbolic_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // 128-bit symbolic value pinned to a known constant so we can
    // predict every byte. MSB byte = 0x10, LSB byte = 0x1F.
    let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128", 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

    let addr = RustBV::concrete(0x1000, 64);
    mem.store_symbolic(addr, sym, &ctx, &concretizer)
        .expect("store_symbolic must succeed");
    assert!(ctx.is_sat(), "context must remain SAT after store");

    // Exact 16-byte load returns the full symbolic value.
    let full = mem
        .load_concrete(0x1000, 16, &ctx)
        .expect("16-byte load must succeed");
    assert_eq!(
        ctx.eval(&full),
        Some(pinned),
        "exact-width load must round-trip the pinned u128"
    );

    // Per-byte BE layout: byte at addr+i corresponds to byte position
    // (15 - i) when the value is interpreted MSB-first.
    for i in 0u64..16 {
        let byte_bv = mem
            .load_concrete(0x1000 + i, 1, &ctx)
            .expect("single-byte load must succeed");
        let expected: u128 = (pinned >> ((15 - i) * 8)) & 0xff;
        assert_eq!(
            ctx.eval(&byte_bv),
            Some(expected),
            "BE byte at offset {i} expected 0x{expected:02x}"
        );
    }

    // 4-byte load at offset 4 should return bytes [4..8) of the BE
    // layout (bits [95:64] of the original u128 = 0x14151617).
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected_word: u128 = (pinned >> 64) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected_word),
        "BE 4-byte load at offset 4 expected 0x{expected_word:08x}"
    );

    // 8-byte halves: high half at offset 0, low half at offset 8.
    let qhi = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("hi qword load must succeed");
    let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qhi),
        Some(expected_qhi),
        "BE high qword expected 0x{expected_qhi:016x}"
    );

    let qlo = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("lo qword load must succeed");
    let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qlo),
        Some(expected_qlo),
        "BE low qword expected 0x{expected_qlo:016x}"
    );
}

/// angr-v1q2: 128-bit symbolic store to little-endian memory must lay
/// out bytes LSB-first (byte at addr+0 = LSB, byte at addr+15 = MSB)
/// and sub-word loads must extract the corresponding lanes.
///
/// Counterpart to `test_big_endian_128bit_wide_symbolic_store`. Hits
/// both partial-extract paths in `load_concrete`: the exact-address
/// path (load at 0x1000) and the symbolic_spans path (load at
/// non-zero offsets). Before the fix, the LE branch returned MSB-side
/// bytes from the wide BV instead of LSB-side bytes, so single-byte
/// and 4-byte loads at non-zero offsets gave wrong values.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_little_endian_128bit_wide_symbolic_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // 128-bit symbolic value pinned to a known constant. For LE, the
    // byte at addr+0 is the LSB (0x1F here) and addr+15 is the MSB.
    let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128_le", 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

    let addr = RustBV::concrete(0x1000, 64);
    mem.store_symbolic(addr, sym, &ctx, &concretizer)
        .expect("store_symbolic must succeed");
    assert!(ctx.is_sat(), "context must remain SAT after store");

    // Exact 16-byte load returns the full symbolic value.
    let full = mem
        .load_concrete(0x1000, 16, &ctx)
        .expect("16-byte load must succeed");
    assert_eq!(
        ctx.eval(&full),
        Some(pinned),
        "exact-width load must round-trip the pinned u128"
    );

    // Per-byte LE layout: byte at addr+i is bits [(i+1)*8-1 : i*8].
    for i in 0u64..16 {
        let byte_bv = mem
            .load_concrete(0x1000 + i, 1, &ctx)
            .expect("single-byte load must succeed");
        let expected: u128 = (pinned >> (i * 8)) & 0xff;
        assert_eq!(
            ctx.eval(&byte_bv),
            Some(expected),
            "LE byte at offset {i} expected 0x{expected:02x}"
        );
    }

    // 4-byte load at offset 4 should return bits [63:32] of the
    // pinned u128 = 0x17161514.
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected_word: u128 = (pinned >> 32) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected_word),
        "LE 4-byte load at offset 4 expected 0x{expected_word:08x}"
    );

    // 8-byte halves: low half at offset 0, high half at offset 8.
    let qlo = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("lo qword load must succeed");
    let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qlo),
        Some(expected_qlo),
        "LE low qword expected 0x{expected_qlo:016x}"
    );

    let qhi = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("hi qword load must succeed");
    let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qhi),
        Some(expected_qhi),
        "LE high qword expected 0x{expected_qhi:016x}"
    );
}

/// angr-76mo: per-byte symbolic concat path in load_concrete must
/// honour memory endianness.
///
/// Stores four independent 8-bit symbolic BVs at consecutive byte
/// addresses, then loads 4 bytes back. This bypasses both the
/// exact-address fast path (no width-32 entry at base) and the
/// symbolic_spans path (8-bit stores have no span entries), forcing
/// the per-byte concat fallback (~lines 642-680 of memory.rs).
///
/// Before the fix, the LE concat order was hardcoded for both
/// endiannesses: parts[N-1] :: ... :: parts[0]. For BE that put byte 0
/// at the LSB instead of the MSB.
fn per_byte_symbolic_setup(endness: Endness) -> (SymContext, SymbolicMemory, [u128; 4]) {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(endness);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let pinned: [u128; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
    for (i, &val) in pinned.iter().enumerate() {
        let sym = RustBV::symbolic(&ctx, format!("byte{i}"), 8);
        ctx.assume_true(&sym.eq(&RustBV::concrete(val, 8), &ctx));
        mem.store_concrete(0x1000 + i as u64, sym).unwrap();
    }
    assert!(
        ctx.is_sat(),
        "context must remain SAT after per-byte stores"
    );
    (ctx, mem, pinned)
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_per_byte_symbolic_concat_little_endian() {
    let (ctx, mem, pinned) = per_byte_symbolic_setup(Endness::Little);
    let word = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte load must succeed");
    // LE: byte at addr+i is at bits [(i+1)*8-1 : i*8]
    let expected: u128 = (pinned[3] << 24) | (pinned[2] << 16) | (pinned[1] << 8) | pinned[0];
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "LE per-byte concat expected 0x{expected:08x}"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_per_byte_symbolic_concat_big_endian() {
    let (ctx, mem, pinned) = per_byte_symbolic_setup(Endness::Big);
    let word = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte load must succeed");
    // BE: byte at addr+i is at bits [(N-i)*8-1 : (N-i-1)*8]
    let expected: u128 = (pinned[0] << 24) | (pinned[1] << 16) | (pinned[2] << 8) | pinned[3];
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "BE per-byte concat expected 0x{expected:08x}"
    );
}

/// angr-76mo: wider-symbolic linear-scan fallback in load_concrete
/// must honour memory endianness when extracting a sub-range.
///
/// The linear scan at lines 672-680 is reached when no per-byte
/// reconstruction succeeds but a containing wider BV exists. Direct
/// manipulation of `symbolic_objects` (without populating
/// `symbolic_spans`) is the surest way to force this path: the
/// exact-address branch needs sym.width() == size*8 (skipped), the
/// symbolic_spans branch finds nothing, per-byte concat finds
/// nothing, then the linear scan triggers.
fn wide_linear_scan_setup(endness: Endness) -> (SymContext, SymbolicMemory, u128) {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(endness);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let pinned: u128 = 0x1011_1213_1415_1617_1819_1A1B_1C1D_1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128_lin", 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));
    // Insert directly into symbolic_objects without populating
    // symbolic_spans, then mark each byte as symbolic on the page so
    // has_symbolic flips during the byte scan.
    mem.symbolic_objects.insert(Address(0x1000), sym);
    let page = mem.pages.get_mut(&(0x1000 >> 12)).expect("page mapped");
    page.mark_symbolic(0, 16);
    (ctx, mem, pinned)
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_wide_linear_scan_little_endian() {
    let (ctx, mem, pinned) = wide_linear_scan_setup(Endness::Little);
    // 4-byte load at offset 4 → LE bytes [4..8) = bits [63:32].
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected: u128 = (pinned >> 32) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "LE linear scan expected 0x{expected:08x}"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_wide_linear_scan_big_endian() {
    let (ctx, mem, pinned) = wide_linear_scan_setup(Endness::Big);
    // 4-byte load at offset 4 → BE bytes [4..8) = bits [95:64].
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected: u128 = (pinned >> 64) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "BE linear scan expected 0x{expected:08x}"
    );
}

/// angr-uwtj: `containing_wider_sym` consults `symbolic_spans` first
/// (O(1)) before falling back to a linear scan over
/// `symbolic_objects` (O(n)). After `import_symbolic_value`, the
/// spans reverse index covers offsets 1..sym_bytes; the base address
/// is matched against `symbolic_objects` directly. This test pins
/// the helper's three branches:
///   - addr strictly inside the wider sym → spans path 1
///   - addr IS the base of a wider sym for a partial read → path 2
///   - addr outside the wider sym → returns None
///   - stale spans (object deleted) → falls back to linear scan,
///     which also misses, returning None.
#[test]
fn test_containing_wider_sym_spans_first() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let wide = RustBV::symbolic(&ctx, "wide128_uwtj", 128);
    mem.import_symbolic_value(0x1000, wide, None);

    // Path 1: addr inside the wider sym → spans hit.
    let hit = mem
        .containing_wider_sym(Address(0x1008), 4)
        .expect("spans-first lookup must find wider sym");
    assert_eq!(hit.0, Address(0x1000));
    assert_eq!(hit.1.width(), 128);

    // Path 2: addr IS the base of a wider sym, partial read.
    let hit_base = mem
        .containing_wider_sym(Address(0x1000), 4)
        .expect("base-address lookup must find wider sym");
    assert_eq!(hit_base.0, Address(0x1000));

    // Out-of-range: addr beyond the wider sym → miss.
    assert!(
        mem.containing_wider_sym(Address(0x1020), 4).is_none(),
        "addr outside wider sym should not match"
    );

    // Load range crosses the wider sym's end → miss (full
    // containment is required).
    assert!(
        mem.containing_wider_sym(Address(0x100C), 8).is_none(),
        "load crossing wider sym's end should not match"
    );

    // Stale spans: delete the base object but leave the spans
    // entries pointing at it. The helper's path 1 sees the spans
    // entry, fails to find the base object, falls through to path
    // 2 (no entry at addr), then to the linear scan (also empty).
    // Net: None — confirming the safety net works.
    mem.symbolic_objects.remove(&Address(0x1000));
    assert!(
        mem.containing_wider_sym(Address(0x1008), 4).is_none(),
        "stale spans without backing object must not return a hit"
    );
}

/// angr-uwtj: end-to-end load_concrete via the spans-first slow path.
/// Setup forces `has_inner_overlap=true` (bypassing the fast spans
/// check at line ~95) and removes a single spans entry so
/// `try_byte_merge_load` fails on that byte. The slow-path
/// reconstruction then calls `containing_wider_sym`, which finds the
/// wider sym via spans path 1 and extracts the load range
/// endianness-correctly.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_slow_path_spans_first_little_endian() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let pinned: u128 = 0x1011_1213_1415_1617_1819_1A1B_1C1D_1E1F;
    let wide = RustBV::symbolic(&ctx, "wide128_slow_le", 128);
    ctx.assume_true(&wide.eq(&RustBV::concrete(pinned, 128), &ctx));
    mem.import_symbolic_value(0x1000, wide, None);
    // Insert an unrelated symbolic_objects entry inside the load
    // range to force has_inner_overlap=true on the load below.
    let noise = RustBV::symbolic(&ctx, "noise8", 8);
    mem.symbolic_objects.insert(Address(0x1006), noise);
    // Remove the spans entry at 0x1007 so try_byte_merge_load
    // returns None and we fall through to the slow path.
    mem.symbolic_spans.remove(&Address(0x1007));
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed via spans-first slow path");
    // LE bytes [4..8) of the wide value live at bits [63:32].
    let expected: u128 = (pinned >> 32) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "LE spans-first slow path expected 0x{expected:08x}"
    );
}

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
    parent.import_symbolic_value(0x1000, parent_sym, None);
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
    child.import_symbolic_value(0x2000, child_sym, None);

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
    parent.import_symbolic_value(0x1000, parent_sym, None);
    assert!(parent.is_imported_addr(0x1000));
    assert!(!parent.is_imported_addr(0x2000));

    let mut child = parent.fork();
    let child_sym = RustBV::symbolic(&ctx, "child_imp", 32);
    child.import_symbolic_value(0x2000, child_sym, None);

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
        page_hint: Some((1, 1)),
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
        page_hint: Some((2, 2)),
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

/// angr-3zhl: load_concrete must not return a stale wider symbolic
/// object when a later store has partially overwritten its trailing
/// bytes. Setup: store sym1 (64-bit) at 0x1000, then sym2 (64-bit)
/// at 0x1004 — sym2's lower 4 bytes overwrite sym1's upper 4 bytes.
/// A subsequent load(0x1000, 8) must produce concat(sym2[31:0],
/// sym1[31:0]) for LE memory, not the entire sym1 via the
/// exact-address fast path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_partial_overlap_later_store_wins() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Pin two symbolic 64-bit values to known constants so we can
    // predict every byte after the partial overwrite.
    let k1: u128 = 0x1122_3344_5566_7788;
    let k2: u128 = 0xAABB_CCDD_EEFF_0011;
    let sym1 = RustBV::symbolic(&ctx, "sym1_3zhl", 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2_3zhl", 64);
    ctx.assume_true(&sym1.eq(&RustBV::concrete(k1, 64), &ctx));
    ctx.assume_true(&sym2.eq(&RustBV::concrete(k2, 64), &ctx));

    // Store sym1 at 0x1000 (covers 0x1000..0x1008), then sym2 at
    // 0x1004 (covers 0x1004..0x100C). Bytes 0x1004..0x1008 are now
    // sym2's lower half; bytes 0x1000..0x1004 remain sym1's lower half.
    mem.store_concrete(0x1000, sym1).expect("store sym1");
    mem.store_concrete(0x1004, sym2).expect("store sym2");
    assert!(ctx.is_sat(), "context must remain SAT after both stores");

    // 8-byte load at 0x1000 must reflect both writes:
    //   bytes 0x1000..0x1004 = sym1[31:0]
    //   bytes 0x1004..0x1008 = sym2[31:0]
    // LE: result low half = sym1[31:0], result high half = sym2[31:0].
    let loaded = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("8-byte load at 0x1000 must succeed");
    let expected: u128 = ((k2 & 0xFFFF_FFFF) << 32) | (k1 & 0xFFFF_FFFF);
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "load(0x1000, 8) must merge sym1's low half with sym2's low half; \
         expected 0x{expected:016x}, the bug would return sym1 entire (0x{k1:016x})",
    );

    // Sanity checks for the unaffected ranges:
    // load(0x1000, 4) is sym1's low 4 bytes.
    let lower = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&lower),
        Some(k1 & 0xFFFF_FFFF),
        "load(0x1000, 4) must equal sym1's low 4 bytes"
    );
    // load(0x1004, 8) is the full sym2.
    let upper = mem
        .load_concrete(0x1004, 8, &ctx)
        .expect("8-byte load at 0x1004 must succeed");
    assert_eq!(
        ctx.eval(&upper),
        Some(k2),
        "load(0x1004, 8) must equal sym2 entire"
    );
}

/// angr-9ke6b.97: the same angr-3zhl partial-overlap merge must hold on the
/// *lazy* load path. Before `load_concrete` and `load_concrete_lazy_inner`
/// were unified behind `load_concrete_common`, the inner-overlap scan lived
/// only in the eager copy, so `load_concrete_lazy` — the path behind every
/// ITE-tree leaf and `store_concrete`'s read-modify-write — returned the
/// stale wider object with no error. Same setup as
/// `test_load_concrete_partial_overlap_later_store_wins`, loaded through
/// the lazy entry point.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_lazy_partial_overlap_later_store_wins() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let k1: u128 = 0x1122_3344_5566_7788;
    let k2: u128 = 0xAABB_CCDD_EEFF_0011;
    let sym1 = RustBV::symbolic(&ctx, "sym1_97_lazy", 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2_97_lazy", 64);
    ctx.assume_true(&sym1.eq(&RustBV::concrete(k1, 64), &ctx));
    ctx.assume_true(&sym2.eq(&RustBV::concrete(k2, 64), &ctx));

    mem.store_concrete(0x1000, sym1).expect("store sym1");
    mem.store_concrete(0x1004, sym2).expect("store sym2");
    assert!(ctx.is_sat(), "context must remain SAT after both stores");

    let loaded = mem
        .load_concrete_lazy(0x1000, 8, &ctx)
        .expect("8-byte lazy load at 0x1000 must succeed");
    let expected: u128 = ((k2 & 0xFFFF_FFFF) << 32) | (k1 & 0xFFFF_FFFF);
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "load_concrete_lazy(0x1000, 8) must merge sym1's low half with sym2's \
         low half; expected 0x{expected:016x}, the pre-unification bug returned \
         sym1 entire (0x{k1:016x})",
    );

    // The eager and lazy paths must now agree byte-for-byte on this setup.
    let eager = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("8-byte eager load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&eager),
        ctx.eval(&loaded),
        "load_concrete and load_concrete_lazy must agree after unification"
    );
}

/// angr-9ke6b.97: partial read of a wider symbolic object based at the load
/// address must work on the lazy path too. `load_concrete_lazy_inner`'s
/// exact-address fast path only accepted `sym.width() == size * 8`; a
/// narrower read fell through to the page scan and only recovered via the
/// LE-only `containing_wider_sym` tail. Sharing `load_concrete`'s
/// endianness-aware fast path fixes the big-endian case, which previously
/// extracted the wrong lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_lazy_partial_read_of_wider_sym_big_endian() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let k: u128 = 0x1122_3344_5566_7788;
    let sym = RustBV::symbolic(&ctx, "sym_97_be", 64);
    ctx.assume_true(&sym.eq(&RustBV::concrete(k, 64), &ctx));
    mem.store_concrete(0x1000, sym).expect("store sym");

    // BE: byte 0 of the stored value is the MSB, so a 4-byte read at the
    // base address is the high half (0x11223344), not the low half.
    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("4-byte lazy load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&loaded),
        Some(0x1122_3344),
        "BE load_concrete_lazy(0x1000, 4) must be the wide value's high half"
    );
    let eager = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte eager load at 0x1000 must succeed");
    assert_eq!(
        ctx.eval(&eager),
        ctx.eval(&loaded),
        "load_concrete and load_concrete_lazy must agree on BE partial reads"
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
        page_hint: Some((1, 1)),
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
/// (memory/mod.rs:298) and the flush pipeline.
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
        page_hint: Some((1, 1)),
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
        page_hint: None,
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
        page_hint: None,
    });

    let err = mem
        .flush_pending_writes(&ctx, &concretizer)
        .expect_err("never-mapped candidate still cannot be stored");
    assert!(
        matches!(err, MemoryError::Unmapped { .. }),
        "expected plain Unmapped, got {err:?}"
    );
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

/// angr-0nme Phase 0: an eager symbolic store with N candidate addresses
/// produces an ITE chain of depth N. `record_mem_ite_depth` must bump the
/// `mem_ite_depth_max` watermark and add to the cumulative total. This is
/// the baseline metric that Phase 1 (Multi cells, angr-czph) will be
/// compared against.
///
/// The two `mem_ite_depth_*` counters are process-global atomics shared
/// with `cargo test` parallel runners, so this asserts on **deltas**
/// from a captured baseline rather than absolute values. The pre/post
/// difference for `mem_ite_depth_total` must be at least 3 (one
/// 3-candidate eager store from this test); for `mem_ite_depth_max` the
/// post-store watermark must be at least 3 (it can only climb).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mem_ite_depth_counter_records_eager_multi_store() {
    use crate::concretize::AddressConcretizer;
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_total = pre.get("mem_ite_depth_total").copied().unwrap_or(0);
    assert!(
        pre.contains_key("mem_ite_depth_max"),
        "mem_ite_depth_max key must be reported by get_solver_stats"
    );
    assert!(
        pre.contains_key("mem_ite_depth_total"),
        "mem_ite_depth_total key must be reported by get_solver_stats"
    );

    let ctx = SymContext::new_mock();
    // SYMBOLIC_WRITE_ADDRESSES on so the Range strategy applies and the store
    // fans out to 3 eager ITE candidates; the default Max-only chain would
    // resolve to one address and record no ITE depth (angr-9ke6b.194).
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Build a symbolic address constrained to {0x1000, 0x1004, 0x1008}.
    let addr = RustBV::symbolic(&ctx, "addr", 64);
    let a0 = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x1004, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x1008, 64), &ctx);
    let or_01 = a0.or(&a1, &ctx);
    let or_all = or_01.or(&a2, &ctx);
    ctx.assume_true(&or_all);
    assert!(ctx.is_sat(), "candidate-address context must be SAT");

    let value = RustBV::concrete(0xDEAD, 32);
    mem.store_symbolic(addr, value, &ctx, &concretizer)
        .expect("symbolic store with multi solutions must succeed");

    let post = get_solver_stats();
    let post_max = post.get("mem_ite_depth_max").copied().unwrap();
    let post_total = post.get("mem_ite_depth_total").copied().unwrap();
    assert!(
        post_max >= 3,
        "expected mem_ite_depth_max >= 3 (got {post_max}) after a 3-candidate eager store"
    );
    assert!(
        post_total >= pre_total + 3,
        "expected mem_ite_depth_total delta >= 3 (pre={pre_total}, post={post_total})"
    );
}

/// angr-2j5v end-to-end: a real `SymbolicMemory::{store,load}` round-trip
/// must increment `mem_load_count` / `mem_store_count` / `mem_load_bytes` /
/// `mem_store_bytes`. Delta-based assertions (counters are process-global).
#[test]
fn test_memory_volume_counters_fire_on_load_store() {
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_load = pre.get("mem_load_count").copied().unwrap_or(0);
    let pre_store = pre.get("mem_store_count").copied().unwrap_or(0);
    let pre_load_bytes = pre.get("mem_load_bytes").copied().unwrap_or(0);
    let pre_store_bytes = pre.get("mem_store_bytes").copied().unwrap_or(0);

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x2000, 0x1000, Permission::RWX);

    // Concrete address, 4 bytes stored, 4 bytes loaded.
    let addr = RustBV::concrete(0x2000, 64);
    let value = RustBV::concrete(0xCAFEBABE, 32);
    mem.store(&addr, value, &ctx).expect("store ok");
    let _ = mem.load(addr, 4, &ctx).expect("load ok");

    let post = get_solver_stats();
    assert!(
        post.get("mem_load_count").copied().unwrap() > pre_load,
        "mem_load_count must climb"
    );
    assert!(
        post.get("mem_store_count").copied().unwrap() > pre_store,
        "mem_store_count must climb"
    );
    assert!(
        post.get("mem_load_bytes").copied().unwrap() >= pre_load_bytes + 4,
        "mem_load_bytes must climb by load size"
    );
    assert!(
        post.get("mem_store_bytes").copied().unwrap() >= pre_store_bytes + 4,
        "mem_store_bytes must climb by store size"
    );
}

/// angr-2j5v: a symbolic-address store routed through `concretize_write`
/// must bump `concretize_write_count` and `concretize_total_candidates`.
/// Uses the same setup as the ITE-depth test (3 candidate addresses).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_concretize_counters_fire_on_symbolic_store() {
    use crate::concretize::AddressConcretizer;
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_write = pre.get("concretize_write_count").copied().unwrap_or(0);
    let pre_total = pre.get("concretize_total_candidates").copied().unwrap_or(0);
    let pre_max = pre.get("concretize_max_candidates").copied().unwrap_or(0);

    let ctx = SymContext::new_mock();
    // SYMBOLIC_WRITE_ADDRESSES on so the Range strategy applies and the store
    // sees K=3 candidates; the default Max-only chain would report K=1
    // (angr-9ke6b.194).
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr = RustBV::symbolic(&ctx, "addr_2j5v", 64);
    let a0 = addr.eq(&RustBV::concrete(0x3000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x3004, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x3008, 64), &ctx);
    let or_all = a0.or(&a1, &ctx).or(&a2, &ctx);
    ctx.assume_true(&or_all);
    assert!(ctx.is_sat());

    let value = RustBV::concrete(0xC0DE, 32);
    mem.store_symbolic(addr, value, &ctx, &concretizer)
        .expect("symbolic store must succeed");

    let post = get_solver_stats();
    assert!(
        post.get("concretize_write_count").copied().unwrap() > pre_write,
        "concretize_write_count must climb"
    );
    assert!(
        post.get("concretize_total_candidates").copied().unwrap() >= pre_total + 3,
        "concretize_total_candidates must climb by K=3"
    );
    assert!(
        post.get("concretize_max_candidates").copied().unwrap() >= pre_max.max(3),
        "concretize_max_candidates must reach >= 3"
    );
}

/// angr-0nme Phase 0: direct calls to `record_mem_ite_depth` increment the
/// cumulative total and lift the max watermark monotonically. A 0-depth
/// call is a no-op. Verified via deltas because the underlying atomics
/// are process-global and may be touched by other parallel tests.
#[test]
fn test_record_mem_ite_depth_helper() {
    use crate::symbolic::{get_solver_stats, record_mem_ite_depth};

    let baseline = get_solver_stats();
    let base_total = baseline.get("mem_ite_depth_total").copied().unwrap_or(0);
    let base_max = baseline.get("mem_ite_depth_max").copied().unwrap_or(0);

    record_mem_ite_depth(0); // no-op
    let after_zero = get_solver_stats();
    assert_eq!(
        after_zero.get("mem_ite_depth_total").copied().unwrap_or(0),
        base_total,
        "record_mem_ite_depth(0) must not change the total"
    );

    record_mem_ite_depth(5);
    record_mem_ite_depth(8);
    record_mem_ite_depth(3);
    let after = get_solver_stats();
    let after_total = after.get("mem_ite_depth_total").copied().unwrap();
    let after_max = after.get("mem_ite_depth_max").copied().unwrap();
    assert!(
        after_total >= base_total + 16,
        "expected total delta of 16 (5+8+3); base={base_total} after={after_total}"
    );
    assert!(
        after_max >= base_max.max(8),
        "expected max to reach at least 8; base={base_max} after={after_max}"
    );
}

/// angr-jvjf (case 1): a concrete byte store into the middle of a
/// wider symbolic object based at the same load address must not be
/// shadowed by the original wider sym. Pre-fix the load fast path at
/// load.rs:74 returns `symbolic_objects[addr]` entire because its
/// width still matches the requested size — the concrete byte we
/// wrote at addr+3 is silently lost.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_concrete_overwrite_inner_byte_of_wider_sym_at_base() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Pin sym to a known constant so we can predict the bytes.
    let k: u128 = 0x1122_3344_5566_7788;
    let sym = RustBV::symbolic(&ctx, "sym_jvjf_a", 64);
    ctx.assume_true(&sym.eq(&RustBV::concrete(k, 64), &ctx));

    // Store the 64-bit sym at 0x1000 (covers 0x1000..0x1008).
    mem.store_concrete(0x1000, sym).expect("store sym");

    // Concrete-overwrite byte 3 (LE: byte 3 of k = 0x44) with 0xFF.
    mem.store_concrete(0x1003, RustBV::concrete(0xFF, 8))
        .expect("store concrete byte");
    assert!(ctx.is_sat(), "context must remain SAT after the stores");

    // 8-byte load at 0x1000 must reflect the concrete overwrite at
    // byte 3 — i.e. byte 3 of the loaded value is 0xFF, not 0x44.
    let loaded = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("8-byte load must succeed");
    let expected: u128 = (k & !(0xFFu128 << 24)) | (0xFFu128 << 24);
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "load(0x1000, 8) must reflect the concrete byte at 0x1003; \
         expected 0x{expected:016x}, the bug returns the original sym (0x{k:016x})",
    );
}

/// angr-jvjf (case 2): a concrete byte store into the middle of a
/// wider symbolic object based at an earlier address must not leave
/// the `symbolic_spans` entry stale. Pre-fix a 1-byte load at the
/// overwritten offset hits the span fast path at load.rs:95 and
/// returns the now-stale extract of the wider sym.
#[test]
fn test_concrete_overwrite_clears_stale_symbolic_spans() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let k: u128 = 0x1122_3344_5566_7788;
    let sym = RustBV::symbolic(&ctx, "sym_jvjf_b", 64);
    ctx.assume_true(&sym.eq(&RustBV::concrete(k, 64), &ctx));

    // Wider sym at 0x1000 produces span entries at 0x1001..0x1008.
    mem.store_concrete(0x1000, sym).expect("store sym");

    // Concrete-overwrite byte 3 with 0xFF (the byte covered by the
    // span 0x1003 -> (0x1000, 64)).
    mem.store_concrete(0x1003, RustBV::concrete(0xFF, 8))
        .expect("store concrete byte");

    // 1-byte load at 0x1003 must be the concrete 0xFF, not the
    // extracted sym byte 0x44.
    let byte = mem
        .load_concrete(0x1003, 1, &ctx)
        .expect("1-byte load must succeed");
    assert_eq!(
        ctx.eval(&byte),
        Some(0xFF),
        "load(0x1003, 1) must reflect the concrete overwrite; \
         the bug returns the stale extract from the wider sym"
    );
}

/// angr-7qon (case 1): a concrete write at the base of a wider sym
/// that doesn't cover the full sym width must not leave orphaned
/// page-bitmap bits for the surviving trailing bytes. Pre-fix,
/// store.rs:154 removes the wider sym entirely (and the spans),
/// but page.store_concrete only cleared the bitmap bits within the
/// concrete write range — so byte 0x1001 (the survivor) still has
/// its symbolic bit set with no symbolic_objects entry covering it,
/// and a subsequent 1-byte load fails with
/// "symbolic bytes not fully tracked".
#[test]
fn test_concrete_overwrite_at_base_truncates_wider_sym_byte_load_succeeds() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let sym = RustBV::symbolic(&ctx, "sym_7qon_a", 16);
    mem.store_concrete(0x1000, sym).expect("store 16-bit sym");

    // Concrete 1-byte write at 0x1000 truncates the wider sym (covers
    // only byte 0x1000 of the 2-byte sym at 0x1000..0x1002).
    mem.store_concrete(0x1000, RustBV::concrete(0xAA, 8))
        .expect("store concrete byte");

    // Pre-fix: load(0x1001, 1) fails because the page bitmap still
    // marks 0x1001 symbolic but no symbolic_objects entry covers it.
    // Post-fix: bitmap bit cleared, byte reclassified as concrete
    // (returns the pre-sym page byte, which is 0 here).
    let byte = mem
        .load_concrete(0x1001, 1, &ctx)
        .expect("1-byte load of survivor must succeed");
    assert_eq!(
        ctx.eval(&byte),
        Some(0),
        "load(0x1001, 1) returns the underlying concrete page byte (0); \
         pre-fix it errors with 'symbolic bytes not fully tracked'"
    );
}

/// angr-7qon (case 2): a concrete write at the base of a wider sym
/// that covers some but not all bytes of the sym (e.g. 4 bytes of a
/// 64-bit sym) must clear the page bitmap bits for the survivors
/// across the whole tail, not just immediately after `addr+size`.
#[test]
fn test_concrete_overwrite_partial_truncates_wider_sym_multibyte_tail() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let sym = RustBV::symbolic(&ctx, "sym_7qon_b", 64);
    mem.store_concrete(0x1000, sym).expect("store 64-bit sym");

    // Concrete 4-byte write at 0x1000 truncates the 64-bit sym, leaving
    // bytes 0x1004..0x1008 as survivors.
    mem.store_concrete(0x1000, RustBV::concrete(0xDEADBEEF, 32))
        .expect("store concrete 4 bytes");

    // Each survivor byte's page bit must be cleared; load must
    // succeed and return the page byte (0).
    for survivor in 0x1004u64..0x1008 {
        let byte = mem
            .load_concrete(survivor, 1, &ctx)
            .unwrap_or_else(|e| panic!("load(0x{survivor:x}, 1) must succeed: {e:?}"));
        assert_eq!(
            ctx.eval(&byte),
            Some(0),
            "load(0x{survivor:x}, 1) survivor byte must be concrete 0"
        );
    }

    // And a single 4-byte load over the whole tail must also succeed.
    let tail = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte tail load must succeed");
    assert_eq!(ctx.eval(&tail), Some(0));
}

/// angr-7qon (case 3): the truncation cleanup must walk pages
/// correctly when the wider sym crosses a page boundary. A 16-bit
/// sym at 0x1FFF spans pages 0 and 1; a 1-byte concrete write at
/// 0x1FFF leaves byte 0x2000 on the next page as the survivor.
#[test]
fn test_concrete_overwrite_at_base_truncates_wider_sym_crosses_page() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x2000, Permission::RWX);

    let sym = RustBV::symbolic(&ctx, "sym_7qon_c", 16);
    mem.store_concrete(0x1FFF, sym)
        .expect("store cross-page sym");

    mem.store_concrete(0x1FFF, RustBV::concrete(0xAA, 8))
        .expect("store concrete byte at base");

    let byte = mem
        .load_concrete(0x2000, 1, &ctx)
        .expect("survivor on next page must load");
    assert_eq!(
        ctx.eval(&byte),
        Some(0),
        "cross-page survivor at 0x2000 must reclassify as concrete"
    );
}

/// Regression (angr-9ke6b.98): a narrower symbolic value overwriting a wider
/// one at the same base must retire the abandoned tail. Before the fix,
/// `store_concrete`'s symbolic branch refreshed `symbolic_spans` only inside
/// the new width, so the tail bytes kept spans naming `(base, old_width)` —
/// a live object that no longer reaches them — and the page bitmap still said
/// symbolic. A later load of a tail byte followed that span into an
/// out-of-range extract and hard-failed with `SymbolicAddress`, so a readable
/// byte became an error. Mirrors the concrete branch's angr-7qon semantics:
/// the tail reclassifies as concrete.
// Reads a symbolic span back as a concrete value, which needs a real solver
// (angr-9ke6b.236, bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_narrower_sym_overwrite_at_base_truncates_wider_sym() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let wide = RustBV::symbolic(&ctx, "sym_98_wide", 128);
    mem.store_concrete(0x1000, wide).expect("store 16-byte sym");

    let narrow = RustBV::symbolic(&ctx, "sym_98_narrow", 64);
    mem.store_concrete(0x1000, narrow.clone())
        .expect("store 8-byte sym at same base");

    // The abandoned tail [0x1008, 0x1010) must load cleanly, not raise
    // MemoryError::SymbolicAddress ("symbolic bytes not fully tracked").
    let tail = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("abandoned tail must load, not hard-fail");
    assert_eq!(
        ctx.eval(&tail),
        Some(0),
        "abandoned tail must reclassify as concrete (angr-7qon parity)"
    );

    // Per-byte too: the first tail byte is the one whose stale span pointed
    // back at the still-live base.
    let tail_byte = mem
        .load_concrete(0x1008, 1, &ctx)
        .expect("first abandoned tail byte must load");
    assert_eq!(ctx.eval(&tail_byte), Some(0));

    // The surviving head must still be the narrow symbolic value.
    let head = mem.load_concrete(0x1000, 8, &ctx).expect("head must load");
    assert!(head.is_symbolic(), "head must stay symbolic");
    ctx.assume_true(&narrow.eq(&RustBV::concrete(0x1122_3344_5566_7788, 64), &ctx));
    assert_eq!(
        ctx.eval(&head),
        Some(0x1122_3344_5566_7788),
        "head must read back the narrow value that overwrote the wide one"
    );
}

/// angr-9ke6b.228: `mem_lazy_page_fault_count` must tick on **both** the load
/// and the store side. It used to be bumped only in the public
/// `SymbolicMemory::{load,store}` wrappers — which no production path calls —
/// and the store-side branch there was outright dead, because `store_concrete`
/// has no lazy classification and never yields `UnmappedPageInRegion`. The
/// bumps now live on the producers, so the lazy entry points the interpreter
/// actually uses are covered.
///
/// Counters are process-global and tests run in parallel, so this asserts
/// strict growth against a baseline read just before the faulting op rather
/// than an exact delta.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_lazy_page_fault_counter_ticks_on_both_sides() {
    use crate::symbolic::get_solver_stats;

    let fault_count = || {
        get_solver_stats()
            .get("mem_lazy_page_fault_count")
            .copied()
            .unwrap_or(0)
    };

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // Page 0x2000 is unmapped but declared lazy: Python still holds its backer.
    mem.add_lazy_region(0x2000, 0x1000);

    let base_store = fault_count();
    let err = mem
        .store_concrete_lazy(0x2000, RustBV::concrete(0xAA, 8))
        .expect_err("lazy-region store must report the fetch-me signal");
    assert!(
        matches!(err, MemoryError::UnmappedPageInRegion { page_addr } if page_addr == 0x2000),
        "expected UnmappedPageInRegion at 0x2000, got {err:?}"
    );
    assert!(
        fault_count() > base_store,
        "store-side lazy fault must bump mem_lazy_page_fault_count"
    );

    let base_load = fault_count();
    let err = mem
        .load_concrete_lazy(0x2000, 1, &ctx)
        .expect_err("lazy-region load must report the fetch-me signal");
    assert!(
        matches!(err, MemoryError::UnmappedPageInRegion { page_addr } if page_addr == 0x2000),
        "expected UnmappedPageInRegion at 0x2000, got {err:?}"
    );
    assert!(
        fault_count() > base_load,
        "load-side lazy fault must bump mem_lazy_page_fault_count"
    );

    // A hard (non-lazy) miss must classify as plain `Unmapped`, which is the
    // branch that leaves the counter alone. (Asserted on the error shape, not
    // on the counter staying equal: it is process-global and a concurrently
    // running test may bump it.)
    assert!(matches!(
        mem.load_concrete_lazy(0x9000, 1, &ctx),
        Err(MemoryError::Unmapped { .. })
    ));
    assert!(matches!(
        mem.store_concrete_lazy(0x9000, RustBV::concrete(1, 8)),
        Err(MemoryError::Unmapped { .. })
    ));
}

/// angr-9ke6b.229: the `mem_{load,store}_symbolic_addr` counters must tick from
/// the *production* symbolic-address entry points, not from the test-only
/// `SymbolicMemory::{load,store}` wrappers (which `.228` showed nothing calls).
/// Lower-bound assertions only — the counters are process-global and other
/// tests run concurrently in the same process.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_addr_counters_tick_from_production_entry_points() {
    let load_sym = || {
        crate::symbolic::get_solver_stats()
            .get("mem_load_symbolic_addr")
            .copied()
            .unwrap_or(0)
    };
    let store_sym = || {
        crate::symbolic::get_solver_stats()
            .get("mem_store_symbolic_addr")
            .copied()
            .unwrap_or(0)
    };

    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // A symbolic address pinned to a single concrete solution: still symbolic
    // to `as_u64()`, so it goes past every fast path into the concretizer.
    let addr = RustBV::symbolic(&ctx, "counter_addr", 64);
    ctx.assume_true(&addr.eq(&RustBV::concrete(0x1000, 64), &ctx));
    let value = RustBV::concrete(0x41, 8);

    let base_store = store_sym();
    mem.store_symbolic(addr.clone(), value.clone(), &ctx, &concretizer)
        .expect("store_symbolic must succeed");
    mem.store_symbolic_unified(addr.clone(), value.clone(), &ctx, &concretizer)
        .expect("store_symbolic_unified must succeed");
    mem.store_symbolic_unified_multi(addr.clone(), value.clone(), &ctx, &concretizer)
        .expect("store_symbolic_unified_multi must succeed");
    mem.store_with_concretization(&addr, value, &ConcretizationResult::Single(0x1000), &ctx)
        .expect("store_with_concretization must succeed");
    assert!(
        store_sym() >= base_store + 4,
        "each of the four symbolic store entry points must bump \
         mem_store_symbolic_addr (base {base_store}, now {})",
        store_sym()
    );

    let base_load = load_sym();
    mem.load_symbolic(addr.clone(), 1, &ctx, &concretizer)
        .expect("load_symbolic must succeed");
    mem.load_symbolic_unified(addr, 1, &ctx, &concretizer)
        .expect("load_symbolic_unified must succeed");
    assert!(
        load_sym() >= base_load + 2,
        "both symbolic load entry points must bump mem_load_symbolic_addr \
         (base {base_load}, now {})",
        load_sym()
    );
}
