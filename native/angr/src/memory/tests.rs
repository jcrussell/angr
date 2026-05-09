use super::*;

#[test]
fn test_memory_concrete() {
    let ctx = SymContext::new_mock();

    let mut mem = SymbolicMemory::new(Endness::Little);

    // Map a page
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Store and load
    let addr = RustBV::concrete(0x1000, 64);
    let value = RustBV::concrete(0x12345678, 32);
    mem.store(addr.clone(), value.clone(), &ctx).unwrap();

    let loaded = mem.load(addr, 4, &ctx).unwrap();
    assert_eq!(loaded.as_u64(), Some(0x12345678));
}

#[test]
fn test_memory_endianness() {
    let ctx = SymContext::new_mock();

    // Little-endian memory
    let mut mem_le = SymbolicMemory::new(Endness::Little);
    mem_le.map(0x1000, 0x1000, Permission::RWX);
    mem_le
        .store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
        .unwrap();

    // First byte should be 0x78 (low byte)
    let byte0 = mem_le.load_concrete(0x1000, 1, &ctx).unwrap();
    assert_eq!(byte0.as_u64(), Some(0x78));

    // Big-endian memory
    let mut mem_be = SymbolicMemory::new(Endness::Big);
    mem_be.map(0x1000, 0x1000, Permission::RWX);
    mem_be
        .store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
        .unwrap();

    // First byte should be 0x12 (high byte)
    let byte0 = mem_be.load_concrete(0x1000, 1, &ctx).unwrap();
    assert_eq!(byte0.as_u64(), Some(0x12));
}

#[test]
fn test_memory_fork() {
    let ctx = SymContext::new_mock();

    let mut mem1 = SymbolicMemory::new(Endness::Little);
    mem1.map(0x1000, 0x1000, Permission::RWX);
    mem1.store_concrete(0x1000, RustBV::concrete(0xAAAA, 16))
        .unwrap();

    // Fork
    let mut mem2 = mem1.fork();

    // Modify mem2
    mem2.store_concrete(0x1000, RustBV::concrete(0xBBBB, 16))
        .unwrap();

    // mem1 should still have original value
    let val1 = mem1.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(val1.as_u64(), Some(0xAAAA));

    // mem2 should have new value
    let val2 = mem2.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(val2.as_u64(), Some(0xBBBB));
}

#[test]
fn test_memory_map_data() {
    let ctx = SymContext::new_mock();

    let mut mem = SymbolicMemory::new(Endness::Little);

    // Map data
    let data: Vec<u8> = (0..=255u8).collect();
    mem.map_data(0x1000, &data, Permission::RX);

    // Check some bytes
    let byte0 = mem.load_concrete(0x1000, 1, &ctx).unwrap();
    assert_eq!(byte0.as_u64(), Some(0));

    let byte100 = mem.load_concrete(0x1064, 1, &ctx).unwrap();
    assert_eq!(byte100.as_u64(), Some(100));
}

#[test]
fn test_symbolic_load_concrete_fast_path() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x12345678, 32)).unwrap();

    // Load with concrete address should work
    let addr = RustBV::concrete(0x1000, 64);
    let loaded = mem.load_symbolic(addr, 4, &ctx, &concretizer).unwrap();
    assert_eq!(loaded.as_u64(), Some(0x12345678));
}

#[test]
fn test_symbolic_store_concrete_fast_path() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Store with concrete address should work
    let addr = RustBV::concrete(0x1000, 64);
    let value = RustBV::concrete(0xDEADBEEF, 32);
    mem.store_symbolic(addr, value, &ctx, &concretizer).unwrap();

    let loaded = mem.load_concrete(0x1000, 4, &ctx).unwrap();
    assert_eq!(loaded.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_permission_enforcement_disabled_by_default() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // Read-only page; without enforcement, stores should still succeed.
    mem.map(0x1000, 0x1000, Permission::R);
    assert!(!mem.enforce_permissions());

    mem.store_concrete(0x1000, RustBV::concrete(0xCAFE, 16)).unwrap();
    let loaded = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(loaded.as_u64(), Some(0xCAFE));
}

#[test]
fn test_permission_enforcement_blocks_write_to_readonly() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::R);
    mem.set_enforce_permissions(true);

    let err = mem
        .store_concrete(0x1000, RustBV::concrete(0xCAFE, 16))
        .unwrap_err();
    match err {
        MemoryError::Permission { addr, required, actual } => {
            assert_eq!(addr, 0x1000);
            assert!(required.write);
            assert!(!actual.write);
            assert!(actual.read);
        }
        other => panic!("expected Permission error, got {:?}", other),
    }
}

#[test]
fn test_permission_enforcement_blocks_read_from_writeonly() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // Write-only page (unusual but exercises the R check independently).
    mem.map(0x1000, 0x1000, Permission::W);
    mem.set_enforce_permissions(true);

    let err = mem.load_concrete(0x1000, 2, &ctx).unwrap_err();
    match err {
        MemoryError::Permission { addr, required, actual } => {
            assert_eq!(addr, 0x1000);
            assert!(required.read);
            assert!(!actual.read);
        }
        other => panic!("expected Permission error, got {:?}", other),
    }
}

#[test]
fn test_permission_enforcement_allows_rwx() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.set_enforce_permissions(true);

    mem.store_concrete(0x1000, RustBV::concrete(0xBEEF, 16)).unwrap();
    let loaded = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(loaded.as_u64(), Some(0xBEEF));
}

#[test]
fn test_permission_enforcement_cross_page_write() {
    // First page RW, second page R. A 4-byte store straddling the
    // boundary at 0x1ffe should fail because the second page is R-only.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RW);
    mem.map(0x2000, 0x1000, Permission::R);
    mem.set_enforce_permissions(true);

    let err = mem
        .store_concrete(0x1ffe, RustBV::concrete(0x11223344, 32))
        .unwrap_err();
    assert!(matches!(err, MemoryError::Permission { .. }));
}

#[test]
fn test_permission_enforcement_propagates_through_fork() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::R);
    mem.set_enforce_permissions(true);

    let mut forked = mem.fork();
    assert!(forked.enforce_permissions());
    let err = forked
        .store_concrete(0x1000, RustBV::concrete(0xDEAD, 16))
        .unwrap_err();
    assert!(matches!(err, MemoryError::Permission { .. }));
}

#[test]
fn test_check_executable_rejects_non_x_page() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RW);
    mem.set_enforce_permissions(true);
    let err = mem.check_executable(0x1234).unwrap_err();
    match err {
        MemoryError::Permission { addr, required, actual } => {
            assert_eq!(addr, 0x1234);
            assert_eq!(required, Permission::X);
            assert_eq!(actual, Permission::RW);
        }
        _ => panic!("expected Permission error, got {:?}", err),
    }
}

#[test]
fn test_check_executable_allows_x_page() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x2000, 0x1000, Permission::RX);
    mem.set_enforce_permissions(true);
    mem.check_executable(0x2010).unwrap();
    // RWX also allows execute.
    mem.map(0x3000, 0x1000, Permission::RWX);
    mem.check_executable(0x3000).unwrap();
}

#[test]
fn test_check_executable_skips_unmapped() {
    // Unmapped pages must NOT error here — callers (native lift /
    // Python lift_block) handle resolution. Only mapped-without-X
    // is a violation.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_enforce_permissions(true);
    mem.check_executable(0xdeadbeef).unwrap();
}

#[test]
fn test_check_executable_noop_when_disabled() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RW);
    // enforce_permissions defaults to false: even RW page must pass.
    mem.check_executable(0x1000).unwrap();
}

/// angr-wyxb: when two symbolic stores partially overlap, the address
/// constraint on each store's address expression and any value
/// constraints must remain in the solver after the stores complete.
#[test]
fn test_symbolic_store_partial_overlap_constraint_propagation() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Symbolic addresses, each pinned to a specific value via a
    // constraint added to the solver up-front.
    let addr1 = RustBV::symbolic(&ctx, "addr1".to_string(), 64);
    let addr2 = RustBV::symbolic(&ctx, "addr2".to_string(), 64);
    ctx.assume_true(&addr1.eq(&RustBV::concrete(0x1000, 64), &ctx));
    ctx.assume_true(&addr2.eq(&RustBV::concrete(0x1004, 64), &ctx));

    // Two 64-bit symbolic values; sym1 carries an additional constraint
    // (a specific u128 value) so we can verify that this value-side
    // constraint also survives the partial overlap.
    let sym1 = RustBV::symbolic(&ctx, "sym1".to_string(), 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2".to_string(), 64);
    let pinned_sym1: u128 = 0xDEAD_BEEF_F00D_BABE;
    ctx.assume_true(&sym1.eq(&RustBV::concrete(pinned_sym1, 64), &ctx));

    // Partial overlap: sym1 covers [0x1000, 0x1008); sym2 covers
    // [0x1004, 0x100C). Bytes [0x1004, 0x1008) are written by both.
    mem.store_symbolic(addr1.clone(), sym1.clone(), &ctx, &concretizer)
        .expect("store_symbolic addr1 must succeed");
    mem.store_symbolic(addr2.clone(), sym2.clone(), &ctx, &concretizer)
        .expect("store_symbolic addr2 must succeed");

    // The base context must remain satisfiable.
    assert!(
        ctx.is_sat(),
        "context must stay SAT after partial-overlap stores"
    );

    // addr1's solution constraint (== 0x1000) must survive: probing
    // an alternative value in a forked context must be UNSAT.
    let probe_addr = ctx.fork();
    probe_addr.assume_true(
        &addr1.eq(&RustBV::concrete(0x2000, 64), &probe_addr),
    );
    assert!(
        !probe_addr.is_sat(),
        "addr1==0x1000 must survive partial-overlap stores; \
         probing addr1==0x2000 was unexpectedly SAT"
    );

    // sym1's value constraint must survive: probing sym1 == 0 must
    // be UNSAT (sym1 is pinned to 0xDEAD_BEEF_F00D_BABE).
    let probe_sym1 = ctx.fork();
    probe_sym1
        .assume_true(&sym1.eq(&RustBV::concrete(0, 64), &probe_sym1));
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
#[test]
fn test_big_endian_128bit_wide_symbolic_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // 128-bit symbolic value pinned to a known constant so we can
    // predict every byte. MSB byte = 0x10, LSB byte = 0x1F.
    let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128".to_string(), 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

    let addr = RustBV::concrete(0x1000, 64);
    mem.store_symbolic(addr, sym.clone(), &ctx, &concretizer)
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
            "BE byte at offset {} expected 0x{:02x}",
            i, expected
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
        "BE 4-byte load at offset 4 expected 0x{:08x}",
        expected_word
    );

    // 8-byte halves: high half at offset 0, low half at offset 8.
    let qhi = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("hi qword load must succeed");
    let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qhi),
        Some(expected_qhi),
        "BE high qword expected 0x{:016x}",
        expected_qhi
    );

    let qlo = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("lo qword load must succeed");
    let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qlo),
        Some(expected_qlo),
        "BE low qword expected 0x{:016x}",
        expected_qlo
    );
}

/// angr-v1q2: 128-bit symbolic store to little-endian memory must lay
/// out bytes LSB-first (byte at addr+0 = LSB, byte at addr+15 = MSB)
/// and sub-word loads must extract the corresponding lanes.
///
/// Counterpart to `test_big_endian_128bit_wide_symbolic_store`. Hits
/// both partial-extract paths in `load_concrete`: the exact-address
/// path (load at 0x1000) and the symbolic_spans path (load at offsets
/// > 0). Before the fix, the LE branch returned MSB-side bytes from
/// the wide BV instead of LSB-side bytes, so single-byte and 4-byte
/// loads at non-zero offsets gave wrong values.
#[test]
fn test_little_endian_128bit_wide_symbolic_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // 128-bit symbolic value pinned to a known constant. For LE, the
    // byte at addr+0 is the LSB (0x1F here) and addr+15 is the MSB.
    let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128_le".to_string(), 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

    let addr = RustBV::concrete(0x1000, 64);
    mem.store_symbolic(addr, sym.clone(), &ctx, &concretizer)
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
            "LE byte at offset {} expected 0x{:02x}",
            i, expected
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
        "LE 4-byte load at offset 4 expected 0x{:08x}",
        expected_word
    );

    // 8-byte halves: low half at offset 0, high half at offset 8.
    let qlo = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("lo qword load must succeed");
    let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qlo),
        Some(expected_qlo),
        "LE low qword expected 0x{:016x}",
        expected_qlo
    );

    let qhi = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("hi qword load must succeed");
    let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qhi),
        Some(expected_qhi),
        "LE high qword expected 0x{:016x}",
        expected_qhi
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
        let sym = RustBV::symbolic(&ctx, format!("byte{}", i), 8);
        ctx.assume_true(&sym.eq(&RustBV::concrete(val, 8), &ctx));
        mem.store_concrete(0x1000 + i as u64, sym).unwrap();
    }
    assert!(ctx.is_sat(), "context must remain SAT after per-byte stores");
    (ctx, mem, pinned)
}

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
        "LE per-byte concat expected 0x{:08x}",
        expected
    );
}

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
        "BE per-byte concat expected 0x{:08x}",
        expected
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
    let sym = RustBV::symbolic(&ctx, "wide128_lin".to_string(), 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));
    // Insert directly into symbolic_objects without populating
    // symbolic_spans, then mark each byte as symbolic on the page so
    // has_symbolic flips during the byte scan.
    mem.symbolic_objects.insert(0x1000, sym);
    let page = mem.pages.get_mut(&(0x1000 >> 12)).expect("page mapped");
    page.mark_symbolic(0, 16);
    (ctx, mem, pinned)
}

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
        "LE linear scan expected 0x{:08x}",
        expected
    );
}

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
        "BE linear scan expected 0x{:08x}",
        expected
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
    let addr = RustBV::symbolic(&ctx, "load_addr".to_string(), 64);
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
         got {:?}",
        solutions
    );
    assert!(
        solutions.contains(&expected_at_2ffc),
        "solver must enumerate the 0x2FFC slice in loaded value; \
         got {:?}",
        solutions
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
    let parent_sym = RustBV::symbolic(&ctx, "parent_wide".to_string(), 64);
    parent.import_symbolic_value(0x1000, parent_sym, None);
    // Sanity: spans for 0x1001..0x1008 exist on parent.
    for off in 1..8u64 {
        assert!(
            parent.symbolic_spans.contains_key(&(0x1000 + off)),
            "parent must have span entry for byte 0x{:x}",
            0x1000 + off
        );
    }
    let parent_span_count_before = parent.symbolic_spans.len();

    // Fork; then write a fresh wide symbolic in the child at a different
    // base. This must NOT add 0x2001..0x2008 to the parent's spans.
    let mut child = parent.fork();
    let child_sym = RustBV::symbolic(&ctx, "child_wide".to_string(), 64);
    child.import_symbolic_value(0x2000, child_sym, None);

    // Parent's symbolic_spans is unchanged.
    assert_eq!(
        parent.symbolic_spans.len(),
        parent_span_count_before,
        "parent symbolic_spans grew after child mutation"
    );
    for off in 1..8u64 {
        assert!(
            !parent.symbolic_spans.contains_key(&(0x2000 + off)),
            "parent leaked span entry for child-only byte 0x{:x}",
            0x2000 + off
        );
        assert!(
            child.symbolic_spans.contains_key(&(0x2000 + off)),
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

    let parent_sym = RustBV::symbolic(&ctx, "parent_imp".to_string(), 32);
    parent.import_symbolic_value(0x1000, parent_sym, None);
    assert!(parent.is_imported_addr(0x1000));
    assert!(!parent.is_imported_addr(0x2000));

    let mut child = parent.fork();
    let child_sym = RustBV::symbolic(&ctx, "child_imp".to_string(), 32);
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
    let p_addr = RustBV::symbolic(&ctx, "p_addr".to_string(), 64);
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

    let c_addr = RustBV::symbolic(&ctx, "c_addr".to_string(), 64);
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
        MemoryError::Permission { addr, required, actual } => {
            assert_eq!(
                addr, 0x2000,
                "Permission error should point at R-only middle page"
            );
            assert_eq!(required, Permission::W);
            assert_eq!(actual, Permission::R);
        }
        other => panic!("expected Permission error, got {:?}", other),
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
        MemoryError::Permission { addr, required, actual } => {
            assert_eq!(
                addr, 0x2000,
                "Permission error should point at R-only middle page \
                 when iterating 3-page range"
            );
            assert_eq!(required, Permission::W);
            assert_eq!(actual, Permission::R);
        }
        other => panic!("expected Permission error, got {:?}", other),
    }
}

/// angr-3zhl: load_concrete must not return a stale wider symbolic
/// object when a later store has partially overwritten its trailing
/// bytes. Setup: store sym1 (64-bit) at 0x1000, then sym2 (64-bit)
/// at 0x1004 — sym2's lower 4 bytes overwrite sym1's upper 4 bytes.
/// A subsequent load(0x1000, 8) must produce concat(sym2[31:0],
/// sym1[31:0]) for LE memory, not the entire sym1 via the
/// exact-address fast path.
#[test]
fn test_load_concrete_partial_overlap_later_store_wins() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Pin two symbolic 64-bit values to known constants so we can
    // predict every byte after the partial overwrite.
    let k1: u128 = 0x1122_3344_5566_7788;
    let k2: u128 = 0xAABB_CCDD_EEFF_0011;
    let sym1 = RustBV::symbolic(&ctx, "sym1_3zhl".to_string(), 64);
    let sym2 = RustBV::symbolic(&ctx, "sym2_3zhl".to_string(), 64);
    ctx.assume_true(&sym1.eq(&RustBV::concrete(k1, 64), &ctx));
    ctx.assume_true(&sym2.eq(&RustBV::concrete(k2, 64), &ctx));

    // Store sym1 at 0x1000 (covers 0x1000..0x1008), then sym2 at
    // 0x1004 (covers 0x1004..0x100C). Bytes 0x1004..0x1008 are now
    // sym2's lower half; bytes 0x1000..0x1004 remain sym1's lower half.
    mem.store_concrete(0x1000, sym1.clone()).expect("store sym1");
    mem.store_concrete(0x1004, sym2.clone()).expect("store sym2");
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
         expected 0x{:016x}, the bug would return sym1 entire (0x{:016x})",
        expected,
        k1,
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
    mem.store_concrete(0x1000, baseline.clone()).unwrap();
    let pre = mem.load_concrete(0x1000, 2, &ctx).unwrap();
    assert_eq!(pre.as_u64(), Some(0xAAAA));

    // Defer a new value at the same concrete address.
    let new_val = RustBV::concrete(0xBBBB, 16);
    mem.add_pending_write(PendingWrite {
        addr: RustBV::concrete(0x1000, 64),
        value: new_val.clone(),
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
        MemoryError::Permission { addr, required, actual } => {
            assert_eq!(
                addr, 0x2000,
                "Permission error should point at W-only middle page"
            );
            assert_eq!(required, Permission::R);
            assert_eq!(actual, Permission::W);
        }
        other => panic!("expected Permission error, got {:?}", other),
    }
}
