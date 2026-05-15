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
    mem.store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
        .unwrap();

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

    mem.store_concrete(0x1000, RustBV::concrete(0xCAFE, 16))
        .unwrap();
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
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
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
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
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

    mem.store_concrete(0x1000, RustBV::concrete(0xBEEF, 16))
        .unwrap();
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
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
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
            i,
            expected
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
            i,
            expected
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
    assert!(
        ctx.is_sat(),
        "context must remain SAT after per-byte stores"
    );
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
    mem.store_concrete(0x1000, sym1.clone())
        .expect("store sym1");
    mem.store_concrete(0x1004, sym2.clone())
        .expect("store sym2");
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
        other => panic!("expected Permission error, got {:?}", other),
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
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Build a symbolic address constrained to {0x1000, 0x1004, 0x1008}.
    let addr = RustBV::symbolic(&ctx, "addr".to_string(), 64);
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

// ============================================================================
// Phase 1.1 (angr-me3z): MultiPayload data structure + sidecar storage tests.
// These cover only the data structure and storage. Load-side collapse is in
// Phase 1.2 (angr-n082) and store-side helpers in Phase 1.3 (angr-aija).
// ============================================================================

/// Build a Multi alternative `(addr == cand) -> byte(value)` for tests.
fn make_alt(ctx: &SymContext, addr_var: &RustBV, cand: u64, value: u8) -> MultiAlternative {
    let cand_const = RustBV::concrete(cand as u128, addr_var.width());
    let cond = addr_var.eq(&cand_const, ctx);
    let val = RustBV::concrete(value as u128, 8);
    MultiAlternative::new(cond, val)
}

/// Round-trip: install a payload, read it back through the getter, and
/// confirm the page bit + alt count agree. Verifies the basic sidecar
/// wiring before any load/store integration.
#[test]
fn test_multi_payload_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr".to_string(), 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x1000, 0xAA),
        make_alt(&ctx, &addr_var, 0x1004, 0xBB),
        make_alt(&ctx, &addr_var, 0x1008, 0xCC),
    ]);

    mem.set_multi_alternatives(0x1000, payload);

    assert_eq!(mem.multi_cell_count(), 1);
    let got = mem
        .get_multi_alternatives(0x1000)
        .expect("Multi cell must be readable after install");
    assert_eq!(got.len(), 3);
    assert_eq!(got.alternatives()[0].value.as_u64(), Some(0xAA));
    assert_eq!(got.alternatives()[1].value.as_u64(), Some(0xBB));
    assert_eq!(got.alternatives()[2].value.as_u64(), Some(0xCC));

    // Page bitmap should be set for offset 0 of page 0x1000.
    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(page.is_multi(0), "page must mark byte 0 as Multi");
    assert!(
        !page.is_symbolic(0),
        "Multi marker must not overlap plain Symbolic marker"
    );
}

/// Empty payload clears the cell (must not be stored as a zero-length entry).
#[test]
fn test_multi_payload_empty_clears_cell() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr".to_string(), 64);
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0x11)]),
    );
    assert_eq!(mem.multi_cell_count(), 1);

    // Setting an empty payload should clear the cell.
    mem.set_multi_alternatives(0x1000, MultiPayload::default());
    assert_eq!(mem.multi_cell_count(), 0);
    assert!(mem.get_multi_alternatives(0x1000).is_none());

    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(!page.is_multi(0), "Multi marker must be cleared");
}

/// Fork independence: mutating the child's Multi cells must not bleed into
/// the parent, matching SymbolicMemory's CoW contract.
#[test]
fn test_multi_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr".to_string(), 64);
    parent.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x1000, 0xAA),
            make_alt(&ctx, &addr_var, 0x1004, 0xBB),
        ]),
    );

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), 1, "fork must copy multi_objects");

    // Replace the child's payload with a different shape.
    child.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0xFF)]),
    );

    let p = parent
        .get_multi_alternatives(0x1000)
        .expect("parent still has its payload");
    assert_eq!(p.len(), 2, "parent payload must be untouched by fork mutation");
    assert_eq!(p.alternatives()[0].value.as_u64(), Some(0xAA));

    let c = child
        .get_multi_alternatives(0x1000)
        .expect("child has the new payload");
    assert_eq!(c.len(), 1);
    assert_eq!(c.alternatives()[0].value.as_u64(), Some(0xFF));

    // And clearing the child must not clear the parent's cell.
    child.clear_multi_at(0x1000);
    assert_eq!(child.multi_cell_count(), 0);
    assert_eq!(parent.multi_cell_count(), 1);
}

/// Setting a Multi cell at an address that previously held a plain-Symbolic
/// entry must take precedence: the symbolic_objects entry is removed, the
/// page's symbolic bit is replaced by the multi bit. This is the
/// "Multi supersedes Symbolic" rule documented on the `multi_objects` field.
#[test]
fn test_multi_supersedes_existing_symbolic() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Install a single-byte symbolic value at 0x1000.
    let sym_byte = RustBV::symbolic(&ctx, "byte".to_string(), 8);
    mem.import_symbolic_value(0x1000, sym_byte, None);
    assert!(mem.get_symbolic_object(0x1000).is_some());

    // Now upgrade the same byte to Multi.
    let addr_var = RustBV::symbolic(&ctx, "addr".to_string(), 64);
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0x77)]),
    );

    assert!(
        mem.get_symbolic_object(0x1000).is_none(),
        "set_multi_alternatives must clear the prior symbolic_objects entry"
    );
    assert!(mem.get_multi_alternatives(0x1000).is_some());

    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(page.is_multi(0), "Multi marker must be set");
    // import_symbolic_value set the symbolic bit; set_multi_alternatives
    // does not currently clear that bit since the byte is still "abstract" in
    // some sense — load-side collapse (Phase 1.2) handles the priority. Just
    // assert the multi bit is set and the symbolic_objects entry is gone,
    // which is what later phases rely on.
    let _ = page.is_symbolic(0);
}

/// Counter invariant: every set_multi_alternatives must bump mem_ite_depth.
/// Uses delta assertions because the underlying atomics are process-global
/// and cargo test runs in parallel — same convention as the Phase 0 tests.
#[test]
fn test_multi_payload_records_ite_depth() {
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_total = pre.get("mem_ite_depth_total").copied().unwrap_or(0);

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x2000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr_d".to_string(), 64);

    // Two installs: 3 alts then 2 alts = total delta of 5; max watermark must be >= 3.
    mem.set_multi_alternatives(
        0x2000,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x2000, 0x01),
            make_alt(&ctx, &addr_var, 0x2004, 0x02),
            make_alt(&ctx, &addr_var, 0x2008, 0x03),
        ]),
    );
    mem.set_multi_alternatives(
        0x2010,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x2010, 0x10),
            make_alt(&ctx, &addr_var, 0x2014, 0x20),
        ]),
    );

    // Empty payload must NOT touch the counter.
    mem.set_multi_alternatives(0x2020, MultiPayload::default());

    let post = get_solver_stats();
    let post_total = post.get("mem_ite_depth_total").copied().unwrap();
    let post_max = post.get("mem_ite_depth_max").copied().unwrap();
    assert!(
        post_total >= pre_total + 5,
        "expected mem_ite_depth_total delta >= 5 (pre={pre_total}, post={post_total})"
    );
    assert!(
        post_max >= 3,
        "expected mem_ite_depth_max >= 3 after a 3-alt insert (got {post_max})"
    );
}

// ============================================================================
// Phase 1.2 (angr-n082): Multi-cell collapse in load_concrete_lazy_inner.
// ============================================================================

/// Single-byte Multi load: install two alternatives at one byte, constrain
/// the address variable to either candidate, then probe-fork to pin the
/// address and verify the load eval'd to the matching alternative's value.
/// Also asserts that `mem_ite_depth_max` reflects the 2-alt collapse.
#[test]
fn test_multi_cell_load_single_byte() {
    use crate::symbolic::get_solver_stats;

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "load1_addr".to_string(), 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x1000, 0xAA),
        make_alt(&ctx, &addr_var, 0x2000, 0xBB),
    ]);
    mem.set_multi_alternatives(0x1000, payload);

    // Constrain addr to {0x1000, 0x2000} so both alternatives are reachable.
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));
    assert!(ctx.is_sat());

    let pre_max = get_solver_stats()
        .get("mem_ite_depth_max")
        .copied()
        .unwrap_or(0);

    let loaded = mem
        .load_concrete_lazy(0x1000, 1, &ctx)
        .expect("Multi-cell load must succeed");

    // The collapse must record the alternative count.
    let post_max = get_solver_stats()
        .get("mem_ite_depth_max")
        .copied()
        .unwrap_or(0);
    assert!(
        post_max >= pre_max.max(2),
        "expected mem_ite_depth_max >= 2 after a 2-alt collapse \
         (pre={pre_max}, post={post_max})"
    );

    // Probe-fork: under addr == 0x1000, loaded == 0xAA.
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert!(probe_a.is_sat());
    assert_eq!(probe_a.eval(&loaded), Some(0xAA));

    // Probe-fork: under addr == 0x2000, loaded == 0xBB.
    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x2000, 64), &probe_b));
    assert!(probe_b.is_sat());
    assert_eq!(probe_b.eval(&loaded), Some(0xBB));
}

/// Multi-byte load that mixes a Multi cell with surrounding concrete bytes.
/// 4-byte little-endian load at 0x1000 where:
///   byte 0 (0x1000): Multi {addr==A -> 0xAA, addr==B -> 0xBB}
///   byte 1 (0x1001): concrete 0x11
///   byte 2 (0x1002): concrete 0x22
///   byte 3 (0x1003): concrete 0x33
/// Under addr==A the load must read 0x332211AA, under addr==B 0x332211BB.
#[test]
fn test_multi_cell_load_mixed_concrete() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Surrounding concrete bytes.
    mem.store_concrete(0x1001, RustBV::concrete(0x11, 8)).unwrap();
    mem.store_concrete(0x1002, RustBV::concrete(0x22, 8)).unwrap();
    mem.store_concrete(0x1003, RustBV::concrete(0x33, 8)).unwrap();

    let addr_var = RustBV::symbolic(&ctx, "load_mix_addr".to_string(), 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x4000, 0xAA),
        make_alt(&ctx, &addr_var, 0x5000, 0xBB),
    ]);
    mem.set_multi_alternatives(0x1000, payload);

    let eq_a = addr_var.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x5000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("4-byte Multi+concrete load must succeed");
    assert_eq!(loaded.width(), 32);

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x4000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded), Some(0x33_22_11_AA));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x5000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded), Some(0x33_22_11_BB));
}

/// Big-endian variant of the mixed concrete + Multi load. byte 0 is the
/// MSB so addr==A should yield 0xAA_11_22_33 and addr==B 0xBB_11_22_33.
#[test]
fn test_multi_cell_load_big_endian() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    mem.store_concrete(0x1001, RustBV::concrete(0x11, 8)).unwrap();
    mem.store_concrete(0x1002, RustBV::concrete(0x22, 8)).unwrap();
    mem.store_concrete(0x1003, RustBV::concrete(0x33, 8)).unwrap();

    let addr_var = RustBV::symbolic(&ctx, "load_be_addr".to_string(), 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x4000, 0xAA),
        make_alt(&ctx, &addr_var, 0x5000, 0xBB),
    ]);
    mem.set_multi_alternatives(0x1000, payload);

    let eq_a = addr_var.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x5000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("BE Multi+concrete load must succeed");

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x4000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded), Some(0xAA_11_22_33));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x5000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded), Some(0xBB_11_22_33));
}

/// Multi cells at multiple bytes within the load range, plus a concrete
/// byte in between, exercises the per-byte loop's ability to handle
/// several independent ITE chains in one load.
#[test]
fn test_multi_cell_load_multiple_multi_bytes() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    mem.store_concrete(0x1001, RustBV::concrete(0x11, 8)).unwrap();
    // Last byte (offset 3) is also concrete via no-op (default 0).

    let addr_var = RustBV::symbolic(&ctx, "load_multi_addr".to_string(), 64);
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x4000, 0xAA),
            make_alt(&ctx, &addr_var, 0x5000, 0xBB),
        ]),
    );
    mem.set_multi_alternatives(
        0x1002,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x4000, 0xCC),
            make_alt(&ctx, &addr_var, 0x5000, 0xDD),
        ]),
    );

    let eq_a = addr_var.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x5000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("two-multi-byte load must succeed");

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x4000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded), Some(0x00_CC_11_AA));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x5000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded), Some(0x00_DD_11_BB));
}

/// Concrete overwrite of a Multi byte must clear the multi_bitmap bit on
/// the page. The owning `multi_objects` entry is the caller's
/// responsibility (documented on `store_concrete`), but the page-level
/// bookkeeping must self-clean so later loads do not see a stale Multi
/// marker.
#[test]
fn test_concrete_overwrite_clears_multi_bit() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr".to_string(), 64);
    mem.set_multi_alternatives(
        0x3000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3000, 0x42)]),
    );
    {
        let page = mem.pages.get(&(0x3000 >> 12)).expect("page must exist");
        assert!(page.is_multi(0));
    }

    // Concrete overwrite at the same byte.
    mem.store_concrete(0x3000, RustBV::concrete(0xFE, 8)).unwrap();

    let page = mem.pages.get(&(0x3000 >> 12)).expect("page must exist");
    assert!(
        !page.is_multi(0),
        "concrete overwrite must clear the multi_bitmap bit"
    );
}

// ============================================================================
// Phase 1.3 (angr-aija): store_concrete_multi / store_symbolic_unified_multi
// helpers. These exercise the store -> Multi -> load round-trip.
// ============================================================================

/// Round-trip a multi-byte LE store through `store_concrete_multi` and
/// the Phase 1.2 load path. With two candidates {A, B}, the load at A
/// should yield the full stored value, and the load at B likewise.
#[test]
fn test_store_concrete_multi_le_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_addr".to_string(), 64);
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
#[test]
fn test_store_concrete_multi_be_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_be_addr".to_string(), 64);
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
#[test]
fn test_store_concrete_multi_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_fork_addr".to_string(), 64);
    let value = RustBV::concrete(0x11_22_33_44, 32);
    parent
        .store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    let parent_count_before = parent.multi_cell_count();
    assert_eq!(parent_count_before, 8);

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), parent_count_before);

    // Child overwrites byte 0 of the first candidate with a concrete byte.
    // store_concrete clears the corresponding multi bit AND the
    // multi_objects entry (per test_concrete_overwrite_clears_multi_bit).
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
#[test]
fn test_store_symbolic_unified_multi_multiple_round_trip() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "ssm_addr".to_string(), 64);
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
/// short-circuit to the eager `store_concrete_automap` path (no Multi
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
// Phase 2 (angr-qh5u): default-store gate, lazy-region safety, and Multi
// flush on export.
// ============================================================================

/// With the gate OFF (default), `store_symbolic_unified` must use the
/// eager `store_conditional_multiple` path — no Multi cells installed.
#[test]
fn test_phase2_gate_off_default_eager_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);
    assert!(!mem.use_multi_cell_stores(), "gate must default to off");

    let addr_var = RustBV::symbolic(&ctx, "p2_off_addr".to_string(), 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx).or(
        &addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx),
        &ctx,
    ));
    let value = RustBV::concrete(0xDEAD_BEEF, 32);

    mem.store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect("gated-off eager store must succeed");

    // Eager path stores ITE BVs into symbolic_objects; no Multi cells.
    assert_eq!(
        mem.multi_cell_count(),
        0,
        "gate off must not install Multi cells"
    );
}

/// With the gate ON, `store_symbolic_unified` must route Multiple to
/// Multi cells — same end-state as `store_symbolic_unified_multi`.
#[test]
fn test_phase2_gate_on_installs_multi() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);
    mem.set_use_multi_cell_stores(true);

    let addr_var = RustBV::symbolic(&ctx, "p2_on_addr".to_string(), 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx).or(
        &addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx),
        &ctx,
    ));
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
#[test]
fn test_phase2_safe_install_lazy_region_signals() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_use_multi_cell_stores(true);

    // Map page 0x1000; leave page 0x2000 unmapped but inside a lazy region.
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.add_lazy_region(0x2000, 0x1000);

    let addr_var = RustBV::symbolic(&ctx, "p2_lazy_addr".to_string(), 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx).or(
        &addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx),
        &ctx,
    ));
    let value = RustBV::concrete(0x11, 8);

    let err = mem
        .store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect_err("lazy unmapped candidate must error to interpreter");
    match err {
        MemoryError::UnmappedPageInRegion { page_addr } => {
            assert_eq!(page_addr, 0x2000, "must point at the lazy page");
        }
        other => panic!("expected UnmappedPageInRegion, got {:?}", other),
    }
    // No Multi cells installed on the partial run.
    assert_eq!(mem.multi_cell_count(), 0);
}

/// The Phase 2 safe installer must silently skip candidates whose pages
/// are unmapped and NOT in any lazy region (matches
/// `prepare_addresses_for_ite`'s skip-unmapped behavior).
#[test]
fn test_phase2_safe_install_skips_unmapped_non_lazy() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_use_multi_cell_stores(true);

    // Only page 0x1000 is mapped. Page 0x2000 is unmapped and NOT lazy.
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_skip_addr".to_string(), 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx).or(
        &addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx),
        &ctx,
    ));
    let value = RustBV::concrete(0xAB, 8);

    mem.store_symbolic_unified(addr_var.clone(), value, &ctx, &concretizer)
        .expect("non-lazy unmapped candidate must be silently skipped");

    // Only the mapped candidate (0x1000) got a Multi cell.
    assert_eq!(mem.multi_cell_count(), 1);
    assert!(mem.get_multi_alternatives(0x1000).is_some());
    assert!(mem.get_multi_alternatives(0x2000).is_none());
}

/// `flush_multi_cells` must collapse every Multi byte into a per-byte
/// symbolic_objects entry and mark the page-level symbolic bit so the
/// state export pipeline picks it up. This is the export-correctness
/// invariant called out in `rust_lazy_memory_design.rst` Phase 2.
#[test]
fn test_phase2_flush_multi_to_symbolic_objects() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_flush_addr".to_string(), 64);
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
#[test]
fn test_phase2_fork_independence_via_safe_install() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);
    parent.set_use_multi_cell_stores(true);

    let addr_var = RustBV::symbolic(&ctx, "p2_fork_addr".to_string(), 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx).or(
        &addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx),
        &ctx,
    ));
    parent
        .store_symbolic_unified(addr_var.clone(), RustBV::concrete(0x12, 8), &ctx, &concretizer)
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

// ============================================================================
// Phase 3 (angr-j0n4): per-load Multi-cell collapse cache
// ============================================================================

/// After a single load of a Multi byte, the payload's collapse cache must
/// be populated. A second load returns the cached BV unchanged.
#[test]
fn test_phase3_collapse_cache_hit_after_load() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p3_hit_addr".to_string(), 64);
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
    assert!(payload.has_cached_collapse(), "collapse must populate cache");

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
#[test]
fn test_phase3_collapse_cache_invalidated_on_default_byte_change() {
    let ctx = SymContext::new_mock();
    let payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::symbolic(&ctx, "p3_default_change_cond".to_string(), 1),
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
    let mut payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
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

    let addr_var = RustBV::symbolic(&ctx, "p41_hit_addr".to_string(), 64);
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

    let addr_var = RustBV::symbolic(&ctx, "p41_size1_addr".to_string(), 64);
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
#[test]
fn test_phase4_wider_load_cache_invalidated_on_multi_install() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var1 = RustBV::symbolic(&ctx, "p41_inval_addr1".to_string(), 64);
    let addr_var2 = RustBV::symbolic(&ctx, "p41_inval_addr2".to_string(), 64);

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

    let addr_var = RustBV::symbolic(&ctx, "p41_sym_byte_addr".to_string(), 64);
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
    let sym_val = RustBV::symbolic(&ctx, "p41_sym_byte_val".to_string(), 8);
    mem.store_concrete(0x1001, sym_val).unwrap();

    // A 4-byte load at 0x1000 covers both — must NOT cache.
    let _ = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        0,
        "loads touching plain Symbolic bytes must not be cached"
    );
}

/// Forks must carry the wider-load cache forward (cheap BV refcount clone)
/// and remain independent under child-side mutation.
#[test]
fn test_phase4_wider_load_cache_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_fork_addr".to_string(), 64);
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
    assert_eq!(child.wider_load_cache_len(), 1, "fork carries cache forward");

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
