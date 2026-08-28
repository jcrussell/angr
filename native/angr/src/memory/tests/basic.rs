use super::super::*;

#[test]
fn test_memory_concrete() {
    let ctx = SymContext::new_mock();

    let mut mem = SymbolicMemory::new(Endness::Little);

    // Map a page
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Store and load
    let addr = RustBV::concrete(0x1000, 64);
    let value = RustBV::concrete(0x12345678, 32);
    mem.store(&addr, value, &ctx).unwrap();

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
fn test_wide_concrete_store_roundtrip_le() {
    // 32-byte (V256) concrete store must round-trip byte-for-byte; the old
    // u128-funnel path would corrupt bytes >=16 via a masked shift.
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let le_bytes: Vec<u8> = (0..32).map(|i| 0xA0u8 ^ i as u8).collect();
    mem.store_concrete_le_bytes_automap_internal(0x1000u64, &le_bytes)
        .unwrap();

    // Little-endian: value byte i lands at addr+i.
    for (i, &b) in le_bytes.iter().enumerate() {
        let loaded = mem.load_concrete(0x1000 + i as u64, 1, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(b as u64), "LE byte {i} mismatch");
    }
}

#[test]
fn test_wide_concrete_store_roundtrip_be() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let le_bytes: Vec<u8> = (0..32).map(|i| 0xA0u8 ^ i as u8).collect();
    mem.store_concrete_le_bytes_automap_internal(0x1000u64, &le_bytes)
        .unwrap();

    // Big-endian: value byte i lands at addr+(len-1-i).
    let len = le_bytes.len();
    for (i, &b) in le_bytes.iter().enumerate() {
        let dst = 0x1000 + (len - 1 - i) as u64;
        let loaded = mem.load_concrete(dst, 1, &ctx).unwrap();
        assert_eq!(loaded.as_u64(), Some(b as u64), "BE byte {i} mismatch");
    }
}

#[test]
fn test_wide_concrete_load_exact() {
    // Regression (angr-aca6y investigation): a concrete load wider than 16
    // bytes must return the exact bytes. The old u128-funnel path OR-folded
    // bytes >=16 back over the low bytes (`v |= byte << (i*8)` wraps the shift
    // mod 128 in release), so a 24-byte read of "-maxmem\0--debug\0--shell\0"
    // came back as "-msxmmm\0--debug\0-msxmmm\0". The fix assembles wide
    // concrete loads as a Concat of per-byte concretes.
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let bytes: Vec<u8> = (0..32u32)
        .map(|i| (0x11u8).wrapping_mul(i as u8 + 1))
        .collect();
    mem.store_concrete_le_bytes_automap_internal(0x1000u64, &bytes)
        .unwrap();

    // Wide loads (>16 bytes) return a Concat expression — verify byte-for-byte
    // via per-byte extract, which would catch any shift-overflow corruption.
    for &size in &[17u32, 24, 32] {
        let loaded = mem.load_concrete(0x1000, size, &ctx).unwrap();
        assert_eq!(loaded.width(), size * 8, "width for size {size}");
        for i in 0..size {
            let byte = loaded.extract(i * 8 + 7, i * 8, &ctx);
            assert_eq!(
                byte.as_u64(),
                Some(bytes[i as usize] as u64),
                "size {size} byte {i} corrupted"
            );
        }
    }

    // <=16-byte loads still take the fast u128 path and stay exact.
    let lo16 = mem.load_concrete(0x1000, 16, &ctx).unwrap();
    for i in 0..16u32 {
        let byte = lo16.extract(i * 8 + 7, i * 8, &ctx);
        assert_eq!(
            byte.as_u64(),
            Some(bytes[i as usize] as u64),
            "16B byte {i}"
        );
    }
}

#[test]
fn test_wide_concrete_store_unmapped_errors() {
    // A store into an unmapped, non-lazy region must surface an error rather
    // than silently vanish (the old `let _ =` discarded the Result).
    let mut mem = SymbolicMemory::new(Endness::Little);
    let le_bytes = vec![0xFFu8; 32];
    let err = mem.store_concrete_le_bytes_automap_internal(0x9000u64, &le_bytes);
    assert!(err.is_err(), "store into unmapped region should error");
}

#[test]
fn test_map_zero_size_is_noop_regardless_of_alignment() {
    // Regression (angr-n0irt.5): a zero-length map must be a true no-op
    // matching Python's paged_memory_mixin (`while size_done < length` never
    // runs). Previously a non-page-aligned addr with size==0 computed
    // end_page = (addr+0+PAGE_SIZE-1)>>12 > start_page and spuriously mapped
    // exactly one page.
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Non-aligned addr, size 0 — the bug case.
    mem.map(0x1001, 0, Permission::RWX);
    assert!(
        !mem.is_mapped(0x1000),
        "zero-size map at a non-aligned addr must not map any page"
    );

    // Aligned addr, size 0 — always was a no-op; keep it a no-op.
    mem.map(0x2000, 0, Permission::RWX);
    assert!(
        !mem.is_mapped(0x2000),
        "zero-size map at an aligned addr must not map any page"
    );
}

#[test]
fn test_zero_size_access_is_rejected_not_looped() {
    // Regression (angr-9ke6b.99): `end_page = (addr + size - 1) >> 12` underflows
    // at size == 0. The workspace release profile leaves `overflow-checks` off,
    // so instead of panicking it wrapped to u64::MAX >> 12 and `check_perms_range`
    // iterated ~4.5e15 pages — a hang, not an error. addr == 0 is the worst case
    // (it underflows even in a debug build), and enforce_permissions is what turns
    // the bad end_page into the loop. If this test ever hangs, the guard is gone.
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x0, 0x1000, Permission::RWX);
    mem.set_enforce_permissions(true);

    for &addr in &[0u64, 0x400, 0x1000] {
        assert!(
            matches!(
                mem.load_concrete(addr, 0, &ctx),
                Err(MemoryError::ZeroSize { .. })
            ),
            "zero-size load at 0x{addr:x} must be a clean error"
        );
        assert!(
            matches!(
                mem.load_concrete_lazy(addr, 0, &ctx),
                Err(MemoryError::ZeroSize { .. })
            ),
            "zero-size lazy load at 0x{addr:x} must be a clean error"
        );
    }

    // Store side: `size = value.width() / 8`, so any sub-byte-width BV lands on
    // the same underflow. Nothing may be written before the rejection.
    let one_bit = RustBV::concrete(1, 1);
    assert!(matches!(
        mem.store_concrete(0x0u64, one_bit),
        Err(MemoryError::ZeroSize { .. })
    ));
    assert_eq!(
        mem.load_concrete(0x0u64, 1, &ctx).unwrap().as_u64(),
        Some(0),
        "a rejected zero-size store must not have written anything"
    );
}

#[test]
fn test_oversize_access_is_rejected_not_width_wrapped() {
    // angr-0jh0j.35, the upper-bound sibling of the zero-size test above: the
    // load family computes its result width as `size * 8` in u32, which wraps
    // in release past `MAX_ACCESS_BYTES` (u32::MAX / 8). `size == 1 << 29`
    // wraps to width 0 — a BV Z3 cannot represent — and `size * 8 - 1` in the
    // wider-sym extract underflows on top of that.
    //
    // Mutation check: this test DOES differentiate the fix. Without the
    // `check_access_size` guard, `load_concrete` would run the byte loop over
    // half a billion addresses before minting an invalid BV, so the pre-fix
    // behaviour is "hangs then panics", not "returns a wrong value" — which is
    // also why the assertion below is on the error variant and not on a value.
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x0, 0x1000, Permission::RWX);

    let too_big = MAX_ACCESS_BYTES + 1; // 1 << 29, the first wrapping size
    for &addr in &[0u64, 0x400] {
        assert!(
            matches!(
                mem.load_concrete(addr, too_big, &ctx),
                Err(MemoryError::SizeTooLarge { size, .. }) if size == too_big
            ),
            "oversize load at 0x{addr:x} must be a clean error"
        );
        assert!(
            matches!(
                mem.load_concrete_lazy(addr, too_big, &ctx),
                Err(MemoryError::SizeTooLarge { .. })
            ),
            "oversize lazy load at 0x{addr:x} must be a clean error"
        );
    }

    // The bound is exactly the arithmetic one, so it rejects nothing that used
    // to work: the largest width-representable size passes the guard itself.
    // (Asserted on the guard rather than on a `load_concrete` call — a 512 MiB
    // load that got past the guard would walk half a billion byte addresses.)
    assert!(check_access_size(0, MAX_ACCESS_BYTES).is_ok());
    assert_eq!(MAX_ACCESS_BYTES.checked_mul(8), Some(u32::MAX - 7));

    // The infallible fabricate path clamps rather than wrapping: an oversize
    // `size` yields a too-narrow-but-valid BV, not a 0-width one.
    assert_eq!(fabricate_width_bits(too_big), MAX_ACCESS_BYTES * 8);
    assert_eq!(fabricate_width_bits(4), 32);
}

#[test]
fn test_wraparound_access_is_rejected_not_silently_redirected() {
    // Regression (angr-03vl4.34/.35): the mirror image of the size == 0
    // underflow above. `end_page_inclusive`/`end_page_exclusive` used bare `+`,
    // so an access near u64::MAX *overflowed* — and with release overflow-checks
    // off it wrapped to an end_page far below start_page. Every mapped-page and
    // permission loop (`start_page..=end_page`, `check_perms_range`) is then
    // empty, and the byte loop's wrapping `Address` addition walks straight into
    // whatever low page is really mapped. Reachable: `write_fallback_max`
    // concretizes an under-constrained store pointer to exactly u64::MAX.
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // Page 0 is the page a wrapped access lands on. Nothing near u64::MAX is
    // mapped, so a correct engine can only error here.
    mem.map(0x0u64, 0x1000, Permission::RWX);
    mem.set_enforce_permissions(true);
    mem.store_concrete(0x0u64, RustBV::concrete(0x1111_1111, 32))
        .unwrap();

    let wrapping = u64::MAX - 1; // [u64::MAX-1, u64::MAX+2) — overflows by 2.

    assert!(
        matches!(
            mem.load_concrete(wrapping, 4, &ctx),
            Err(MemoryError::OutOfBounds { .. })
        ),
        "a load whose range wraps past u64::MAX must error, not read page 0"
    );
    assert!(matches!(
        mem.load_concrete_lazy(wrapping, 4, &ctx),
        Err(MemoryError::OutOfBounds { .. })
    ));

    // Store side: the eager path (`end_page_inclusive`) and both lazy
    // wrappers (`end_page_exclusive` via `check_pages_mapped_lazy` /
    // the automap loop).
    let poison = RustBV::concrete(0xdead_beef, 32);
    assert!(matches!(
        mem.store_concrete(wrapping, poison.clone()),
        Err(MemoryError::OutOfBounds { .. })
    ));
    assert!(matches!(
        mem.store_concrete_lazy(wrapping, poison.clone()),
        Err(MemoryError::OutOfBounds { .. })
    ));
    assert!(matches!(
        mem.store_concrete_automap_internal(wrapping, poison),
        Err(MemoryError::OutOfBounds { .. })
    ));
    assert_eq!(
        mem.load_concrete(0x0u64, 4, &ctx).unwrap().as_u64(),
        Some(0x1111_1111),
        "no rejected wraparound store may have corrupted the low page"
    );

    // A single byte *at* u64::MAX does not wrap, so it stays a plain unmapped
    // error — the guard must not over-reject the last addressable byte.
    assert!(matches!(
        mem.load_concrete(u64::MAX, 1, &ctx),
        Err(MemoryError::Unmapped { .. })
    ));
}

#[test]
fn test_map_of_last_page_and_of_wrapping_region() {
    // `end_page_exclusive`'s two boundary cases. Mapping the final page has
    // `addr + size == 2^64`, which is a legal range ending exactly at the top
    // of the address space, not an overflow.
    let mut mem = SymbolicMemory::new(Endness::Little);
    let last_page = u64::MAX - 0xFFF;
    mem.map(last_page, 0x1000, Permission::RWX);
    assert!(
        mem.is_mapped(last_page),
        "a region ending exactly at u64::MAX must still map"
    );

    // A genuinely wrapping map is a caller bug: skipped (and warned), never
    // clamped to the last page — clamping would turn map(0, u64::MAX) into 2^52
    // page insertions. If this ever hangs, the reject-don't-clamp rule is gone.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000u64, u64::MAX, Permission::RWX);
    assert!(!mem.is_mapped(0x1000), "a wrapping map must be a no-op");
    mem.map(0x0u64, 0x1000, Permission::RWX);
    mem.unmap(0x1000u64, u64::MAX);
    assert!(
        mem.is_mapped(0x0),
        "a wrapping unmap must not drop unrelated pages"
    );
}

#[test]
fn test_add_lazy_region_of_last_page_and_of_wrapping_region() {
    // angr-03vl4.36: the third `end_page_exclusive` caller, alongside the
    // `map`/`unmap` pair covered by the test above. A wrapping region used to
    // push `(start_page, end_page)` with `start_page > end_page`, which
    // `is_in_lazy_region`'s `page >= start && page < end` can never match — the
    // region was registered but permanently inert, and the pages inside it kept
    // reporting a hard `Unmapped` instead of the intended fetch request.
    let ctx = SymContext::new_mock();

    // Boundary case that must keep working: a region ending exactly at
    // u64::MAX is legal, not an overflow.
    let mut mem = SymbolicMemory::new(Endness::Little);
    let last_page = u64::MAX - 0xFFF;
    mem.add_lazy_region(last_page, 0x1000);
    assert_eq!(mem.lazy_region_count(), 1);
    assert!(
        mem.is_in_lazy_region(u64::MAX >> 12),
        "a lazy region ending exactly at u64::MAX must cover the last page"
    );
    assert!(
        matches!(
            mem.load_concrete_lazy(last_page, 1, &ctx),
            Err(MemoryError::UnmappedPageInRegion { .. })
        ),
        "a miss in the last-page lazy region must ask for a fetch"
    );

    // A genuinely wrapping region is a caller bug: rejected outright, never
    // pushed as an inert entry that silently matches nothing.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.add_lazy_region(0x1000u64, u64::MAX);
    assert_eq!(
        mem.lazy_region_count(),
        0,
        "a wrapping lazy region must not be registered"
    );
    assert!(!mem.is_in_lazy_region(1));

    // Zero size is a no-op even when `start_addr` is not page-aligned — the
    // ceil-div would otherwise round `end_page` up over a page nobody asked for.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.add_lazy_region(0x1800u64, 0);
    assert_eq!(mem.lazy_region_count(), 0);
    assert!(!mem.is_in_lazy_region(1));
    assert!(
        matches!(
            mem.load_concrete_lazy(0x1800u64, 1, &ctx),
            Err(MemoryError::Unmapped { .. })
        ),
        "a zero-size lazy region must not make an unmapped page fetchable"
    );
}

#[test]
fn test_unmap_zero_size_is_noop_regardless_of_alignment() {
    // Sibling of the map() guard: a zero-length unmap must not drop a page.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    assert!(mem.is_mapped(0x1000));

    // Non-aligned addr, size 0 — previously removed page 0x1.
    mem.unmap(0x1001, 0);
    assert!(
        mem.is_mapped(0x1000),
        "zero-size unmap at a non-aligned addr must not drop a mapped page"
    );
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
        other => panic!("expected Permission error, got {other:?}"),
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
        other => panic!("expected Permission error, got {other:?}"),
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
    mem.set_enforce_nx(true);
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
        _ => panic!("expected Permission error, got {err:?}"),
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

#[test]
fn test_snapshot_preserves_dirty_pages() {
    // angr-ype54: `dirty_pages` is NOT a rebuildable cache. The Python
    // callback state is refreshed by replaying these pages
    // (rust_state_sync::_replay_rust_dirty_pages), so a migration
    // round-trip that dropped them stranded every write made since the
    // last callback — the callback state then read stale zeros.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x12345678, 32))
        .unwrap();

    let dirty_before = {
        let mut d = mem.get_dirty_pages();
        d.sort_unstable();
        d
    };
    assert!(!dirty_before.is_empty(), "store must dirty a page");

    let ctx = SymContext::new();
    let proof = mem.flush_multi_cells(&ctx);
    let restored = SymbolicMemory::from_snapshot(mem.to_snapshot(&proof));
    let dirty_after = {
        let mut d = restored.get_dirty_pages();
        d.sort_unstable();
        d
    };
    assert_eq!(dirty_before, dirty_after);
}

// =============================================================================
// Harness 6 (integer-overflow/wraparound boundary sweep). Each historical fix
// below shipped a regression test pinned to the one address its bug report
// used; these sweep the shared `test_boundary_values` table instead, so the
// same shape is checked at every entry rather than only the original one.
// =============================================================================

/// `end_page_inclusive`/`end_page_exclusive` (angr-9ke6b.99, angr-03vl4.34)
/// directly, against an independent `checked_add`-based reference — not just
/// the wrapper `store_concrete`/`map` call sites below, since a future third
/// caller would otherwise only be covered indirectly.
#[test]
fn end_page_helpers_boundary_sweep_matches_checked_add_reference() {
    for &addr in &crate::test_boundary_values::boundary_addresses() {
        for &size in &[0u64, 1, 8, 0x1000, 0x1_0000] {
            let inclusive = super::super::end_page_inclusive(addr, size);
            if size == 0 {
                assert!(
                    matches!(inclusive, Err(MemoryError::ZeroSize { .. })),
                    "size=0 must be ZeroSize for addr={addr:#x}"
                );
            } else {
                match addr.checked_add(size - 1) {
                    // `MemoryError` has no `PartialEq` (angr-irwe `#[non_exhaustive]`),
                    // so unwrap the `Ok` payload rather than `assert_eq!`-ing the
                    // whole `Result`.
                    Some(last) => match inclusive {
                        Ok(got) => assert_eq!(got, last >> 12, "addr={addr:#x} size={size:#x}"),
                        Err(e) => panic!(
                            "expected Ok({:#x}) for addr={addr:#x} size={size:#x}, got Err({e:?})",
                            last >> 12
                        ),
                    },
                    None => assert!(
                        matches!(inclusive, Err(MemoryError::OutOfBounds { .. })),
                        "overflowing addr={addr:#x} size={size:#x} must be OutOfBounds, got {inclusive:?}"
                    ),
                }
            }

            let exclusive = super::super::end_page_exclusive(addr, size);
            match addr.checked_add(size) {
                Some(end) => match exclusive {
                    Ok(got) => assert_eq!(
                        got,
                        end.div_ceil(0x1000),
                        "addr={addr:#x} size={size:#x}"
                    ),
                    Err(e) => panic!(
                        "expected Ok for addr={addr:#x} size={size:#x}, got Err({e:?})"
                    ),
                },
                None if addr.wrapping_add(size) == 0 => {
                    // Region ends exactly at 2^64 — legal, not an overflow.
                    match exclusive {
                        Ok(got) => assert_eq!(got, (u64::MAX >> 12) + 1, "addr={addr:#x} size={size:#x}"),
                        Err(e) => panic!(
                            "expected Ok for exact-top-of-space addr={addr:#x} size={size:#x}, got Err({e:?})"
                        ),
                    }
                }
                None => assert!(
                    matches!(exclusive, Err(MemoryError::OutOfBounds { .. })),
                    "overflowing addr={addr:#x} size={size:#x} must be OutOfBounds, got {exclusive:?}"
                ),
            }
        }
    }
}

/// `store_concrete`/`load_concrete` (angr-03vl4.34/.35): a wraparound near
/// the top of the address space must be rejected as `OutOfBounds`, never
/// silently redirected into the low pages that a `u64` wrap would land on.
#[test]
fn store_and_load_concrete_reject_wraparound_near_top_of_address_space() {
    let ctx = SymContext::new_mock();
    for &addr in &crate::test_boundary_values::boundary_addresses() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        // A real low page, so a wraparound bug (writing through address 0)
        // would leave an observable trace instead of hitting `Unmapped`.
        mem.map(0u64, 0x1000, Permission::RWX);

        let value = RustBV::concrete(0x1122_3344_5566_7788, 64);
        let would_overflow = addr.checked_add(7).is_none();
        match mem.store_concrete(addr, value) {
            Ok(()) => assert!(
                !would_overflow,
                "store_concrete succeeded for overflowing addr={addr:#x}"
            ),
            Err(MemoryError::OutOfBounds { .. }) => assert!(
                would_overflow,
                "store_concrete rejected a non-overflowing addr={addr:#x} as OutOfBounds"
            ),
            // A non-overflowing addr with no page mapped there is a
            // legitimate Unmapped, not an overflow rejection.
            Err(MemoryError::Unmapped { .. }) => assert!(!would_overflow),
            Err(other) => panic!("unexpected error for addr={addr:#x}: {other:?}"),
        }

        // `addr` itself may legitimately target page 0 (the table includes
        // 0/1/0xFFF), in which case the store above is *supposed* to have
        // written there — only check the low page stays untouched for a
        // `addr` whose own page is elsewhere, i.e. an actual wraparound
        // would have to jump there.
        if addr >= 0x1000 {
            let low_byte = mem
                .load_concrete(0u64, 1, &ctx)
                .expect("low page stays mapped and readable");
            assert_eq!(
                low_byte.as_u64(),
                Some(0),
                "a wraparound store must not have landed on page 0 for addr={addr:#x}"
            );
        }
    }
}

/// `map`/`unmap` (angr-03vl4.36): a wrapping request must silently install
/// nothing — but a legal one must still map/unmap normally, so the guard
/// isn't over-broad.
#[test]
fn map_and_unmap_boundary_sweep_noop_on_overflow_else_installs_and_clears() {
    for &addr in &crate::test_boundary_values::boundary_addresses() {
        let size = 0x2000u64; // two pages
        let overflows = addr.checked_add(size).is_none() && addr.wrapping_add(size) != 0;

        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(addr, size, Permission::RWX);
        if overflows {
            assert!(
                !mem.is_mapped(addr),
                "map must not install the start page for overflowing addr={addr:#x}"
            );
        } else {
            assert!(
                mem.is_mapped(addr),
                "map must install the start page for legal addr={addr:#x}"
            );
            mem.unmap(addr, size);
            assert!(
                !mem.is_mapped(addr),
                "unmap must clear the start page for legal addr={addr:#x}"
            );
        }
    }
}

/// `add_lazy_region` (angr-03vl4.36 sibling): same overflow guard as
/// `map`/`unmap`, checked against `lazy_region_count`/`is_addr_in_lazy_region`
/// rather than the page table.
#[test]
fn add_lazy_region_boundary_sweep_matches_overflow_guard() {
    for &addr in &crate::test_boundary_values::boundary_addresses() {
        let size = 0x2000u64;
        let overflows = addr.checked_add(size).is_none() && addr.wrapping_add(size) != 0;

        let mut mem = SymbolicMemory::new(Endness::Little);
        let before = mem.lazy_region_count();
        mem.add_lazy_region(addr, size);
        if overflows {
            assert_eq!(
                mem.lazy_region_count(),
                before,
                "a wrapping lazy region must not be registered for addr={addr:#x}"
            );
        } else {
            assert_eq!(mem.lazy_region_count(), before + 1, "addr={addr:#x}");
            assert!(
                mem.is_addr_in_lazy_region(addr),
                "addr={addr:#x} must be recognized as inside the region it just registered"
            );
        }
    }
}
