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
