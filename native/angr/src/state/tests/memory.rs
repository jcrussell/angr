//! State-level memory behaviour: the state's own load/store surface, CoW
//! isolation of memory across a fork, reads that must see not-yet-flushed
//! Multi cells, and `apply_changes`'s 16-byte chunking of wide Python
//! SimProcedure writes (memory and register) plus its per-write error handling.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

#[test]
fn test_state_memory() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map and write
    state.map_memory(0x1000, 0x1000, Permission::RWX);
    state
        .memory_store(0x1000, RustBV::concrete(0xDEADBEEF, 32))
        .unwrap();

    // Read back
    let val = state.memory_load(0x1000, 4).unwrap();
    assert_eq!(val.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_state_fork_memory_cow() {
    let mut state1 = RustSimState::new("amd64").unwrap();
    state1.map_memory(0x1000, 0x1000, Permission::RWX);
    state1
        .memory_store(0x1000, RustBV::concrete(0xAAAA, 16))
        .unwrap();

    let mut state2 = state1.fork();

    // Modify state2
    state2
        .memory_store(0x1000, RustBV::concrete(0xBBBB, 16))
        .unwrap();

    // state1 should still have original value
    let val1 = state1.memory_load(0x1000, 2).unwrap();
    assert_eq!(val1.as_u64(), Some(0xAAAA));

    // state2 should have new value
    let val2 = state2.memory_load(0x1000, 2).unwrap();
    assert_eq!(val2.as_u64(), Some(0xBBBB));
}

/// angr-9ke6b.96: `memory_load` — the SimProcedure-facing load exposed as
/// the `memory_load` pymethod — goes through `SymbolicMemory::load_concrete`,
/// not the `_lazy` variant. A prior symbolic-address store that concretized
/// to Multiple/Strided leaves the value in un-flushed Multi cells with the
/// page's symbolic bitmap clear, so before the fix this returned the stale
/// concrete placeholder bytes with no error.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_memory_load_sees_unflushed_multi_cells() {
    let mut state = RustSimState::new("amd64").unwrap();
    // SYMBOLIC_WRITE_ADDRESSES on: without it an unannotated symbolic-address
    // store uses Python's Max-only chain and lands on one address, installing
    // no Multi cells (angr-9ke6b.194).
    state.set_concretizer(crate::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::concretize::AddressConcretizer::new()
    });
    state.map_memory(0x1000, 0x4000, crate::memory::Permission::RWX);

    let addr_var = {
        let ctx = state.solver().borrow();
        let addr_var = RustBV::symbolic(&ctx, "state_multi_addr", 64);
        ctx.assume_true(
            &addr_var
                .eq(&RustBV::concrete(0x1000, 64), &ctx)
                .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
        );
        addr_var
    };

    state
        .memory_store_symbolic(addr_var.clone(), RustBV::concrete(0xDEADBEEF, 32))
        .expect("symbolic-address store must succeed");
    assert_eq!(
        state.memory().multi_cell_count(),
        8,
        "Multiple/Strided concretization installs per-byte Multi cells"
    );

    let loaded = state
        .memory_load(0x1000, 4)
        .expect("memory_load must succeed");
    assert!(
        loaded.as_u64().is_none(),
        "a Multi-covered load must be symbolic, not the placeholder concrete byte"
    );
    let probe = state.solver().borrow().fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(
        probe.eval(&loaded),
        Some(0xDEADBEEF),
        "memory_load must reconstruct the Multi alternative"
    );
}

/// `apply_changes` splits a Python SimProcedure memory write into 16-byte
/// chunks (RustBV is u128-backed), so a buffer wider than one chunk must land
/// byte-for-byte rather than repeating the first 16 bytes.
#[test]
fn apply_changes_chunks_memory_writes_wider_than_16_bytes() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x1000, Permission::RWX);

    let data: Vec<u8> = (0..40u8).collect();
    let mut changes = StateChanges::new();
    changes.memory_writes.push((0x1000, data.clone()));
    state.apply_changes(&changes);

    for (i, &expected) in data.iter().enumerate() {
        let byte = state
            .memory_load(0x1000 + i as u64, 1)
            .expect("mapped load")
            .as_u64();
        assert_eq!(byte, Some(expected as u64), "byte {i} did not land");
    }
}

/// The register file is byte-addressed like memory, so `apply_changes` chunks
/// a register write the same way. A 32-byte YMM slot must land byte-for-byte
/// rather than shift-overflowing the pack loop and truncating to `u128`
/// (angr-xdjsx).
#[test]
fn apply_changes_chunks_register_writes_wider_than_16_bytes() {
    let mut state = RustSimState::new("amd64").unwrap();
    // VEX lays amd64 out with 256-bit YMM slots; XMM0 is the low half of YMM0,
    // so its offset is the base of a 32-byte register.
    let ymm0 = crate::arch::amd64::offsets::XMM0;

    let data: Vec<u8> = (1..=32u8).collect();
    let mut changes = StateChanges::new();
    changes.register_writes.push((ymm0, 32, data.clone()));
    state.apply_changes(&changes);

    for (i, &expected) in data.iter().enumerate() {
        let byte = state.get_register_by_offset(ymm0 + i as u32, 1).as_u64();
        assert_eq!(
            byte,
            Some(u64::from(expected)),
            "register byte {i} did not land"
        );
    }
}

/// A write whose address is unmapped fails inside `store_concrete`.
/// `apply_changes` has no error channel to its callers, so it must warn and
/// keep applying the *remaining* changes instead of aborting the whole sync
/// (angr-sqfj8.87).
#[test]
fn apply_changes_continues_past_a_failed_memory_write() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x1000, 0x1000, Permission::RWX);

    let mut changes = StateChanges::new();
    // Unmapped — store_concrete returns MemoryError::Unmapped.
    changes.memory_writes.push((0xdead_0000, vec![0xAA; 4]));
    // Mapped — must still be applied after the failure above.
    changes
        .memory_writes
        .push((0x1000, vec![0x11, 0x22, 0x33, 0x44]));
    changes.new_pc = Some(0x2000);
    state.apply_changes(&changes);

    assert_eq!(
        state.memory_load(0x1000, 4).expect("mapped load").as_u64(),
        Some(0x4433_2211),
        "a later write must survive an earlier unmapped write"
    );
    assert_eq!(
        state.pc(),
        0x2000,
        "PC update must survive the failed write"
    );
}
