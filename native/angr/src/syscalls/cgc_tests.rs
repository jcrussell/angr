// angr-cgc-tests: CGC syscall handler unit tests, extracted out of the former
// in-file `mod tests` (~666 lines) into a sibling file to shrink
// syscalls/cgc.rs below the god-object threshold. Declared as a direct child
// of `cgc` so `use super::*` reaches the module's private items.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

fn x86_state_with_buf() -> RustSimState {
    let mut state = RustSimState::new("x86").expect("x86 state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
}

#[test]
fn terminate_returns_exit_outcome() {
    let h = NativeTerminateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h.call(&mut state, &[]).expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Exit));
}

#[test]
fn transmit_writes_to_stdout_and_stores_count() {
    let h = NativeTransmitSyscall;
    let mut state = x86_state_with_buf();
    state.map_memory_data(0x3000, b"hello", Permission::RWX);

    // fd=1 (stdout), buf=0x3000, count=5, tx_bytes=0x2000.
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 32),
                RustBV::concrete(0x3000, 32),
                RustBV::concrete(5, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.stdout_buffer(), b"hello");
    let stored = state.memory_load(0x2000, 4).expect("load").as_u64();
    assert_eq!(stored, Some(5));
}

#[test]
fn transmit_with_null_tx_bytes_skips_store() {
    let h = NativeTransmitSyscall;
    let mut state = x86_state_with_buf();
    state.map_memory_data(0x3000, b"x", Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 32),
                RustBV::concrete(0x3000, 32),
                RustBV::concrete(1, 32),
                RustBV::concrete(0, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    assert_eq!(state.stdout_buffer(), b"x");
}

#[test]
fn transmit_symbolic_byte_falls_back() {
    let h = NativeTransmitSyscall;
    let mut state = x86_state_with_buf();
    state.map_memory(0x3000, 0x1000, Permission::RWX);
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "b", 8);
    state.memory_store(0x3000, sym).expect("store");
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 32),
                RustBV::concrete(0x3000, 32),
                RustBV::concrete(1, 32),
                RustBV::concrete(0, 32),
            ],
        )
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    // No partial transmit.
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn transmit_unknown_fd_falls_back() {
    let h = NativeTransmitSyscall;
    let mut state = x86_state_with_buf();
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(7, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(1, 32),
                RustBV::concrete(0, 32),
            ],
        )
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn receive_stdin_writes_symbolic_and_stores_count() {
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(4, 32),
                RustBV::concrete(0x2800, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    for i in 0..4u64 {
        let byte = state.memory_load(0x2000 + i, 1).expect("load");
        assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
    }
    let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
    assert_eq!(stored, Some(4));
}

#[test]
fn receive_stdin_records_stdin_symbols() {
    // Regression (angr-vx8p.3): each fresh symbolic byte from a CGC
    // stdin `receive` must be tracked under state.stdin_symbols so the
    // Python-side _inject_rust_stdin can feed posix.dumps(0). Mirrors
    // read.rs::test_read_records_stdin_symbols.
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    assert!(!state.has_stdin_symbols());
    let _ = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(4, 32),
                RustBV::concrete(0x2800, 32),
            ],
        )
        .expect("ok");
    let symbols = state.stdin_symbols();
    assert_eq!(symbols.len(), 4);
    for (name, bits) in symbols {
        assert_eq!(*bits, 8);
        assert!(name.starts_with("cgc_receive_"), "got {name}");
    }
}

#[test]
fn receive_non_stdin_fd_records_no_stdin_symbols() {
    // A non-stdin receive falls back to Python and must NOT pollute
    // stdin_symbols.
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    let _ = h.call(
        &mut state,
        &[
            RustBV::concrete(1, 32),
            RustBV::concrete(0x2000, 32),
            RustBV::concrete(4, 32),
            RustBV::concrete(0x2800, 32),
        ],
    );
    assert!(!state.has_stdin_symbols());
}

#[test]
fn receive_zero_count_writes_zero_count_and_returns() {
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    state.map_memory_data(0x2000, b"abcd", Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2800, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    // The buffer at 0x2000 is untouched.
    for (i, &b) in b"abcd".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(b as u64));
    }
    let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
    assert_eq!(stored, Some(0));
}

#[test]
fn receive_non_stdin_fd_falls_back() {
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(4, 32),
                RustBV::concrete(0, 32),
            ],
        )
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

/// A zero-length receive on a non-stdin fd must still fall back to Python
/// rather than reporting success: Python's `procedures/cgc/receive.py::run`
/// resolves the fd and returns -1 for an unopened one before it looks at
/// `count`, so short-circuiting on `count == 0` would silently model a
/// never-opened fd as valid (angr-fs583).
#[test]
fn receive_zero_count_non_stdin_fd_falls_back() {
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    state.map_memory_data(0x2800, &[0xAA; 4], Permission::RWX);

    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(999, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2800, 32),
            ],
        )
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    // The rx_bytes out-param must be untouched — writing 0 there would leave
    // the state claiming a successful zero-byte read on a bogus fd.
    let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
    assert_eq!(stored, Some(0xAAAA_AAAA));
}

/// fdwait's native stub only models the `CGC_NON_BLOCKING_FDS` mode, so
/// every fdwait happy-path test must opt into it (angr-op0dn.14.8).
fn nonblocking_state() -> RustSimState {
    let mut state = x86_state_with_buf();
    state.set_option("CGC_NON_BLOCKING_FDS", true);
    state
}

#[test]
fn fdwait_falls_back_without_non_blocking_fds() {
    let h = NativeFdwaitSyscall;
    let mut state = x86_state_with_buf();
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(4, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(0x2100, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2200, 32),
            ],
        )
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn fdwait_sets_masks_and_total() {
    let h = NativeFdwaitSyscall;
    let mut state = nonblocking_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(4, 32),      // nfds
                RustBV::concrete(0x2000, 32), // readfds
                RustBV::concrete(0x2100, 32), // writefds
                RustBV::concrete(0, 32),      // timeout (null)
                RustBV::concrete(0x2200, 32), // readyfds
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let rd = state.memory_load(0x2000, 4).expect("load").as_u64();
    assert_eq!(rd, Some(0b1111));
    let wr = state.memory_load(0x2100, 4).expect("load").as_u64();
    assert_eq!(wr, Some(0b1111));
    let total = state.memory_load(0x2200, 4).expect("load").as_u64();
    assert_eq!(total, Some(8));
}

#[test]
fn fdwait_nfds_zero_yields_zero_mask() {
    let h = NativeFdwaitSyscall;
    let mut state = nonblocking_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2200, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let rd = state.memory_load(0x2000, 4).expect("load").as_u64();
    assert_eq!(rd, Some(0));
    let total = state.memory_load(0x2200, 4).expect("load").as_u64();
    assert_eq!(total, Some(0));
}

#[test]
fn fdwait_clamps_to_32_fds() {
    let h = NativeFdwaitSyscall;
    let mut state = nonblocking_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(100, 32),
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2200, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let rd = state.memory_load(0x2000, 4).expect("load").as_u64();
    assert_eq!(rd, Some(u32::MAX as u64));
    // 32 fds counted once per mask loop; the null `writefds` suppresses the
    // *store*, not the count -- see the Python proc (angr-cslvl).
    let total = state.memory_load(0x2200, 4).expect("load").as_u64();
    assert_eq!(total, Some(64));
}

/// `procedures/cgc/fdwait.py` accumulates `total_ready` over both the read
/// and the write fd loop unconditionally, and only guards the mask stores on
/// a non-null pointer. So a null `readfds` must still contribute its
/// `min(nfds, 32)` to the count -- the native stub used to skip it, which
/// under-reported readiness by half (angr-cslvl).
#[test]
fn fdwait_counts_null_masks_like_python() {
    let h = NativeFdwaitSyscall;
    let mut state = nonblocking_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(4, 32),      // nfds
                RustBV::concrete(0, 32),      // readfds (null)
                RustBV::concrete(0x2100, 32), // writefds
                RustBV::concrete(0, 32),      // timeout (null)
                RustBV::concrete(0x2200, 32), // readyfds
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let wr = state.memory_load(0x2100, 4).expect("load").as_u64();
    assert_eq!(wr, Some(0b1111));
    let total = state.memory_load(0x2200, 4).expect("load").as_u64();
    assert_eq!(total, Some(8));
}

/// Both masks null: nothing is stored, but the count is still `nfds * 2`.
#[test]
fn fdwait_both_masks_null_still_counts() {
    let h = NativeFdwaitSyscall;
    let mut state = nonblocking_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2200, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let total = state.memory_load(0x2200, 4).expect("load").as_u64();
    assert_eq!(total, Some(6));
}

#[test]
fn random_writes_symbolic_and_stores_count() {
    let h = NativeRandomSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(8, 32),
                RustBV::concrete(0x2800, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    for i in 0..8u64 {
        let byte = state.memory_load(0x2000 + i, 1).expect("load");
        assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
    }
    let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
    assert_eq!(stored, Some(8));
}

#[test]
fn allocate_zero_length_returns_einval() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(
        outcome,
        SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
    ));
}

#[test]
fn allocate_oversize_length_returns_einval() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(CGC_MAX_ALLOCATION as u128 + 1, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(
        outcome,
        SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
    ));
}

#[test]
fn allocate_null_addr_returns_efault() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(
        outcome,
        SyscallOutcome::Continue { ret } if ret == CGC_EFAULT
    ));
}

#[test]
fn allocate_bumps_and_maps_region() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    let base_before = state.cgc_allocation_base();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    // allocation_base shrinks by one page.
    assert_eq!(state.cgc_allocation_base(), base_before - 0x1000);
    let chosen = base_before - 0x1000;
    // The chosen address is written back to *addr.
    let stored = state.memory_load(0x2000, 4).expect("load").as_u64();
    assert_eq!(stored, Some(chosen));
    // The chosen page is now mapped (sanity-check via a store).
    state
        .memory_store(chosen, RustBV::concrete(42, 8))
        .expect("store on freshly allocated page");
}

#[test]
fn allocate_with_is_x_grants_execute() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    h.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 32),
            RustBV::concrete(1, 32), // is_x = true
            RustBV::concrete(0x2000, 32),
        ],
    )
    .expect("ok");
    let chosen = 0xB800_0000 - 0x1000;
    let page_num = chosen >> 12;
    let perm = state.memory().page_permissions(page_num).expect("mapped");
    assert!(perm.execute);
    assert!(perm.read);
    assert!(perm.write);
}

#[test]
fn allocate_round_up_length_to_page() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    let base_before = state.cgc_allocation_base();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 32), // 1 byte → 1 page
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    assert_eq!(state.cgc_allocation_base(), base_before - 0x1000);
}

#[test]
fn allocate_reuses_sinkhole_first_fit() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    // Pre-seed two sinkholes; first-fit picks the highest-addr one.
    state.cgc_add_sinkhole(0x9000_0000, 0x2000);
    state.cgc_add_sinkhole(0xA000_0000, 0x2000);
    let base_before = state.cgc_allocation_base();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    // allocation_base unchanged (came from sinkhole).
    assert_eq!(state.cgc_allocation_base(), base_before);
    // chosen is HIGH end of highest-addr sinkhole: 0xA0000000 + 0x1000.
    let chosen = state.memory_load(0x2000, 4).expect("load").as_u64();
    assert_eq!(chosen, Some(0xA000_1000));
    // The remaining 0x1000 of the 0xA000_0000 sinkhole stays in the list.
    let sinks = state.cgc_sinkholes();
    let has_lo_a = sinks.iter().any(|&(a, l)| a == 0xA000_0000 && l == 0x1000);
    let has_lo_9 = sinks.iter().any(|&(a, l)| a == 0x9000_0000 && l == 0x2000);
    assert!(has_lo_a, "remainder of A sinkhole should be retained");
    assert!(has_lo_9, "untouched 9 sinkhole should be retained");
}

#[test]
fn allocate_symbolic_length_falls_back() {
    let h = NativeAllocateSyscall;
    let mut state = x86_state_with_buf();
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "len", 32);
    let err = h
        .call(
            &mut state,
            &[sym, RustBV::concrete(0, 32), RustBV::concrete(0x2000, 32)],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn deallocate_unaligned_addr_returns_einval() {
    let h = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x1001, 32), RustBV::concrete(0x1000, 32)],
        )
        .expect("ok");
    assert!(matches!(
        outcome,
        SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
    ));
}

#[test]
fn deallocate_zero_length_returns_einval() {
    let h = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 32), RustBV::concrete(0, 32)],
        )
        .expect("ok");
    assert!(matches!(
        outcome,
        SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
    ));
}

#[test]
fn deallocate_null_addr_returns_einval() {
    let h = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0, 32), RustBV::concrete(0x1000, 32)],
        )
        .expect("ok");
    assert!(matches!(
        outcome,
        SyscallOutcome::Continue { ret } if ret == CGC_EINVAL
    ));
}

#[test]
fn deallocate_unmaps_region_and_records_sinkhole() {
    let h = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    // x86_state_with_buf maps [0x2000, 0x3000) RWX; deallocate it.
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 32), RustBV::concrete(0x1000, 32)],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    // Page is gone.
    assert!(!state.memory().is_mapped(0x2000u64));
    // Sinkhole records the freed run.
    let sinks = state.cgc_sinkholes();
    assert!(sinks.iter().any(|&(a, l)| a == 0x2000 && l == 0x1000));
}

#[test]
fn deallocate_unmapped_region_is_noop_with_zero_ret() {
    let h = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    // 0x9000 is not mapped — Python procedure returns 0 with no work.
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x9000, 32), RustBV::concrete(0x1000, 32)],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    // No sinkhole entry added.
    assert!(state.cgc_sinkholes().is_empty());
}

#[test]
fn deallocate_partial_run_unmaps_what_it_can() {
    let h = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    // Map a single page at 0x4000, ask to deallocate two pages — the
    // procedure should only free the one that's mapped.
    state.map_memory(0x4000, 0x1000, Permission::RWX);
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x4000, 32), RustBV::concrete(0x2000, 32)],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    assert!(!state.memory().is_mapped(0x4000u64));
    let sinks = state.cgc_sinkholes();
    assert!(sinks.iter().any(|&(a, l)| a == 0x4000 && l == 0x1000));
}

#[test]
fn deallocate_then_allocate_reuses_freed_region() {
    let alloc = NativeAllocateSyscall;
    let dealloc = NativeDeallocateSyscall;
    let mut state = x86_state_with_buf();
    // First allocate to bump the high-water down by a page.
    alloc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("alloc ok");
    let first_chosen = state
        .memory_load(0x2000, 4)
        .expect("load")
        .as_u64()
        .expect("concrete");
    // Deallocate that page.
    dealloc
        .call(
            &mut state,
            &[
                RustBV::concrete(first_chosen as u128, 32),
                RustBV::concrete(0x1000, 32),
            ],
        )
        .expect("dealloc ok");
    // Allocate again — should hand back the same page from the sinkhole.
    alloc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2000, 32),
            ],
        )
        .expect("realloc ok");
    let second_chosen = state
        .memory_load(0x2000, 4)
        .expect("load")
        .as_u64()
        .expect("concrete");
    assert_eq!(first_chosen, second_chosen, "sinkhole should be reused");
}

#[test]
fn random_zero_count_writes_zero_and_returns() {
    let h = NativeRandomSyscall;
    let mut state = x86_state_with_buf();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 32),
                RustBV::concrete(0, 32),
                RustBV::concrete(0x2800, 32),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let stored = state.memory_load(0x2800, 4).expect("load").as_u64();
    assert_eq!(stored, Some(0));
}

/// angr-ptf54: CGC `receive` on fd 0 consumes the harness-seeded stdin bytes
/// (bound by constraint) instead of minting unconstrained ones.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn receive_consumes_seeded_stdin() {
    let h = NativeReceiveSyscall;
    let mut state = x86_state_with_buf();
    let seed: Vec<RustBV> = vec![RustBV::concrete(0x41, 8), RustBV::concrete(0x42, 8)];
    state.file_system().set_fd_content_sym(0, seed);

    h.call(
        &mut state,
        &[
            RustBV::concrete(0, 32),
            RustBV::concrete(0x2000, 32),
            RustBV::concrete(4, 32),
            RustBV::concrete(0x2800, 32),
        ],
    )
    .expect("ok");

    for (i, want) in [0x41u128, 0x42].iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {i} is still a leaf symbol");
        assert_eq!(state.eval(&byte), Some(*want), "byte {i} binds to the seed");
    }
    // Only the 2 bytes past the seed are recorded for posix.dumps(0).
    assert_eq!(state.stdin_symbols().len(), 2);
}
