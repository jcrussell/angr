// Tests for exploration/helpers.rs — extracted from the former inline
// `#[cfg(test)] mod tests` block (see rust-mod-tests-sibling-extraction).
use super::*;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

// MIPS32 VEX guest-state offsets for $a0-$a3 (R4-R7).
const A0: u32 = 24;
const A1: u32 = 28;
const A2: u32 = 32;
const A3: u32 = 36;

/// O32 syscalls with >4 args read args 5+ from the stack at [sp+16].
/// A 6-arg MIPS32 syscall (futex/epoll_pwait/mmap2) must dispatch
/// natively when SP is concrete (angr-tvod).
#[test]
fn mips_o32_extract_syscall_args_reads_stack_window() {
    let mgr = RustExplorationManager::new("mips32", None).expect("mips32 mgr");
    let mut state = RustSimState::new("mips32").expect("mips32 state");

    // Register args $a0-$a3 = 0xa0..0xa3.
    state.set_register_by_offset(A0, RustBV::concrete(0xa0, 32));
    state.set_register_by_offset(A1, RustBV::concrete(0xa1, 32));
    state.set_register_by_offset(A2, RustBV::concrete(0xa2, 32));
    state.set_register_by_offset(A3, RustBV::concrete(0xa3, 32));

    // Stack-resident args 5,6 at [sp+16], [sp+20].
    let sp: u64 = 0x7fff_f000;
    state.map_memory(sp, 0x1000, crate::memory::Permission::RW);
    state.set_sp(RustBV::concrete(sp as u128, 32));
    state
        .memory_store(sp + 16, RustBV::concrete(0xa4, 32))
        .expect("store arg5");
    state
        .memory_store(sp + 20, RustBV::concrete(0xa5, 32))
        .expect("store arg6");

    let args = mgr.extract_syscall_args(&state, 6).expect("6 args");
    assert_eq!(args.len(), 6);
    let vals: Vec<u64> = args.iter().map(|a| a.as_u64().expect("concrete")).collect();
    assert_eq!(vals, vec![0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5]);
}

/// A symbolic SP must NOT fabricate stack args; it returns SpSymbolic so
/// the dispatcher falls through to the Python syscall callback.
#[test]
fn mips_o32_extract_syscall_args_symbolic_sp_falls_back() {
    let mgr = RustExplorationManager::new("mips32", None).expect("mips32 mgr");
    let mut state = RustSimState::new("mips32").expect("mips32 state");
    let ctx = SymContext::new();
    state.set_sp(RustBV::symbolic(&ctx, "sp", 32));

    match mgr.extract_syscall_args(&state, 6) {
        Err(ExtractionError::SpSymbolic) => {}
        other => panic!("expected SpSymbolic, got {other:?}"),
    }
}

/// ABIs that do not spill syscall args to the stack (no
/// syscall_stack_arg_offset) report RegisterOverflow, never a stack read.
#[test]
fn amd64_extract_syscall_args_overflow_is_register_overflow() {
    let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = RustSimState::new("amd64").expect("amd64 state");
    // amd64 exposes 6 syscall arg registers; a 7-arg request overflows.
    match mgr.extract_syscall_args(&state, 7) {
        Err(ExtractionError::RegisterOverflow {
            requested: 7,
            available: 6,
        }) => {}
        other => panic!("expected RegisterOverflow, got {other:?}"),
    }
}
