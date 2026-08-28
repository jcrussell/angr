//! Shared fixtures for the `file_path_*_tests` sibling modules (angr-5mnx3.70).
//!
//! Split out of the former single `file_path_tests.rs` so a helper used by
//! more than one topical module lives in exactly one place. Items are
//! `pub(super)` because every consumer is a sibling child of `file_path`.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::RustBV;


/// Map an RW page at 0x2000 and stage a NUL-terminated path byte-by-byte.
pub(super) fn stage_path(state: &mut RustSimState, addr: u64, path: &[u8]) {
    state.map_memory(addr & !0xfff, 0x1000, Permission::RWX);
    for (i, b) in path.iter().enumerate() {
        state
            .memory_store(addr + i as u64, RustBV::concrete(*b as u128, 8))
            .expect("store path byte");
    }
    state
        .memory_store(addr + path.len() as u64, RustBV::concrete(0, 8))
        .expect("store NUL");
}

/// Build an amd64 state with `path` staged (NUL-terminated) at 0x2000 —
/// the default staging address shared by most file_path syscall tests.
pub(super) fn state_with_path(path: &[u8]) -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("state");
    stage_path(&mut state, 0x2000, path);
    state
}

/// Unwrap a `Continue` outcome's return value, panicking with context
/// otherwise.
pub(super) fn expect_continue(outcome: SyscallOutcome) -> u64 {
    match outcome {
        SyscallOutcome::Continue { ret } => ret,
        other => panic!("expected Continue, got {other:?}"),
    }
}

/// Assert that `err` is a `SymbolicArgument` whose message contains
/// `needle`.
pub(super) fn assert_symbolic_arg(err: SyscallError, needle: &str) {
    match err {
        SyscallError::SymbolicArgument(msg) => assert!(
            msg.contains(needle),
            "expected SymbolicArgument message to contain {needle:?}, got {msg:?}",
        ),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

/// Assert that `err` is a `Memory` error (message not inspected).
pub(super) fn assert_memory_err(err: SyscallError) {
    match err {
        SyscallError::Memory(_) => {}
        other => panic!("expected Memory error, got {other:?}"),
    }
}

/// `AT_FDCWD` reinterpreted as unsigned 64-bit — same constant the
/// handler matches on. Kept inline so the test stays self-contained.
pub(super) const TEST_AT_FDCWD: u64 = AT_FDCWD_UNSIGNED;

/// Read `size` bytes of LE-packed u64 from memory at `addr`.
pub(super) fn read_u64_le(state: &RustSimState, addr: u64) -> u64 {
    let bv = state.memory_load(addr, 8).expect("load");
    bv.as_u64().expect("concrete")
}

pub(super) fn read_u32_le(state: &RustSimState, addr: u64) -> u32 {
    let bv = state.memory_load(addr, 4).expect("load");
    bv.as_u64().expect("concrete") as u32
}
